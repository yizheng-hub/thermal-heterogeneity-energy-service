"""Preserve every completed or interrupted attempt before disposable cleanup."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from thermal_heterogeneity import energyplus_launcher as launcher


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    for name in ["model.idf", "weather.epw", "energyplus.exe", "EnergyPlusAPI.dll"]:
        (project / name).write_text("locked " + name)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(launcher, "ROOT", project)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(temporary))
    folder = project / "output/case"
    arguments = ["-w", str(project / "weather.epw"), "-d", str(folder), "-r", str(project / "model.idf")]
    return project, temporary, folder, arguments


def invoke(case):
    project, _, folder, arguments = case
    return launcher.run_with_scratch(project / "energyplus.exe", arguments, folder,
                                    stdout=subprocess.DEVNULL, timeout_seconds=900)


@pytest.mark.parametrize("code,status", [(0, "finished"), (124, "timeout"), (127, "process_failed")])
def test_verified_commit_precedes_scratch_cleanup(case, monkeypatch, code, status):
    project, temporary, folder, _ = case

    def run(command, *, cwd, **kwargs):
        assert Path(command[-1]).is_absolute()
        assert Path(command[command.index("-w") + 1]).is_absolute()
        assert cwd.parent == temporary and cwd != folder
        (cwd / "eplusout.err").write_text("complete or partial retained error log")
        (cwd / "nested").mkdir()
        (cwd / "nested/state.bin").write_bytes(b"\x00\x01\xff")
        return code, status

    monkeypatch.setattr(launcher, "_run_process", run)
    result = invoke(case)
    audit = json.loads(Path(result.execution_audit_path).read_text())
    assert result.returncode == code and result.execution_status == status
    assert audit["status"] == "verified_outputs_committed"
    assert audit["timeout_seconds"] == 900
    assert (folder / "nested/state.bin").read_bytes() == b"\x00\x01\xff"
    assert audit["retained_sha256"]["nested/state.bin"] == launcher._sha(folder / "nested/state.bin")
    assert not list(temporary.iterdir())


def test_existing_different_output_preserves_both_attempts(case, monkeypatch):
    _, temporary, folder, _ = case
    folder.mkdir(parents=True)
    (folder / "eplusout.err").write_text("earlier attempt")

    def run(command, *, cwd, **kwargs):
        (cwd / "eplusout.err").write_text("new attempt")
        return 0, "finished"

    monkeypatch.setattr(launcher, "_run_process", run)
    with pytest.raises(FileExistsError, match="earlier retained output"):
        invoke(case)
    scratch, = temporary.iterdir()
    assert (scratch / "eplusout.err").read_text() == "new attempt"
    assert (folder / "eplusout.err").read_text() == "earlier attempt"
    audit, = folder.glob("ENGINE_EXECUTION_*.json")
    assert json.loads(audit.read_text())["status"] == "copy_not_committed_scratch_retained"


def test_unverified_process_termination_never_copies_or_deletes(case, monkeypatch):
    _, temporary, folder, _ = case

    def run(command, *, cwd, **kwargs):
        (cwd / "active.sql").write_text("still possibly open")
        raise RuntimeError("Cannot verify termination")

    monkeypatch.setattr(launcher, "_run_process", run)
    with pytest.raises(RuntimeError, match="termination"):
        invoke(case)
    scratch, = temporary.iterdir()
    assert (scratch / "active.sql").exists() and not (folder / "active.sql").exists()
    audit, = folder.glob("ENGINE_EXECUTION_*.json")
    assert json.loads(audit.read_text())["status"] == "termination_failed_scratch_retained"


def test_taskkill_oserror_is_a_termination_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="nt", environ={"SystemRoot": str(tmp_path)}))
    class Child:
        pid = 12345

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("owned child", timeout)

        def poll(self):
            return None

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("taskkill unavailable")))
    with pytest.raises(RuntimeError, match="owned child process tree"):
        launcher._run_process(["owned child"], stdout=subprocess.DEVNULL, cwd=Path.cwd(), timeout_seconds=1)


def test_taskkill_failure_retains_scratch_even_if_parent_has_exited(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name="nt", environ={"SystemRoot": str(tmp_path)}))
    class Child:
        pid = 12345
        polls = 0

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("owned child", timeout)

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0

    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: Child())
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1))
    with pytest.raises(RuntimeError, match="owned child process tree"):
        launcher._run_process(["owned child"], stdout=subprocess.DEVNULL, cwd=Path.cwd(), timeout_seconds=1)


def test_compact_output_cannot_escape_project(case, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/global_extension"))
    import global_matrix

    project, temporary, folder, _ = case
    with pytest.raises(ValueError, match="Retained simulation outputs"):
        global_matrix.run_energyplus(project / "model.idf", project / "weather.epw",
            project / "energyplus.exe", folder, temporary / "outside.npz", [])
    assert not folder.exists() and not list(temporary.iterdir())


def test_outside_destination_is_rejected_before_creation(case):
    project, temporary, _, arguments = case
    outside = temporary / "not_allowed"
    arguments[arguments.index("-d") + 1] = str(outside)
    with pytest.raises(ValueError, match="inside this project"):
        launcher.run_with_scratch(project / "energyplus.exe", arguments, outside, stdout=subprocess.DEVNULL)
    assert not outside.exists() and not list(temporary.iterdir())


def test_failed_preflights_retain_separate_attempt_reports(case, monkeypatch):
    project, _, _, _ = case
    report = project / "audit/startup.json"
    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("startup")))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="startup check failed"):
            launcher.verify_runtime(project / "energyplus.exe", report)
    attempts = list(report.parent.glob("startup*.json"))
    assert len(attempts) == 2
    assert all(json.loads(p.read_text())["status"] == "startup_failed" for p in attempts)


def test_source_content_change_during_collection_retains_scratch(case, monkeypatch):
    _, temporary, folder, _ = case

    def run(command, *, cwd, **kwargs):
        (cwd / "output.bin").write_text("before")
        return 0, "finished"

    original_copy = launcher.shutil.copy2

    def copy_then_change(source, target):
        original_copy(source, target)
        source.write_text("after")

    monkeypatch.setattr(launcher, "_run_process", run)
    monkeypatch.setattr(launcher.shutil, "copy2", copy_then_change)
    with pytest.raises(ValueError, match="changed during collection"):
        invoke(case)
    assert len(list(temporary.iterdir())) == 1
    audit, = folder.glob("ENGINE_EXECUTION_*.json")
    assert json.loads(audit.read_text())["status"] == "copy_not_committed_scratch_retained"


def test_stale_adjacent_readvars_controls_are_not_reused(case, monkeypatch):
    project, _, folder, _ = case
    stale = project / "model.rvi"
    old_output = project.parent / "an_old_output"
    stale.write_text(f"{old_output / 'eplusout.eso'}\n{old_output / 'eplusout.csv'}\n")

    def run(command, *, cwd, **kwargs):
        input_path = Path(command[-1])
        assert input_path == cwd / "in.idf"
        assert input_path.read_bytes() == (project / "model.idf").read_bytes()
        assert not input_path.with_suffix(".rvi").exists()
        (cwd / "eplusout.csv").write_text("fresh conversion")
        return 0, "finished"

    monkeypatch.setattr(launcher, "_run_process", run)
    invoke(case)
    assert (folder / "eplusout.csv").read_text() == "fresh conversion"
    assert stale.read_text().startswith(str(old_output))
