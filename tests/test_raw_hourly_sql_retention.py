"""The public fresh matrix retains diagnostic SQL without changing physical calls."""

from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
import run_global_matrix_retained_sql as module  # noqa: E402 - initialize local imports before loading project modules


def test_only_the_postprocessing_cleanup_flag_changes(monkeypatch):
    positions = tuple(object() for _ in range(6))
    captured = {}
    result = {"completed": True}

    def execute(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return result

    monkeypatch.setattr(module, "original_run_energyplus", execute)
    assert module.run_with_retained_sql(*positions, cleanup_raw=True, timeout_seconds=900) is result
    assert captured["args"] == positions
    assert captured["kwargs"] == {"cleanup_raw": False, "timeout_seconds": 900}


def test_cli_restores_the_original_binding_when_the_matrix_raises(monkeypatch):
    previous = module.matrix.run_energyplus

    def execute():
        assert module.matrix.run_energyplus is module.run_with_retained_sql
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(module.matrix, "main", execute)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        module.main()
    assert module.matrix.run_energyplus is previous


def test_fresh_raw_entrypoint_uses_the_retaining_cli_in_its_own_project():
    workflow = runpy.run_path(str(ROOT / "reproduce_from_raw.py"))
    output = ROOT / "output/fresh_not_executed"
    runtime = ROOT / "tools/EnergyPlus-22.1.0/energyplus.exe"
    command, = workflow["commands"](output, runtime, 4)["matrix"]
    assert Path(command[1]) == ROOT / "src/analysis/global_extension/run_global_matrix_retained_sql.py"
    assert command[command.index("--output-root") + 1] == str(output / "matrix")
    assert command[command.index("--runtime") + 1] == str(runtime)
