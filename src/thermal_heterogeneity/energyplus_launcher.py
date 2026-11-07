"""Run the contained EnergyPlus library in a separate Python process per case."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def energyplus_command(executable, arguments):
    return [sys.executable, str(Path(__file__).resolve()),
        "--runtime-root", str(Path(executable).resolve().parent), "--", *map(str, arguments)]


def runtime_provenance(executable):
    library = Path(executable).parent / "EnergyPlusAPI.dll"
    with library.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = ROOT / "SELF_CONTAINED_PROJECT_MANIFEST.json"
    package_checksum = None
    if manifest.is_file() and library.resolve().is_relative_to(ROOT):
        prefix = library.resolve().parent.relative_to(ROOT).as_posix() + "/"
        files = json.loads(manifest.read_text())["files"]
        entries = {name: info["sha256"] for name, info in files.items() if name.startswith(prefix)}
        package_checksum = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"runtime_backend": "official_python_c_api_subprocess",
        "runtime_api_sha256": checksum, "runtime_package_manifest_sha256": package_checksum}


def verify_runtime(executable, report_path=None):
    if report_path is not None:
        report_path = Path(report_path)
        count = 1
        original = report_path
        while report_path.exists():
            count += 1
            report_path = original.with_name(original.stem + f"_{count:03d}" + original.suffix)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "command": energyplus_command(executable, ["--version"])}
    if report_path is not None:
        _atomic_json(report_path, report)
    try:
        report.update(runtime_provenance(executable))
        result = subprocess.run(report["command"], capture_output=True,
            text=True, errors="replace", timeout=120)
        report.update(status="completed" if result.returncode == 0 else "startup_failed",
            exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        report.update(status="startup_failed", exit_code=124 if isinstance(error, subprocess.TimeoutExpired) else 127,
            exception=repr(error))
    if report_path is not None:
        _atomic_json(report_path, report)
    if report["exit_code"]:
        raise RuntimeError("Contained EnergyPlus API startup check failed: " + str(report))
    return report


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_json(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def _run_process(command, *, stdout, cwd, timeout_seconds):
    child = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
        cwd=cwd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        code = child.wait(timeout=timeout_seconds)
        return code, "finished" if code == 0 else "process_failed"
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        status = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
        if child.poll() is None:
            if os.name == "nt":
                tool = Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"
                try:
                    stopped = subprocess.run([str(tool), "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True, check=False, timeout=15)
                except (OSError, subprocess.TimeoutExpired) as error:
                    raise RuntimeError("Cannot verify termination of the owned child process tree") from error
                if stopped.returncode:
                    raise RuntimeError("Cannot verify termination of the owned child process tree")
            else:
                child.kill()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Owned child process termination did not complete") from error
        return (124 if status == "timeout" else 130), status


def run_with_scratch(executable, arguments, run_folder, *, stdout, timeout_seconds=900):
    """Create disposable working files; commit their verified copies before cleanup."""
    final = Path(run_folder).resolve()
    args = list(map(str, arguments))
    output_index = args.index("-d") + 1
    if Path(args[output_index]).resolve() != final:
        raise ValueError("The retained output folder differs from the declared invocation")
    inputs = [Path(args[-1]).resolve(), Path(args[args.index("-w") + 1]).resolve()]
    if not all(path.is_relative_to(ROOT) for path in [final, *inputs, Path(executable).resolve()]):
        raise ValueError("Scientific inputs, runtime and retained outputs must remain inside this project")
    final.mkdir(parents=True, exist_ok=True)
    args[-1] = str(inputs[0])
    args[args.index("-w") + 1] = str(inputs[1])
    locked = {"scientific_input_sha256": {str(p): _sha(p) for p in inputs},
        **runtime_provenance(executable), "runtime_executable_sha256": _sha(Path(executable))}
    parent = Path(tempfile.gettempdir()).resolve()
    filesystem = "non_windows"
    if os.name == "nt":
        name = ctypes.create_unicode_buffer(64)
        if not ctypes.windll.kernel32.GetVolumeInformationW(str(parent.anchor), None, 0, None, None, None, name, len(name)):
            raise OSError("Cannot inspect the temporary working volume")
        filesystem = name.value
        if filesystem.upper() != "NTFS":
            raise ValueError("The automatically created EnergyPlus working directory requires an NTFS temporary volume")
    scratch = Path(tempfile.mkdtemp(prefix="hem_energyplus_", dir=parent)).resolve()
    if scratch.parent != parent or not scratch.name.startswith("hem_energyplus_"):
        raise ValueError("Temporary directory ownership boundary is invalid")
    token = scratch.name + ":" + str(os.getpid())
    owner = scratch / ".hem_scratch_owner"
    owner.write_text(token)
    audit = final / ("ENGINE_EXECUTION_" + scratch.name + ".json")
    report = {"status": "staging_input", "logical_run_id": final.name, "attempt_id": scratch.name,
        "temporary_working_directory": str(scratch), "temporary_filesystem": filesystem,
        "retained_run_folder": str(final), **locked, "timeout_seconds": timeout_seconds}
    _atomic_json(audit, report)
    # ReadVarsESO reuses adjacent .rvi/.mvi files. A clean local copy prevents
    # stale output paths from an earlier execution directory being reused.
    staged_input = scratch / "in.idf"
    try:
        shutil.copy2(inputs[0], staged_input)
        if _sha(staged_input) != locked["scientific_input_sha256"][str(inputs[0])]:
            raise ValueError("The staged solver input differs from the retained scientific input")
    except BaseException as error:
        report.update(status="input_staging_failed_scratch_retained", exception=repr(error))
        _atomic_json(audit, report)
        raise
    args[-1] = str(staged_input)
    args[output_index] = str(scratch)
    command = energyplus_command(executable, args)
    report.update(status="running", command=command)
    _atomic_json(audit, report)
    started = time.monotonic()
    try:
        code, status = _run_process(command, stdout=stdout, cwd=scratch, timeout_seconds=timeout_seconds)
    except OSError as error:
        code, status = 127, "infrastructure_startup_failure"
        report["exception"] = repr(error)
    except RuntimeError as error:
        report.update(status="termination_failed_scratch_retained", exception=repr(error))
        _atomic_json(audit, report)
        raise
    report.update(exit_code=code, execution_status=status, seconds=time.monotonic() - started)
    source = {}
    try:
        for path in sorted(scratch.rglob("*")):
            if not path.is_file() or path == owner:
                continue
            if not path.resolve().is_relative_to(scratch):
                raise ValueError("A temporary output resolves outside its owned directory")
            relative = path.relative_to(scratch)
            source[relative.as_posix()] = _sha(path)
            target = final / relative
            if not target.resolve().is_relative_to(final):
                raise ValueError("A retained output resolves outside its project run directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha(target) != source[relative.as_posix()]:
                    raise FileExistsError("An earlier retained output is preserved: " + str(target))
            else:
                shutil.copy2(path, target)
            if _sha(target) != source[relative.as_posix()]:
                raise ValueError("Temporary output copy did not verify: " + str(relative))
        after = {p.relative_to(scratch).as_posix(): _sha(p) for p in scratch.rglob("*") if p.is_file() and p != owner}
        if after != source:
            raise ValueError("Temporary outputs changed during collection")
    except BaseException as error:
        report.update(status="copy_not_committed_scratch_retained", exception=repr(error), source_output_sha256=source)
        _atomic_json(audit, report)
        raise
    report.update(status="verified_outputs_committed", retained_sha256=source)
    _atomic_json(audit, report)
    if scratch.resolve().parent != parent or owner.read_text() != token:
        raise ValueError("Owned temporary cleanup check failed")
    shutil.rmtree(scratch)
    result = subprocess.CompletedProcess(command, code)
    result.execution_status = status
    result.execution_audit_path = str(audit)
    if status == "interrupted":
        raise KeyboardInterrupt("Interrupted attempt was retained before temporary cleanup")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args, engine_args = parser.parse_known_args()
    runtime = args.runtime_root.resolve()
    if not runtime.is_relative_to(ROOT) or not Path(sys.executable).resolve().is_relative_to(ROOT):
        raise ValueError("EnergyPlus and Python must belong to this contained project")
    if engine_args[:1] == ["--"]:
        engine_args = engine_args[1:]
    sys.path.insert(0, str(runtime))
    directory = os.add_dll_directory(str(runtime))
    try:
        from pyenergyplus.api import EnergyPlusAPI

        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        try:
            api.verify_api_version_match(state)
            return api.runtime.run_energyplus(state, engine_args)
        finally:
            api.state_manager.delete_state(state)
    finally:
        directory.close()


if __name__ == "__main__":
    raise SystemExit(main())
