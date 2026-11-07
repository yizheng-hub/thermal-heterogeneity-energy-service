"""Recover ReadVars conversion from a hash-bound completed annual physical run."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
from global_matrix import extract_hourly  # noqa: E402 - initialize local imports before loading project modules
from run_global_matrix import ZONES  # noqa: E402 - initialize local imports before loading project modules
from thermal_heterogeneity.energyplus_launcher import runtime_provenance  # noqa: E402 - initialize local imports before loading project modules


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def recover(folder, idf, weather, compact):
    folder, idf, weather, compact = map(lambda p: Path(p).resolve(), [folder, idf, weather, compact])
    if not all(p.is_relative_to(ROOT) for p in [folder, idf, weather, compact]):
        raise ValueError("Recovery inputs and outputs must belong to this project")
    if (folder / "RUN_QA.json").exists():
        raise FileExistsError("An existing quality record is preserved")
    if compact.exists() or (folder / "eplusout.csv").exists():
        raise FileExistsError("Existing partial postprocessing outputs are preserved; use a separate recovery attempt")
    audits = list(folder.glob("ENGINE_EXECUTION_*.json"))
    if len(audits) != 1:
        raise ValueError("Recovery requires one unambiguous execution audit")
    execution = json.loads(audits[0].read_text())
    runtime = ROOT / "tools/EnergyPlus-22.1.0/energyplus.exe"
    info = runtime_provenance(runtime)
    if execution.get("status") != "verified_outputs_committed" or execution.get("exit_code") != 0:
        raise ValueError("Physical execution did not complete and commit its outputs")
    if any(execution.get(k) != v for k, v in info.items()) or execution["runtime_executable_sha256"] != sha(runtime):
        raise ValueError("Retained physical runtime differs from the recovery runtime")
    inputs = {Path(k).suffix.lower(): v for k, v in execution["scientific_input_sha256"].items()}
    if inputs.get(".idf") != sha(idf) or inputs.get(".epw") != sha(weather):
        raise ValueError("Retained physical inputs differ from the recovery inputs")
    for name in ["eplusout.eso", "eplusout.sql", "eplusout.err", "eplusout.eio"]:
        if sha(folder / name) != execution["retained_sha256"][name]:
            raise ValueError("A completed physical output changed: " + name)
    text = (folder / "eplusout.err").read_text(errors="replace")
    counts = re.findall(r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;", text)
    if not counts:
        raise ValueError("Retained physical error report has no successful completion")
    converter = runtime.parent / "PostProcess/ReadVarsESO.exe"
    converter_sha256 = sha(converter)
    work = Path(tempfile.mkdtemp(prefix="hem_readvars_")).resolve()
    if work.parent != Path(tempfile.gettempdir()).resolve():
        raise ValueError("Owned conversion workspace boundary is invalid")
    token = work.name + ":" + str(os.getpid())
    owner = work / ".hem_readvars_owner"
    owner.write_text(token)
    control = work / "convert.rvi"
    csv = work / "eplusout.csv"
    control.write_text(str(folder / "eplusout.eso") + "\n" + str(csv) + "\n")
    command = [str(converter), str(control), "unlimited"]
    attempt = folder / ("CONVERSION_RECOVERY_" + work.name)
    attempt.mkdir()
    report_path = attempt / "AUDIT.json"
    report = {"status": "running", "scope": "Output conversion only; completed physical simulation is not rerun",
              "execution_audit": audits[0].name, "input_idf_sha256": sha(idf), "weather_sha256": sha(weather),
              "source_eso_sha256": sha(folder / "eplusout.eso"), "readvars_sha256": converter_sha256,
              "command": command, "temporary_workspace": str(work)}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, timeout=120)
        (attempt / "CONSOLE.log").write_bytes(result.stdout + result.stderr)
        if result.returncode or not csv.is_file():
            raise RuntimeError("Retained ESO conversion did not complete")
        target = attempt / "eplusout.csv"
        shutil.copy2(csv, target)
        if sha(csv) != sha(target):
            raise ValueError("Recovered CSV copy did not verify")
        for name in ["eplusout.sql", "eplusout.err"]:
            shutil.copy2(folder / name, attempt / name)
            if sha(attempt / name) != execution["retained_sha256"][name]:
                raise ValueError("Physical output copy for extraction did not verify: " + name)
        qa = {"completed": True, "exit_code": 0, "seconds": execution["seconds"],
              "reported_warning_count": int(counts[-1][0]), "reported_severe_count": int(counts[-1][1]),
              "fatal_occurrences": len(re.findall(r"\*\*\s+Fatal\s+\*\*", text)),
              "input_idf_sha256": sha(idf), "weather_sha256": sha(weather), "runtime_sha256": sha(runtime),
              "compact_output": str(compact), **info, "execution_audit_path": str(audits[0]),
              "execution_status": "completed_physical_run_with_recovered_readvars_conversion",
              "postprocess_recovery_audit_relative_to_run_folder": report_path.relative_to(folder).as_posix()}
        staged_compact = attempt / "hourly.npz"
        qa.update(extract_hourly(attempt, ZONES, staged_compact))
        compact.parent.mkdir(parents=True, exist_ok=True)
        for source, destination in [(target, folder / "eplusout.csv"), (staged_compact, compact)]:
            if destination.exists():
                raise FileExistsError("A simultaneous recovery output is preserved: " + str(destination))
            shutil.copy2(source, destination)
            if sha(destination) != sha(source):
                raise ValueError("Committed recovery output copy did not verify")
        (folder / "RUN_QA.json").write_text(json.dumps(qa, indent=2) + "\n")
        report.update(status="completed", recovered_csv_sha256=sha(target), hourly_sha256=sha(compact))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        report.update(status="not_completed_workspace_retained", exception=repr(error))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        raise
    if (work.parent != Path(tempfile.gettempdir()).resolve() or not work.name.startswith("hem_readvars_")
            or owner.read_text() != token):
        raise ValueError("Owned conversion cleanup boundary is invalid")
    shutil.rmtree(work)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run-folder", "idf", "weather", "compact"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.run_folder, args.idf, args.weather, args.compact)), flush=True)
