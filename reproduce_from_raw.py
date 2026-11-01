"""Rebuild demand, annual, spatial and response evidence from retained raw inputs.

Every stage writes its own log and status. No archived result table or old
assignment file is an input. EnergyPlus is an explicit versioned dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
GLOBAL = ROOT / "src/analysis/global_extension"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commands(output, runtime, workers):
    demand = output / "demand"
    matrix = output / "matrix"
    spatial = output / "spatial"
    response = output / "response"
    results = output / "results"
    climate = GLOBAL / "GLOBAL_CLIMATES.json"
    carbon = ROOT / "data/metadata/global_electricity_carbon_factors_ember_2025.csv"
    profiles = matrix / "GLOBAL_PROFILES.csv"
    annual = matrix / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv"
    probe = matrix / "probe_runs/PROBE_RUN_MANIFEST.csv"

    def call(script, *args):
        return [sys.executable, str(script), *map(str, args)]

    return {
        "demand": [
            call(
                ROOT / "src/utilities/rebuild_profile_assignments.py",
                "--output-root",
                demand,
                "--workers",
                workers,
            )
        ],
        "observations": [
            call(
                ROOT / "src/utilities/summarize_raw_observations.py",
                "--demand-root",
                demand,
                "--output-root",
                results / "observations",
            ),
            call(
                ROOT / "src/analysis/validation/shared_room_reproduction.py",
                "--upstream-root",
                ROOT / "data/upstream/shared_room_observations",
                "--output-root",
                results / "shared_room",
            ),
            call(
                ROOT / "src/utilities/observation_figure_sources.py",
                "--demand-root", demand,
                "--output-root", results / "observation_figures",
            ),
        ],
        "matrix": [
            call(
                GLOBAL / "run_global_matrix_retained_sql.py",
                "--prototype-root",
                ROOT / "data/upstream/global_climate_matrix/prototypes_22_1",
                "--weather-root",
                ROOT / "data/upstream/global_climate_matrix/weather",
                "--reference-idf",
                ROOT / "data/upstream/frozen_buildings/Miami/baseline.idf",
                "--ashrae-profiles",
                demand / "ASHRAE_ASSIGNMENTS.npz",
                "--ctcd-profiles",
                demand / "CTCD_ASSIGNMENTS.csv.gz",
                "--prototype-mode",
                "fixed_reference",
                "--candidate-scope",
                "all",
                "--automatic-solver-stability",
                "--runtime",
                runtime,
                "--output-root",
                matrix,
                "--workers",
                workers,
            )
        ],
        "annual": [
            call(
                GLOBAL / "analyse_global_results.py",
                "--run-manifest",
                annual,
                "--anchor-manifest",
                probe,
                "--profiles",
                profiles,
                "--climates",
                climate,
                "--carbon-factors",
                carbon,
                "--output",
                results / "global_final",
            )
        ],
        "seasonal": [
            call(
                GLOBAL / "reproduce_seasonal.py",
                "--raw-run-root",
                output,
                "--runtime",
                runtime,
                "--workers",
                workers,
            )
        ],
        "spatial": [
            call(
                GLOBAL / "global_spatial_extension.py",
                "plan",
                "--controller-root",
                matrix / "control_generation/controller_input",
                "--profiles",
                profiles,
                "--model-manifest",
                matrix / "models/MODEL_MANIFEST.json",
                "--runtime",
                runtime,
                "--output-root",
                spatial,
            ),
            call(
                GLOBAL / "automatic_solver_stability.py",
                "--stage",
                "spatial",
                "--raw-run-root",
                output,
                "--runtime",
                runtime,
                "--workers",
                workers,
            ),
            call(
                GLOBAL / "global_spatial_extension.py",
                "score",
                "--logical-plan",
                spatial / "GLOBAL_SPATIAL_LOGICAL_PLAN.csv",
                "--completed-manifest",
                spatial / "GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv",
                "--annual-manifest",
                annual,
                "--profiles",
                spatial / "GLOBAL_SPATIAL_PROFILES.csv",
                "--reference-manifest",
                probe,
                "--carbon-factors",
                carbon,
                "--output-root",
                results / "spatial_final",
            ),
        ],
        "response": [
            call(
                GLOBAL / "multi_event_dr_extension.py",
                "reference",
                "--probe-manifest",
                probe,
                "--climates",
                climate,
                "--output",
                response / "REFERENCE_MANIFEST.csv",
            ),
            call(
                GLOBAL / "multi_event_dr_extension.py",
                "protocol",
                "--reference-manifest",
                response / "REFERENCE_MANIFEST.csv",
                "--output",
                response / "EVENT_PROTOCOL.csv",
            ),
            call(
                GLOBAL / "multi_event_dr_extension.py",
                "plan",
                "--annual-manifest",
                annual,
                "--protocol",
                response / "EVENT_PROTOCOL.csv",
                "--profiles",
                profiles,
                "--output-root",
                response,
            ),
            call(
                GLOBAL / "automatic_solver_stability.py",
                "--stage",
                "response",
                "--raw-run-root",
                output,
                "--runtime",
                runtime,
                "--workers",
                workers,
            ),
            call(
                GLOBAL / "multi_event_dr_extension.py",
                "score",
                "--run-manifest",
                response / "EVENT_RUN_MANIFEST_COMPLETED.csv",
                "--event-manifest",
                response / "EVENT_EVALUATION_MANIFEST.csv",
                "--profiles",
                profiles,
                "--reference-manifest",
                response / "REFERENCE_MANIFEST.csv",
                "--output",
                results / "multi_event_dr_final/EVENT_RESPONSE_SCORES.csv",
            ),
        ],
        "secondary": [call(ROOT / "reproduce_secondary_analyses.py", "--raw-run-root", output)],
        "strengthening": [
            call(
                ROOT / "reproduce_strengthening.py",
                "--raw-run-root",
                output,
                "--energyplus",
                runtime,
                "--workers",
                workers,
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--energyplus", type=Path, default=ROOT / "tools/EnergyPlus-22.1.0/energyplus.exe"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "demand",
            "observations",
            "matrix",
            "annual",
            "seasonal",
            "spatial",
            "response",
            "secondary",
            "strengthening",
        ),
        default=[
            "demand",
            "observations",
            "matrix",
            "annual",
            "seasonal",
            "spatial",
            "response",
            "secondary",
            "strengthening",
        ],
    )
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    runtime = args.energyplus.resolve()
    if not output.is_relative_to(ROOT) or not runtime.is_relative_to(ROOT):
        raise ValueError(
            "Keep the output directory and EnergyPlus installation inside this project. Move the whole project to a drive with sufficient space before running."
        )
    jobs = commands(output, runtime, args.workers)
    plan = {
        "stages": {stage: jobs[stage] for stage in args.stages},
        "scope": "Raw observations and retained physical inputs through new annual, spatial and DR trajectories and scores",
        "runtime": str(runtime),
        "historical_archive_inputs": [],
        "storage_preflight": {
            "minimum_free_gib": 250 if "strengthening" in args.stages else 125 if "matrix" in args.stages else 25,
            "scope": "Conservative startup working-space allowance for full hourly SQL, trajectories and later archives; not a total-storage or runtime guarantee. Monitor free space during each stage.",
        },
    }
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    required = plan["storage_preflight"]["minimum_free_gib"] * 2**30
    if {"matrix", "seasonal", "spatial", "response", "strengthening"} & set(
        args.stages
    ) and shutil.disk_usage(ROOT).free < required:
        raise OSError(
            "Insufficient free space for the selected simulation stages. Move the complete project to a drive with at least "
            + str(required // 2**30)
            + " GiB free, keeping its inputs and outputs together."
        )
    if not runtime.is_file():
        raise FileNotFoundError(
            "EnergyPlus 22.1 is required. Supply --energyplus or install tools/EnergyPlus-22.1.0."
        )
    expected = "8e56b46647e109944d8762df6f35eff2fa149c94dc43a78cbf17c69f56e3ee27"
    if sha(runtime) != expected:
        raise ValueError("Runtime differs from the locked EnergyPlus 22.1 executable")
    for dependency in (
        "Energy+.idd",
        "python_standard_lib/encodings/__init__.py",
        "PostProcess/ReadVarsESO.exe",
    ):
        if not (runtime.parent / dependency).is_file():
            raise FileNotFoundError(f"Incomplete EnergyPlus installation: {dependency}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from utilities.audit_reproduction_inputs import inventory
    from thermal_heterogeneity.energyplus_launcher import verify_runtime

    if {"matrix", "seasonal", "spatial", "response", "strengthening"} & set(args.stages):
        verify_runtime(runtime, output / "RUNTIME_API_PREFLIGHT.json")

    version = 1
    lock = output / "provenance" / f"INPUT_SOURCE_RUNTIME_LOCK_{version:03d}.json"
    while lock.exists():
        version += 1
        lock = lock.with_name(f"INPUT_SOURCE_RUNTIME_LOCK_{version:03d}.json")
    inventory(
        ROOT,
        runtime,
        lock,
        include_validation=bool({"strengthening", "observations"} & set(args.stages)),
    )
    plan["input_source_runtime_lock"] = str(lock.relative_to(output))
    (output / "EXECUTION_PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    status_path = output / "STAGE_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    for stage in args.stages:
        if stage == "demand" and (output / "demand/PROFILE_REBUILD_AUDIT.json").is_file():
            raise FileExistsError(
                "Demand outputs already exist; omit demand from --stages to resume downstream stages"
            )
        for index, command in enumerate(jobs[stage]):
            attempt = 1
            log = logs / f"{stage}_{index}.log"
            while log.exists():
                attempt += 1
                log = logs / f"{stage}_{index}_attempt{attempt}.log"
            started = time.time()
            status[stage] = {
                "status": "running",
                "step": index,
                "attempt": attempt,
                "command": command,
                "source_sha256": sha(command[1]),
                "log": str(log.relative_to(output)),
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(stage, index, "running", flush=True)
            with log.open("wb") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            status[stage].update(
                status="completed" if result.returncode == 0 else "failed",
                returncode=result.returncode,
                seconds=time.time() - started,
            )
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            if result.returncode:
                raise RuntimeError(
                    f"{stage} step {index} failed. Inspect {log}; no stage is silently skipped."
                )
        if stage == "response":
            for name in ["REFERENCE_MANIFEST.csv", "EVENT_PROTOCOL.csv"]:
                shutil.copy2(
                    output / "response" / name, output / "results/multi_event_dr_final" / name
                )
        print(stage, "completed", flush=True)


if __name__ == "__main__":
    main()
