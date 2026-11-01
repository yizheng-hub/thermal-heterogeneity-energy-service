"""Generate solver-only evidence from the active fresh run, without old archives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from global_matrix import run_energyplus, sha256
from run_global_matrix import ZONES, set_maximum_hvac_iterations, warning_profile, _cached_run
from reconstruct_probe_manifest import reconstruct_probe_manifest
from compare_solver_iterations import compare_probe_manifests


def audit_probes(batch, runtime, workers):
    first = reconstruct_probe_manifest(batch, expected_solver_iterations=40)
    failed = first.loc[~first.relative_engineering_pass].copy()
    if failed.empty:
        raise ValueError("Probe failure did not identify an eligible solver-only case")
    for row in failed.itertuples():
        categories = set(json.loads(row.new_warning_categories))
        critical = set(json.loads(row.critical_warning_count_increases))
        qa = json.loads((batch / "probe_runs/runs" / row.run_id / "RUN_QA.json").read_text())
        if (
            not categories.issubset({"hvac_iterations"})
            or critical != {"hvac_iterations"}
            or qa.get("reported_severe_count") != 0
            or qa.get("fatal_occurrences") != 0
            or qa.get("unexplained_recurring_groups")
        ):
            raise ValueError(
                f"Probe {row.run_id} needs engineering review beyond solver iteration limits"
            )
    models = json.loads((batch / "models/MODEL_MANIFEST.json").read_text())
    output = batch / "solver_stability/probes"
    output.mkdir(parents=True, exist_ok=True)

    def execute(row):
        original = batch / "probe_runs/inputs" / (row.run_id + ".idf")
        text = set_maximum_hvac_iterations(original.read_text(), 100)
        folder = output / row.run_id
        folder.mkdir(exist_ok=True)
        idf = folder / "input.idf"
        idf.write_text(text)
        weather = Path(models[row.climate]["weather_path"])
        job = {
            "run_id": row.run_id,
            "run_folder": folder / "run",
            "compact": folder / "hourly.npz",
            "input_sha256": sha256(idf),
            "weather_sha256": sha256(weather),
        }
        qa = _cached_run(job, sha256(runtime))
        if qa is None:
            qa = run_energyplus(
                idf, weather, runtime, job["run_folder"], job["compact"], ZONES, cleanup_raw=True
            )
        if not qa["completed"] or qa["reported_severe_count"] != 0 or qa["fatal_occurrences"] != 0:
            raise ValueError(f"Higher-iteration probe did not complete cleanly: {row.run_id}")
        return {
            "climate": row.climate,
            "target_c": row.target_c,
            "maximum_hvac_iterations": 100,
            "hourly_path": str(job["compact"]),
            "treatment_warning_profile": json.dumps(
                warning_profile(job["run_folder"] / "eplusout.err")
            ),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        second = pd.DataFrame(pool.map(execute, failed.itertuples()))
    evidence = compare_probe_manifests(failed, second)
    target = output / "PROBE_SOLVER_STABILITY.csv"
    evidence.to_csv(target, index=False)
    # The normal runner applies the existing strict exception rules to this
    # freshly generated evidence. No service or energy direction is a gate.
    return target


def audit_annual(manifest, runtime, output, workers):
    from audit_annual_solver_stability import audit_annual_solver_stability

    result = audit_annual_solver_stability(pd.read_csv(manifest), runtime, output, workers=workers)
    if not result.solver_stability_pass.all():
        raise ValueError("Fresh annual solver comparison did not pass every declared check")
    return output / "ANNUAL_SOLVER_STABILITY.csv"


def ensure_reference_metadata(batch, runtime, workers):
    """Repair missing sizing metadata using exact fresh reference reruns."""
    manifest = pd.read_csv(batch / "probe_runs/PROBE_RUN_MANIFEST.csv")
    anchors = manifest.loc[manifest.target_c.eq(25)]
    models = json.loads((batch / "models/MODEL_MANIFEST.json").read_text())

    def repair(row):
        source = batch / "probe_runs/runs" / row.run_id
        metadata = source / "simulation_metadata.sql"
        if metadata.is_file() or (source / "eplusout.sql").is_file():
            return {"climate": row.climate, "status": "already_available"}
        folder = batch / "reference_metadata_replay" / row.run_id
        idf = batch / "probe_runs/inputs" / (row.run_id + ".idf")
        compact = folder / "hourly.npz"
        weather = Path(models[row.climate]["weather_path"])
        job = {
            "run_id": row.run_id,
            "run_folder": folder / "run",
            "compact": compact,
            "input_sha256": sha256(idf),
            "weather_sha256": sha256(weather),
        }
        qa = _cached_run(job, sha256(runtime))
        if qa is None:
            qa = run_energyplus(
                idf, weather, runtime, job["run_folder"], compact, ZONES, cleanup_raw=True
            )
        if not qa["completed"]:
            raise ValueError(f"Reference metadata replay did not complete: {row.climate}")
        with (
            np.load(row.hourly_path, allow_pickle=False) as original,
            np.load(compact, allow_pickle=False) as replay,
        ):
            if set(original.files) != set(replay.files) or not all(
                np.array_equal(original[k], replay[k]) for k in original.files
            ):
                raise ValueError(f"Reference replay changed a retained hourly array: {row.climate}")
        shutil.copy2(qa["sql_metadata_path"], metadata)
        return {
            "climate": row.climate,
            "status": "exact_reference_replay",
            "metadata_sha256": sha256(metadata),
            "replay_qa": str(job["run_folder"] / "RUN_QA.json"),
            "original_hourly_sha256": sha256(Path(row.hourly_path)),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(repair, anchors.itertuples()))
    (batch / "REFERENCE_METADATA_AUDIT.json").write_text(json.dumps(rows, indent=2) + "\n")


def execute_spatial(raw, runtime, workers):
    from global_spatial_extension import execute_plan

    output = raw / "spatial"
    planned = pd.read_csv(output / "GLOBAL_SPATIAL_EXECUTION_PLAN.csv")
    probes = pd.read_csv(raw / "matrix/probe_runs/PROBE_RUN_MANIFEST.csv")
    args = (planned, probes, raw / "matrix/probe_runs", runtime, output)
    result = execute_plan(*args, workers=workers)
    if not result.relative_engineering_pass.all():
        evidence = audit_annual(
            output / "GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv", runtime,
            output / "solver_stability", workers,
        )
        result = execute_plan(*args, workers=workers, solver_stability_manifest=evidence)
    if not result.completed.all() or not result.relative_engineering_pass.all():
        raise RuntimeError("Fresh spatial execution did not satisfy its engineering checks")


def execute_response(raw, runtime, workers):
    from multi_event_dr_extension import execute_event_plan, attach_planned_event_metadata
    from audit_event_solver_stability import audit_event_solver_stability

    output = raw / "response"
    planned = pd.read_csv(output / "EVENT_RUN_MANIFEST.csv")
    try:
        result = execute_event_plan(planned, output, runtime, workers=workers)
    except RuntimeError:
        evidence_root = output / "solver_stability"
        audit = audit_event_solver_stability(planned, output, runtime, evidence_root, workers=workers)
        if not audit.solver_stability_pass.all():
            raise RuntimeError("Fresh response solver evidence did not pass")
        result = execute_event_plan(
            planned, output, runtime, workers=workers,
            solver_stability_manifest=evidence_root / "EVENT_SOLVER_STABILITY.csv",
        )
    result = attach_planned_event_metadata(result, planned)
    result.to_csv(output / "EVENT_RUN_MANIFEST_COMPLETED.csv", index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["spatial", "response"], required=True)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run = execute_spatial if args.stage == "spatial" else execute_response
    run(args.raw_run_root.resolve(), args.runtime.resolve(), args.workers)


if __name__ == "__main__":
    main()
