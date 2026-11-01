"""Regenerate seasonal commands and scores from the fresh raw-input run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import seasonal_command_extension as seasonal
from audit_seasonal_solver_stability import audit_seasonal_solver_stability
from seasonal_facility_energy import (
    extract_offset_zero_monthly_gas,
    enrich_seasonal_facility_energy,
)
from global_matrix import sha256


def run(raw, runtime, workers):
    matrix = raw / "matrix"
    output = raw / "seasonal"
    output.mkdir(parents=True, exist_ok=True)
    ashrae = pd.read_csv(raw / "demand/ASHRAE_RECORDS.csv.gz", low_memory=False)
    ashrae["centre_c"] = ashrae.reference_g050_c
    china = pd.read_csv(raw / "demand/CTCD_RECORDS.csv.gz", low_memory=False)
    summary, effects = seasonal.derive_source_contrasts(ashrae, china)
    contrasts = output / "SEASONAL_SOURCE_CONTRASTS.csv"
    summary.to_csv(contrasts, index=False)
    effects.to_csv(output / "SEASONAL_PLACE_EFFECTS.csv", index=False)
    climate = Path(__file__).with_name("GLOBAL_CLIMATES.json")
    runs, _ = seasonal.build_plan(
        matrix / "models/MODEL_MANIFEST.json",
        matrix / "GLOBAL_PROFILES.csv",
        matrix / "probe_runs",
        climate,
        contrasts,
        output,
    )
    probes = pd.read_csv(matrix / "probe_runs/PROBE_RUN_MANIFEST.csv")
    args = (runs, probes, matrix / "probe_runs", output, runtime)
    try:
        seasonal.execute_plan(*args, workers=workers)
    except RuntimeError:
        evidence_root = output / "solver_stability"
        audit = audit_seasonal_solver_stability(
            runs, runtime, output, evidence_root, workers=workers
        )
        if not (
            audit.trajectory_stable_within_declared_tolerance
            & audit.higher_iteration_completed
            & audit.warning_profile_nonworsening
            & audit.higher_iteration_severe_count.eq(0)
            & audit.higher_iteration_fatal_occurrences.eq(0)
        ).all():
            raise RuntimeError("Fresh seasonal solver comparison did not pass")
        seasonal.execute_plan(
            *args,
            workers=workers,
            solver_stability_manifest=evidence_root / "SEASONAL_SOLVER_STABILITY.csv",
        )
    destination = raw / "results/seasonal_command/final"
    seasonal.score_completed_plan(
        output / "SEASONAL_COMMAND_RUN_MANIFEST_COMPLETED.csv",
        matrix / "GLOBAL_PROFILES.csv",
        climate,
        contrasts,
        matrix / "probe_runs/PROBE_RUN_MANIFEST.csv",
        destination,
    )
    monthly, extraction = extract_offset_zero_monthly_gas(
        matrix / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv",
        sql_search_roots=(matrix,),
        workers=workers,
    )
    metrics = destination / "SEASONAL_COMMAND_SEASON_METRICS.csv"
    blocks, annual, reconstruction = enrich_seasonal_facility_energy(pd.read_csv(metrics), monthly)
    facility = raw / "results/seasonal_facility_energy_final"
    facility.mkdir(parents=True, exist_ok=True)
    files = {
        "OFFSET_ZERO_MONTHLY_NATURAL_GAS.csv": monthly,
        "SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv": blocks,
        "SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv": annual,
    }
    for name, frame in files.items():
        frame.to_csv(facility / name, index=False)
    audit = {
        "status": "PASS",
        "source_seasonal_metrics_sha256": sha256(metrics),
        "extraction": extraction,
        "reconstruction": reconstruction,
        "outputs": {
            name: {"rows": len(frame), "sha256": sha256(facility / name)}
            for name, frame in files.items()
        },
    }
    (facility / "SEASONAL_FACILITY_ENERGY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    for name in ["SEASONAL_SOURCE_CONTRASTS.csv", "SEASONAL_PLACE_EFFECTS.csv"]:
        (destination.parent / name).write_bytes((output / name).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.raw_run_root.resolve(), args.runtime.resolve(), args.workers)


if __name__ == "__main__":
    main()
