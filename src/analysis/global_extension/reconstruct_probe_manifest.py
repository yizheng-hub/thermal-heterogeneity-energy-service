"""Reconstruct a missing probe manifest from immutable run evidence.

The utility does not read or derive energy or thermal-service results.  It
links each completed run to its declared climate and uniform thermostat target
by matching the stored input IDF against inputs regenerated from the prepared
model.  The output path is explicit and must not already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
GLOBAL = HERE.parent
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(GLOBAL))
sys.path.insert(0, str(ENERGY_SERVICE))

from generate_model_inputs import annual_input, signature  # noqa: E402
from global_matrix import PROBE_TARGETS_C, parse_idf, sha256  # noqa: E402
from run_global_matrix import relative_engineering_qa, warning_profile  # noqa: E402


EXPECTED_CLIMATES = 9
EXPECTED_RUNS = EXPECTED_CLIMATES * len(PROBE_TARGETS_C)
REQUIRED_COMPACT_ARRAYS = {
    "zone_ids",
    "temperatures",
    "people",
    "occupied",
    "hvac_electric_kwh",
    "gas_hhv_kwh",
}


def _solver_iterations(text: str) -> int:
    rows = [row for row in parse_idf(text) if row[0].lower() == "convergencelimits"]
    if len(rows) != 1 or len(rows[0]) < 3:
        raise ValueError("Expected one complete ConvergenceLimits object")
    return int(float(rows[0][2]))


def _controlled_zones(text: str) -> tuple[str, ...]:
    rows = [
        row[2]
        for row in parse_idf(text)
        if row[0].lower() == "zonecontrol:thermostat" and len(row) > 5
    ]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("Prepared model has missing or duplicate controlled zones")
    return tuple(rows)


def _expected_signatures(
    model_manifest: dict[str, dict],
    targets_c: Sequence[float],
    expected_solver_iterations: int,
) -> tuple[dict[str, tuple[str, float]], dict[str, str]]:
    mapping: dict[str, tuple[str, float]] = {}
    weather_to_climate: dict[str, str] = {}
    for climate, cfg in model_manifest.items():
        model = Path(cfg["prepared_idf"])
        if not model.is_file() or sha256(model) != cfg["prepared_idf_sha256"]:
            raise ValueError(f"Prepared-model evidence changed for {climate}")
        text = model.read_text(encoding="utf-8", errors="replace")
        if _solver_iterations(text) != expected_solver_iterations:
            raise ValueError(
                f"{climate} uses {_solver_iterations(text)} HVAC iterations; "
                f"expected {expected_solver_iterations}"
            )
        weather_hash = str(cfg["weather_sha256"])
        if weather_hash in weather_to_climate:
            raise ValueError("Weather hashes must identify one climate each")
        weather_to_climate[weather_hash] = climate
        zones = _controlled_zones(text)
        half_deadband = float(cfg["half_deadband_c"])
        for target in targets_c:
            generated = annual_input(
                text,
                dict.fromkeys(zones, float(target)),
                half_deadband,
            )
            key = signature(generated)
            if key in mapping:
                raise ValueError("Expected probe signatures are not unique")
            mapping[key] = (climate, float(target))
    return mapping, weather_to_climate


def _validate_compact(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"Compact trajectory evidence changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        missing = REQUIRED_COMPACT_ARRAYS - set(saved.files)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")


def reconstruct_probe_manifest(
    batch_root: Path,
    *,
    targets_c: Sequence[float] = PROBE_TARGETS_C,
    expected_climates: int = EXPECTED_CLIMATES,
    expected_solver_iterations: int = 20,
) -> pd.DataFrame:
    """Return a validated manifest without modifying the source batch."""
    batch_root = batch_root.resolve()
    model_manifest_path = batch_root / "models" / "MODEL_MANIFEST.json"
    probe_root = batch_root / "probe_runs"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if len(model_manifest) != expected_climates:
        raise ValueError(
            f"Expected {expected_climates} climates, found {len(model_manifest)}"
        )
    expected, weather_to_climate = _expected_signatures(
        model_manifest, targets_c, expected_solver_iterations
    )
    expected_runs = expected_climates * len(targets_c)
    run_folders = sorted(path for path in (probe_root / "runs").iterdir() if path.is_dir())
    if len(run_folders) != expected_runs:
        raise ValueError(f"Expected {expected_runs} run folders, found {len(run_folders)}")

    preliminary: list[dict] = []
    for run_folder in run_folders:
        run_id = run_folder.name
        input_idf = probe_root / "inputs" / f"{run_id}.idf"
        compact = probe_root / "hourly" / f"{run_id}.npz"
        qa_path = run_folder / "RUN_QA.json"
        error_path = run_folder / "eplusout.err"
        if not input_idf.is_file() or not qa_path.is_file() or not error_path.is_file():
            raise ValueError(f"Incomplete run evidence: {run_id}")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if not qa.get("completed") or qa.get("reported_severe_count") != 0:
            raise ValueError(f"Run was not completed without severe errors: {run_id}")
        if qa.get("fatal_occurrences") != 0:
            raise ValueError(f"Run contains a fatal error: {run_id}")
        input_hash = sha256(input_idf)
        if input_hash != qa.get("input_idf_sha256"):
            raise ValueError(f"Input hash does not match RUN_QA.json: {run_id}")
        pair = expected.get(signature(input_idf.read_text(encoding="utf-8", errors="replace")))
        if pair is None:
            raise ValueError(f"Input is not one of the declared probes: {run_id}")
        climate, target_c = pair
        if weather_to_climate.get(qa.get("weather_sha256")) != climate:
            raise ValueError(f"Weather and prepared model identify different climates: {run_id}")
        hourly_hash = qa.get("hourly_sha256")
        if not isinstance(hourly_hash, str):
            raise ValueError(f"RUN_QA.json lacks a compact-output hash: {run_id}")
        _validate_compact(compact, hourly_hash)
        preliminary.append({
            "run_id": run_id,
            "climate": climate,
            "target_c": target_c,
            "completed": True,
            "maximum_hvac_iterations": expected_solver_iterations,
            "run_qa": qa,
            "run_folder": run_folder,
            "input_sha256": input_hash,
            "hourly_path": str(compact.resolve()),
            "hourly_sha256": hourly_hash,
        })

    keys = {(row["climate"], row["target_c"]) for row in preliminary}
    if len(keys) != expected_runs or keys != set(expected.values()):
        raise ValueError("Runs do not form the complete unique climate-target matrix")
    baseline_profiles = {
        row["climate"]: warning_profile(row["run_folder"] / "eplusout.err")
        for row in preliminary
        if row["target_c"] == 25.0
    }
    if len(baseline_profiles) != expected_climates:
        raise ValueError("Each climate needs one 25 C engineering baseline")

    rows = []
    for row in preliminary:
        treatment = warning_profile(row["run_folder"] / "eplusout.err")
        relative = relative_engineering_qa(
            row["run_qa"], treatment, baseline_profiles[row["climate"]]
        )
        rows.append({
            "run_id": row["run_id"],
            "climate": row["climate"],
            "target_c": row["target_c"],
            "completed": row["completed"],
            "maximum_hvac_iterations": row["maximum_hvac_iterations"],
            "relative_engineering_pass": relative[
                "engineering_pass_relative_to_prototype_baseline"
            ],
            "new_warning_categories": json.dumps(relative["new_warning_categories"]),
            "warning_count_increases": json.dumps(relative["warning_count_increases"]),
            "critical_warning_count_increases": json.dumps(
                relative["critical_warning_count_increases"]
            ),
            "inherited_environmental_warning_count_drift": json.dumps(
                relative["inherited_environmental_warning_count_drift"]
            ),
            "baseline_warning_profile": json.dumps(relative["baseline_warning_profile"]),
            "treatment_warning_profile": json.dumps(
                relative["treatment_warning_profile"]
            ),
            "input_sha256": row["input_sha256"],
            "hourly_path": row["hourly_path"],
            "hourly_sha256": row["hourly_sha256"],
        })
    return pd.DataFrame(rows).sort_values(
        ["climate", "target_c"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reconstruct_probe_manifest(args.batch_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        result.to_csv(stream, index=False)


if __name__ == "__main__":
    main()
