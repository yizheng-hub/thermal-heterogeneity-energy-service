"""Rerun annual controls with solver warnings at a higher iteration limit.

Only completed annual cases whose sole relative engineering failure is a
``SimHVAC`` maximum-iteration warning are eligible.  The higher-iteration
trajectories are written to a separate audit directory and compared with the
original annual results; this command never changes the primary run folders or
their quality-assurance decisions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
GLOBAL = HERE.parent
sys.path.insert(0, str(GLOBAL))

from global_matrix import run_energyplus, sha256  # noqa: E402
from run_global_matrix import (  # noqa: E402
    ZONES,
    compare_solver_warning_categories,
    relative_engineering_qa,
    set_maximum_hvac_iterations,
    solver_configuration,
    warning_profile,
)


REQUIRED_ARRAYS = (
    "zone_ids", "temperatures", "people", "occupied",
    "hvac_electric_kwh", "gas_hhv_kwh", "facility_electric_kwh",
    "facility_gas_hhv_kwh_annual",
)


def _json_object(value: object, label: str) -> dict:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _json_list(value: object, label: str) -> list:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a JSON list")
    return parsed


def _truth(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{label} is not a boolean")


def select_hvac_only_failures(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return completed annual rows rejected solely for HVAC iterations."""
    required = {
        "run_id", "climate", "composition_id", "control", "offset_c",
        "completed", "relative_engineering_pass", "new_warning_categories",
        "critical_warning_count_increases", "baseline_warning_profile",
        "treatment_warning_profile", "input_idf_path", "weather_path",
        "hourly_path", "qa_path", "maximum_hvac_iterations",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Annual manifest lacks {sorted(missing)}")
    if manifest.run_id.duplicated().any():
        raise ValueError("Annual manifest duplicates a run identifier")

    selected: list[dict] = []
    unsupported: list[str] = []
    for row in manifest.to_dict("records"):
        if _truth(row["relative_engineering_pass"], "relative_engineering_pass"):
            continue
        if not _truth(row["completed"], "completed"):
            unsupported.append(str(row["run_id"]))
            continue
        qa_path = Path(row["qa_path"])
        if not qa_path.is_file():
            raise FileNotFoundError(qa_path)
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        clean_completion = bool(
            qa.get("completed")
            and qa.get("reported_severe_count") == 0
            and qa.get("fatal_occurrences") == 0
            and not qa.get("unexplained_recurring_groups")
        )
        categories = set(_json_list(
            row["new_warning_categories"], "new_warning_categories"
        ))
        critical = set(_json_object(
            row["critical_warning_count_increases"],
            "critical_warning_count_increases",
        ))
        if (
            clean_completion
            and categories == {"hvac_iterations"}
            and critical <= {"hvac_iterations"}
        ):
            selected.append(row)
        else:
            unsupported.append(str(row["run_id"]))
    if unsupported:
        raise ValueError(
            "Annual failures include cases not attributable solely to HVAC "
            f"iterations: {unsupported}"
        )
    return pd.DataFrame(selected, columns=manifest.columns)


def _load_compact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    missing = set(REQUIRED_ARRAYS) - set(data)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    return data


def _relative_difference(first: float, second: float) -> float:
    if abs(first) <= 1e-12:
        return 0.0 if abs(second) <= 1e-12 else float("inf")
    return abs(second - first) / abs(first)


def compare_annual_trajectories(
    first_path: Path,
    second_path: Path,
    *,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> dict:
    """Compare occupied temperatures, HVAC energy and complete facility energy."""
    if temperature_tolerance_c <= 0 or energy_relative_tolerance <= 0:
        raise ValueError("Stability tolerances must be positive")
    first = _load_compact(first_path)
    second = _load_compact(second_path)
    for name in ("zone_ids", "occupied", "people"):
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"Annual solver comparison changed {name}")
    occupied = first["occupied"].astype(bool)
    difference = np.abs(
        first["temperatures"].astype(float)[occupied]
        - second["temperatures"].astype(float)[occupied]
    )
    if difference.size == 0 or not np.isfinite(difference).all():
        raise ValueError("Occupied temperature comparison is empty or non-finite")

    first_electricity = float(first["hvac_electric_kwh"].astype(float).sum())
    second_electricity = float(second["hvac_electric_kwh"].astype(float).sum())
    first_gas = float(first["gas_hhv_kwh"].astype(float).sum())
    second_gas = float(second["gas_hhv_kwh"].astype(float).sum())
    electricity_difference = _relative_difference(
        first_electricity, second_electricity
    )
    gas_difference = _relative_difference(first_gas, second_gas)
    total_difference = _relative_difference(
        first_electricity + first_gas,
        second_electricity + second_gas,
    )
    first_facility_electricity = float(
        first["facility_electric_kwh"].astype(float).sum()
    )
    second_facility_electricity = float(
        second["facility_electric_kwh"].astype(float).sum()
    )
    first_facility_gas = float(
        first["facility_gas_hhv_kwh_annual"].astype(float).sum()
    )
    second_facility_gas = float(
        second["facility_gas_hhv_kwh_annual"].astype(float).sum()
    )
    facility_electricity_difference = _relative_difference(
        first_facility_electricity,
        second_facility_electricity,
    )
    facility_gas_difference = _relative_difference(
        first_facility_gas,
        second_facility_gas,
    )
    facility_total_difference = _relative_difference(
        first_facility_electricity + first_facility_gas,
        second_facility_electricity + second_facility_gas,
    )
    maximum_temperature = float(difference.max())
    stable = bool(
        maximum_temperature <= temperature_tolerance_c
        and electricity_difference <= energy_relative_tolerance
        and gas_difference <= energy_relative_tolerance
        and total_difference <= energy_relative_tolerance
        and facility_total_difference <= energy_relative_tolerance
    )
    return {
        "operative_temperature_max_abs_difference_c": maximum_temperature,
        "operative_temperature_p99_abs_difference_c": float(
            np.quantile(difference, 0.99)
        ),
        "hvac_electricity_relative_difference": electricity_difference,
        "gas_hhv_relative_difference": gas_difference,
        "total_purchased_energy_relative_difference": total_difference,
        "facility_electricity_relative_difference": (
            facility_electricity_difference
        ),
        "facility_gas_hhv_relative_difference": facility_gas_difference,
        "facility_purchased_energy_relative_difference": (
            facility_total_difference
        ),
        "trajectory_stable_within_declared_tolerance": stable,
        "declared_temperature_tolerance_c": temperature_tolerance_c,
        "declared_energy_relative_tolerance": energy_relative_tolerance,
    }


def _prepare_input(source: Path, destination: Path, higher_iterations: int) -> int:
    text = source.read_text(encoding="utf-8", errors="replace")
    configuration = solver_configuration(text)
    original = configuration["maximum_hvac_iterations"]
    if original is None or higher_iterations <= int(original):
        raise ValueError("Higher HVAC iteration limit must exceed the annual input")
    updated = set_maximum_hvac_iterations(text, higher_iterations)
    if set_maximum_hvac_iterations(updated, int(original)) != text:
        raise AssertionError("Solver audit changed fields beyond the iteration limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated, encoding="utf-8")
    return int(original)


def _run_or_reuse(job: dict, runtime: Path) -> dict:
    qa_path = job["run_folder"] / "RUN_QA.json"
    if qa_path.is_file() and job["compact"].is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        expected = {
            "input_idf_sha256": sha256(job["idf"]),
            "weather_sha256": sha256(job["weather"]),
            "runtime_sha256": sha256(runtime),
            "hourly_sha256": sha256(job["compact"]),
        }
        if qa.get("completed") and all(qa.get(key) == value for key, value in expected.items()):
            return qa
        raise ValueError(f"Solver-audit cache mismatch for {job['run_id']}")
    return run_energyplus(
        job["idf"], job["weather"], runtime, job["run_folder"],
        job["compact"], ZONES, cleanup_raw=True,
    )


def audit_annual_solver_stability(
    manifest: pd.DataFrame,
    runtime: Path,
    output_root: Path,
    *,
    higher_iterations: int = 100,
    workers: int = 6,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> pd.DataFrame:
    """Rerun and compare every eligible annual solver-warning case."""
    selected = select_hvac_only_failures(manifest)
    if selected.empty:
        raise ValueError("No HVAC-only annual failures require a stability audit")
    if workers < 1:
        raise ValueError("At least one worker is required")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for row in selected.to_dict("records"):
        run_id = str(row["run_id"])
        source = Path(row["input_idf_path"])
        idf = output_root / "inputs" / f"{run_id}_{higher_iterations}.idf"
        original_iterations = _prepare_input(source, idf, higher_iterations)
        jobs.append({
            "row": row,
            "run_id": run_id,
            "original_iterations": original_iterations,
            "idf": idf,
            "weather": Path(row["weather_path"]),
            "run_folder": output_root / "runs" / run_id,
            "compact": output_root / "hourly" / f"{run_id}.npz",
        })

    records: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_run_or_reuse, job, runtime): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            records.append((job, future.result()))
            print(json.dumps({
                "run_id": job["run_id"],
                "completed": records[-1][1].get("completed"),
            }), flush=True)

    rows: list[dict] = []
    for job, qa in records:
        original = job["row"]
        baseline_profile = _json_object(
            original["baseline_warning_profile"], "baseline_warning_profile"
        )
        higher_profile = warning_profile(job["run_folder"] / "eplusout.err")
        relative = relative_engineering_qa(qa, higher_profile, baseline_profile)
        comparison = compare_annual_trajectories(
            Path(original["hourly_path"]), job["compact"],
            temperature_tolerance_c=temperature_tolerance_c,
            energy_relative_tolerance=energy_relative_tolerance,
        )
        lower_profile = _json_object(
            original["treatment_warning_profile"], "treatment_warning_profile"
        )
        warning_comparison = compare_solver_warning_categories(
            lower_profile, higher_profile
        )
        solver_stability_pass = bool(
            comparison["trajectory_stable_within_declared_tolerance"]
            and qa.get("completed")
            and int(qa.get("reported_severe_count") or 0) == 0
            and int(qa.get("fatal_occurrences") or 0) == 0
            and warning_comparison["warning_categories_stable"]
        )
        rows.append({
            "run_id": job["run_id"],
            "climate": original["climate"],
            "composition_id": int(original["composition_id"]),
            "control": original["control"],
            "offset_c": float(original["offset_c"]),
            "first_maximum_hvac_iterations": job["original_iterations"],
            "second_maximum_hvac_iterations": higher_iterations,
            **comparison,
            "higher_iteration_completed": bool(qa.get("completed")),
            "higher_iteration_severe_count": qa.get("reported_severe_count"),
            "higher_iteration_fatal_occurrences": qa.get("fatal_occurrences"),
            "higher_iteration_relative_engineering_pass": relative[
                "engineering_pass_relative_to_prototype_baseline"
            ],
            "first_treatment_warning_profile": json.dumps(lower_profile, sort_keys=True),
            "second_treatment_warning_profile": json.dumps(higher_profile, sort_keys=True),
            "warning_profile_nonworsening": warning_comparison[
                "warning_profile_nonworsening"
            ],
            "warning_categories_stable": warning_comparison[
                "warning_categories_stable"
            ],
            "higher_iteration_new_warning_categories": json.dumps(
                warning_comparison["higher_iteration_new_warning_categories"],
                sort_keys=True,
            ),
            "warning_count_drift": json.dumps(
                warning_comparison["warning_count_drift"], sort_keys=True
            ),
            "solver_stability_pass": solver_stability_pass,
            "first_hourly_sha256": sha256(Path(original["hourly_path"])),
            "first_input_sha256": sha256(Path(original["input_idf_path"])),
            "second_hourly_sha256": sha256(job["compact"]),
            "second_input_sha256": sha256(job["idf"]),
            "second_qa_path": str((job["run_folder"] / "RUN_QA.json").resolve()),
            "second_hourly_path": str(job["compact"].resolve()),
        })
    result = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "offset_c"], kind="stable"
    ).reset_index(drop=True)
    evidence_path = output_root / "ANNUAL_SOLVER_STABILITY.csv"
    result.to_csv(evidence_path, index=False)
    audit = {
        "selected_runs": len(result),
        "completed_higher_iteration_runs": int(result.higher_iteration_completed.sum()),
        "stable_trajectories": int(
            result.trajectory_stable_within_declared_tolerance.sum()
        ),
        "nonworsening_warning_profiles": int(result.warning_profile_nonworsening.sum()),
        "stable_warning_categories": int(result.warning_categories_stable.sum()),
        "solver_stability_passes": int(result.solver_stability_pass.sum()),
        "maximum_occupied_temperature_difference_c": float(
            result.operative_temperature_max_abs_difference_c.max()
        ),
        "maximum_facility_purchased_energy_relative_difference": float(
            result.facility_purchased_energy_relative_difference.max()
        ),
        "declared_temperature_tolerance_c": temperature_tolerance_c,
        "declared_energy_relative_tolerance": energy_relative_tolerance,
        "stability_rule": (
            "completed; zero severe/fatal; no warning category introduced at the "
            "higher iteration limit; temperature and energy trajectories within "
            "the declared tolerances; warning-count drift retained as a diagnostic"
        ),
        "all_completed": bool(result.higher_iteration_completed.all()),
        "all_stable": bool(result.trajectory_stable_within_declared_tolerance.all()),
        "all_zero_severe": bool(result.higher_iteration_severe_count.eq(0).all()),
        "all_zero_fatal": bool(result.higher_iteration_fatal_occurrences.eq(0).all()),
        "all_warning_categories_stable": bool(
            result.warning_categories_stable.all()
        ),
        "evidence_sha256": sha256(evidence_path),
        "primary_outputs_modified": False,
    }
    (output_root / "ANNUAL_SOLVER_STABILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--higher-iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--temperature-tolerance-c", type=float, default=0.01)
    parser.add_argument("--energy-relative-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    result = audit_annual_solver_stability(
        pd.read_csv(args.annual_manifest), args.runtime.resolve(),
        args.output_root.resolve(), higher_iterations=args.higher_iterations,
        workers=args.workers,
        temperature_tolerance_c=args.temperature_tolerance_c,
        energy_relative_tolerance=args.energy_relative_tolerance,
    )
    if (
        not result.higher_iteration_completed.all()
        or not result["trajectory_stable_within_declared_tolerance"].all()
        or not result.solver_stability_pass.all()
    ):
        raise RuntimeError("Annual solver-stability audit did not pass every selected run")


if __name__ == "__main__":
    main()
