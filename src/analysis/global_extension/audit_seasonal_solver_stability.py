"""Audit seasonal controls rejected only for HVAC-iteration warnings.

The audit reruns only clean, completed 40-iteration cases whose sole relative
engineering failure is ``SimHVAC`` maximum iterations.  Each rerun changes
only ``ConvergenceLimits.Maximum HVAC Iterations`` to 100 and is stored apart
from the primary seasonal outputs.  Qualification remains the responsibility
of :mod:`seasonal_command_extension`, which verifies this evidence against the
active input, compact trajectory and error log before applying it.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from global_matrix import run_energyplus, sha256  # noqa: E402
from run_global_matrix import (  # noqa: E402
    relative_engineering_qa,
    set_maximum_hvac_iterations,
    solver_configuration,
    warning_profile,
)


REQUIRED_ARRAYS = (
    "zone_ids",
    "temperatures",
    "people",
    "occupied",
    "month",
    "day",
    "hour",
    "dst",
    "facility_electric_kwh",
    "hvac_electric_kwh",
    "gas_hhv_kwh",
    "facility_gas_hhv_kwh_annual",
)

ENERGY_DIFFERENCE_COLUMNS = (
    "facility_electricity_relative_difference",
    "hvac_electricity_relative_difference",
    "space_heating_gas_relative_difference",
    "facility_electricity_trajectory_relative_l1",
    "hvac_electricity_trajectory_relative_l1",
    "space_heating_gas_trajectory_relative_l1",
    "facility_gas_annual_relative_difference",
    "facility_electric_plus_space_heating_gas_relative_difference",
)


def _truth(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    parsed = str(value).strip().lower()
    if parsed in {"true", "1"}:
        return True
    if parsed in {"false", "0"}:
        return False
    raise ValueError(f"{label} is not a boolean")


def _json_object(value: object, label: str) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _json_list(value: object, label: str) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must be a JSON list")
    return parsed


def _load_compact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    missing = set(REQUIRED_ARRAYS) - set(data)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    return data


def _relative_difference(first: float, second: float) -> float:
    denominator = abs(first)
    if denominator <= 1e-12:
        return 0.0 if abs(second) <= 1e-12 else float("inf")
    return abs(second - first) / denominator


def _trajectory_relative_l1(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape:
        raise ValueError("Solver comparison changed an energy trajectory shape")
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        raise ValueError("Solver comparison found a non-finite energy trajectory")
    denominator = float(np.abs(first_values).sum())
    difference = float(np.abs(second_values - first_values).sum())
    if denominator <= 1e-12:
        return 0.0 if difference <= 1e-12 else float("inf")
    return difference / denominator


def compare_seasonal_trajectories(
    first_path: Path,
    second_path: Path,
    *,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> dict:
    """Compare occupied temperatures and exact purchased-energy streams."""
    if temperature_tolerance_c <= 0 or energy_relative_tolerance <= 0:
        raise ValueError("Stability tolerances must be positive")
    first = _load_compact(first_path)
    second = _load_compact(second_path)
    for name in ("zone_ids", "occupied", "people", "month", "day", "hour", "dst"):
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"Seasonal solver comparison changed {name}")

    occupied = first["occupied"].astype(bool)
    temperature_difference = np.abs(
        first["temperatures"].astype(float)[occupied]
        - second["temperatures"].astype(float)[occupied]
    )
    if temperature_difference.size == 0 or not np.isfinite(
        temperature_difference
    ).all():
        raise ValueError("Occupied temperature comparison is empty or non-finite")

    streams = (
        "facility_electric_kwh",
        "hvac_electric_kwh",
        "gas_hhv_kwh",
    )
    totals = {
        name: _relative_difference(
            float(first[name].astype(float).sum()),
            float(second[name].astype(float).sum()),
        )
        for name in streams
    }
    trajectory_l1 = {
        name: _trajectory_relative_l1(first[name], second[name])
        for name in streams
    }
    first_combined = float(
        first["facility_electric_kwh"].sum() + first["gas_hhv_kwh"].sum()
    )
    second_combined = float(
        second["facility_electric_kwh"].sum() + second["gas_hhv_kwh"].sum()
    )
    combined_difference = _relative_difference(first_combined, second_combined)
    facility_gas_difference = _relative_difference(
        float(np.asarray(first["facility_gas_hhv_kwh_annual"])),
        float(np.asarray(second["facility_gas_hhv_kwh_annual"])),
    )
    maximum_temperature = float(temperature_difference.max())
    stable = bool(
        maximum_temperature <= temperature_tolerance_c
        and all(value <= energy_relative_tolerance for value in totals.values())
        and all(value <= energy_relative_tolerance for value in trajectory_l1.values())
        and combined_difference <= energy_relative_tolerance
        and facility_gas_difference <= energy_relative_tolerance
    )
    return {
        "operative_temperature_max_abs_difference_c": maximum_temperature,
        "operative_temperature_p99_abs_difference_c": float(
            np.quantile(temperature_difference, 0.99)
        ),
        "facility_electricity_relative_difference": totals[
            "facility_electric_kwh"
        ],
        "hvac_electricity_relative_difference": totals["hvac_electric_kwh"],
        "space_heating_gas_relative_difference": totals["gas_hhv_kwh"],
        "facility_electricity_trajectory_relative_l1": trajectory_l1[
            "facility_electric_kwh"
        ],
        "hvac_electricity_trajectory_relative_l1": trajectory_l1[
            "hvac_electric_kwh"
        ],
        "space_heating_gas_trajectory_relative_l1": trajectory_l1["gas_hhv_kwh"],
        "facility_gas_annual_relative_difference": facility_gas_difference,
        "facility_electric_plus_space_heating_gas_relative_difference": (
            combined_difference
        ),
        "trajectory_stable_within_declared_tolerance": stable,
        "declared_temperature_tolerance_c": temperature_tolerance_c,
        "declared_energy_relative_tolerance": energy_relative_tolerance,
    }


def read_seasonal_solver_stability_evidence(
    path: Path | None,
) -> tuple[dict[str, dict], str | None]:
    """Load exact 40-vs-100 seasonal evidence without hiding failed audits."""
    if path is None:
        return {}, None
    frame = pd.read_csv(path)
    required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "seasonal_contrast_fraction",
        "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        *ENERGY_DIFFERENCE_COLUMNS,
        "trajectory_stable_within_declared_tolerance",
        "declared_temperature_tolerance_c",
        "declared_energy_relative_tolerance",
        "higher_iteration_completed",
        "higher_iteration_severe_count",
        "higher_iteration_fatal_occurrences",
        "higher_iteration_relative_engineering_pass",
        "warning_profile_nonworsening",
        "first_treatment_warning_profile",
        "second_treatment_warning_profile",
        "first_input_sha256",
        "first_hourly_sha256",
        "first_error_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(
            f"Seasonal solver-stability evidence lacks {sorted(missing)}"
        )
    if frame.run_id.duplicated().any():
        raise ValueError(
            "Seasonal solver-stability evidence duplicates a run identifier"
        )
    boolean_columns = (
        "trajectory_stable_within_declared_tolerance",
        "higher_iteration_completed",
        "higher_iteration_relative_engineering_pass",
        "warning_profile_nonworsening",
    )
    for column in boolean_columns:
        frame[column] = frame[column].map(lambda value: _truth(value, column))
    numeric_columns = (
        "composition_id",
        "seasonal_contrast_fraction",
        "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        *ENERGY_DIFFERENCE_COLUMNS,
        "declared_temperature_tolerance_c",
        "declared_energy_relative_tolerance",
        "higher_iteration_severe_count",
        "higher_iteration_fatal_occurrences",
    )
    numeric = frame[list(numeric_columns)].apply(pd.to_numeric, errors="raise")
    if np.isnan(numeric.to_numpy()).any():
        raise ValueError("Seasonal solver-stability evidence contains NaN values")
    frame[list(numeric_columns)] = numeric
    finite_columns = [
        "composition_id",
        "seasonal_contrast_fraction",
        "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "declared_temperature_tolerance_c",
        "declared_energy_relative_tolerance",
        "higher_iteration_severe_count",
        "higher_iteration_fatal_occurrences",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy()).all():
        raise ValueError(
            "Seasonal solver-stability identities and limits must be finite"
        )
    differences = frame[[
        "operative_temperature_max_abs_difference_c",
        *ENERGY_DIFFERENCE_COLUMNS,
    ]]
    if (differences < 0).any().any():
        raise ValueError("Seasonal solver-stability differences cannot be negative")
    if (
        (frame.declared_temperature_tolerance_c <= 0).any()
        or (frame.declared_energy_relative_tolerance <= 0).any()
        or (
            frame.second_maximum_hvac_iterations
            <= frame.first_maximum_hvac_iterations
        ).any()
    ):
        raise ValueError("Seasonal solver-stability limits are invalid")
    within = (
        frame.operative_temperature_max_abs_difference_c.le(
            frame.declared_temperature_tolerance_c
        )
        & frame[list(ENERGY_DIFFERENCE_COLUMNS)].le(
            frame.declared_energy_relative_tolerance, axis="index"
        ).all(axis=1)
    )
    if not frame.trajectory_stable_within_declared_tolerance.eq(within).all():
        raise ValueError(
            "Seasonal solver-stability flags contradict their declared limits"
        )
    if (
        (frame.higher_iteration_severe_count < 0).any()
        or (frame.higher_iteration_fatal_occurrences < 0).any()
    ):
        raise ValueError("Seasonal higher-iteration error counts cannot be negative")
    for item in frame.to_dict("records"):
        first = _json_object(
            item["first_treatment_warning_profile"],
            "first_treatment_warning_profile",
        )
        second = _json_object(
            item["second_treatment_warning_profile"],
            "second_treatment_warning_profile",
        )
        nonworsening = all(
            value <= first.get(name, 0) for name, value in second.items()
        )
        if bool(item["warning_profile_nonworsening"]) != nonworsening:
            raise ValueError(
                "Seasonal warning-profile flag contradicts the supplied profiles"
            )
        for column in (
            "first_input_sha256",
            "first_hourly_sha256",
            "first_error_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item[column]).lower()):
                raise ValueError(f"Seasonal solver-stability {column} is invalid")
    return {
        str(row["run_id"]): row for row in frame.to_dict("records")
    }, sha256(path)


def apply_seasonal_solver_stability_evidence(
    relative: dict,
    run_qa: dict,
    evidence: dict | None,
    *,
    current_maximum_hvac_iterations: int,
    active_input_sha256: str,
    active_hourly_sha256: str,
    active_error_sha256: str,
) -> tuple[bool, dict]:
    """Qualify one exact seasonal run only when its audit is verified stable."""
    raw_pass = bool(relative["engineering_pass_relative_to_prototype_baseline"])
    audit = {
        "raw_relative_engineering_pass": raw_pass,
        "seasonal_solver_stability_exception_applied": False,
    }
    if raw_pass or evidence is None:
        return raw_pass, audit
    only_hvac = bool(
        set(relative["new_warning_categories"]) == {"hvac_iterations"}
        and set(relative["critical_warning_count_increases"])
        == {"hvac_iterations"}
    )
    clean = bool(
        run_qa.get("completed")
        and run_qa.get("reported_severe_count") == 0
        and run_qa.get("fatal_occurrences") == 0
        and not run_qa.get("unexplained_recurring_groups")
    )
    if not only_hvac or not clean:
        return False, audit
    first_profile = _json_object(
        evidence["first_treatment_warning_profile"],
        "first_treatment_warning_profile",
    )
    if first_profile != relative["treatment_warning_profile"]:
        raise ValueError(
            "Seasonal stability evidence does not match the active warning profile"
        )
    hashes_match = all(
        str(evidence[name]) == value
        for name, value in {
            "first_input_sha256": active_input_sha256,
            "first_hourly_sha256": active_hourly_sha256,
            "first_error_sha256": active_error_sha256,
        }.items()
    )
    if (
        int(evidence["first_maximum_hvac_iterations"])
        != int(current_maximum_hvac_iterations)
        or not hashes_match
    ):
        raise ValueError(
            "Seasonal stability evidence does not match the active primary run"
        )
    verified = bool(
        evidence["trajectory_stable_within_declared_tolerance"]
        and evidence["higher_iteration_completed"]
        and evidence["warning_profile_nonworsening"]
        and int(evidence["higher_iteration_severe_count"]) == 0
        and int(evidence["higher_iteration_fatal_occurrences"]) == 0
    )
    audit["seasonal_solver_stability_evidence_verified"] = verified
    if not verified:
        return False, audit
    audit.update({
        "seasonal_solver_stability_exception_applied": True,
        "seasonal_solver_stability_higher_iteration_limit": int(
            evidence["second_maximum_hvac_iterations"]
        ),
        "seasonal_solver_stability_temperature_difference_c": float(
            evidence["operative_temperature_max_abs_difference_c"]
        ),
        "seasonal_solver_stability_facility_electricity_relative_difference": (
            float(evidence["facility_electricity_relative_difference"])
        ),
        "seasonal_solver_stability_hvac_electricity_relative_difference": (
            float(evidence["hvac_electricity_relative_difference"])
        ),
        "seasonal_solver_stability_space_heating_gas_relative_difference": (
            float(evidence["space_heating_gas_relative_difference"])
        ),
        "seasonal_solver_stability_higher_iteration_relative_pass": bool(
            evidence["higher_iteration_relative_engineering_pass"]
        ),
    })
    return True, audit


def select_hvac_only_failures(
    runs: pd.DataFrame,
    primary_root: Path,
    *,
    expected_first_iterations: int = 40,
) -> pd.DataFrame:
    """Select exact primary cases whose sole failure is HVAC iterations."""
    required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "seasonal_contrast_fraction",
        "targets_json",
        "input_idf_path",
        "weather_path",
    }
    missing = required - set(runs)
    if missing:
        raise ValueError(f"Seasonal run manifest lacks {sorted(missing)}")
    if runs.run_id.duplicated().any():
        raise ValueError("Seasonal run manifest duplicates run_id")

    selected: list[dict] = []
    unsupported: list[str] = []
    for row in runs.to_dict("records"):
        run_id = str(row["run_id"])
        qa_path = primary_root / "runs" / run_id / "RUN_QA.json"
        compact = primary_root / "hourly" / f"{run_id}.npz"
        error_path = primary_root / "runs" / run_id / "eplusout.err"
        for path in (qa_path, compact, error_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        input_path = Path(row["input_idf_path"])
        weather_path = Path(row["weather_path"])
        expected_hashes = {
            "input_idf_sha256": sha256(input_path),
            "weather_sha256": sha256(weather_path),
            "hourly_sha256": sha256(compact),
        }
        if any(qa.get(name) != value for name, value in expected_hashes.items()):
            raise ValueError(f"Primary seasonal provenance mismatch for {run_id}")
        first_iterations = solver_configuration(
            input_path.read_text(encoding="utf-8", errors="replace")
        )["maximum_hvac_iterations"]
        if int(first_iterations) != int(expected_first_iterations):
            raise ValueError(
                f"Seasonal run {run_id} uses {first_iterations} HVAC iterations, "
                f"not the declared {expected_first_iterations}"
            )
        if _truth(qa.get("relative_engineering_pass"), "relative_engineering_pass"):
            continue
        clean = bool(
            qa.get("completed")
            and qa.get("reported_severe_count") == 0
            and qa.get("fatal_occurrences") == 0
            and not qa.get("unexplained_recurring_groups")
        )
        categories = set(_json_list(
            qa.get("new_warning_categories"), "new_warning_categories"
        ))
        critical = set(_json_object(
            qa.get("critical_warning_count_increases"),
            "critical_warning_count_increases",
        ))
        if clean and categories == {"hvac_iterations"} and critical == {
            "hvac_iterations"
        }:
            selected.append(
                row
                | {
                    "primary_qa_path": str(qa_path.resolve()),
                    "primary_hourly_path": str(compact.resolve()),
                    "primary_error_path": str(error_path.resolve()),
                    "first_maximum_hvac_iterations": int(first_iterations),
                }
            )
        else:
            unsupported.append(run_id)
    if unsupported:
        raise ValueError(
            "Seasonal failures include cases not attributable solely to HVAC "
            f"iterations: {unsupported}"
        )
    return pd.DataFrame(selected)


def _prepare_input(
    source: Path,
    destination: Path,
    *,
    first_iterations: int,
    higher_iterations: int,
) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    current = solver_configuration(text)["maximum_hvac_iterations"]
    if int(current) != int(first_iterations):
        raise ValueError("Primary input no longer matches the declared iteration limit")
    if higher_iterations <= first_iterations:
        raise ValueError("Higher HVAC iteration limit must exceed the primary limit")
    updated = set_maximum_hvac_iterations(text, higher_iterations)
    if set_maximum_hvac_iterations(updated, first_iterations) != text:
        raise AssertionError("Stability audit changed fields beyond the iteration limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated, encoding="utf-8")


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
        if qa.get("completed") and all(
            qa.get(name) == value for name, value in expected.items()
        ):
            return qa
        raise ValueError(f"Seasonal stability cache mismatch for {job['run_id']}")
    zones = list(_json_object(job["row"]["targets_json"], "targets_json")["DJF"])
    return run_energyplus(
        job["idf"],
        job["weather"],
        runtime,
        job["run_folder"],
        job["compact"],
        zones,
        cleanup_raw=True,
    )


def audit_seasonal_solver_stability(
    runs: pd.DataFrame,
    runtime: Path,
    primary_root: Path,
    output_root: Path,
    *,
    first_iterations: int = 40,
    higher_iterations: int = 100,
    workers: int = 4,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> pd.DataFrame:
    """Rerun and compare every eligible seasonal solver-warning case."""
    if workers < 1:
        raise ValueError("At least one worker is required")
    selected = select_hvac_only_failures(
        runs, primary_root, expected_first_iterations=first_iterations
    )
    if selected.empty:
        raise ValueError("No HVAC-only seasonal failures require a stability audit")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for row in selected.to_dict("records"):
        run_id = str(row["run_id"])
        idf = output_root / "inputs" / f"{run_id}_{higher_iterations}.idf"
        _prepare_input(
            Path(row["input_idf_path"]),
            idf,
            first_iterations=first_iterations,
            higher_iterations=higher_iterations,
        )
        jobs.append({
            "row": row,
            "run_id": run_id,
            "idf": idf,
            "weather": Path(row["weather_path"]),
            "run_folder": output_root / "runs" / run_id,
            "compact": output_root / "hourly" / f"{run_id}.npz",
        })

    completed: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_run_or_reuse, job, runtime): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            qa = future.result()
            completed.append((job, qa))
            print(json.dumps({
                "run_id": job["run_id"], "completed": qa.get("completed")
            }), flush=True)

    rows: list[dict] = []
    for job, qa in completed:
        original = job["row"]
        primary_qa = json.loads(
            Path(original["primary_qa_path"]).read_text(encoding="utf-8")
        )
        baseline_profile = _json_object(
            primary_qa["baseline_warning_profile"], "baseline_warning_profile"
        )
        first_profile = _json_object(
            primary_qa["treatment_warning_profile"], "treatment_warning_profile"
        )
        second_profile = warning_profile(job["run_folder"] / "eplusout.err")
        higher_relative = relative_engineering_qa(
            qa, second_profile, baseline_profile
        )
        comparison = compare_seasonal_trajectories(
            Path(original["primary_hourly_path"]),
            job["compact"],
            temperature_tolerance_c=temperature_tolerance_c,
            energy_relative_tolerance=energy_relative_tolerance,
        )
        rows.append({
            "run_id": job["run_id"],
            "climate": original["climate"],
            "composition_id": int(original["composition_id"]),
            "control": original["control"],
            "seasonal_contrast_fraction": float(
                original["seasonal_contrast_fraction"]
            ),
            "first_maximum_hvac_iterations": first_iterations,
            "second_maximum_hvac_iterations": higher_iterations,
            **comparison,
            "higher_iteration_completed": bool(qa.get("completed")),
            "higher_iteration_severe_count": qa.get("reported_severe_count"),
            "higher_iteration_fatal_occurrences": qa.get("fatal_occurrences"),
            "higher_iteration_relative_engineering_pass": higher_relative[
                "engineering_pass_relative_to_prototype_baseline"
            ],
            "warning_profile_nonworsening": all(
                value <= first_profile.get(name, 0)
                for name, value in second_profile.items()
            ),
            "first_treatment_warning_profile": json.dumps(
                first_profile, sort_keys=True
            ),
            "second_treatment_warning_profile": json.dumps(
                second_profile, sort_keys=True
            ),
            "first_input_sha256": sha256(Path(original["input_idf_path"])),
            "first_hourly_sha256": sha256(
                Path(original["primary_hourly_path"])
            ),
            "first_error_sha256": sha256(Path(original["primary_error_path"])),
            "second_input_sha256": sha256(job["idf"]),
            "second_hourly_sha256": sha256(job["compact"]),
            "second_qa_path": str(
                (job["run_folder"] / "RUN_QA.json").resolve()
            ),
            "second_hourly_path": str(job["compact"].resolve()),
        })
    result = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "seasonal_contrast_fraction"],
        kind="stable",
    ).reset_index(drop=True)
    evidence_path = output_root / "SEASONAL_SOLVER_STABILITY.csv"
    result.to_csv(evidence_path, index=False)
    verified = (
        result.trajectory_stable_within_declared_tolerance
        & result.higher_iteration_completed
        & result.warning_profile_nonworsening
        & result.higher_iteration_severe_count.eq(0)
        & result.higher_iteration_fatal_occurrences.eq(0)
    )
    audit = {
        "selected_runs": len(result),
        "verified_stable_runs": int(verified.sum()),
        "all_verified": bool(verified.all()),
        "primary_outputs_modified": False,
        "first_maximum_hvac_iterations": first_iterations,
        "second_maximum_hvac_iterations": higher_iterations,
        "evidence_path": str(evidence_path.resolve()),
        "evidence_sha256": sha256(evidence_path),
    }
    (output_root / "SEASONAL_SOLVER_STABILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--first-iterations", type=int, default=40)
    parser.add_argument("--higher-iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature-tolerance-c", type=float, default=0.01)
    parser.add_argument("--energy-relative-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    result = audit_seasonal_solver_stability(
        pd.read_csv(args.run_manifest),
        args.runtime.resolve(),
        args.primary_root.resolve(),
        args.output_root.resolve(),
        first_iterations=args.first_iterations,
        higher_iterations=args.higher_iterations,
        workers=args.workers,
        temperature_tolerance_c=args.temperature_tolerance_c,
        energy_relative_tolerance=args.energy_relative_tolerance,
    )
    verified = (
        result.trajectory_stable_within_declared_tolerance
        & result.higher_iteration_completed
        & result.warning_profile_nonworsening
        & result.higher_iteration_severe_count.eq(0)
        & result.higher_iteration_fatal_occurrences.eq(0)
    )
    if not verified.all():
        raise RuntimeError(
            "Seasonal solver-stability audit did not verify every selected run"
        )


if __name__ == "__main__":
    main()
