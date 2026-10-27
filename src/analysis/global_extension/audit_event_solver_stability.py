"""Audit event actions with HVAC iteration warnings at a higher solver limit.

Only completed actions whose sole relative engineering failure is an increase
in ``SimHVAC`` maximum-iteration warnings are rerun.  The audit changes only
the EnergyPlus HVAC iteration limit, writes separate evidence, and never
replaces the primary event trajectories.
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
    compare_solver_warning_categories,
    relative_engineering_qa,
    set_maximum_hvac_iterations,
    solver_configuration,
    warning_profile,
)


REQUIRED_ARRAYS = (
    "zone_ids", "temperatures", "people", "occupied",
    "hvac_electric_kwh", "gas_hhv_kwh",
)


def _load_compact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    missing = set(REQUIRED_ARRAYS) - set(data)
    if missing:
        raise ValueError(f"Event compact lacks {sorted(missing)}: {path}")
    return data


def _relative_difference(first: float, second: float) -> float:
    if abs(first) <= 1e-12:
        return 0.0 if abs(second) <= 1e-12 else float("inf")
    return abs(second - first) / abs(first)


def compare_event_trajectories(
    first_path: Path,
    second_path: Path,
    *,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> dict[str, object]:
    """Compare occupied temperatures and purchased energy for one action."""
    if temperature_tolerance_c <= 0 or energy_relative_tolerance <= 0:
        raise ValueError("Stability tolerances must be positive")
    first = _load_compact(first_path)
    second = _load_compact(second_path)
    for name in ("zone_ids", "occupied", "people"):
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"Event solver comparison changed {name}")
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
    first_has_facility = "facility_electric_kwh" in first
    second_has_facility = "facility_electric_kwh" in second
    if first_has_facility != second_has_facility:
        raise ValueError(
            "Facility electricity is present in only one event trajectory"
        )
    facility_difference = 0.0
    if first_has_facility:
        facility_difference = _relative_difference(
            float(first["facility_electric_kwh"].astype(float).sum()),
            float(second["facility_electric_kwh"].astype(float).sum()),
        )
    maximum_temperature = float(difference.max())
    stable = bool(
        maximum_temperature <= temperature_tolerance_c
        and electricity_difference <= energy_relative_tolerance
        and gas_difference <= energy_relative_tolerance
        and total_difference <= energy_relative_tolerance
        and (
            not first_has_facility
            or facility_difference <= energy_relative_tolerance
        )
    )
    return {
        "operative_temperature_max_abs_difference_c": maximum_temperature,
        "operative_temperature_p99_abs_difference_c": float(
            np.quantile(difference, 0.99)
        ),
        "hvac_electricity_relative_difference": electricity_difference,
        "gas_hhv_relative_difference": gas_difference,
        "total_purchased_energy_relative_difference": total_difference,
        "facility_electricity_compared": first_has_facility,
        "facility_electricity_relative_difference": facility_difference,
        "trajectory_stable_within_declared_tolerance": stable,
        "declared_temperature_tolerance_c": temperature_tolerance_c,
        "declared_energy_relative_tolerance": energy_relative_tolerance,
    }


def _load_action_summaries(
    event_plan: pd.DataFrame, execution_root: Path
) -> list[tuple[dict, dict]]:
    if event_plan.run_id.duplicated().any():
        raise ValueError("Event plan duplicates run_id")
    records: list[tuple[dict, dict]] = []
    for plan in event_plan.to_dict("records"):
        run_id = str(plan["run_id"])
        path = execution_root / "summaries" / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Event action summary is missing; finish the primary batch first: {path}"
            )
        summary = json.loads(path.read_text(encoding="utf-8"))
        if str(summary.get("run_id")) != run_id:
            raise ValueError(f"Event action summary has the wrong run_id: {path}")
        records.append((plan, summary))
    return records


def select_event_hvac_only_failures(
    event_plan: pd.DataFrame, execution_root: Path
) -> list[tuple[dict, dict]]:
    """Select completed actions rejected solely for HVAC iteration warnings."""
    required = {
        "run_id", "climate", "composition_id", "control", "action_id",
        "input_idf_path", "weather_path", "zones_json",
    }
    missing = required - set(event_plan)
    if missing:
        raise ValueError(f"Event plan lacks {sorted(missing)}")
    selected: list[tuple[dict, dict]] = []
    unsupported: list[str] = []
    for plan, summary in _load_action_summaries(event_plan, execution_root):
        raw_pass = bool(
            summary.get(
                "raw_baseline_relative_engineering_pass",
                summary.get("baseline_relative_engineering_pass"),
            )
        )
        if raw_pass:
            continue
        clean = bool(
            summary.get("completed")
            and int(summary.get("reported_severe_count") or 0) == 0
            and int(summary.get("fatal_occurrences") or 0) == 0
            and not summary.get("unexplained_recurring_groups")
        )
        new_categories = set(
            summary.get("new_numerical_warning_categories", [])
        )
        critical = set(summary.get("critical_warning_count_increases", {}))
        if (
            clean
            and new_categories.issubset({"hvac_iterations"})
            and critical == {"hvac_iterations"}
        ):
            selected.append((plan, summary))
        else:
            unsupported.append(str(plan["run_id"]))
    if unsupported:
        raise ValueError(
            "Event failures include cases not attributable solely to HVAC "
            f"iterations: {unsupported}"
        )
    return selected


def _prepare_input(source: Path, destination: Path, higher_iterations: int) -> int:
    text = source.read_text(encoding="utf-8", errors="replace")
    configuration = solver_configuration(text)
    original = configuration["maximum_hvac_iterations"]
    if original is None or higher_iterations <= int(original):
        raise ValueError("Higher HVAC iteration limit must exceed the event input")
    updated = set_maximum_hvac_iterations(text, higher_iterations)
    if set_maximum_hvac_iterations(updated, int(original)) != text:
        raise AssertionError("Event solver audit changed fields beyond the iteration limit")
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
        if qa.get("completed") and all(
            qa.get(key) == value for key, value in expected.items()
        ):
            return qa
        raise ValueError(f"Event solver-audit cache mismatch for {job['run_id']}")
    return run_energyplus(
        job["idf"], job["weather"], runtime, job["run_folder"],
        job["compact"], job["zones"], cleanup_raw=True,
    )


def audit_event_solver_stability(
    event_plan: pd.DataFrame,
    execution_root: Path,
    runtime: Path,
    output_root: Path,
    *,
    higher_iterations: int = 100,
    workers: int = 6,
    temperature_tolerance_c: float = 0.01,
    energy_relative_tolerance: float = 0.001,
) -> pd.DataFrame:
    """Rerun and compare every eligible event solver-warning case."""
    selected = select_event_hvac_only_failures(event_plan, execution_root)
    if not selected:
        raise ValueError("No HVAC-only event failures require a stability audit")
    if workers < 1:
        raise ValueError("At least one worker is required")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict] = []
    for plan, summary in selected:
        run_id = str(plan["run_id"])
        source = Path(plan["input_idf_path"])
        if sha256(source) != str(summary["input_idf_sha256"]):
            raise ValueError(f"Primary event input hash changed: {run_id}")
        primary = Path(summary["hourly_path"])
        if sha256(primary) != str(summary["hourly_sha256"]):
            raise ValueError(f"Primary event trajectory hash changed: {run_id}")
        idf = output_root / "inputs" / f"{run_id}_{higher_iterations}.idf"
        original_iterations = _prepare_input(source, idf, higher_iterations)
        jobs.append({
            "plan": plan,
            "summary": summary,
            "run_id": run_id,
            "original_iterations": original_iterations,
            "idf": idf,
            "weather": Path(plan["weather_path"]),
            "zones": json.loads(str(plan["zones_json"])),
            "run_folder": output_root / "runs" / run_id,
            "compact": output_root / "hourly" / f"{run_id}.npz",
        })
    records: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(_run_or_reuse, job, runtime): job for job in jobs
        }
        for future in as_completed(pending):
            job = pending[future]
            qa = future.result()
            records.append((job, qa))
            print(json.dumps({
                "run_id": job["run_id"], "completed": qa.get("completed")
            }), flush=True)

    rows: list[dict] = []
    for job, qa in records:
        plan = job["plan"]
        summary = job["summary"]
        baseline_profile = dict(summary["baseline_warning_profile"])
        first_action_profile = dict(summary["action_warning_profile"])
        second_action_profile = warning_profile(job["run_folder"] / "eplusout.err")
        relative = relative_engineering_qa(
            qa, second_action_profile, baseline_profile
        )
        comparison = compare_event_trajectories(
            Path(summary["hourly_path"]), job["compact"],
            temperature_tolerance_c=temperature_tolerance_c,
            energy_relative_tolerance=energy_relative_tolerance,
        )
        warning_comparison = compare_solver_warning_categories(
            first_action_profile, second_action_profile
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
            "climate": plan["climate"],
            "composition_id": int(plan["composition_id"]),
            "control": plan["control"],
            "action_id": plan["action_id"],
            "first_maximum_hvac_iterations": job["original_iterations"],
            "second_maximum_hvac_iterations": higher_iterations,
            **comparison,
            "higher_iteration_completed": bool(qa.get("completed")),
            "higher_iteration_severe_count": int(
                qa.get("reported_severe_count") or 0
            ),
            "higher_iteration_fatal_occurrences": int(
                qa.get("fatal_occurrences") or 0
            ),
            "higher_iteration_relative_engineering_pass": relative[
                "engineering_pass_relative_to_prototype_baseline"
            ],
            "first_action_warning_profile": json.dumps(
                first_action_profile, sort_keys=True
            ),
            "second_action_warning_profile": json.dumps(
                second_action_profile, sort_keys=True
            ),
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
            "first_input_sha256": str(summary["input_idf_sha256"]),
            "first_hourly_sha256": str(summary["hourly_sha256"]),
            "second_input_sha256": sha256(job["idf"]),
            "second_hourly_sha256": sha256(job["compact"]),
            "second_qa_path": str(
                (job["run_folder"] / "RUN_QA.json").resolve()
            ),
            "second_hourly_path": str(job["compact"].resolve()),
        })
    result = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "action_id"], kind="stable"
    ).reset_index(drop=True)
    evidence_path = output_root / "EVENT_SOLVER_STABILITY.csv"
    result.to_csv(evidence_path, index=False)
    audit = {
        "selected_runs": len(result),
        "completed_higher_iteration_runs": int(
            result.higher_iteration_completed.sum()
        ),
        "stable_trajectories": int(
            result.trajectory_stable_within_declared_tolerance.sum()
        ),
        "nonworsening_warning_profiles": int(
            result.warning_profile_nonworsening.sum()
        ),
        "stable_warning_categories": int(result.warning_categories_stable.sum()),
        "solver_stability_passes": int(result.solver_stability_pass.sum()),
        "stability_rule": (
            "completed; zero severe/fatal; no warning category introduced at the "
            "higher iteration limit; temperature and energy trajectories within "
            "the declared tolerances; warning-count drift retained as a diagnostic"
        ),
        "facility_electricity_compared": int(
            result.facility_electricity_compared.sum()
        ),
        "all_completed": bool(result.higher_iteration_completed.all()),
        "all_stable": bool(
            result.trajectory_stable_within_declared_tolerance.all()
        ),
        "primary_outputs_modified": False,
        "evidence_sha256": sha256(evidence_path),
    }
    (output_root / "EVENT_SOLVER_STABILITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-plan", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--higher-iterations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--temperature-tolerance-c", type=float, default=0.01)
    parser.add_argument("--energy-relative-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    result = audit_event_solver_stability(
        pd.read_csv(args.event_plan),
        args.execution_root.resolve(),
        args.runtime.resolve(),
        args.output_root.resolve(),
        higher_iterations=args.higher_iterations,
        workers=args.workers,
        temperature_tolerance_c=args.temperature_tolerance_c,
        energy_relative_tolerance=args.energy_relative_tolerance,
    )
    if (
        not result.higher_iteration_completed.all()
        or not result.trajectory_stable_within_declared_tolerance.all()
        or not result.solver_stability_pass.all()
    ):
        raise RuntimeError(
            "Event solver-stability audit did not pass every selected run"
        )


if __name__ == "__main__":
    main()
