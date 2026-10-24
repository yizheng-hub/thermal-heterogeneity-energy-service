"""Source-building-preserving robustness analysis on frozen trajectories.

This program does not run EnergyPlus or change building physics.  It samples
archive units within each protocol-eligible source building, places them in the
frozen zone capacities, and re-scores the existing annual and July trajectory
libraries under room-mean and full demand representations.

The public interface deliberately separates the existing paper data bundle
(`--data-root`) from this extension's inputs (`--input-root`).  All trajectory
paths are resolved relative to `--input-root/hourly_inputs`; absolute source
paths in the local provenance index are never used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORE_TOL_C = 1e-9
QA_TOL_C = 1e-8
ENERGY_ABS_TOL_KWH = 1e-6
ENERGY_REL_TOL = 1e-9
RESPONSE_ABS_TOL_KW = 1e-9
RESPONSE_REL_TOL = 1e-9
PROFILE_COUNT = 267
TAIL_COUNT = 27
LAMBDAS = (0.0, 1.0)
FRACTIONS = (0.50, 0.75, 0.90, 1.00)
REPRESENTATIONS = ("mean", "full")
PRECONDITION_SCENARIO = "multinational_source_balanced"
PRECONDITION_CONTEXT = "Miami"
PRECONDITION_SEED = 2026102001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--precondition-only",
        action="store_true",
        help="Reproduce one frozen annual and July group, then stop.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_zone(value: str) -> str:
    return str(value).strip().upper()


def energy_tolerance(a: float, b: float) -> float:
    return max(ENERGY_ABS_TOL_KWH, ENERGY_REL_TOL * max(abs(a), abs(b)))


def response_tolerance(a: float, b: float) -> float:
    return max(RESPONSE_ABS_TOL_KW, RESPONSE_REL_TOL * max(abs(a), abs(b)))


def trajectory_path(input_root: Path, run_class: str, action_id: str) -> Path:
    prefix = "anchor" if run_class == "common25_anchor" else run_class
    name = f"{prefix}_{action_id}.npz"
    path = input_root / "hourly_inputs" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        result = {key: stored[key].copy() for key in stored.files}
    required = {"temperatures", "people", "zone_ids", "occupied", "month", "day", "hour", "dst"}
    if set(result) != required:
        raise RuntimeError(f"Unexpected trajectory arrays in {path}: {sorted(result)}")
    if result["temperatures"].shape != result["people"].shape:
        raise RuntimeError(f"Temperature/people shape mismatch: {path}")
    if result["temperatures"].shape[0] != 8760:
        raise RuntimeError(f"Expected 8760 rows: {path}")
    if not np.isfinite(result["temperatures"]).all() or not np.isfinite(result["people"]).all():
        raise RuntimeError(f"Non-finite trajectory: {path}")
    return result


def event_and_recovery_masks(data: dict[str, np.ndarray], month: int, day: int) -> tuple[np.ndarray, np.ndarray]:
    local_hour = data["hour"] - 1 + data["dst"]
    event = (
        (data["month"] == month)
        & (data["day"] == day)
        & (local_hour >= 14)
        & (local_hour < 16)
    )
    indices = np.flatnonzero(event)
    if len(indices) != 2:
        raise RuntimeError(f"Expected two event hours for month={month}, day={day}; found {len(indices)}")
    recovery = np.zeros(len(event), dtype=bool)
    recovery[indices[-1] + 1 : indices[-1] + 49] = True
    if int(recovery.sum()) != 48 or np.any(event & recovery):
        raise RuntimeError("Invalid 48-hour recovery mask")
    occupied = data["occupied"].astype(bool)
    return event & occupied, recovery & occupied


def zone_distributions(data: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if mask is None:
        mask = data["occupied"].astype(bool)
    if mask.shape != (8760,):
        raise RuntimeError("Service mask must contain 8760 Boolean rows")
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for column, zone in enumerate(data["zone_ids"]):
        temperature = data["temperatures"][mask, column]
        weight = data["people"][mask, column]
        if not len(temperature) or float(weight.sum()) <= 0:
            continue
        order = np.argsort(temperature, kind="mergesort")
        temperature = temperature[order]
        weight = weight[order]
        result[norm_zone(str(zone))] = (
            temperature,
            np.r_[0.0, np.cumsum(weight)],
            np.r_[0.0, np.cumsum(weight * temperature)],
        )
    return result


def score_profiles(
    distributions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    profiles: pd.DataFrame,
    spread_lambda: float,
    representation: str,
    fixed_ids: set[str] | None = None,
) -> tuple[float, float | None, pd.DataFrame]:
    if representation not in REPRESENTATIONS:
        raise ValueError(representation)
    group = profiles.reset_index(drop=True).copy()
    group["zone_mean_c"] = group.groupby("zone_id")["demand_centre_c"].transform("mean")
    centres = group.zone_mean_c.to_numpy(dtype=float)
    if representation == "full":
        centres = centres + spread_lambda * (group.demand_centre_c.to_numpy(dtype=float) - centres)
    distances = np.full(len(group), np.nan)
    weights = np.full(len(group), np.nan)
    for zone, local in group.groupby("zone_id", sort=False):
        key = norm_zone(zone)
        if key not in distributions:
            raise RuntimeError(f"No occupied trajectory support for zone {zone}")
        temperature, cumulative_weight, cumulative_temperature = distributions[key]
        total_weight = cumulative_weight[-1]
        total_temperature = cumulative_temperature[-1]
        indices = local.index.to_numpy(dtype=int)
        local_centres = centres[indices]
        cuts = np.searchsorted(temperature, local_centres, side="right")
        distances[indices] = (
            local_centres * (2 * cumulative_weight[cuts] - total_weight)
            + total_temperature
            - 2 * cumulative_temperature[cuts]
        ) / total_weight
        weights[indices] = total_weight / len(indices)
    if not np.isfinite(distances).all() or not np.isfinite(weights).all():
        raise RuntimeError("Profile scoring produced non-finite values")
    people = pd.DataFrame(
        {
            "profile_id": group.profile_id.astype(str),
            "zone_id": group.zone_id.astype(str),
            "distance_c": distances,
            "profile_person_hour_weight": weights,
        }
    )
    mean_c = float(np.average(distances, weights=weights))
    tail_c = None
    if fixed_ids is not None:
        chosen = people[people.profile_id.isin(fixed_ids)]
        if len(chosen) != TAIL_COUNT:
            raise RuntimeError(f"Expected {TAIL_COUNT} fixed-tail profiles; found {len(chosen)}")
        tail_c = float(chosen.distance_c.mean())
    return mean_c, tail_c, people


def make_anchor(
    common_data: dict[str, np.ndarray], profiles: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    distributions = zone_distributions(common_data)
    _, _, people = score_profiles(distributions, profiles, 1.0, "full")
    fixed = (
        people.sort_values(["distance_c", "profile_id"], ascending=[False, True], kind="mergesort")
        .head(TAIL_COUNT)
        .profile_id.astype(str)
        .tolist()
    )
    mean_c, tail_c, scored = score_profiles(distributions, profiles, 1.0, "full", set(fixed))
    return {"mean_c": mean_c, "fixed_tail_c": tail_c, "fixed_ids": fixed}, scored


def validate_inputs(data_root: Path, input_root: Path, output_root: Path) -> dict[str, Any]:
    required_input = [
        "PROTOCOL.json",
        "ELIGIBLE_SOURCE_BUILDING_POOL.csv.gz",
        "ELIGIBLE_SOURCE_BUILDING_METRIC_STRATA.csv",
        "ANNUAL_CANDIDATE_SUPPORT.csv",
        "JULY_CANDIDATE_SUPPORT.csv",
        "EVENT_BASELINE_MAP.csv",
        "ZONE_CAPACITIES.csv",
    ]
    required_data = [
        "building/annual_action_library.csv.gz",
        "building/annual_service_scores.csv.gz",
        "building/annual_service_anchors.csv",
        "building/dynamic_action_library.csv.gz",
        "building/dynamic_service_scores.csv.gz",
        "building/dynamic_service_anchors.csv",
        "population/population_assignments_minimized.csv.gz",
        "population/fixed_tail_memberships_minimized.csv.gz",
        "expected/annual_decisions_reference.csv.gz",
        "expected/dynamic_decisions_reference.csv.gz",
    ]
    missing = [str(input_root / name) for name in required_input if not (input_root / name).is_file()]
    missing += [str(data_root / name) for name in required_data if not (data_root / name).is_file()]
    index_path = input_root / "INPUT_INDEX_PORTABLE.csv"
    if not index_path.is_file():
        index_path = input_root / "TRAJECTORY_INPUT_INDEX.csv"
    if not index_path.is_file():
        missing.append(str(index_path))
    output_root.mkdir(parents=True, exist_ok=True)
    failure_path = output_root / "MISSING_FAILURE_LEDGER.csv"
    if missing:
        pd.DataFrame({"failure_type": "missing_required_input", "item": missing}).to_csv(failure_path, index=False)
        raise RuntimeError(f"Missing {len(missing)} required inputs; see {failure_path}")

    protocol = json.loads((input_root / "PROTOCOL.json").read_text(encoding="utf-8"))
    preflight_path = input_root / "PREFLIGHT_RESULT.json"
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight_path.is_file()
        else None
    )
    if protocol["profile_count"] != PROFILE_COUNT or protocol["fixed_tail_count"] != TAIL_COUNT:
        raise RuntimeError("Protocol profile/tail counts differ from the locked analysis")
    if protocol["spread_lambda"] != [0, 1] or protocol["service_budget_fractions"] != list(FRACTIONS):
        raise RuntimeError("Protocol lambda or service fractions differ from the locked analysis")
    if protocol["annual_design_cases"] != 960 or protocol["july_design_cases"] != 960:
        raise RuntimeError("Protocol design-case counts differ from 960 annual + 960 July")
    if preflight is not None and (
        preflight.get("outcomes_generated") != 0 or preflight.get("energyplus_runs") != 0
    ):
        raise RuntimeError("Preflight unexpectedly reports prior outcome scoring or physical runs")
    if preflight is None and not protocol.get("preflight_only_no_outcomes_generated", False):
        raise RuntimeError("Portable protocol lacks the locked no-outcome preflight declaration")

    index = pd.read_csv(index_path)
    if len(index) != 416 or set(index.run_class) != {"annual", "event", "common25_anchor"}:
        raise RuntimeError("Trajectory index is not the locked 416-run cache")
    cache_column = "cache_path"
    failures = []
    for row in index.itertuples(index=False):
        cache = input_root / "hourly_inputs" / Path(getattr(row, cache_column)).name
        if not cache.is_file():
            failures.append({"failure_type": "missing_cache", "item": str(cache)})
        elif hasattr(row, "cache_sha256") and sha256(cache) != row.cache_sha256:
            failures.append({"failure_type": "cache_sha256_mismatch", "item": str(cache)})
    pd.DataFrame(failures, columns=["failure_type", "item"]).to_csv(failure_path, index=False)
    if failures:
        raise RuntimeError(f"Trajectory cache validation failed; see {failure_path}")
    return {
        "protocol_sha256": sha256(input_root / "PROTOCOL.json"),
        "preflight_sha256": sha256(preflight_path) if preflight_path.is_file() else None,
        "trajectory_index": str(index_path),
        "trajectory_index_sha256": sha256(index_path),
        "event_baseline_map_sha256": sha256(input_root / "EVENT_BASELINE_MAP.csv"),
        "cache_files_verified": len(index),
        "energyplus_runs": 0,
        "physical_parameter_changes": 0,
    }


def load_libraries(data_root: Path, input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_support = pd.read_csv(input_root / "ANNUAL_CANDIDATE_SUPPORT.csv")
    annual_library = pd.read_csv(data_root / "building/annual_action_library.csv.gz")
    annual = annual_support.merge(
        annual_library,
        left_on=["city", "physical_action_id"],
        right_on=["context_id", "physical_action_id"],
        how="left",
        validate="many_to_one",
    )
    if annual.purchased_site_energy_kwh.isna().any():
        raise RuntimeError("Annual support is missing frozen action metadata")

    event_map = pd.read_csv(input_root / "EVENT_BASELINE_MAP.csv")
    july_support = pd.read_csv(input_root / "JULY_CANDIDATE_SUPPORT.csv")
    if not july_support.merge(
        event_map,
        on=["city", "composition_seed", "event_month", "event_day", "event_action_id"],
        how="left",
        validate="one_to_one",
    ).baseline_physical_action_id.notna().all():
        raise RuntimeError("July support is missing event-to-baseline mappings")
    dynamic_library = pd.read_csv(data_root / "building/dynamic_action_library.csv.gz")
    dynamic_library = dynamic_library[dynamic_library.event_id.str.contains(":m07:", regex=False)].copy()
    dynamic = event_map.merge(
        dynamic_library,
        left_on=["city", "event_action_id"],
        right_on=["context_id", "physical_action_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_library"),
    )
    if dynamic.event_average_reduction_kw.isna().any():
        raise RuntimeError("July support is missing frozen response metadata")
    return annual, dynamic


def original_profiles(data_root: Path, scenario: str, context: str, seed: int) -> tuple[pd.DataFrame, set[str]]:
    assignments = pd.read_csv(data_root / "population/population_assignments_minimized.csv.gz")
    group = assignments[
        assignments.scenario_id.eq(scenario)
        & assignments.context.eq(context)
        & assignments.composition_seed.eq(seed)
    ].copy()
    if len(group) != PROFILE_COUNT:
        raise RuntimeError("Original reproduction group is not 267 profiles")
    profiles = group.rename(
        columns={"release_profile_id": "profile_id", "zone_id": "zone_id", "demand_centre_c": "demand_centre_c"}
    )[["profile_id", "zone_id", "demand_centre_c"]].reset_index(drop=True)
    tails = pd.read_csv(data_root / "population/fixed_tail_memberships_minimized.csv.gz")
    fixed = set(
        tails[
            tails.scenario_id.eq(scenario)
            & tails.context.eq(context)
            & tails.composition_seed.eq(seed)
        ].release_profile_id.astype(str)
    )
    if len(fixed) != TAIL_COUNT:
        raise RuntimeError("Original reproduction fixed tail is not 27 profiles")
    return profiles, fixed


def annual_anchor_for_profiles(input_root: Path, context: str, profiles: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    data = load_trajectory(trajectory_path(input_root, "common25_anchor", context))
    return make_anchor(data, profiles)


def score_annual_group(
    input_root: Path,
    annual_actions: pd.DataFrame,
    cohort_id: str,
    profiles: pd.DataFrame,
    anchor: dict[str, Any],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    fixed = set(anchor["fixed_ids"])
    for action in annual_actions.to_dict("records"):
        data = load_trajectory(trajectory_path(input_root, "annual", action["physical_action_id"]))
        distributions = zone_distributions(data)
        for spread_lambda in LAMBDAS:
            for representation in REPRESENTATIONS:
                mean_c, tail_c, _ = score_profiles(
                    distributions, profiles, spread_lambda, representation, fixed
                )
                rows.append(
                    {
                        "cohort_id": cohort_id,
                        "context": action["city"],
                        "composition_seed": int(action["composition_seed"]),
                        "lambda": spread_lambda,
                        "representation": representation,
                        "physical_action_id": action["physical_action_id"],
                        "mean_c": mean_c,
                        "fixed_tail_c": tail_c,
                        "site_kwh": float(action["site_kwh"]),
                        "electric_kwh": float(action["electric_kwh"]),
                        "gas_kwh_fuel": float(action["gas_kwh_fuel"]),
                        "operational_co2_kg": float(action["operational_co2_kg"]),
                        "common25_mean_anchor_c": float(anchor["mean_c"]),
                        "common25_fixed_tail_anchor_c": float(anchor["fixed_tail_c"]),
                        **metadata,
                    }
                )
    return pd.DataFrame(rows)


def select_annual(pool: pd.DataFrame) -> pd.Series | None:
    eligible = pool[pool.service_eligible]
    if not len(eligible):
        return None
    minimum = float(eligible.site_kwh.min())
    tied = eligible[
        eligible.site_kwh.apply(lambda value: abs(float(value) - minimum) <= energy_tolerance(float(value), minimum))
    ]
    return tied.sort_values("physical_action_id", kind="mergesort").iloc[0]


def annual_decisions(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    keys = ["cohort_id", "context", "composition_seed", "lambda"]
    for key, local in scores.groupby(keys, sort=True):
        cohort_id, context, seed, spread_lambda = key
        mean_anchor = float(local.common25_mean_anchor_c.iloc[0])
        tail_anchor = float(local.common25_fixed_tail_anchor_c.iloc[0])
        for fraction in FRACTIONS:
            pools: dict[str, pd.DataFrame] = {}
            selected: dict[str, pd.Series | None] = {}
            for representation in REPRESENTATIONS:
                pool = local[local.representation.eq(representation)].copy()
                pool["service_fraction"] = fraction
                pool["mean_budget_c"] = fraction * mean_anchor
                pool["fixed_tail_budget_c"] = fraction * tail_anchor
                pool["mean_eligible"] = pool.mean_c.le(pool.mean_budget_c + SCORE_TOL_C)
                pool["tail_eligible"] = pool.fixed_tail_c.le(pool.fixed_tail_budget_c + SCORE_TOL_C)
                pool["service_eligible"] = pool.mean_eligible & pool.tail_eligible
                pools[representation] = pool
                selected[representation] = select_annual(pool)
                eligibility.extend(pool.to_dict("records"))
            mean_pool = pools["mean"].set_index("physical_action_id")
            full_pool = pools["full"].set_index("physical_action_id")
            paired = mean_pool[["service_eligible"]].rename(columns={"service_eligible": "mean_eligible"}).join(
                full_pool[["service_eligible"]].rename(columns={"service_eligible": "full_eligible"}),
                validate="one_to_one",
            )
            mean_choice, full_choice = selected["mean"], selected["full"]
            if mean_choice is None and full_choice is None:
                feasibility = "neither_feasible"
            elif mean_choice is not None and full_choice is None:
                feasibility = "mean_only_feasible"
            elif mean_choice is None and full_choice is not None:
                feasibility = "full_only_feasible"
            else:
                feasibility = "both_feasible"
            if mean_choice is not None and full_choice is not None:
                if mean_choice.physical_action_id == full_choice.physical_action_id:
                    selection = "same_action"
                elif abs(float(mean_choice.site_kwh) - float(full_choice.site_kwh)) <= energy_tolerance(
                    float(mean_choice.site_kwh), float(full_choice.site_kwh)
                ):
                    selection = "different_action_energy_tie"
                else:
                    selection = "different_action_energy_nontie"
            else:
                selection = "not_applicable"
            if mean_choice is None:
                mean_full = None
                reaudit = "no_mean_selection"
            else:
                mean_full = full_pool.loc[mean_choice.physical_action_id]
                reaudit = "mean_selected_full_eligible" if bool(mean_full.service_eligible) else "mean_selected_false_accept"
            first = local.iloc[0]
            row: dict[str, Any] = {
                "cohort_id": cohort_id,
                "context": context,
                "composition_seed": int(seed),
                "lambda": float(spread_lambda),
                "service_fraction": fraction,
                "annual_mean_budget_c": fraction * mean_anchor,
                "annual_fixed_tail_budget_c": fraction * tail_anchor,
                "candidate_count_each_representation": len(mean_pool),
                "mean_eligible_action_count": int(mean_pool.service_eligible.sum()),
                "full_eligible_action_count": int(full_pool.service_eligible.sum()),
                "candidate_false_accept_count": int((paired.mean_eligible & ~paired.full_eligible).sum()),
                "candidate_false_reject_count": int((~paired.mean_eligible & paired.full_eligible).sum()),
                "paired_feasibility_status": feasibility,
                "selection_status": selection,
                "mean_selected_full_reaudit_status": reaudit,
            }
            for column in [
                "source_country",
                "source_city",
                "known_id_category",
                "observed_temperature_source",
                "sample_reference_mean_c",
                "sample_reference_sd_c",
            ]:
                if column in first:
                    row[column] = first[column]
            for label, choice in [("mean", mean_choice), ("full", full_choice)]:
                row[f"{label}_selected_action_id"] = None if choice is None else choice.physical_action_id
                for output, source in [
                    ("site_kwh", "site_kwh"),
                    ("electric_kwh", "electric_kwh"),
                    ("gas_kwh_fuel", "gas_kwh_fuel"),
                    ("operational_co2_kg", "operational_co2_kg"),
                    ("mean_c", "mean_c"),
                    ("fixed_tail_c", "fixed_tail_c"),
                ]:
                    row[f"{label}_selected_{output}"] = None if choice is None else float(choice[source])
            row["mean_selected_full_eligible"] = None if mean_full is None else bool(mean_full.service_eligible)
            if mean_full is not None:
                row["mean_selected_full_mean_c"] = float(mean_full.mean_c)
                row["mean_selected_full_fixed_tail_c"] = float(mean_full.fixed_tail_c)
                row["mean_selected_full_mean_excess_c"] = max(float(mean_full.mean_c - fraction * mean_anchor), 0.0)
                row["mean_selected_full_tail_excess_c"] = max(float(mean_full.fixed_tail_c - fraction * tail_anchor), 0.0)
            if mean_choice is not None and full_choice is not None:
                row["full_minus_mean_selected_site_kwh"] = float(full_choice.site_kwh - mean_choice.site_kwh)
                row["full_minus_mean_selected_site_pct"] = 100 * float(full_choice.site_kwh - mean_choice.site_kwh) / float(mean_choice.site_kwh)
            decisions.append(row)
    return pd.DataFrame(decisions), pd.DataFrame(eligibility)


def dynamic_anchor_for_profiles(
    input_root: Path,
    context: str,
    month: int,
    day: int,
    profiles: pd.DataFrame,
    fixed_ids: set[str],
) -> dict[str, float]:
    common = load_trajectory(trajectory_path(input_root, "common25_anchor", context))
    event_mask, recovery_mask = event_and_recovery_masks(common, month, day)
    result: dict[str, float] = {}
    for window, mask in [("event", event_mask), ("recovery", recovery_mask)]:
        mean_c, tail_c, _ = score_profiles(zone_distributions(common, mask), profiles, 1.0, "full", fixed_ids)
        result[f"{window}_mean_c"] = mean_c
        result[f"{window}_fixed_tail_c"] = float(tail_c)
    return result


def score_dynamic_group(
    input_root: Path,
    dynamic_actions: pd.DataFrame,
    cohort_id: str,
    profiles: pd.DataFrame,
    annual_anchor: dict[str, Any],
    dynamic_anchor: dict[str, float],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    fixed_ids = set(annual_anchor["fixed_ids"])
    for action in dynamic_actions.to_dict("records"):
        baseline = load_trajectory(trajectory_path(input_root, "annual", action["baseline_physical_action_id"]))
        perturbed = load_trajectory(trajectory_path(input_root, "event", action["event_action_id"]))
        event_mask, recovery_mask = event_and_recovery_masks(perturbed, int(action["event_month"]), int(action["event_day"]))
        base_event_mask, base_recovery_mask = event_and_recovery_masks(baseline, int(action["event_month"]), int(action["event_day"]))
        if not np.array_equal(event_mask, base_event_mask) or not np.array_equal(recovery_mask, base_recovery_mask):
            raise RuntimeError(f"Baseline/event service masks differ: {action['event_action_id']}")
        distributions = {
            "event_baseline": zone_distributions(baseline, event_mask),
            "event_perturbed": zone_distributions(perturbed, event_mask),
            "recovery_baseline": zone_distributions(baseline, recovery_mask),
            "recovery_perturbed": zone_distributions(perturbed, recovery_mask),
        }
        for spread_lambda in LAMBDAS:
            for representation in REPRESENTATIONS:
                values: dict[str, float] = {}
                for label, distribution in distributions.items():
                    mean_c, tail_c, _ = score_profiles(
                        distribution, profiles, spread_lambda, representation, fixed_ids
                    )
                    values[f"{label}_mean_c"] = mean_c
                    values[f"{label}_fixed_tail_c"] = float(tail_c)
                rows.append(
                    {
                        "cohort_id": cohort_id,
                        "context": action["city"],
                        "composition_seed": int(action["composition_seed"]),
                        "event_month": int(action["event_month"]),
                        "event_day": int(action["event_day"]),
                        "lambda": spread_lambda,
                        "representation": representation,
                        "event_action_id": action["event_action_id"],
                        "baseline_physical_action_id": action["baseline_physical_action_id"],
                        "amplitude_c": float(action["amplitude_c"]),
                        "source_strategies": action["source_strategies"],
                        **values,
                        "event_anchor_mean_c": dynamic_anchor["event_mean_c"],
                        "event_anchor_fixed_tail_c": dynamic_anchor["event_fixed_tail_c"],
                        "recovery_anchor_mean_c": dynamic_anchor["recovery_mean_c"],
                        "recovery_anchor_fixed_tail_c": dynamic_anchor["recovery_fixed_tail_c"],
                        "event_average_reduction_kw": float(action["event_average_reduction_kw"]),
                        "recovery_net_extra_electric_kwh": float(action["recovery_net_extra_electric_kwh"]),
                        "recovery_peak_hour_extra_kw": float(action["recovery_peak_hour_extra_kw"]),
                        "event_plus48h_net_electric_saving_kwh": float(action["event_plus48h_net_electric_saving_kwh"]),
                        "event_plus48h_gas_change_kwh_fuel": float(action["event_plus48h_gas_change_kwh_fuel"]),
                        **metadata,
                    }
                )
    return pd.DataFrame(rows)


def select_dynamic(pool: pd.DataFrame, threshold: float) -> pd.Series | None:
    eligible = pool[pool.service_eligible & pool.event_average_reduction_kw.gt(threshold)]
    if not len(eligible):
        return None
    maximum = float(eligible.event_average_reduction_kw.max())
    tied = eligible[
        eligible.event_average_reduction_kw.apply(
            lambda value: abs(float(value) - maximum) <= response_tolerance(float(value), maximum)
        )
    ]
    return tied.sort_values("event_action_id", kind="mergesort").iloc[0]


def dynamic_decisions(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    keys = ["cohort_id", "context", "composition_seed", "event_month", "lambda"]
    for key, local in scores.groupby(keys, sort=True):
        cohort_id, context, seed, event_month, spread_lambda = key
        max_abs = float(local.event_average_reduction_kw.abs().max())
        threshold = max(RESPONSE_ABS_TOL_KW, RESPONSE_REL_TOL * max_abs)
        for fraction in FRACTIONS:
            pools: dict[str, pd.DataFrame] = {}
            selected: dict[str, pd.Series | None] = {}
            states: dict[str, str] = {}
            for representation in REPRESENTATIONS:
                pool = local[local.representation.eq(representation)].copy()
                pool["service_fraction"] = fraction
                for window in ["event", "recovery"]:
                    pool[f"{window}_mean_budget_c"] = fraction * pool[f"{window}_anchor_mean_c"]
                    pool[f"{window}_fixed_tail_budget_c"] = fraction * pool[f"{window}_anchor_fixed_tail_c"]
                checks = []
                for window in ["event", "recovery"]:
                    for state in ["baseline", "perturbed"]:
                        checks.append(pool[f"{window}_{state}_mean_c"].le(pool[f"{window}_mean_budget_c"] + SCORE_TOL_C))
                        checks.append(pool[f"{window}_{state}_fixed_tail_c"].le(pool[f"{window}_fixed_tail_budget_c"] + SCORE_TOL_C))
                pool["service_eligible"] = np.logical_and.reduce(checks)
                pool["positive_response"] = pool.event_average_reduction_kw.gt(threshold)
                pools[representation] = pool
                selected[representation] = select_dynamic(pool, threshold)
                if not pool.service_eligible.any():
                    states[representation] = "no_service_eligible_action"
                elif not (pool.service_eligible & pool.positive_response).any():
                    states[representation] = "eligible_but_no_positive_response"
                else:
                    states[representation] = "positive_action_selected"
                eligibility.extend(pool.to_dict("records"))
            mean_pool = pools["mean"].set_index("event_action_id")
            full_pool = pools["full"].set_index("event_action_id")
            paired = mean_pool[["service_eligible"]].rename(columns={"service_eligible": "mean_eligible"}).join(
                full_pool[["service_eligible"]].rename(columns={"service_eligible": "full_eligible"}),
                validate="one_to_one",
            )
            mean_choice, full_choice = selected["mean"], selected["full"]
            if mean_choice is None:
                mean_full = None
                reaudit = "no_mean_selection"
            else:
                mean_full = full_pool.loc[mean_choice.event_action_id]
                reaudit = "mean_selected_full_eligible" if bool(mean_full.service_eligible) else "mean_selected_false_accept"
            if mean_choice is not None and full_choice is not None:
                if mean_choice.event_action_id == full_choice.event_action_id:
                    selection = "same_action"
                elif abs(float(mean_choice.event_average_reduction_kw) - float(full_choice.event_average_reduction_kw)) <= response_tolerance(
                    float(mean_choice.event_average_reduction_kw), float(full_choice.event_average_reduction_kw)
                ):
                    selection = "different_action_response_tie"
                else:
                    selection = "different_action_response_nontie"
            else:
                selection = "not_applicable"
            first = local.iloc[0]
            row: dict[str, Any] = {
                "cohort_id": cohort_id,
                "context": context,
                "composition_seed": int(seed),
                "event_month": int(event_month),
                "lambda": float(spread_lambda),
                "service_fraction": fraction,
                "event_mean_budget_c": fraction * float(first.event_anchor_mean_c),
                "event_fixed_tail_budget_c": fraction * float(first.event_anchor_fixed_tail_c),
                "recovery_mean_budget_c": fraction * float(first.recovery_anchor_mean_c),
                "recovery_fixed_tail_budget_c": fraction * float(first.recovery_anchor_fixed_tail_c),
                "candidate_count_each_representation": len(mean_pool),
                "mean_service_eligible_action_count": int(mean_pool.service_eligible.sum()),
                "full_service_eligible_action_count": int(full_pool.service_eligible.sum()),
                "mean_positive_action_count": int((mean_pool.service_eligible & mean_pool.positive_response).sum()),
                "full_positive_action_count": int((full_pool.service_eligible & full_pool.positive_response).sum()),
                "candidate_false_accept_count": int((paired.mean_eligible & ~paired.full_eligible).sum()),
                "candidate_false_reject_count": int((~paired.mean_eligible & paired.full_eligible).sum()),
                "mean_selection_state": states["mean"],
                "full_selection_state": states["full"],
                "selection_status": selection,
                "mean_selected_full_reaudit_status": reaudit,
                "positive_response_threshold_kw": threshold,
            }
            for column in [
                "source_country",
                "source_city",
                "known_id_category",
                "observed_temperature_source",
                "sample_reference_mean_c",
                "sample_reference_sd_c",
            ]:
                if column in first:
                    row[column] = first[column]
            for label, choice in [("mean", mean_choice), ("full", full_choice)]:
                row[f"{label}_selected_action_id"] = None if choice is None else choice.event_action_id
                for output in [
                    "event_average_reduction_kw",
                    "recovery_net_extra_electric_kwh",
                    "recovery_peak_hour_extra_kw",
                    "event_plus48h_net_electric_saving_kwh",
                    "event_plus48h_gas_change_kwh_fuel",
                ]:
                    row[f"{label}_selected_{output}"] = None if choice is None else float(choice[output])
            row["mean_selected_full_eligible"] = None if mean_full is None else bool(mean_full.service_eligible)
            if mean_full is not None:
                excesses = []
                for window in ["event", "recovery"]:
                    for state in ["baseline", "perturbed"]:
                        for metric in ["mean", "fixed_tail"]:
                            score = float(mean_full[f"{window}_{state}_{metric}_c"])
                            budget = float(mean_full[f"{window}_{metric}_budget_c"])
                            excess = max(score - budget, 0.0)
                            row[f"mean_selected_full_{window}_{state}_{metric}_excess_c"] = excess
                            excesses.append(excess)
                row["mean_selected_full_max_service_excess_c"] = max(excesses)
            if mean_choice is not None and full_choice is not None:
                row["full_minus_mean_selected_response_kw"] = float(
                    full_choice.event_average_reduction_kw - mean_choice.event_average_reduction_kw
                )
            decisions.append(row)
    return pd.DataFrame(decisions), pd.DataFrame(eligibility)


def equal_or_both_missing(left: pd.Series, right: pd.Series) -> bool:
    return bool(((left.astype("string") == right.astype("string")) | (left.isna() & right.isna())).all())


def run_precondition(
    data_root: Path,
    input_root: Path,
    annual_actions: pd.DataFrame,
    dynamic_actions: pd.DataFrame,
) -> dict[str, Any]:
    profiles, expected_fixed = original_profiles(
        data_root, PRECONDITION_SCENARIO, PRECONDITION_CONTEXT, PRECONDITION_SEED
    )
    anchor, _ = annual_anchor_for_profiles(input_root, PRECONDITION_CONTEXT, profiles)
    fixed_match = set(anchor["fixed_ids"]) == expected_fixed
    local_annual = annual_actions[
        annual_actions.city.eq(PRECONDITION_CONTEXT)
        & annual_actions.composition_seed.eq(PRECONDITION_SEED)
    ]
    annual_scores = score_annual_group(
        input_root,
        local_annual,
        PRECONDITION_SCENARIO,
        profiles,
        anchor,
        {},
    )
    expected_annual = pd.read_csv(data_root / "building/annual_service_scores.csv.gz")
    expected_annual = expected_annual[
        expected_annual.population_scenario.eq(PRECONDITION_SCENARIO)
        & expected_annual.city.eq(PRECONDITION_CONTEXT)
        & expected_annual.composition_seed.eq(PRECONDITION_SEED)
        & expected_annual.delta_c.eq(0)
        & expected_annual["lambda"].isin(LAMBDAS)
        & expected_annual.physical_action_id.isin(local_annual.physical_action_id)
    ]
    annual_merged = annual_scores.merge(
        expected_annual,
        left_on=["context", "composition_seed", "lambda", "representation", "physical_action_id"],
        right_on=["city", "composition_seed", "lambda", "representation", "physical_action_id"],
        validate="one_to_one",
        suffixes=("_new", "_expected"),
    )
    annual_score_error = float(
        np.abs(
            annual_merged[["mean_c_new", "fixed_tail_c_new"]].to_numpy()
            - annual_merged[["mean_c_expected", "fixed_tail_c_expected"]].to_numpy()
        ).max()
    )
    annual_decision, _ = annual_decisions(annual_scores)
    expected_annual_decision = pd.read_csv(data_root / "expected/annual_decisions_reference.csv.gz")
    expected_annual_decision = expected_annual_decision[
        expected_annual_decision.population_scenario.eq(PRECONDITION_SCENARIO)
        & expected_annual_decision.city.eq(PRECONDITION_CONTEXT)
        & expected_annual_decision.composition_seed.eq(PRECONDITION_SEED)
        & expected_annual_decision.delta_c.eq(0)
        & expected_annual_decision["lambda"].isin(LAMBDAS)
    ]
    annual_compare = annual_decision.merge(
        expected_annual_decision,
        left_on=["context", "composition_seed", "lambda", "service_fraction"],
        right_on=["city", "composition_seed", "lambda", "service_fraction"],
        validate="one_to_one",
        suffixes=("_new", "_expected"),
    )
    annual_selection_columns = [
        "paired_feasibility_status",
        "selection_status",
        "mean_selected_action_id",
        "full_selected_action_id",
        "mean_selected_full_reaudit_status",
    ]
    annual_selection_match = all(
        equal_or_both_missing(annual_compare[f"{column}_new"], annual_compare[f"{column}_expected"])
        for column in annual_selection_columns
    )
    annual_count_match = all(
        np.array_equal(annual_compare[f"{column}_new"], annual_compare[f"{column}_expected"])
        for column in [
            "mean_eligible_action_count",
            "full_eligible_action_count",
            "candidate_false_accept_count",
            "candidate_false_reject_count",
        ]
    )

    local_dynamic = dynamic_actions[
        dynamic_actions.city.eq(PRECONDITION_CONTEXT)
        & dynamic_actions.composition_seed.eq(PRECONDITION_SEED)
    ]
    month = int(local_dynamic.event_month.iloc[0])
    day = int(local_dynamic.event_day.iloc[0])
    dynamic_anchor = dynamic_anchor_for_profiles(
        input_root, PRECONDITION_CONTEXT, month, day, profiles, expected_fixed
    )
    dynamic_scores = score_dynamic_group(
        input_root,
        local_dynamic,
        PRECONDITION_SCENARIO,
        profiles,
        anchor,
        dynamic_anchor,
        {},
    )
    expected_dynamic = pd.read_csv(data_root / "building/dynamic_service_scores.csv.gz")
    expected_dynamic = expected_dynamic[
        expected_dynamic.population_scenario.eq(PRECONDITION_SCENARIO)
        & expected_dynamic.city.eq(PRECONDITION_CONTEXT)
        & expected_dynamic.composition_seed.eq(PRECONDITION_SEED)
        & expected_dynamic.event_month.eq(7)
        & expected_dynamic.delta_c.eq(0)
        & expected_dynamic["lambda"].isin(LAMBDAS)
        & expected_dynamic.event_action_id.isin(local_dynamic.event_action_id)
    ]
    dynamic_merged = dynamic_scores.merge(
        expected_dynamic,
        left_on=["context", "composition_seed", "event_month", "lambda", "representation", "event_action_id"],
        right_on=["city", "composition_seed", "event_month", "lambda", "representation", "event_action_id"],
        validate="one_to_one",
        suffixes=("_new", "_expected"),
    )
    dynamic_columns = [
        f"{window}_{state}_{metric}_c"
        for window in ["event", "recovery"]
        for state in ["baseline", "perturbed"]
        for metric in ["mean", "fixed_tail"]
    ]
    dynamic_score_error = max(
        float(np.abs(dynamic_merged[f"{column}_new"] - dynamic_merged[f"{column}_expected"]).max())
        for column in dynamic_columns
    )
    dynamic_decision, _ = dynamic_decisions(dynamic_scores)
    expected_dynamic_decision = pd.read_csv(data_root / "expected/dynamic_decisions_reference.csv.gz")
    expected_dynamic_decision = expected_dynamic_decision[
        expected_dynamic_decision.population_scenario.eq(PRECONDITION_SCENARIO)
        & expected_dynamic_decision.city.eq(PRECONDITION_CONTEXT)
        & expected_dynamic_decision.composition_seed.eq(PRECONDITION_SEED)
        & expected_dynamic_decision.event_month.eq(7)
        & expected_dynamic_decision.delta_c.eq(0)
        & expected_dynamic_decision["lambda"].isin(LAMBDAS)
    ]
    dynamic_compare = dynamic_decision.merge(
        expected_dynamic_decision,
        left_on=["context", "composition_seed", "event_month", "lambda", "service_fraction"],
        right_on=["city", "composition_seed", "event_month", "lambda", "service_fraction"],
        validate="one_to_one",
        suffixes=("_new", "_expected"),
    )
    dynamic_selection_columns = [
        "mean_selection_state",
        "full_selection_state",
        "mean_selected_action_id",
        "full_selected_action_id",
        "mean_selected_full_reaudit_status",
    ]
    dynamic_selection_match = all(
        equal_or_both_missing(dynamic_compare[f"{column}_new"], dynamic_compare[f"{column}_expected"])
        for column in dynamic_selection_columns
    )
    dynamic_count_match = all(
        np.array_equal(dynamic_compare[f"{column}_new"], dynamic_compare[f"{column}_expected"])
        for column in [
            "mean_service_eligible_action_count",
            "full_service_eligible_action_count",
            "mean_positive_action_count",
            "full_positive_action_count",
            "candidate_false_accept_count",
            "candidate_false_reject_count",
        ]
    )
    passed = bool(
        fixed_match
        and annual_score_error <= QA_TOL_C
        and dynamic_score_error <= QA_TOL_C
        and annual_selection_match
        and annual_count_match
        and dynamic_selection_match
        and dynamic_count_match
    )
    return {
        "passed": passed,
        "scenario": PRECONDITION_SCENARIO,
        "context": PRECONDITION_CONTEXT,
        "composition_seed": PRECONDITION_SEED,
        "annual_candidate_actions": len(local_annual),
        "annual_score_rows_recomputed": len(annual_scores),
        "annual_decision_rows_recomputed": len(annual_decision),
        "annual_maximum_score_error_c": annual_score_error,
        "annual_selection_match": annual_selection_match,
        "annual_counts_match": annual_count_match,
        "july_candidate_actions": len(local_dynamic),
        "july_score_rows_recomputed": len(dynamic_scores),
        "july_decision_rows_recomputed": len(dynamic_decision),
        "july_maximum_score_error_c": dynamic_score_error,
        "july_selection_match": dynamic_selection_match,
        "july_counts_match": dynamic_count_match,
        "annual_fixed27_identity_match": fixed_match,
        "score_tolerance_c": QA_TOL_C,
    }


def build_source_profiles(
    input_root: Path, protocol: dict[str, Any]
) -> tuple[dict[tuple[str, int, str], pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(input_root / "ELIGIBLE_SOURCE_BUILDING_POOL.csv.gz")
    strata = pd.read_csv(input_root / "ELIGIBLE_SOURCE_BUILDING_METRIC_STRATA.csv")
    capacities = pd.read_csv(input_root / "ZONE_CAPACITIES.csv")
    profiles: dict[tuple[str, int, str], pd.DataFrame] = {}
    sample_rows = []
    assignment_rows = []
    qa_rows = []
    eligible = [row["release_building_group_id"] for row in protocol["eligible_source_buildings"]]
    if set(eligible) != set(pool.release_building_group_id.unique()) or len(eligible) != 10:
        raise RuntimeError("Eligible source-building pool does not match the ten-building protocol")
    for building_id in eligible:
        building = pool[pool.release_building_group_id.eq(building_id)].sort_values(
            "robustness_unit_id", kind="mergesort"
        ).reset_index(drop=True)
        stratum = strata[strata.release_building_group_id.eq(building_id)]
        if len(stratum) != 1:
            raise RuntimeError(f"Missing metric stratum for {building_id}")
        stratum = stratum.iloc[0]
        if building.observed_temperature_source.nunique() != 1:
            raise RuntimeError(f"Mixed temperature source within {building_id}")
        for seed in protocol["seeds"]:
            rng = np.random.default_rng(int(seed))
            chosen = rng.choice(len(building), size=PROFILE_COUNT, replace=False)
            sample = building.iloc[chosen].copy().reset_index(drop=True)
            if sample.robustness_unit_id.nunique() != PROFILE_COUNT:
                raise RuntimeError(f"Duplicate sampled unit: {building_id} {seed}")
            sample["composition_seed"] = int(seed)
            sample_rows.extend(sample.to_dict("records"))
            permutation = np.random.default_rng(int(seed) + 1000).permutation(PROFILE_COUNT)
            ordered = sample.iloc[permutation].reset_index(drop=True)
            for context, capacity in capacities.groupby("context", sort=False):
                if int(capacity.profiles.sum()) != PROFILE_COUNT:
                    raise RuntimeError(f"Zone capacity does not sum to 267: {context}")
                rows = []
                offset = 0
                for zone_row in capacity.itertuples(index=False):
                    local = ordered.iloc[offset : offset + int(zone_row.profiles)].copy()
                    local["zone_id"] = str(zone_row.zone_id)
                    local["assignment_order"] = np.arange(len(local), dtype=int)
                    rows.append(local)
                    offset += len(local)
                assigned = pd.concat(rows, ignore_index=True)
                assigned["profile_id"] = assigned.robustness_unit_id.astype(str)
                assigned["context"] = context
                assigned["source_building_id"] = building_id
                profiles[building_id, int(seed), context] = assigned[
                    ["profile_id", "zone_id", "demand_centre_c"]
                ].copy()
                assignment_rows.extend(
                    assigned[
                        [
                            "source_building_id",
                            "composition_seed",
                            "context",
                            "zone_id",
                            "assignment_order",
                            "profile_id",
                            "demand_centre_c",
                            "known_subject_id",
                            "observed_temperature_source",
                        ]
                    ].to_dict("records")
                )
                qa_rows.append(
                    {
                        "source_building_id": building_id,
                        "source_country": stratum.country,
                        "source_city": stratum.city,
                        "composition_seed": int(seed),
                        "context": context,
                        "profiles": len(assigned),
                        "zones": assigned.zone_id.nunique(),
                        "known_id_units": int(assigned.known_subject_id.sum()),
                        "known_id_category": "known_id_units" if assigned.known_subject_id.all() else "unknown_id_record_units",
                        "observed_temperature_source": stratum.observed_temperature_source,
                        "sample_reference_mean_c": float(assigned.demand_centre_c.mean()),
                        "sample_reference_sd_c": float(assigned.demand_centre_c.std(ddof=1)),
                        "sample_reference_min_c": float(assigned.demand_centre_c.min()),
                        "sample_reference_max_c": float(assigned.demand_centre_c.max()),
                    }
                )
    if len(profiles) != 10 * 3 * 4:
        raise RuntimeError("Expected 120 source-building/seed/context profile groups")
    return profiles, pd.DataFrame(sample_rows), pd.DataFrame(assignment_rows), pd.DataFrame(qa_rows)


def source_metadata(qa: pd.DataFrame, building_id: str, seed: int, context: str) -> dict[str, Any]:
    row = qa[
        qa.source_building_id.eq(building_id)
        & qa.composition_seed.eq(seed)
        & qa.context.eq(context)
    ]
    if len(row) != 1:
        raise RuntimeError("Source metadata key is not unique")
    record = row.iloc[0]
    return {
        "source_country": record.source_country,
        "source_city": record.source_city,
        "known_id_category": record.known_id_category,
        "observed_temperature_source": record.observed_temperature_source,
        "sample_reference_mean_c": float(record.sample_reference_mean_c),
        "sample_reference_sd_c": float(record.sample_reference_sd_c),
    }


def summarize_results(
    annual: pd.DataFrame,
    july: pd.DataFrame,
    qa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    annual_grouped = annual.groupby(
        ["cohort_id", "source_country", "source_city", "known_id_category", "observed_temperature_source"],
        dropna=False,
    ).agg(
        annual_design_cases=("service_fraction", "size"),
        annual_mean_selected_false_accepts=(
            "mean_selected_full_reaudit_status",
            lambda x: int(x.eq("mean_selected_false_accept").sum()),
        ),
        annual_both_feasible=("paired_feasibility_status", lambda x: int(x.eq("both_feasible").sum())),
        annual_mean_only_feasible=("paired_feasibility_status", lambda x: int(x.eq("mean_only_feasible").sum())),
        annual_neither_feasible=("paired_feasibility_status", lambda x: int(x.eq("neither_feasible").sum())),
        annual_candidate_false_accepts=("candidate_false_accept_count", "sum"),
        annual_candidate_false_rejects=("candidate_false_reject_count", "sum"),
    ).reset_index()
    july_grouped = july.groupby("cohort_id", dropna=False).agg(
        july_design_cases=("service_fraction", "size"),
        july_mean_selected_false_accepts=(
            "mean_selected_full_reaudit_status",
            lambda x: int(x.eq("mean_selected_false_accept").sum()),
        ),
        july_mean_positive_selections=("mean_selection_state", lambda x: int(x.eq("positive_action_selected").sum())),
        july_full_positive_selections=("full_selection_state", lambda x: int(x.eq("positive_action_selected").sum())),
        july_candidate_false_accepts=("candidate_false_accept_count", "sum"),
        july_candidate_false_rejects=("candidate_false_reject_count", "sum"),
    ).reset_index()
    per_source = annual_grouped.merge(july_grouped, on="cohort_id", validate="one_to_one")
    annual_heterogeneous = annual[annual["lambda"].eq(1.0)].copy()
    annual_heterogeneous["false_accept"] = annual_heterogeneous.mean_selected_full_reaudit_status.eq(
        "mean_selected_false_accept"
    )
    annual_heterogeneous["mean_selected"] = annual_heterogeneous.mean_selected_action_id.notna()
    annual_heterogeneous["selected_co2_difference_kg"] = (
        annual_heterogeneous.full_selected_operational_co2_kg
        - annual_heterogeneous.mean_selected_operational_co2_kg
    )
    annual_lambda1 = annual_heterogeneous.groupby("cohort_id", dropna=False).agg(
        annual_lambda1_design_cases=("service_fraction", "size"),
        annual_lambda1_mean_selections=("mean_selected", "sum"),
        annual_lambda1_mean_selected_false_accepts=("false_accept", "sum"),
        annual_lambda1_paired_selections=("full_minus_mean_selected_site_kwh", "count"),
        annual_lambda1_median_energy_correction_kwh=("full_minus_mean_selected_site_kwh", "median"),
        annual_lambda1_maximum_energy_correction_kwh=("full_minus_mean_selected_site_kwh", "max"),
        annual_lambda1_median_carbon_correction_kg=("selected_co2_difference_kg", "median"),
    ).reset_index()
    july_heterogeneous = july[july["lambda"].eq(1.0)].copy()
    july_heterogeneous["false_accept"] = july_heterogeneous.mean_selected_full_reaudit_status.eq(
        "mean_selected_false_accept"
    )
    july_heterogeneous["mean_selected"] = july_heterogeneous.mean_selected_action_id.notna()
    july_lambda1 = july_heterogeneous.groupby("cohort_id", dropna=False).agg(
        july_lambda1_design_cases=("service_fraction", "size"),
        july_lambda1_mean_selections=("mean_selected", "sum"),
        july_lambda1_mean_selected_false_accepts=("false_accept", "sum"),
        july_lambda1_paired_selections=("full_minus_mean_selected_response_kw", "count"),
        july_lambda1_median_response_correction_kw=("full_minus_mean_selected_response_kw", "median"),
        july_lambda1_minimum_response_correction_kw=("full_minus_mean_selected_response_kw", "min"),
    ).reset_index()
    per_source = per_source.merge(annual_lambda1, on="cohort_id", validate="one_to_one")
    per_source = per_source.merge(july_lambda1, on="cohort_id", validate="one_to_one")
    per_source["annual_selection_false_accept_fraction"] = (
        per_source.annual_mean_selected_false_accepts / per_source.annual_design_cases
    )
    per_source["july_selection_false_accept_fraction"] = (
        per_source.july_mean_selected_false_accepts / per_source.july_design_cases
    )
    per_source["annual_lambda1_selection_false_accept_fraction"] = (
        per_source.annual_lambda1_mean_selected_false_accepts / per_source.annual_lambda1_design_cases
    )
    per_source["annual_lambda1_conditional_false_accept_fraction"] = (
        per_source.annual_lambda1_mean_selected_false_accepts / per_source.annual_lambda1_mean_selections
    )
    per_source["july_lambda1_selection_false_accept_fraction"] = (
        per_source.july_lambda1_mean_selected_false_accepts / per_source.july_lambda1_design_cases
    )
    per_source["july_lambda1_conditional_false_accept_fraction"] = (
        per_source.july_lambda1_mean_selected_false_accepts / per_source.july_lambda1_mean_selections
    )
    strata = per_source.groupby(["observed_temperature_source", "known_id_category"], dropna=False).agg(
        source_buildings=("cohort_id", "size"),
        annual_design_cases=("annual_design_cases", "sum"),
        annual_mean_selected_false_accepts=("annual_mean_selected_false_accepts", "sum"),
        july_design_cases=("july_design_cases", "sum"),
        july_mean_selected_false_accepts=("july_mean_selected_false_accepts", "sum"),
        annual_lambda1_design_cases=("annual_lambda1_design_cases", "sum"),
        annual_lambda1_mean_selections=("annual_lambda1_mean_selections", "sum"),
        annual_lambda1_mean_selected_false_accepts=("annual_lambda1_mean_selected_false_accepts", "sum"),
        july_lambda1_design_cases=("july_lambda1_design_cases", "sum"),
        july_lambda1_mean_selections=("july_lambda1_mean_selections", "sum"),
        july_lambda1_mean_selected_false_accepts=("july_lambda1_mean_selected_false_accepts", "sum"),
    ).reset_index()
    strata["annual_selection_false_accept_fraction"] = (
        strata.annual_mean_selected_false_accepts / strata.annual_design_cases
    )
    strata["july_selection_false_accept_fraction"] = (
        strata.july_mean_selected_false_accepts / strata.july_design_cases
    )
    strata["annual_lambda1_design_false_accept_fraction"] = (
        strata.annual_lambda1_mean_selected_false_accepts / strata.annual_lambda1_design_cases
    )
    strata["annual_lambda1_conditional_false_accept_fraction"] = (
        strata.annual_lambda1_mean_selected_false_accepts / strata.annual_lambda1_mean_selections
    )
    strata["july_lambda1_design_false_accept_fraction"] = (
        strata.july_lambda1_mean_selected_false_accepts / strata.july_lambda1_design_cases
    )
    strata["july_lambda1_conditional_false_accept_fraction"] = (
        strata.july_lambda1_mean_selected_false_accepts / strata.july_lambda1_mean_selections
    )
    summary = {
        "completed": True,
        "interpretation": "Source-building-preserved archive-unit robustness on frozen trajectories; not local workforce prevalence or an observed intervention.",
        "source_buildings": int(per_source.cohort_id.nunique()),
        "known_id_source_buildings": int((per_source.known_id_category == "known_id_units").sum()),
        "unknown_id_record_source_buildings": int((per_source.known_id_category == "unknown_id_record_units").sum()),
        "operative_source_buildings": int((per_source.observed_temperature_source == "operative").sum()),
        "air_fallback_source_buildings": int((per_source.observed_temperature_source == "air_fallback").sum()),
        "sampled_archive_units_per_source_seed": PROFILE_COUNT,
        "annual_design_cases": len(annual),
        "july_design_cases": len(july),
        "annual_mean_selected_false_accepts": int(annual.mean_selected_full_reaudit_status.eq("mean_selected_false_accept").sum()),
        "july_mean_selected_false_accepts": int(july.mean_selected_full_reaudit_status.eq("mean_selected_false_accept").sum()),
        "annual_lambda1_design_cases": int(len(annual_heterogeneous)),
        "annual_lambda1_mean_selections": int(annual_heterogeneous.mean_selected.sum()),
        "annual_lambda1_mean_selected_false_accepts": int(annual_heterogeneous.false_accept.sum()),
        "annual_lambda1_design_false_accept_fraction": float(annual_heterogeneous.false_accept.mean()),
        "annual_lambda1_conditional_false_accept_fraction": float(
            annual_heterogeneous.false_accept.sum() / annual_heterogeneous.mean_selected.sum()
        ),
        "july_lambda1_design_cases": int(len(july_heterogeneous)),
        "july_lambda1_mean_selections": int(july_heterogeneous.mean_selected.sum()),
        "july_lambda1_mean_selected_false_accepts": int(july_heterogeneous.false_accept.sum()),
        "july_lambda1_design_false_accept_fraction": float(july_heterogeneous.false_accept.mean()),
        "july_lambda1_conditional_false_accept_fraction": float(
            july_heterogeneous.false_accept.sum() / july_heterogeneous.mean_selected.sum()
        ),
        "annual_candidate_false_accepts": int(annual.candidate_false_accept_count.sum()),
        "annual_candidate_false_rejects": int(annual.candidate_false_reject_count.sum()),
        "july_candidate_false_accepts": int(july.candidate_false_accept_count.sum()),
        "july_candidate_false_rejects": int(july.candidate_false_reject_count.sum()),
        "profile_groups": len(qa),
        "energyplus_runs": 0,
        "physical_parameter_changes": 0,
    }
    return per_source, strata, summary


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    input_qa = validate_inputs(data_root, input_root, output_root)
    annual_actions, dynamic_actions = load_libraries(data_root, input_root)
    precondition = run_precondition(data_root, input_root, annual_actions, dynamic_actions)
    write_json(output_root / "REPRODUCTION_PRECONDITION.json", precondition)
    print(json.dumps({"reproduction_precondition": precondition}, ensure_ascii=False), flush=True)
    if not precondition["passed"]:
        raise RuntimeError("Frozen annual/July reproduction precondition failed; new outcome scoring halted")
    if args.precondition_only:
        return

    protocol = json.loads((input_root / "PROTOCOL.json").read_text(encoding="utf-8"))
    profiles, samples, assignments, profile_qa = build_source_profiles(input_root, protocol)
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    samples.to_csv(output_root / "SOURCE_BUILDING_SAMPLES.csv.gz", index=False, compression=deterministic_gzip)
    assignments.to_csv(output_root / "SOURCE_BUILDING_ASSIGNMENTS.csv.gz", index=False, compression=deterministic_gzip)
    profile_qa.to_csv(output_root / "PROFILE_CONSTRUCTION_QA.csv", index=False)

    annual_anchor_rows = []
    anchors: dict[tuple[str, int, str], dict[str, Any]] = {}
    annual_score_frames = []
    for context in pd.read_csv(input_root / "ZONE_CAPACITIES.csv").context.drop_duplicates():
        local_actions = annual_actions[annual_actions.city.eq(context)]
        for seed in protocol["seeds"]:
            seed_actions = local_actions[local_actions.composition_seed.eq(seed)]
            for building_id in [item["release_building_group_id"] for item in protocol["eligible_source_buildings"]]:
                group = profiles[building_id, int(seed), context]
                anchor, _ = annual_anchor_for_profiles(input_root, context, group)
                anchors[building_id, int(seed), context] = anchor
                metadata = source_metadata(profile_qa, building_id, int(seed), context)
                annual_anchor_rows.append(
                    {
                        "source_building_id": building_id,
                        "composition_seed": int(seed),
                        "context": context,
                        "anchor_mean_c": anchor["mean_c"],
                        "anchor_fixed_tail_c": anchor["fixed_tail_c"],
                        "fixed_tail_count": len(anchor["fixed_ids"]),
                        "fixed_tail_profile_ids": "|".join(anchor["fixed_ids"]),
                        **metadata,
                    }
                )
                annual_score_frames.append(
                    score_annual_group(
                        input_root,
                        seed_actions,
                        building_id,
                        group,
                        anchor,
                        metadata,
                    )
                )
        print(json.dumps({"annual_context_scored": context}, ensure_ascii=False), flush=True)
    annual_scores = pd.concat(annual_score_frames, ignore_index=True)
    annual_decision, annual_eligibility = annual_decisions(annual_scores)
    pd.DataFrame(annual_anchor_rows).to_csv(output_root / "ANNUAL_ANCHORS.csv", index=False)
    annual_scores.to_csv(output_root / "ANNUAL_SCORES.csv.gz", index=False, compression=deterministic_gzip)
    annual_eligibility.to_csv(output_root / "ANNUAL_CANDIDATE_ELIGIBILITY.csv.gz", index=False, compression=deterministic_gzip)
    annual_decision.to_csv(output_root / "ANNUAL_DECISIONS.csv", index=False)

    dynamic_anchor_rows = []
    dynamic_score_frames = []
    for context in pd.read_csv(input_root / "ZONE_CAPACITIES.csv").context.drop_duplicates():
        local_actions = dynamic_actions[dynamic_actions.city.eq(context)]
        for seed in protocol["seeds"]:
            seed_actions = local_actions[local_actions.composition_seed.eq(seed)]
            if not len(seed_actions):
                raise RuntimeError(f"No July actions for {context} seed {seed}")
            month = int(seed_actions.event_month.iloc[0])
            day = int(seed_actions.event_day.iloc[0])
            for building_id in [item["release_building_group_id"] for item in protocol["eligible_source_buildings"]]:
                group = profiles[building_id, int(seed), context]
                annual_anchor = anchors[building_id, int(seed), context]
                dynamic_anchor = dynamic_anchor_for_profiles(
                    input_root,
                    context,
                    month,
                    day,
                    group,
                    set(annual_anchor["fixed_ids"]),
                )
                metadata = source_metadata(profile_qa, building_id, int(seed), context)
                dynamic_anchor_rows.append(
                    {
                        "source_building_id": building_id,
                        "composition_seed": int(seed),
                        "context": context,
                        "event_month": month,
                        "event_day": day,
                        **dynamic_anchor,
                        **metadata,
                    }
                )
                dynamic_score_frames.append(
                    score_dynamic_group(
                        input_root,
                        seed_actions,
                        building_id,
                        group,
                        annual_anchor,
                        dynamic_anchor,
                        metadata,
                    )
                )
        print(json.dumps({"july_context_scored": context}, ensure_ascii=False), flush=True)
    dynamic_scores = pd.concat(dynamic_score_frames, ignore_index=True)
    july_decision, july_eligibility = dynamic_decisions(dynamic_scores)
    pd.DataFrame(dynamic_anchor_rows).to_csv(output_root / "JULY_ANCHORS.csv", index=False)
    dynamic_scores.to_csv(output_root / "JULY_SCORES.csv.gz", index=False, compression=deterministic_gzip)
    july_eligibility.to_csv(output_root / "JULY_CANDIDATE_ELIGIBILITY.csv.gz", index=False, compression=deterministic_gzip)
    july_decision.to_csv(output_root / "JULY_DECISIONS.csv", index=False)

    if len(annual_decision) != protocol["annual_design_cases"]:
        raise RuntimeError(f"Annual design-case count {len(annual_decision)} != 960")
    if len(july_decision) != protocol["july_design_cases"]:
        raise RuntimeError(f"July design-case count {len(july_decision)} != 960")
    per_source, strata, summary = summarize_results(annual_decision, july_decision, profile_qa)
    per_source.to_csv(output_root / "SOURCE_BUILDING_RESULT_SUMMARY.csv", index=False)
    per_source.to_csv(output_root / "TABLE_S19_SOURCE_BUILDING_ROBUSTNESS.csv", index=False)
    strata.to_csv(output_root / "METRIC_AND_ID_STRATA_SUMMARY.csv", index=False)
    write_json(output_root / "SUMMARY.json", summary)
    write_json(
        output_root / "QA.json",
        {
            "all_checks_pass": True,
            "input_validation": input_qa,
            "reproduction_precondition": precondition,
            "annual_score_rows": len(annual_scores),
            "annual_candidate_eligibility_rows": len(annual_eligibility),
            "annual_design_cases": len(annual_decision),
            "july_score_rows": len(dynamic_scores),
            "july_candidate_eligibility_rows": len(july_eligibility),
            "july_design_cases": len(july_decision),
            "source_buildings": int(per_source.cohort_id.nunique()),
            "new_physical_runs": 0,
        },
    )
    write_json(
        output_root / "RUN_MANIFEST.json",
        {
            "program": str(Path(__file__).resolve()),
            "program_sha256": sha256(Path(__file__).resolve()),
            "data_root": str(data_root),
            "input_root": str(input_root),
            "output_root": str(output_root),
            "input_qa": input_qa,
            "outputs": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in sorted(output_root.iterdir())
                if path.is_file() and path.name != "RUN_MANIFEST.json"
            },
        },
    )
    print(json.dumps({"completed": True, **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
