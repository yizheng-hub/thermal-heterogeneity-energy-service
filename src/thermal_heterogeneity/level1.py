"""Complete deterministic Level-1 reproduction from minimized anonymous inputs.

The module deliberately never reads ``expected/`` products.  Those products are
reserved for independent regression checks after the scientific payload has
been regenerated.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .building.annual import energy_tolerance
from .config import resolve_asset
from .human.hierarchy import archive_scope, weighted_nested_variance_decomposition
from .human.shared_room_validation import threshold_table
from .io import write_json, write_table


SCORE_TOL_C = 1e-9
RESPONSE_TOL_KW = 1e-9
REPRESENTATIONS = ("mean_only", "quantile3", "quantile5", "full")
QUANTILE_APPROXIMATIONS = ("mean_only", "quantile3", "quantile5")
MARGIN_STRATEGIES = (
    "none",
    "pooled_q90",
    "pooled_q95",
    "pooled_q99",
    "groupmax_q90",
    "training_max",
)
LOCAL_FEEDBACK_SIZES = (0, 5, 10, 20, 40)
LOCAL_REPETITIONS = 200
LOCAL_BOOTSTRAPS = 2000
LOCAL_MIN_UNITS = 60
LOCAL_PRIOR_STRENGTH = 10.0
PREFERENCE_CLASSES = ("warmer", "no change", "cooler")
PROBABILITY_COLUMNS = ("p_warmer", "p_no_change", "p_cooler")


def _read(config: dict[str, Any], asset_id: str) -> pd.DataFrame:
    path = resolve_asset(config, asset_id)
    if not path.is_file():
        raise FileNotFoundError(f"missing Level-1 asset {asset_id}: {path}")
    return pd.read_csv(path, low_memory=False)


def _choose_minimum(pool: pd.DataFrame, action: str) -> pd.Series | None:
    eligible = pool.loc[pool["service_eligible"]].copy()
    if eligible.empty:
        return None
    minimum = float(eligible["site_kwh"].min())
    tied = eligible.loc[
        eligible["site_kwh"].apply(
            lambda value: abs(float(value) - minimum)
            <= energy_tolerance(float(value), minimum)
        )
    ]
    return tied.sort_values(action, kind="mergesort").iloc[0]


def _choose_response(pool: pd.DataFrame) -> pd.Series | None:
    eligible = pool.loc[
        pool["service_eligible"]
        & pool["event_average_reduction_kw"].gt(RESPONSE_TOL_KW)
    ].copy()
    if eligible.empty:
        return None
    maximum = float(eligible["event_average_reduction_kw"].max())
    tied = eligible.loc[
        eligible["event_average_reduction_kw"].ge(maximum - RESPONSE_TOL_KW)
    ]
    return tied.sort_values("event_action_id", kind="mergesort").iloc[0]


def result1(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    profiles = _read(config, "human_profiles")
    scope = pd.DataFrame([archive_scope(profiles)])
    variance = weighted_nested_variance_decomposition(
        profiles,
        value="reference_g050_c",
        weight="unit_equal_weight",
        levels=(
            "country",
            "city",
            "release_building_group_id",
            "release_unit_id",
        ),
    )

    support_input = _read(config, "city_season_support")
    coordinate = (
        support_input.groupby(["country", "city"], sort=True, as_index=False)
        .agg(
            map_latitude_0p5deg=("map_latitude_0p5deg", "first"),
            map_longitude_0p5deg=("map_longitude_0p5deg", "first"),
        )
    )
    rows: list[dict[str, Any]] = []
    quantiles = (0.1, 0.5, 0.9)
    for key, group in profiles.groupby(["country", "city", "season"], sort=True):
        country, city, season = key
        row: dict[str, Any] = {
            "country": country,
            "city": city,
            "season": season,
            "records": len(group),
            "analysis_units": group["release_unit_id"].nunique(),
            "buildings": group["release_building_group_id"].nunique(),
            "sources": group["release_source_group_id"].nunique(),
        }
        for column, label in (
            ("reference_g033_c", "g033"),
            ("reference_g050_c", "g050"),
            ("reference_g067_c", "g067"),
        ):
            values = group[column].dropna()
            q = values.quantile(quantiles).to_numpy(dtype=float)
            row.update(
                {
                    f"{label}_p10_c": q[0],
                    f"{label}_p50_c": q[1],
                    f"{label}_p90_c": q[2],
                    f"{label}_p10_p90_width_c": q[2] - q[0],
                }
            )
        rows.append(row)
    city_season = pd.DataFrame(rows).merge(
        coordinate, on=["country", "city"], how="left", validate="many_to_one"
    )
    policy = _read(config, "policy_estimates")
    policy_support = _read(config, "policy_support")
    return {
        "result1_scope": scope,
        "result1_variance": variance,
        "result1_city_season": city_season,
        "result1_policy_coverage": policy,
        "result1_policy_support": policy_support,
    }


def result2(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    windows = _read(config, "shared_room_windows")
    eligible = windows.loc[windows["eligible"]].copy()
    summaries = []
    threshold_rows = []
    for dataset, group in eligible.groupby("dataset_code", sort=True):
        summaries.append(
            {
                "dataset_code": dataset,
                "eligible_windows": len(group),
                "votes": int(group["vote_count"].sum()),
                "disagreement_windows": int(group["disagreement_present"].sum()),
                "opposing_direction_windows": int(
                    group["simultaneous_warmer_cooler"].sum()
                ),
                "exact_cancellation_windows": int(
                    (group["compressed_burden"].eq(0) & group["full_burden"].gt(0)).sum()
                ),
                "mean_jensen_gap": float(group["jensen_gap"].mean()),
            }
        )
        local = threshold_table(
            group, tuple(float(x) for x in config["science"]["direct_vote_thresholds"])
        )
        local.insert(0, "dataset_code", dataset)
        threshold_rows.append(local)
    return {
        "result2_windows": eligible,
        "result2_summary": pd.DataFrame(summaries),
        "result2_thresholds": pd.concat(threshold_rows, ignore_index=True),
    }


def annual_decisions(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _read(config, "annual_scores")
    anchors = _read(config, "annual_anchors").rename(
        columns={"composition_seed": "seed"}
    )
    anchor_index = anchors.set_index(["population_scenario", "city", "seed"])
    fractions = tuple(float(x) for x in config["science"]["service_fractions"])
    decision_rows: list[dict[str, Any]] = []
    eligibility_rows: list[pd.DataFrame] = []
    keys = ["population_scenario", "city", "composition_seed", "delta_c", "lambda"]
    for key, local in scores.groupby(keys, sort=True):
        population, city, seed, delta, spread = key
        anchor = anchor_index.loc[(population, city, int(seed))]
        for fraction in fractions:
            mean_budget = float(anchor["anchor_mean_c"]) * fraction
            tail_budget = float(anchor["anchor_fixed_tail_c"]) * fraction
            pools: dict[str, pd.DataFrame] = {}
            choices: dict[str, pd.Series | None] = {}
            for representation in ("mean", "full"):
                pool = local.loc[local["representation"].eq(representation)].copy()
                pool["mean_budget_c"] = mean_budget
                pool["fixed_tail_budget_c"] = tail_budget
                pool["service_fraction"] = fraction
                pool["service_eligible"] = pool["mean_c"].le(
                    mean_budget + SCORE_TOL_C
                ) & pool["fixed_tail_c"].le(tail_budget + SCORE_TOL_C)
                pools[representation] = pool.set_index("physical_action_id", drop=False)
                choices[representation] = _choose_minimum(pool, "physical_action_id")
                eligibility_rows.append(pool)
            if not pools["mean"].index.sort_values().equals(
                pools["full"].index.sort_values()
            ):
                raise ValueError(f"annual action support mismatch: {key}")
            mean_ok = pools["mean"]["service_eligible"]
            full_ok = pools["full"]["service_eligible"]
            mean_choice, full_choice = choices["mean"], choices["full"]
            mean_available, full_available = mean_choice is not None, full_choice is not None
            if mean_available and full_available:
                paired_status = "both_feasible"
                same = str(mean_choice["physical_action_id"]) == str(
                    full_choice["physical_action_id"]
                )
                selection = "same_action" if same else "different_action_energy_nontie"
            elif mean_available:
                paired_status, selection = "mean_only_feasible", "not_applicable"
            elif full_available:
                paired_status, selection = "full_only_feasible", "not_applicable"
            else:
                paired_status, selection = "neither_feasible", "not_applicable"
            row: dict[str, Any] = {
                "population_scenario": population,
                "city": city,
                "composition_seed": int(seed),
                "delta_c": float(delta),
                "lambda": float(spread),
                "service_fraction": fraction,
                "annual_mean_budget_c": mean_budget,
                "annual_fixed_tail_budget_c": tail_budget,
                "mean_eligible_action_count": int(mean_ok.sum()),
                "full_eligible_action_count": int(full_ok.sum()),
                "candidate_false_accept_count": int((mean_ok & ~full_ok).sum()),
                "candidate_false_reject_count": int((~mean_ok & full_ok).sum()),
                "paired_feasibility_status": paired_status,
                "selection_status": selection,
            }
            for label, choice in (("mean", mean_choice), ("full", full_choice)):
                for output_name, source_name in (
                    ("action_id", "physical_action_id"),
                    ("site_kwh", "site_kwh"),
                    ("electric_kwh", "electric_kwh"),
                    ("gas_kwh_fuel", "gas_kwh_fuel"),
                    ("operational_co2_kg", "operational_co2_kg"),
                    ("mean_c", "mean_c"),
                    ("fixed_tail_c", "fixed_tail_c"),
                ):
                    row[f"{label}_selected_{output_name}"] = (
                        None if choice is None else choice[source_name]
                    )
            if mean_choice is None:
                row["mean_selected_full_eligible"] = None
                row["mean_selected_full_reaudit_status"] = "no_mean_selection"
            else:
                action_id = mean_choice["physical_action_id"]
                full_eligible = bool(full_ok.loc[action_id])
                row["mean_selected_full_eligible"] = full_eligible
                row["mean_selected_full_reaudit_status"] = (
                    "mean_selected_full_eligible"
                    if full_eligible
                    else "mean_selected_false_accept"
                )
            decision_rows.append(row)
    return pd.DataFrame(decision_rows), pd.concat(eligibility_rows, ignore_index=True)


def _dynamic_eligible(pool: pd.DataFrame) -> pd.Series:
    checks = []
    for window in ("event", "recovery"):
        for trajectory in ("baseline", "perturbed"):
            checks.append(
                pool[f"{window}_{trajectory}_mean_c"].le(
                    pool[f"{window}_mean_budget_c"] + SCORE_TOL_C
                )
            )
            checks.append(
                pool[f"{window}_{trajectory}_fixed_tail_c"].le(
                    pool[f"{window}_fixed_tail_budget_c"] + SCORE_TOL_C
                )
            )
    return pd.concat(checks, axis=1).all(axis=1)


def dynamic_decisions(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = _read(config, "dynamic_scores")
    anchors = _read(config, "dynamic_anchors").rename(
        columns={"composition_seed": "seed"}
    )
    anchor_index = anchors.pivot_table(
        index=["population_scenario", "city", "seed", "event_month"],
        columns="window",
        values=["anchor_mean_c", "anchor_fixed_tail_c"],
        aggfunc="first",
    )
    fractions = tuple(float(x) for x in config["science"]["service_fractions"])
    decisions: list[dict[str, Any]] = []
    eligibilities: list[pd.DataFrame] = []
    keys = [
        "population_scenario",
        "city",
        "composition_seed",
        "event_month",
        "delta_c",
        "lambda",
    ]
    for key, local in scores.groupby(keys, sort=True):
        population, city, seed, month, delta, spread = key
        anchor = anchor_index.loc[(population, city, int(seed), int(month))]
        for fraction in fractions:
            budgets = {
                "event_mean_budget_c": float(anchor[("anchor_mean_c", "event")])
                * fraction,
                "event_fixed_tail_budget_c": float(
                    anchor[("anchor_fixed_tail_c", "event")]
                )
                * fraction,
                "recovery_mean_budget_c": float(
                    anchor[("anchor_mean_c", "recovery")]
                )
                * fraction,
                "recovery_fixed_tail_budget_c": float(
                    anchor[("anchor_fixed_tail_c", "recovery")]
                )
                * fraction,
            }
            pools: dict[str, pd.DataFrame] = {}
            selected: dict[str, pd.Series | None] = {}
            for representation in ("mean", "full"):
                pool = local.loc[local["representation"].eq(representation)].copy()
                for name, value in budgets.items():
                    pool[name] = value
                pool["service_fraction"] = fraction
                pool["service_eligible"] = _dynamic_eligible(pool)
                pools[representation] = pool.set_index("event_action_id", drop=False)
                selected[representation] = _choose_response(pool)
                eligibilities.append(pool)
            if not pools["mean"].index.sort_values().equals(
                pools["full"].index.sort_values()
            ):
                raise ValueError(f"dynamic action support mismatch: {key}")
            mean_ok = pools["mean"]["service_eligible"]
            full_ok = pools["full"]["service_eligible"]
            row: dict[str, Any] = {
                "population_scenario": population,
                "city": city,
                "composition_seed": int(seed),
                "event_month": int(month),
                "delta_c": float(delta),
                "lambda": float(spread),
                "service_fraction": fraction,
                **budgets,
                "mean_service_eligible_action_count": int(mean_ok.sum()),
                "full_service_eligible_action_count": int(full_ok.sum()),
                "mean_positive_action_count": int(
                    (mean_ok & pools["mean"]["event_average_reduction_kw"].gt(RESPONSE_TOL_KW)).sum()
                ),
                "full_positive_action_count": int(
                    (full_ok & pools["full"]["event_average_reduction_kw"].gt(RESPONSE_TOL_KW)).sum()
                ),
                "candidate_false_accept_count": int((mean_ok & ~full_ok).sum()),
                "candidate_false_reject_count": int((~mean_ok & full_ok).sum()),
            }
            for label in ("mean", "full"):
                choice = selected[label]
                for output_name, source_name in (
                    ("action_id", "event_action_id"),
                    ("event_average_reduction_kw", "event_average_reduction_kw"),
                    ("recovery_net_extra_electric_kwh", "recovery_net_extra_electric_kwh"),
                    ("recovery_peak_hour_extra_kw", "recovery_peak_hour_extra_kw"),
                    ("event_plus48h_net_electric_saving_kwh", "event_plus48h_net_electric_saving_kwh"),
                    ("event_plus48h_gas_change_kwh_fuel", "event_plus48h_gas_change_kwh_fuel"),
                ):
                    row[f"{label}_selected_{output_name}"] = (
                        None if choice is None else choice[source_name]
                    )
                eligible_count = int(row[f"{label}_service_eligible_action_count"])
                if choice is not None:
                    state = "positive_action_selected"
                elif eligible_count:
                    state = "eligible_but_no_positive_response"
                else:
                    state = "no_service_eligible_action"
                row[f"{label}_selection_state"] = state
            mean_choice = selected["mean"]
            if mean_choice is None:
                row["mean_selected_full_reaudit_status"] = "no_mean_selection"
                row["mean_selected_full_eligible"] = None
            else:
                action_id = mean_choice["event_action_id"]
                full_eligible = bool(full_ok.loc[action_id])
                row["mean_selected_full_eligible"] = full_eligible
                row["mean_selected_full_reaudit_status"] = (
                    "mean_selected_full_eligible"
                    if full_eligible
                    else "mean_selected_false_accept"
                )
            decisions.append(row)
    return pd.DataFrame(decisions), pd.concat(eligibilities, ignore_index=True)


def _population_sd(config: dict[str, Any]) -> pd.DataFrame:
    assignments = _read(config, "population_assignments")
    assignments = assignments.loc[assignments["allocation"].eq("random_mixing")].copy()
    zone_keys = ["scenario_id", "composition_seed", "context", "zone_id"]
    assignments["_deviation"] = assignments["demand_centre_c"] - assignments.groupby(
        zone_keys, sort=True
    )["demand_centre_c"].transform("mean")
    result = (
        assignments.groupby(
            ["scenario_id", "composition_seed", "context"], sort=True
        )["_deviation"]
        .apply(lambda x: float(np.sqrt(np.mean(np.square(x.to_numpy(dtype=float))))))
        .rename("lambda1_sd_c")
        .reset_index()
        .rename(
            columns={
                "scenario_id": "population_scenario",
                "composition_seed": "seed",
                "context": "city",
            }
        )
    )
    return result


def phase_map(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = _read(config, "representation_scores")
    sds = _population_sd(config).set_index(["population_scenario", "city", "seed"])
    fractions = tuple(float(x) for x in config["science"]["service_fractions"])
    decisions: list[dict[str, Any]] = []
    approximations: list[dict[str, Any]] = []
    eligibility_rows: list[pd.DataFrame] = []
    keys = ["population_scenario", "city", "seed", "allocation", "lambda"]
    for key, local in scores.groupby(keys, sort=True):
        population, city, seed, allocation, spread = key
        if allocation != "random_mixing":
            continue
        anchor_mean = float(local["common25_mean_anchor_c"].iloc[0])
        anchor_tail = float(local["common25_fixed_tail_anchor_c"].iloc[0])
        full_source = local.loc[local["representation"].eq("full")]
        best_required = float(
            np.maximum(
                full_source["mean_c"] / anchor_mean,
                full_source["fixed_tail_c"] / anchor_tail,
            ).min()
        )
        heterogeneity = float(sds.loc[(population, city, int(seed)), "lambda1_sd_c"])
        heterogeneity *= float(spread)
        for fraction in fractions:
            mean_budget = fraction * anchor_mean
            tail_budget = fraction * anchor_tail
            pools: dict[str, pd.DataFrame] = {}
            selected: dict[str, pd.Series | None] = {}
            for representation in REPRESENTATIONS:
                pool = local.loc[local["representation"].eq(representation)].copy()
                pool["service_fraction"] = fraction
                pool["mean_budget_c"] = mean_budget
                pool["fixed_tail_budget_c"] = tail_budget
                pool["service_eligible"] = pool["mean_c"].le(
                    mean_budget + SCORE_TOL_C
                ) & pool["fixed_tail_c"].le(tail_budget + SCORE_TOL_C)
                pools[representation] = pool.set_index("physical_action_id", drop=False)
                selected[representation] = _choose_minimum(pool, "physical_action_id")
            full_pool = pools["full"]
            full_selected = selected["full"]
            row: dict[str, Any] = {
                "population_scenario": population,
                "city": city,
                "seed": int(seed),
                "allocation": allocation,
                "lambda": float(spread),
                "heterogeneity_sd_c": heterogeneity,
                "service_fraction": fraction,
                "mean_budget_c": mean_budget,
                "fixed_tail_budget_c": tail_budget,
                "best_full_required_fraction": best_required,
                "full_library_service_margin_fraction": fraction - best_required,
                "candidate_count": len(full_pool),
                "full_feasible_candidates": int(full_pool["service_eligible"].sum()),
            }
            for representation in REPRESENTATIONS:
                pool = pools[representation]
                chosen = selected[representation]
                false_accept = pool["service_eligible"] & ~full_pool["service_eligible"]
                false_reject = ~pool["service_eligible"] & full_pool["service_eligible"]
                row[f"{representation}_feasible_candidates"] = int(
                    pool["service_eligible"].sum()
                )
                row[f"{representation}_candidate_false_accept_count"] = int(
                    false_accept.sum()
                )
                row[f"{representation}_candidate_false_reject_count"] = int(
                    false_reject.sum()
                )
                row[f"{representation}_eligibility_exact"] = bool(
                    pool["service_eligible"].equals(full_pool["service_eligible"])
                )
                action_id = None if chosen is None else str(chosen["physical_action_id"])
                row[f"{representation}_selected_action_id"] = action_id
                row[f"{representation}_selected_site_kwh"] = (
                    None if chosen is None else float(chosen["site_kwh"])
                )
                if chosen is None:
                    row[f"{representation}_selected_full_eligible"] = None
                    row[f"{representation}_selected_full_mean_excess_c"] = None
                    row[f"{representation}_selected_full_tail_excess_c"] = None
                else:
                    full_reaudit = full_pool.loc[action_id]
                    row[f"{representation}_selected_full_eligible"] = bool(
                        full_reaudit["service_eligible"]
                    )
                    row[f"{representation}_selected_full_mean_excess_c"] = max(
                        float(full_reaudit["mean_c"] - mean_budget), 0.0
                    )
                    row[f"{representation}_selected_full_tail_excess_c"] = max(
                        float(full_reaudit["fixed_tail_c"] - tail_budget), 0.0
                    )
                if chosen is not None and full_selected is not None:
                    same = action_id == str(full_selected["physical_action_id"])
                    correction = float(full_selected["site_kwh"] - chosen["site_kwh"])
                    row[f"{representation}_selected_action_same_as_full"] = same
                    row[f"{representation}_selected_action_nontie_change"] = (
                        (not same)
                        and abs(correction)
                        > energy_tolerance(
                            float(full_selected["site_kwh"]), float(chosen["site_kwh"])
                        )
                    )
                    row[f"{representation}_energy_correction_kwh"] = correction
                    row[f"{representation}_energy_correction_pct_of_rep"] = (
                        100 * correction / float(chosen["site_kwh"])
                    )
                else:
                    row[f"{representation}_selected_action_same_as_full"] = None
                    row[f"{representation}_selected_action_nontie_change"] = None
                    row[f"{representation}_energy_correction_kwh"] = None
                    row[f"{representation}_energy_correction_pct_of_rep"] = None
                mean_error = (
                    pool["mean_c"].sort_index() - full_pool["mean_c"].sort_index()
                ).abs()
                tail_error = (
                    pool["fixed_tail_c"].sort_index()
                    - full_pool["fixed_tail_c"].sort_index()
                ).abs()
                approximations.append(
                    {
                        "population_scenario": population,
                        "city": city,
                        "seed": int(seed),
                        "lambda": float(spread),
                        "service_fraction": fraction,
                        "representation": representation,
                        "candidate_mean_mae_c": float(mean_error.mean()),
                        "candidate_mean_max_abs_error_c": float(mean_error.max()),
                        "candidate_tail_mae_c": float(tail_error.mean()),
                        "candidate_tail_max_abs_error_c": float(tail_error.max()),
                        "eligibility_disagreement_rate": float(
                            (pool["service_eligible"] != full_pool["service_eligible"]).mean()
                        ),
                        "candidate_false_accept_count": int(false_accept.sum()),
                        "candidate_false_reject_count": int(false_reject.sum()),
                    }
                )
                eligibility = pool.reset_index(drop=True)[
                    ["physical_action_id", "service_eligible"]
                ].copy()
                eligibility["population_scenario"] = population
                eligibility["city"] = city
                eligibility["seed"] = int(seed)
                eligibility["lambda"] = float(spread)
                eligibility["service_fraction"] = fraction
                eligibility["representation"] = representation
                eligibility["full_service_eligible"] = eligibility[
                    "physical_action_id"
                ].map(full_pool["service_eligible"])
                eligibility_rows.append(eligibility)
            if full_selected is None:
                row["classification"] = "finite-library-unresolved"
            elif bool(row["mean_only_eligibility_exact"]) and bool(
                row["mean_only_selected_action_same_as_full"]
            ):
                row["classification"] = "mean-safe"
            else:
                row["classification"] = "distribution-required"
            decisions.append(row)
    return (
        pd.DataFrame(decisions),
        pd.DataFrame(approximations),
        pd.concat(eligibility_rows, ignore_index=True),
    )


def _higher_quantile(values: pd.Series, quantile: float) -> float:
    return float(np.quantile(values.to_numpy(dtype=float), quantile, method="higher"))


def _paired_representation_scores(scores: pd.DataFrame) -> pd.DataFrame:
    keys = ["population_scenario", "city", "seed", "lambda", "physical_action_id"]
    full = scores.loc[scores["representation"].eq("full"), keys + ["mean_c", "fixed_tail_c"]].rename(
        columns={"mean_c": "full_mean_c", "fixed_tail_c": "full_tail_c"}
    )
    pieces = []
    fixed = ["site_kwh", "common25_mean_anchor_c", "common25_fixed_tail_anchor_c"]
    for representation in QUANTILE_APPROXIMATIONS:
        local = scores.loc[
            scores["representation"].eq(representation),
            keys + fixed + ["mean_c", "fixed_tail_c"],
        ].copy()
        local["representation"] = representation
        local = local.merge(full, on=keys, how="inner", validate="one_to_one")
        local["mean_under_c"] = (local["full_mean_c"] - local["mean_c"]).clip(lower=0)
        local["tail_under_c"] = (local["full_tail_c"] - local["fixed_tail_c"]).clip(lower=0)
        pieces.append(local)
    return pd.concat(pieces, ignore_index=True)


def _margin(train: pd.DataFrame, representation: str, strategy: str) -> tuple[float, float]:
    local = train.loc[train["representation"].eq(representation)]
    if strategy == "none":
        return 0.0, 0.0
    if strategy.startswith("pooled_q"):
        q = float(strategy.removeprefix("pooled_q")) / 100
        return _higher_quantile(local["mean_under_c"], q), _higher_quantile(
            local["tail_under_c"], q
        )
    if strategy == "groupmax_q90":
        maxima = local.groupby(
            ["population_scenario", "city", "seed"], sort=True
        )[["mean_under_c", "tail_under_c"]].max()
        return _higher_quantile(maxima["mean_under_c"], 0.9), _higher_quantile(
            maxima["tail_under_c"], 0.9
        )
    if strategy == "training_max":
        return float(local["mean_under_c"].max()), float(local["tail_under_c"].max())
    raise ValueError(strategy)


def quantile_certification(
    config: dict[str, Any], phase: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = _paired_representation_scores(_read(config, "representation_scores"))
    decision_keys = ["population_scenario", "city", "seed", "lambda"]
    reference_columns = decision_keys + [
        "service_fraction",
        "mean_budget_c",
        "fixed_tail_budget_c",
        "full_feasible_candidates",
        "full_selected_action_id",
        "full_selected_site_kwh",
    ]
    reference = phase[reference_columns]
    folds: list[tuple[str, str, pd.Series, pd.Series]] = []
    for city, seed in sorted(
        {(str(x.city), int(x.seed)) for x in paired[["city", "seed"]].drop_duplicates().itertuples(index=False)}
    ):
        test = paired["city"].eq(city) & paired["seed"].eq(seed)
        folds.append(("leave_one_context_seed_out", f"{city}__{seed}", ~test, test))
    for population in sorted(paired["population_scenario"].unique()):
        test = paired["population_scenario"].eq(population)
        folds.append(("leave_one_population_out", str(population), ~test, test))

    accumulator: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    margins: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for scheme, fold_id, train_mask, test_mask in folds:
        train, test = paired.loc[train_mask], paired.loc[test_mask]
        test_index = test[decision_keys].drop_duplicates()
        local_reference = reference.merge(
            test_index, on=decision_keys, how="inner", validate="many_to_one"
        )
        for representation in QUANTILE_APPROXIMATIONS:
            for strategy in MARGIN_STRATEGIES:
                mean_margin, tail_margin = _margin(train, representation, strategy)
                margins.append(
                    {
                        "split_scheme": scheme,
                        "fold_id": fold_id,
                        "representation": representation,
                        "strategy": strategy,
                        "train_candidate_score_rows": int(
                            train["representation"].eq(representation).sum()
                        ),
                        "test_candidate_score_rows": int(
                            test["representation"].eq(representation).sum()
                        ),
                        "mean_margin_c": mean_margin,
                        "tail_margin_c": tail_margin,
                    }
                )
                base = test.loc[test["representation"].eq(representation)].copy()
                expanded = base.merge(
                    local_reference, on=decision_keys, how="inner", validate="many_to_many"
                )
                expanded["full_service_eligible"] = expanded["full_mean_c"].le(
                    expanded["mean_budget_c"] + SCORE_TOL_C
                ) & expanded["full_tail_c"].le(
                    expanded["fixed_tail_budget_c"] + SCORE_TOL_C
                )
                expanded["certified_eligible"] = (
                    expanded["mean_c"] + mean_margin
                ).le(expanded["mean_budget_c"] + SCORE_TOL_C) & (
                    expanded["fixed_tail_c"] + tail_margin
                ).le(expanded["fixed_tail_budget_c"] + SCORE_TOL_C)
                key = (scheme, representation, strategy)
                acc = accumulator[key]
                certified = expanded["certified_eligible"]
                full = expanded["full_service_eligible"]
                acc["candidate_rows"] += len(expanded)
                acc["certified_candidates"] += int(certified.sum())
                acc["full_eligible_candidates"] += int(full.sum())
                acc["true_positive_candidates"] += int((certified & full).sum())
                acc["false_accept_candidates"] += int((certified & ~full).sum())
                acc["false_reject_candidates"] += int((~certified & full).sum())
                group_keys = decision_keys + ["service_fraction"]
                for decision_key, group in expanded.groupby(group_keys, sort=True):
                    eligible = group.loc[group["certified_eligible"]].copy()
                    if eligible.empty:
                        choice = None
                    else:
                        minimum = float(eligible["site_kwh"].min())
                        choice = eligible.loc[
                            eligible["site_kwh"].apply(
                                lambda value: abs(float(value) - minimum)
                                <= energy_tolerance(float(value), minimum)
                            )
                        ].sort_values("physical_action_id", kind="mergesort").iloc[0]
                    full_available = bool(int(group["full_feasible_candidates"].iloc[0]) > 0)
                    row = dict(zip(group_keys, decision_key, strict=True))
                    row.update(
                        {
                            "split_scheme": scheme,
                            "fold_id": fold_id,
                            "representation": representation,
                            "strategy": strategy,
                            "mean_margin_c": mean_margin,
                            "tail_margin_c": tail_margin,
                            "full_available": full_available,
                            "certified_available": choice is not None,
                            "full_selected_action_id": group["full_selected_action_id"].iloc[0],
                            "full_selected_site_kwh": group["full_selected_site_kwh"].iloc[0],
                        }
                    )
                    if choice is None:
                        row.update(
                            {
                                "certified_selected_action_id": None,
                                "certified_selected_site_kwh": None,
                                "selected_full_eligible": None,
                                "selected_false_accept": None,
                                "selected_same_as_full": None,
                                "selected_nontie_change": None,
                                "conservative_energy_penalty_kwh": None,
                            }
                        )
                    else:
                        ref_id = row["full_selected_action_id"]
                        ref_energy = row["full_selected_site_kwh"]
                        action_id = str(choice["physical_action_id"])
                        energy = float(choice["site_kwh"])
                        full_ok = bool(choice["full_service_eligible"])
                        same = full_available and action_id == str(ref_id)
                        delta = None if not full_available else energy - float(ref_energy)
                        row.update(
                            {
                                "certified_selected_action_id": action_id,
                                "certified_selected_site_kwh": energy,
                                "selected_full_eligible": full_ok,
                                "selected_false_accept": not full_ok,
                                "selected_same_as_full": same,
                                "selected_nontie_change": None
                                if delta is None
                                else (
                                    (not same)
                                    and abs(delta)
                                    > energy_tolerance(energy, float(ref_energy))
                                ),
                                "conservative_energy_penalty_kwh": delta
                                if full_available and full_ok
                                else None,
                            }
                        )
                    decisions.append(row)
    decision_frame = pd.DataFrame(decisions)
    summary_rows = []
    for key, acc in sorted(accumulator.items()):
        scheme, representation, strategy = key
        local = decision_frame.loc[
            decision_frame["split_scheme"].eq(scheme)
            & decision_frame["representation"].eq(representation)
            & decision_frame["strategy"].eq(strategy)
        ]
        full_available = local.loc[local["full_available"]]
        available = local.loc[local["certified_available"]]
        comparable = local.loc[local["full_available"] & local["certified_available"]]
        conservative = comparable.loc[comparable["selected_full_eligible"].eq(True)]
        energy = conservative["conservative_energy_penalty_kwh"].dropna()
        certified = int(acc["certified_candidates"])
        full = int(acc["full_eligible_candidates"])
        true_positive = int(acc["true_positive_candidates"])
        row = {
            "split_scheme": scheme,
            "representation": representation,
            "strategy": strategy,
            **{name: int(value) for name, value in acc.items()},
            "false_accept_rate_among_certified": (
                np.nan if not certified else acc["false_accept_candidates"] / certified
            ),
            "false_accept_rate_all_candidates": acc["false_accept_candidates"]
            / acc["candidate_rows"],
            "false_reject_rate_among_full_eligible": (
                np.nan if not full else acc["false_reject_candidates"] / full
            ),
            "retained_full_eligible_fraction": (
                np.nan if not full else true_positive / full
            ),
            "certified_library_fraction": certified / acc["candidate_rows"],
            "decision_rows": len(local),
            "full_available_decisions": len(full_available),
            "certified_available_decisions": len(available),
            "certified_coverage_of_full_available": (
                np.nan if full_available.empty else len(comparable) / len(full_available)
            ),
            "unresolved_despite_full_available": int(
                (local["full_available"] & ~local["certified_available"]).sum()
            ),
            "selected_false_accept_decisions": int(
                available["selected_false_accept"].eq(True).sum()
            ),
            "selected_same_as_full_count": int(
                comparable["selected_same_as_full"].eq(True).sum()
            ),
            "selected_nontie_change_count": int(
                comparable["selected_nontie_change"].eq(True).sum()
            ),
            "conservative_comparable_decisions": len(conservative),
            "energy_penalty_median_kwh": np.nan if energy.empty else float(energy.median()),
            "energy_penalty_p95_kwh": np.nan
            if energy.empty
            else float(energy.quantile(0.95)),
            "energy_penalty_max_kwh": np.nan if energy.empty else float(energy.max()),
        }
        row["selected_false_accept_rate_among_certified"] = (
            np.nan
            if available.empty
            else row["selected_false_accept_decisions"] / len(available)
        )
        row["selected_same_as_full_rate"] = (
            np.nan
            if comparable.empty
            else row["selected_same_as_full_count"] / len(comparable)
        )
        row["selected_nontie_change_rate"] = (
            np.nan
            if comparable.empty
            else row["selected_nontie_change_count"] / len(comparable)
        )
        row["gate_zero_selected_false_accept"] = (
            row["selected_false_accept_decisions"] == 0
        )
        row["gate_retention_at_least_50pct"] = row[
            "retained_full_eligible_fraction"
        ] >= 0.5
        row["gate_decision_coverage_at_least_80pct"] = row[
            "certified_coverage_of_full_available"
        ] >= 0.8
        row["passes_predefined_certification_gate"] = bool(
            row["gate_zero_selected_false_accept"]
            and row["gate_retention_at_least_50pct"]
            and row["gate_decision_coverage_at_least_80pct"]
        )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(margins)


def _unit_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("release_unit_id")["release_record_id"].transform("count")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.sum()


def _preference_score(
    observed: np.ndarray, probability: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Return normalized ordinal RPS and tail-share MAE, both dimensionless.

    The historical output key ``crps`` is retained for frozen file stability.
    RPS is divided by the two cumulative class boundaries for three classes.
    """
    lookup = {name: index for index, name in enumerate(PREFERENCE_CLASSES)}
    index = np.array([lookup[str(value)] for value in observed], dtype=int)
    onehot = np.eye(3)[index]
    row_rps = np.mean(
        (
            np.cumsum(probability, axis=1)[:, :-1]
            - np.cumsum(onehot, axis=1)[:, :-1]
        )
        ** 2,
        axis=1,
    )
    observed_warmer = float(np.average(index == 0, weights=weights))
    observed_cooler = float(np.average(index == 2, weights=weights))
    predicted_warmer = float(np.average(probability[:, 0], weights=weights))
    predicted_cooler = float(np.average(probability[:, 2], weights=weights))
    warmer_error = abs(predicted_warmer - observed_warmer)
    cooler_error = abs(predicted_cooler - observed_cooler)
    return {
        "crps": float(np.average(row_rps, weights=weights)),
        "warmer_error": warmer_error,
        "cooler_error": cooler_error,
        "tail_mae": (warmer_error + cooler_error) / 2,
    }


def _local_update(
    feedback: pd.DataFrame, probability: np.ndarray
) -> np.ndarray:
    feedback_probability = feedback[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
    prior = feedback_probability.mean(axis=0)
    counts = np.array(
        [
            int(feedback["observed"].eq(label).sum())
            for label in PREFERENCE_CLASSES
        ],
        dtype=float,
    )
    posterior = (LOCAL_PRIOR_STRENGTH * prior + counts) / (
        LOCAL_PRIOR_STRENGTH + len(feedback)
    )
    factor = posterior / np.clip(prior, 1e-12, None)
    updated = probability * factor[None, :]
    updated /= updated.sum(axis=1, keepdims=True)
    return updated


def _bootstrap_group_means(
    frame: pd.DataFrame,
    columns: list[str],
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    groups = sorted(frame["sampling_group_rank"].unique())
    tables = {
        group: frame.loc[frame["sampling_group_rank"].eq(group)].reset_index(drop=True)
        for group in groups
    }
    draws = {column: np.empty(LOCAL_BOOTSTRAPS, dtype=float) for column in columns}
    for repetition in range(LOCAL_BOOTSTRAPS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        values = {column: [] for column in columns}
        for group in sampled:
            selected = tables[group].iloc[int(rng.integers(0, len(tables[group])))]
            for column in columns:
                values[column].append(float(selected[column]))
        for column in columns:
            draws[column][repetition] = float(np.mean(values[column]))
    return {
        column: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
        for column, values in draws.items()
    }


def local_update_learning_curve(
    config: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = _read(config, "regional_predictions").rename(
        columns={
            "held_out_group_id": "held_out",
            "observed_preference": "observed",
        }
    )
    sampling_plan = _read(config, "local_update_sampling_plan")
    counts = (
        predictions.groupby(["split", "held_out"], sort=True)
        .agg(
            analysis_units=("release_unit_id", "nunique"),
            records=("release_record_id", "size"),
        )
        .reset_index()
    )
    counts["eligible_common_curve"] = counts["analysis_units"].ge(LOCAL_MIN_UNITS)
    eligible = counts.loc[counts["eligible_common_curve"], ["split", "held_out"]]
    work = predictions.merge(
        eligible, on=["split", "held_out"], how="inner", validate="many_to_one"
    )
    rows: list[dict[str, Any]] = []
    for (split, held_out), group in work.groupby(["split", "held_out"], sort=True):
        group = group.reset_index(drop=True)
        group_ranks = group["sampling_group_rank"].unique()
        if len(group_ranks) != 1:
            raise AssertionError("held-out group must map to one sampling_group_rank")
        group_rank = int(group_ranks[0])
        plan_group = sampling_plan.loc[
            sampling_plan["split"].eq(split)
            & sampling_plan["sampling_group_rank"].eq(group_rank)
        ]
        for repetition in range(LOCAL_REPETITIONS):
            draw = plan_group.loc[plan_group["repetition"].eq(repetition)].sort_values(
                "feedback_position"
            )
            if len(draw) != max(LOCAL_FEEDBACK_SIZES):
                raise AssertionError("incomplete frozen local-update draw")
            for feedback_n in LOCAL_FEEDBACK_SIZES:
                selected = draw.head(feedback_n)
                feedback_unit_ranks = set(selected["sampling_unit_rank"])
                feedback_row_ranks = set(selected["sampling_group_row_rank"])
                feedback = group.loc[
                    group["sampling_group_row_rank"].isin(feedback_row_ranks)
                ].copy()
                if len(feedback) != feedback_n:
                    raise AssertionError("frozen feedback rows did not resolve uniquely")
                evaluation = group.loc[
                    ~group["sampling_unit_rank"].isin(feedback_unit_ranks)
                ].copy()
                probability = evaluation[list(PROBABILITY_COLUMNS)].to_numpy(dtype=float)
                weights = _unit_weights(evaluation)
                base = _preference_score(
                    evaluation["observed"].to_numpy(dtype=str), probability, weights
                )
                updated_probability = (
                    probability.copy()
                    if feedback_n == 0
                    else _local_update(feedback, probability)
                )
                updated = _preference_score(
                    evaluation["observed"].to_numpy(dtype=str),
                    updated_probability,
                    weights,
                )
                row: dict[str, Any] = {
                    "split": split,
                    "held_out": held_out,
                    "sampling_group_rank": group_rank,
                    "repetition": repetition,
                    "feedback_n": feedback_n,
                    "evaluation_records": len(evaluation),
                    "evaluation_units": evaluation["release_unit_id"].nunique(),
                }
                for metric in ("crps", "warmer_error", "cooler_error", "tail_mae"):
                    row[f"base_{metric}"] = base[metric]
                    row[f"updated_{metric}"] = updated[metric]
                    row[f"improvement_{metric}"] = base[metric] - updated[metric]
                rows.append(row)
    metrics = pd.DataFrame(rows)
    bootstrap_columns = [
        "base_crps",
        "updated_crps",
        "improvement_crps",
        "base_tail_mae",
        "updated_tail_mae",
        "improvement_tail_mae",
        "improvement_warmer_error",
        "improvement_cooler_error",
    ]
    summaries: list[dict[str, Any]] = []
    for (split, feedback_n), group in metrics.groupby(["split", "feedback_n"], sort=True):
        group_means = group.groupby("sampling_group_rank", as_index=False)[
            bootstrap_columns
        ].mean()
        ci = _bootstrap_group_means(
            group,
            bootstrap_columns,
            np.random.default_rng(seed + (0 if split == "source" else 100000) + int(feedback_n)),
        )
        row: dict[str, Any] = {
            "split": split,
            "feedback_n": int(feedback_n),
            "outer_groups": group["held_out"].nunique(),
            "repeated_draws_per_group": group["repetition"].nunique(),
            "fraction_groups_crps_improved": float(
                (group_means["improvement_crps"] > 0).mean()
            )
            if feedback_n
            else 0.0,
            "fraction_groups_tail_improved": float(
                (group_means["improvement_tail_mae"] > 0).mean()
            )
            if feedback_n
            else 0.0,
        }
        for column in bootstrap_columns:
            row[column] = float(group_means[column].mean())
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = ci[column]
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    for split in sorted(summary["split"].unique()):
        mask = summary["split"].eq(split)
        gain40_crps = float(
            summary.loc[mask & summary["feedback_n"].eq(40), "improvement_crps"].iloc[0]
        )
        gain40_tail = float(
            summary.loc[
                mask & summary["feedback_n"].eq(40), "improvement_tail_mae"
            ].iloc[0]
        )
        summary.loc[mask, "fraction_of_40_crps_gain"] = (
            summary.loc[mask, "improvement_crps"] / gain40_crps
        )
        summary.loc[mask, "fraction_of_40_tail_gain"] = (
            summary.loc[mask, "improvement_tail_mae"] / gain40_tail
        )
        base0_crps = float(
            summary.loc[mask & summary["feedback_n"].eq(0), "updated_crps"].iloc[0]
        )
        base0_tail = float(
            summary.loc[
                mask & summary["feedback_n"].eq(0), "updated_tail_mae"
            ].iloc[0]
        )
        summary.loc[mask, "updated_crps_change_vs_full_group_zero"] = (
            base0_crps - summary.loc[mask, "updated_crps"]
        )
        summary.loc[mask, "updated_tail_change_vs_full_group_zero"] = (
            base0_tail - summary.loc[mask, "updated_tail_mae"]
        )
    return summary, metrics.drop(columns=["sampling_group_rank"])


def centre_and_spread_effects(
    annual: pd.DataFrame, dynamic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    annual_keys = ["population_scenario", "city", "composition_seed", "service_fraction"]
    annual_metrics = [
        "full_eligible_action_count",
        "full_selected_site_kwh",
        "full_selected_electric_kwh",
        "full_selected_gas_kwh_fuel",
        "full_selected_operational_co2_kg",
    ]
    centre_rows = []
    reference = annual.loc[annual["lambda"].eq(1.0) & annual["delta_c"].eq(0)]
    for delta in (-2.0, -1.0, 1.0, 2.0):
        target = annual.loc[annual["lambda"].eq(1.0) & annual["delta_c"].eq(delta)]
        merged = target.merge(
            reference,
            on=annual_keys,
            how="outer",
            suffixes=("_target", "_reference"),
            validate="one_to_one",
        )
        for metric in annual_metrics:
            merged[f"change_{metric}"] = (
                merged[f"{metric}_target"] - merged[f"{metric}_reference"]
            )
        merged["target_delta_c"] = delta
        centre_rows.append(merged)
    centre = pd.concat(centre_rows, ignore_index=True)

    spread_rows = []
    reference = annual.loc[annual["lambda"].eq(1.0) & annual["delta_c"].eq(0)]
    for spread in (0.0, 0.5, 1.5):
        target = annual.loc[annual["lambda"].eq(spread) & annual["delta_c"].eq(0)]
        merged = target.merge(
            reference,
            on=annual_keys,
            how="outer",
            suffixes=("_target", "_reference"),
            validate="one_to_one",
        )
        for metric in annual_metrics:
            merged[f"change_{metric}"] = (
                merged[f"{metric}_target"] - merged[f"{metric}_reference"]
            )
        merged["target_lambda"] = spread
        spread_rows.append(merged)
    spread_effects = pd.concat(spread_rows, ignore_index=True)

    interaction_rows = []
    lookup = annual.set_index(annual_keys + ["delta_c", "lambda"])
    for key in annual[annual_keys].drop_duplicates().itertuples(index=False, name=None):
        for delta in (-2.0, -1.0, 1.0, 2.0):
            for spread in (0.0, 0.5, 1.5):
                a = lookup.loc[key + (delta, spread)]
                b = lookup.loc[key + (delta, 1.0)]
                c = lookup.loc[key + (0.0, spread)]
                d = lookup.loc[key + (0.0, 1.0)]
                row = dict(zip(annual_keys, key, strict=True)) | {
                    "delta_c": delta,
                    "lambda": spread,
                }
                for metric in annual_metrics:
                    row[f"interaction_{metric}"] = (
                        float(a[metric])
                        - float(b[metric])
                        - float(c[metric])
                        + float(d[metric])
                    ) if all(pd.notna(x[metric]) for x in (a, b, c, d)) else np.nan
                interaction_rows.append(row)
    interactions = pd.DataFrame(interaction_rows)

    dynamic_keys = annual_keys + ["event_month"]
    dynamic_metrics = [
        "full_positive_action_count",
        "full_selected_event_average_reduction_kw",
        "full_selected_recovery_net_extra_electric_kwh",
        "full_selected_recovery_peak_hour_extra_kw",
        "full_selected_event_plus48h_net_electric_saving_kwh",
    ]
    dynamic_rows = []
    reference = dynamic.loc[dynamic["lambda"].eq(1.0) & dynamic["delta_c"].eq(0)]
    for spread in (0.0, 0.5, 1.5):
        target = dynamic.loc[dynamic["lambda"].eq(spread) & dynamic["delta_c"].eq(0)]
        merged = target.merge(
            reference,
            on=dynamic_keys,
            how="outer",
            suffixes=("_target", "_reference"),
            validate="one_to_one",
        )
        for metric in dynamic_metrics:
            merged[f"change_{metric}"] = (
                merged[f"{metric}_target"] - merged[f"{metric}_reference"]
            )
        merged["target_lambda"] = spread
        dynamic_rows.append(merged)
    dynamic_effects = pd.concat(dynamic_rows, ignore_index=True)
    return centre, spread_effects, interactions, dynamic_effects


def failure_ledgers(
    annual: pd.DataFrame, dynamic: pd.DataFrame, dynamic_eligibility: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    annual_failure = annual.loc[
        annual["paired_feasibility_status"].ne("both_feasible")
        | annual["mean_selected_full_reaudit_status"].eq("mean_selected_false_accept")
    ].copy()
    annual_failure["failure_reason"] = np.select(
        [
            annual_failure["paired_feasibility_status"].eq("neither_feasible"),
            annual_failure["paired_feasibility_status"].eq("mean_only_feasible"),
            annual_failure["paired_feasibility_status"].eq("full_only_feasible"),
        ],
        ["neither_feasible", "mean_only_feasible", "full_only_feasible"],
        default="mean_selected_false_accept",
    )
    dynamic_failure = dynamic.loc[
        dynamic["mean_selection_state"].ne("positive_action_selected")
        | dynamic["full_selection_state"].ne("positive_action_selected")
        | dynamic["mean_selected_full_reaudit_status"].eq("mean_selected_false_accept")
    ].copy()
    dynamic_failure["failure_reason"] = (
        dynamic_failure["mean_selection_state"].astype(str)
        + "|"
        + dynamic_failure["full_selection_state"].astype(str)
        + "|"
        + dynamic_failure["mean_selected_full_reaudit_status"].astype(str)
    )
    nonpositive = dynamic_eligibility.loc[
        dynamic_eligibility["event_average_reduction_kw"].le(RESPONSE_TOL_KW)
    ].copy()
    nonpositive = nonpositive.drop_duplicates(
        [
            "population_scenario",
            "city",
            "composition_seed",
            "event_month",
            "delta_c",
            "lambda",
            "representation",
            "event_action_id",
            "service_fraction",
        ]
    )
    return {
        "annual_failure_and_unmatched": annual_failure,
        "dynamic_failure_and_unmatched": dynamic_failure,
        "dynamic_nonpositive_response": nonpositive,
    }


def headline_results(
    tables: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(result: str, metric: str, value: Any, unit: str, denominator: Any = None) -> None:
        rows.append(
            {
                "result": result,
                "metric": metric,
                "value": value,
                "unit": unit,
                "denominator": denominator,
            }
        )

    scope = tables["result1_scope"].iloc[0]
    for metric in ("records", "units", "buildings", "cities", "countries"):
        add("Result1", f"archive_{metric}", int(scope[metric]), "count")
    for row in tables["result1_variance"].itertuples(index=False):
        add("Result1", f"variance_share_{row.component}", float(row.share), "fraction")

    direct = tables["result2_summary"].set_index("dataset_code").loc["NUS_HAAS"]
    for metric in (
        "eligible_windows",
        "votes",
        "disagreement_windows",
        "opposing_direction_windows",
        "exact_cancellation_windows",
    ):
        add("Result2", f"nus_{metric}", int(direct[metric]), "count", 63)
    add("Result2", "nus_mean_jensen_gap", float(direct["mean_jensen_gap"]), "burden")

    centre = tables["result3_centre_effects"]
    for delta in (-2.0, 2.0):
        for city, group in centre.loc[centre["target_delta_c"].eq(delta)].groupby("city", sort=True):
            energy = group["change_full_selected_site_kwh"].dropna()
            carbon = group["change_full_selected_operational_co2_kg"].dropna()
            add(
                "Result3",
                f"delta_{delta:+g}_{city}_median_energy_change",
                float(energy.median()),
                "kWh yr-1",
                len(energy),
            )
            add(
                "Result3",
                f"delta_{delta:+g}_{city}_median_carbon_change",
                float(carbon.median()),
                "kgCO2 yr-1",
                len(carbon),
            )

    annual = tables["result3_annual_decisions"]
    for spread in (0.0, 0.5, 1.0, 1.5):
        group = annual.loc[annual["delta_c"].eq(0) & annual["lambda"].eq(spread)]
        add(
            "Result4",
            f"lambda_{spread:g}_full_selection_cells",
            int(group["full_selected_action_id"].notna().sum()),
            "decision cells",
            len(group),
        )
        mean_selected = group.loc[group["mean_selected_action_id"].notna()]
        add(
            "Result4",
            f"lambda_{spread:g}_mean_selected_false_accept",
            int(mean_selected["mean_selected_full_eligible"].eq(False).sum()),
            "decision cells",
            len(mean_selected),
        )

    dynamic = tables["result5_dynamic_decisions"]
    for spread in (0.0, 0.5, 1.0, 1.5):
        group = dynamic.loc[
            dynamic["event_month"].eq(7)
            & dynamic["delta_c"].eq(0)
            & dynamic["lambda"].eq(spread)
        ]
        add(
            "Result5",
            f"july_lambda_{spread:g}_full_positive_selection_cells",
            int(group["full_selected_action_id"].notna().sum()),
            "decision cells",
            len(group),
        )
        mean_selected = group.loc[group["mean_selected_action_id"].notna()]
        add(
            "Result5",
            f"july_lambda_{spread:g}_mean_selected_false_accept",
            int(mean_selected["mean_selected_full_eligible"].eq(False).sum()),
            "decision cells",
            len(mean_selected),
        )
    february = dynamic.loc[
        dynamic["event_month"].eq(2)
        & dynamic["delta_c"].eq(0)
        & dynamic["lambda"].eq(1.0)
    ]
    add(
        "Result5",
        "february_central_full_positive_selection_cells",
        int(february["full_selected_action_id"].notna().sum()),
        "decision cells",
        len(february),
    )

    phase = tables["result6_phase_map"]
    for classification, count in phase["classification"].value_counts().sort_index().items():
        metric = classification.replace("-", "_")
        add("Result6", f"phase_{metric}", int(count), "decision cells", len(phase))
    resolved = phase.loc[phase["classification"].ne("finite-library-unresolved")]
    add("Result6", "phase_resolved_cells", len(resolved), "decision cells", len(phase))
    gate = tables["result6_quantile_gate"]
    recommended = gate.loc[
        gate["split_scheme"].eq("leave_one_context_seed_out")
        & gate["representation"].eq("quantile5")
        & gate["strategy"].eq("pooled_q95")
    ].iloc[0]
    for metric in (
        "false_accept_candidates",
        "selected_false_accept_decisions",
        "retained_full_eligible_fraction",
        "certified_coverage_of_full_available",
    ):
        add("Result6", f"protected_q5_{metric}", recommended[metric], "count_or_fraction")

    headline = pd.DataFrame(rows)
    nested: dict[str, Any] = {}
    for result, group in headline.groupby("result", sort=True):
        nested[result] = {
            row.metric: {
                "value": None if pd.isna(row.value) else row.value,
                "unit": row.unit,
                "denominator": None if pd.isna(row.denominator) else row.denominator,
            }
            for row in group.itertuples(index=False)
        }
    return headline, nested


def validate_frozen_sensitivity(config: dict[str, Any]) -> None:
    """Check that the declared coefficients describe the frozen input grid."""
    actual = sorted(_read(config, "three_definition_matches")["griffiths_coefficient"].unique())
    declared = sorted(config["science"]["griffiths_coefficients"])
    if actual != declared:
        raise ValueError(
            "Griffiths coefficient declaration does not match frozen inputs: "
            f"declared={declared}, actual={actual}"
        )


def reproduce_all(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    validate_frozen_sensitivity(config)
    tables: dict[str, pd.DataFrame] = {}
    tables.update(result1(config))
    tables.update(result2(config))
    annual, annual_eligibility = annual_decisions(config)
    dynamic, dynamic_eligibility = dynamic_decisions(config)
    tables["result3_annual_decisions"] = annual
    tables["result3_annual_eligibility"] = annual_eligibility
    tables["result5_dynamic_decisions"] = dynamic
    tables["result5_dynamic_eligibility"] = dynamic_eligibility
    centre, spread, interactions, dynamic_effects = centre_and_spread_effects(
        annual, dynamic
    )
    tables["result3_centre_effects"] = centre
    tables["result4_spread_effects"] = spread
    tables["result4_interactions"] = interactions
    tables["result5_spread_effects"] = dynamic_effects
    phase, approximations, phase_eligibility = phase_map(config)
    tables["result6_phase_map"] = phase
    tables["result6_approximation"] = approximations
    tables["result6_phase_eligibility"] = phase_eligibility
    gate, margins = quantile_certification(config, phase)
    tables["result6_quantile_gate"] = gate
    tables["result6_quantile_margins"] = margins
    learning, learning_replicates = local_update_learning_curve(
        config, int(config["science"]["seed"])
    )
    tables["result6_local_update"] = learning
    tables["result6_local_update_replicates"] = learning_replicates
    for name, ledger in failure_ledgers(annual, dynamic, dynamic_eligibility).items():
        tables[f"ledger_{name}"] = ledger
    headline, headline_json = headline_results(tables)
    tables["headline_results"] = headline
    return tables, headline_json


def write_scientific_tables(
    tables: dict[str, pd.DataFrame], root: Path
) -> list[Path]:
    written = []
    for name, frame in sorted(tables.items()):
        subdirectory = "ledgers" if name.startswith("ledger_") else "tables"
        target = root / subdirectory / f"{name}.csv"
        columns = list(frame.columns)
        written.append(write_table(frame, target, sort_by=columns))
    return written
