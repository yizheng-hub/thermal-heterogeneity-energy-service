"""Evaluate service thresholds and recovery windows using retained trajectories.

This analysis never changes a controller, building input or EnergyPlus output.
It tests dependence on the shared common-25 candidate, summarizes July response
qualification, extends recovery accounting on saved 8760-hour trajectories and
restricts direct comparisons to conditions qualified under all three controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = ("uniform_average", "personal_center", "ltpref_interval")
ALTERNATIVES = CONTROLS[1:]
LAYOUTS = ("random_mixing", "demand_ordered")
QUALIFIED = {"cached_qualified", "new_qualified"}
KEYS = ["context", "composition_id", "layout", "permission", "control"]
CONDITION = ["data_source", "context", "composition_id", "layout"]
SERVICE = ("mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c")
HORIZONS = (24, 48, 72, 96, 120, 168)
TOL = 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_energy(frame: pd.DataFrame) -> dict | None:
    """Match the frozen annual tie-breaking rule."""
    best = None
    for row in frame.sort_values(["run_id", "variant"]).to_dict("records"):
        if best is None:
            best = row
            continue
        a, b = row["annual_site_kwh"], best["annual_site_kwh"]
        tolerance = max(1e-6, 1e-9 * max(abs(a), abs(b)))
        if a < b - tolerance:
            best = row
    return best


def pair_selections(selections: pd.DataFrame) -> pd.DataFrame:
    index_keys = ["data_source", "context", "composition_id", "layout", "permission"]
    reference = selections[selections.control.eq(CONTROLS[0])].set_index(index_keys)
    rows = []
    for alternative in ALTERNATIVES:
        other = selections[selections.control.eq(alternative)].set_index(index_keys)
        for key in reference.index:
            ref, alt = reference.loc[key], other.loc[key]
            matched = ref.selection_status == alt.selection_status == "selected"
            item = dict(zip(index_keys, key)) | {
                "reference": CONTROLS[0], "alternative": alternative,
                "reference_status": ref.selection_status,
                "alternative_status": alt.selection_status,
                "status": "matched" if matched else "unmatched",
                "reference_run_id": ref.get("run_id", ""),
                "alternative_run_id": alt.get("run_id", ""),
                "reference_variant": ref.get("variant", np.nan),
                "alternative_variant": alt.get("variant", np.nan),
            }
            item["same_physical_run"] = bool(
                matched and item["reference_run_id"] == item["alternative_run_id"])
            item["both_common25"] = bool(
                item["same_physical_run"] and item["reference_variant"] == 3
                and item["alternative_variant"] == 3)
            if matched:
                delta = alt.annual_site_kwh - ref.annual_site_kwh
                item["delta_annual_site_kwh"] = delta
                item["delta_site_pct"] = 100 * delta / ref.annual_site_kwh
            else:
                item["delta_annual_site_kwh"] = np.nan
                item["delta_site_pct"] = np.nan
            rows.append(item)
    return pd.DataFrame(rows)


def reference_limit_sensitivity(candidates: pd.DataFrame, selections: pd.DataFrame):
    cohort = selections[
        selections.service_fraction.eq(1)
        & selections.permission.eq("native_zone")
        & selections.layout.isin(LAYOUTS)
    ].copy()
    original_pairs = pair_selections(cohort)

    rows = []
    for budget in cohort.to_dict("records"):
        group = candidates
        for key in KEYS:
            group = group[group[key].eq(budget[key])]
        valid = group[
            group.engineering_status.isin(QUALIFIED)
            & group.variant.ne(3)
            & group[SERVICE[0]].le(budget["mean_budget_c"] + TOL)
            & group[SERVICE[1]].le(budget["fixed_tail_budget_c"] + TOL)
        ]
        chosen = choose_energy(valid)
        item = {key: budget[key] for key in KEYS} | {
            "data_source": budget["data_source"],
            "mean_budget_c": budget["mean_budget_c"],
            "fixed_tail_budget_c": budget["fixed_tail_budget_c"],
            "engineering_and_service_qualified_noncommon_candidates": len(valid),
            "selection_status": "selected" if chosen else "service_unavailable",
        }
        if chosen:
            item.update({name: chosen[name] for name in (
                "variant", "run_id", "annual_site_kwh", "annual_operational_co2_kg",
                *SERVICE)})
        rows.append(item)
    no_common = pd.DataFrame(rows)
    no_common_pairs = pair_selections(no_common)

    selection_summary = no_common.groupby("control", sort=True).agg(
        designed_conditions=("selection_status", "size"),
        selected_conditions=("selection_status", lambda x: int(x.eq("selected").sum())),
        unavailable_conditions=("selection_status", lambda x: int(x.ne("selected").sum())),
    ).reset_index()
    pair_rows = []
    for control, group in no_common_pairs.groupby("alternative", sort=True):
        matched = group[group.status.eq("matched")]
        pair_rows.append({
            "alternative": control, "designed_pairs": len(group),
            "matched_pairs": len(matched),
            "same_physical_run_pairs": int(matched.same_physical_run.sum()),
            "lower_energy_pairs": int(matched.delta_site_pct.lt(-1e-9).sum()),
            "higher_energy_pairs": int(matched.delta_site_pct.gt(1e-9).sum()),
            "delta_site_pct_median": float(matched.delta_site_pct.median()),
            "delta_site_pct_min": float(matched.delta_site_pct.min()),
            "delta_site_pct_max": float(matched.delta_site_pct.max()),
        })
    return original_pairs, no_common, no_common_pairs, selection_summary, pd.DataFrame(pair_rows)


def observed_service_nodes(frontiers: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    frame = frontiers[
        frontiers.permission.eq("native_zone") & frontiers.layout.isin(LAYOUTS)
    ].copy()
    frame["distinct_noncommon_match"] = (
        frame.status.eq("matched") & ~frame.same_physical_run
        & ~frame.reference_common25 & ~frame.alternative_common25)
    keys = ["context", "composition_id", "layout", "control"]
    sources = selections[["context", "composition_id", "layout", "data_source"]].drop_duplicates()
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        rows.append(dict(zip(keys, key)) | {
            "observed_service_nodes": len(group),
            "matched_nodes": int(group.status.eq("matched").sum()),
            "any_distinct_noncommon_match": bool(group.distinct_noncommon_match.any()),
            "distinct_noncommon_matches": int(group.distinct_noncommon_match.sum()),
        })
    return pd.DataFrame(rows).merge(
        sources, on=["context", "composition_id", "layout"], validate="many_to_one")


def strict_summary(selections: pd.DataFrame) -> pd.DataFrame:
    strict = selections[
        selections.service_fraction.eq(.5)
        & selections.permission.eq("native_zone")
        & selections.layout.isin(LAYOUTS)
    ]
    rows = []
    for control, group in strict.groupby("control", sort=True):
        selected = group.selection_status.eq("selected")
        miami_ordered = selected & group.context.eq("Miami") & group.layout.eq("demand_ordered")
        rows.append({
            "control": control, "designed_conditions": len(group),
            "selected_conditions": int(selected.sum()),
            "service_unavailable_conditions": int(group.selection_status.eq("service_unavailable").sum()),
            "selected_miami_ordered": int(miami_ordered.sum()),
            "selected_outside_miami_ordered": int((selected & ~miami_ordered).sum()),
        })
    return pd.DataFrame(rows)


def direct_common_support(direct: pd.DataFrame):
    frame = direct[
        direct.permission.eq("native_zone") & direct.layout.isin(LAYOUTS)
    ].copy()
    frame["engineering_qualified"] = frame.engineering_status.isin(QUALIFIED)
    condition_rows = []
    pair_rows = []
    for key, group in frame.groupby(CONDITION, sort=True):
        controls = group.set_index("control")
        all_three = all(control in controls.index and controls.loc[control].engineering_qualified
                        for control in CONTROLS)
        item = dict(zip(CONDITION, key)) | {"all_three_engineering_qualified": bool(all_three)}
        for control in CONTROLS:
            item[f"{control}_engineering_status"] = (
                controls.loc[control].engineering_status if control in controls.index else "missing")
        condition_rows.append(item)
        if not all_three:
            continue
        reference = controls.loc[CONTROLS[0]]
        for control in ALTERNATIVES:
            alternative = controls.loc[control]
            delta = alternative.annual_site_kwh - reference.annual_site_kwh
            pair_rows.append(dict(zip(CONDITION, key)) | {
                "reference": CONTROLS[0], "alternative": control,
                "reference_run_id": reference.run_id,
                "alternative_run_id": alternative.run_id,
                "delta_annual_site_kwh": delta,
                "delta_site_pct": 100 * delta / reference.annual_site_kwh,
                "delta_operational_co2_kg": (
                    alternative.annual_operational_co2_kg - reference.annual_operational_co2_kg),
                "delta_interval_coverage": (
                    alternative.modeled_interval_person_hour_coverage
                    - reference.modeled_interval_person_hour_coverage),
            })
    conditions, pairs = pd.DataFrame(condition_rows), pd.DataFrame(pair_rows)
    summaries = []
    for control, group in pairs.groupby("alternative", sort=True):
        summaries.append({
            "alternative": control, "common_support_conditions": len(group),
            "energy_increase_conditions": int(group.delta_site_pct.gt(1e-9).sum()),
            "energy_decrease_conditions": int(group.delta_site_pct.lt(-1e-9).sum()),
            "coverage_improvement_conditions": int(group.delta_interval_coverage.gt(1e-12).sum()),
            "delta_site_pct_median": float(group.delta_site_pct.median()),
            "delta_site_pct_min": float(group.delta_site_pct.min()),
            "delta_site_pct_max": float(group.delta_site_pct.max()),
        })
    return conditions, pairs, pd.DataFrame(summaries)


def july_dr_audit(gates: pd.DataFrame, cases: pd.DataFrame, portfolio: pd.DataFrame):
    frame = gates[
        gates.event_month.eq(7) & gates.service_fraction.eq(1)
        & gates.permission.isin(("building", "native_zone"))
    ].copy()
    ratio_columns = [
        f"{window}_{trajectory}_{metric}_ratio"
        for window in ("event", "recovery")
        for trajectory in ("baseline", "perturbed")
        for metric in ("mean", "fixed_tail")
    ]
    frame["maximum_service_ratio"] = frame[ratio_columns].max(axis=1)
    frame["maximum_service_excess_pct"] = 100 * (frame.maximum_service_ratio - 1)
    frame["binding_service_ratio"] = frame[ratio_columns].idxmax(axis=1)
    audit_columns = [
        "data_source", "context", "composition_id", "layout", "permission", "control",
        "amplitude_c", "run_id", "baseline_run_id", "engineering_status",
        "engineering_qualified", "raw_positive", "baseline_qualified", "event_qualified",
        "recovery_qualified", "service_eligible", "first_exclusion",
        "event_average_reduction_kw", "maximum_service_ratio",
        "maximum_service_excess_pct", "binding_service_ratio", *ratio_columns,
    ]
    candidate_audit = frame[audit_columns].copy()

    gate_summary_source = frame.groupby(
        ["data_source", "context", "permission", "control"], sort=True).agg(
        logical_candidate_actions=("run_id", "size"),
        engineering_qualified_actions=("engineering_qualified", "sum"),
        positive_physical_actions=("raw_positive", "sum"),
        baseline_qualified_actions=("baseline_qualified", "sum"),
        event_qualified_actions=("event_qualified", "sum"),
        recovery_qualified_actions=("recovery_qualified", "sum"),
    ).reset_index()
    gate_summary_all = frame.groupby(
        ["context", "permission", "control"], sort=True).agg(
        logical_candidate_actions=("run_id", "size"),
        engineering_qualified_actions=("engineering_qualified", "sum"),
        positive_physical_actions=("raw_positive", "sum"),
        baseline_qualified_actions=("baseline_qualified", "sum"),
        event_qualified_actions=("event_qualified", "sum"),
        recovery_qualified_actions=("recovery_qualified", "sum"),
    ).reset_index().assign(data_source="ALL")
    gate_summary = pd.concat([gate_summary_source, gate_summary_all], ignore_index=True)

    excluded = frame[
        frame.engineering_qualified & frame.raw_positive & ~frame.service_eligible
    ]
    excess_summary = excluded.groupby(
        ["context", "permission", "control"], sort=True).maximum_service_excess_pct.agg(
        excluded_positive_actions="size", minimum_excess_pct="min",
        median_excess_pct="median", maximum_excess_pct="max").reset_index()

    selected = cases[
        cases.event_month.eq(7) & cases.service_fraction.eq(1)
        & cases.permission.isin(("building", "native_zone"))
    ].copy()
    identity = ["data_source", "context", "composition_id", "layout", "control"]
    building = selected[selected.permission.eq("building")].set_index(identity)
    native = selected[selected.permission.eq("native_zone")].set_index(identity)
    rows = []
    for key in building.index:
        left, right = building.loc[key], native.loc[key]
        rows.append(dict(zip(identity, key)) | {
            "building_baseline_run_id": left.baseline_run_id,
            "native_zone_baseline_run_id": right.baseline_run_id,
            "same_baseline_run": bool(left.baseline_run_id == right.baseline_run_id),
            "building_selection_status": left.selection_status,
            "native_zone_selection_status": right.selection_status,
            "building_qualified_response_kw": left.service_qualified_available_response_kw,
            "native_zone_qualified_response_kw": right.service_qualified_available_response_kw,
        })
    baseline_pairs = pd.DataFrame(rows)
    baseline_summary_source = baseline_pairs.groupby(
        ["data_source", "context", "control"], sort=True).agg(
        conditions=("same_baseline_run", "size"),
        same_baseline_conditions=("same_baseline_run", "sum"),
        building_selected_conditions=("building_selection_status", lambda x: int(x.eq("selected").sum())),
        native_zone_selected_conditions=("native_zone_selection_status", lambda x: int(x.eq("selected").sum())),
        building_sum_qualified_response_kw=("building_qualified_response_kw", "sum"),
        native_zone_sum_qualified_response_kw=("native_zone_qualified_response_kw", "sum"),
    ).reset_index()
    baseline_summary_all = baseline_pairs.groupby(
        ["context", "control"], sort=True).agg(
        conditions=("same_baseline_run", "size"),
        same_baseline_conditions=("same_baseline_run", "sum"),
        building_selected_conditions=("building_selection_status", lambda x: int(x.eq("selected").sum())),
        native_zone_selected_conditions=("native_zone_selection_status", lambda x: int(x.eq("selected").sum())),
        building_sum_qualified_response_kw=("building_qualified_response_kw", "sum"),
        native_zone_sum_qualified_response_kw=("native_zone_qualified_response_kw", "sum"),
    ).reset_index().assign(data_source="ALL")
    baseline_summary = pd.concat([baseline_summary_source, baseline_summary_all], ignore_index=True)

    raw = frame[frame.engineering_qualified & frame.raw_positive]
    raw_condition = raw.groupby(
        ["data_source", "context", "composition_id", "layout", "permission", "control"],
        sort=True).event_average_reduction_kw.max().reset_index()
    raw_summary = raw_condition.groupby(
        ["context", "permission", "control"], sort=True).agg(
        conditions_with_positive_physical_action=("event_average_reduction_kw", "size"),
        maximum_positive_physical_response_sum_kw=("event_average_reduction_kw", "sum"),
    ).reset_index()
    qualified_summary = selected.groupby(
        ["context", "permission", "control"], sort=True).agg(
        logical_decisions=("selection_status", "size"),
        qualified_selected_decisions=("selection_status", lambda x: int(x.eq("selected").sum())),
        qualified_selected_response_sum_kw=("service_qualified_available_response_kw", "sum"),
    ).reset_index()
    response_summary = raw_summary.merge(
        qualified_summary, on=["context", "permission", "control"], validate="one_to_one")

    portfolio_scope = portfolio[
        portfolio.event_id.eq("month_07")
        & portfolio.dr_participation_fraction.eq(1)
        & portfolio.scope.isin(("building", "native_zone"))
        & portfolio.result_scope.eq("joint_144_offices")
    ].copy()
    return (candidate_audit, gate_summary, excess_summary, baseline_pairs,
            baseline_summary, response_summary, portfolio_scope)


def _path_map(manifests: list[Path]) -> dict[str, Path]:
    paths = {}
    for manifest in manifests:
        frame = pd.read_csv(manifest)
        for row in frame[["run_id", "path_hourly"]].drop_duplicates().itertuples(index=False):
            if pd.isna(row.path_hourly) or not str(row.path_hourly).strip():
                continue
            path = (manifest.parent / str(row.path_hourly)).resolve()
            previous = paths.get(str(row.run_id))
            if previous is not None and previous != path:
                raise ValueError(f"Run {row.run_id} has conflicting hourly paths")
            paths[str(row.run_id)] = path
    return paths


@lru_cache(maxsize=None)
def _trajectory(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def recovery_horizon_metrics(base: dict[str, np.ndarray], event: dict[str, np.ndarray],
                             event_start: int, horizons=HORIZONS) -> list[dict]:
    difference = event["hvac_electric_kwh"] - base["hvac_electric_kwh"]
    event_saving = float(-difference[event_start:event_start + 2].sum())
    rows = []
    for horizon in horizons:
        end = event_start + 2 + int(horizon)
        recovery = float(difference[event_start + 2:end].sum())
        net = float(-difference[event_start:end].sum())
        rows.append({
            "recovery_horizon_hours": int(horizon),
            "event_electric_saving_kwh": event_saving,
            "cumulative_recovery_extra_electric_kwh": recovery,
            "event_plus_recovery_net_electric_saving_kwh": net,
            "terminal_max_abs_zone_temperature_difference_c": float(
                np.max(np.abs(event["temperatures"][end - 1] - base["temperatures"][end - 1]))),
            "terminal_abs_hvac_electric_difference_kwh": float(abs(difference[end - 1])),
            "electricity_closure_error_kwh": float(net - (event_saving - recovery)),
        })
    return rows


def selected_recovery_sensitivity(cases: pd.DataFrame, annual_manifests: list[Path],
                                  event_manifests: list[Path]):
    selected = cases[
        cases.event_month.eq(7) & cases.service_fraction.eq(1)
        & cases.selection_status.eq("selected")
    ].copy()
    annual_paths, event_paths = _path_map(annual_manifests), _path_map(event_manifests)
    rows = []
    for item in selected.to_dict("records"):
        base = _trajectory(str(annual_paths[str(item["baseline_run_id"])]))
        event = _trajectory(str(event_paths[str(item["selected_run_id"])]))
        civil = base["hour"] - 1 + base["dst"]
        indices = np.flatnonzero(
            (base["month"] == int(item["event_month"]))
            & (base["day"] == int(item["event_day"]))
            & (civil >= 14) & (civil < 16))
        if len(indices) != 2 or indices[1] != indices[0] + 1:
            raise ValueError("Selected event does not contain the declared two consecutive hours")
        identity = {name: item[name] for name in (
            "data_source", "context", "composition_id", "layout", "permission", "control",
            "event_month", "event_day", "baseline_run_id", "selected_run_id")}
        for result in recovery_horizon_metrics(base, event, int(indices[0])):
            if result["recovery_horizon_hours"] == 48:
                result["reported_48h_recovery_extra_electric_kwh"] = item["recovery_extra_electric_kwh"]
                result["reported_event_plus48h_net_electric_saving_kwh"] = item["event_plus48h_net_electric_saving_kwh"]
                result["reported_recovery_difference_kwh"] = (
                    result["cumulative_recovery_extra_electric_kwh"]
                    - item["recovery_extra_electric_kwh"])
                result["reported_net_difference_kwh"] = (
                    result["event_plus_recovery_net_electric_saving_kwh"]
                    - item["event_plus48h_net_electric_saving_kwh"])
            else:
                result.update({
                    "reported_48h_recovery_extra_electric_kwh": np.nan,
                    "reported_event_plus48h_net_electric_saving_kwh": np.nan,
                    "reported_recovery_difference_kwh": np.nan,
                    "reported_net_difference_kwh": np.nan,
                })
            rows.append(identity | result)
    detail = pd.DataFrame(rows)
    summaries = []
    for horizon, group in detail.groupby("recovery_horizon_hours", sort=True):
        summaries.append({
            "recovery_horizon_hours": horizon, "selected_cases": len(group),
            "unique_event_trajectories": group.selected_run_id.nunique(),
            "recovery_extra_kwh_min": float(group.cumulative_recovery_extra_electric_kwh.min()),
            "recovery_extra_kwh_median": float(group.cumulative_recovery_extra_electric_kwh.median()),
            "recovery_extra_kwh_max": float(group.cumulative_recovery_extra_electric_kwh.max()),
            "net_saving_kwh_min": float(group.event_plus_recovery_net_electric_saving_kwh.min()),
            "net_saving_kwh_median": float(group.event_plus_recovery_net_electric_saving_kwh.median()),
            "net_saving_kwh_max": float(group.event_plus_recovery_net_electric_saving_kwh.max()),
            "positive_net_saving_cases": int(group.event_plus_recovery_net_electric_saving_kwh.gt(0).sum()),
            "terminal_max_zone_delta_c_median": float(group.terminal_max_abs_zone_temperature_difference_c.median()),
            "terminal_max_zone_delta_c_max": float(group.terminal_max_abs_zone_temperature_difference_c.max()),
            "terminal_zone_delta_above_0_01_c": int(group.terminal_max_abs_zone_temperature_difference_c.gt(.01).sum()),
            "terminal_zone_delta_above_0_05_c": int(group.terminal_max_abs_zone_temperature_difference_c.gt(.05).sum()),
            "maximum_abs_closure_error_kwh": float(group.electricity_closure_error_kwh.abs().max()),
        })
    return detail, pd.DataFrame(summaries)


def run(repository_root: Path, output_root: Path) -> dict:
    repo, output = repository_root.resolve(), output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    annual = repo / "results/Joint_main/annual"
    events = repo / "results/Joint_main/joint_events"
    service = repo / "results/Joint_main/service"
    paths = {
        "candidates": annual / "ANNUAL_ALL_CANDIDATES.csv",
        "selections": annual / "ANNUAL_SERVICE_SELECTIONS.csv",
        "direct": annual / "ANNUAL_DIRECT.csv",
        "frontiers": service / "OBSERVED_SERVICE_FRONTIERS.csv",
        "gates": events / "JOINT_EVENT_CANDIDATE_GATES.csv",
        "cases": events / "JOINT_EVENT_CASE_SERVICE_ENERGY.csv",
        "portfolio": events / "JOINT_EVENT_PORTFOLIO_RESPONSE.csv",
    }
    frames = {name: pd.read_csv(path) for name, path in paths.items()}

    original, no_common, no_common_pairs, no_common_selections, no_common_summary = (
        reference_limit_sensitivity(frames["candidates"], frames["selections"]))
    nodes = observed_service_nodes(frames["frontiers"], frames["selections"])
    strict = strict_summary(frames["selections"])
    common_conditions, common_pairs, common_summary = direct_common_support(frames["direct"])
    dr = july_dr_audit(frames["gates"], frames["cases"], frames["portfolio"])
    annual_manifests = [
        repo / "data/inputs/energy_service/engineering/ANNUAL_RUN_MANIFEST.csv",
        repo / "data/inputs/ctcd_parallel/events/engineering/ANNUAL_RUN_MANIFEST.csv",
    ]
    event_manifests = [
        repo / "data/inputs/energy_service/engineering/EVENT_RUN_MANIFEST.csv",
        repo / "data/inputs/ctcd_parallel/events/engineering/EVENT_RUN_MANIFEST.csv",
    ]
    recovery_detail, recovery_summary = selected_recovery_sensitivity(
        frames["cases"], annual_manifests, event_manifests)

    outputs = {
        "REFERENCE_LIMIT_COMMON25_AUDIT.csv": original,
        "REFERENCE_LIMIT_NO_COMMON25_SELECTIONS.csv": no_common,
        "REFERENCE_LIMIT_NO_COMMON25_PAIRS.csv": no_common_pairs,
        "REFERENCE_LIMIT_NO_COMMON25_SELECTION_SUMMARY.csv": no_common_selections,
        "REFERENCE_LIMIT_NO_COMMON25_PAIR_SUMMARY.csv": no_common_summary,
        "OBSERVED_SERVICE_NODE_CONDITIONS.csv": nodes,
        "STRICT_HALF_REFERENCE_SUMMARY.csv": strict,
        "DIRECT_COMMON_ENGINEERING_CONDITIONS.csv": common_conditions,
        "DIRECT_COMMON_ENGINEERING_PAIRS.csv": common_pairs,
        "DIRECT_COMMON_ENGINEERING_SUMMARY.csv": common_summary,
        "JULY_DR_CANDIDATE_AUDIT.csv": dr[0],
        "JULY_DR_GATE_SUMMARY.csv": dr[1],
        "JULY_DR_SERVICE_EXCESS_SUMMARY.csv": dr[2],
        "JULY_DR_PERMISSION_BASELINE_PAIRS.csv": dr[3],
        "JULY_DR_PERMISSION_BASELINE_SUMMARY.csv": dr[4],
        "JULY_DR_RAW_AND_QUALIFIED_RESPONSE_SUMMARY.csv": dr[5],
        "JULY_DR_JOINT_PORTFOLIO_SCOPE.csv": dr[6],
        "JULY_SELECTED_RECOVERY_HORIZONS.csv": recovery_detail,
        "JULY_SELECTED_RECOVERY_SUMMARY.csv": recovery_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)

    same = original[original.same_physical_run]
    no_common_availability = no_common_selections.set_index("control").selected_conditions.to_dict()
    strict_counts = strict.set_index("control").selected_conditions.to_dict()
    common_counts = common_summary.set_index("alternative").to_dict("index")
    seattle_excess = dr[2][dr[2].context.eq("Seattle_small2019")].to_dict("records")
    horizon_48 = recovery_summary[recovery_summary.recovery_horizon_hours.eq(48)].iloc[0]
    horizon_168 = recovery_summary[recovery_summary.recovery_horizon_hours.eq(168)].iloc[0]
    qa = {
        "status": "PASS", "new_energyplus_runs": 0, "building_physics_changed": False,
        "reference_limit_pairs": len(original),
        "reference_limit_same_physical_pairs": int(original.same_physical_run.sum()),
        "reference_limit_same_pairs_both_common25": int(same.both_common25.sum()),
        "no_common25_selected_conditions": {key: int(value) for key, value in no_common_availability.items()},
        "observed_service_control_conditions": len(nodes),
        "observed_service_conditions_with_distinct_noncommon_match": int(nodes.any_distinct_noncommon_match.sum()),
        "strict_half_reference_selected_conditions": {key: int(value) for key, value in strict_counts.items()},
        "direct_conditions": len(common_conditions),
        "direct_all_three_engineering_qualified_conditions": int(common_conditions.all_three_engineering_qualified.sum()),
        "direct_common_support_summary": common_counts,
        "july_dr_candidate_actions_building_and_native": len(dr[0]),
        "seattle_july_service_excess_summary": seattle_excess,
        "selected_july_response_cases": int(recovery_detail[recovery_detail.recovery_horizon_hours.eq(48)].shape[0]),
        "selected_july_unique_event_trajectories": int(recovery_detail.selected_run_id.nunique()),
        "recovery_48h": horizon_48.to_dict(),
        "recovery_168h": horizon_168.to_dict(),
        "maximum_abs_48h_reported_recovery_difference_kwh": float(
            recovery_detail.reported_recovery_difference_kwh.abs().max()),
        "maximum_abs_48h_reported_net_difference_kwh": float(
            recovery_detail.reported_net_difference_kwh.abs().max()),
        "input_sha256": {path.relative_to(repo).as_posix(): sha256(path)
                         for path in [*paths.values(), *annual_manifests, *event_manifests]},
        "output_sha256": {name: sha256(output / name) for name in outputs},
        "interpretation_limits": [
            "Reference-limit reselection uses the retained finite candidate library.",
            "Observed service nodes are finite-library diagnostics, not a continuous global optimum.",
            "Building/native response comparisons jointly change annual baselines and command permissions.",
            "Recovery horizons extend electricity and terminal-temperature accounting; they do not redefine service qualification.",
            "Engineering-common direct comparisons remain conditional on the retained qualified trajectories.",
        ],
    }
    if not (len(original) == 96 and int(original.same_physical_run.sum()) == 67
            and int(same.both_common25.sum()) == 67):
        raise AssertionError("Frozen common-25 dependence counts changed")
    if no_common_availability != {"ltpref_interval": 44, "personal_center": 48,
                                  "uniform_average": 34}:
        raise AssertionError("No-common-25 availability changed")
    if int(nodes.any_distinct_noncommon_match.sum()) != 89:
        raise AssertionError("Observed service-node count changed")
    if strict_counts != {"ltpref_interval": 4, "personal_center": 4, "uniform_average": 0}:
        raise AssertionError("Strict-service count changed")
    if int(common_conditions.all_three_engineering_qualified.sum()) != 24:
        raise AssertionError("Engineering-common support count changed")
    if len(recovery_detail) != 64 * len(HORIZONS) or recovery_detail.selected_run_id.nunique() != 28:
        raise AssertionError("Selected recovery cohort changed")
    if recovery_detail.electricity_closure_error_kwh.abs().max() > 1e-10:
        raise AssertionError("Recovery electricity does not close")
    if qa["maximum_abs_48h_reported_recovery_difference_kwh"] > 1e-8:
        raise AssertionError("Derived 48-hour recovery differs from the retained result")
    if qa["maximum_abs_48h_reported_net_difference_kwh"] > 1e-8:
        raise AssertionError("Derived 48-hour net saving differs from the retained result")
    (output / "SERVICE_RECOVERY_SENSITIVITY_QA.json").write_text(
        json.dumps(qa, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path,
                        default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = args.output_root or args.repository_root / "results/Joint_main/service_recovery_sensitivity"
    print(json.dumps(run(args.repository_root, output), indent=2))


if __name__ == "__main__":
    main()
