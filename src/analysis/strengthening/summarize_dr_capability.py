"""Account for one response per event with explicit normal-operation support."""

from common import *
import numpy as np
import pandas as pd


EVENT_KEYS = ["climate", "composition_id", "event_id"]
CELL_KEYS = EVENT_KEYS + ["normal_energy_reference_factor"]
VALID_STATUSES = {"selected", "infeasible_service_preserving_positive_response", "unresolved_engineering_operation",
                  "infeasible_normal_energy_budget", "infeasible_service_budget"}
RESOLVED = {"selected", "infeasible_service_preserving_positive_response"}
ACTION_FIELDS = ["event_facility_mean_power_reduction_kw", "event_facility_electricity_reduction_kwh",
    "preconditioning_facility_electricity_change_kwh", "recovery_facility_electricity_change_kwh",
    "precondition_event_recovery_facility_net_reduction_kwh", "event_gas_reduction_kwh_hhv",
    "preconditioning_gas_change_kwh_hhv", "recovery_gas_change_kwh_hhv",
    "precondition_event_recovery_net_gas_reduction_kwh_hhv"]


def availability_table(choices):
    frame = choices.copy()
    unknown = set(frame.status) - VALID_STATUSES
    if unknown:
        raise ValueError("Unrecognized response status: " + str(unknown))
    frame["normal_eligible"] = frame.baseline_status.eq("feasible")
    selected = frame.status.eq("selected")
    no_response = frame.status.eq("infeasible_service_preserving_positive_response")
    if ((selected | no_response | frame.status.eq("unresolved_engineering_operation")) & ~frame.normal_eligible).any():
        raise ValueError("Response outcomes have an ineligible normal operation")
    if (~frame.normal_eligible & frame.status.ne(frame.baseline_status)).any():
        raise ValueError("Normal-infeasible response status differs from its baseline status")
    frame["engineering_resolved_normal_eligible"] = frame.normal_eligible & frame.status.isin(RESOLVED)
    power = pd.to_numeric(frame.event_power_reduction_kw, errors="raise")
    if not np.isfinite(power[selected]).all() or not power[selected].gt(0).all():
        raise ValueError("Selected event action lacks a positive finite power reduction")
    frame["available_event_power_kw"] = np.nan
    frame.loc[selected, "available_event_power_kw"] = power[selected]
    frame.loc[no_response, "available_event_power_kw"] = 0.0
    positive_net = selected & frame.positive_net_selected_action.notna()
    net_power = pd.to_numeric(frame.positive_net_event_power_kw, errors="raise")
    net_kwh = pd.to_numeric(frame.positive_net_full_window_kwh, errors="raise")
    if not np.isfinite(net_power[positive_net]).all() or not np.isfinite(net_kwh[positive_net]).all() or not (
        net_power[positive_net].gt(0).all() and net_kwh[positive_net].gt(0).all()
    ):
        raise ValueError("The positive-net action lacks positive finite power or net electricity")
    frame["positive_net_action_available"] = positive_net
    frame["available_positive_net_event_power_kw"] = np.nan
    frame.loc[frame.engineering_resolved_normal_eligible, "available_positive_net_event_power_kw"] = 0.0
    frame.loc[positive_net, "available_positive_net_event_power_kw"] = net_power[positive_net]
    return frame


def summarize_events(frame, grouping):
    rows = []
    for identity, group in frame.groupby(grouping, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        available = group[group.engineering_resolved_normal_eligible]
        selected = group[group.status.eq("selected")]
        eligible = group[group.normal_eligible]
        rows.append(dict(zip(grouping, identity)) | {
            "planned_normal_settings": len(group.drop_duplicates(["climate", "composition_id"])),
            "eligible_normal_settings": len(eligible.drop_duplicates(["climate", "composition_id"])),
            "planned_events": len(group), "normal_eligible_events": len(eligible),
            "engineering_resolved_events": len(available),
            "engineering_unresolved_events": int(group.status.eq("unresolved_engineering_operation").sum()),
            "selected_positive_response_events": len(selected),
            "resolved_events_without_service_preserving_positive_response": int(group.status.eq("infeasible_service_preserving_positive_response").sum()),
            "selected_positive_net_response_events": int(group.positive_net_action_available.sum()),
            "positive_response_fraction_among_resolved": len(selected) / len(available) if len(available) else np.nan,
            "available_event_power_mean_kw": available.available_event_power_kw.mean(),
            "available_event_power_median_kw": available.available_event_power_kw.median(),
            "selected_event_power_mean_kw": selected.event_power_reduction_kw.mean(),
            "available_positive_net_event_power_mean_kw": available.available_positive_net_event_power_kw.mean(),
            "selected_peak_action_net_electricity_mean_kwh": selected.net_full_window_electricity_reduction_kwh.mean(),
            "positive_net_action_net_electricity_mean_kwh": selected.positive_net_full_window_kwh.mean(),
        })
    return pd.DataFrame(rows)


def paired_events(frame):
    pairs = []
    for expanded, reference in [("zone_summary", "building_search"), ("zone_full", "building_search"), ("zone_full", "zone_summary")]:
        pair = frame[frame.control.eq(expanded)].merge(frame[frame.control.eq(reference)],
            on=CELL_KEYS, suffixes=("_expanded", "_reference"), validate="one_to_one")
        if len(pair) != 1296:
            raise ValueError("Incomplete planned paired response events")
        pair["compared_library"], pair["reference_library"] = expanded, reference
        support = pair.engineering_resolved_normal_eligible_expanded & pair.engineering_resolved_normal_eligible_reference
        pair["common_normal_eligible_engineering_resolved_support"] = support
        for endpoint in ["available_event_power_kw", "available_positive_net_event_power_kw"]:
            pair[endpoint + "_paired_difference"] = np.where(support,
                pair[endpoint + "_expanded"] - pair[endpoint + "_reference"], np.nan)
        pairs.append(pair)
    return pd.concat(pairs, ignore_index=True)


def summarize_pairs(frame, grouping):
    rows = []
    for identity, group in frame.groupby(grouping, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        common = group[group.common_normal_eligible_engineering_resolved_support]
        row = dict(zip(grouping, identity)) | {"planned_paired_events": len(group), "common_supported_events": len(common)}
        for endpoint in ["available_event_power_kw", "available_positive_net_event_power_kw"]:
            delta = common[endpoint + "_paired_difference"]
            row[endpoint + "_paired_difference_mean"] = delta.mean()
            row[endpoint + "_paired_difference_median"] = delta.median()
            row[endpoint + "_expanded_mean_on_common_support"] = common[endpoint + "_expanded"].mean()
            row[endpoint + "_reference_mean_on_common_support"] = common[endpoint + "_reference"].mean()
            row[endpoint + "_expanded_greater_events"] = int(delta.gt(1e-9).sum())
            row[endpoint + "_reference_greater_events"] = int(delta.lt(-1e-9).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def add_action_details(choices, scores):
    keys = EVENT_KEYS + ["control"]
    if scores.duplicated(keys + ["action_id"]).any():
        raise ValueError("Duplicated event-action scores")
    evaluated = scores[scores.status.eq("evaluated")]
    baseline = evaluated.groupby(keys).baseline_facility_mean_kw.agg(["min", "max"]).reset_index()
    if not np.allclose(baseline["min"], baseline["max"], atol=1e-8, rtol=0):
        raise ValueError("The same normal baseline has inconsistent event power")
    frame = choices.merge(baseline[keys + ["min"]].rename(columns={"min": "normal_event_power_kw"}), on=keys, how="left", validate="many_to_one")
    frame.loc[~frame.normal_eligible, "normal_event_power_kw"] = np.nan
    for label, selector in [("peak", "selected_action"), ("positive_net", "positive_net_selected_action")]:
        source = evaluated[keys + ["action_id", *ACTION_FIELDS]].rename(
            columns={name: label + "_" + name for name in ["action_id", *ACTION_FIELDS]})
        frame = frame.merge(source, left_on=keys + [selector], right_on=keys + [label + "_action_id"],
                            how="left", validate="many_to_one")
        selected = frame[selector].notna()
        if frame.loc[selected, label + "_action_id"].isna().any():
            raise ValueError("A chosen response has no engineering-evaluated action score")
        expected = frame.event_power_reduction_kw if label == "peak" else frame.positive_net_event_power_kw
        if not np.allclose(frame.loc[selected, label + "_event_facility_mean_power_reduction_kw"], expected[selected], atol=1e-8, rtol=0):
            raise ValueError("Chosen action power does not match the physical action score")
        expected_net = frame.net_full_window_electricity_reduction_kwh if label == "peak" else frame.positive_net_full_window_kwh
        if not np.allclose(frame.loc[selected, label + "_precondition_event_recovery_facility_net_reduction_kwh"], expected_net[selected], atol=1e-8, rtol=0):
            raise ValueError("Chosen action net electricity differs from its physical score")
        for event, pre, recovery, net in [
            ("event_facility_electricity_reduction_kwh", "preconditioning_facility_electricity_change_kwh",
             "recovery_facility_electricity_change_kwh", "precondition_event_recovery_facility_net_reduction_kwh"),
            ("event_gas_reduction_kwh_hhv", "preconditioning_gas_change_kwh_hhv",
             "recovery_gas_change_kwh_hhv", "precondition_event_recovery_net_gas_reduction_kwh_hhv")]:
            values = frame.loc[selected]
            if not np.allclose(values[label + "_" + event] - values[label + "_" + pre] - values[label + "_" + recovery],
                               values[label + "_" + net], atol=1e-8, rtol=0):
                raise ValueError("Response event/precondition/recovery carrier accounting does not close")
    selected = frame.status.eq("selected")
    if not np.allclose(frame.loc[selected, "normal_event_power_kw"] - frame.loc[selected, "absolute_event_facility_mean_kw"],
                       frame.loc[selected, "event_power_reduction_kw"], atol=1e-8, rtol=0):
        raise ValueError("Absolute event power and power reduction do not close")
    return frame


def main():
    folder = ROOT / "results/dr_capability"
    out = ROOT / "results/dr_capability_summary"
    paths = {name: folder / name for name in ["CAPABILITY_AUDIT.json", "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv",
        "ISOLATED_EVENT_ACTION_SCORES.csv", "NORMAL_OPERATION_BUDGET_STATUS.csv", "PLAN_AUDIT.json"]}
    paths["protocol"] = ROOT / "response_accounting.yaml"
    paths["event_protocol"] = RAW_RUN / "response/EVENT_PROTOCOL.csv"
    paths["source"] = Path(__file__)
    hashes = {path.relative_to(CODE).as_posix(): digest(path) for path in paths.values()}
    audit = json.loads(paths["CAPABILITY_AUDIT.json"].read_text())
    if audit["logical_event_actions"] != 6480 or audit["budget_event_cells"] != 3888:
        raise ValueError("The complete frozen response inventory is required")
    choices = pd.read_csv(paths["ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv"])
    if len(choices) != 3888 or choices.duplicated(CELL_KEYS + ["control"]).any():
        raise ValueError("Incomplete or duplicated response choices")
    budget = pd.read_csv(paths["NORMAL_OPERATION_BUDGET_STATUS.csv"])
    settings = ["climate", "composition_id", "control", "normal_energy_reference_factor"]
    if len(budget) != 486 or budget.duplicated(settings).any():
        raise ValueError("Incomplete normal-operation budget inventory")
    choice_settings = choices[settings + ["baseline_status"]].drop_duplicates()
    comparison = choice_settings.merge(budget[settings + ["status"]], on=settings, validate="one_to_one")
    if len(comparison) != 486 or not comparison.baseline_status.eq(comparison.status).all():
        raise ValueError("Response choices do not match frozen normal-operation feasibility")
    protocol = pd.read_csv(paths["event_protocol"])
    if digest(paths["event_protocol"]) != json.loads(paths["PLAN_AUDIT.json"].read_text())["event_protocol_sha256"]:
        raise ValueError("Response event protocol changed after normal-operation selection")
    for identity, group in choices.groupby(settings):
        expected = set(protocol[protocol.climate.eq(identity[0])].event_id)
        if len(group) != 8 or set(group.event_id) != expected:
            raise ValueError("A normal-operation setting lacks its8frozen events")
    frame = add_action_details(availability_table(choices), pd.read_csv(paths["ISOLATED_EVENT_ACTION_SCORES.csv"]))
    pairs = paired_events(frame)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "ONE_ACTION_RESOURCE_STATUS.csv", index=False)
    pairs.to_csv(out / "PAIRED_AVAILABLE_RESPONSE.csv", index=False)
    grouping = ["control", "normal_energy_reference_factor"]
    pair_grouping = ["compared_library", "reference_library", "normal_energy_reference_factor"]
    for suffix, extra in [("ALL_SETTINGS", []), ("BY_CLIMATE", ["climate"]), ("BY_COMPOSITION", ["composition_id"])]:
        summarize_events(frame, grouping + extra).to_csv(out / ("RESOURCE_SUMMARY_" + suffix + ".csv"), index=False)
        summarize_pairs(pairs, pair_grouping + extra).to_csv(out / ("PAIRED_SUMMARY_" + suffix + ".csv"), index=False)
    leave_events, leave_pairs = [], []
    for dimension in ["climate", "composition_id"]:
        for value in sorted(frame[dimension].unique()):
            event_summary = summarize_events(frame[frame[dimension].ne(value)], grouping)
            pair_summary = summarize_pairs(pairs[pairs[dimension].ne(value)], pair_grouping)
            for summary, destination in [(event_summary, leave_events), (pair_summary, leave_pairs)]:
                summary["omitted_dimension"], summary["omitted_value"] = dimension, str(value)
                destination.append(summary)
    pd.concat(leave_events, ignore_index=True).to_csv(out / "LEAVE_ONE_GROUP_RESOURCE_SUMMARIES.csv", index=False)
    pd.concat(leave_pairs, ignore_index=True).to_csv(out / "LEAVE_ONE_GROUP_PAIRED_SUMMARIES.csv", index=False)
    for name, expected in hashes.items():
        if digest(CODE / name) != expected:
            raise ValueError("Response summary input changed during execution")
    write_json(out / "INPUT_SHA256.json", hashes)
    report = {"status": "partial" if frame.status.eq("unresolved_engineering_operation").any() else "completed",
        "planned_normal_budget_settings": 486, "planned_event_budget_cells": len(frame),
        "baseline_ineligible_events": int((~frame.normal_eligible).sum()),
        "engineering_unresolved_events": int(frame.status.eq("unresolved_engineering_operation").sum()),
        "normal_eligible_resolved_events": int(frame.engineering_resolved_normal_eligible.sum()),
        "selected_event_budget_cells": int(frame.status.eq("selected").sum()),
        "paired_comparison_rows": len(pairs),
        "scope": "Each budget separately; one action per event; available-response zero means no qualifying positive action, not zero building power. Unavailable normal operation or engineering remains missing. Leave-one groups describe fixed-design sensitivity, not global confidence intervals.",
        "output_sha256": {p.name: digest(p) for p in out.glob("*.csv")}}
    write_json(out / "AUDIT.json", report)
    record_status("common_budget_dr_resource_accounting", report["status"], report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
