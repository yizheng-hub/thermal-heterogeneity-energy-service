"""Test whether equally representative event dates change service-qualified response."""

from pathlib import Path
import subprocess
import sys

import pandas as pd

from common import RAW_RUN, ROOT
from multi_event_dr_extension import select_protocols
import dr_information_review


def paired_calendars():
    response = RAW_RUN / "response"
    reference = pd.read_csv(response / "REFERENCE_MANIFEST.csv")
    runs = RAW_RUN / "matrix/probe_runs/runs"
    primary = pd.read_csv(response / "EVENT_PROTOCOL.csv", parse_dates=["start_timestamp"])
    alternate = select_protocols(reference, runs, equivalent_dates=True)
    keys = ["climate", "season_block", "load_level"]
    columns = keys + ["event_id", "start_timestamp", "reference_mean_kw", "target_reference_mean_kw"]
    pairs = primary[columns].merge(alternate[columns], on=keys, validate="one_to_one",
                                   suffixes=("_primary", "_alternate"))
    if len(pairs) != len(primary) or len(pairs) != len(alternate):
        raise ValueError("The calendars must cover the same weather, season and load strata")
    pairs["date_changed"] = pairs.event_id_primary != pairs.event_id_alternate
    a = (pairs.reference_mean_kw_primary - pairs.target_reference_mean_kw_primary).abs()
    b = (pairs.reference_mean_kw_alternate - pairs.target_reference_mean_kw_alternate).abs()
    if ((a - b).abs() > 1e-9).any():
        raise ValueError("The two windows must be equally close to the target load")
    return pairs, alternate


def compare_power(results, mapping, calendar):
    data = pd.read_csv(results / "PAIRED_EVENT_INFORMATION_REVIEW.csv")
    rows = data.loc[data.control.eq("building_search") & data.normal_energy_reference_factor.eq(1)
                    & data.policy.eq("maximum_power")].copy()
    events = mapping[["climate", "season_block", "load_level", "date_changed", "event_id_" + calendar]]
    rows = rows.merge(events.rename(columns={"event_id_" + calendar: "event_id"}),
                      on=["climate", "event_id"], how="inner", validate="many_to_one")
    if len(rows) != 432 or rows.duplicated(["climate", "composition_id", "season_block", "load_level"]).any():
        raise ValueError("Expected one observation per setting, season and load stratum")
    rows["zone_average_kw"] = rows.mean_available_power_kw
    rows["individual_kw"] = rows.full_available_power_kw
    rows["rechecked_kw"] = rows.mean_available_power_kw.where(rows.mean_selected_full_service_qualified.eq(True), 0)
    supported = rows.normal_eligible_engineering_resolved_support.eq(True)
    power = ["zone_average_kw", "rechecked_kw", "individual_kw"]
    rows.loc[~supported, power] = float("nan")
    if rows.loc[supported, power].isna().any().any():
        raise ValueError("An engineering-resolved event has missing power")
    rows["calendar"] = calendar
    rows["resolved"] = supported
    return rows[["calendar", "climate", "composition_id", "season_block", "load_level", "event_id",
                 "date_changed", "resolved", *power]]


def main():
    out = ROOT / "results/event_date_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    mapping, alternate = paired_calendars()
    mapping.to_csv(out / "PAIRED_EVENT_DATES.csv", index=False)
    calendar = out / "ALTERNATE_EVENT_PROTOCOL.csv"
    alternate.to_csv(calendar, index=False)
    capability = ROOT / "results/equivalent_date_capability"
    for stage in ("plan", "baseline", "pilot", "full", "score"):
        subprocess.run([sys.executable, str(Path(__file__).with_name("dr_capability.py")),
                        stage, "--equivalent-dates"], check=True)
        if stage == "plan":
            binding = "NORMAL_OPERATION_BUDGET_STATUS.csv"
            standard = pd.read_csv(ROOT / "results/dr_capability" / binding)
            pd.testing.assert_frame_equal(standard, pd.read_csv(capability / binding))
    assessments = ROOT / "results/equivalent_date_information"
    dr_information_review.main(capability, assessments, calendar)
    primary = compare_power(ROOT / "results/dr_information_review", mapping, "primary")
    alternative = compare_power(assessments, mapping, "alternate")
    choices = []
    for label, folder in [("primary", ROOT / "results/dr_capability"), ("alternate", capability)]:
        actions = pd.read_csv(folder / "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")
        actions = actions.loc[actions.control.eq("building_search") & actions.normal_energy_reference_factor.eq(1)].copy()
        actions["calendar"] = label
        choices.append(actions[["calendar", "climate", "composition_id", "event_id", "positive_net_event_power_kw"]])
    rows = pd.concat([primary, alternative], ignore_index=True).merge(
        pd.concat(choices, ignore_index=True), on=["calendar", "climate", "composition_id", "event_id"],
        how="left", validate="one_to_one", indicator=True)
    if not rows["_merge"].eq("both").all():
        raise ValueError("Missing full-window electricity assessment for a calendar opportunity")
    rows.drop(columns="_merge", inplace=True)
    rows.loc[rows.resolved, "positive_net_event_power_kw"] = rows.loc[rows.resolved, "positive_net_event_power_kw"].fillna(0)
    rows.loc[~rows.resolved, "positive_net_event_power_kw"] = float("nan")
    strata = ["climate", "composition_id", "season_block", "load_level"]
    rows["paired_support"] = rows.groupby(strata).resolved.transform("all")
    rows.to_csv(out / "PAIRED_EVENT_POWER.csv", index=False)
    summary = []
    for label, subset in rows.groupby("calendar"):
        for scope, mask in [("all", subset.date_changed.notna()), ("exchanged", subset.date_changed)]:
            group = subset.loc[mask]
            matched = group.loc[group.paired_support]
            summary.append(dict(calendar=label, scope=scope, opportunities=len(group),
                                resolved_both=len(matched), **matched[["zone_average_kw", "rechecked_kw",
                                "individual_kw", "positive_net_event_power_kw"]].mean().to_dict()))
    pd.DataFrame(summary).to_csv(out / "EVENT_DATE_POWER_SUMMARY.csv", index=False)


if __name__ == "__main__":
    main()
