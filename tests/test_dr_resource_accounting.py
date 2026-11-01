import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from summarize_dr_capability import availability_table, summarize_events, paired_events, add_action_details


def choice(status, baseline="feasible", **values):
    return {"status": status, "baseline_status": baseline, "event_power_reduction_kw": np.nan,
        "positive_net_selected_action": None, "positive_net_event_power_kw": np.nan,
        "positive_net_full_window_kwh": np.nan, "net_full_window_electricity_reduction_kwh": np.nan,
        "selected_action": None, "absolute_event_facility_mean_kw": np.nan, **values}


def test_zero_available_response_is_not_a_zero_building_or_missing_baseline():
    rows = [choice("selected", event_power_reduction_kw=10.0, selected_action="shed",
                   positive_net_selected_action="precondition", positive_net_event_power_kw=8.0,
                   positive_net_full_window_kwh=2.0, net_full_window_electricity_reduction_kwh=-1.0),
        choice("infeasible_service_preserving_positive_response"),
        choice("unresolved_engineering_operation"),
        choice("infeasible_normal_energy_budget", "infeasible_normal_energy_budget")]
    frame = availability_table(pd.DataFrame(rows).assign(control="full", normal_energy_reference_factor=1.0,
        climate=["A", "B", "C", "D"], composition_id=1))
    assert frame.available_event_power_kw.iloc[:2].tolist() == [10, 0]
    assert frame.available_event_power_kw.iloc[2:].isna().all()
    assert frame.available_positive_net_event_power_kw.iloc[:2].tolist() == [8, 0]
    row = summarize_events(frame, ["control", "normal_energy_reference_factor"]).iloc[0]
    assert row.planned_events == 4 and row.normal_eligible_events == 3 and row.engineering_resolved_events == 2
    assert row.available_event_power_mean_kw == 5 and row.selected_event_power_mean_kw == 10
    assert row.available_positive_net_event_power_mean_kw == 4
    assert row.selected_peak_action_net_electricity_mean_kwh == -1
    empty_support = summarize_events(frame.iloc[2:], ["control", "normal_energy_reference_factor"]).iloc[0]
    assert np.isnan(empty_support.available_event_power_mean_kw)
    assert np.isnan(empty_support.positive_response_fraction_among_resolved)


def test_pairing_includes_unilateral_no_response_and_excludes_unresolved_support():
    rows = []
    for control, power in [("building_search", 0), ("zone_summary", 5), ("zone_full", 10)]:
        for budget in [0.95, 1.0, 1.05]:
            for i in range(432):
                item = choice("infeasible_normal_energy_budget", "infeasible_normal_energy_budget") if budget == 0.95 else (
                    choice("selected", event_power_reduction_kw=power, selected_action="shed") if power else
                    choice("infeasible_service_preserving_positive_response"))
                if control == "zone_full" and budget == 1.0 and i == 0:
                    item = choice("unresolved_engineering_operation")
                if budget != 0.95 and i == 1 and control == "building_search":
                    item = choice("selected", event_power_reduction_kw=3.0, selected_action="shed")
                if budget != 0.95 and i == 1 and control == "zone_full":
                    item = choice("infeasible_service_preserving_positive_response")
                rows.append({**item, "control": control, "climate": "C" + str(i // 48),
                    "composition_id": (i // 8) % 6, "event_id": "E" + str(i % 8),
                    "normal_energy_reference_factor": budget})
    result = paired_events(availability_table(pd.DataFrame(rows)))
    pair = result[result.compared_library.eq("zone_full") & result.reference_library.eq("building_search")]
    assert len(pair) == 1296
    assert pair.common_normal_eligible_engineering_resolved_support.sum() == 863
    assert pair.available_event_power_kw_paired_difference.dropna().value_counts().to_dict() == {10.0: 861, -3.0: 2}
    assert pair.loc[pair.normal_energy_reference_factor.eq(0.95), "available_event_power_kw_paired_difference"].isna().all()


def test_selected_action_carriers_are_taken_from_its_own_physical_score():
    keys = {"climate": "A", "composition_id": 1, "control": "zone_full", "event_id": "E"}
    selected = choice("selected", selected_action="peak", positive_net_selected_action="net",
        event_power_reduction_kw=2.0, positive_net_event_power_kw=1.0, absolute_event_facility_mean_kw=8.0,
        net_full_window_electricity_reduction_kwh=-1.0, positive_net_full_window_kwh=1.0)
    choices = availability_table(pd.DataFrame([{**keys, **selected}]))
    scores = pd.DataFrame([{**keys, "status": "evaluated", "action_id": action, "baseline_facility_mean_kw": 10.0,
        "event_facility_mean_power_reduction_kw": power, "event_facility_electricity_reduction_kwh": 2 * power,
        "preconditioning_facility_electricity_change_kwh": pre, "recovery_facility_electricity_change_kwh": 1.0,
        "precondition_event_recovery_facility_net_reduction_kwh": 2 * power - pre - 1,
        "event_gas_reduction_kwh_hhv": 2.0, "preconditioning_gas_change_kwh_hhv": 0.0,
        "recovery_gas_change_kwh_hhv": 1.0, "precondition_event_recovery_net_gas_reduction_kwh_hhv": 1.0}
        for action, power, pre in [("peak", 2.0, 4.0), ("net", 1.0, 0.0)]])
    result = add_action_details(choices, scores)
    assert result.peak_preconditioning_facility_electricity_change_kwh.iloc[0] == 4
    assert result.positive_net_preconditioning_facility_electricity_change_kwh.iloc[0] == 0
    assert result.normal_event_power_kw.iloc[0] == 10
    scores.loc[0, "recovery_gas_change_kwh_hhv"] = 99
    with pytest.raises(ValueError, match="carrier accounting"):
        add_action_details(choices, scores)
