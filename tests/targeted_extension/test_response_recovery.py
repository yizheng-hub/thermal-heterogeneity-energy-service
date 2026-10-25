"""Independent synthetic checks of signed carriers and sequential service gates."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/analysis/targeted_extension/response_recovery.py"
SPEC = importlib.util.spec_from_file_location("response_recovery", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(amplitude, power, **changed):
    row = {"source_scope": "test", "context": "test", "composition_id": 1, "layout": "random",
           "permission": "zone", "control": "uniform_average", "date_class": "test", "event_month": 7,
           "event_day": 1, "service_fraction": 1., "baseline_run_id": "own", "run_id": str(amplitude),
           "amplitude_c": amplitude, "engineering_status": "new_qualified", "baseline_available": True}
    for window in ["event", "recovery"]:
        for metric in ["mean", "fixed_tail"]:
            row[f"{window}_{metric}_budget_c"] = 1.
            for trajectory in ["baseline", "perturbed"]:
                row[f"{window}_{trajectory}_{metric}_c"] = .5
    row.update({metric: 0. for metric in MODULE.METRICS})
    row.update(event_average_reduction_kw=power, event_electric_saving_kwh=2 * power)
    return row | changed


def test_signed_rebound_can_reverse_event_electric_saving_and_gas_is_separate():
    hours = pd.DataFrame({"hour_index": np.arange(50), "baseline_electric_kwh": 10.,
                          "perturbed_electric_kwh": [5.] * 2 + [11.] * 48,
                          "baseline_gas_hhv_kwh": 2., "perturbed_gas_hhv_kwh": [1.] * 2 + [3.] * 48})
    metrics = MODULE.energy_accounting(hours.sample(frac=1, random_state=0))
    assert metrics["event_average_reduction_kw"] == 5.
    assert metrics["event_electric_saving_kwh"] == 10.
    assert metrics["recovery_extra_electric_kwh"] == 48.
    assert metrics["event_plus48h_net_electric_saving_kwh"] == -38.
    assert metrics["event_gas_change_kwh_hhv"] == -2.
    assert metrics["recovery_gas_change_kwh_hhv"] == 48.
    assert metrics["event_plus48h_gas_change_kwh_hhv"] == 46.
    with pytest.raises(ValueError, match="exactly"):
        MODULE.energy_accounting(hours.iloc[:-1])


def test_baseline_recovery_is_not_silently_treated_as_eligible_zero():
    frame = MODULE.classify_gates(pd.DataFrame([candidate(.5, 5., recovery_baseline_fixed_tail_c=1.1)]))
    assert frame.raw_positive.iloc[0]
    assert not frame.baseline_qualified.iloc[0]
    stages = MODULE.select_stages(frame)
    assert stages.loc[stages.stage.eq("raw_positive"), "selected"].iloc[0]
    final = stages.loc[stages.stage.eq("recovery_qualified")].iloc[0]
    assert final.response_kw_with_no_action_zero == 0.
    assert np.isnan(final.service_qualified_available_kw)


def test_event_and_recovery_filters_reselect_amplitude_without_adding_powers():
    frame = MODULE.classify_gates(pd.DataFrame([
        candidate(1., 10., recovery_perturbed_mean_c=1.1), candidate(.5, 4.)]))
    stages = MODULE.select_stages(frame).set_index("stage")
    assert stages.loc["event_qualified", "response_kw_with_no_action_zero"] == 10.
    assert stages.loc["recovery_qualified", "response_kw_with_no_action_zero"] == 4.
    assert stages.loc["recovery_qualified", "response_kw_lost_from_previous_stage"] == 6.
    assert stages.loc["recovery_qualified", "selected_amplitude_c"] == .5


def test_failed_and_nonpositive_candidates_remain_in_denominator():
    frame = MODULE.classify_gates(pd.DataFrame([
        candidate(.5, -2.), candidate(1., 20., engineering_status="execution_failed")]))
    assert len(frame) == 2
    assert frame.first_exclusion.tolist() == ["nonpositive_response", "engineering"]
    final = MODULE.select_stages(frame).iloc[-1]
    assert final.logical_candidates == 2
    assert final.engineering_candidates == 1
    assert not final.selected
    assert final.service_qualified_available_kw == 0.


def test_single_floor_permission_alias_is_verified_before_deduplication():
    frame = MODULE.classify_gates(pd.DataFrame([
        candidate(.5, 4., permission="building"), candidate(.5, 4., permission="floor")]))
    settings = {"test": {"floor_binding_equals_building": True}}
    marked = MODULE.mark_spatial_aliases(frame, settings)
    assert marked.distinct_spatial_condition.tolist() == [True, False]
    frame.loc[frame.permission.eq("floor"), "run_id"] = "different"
    with pytest.raises(ValueError, match="aliases differ"):
        MODULE.mark_spatial_aliases(frame, settings)
