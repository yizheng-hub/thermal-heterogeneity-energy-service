"""Reviewer sensitivities must remain derived from frozen results."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/analysis/targeted_extension/reviewer_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("reviewer_sensitivity", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_reviewer_sensitivity_counts_and_closure(tmp_path):
    if not (ROOT / "results/Joint_main/annual/ANNUAL_ALL_CANDIDATES.csv").is_file():
        pytest.skip("requires retained local simulation results")
    report = MODULE.run(ROOT, tmp_path)
    assert report["reference_limit_pairs"] == 96
    assert report["reference_limit_same_physical_pairs"] == 67
    assert report["reference_limit_same_pairs_both_common25"] == 67
    assert report["no_common25_selected_conditions"] == {
        "ltpref_interval": 44, "personal_center": 48, "uniform_average": 34}
    assert report["observed_service_conditions_with_distinct_noncommon_match"] == 89
    assert report["strict_half_reference_selected_conditions"] == {
        "ltpref_interval": 4, "personal_center": 4, "uniform_average": 0}
    assert report["direct_all_three_engineering_qualified_conditions"] == 24
    assert report["selected_july_response_cases"] == 64
    assert report["selected_july_unique_event_trajectories"] == 28
    assert report["maximum_abs_48h_reported_recovery_difference_kwh"] < 1e-10
    assert report["maximum_abs_48h_reported_net_difference_kwh"] < 1e-10


def test_horizon_energy_identity_and_terminal_state():
    hours = 200
    base = {"hvac_electric_kwh": np.full(hours, 10.),
            "temperatures": np.full((hours, 2), 24.)}
    event = {"hvac_electric_kwh": np.full(hours, 10.),
             "temperatures": np.full((hours, 2), 24.)}
    event["hvac_electric_kwh"][10:12] = 8.
    event["hvac_electric_kwh"][12:16] = 10.5
    event["temperatures"][15] = [24.1, 23.9]
    row = MODULE.recovery_horizon_metrics(base, event, 10, horizons=(4,))[0]
    assert row["event_electric_saving_kwh"] == 4.
    assert row["cumulative_recovery_extra_electric_kwh"] == 2.
    assert row["event_plus_recovery_net_electric_saving_kwh"] == 2.
    assert np.isclose(row["terminal_max_abs_zone_temperature_difference_c"], .1)
    assert abs(row["electricity_closure_error_kwh"]) < 1e-12
