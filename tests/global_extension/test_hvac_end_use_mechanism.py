"""Tests for HVAC end-use decomposition of direct control comparisons."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.global_extension.hvac_end_use_mechanism import analyse


def test_analyse_joins_runs_and_reports_signed_end_use_changes() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "uniform",
                "heating_electric_kwh": 10.0,
                "cooling_electric_kwh": 40.0,
                "fans_electric_kwh": 5.0,
                "gas_hhv_kwh": 8.0,
                "facility_purchased_kwh": 55.0,
            },
            {
                "run_id": "centre",
                "heating_electric_kwh": 18.0,
                "cooling_electric_kwh": 43.0,
                "fans_electric_kwh": 6.0,
                "gas_hhv_kwh": 7.0,
                "facility_purchased_kwh": 67.0,
            },
            {
                "run_id": "range",
                "heating_electric_kwh": 14.0,
                "cooling_electric_kwh": 45.0,
                "fans_electric_kwh": 7.0,
                "gas_hhv_kwh": 6.0,
                "facility_purchased_kwh": 66.0,
            },
        ]
    )
    direct = pd.DataFrame(
        [
            {
                "alternative_run_id": "centre",
                "reference_run_id": "uniform",
                "alternative_control": "personal_center",
                "climate": "Beijing",
                "composition": "composition_1",
                "delta_facility_purchased_kwh": 12.0,
            },
            {
                "alternative_run_id": "range",
                "reference_run_id": "uniform",
                "alternative_control": "personal_range",
                "climate": "Beijing",
                "composition": "composition_2",
                "delta_facility_purchased_kwh": 11.0,
            },
        ]
    )

    detail, summary = analyse(direct, runs)

    assert detail.loc[0, "delta_terminal_heating_electric_kwh"] == pytest.approx(8.0)
    assert detail.loc[0, "delta_cooling_electric_kwh"] == pytest.approx(3.0)
    assert detail.loc[0, "delta_fans_electric_kwh"] == pytest.approx(1.0)
    assert detail.loc[0, "delta_space_heating_gas_kwh"] == pytest.approx(-1.0)
    assert detail.loc[0, "delta_whole_building_purchased_kwh"] == pytest.approx(12.0)
    assert summary["comparisons"] == 2
    assert summary["whole_building_energy_increases"] == 2


def test_analyse_rejects_inconsistent_whole_building_change() -> None:
    runs = pd.DataFrame(
        [
            {"run_id": "u", "heating_electric_kwh": 0, "cooling_electric_kwh": 0,
             "fans_electric_kwh": 0, "gas_hhv_kwh": 0, "facility_purchased_kwh": 1},
            {"run_id": "p", "heating_electric_kwh": 0, "cooling_electric_kwh": 0,
             "fans_electric_kwh": 0, "gas_hhv_kwh": 0, "facility_purchased_kwh": 2},
        ]
    )
    direct = pd.DataFrame(
        [{"alternative_run_id": "p", "reference_run_id": "u",
          "alternative_control": "personal_range", "climate": "Delhi",
          "composition": "composition_2", "delta_facility_purchased_kwh": 2.0}]
    )

    with pytest.raises(ValueError, match="does not match"):
        analyse(direct, runs)
