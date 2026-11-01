import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from dr_terminal_recovery import residuals


def annual():
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    return {"month": clock.month.to_numpy(), "day": clock.day.to_numpy(),
        "hour": clock.hour.to_numpy() + 1, "zone_ids": np.array(["A", "B"]),
        "occupied": np.ones(8760, bool), "dst": np.zeros(8760, int), "people": np.ones((8760, 2)),
        "temperatures": np.full((8760, 2), 23.0),
        "facility_electric_kwh": np.zeros(8760), "hvac_electric_kwh": np.zeros(8760),
        "gas_hhv_kwh": np.zeros(8760)}


def test_recovery_hours_preserve_signed_carriers_and_zone_aggregation():
    normal, action = annual(), annual()
    action["temperatures"] += np.array([1.0, -3.0])
    action["facility_electric_kwh"] += 2.0
    action["hvac_electric_kwh"] += 1.0
    action["gas_hhv_kwh"] -= 4.0
    event = {"start_timestamp": "2001-01-02 14:00", "duration_hours": 2, "recovery_hours": 48}
    rows = residuals(action, normal, event)
    assert len(rows) == 7
    assert rows[0]["audit_timestamp"] == "2001-01-04 15:00:00"
    assert rows[-1]["audit_timestamp"] == "2001-01-04 21:00:00"
    assert all(row["operative_temperature_max_abs_difference_c"] == 3.0 and
               row["operative_temperature_zone_mean_abs_difference_c"] == 2.0 for row in rows)
    assert sum(row["space_heating_gas_hhv_interval_mean_kw_difference"] for row in rows[1:]) == -24
    assert sum(row["facility_electricity_interval_mean_kw_difference"] for row in rows[1:]) == 12


def test_recovery_does_not_wrap_missing_year_end_followup_or_accept_shifted_clocks():
    normal, action = annual(), annual()
    event = {"start_timestamp": "2001-12-29 20:00", "duration_hours": 2, "recovery_hours": 48}
    rows = residuals(action, normal, event)
    assert [row["audit_hour_index"] for row in rows] == [0, 1, 2]
    action["hour"][0] = 2
    with pytest.raises(ValueError, match="unique|hourly"):
        residuals(action, normal, event)
    action = annual()
    action["people"][0, 0] = 2
    with pytest.raises(ValueError, match="people loads"):
        residuals(action, normal, event)
    action = annual()
    action["dst"][0] = 1
    with pytest.raises(ValueError, match="daylight-saving"):
        residuals(action, normal, event)
