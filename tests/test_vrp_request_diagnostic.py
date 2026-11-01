import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import vrp_request_diagnostic as module


def database():
    connection = sqlite3.connect(":memory:")
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    frame = pd.DataFrame({"TimeIndex": np.arange(1, 8761), "Month": clock.month,
        "Day": clock.day, "Hour": clock.hour + 1, "Dst": 0, "Interval": 60,
        "EnvironmentPeriodIndex": 1, "WarmupFlag": 0})
    frame.to_sql("Time", connection, index=False)
    pd.DataFrame({"EnvironmentPeriodIndex": [1], "EnvironmentType": [3]}).to_sql("EnvironmentPeriods", connection, index=False)
    dictionary, values = [], []
    for kind, variable in enumerate([module.ACTUAL, module.REQUESTED]):
        for loop, key in enumerate(["BOTTOM", "MIDDLE", "TOP"]):
            index = kind * 3 + loop + 1
            dictionary.append({"ReportDataDictionaryIndex": index, "KeyValue": key,
                "Name": variable, "Units": "kg/s", "ReportingFrequency": "Hourly"})
            value = np.ones(8760)
            if kind == 0 and loop == 1:
                value[:2] = [.98, 1.2]
            values.extend({"TimeIndex": number, "ReportDataDictionaryIndex": index, "Value": v}
                          for number, v in enumerate(value, 1))
    pd.DataFrame(dictionary).to_sql("ReportDataDictionary", connection, index=False)
    pd.DataFrame(values).to_sql("ReportData", connection, index=False)
    data = {"month": clock.month.to_numpy(), "day": clock.day.to_numpy(),
            "hour": clock.hour.to_numpy() + 1, "dst": np.zeros(8760, int),
            "occupied": np.ones(8760, bool)}
    return connection, data


def test_vrp_comparison_pairs_native_airloops_and_keeps_economizer_excess():
    connection, data = database()
    try:
        values = pd.DataFrame(module.loop_statistics(connection, data)).set_index("airloop")
        assert len(values) == 3
        assert values.loc["MIDDLE", "hours_below99pct_controller_request"] == 1
        assert values.loc["MIDDLE", "actual_to_vrp_request_ratio_min"] == .98
        assert values.loc["MIDDLE", "actual_intake_mass_kg"] > values.loc["MIDDLE", "controller_requested_mass_kg"]
        assert values.loc["BOTTOM", "hours_below99pct_controller_request"] == 0
    finally:
        connection.close()


def test_vrp_rejects_a_calendar_shift_and_wrong_units():
    connection, data = database()
    try:
        data["dst"][0] = 1
        with pytest.raises(ValueError, match="calendars differ"):
            module.loop_statistics(connection, data)
        data["dst"][0] = 0
        connection.execute("update ReportDataDictionary set Units='m3/s' where ReportDataDictionaryIndex=1")
        with pytest.raises(ValueError, match="incorrect-unit"):
            module.loop_statistics(connection, data)
    finally:
        connection.close()
