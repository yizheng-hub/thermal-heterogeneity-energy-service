"""Protect ventilation-probe unit handling and retention of HVAC warnings."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "src/utilities/audit_ventilation_adaptation.py"
SPEC = importlib.util.spec_from_file_location("ventilation_probe", PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_oa_and_vrp_request_are_not_confused():
    frame = pd.DataFrame({
        "AHU:Air System Outdoor Air Mass Flow Rate [kg/s](Hourly)": [1.0],
        "AHU:Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate [kg/s](Hourly)": [1.1],
        "Heating:NaturalGas [J](Hourly)": [3.6e6],
    })
    assert PROBE.series(frame, "AHU", "Air System Outdoor Air Mass Flow Rate")[0] == 1.0
    assert PROBE.series(frame, "AHU", "Mechanical Ventilation Requested")[0] == 1.1
    assert PROBE.meter(frame, "Heating:NaturalGas")[0] == 1.0


def test_unexpected_duplicate_meter_is_rejected():
    frame = pd.DataFrame({"Fans:Electricity [J](Hourly)": [1], "Fans:Electricity [J](Timestep)": [2]})
    with pytest.raises(AssertionError):
        PROBE.meter(frame, "Fans:Electricity")


def test_source_dhw_warning_does_not_hide_hvac_numerical_warning(tmp_path):
    (tmp_path / "eplusout.err").write_text(
        '   *************  ** Warning ** "CORE_BOTTOM SUB CAT" - Target water temperature should be less than or equal to the hot water temperature error continues...\n'
        '   *************  **   ~~~   **   This error occurred 593143 total times;\n'
        '   ** Warning ** SimHVAC: Maximum iterations exceeded\n'
        '   *************  ** Warning ** SimHVAC maximum iterations continues...\n'
        '   *************  **   ~~~   **   This error occurred 10 total times;\n'
    )
    report = PROBE.warnings(tmp_path)
    assert len(report["recurring_groups"]) == 2
    assert len(report["unexplained_recurring_groups"]) == 1
    assert "SimHVAC" in report["unexplained_recurring_groups"][0]["message"]
    assert len(report["direct_hvac_numerical_messages"]) == 1
