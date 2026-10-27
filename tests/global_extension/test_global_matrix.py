from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

from global_matrix import (  # noqa: E402
    ClimateSpec,
    PROBE_TARGETS_C,
    control_manifest,
    extract_hourly,
    extreme_weeks,
    non_interface_fingerprint,
    parse_epw_header,
    parse_idf,
    patch_control_interface,
    probe_manifest,
    replace_site_and_sizing,
)


def _epw(path: Path) -> None:
    headers = [
        "LOCATION,Beijing,Beijing,CHN,TEST,545110,39.8,116.47,8,31.3",
        "DESIGN CONDITIONS,0",
        ("TYPICAL/EXTREME PERIODS,2,Summer - Week Nearest Max Temperature For Period,"
         "Extreme,7/27,8/2,Winter - Week Nearest Min Temperature For Period,"
         "Extreme,1/13,1/19"),
        "GROUND TEMPERATURES,0",
        "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
        "COMMENTS 1,test",
        "COMMENTS 2,test",
        "DATA PERIODS,1,1,Data,Sunday,1/1,12/31",
    ]
    data = ["2020,1,1,1,60,0,-5,0,0"] * 24 * 14
    path.write_text("\n".join(headers + data) + "\n", encoding="utf-8")


def test_epw_location_and_extreme_weeks(tmp_path: Path):
    weather = tmp_path / "weather.epw"
    _epw(weather)
    header = parse_epw_header(weather)
    assert header["city"] == "Beijing"
    assert header["country"] == "CHN"
    assert extreme_weeks(weather) == {
        "heating": (1, 13, 1, 19),
        "cooling": (7, 27, 8, 2),
    }


def test_site_sizing_replaces_old_location_and_design_days(tmp_path: Path):
    weather = tmp_path / "weather.epw"
    _epw(weather)
    source = """Version,22.1;
Site:Location,Old,1,2,3,4;
SizingPeriod:DesignDay,Old heat;
SizingPeriod:DesignDay,Old cool;
Building,Office;
"""
    result = replace_site_and_sizing(source, weather)
    objects = parse_idf(result)
    assert len([x for x in objects if x[0].lower() == "site:location"]) == 1
    assert not [x for x in objects if x[0].lower() == "sizingperiod:designday"]
    periods = [x for x in objects if x[0].lower() == "sizingperiod:weatherfiledays"]
    assert len(periods) == 2
    assert periods[0][1] == "EPW cold extreme week"


def test_interface_patch_preserves_undeclared_objects():
    zones = [f"Zone {index}" for index in range(15)]
    source = ["Version,22.1;", "Building,Office;"]
    reference = ["Version,22.1;", "Building,Office;"]
    for zone in zones:
        source.append(
            f"ThermostatSetpoint:DualSetpoint,{zone} control,Shared heat,Shared cool;"
        )
        reference.extend([
            (f"ThermostatSetpoint:DualSetpoint,{zone} control,"
             f"{zone} independent heating,{zone} independent cooling;"),
            (f"Schedule:Compact,{zone} independent heating,Temperature,"
             "Through:12/31,For:Weekdays,Until:24:00,23.5;"),
            (f"Schedule:Compact,{zone} independent cooling,Temperature,"
             "Through:12/31,For:Weekdays,Until:24:00,26.5;"),
            (f"EnergyManagementSystem:Actuator,{zone}_independent_2_actuator,"
             f"{zone} independent heating,Schedule:Compact,Schedule Value;"),
            (f"EnergyManagementSystem:Actuator,{zone}_independent_3_actuator,"
             f"{zone} independent cooling,Schedule:Compact,Schedule Value;"),
        ])
    for name in (
        "OA_BASED_OPTIMUM_START_EMS_Program",
        "pacu_vav_bot_MAX_OA_FRAC",
        "pacu_vav_mid_MAX_OA_FRAC",
        "pacu_vav_top_MAX_OA_FRAC",
    ):
        source.append(f"EnergyManagementSystem:Program,{name},SET x = 0;")
        reference.append(f"EnergyManagementSystem:Program,{name},SET x = 1;")
    reference.extend([
        "Output:Variable,*,Zone Operative Temperature,Hourly;",
        "Output:Variable,*,Zone People Occupant Count,Hourly;",
        "Output:Meter,Heating:Electricity,Hourly;",
        "Output:SQLite,SimpleAndTabular;",
    ])
    source_text = "\n".join(source)
    result = patch_control_interface(source_text, "\n".join(reference))
    assert non_interface_fingerprint(source_text) == non_interface_fingerprint(result)
    objects = parse_idf(result)
    assert len([x for x in objects if x[0].lower() == "output:variable"]) == 2
    assert len([x for x in objects if x[0].lower() == "schedule:compact"]) == 30


def test_declared_matrix_counts():
    climates = [
        ClimateSpec(f"c{i}", "country", "continent", "variant", "weather.epw", i < 3)
        for i in range(9)
    ]
    probes = probe_manifest(climates)
    controls = control_manifest(climates, [f"p{i}" for i in range(6)])
    assert len(PROBE_TARGETS_C) == 31
    assert len(probes) == 279
    assert len(controls) == 378
    assert not probes.run_id.duplicated().any()
    assert not controls.run_id.duplicated().any()
    assert set(controls[~controls.sentinel].offset_c) == {0.0}
    assert set(controls[controls.sentinel].offset_c) == {-1.0, -0.5, 0.0, 0.5, 1.0}

    full_controls = control_manifest(climates, [f"p{i}" for i in range(6)], "all")
    assert len(full_controls) == 810
    assert set(full_controls.offset_c) == {-1.0, -0.5, 0.0, 0.5, 1.0}


def test_control_manifest_rejects_unknown_candidate_scope():
    climate = ClimateSpec("c", "country", "continent", "variant", "weather.epw", True)
    with pytest.raises(ValueError, match="Unknown candidate scope"):
        control_manifest([climate], ["p"], "unknown")


def test_compact_extraction_includes_whole_building_electricity(tmp_path: Path):
    hours = 8760
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    frame = pd.DataFrame({
        "Zone A:Zone Operative Temperature [C](Hourly)": np.full(hours, 25.0),
        "Zone A:Zone People Occupant Count [](Hourly)": np.ones(hours),
        "Electricity:Facility [J](Hourly)": np.full(hours, 18.0e6),
        "Electricity:HVAC [J](Hourly)": np.full(hours, 16.2e6),
        "Heating:Electricity [J](Hourly)": np.full(hours, 3.6e6),
        "Cooling:Electricity [J](Hourly)": np.full(hours, 7.2e6),
        "Fans:Electricity [J](Hourly)": np.full(hours, 3.6e6),
        "Heating:NaturalGas [J](Hourly)": np.full(hours, 10.8e6),
    })
    frame.to_csv(run_folder / "eplusout.csv", index=False)
    with sqlite3.connect(run_folder / "eplusout.sql") as connection:
        connection.execute(
            "create table Time (TimeIndex integer, Month integer, Day integer, "
            "Hour integer, Dst integer, Interval integer, DayType text)"
        )
        connection.executemany(
            "insert into Time values (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    index + 1,
                    (index // 730) + 1,
                    1,
                    index % 24 + 1,
                    0,
                    60,
                    "Monday",
                )
                for index in range(hours)
            ],
        )
        connection.execute(
            "create table TabularDataWithStrings ("
            "ReportName text, ReportForString text, TableName text, "
            "RowName text, ColumnName text, Units text, Value text)"
        )
        connection.execute(
            "insert into TabularDataWithStrings values (?, ?, ?, ?, ?, ?, ?)",
            (
                "AnnualBuildingUtilityPerformanceSummary",
                "Entire Facility",
                "End Uses",
                "Total End Uses",
                "Natural Gas",
                "GJ",
                "108.0",
            ),
        )
    (run_folder / "eplusout.err").write_text(
        "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;\n",
        encoding="utf-8",
    )

    compact = tmp_path / "annual.npz"
    extract_hourly(run_folder, ["Zone A"], compact)
    with np.load(compact, allow_pickle=False) as stored:
        assert np.allclose(stored["facility_electric_kwh"], 5.0)
        assert np.isclose(
            np.asarray(stored["facility_gas_hhv_kwh_annual"]).reshape(-1)[0],
            30_000.0,
        )
        assert np.allclose(stored["hvac_electric_kwh"], 4.5)
        assert np.allclose(stored["gas_hhv_kwh"], 3.0)
