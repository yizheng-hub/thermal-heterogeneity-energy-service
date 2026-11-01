from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.energy_service.generate_model_inputs import physical_hash
from analysis.global_extension.seasonal_command_extension import (
    BLOCKS,
    demand_shift,
    optimise_seasonal_targets,
    fixed_tail_for_seasonal_demand,
    require_complete_seasonal_execution,
    recover_existing_plan,
    score_completed_plan,
    score_seasonal_trajectory,
    seasonal_annual_input,
)
from analysis.global_extension.global_matrix import sha256


def test_fresh_seasonal_execution_passes_an_explicit_string_zone_axis(tmp_path, monkeypatch):
    import analysis.global_extension.seasonal_command_extension as module
    runtime = tmp_path / "energyplus.exe"
    runtime.write_bytes(b"test runtime")
    def inspect_zones(_idf, _weather, _runtime, _folder, _compact, zones, **kwargs):
        values = np.asarray(zones)
        assert values.ndim == 1 and values.dtype.kind == "U"
        assert values.tolist() == ["Zone B", "Zone A"]
        raise RuntimeError("zone interface verified before simulation")
    monkeypatch.setattr(module, "run_energyplus", inspect_zones)
    runs = pd.DataFrame([{"run_id": "test", "climate": "c", "input_idf_path": str(tmp_path / "in.idf"),
        "weather_path": str(tmp_path / "weather.epw"), "targets_json": json.dumps({"DJF": {"Zone B": 25., "Zone A": 24.}})}])
    probes = pd.DataFrame([{"run_id": "probe", "climate": "c", "target_c": 25.}])
    with pytest.raises(RuntimeError, match="zone interface verified"):
        module.execute_plan(runs, probes, tmp_path / "probes", tmp_path / "output", runtime, workers=1)


def test_recover_existing_plan_extracts_completed_raw_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "seasonal"
    run_id = "seasonal_test"
    folder = output / "runs" / run_id
    folder.mkdir(parents=True)
    for name in ("eplusout.csv", "eplusout.sql"):
        (folder / name).write_text("raw", encoding="utf-8")
    (folder / "eplusout.err").write_text(
        "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.idf"
    weather_path = tmp_path / "weather.epw"
    runtime = tmp_path / "energyplus.exe"
    for path in (input_path, weather_path, runtime):
        path.write_text(path.name, encoding="utf-8")
    runs = pd.DataFrame([{
        "run_id": run_id,
        "climate": "Test",
        "composition_id": 1,
        "control": "uniform_average",
        "seasonal_contrast_fraction": 1.0,
        "targets_json": json.dumps({"DJF": {"Zone A": 24.0}}),
        "input_idf_path": input_path,
        "weather_path": weather_path,
    }])
    probe_root = tmp_path / "probes"
    baseline = probe_root / "runs" / "reference"
    baseline.mkdir(parents=True)
    (baseline / "eplusout.err").write_text("baseline", encoding="utf-8")
    probes = pd.DataFrame([{
        "run_id": "reference", "climate": "Test", "target_c": 25.0,
    }])

    def fake_extract(_folder, zones, compact):
        assert zones == ["Zone A"]
        compact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            compact,
            month=np.ones(8760, dtype=int),
            facility_electric_kwh=np.ones(8760),
            facility_gas_hhv_kwh_annual=np.asarray(2.0),
        )
        return {
            "completed": True, "engineering_pass": True,
            "hourly_sha256": sha256(compact),
        }

    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.extract_hourly",
        fake_extract,
    )
    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.warning_profile",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.relative_engineering_qa",
        lambda *_args: {
            "engineering_pass_relative_to_prototype_baseline": True,
            "relative_engineering_pass": True,
        },
    )
    recovered = recover_existing_plan(
        runs, probes, probe_root, output, runtime, cleanup_raw=False,
    )
    assert len(recovered) == 1
    assert bool(recovered.relative_engineering_pass.iloc[0])
    assert Path(recovered.hourly_path.iloc[0]).is_file()
    assert (output / "SEASONAL_COMMAND_RECOVERY_AUDIT.csv").is_file()
    assert (output / "SEASONAL_COMMAND_RUN_MANIFEST_COMPLETED.csv").is_file()


def test_failed_recovery_gate_preserves_primary_raw_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "seasonal"
    run_id = "seasonal_warning"
    folder = output / "runs" / run_id
    folder.mkdir(parents=True)
    for name in ("eplusout.csv", "eplusout.sql"):
        (folder / name).write_text("raw", encoding="utf-8")
    (folder / "eplusout.err").write_text(
        "EnergyPlus Completed Successfully-- 1 Warning; 0 Severe Errors;\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.idf"
    weather_path = tmp_path / "weather.epw"
    runtime = tmp_path / "energyplus.exe"
    for path in (input_path, weather_path, runtime):
        path.write_text(path.name, encoding="utf-8")
    runs = pd.DataFrame([{
        "run_id": run_id,
        "climate": "Test",
        "composition_id": 1,
        "control": "personal_center",
        "seasonal_contrast_fraction": 1.0,
        "targets_json": json.dumps({"DJF": {"Zone A": 24.0}}),
        "input_idf_path": input_path,
        "weather_path": weather_path,
    }])
    probe_root = tmp_path / "probes"
    baseline = probe_root / "runs" / "reference"
    baseline.mkdir(parents=True)
    (baseline / "eplusout.err").write_text("baseline", encoding="utf-8")
    probes = pd.DataFrame([{
        "run_id": "reference", "climate": "Test", "target_c": 25.0,
    }])

    def fake_extract(_folder, _zones, compact):
        compact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(compact, month=np.ones(8760, dtype=int))
        return {
            "completed": True,
            "reported_severe_count": 0,
            "fatal_occurrences": 0,
            "unexplained_recurring_groups": [],
            "hourly_sha256": sha256(compact),
        }

    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.extract_hourly",
        fake_extract,
    )
    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.warning_profile",
        lambda _path: {},
    )
    monkeypatch.setattr(
        "analysis.global_extension.seasonal_command_extension.relative_engineering_qa",
        lambda *_args: {
            "engineering_pass_relative_to_prototype_baseline": False,
            "new_warning_categories": ["hvac_iterations"],
            "critical_warning_count_increases": {
                "hvac_iterations": {"baseline": 0, "treatment": 1}
            },
            "treatment_warning_profile": {"hvac_iterations": 1},
        },
    )
    with pytest.raises(RuntimeError, match="completeness/engineering gate"):
        recover_existing_plan(
            runs, probes, probe_root, output, runtime, cleanup_raw=True,
        )
    assert (folder / "eplusout.sql").is_file()
    assert (folder / "eplusout.csv").is_file()
    assert (output / "SEASONAL_COMMAND_RECOVERY_AUDIT.csv").is_file()
    assert not (output / "SEASONAL_COMMAND_RUN_MANIFEST_COMPLETED.csv").exists()


def test_seasonal_execution_gate_rejects_failed_or_missing_runs():
    failed = pd.DataFrame([
        {"run_id": "a", "completed": True, "relative_engineering_pass": True},
        {"run_id": "b", "completed": True, "relative_engineering_pass": False},
    ])
    with pytest.raises(RuntimeError, match="completeness/engineering gate"):
        require_complete_seasonal_execution(failed, 2)

    incomplete = failed.iloc[:1].assign(relative_engineering_pass=True)
    with pytest.raises(RuntimeError, match="expected=2, observed=1"):
        require_complete_seasonal_execution(incomplete, 2)


def _profiles() -> pd.DataFrame:
    return pd.DataFrame({
        "profile_id": ["a", "b"],
        "zone_id": ["Zone A", "Zone A"],
        "centre_c": [23.0, 25.0],
        "controller_lower_c": [22.0, 24.0],
        "controller_upper_c": [24.0, 26.0],
    })


def _probes() -> dict[str, np.ndarray]:
    grid = np.array([20.0, 25.0, 27.0])
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    indices = np.flatnonzero((clock.hour >= 8) & (clock.hour < 18))
    operative = np.empty((len(grid), len(indices), 1))
    for index, target in enumerate(grid):
        operative[index, :, 0] = target
    return {
        "uniform_target_c": grid,
        "operative_c": operative,
        "native_people": np.ones((len(indices), 1)),
        "zone_ids": np.array(["Zone A"]),
        "occupied_source_indices": indices,
    }


def test_seasonal_targets_use_local_season_shift_and_probe_subset():
    targets, diagnostics = optimise_seasonal_targets(
        _probes(), _profiles(), latitude=40.0, contrast_c=2.0, fraction=1.0
    )
    assert set(targets) == set(BLOCKS)
    assert demand_shift(2.0, 1.0, "winter") == -1.0
    assert demand_shift(2.0, 1.0, "summer") == 1.0
    assert targets["DJF"]["uniform_average"]["Zone A"] < targets["JJA"][
        "uniform_average"
    ]["Zone A"]
    assert {row["local_season"] for row in diagnostics} == {
        "winter", "spring", "summer", "autumn"
    }
    assert len(diagnostics) == len(BLOCKS) * 3
    assert {row["control"] for row in diagnostics} == {
        "uniform_average", "personal_center", "personal_range"
    }
    assert all(row["zone_count"] == 1 for row in diagnostics)
    assert all(0.0 <= row["share_at_grid_boundary"] <= 1.0 for row in diagnostics)
    for row in diagnostics:
        assert row["zones_at_grid_boundary"] == (
            row["zones_at_grid_min"] + row["zones_at_grid_max"]
        )


def test_seasonal_schedule_changes_only_thermostat_schedules():
    source = """
Schedule:Compact, Heat, Temperature, Through: 12/31, For: Weekdays Saturday,
 Until: 06:00, 18, Until: 22:00, 23.5, Until: 24:00, 18;
Schedule:Compact, Cool, Temperature, Through: 12/31, For: Weekdays Saturday,
 Until: 06:00, 30, Until: 22:00, 26.5, Until: 24:00, 30;
ThermostatSetpoint:DualSetpoint, Dual, Heat, Cool;
ZoneControl:Thermostat, Tstat, Zone A, ThermostatSetpoint:DualSetpoint, , Dual;
Zone, Zone A;
"""
    targets = {
        "DJF": {"Zone A": 23.0}, "MAM": {"Zone A": 24.0},
        "JJA": {"Zone A": 25.0}, "SON": {"Zone A": 24.0},
    }
    result = seasonal_annual_input(source, targets, 1.5)
    assert result.count("Through:") == 10
    assert "Through: 2/28" in result and "Through: 12/31" in result
    assert physical_hash(result) == physical_hash(source)


def test_seasonal_scoring_fixes_tail_from_anchor_and_closes_annual_energy():
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    data = {
        "zone_ids": np.array(["Zone A"]),
        "temperatures": np.full((8760, 1), 24.0),
        "people": np.ones((8760, 1)),
        "occupied": np.ones(8760, dtype=bool),
        "month": clock.month.to_numpy(),
        "facility_gas_hhv_kwh_annual": np.asarray(0.0),
        "facility_electric_kwh": np.full(8760, 1.5),
        "hvac_electric_kwh": np.ones(8760),
        "heating_electric_kwh": np.full(8760, 0.25),
        "cooling_electric_kwh": np.full(8760, 0.50),
        "fans_electric_kwh": np.full(8760, 0.25),
        "gas_hhv_kwh": np.zeros(8760),
    }
    tail = fixed_tail_for_seasonal_demand(
        data, _profiles(), latitude=40.0, contrast_c=2.0, fraction=1.0,
        tail_fraction=0.5,
    )
    annual, seasons = score_seasonal_trajectory(
        data, _profiles(), latitude=40.0, contrast_c=2.0, fraction=1.0,
        fixed_tail=tail,
    )
    assert len(tail) == 1
    assert len(seasons) == 4
    assert annual["total_purchased_kwh"] == annual["facility_purchased_kwh"]
    assert all(np.isnan(item["total_purchased_kwh"]) for item in seasons)
    assert np.isclose(
        annual["conditioning_purchased_kwh"],
        sum(item["conditioning_purchased_kwh"] for item in seasons),
    )
    assert np.isclose(
        annual["facility_electric_plus_space_heating_gas_kwh"],
        sum(
            item["facility_electric_plus_space_heating_gas_kwh"]
            for item in seasons
        ),
    )


def test_completed_seasonal_scoring_refuses_manifest_hash_mismatch(
    tmp_path: Path,
):
    hours = 8760
    compact = tmp_path / "trajectory.npz"
    np.savez_compressed(
        compact,
        zone_ids=np.asarray(["Zone A"]),
        temperatures=np.full((hours, 1), 24.0),
        people=np.ones((hours, 1)),
        occupied=np.ones(hours, dtype=bool),
        month=np.resize(np.arange(1, 13), hours),
        facility_gas_hhv_kwh_annual=np.asarray(0.0),
        facility_electric_kwh=np.full(hours, 1.5),
        hvac_electric_kwh=np.ones(hours),
        heating_electric_kwh=np.full(hours, 0.25),
        cooling_electric_kwh=np.full(hours, 0.50),
        fans_electric_kwh=np.full(hours, 0.25),
        gas_hhv_kwh=np.zeros(hours),
    )
    runs = tmp_path / "runs.csv"
    pd.DataFrame([
        {
            "run_id": control,
            "climate": "Test",
            "composition_id": 1,
            "cohort_label": "Test cohort",
            "control": control,
            "seasonal_contrast_fraction": 1.0,
            "completed": True,
            "relative_engineering_pass": True,
            "hourly_path": compact,
            "hourly_sha256": "not-the-current-hash",
        }
        for control in ("uniform_average", "personal_center", "personal_range")
    ]).to_csv(runs, index=False)
    profiles = tmp_path / "profiles.csv"
    pd.DataFrame([{
        "context": "Test", "composition_id": 1, "profile_id": "p1",
        "zone_id": "Zone A", "centre_c": 24.0,
        "controller_lower_c": 23.0, "controller_upper_c": 25.0,
        "demand_source": "ASHRAE",
    }]).to_csv(profiles, index=False)
    climates = tmp_path / "climates.json"
    climates.write_text(
        json.dumps({"climates": [{"climate": "Test", "latitude": 40.0}]}),
        encoding="utf-8",
    )
    contrasts = tmp_path / "contrasts.csv"
    pd.DataFrame([{
        "demand_source": "ASHRAE", "warm_minus_cool_c": 1.0,
    }]).to_csv(contrasts, index=False)
    references = tmp_path / "references.csv"
    pd.DataFrame([{
        "climate": "Test", "target_c": 25.0, "hourly_path": compact,
        "hourly_sha256": sha256(compact),
    }]).to_csv(references, index=False)

    with pytest.raises((ValueError, RuntimeError), match="(?i)hash"):
        score_completed_plan(
            runs, profiles, climates, contrasts, references, tmp_path / "results"
        )
