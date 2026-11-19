from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.multi_event_dr_extension import (
    action_variants,
    apply_event_solver_stability_exception,
    attach_planned_event_metadata,
    baseline_relative_qa,
    baseline_artifact_hashes,
    compact_timestamps,
    directional_protected_zones,
    event_specific_action_input,
    prepare_reference_manifest,
    read_event_solver_stability_evidence,
    require_complete_event_execution,
    score_event,
    score_event_manifest,
    select_protocols,
)
from analysis.global_extension.global_matrix import sha256
from analysis.energy_service.generate_model_inputs import physical_hash


def test_event_execution_gate_rejects_failed_or_missing_runs():
    qualified = pd.DataFrame([{
        "run_id": "a", "completed": True, "engineering_pass": False,
        "final_engineering_qualified": True,
    }])
    require_complete_event_execution(qualified, 1)

    failed = pd.DataFrame([
        {"run_id": "a", "completed": True,
         "final_engineering_qualified": True},
        {"run_id": "b", "completed": True,
         "final_engineering_qualified": False},
    ])
    with pytest.raises(RuntimeError, match="completeness/engineering gate"):
        require_complete_event_execution(failed, 2)

    incomplete = failed.iloc[:1].copy()
    with pytest.raises(RuntimeError, match="expected=2, observed=1"):
        require_complete_event_execution(incomplete, 2)


def test_completed_event_manifest_restores_scoring_metadata():
    executed = pd.DataFrame([{"run_id": "r1", "completed": True}])
    planned = pd.DataFrame([{
        "run_id": "r1", "baseline_hourly_path": "baseline.npz",
        "baseline_hourly_sha256": "hourly-hash",
        "baseline_qa_path": "baseline.json", "baseline_qa_sha256": "qa-hash",
        "action_family": "uniform_shift", "permission": "building",
        "cohort_label": "China A",
    }])
    result = attach_planned_event_metadata(executed, planned)
    assert result.loc[0, "cohort_label"] == "China A"
    assert result.loc[0, "baseline_hourly_path"] == "baseline.npz"


def test_baseline_artifacts_must_match_event_plan(tmp_path: Path):
    hourly = tmp_path / "baseline.npz"
    qa = tmp_path / "baseline.json"
    hourly.write_bytes(b"hourly")
    qa.write_text("{}", encoding="utf-8")
    row = {
        "baseline_hourly_path": str(hourly),
        "baseline_hourly_sha256": sha256(hourly),
        "baseline_qa_path": str(qa),
        "baseline_qa_sha256": sha256(qa),
    }
    assert baseline_artifact_hashes(row) == {
        "baseline_hourly_sha256": sha256(hourly),
        "baseline_qa_sha256": sha256(qa),
    }
    hourly.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed after event planning"):
        baseline_artifact_hashes(row)


def _annual_compact(path: Path) -> None:
    clock = pd.date_range("2001-01-01", "2001-12-31 23:00", freq="h")
    hour = clock.hour.to_numpy()
    occupied = (clock.weekday.to_numpy() < 5) & (hour >= 6) & (hour < 22)
    seasonal = np.sin(2 * np.pi * (clock.dayofyear.to_numpy() - 1) / 365)
    load = 20 + 10 * seasonal + np.where((hour >= 8) & (hour < 10), 20, 0)
    np.savez_compressed(
        path,
        month=clock.month.to_numpy(), day=clock.day.to_numpy(), hour=hour + 1,
        dst=np.zeros(len(clock), dtype=int), occupied=occupied,
        facility_gas_hhv_kwh_annual=np.asarray(0.0),
        facility_electric_kwh=load + 5.0,
        hvac_electric_kwh=load,
        cooling_electric_kwh=np.where(seasonal >= 0, load, 0),
        heating_electric_kwh=np.where(seasonal < 0, load, 0),
        fans_electric_kwh=np.full(len(clock), 2.0),
        gas_hhv_kwh=np.zeros(len(clock)),
    )


def _profiles(zones: list[str]) -> pd.DataFrame:
    rows = []
    for index, zone in enumerate(zones):
        rows.append({
            "profile_id": f"p{index}", "zone_id": zone,
            "centre_c": 22.0 + index,
            "controller_lower_c": 20.0 + index,
            "controller_upper_c": 24.0 + index,
        })
    return pd.DataFrame(rows)


def test_protocol_selects_eight_events_from_reference_manifest(tmp_path: Path):
    compact = tmp_path / "reference.npz"
    _annual_compact(compact)
    manifest = pd.DataFrame([{
        "climate": "Beijing", "latitude": 39.9,
        "hourly_path": str(compact), "hourly_sha256": sha256(compact),
        "run_id": "reference",
    }])
    result = select_protocols(manifest)
    assert len(result) == 8
    assert result.groupby("season_block").size().eq(2).all()
    assert set(result.load_level) == {"typical", "high"}
    assert result.selection_load_scope.eq("facility electricity").all()
    with np.load(compact) as saved:
        assert len(compact_timestamps(saved)) == 8760
        clock = compact_timestamps(saved)
        for row in result.itertuples(index=False):
            start = int(clock.get_loc(pd.Timestamp(row.start_timestamp)))
            expected = float(saved["facility_electric_kwh"][start:start + 2].mean())
            assert np.isclose(row.reference_mean_kw, expected)


def test_protocol_ranks_using_native_meter_precision(tmp_path, monkeypatch):
    from analysis.global_extension import multi_event_dr_extension as events

    compact = tmp_path / "reference.npz"
    _annual_compact(compact)
    with np.load(compact) as saved:
        meter = np.nextafter(saved["facility_electric_kwh"], np.inf)
        clock = compact_timestamps(saved)
    manifest = pd.DataFrame([{
        "climate": "Beijing", "latitude": 39.9, "run_id": "reference",
        "hourly_path": str(compact), "hourly_sha256": sha256(compact),
    }])
    def read_meter(path):
        assert path == tmp_path / "reference/eplusout.sql"
        return meter

    monkeypatch.setattr(events, "extract_facility_electric_kwh", read_meter)
    result = select_protocols(manifest, tmp_path)
    for row in result.itertuples():
        start = clock.get_loc(row.start_timestamp)
        assert row.reference_mean_kw == meter[start:start + 2].mean()
    meter += 1
    with pytest.raises(ValueError, match="meters disagree"):
        select_protocols(manifest, tmp_path)


def test_protocol_uses_gas_when_classifying_heating_events(tmp_path: Path):
    compact = tmp_path / "gas_heating_reference.npz"
    _annual_compact(compact)
    with np.load(compact, allow_pickle=False) as saved:
        values = {name: np.asarray(saved[name]) for name in saved.files}
    values["cooling_electric_kwh"] = np.zeros(8760)
    values["heating_electric_kwh"] = np.zeros(8760)
    values["gas_hhv_kwh"] = np.full(8760, 20.0)
    np.savez_compressed(compact, **values)
    manifest = pd.DataFrame([{
        "climate": "Montreal", "latitude": 45.5,
        "hourly_path": str(compact), "hourly_sha256": sha256(compact),
        "run_id": "reference",
    }])
    result = select_protocols(manifest)
    assert result.sign.eq(-1.0).all()
    assert result.dominant_mode.eq("heating").all()


def test_protocol_uses_dst_adjusted_schedule_clock(tmp_path: Path):
    compact = tmp_path / "dst_reference.npz"
    _annual_compact(compact)
    with np.load(compact, allow_pickle=False) as saved:
        values = {name: np.asarray(saved[name]) for name in saved.files}
    values["dst"] = np.ones(8760, dtype=int)
    np.savez_compressed(compact, **values)
    result = select_protocols(pd.DataFrame([{
        "climate": "Test", "latitude": 40.0,
        "hourly_path": str(compact), "hourly_sha256": sha256(compact),
        "run_id": "reference",
    }]))
    expected = (result.start_timestamp.dt.hour * 60 + 60) % 1440
    assert result.start_minute.tolist() == expected.tolist()


def test_reference_manifest_selects_qualified_25c_and_coordinates(tmp_path: Path):
    probes = pd.DataFrame([
        {"climate": "A", "target_c": 24.0, "hourly_path": "a24.npz",
         "hourly_sha256": "hash-24", "completed": True,
         "relative_engineering_pass": True},
        {"climate": "A", "target_c": 25.0, "hourly_path": "a25.npz",
         "hourly_sha256": "hash-25", "completed": True,
         "relative_engineering_pass": True},
    ])
    climates = tmp_path / "climates.json"
    climates.write_text(json.dumps({"climates": [
        {"climate": "A", "latitude": 1.0, "longitude": 2.0}
    ]}), encoding="utf-8")
    result = prepare_reference_manifest(probes, climates)
    assert result.hourly_path.tolist() == ["a25.npz"]
    assert result.latitude.tolist() == [1.0]


def test_directional_zone_protection_and_action_library():
    zones = ["a", "b", "c", "d"]
    profiles = _profiles(zones)
    warm, warm_audit = directional_protected_zones(
        profiles, zones, 1.0, protected_fraction=0.25
    )
    cool, _ = directional_protected_zones(
        profiles, zones, -1.0, protected_fraction=0.25
    )
    assert warm == ["a"]
    assert cool == ["d"]
    assert warm_audit.protected.sum() == 1

    protocol = pd.DataFrame([
        {"event_id": "warm", "month": 7, "day": 5, "sign": 1.0,
         "start_minute": 840, "duration_hours": 2, "precondition_hours": 2},
        {"event_id": "cool", "month": 1, "day": 5, "sign": -1.0,
         "start_minute": 480, "duration_hours": 2, "precondition_hours": 2},
    ])
    variants, audit = action_variants(
        protocol, profiles, zones, protected_fraction=0.25
    )
    assert {row["action_id"] for row in variants} == {
        "uniform_0.5", "uniform_1.0", "precondition_0.5_uniform_1.0",
        "protected_1.0", "precondition_0.5_protected_1.0",
    }
    protected = next(row for row in variants if row["action_id"] == "protected_1.0")
    by_event = {row["event_id"]: row for row in protected["events"]}
    assert by_event["warm"]["event_magnitude_c"]["a"] == 0
    assert by_event["cool"]["event_magnitude_c"]["d"] == 0
    assert not audit.empty


def test_event_input_changes_only_thermostat_schedules():
    source = """
Version,22.1;
Building,Office;
Zone,Zone A;
Schedule:Compact,Heat A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,23.5,For: Saturday,Until: 24:00,23.5;
Schedule:Compact,Cool A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,26.5,For: Saturday,Until: 24:00,26.5;
ThermostatSetpoint:DualSetpoint,Zone A dual,Heat A,Cool A;
ZoneControl:Thermostat,Zone A thermostat,Zone A,Control Type,4,Zone A dual;
"""
    event = {
        "month": 7, "day": 5, "sign": 1.0, "start_minute": 840,
        "duration_hours": 2, "precondition_hours": 2,
        "precondition_magnitude_c": {"Zone A": 0.5},
        "event_magnitude_c": {"Zone A": 1.0},
    }
    result = event_specific_action_input(source, [event])
    assert physical_hash(result) == physical_hash(source)
    assert "Until: 12:00" in result
    assert "Until: 14:00" in result
    assert "Until: 16:00" in result
    heat = result.split("Schedule:Compact,Heat A", 1)[1].split(";", 1)[0]
    cool = result.split("Schedule:Compact,Cool A", 1)[1].split(";", 1)[0]
    assert ",24.0" not in heat and ",22.5" not in heat
    assert ",26.0" in cool and ",27.5" in cool


def test_heating_event_changes_only_heating_boundary():
    source = """
Version,22.1;
Building,Office;
Zone,Zone A;
Schedule:Compact,Heat A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,23.5,For: Saturday,Until: 24:00,23.5;
Schedule:Compact,Cool A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,26.5,For: Saturday,Until: 24:00,26.5;
ThermostatSetpoint:DualSetpoint,Zone A dual,Heat A,Cool A;
ZoneControl:Thermostat,Zone A thermostat,Zone A,Control Type,4,Zone A dual;
"""
    event = {
        "month": 1, "day": 5, "sign": -1.0, "start_minute": 480,
        "duration_hours": 2, "precondition_hours": 2,
        "precondition_magnitude_c": {"Zone A": 0.5},
        "event_magnitude_c": {"Zone A": 1.0},
    }
    result = event_specific_action_input(source, [event])
    heat = result.split("Schedule:Compact,Heat A", 1)[1].split(";", 1)[0]
    cool = result.split("Schedule:Compact,Cool A", 1)[1].split(";", 1)[0]
    assert ",24.0" in heat and ",22.5" in heat
    assert ",26.0" not in cool and ",27.5" not in cool


def test_preconditioning_cannot_collapse_deadband():
    source = """
Version,22.1;
Building,Office;
Zone,Zone A;
Schedule:Compact,Heat A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,24.0,For: Saturday,Until: 24:00,24.0;
Schedule:Compact,Cool A,Temperature,Through: 12/31,For: Weekdays,Until: 24:00,25.2,For: Saturday,Until: 24:00,25.2;
ThermostatSetpoint:DualSetpoint,Zone A dual,Heat A,Cool A;
ZoneControl:Thermostat,Zone A thermostat,Zone A,Control Type,4,Zone A dual;
"""
    event = {
        "month": 7, "day": 5, "sign": 1.0, "start_minute": 840,
        "duration_hours": 2, "precondition_hours": 2,
        "precondition_magnitude_c": {"Zone A": 0.5},
        "event_magnitude_c": {"Zone A": 1.0},
    }
    with pytest.raises(ValueError, match="deadband"):
        event_specific_action_input(source, [event])


def test_baseline_relative_qa_rejects_new_or_more_warnings():
    baseline = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "direct_numerical_warnings": [
            "** Warning ** SimHVAC: Maximum iterations, at 01/01",
            "** Warning ** WetBulb not converged after 101 iterations(PsyTwbFnTdbWPb)",
        ],
    }
    same = baseline | {
        "direct_numerical_warnings": [
            "** Warning ** SimHVAC: Maximum iterations, at 01/02",
            "** Warning ** WetBulb not converged after 101 iterations(PsyTwbFnTdbWPb)",
        ],
        "unexplained_recurring_groups": [],
    }
    assert baseline_relative_qa(baseline, same)["baseline_relative_engineering_pass"]
    more = same | {"direct_numerical_warnings": same["direct_numerical_warnings"] + [
        "** Warning ** SimHVAC: Maximum iterations, at 01/03"
    ]}
    assert not baseline_relative_qa(baseline, more)["baseline_relative_engineering_pass"]

    profiled = baseline_relative_qa(
        baseline,
        same,
        baseline_profile={
            "hvac_iterations": 0, "wet_bulb": 2,
            "psychrometric_temperature": 7, "coil_operating_range": 0,
        },
        action_profile={
            "hvac_iterations": 0, "wet_bulb": 5,
            "psychrometric_temperature": 9, "coil_operating_range": 0,
        },
    )
    assert profiled["baseline_relative_engineering_pass"]
    assert set(profiled["inherited_environmental_warning_count_drift"]) == {
        "wet_bulb", "psychrometric_temperature",
    }
    failed = baseline_relative_qa(
        baseline,
        same,
        baseline_profile={
            "hvac_iterations": 0, "wet_bulb": 2,
            "psychrometric_temperature": 7, "coil_operating_range": 0,
        },
        action_profile={
            "hvac_iterations": 1, "wet_bulb": 2,
            "psychrometric_temperature": 7, "coil_operating_range": 0,
        },
    )
    assert not failed["baseline_relative_engineering_pass"]


def _event_stability_row() -> dict:
    profile = json.dumps({
        "hvac_iterations": 5,
        "wet_bulb": 0,
        "psychrometric_temperature": 0,
        "coil_operating_range": 0,
    }, sort_keys=True)
    return {
        "run_id": "event-test", "climate": "Beijing",
        "composition_id": 1, "control": "personal_center",
        "action_id": "uniform_1.0",
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.002,
        "hvac_electricity_relative_difference": 0.0002,
        "gas_hhv_relative_difference": 0.0001,
        "total_purchased_energy_relative_difference": 0.0002,
        "facility_electricity_compared": True,
        "facility_electricity_relative_difference": 0.0001,
        "trajectory_stable_within_declared_tolerance": True,
        "declared_temperature_tolerance_c": 0.01,
        "declared_energy_relative_tolerance": 0.001,
        "higher_iteration_completed": True,
        "higher_iteration_severe_count": 0,
        "higher_iteration_fatal_occurrences": 0,
        "higher_iteration_relative_engineering_pass": False,
        "warning_profile_nonworsening": True,
        "first_action_warning_profile": profile,
        "second_action_warning_profile": profile,
        "first_input_sha256": "a" * 64,
        "first_hourly_sha256": "b" * 64,
    }


def test_event_solver_stability_evidence_is_exact_and_hash_bound(tmp_path: Path):
    path = tmp_path / "event_stability.csv"
    pd.DataFrame([_event_stability_row()]).to_csv(path, index=False)
    evidence, evidence_hash = read_event_solver_stability_evidence(path)
    assert evidence_hash == sha256(path)
    relative = {
        "baseline_relative_engineering_pass": False,
        "new_numerical_warning_categories": [],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 4, "treatment": 5}
        },
        "baseline_warning_profile": {"hvac_iterations": 4},
        "action_warning_profile": json.loads(
            _event_stability_row()["first_action_warning_profile"]
        ),
    }
    qa = {
        "completed": True, "reported_severe_count": 0,
        "fatal_occurrences": 0, "unexplained_recurring_groups": [],
    }
    passed, audit = apply_event_solver_stability_exception(
        relative, qa, evidence["event-test"],
        current_maximum_hvac_iterations=40,
        active_input_sha256="a" * 64,
        active_hourly_sha256="b" * 64,
    )
    assert passed
    assert audit["event_solver_stability_exception_applied"]
    assert audit["event_solver_stability_numerical_stability_pass"]
    assert not audit[
        "event_solver_stability_higher_iteration_baseline_warning_pass"
    ]
    with pytest.raises(ValueError, match="active trajectory"):
        apply_event_solver_stability_exception(
            relative, qa, evidence["event-test"],
            current_maximum_hvac_iterations=40,
            active_input_sha256="a" * 64,
            active_hourly_sha256="c" * 64,
        )


def test_event_solver_stability_reader_checks_facility_electricity(tmp_path: Path):
    row = _event_stability_row()
    row["facility_electricity_relative_difference"] = 0.002
    path = tmp_path / "unstable_facility.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="fails its declared limits"):
        read_event_solver_stability_evidence(path)


def test_event_solver_stability_accepts_count_drift_without_new_category(
    tmp_path: Path,
):
    row = _event_stability_row()
    second = json.loads(row["second_action_warning_profile"])
    second["hvac_iterations"] += 3
    row["second_action_warning_profile"] = json.dumps(second, sort_keys=True)
    row["warning_profile_nonworsening"] = False
    row["higher_iteration_relative_engineering_pass"] = False
    path = tmp_path / "event_count_drift.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    evidence, _ = read_event_solver_stability_evidence(path)
    selected = evidence["event-test"]

    assert selected["warning_categories_stable"]
    assert selected["solver_stability_pass"]
    assert json.loads(selected["higher_iteration_new_warning_categories"]) == []
    assert json.loads(selected["warning_count_drift"])["hvac_iterations"]["delta"] == 3


def test_event_solver_stability_rejects_new_warning_category(tmp_path: Path):
    row = _event_stability_row()
    second = json.loads(row["second_action_warning_profile"])
    second["wet_bulb"] = 1
    row["second_action_warning_profile"] = json.dumps(second, sort_keys=True)
    path = tmp_path / "event_new_category.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="category-and-trajectory"):
        read_event_solver_stability_evidence(path)


def _short_compact(temperature: np.ndarray, electricity: np.ndarray) -> dict[str, np.ndarray]:
    clock = pd.date_range("2001-01-01", periods=len(temperature), freq="h")
    return {
        "month": clock.month.to_numpy(), "day": clock.day.to_numpy(),
        "hour": clock.hour.to_numpy() + 1, "dst": np.zeros(len(clock), dtype=int),
        "zone_ids": np.array(["zone"]),
        "temperatures": temperature[:, None],
        "people": np.ones((len(clock), 1)), "occupied": np.ones(len(clock), dtype=bool),
        "facility_gas_hhv_kwh_annual": np.asarray(0.0),
        "facility_electric_kwh": electricity + 4.0,
        "hvac_electric_kwh": electricity,
        "heating_electric_kwh": electricity * 0.2,
        "cooling_electric_kwh": electricity * 0.5,
        "fans_electric_kwh": electricity * 0.3,
        "gas_hhv_kwh": np.zeros(len(clock)),
    }


def test_event_scoring_uses_event_only_service_allowance():
    hours = 72
    baseline = _short_compact(np.full(hours, 25.0), np.full(hours, 10.0))
    action_temperature = np.full(hours, 25.0)
    action_temperature[8:10] = 25.5
    action_temperature[10:58] = 25.0
    action_electricity = np.full(hours, 10.0)
    action_electricity[6:8] = 11.0
    action_electricity[8:10] = 7.0
    action_electricity[10:58] = 10.05
    action = _short_compact(action_temperature, action_electricity)
    reference = _short_compact(np.full(hours, 25.0), np.full(hours, 10.0))
    profiles = pd.DataFrame([{
        "profile_id": "p", "zone_id": "zone", "centre_c": 24.0,
        "controller_lower_c": 23.0, "controller_upper_c": 26.0,
    }])
    event = {
        "event_id": "e", "start_timestamp": pd.Timestamp("2001-01-01 08:00"),
        "duration_hours": 2, "precondition_hours": 2, "recovery_hours": 48,
    }
    result = score_event(
        baseline, action, reference, profiles, event, limit_factors=(1.0, 1.5)
    )
    assert len(result) == 2
    assert np.isclose(result[0]["event_electricity_reduction_kwh"], 6.0)
    assert np.isclose(result[0]["event_facility_electricity_reduction_kwh"], 6.0)
    assert np.isclose(result[0]["event_hvac_electricity_reduction_kwh"], 6.0)
    assert np.isclose(result[0]["event_cooling_electricity_reduction_kwh"], 3.0)
    assert np.isclose(result[0]["event_heating_electricity_reduction_kwh"], 1.2)
    assert np.isclose(result[0]["event_fans_electricity_reduction_kwh"], 1.8)
    assert np.isclose(result[0]["recovery_electricity_change_kwh"], 2.4)
    assert np.isclose(result[0]["preconditioning_electricity_change_kwh"], 2.0)
    assert np.isclose(
        result[0]["precondition_event_recovery_net_reduction_kwh"], 1.6
    )
    assert result[0]["qualification_checks"] == 12
    assert result[0]["strict_qualification_checks"] == 10
    assert result[0]["event_allowance_checks"] == 2
    assert result[0]["event_service_limit_factor"] == 1.0
    assert result[0]["positive_full_window_response"]
    assert not result[0]["service_qualified"]
    assert result[1]["service_qualified"]

    action_temperature[10:58] = 25.1
    recovery_failed = score_event(
        baseline,
        _short_compact(action_temperature, action_electricity),
        reference,
        profiles,
        event,
        limit_factors=(1.5,),
    )[0]
    assert recovery_failed["action_event_service_qualified"]
    assert not recovery_failed["action_non_event_service_qualified"]
    assert not recovery_failed["service_qualified"]


def test_event_scoring_skips_service_check_for_unoccupied_precondition_window():
    hours = 72
    baseline = _short_compact(np.full(hours, 25.0), np.full(hours, 10.0))
    action_temperature = np.full(hours, 25.0)
    action_temperature[8:10] = 25.5
    action = _short_compact(action_temperature, np.full(hours, 9.0))
    reference = _short_compact(np.full(hours, 25.0), np.full(hours, 10.0))
    for compact in (baseline, action, reference):
        compact["people"][6:8] = 0.0
        compact["occupied"][6:8] = False
    profiles = pd.DataFrame([{
        "profile_id": "p", "zone_id": "zone", "centre_c": 24.0,
        "controller_lower_c": 23.0, "controller_upper_c": 26.0,
    }])
    event = {
        "event_id": "e", "start_timestamp": pd.Timestamp("2001-01-01 08:00"),
        "duration_hours": 2, "precondition_hours": 2, "recovery_hours": 48,
    }
    result = score_event(
        baseline, action, reference, profiles, event, limit_factors=(1.5,)
    )[0]
    assert not result["precondition_service_evaluated"]
    assert result["qualification_checks"] == 8
    assert result["strict_qualification_checks"] == 6
    assert result["service_qualified"]


def test_scoring_refuses_runs_that_fail_relative_engineering_qa():
    runs = pd.DataFrame([{
        "run_id": "bad", "climate": "Test", "composition_id": 1,
        "control": "uniform_average", "action_id": "uniform_0.5",
        "action_family": "uniform_shift", "permission": "building",
        "completed": True, "baseline_relative_engineering_pass": False,
        "final_engineering_qualified": False,
        "hourly_path": "missing.npz", "baseline_hourly_path": "missing-base.npz",
    }])
    reference = pd.DataFrame([{"climate": "Test", "hourly_path": "missing-ref.npz"}])
    with pytest.raises(RuntimeError, match="completeness/engineering gate"):
        score_event_manifest(
            runs, pd.DataFrame({"run_id": ["bad"]}),
            pd.DataFrame({"context": []}), reference,
        )


def test_scoring_refuses_action_compact_hash_mismatch(tmp_path: Path):
    action = tmp_path / "action.npz"
    baseline = tmp_path / "baseline.npz"
    reference = tmp_path / "reference.npz"
    qa = tmp_path / "baseline.json"
    action.write_bytes(b"changed action")
    baseline.write_bytes(b"baseline")
    reference.write_bytes(b"reference")
    qa.write_text("{}", encoding="utf-8")
    runs = pd.DataFrame([{
        "run_id": "r1", "climate": "Test", "composition_id": 1,
        "cohort_label": "Test cohort", "control": "uniform_average",
        "action_id": "uniform_0.5", "action_family": "uniform_shift",
        "permission": "building", "completed": True,
        "baseline_relative_engineering_pass": True,
        "final_engineering_qualified": True,
        "hourly_path": str(action), "hourly_sha256": "not-the-current-hash",
        "baseline_hourly_path": str(baseline),
        "baseline_hourly_sha256": sha256(baseline),
        "baseline_qa_path": str(qa), "baseline_qa_sha256": sha256(qa),
    }])
    with pytest.raises((ValueError, RuntimeError), match="SHA-256|(?i:hash)"):
        score_event_manifest(
            runs,
            pd.DataFrame({"run_id": ["r1"]}),
            pd.DataFrame({"context": []}),
            pd.DataFrame({
                "climate": ["Test"], "hourly_path": [reference],
                "hourly_sha256": [sha256(reference)],
            }),
        )
