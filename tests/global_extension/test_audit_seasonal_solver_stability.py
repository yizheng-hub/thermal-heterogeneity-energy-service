from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.audit_seasonal_solver_stability import (
    apply_seasonal_solver_stability_evidence,
    compare_seasonal_trajectories,
    read_seasonal_solver_stability_evidence,
    select_hvac_only_failures,
)
from analysis.global_extension.global_matrix import sha256


def _compact(path: Path, *, temperature_shift: float = 0.0) -> None:
    hours = 4
    np.savez_compressed(
        path,
        zone_ids=np.asarray(["Zone A"]),
        temperatures=np.full((hours, 1), 24.0 + temperature_shift),
        people=np.ones((hours, 1)),
        occupied=np.asarray([True, True, False, False]),
        month=np.ones(hours, dtype=int),
        day=np.ones(hours, dtype=int),
        hour=np.arange(1, hours + 1),
        dst=np.zeros(hours, dtype=int),
        facility_electric_kwh=np.full(hours, 2.0),
        hvac_electric_kwh=np.full(hours, 1.0),
        gas_hhv_kwh=np.full(hours, 0.5),
        facility_gas_hhv_kwh_annual=np.asarray(2.5),
    )


def _profile(hvac: int) -> dict[str, int]:
    return {
        "hvac_iterations": hvac,
        "wet_bulb": 0,
        "psychrometric_temperature": 0,
        "coil_operating_range": 0,
    }


def test_compare_seasonal_trajectories_checks_facility_and_hvac_streams(
    tmp_path: Path,
):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _compact(first)
    _compact(second, temperature_shift=0.005)
    result = compare_seasonal_trajectories(first, second)
    assert result["trajectory_stable_within_declared_tolerance"]
    assert result["operative_temperature_max_abs_difference_c"] == pytest.approx(
        0.005
    )
    assert result["facility_electricity_relative_difference"] == 0
    assert result["facility_electricity_trajectory_relative_l1"] == 0
    assert result["facility_gas_annual_relative_difference"] == 0


def test_select_hvac_only_failures_requires_exact_clean_primary_evidence(
    tmp_path: Path,
):
    primary = tmp_path / "primary"
    run_id = "seasonal_test"
    run_folder = primary / "runs" / run_id
    run_folder.mkdir(parents=True)
    compact = primary / "hourly" / f"{run_id}.npz"
    compact.parent.mkdir()
    _compact(compact)
    input_path = tmp_path / "input.idf"
    input_path.write_text("ConvergenceLimits,1,40;\n", encoding="utf-8")
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    error = run_folder / "eplusout.err"
    error.write_text(
        "** Warning ** SimHVAC: Maximum iterations exceeded\n",
        encoding="utf-8",
    )
    qa = {
        "completed": True,
        "reported_severe_count": 0,
        "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
        "relative_engineering_pass": False,
        "new_warning_categories": ["hvac_iterations"],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 0, "treatment": 1}
        },
        "baseline_warning_profile": _profile(0),
        "treatment_warning_profile": _profile(1),
        "input_idf_sha256": sha256(input_path),
        "weather_sha256": sha256(weather),
        "hourly_sha256": sha256(compact),
    }
    (run_folder / "RUN_QA.json").write_text(
        json.dumps(qa), encoding="utf-8"
    )
    runs = pd.DataFrame([{
        "run_id": run_id,
        "climate": "Test",
        "composition_id": 1,
        "control": "personal_center",
        "seasonal_contrast_fraction": 1.0,
        "targets_json": json.dumps({"DJF": {"Zone A": 24.0}}),
        "input_idf_path": input_path,
        "weather_path": weather,
    }])
    selected = select_hvac_only_failures(runs, primary)
    assert selected.run_id.tolist() == [run_id]
    qa["reported_severe_count"] = 1
    (run_folder / "RUN_QA.json").write_text(
        json.dumps(qa), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not attributable solely"):
        select_hvac_only_failures(runs, primary)


def _evidence_row(tmp_path: Path) -> dict:
    first_profile = json.dumps(_profile(1), sort_keys=True)
    second_profile = json.dumps(_profile(0), sort_keys=True)
    return {
        "run_id": "seasonal_test",
        "climate": "Test",
        "composition_id": 1,
        "control": "personal_center",
        "seasonal_contrast_fraction": 1.0,
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.005,
        "facility_electricity_relative_difference": 0.0001,
        "hvac_electricity_relative_difference": 0.0001,
        "space_heating_gas_relative_difference": 0.0001,
        "facility_electricity_trajectory_relative_l1": 0.0001,
        "hvac_electricity_trajectory_relative_l1": 0.0001,
        "space_heating_gas_trajectory_relative_l1": 0.0001,
        "facility_gas_annual_relative_difference": 0.0001,
        "facility_electric_plus_space_heating_gas_relative_difference": 0.0001,
        "trajectory_stable_within_declared_tolerance": True,
        "declared_temperature_tolerance_c": 0.01,
        "declared_energy_relative_tolerance": 0.001,
        "higher_iteration_completed": True,
        "higher_iteration_severe_count": 0,
        "higher_iteration_fatal_occurrences": 0,
        "higher_iteration_relative_engineering_pass": True,
        "warning_profile_nonworsening": True,
        "first_treatment_warning_profile": first_profile,
        "second_treatment_warning_profile": second_profile,
        "first_input_sha256": "a" * 64,
        "first_hourly_sha256": "b" * 64,
        "first_error_sha256": "c" * 64,
        "second_input_sha256": "d" * 64,
        "second_hourly_sha256": "e" * 64,
        "second_qa_path": str(tmp_path / "qa.json"),
        "second_hourly_path": str(tmp_path / "second.npz"),
    }


def test_read_and_apply_seasonal_stability_evidence_is_hash_bound(
    tmp_path: Path,
):
    path = tmp_path / "stability.csv"
    pd.DataFrame([_evidence_row(tmp_path)]).to_csv(path, index=False)
    evidence, evidence_hash = read_seasonal_solver_stability_evidence(path)
    assert evidence_hash == sha256(path)
    relative = {
        "engineering_pass_relative_to_prototype_baseline": False,
        "new_warning_categories": ["hvac_iterations"],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 0, "treatment": 1}
        },
        "treatment_warning_profile": _profile(1),
    }
    qa = {
        "completed": True,
        "reported_severe_count": 0,
        "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    passed, audit = apply_seasonal_solver_stability_evidence(
        relative,
        qa,
        evidence["seasonal_test"],
        current_maximum_hvac_iterations=40,
        active_input_sha256="a" * 64,
        active_hourly_sha256="b" * 64,
        active_error_sha256="c" * 64,
    )
    assert passed
    assert audit["seasonal_solver_stability_exception_applied"]
    with pytest.raises(ValueError, match="active primary run"):
        apply_seasonal_solver_stability_evidence(
            relative,
            qa,
            evidence["seasonal_test"],
            current_maximum_hvac_iterations=40,
            active_input_sha256="f" * 64,
            active_hourly_sha256="b" * 64,
            active_error_sha256="c" * 64,
        )
