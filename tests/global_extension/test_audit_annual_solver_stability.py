from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

import audit_annual_solver_stability as audit  # noqa: E402


def _compact(path: Path, temperature: float = 25.0, electricity: float = 1.0) -> None:
    hours = 8
    np.savez_compressed(
        path,
        zone_ids=np.array(["Z1", "Z2"]),
        temperatures=np.full((hours, 2), temperature),
        people=np.ones((hours, 2)),
        occupied=np.array([False, True, True, False, True, True, False, True]),
        hvac_electric_kwh=np.full(hours, electricity),
        gas_hhv_kwh=np.full(hours, 0.5),
        facility_electric_kwh=np.full(hours, electricity + 1.0),
        facility_gas_hhv_kwh_annual=np.array([4.0]),
    )


def _manifest_row(tmp_path: Path, hourly: Path) -> dict:
    source = tmp_path / "annual.idf"
    source.write_text(
        "ConvergenceLimits,1,40,2,8;\nBuilding,Office;", encoding="utf-8"
    )
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    qa_path = tmp_path / "RUN_QA.json"
    qa_path.write_text(json.dumps({
        "completed": True,
        "reported_severe_count": 0,
        "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }), encoding="utf-8")
    return {
        "run_id": "annual_test",
        "climate": "Beijing",
        "composition_id": 1,
        "control": "personal_center",
        "offset_c": 0.0,
        "completed": True,
        "relative_engineering_pass": False,
        "new_warning_categories": '["hvac_iterations"]',
        "critical_warning_count_increases": (
            '{"hvac_iterations": {"baseline": 0, "treatment": 1}}'
        ),
        "baseline_warning_profile": (
            '{"hvac_iterations": 0, "wet_bulb": 0, '
            '"psychrometric_temperature": 0, "coil_operating_range": 0}'
        ),
        "treatment_warning_profile": (
            '{"hvac_iterations": 1, "wet_bulb": 0, '
            '"psychrometric_temperature": 0, "coil_operating_range": 0}'
        ),
        "input_idf_path": str(source),
        "weather_path": str(weather),
        "hourly_path": str(hourly),
        "qa_path": str(qa_path),
        "maximum_hvac_iterations": 40,
    }


def test_selection_refuses_non_hvac_annual_failure(tmp_path: Path) -> None:
    hourly = tmp_path / "first.npz"
    _compact(hourly)
    row = _manifest_row(tmp_path, hourly)
    row["new_warning_categories"] = '["coil_operating_range"]'
    row["critical_warning_count_increases"] = (
        '{"coil_operating_range": {"baseline": 0, "treatment": 1}}'
    )
    with pytest.raises(ValueError, match="not attributable solely"):
        audit.select_hvac_only_failures(pd.DataFrame([row]))


def test_comparison_uses_only_occupied_temperature_hours(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _compact(first)
    _compact(second)
    with np.load(second, allow_pickle=False) as stored:
        values = {name: np.asarray(stored[name]) for name in stored.files}
    values["temperatures"] = values["temperatures"].copy()
    values["temperatures"][0] += 5.0
    np.savez_compressed(second, **values)
    result = audit.compare_annual_trajectories(first, second)
    assert result["operative_temperature_max_abs_difference_c"] == 0.0
    assert result["trajectory_stable_within_declared_tolerance"]


def test_comparison_checks_complete_facility_energy(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _compact(first)
    _compact(second)
    with np.load(second, allow_pickle=False) as stored:
        values = {name: np.asarray(stored[name]) for name in stored.files}
    values["facility_electric_kwh"] = (
        values["facility_electric_kwh"].astype(float) * 1.01
    )
    np.savez_compressed(second, **values)

    result = audit.compare_annual_trajectories(first, second)

    assert result["facility_purchased_energy_relative_difference"] > 0.001
    assert not result["trajectory_stable_within_declared_tolerance"]


def test_audit_writes_separate_stability_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.npz"
    _compact(first)
    row = _manifest_row(tmp_path, first)

    def fake_run(job: dict, runtime: Path) -> dict:
        del runtime
        job["run_folder"].mkdir(parents=True, exist_ok=True)
        job["compact"].parent.mkdir(parents=True, exist_ok=True)
        _compact(job["compact"], temperature=25.005, electricity=1.0001)
        (job["run_folder"] / "eplusout.err").write_text(
            "** Warning ** SimHVAC: Maximum iterations exceeded\n"
            "EnergyPlus Completed Successfully-- 1 Warning; 0 Severe Errors;\n",
            encoding="utf-8",
        )
        qa = {
            "completed": True,
            "reported_severe_count": 0,
            "fatal_occurrences": 0,
            "unexplained_recurring_groups": [],
        }
        (job["run_folder"] / "RUN_QA.json").write_text(
            json.dumps(qa), encoding="utf-8"
        )
        return qa

    monkeypatch.setattr(audit, "_run_or_reuse", fake_run)
    runtime = tmp_path / "energyplus.exe"
    runtime.write_bytes(b"runtime")
    output = tmp_path / "audit"
    result = audit.audit_annual_solver_stability(
        pd.DataFrame([row]), runtime, output, workers=1
    )
    assert len(result) == 1
    assert result.trajectory_stable_within_declared_tolerance.iloc[0]
    assert result.warning_profile_nonworsening.iloc[0]
    assert result.warning_categories_stable.iloc[0]
    assert result.solver_stability_pass.iloc[0]
    assert json.loads(result.higher_iteration_new_warning_categories.iloc[0]) == []
    assert (output / "ANNUAL_SOLVER_STABILITY.csv").is_file()
    report = json.loads(
        (output / "ANNUAL_SOLVER_STABILITY_AUDIT.json").read_text(encoding="utf-8")
    )
    assert report["primary_outputs_modified"] is False
    assert report["solver_stability_passes"] == 1
    assert "warning-count drift" in report["stability_rule"]
    original = Path(row["input_idf_path"]).read_text(encoding="utf-8")
    assert ",40," in original
    rerun = next((output / "inputs").glob("*.idf")).read_text(encoding="utf-8")
    assert ",100," in rerun
