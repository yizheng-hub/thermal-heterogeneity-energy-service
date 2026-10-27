from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.audit_event_solver_stability import (
    compare_event_trajectories,
    select_event_hvac_only_failures,
)


def _compact(path: Path, *, temperature: float, facility_scale: float = 1.0) -> None:
    hours = 24
    hvac = np.full(hours, 10.0)
    np.savez_compressed(
        path,
        zone_ids=np.asarray(["zone"]),
        temperatures=np.full((hours, 1), temperature),
        people=np.ones((hours, 1)),
        occupied=np.ones(hours, dtype=bool),
        hvac_electric_kwh=hvac,
        gas_hhv_kwh=np.full(hours, 2.0),
        facility_electric_kwh=(hvac + 5.0) * facility_scale,
    )


def test_event_comparison_includes_facility_electricity(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _compact(first, temperature=25.0)
    _compact(second, temperature=25.005)
    result = compare_event_trajectories(first, second)
    assert result["trajectory_stable_within_declared_tolerance"]
    assert result["facility_electricity_compared"]

    _compact(second, temperature=25.005, facility_scale=1.01)
    result = compare_event_trajectories(first, second)
    assert not result["trajectory_stable_within_declared_tolerance"]
    assert result["facility_electricity_relative_difference"] > 0.001


def test_event_selection_accepts_only_clean_hvac_iteration_failures(tmp_path: Path):
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    plan = pd.DataFrame([{
        "run_id": "event-a", "climate": "Beijing", "composition_id": 1,
        "control": "personal_center", "action_id": "uniform_1.0",
        "input_idf_path": "input.idf", "weather_path": "weather.epw",
        "zones_json": json.dumps(["zone"]),
    }])
    summary = {
        "run_id": "event-a", "completed": True,
        "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
        "baseline_relative_engineering_pass": False,
        "new_numerical_warning_categories": [],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 4, "treatment": 5}
        },
    }
    (summaries / "event-a.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    selected = select_event_hvac_only_failures(plan, tmp_path)
    assert len(selected) == 1

    summary["critical_warning_count_increases"] = {
        "coil_operating_range": {"baseline": 0, "treatment": 1}
    }
    (summaries / "event-a.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not attributable solely"):
        select_event_hvac_only_failures(plan, tmp_path)
