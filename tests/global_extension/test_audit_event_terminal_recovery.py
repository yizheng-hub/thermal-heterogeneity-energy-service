from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.audit_event_terminal_recovery import (
    audit_event_terminal_recovery,
    sha256,
)


def _compact(path: Path, *, action: bool) -> None:
    clock = pd.date_range("2001-01-01", periods=100, freq="h")
    temperatures = np.full((len(clock), 2), 25.0)
    facility = np.full(len(clock), 10.0)
    hvac = np.full(len(clock), 4.0)
    gas = np.full(len(clock), 2.0)
    if action:
        # First event: recovery ends at 14:00, so 13:00 is its final recovery hour.
        index = int(np.flatnonzero(clock == pd.Timestamp("2001-01-01 13:00"))[0])
        temperatures[index] = [25.1, 24.8]
        facility[index] = 11.0
        hvac[index] = 3.5
        gas[index] = 2.3
        # Leave a smaller residual in the first hour after the recovery horizon.
        following = int(np.flatnonzero(clock == pd.Timestamp("2001-01-01 14:00"))[0])
        temperatures[following] = [25.05, 25.0]
        facility[following] = 10.2
    np.savez_compressed(
        path,
        month=clock.month.to_numpy(int),
        day=clock.day.to_numpy(int),
        hour=(clock.hour + 1).to_numpy(int),
        zone_ids=np.asarray(["Zone A", "Zone B"]),
        temperatures=temperatures,
        facility_electric_kwh=facility,
        hvac_electric_kwh=hvac,
        gas_hhv_kwh=gas,
    )


def _manifests(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = tmp_path / "baseline.npz"
    action = tmp_path / "action.npz"
    _compact(baseline, action=False)
    _compact(action, action=True)
    runs = pd.DataFrame([{
        "run_id": "run-1",
        "climate": "Test",
        "composition_id": 1,
        "control": "personal_center",
        "action_id": "protected_1.0",
        "action_family": "service_protected_shift",
        "permission": "native_zone",
        "hourly_path": str(action),
        "hourly_sha256": sha256(action),
        "baseline_hourly_path": str(baseline),
        "baseline_hourly_sha256": sha256(baseline),
        "completed": True,
        "baseline_relative_engineering_pass": True,
        "final_engineering_qualified": True,
    }])
    events = pd.DataFrame([
        {
            "run_id": "run-1",
            "event_id": "first",
            "start_timestamp": "2001-01-01 10:00:00",
            "duration_hours": 2,
            "precondition_hours": 2,
            "recovery_hours": 2,
        },
        {
            "run_id": "run-1",
            "event_id": "later",
            "start_timestamp": "2001-01-01 16:00:00",
            "duration_hours": 2,
            "precondition_hours": 2,
            "recovery_hours": 2,
        },
    ])
    return runs, events


def test_terminal_recovery_audit_reports_residuals_and_later_event_overlap(
    tmp_path: Path,
) -> None:
    runs, events = _manifests(tmp_path)
    frame, report = audit_event_terminal_recovery(
        runs, events, tmp_path / "audit", follow_up_hours=6
    )

    final = frame[
        frame.event_id.eq("first") & frame.audit_hour_index.eq(0)
    ].iloc[0]
    assert np.isclose(final.operative_temperature_max_abs_difference_c, 0.2)
    assert np.isclose(final.operative_temperature_mean_abs_difference_c, 0.15)
    assert np.isclose(final.facility_electricity_kw_difference, 1.0)
    assert np.isclose(final.hvac_electricity_kw_difference, -0.5)
    assert np.isclose(final.gas_input_kw_hhv_difference, 0.3)
    assert bool(final.later_event_overlaps_post_recovery_audit)

    first_post = frame[
        frame.event_id.eq("first") & frame.audit_hour_index.eq(1)
    ].iloc[0]
    assert bool(first_post.overlaps_later_event_window)
    assert first_post.overlapping_later_event_ids == "later"
    assert np.isclose(first_post.operative_temperature_max_abs_difference_c, 0.05)

    later_final = frame[
        frame.event_id.eq("later") & frame.audit_hour_index.eq(0)
    ].iloc[0]
    assert not bool(later_final.later_event_overlaps_post_recovery_audit)
    assert report["threshold_applied"] is False
    assert report["full_recovery_claimed"] is False
    assert report["events"] == 2
    assert (tmp_path / "audit" / "EVENT_TERMINAL_RECOVERY_RESIDUALS.csv").is_file()
    audit_path = tmp_path / "audit" / "EVENT_TERMINAL_RECOVERY_AUDIT.json"
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert saved["groups_by_climate_action_control"][0]["climate"] == "Test"
    assert "PASS" not in audit_path.read_text(encoding="utf-8")


def test_terminal_recovery_audit_rejects_changed_action_trajectory(tmp_path: Path) -> None:
    runs, events = _manifests(tmp_path)
    runs.loc[0, "hourly_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit_event_terminal_recovery(runs, events, tmp_path / "audit")
