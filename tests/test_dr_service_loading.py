import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))


def test_selected_action_rescoring_accepts_valid_dr_schema_without_annual_gas_total(tmp_path, monkeypatch):
    import dr_selected_uncertainty as module
    folder = tmp_path / "results/dr_capability"
    folder.mkdir(parents=True)
    matrix = tmp_path / "matrix"
    (matrix / "probe_runs").mkdir(parents=True)
    (tmp_path / "results/base/annual").mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "MATRIX", matrix)
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    data = {"month": clock.month.to_numpy(), "day": clock.day.to_numpy(), "hour": clock.hour.to_numpy() + 1,
            "dst": np.zeros(8760, int), "occupied": np.ones(8760, bool), "zone_ids": np.array(["A"]),
            "temperatures": np.full((8760, 1), 22.0), "people": np.ones((8760, 1))}
    for key in ["facility_electric_kwh", "hvac_electric_kwh", "heating_electric_kwh",
                "cooling_electric_kwh", "fans_electric_kwh", "gas_hhv_kwh"]:
        data[key] = np.zeros(8760)
    baseline, action, reference = [tmp_path / name for name in ["baseline.npz", "action.npz", "reference.npz"]]
    np.savez_compressed(baseline, **data)
    np.savez_compressed(action, **data)
    np.savez_compressed(reference, **data, facility_gas_hhv_kwh_annual=np.array([0.0]))
    profiles = pd.DataFrame({"context": ["A"] * 267, "composition_id": [2026102001] * 267,
        "cohort_label": ["Multinational A"] * 267, "profile_id": ["P" + str(i) for i in range(267)],
        "zone_id": ["A"] * 267, "centre_c": [22.0] * 267,
        "controller_lower_c": [21.0] * 267, "controller_upper_c": [23.0] * 267})
    profiles.to_csv(matrix / "GLOBAL_PROFILES.csv", index=False)
    profiles.iloc[:27].rename(columns={"context": "climate", "cohort_label": "composition"})[
        ["climate", "composition", "profile_id"]].to_csv(tmp_path / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    pd.DataFrame([{"climate": "A", "target_c": 25, "hourly_path": str(reference)}]).to_csv(
        matrix / "probe_runs/PROBE_RUN_MANIFEST.csv", index=False)
    common = {"climate": "A", "composition_id": 2026102001, "control": "building_search", "event_id": "E"}
    pd.DataFrame([{**common, "status": "selected", "selected_action": "shed", "event_power_reduction_kw": 1.0,
                   "net_full_window_electricity_reduction_kwh": 2.0}]).to_csv(folder / "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv", index=False)
    event = {"start_timestamp": "2001-01-02 14:00:00", "duration_hours": 2, "recovery_hours": 48, "precondition_hours": 2}
    pd.DataFrame([{**common, "action_id": "shed", "normal_id": "N", "run_id": "R", "event_json": json.dumps(event)}]).to_csv(
        folder / "SINGLE_EVENT_ACTION_PLAN.csv", index=False)
    pd.DataFrame([{"normal_id": "N", "hourly_path": str(baseline)}]).to_csv(folder / "NORMAL_OPERATION_REPLAY_COMPLETED.csv", index=False)
    pd.DataFrame([{"run_id": "R", "hourly_path": str(action)}]).to_csv(folder / "ACTION_FULL_COMPLETED.csv", index=False)
    monkeypatch.setattr(module, "person_keys", lambda ids: list(ids))
    monkeypatch.setattr(module, "keyed_normal", lambda key: np.zeros(module.DRAWS))
    monkeypatch.setattr(module, "shared_normal", lambda ids, archive, study_shared: np.zeros((module.DRAWS, len(ids))))
    monkeypatch.setattr(module, "empirical_temporal_shifts", lambda ids, single_record=False: np.zeros((module.DRAWS, len(ids))))
    captured = []
    monkeypatch.setattr(module, "write_results", lambda rows, windows, count: captured.append((rows, windows, count)))
    module.main()
    rows, windows, count = captured[0]
    assert count == 1 and len(rows) == 22
    assert all(row["conditional_any_service_violation_frequency"] == 0 for row in rows)
    assert {row["window"] for row in windows} == {"annual", "baseline_precondition", "action_precondition",
        "baseline_event", "action_event", "baseline_recovery", "action_recovery"}
    assert len(module.CONSUMED_INPUTS) == 10
    assert all(module.digest(path) == sha for path, sha in module.CONSUMED_INPUTS.items())
    assert module.EXECUTED_SOURCES


def test_selected_action_receipt_preserves_outputs_when_an_input_changes(tmp_path, monkeypatch):
    import dr_selected_uncertainty as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "MATRIX", tmp_path / "matrix")
    monkeypatch.setattr(module, "CONSUMED_INPUTS", {})
    monkeypatch.setattr(module, "EXECUTED_SOURCES", {})
    source = tmp_path / "input.csv"
    source.write_text("value\n1\n")
    module.tracked(source)
    module.write_results([], [], 0)
    output = tmp_path / "results/dr_capability_uncertainty_identity"
    receipt = json.loads((output / "AUDIT.json").read_text())
    assert receipt["input_sha256"] == {"input.csv": module.digest(source)}
    assert set(receipt["output_sha256"]) == {"FROZEN_ACTION_SERVICE_VIOLATIONS.csv", "SERVICE_WINDOW_VIOLATIONS.csv"}
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    source.write_text("value\n2\n")
    with pytest.raises(ValueError, match="changed during execution"):
        module.write_results([], [], 0)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_selected_action_cache_reuses_unchanged_results_and_rejects_fresh_changed_inputs(tmp_path, monkeypatch):
    import dr_selected_uncertainty as module
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "MATRIX", tmp_path / "matrix")
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    data = tmp_path / "input.csv"
    data.write_text("value\n1\n")
    monkeypatch.setattr(module, "CONSUMED_INPUTS", {data: module.digest(data)})
    monkeypatch.setattr(module, "EXECUTED_SOURCES", {source: module.digest(source)})
    module.write_results([], [], 0)
    output = tmp_path / "results/dr_capability_uncertainty_identity"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    assert module.completed_cache(output)
    changed_source = tmp_path / "source_copy.py"
    changed_source.write_text("value = 2\n")
    module.EXECUTED_SOURCES.clear()
    module.EXECUTED_SOURCES[changed_source] = module.digest(changed_source)
    with pytest.raises(ValueError, match="executable sources changed"):
        module.completed_cache(output)
    module.EXECUTED_SOURCES.clear()
    module.EXECUTED_SOURCES[source] = module.digest(source)
    data.write_text("value\n2\n")
    module.CONSUMED_INPUTS.clear()
    module.tracked(data)
    monkeypatch.setattr(module.inspect, "getfile", lambda function: str(source))
    with pytest.raises(ValueError, match="prior attempt"):
        module.main()
    with pytest.raises(FileExistsError, match="prior response sensitivity attempt"):
        module.write_results([], [], 0)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
