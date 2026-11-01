import sys
from pathlib import Path
import json
import pandas as pd
import pytest
import os
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import dr_capability as module


@pytest.mark.skipif(os.name != "nt", reason="Bundled response execution uses Windows process synchronization")
def test_response_command_lock_serializes_processes_sharing_one_output(tmp_path):
    script = tmp_path / "lock_worker.py"
    script.write_text("\n".join([
        "import sys,time", "from pathlib import Path",
        "sys.path.insert(0,sys.argv[1])", "import dr_capability as module",
        "module.OUT=Path(sys.argv[2])", "marker=Path(sys.argv[3])",
        "marker.with_suffix('.ready').write_text('ready')",
        "with module.response_command_lock():", "    marker.write_text('entered')",
        "    while not Path(sys.argv[4]).exists(): time.sleep(0.02)",
    ]))
    first_marker, second_marker = tmp_path / "first", tmp_path / "second"
    release = tmp_path / "release"
    commands = [sys.executable, str(script), str(Path(module.__file__).parent), str(tmp_path)]
    first = subprocess.Popen(commands + [str(first_marker), str(release)])
    second = None
    try:
        deadline = time.monotonic() + 15
        while not first_marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert first_marker.exists()
        second = subprocess.Popen(commands + [str(second_marker), str(release)])
        deadline = time.monotonic() + 15
        while not second_marker.with_suffix('.ready').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert second_marker.with_suffix('.ready').exists()
        time.sleep(0.3)
        assert not second_marker.exists()
        release.touch()
        assert first.wait(timeout=15) == 0 and second.wait(timeout=15) == 0
        assert second_marker.exists()
    finally:
        release.touch()
        for process in [first, second]:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


def test_summary_benchmark_is_nested_and_full_floor_information_is_separate():
    group = pd.DataFrame({
        "run_id": ["b", "u", "zm", "zd", "fp", "pc", "pr", "zm_duplicate"],
        "physical_plan_id": ["b", "u", "zm", "zd", "fp", "pc", "pr", "zm"],
        "permission": ["building", "building", "native_zone", "native_zone", "floor", "native_zone", "native_zone", "native_zone"],
        "control": ["building_mean", "uniform_average", "native_zone_mean", "native_zone_median", "personal_center", "personal_center", "personal_range", "native_zone_mean"],
    })
    building = module.baseline_library(group, "building_search")
    summary = module.baseline_library(group, "zone_summary")
    full = module.baseline_library(group, "zone_full")
    assert set(building.physical_plan_id) == {"b", "u"}
    assert set(summary.physical_plan_id) == {"b", "u", "zm", "zd"}
    assert set(full.physical_plan_id) == {"b", "u", "zm", "zd", "fp", "pc", "pr"}
    assert len(summary) == 4 and not summary.permission.eq("floor").any()
    with pytest.raises(ValueError, match="Unknown normal-operation library"):
        module.baseline_library(group, "unfrozen_policy")


def test_response_plan_waits_for_completed_strong_search(tmp_path, monkeypatch):
    audit = tmp_path / "results/budget_frontiers/BUDGET_FRONTIER_AUDIT.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"status": "partial"}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Complete the declared strong-baseline search"):
        module.plan()
    assert not (tmp_path / "results/dr_capability").exists()


def test_all_unavailable_actions_retain_status_and_empty_power_fields(tmp_path, monkeypatch):
    out = tmp_path / "results/dr_capability"
    matrix = tmp_path / "matrix"
    out.mkdir(parents=True)
    (matrix / "probe_runs").mkdir(parents=True)
    (tmp_path / "response").mkdir()
    (tmp_path / "results/base/annual").mkdir(parents=True)
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "MATRIX", matrix)
    monkeypatch.setattr(module, "RAW_RUN", tmp_path)
    monkeypatch.setattr(module, "enable", lambda dr: None)
    monkeypatch.setattr(module.dr, "fixed_tail_profile_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(module, "record_status", lambda *args: None)
    rows = [{"climate": "Context", "composition_id": 1, "control": library,
             "event_id": "event", "action_id": str(action), "run_id": library + str(action)}
            for library in module.NORMAL_LIBRARIES for action in range(5)]
    pd.DataFrame(rows).to_csv(out / "SINGLE_EVENT_ACTION_PLAN.csv", index=False)
    pd.DataFrame([{"run_id": row["run_id"], "completed": False,
                   "final_engineering_qualified": False} for row in rows]).to_csv(
        out / "ACTION_FULL_COMPLETED.csv", index=False)
    pd.DataFrame({"normal_id": ["normal"]}).to_csv(out / "NORMAL_OPERATION_REPLAY_COMPLETED.csv", index=False)
    pd.DataFrame({"profile_id": []}).to_csv(matrix / "GLOBAL_PROFILES.csv", index=False)
    pd.DataFrame({"climate": ["Context"], "target_c": [25]}).to_csv(matrix / "probe_runs/PROBE_RUN_MANIFEST.csv", index=False)
    pd.DataFrame({"profile_id": []}).to_csv(tmp_path / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    pd.DataFrame([{"climate": "Context", "composition_id": 1, "control": library,
                   "normal_energy_reference_factor": 1., "status": "feasible"}
                  for library in module.NORMAL_LIBRARIES]).to_csv(out / "NORMAL_OPERATION_BUDGET_STATUS.csv", index=False)
    protocol = tmp_path / "response/EVENT_PROTOCOL.csv"
    pd.DataFrame({"climate": ["Context"], "event_id": ["event"]}).to_csv(protocol, index=False)
    (out / "PLAN_AUDIT.json").write_text(json.dumps({"event_protocol_sha256": module.digest(protocol)}))
    module.score()
    choices = pd.read_csv(out / "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")
    assert choices.status.eq("unresolved_engineering_operation").all()
    assert choices.selected_action.isna().all() and choices.event_power_reduction_kw.isna().all()
    paired = pd.read_csv(out / "PAIRED_COMMON_BUDGET_POWER_DIFFERENCES.csv")
    assert len(paired) == 3 and not paired.both_have_selected_action.any()
    assert paired.paired_power_difference_kw.isna().all()
    import dr_selected_uncertainty as uncertainty
    monkeypatch.setattr(uncertainty, "ROOT", tmp_path)
    monkeypatch.setattr(uncertainty, "record_status", lambda *args: None)
    uncertainty.main()
    report = json.loads((tmp_path / "results/dr_capability_uncertainty_identity/AUDIT.json").read_text())
    assert report["selected_operations"] == report["operation_draw_evaluations"] == 0
    empty = pd.read_csv(tmp_path / "results/dr_capability_uncertainty_identity/FROZEN_ACTION_SERVICE_VIOLATIONS.csv")
    assert empty.empty and "event_power_kw_fixed_operation" in empty
