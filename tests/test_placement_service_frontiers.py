import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from placement_service_frontiers import frontiers, select_operation, validate_inventory, verified_cache


def operations():
    return pd.DataFrame([
        {"operation_id": "mixed", "control": "mean", "layout": "random_mixing", "moved_profiles": 0,
         "engineering_pass": True, "population_mismatch_c": 2.0, "fixed_tail_mismatch_c": 4.0,
         "facility_purchased_kwh": 100.0, "facility_electric_kwh": 80.0, "facility_gas_hhv_kwh": 20.0},
        {"operation_id": "arranged", "control": "center", "layout": "within_floor_relocation_25pct", "moved_profiles": 66,
         "engineering_pass": True, "population_mismatch_c": 1.0, "fixed_tail_mismatch_c": 2.0,
         "facility_purchased_kwh": 120.0, "facility_electric_kwh": 115.0, "facility_gas_hhv_kwh": 5.0},
        {"operation_id": "invalid", "control": "range", "layout": "within_floor_relocation_25pct", "moved_profiles": 66,
         "engineering_pass": False, "population_mismatch_c": 0.0, "fixed_tail_mismatch_c": 0.0,
         "facility_purchased_kwh": 1.0, "facility_electric_kwh": 1.0, "facility_gas_hhv_kwh": 0.0},
    ]).assign(system="original_vav", climate="A", composition="B", hourly_path="unused")


def test_nested_frontiers_keep_new_feasibility_separate_from_energy_savings():
    frame = operations()
    anchors = pd.DataFrame([{"climate": "A", "composition": "B", "anchor_mean_mismatch_c": 2.0,
                             "anchor_tail_mismatch_c": 4.0}])
    result = frontiers(frame, anchors, "shared_command_menu")
    assert len(result) == 31 * 4
    strict = result[result.budget_type.eq("reference_relative") & result.reference_fraction.eq(0.5)]
    assert strict.status.tolist() == ["infeasible", "feasible", "feasible", "feasible"]
    assert strict.newly_feasible_with_movement.tolist() == [False, True, True, True]
    assert strict.energy_change_from_zero_pct.isna().all()
    relaxed = result[result.budget_type.eq("reference_relative") & result.reference_fraction.eq(1.0)]
    assert relaxed.selected_operation_id.tolist() == ["mixed"] * 4
    assert relaxed.energy_change_from_zero_pct.eq(0).all()


def test_energy_ties_prefer_actual_moves_and_record_exact_minimum():
    frame = operations().iloc[:2].copy()
    frame.loc[0, "facility_purchased_kwh"] = 120.0000005
    choice = select_operation(frame, 2, 4, 1)
    assert choice["selected_operation_id"] == "mixed"
    assert choice["finite_minimum_energy_kwh"] == 120
    assert choice["selected_energy_kwh"] == 120.0000005
    # A nominal50% layout with fewer actual moves wins a tie over a25% label.
    frame.loc[0, ["moved_profiles", "layout"]] = [50, "within_floor_relocation_50pct"]
    assert select_operation(frame, 2, 4, 0.25)["selected_moved_profiles"] == 50


def test_invalid_engineering_and_energy_carriers_cannot_be_silently_used():
    frame = operations()
    assert select_operation(frame, 0.5, 1, 1)["status"] == "infeasible"
    frame.loc[0, "facility_gas_hhv_kwh"] = 200
    with pytest.raises(ValueError, match="carriers do not close"):
        validate_inventory(frame, {})


def test_completed_cache_rejects_changed_scientific_inputs(tmp_path, monkeypatch):
    import json
    import placement_service_frontiers as module
    monkeypatch.setattr(module, "CODE", tmp_path)
    (tmp_path / "input").write_text("original")
    out = tmp_path / "result"
    out.mkdir()
    (out / "AUDIT.json").write_text(json.dumps({"status": "completed", "output_sha256": {}}))
    (out / "INPUT_SHA256.json").write_text(json.dumps({"input": module.digest(tmp_path / "input")}))
    assert verified_cache(out)
    (tmp_path / "input").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        verified_cache(out)
