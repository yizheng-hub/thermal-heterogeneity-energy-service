import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import aligned_grid_sensitivity as module


def operation(physical, offset=0, energy=100, **updates):
    return {"climate": "A", "composition": "B", "layout": "random_mixing",
        "control": "personal_center", "permission": "native_zone", "offset_c": offset,
        "run_id": "run_" + physical, "physical_plan_id": physical, "source_kind": None,
        "engineering_pass": True, "facility_purchased_kwh": energy, "hourly_path": "unused",
        "mean_reference_mismatch_c": 0.5, "fixed_tail_reference_mismatch_c": 1.0,
        "zone_mean_mean_reference_mismatch_c": 0.2, "zone_mean_fixed_tail_reference_mismatch_c": 0.4,
        **updates}


def test_filter_precedes_alias_dedup_and_keeps_uniform_and_original_spatial():
    frame = pd.DataFrame([
        operation("alias", .125, run_id="a_fine"), operation("alias", 0, run_id="z_coarse"),
        operation("uniform", -4, control="uniform_command_search"),
        operation("spatial", .1, source_kind="new_spatial_run", permission="floor"),
        operation("quarter", .25, control="native_zone_median"),
        operation("eighth", .875, control="building_mean"),
    ])
    half = module.level_operations(frame, .5)
    assert set(half.physical_plan_id) == {"alias", "uniform", "spatial"}
    assert half.set_index("physical_plan_id").loc["alias", "run_id"] == "z_coarse"
    assert module.level_operations(frame, .125).set_index("physical_plan_id").loc["alias", "run_id"] == "a_fine"
    assert "quarter" in set(module.level_operations(frame, .25).physical_plan_id)
    assert frame.loc[3, "offset_c"] == .1


def test_missing_either_mean_endpoint_is_scored_once_without_mutating_history():
    frame = pd.DataFrame([operation("a", zone_mean_mean_reference_mismatch_c=np.nan)])
    calls = []

    def direct(row):
        calls.append(row.physical_plan_id)
        return {"mean_reference_mismatch_c": .2, "fixed_tail_reference_mismatch_c": .4}

    levels, derived = module.prepare_levels(frame, direct)
    assert calls == ["a"] and len(derived) == 1
    assert all(level[module.MEAN_COLUMNS].notna().all().all() for level in levels.values())
    assert pd.isna(frame.loc[0, module.MEAN_COLUMNS[0]])
    frame.loc[0, module.MEAN_COLUMNS[1]] = 9
    with pytest.raises(ValueError, match="scores disagree"):
        module.prepare_levels(frame, direct)


def test_original_energy_tie_can_change_physical_choice_and_raise_selected_energy():
    frame = pd.DataFrame([operation("a", run_id="z_old"),
                          operation("b", .125, 100.0000005, run_id="a_new")])
    levels, _ = module.prepare_levels(frame, lambda row: pytest.fail("Unexpected scoring"))
    rows, changes = module.evaluate_levels(levels, 1, 2)
    assert [row["full_choice"] for row in rows] == ["a", "a", "b"]
    assert [row["full_finite_minimum_energy_kwh"] for row in rows] == [100] * 3
    assert changes[1]["full_physical_choice_changed"]
    assert changes[1]["full_selected_energies_tied"]
    assert changes[1]["full_selected_energy_change_kwh"] > 0


def test_common_full_budgets_engineering_and_infeasibility_never_get_fallback():
    frame = pd.DataFrame([
        operation("mean_only", mean_reference_mismatch_c=2),
        operation("invalid", energy=1, engineering_pass=False),
    ])
    levels, _ = module.prepare_levels(frame, lambda row: pytest.fail("Unexpected scoring"))
    rows, _ = module.evaluate_levels(levels, 1, 2)
    assert all(row["full_status"] == "infeasible" and row["full_choice"] is None for row in rows)
    assert all(row["zone_mean_choice"] == "mean_only" for row in rows)
    assert all(row["zone_mean_selected_full_service_excess_c"] == 1 for row in rows)
    assert all(row["energy_difference_kwh"] is None for row in rows)
    frame = pd.DataFrame([operation("old", mean_reference_mismatch_c=2, zone_mean_mean_reference_mismatch_c=2),
                          operation("new", .125)])
    levels, _ = module.prepare_levels(frame, lambda row: pytest.fail("Unexpected scoring"))
    _, changes = module.evaluate_levels(levels, 1, 2)
    assert changes[1]["full_feasibility_changed"]
    assert changes[1]["full_selected_energy_change_kwh"] is None


def test_fine_reproduction_checks_each_setting_not_only_aggregate_count():
    levels, _ = module.prepare_levels(pd.DataFrame([operation("a")]), lambda row: None)
    rows, _ = module.evaluate_levels(levels, 1, 2)
    choices = pd.DataFrame(rows).assign(climate="A", composition="B")
    expected = choices[choices.level.eq("eighth")][["climate", "composition", "full_choice", "population_budget_c"]].copy()
    module.validate_fine(choices, expected)
    expected.loc[:, "full_choice"] = "different"
    with pytest.raises(ValueError, match="full_choice"):
        module.validate_fine(choices, expected)


@pytest.mark.parametrize("changed", ["input", "source", "output"])
def test_completed_cache_binds_inputs_source_and_outputs(tmp_path, monkeypatch, changed):
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "source_hashes", lambda: {"module": "original"})
    (tmp_path / "input").write_text("original")
    out = tmp_path / "result"
    out.mkdir()
    (out / "output.csv").write_text("original")
    (out / "INPUT_SHA256.json").write_text(json.dumps({"input": module.digest(tmp_path / "input")}))
    (out / "AUDIT.json").write_text(json.dumps({"status": "completed", "source_sha256": {"module": "original"},
        "output_sha256": {"output.csv": module.digest(out / "output.csv")}}))
    assert module.verified_cache(out)
    if changed == "source":
        monkeypatch.setattr(module, "source_hashes", lambda: {"module": "changed"})
    else:
        (tmp_path / "input" if changed == "input" else out / "output.csv").write_text("changed")
    with pytest.raises(ValueError, match=changed + " changed"):
        module.verified_cache(out)


def test_nonempty_incomplete_directory_is_preserved(tmp_path):
    (tmp_path / "partial.csv").write_text("partial")
    with pytest.raises(FileExistsError, match="Preserve"):
        module.verified_cache(tmp_path)


def test_synthetic_main_writes_all_levels_derived_hashes_and_reuses_completed_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path / "out")
    monkeypatch.setattr(module, "MATRIX", tmp_path / "raw/matrix")
    monkeypatch.setattr(module, "EXPECTED_SETTINGS", 1)
    monkeypatch.setattr(module, "source_hashes", lambda: {"synthetic_module": "fixed"})
    monkeypatch.setattr(module.scoring, "normalise_profiles", lambda frame: frame)
    monkeypatch.setattr(module.scoring, "zone_mean_profiles", lambda frame: frame)
    monkeypatch.setattr(module.scoring, "load_hourly", lambda path: path)
    calls = []

    def score(path, group, tail):
        calls.append((path, len(group), len(tail)))
        return {"summary": {"mean_reference_mismatch_c": .2, "fixed_tail_reference_mismatch_c": .4}}

    monkeypatch.setattr(module.scoring, "score_trajectory", score)
    hourly = tmp_path / "raw/retained.npz"
    hourly.parent.mkdir(parents=True)
    hourly.write_bytes(b"synthetic retained trajectory")
    library = pd.DataFrame([operation("a"), operation("b", .125, 90, hourly_path=str(hourly),
        hourly_sha256=module.digest(hourly), mean_reference_mismatch_c=1.2,
        zone_mean_mean_reference_mismatch_c=np.nan)] + [
        operation("u" + str(i), target - 25, 1000 + i, control="uniform_command_search")
        for i, target in enumerate(np.linspace(20, 27, 31))])
    expected = pd.DataFrame([{"climate": "A", "composition": "B", "candidate_operations": 33,
        "full_qualified_operations": 32, "zone_mean_qualified_operations": 33,
        "qualified_by_both": 32, "qualified_only_by_full": 0, "qualified_only_by_mean": 1,
        "qualified_operation_sets_differ": True, "choices_differ": True, "full_choice": "a",
        "zone_mean_choice": "b", "zone_mean_selected_full_service_excess_c": .2,
        "energy_difference_kwh": 10, "population_budget_c": 1.0, "fixed_tail_budget_c": 2.0}])
    paths = {"library": module.ROOT / "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv",
        "profiles": module.MATRIX / "GLOBAL_PROFILES.csv",
        "tails": module.ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
        "nominal": module.ROOT / "results/nominal_information/NOMINAL_QUALIFICATION_COUNTS.csv"}
    profiles = pd.DataFrame({"climate": ["A"] * 267, "composition": ["B"] * 267,
                             "profile_id": [str(i) for i in range(267)]})
    for key, frame in [("library", library), ("profiles", profiles), ("tails", profiles.iloc[:27]), ("nominal", expected)]:
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[key], index=False)
    anchor = paths["tails"].with_name("GLOBAL_25C_SERVICE_ANCHORS.csv")
    pd.DataFrame([{"climate": "A", "composition": "B", "anchor_mean_mismatch_c": 1, "anchor_tail_mismatch_c": 2}]).to_csv(anchor, index=False)
    nominal = paths["nominal"].parent
    (nominal / "AUDIT.json").write_text(json.dumps({"status": "completed", "output_sha256": module.digest(paths["nominal"])}))
    (nominal / "INPUT_SHA256.json").write_text(json.dumps({
        paths[k].relative_to(tmp_path).as_posix(): module.digest(paths[k]) for k in ["library", "profiles", "tails"]}))
    protocol = tmp_path / "data/metadata/strengthening_protocol/analysis_plan_amendment_31_aligned_grid_sensitivity.yaml"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("synthetic frozen protocol")
    module.main()
    out = module.ROOT / "results/aligned_grid_sensitivity"
    assert len(pd.read_csv(out / "ALIGNED_GRID_CHOICES.csv")) == 3
    assert len(pd.read_csv(out / "ALIGNED_GRID_LEVEL_CHANGES.csv")) == 2
    assert len(pd.read_csv(out / "DERIVED_ZONE_MEAN_SCORES.csv")) == 1
    assert len(calls) == 1 and calls[0][1:] == (267, 27)
    assert "raw/retained.npz" in json.loads((out / "INPUT_SHA256.json").read_text())
    module.main()
    assert len(calls) == 1
