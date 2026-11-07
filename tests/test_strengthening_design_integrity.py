import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from demand_validation import chronological_splits
from strong_baselines_and_placement import relocation, FLOOR_GROUPS, ZONES
import hvac_models
from hvac_confirmation import operation_label


def test_supply_reset_confirms_only_complete_independent_climate_pilots():
    from independent_supply_reset import eligible_climates

    records = []
    for climate in ("A", "B"):
        records.append({"climate": climate, "composition_id": None, "control": "reference", "validation_pass": True})
        records.extend({"climate": climate, "composition_id": composition, "control": control, "validation_pass": True}
            for composition in (1, 2) for control in ("uniform_average", "personal_center", "personal_range"))
    records[-1]["validation_pass"] = False
    assert eligible_climates(records, ["A", "B"], [1, 2]) == {"A"}
    records[1] = records[2].copy()
    assert eligible_climates(records, ["A", "B"], [1, 2]) == set()


def test_retuning_requires_every_declared_probe_within_each_independent_climate():
    from retune_heatpump_controls import complete_probe_climates

    rows = [{"climate": climate, "probe_target_c": target, "validation_pass": True}
        for climate in ("A", "B", "C") for target in (20, 25, 27)]
    rows[0]["validation_pass"] = False
    rows.pop()
    quality = {r["climate"]: r for r in complete_probe_climates(rows, [20, 25, 27], ["A", "B", "C"])}
    assert not quality["A"]["retuning_available"]
    assert quality["B"]["retuning_available"]
    assert not quality["C"]["retuning_available"]
    assert quality["A"]["passed_probes"] == 2
    assert quality["C"]["returned_probes"] == 2


def test_frontier_search_status_distinguishes_complete_uniform_benchmark_from_missing_search(monkeypatch, tmp_path):
    import budget_frontiers as module
    import json

    (tmp_path / "audit").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "audit/GRID_STOP.json").write_text(json.dumps({"further_refinement_executed": True}))
    pd.DataFrame({"completed": [True] * 1674, "relative_engineering_pass": [True] * 1674}).to_csv(
        tmp_path / "results/UNIFORM_SEARCH_FULL_MANIFEST.csv", index=False)
    pd.DataFrame({"score": range(1674)}).to_csv(tmp_path / "results/UNIFORM_SEARCH_METRICS.csv", index=False)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    coverage = {row["candidate_family"]: row for row in module.search_coverage()}
    assert coverage["UNIFORM_SEARCH"]["complete"]
    assert not coverage["BASELINE_OFFSETS"]["complete"]
    assert coverage["BASELINE_OFFSETS"]["expected_logical_operations"] == 3456
    assert coverage["BASELINE_OFFSETS"]["returned_operations"] == 0


def test_spatial_numeric_cohorts_and_reused_operations_join_the_shared_library():
    from budget_frontiers import align_spatial_operations
    import pytest

    profiles = pd.DataFrame({"composition_id": [123, 123], "cohort_label": ["Composition A"] * 2})
    spatial = pd.DataFrame({"composition": [123, 123], "run_id": ["annual", "floor"]})
    fields = ["input_sha256", "input_idf_path", "qa_path", "weather_path", "hourly_path", "hourly_sha256"]
    annual = pd.DataFrame([{"run_id": "annual", **{field: "annual_" + field for field in fields}}])
    floor = pd.DataFrame([{"run_id": "floor", **{field: "floor_" + field for field in fields}}])
    aligned = align_spatial_operations(spatial, profiles, [annual, floor])
    assert aligned.composition.tolist() == ["Composition A", "Composition A"]
    assert aligned.physical_plan_id.tolist() == ["annual_input_sha256", "floor_input_sha256"]
    assert aligned.qa_path.tolist() == ["annual_qa_path", "floor_qa_path"]
    with pytest.raises(ValueError, match="metadata are missing"):
        align_spatial_operations(spatial, profiles, [annual, floor.iloc[:0]])


def test_chronological_calibration_never_splits_tied_time_blocks_or_undated_records():
    rows = []
    for i, day in enumerate([1, 1, 2, 2, 3, 3]):
        rows.append({"person": "A", "time": pd.Timestamp(2001, 1, day, tz="UTC"), "record_id": i})
    rows += [{"person": "B", "time": pd.NaT, "record_id": i + 10} for i in range(6)]
    splits = chronological_splits(pd.DataFrame(rows))
    assert len(splits) == 1
    person, partition, early, late = splits[0]
    assert person == "A" and len(early) == 4 and len(late) == 2
    assert early.time.max() < late.time.min() and set(early.record_id).isdisjoint(late.record_id)


def test_relocation_preserves_floor_and_zone_capacity_under_partial_budget():
    rng = np.random.default_rng(24)
    zones = [z for z in ZONES for _ in range(3)]
    p = pd.DataFrame(
        {
            "profile_id": [f"p{i:03}" for i in range(len(zones))],
            "zone_id": zones,
            "centre_c": rng.normal(24, 2, len(zones)),
        }
    )
    moved, audit = relocation(p, 0.25)
    assert audit["profiles_moved"] <= int(len(p) * 0.25)
    assert p.zone_id.value_counts().sort_index().equals(moved.zone_id.value_counts().sort_index())
    assert all(FLOOR_GROUPS[a] == FLOOR_GROUPS[b] for a, b in zip(p.zone_id, moved.zone_id))


def test_runtime_idd_parser_retains_fields_with_spaces_before_separators(tmp_path, monkeypatch):
    (tmp_path / "Energy+.idd").write_text(
        "Coil:Cooling:DX:SingleSpeed,\n A1 , \\field Name\n N1 ; \\field Minimum Outdoor Dry-Bulb Temperature for Compressor Operation\n"
    )
    monkeypatch.setattr(hvac_models, "RUNTIME", tmp_path / "energyplus.exe")
    fields = hvac_models.idd_fields()
    assert fields["coil:cooling:dx:singlespeed"] == [
        "Name",
        "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation",
    ]


def test_decimal_probe_commands_have_distinct_readvars_temporary_file_stems():
    labels = [
        operation_label(
            {"composition_id": 0, "control": f"probe_{t}", "scenario_tag": "own_response"}
        )
        for t in [22.6, 22.7]
    ]
    assert len({label.split(".")[0] for label in labels}) == 2
    assert all("." not in label for label in labels)


def test_budget_frontier_excludes_an_engineering_unqualified_low_energy_operation():
    from budget_frontiers import select

    frame = pd.DataFrame({"engineering_pass": [False, True],
        "mean_reference_mismatch_c": [0.5, 0.6], "fixed_tail_reference_mismatch_c": [1., 1.1],
        "facility_purchased_kwh": [100., 200.], "run_id": ["unqualified", "qualified"],
        "offset_c": [0., 0.], "physical_plan_id": ["unqualified", "qualified"],
        "control": ["building_mean"] * 2, "permission": ["building"] * 2})
    result = select(frame, 1., 2., "full")
    assert result["selected_physical_plan"] == "qualified"
    assert result["qualifying_plans"] == 1 and result["finite_library_minimum_energy_kwh"] == 200.
    assert select(frame.iloc[:1], 1., 2., "full")["status"] == "infeasible"
