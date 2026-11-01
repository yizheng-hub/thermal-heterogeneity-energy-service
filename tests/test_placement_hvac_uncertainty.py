import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import placement_uncertainty as module


def planned_pairs():
    rows = []
    for climate in range(9):
        for composition in range(6):
            for control in ["personal_center", "personal_range"]:
                for budget in [25, 50, 100]:
                    rows.append({"climate": str(climate), "composition_id": composition,
                        "composition": ("Multinational " if composition < 3 else "China ") + str(composition),
                        "control": control, "layout": f"within_floor_relocation_{budget}pct",
                        "moved_profiles": budget, "engineering_pass_mixed": True,
                        "engineering_pass_arranged": True,
                        "hourly_path_mixed": f"mixed{climate}{composition}{control}",
                        "hourly_path_arranged": f"arranged{climate}{composition}{control}{budget}",
                        "hourly_sha256_mixed": "same", "hourly_sha256": "same",
                        "facility_purchased_kwh_mixed": 100., "facility_purchased_kwh_arranged": 101.,
                        "population_gain_c": 0.4, "fixed_tail_gain_c": 0.8, "energy_change_pct": 1.})
    return pd.DataFrame(rows)


def test_hvac_inputs_filter_both_qualification_flags_without_losing_planned_denominators(tmp_path):
    base = planned_pairs()
    vav = tmp_path / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv"
    vav.parent.mkdir(parents=True)
    base.rename(columns={"layout": "layout_arranged"}).to_csv(vav, index=False)
    folder = tmp_path / "results/placement_hvac_interaction"
    folder.mkdir()
    alternative = base.drop(columns=["composition", "moved_profiles"]).copy()
    alternative.loc[0, "engineering_pass_arranged"] = False
    for system in ["independent_pthp", "vav_coordinated_outdoor_reset"]:
        alternative.to_csv(folder / (system + "_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv"), index=False)
    operations, mixed, denominators = module.hvac_operations(tmp_path, single_record=True)
    assert len(operations) == 324 * 3 - 2
    assert len(mixed) == 108 * 3
    assert denominators.planned_settings.sum() == 1296
    assert denominators.available_settings.sum() == 1294
    assert denominators.scenario_eligible_planned_settings.sum() == 648
    assert denominators.scenario_eligible_available_settings.sum() == 646
    alternative.loc[1, "facility_purchased_kwh_mixed"] = 99.
    alternative.to_csv(folder / "independent_pthp_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv", index=False)
    with pytest.raises(ValueError, match="Conflicting repeated mixed baseline"):
        module.hvac_operations(tmp_path)


def test_hvac_interaction_is_calculated_within_matched_draws(monkeypatch):
    monkeypatch.setattr(module, "DRAWS", 4)
    key = {"climate": "c", "composition": "p", "control": "center", "layout": "25pct", "scenario": "s"}
    rows = []
    for system, values in [("original_vav", [0., 0., 100., 100.]), ("independent_pthp", [0., 100., 100., 100.])]:
        rows.extend(key | {"system": system, "draw": i, "population_gain_c": v, "fixed_tail_gain_c": 2 * v}
                    for i, v in enumerate(values))
    frame = pd.DataFrame(rows)
    result = module.matched_hvac_draws(frame)[0]
    assert result["population_gain_c_paired_system_difference_median"] == 0.
    assert result["fixed_tail_gain_c_paired_system_difference_median"] == 0.
    assert result["population_gain_c_conditional_fraction_greater_than_vav"] == 0.25
    with pytest.raises(ValueError, match="Incomplete paired HVAC draw indices"):
        module.matched_hvac_draws(frame.drop(index=7))


def test_hvac_trajectory_pairing_rejects_changed_native_clock_or_occupancy():
    a = {key: np.array([0, 1]) for key in ["occupied", "zone_ids", "month", "day", "hour", "dst", "people"]}
    b = {key: value.copy() for key, value in a.items()}
    module.same_physical_calendar(a, b)
    b["hour"][0] = 9
    with pytest.raises(ValueError, match="calendar or zone changed"):
        module.same_physical_calendar(a, b)
    b["hour"] = a["hour"]
    b["people"] = np.array([0., 1.1])
    with pytest.raises(ValueError, match="native occupancy changed"):
        module.same_physical_calendar(a, b)


def test_completed_hvac_cache_checks_contained_inputs_executed_code_and_outputs(tmp_path, monkeypatch):
    source = tmp_path / "method.py"
    source.write_text("frozen method")
    input_file = tmp_path / "input.csv"
    input_file.write_text("frozen scientific input")
    out = tmp_path / "results"
    out.mkdir()
    (out / "summary.csv").write_text("result")
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "executed_sources", lambda: {"method": source})
    module.write_json(out / "INPUT_SHA256.json", {"input.csv": module.digest(input_file)})
    module.write_json(out / "AUDIT.json", {"status": "completed",
        "executed_source_sha256": {"method": module.digest(source)},
        "output_sha256": {"summary.csv": module.digest(out / "summary.csv")}})
    assert module.verified_hvac_cache(out)
    input_file.write_text("changed input")
    with pytest.raises(ValueError, match="input changed"):
        module.verified_hvac_cache(out)
