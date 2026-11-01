import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from placement_hvac_summary import matched_effects


def test_system_contrast_uses_identical_qualified_settings_and_preserves_denominator():
    base = pd.DataFrame({"climate": ["A", "B"], "composition_id": [1, 1], "control": ["center"] * 2,
        "layout_arranged": ["25pct"] * 2, "population_gain_c": [0.2, 9.0],
        "fixed_tail_gain_c": [0.4, 10.0], "energy_change_pct": [2.0, 100.0],
        "actual_relocation_fraction": [0.24, 0.24]})
    other = base.rename(columns={"layout_arranged": "layout"}).copy()
    other["engineering_pass_arranged"] = [True, False]
    other["engineering_pass_mixed"] = True
    other["energy_change_pct"] = [-1.0, -100.0]
    base["heating_electric_kwh_arranged"] = [2500.0, 999999.0]
    base["heating_electric_kwh_mixed"] = [500.0, 0.0]
    other["heating_electric_kwh_arranged"] = [1500.0, 999999.0]
    other["heating_electric_kwh_mixed"] = [1000.0, 0.0]
    pairs, summary = matched_effects(base, other)
    assert pairs.climate.tolist() == ["A"]
    row = summary.iloc[0]
    assert row.planned_settings == 2 and row.matched_qualified_settings == 1 and row.unavailable_settings == 1
    assert row.energy_change_pct_vav_median == 2.0
    assert row.energy_change_pct_paired_system_difference_median == -3.0
    assert row.vav_energy_increase_cases == 1 and row.alternative_energy_decrease_cases == 1
    assert row.heating_electric_change_mwh_vav_median == 2.0
    assert row.heating_electric_change_mwh_alternative_median == 0.5
    with pytest.raises(ValueError, match="duplicate settings"):
        matched_effects(pd.concat([base, base]), other)
