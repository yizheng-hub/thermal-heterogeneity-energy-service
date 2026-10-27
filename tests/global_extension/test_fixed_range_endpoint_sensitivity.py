from __future__ import annotations

import pandas as pd
import pytest

from analysis.global_extension.fixed_range_endpoint_sensitivity import (
    direct_comparisons,
    fixed_width_profiles,
    summarise_direct,
)


def test_fixed_width_profiles_preserve_centres_and_set_symmetric_bounds():
    profiles = pd.DataFrame({
        "centre_c": [22.0, 25.0],
        "lower_c": [20.0, 21.0],
        "upper_c": [24.0, 29.0],
    })
    result = fixed_width_profiles(profiles, 1.5)

    assert result.centre_c.tolist() == [22.0, 25.0]
    assert result.lower_c.tolist() == [20.5, 23.5]
    assert result.upper_c.tolist() == [23.5, 26.5]
    with pytest.raises(ValueError, match="positive and finite"):
        fixed_width_profiles(profiles, 0.0)


def _metrics() -> pd.DataFrame:
    rows = []
    for width in (1.5, 2.0):
        for climate in range(9):
            for composition in range(6):
                for index, control in enumerate(
                    ("uniform_average", "personal_center", "personal_range")
                ):
                    rows.append({
                        "fixed_half_width_c": width,
                        "climate": f"Climate {climate}",
                        "composition": f"Composition {composition}",
                        "layout": "random_mixing",
                        "permission": "native_zone",
                        "control": control,
                        "offset_c": 0.0,
                        "run_id": f"{width}-{climate}-{composition}-{control}",
                        "mean_reference_mismatch_c": 2.0 - 0.1 * index,
                        "fixed_tail_reference_mismatch_c": 3.0 - 0.1 * index,
                        "modeled_interval_person_hour_coverage": 0.5 + 0.1 * index,
                        "modeled_interval_mean_excess_c": 1.0 - 0.1 * index,
                        "facility_purchased_kwh": 1_000.0 + 10.0 * index,
                    })
    return pd.DataFrame(rows)


def test_direct_fixed_width_summary_covers_two_widths_and_all_groups():
    comparisons = direct_comparisons(_metrics())
    summary = summarise_direct(comparisons)

    assert len(comparisons) == 216
    assert len(summary) == 4
    assert all(row["comparisons"] == 54 for row in summary)
    assert all(row["fixed_interval_coverage_higher"] == 54 for row in summary)
    assert all(row["fixed_interval_mean_excess_lower"] == 54 for row in summary)
    assert all(row["facility_energy_increase"] == 54 for row in summary)
