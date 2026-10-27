import numpy as np
import pandas as pd

from analysis.seasonal.seasonal_adaptation import (
    estimate_within_place_warm_cool_contrast,
    shift_demand_coordinates,
    warm_cool_month_adjustments,
)


def test_within_place_contrast_uses_only_paired_places():
    observations = pd.DataFrame({
        "city": ["a", "a", "a", "a", "b", "b", "c"],
        "season": ["winter", "winter", "summer", "summer", "winter", "summer", "summer"],
        "centre_c": [20.0, 22.0, 24.0, 26.0, 18.0, 24.0, 99.0],
    })
    summary, effects = estimate_within_place_warm_cool_contrast(
        observations, place_columns=["city"]
    )
    # City a: +4 C with equal 2/2 counts; city b: +6 C with 1/1 counts.
    # Fixed-effect weights are 1 and 0.5, giving 4.666... C.
    assert np.isclose(summary["warm_minus_cool_c"], 14 / 3)
    assert summary["eligible_places"] == 2
    assert set(effects.city) == {"a", "b"}


def test_seasonal_shift_preserves_interval_widths_and_moves_centres():
    profiles = pd.DataFrame({
        "centre_c": [23.0, 25.0],
        "controller_lower_c": [21.0, 22.5],
        "controller_upper_c": [25.0, 27.5],
    })
    shifted = shift_demand_coordinates(profiles, np.array([-1.0, 1.5]))
    assert shifted.centre_c.tolist() == [22.0, 26.5]
    assert np.allclose(
        shifted.controller_upper_c - shifted.controller_lower_c,
        profiles.controller_upper_c - profiles.controller_lower_c,
    )


def test_warm_cool_month_adjustments_reverse_between_hemispheres():
    months = [1, 4, 7, 10]
    north = warm_cool_month_adjustments(months, 2.0, latitude=40)
    south = warm_cool_month_adjustments(months, 2.0, latitude=-34)
    assert north.tolist() == [-1.0, 0.0, 1.0, 0.0]
    assert south.tolist() == [1.0, 0.0, -1.0, 0.0]
