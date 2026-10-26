import numpy as np
import pandas as pd

from analysis.targeted_extension.analyse_common_baseline_dr import (
    information_centres,
    phase_indices,
)


def test_information_levels_are_nested_descriptions():
    profiles = pd.DataFrame({
        "zone_id": ["a", "a", "b", "b"],
        "centre_c": [20.0, 22.0, 24.0, 26.0],
    })
    assert np.allclose(information_centres(profiles, "building_mean"), 23.0)
    assert np.allclose(information_centres(profiles, "zone_mean"), [21, 21, 25, 25])
    assert np.allclose(information_centres(profiles, "full_distribution"), [20, 22, 24, 26])


def test_phase_indices_cover_two_plus_two_plus_48_hours():
    month = np.ones(100, dtype=int)
    day = np.repeat(np.arange(1, 6), 24)[:100]
    hour = np.tile(np.arange(1, 25), 5)[:100]
    dst = np.zeros(100, dtype=int)
    phases = phase_indices({"month": month, "day": day, "hour": hour, "dst": dst},
                            {"month": 1, "day": 2})
    assert [len(phases[name]) for name in ("precondition", "event", "recovery")] == [2, 2, 48]
    assert phases["event"][0] == phases["precondition"][-1] + 1
