import numpy as np
import pandas as pd

from thermal_heterogeneity.human.hierarchy import nested_variance_decomposition, weighted_quantile
from thermal_heterogeneity.human.shared_room_validation import window_statistics


def test_weighted_quantile_is_stable_at_threshold():
    assert weighted_quantile(np.array([2.0, 1.0, 3.0]), np.array([1.0, 1.0, 2.0]), 0.5) == 2.0


def test_variance_components_close():
    frame = pd.DataFrame({"country": ["a", "a", "b", "b"], "city": ["x", "x", "y", "y"], "building": ["p", "p", "q", "q"], "unit": ["u", "u", "v", "v"], "value": [0.0, 2.0, 4.0, 6.0]})
    result = nested_variance_decomposition(frame, "value")
    assert np.isclose(result["share"].sum(), 1.0)


def test_jensen_gap_detects_opposing_votes():
    result = window_statistics(np.array([-1.0, 1.0]))
    assert result["compressed_burden"] == 0.0
    assert result["full_burden"] == 1.0
    assert result["opposing_directions"]
