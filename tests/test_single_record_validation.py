"""Check the calibration-information boundary and executable range bounds."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from single_record_validation import frozen_splits, one_record_features, signed_margin


def test_uses_first_early_record_without_evaluation_or_later_calibration_information():
    frame = pd.DataFrame({
        "record_id": ["a", "b", "c"], "person": ["p"] * 3,
        "time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"], utc=True),
        "reference_centre": [22., 28., 30.], "source_group": ["s"] * 3,
        "building_id": ["house"] * 3, "temperature": [22., 24., 25.],
        "acceptability": [1., 0., 1.], "preference": [0., 1., 0.],
    })
    assignment = pd.DataFrame({"person": ["p"] * 3, "record_id": ["b", "a", "c"],
        "partition": ["final_test"] * 3,
        "stage": ["early_calibration", "early_calibration", "later_evaluation"]})
    splits = frozen_splits(frame, assignment, timed=True)
    values = one_record_features(splits, pd.Series({"a": 1., "b": 4.}), {"s": 24., "__all__": 24.}, 2)
    assert len(values) == 1
    assert values.calibration_record_id.iloc[0] == "a"
    assert values.raw_center.iloc[0] == 22.
    assert values.shrink_center.iloc[0] == (22. + 2 * 24.) / 3
    assert values.published_half_width.iloc[0] == 1.
    undated = one_record_features(frozen_splits(frame, assignment, timed=False),
        pd.Series({"a": 1., "b": 4.}), {"s": 24., "__all__": 24.}, 2)
    assert undated.calibration_record_id.iloc[0] == "b"


def test_range_inclusion_obeys_original_clipping_at_both_limits():
    margin = signed_margin(np.array([15.5, 16., 29.5, 30.5]),
        np.array([17., 17., 29., 29.]), np.array([3., 3., 3., 3.]))
    np.testing.assert_array_equal(margin >= 0, [False, True, True, False])
    np.testing.assert_allclose(margin, [-.5, 0., .5, -.5])
