import numpy as np
import pandas as pd

from analysis.targeted_extension.repeated_observation_audit import (
    binary_metrics,
    support_table,
    time_ordered_holdout,
)


def test_binary_metrics_counts_both_classes():
    result = binary_metrics(
        pd.Series([True, True, False, False]),
        pd.Series([True, False, True, False]),
    )
    assert result["accuracy"] == 0.5
    assert result["balanced_accuracy"] == 0.5


def test_support_and_time_ordered_holdout_do_not_use_future_records():
    frame = pd.DataFrame({
        "subject_key": ["a"] * 5,
        "record_id": np.arange(5),
        "timestamp_parsed": pd.date_range("2020-01-01", periods=5),
        "observed_temperature_c": [20.0, 21.0, 22.0, 23.0, 24.0],
        "thermal_acceptability": ["acceptable"] * 5,
        "thermal_preference": ["no change"] * 5,
        "thermal_sensation": [0.0] * 5,
        "interval_centre_c": [20.0, 21.0, 22.0, 99.0, 99.0],
        "interval_half_width_c": [1.0, 2.0, 3.0, 99.0, 99.0],
    })
    support = support_table(frame)
    assert support.loc[support.minimum_records.eq(5), "subject_within_building_series"].item() == 1
    heldout = time_ordered_holdout(frame)
    assert len(heldout) == 2
    assert heldout.training_centre_c.eq(21.0).all()
    assert heldout.training_model_half_width_c.eq(2.0).all()
