from __future__ import annotations

import pandas as pd

from src.analysis.validation.personal_range_validation import (
    compare_variable_with_fixed_widths,
    external_enth_validation,
)


def test_variable_width_is_not_labelled_supported_when_fixed_band_is_better() -> None:
    summary = pd.DataFrame([
        {"outcome": "acceptability", "range_model": "virtual_width",
         "balanced_accuracy": 0.63, "roc_auc": 0.70},
        {"outcome": "acceptability", "range_model": "fixed_2C",
         "balanced_accuracy": 0.65, "roc_auc": 0.71},
        {"outcome": "preference_binary", "range_model": "virtual_width",
         "balanced_accuracy": 0.58, "roc_auc": 0.64},
        {"outcome": "preference_binary", "range_model": "fixed_1.5C",
         "balanced_accuracy": 0.62, "roc_auc": 0.66},
    ])

    rows, supported = compare_variable_with_fixed_widths(summary)

    assert not supported
    assert {row["best_fixed_model"] for row in rows} == {"fixed_2C", "fixed_1.5C"}
    assert all(row["balanced_accuracy_difference"] < 0 for row in rows)


def test_external_summary_uses_defined_person_equal_metrics() -> None:
    frame = pd.DataFrame(
        {
            "person": ["p1"] * 8,
            "day": [1, 1, 1, 2, 2, 2, 3, 3],
            "time": pd.date_range("2024-01-01", periods=8, freq="h"),
            "temperature": [22.0, 24.0, 26.0, 22.5, 24.5, 26.5, 22.0, 27.0],
            "preference": [-1, 0, 1, -1, 0, 1, -1, 1],
        }
    )

    predictions, summary = external_enth_validation(frame, half_width=2.0)

    assert len(predictions) == 2
    assert summary.people_with_multiple_observed_classes.eq(1).all()
    assert summary.person_equal_accuracy.notna().all()
    assert summary.person_equal_balanced_accuracy.notna().all()
