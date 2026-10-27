import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.tail_fraction_sensitivity import (
    _direct_summary,
    tail_indices,
)


def test_tail_indices_are_nested_and_use_ceil_counts():
    distances = np.arange(267, dtype=float)
    ids = pd.Series([f"p{index:03d}" for index in range(267)])
    tails = tail_indices(distances, ids)
    assert [len(tails[value]) for value in (0.05, 0.10, 0.20)] == [14, 27, 54]
    assert set(tails[0.05]).issubset(tails[0.10])
    assert set(tails[0.10]).issubset(tails[0.20])


def test_tail_indices_reject_unsorted_or_invalid_fractions():
    distances = np.arange(4, dtype=float)
    ids = pd.Series(list("abcd"))
    with pytest.raises(ValueError, match="unique and increasing"):
        tail_indices(distances, ids, (0.2, 0.1))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        tail_indices(distances, ids, (0.0,))


def test_direct_summary_counts_service_and_energy_directions():
    frame = pd.DataFrame(
        {
            "alternative_control": ["personal_center"] * 3,
            "status": ["matched"] * 3,
            "delta_mean_reference_mismatch_c": [-0.2, -0.1, 0.1],
            "delta_fixed_tail_reference_mismatch_c": [-0.3, 0.2, -0.1],
            "delta_facility_purchased_kwh": [2.0, -1.0, 0.0],
        }
    )
    result = _direct_summary(frame)["personal_center"]
    assert result["population_mean_mismatch_lower"] == 2
    assert result["fixed_tail_mismatch_lower"] == 2
    assert result["both_mismatch_endpoints_lower"] == 1
    assert result["facility_energy_direction"] == {
        "increase": 1,
        "decrease": 1,
        "tie": 1,
    }
