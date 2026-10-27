import math

import pandas as pd
import pytest

from analysis.global_extension.reference_anchor_sensitivity import (
    headline_row,
    reference_slug,
    selection_failures_by_climate,
)


def _result():
    direct = {
        control: {
            "population_mean_mismatch_lower": 50,
            "fixed_tail_mismatch_lower": 30,
            "facility_energy_direction": {
                "increase": 45,
                "decrease": 9,
                "tie": 0,
            },
        }
        for control in ("personal_center", "personal_range")
    }
    alternatives = {
        control: {"control_service_available": 12}
        for control in ("personal_center", "personal_range")
    }
    return {
        "direct": direct,
        "common_service": {
            "0.75": {
                "uniform_service_available": 10,
                "alternatives": alternatives,
            },
            "0.5": {
                "uniform_service_available": 0,
                "alternatives": alternatives,
            },
        },
        "information": {
            "by_service_fraction": {
                "1": {
                    "full_distribution_resolvable": {
                        "scenarios": 54,
                        "eligibility_disagreement": 23,
                        "selection_disagreement": 10,
                        "zone_mean_selected_plan_fails_full_review": 10,
                    }
                }
            }
        },
    }


def test_reference_slug_is_stable_and_rejects_nonfinite_values():
    assert reference_slug(24.0) == "reference_24C"
    assert reference_slug(24.5) == "reference_24p5C"
    assert reference_slug(-1.0) == "reference_m1C"
    with pytest.raises(ValueError, match="finite"):
        reference_slug(math.nan)


def test_headline_row_flattens_predeclared_endpoints():
    row = headline_row(25.0, _result())
    assert row["reference_target_c"] == 25.0
    assert row["eligibility_disagreement_at_fraction_1"] == 23
    assert row["zone_mean_selected_plan_fails_full_review_at_fraction_1"] == 10
    assert row["personal_center_direct_energy_increase"] == 45
    assert row["personal_range_available_at_fraction_0p75"] == 12


def test_selection_failures_by_climate_uses_fraction_one_resolvable_scope():
    frame = pd.DataFrame(
        {
            "climate": ["Beijing", "Beijing", "Malmo", "Malmo"],
            "service_fraction": [1.0, 1.0, 1.0, 0.75],
            "full_unresolved": [False, True, False, False],
            "eligibility_disagreement": [True, True, False, True],
            "selection_disagreement": [True, True, False, True],
            "zone_mean_selected_plan_fails_full_review": [True, True, False, True],
        }
    )
    result = selection_failures_by_climate(25.0, frame).set_index("climate")
    assert result.loc["Beijing", "full_resolvable_scenarios_at_fraction_1"] == 1
    assert (
        result.loc[
            "Beijing",
            "zone_mean_selected_plan_fails_full_review_at_fraction_1",
        ]
        == 1
    )
    assert result.loc["Malmo", "selection_disagreement_at_fraction_1"] == 0
