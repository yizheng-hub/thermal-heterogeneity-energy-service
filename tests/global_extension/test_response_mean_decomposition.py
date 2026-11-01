"""Check response availability, conditional magnitude, and matched-case accounting."""
import pandas as pd
import pytest

from analysis.global_extension.response_mean_decomposition import analyse


def _scores():
    powers = {"uniform_average": [12, 8, -2, 4],
              "personal_center": [10, 6, 0, 5],
              "personal_range": [9, 7, -1, 3]}
    qualifies = {"uniform_average": [True, False, False, True],
                 "personal_center": [True, True, False, True],
                 "personal_range": [True, True, False, False]}
    rows = []
    for control, values in powers.items():
        for index, power in enumerate(values):
            rows.append({"control": control, "climate": "Warm" if index < 2 else "Cold",
                         "composition_id": 1, "event_id": str(index),
                         "action_id": "a" if index % 2 == 0 else "b",
                         "event_service_limit_factor": 1.0,
                         "service_qualified": qualifies[control][index],
                         "positive_event_response": power > 0,
                         "qualified_positive_event_response": qualifies[control][index],
                         "event_mean_power_reduction_kw": power})
    return pd.DataFrame(rows)


def test_conditional_mean_excludes_failed_cases_and_raw_mean_keeps_all_denominators():
    scores = _scores()
    relaxed = scores.assign(event_service_limit_factor=1.1, event_mean_power_reduction_kw=1000)
    means, pairs = analyse(pd.concat([scores, relaxed], ignore_index=True))
    overall = means[means.group_dimension.eq("overall")].set_index("control")
    uniform = overall.loc["uniform_average"]
    center = overall.loc["personal_center"]
    assert uniform.case_count == 4
    assert uniform.qualified_positive_fraction == pytest.approx(0.5)
    assert uniform.qualified_positive_conditional_mean_kw == pytest.approx(8)
    assert uniform.service_preserving_zero_filled_mean_kw == pytest.approx(4)
    assert uniform.raw_positive_zero_filled_mean_kw == pytest.approx(6)
    assert uniform.raw_positive_conditional_mean_kw == pytest.approx(8)
    assert center.qualified_positive_fraction == pytest.approx(0.75)
    assert center.qualified_positive_conditional_mean_kw == pytest.approx(7)
    assert center.service_preserving_zero_filled_mean_kw == pytest.approx(5.25)
    assert means.mean_identity_residual_kw.abs().max() < 1e-12
    paired = pairs[pairs.group_dimension.eq("overall")].set_index("alternative_control")
    assert paired.loc["personal_center", "alternative_only_contribution_kw"] == pytest.approx(1.5)
    assert paired.loc["personal_center", "both_qualified_power_difference_contribution_kw"] == pytest.approx(-0.25)
    assert paired.loc["personal_center", "observed_zero_filled_mean_difference_kw"] == pytest.approx(1.25)
    assert paired.loc["personal_range", "uniform_only_contribution_kw"] == pytest.approx(-1)
    assert pairs.paired_identity_residual_kw.abs().max() < 1e-12


def test_empty_qualified_group_has_zero_contribution_and_undefined_conditional_mean():
    scores = _scores()
    selected = scores.control.eq("uniform_average") & scores.climate.eq("Cold")
    scores.loc[selected, ["service_qualified", "qualified_positive_event_response"]] = False
    means, _ = analyse(scores)
    row = means[means.group_dimension.eq("climate") & means.group_value.eq("Cold")
                & means.control.eq("uniform_average")].iloc[0]
    assert row.qualified_positive_count == 0
    assert pd.isna(row.qualified_positive_conditional_mean_kw)
    assert row.fraction_times_conditional_mean_kw == 0
    assert row.service_preserving_zero_filled_mean_kw == 0


def test_inconsistent_retained_qualification_is_rejected():
    scores = _scores()
    scores.loc[0, "service_qualified"] = False
    with pytest.raises(ValueError, match="flags are inconsistent"):
        analyse(scores)
