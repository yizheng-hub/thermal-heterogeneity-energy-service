"""Check staged gating, additive window allowances, and baseline matched subsets."""
import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.response_qualification_sensitivity import (
    CONTROLS,
    DELTAS,
    FLAGS,
    METRICS,
    STAGES,
    WINDOWS,
    analyse,
    rescore,
)


def _scores(count=8):
    rows = []
    for control in CONTROLS:
        for index in range(count):
            row = {"control": control, "climate": "Warm" if index < 4 else "Cold",
                   "composition_id": 1, "event_id": str(index), "action_id": "a",
                   "event_service_limit_factor": 1.0,
                   "reference_precondition_weighted_profile_hours": 1.0,
                   "precondition_service_evaluated": True,
                   "event_mean_power_reduction_kw": 10.0,
                   "event_electricity_reduction_kwh": 20.0,
                   "positive_event_response": True, **dict.fromkeys(FLAGS, True)}
            for party in ("baseline", "action", "reference"):
                for window in WINDOWS:
                    for metric in METRICS:
                        row[f"{party}_{window}_{metric}"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _set_gates(frame, baseline=True, event=True, non_event=True):
    """Write independently specified retained source flags for synthetic cases."""
    frame.loc[:, "baseline_service_qualified"] = baseline
    frame.loc[:, "action_event_service_qualified"] = event
    frame.loc[:, "action_non_event_service_qualified"] = non_event
    frame.loc[:, "service_qualified"] = baseline & event & non_event
    frame.loc[:, "qualified_positive_event_response"] = (
        frame.service_qualified & frame.positive_event_response)


def _staged_scores():
    scores = _scores()
    cases = scores.event_id.astype(int)
    for index, fields in {
        0: {"baseline_recovery_population_mean_mismatch_c": 1.02,
            "action_event_fixed_tail_mean_mismatch_c": 1.03},
        1: {"action_event_population_mean_mismatch_c": 1.02,
            "action_recovery_fixed_tail_mean_mismatch_c": 1.03},
        2: {"action_precondition_population_mean_mismatch_c": 1.02},
        6: {"baseline_precondition_fixed_tail_mean_mismatch_c": 1.08,
            "baseline_event_population_mean_mismatch_c": 1.04,
            "action_recovery_population_mean_mismatch_c": 1.08},
    }.items():
        for field, value in fields.items():
            scores.loc[cases.eq(index), field] = value
    scores.loc[cases.isin((4, 7)), "positive_event_response"] = False
    scores.loc[cases.eq(4), "event_mean_power_reduction_kw"] = -2.0
    scores.loc[cases.eq(4), "event_electricity_reduction_kwh"] = -4.0
    scores.loc[cases.eq(7), ["event_mean_power_reduction_kw", "event_electricity_reduction_kwh"]] = 0
    scores.loc[cases.eq(5), "reference_precondition_weighted_profile_hours"] = 0
    scores.loc[cases.eq(5), "precondition_service_evaluated"] = False
    for party in ("baseline", "action", "reference"):
        for metric in METRICS:
            scores.loc[cases.eq(5), f"{party}_precondition_{metric}"] = np.nan
    _set_gates(scores, ~cases.isin((0, 6)), ~cases.isin((0, 1)), ~cases.isin((1, 2, 6)))
    return scores


def test_failure_stage_priority_nonpositive_and_overlapping_windows():
    scores = _staged_scores()
    strict = rescore(scores, 0)
    expected = [STAGES[i] for i in (0, 1, 2, 3, 4, 3, 0, 4)]
    assert strict[strict.control.eq(CONTROLS[0])].qualification_stage.tolist() == expected
    means, overlaps, _, _ = analyse(scores)
    row = means[(means.delta_c.eq(0)) & means.group_dimension.eq("overall")
                & means.control.eq(CONTROLS[0])].iloc[0]
    assert sum(row[f"{stage}_count"] for stage in STAGES) == 8
    assert row.baseline_failed_count == 2
    assert row.raw_nonpositive_count == 2
    selected = overlaps[(overlaps.delta_c.eq(0)) & overlaps.group_dimension.eq("overall")
                        & overlaps.control.eq(CONTROLS[0])]
    assert selected.either_metric_failure_count.sum() == 8
    assert selected.loc[selected.window.eq("precondition"), "applicable_count"].tolist() == [7, 7]


def test_delta_is_additive_for_both_metrics_parties_and_every_applicable_window():
    scores = _staged_scores()
    raw = scores.event_mean_power_reduction_kw.copy()
    low = rescore(scores, .025)
    medium = rescore(scores, .05)
    high = rescore(scores, .10)
    case = scores.event_id.astype(int)
    assert low.loc[case.eq(0), "baseline_service_qualified"].all()
    assert not low.loc[case.eq(0), "action_event_service_qualified"].any()
    assert not low.loc[case.eq(1), "action_non_event_service_qualified"].any()
    assert low.loc[case.eq(2), "service_qualified"].all()
    assert medium.loc[case.eq(0) | case.eq(1), "service_qualified"].all()
    assert not medium.loc[case.eq(6), "baseline_service_qualified"].any()
    assert not medium.loc[case.eq(6), "action_non_event_service_qualified"].any()
    assert high.service_qualified.all()
    pd.testing.assert_series_equal(high.event_mean_power_reduction_kw, raw)
    assert high.positive_event_response.equals(scores.positive_event_response)
    assert high.loc[case.eq(5), "service_qualified"].all()


def test_original_tolerance_and_retained_flag_mismatch_are_checked():
    scores = _scores(1)
    scores["baseline_event_population_mean_mismatch_c"] += .5e-9
    assert rescore(scores, 0).service_qualified.all()
    scores["baseline_event_population_mean_mismatch_c"] += 2e-9
    with pytest.raises(ValueError, match="retained baseline_service_qualified"):
        rescore(scores, 0)


def test_common_baseline_subset_denominator_and_full_mean_partition():
    scores = _scores(4)
    uniform = scores.control.eq(CONTROLS[0])
    first = scores.event_id.eq("0")
    second = scores.event_id.eq("1")
    scores.loc[uniform & first, "baseline_recovery_population_mean_mismatch_c"] = 1.1
    scores.loc[uniform & second, "action_event_population_mean_mismatch_c"] = 1.1
    scores.loc[uniform, "event_mean_power_reduction_kw"] = [12, 8, 10, 4]
    scores.loc[scores.control.eq(CONTROLS[1]), "event_mean_power_reduction_kw"] = [10, 6, 10, 5]
    _set_gates(scores, ~(uniform & first), ~(uniform & second), True)
    _, _, pairs, _ = analyse(scores)
    row = pairs[pairs.delta_c.eq(0) & pairs.group_dimension.eq("overall")
                & pairs.alternative_control.eq(CONTROLS[1])
                & pairs.subset_kind.eq("pairwise_common_baseline")].iloc[0]
    assert row.full_pair_count == 4
    assert row.subset_pair_count == 3
    assert row.alternative_qualified_positive_event_response_fraction == 1
    assert row.uniform_qualified_positive_event_response_fraction == pytest.approx(2 / 3)
    assert row.subset_zero_filled_mean_difference_kw == pytest.approx(7 / 3)
    assert row.pairwise_both_baseline_difference_contribution_kw == pytest.approx(7 / 4)
    assert row.pairwise_discordant_baseline_difference_contribution_kw == pytest.approx(10 / 4)
    assert row.full_zero_filled_mean_difference_kw == pytest.approx(17 / 4)
    assert pairs.baseline_partition_identity_residual_kw.abs().max() < 1e-12


def test_empty_common_baseline_subset_has_no_invented_conditional_mean():
    scores = _scores(1)
    uniform = scores.control.eq(CONTROLS[0])
    scores.loc[uniform, "baseline_recovery_population_mean_mismatch_c"] = 1.2
    _set_gates(scores, ~uniform, True, True)
    means, _, pairs, _ = analyse(scores)
    assert len(means.delta_c.unique()) == len(DELTAS)
    assert pairs.subset_pair_count.eq(0).all()
    assert pairs.subset_zero_filled_mean_difference_kw.isna().all()
    assert pairs.uniform_qualified_positive_event_response_fraction.isna().all()
    assert pairs.subset_difference_contribution_to_full_mean_kw.eq(0).all()


def test_precondition_applicability_and_original_positivity_cannot_be_redefined():
    scores = _scores(1)
    scores.loc[0, "precondition_service_evaluated"] = False
    with pytest.raises(ValueError, match="Precondition applicability"):
        rescore(scores, .01)
    scores.loc[0, "precondition_service_evaluated"] = True
    scores.loc[0, "positive_event_response"] = False
    with pytest.raises(ValueError, match="Original positivity"):
        rescore(scores, .01)
