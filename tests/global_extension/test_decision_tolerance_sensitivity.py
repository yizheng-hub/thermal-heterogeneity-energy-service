"""Check absolute margins, full-library reselection and unchanged tie rules."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.global_extension import decision_tolerance_sensitivity as module
from tests.global_extension.test_decision_effect_sizes import candidate, inputs


def test_reselects_previously_unselected_candidates_at_every_delta():
    metrics, anchors, _, _ = inputs([
        candidate("old-mean-choice", 100, 1.04, 1.04, 0.9, 0.9),
        candidate("previously-unselected", 90, 1.02, 1.02, 1.02, 1.02),
        candidate("full-choice", 110, 0.9, 0.9, 0.9, 0.9),
    ])
    detail, qualification, totals = module.scan(metrics, anchors)
    assert detail.full_selected_plan_id.tolist() == [
        "full-choice", "full-choice", "previously-unselected",
        "previously-unselected", "previously-unselected",
    ]
    assert detail.zone_mean_selected_plan_id.tolist() == [
        "old-mean-choice", "old-mean-choice", "previously-unselected",
        "previously-unselected", "previously-unselected",
    ]
    assert totals.selected_physical_plan_changed_settings.tolist() == [1, 1, 0, 0, 0]
    assert totals.eligibility_set_changed_settings.tolist() == [1, 1, 1, 0, 0]
    assert detail.delta_purchased_kwh_full_minus_zone_mean.tolist() == [10, 10, 0, 0, 0]
    assert len(qualification) == 15
    for _, rows in qualification.groupby("physical_plan_id"):
        assert rows.full_qualified.astype(int).is_monotonic_increasing
        assert rows.zone_mean_qualified.astype(int).is_monotonic_increasing


def test_adds_same_absolute_delta_to_both_unequal_common_limits():
    metrics, anchors, _, _ = inputs([
        candidate("population-failure", 90, 1.026, 1.9, 0.9, 1.9),
        candidate("tail-failure", 95, 0.9, 2.026, 0.9, 1.9),
        candidate("safe", 110, 0.9, 1.9, 0.9, 1.9),
    ])
    anchors["anchor_tail_mismatch_c"] = 2.0
    detail, qualification, _ = module.scan(metrics, anchors)
    row = detail.loc[detail.delta_c.eq(0.025)].iloc[0]
    assert row.mean_limit_c == pytest.approx(1.025)
    assert row.tail_limit_c == pytest.approx(2.025)
    assert row.full_selected_plan_id == "safe"
    state = qualification.loc[qualification.delta_c.eq(0.025)]
    assert state.full_qualified.tolist() == [False, True, False]  # sorted physical IDs
    assert state.zone_mean_qualified.all()
    assert detail.loc[detail.delta_c.eq(0.05), "full_selected_plan_id"].item() == "population-failure"


def test_new_disagreement_can_appear_in_an_initially_unchanged_setting():
    metrics, anchors, _, _ = inputs([
        candidate("safe", 100, 0.9, 0.9, 0.9, 0.9),
        candidate("cheap", 90, 1.04, 1.04, 1.02, 1.02),
    ])
    detail, _, totals = module.scan(metrics, anchors)
    assert totals.selected_physical_plan_changed_settings.tolist() == [0, 0, 1, 0, 0]
    assert detail.delta_purchased_kwh_full_minus_zone_mean.iloc[2] == 10


@pytest.mark.parametrize("minimum", [100.0, 1_000_000.0])
def test_keeps_numerical_boundary_aliases_engineering_and_energy_ties(minimum):
    tie = max(1e-6, 1e-9 * minimum)
    rows = [
        candidate("z-minimum", minimum, 1.025 + 0.5e-9, 0.9, 0.9, 0.9),
        candidate("a-tied", minimum + 0.5 * tie, 0.9, 0.9, 0.9, 0.9),
        candidate("outside-score-tolerance", 1, 1.025 + 2e-9, 0.9, 0.9, 0.9),
        candidate("failed-engineering", 0, 0.1, 0.1, 0.1, 0.1),
    ]
    rows[-1]["engineering_pass"] = False
    alias = rows[1] | {"run_id": "later-alias", "control": "personal_center"}
    metrics, anchors, _, _ = inputs([*rows, alias])
    detail, qualification, totals = module.scan(metrics, anchors)
    row = detail.loc[detail.delta_c.eq(0.025)].iloc[0]
    assert row.physical_plan_count == 4
    assert row.physical_alias_count == 1
    assert row.full_qualified_plan_count == 2
    assert row.full_selected_plan_id == "a-tied"
    assert row.full_selected_facility_purchased_kwh == minimum + 0.5 * tie
    assert totals.candidate_rows.eq(5).all()
    assert not qualification.loc[
        qualification.physical_plan_id.eq("failed-engineering"), "full_qualified"
    ].any()


def test_separates_unresolved_cases_from_two_different_physical_choices():
    metrics, anchors, _, _ = inputs([
        candidate("only-choice", 100, 1.04, 1.04, 1.02, 1.02),
    ])
    detail, _, totals = module.scan(metrics, anchors)
    assert totals.full_unresolved_settings.tolist() == [1, 1, 1, 0, 0]
    assert totals.zone_mean_unresolved_settings.tolist() == [1, 1, 0, 0, 0]
    assert totals.availability_disagreement_settings.tolist() == [0, 0, 1, 0, 0]
    assert totals.selected_physical_plan_changed_settings.eq(0).all()
    assert pd.isna(detail.delta_purchased_kwh_full_minus_zone_mean.iloc[2])
    assert detail.delta_purchased_kwh_full_minus_zone_mean.iloc[3] == 0


def test_zero_delta_exactly_preserves_all_original_decision_fields_and_inputs():
    metrics, anchors, expected, _ = inputs([
        candidate("cheap", 100, 1.02, 0.9, 0.9, 0.9),
        candidate("safe", 110, 0.9, 0.9, 0.9, 0.9),
    ])
    metrics_before, anchors_before = metrics.copy(deep=True), anchors.copy(deep=True)
    detail, _, _ = module.scan(metrics, anchors)
    pd.testing.assert_frame_equal(
        detail.loc[detail.delta_c.eq(0), expected.columns].reset_index(drop=True),
        expected, check_exact=True,
    )
    pd.testing.assert_frame_equal(metrics, metrics_before, check_exact=True)
    pd.testing.assert_frame_equal(anchors, anchors_before, check_exact=True)
