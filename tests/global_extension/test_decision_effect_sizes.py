"""Boundary and interpretation checks for the fixed-plan effect-size audit."""

from __future__ import annotations

import pandas as pd
import pytest

from src.analysis.global_extension import decision_effect_sizes as module


def candidate(plan, energy, population, tail, mean_population, mean_tail):
    return {
        "climate": "test", "composition": "group", "layout": "random_mixing",
        "permission": "native_zone", "physical_plan_id": plan, "run_id": plan,
        "control": "uniform_average", "offset_c": 0.0, "engineering_pass": True,
        "assigned_profile_count": 10, "facility_purchased_kwh": energy,
        "mean_reference_mismatch_c": population, "fixed_tail_reference_mismatch_c": tail,
        "zone_mean_mean_reference_mismatch_c": mean_population,
        "zone_mean_fixed_tail_reference_mismatch_c": mean_tail,
    }


def inputs(rows):
    metrics = pd.DataFrame(rows)
    anchors = pd.DataFrame([{
        "climate": "test", "composition": "group", "layout": "random_mixing",
        "anchor_mean_mismatch_c": 1.0, "anchor_tail_mismatch_c": 1.0,
        "tail_profile_count": 1,
    }])
    memberships = anchors[module.PROFILE_KEYS].assign(profile_id="fixed-profile")
    reported = module.original.fixed_plan_information_diagnostic(
        metrics, anchors, fractions=(1.0,)
    )
    return metrics, anchors, reported, memberships


def test_rechecks_both_endpoints_and_keeps_signed_margins_and_energy_direction():
    data = inputs([
        candidate("cheap", 100.0, 0.8, 1.2, 0.5, 0.5),
        candidate("safe", 105.0, 0.9, 0.9, 0.9, 0.9),
    ])
    detail, replay, summary = module.analyse(*data)
    row = detail.iloc[0]
    assert len(replay) == 1
    assert row.choice_change_type == "mean_choice_fails_full_review"
    assert row.zone_mean_selected_full_population_pass
    assert not row.zone_mean_selected_full_tail_pass
    assert row.full_selected_full_review_pass
    assert row.zone_mean_selected_full_population_margin_c == pytest.approx(-0.2)
    assert row.zone_mean_selected_full_tail_margin_c == pytest.approx(0.2)
    assert row.delta_purchased_kwh_full_minus_zone_mean == pytest.approx(5.0)
    assert row.delta_purchased_pct_full_minus_zone_mean == pytest.approx(5.0)
    assert row.zone_mean_selected_max_excess_in_score_tolerances == pytest.approx(2e8)
    assert row.energy_change_exceeds_both_tie_tolerances
    assert summary["counts"]["tail_limit_failures"] == 1
    assert summary["counts"]["population_limit_failures"] == 0


def test_different_choice_is_not_automatically_a_failed_full_review():
    data = inputs([
        candidate("full-cheap", 100.0, 0.9, 0.9, 1.1, 1.1),
        candidate("mean-choice", 105.0, 0.95, 0.95, 0.9, 0.9),
    ])
    detail, _, summary = module.analyse(*data)
    assert detail.iloc[0].choice_change_type == "different_choice_both_full_qualified"
    assert detail.iloc[0].delta_purchased_kwh_full_minus_zone_mean == pytest.approx(-5.0)
    assert summary["counts"]["changed_mean_choices_failing_full_review"] == 0
    assert summary["counts"]["changed_choices_both_full_qualified"] == 1


@pytest.mark.parametrize("minimum", [100.0, 1_000_000.0])
def test_preserves_service_boundary_and_both_original_energy_tie_scales(minimum):
    tie_tolerance = max(1e-6, 1e-9 * minimum)
    data = inputs([
        candidate("mean-cheap", minimum - 1, 1 + 2e-9, 0.9, 0.9, 0.9),
        candidate("z-pool-minimum", minimum, 1 + 0.5e-9, 0.9, 1.1, 1.1),
        candidate("a-tied-choice", minimum + 0.5 * tie_tolerance, 0.9, 0.9, 1.1, 1.1),
    ])
    detail, _, _ = module.analyse(*data)
    row = detail.iloc[0]
    assert row.full_qualified_plan_count == 2
    assert row.full_selected_run_id == "a-tied-choice"
    assert row.zone_mean_selected_max_full_service_excess_c == pytest.approx(2e-9)
    assert row.full_selected_pool_minimum_purchased_kwh == minimum
    assert row.full_selected_energy_tie_tolerance_kwh == pytest.approx(tie_tolerance)
    assert row.full_selected_energy_above_pool_minimum_kwh == pytest.approx(
        0.5 * tie_tolerance, rel=1e-6, abs=1e-12
    )


def test_rejects_drift_from_the_recorded_selection_and_fixed_membership():
    data = inputs([
        candidate("cheap", 100.0, 0.8, 1.2, 0.5, 0.5),
        candidate("safe", 105.0, 0.9, 0.9, 0.9, 0.9),
    ])
    metrics, anchors, reported, memberships = data
    altered = reported.copy()
    altered.loc[0, "full_selected_run_id"] = "different"
    with pytest.raises(ValueError, match="replay differs"):
        module.analyse(metrics, anchors, altered, memberships)
    with pytest.raises(ValueError, match="membership counts differ"):
        module.analyse(metrics, anchors, reported, memberships.iloc[:0])
