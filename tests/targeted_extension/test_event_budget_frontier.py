import pandas as pd
import pytest

from analysis.targeted_extension.event_budget_frontier import (
    calculate_frontier,
    prepare_actions,
)


def _row(run_id, amplitude, response, event_score, recovery_score, original):
    return {
        "data_source": "ASHRAE",
        "context": "Example",
        "composition_id": 1,
        "layout": "random",
        "permission": "building",
        "control": "uniform_average",
        "event_month": 7,
        "service_fraction": 1.0,
        "run_id": run_id,
        "amplitude_c": amplitude,
        "engineering_qualified": True,
        "raw_positive": True,
        "baseline_qualified": True,
        "event_qualified": original,
        "recovery_qualified": original,
        "event_average_reduction_kw": response,
        "event_perturbed_mean_c": event_score,
        "event_mean_budget_c": 2.0,
        "event_perturbed_fixed_tail_c": 4.0,
        "event_fixed_tail_budget_c": 4.0,
        "recovery_perturbed_mean_c": recovery_score,
        "recovery_mean_budget_c": 2.0,
        "recovery_perturbed_fixed_tail_c": 4.0,
        "recovery_fixed_tail_budget_c": 4.0,
    }


def test_zero_reproduces_original_and_frontier_is_discrete_and_nested():
    source = pd.DataFrame(
        [
            _row("small", 0.5, 4.0, 2.0, 2.0, True),
            _row("large", 1.0, 8.0, 2.2, 2.0, False),
        ]
    )
    actions = prepare_actions(source)
    decisions, _ = calculate_frontier(actions, (0.0, 0.1, 0.2))
    assert decisions.selected_response_kw.tolist() == pytest.approx([4.0, 4.0, 8.0])


def test_recovery_failure_cannot_be_unlocked_by_event_allowance():
    source = pd.DataFrame([_row("blocked", 1.0, 8.0, 2.1, 2.1, False)])
    actions = prepare_actions(source)
    decisions, _ = calculate_frontier(actions, (0.0, 1.0))
    assert not decisions.selected.any()
