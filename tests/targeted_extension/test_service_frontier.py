"""Service comparisons must keep shared fallback and never create a trajectory."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SPEC = spec_from_file_location("service_frontier", Path(__file__).resolve().parents[2]
                              / "src/analysis/targeted_extension/service_frontier.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidates():
    rows = []
    for control in MODULE.CONTROLS:
        rows.append({"context": "x", "composition_id": 1, "layout": "mixed",
                     "permission": "zone", "control": control, "variant": 3,
                     "run_id": "fallback", "permission_alias": "",
                     "engineering_status": "cached_qualified",
                     MODULE.SERVICE[0]: 1.0, MODULE.SERVICE[1]: 2.0,
                     MODULE.ENERGY: 100.0, MODULE.CARBON: 30.0})
    personal = rows[1] | {"run_id": "personal", "variant": 0,
                          MODULE.SERVICE[0]: 0.8, MODULE.SERVICE[1]: 1.8,
                          MODULE.ENERGY: 110.0, MODULE.CARBON: 33.0}
    return pd.DataFrame(rows + [personal])


def test_pareto_uses_all_three_objectives_and_keeps_equal_points():
    values = [[100, 1, 2], [110, .8, 1.8], [120, 1.1, 2.1], [100, 1, 2]]
    assert MODULE.pareto_mask(values).tolist() == [True, True, False, True]
    with pytest.raises(ValueError, match="finite"):
        MODULE.pareto_mask([[np.nan, 1, 2]])


def test_only_duplicate_identity_and_permission_alias_are_removed():
    frame = candidates()
    duplicate = frame.iloc[[0]].copy()
    duplicate["variant"] = 1
    alias = frame.iloc[[0]].copy()
    alias["permission_alias"] = "building"
    rejected = frame.iloc[[0]].copy()
    rejected["engineering_status"] = "new_rejected"
    rejected["run_id"] = "failed"
    result = MODULE.canonical_candidates(pd.concat([frame, duplicate, alias, rejected]))
    assert len(result) == 4
    assert len(result[result.run_id.eq("fallback")]) == 3


def test_observed_nodes_keep_fallback_and_report_unavailable_uniform():
    frame = candidates()
    anchors = pd.DataFrame([{"context": "x", "composition_id": 1,
                             "layout": "mixed", "anchor_run_id": "fallback"}])
    result = MODULE.observed_requirement_frontiers(frame, anchors)
    strict = result[result.mean_budget_c.eq(.8)]
    assert strict.reference_status.eq("service_unavailable").all()
    assert strict[strict.control.eq("personal_center")].alternative_run_id.iloc[0] == "personal"
    loose = result[result.mean_budget_c.eq(1)]
    assert loose.both_common25.all()
    assert set(result.alternative_run_id) <= {"", "fallback", "personal"}


def test_near_service_preserves_same_run_and_distinguishes_tolerance():
    frame = candidates()
    frame.loc[frame.run_id.eq("personal"), MODULE.SERVICE] = [1.04, 2.04]
    result = MODULE.near_service_pairs(frame)
    pc = result[result.control.eq("personal_center")]
    assert pc[pc.tolerance_c.eq(.02)].same_physical_run.all()
    assert len(pc[(pc.tolerance_c.eq(.05)) & ~pc.same_physical_run]) == 1


def test_no_match_is_explicit_and_summaries_count_conditions():
    frame = candidates()
    frame = frame[~((frame.control.eq("personal_center")) & frame.run_id.eq("fallback"))]
    result = MODULE.near_service_pairs(frame)
    pc = result[result.control.eq("personal_center")]
    assert pc.status.eq("no_match").all()
    summary = MODULE.summarize_pairs(result, ["control", "tolerance_c"])
    pc_summary = summary[summary.control.eq("personal_center")]
    assert pc_summary.conditions.eq(1).all()
    assert pc_summary.matched_conditions.eq(0).all()
    condition_summary = MODULE.near_condition_summary(result)
    assert condition_summary[condition_summary.control.eq("personal_center")].status.eq("no_match").all()


def test_energy_choice_uses_real_run_and_stable_tie_breaking():
    frame = candidates()
    chosen = MODULE.energy_choice(frame[frame.control.eq("personal_center")])
    assert chosen["run_id"] == "fallback"
    assert MODULE.energy_choice(frame.iloc[:0]) is None


def test_original_selection_is_verified_not_trusted():
    frame = candidates()
    selection = frame.iloc[1].to_dict() | {"mean_budget_c": 1.0,
                                         "fixed_tail_budget_c": 2.0,
                                         "selection_status": "selected"}
    MODULE.verify_existing_selections(frame, pd.DataFrame([selection]))
    selection["run_id"] = "personal"
    with pytest.raises(ValueError, match="trajectory differs"):
        MODULE.verify_existing_selections(frame, pd.DataFrame([selection]))
