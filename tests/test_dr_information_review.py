import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import dr_information_review as module


def fixture_data():
    scorer = module.explicit_scorer()
    profiles = pd.DataFrame({"profile_id": ["1", "2", "3", "4"], "zone_id": ["A", "A", "B", "B"],
        "centre_c": [20., 24., 26., 30.], "controller_lower_c": [19., 23., 25., 29.],
        "controller_upper_c": [21., 25., 27., 31.]})
    reference = {"month": np.ones(4, int), "day": np.ones(4, int), "hour": np.arange(1, 5),
        "zone_ids": np.array(["A", "B"]), "temperatures": np.tile([22., 28.], (4, 1)),
        "people": np.tile([1., 3.], (4, 1)), "occupied": np.ones(4, bool)}
    for name in ["facility_electric_kwh", "hvac_electric_kwh", "heating_electric_kwh",
                 "cooling_electric_kwh", "fans_electric_kwh", "gas_hhv_kwh"]:
        reference[name] = np.full(4, 10. if name == "facility_electric_kwh" else 1.)
    baseline = {k: v.copy() for k, v in reference.items()}
    baseline["temperatures"][0] = [20., 26.]
    plan = []
    contexts = {}
    for i, name in enumerate("abcde"):
        event = {"event_id": "E", "start_timestamp": "2001-01-01 01:00:00", "duration_hours": 2,
            "recovery_hours": 1, "precondition_hours": int(name in "ce")}
        action = {k: v.copy() for k, v in baseline.items()}
        action["facility_electric_kwh"][1:3] -= (i + 1) / 2
        if name == "b":
            action["temperatures"][1:3] = [20., 26.]
        if name == "c":
            action["facility_electric_kwh"][0] += 4
        if name == "d":
            action["temperatures"][3] = [20., 26.]
        plan.append({"climate": "A", "composition_id": 1, "composition": "Test", "control": "building_search",
            "event_id": "E", "action_id": name, "run_id": "run_" + name, "normal_id": "N"})
        contexts[name] = (baseline, action, reference, profiles, {"1", "4"}, event)
    data = {"plan": pd.DataFrame(plan), "completed": pd.DataFrame({"run_id": ["run_" + x for x in "abcde"],
        "completed": True, "final_engineering_qualified": True}),
        "budgets": pd.DataFrame([{"climate": "A", "composition_id": 1, "control": "building_search",
            "normal_energy_reference_factor": b, "status": "feasible" if b != .95 else "infeasible_normal_energy_budget"}
            for b in [.95, 1., 1.05]]), "protocol": pd.DataFrame([{"climate": "A", "event_id": "E"}])}
    def context(row):
        return contexts[row.action_id]
    data["scores"] = module.replay_actions(data, context, scorer)
    data["choices"] = module.select_choices(data["scores"], data["budgets"], data["protocol"])
    return data, context, scorer


def test_explicit_tail_is_fixed_window_weights_are_retained_and_reference_is_not_averaged():
    data, context, scorer = fixture_data()
    full = data["scores"]
    assert full.set_index("action_id").loc["b", "action_event_fixed_tail_mean_mismatch_c"] == 3
    assert not full.set_index("action_id").loc["b", "service_qualified"]
    mean = module.mean_actions(full, data, context, scorer)
    assert mean.set_index("action_id").loc["b", "action_event_fixed_tail_mean_mismatch_c"] == 2
    assert mean.set_index("action_id").loc["b", "service_qualified"]
    reference_columns = [c for c in full if c.startswith("reference_")]
    pd.testing.assert_frame_equal(full[reference_columns], mean[reference_columns])
    interval_columns = [c for c in full if "interval_" in c]
    pd.testing.assert_frame_equal(full[interval_columns], mean[interval_columns])
    assert full.set_index("action_id").loc["e", "baseline_window_excess_c"] == 1
    assert full.set_index("action_id").loc["e", "action_precondition_excess_c"] == 1
    assert full.set_index("action_id").loc["d", "action_recovery_excess_c"] == 1
    assert data["choices"].loc[data["choices"].status.eq("selected"), "selected_action"].tolist() == ["a", "a"]
    assert module.select_choices(mean, data["budgets"], data["protocol"]).selected_action.dropna().tolist() == ["e", "e"]


def test_pre_is_skipped_only_by_reference_occupancy_and_event_recovery_still_require_values():
    data, _, _ = fixture_data()
    row = data["scores"].iloc[0].to_dict()
    assert row["reference_precondition_weighted_profile_hours"] == 0
    assert module.assess_service(row)["service_qualified"]
    row["reference_event_population_mean_mismatch_c"] = np.nan
    assert not module.assess_service(row)["service_qualified"]
    row = data["scores"].iloc[0].to_dict()
    row["action_recovery_fixed_tail_mean_mismatch_c"] = np.nan
    assert not module.assess_service(row)["service_qualified"]


def test_kwh_gate_strict_positive_net_and_original_exact_power_order():
    data, _, _ = fixture_data()
    rows = []
    for name in "abcde":
        row = data["scores"].iloc[0].to_dict()
        row.update(action_id=name, event_facility_electricity_reduction_kwh=1.5e-9,
            event_facility_mean_power_reduction_kw=.75e-9,
            precondition_event_recovery_facility_net_reduction_kwh=1e-12)
        rows.append(module.assess_service(row))
    result = module.select_choices(pd.DataFrame(rows), data["budgets"], data["protocol"])
    assert result.selected_action.dropna().eq("a").all()
    assert result.positive_net_selected_action.dropna().eq("a").all()
    assert result.event_power_reduction_kw.dropna().lt(1e-9).all()
    rows[1][module.POWER] += 1e-15
    result = module.select_choices(pd.DataFrame(rows), data["budgets"], data["protocol"])
    assert result.selected_action.dropna().eq("b").all()
    for row in rows:
        row[module.NET] = 0
    result = module.select_choices(pd.DataFrame(rows), data["budgets"], data["protocol"])
    assert result.selected_action.dropna().eq("b").all() and result.positive_net_selected_action.isna().all()


def test_service_tolerance_preserves_original_addition_at_floating_point_boundary():
    data, _, _ = fixture_data()
    row = data["scores"].iloc[0].to_dict()
    for metric in module.METRICS:
        row["reference_event_" + metric] = 1.0
        row["action_event_" + metric] = 1.0 + 1e-9
    assert module.assess_service(row)["action_event_service_qualified"]
    row["action_event_population_mean_mismatch_c"] = np.nextafter(1.0 + 1e-9, np.inf)
    assert not module.assess_service(row)["action_event_service_qualified"]


def test_positive_net_policy_maximizes_power_instead_of_net_energy():
    data, _, _ = fixture_data()
    rows = []
    for name, power, net in [("a", 2., 1.), ("b", 3., -1.), ("c", 1., 10.), ("d", .5, .1), ("e", .1, .1)]:
        row = data["scores"].iloc[0].to_dict()
        row.update(action_id=name, event_facility_mean_power_reduction_kw=power,
                   precondition_event_recovery_facility_net_reduction_kwh=net)
        rows.append(row)
    choices = module.select_choices(pd.DataFrame(rows), data["budgets"], data["protocol"])
    assert choices.selected_action.dropna().eq("b").all()
    assert choices.positive_net_selected_action.dropna().eq("a").all()
    assert choices.positive_net_full_window_kwh.dropna().eq(1).all()


def test_unilateral_zero_unknown_and_common_baseline_subset_have_separate_denominators():
    data, context, scorer = fixture_data()
    full, full_choices = data["scores"], data["choices"]
    mean = module.mean_actions(full, data, context, scorer)
    mean_choices = module.select_choices(mean, data["budgets"], data["protocol"])
    action, pairs = module.compare_representations(full, mean, full_choices, mean_choices)
    assert action.service_qualified_mean_only.sum() == 4
    assert pairs.normal_eligible_engineering_resolved_support.sum() == 4
    assert not pairs.common_baseline_window_support.any()
    assert pairs.loc[pairs.normal_energy_reference_factor.eq(.95), "full_available_power_kw"].isna().all()
    assert pairs.mean_selected_fails_full_review.dropna().all()
    # Keep full qualification, but no full-approved action has positive event energy.
    changed = full.copy()
    changed["qualified_positive_event_response"] = False
    full_zero = module.select_choices(changed, data["budgets"], data["protocol"])
    _, zero_pairs = module.compare_representations(changed, mean, full_zero, mean_choices)
    supported = zero_pairs[zero_pairs.normal_eligible_engineering_resolved_support]
    assert supported.full_available_power_kw.eq(0).all()
    assert supported.mean_minus_full_available_power_kw.gt(0).all()
    empty = module.summarize(pairs, ["control", "normal_energy_reference_factor", "policy"])
    assert empty.loc[empty.support.eq("common_baseline_windows"), "full_available_power_mean_kw"].isna().all()


def test_all_engineering_unavailable_retains_explicit_empty_choice_schema():
    data, context, scorer = fixture_data()
    data["completed"]["final_engineering_qualified"] = False
    full = module.replay_actions(data, lambda row: pytest.fail("Unavailable action must not load"), scorer)
    choices = module.select_choices(full, data["budgets"], data["protocol"])
    assert choices.columns.tolist() == module.CHOICE_COLUMNS
    assert choices.loc[choices.normal_energy_reference_factor.ne(.95), "status"].eq("unresolved_engineering_operation").all()
    mean = module.mean_actions(full, data, lambda row: pytest.fail("Unavailable action must not load"), scorer)
    _, pairs = module.compare_representations(full, mean, choices, choices)
    assert pairs.full_available_power_kw.isna().all()


def test_annual_infeasibility_without_action_rows_remains_unknown_not_zero():
    data, _, _ = fixture_data()
    scores = data["scores"].iloc[:0]
    budgets = data["budgets"].copy()
    budgets["status"] = "infeasible_normal_energy_budget"
    choices = module.select_choices(scores, budgets, data["protocol"])
    _, pairs = module.compare_representations(scores, scores, choices, choices)
    assert len(pairs) == 6
    assert not pairs.normal_eligible_engineering_resolved_support.any()
    assert pairs.full_available_power_kw.isna().all()
    assert pairs.mean_available_power_kw.isna().all()


def test_fixed_tail_wrapper_is_restored_even_when_explicit_scoring_raises(monkeypatch):
    data, context, scorer = fixture_data()
    original = scorer.fixed_tail_profile_ids

    def fail(baseline, action, reference, profiles, event, **kwargs):
        assert scorer.fixed_tail_profile_ids(reference, profiles) == {"1", "4"}
        raise ValueError("synthetic score failure")

    monkeypatch.setattr(scorer, "score_event", fail)
    with pytest.raises(ValueError, match="synthetic score failure"):
        module.replay_actions(data, context, scorer)
    assert scorer.fixed_tail_profile_ids is original


def bindings(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "EXPECTED_ACTIONS", 5)
    monkeypatch.setattr(module, "EXPECTED_CELLS", 3)
    monkeypatch.setattr(module, "source_hashes", lambda: {"synthetic": "fixed"})
    (tmp_path / "input").write_text("frozen")
    return {"input": module.digest(tmp_path / "input")}, module.source_hashes()


@pytest.mark.parametrize("mismatch", ["score", "primary", "positive_net"])
def test_full_gate_failure_never_computes_mean_or_writes_pass_receipt(tmp_path, monkeypatch, mismatch):
    data, context, scorer = fixture_data()
    inputs, sources = bindings(tmp_path, monkeypatch)
    if mismatch == "score":
        data["scores"].loc[0, module.POWER] += .01
    else:
        name = "selected_action" if mismatch == "primary" else "positive_net_selected_action"
        data["choices"].loc[data["choices"].status.eq("selected"), name] = "incorrect"
    monkeypatch.setattr(module, "mean_actions", lambda *args: pytest.fail("Mean stage ran before full gate"))
    out = tmp_path / "review"
    with pytest.raises(ValueError, match="Full replay differs"):
        module.review(out, data, context, scorer, inputs, sources)
    assert not (out / "FULL_REPLAY_RECEIPT.json").exists()
    with pytest.raises(FileExistsError, match="Preserve"):
        module.verified_cache(out)


def test_receipt_precedes_mean_and_completed_cache_binds_outputs_and_sources(tmp_path, monkeypatch):
    data, context, scorer = fixture_data()
    inputs, sources = bindings(tmp_path, monkeypatch)
    original = module.mean_actions
    out = tmp_path / "review"

    def mean_after_receipt(*args):
        receipt = json.loads((out / "FULL_REPLAY_RECEIPT.json").read_text())
        assert receipt["status"] == "completed" and receipt["zone_mean_scores_computed"] == 0
        assert receipt["budget_event_cells"] == 3 and receipt["policy_cell_agreements"] == 6
        return original(*args)

    monkeypatch.setattr(module, "mean_actions", mean_after_receipt)
    module.review(out, data, context, scorer, inputs, sources)
    assert module.verified_cache(out)
    assert len(pd.read_csv(out / "PAIRED_EVENT_INFORMATION_REVIEW.csv")) == 6
    monkeypatch.setattr(module, "source_hashes", lambda: {"synthetic": "changed"})
    with pytest.raises(ValueError, match="sources changed"):
        module.verified_cache(out)
    monkeypatch.setattr(module, "source_hashes", lambda: sources)
    (tmp_path / "input").write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        module.verified_cache(out)
    (tmp_path / "input").write_text("frozen")
    (out / "MEAN_REVIEW_CHOICES.csv").write_text("changed")
    with pytest.raises(ValueError, match="output changed"):
        module.verified_cache(out)


def test_mean_can_also_reject_full_service_and_does_not_use_interval_limits():
    data, _, _ = fixture_data()
    full = data["scores"].copy()
    mean = full.copy()
    mean.loc[0, "action_event_fixed_tail_mean_mismatch_c"] = 99
    mean.loc[0] = pd.Series(module.assess_service(mean.loc[0].to_dict()))
    full.loc[0, "action_event_interval_excess_c"] = 999
    assert module.assess_service(full.loc[0].to_dict())["service_qualified"]
    choices = module.select_choices(full, data["budgets"], data["protocol"])
    action, _ = module.compare_representations(full, mean, choices,
        module.select_choices(mean, data["budgets"], data["protocol"]))
    assert action.service_qualified_full_only.iloc[0] == 1


def test_cache_rejects_a_different_current_raw_branch_even_when_old_inputs_are_intact(tmp_path, monkeypatch):
    data, context, scorer = fixture_data()
    inputs, sources = bindings(tmp_path, monkeypatch)
    primary = {"profiles": {"path": "input", "sha256": inputs["input"]}}
    out = tmp_path / "review"
    module.review(out, data, context, scorer, inputs, sources, primary)
    assert module.verified_cache(out, primary)
    (tmp_path / "other_raw_branch").write_text("frozen")
    other = {"profiles": {"path": "other_raw_branch", "sha256": inputs["input"]}}
    with pytest.raises(ValueError, match="current primary inputs differ"):
        module.verified_cache(out, other)


def test_full_replay_matches_serialized_original_timestamps_flags_and_empty_values():
    from io import StringIO

    data, context, scorer = fixture_data()
    expected_scores = pd.read_csv(StringIO(data["scores"].to_csv(index=False)))
    expected_choices = pd.read_csv(StringIO(data["choices"].to_csv(index=False)))
    actual = module.replay_actions(data, context, scorer)
    assert module.verify_table(actual, expected_scores, module.ACTION_KEYS) < 1e-12
    choices = module.select_choices(actual, data["budgets"], data["protocol"])
    assert module.verify_table(choices, expected_choices, module.CELL_KEYS) < 1e-12


@pytest.mark.parametrize("changed", [None, "normal_state_reference_sha256", "input_idf_sha256", "hourly_sha256"])
def test_actual_completion_schema_binds_the_normal_through_qa_not_an_absent_normal_id(changed):
    from types import SimpleNamespace

    qa = {"normal_state_reference_sha256": "normal-states", "input_idf_sha256": "action-input",
          "hourly_sha256": "action-hourly", "weather_sha256": "weather"}
    if changed:
        qa[changed] = "different"
    done = pd.Series({"run_id": "run", "qa": json.dumps(qa), "hourly_sha256": "action-hourly",
                      "history_check": json.dumps({"history_matches": True})})
    normal = pd.Series({"normal_replay_matches_original": True, "states_sha256": "normal-states"})
    plan = SimpleNamespace(input_sha256="action-input")
    assert "normal_id" not in done.index
    if changed:
        with pytest.raises(ValueError, match="planned input and normal state reference"):
            module.validate_context_binding(plan, done, normal)
    else:
        assert module.validate_context_binding(plan, done, normal) == qa
