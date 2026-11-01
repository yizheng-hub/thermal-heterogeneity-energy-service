import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import dr_information_uncertainty as module


def test_no_action_zero_is_distinct_from_unavailable():
    assert np.all(module.frozen_power(False, True, None, np.nan) == 0)
    assert np.isnan(module.frozen_power(False, False, None, np.nan)).all()
    mask = np.arange(256) % 2 == 0
    np.testing.assert_array_equal(module.frozen_power(True, True, mask, 4), mask * 4)


def test_score_uses_original_addition_and_paired_reference():
    class Curve:
        def __init__(self, offset):
            self.offset = offset

        def score(self, centers, tail):
            return centers[:, 0] + self.offset, centers[:, 0] + self.offset

    centers = np.array([[1.], [2.]])
    passed, excess, _, _ = module.checked_scores(Curve(0), Curve(0), centers, [0], annual=True)
    assert passed.all() and np.all(excess == 0)
    passed, _, _, _ = module.checked_scores(Curve(2e-9), Curve(0), centers, [0], annual=True)
    assert not passed.any()
    with pytest.raises(ValueError, match="Nonfinite"):
        module.checked_scores(Curve(np.nan), Curve(0), centers, [0], annual=True)


def test_nominal_window_pass_does_not_hide_behind_other_failed_stages():
    saved = {"action_event_" + metric: 2. for metric in module.METRICS}
    saved.update({"reference_event_" + metric: 2. for metric in module.METRICS})
    assert module.expected_window_pass(saved, "action_event")
    saved["action_event_" + module.METRICS[1]] += 2e-9
    assert not module.expected_window_pass(saved, "action_event")


def test_primary_assets_bind_current_semantic_path_even_with_equal_content(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "CODE", tmp_path)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text("same data")
    second.write_text("same data")
    module.INPUTS.clear()
    a, b = module.primary_assets({"profiles": first}), module.primary_assets({"profiles": second})
    assert a["profiles"]["sha256"] == b["profiles"]["sha256"]
    assert a != b


def test_context_dedup_keeps_demand_identity_and_exact_window():
    pairs, plans = [], []
    for comp in [1, 2]:
        for control in ["building_search", "zone_summary"]:
            cell = dict(climate="C", composition_id=comp, control=control, event_id="E",
                normal_energy_reference_factor=1., full_selected_action="a", mean_selected_action="a")
            pairs.append(cell)
            plans.append({k: cell[k] for k in module.information.EVENT_KEYS} | dict(action_id="a",
                normal_id="N", run_id="R", event_json='{"precondition_hours":2}'))
    contexts, mapping = module.choice_inventory(pd.DataFrame(pairs), pd.DataFrame(plans))
    assert len(contexts) == 2 and len(mapping) == 8
    plans[-1]["event_json"] = '{"precondition_hours":0}'
    contexts, _ = module.choice_inventory(pd.DataFrame(pairs), pd.DataFrame(plans))
    assert len(contexts) == 3


def test_draws_keep_cross_climate_identity_pairing_and_source_specific_support(monkeypatch):
    monkeypatch.setattr(module, "person_keys", lambda ids: ["P1", "P2"])
    monkeypatch.setattr(module, "shared_normal", lambda ids, archive, study_shared: np.zeros((256, 2)))
    monkeypatch.setattr(module, "empirical_temporal_shifts", lambda ids, single_record=False: np.ones((256, 2)))
    g = pd.DataFrame(dict(centre_c=[22., 24.], profile_id=["a", "b"], composition=["Multinational A"] * 2))
    first, second = dict(module.demand_draws(g)), dict(module.demand_draws(g.copy()))
    assert len(first) == 22
    np.testing.assert_array_equal(first["sourceSD0.5_personSD0.5"], second["sourceSD0.5_personSD0.5"])
    g["composition"] = "China A"
    assert len(module.demand_draws(g)) == 18


def test_resolved_no_selection_keeps_zero_power_in_supported_summaries():
    cell = dict(climate="C", composition_id=1, control="building_search", event_id="E",
        normal_energy_reference_factor=1., normal_eligible_engineering_resolved_support=True,
        common_baseline_window_support=True, full_selected_action=np.nan, mean_selected_action=np.nan,
        full_available_power_kw=0., mean_available_power_kw=0.)
    mapping = pd.DataFrame(columns=module.information.CELL_KEYS + ["representation", "context_id"])
    summary = pd.DataFrame([dict(context_id="other", scenario="sourceSD0_personSD0", composition_id=1)])
    aggregate, draws = module.paired_summaries(pd.DataFrame([cell]), mapping, summary, {})
    assert len(draws) == 512
    assert aggregate.full_selection_full_checked_power_mean_kw.eq(0).all()
    assert aggregate.mean_selection_full_checked_power_mean_kw.eq(0).all()
    assert aggregate.supported_events.eq(1).all()
