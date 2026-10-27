from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.global_extension import global_information_compression as module
from src.analysis.global_extension.global_information_compression import (
    calibrate_double_holdout_margins,
    five_bin_profiles,
    score_five_bin_candidates,
    select_information_decisions,
    summarise_decisions,
    validate_support_tables,
)


def test_five_bin_profiles_are_deterministic_and_count_preserving():
    profiles = pd.DataFrame({
        "profile_id": [f"p{i:02d}" for i in range(11)],
        "zone_id": ["z"] * 11,
        "centre_c": [27, 20, 26, 21, 25, 22, 24, 23, 28, 19, 29],
        "lower_c": [18.0] * 11,
        "upper_c": [30.0] * 11,
    })
    first, bins = five_bin_profiles(profiles, ["p00", "p10"])
    second, repeated = five_bin_profiles(profiles.sample(frac=1, random_state=7), ["p10", "p00"])
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(bins, repeated)
    assert len(bins) == 5
    assert bins.profile_count.sum() == 11
    assert bins.fixed_tail_count.sum() == 2


def test_margins_exclude_both_heldout_dimensions():
    rows = []
    for climate in ("a", "b", "c"):
        for composition in ("x", "y", "z"):
            error = 9.0 if (climate, composition) == ("a", "x") else 0.2
            rows.append({
                "climate": climate,
                "composition": composition,
                "mean_reference_mismatch_c": 1.0 + error,
                "fixed_tail_reference_mismatch_c": 2.0 + error,
                "five_bin_mean_mismatch_c": 1.0,
                "five_bin_tail_mismatch_c": 2.0,
            })
    candidates = pd.DataFrame(rows)
    margins = calibrate_double_holdout_margins(candidates, quantile=1.0)
    held = margins.loc[
        margins.climate.eq("a") & margins.composition.eq("x")
    ].iloc[0]
    assert np.isclose(held.mean_margin_c, 0.2)
    assert held.training_climates == 2
    assert held.training_compositions == 2


def test_reused_hourly_path_is_rescored_for_each_demand_composition(
    tmp_path, monkeypatch
):
    path = tmp_path / "shared.npz"
    metrics = pd.DataFrame([
        {
            "climate": "c", "composition": composition,
            "layout": "random_mixing", "permission": "native_zone",
            "run_id": composition, "hourly_path": str(path),
            "hourly_sha256": "a" * 64, "engineering_pass": True,
        }
        for composition in ("cool", "warm")
    ])
    profiles = pd.DataFrame([
        {
            "climate": "c", "composition": composition,
            "layout": "random_mixing", "profile_id": f"{composition}-p{index:03d}",
            "zone_id": "z", "centre_c": centre, "lower_c": 20.0,
            "upper_c": 30.0,
        }
        for composition, centre in (("cool", 22.0), ("warm", 28.0))
        for index in range(267)
    ])
    tails = profiles.groupby(["climate", "composition", "layout"], sort=True).head(27)[
        ["climate", "composition", "layout", "profile_id"]
    ]
    calls = []
    monkeypatch.setattr(module, "verify_hourly_hash", lambda *args: "a" * 64)
    monkeypatch.setattr(module, "load_hourly", lambda *args: {})

    def fake_score(_data, group, _tail):
        calls.append(float(group.centre_c.mean()))
        return {"summary": {
            "mean_reference_mismatch_c": calls[-1],
            "fixed_tail_reference_mismatch_c": calls[-1],
        }}

    monkeypatch.setattr(module, "score_trajectory", fake_score)
    scored, _ = score_five_bin_candidates(
        metrics, profiles, tails, metrics_source=tmp_path / "metrics.csv"
    )
    assert calls == [22.0, 28.0]
    assert scored.five_bin_mean_mismatch_c.tolist() == [22.0, 28.0]


def test_reused_hourly_path_requires_consistent_declared_hash(tmp_path, monkeypatch):
    path = tmp_path / "shared.npz"
    metrics = pd.DataFrame([
        {
            "climate": "c", "composition": "x", "layout": "random_mixing",
            "permission": "native_zone", "run_id": run_id,
            "hourly_path": str(path), "hourly_sha256": declared,
            "engineering_pass": True,
        }
        for run_id, declared in (("one", "a" * 64), ("two", "b" * 64))
    ])
    profiles = pd.DataFrame([{
        "climate": "c", "composition": "x", "layout": "random_mixing",
        "profile_id": f"p{index:03d}", "zone_id": "z", "centre_c": 25.0,
        "lower_c": 20.0, "upper_c": 30.0,
    } for index in range(267)])
    tails = profiles.head(27)[
        ["climate", "composition", "layout", "profile_id"]
    ].copy()
    monkeypatch.setattr(module, "verify_hourly_hash", lambda *args: "a" * 64)
    monkeypatch.setattr(module, "load_hourly", lambda *args: {})
    monkeypatch.setattr(module, "score_trajectory", lambda *args: {"summary": {
        "mean_reference_mismatch_c": 1.0,
        "fixed_tail_reference_mismatch_c": 1.0,
    }})
    with pytest.raises(ValueError, match="different SHA-256"):
        score_five_bin_candidates(
            metrics, profiles, tails, metrics_source=tmp_path / "metrics.csv"
        )


def test_support_tables_must_cover_the_same_profile_cells():
    profiles = pd.DataFrame([{
        "climate": "c", "composition": "x", "layout": "random_mixing",
        "profile_id": f"p{index:03d}",
    } for index in range(267)])
    anchors = pd.DataFrame([{
        "climate": "c", "composition": "x", "layout": "random_mixing",
        "anchor_mean_mismatch_c": 1.0, "anchor_tail_mismatch_c": 1.0,
    }])
    tails = profiles.head(27).copy()
    validate_support_tables(profiles, anchors, tails)
    anchors.loc[0, "climate"] = "old-context"
    with pytest.raises(ValueError, match="do not match"):
        validate_support_tables(profiles, anchors, tails)


def _candidate(plan: str, energy: float, full: float, mean: float, five: float) -> dict:
    return {
        "climate": "a",
        "composition": "x",
        "layout": "random_mixing",
        "permission": "native_zone",
        "physical_plan_id": plan,
        "run_id": plan,
        "control": "uniform_average",
        "offset_c": 0.0,
        "facility_purchased_kwh": energy,
        "mean_reference_mismatch_c": full,
        "fixed_tail_reference_mismatch_c": full,
        "zone_mean_mean_reference_mismatch_c": mean,
        "zone_mean_fixed_tail_reference_mismatch_c": mean,
        "five_bin_mean_mismatch_c": five,
        "five_bin_tail_mismatch_c": five,
        "physical_alias_count": 1,
    }


def test_decision_audit_identifies_false_acceptance_and_protection():
    candidates = pd.DataFrame([
        _candidate("cheap", 90.0, 1.2, 0.8, 1.0),
        _candidate("safe", 100.0, 0.9, 0.9, 0.8),
    ])
    anchors = pd.DataFrame([{
        "climate": "a", "composition": "x", "layout": "random_mixing",
        "anchor_mean_mismatch_c": 1.0, "anchor_tail_mismatch_c": 1.0,
    }])
    margins = pd.DataFrame([{
        "climate": "a", "composition": "x",
        "mean_margin_c": 0.2, "tail_margin_c": 0.2,
    }])
    decisions = select_information_decisions(
        candidates, anchors, margins, fractions=(1.0,)
    )
    zone = decisions.loc[decisions.representation.eq("zone mean")].iloc[0]
    raw = decisions.loc[decisions.representation.eq("five bins")].iloc[0]
    protected = decisions.loc[
        decisions.representation.eq("protected five bins")
    ].iloc[0]
    assert zone.selected_plan_false_accept
    assert raw.selected_plan_false_accept
    assert protected.selected_plan_id == "safe"
    assert protected.selected_plan_passes_full_review


def test_summary_uses_only_fraction_one_full_resolvable_headline():
    rows = []
    for climate in range(9):
        for composition in range(6):
            for representation in (
                "full profiles", "zone mean", "five bins", "protected five bins"
            ):
                rows.append({
                    "climate": f"c{climate}",
                    "composition": f"p{composition}",
                    "service_fraction": 1.0,
                    "representation": representation,
                    "full_resolvable": climate != 0,
                    "selection_status": "selected",
                    "selected_plan_false_accept": False,
                        "selected_plan_passes_full_review": True,
                    "false_reject_no_selection": False,
                    "exact_selection_agreement": True,
                    "candidate_false_accept_count": 0,
                    "candidate_false_reject_count": 0,
                    "qualified_under_both_count": 1,
                    "full_qualified_plan_count": 1,
                    "safe_energy_penalty_percent": 0.0,
                })
    summary = summarise_decisions(pd.DataFrame(rows))
    assert summary["primary_full_resolvable_scenarios"] == 48
    assert summary["primary"]["five bins"]["scenarios"] == 48
    assert summary["primary"]["five bins"]["safe_decision_coverage"] == 1.0
    assert summary["information_values_per_scenario"]["five bins"]["total_values"] == 225
