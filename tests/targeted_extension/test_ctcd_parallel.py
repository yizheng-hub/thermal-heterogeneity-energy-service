"""Parallel CTCD design preserves draws, exact cached decisions and source physics."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/targeted_extension"))
sys.path.insert(0, str(ROOT / "src/utilities"))
import ctcd_parallel as parallel  # noqa: E402
from run_population_controls import retained_result, sha  # noqa: E402


def test_ordered_placement_preserves_exact_original_draw_and_capacities(tmp_path):
    contexts = {}
    for context in ("one", "two"):
        contexts[context] = {"zones": ["A", "B"], "profiles": context + ".npz"}
        np.savez(tmp_path / (context + ".npz"), layout=np.array(["random_mixing"] * 267),
                 composition_id=np.ones(267), zone_id=np.array(["A"] * 100 + ["B"] * 167))
    pool = pd.DataFrame({"profile_id": [f"p{i:04d}" for i in range(400)],
                         "centre_c": np.linspace(20, 29, 400), "lower_c": 19., "upper_c": 30.,
                         "controller_lower_c": 19., "controller_upper_c": 30.})
    original = parallel.assign_external_profiles(pool, tmp_path, contexts)
    profiles = parallel.parallel_profiles(pool, tmp_path, contexts, original)
    pd.testing.assert_frame_equal(profiles[profiles.layout.eq("random_mixing")].reset_index(drop=True), original)
    for (context, draw), group in profiles.groupby(["context", "composition_id"]):
        random = group[group.layout.eq("random_mixing")].set_index("profile_id")
        ordered = group[group.layout.eq("demand_ordered")].set_index("profile_id")
        pd.testing.assert_frame_equal(random.drop(columns="layout").drop(columns="zone_id").sort_index(),
                                      ordered.drop(columns="layout").drop(columns="zone_id").sort_index())
        assert ordered.groupby("zone_id").size().to_dict() == {"A": 100, "B": 167}
        assert ordered[ordered.zone_id.eq("A")].centre_c.max() <= ordered[ordered.zone_id.eq("B")].centre_c.min()
    damaged = original.copy()
    damaged.loc[0, "centre_c"] += .1
    with pytest.raises(AssertionError):
        parallel.parallel_profiles(pool, tmp_path, contexts, damaged)


def cached_job(tmp_path, passed):
    source = tmp_path / "new.idf"
    source.write_text("Material,Wall,MediumRough,0.1,0.2,800,900;")
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps({"engineering_pass": passed}))
    job = {"run_id": "new", "context": "city", "cached_run_id": "old",
           "cached_qa_path": str(qa), "cached_qa_sha256": sha(qa),
           "cached_engineering_status": "new_qualified" if passed else "new_rejected",
           "normalized_idf_sha256": parallel.signature(source.read_text()),
           "cached_normalized_idf_sha256": parallel.signature(source.read_text()),
           "cached_hourly_path": "", "cached_hourly_sha256": ""}
    return source, qa, job


def test_rejected_exact_input_without_hourly_stays_rejected(tmp_path):
    source, qa, job = cached_job(tmp_path, False)
    result = retained_result(job, tmp_path, tmp_path / "models", source)
    assert result["engineering_status"] == "cached_rejected"
    assert result["path_hourly"] == result["hourly_sha256"] == ""
    assert result["cache_reused"] is True
    qa.write_text(json.dumps({"engineering_pass": True}))
    with pytest.raises(ValueError, match="QA checksum"):
        retained_result(job, tmp_path, tmp_path / "models", source)


def test_qualified_cache_requires_unchanged_actual_hourly_data(tmp_path):
    source, _, job = cached_job(tmp_path, True)
    with pytest.raises(ValueError, match="actual hourly"):
        retained_result(job, tmp_path, tmp_path / "models", source)
    hourly = tmp_path / "hourly.npz"
    np.savez(hourly, temperatures=np.zeros((8760, 1)))
    job.update(cached_hourly_path=str(hourly), cached_hourly_sha256=sha(hourly))
    assert retained_result(job, tmp_path, tmp_path / "models", source)["engineering_status"] == "cached_qualified"
    np.savez(hourly, temperatures=np.ones((8760, 1)))
    with pytest.raises(ValueError, match="hourly checksum"):
        retained_result(job, tmp_path, tmp_path / "models", source)


def test_cached_rejection_cannot_be_relabelled_qualified(tmp_path):
    source, _, job = cached_job(tmp_path, False)
    job["cached_engineering_status"] = "new_qualified"
    with pytest.raises(ValueError, match="original engineering decision"):
        retained_result(job, tmp_path, tmp_path / "models", source)
    job["cached_engineering_status"] = "new_rejected"
    job["cached_normalized_idf_sha256"] = "another_model"
    with pytest.raises(ValueError, match="normalized identity"):
        retained_result(job, tmp_path, tmp_path / "models", source)


def test_parallel_replay_checks_original_hourly_identity_not_only_plan_shape(tmp_path):
    from replay_ctcd_parallel import verify_original_native

    original, expanded = tmp_path / "original", tmp_path / "expanded"
    (original / "engineering").mkdir(parents=True)
    expanded.mkdir()
    rows, profiles = [], []
    for context in range(4):
        for draw in range(3):
            for profile in range(267):
                profiles.append({"context": context, "composition_id": draw,
                                 "layout": "random_mixing", "profile_id": f"p{profile}"})
            for control in ("uniform_average", "personal_center", "ltpref_interval"):
                for variant in range(4):
                    identity = f"{context}-{draw}-{control}-{variant}"
                    rows.append({"context": context, "composition_id": draw,
                                 "layout": "random_mixing", "permission": "native_zone",
                                 "control": control, "variant": variant, "run_id": identity,
                                 "normalized_idf_sha256": identity, "physical_sha256": "physical",
                                 "weather_sha256": "weather", "hourly_sha256": identity,
                                 "qa_sha256": "qa", "engineering_status": "new_qualified"})
    old = pd.DataFrame(rows)
    old.to_csv(original / "engineering/ANNUAL_RUN_MANIFEST.csv", index=False)
    original_profiles = pd.DataFrame(profiles)
    original_profiles.to_csv(original / "engineering/PROFILE_ASSIGNMENTS.csv.gz", index=False)
    pd.concat([original_profiles, original_profiles.assign(layout="demand_ordered")]).to_csv(
        expanded / "PROFILE_ASSIGNMENTS.csv.gz", index=False)
    current = old.assign(engineering_status="cached_qualified")
    assert verify_original_native(current, expanded, original)["logical_candidates"] == 144
    current.loc[0, "hourly_sha256"] = "different_trajectory"
    with pytest.raises(ValueError, match="physical output or QA"):
        verify_original_native(current, expanded, original)


def test_parallel_contract_retains_24_layout_anchors(tmp_path):
    from replay_ctcd_external import verify_contract

    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    anchors, budgets, tails = [], [], []
    for context in range(4):
        for draw in range(3):
            for layout in ("random_mixing", "demand_ordered"):
                key = {"context": context, "composition_id": draw, "layout": layout}
                anchors.append(key | {"anchor_mean_c": 1., "anchor_fixed_tail_c": 2.})
                tails.extend(key | {"profile_id": f"p{i}"} for i in range(27))
                for fraction in (.5, .75, .9, 1.):
                    budgets.append(key | {"anchor_mean_c": 1., "anchor_fixed_tail_c": 2.,
                                          "service_fraction": fraction, "mean_budget_c": fraction,
                                          "fixed_tail_budget_c": 2 * fraction})
    pd.DataFrame(anchors).to_csv(after / "ANNUAL_ANCHORS.csv", index=False)
    pd.DataFrame(budgets).to_csv(before / "BASELINE_DECISION_CONTRACT.csv", index=False)
    pd.DataFrame(budgets).drop(columns=["anchor_mean_c", "anchor_fixed_tail_c"]).to_csv(
        after / "ANNUAL_SERVICE_SELECTIONS.csv", index=False)
    for directory in (before, after):
        pd.DataFrame(tails).to_csv(directory / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    assert verify_contract(after, before, expected_anchors=24)["fixed_tail_memberships"] == 648
