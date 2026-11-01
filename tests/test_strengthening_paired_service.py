import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from common import *
from service_distribution import ServiceDistribution, PairedWorseningHours
from analyse_global_results import score_trajectory


def test_source_groups_are_consistent_for_repeated_profile_representations(tmp_path, monkeypatch):
    import profile_identity as module
    import pytest

    path = tmp_path / "audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz"
    path.parent.mkdir()
    frame = pd.DataFrame({"profile_id": ["p1", "p2", "p3", "p1"],
                          "source_key": ["study1", "study1", "study2", "study1"]})
    frame.to_csv(path, index=False)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.source_map.cache_clear()
    try:
        assert module.source_keys(["p3", "p1", "p2", "p1"]) == ["study2", "study1", "study1", "study1"]
        frame.loc[3, "source_key"] = "conflicting_study"
        frame.to_csv(path, index=False)
        module.source_map.cache_clear()
        with pytest.raises(ValueError, match="conflicting source"):
            module.source_keys(["p1"])
    finally:
        module.source_map.cache_clear()


def test_study_shared_draws_preserve_pairing_without_adding_archive_variance(monkeypatch):
    import paired_uncertainty as module

    groups = {"p1": "study1", "p2": "study1", "p3": "study2"}
    monkeypatch.setattr(module, "source_keys", lambda ids: [groups[key] for key in ids])
    ids = ["p1", "p2", "p3"]
    draws = module.shared_normal(ids, "ASHRAE", study_shared=True)
    assert draws.shape == (module.DRAWS, 3)
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    assert not np.array_equal(draws[:, 0], draws[:, 2])
    np.testing.assert_array_equal(draws, module.shared_normal(ids, "different_archive", study_shared=True))
    np.testing.assert_array_equal(draws[:, [2, 0, 1]], module.shared_normal(["p3", "p1", "p2"], "ASHRAE", study_shared=True))
    archive = module.shared_normal(ids, "ASHRAE")
    np.testing.assert_array_equal(archive[:, 0], module.keyed_normal("source|ASHRAE"))


def test_single_record_temporal_transport_excludes_final_people_and_keeps_identity_pairing(monkeypatch):
    import paired_uncertainty as module

    pool = pd.DataFrame({"person": ["dev", "test", "dev"],
        "partition": ["development", "final_test", "development"],
        "center_model": ["raw_center", "raw_center", "shrink_center"],
        "source": ["s", "s", "s"], "signed_later_proxy_shift_c": [2.5, 999., -999.]})
    monkeypatch.setattr(module.pd, "read_csv", lambda path: pool.copy())
    monkeypatch.setattr(module, "person_keys", lambda ids: ["same_person", "same_person"])
    shifts = module.empirical_temporal_shifts(["slot_a", "slot_b"], single_record=True)
    np.testing.assert_array_equal(shifts, np.full((module.DRAWS, 2), 2.5))


def test_exact_occupancy_weighted_service_matches_hourly_scoring():
    rng = np.random.default_rng(42)
    data = {
        "zone_ids": np.array(["A", "B"]),
        "temperatures": rng.normal(24, 2, (48, 2)),
        "people": rng.uniform(0.1, 15, (48, 2)),
        "occupied": np.arange(48) % 4 != 0,
    }
    profiles = pd.DataFrame(
        {
            "zone_id": ["A", "A", "A", "B", "B"],
            "centre_c": [20, 23, 25, 27, 29],
            "lower_c": [19, 22, 24, 26, 28],
            "upper_c": [21, 24, 26, 28, 30],
        }
    )
    tail = np.array([1, 4])
    reference = score_trajectory(data, profiles, tail)
    fast = ServiceDistribution(data, profiles)
    population, least = fast.score(profiles.centre_c.to_numpy(), tail)
    assert np.isclose(population, reference["summary"]["mean_reference_mismatch_c"], atol=1e-12)
    assert np.isclose(least, reference["summary"]["fixed_tail_reference_mismatch_c"], atol=1e-12)
    assert np.allclose(
        fast.coverage(profiles.lower_c, profiles.upper_c),
        reference["profiles"]["modeled_interval_coverage"],
        atol=1e-12,
    )


def test_identical_candidate_and_reference_have_zero_paired_excess_under_correlated_draws():
    data = {
        "zone_ids": np.array(["A"]),
        "temperatures": np.array([[22.0], [26.0]]),
        "people": np.array([[1.0], [3.0]]),
        "occupied": np.array([True, True]),
    }
    p = pd.DataFrame({"zone_id": ["A", "A"]})
    model = ServiceDistribution(data, p)
    rng = np.random.default_rng(4)
    centers = rng.normal(24, 1, (256, 2))
    a, b = model.score(centers, [0])
    c, d = model.score(centers, [0])
    assert np.max(abs(a - c)) == 0 and np.max(abs(b - d)) == 0


def test_quarter_degree_commands_are_not_rounded_to_tenths():
    # A quarter step from a tenth-degree base must remain an exact half-tenth.
    base = 23.1
    assert round(base + 0.25, 8) == 23.35
    assert round(base - 0.75, 8) == 22.35


def test_paired_worsening_person_hours_match_direct_hourly_inequality():
    rng = np.random.default_rng(14)
    reference = {
        "zone_ids": np.array(["A", "B"]),
        "temperatures": rng.normal(24, 2, (37, 2)),
        "people": rng.uniform(0.1, 4, (37, 2)),
        "occupied": np.arange(37) % 3 != 0,
    }
    candidate = {**reference, "temperatures": rng.normal(24, 2, (37, 2))}
    p = pd.DataFrame({"zone_id": ["A", "A", "B"]})
    centers = rng.normal(24, 2, (11, 3))
    direct = np.zeros(11)
    for j, z in enumerate([0, 0, 1]):
        count = 2 if z == 0 else 1
        worse = (
            abs(candidate["temperatures"][:, z, None] - centers[:, j])
            > abs(reference["temperatures"][:, z, None] - centers[:, j]) + 1e-9
        )
        direct += (
            worse * (reference["people"][:, z] * reference["occupied"] / count)[:, None]
        ).sum(axis=0)
    assert np.allclose(
        PairedWorseningHours(candidate, reference, p).person_hours(centers), direct, atol=1e-10
    )
    assert np.all(PairedWorseningHours(reference, reference, p).person_hours(centers) == 0)


def test_selection_uses_declared_energy_tolerance_and_run_identifier_tie_rule():
    from paired_uncertainty import choose

    energy = np.array([1000.0, 1000.0000005, 1010.0])
    eligible = np.array([[True, True, True], [False, False, False], [True, False, True]])
    assert np.array_equal(choose(eligible, energy, np.array(["z", "a", "b"])), [1, -1, 0])


def test_dr_window_tail_keeps_original_person_hour_weights():
    from dr_selected_uncertainty import window_score
    from multi_event_dr_extension import service_metrics

    rng = np.random.default_rng(318)
    data = {
        "zone_ids": np.array(["A", "B"]),
        "temperatures": rng.normal(24, 2, (50, 2)),
        "people": rng.uniform(0.1, 10, (50, 2)),
        "occupied": np.arange(50) % 4 != 0,
    }
    profiles = pd.DataFrame(
        {
            "zone_id": ["A", "A", "B"],
            "profile_id": ["p1", "p2", "p3"],
            "centre_c": [22.0, 24.0, 27.0],
            "controller_lower_c": [20.0, 22.0, 25.0],
            "controller_upper_c": [24.0, 26.0, 29.0],
        }
    )
    mask = (np.arange(50) >= 11) & (np.arange(50) < 23)
    exact = service_metrics(data, profiles, mask.copy(), {"p1", "p3"})
    population, least = window_score(
        ServiceDistribution(data, profiles, mask), profiles.centre_c.to_numpy(), [0, 2]
    )
    assert np.isclose(population, exact["population_mean_mismatch_c"], atol=1e-12)
    assert np.isclose(least, exact["fixed_tail_mean_mismatch_c"], atol=1e-12)


def test_dr_baseline_libraries_include_identical_lower_scope_operations():
    from dr_capability import baseline_library

    frame = pd.DataFrame(
        {
            "permission": ["building", "building", "floor", "native_zone", "native_zone"],
            "control": [
                "uniform_command_search",
                "building_median",
                "personal_center",
                "personal_center",
                "personal_range",
            ],
            "run_id": ["a", "b", "c", "d", "e"],
            "physical_plan_id": ["1", "2", "3", "4", "5"],
        }
    )
    building = set(baseline_library(frame, "building_search").physical_plan_id)
    summary = set(baseline_library(frame, "zone_summary").physical_plan_id)
    full = set(baseline_library(frame, "zone_full").physical_plan_id)
    assert building == {"1", "2"}
    assert summary == building
    assert building < full and full == {"1", "2", "3", "4", "5"}


def test_bit_exact_state_deltas_survive_moving_the_whole_run_folder(tmp_path):
    from lossless_history_storage import save_states_against, load_states

    source = tmp_path / "original"
    source.mkdir()
    actions = source / "actions"
    actions.mkdir()
    base = source / "normal.npz"
    delta = actions / "response.npz"
    keys = np.array(["air", "surface"])
    normal = np.array([[0.0, -0.0], [23.0, 24.0]])
    response = {"keys": keys, "values": np.array([[-0.0, 0.0], [23.1, 24.3]])}
    np.savez_compressed(base, keys=keys, values=normal)
    save_states_against(delta, response, base)
    moved = tmp_path / "relocated"
    source.rename(moved)
    restored = load_states(moved / "actions/response.npz")
    assert np.array_equal(restored["values"].view(np.uint64), response["values"].view(np.uint64))
