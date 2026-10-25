"""External populations must regenerate commands without changing physics."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/energy_service"))
sys.path.insert(0, str(ROOT / "src/analysis/targeted_extension"))
SPEC = importlib.util.spec_from_file_location(
    "ctcd_controls", ROOT / "src/analysis/targeted_extension/ctcd_controls.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_new_profile_information_changes_actual_commands(tmp_path):
    import json
    controller = tmp_path / "controller"
    controller.mkdir()
    config = {"city": {"zones": ["A", "B"], "floor_groups": {"A": "one", "B": "one"},
                       "floor_binding_equals_building": True, "probes": "controller/probes.npz"}}
    (controller / "CONTEXTS.json").write_text(json.dumps(config))
    np.savez(controller / "probes.npz", uniform_target_c=np.array([20., 25., 30.]),
             operative_c=np.broadcast_to(np.array([20., 25., 30.])[:, None, None], (3, 1, 2)),
             native_people=np.ones((1, 2)), purchased_site_kwh=np.array([1., 2., 3.]))
    profiles = pd.DataFrame({"composition_id": [1, 1], "layout": ["random_mixing"] * 2,
                             "zone_id": ["A", "B"], "centre_c": [30., 30.],
                             "controller_lower_c": [29., 29.], "controller_upper_c": [30., 30.]})
    warm = MODULE.generate(tmp_path, profile_tables={"city": profiles},
                           permissions=("native_zone",), expected_commands=12)
    cool = profiles.assign(centre_c=20., controller_lower_c=20., controller_upper_c=21.)
    cold = MODULE.generate(tmp_path, profile_tables={"city": cool},
                           permissions=("native_zone",), expected_commands=12)
    assert warm[0]["targets_c"] == {"A": 30., "B": 30.}
    assert cold[0]["targets_c"] == {"A": 20., "B": 20.}
    assert {row["permission"] for row in warm} == {"native_zone"}
    assert all(set(row["targets_c"].values()) == {25.} for row in warm if row["variant"] == 3)
    with pytest.raises(ValueError, match="1296"):
        MODULE.generate(tmp_path, profile_tables={"city": profiles})


def test_generated_idf_moves_both_platforms_and_preserves_physics():
    source = ("ZoneControl:Thermostat,T,Z,Type,ThermostatSetpoint:DualSetpoint,D;"
              "ThermostatSetpoint:DualSetpoint,D,H,C;"
              "Schedule:Compact,H,Temperature,Through:12/31,For:Weekdays,Until:24:00,23.5;"
              "Schedule:Compact,C,Temperature,Through:12/31,For:Weekdays,Until:24:00,26.5;"
              "Material,Wall,MediumRough,0.1,0.2,800,900;")
    changed = MODULE.annual_input(source, {"Z": 27.}, 1.5)
    assert "25.5" in changed and "28.5" in changed
    assert MODULE.signature(source) != MODULE.signature(changed)
    assert MODULE.physical_hash(source) == MODULE.physical_hash(changed)
    assert MODULE.physical_hash(changed.replace("0.1,0.2,800", "0.01,0.2,800")) != MODULE.physical_hash(source)


def test_draw_sampling_keeps_same_people_across_contexts_and_native_capacity(tmp_path):
    config = {}
    for city in ("one", "two"):
        config[city] = {"zones": ["A", "B"], "profiles": city + ".npz"}
        np.savez(tmp_path / (city + ".npz"), layout=np.array(["random_mixing"] * 267),
                 composition_id=np.ones(267), zone_id=np.array(["A"] * 100 + ["B"] * 167))
    pool = pd.DataFrame({"profile_id": ["p" + str(i) for i in range(300)],
                         "centre_c": np.linspace(20, 29, 300), "lower_c": 19., "upper_c": 30.,
                         "controller_lower_c": 19., "controller_upper_c": 30.})
    first = MODULE.assign_external_profiles(pool, tmp_path, config)
    assert first.equals(MODULE.assign_external_profiles(pool, tmp_path, config))
    MODULE.validate_profiles(first, tmp_path, config)
    damaged = first.copy()
    damaged.loc[0, "zone_id"] = "B"
    with pytest.raises(ValueError, match="capacities"):
        MODULE.validate_profiles(damaged, tmp_path, config)


def test_summary_retains_unmatched_denominators_and_increased_energy(tmp_path):
    import replay_ctcd_external as replay
    annual = tmp_path / "annual"
    annual.mkdir()
    base = {"context": "Miami", "comparison_mode": "direct", "service_fraction": -1.,
            "alternative": "personal_center", "reference_run_id": "reference"}
    rows = []
    for draw in range(3):
        matched = draw == 0
        rows.append(base | {"composition_id": draw, "status": "matched" if matched else "unmatched",
                           "delta_site_pct": 4. if matched else np.nan,
                           "delta_interval_coverage_percentage_points": 8. if matched else np.nan,
                           "delta_mean_reference_mismatch_c": -.2 if matched else np.nan,
                           "delta_fixed_tail_reference_mismatch_c": -.3 if matched else np.nan,
                           "delta_annual_operational_co2_kg": 10. if matched else np.nan})
    pd.DataFrame(rows).to_csv(annual / "CONTROL_COMPARISONS.csv", index=False)
    pd.DataFrame([{"run_id": "reference", "annual_operational_co2_kg": 20.}]).to_csv(
        annual / "ANNUAL_ALL_CANDIDATES.csv", index=False)
    pd.DataFrame([{"context": "Miami", "control": "personal_center", "service_fraction": 1.,
                   "selection_status": "selected" if draw == 0 else "service_unavailable"}
                  for draw in range(3)]).to_csv(annual / "ANNUAL_SERVICE_SELECTIONS.csv", index=False)
    replay.summarize(annual, tmp_path)
    summary = pd.read_csv(tmp_path / "CTCD_PAIR_SUMMARY.csv").iloc[0]
    assert summary.declared_draws == 3 and summary.matched_draws == 1 and summary.unmatched_draws == 2
    assert summary.energy_change_pct_median == 4. and summary.energy_increase_draws == 1
    assert summary.carbon_change_pct_median == 50.
    points = pd.read_csv(tmp_path / "CTCD_MAIN_FIGURE_POINTS.csv")
    assert len(points) == 3 and points.status.eq("unmatched").sum() == 2


def test_contract_does_not_allow_tail_reselection_after_runs(tmp_path):
    import replay_ctcd_external as replay
    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    after.mkdir()
    anchors, contracts, tails, selections = [], [], [], []
    for context in range(4):
        for draw in range(3):
            key = {"context": str(context), "composition_id": draw, "layout": "random_mixing"}
            anchors.append(key | {"anchor_mean_c": 1., "anchor_fixed_tail_c": 2.})
            tails.extend(key | {"profile_id": f"p{i}"} for i in range(27))
            for fraction in (.5, .75, .9, 1.):
                budget = key | {"service_fraction": fraction, "mean_budget_c": fraction,
                                "fixed_tail_budget_c": 2 * fraction}
                contracts.append(budget | {"anchor_mean_c": 1., "anchor_fixed_tail_c": 2.})
                selections.append(budget)
    pd.DataFrame(anchors).to_csv(after / "ANNUAL_ANCHORS.csv", index=False)
    pd.DataFrame(contracts).to_csv(before / "BASELINE_DECISION_CONTRACT.csv", index=False)
    pd.DataFrame(selections).to_csv(after / "ANNUAL_SERVICE_SELECTIONS.csv", index=False)
    pd.DataFrame(tails).to_csv(before / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    pd.DataFrame(tails).to_csv(after / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    assert replay.verify_contract(after, before)["status"] == "PASS"
    tails[0]["profile_id"] = "new_tail_member"
    pd.DataFrame(tails).to_csv(after / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    with pytest.raises(ValueError, match="tail identities"):
        replay.verify_contract(after, before)
