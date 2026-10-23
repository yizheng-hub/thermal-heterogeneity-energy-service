import pandas as pd
from pandas.testing import assert_frame_equal

from thermal_heterogeneity.building.annual import compare_representations
from thermal_heterogeneity import level1


def test_false_accept_and_energy_correction_regression():
    rows = []
    for representation in ("compressed", "full"):
        rows.extend([{"case": 1, "representation": representation, "physical_action_id": "low", "site_kwh": 10.0, "mean_c": 0.5 if representation == "compressed" else 1.5, "fixed_tail_c": 0.5 if representation == "compressed" else 1.5, "mean_budget_c": 1.0, "fixed_tail_budget_c": 1.0}, {"case": 1, "representation": representation, "physical_action_id": "safe", "site_kwh": 12.0, "mean_c": 0.5, "fixed_tail_c": 0.5, "mean_budget_c": 1.0, "fixed_tail_budget_c": 1.0}])
    result, _ = compare_representations(pd.DataFrame(rows), decision_keys=["case"])
    row = result.iloc[0]
    assert row["candidate_false_accept_count"] == 1
    assert row["compressed_selected_false_accept"]
    assert row["energy_correction_kwh"] == 2.0


def test_local_update_draws_are_invariant_to_release_identifiers(tmp_path, monkeypatch):
    monkeypatch.setattr(level1, "LOCAL_REPETITIONS", 1)
    monkeypatch.setattr(level1, "LOCAL_BOOTSTRAPS", 2)
    rows = []
    for rank in range(60):
        rows.append(
            {
                "split": "city",
                "held_out_group_id": "anonymous_group_a",
                "sampling_group_rank": 0,
                "sampling_unit_rank": rank,
                "sampling_group_row_rank": rank,
                "release_record_id": f"record_{rank:02d}",
                "release_unit_id": f"unit_{rank:02d}",
                "observed_preference": ("warmer", "no change", "cooler")[rank % 3],
                "p_warmer": 0.2,
                "p_no_change": 0.5,
                "p_cooler": 0.3,
            }
        )
    plan = pd.DataFrame(
        {
            "split": "city",
            "sampling_group_rank": 0,
            "repetition": 0,
            "feedback_position": range(40),
            "sampling_unit_rank": range(40),
            "sampling_group_row_rank": range(40),
        }
    )

    def run(tag: str, reverse_ids: bool) -> pd.DataFrame:
        predictions = pd.DataFrame(rows)
        if reverse_ids:
            predictions["held_out_group_id"] = "anonymous_group_z"
            predictions["release_unit_id"] = list(reversed(predictions["release_unit_id"]))
            predictions["release_record_id"] = list(reversed(predictions["release_record_id"]))
        predictions.to_csv(tmp_path / f"predictions_{tag}.csv", index=False)
        plan.to_csv(tmp_path / f"plan_{tag}.csv", index=False)
        config = {
            "_data_root": tmp_path,
            "assets": {
                "regional_predictions": f"predictions_{tag}.csv",
                "local_update_sampling_plan": f"plan_{tag}.csv",
            },
        }
        summary, _ = level1.local_update_learning_curve(config, seed=20261021)
        return summary.drop(columns="split")

    assert_frame_equal(run("a", False), run("b", True), check_dtype=False)
