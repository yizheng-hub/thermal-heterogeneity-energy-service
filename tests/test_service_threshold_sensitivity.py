import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from service_threshold_sensitivity import (
    SCORE_COLUMNS, evaluate_thresholds, score_representations, summarise,
)


def test_paired_scores_keep_person_weights_and_reference_group_identity():
    data = dict(zone_ids=np.array(["A", "B"]),
                temperatures=np.array([[22., 24.], [26., 28.]]),
                people=np.array([[1., 4.], [3., 2.]]), occupied=np.array([True, True]))
    profiles = pd.DataFrame(dict(zone_id=["A", "A", "B"], centre_c=[20., 28., 24.]))
    # Individual occupied-time scores are 5, 3, 4/3; the fixed group is persons0/2.
    # Its equal-person average differs from both population weighting and reranking.
    expected = [2.4, 19 / 6, 1.6, 5 / 3]
    np.testing.assert_allclose(score_representations(data, profiles, [0, 2]), expected, atol=1e-12)
    np.testing.assert_array_equal(profiles.centre_c, [20, 28, 24])


def candidate(setting, run, energy, values):
    return dict(climate=setting, composition="A", physical_plan_id=run,
                run_id=run, control="shared", permission="building",
                facility_purchased_kwh=energy, **dict(zip(SCORE_COLUMNS, values)))


def limits(settings):
    return pd.DataFrame([dict(climate=s, composition="A", reference_population_c=1.,
                              reference_fixed_group_c=1.) for s in settings])


def test_four_feasibility_states_preserve_missing_comparisons_and_correct_denominator():
    scores = pd.DataFrame([
        candidate("both", "a", 100., [.8, .8, .2, .2]),
        candidate("both", "b", 110., [.4, .4, .3, .3]),
        candidate("zone", "a", 100., [.8, .8, .2, .2]),
        candidate("individual", "a", 100., [.2, .4, .1, .8]),
        candidate("neither", "a", 100., [2., 2., 1., 1.]),
    ])
    selections = evaluate_thresholds(scores, limits(scores.climate.unique()), [.5])
    result = selections.set_index("climate")
    assert result.loc["both", "feasibility"] == "both_feasible"
    assert result.loc["zone", "feasibility"] == "zone_average_only"
    assert result.loc["individual", "feasibility"] == "individual_only"
    assert result.loc["neither", "feasibility"] == "neither_feasible"
    assert result.loc["both", "additional_purchased_energy_kwh"] == 10
    assert result.loc["both", "additional_purchased_energy_pct"] == 10
    assert np.isclose(result.loc["zone", "zone_selected_individual_exceedance_c"], .3)
    assert result.loc[["zone", "individual", "neither"], "additional_purchased_energy_kwh"].isna().all()
    assert result.loc[["zone", "individual", "neither"], "choices_differ"].isna().all()
    summary = summarise(selections).iloc[0]
    assert summary.both_feasible == summary.zone_average_only == summary.individual_only == summary.neither_feasible == 1
    assert summary.different_choices_pct_of_both_feasible == 100


def test_no_common_feasible_setting_has_no_energy_difference_or_choice_percentage():
    scores = pd.DataFrame([candidate("A", "one", 100., [2., 2., .2, .2])])
    summary = summarise(evaluate_thresholds(scores, limits(["A"]), [.5])).iloc[0]
    assert summary.both_feasible == 0
    assert pd.isna(summary.different_choices_pct_of_both_feasible)
    assert pd.isna(summary.both_feasible_additional_purchased_energy_kwh_median)


def test_choice_uses_existing_tolerance_and_identifier_tie_rule_not_input_order():
    scores = pd.DataFrame([
        candidate("A", "z", 1_000_000., [.3] * 4),
        candidate("A", "a", 1_000_000.0005, [.3] * 4),
        candidate("A", "0", 1_000_000.002, [.3] * 4),
    ])
    for ordered in [scores, scores.iloc[::-1]]:
        result = evaluate_thresholds(ordered, limits(["A"]), [1.]).iloc[0]
        assert result.individual_run_id == result.zone_average_run_id == "a"
        assert result.choices_differ == False
def test_default_workflow_includes_thresholds_from_the_new_run():
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    runner = runpy.run_path(str(root / "reproduce_strengthening.py"))
    raw = root / "output" / "another_run"
    output = raw / "strengthening"
    commands = runner["phase_commands"](raw, output)["uncertainty"]
    names = [command[0] for command in commands]
    assert names.index("service_threshold_sensitivity.py") == names.index("aligned_grid_sensitivity.py") + 1
    command = commands[names.index("service_threshold_sensitivity.py")]
    arguments = dict(zip(command[1::2], command[2::2]))
    assert Path(arguments["--matrix-root"]) == raw / "matrix"
    assert Path(arguments["--results-root"]) == output / "results"
    assert Path(arguments["--memberships"]) == output / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv"
    assert Path(arguments["--trajectory-root"]) == raw
    assert Path(arguments["--output-dir"]) == output / "results/service_threshold_sensitivity"


def test_table_keeps_infeasible_comparisons_and_zero_exceedances_separate():
    from src.analysis.figure_tables import service_threshold_rows
    import pandas as pd
    import numpy as np
    frame = pd.DataFrame([
        [0.5, "zone_average_only", np.nan, "a", 2.0, np.nan, np.nan],
        [0.5, "neither_feasible", np.nan, np.nan, np.nan, np.nan, np.nan],
        [1.0, "both_feasible", True, "a", 0.2, 2000.0, 1.0],
        [1.0, "both_feasible", False, "b", 0.0, 0.0, 0.0],
        [1.0, "zone_average_only", np.nan, "c", 0.5, np.nan, np.nan],
    ], columns=["factor", "feasibility", "choices_differ", "zone_average_run_id",
                "zone_selected_individual_exceedance_c", "additional_purchased_energy_kwh",
                "additional_purchased_energy_pct"])
    rows = service_threshold_rows(frame)
    assert rows[0] == ["0.50", "0 / 1 / 1", "N/A", "1/1", "2.000–2.000", "N/A", "N/A"]
    assert rows[1] == ["1.00", "2 / 1 / 0", "1/2 (50.0)", "2/3", "0.200–0.500", "2.00–2.00", "1.00–1.00"]
