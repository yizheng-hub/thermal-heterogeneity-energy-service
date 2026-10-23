import numpy as np
import pandas as pd
import pytest

from thermal_heterogeneity.implementation.local_update import crps_ordinal, local_intercept_update
from thermal_heterogeneity.implementation.phase_map import classify_cell, quantile_approximation
from thermal_heterogeneity.implementation.protected_quantiles import calibrate_margins
from thermal_heterogeneity.level1 import _preference_score, validate_frozen_sensitivity


def test_phase_classification_keeps_unresolved_separate():
    assert (
        classify_cell(mean_available=False, full_available=False, selection_matches=False)
        == "finite_library_unresolved"
    )
    assert (
        classify_cell(mean_available=True, full_available=True, selection_matches=True)
        == "mean_safe"
    )


def test_rank_bin_means_retain_unequal_population_counts_and_mean():
    frame = pd.DataFrame({"value": [3.0, 100.0, 1.0, 7.0, 2.0, 9.0, 8.0], "weight": [1.0] * 7})
    result = quantile_approximation(frame, value="value", bins=3)
    assert result["value"].to_numpy() == pytest.approx([2.0, 7.5, 54.5])
    assert result["weight"].to_numpy() == pytest.approx([3 / 7, 2 / 7, 2 / 7])
    assert result["quantile_bin"].tolist() == [1, 2, 3]
    assert np.average(result["value"], weights=result["weight"]) == pytest.approx(
        frame["value"].mean()
    )


def test_rank_bin_means_preserve_original_weighted_mean():
    frame = pd.DataFrame(
        {"value": [10.0, 1.0, 4.0, 20.0, 2.0], "weight": [1.0, 2.0, 5.0, 3.0, 4.0]}
    )
    result = quantile_approximation(frame, value="value", bins=2)
    assert result["value"].to_numpy() == pytest.approx([30 / 11, 70 / 4])
    assert result["weight"].to_numpy() == pytest.approx([11 / 15, 4 / 15])
    assert np.average(result["value"], weights=result["weight"]) == pytest.approx(
        np.average(frame["value"], weights=frame["weight"])
    )


def test_protected_margin_uses_higher_empirical_quantile():
    frame = pd.DataFrame(
        {"am": [0.0, 0.0, 0.0], "at": [0.0, 0.0, 0.0], "fm": [0.0, 1.0, 2.0], "ft": [1.0, 2.0, 3.0]}
    )
    assert calibrate_margins(
        frame,
        approximate_mean="am",
        approximate_tail="at",
        full_mean="fm",
        full_tail="ft",
        quantile=0.5,
    ) == (1.0, 2.0)


def test_local_update_is_normalized():
    probability = np.array([[0.2, 0.6, 0.2], [0.3, 0.4, 0.3]])
    updated, posterior = local_intercept_update(np.array([0, 0, 1]), probability)
    assert np.allclose(updated.sum(axis=1), 1.0)
    assert np.isclose(posterior.sum(), 1.0)


def test_paper_rps_uses_normalized_dimensionless_scale():
    probability = np.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]])
    weights = np.array([0.2, 0.3, 0.5])
    observed = np.array(["warmer", "no change", "cooler"])
    # Hand-calculated row scores: (0.16+0.01)/2, (0.04+0.04)/2,
    # and (0.01+0.16)/2, followed by the declared weighted average.
    score = _preference_score(observed, probability, weights)["crps"]
    assert score == pytest.approx(0.0715)
    assert crps_ordinal(np.array([0, 1, 2]), probability, weights) == pytest.approx(2 * score)


@pytest.mark.parametrize("declared,valid", [([0.33, 0.5, 0.67], True), ([0.19, 0.5, 0.73], False)])
def test_sensitivity_declaration_matches_frozen_input(tmp_path, declared, valid):
    pd.DataFrame({"griffiths_coefficient": [0.5, 0.33, 0.67, 0.5]}).to_csv(
        tmp_path / "matches.csv", index=False
    )
    config = {
        "_data_root": tmp_path,
        "assets": {"three_definition_matches": "matches.csv"},
        "science": {"griffiths_coefficients": declared},
    }
    if valid:
        validate_frozen_sensitivity(config)
    else:
        with pytest.raises(ValueError, match="does not match frozen inputs"):
            validate_frozen_sensitivity(config)
