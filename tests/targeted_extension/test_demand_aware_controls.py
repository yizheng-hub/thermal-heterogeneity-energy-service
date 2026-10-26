import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/analysis/targeted_extension/demand_aware_controls.py"
SPEC = importlib.util.spec_from_file_location("demand_aware_controls", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
profile_mismatch_curves = MODULE.profile_mismatch_curves
profile_interval_curves = MODULE.profile_interval_curves
choose_grid_value = MODULE.choose_grid_value


def test_profile_mismatch_uses_full_distribution_not_only_mean():
    predicted = np.array([[20.0, 20.0], [24.0, 24.0], [28.0, 28.0]])
    people = np.ones(2)
    symmetric = profile_mismatch_curves(predicted, people, np.array([22.0, 26.0]))
    skewed = profile_mismatch_curves(predicted, people, np.array([20.0, 28.0]))
    assert np.isclose(symmetric[:, 0].mean(), symmetric[:, 1].mean())
    assert not np.allclose(symmetric, skewed)


def test_profile_interval_loss_uses_each_profile_range():
    predicted = np.array([[20.0, 20.0], [24.0, 24.0], [28.0, 28.0]])
    people = np.ones(2)
    narrow = profile_interval_curves(
        predicted, people, np.array([23.0, 23.0]), np.array([25.0, 25.0])
    )
    wide = profile_interval_curves(
        predicted, people, np.array([20.0, 20.0]), np.array([28.0, 28.0])
    )
    assert narrow[1].sum() == 0
    assert np.all(wide == 0)


def test_grid_ties_use_neutral_midpoint():
    grid = np.array([20.0, 20.1, 20.2, 20.3, 20.4])
    loss = np.array([2.0, 1.0, 1.0, 1.0, 2.0])
    assert np.isclose(choose_grid_value(grid, loss), 20.2)
