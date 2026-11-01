import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from group_information_validation import group_centers, paired_bootstrap, compare


def test_group_centers_exclude_personal_test_information_and_development_self():
    frame = pd.DataFrame({"person": ["a", "b", "c", "d", "test"],
        "partition": ["development"] * 4 + ["final_test"], "source": ["s"] * 5,
        "building_id": ["building"] * 5, "raw_center": [20., 22., 24., 26., 50.]})
    values, fits = group_centers(frame)
    assert values.loc[0, "pooled_mean"] == 24
    assert values.loc[4, "building_mean"] == 23
    assert fits[fits.person.eq("a")].development_participants.eq(3).all()
    changed = frame.copy()
    changed.loc[4, "raw_center"] = -999
    doubled = pd.concat([changed, changed.iloc[[0, 0, 1]]], ignore_index=True)
    repeated, _ = group_centers(doubled)
    assert repeated.iloc[4].pooled_mean == 23
    assert repeated.iloc[4].building_median == 23


def test_group_support_fallback_uses_only_development_units():
    frame = pd.DataFrame({"person": ["a", "b", "c", "d", "e", "x", "y"],
        "partition": ["development"] * 5 + ["final_test"] * 2,
        "source": ["A", "A", "B", "B", "B", "A", "B"],
        "building_id": [str(i) for i in range(7)], "raw_center": [20., 22., 24., 26., 28., 0., 0.]})
    values, fits = group_centers(frame)
    assert values.loc[5, "building_mean"] == 24
    assert values.loc[6, "building_mean"] == 26
    assert set(fits[fits.person.eq("x")].support_level) == {"pooled"}
    assert fits[fits.person.eq("y") & fits.model.eq("building_mean")].support_level.iloc[0] == "study"


def test_paired_bootstrap_keeps_model_pairs_and_all_person_responses():
    frame = pd.DataFrame({"person": ["a", "a", "b", "b"], "source": ["s"] * 4,
        "building_id": ["building"] * 4, "truth": [0, 1, 0, 1]})
    predictions = np.column_stack([frame.truth, frame.truth, 1 - frame.truth])
    draws = paired_bootstrap(frame, predictions)
    assert draws.shape == (500, 3)
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert (draws[:, 0] == 1).all() and (draws[:, 2] == 0).all()


def test_final_response_labels_do_not_choose_group_models_widths_or_probabilities(tmp_path):
    rows = []
    for index, center in enumerate([22., 23., 24., 25., 22.5, 24.5]):
        for offset, truth in [(0, 1), (5, 0)]:
            rows.append({"person": str(index), "partition": "development" if index < 4 else "final_test",
                "source": "source", "building_id": "building", "record_id": str(index) + "-" + str(offset),
                "temperature": center + offset, "raw_center": center,
                "shrink_center": (center + 5 * 23.5) / 6, "source_prior_only": 23.5, "no_change": truth})
    frame = pd.DataFrame(rows)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    metrics, contrasts = compare(frame, "no_change", "timed_multi", first)
    changed = frame.copy()
    final = changed.partition.eq("final_test")
    changed.loc[final, "no_change"] = 1 - changed.loc[final, "no_change"]
    compare(changed, "no_change", "timed_multi", second)
    assert (first / "FROZEN_MODEL_SELECTIONS.json").read_bytes() == (second / "FROZEN_MODEL_SELECTIONS.json").read_bytes()
    assert (first / "GROUP_CENTER_FITS.csv").read_bytes() == (second / "GROUP_CENTER_FITS.csv").read_bytes()
    assert len(metrics) == 18 and len(contrasts) == 28
    assert all(item["records"] == 4 and item["participants"] == 2 for item in metrics)
