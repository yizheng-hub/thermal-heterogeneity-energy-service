"""Meaningful upstream input, grouping, privacy and numerical replay contracts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/utilities/generate_demand_inputs.py"
SPEC = importlib.util.spec_from_file_location("public_upstream_replay", SOURCE)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_external_scene_asset_is_locked_before_loading(tmp_path):
    asset = tmp_path / "external_scene.json"
    asset.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="scene_pool SHA-256 differs"):
        REPLAY.numeric_scenes(asset)


def test_released_numeric_scenes_need_no_question_text_or_dialogue():
    path = SOURCE.parents[2] / "data/metadata/numeric_scene_spec_v1.json"
    scenes = REPLAY.numeric_scenes(path, numeric_only=True)
    assert len(scenes) == 19
    assert {scene["scene_id"] for scene in scenes} == set(REPLAY.SCENE_LOGIC)
    for scene in scenes:
        assert set(scene) == {"scene_id", "pmv_range", "physical_parameters",
                              "evidence_strength", "evidence_influence_profile"}
        assert set(scene["physical_parameters"]) == {"air_temp", "mrt", "air_velocity", "relative_humidity"}
        assert set(scene["evidence_influence_profile"]) <= {"secondary_weight", "decay_tau"}


def test_released_numeric_scene_tampering_is_rejected(tmp_path):
    path = SOURCE.parents[2] / "data/metadata/numeric_scene_spec_v1.json"
    altered = tmp_path / "numeric_scene_spec.json"
    altered.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="numeric_scenes SHA-256 differs"):
        REPLAY.numeric_scenes(altered, numeric_only=True)


def test_cohort_groups_subjects_retains_air_fallback_and_exports_no_original_ids():
    measured = pd.DataFrame({
        "record_id": [101, 102, 103, 104, 105], "building_id": [91] * 5,
        "subject_id": [7, 7, np.nan, 9, 10], "met": [1.1, 1.2, 1.1, 1.4, 1.1],
        "thermal_sensation": [1, -1, 0, 0, np.nan], "top": [24., np.nan, 26., 25., 25.],
        "ta": [23., 25., 25., 25., 25.], "clo": [.6, np.nan, .8, .5, .5],
        "season": ["summer"] * 5,
    })
    metadata = pd.DataFrame({
        "building_id": [91], "building_type": [" Office "],
        "cooling_type": ["Air conditioned"], "publication": ["private_source_name"],
        "contributor": ["private_contributor"], "climate": ["warm"],
        "city": ["private_city"], "country": ["private_country"],
    })
    cohort, scope = REPLAY.build_cohort(measured, metadata)
    assert scope["records"] == 3 and scope["analysis_units"] == 2
    assert cohort.temperature_source.tolist() == ["operative", "air_fallback", "operative"]
    assert cohort.reference_g050_c.tolist() == [22., 27., 26.]
    assert cohort.unit_weight.tolist() == [.5, .5, 1.]
    assert np.isclose(cohort.clo_analysis.iloc[1], .7)
    assert cohort.clothing_imputation.iloc[1] == "source_season_median"
    public = REPLAY.anonymous_inputs(cohort)
    assert public.replay_unit.iloc[0] == public.replay_unit.iloc[1]
    assert public.replay_unit.iloc[0] != public.replay_unit.iloc[2]
    assert not {"record_id", "subject_id", "building_id", "source_group", "city", "country"} & set(public)
    assert "private_" not in public.to_csv(index=False)


def test_virtual_interval_replays_nineteen_updates_and_stays_in_executable_domain():
    # Author-generated numeric fixtures contain no original scenario text or asset values.
    scenes = [{"scene_id": key, "pmv_range": [-.2, .2],
               "physical_parameters": {"air_temp": 25., "mrt": 25.,
                                       "air_velocity": .1, "relative_humidity": 50.},
               "evidence_strength": 1., "evidence_influence_profile": {}}
              for key in REPLAY.SCENE_LOGIC]
    result = REPLAY.virtual_interval(24., .6, 1.1, scenes)
    assert result["rounds"] == 19
    assert result["virtual_model_derived"] is True
    assert 16. <= result["interval_lower_c"] < result["interval_upper_c"] <= 30.
    assert result["interval_centre_c"] == 24.
    low_pmv = REPLAY.pmv_fanger(24. - result["interval_half_width_c"],
                               24. - result["interval_half_width_c"], .1, 50., 1.1, .6)
    high_pmv = REPLAY.pmv_fanger(24. + result["interval_half_width_c"],
                                24. + result["interval_half_width_c"], .1, 50., 1.1, .6)
    assert high_pmv > low_pmv
    assert result == REPLAY.virtual_interval(24., .6, 1.1, scenes)
