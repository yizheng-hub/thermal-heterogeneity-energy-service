"""Check that range audits reject wrong reference linkage and retain lost matches."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "src/analysis/targeted_extension/range_assumptions.py"
SPEC = importlib.util.spec_from_file_location("range_assumptions", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_reference_linkage_is_not_silently_accepted():
    profiles = pd.DataFrame({"release_profile_id": ["x"], "reference_g050_c": [24.]})
    intervals = pd.DataFrame({"release_profile_id": ["x"], "interval_lower_c": [22.],
                              "interval_upper_c": [26.], "interval_centre_c": [25.],
                              "virtual_model_derived": [True]})
    with pytest.raises(ValueError, match="misaligned"):
        MOD.range_geometry(profiles, intervals)


def test_missing_variant_is_not_a_zero_energy_difference():
    common = dict(city="a", seed=1, allocation="random", service_fraction=1,
                  strategy="ltpref_interval", support="primary")
    selections = pd.DataFrame([
        common | dict(griffiths_coefficient=.5, interval_width_multiplier=1., change_pct=2.),
        common | dict(griffiths_coefficient=.5, interval_width_multiplier=.5, change_pct=None),
    ])
    _, summary = MOD.sensitivity_transitions(selections)
    narrower = summary[summary.interval_width_multiplier.eq(.5)].iloc[0]
    assert narrower.primary_match_lost == 1
    assert narrower.both_matched == 0
    assert pd.isna(narrower.variant_energy_change_median_pct)
