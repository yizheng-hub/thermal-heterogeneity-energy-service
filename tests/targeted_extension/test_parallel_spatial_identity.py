"""Placement comparisons must preserve profile and fixed-tail identities."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


path = Path(__file__).resolve().parents[2]/"src/analysis/targeted_extension/spatial_effects.py"
spec = spec_from_file_location("parallel_spatial_identity", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def make_inputs(tmp_path, changed=False):
    annual, common = tmp_path/"annual", tmp_path/"common"
    annual.mkdir()
    common.mkdir()
    rows = []
    for layout in ("random_mixing", "demand_ordered"):
        for i in range(267):
            identity = f"profile_{i:03d}"
            if changed and layout == "demand_ordered" and i == 266:
                identity = "replacement"
            rows.append(dict(context="Miami", composition_id=1, layout=layout, profile_id=identity))
    pd.DataFrame(rows).to_csv(annual/"PROFILE_ANNUAL_SERVICE.csv.gz", index=False)
    pd.DataFrame([dict(context="Miami", composition_id=1,
                       fixed_tail_profile_ids="|".join(f"profile_{i:03d}" for i in range(27)))]).to_csv(
                           common/"COMMON_LAYOUT_ANCHORS.csv", index=False)
    return annual, common


def test_two_layout_protocol_retains_all_profiles_and_tail(tmp_path):
    annual, common = make_inputs(tmp_path)
    assert module.verify_people_and_tails(annual, common, expected_layouts=2) == 1
    with pytest.raises(ValueError, match="same 267 identities"):
        module.verify_people_and_tails(annual, common)


def test_two_layout_protocol_rejects_replacing_a_profile(tmp_path):
    annual, common = make_inputs(tmp_path, changed=True)
    with pytest.raises(ValueError, match="same 267 identities"):
        module.verify_people_and_tails(annual, common, expected_layouts=2)
