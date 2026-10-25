"""Contracts that prevent misleading spatial contrasts."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[2] / "src/analysis/targeted_extension/spatial_effects.py"
SPEC = importlib.util.spec_from_file_location("targeted_spatial_effects", PATH)
SPATIAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPATIAL)


def design(statuses=None):
    rows = []
    energies = [100.0, 105.0, 110.0, 108.0]
    for index, (layout, permission) in enumerate([
        ("random_mixing", "building"), ("demand_ordered", "building"),
        ("random_mixing", "native_zone"), ("demand_ordered", "native_zone")
    ]):
        row = dict(context="example", composition_id=1, control="personal_center", weight_view="profile_equal",
                   comparison_mode="equal_service", service_fraction=1.0, layout=layout, permission=permission,
                   mean_budget_c=2.0, fixed_tail_budget_c=4.0, selection_status=(statuses or ["selected"] * 4)[index],
                   run_id=str(index))
        row.update({name: energies[index] if name == "annual_site_kwh" else 0.2 for name in SPATIAL.METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_alias_is_design_property_not_equal_outcome():
    frame = pd.DataFrame({"context": ["small"] * 3, "permission": ["building", "floor", "native_zone"], "energy": [1, 1, 1]})
    aliases = pd.DataFrame({"context": ["small"], "permission": ["floor"]})
    assert SPATIAL.remove_aliases(frame, aliases).permission.tolist() == ["building", "native_zone"]


def test_missing_selection_keeps_denominator_without_filling_effect():
    frame = design(["selected", "service_unavailable", "selected", "selected"])
    pairs = SPATIAL.pair_contrasts(frame)
    pair = pairs[(pairs.axis == "layout") & (pairs.permission == "building")].iloc[0]
    assert pair.status == "unmatched"
    assert np.isnan(pair.delta_annual_site_kwh)
    assert len(pairs) == 4


def test_layout_contract_cannot_change_with_placement():
    frame = design()
    frame.loc[frame.layout == "demand_ordered", "fixed_tail_budget_c"] = 5.0
    with pytest.raises(ValueError, match="absolute service contract"):
        SPATIAL.pair_contrasts(frame)


def test_four_corner_interaction_uses_same_composition():
    row = SPATIAL.interaction_contrasts(design()).iloc[0]
    assert row.status == "matched"
    assert row.interaction_annual_site_kwh == -7.0
    assert row.interaction_site_pct == -7.0
    failed = SPATIAL.interaction_contrasts(design(["selected", "service_unavailable", "selected", "selected"])).iloc[0]
    assert failed.selected_corners == 3
    assert failed.qualification_interaction == 1
    assert np.isnan(failed.interaction_annual_site_kwh)
