"""Seasonal summaries keep finite cohort grids and actual matched support."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'parallel_seasonal', ROOT / 'src/analysis/seasonal/derive_seasonal_comparison.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def selection_grid(layouts):
    rows = []
    for context in MODULE.CONTEXTS:
        for seed in (1, 2, 3):
            for layout in layouts:
                for control in MODULE.CONTROLS:
                    for mode in ('direct', 'equal_service'):
                        rows.append({'context': context, 'composition_id': seed, 'layout': layout,
                                     'permission': 'native_zone', 'control': control,
                                     'comparison_mode': mode})
    table = pd.DataFrame(rows)
    return table, table[table.comparison_mode.eq('direct')]


def test_original_default_scope_and_explicit_parallel_scope_are_distinct():
    original, direct = selection_grid(['random_mixing', 'demand_ordered', 'source_structured'])
    MODULE.validate_selection_grid(original, direct)
    parallel, direct = selection_grid(['random_mixing', 'demand_ordered'])
    MODULE.validate_selection_grid(parallel, direct, expected_layouts=2)
    assert len(parallel) == 144
    with pytest.raises(ValueError, match='3 layouts'):
        MODULE.validate_selection_grid(parallel, direct)


def test_balanced_row_count_cannot_hide_a_missing_or_duplicate_design_cell():
    parallel, direct = selection_grid(['random_mixing', 'demand_ordered'])
    damaged = parallel.copy()
    damaged.iloc[0] = damaged.iloc[1]
    with pytest.raises(ValueError, match='unique native-zone cells'):
        MODULE.validate_selection_grid(damaged, direct, expected_layouts=2)
    damaged = parallel.replace({'layout': {'demand_ordered': 'invented_building_structure'}})
    with pytest.raises(ValueError, match='unique native-zone cells'):
        MODULE.validate_selection_grid(damaged, direct, expected_layouts=2)


def test_parallel_summary_uses_actual_matched_pairs_not_primary_114(tmp_path):
    rows = []
    for seed in (1, 2, 3):
        for mode, fraction, gap in [('direct', -1., 4.), ('equal_service', 1., .5)]:
            rows.append({'context': 'Miami', 'composition_id': seed, 'layout': 'random_mixing',
                         'permission': 'native_zone', 'alternative': 'personal_center',
                         'comparison_mode': mode, 'service_fraction': fraction,
                         'status': 'matched' if seed < 3 else 'unmatched', 'delta_site_pct': gap})
    annual = tmp_path / 'annual'
    annual.mkdir()
    pd.DataFrame(rows).to_csv(annual / 'CONTROL_COMPARISONS.csv', index=False)
    MODULE.export_comparison_regimes(SimpleNamespace(
        annual_root=annual, output=tmp_path, expected_layouts=2))
    summary = json.loads((tmp_path / 'COMPARISON_REGIMES_QA.json').read_text())
    assert summary['matched_conditions'] == 2
    assert summary['median_absolute_direct_energy_difference_pct'] == 4.
    assert summary['median_absolute_common_service_energy_difference_pct'] == .5
    with pytest.raises(ValueError, match='114'):
        MODULE.export_comparison_regimes(SimpleNamespace(annual_root=annual, output=tmp_path))
