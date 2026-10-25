"""Shared-design filtering keeps failed comparisons and removes only declared aliases."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


path = Path(__file__).resolve().parents[2]/"src/analysis/targeted_extension/summarize_parallel_cohorts.py"
spec = spec_from_file_location("parallel_cohort_summary", path)
module = module_from_spec(spec)
spec.loader.exec_module(module)


def test_shared_scope_keeps_rejection_and_ordered_layout():
    frame = pd.DataFrame([
        dict(context="Miami", layout="random_mixing", permission="floor", status="unmatched"),
        dict(context="Miami", layout="demand_ordered", permission="native_zone", status="matched"),
        dict(context="Seattle_small2019", layout="random_mixing", permission="floor", status="matched"),
        dict(context="Seattle_small2019", layout="source_structured", permission="building", status="matched"),
    ])
    result = module.shared_scope(frame)
    assert result.index.tolist() == [0, 1]
    assert result.status.tolist() == ["unmatched", "matched"]


def test_summary_denominator_includes_unavailable_and_signed_effects():
    rows = []
    for status, value in [("matched", -2.), ("matched", 3.), ("unmatched", None)]:
        rows.append({"data_source": "CTCD", "status": status}
                    | dict.fromkeys(module.METRICS, value))
    result = module.summarize(pd.DataFrame(rows), ["data_source"]).iloc[0]
    assert result.designed_pairs == 3
    assert result.matched_pairs == 2
    assert result.unavailable_pairs == 1
    assert result.delta_site_pct_median == .5
    assert result.delta_site_pct_positive == 1
    assert result.delta_site_pct_negative == 1
