"""Regression against retained scored controls, including a lost strict option."""

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_strict_feasibility_keeps_the_low_coefficient_loss_and_permission_boundary():
    spec = importlib.util.spec_from_file_location(
        "strict_service", ROOT / "src/analysis/experiments/candidate_support/strict_service.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = ROOT / "data/analysis_results/method_comparisons/candidate_support"
    detail, summary, context = module.summarize(
        pd.read_csv(source / "candidate_scores.csv"),
        pd.read_csv(source / "service_anchors.csv"))
    assert len(detail) == 1620 and len(summary) == 45 and len(context) == 180
    assert set(detail.permission) == {"native_zone"}
    assert not detail[detail.control.eq("uniform")].strict_service_available.any()
    assert summary.native_zone_conditions.eq(36).all()
    for support in module.SUPPORTS:
        for control in ("personal_center", "ltpref_interval"):
            selected = summary[summary.candidate_support.eq(support)
                               & summary.control.eq(control)]
            expected = [2 if g == .33 else 3 for g in selected.griffiths_coefficient]
            assert selected.strict_service_available_conditions.tolist() == expected
            lost = detail[detail.candidate_support.eq(support) & detail.control.eq(control)
                          & detail.city.eq("Miami") & detail.seed.eq(2026102002)
                          & detail.allocation.eq("demand_ordered")
                          & detail.griffiths_coefficient.eq(.33)]
            assert lost.selection_status.tolist() == ["strict_service_unavailable"]
