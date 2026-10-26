from __future__ import annotations

import pandas as pd

from analysis.targeted_extension.assemble_final_evidence import headline_summary


def test_headline_summary_keeps_direct_and_same_service_denominators_separate() -> None:
    annual = pd.DataFrame(
        [
            {
                "quality_scope": "primary",
                "comparison_mode": "direct",
                "permission": "native_zone",
                "layout": "random_mixing",
                "status": "matched",
                "service_fraction": -1.0,
                "delta_mean_reference_mismatch_c": -0.2,
                "delta_fixed_tail_reference_mismatch_c": -0.1,
                "delta_modeled_interval_person_hour_coverage": 0.1,
                "delta_annual_site_kwh": 10.0,
                "delta_site_pct": 1.0,
            },
            {
                "quality_scope": "primary",
                "comparison_mode": "equal_service",
                "permission": "building",
                "layout": "demand_ordered",
                "status": "matched",
                "service_fraction": 1.0,
                "delta_mean_reference_mismatch_c": 0.0,
                "delta_fixed_tail_reference_mismatch_c": 0.0,
                "delta_modeled_interval_person_hour_coverage": 0.0,
                "delta_annual_site_kwh": -5.0,
                "delta_site_pct": -0.5,
            },
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "quality_scope": "primary",
                "comparison_mode": "equal_service",
                "service_fraction": 0.5,
                "layout": "random_mixing",
                "selection_status": "selected",
                "control": "personal_center",
            }
        ]
    )

    summary = headline_summary(annual, selections)

    assert summary["direct_common_native_zone"] == {
        "matched_pairs": 1,
        "mean_mismatch_improved": 1,
        "fixed_tail_improved": 1,
        "interval_coverage_improved": 1,
        "all_three_service_endpoints_improved": 1,
        "annual_energy_increased": 1,
        "annual_energy_decreased": 0,
    }
    assert summary["same_service_common_design"]["matched_pairs"] == 1
    assert summary["tight_service_selected_conditions"] == {
        "uniform_average": 0,
        "personal_center": 1,
        "personal_range": 0,
    }
