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


def test_headline_response_summary_uses_declared_main_layouts() -> None:
    annual = pd.DataFrame(
        columns=[
            "quality_scope", "comparison_mode", "permission", "layout", "status",
            "service_fraction", "delta_mean_reference_mismatch_c",
            "delta_fixed_tail_reference_mismatch_c",
            "delta_modeled_interval_person_hour_coverage", "delta_annual_site_kwh",
            "delta_site_pct",
        ]
    )
    selections = pd.DataFrame(
        columns=[
            "quality_scope", "comparison_mode", "service_fraction", "layout",
            "selection_status", "control",
        ]
    )
    common = pd.DataFrame(
        [
            {
                "quality_scope": "primary", "layout": layout,
                "information_level": "building_mean", "selected_response_kw": 1.0,
                "full_reviewed_usable_response_kw": 0.0,
                "selected_review_outcome": "false_acceptance", "data_source": "ASHRAE",
                "context": "X", "composition_id": "c", "permission": "building",
                "event_month": 7, "ideal_full_information_response_kw": 1.0,
            }
            for layout in ("random_mixing", "source_structured")
        ]
    )
    own = pd.DataFrame(
        [
            {
                "quality_scope": "primary", "layout": layout,
                "qualified_positive_response": True, "control": "uniform_average",
                "recovery_extra_electric_kwh": 0.5, "event_electric_saving_kwh": 1.0,
                "event_plus_recovery_net_electric_saving_kwh": 0.5,
            }
            for layout in ("demand_ordered", "source_structured")
        ]
    )
    comparisons = pd.DataFrame(
        [
            {
                "quality_scope": "primary", "layout": layout,
                "uniform_status": "selected", "demand_guided_status": "selected",
                "response_difference_kw": -1.0,
            }
            for layout in ("demand_ordered", "source_structured")
        ]
    )

    summary = headline_summary(
        annual, selections, common_dr=common, own_dr=own,
        own_comparisons=comparisons,
    )

    assert summary["common_baseline_response"]["building_mean"]["evaluated_conditions"] == 1
    assert summary["own_baseline_response"]["evaluated_control_conditions"] == 1
    assert summary["own_baseline_response"]["paired_both_positive"] == 1
