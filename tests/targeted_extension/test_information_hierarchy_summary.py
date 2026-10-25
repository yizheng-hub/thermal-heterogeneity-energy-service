from pathlib import Path

import pandas as pd
import pytest

from analysis.experiments.decision_value.summarize_information_hierarchy import (
    build_rows,
    summarize_building_mean_to_zone_mean,
    summarize_zone_mean_to_full,
)


def test_zone_mean_to_full_counts_are_derived_from_rows():
    phase = pd.DataFrame(
        [
            {
                "classification": "mean-safe",
                "full_feasible_candidates": 2,
                "mean_only_eligibility_exact": True,
                "mean_only_selected_action_same_as_full": True,
                "mean_only_selected_full_eligible": True,
            },
            {
                "classification": "distribution-required",
                "full_feasible_candidates": 1,
                "mean_only_eligibility_exact": False,
                "mean_only_selected_action_same_as_full": False,
                "mean_only_selected_full_eligible": False,
            },
            {
                "classification": "finite-library-unresolved",
                "full_feasible_candidates": 0,
                "mean_only_eligibility_exact": False,
                "mean_only_selected_action_same_as_full": False,
                "mean_only_selected_full_eligible": False,
            },
        ]
    )
    result = summarize_zone_mean_to_full(phase)
    assert result == {
        "design_cells": 3,
        "full_profile_resolvable_cells": 2,
        "zone_mean_safe_cells": 1,
        "distribution_required_cells": 1,
        "finite_library_unresolved_cells": 1,
        "zone_mean_eligibility_disagreement_cells": 1,
        "zone_mean_selection_disagreement_cells": 1,
        "zone_mean_selected_plan_fails_full_review_cells": 1,
        "distribution_required_share_of_resolvable": 0.5,
    }


def test_building_to_zone_summary_uses_only_declared_matched_subset():
    comparisons = pd.DataFrame(
        [
            {
                "comparison_mode": "direct",
                "permission": "native_zone",
                "control": "personal_center",
                "status": "matched",
                "delta_annual_site_kwh": 2.0,
                "delta_mean_reference_mismatch_c": -0.2,
                "delta_fixed_tail_reference_mismatch_c": -0.3,
                "delta_modeled_interval_person_hour_coverage": 0.1,
            },
            {
                "comparison_mode": "direct",
                "permission": "native_zone",
                "control": "personal_center",
                "status": "unmatched",
                "delta_annual_site_kwh": float("nan"),
                "delta_mean_reference_mismatch_c": float("nan"),
                "delta_fixed_tail_reference_mismatch_c": float("nan"),
                "delta_modeled_interval_person_hour_coverage": float("nan"),
            },
            {
                "comparison_mode": "equal_service",
                "permission": "native_zone",
                "control": "personal_center",
                "status": "matched",
                "delta_annual_site_kwh": -9.0,
                "delta_mean_reference_mismatch_c": -9.0,
                "delta_fixed_tail_reference_mismatch_c": -9.0,
                "delta_modeled_interval_person_hour_coverage": 9.0,
            },
        ]
    )
    result = summarize_building_mean_to_zone_mean(comparisons)
    assert result["designed_comparisons"] == 2
    assert result["matched_comparisons"] == 1
    assert result["mean_mismatch_improved"] == 1
    assert result["annual_energy_increased"] == 1
    assert result["annual_energy_decreased"] == 0


def test_output_keeps_evidence_blocks_separate():
    building_zone = {
        "designed_comparisons": 2,
        "matched_comparisons": 1,
        "unmatched_comparisons": 1,
        "mean_mismatch_improved": 1,
        "fixed_tail_mismatch_improved": 1,
        "interval_coverage_improved": 1,
        "annual_energy_increased": 1,
        "annual_energy_decreased": 0,
    }
    zone_full = {
        "design_cells": 3,
        "full_profile_resolvable_cells": 2,
        "zone_mean_safe_cells": 1,
        "distribution_required_cells": 1,
        "finite_library_unresolved_cells": 1,
        "zone_mean_eligibility_disagreement_cells": 1,
        "zone_mean_selection_disagreement_cells": 1,
        "zone_mean_selected_plan_fails_full_review_cells": 1,
        "distribution_required_share_of_resolvable": 0.5,
    }
    rows = build_rows(
        building_zone,
        zone_full,
        control_source="controls.csv",
        phase_source="phase.csv",
    )
    assert set(rows["evidence_block"]) == {
        "building_mean_to_zone_mean_direct_control",
        "zone_mean_to_full_profile_decision_review",
    }
    assert not rows["single_staircase_compatible"].any()


def test_current_row_level_results_reproduce_declared_counts():
    project = Path(__file__).resolve().parents[2]
    phase_path = (
        project
        / "output/replay/supporting_evidence/analysis/tables/result6_phase_map.csv"
    )
    control_path = project / "results/Joint_main/annual/CONTROL_COMPARISONS.csv"
    if not phase_path.exists() or not control_path.exists():
        pytest.skip("External replay outputs are not present in the source-only checkout")

    zone_full = summarize_zone_mean_to_full(pd.read_csv(phase_path))
    assert {
        key: zone_full[key]
        for key in (
            "design_cells",
            "full_profile_resolvable_cells",
            "zone_mean_safe_cells",
            "distribution_required_cells",
            "finite_library_unresolved_cells",
        )
    } == {
        "design_cells": 1008,
        "full_profile_resolvable_cells": 606,
        "zone_mean_safe_cells": 275,
        "distribution_required_cells": 331,
        "finite_library_unresolved_cells": 402,
    }
    building_zone = summarize_building_mean_to_zone_mean(pd.read_csv(control_path))
    assert building_zone["designed_comparisons"] == 48
    assert building_zone["matched_comparisons"] == 24
