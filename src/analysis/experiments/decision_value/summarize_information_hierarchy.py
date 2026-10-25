"""Summarize what building-, zone- and full-profile demand information establishes.

The two evidence blocks use existing row-level results and add no physical
simulation.  They are intentionally kept separate because they use different
populations, action supports and decision rules; their denominators are not a
single causal staircase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CLASSIFICATIONS = {
    "mean-safe",
    "distribution-required",
    "finite-library-unresolved",
}


def _bool_series(values: pd.Series) -> pd.Series:
    """Return nullable CSV booleans as ordinary booleans, with missing=False."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    unknown = normalized[normalized.notna() & ~normalized.isin({"true", "false"})]
    if not unknown.empty:
        raise ValueError(f"Unexpected boolean values: {sorted(unknown.unique())}")
    return normalized.eq("true").fillna(False)


def summarize_zone_mean_to_full(phase: pd.DataFrame) -> dict[str, int | float]:
    """Recompute the mean-only versus full-profile decision counts."""

    required = {
        "classification",
        "full_feasible_candidates",
        "mean_only_eligibility_exact",
        "mean_only_selected_action_same_as_full",
        "mean_only_selected_full_eligible",
    }
    missing = required - set(phase.columns)
    if missing:
        raise ValueError(f"Phase map lacks columns: {sorted(missing)}")
    classes = set(phase["classification"].dropna().unique())
    if classes != CLASSIFICATIONS:
        raise ValueError(f"Unexpected phase-map classifications: {sorted(classes)}")

    unresolved = phase["classification"].eq("finite-library-unresolved")
    resolved = phase.loc[~unresolved].copy()
    eligibility_exact = _bool_series(resolved["mean_only_eligibility_exact"])
    selection_same = _bool_series(resolved["mean_only_selected_action_same_as_full"])
    selected_full_eligible = _bool_series(resolved["mean_only_selected_full_eligible"])

    if not phase.loc[unresolved, "full_feasible_candidates"].eq(0).all():
        raise ValueError("Unresolved rows must have no full-profile-qualified action")
    mean_safe = resolved["classification"].eq("mean-safe")
    distribution_required = resolved["classification"].eq("distribution-required")
    if not (eligibility_exact[mean_safe] & selection_same[mean_safe]).all():
        raise ValueError("Mean-safe rows must reproduce full eligibility and selection")
    changed = ~eligibility_exact | ~selection_same
    if not changed.eq(distribution_required).all():
        raise ValueError("Distribution-required rows do not match their declared definition")

    counts: dict[str, int | float] = {
        "design_cells": int(len(phase)),
        "full_profile_resolvable_cells": int(len(resolved)),
        "zone_mean_safe_cells": int(mean_safe.sum()),
        "distribution_required_cells": int(distribution_required.sum()),
        "finite_library_unresolved_cells": int(unresolved.sum()),
        "zone_mean_eligibility_disagreement_cells": int((~eligibility_exact).sum()),
        "zone_mean_selection_disagreement_cells": int((~selection_same).sum()),
        "zone_mean_selected_plan_fails_full_review_cells": int((~selected_full_eligible).sum()),
    }
    counts["distribution_required_share_of_resolvable"] = (
        counts["distribution_required_cells"] / counts["full_profile_resolvable_cells"]
    )
    return counts


def summarize_building_mean_to_zone_mean(comparisons: pd.DataFrame) -> dict[str, int | float]:
    """Summarize the matched direct Uniform-average/Personal-centre contrast."""

    required = {
        "comparison_mode",
        "permission",
        "control",
        "status",
        "delta_annual_site_kwh",
        "delta_mean_reference_mismatch_c",
        "delta_fixed_tail_reference_mismatch_c",
        "delta_modeled_interval_person_hour_coverage",
    }
    missing = required - set(comparisons.columns)
    if missing:
        raise ValueError(f"Control comparison table lacks columns: {sorted(missing)}")
    selected = comparisons.loc[
        comparisons["comparison_mode"].eq("direct")
        & comparisons["permission"].eq("native_zone")
        & comparisons["control"].eq("personal_center")
    ].copy()
    if selected.empty:
        raise ValueError("No native-zone direct Personal-centre comparisons found")
    matched = selected.loc[selected["status"].eq("matched")]
    tolerance = 1e-9
    return {
        "designed_comparisons": int(len(selected)),
        "matched_comparisons": int(len(matched)),
        "unmatched_comparisons": int(len(selected) - len(matched)),
        "mean_mismatch_improved": int(
            matched["delta_mean_reference_mismatch_c"].lt(-tolerance).sum()
        ),
        "fixed_tail_mismatch_improved": int(
            matched["delta_fixed_tail_reference_mismatch_c"].lt(-tolerance).sum()
        ),
        "interval_coverage_improved": int(
            matched["delta_modeled_interval_person_hour_coverage"].gt(tolerance).sum()
        ),
        "annual_energy_increased": int(matched["delta_annual_site_kwh"].gt(tolerance).sum()),
        "annual_energy_decreased": int(matched["delta_annual_site_kwh"].lt(-tolerance).sum()),
    }


def _row(
    block: str,
    transition: str,
    metric: str,
    count: int,
    denominator: int,
    definition: str,
    source_file: str,
    source_filter: str,
) -> dict[str, object]:
    return {
        "evidence_block": block,
        "transition": transition,
        "metric": metric,
        "count": count,
        "denominator": denominator,
        "fraction": count / denominator,
        "definition": definition,
        "source_file": source_file,
        "source_filter": source_filter,
        "single_staircase_compatible": False,
    }


def build_rows(
    building_zone: dict[str, int | float],
    zone_full: dict[str, int | float],
    *,
    control_source: str,
    phase_source: str,
) -> pd.DataFrame:
    """Build a compact long-form table without joining incompatible denominators."""

    direct_n = int(building_zone["designed_comparisons"])
    matched_n = int(building_zone["matched_comparisons"])
    total = int(zone_full["design_cells"])
    resolved = int(zone_full["full_profile_resolvable_cells"])
    rows = [
        _row(
            "building_mean_to_zone_mean_direct_control",
            "building-wide mean -> zone mean",
            "designed_comparisons",
            direct_n,
            direct_n,
            "Direct Uniform-average versus Personal-centre comparisons under native-zone commands.",
            control_source,
            "comparison_mode=direct; permission=native_zone; control=personal_center",
        ),
        _row(
            "building_mean_to_zone_mean_direct_control",
            "building-wide mean -> zone mean",
            "matched_comparisons",
            matched_n,
            direct_n,
            "Both direct control runs passed the engineering and availability checks.",
            control_source,
            "status=matched within the direct native-zone Personal-centre subset",
        ),
    ]
    for metric, definition in (
        ("mean_mismatch_improved", "Personal centre reduced full-profile population-mean reference mismatch."),
        ("fixed_tail_mismatch_improved", "Personal centre reduced mismatch for the fixed reference-tail profiles."),
        ("interval_coverage_improved", "Personal centre increased coverage of the model-derived reference intervals."),
        ("annual_energy_increased", "Personal centre used more annual purchased site HVAC energy."),
        ("annual_energy_decreased", "Personal centre used less annual purchased site HVAC energy."),
    ):
        rows.append(
            _row(
                "building_mean_to_zone_mean_direct_control",
                "building-wide mean -> zone mean",
                metric,
                int(building_zone[metric]),
                matched_n,
                definition,
                control_source,
                "matched direct native-zone Personal-centre versus Uniform-average pairs",
            )
        )

    for metric, denominator, definition in (
        ("design_cells", total, "All declared population-context-seed-spread-budget cells."),
        (
            "full_profile_resolvable_cells",
            total,
            "Cells with at least one action passing full-profile mean and fixed-tail limits.",
        ),
        (
            "zone_mean_safe_cells",
            resolved,
            "Resolved cells where zone-mean and full-profile review agree on every candidate's eligibility and select the same action.",
        ),
        (
            "distribution_required_cells",
            resolved,
            "Resolved cells where zone-mean review changes at least one candidate's eligibility or selects a different action.",
        ),
        (
            "finite_library_unresolved_cells",
            total,
            "Cells with no full-profile-qualified action in the tested physical library.",
        ),
        (
            "zone_mean_eligibility_disagreement_cells",
            resolved,
            "Resolved cells where zone-mean and full-profile candidate eligibility differ.",
        ),
        (
            "zone_mean_selection_disagreement_cells",
            resolved,
            "Resolved cells where zone-mean and full-profile reviews select different actions.",
        ),
        (
            "zone_mean_selected_plan_fails_full_review_cells",
            resolved,
            "Resolved cells where the action selected by zone-mean review fails full-profile review.",
        ),
    ):
        rows.append(
            _row(
                "zone_mean_to_full_profile_decision_review",
                "zone mean -> full within-zone profile distribution",
                metric,
                int(zone_full[metric]),
                denominator,
                definition,
                phase_source,
                "random_mixing; identical physical action library and service budgets across representations",
            )
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-map",
        type=Path,
        default=project
        / "output/replay/supporting_evidence/analysis/tables/result6_phase_map.csv",
    )
    parser.add_argument(
        "--control-comparisons",
        type=Path,
        default=project / "results/Joint_main/annual/CONTROL_COMPARISONS.csv",
    )
    parser.add_argument(
        "--output-root", type=Path, default=project / "results/decision_value"
    )
    args = parser.parse_args()

    phase_path = args.phase_map.resolve()
    control_path = args.control_comparisons.resolve()
    output_root = args.output_root.resolve()
    zone_full = summarize_zone_mean_to_full(pd.read_csv(phase_path))
    building_zone = summarize_building_mean_to_zone_mean(pd.read_csv(control_path))
    rows = build_rows(
        building_zone,
        zone_full,
        control_source=control_path.relative_to(project).as_posix(),
        phase_source=phase_path.relative_to(project).as_posix(),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "INFORMATION_HIERARCHY_SUMMARY.csv"
    json_path = output_root / "INFORMATION_HIERARCHY_SUMMARY.json"
    rows.to_csv(csv_path, index=False, float_format="%.12g")
    report = {
        "status": "PASS",
        "physical_simulations_added": 0,
        "single_causal_staircase": False,
        "reason_not_single_staircase": (
            "The building-to-zone contrast uses the six fixed main-study compositions and direct "
            "controller choices. The zone-to-full contrast uses three archive population strata, "
            "seven imposed spread levels, four service budgets and one fixed physical action library. "
            "Their populations, candidate supports, objectives and denominators differ, so their "
            "counts must not be concatenated as sequential attrition."
        ),
        "building_mean_to_zone_mean": {
            "question": "What changes when direct targets use assigned zone means instead of one building-wide mean?",
            "scope": "Matched direct Uniform-average versus Personal-centre runs under native-zone commands.",
            "counts": building_zone,
        },
        "zone_mean_to_full_profile": {
            "question": "When does one mean per zone reproduce decisions made with the complete within-zone profile distribution?",
            "scope": "Identical physical action library and service budgets; representation changes only the service review.",
            "counts": zone_full,
        },
        "definitions": {
            "zone_mean_safe": "A full-profile-qualified action exists, zone-mean and full-profile eligibility agree for every candidate, and both reviews select the same physical action.",
            "distribution_required": "A full-profile-qualified action exists, but zone-mean review changes at least one candidate's eligibility or selects a different physical action.",
            "finite_library_unresolved": "No tested physical action passes the full-profile mean and fixed-tail limits.",
        },
        "inputs": {
            phase_path.relative_to(project).as_posix(): _sha256(phase_path),
            control_path.relative_to(project).as_posix(): _sha256(control_path),
        },
        "outputs": {csv_path.relative_to(project).as_posix(): _sha256(csv_path)},
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
