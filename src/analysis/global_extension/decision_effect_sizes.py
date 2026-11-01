"""Audit changed annual choices using the existing fixed-plan scoring tables.

No trajectories are simulated or rescored. Selection is replayed by the original
diagnostic, including physical aliases, score tolerance and energy tie-breaking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import analyse_global_results as original
except ImportError:  # direct script execution
    import analyse_global_results as original


PROFILE_KEYS = ["climate", "composition", "layout"]
SCENARIO_KEYS = [*PROFILE_KEYS, "permission"]
INPUT_NAMES = {
    "metrics": "GLOBAL_RUN_METRICS.csv",
    "anchors": "GLOBAL_25C_SERVICE_ANCHORS.csv",
    "reported": "GLOBAL_INFORMATION_DIAGNOSTIC.csv",
    "memberships": "GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
}


def _range(values: pd.Series) -> dict:
    return {"minimum": float(values.min()), "median": float(values.median()),
            "maximum": float(values.max())} if len(values) else {}


def analyse(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    reported: pd.DataFrame,
    memberships: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return changed-choice effects, all primary decisions and an audit summary."""
    if anchors.duplicated(PROFILE_KEYS).any():
        raise ValueError("Service anchors duplicate a profile group")
    if memberships.duplicated([*PROFILE_KEYS, "profile_id"]).any():
        raise ValueError("Fixed-tail membership duplicates a profile")
    counts = memberships.groupby(PROFILE_KEYS).size().sort_index()
    declared = anchors.set_index(PROFILE_KEYS).tail_profile_count.sort_index()
    if not counts.index.equals(declared.index) or not np.array_equal(counts, declared):
        raise ValueError("Fixed-tail membership counts differ from the service anchors")
    assigned = metrics.groupby(PROFILE_KEYS).assigned_profile_count.agg(["min", "max"])
    if (assigned["min"] != assigned["max"]).any() or not assigned.index.equals(counts.index):
        raise ValueError("Annual candidates do not share the declared profile groups")
    if not np.array_equal(np.ceil(assigned["min"] * 0.1).astype(int), counts):
        raise ValueError("Fixed-tail membership is not the declared 10% group")

    replay = original.fixed_plan_information_diagnostic(metrics, anchors, fractions=(1.0,))
    replay = replay.sort_values(SCENARIO_KEYS).reset_index(drop=True)
    expected = reported.loc[reported.service_fraction.eq(1.0)].sort_values(
        SCENARIO_KEYS
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            replay[expected.columns], expected, check_dtype=False,
            check_exact=False, rtol=0, atol=original.SERVICE_TOLERANCE_C,
        )
    except AssertionError as error:
        raise ValueError("Primary selection replay differs from the recorded diagnostic") from error
    if replay[["full_unresolved", "zone_mean_unresolved"]].any().any():
        raise ValueError("This paired effect audit requires both primary choices to exist")

    detail = replay.loc[replay.selection_disagreement].copy().reset_index(drop=True)
    detail = detail.merge(anchors[PROFILE_KEYS + ["tail_profile_count"]],
                          on=PROFILE_KEYS, validate="many_to_one")
    detail["service_tolerance_c"] = original.SERVICE_TOLERANCE_C
    for prefix in ("zone_mean_selected", "full_selected"):
        population = detail[f"{prefix}_mean_reference_mismatch_c"]
        tail = detail[f"{prefix}_fixed_tail_reference_mismatch_c"]
        detail[f"{prefix}_full_population_margin_c"] = population - detail.mean_limit_c
        detail[f"{prefix}_full_tail_margin_c"] = tail - detail.tail_limit_c
        detail[f"{prefix}_full_population_pass"] = population.le(
            detail.mean_limit_c + original.SERVICE_TOLERANCE_C
        )
        detail[f"{prefix}_full_tail_pass"] = tail.le(
            detail.tail_limit_c + original.SERVICE_TOLERANCE_C
        )
        detail[f"{prefix}_full_review_pass"] = (
            detail[f"{prefix}_full_population_pass"] & detail[f"{prefix}_full_tail_pass"]
        )
        detail[f"{prefix}_max_full_service_excess_c"] = detail[
            [f"{prefix}_full_population_margin_c", f"{prefix}_full_tail_margin_c"]
        ].clip(lower=0).max(axis=1)
    if not detail.full_selected_full_review_pass.all():
        raise ValueError("A full-profile selected plan fails its original service limits")
    if not np.array_equal(
        ~detail.zone_mean_selected_full_review_pass,
        detail.zone_mean_selected_plan_fails_full_review,
    ):
        raise ValueError("Full-profile recheck disagrees with the recorded failure flag")
    detail["choice_change_type"] = np.where(
        detail.zone_mean_selected_full_review_pass,
        "different_choice_both_full_qualified", "mean_choice_fails_full_review",
    )
    detail["zone_mean_selected_max_excess_in_score_tolerances"] = (
        detail.zone_mean_selected_max_full_service_excess_c / original.SERVICE_TOLERANCE_C
    )

    groups = dict(tuple(metrics.groupby(SCENARIO_KEYS, sort=True)))
    for index, row in detail.iterrows():
        group = groups[tuple(row[key] for key in SCENARIO_KEYS)]
        for prefix, population_column, tail_column in (
            ("full_selected", "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"),
            ("zone_mean_selected", "zone_mean_mean_reference_mismatch_c",
             "zone_mean_fixed_tail_reference_mismatch_c"),
        ):
            qualified = group[
                group.engineering_pass.astype(bool)
                & group[population_column].le(row.mean_limit_c + original.SERVICE_TOLERANCE_C)
                & group[tail_column].le(row.tail_limit_c + original.SERVICE_TOLERANCE_C)
            ]
            minimum = float(qualified.facility_purchased_kwh.min())
            # This is the existing choose_minimum_energy tolerance, not a new threshold.
            tolerance = max(1e-6, 1e-9 * abs(minimum))
            detail.loc[index, f"{prefix}_pool_minimum_purchased_kwh"] = minimum
            detail.loc[index, f"{prefix}_energy_tie_tolerance_kwh"] = tolerance
            detail.loc[index, f"{prefix}_energy_above_pool_minimum_kwh"] = (
                row[f"{prefix}_facility_purchased_kwh"] - minimum
            )

    detail["delta_purchased_kwh_full_minus_zone_mean"] = (
        detail.full_selected_facility_purchased_kwh
        - detail.zone_mean_selected_facility_purchased_kwh
    )
    detail["delta_purchased_pct_full_minus_zone_mean"] = (
        100 * detail.delta_purchased_kwh_full_minus_zone_mean
        / detail.zone_mean_selected_facility_purchased_kwh
    )
    tie_tolerance = detail[
        ["full_selected_energy_tie_tolerance_kwh", "zone_mean_selected_energy_tie_tolerance_kwh"]
    ].max(axis=1)
    detail["absolute_energy_change_in_max_tie_tolerances"] = (
        detail.delta_purchased_kwh_full_minus_zone_mean.abs() / tie_tolerance
    )
    detail["energy_change_exceeds_both_tie_tolerances"] = (
        detail.delta_purchased_kwh_full_minus_zone_mean.abs() > tie_tolerance
    )
    population_failure = ~detail.zone_mean_selected_full_population_pass
    tail_failure = ~detail.zone_mean_selected_full_tail_pass
    summary = {
        "status": "PASS",
        "definition": "Post-processing of the unchanged annual candidate scoring tables at service factor 1.00.",
        "counts": {
            "primary_scenarios": len(replay),
            "eligibility_set_changes": int(replay.eligibility_disagreement.sum()),
            "selected_physical_plan_changes": len(detail),
            "changed_mean_choices_failing_full_review": int((population_failure | tail_failure).sum()),
            "changed_choices_both_full_qualified": int(detail.zone_mean_selected_full_review_pass.sum()),
            "population_limit_failures": int(population_failure.sum()),
            "tail_limit_failures": int(tail_failure.sum()),
            "both_limit_failures": int((population_failure & tail_failure).sum()),
            "energy_changes_exceeding_both_tie_tolerances": int(detail.energy_change_exceeds_both_tie_tolerances.sum()),
        },
        "changed_scenarios_by_climate": detail.groupby("climate").size().to_dict(),
        "population_excess_c_in_failed_choices": _range(
            detail.loc[population_failure, "zone_mean_selected_full_population_margin_c"]),
        "tail_excess_c_in_failed_choices": _range(
            detail.loc[tail_failure, "zone_mean_selected_full_tail_margin_c"]),
        "maximum_service_excess_c_per_changed_choice": _range(
            detail.zone_mean_selected_max_full_service_excess_c),
        "delta_purchased_kwh_full_minus_zone_mean": _range(
            detail.delta_purchased_kwh_full_minus_zone_mean),
        "delta_purchased_pct_full_minus_zone_mean": _range(
            detail.delta_purchased_pct_full_minus_zone_mean),
        "absolute_energy_change_in_max_tie_tolerances": _range(
            detail.absolute_energy_change_in_max_tie_tolerances),
        "contract": {
            "service_fraction": 1.0,
            "service_tolerance_c": original.SERVICE_TOLERANCE_C,
            "service_limits": "Both reviews use the full-profile annual common-25-C reference scores.",
            "fixed_tail": "The 10% with largest profile-level annual center mismatch in the common-25-C reference; membership is unchanged between reviews and plans.",
            "population_weighting": "Each zone's native occupied person-hours are divided equally among assigned profiles, then population mismatch is weighted by those person-hours.",
            "tail_weighting": "First compute each fixed member's own occupied-person-hour-weighted annual mismatch; then take the unweighted mean over fixed members.",
            "zone_mean_review": "Replace each profile center by its assigned zone's arithmetic mean; retain trajectories, profile slots, weights and fixed-tail membership.",
            "margin_sign": "Full-profile score minus common limit, in C: positive is above the nominal limit; qualification includes the unchanged 1e-9 C tolerance.",
            "energy_tie_tolerance_kwh": "max(1e-6, 1e-9 * abs(minimum qualified facility_purchased_kwh)), separately for each review's pool.",
            "energy_tie_break": "Within the existing tolerance of the pool minimum, sort run_id then offset_c using stable mergesort.",
            "energy_delta_sign": "Full-profile selected energy minus zone-mean selected energy; percent uses zone-mean selected energy as denominator.",
            "energy_boundary": "Annual facility electricity plus complete annual facility natural gas on an HHV basis.",
        },
    }
    return detail, replay, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/global_extension/global_final"))
    parser.add_argument("--output", type=Path, default=Path("results/global_extension/decision_effect_sizes"))
    args = parser.parse_args()
    paths = {key: args.source / name for key, name in INPUT_NAMES.items()}
    audit_path = args.source / "GLOBAL_ANALYSIS_AUDIT.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    hashes = {key: original.sha256(path) for key, path in paths.items()}
    for key, path in paths.items():
        if hashes[key] != audit["output_sha256"][path.name]:
            raise ValueError(f"Input no longer matches the original analysis audit: {path.name}")
    detail, replay, summary = analyse(**{key: pd.read_csv(path) for key, path in paths.items()})
    summary["input_sha256"] = hashes | {"original_analysis_audit": original.sha256(audit_path)}
    summary["source_code_sha256"] = {
        "selection": original.sha256(Path(original.__file__)),
        "thermal_service": original.sha256(original.ENERGY_SERVICE / "thermal_service.py"),
        "effect_size_audit": original.sha256(Path(__file__)),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {"DECISION_EFFECT_SIZES.csv": detail, "DECISION_REPLAY_AUDIT.csv": replay}
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False, float_format="%.17g")
    summary["output_sha256"] = {name: original.sha256(args.output / name) for name in outputs}
    (args.output / "DECISION_EFFECT_SIZES_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
