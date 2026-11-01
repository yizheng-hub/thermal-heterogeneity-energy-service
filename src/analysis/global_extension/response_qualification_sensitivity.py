"""Locate response qualification losses and rescore all service windows additively."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.response_mean_decomposition import (
    CONTROLS,
    PAIR_KEYS,
    _boolean,
    analyse as decompose_means,
)


DELTAS = (0.0, 0.01, 0.025, 0.05, 0.10)
WINDOWS = ("precondition", "event", "recovery")
METRICS = ("population_mean_mismatch_c", "fixed_tail_mean_mismatch_c")
TOLERANCE = 1e-9
STAGES = (
    "baseline_failed",
    "action_event_failed_after_baseline_pass",
    "action_non_event_failed_after_baseline_and_event_pass",
    "all_service_pass_positive_response",
    "all_service_pass_nonpositive_response",
)
FLAGS = (
    "baseline_service_qualified",
    "action_event_service_qualified",
    "action_non_event_service_qualified",
    "service_qualified",
    "qualified_positive_event_response",
)


def rescore(scores: pd.DataFrame, delta_c: float) -> pd.DataFrame:
    """Apply reference + delta + original tolerance in every applicable check."""
    if not np.isfinite(delta_c) or delta_c < 0:
        raise ValueError("The additive service allowance must be finite and nonnegative")
    frame = scores[np.isclose(scores.event_service_limit_factor, 1.0)].copy()
    occupied = frame.reference_precondition_weighted_profile_hours.gt(0)
    if not occupied.eq(_boolean(frame.precondition_service_evaluated,
                                "precondition_service_evaluated")).all():
        raise ValueError("Precondition applicability differs from the original rule")
    positive = _boolean(frame.positive_event_response, "positive_event_response")
    if not positive.eq(frame.event_electricity_reduction_kwh.gt(TOLERANCE)).all():
        raise ValueError("Original positivity differs from the event-energy rule")
    for party in ("baseline", "action"):
        for window in WINDOWS:
            applicable = occupied if window == "precondition" else True
            for metric in METRICS:
                observed = frame[f"{party}_{window}_{metric}"]
                reference = frame[f"reference_{window}_{metric}"]
                passed = (np.isfinite(observed) & np.isfinite(reference)
                          & observed.le(reference + delta_c + TOLERANCE))
                frame[f"{party}_{window}_{metric}_failed"] = applicable & ~passed
            failures = [f"{party}_{window}_{metric}_failed" for metric in METRICS]
            frame[f"{party}_{window}_passed"] = ~frame[failures].any(axis=1)
    baseline = frame[[f"baseline_{w}_passed" for w in WINDOWS]].all(axis=1)
    event = frame.action_event_passed
    non_event = frame[["action_precondition_passed", "action_recovery_passed"]].all(axis=1)
    values = (baseline, event, non_event, baseline & event & non_event,
              baseline & event & non_event & positive)
    for name, value in zip(FLAGS, values):
        if delta_c == 0 and not value.eq(_boolean(frame[name], name)).all():
            raise ValueError(f"Zero-allowance recomputation differs from retained {name}")
        frame[name] = value
    frame["qualification_stage"] = np.select(
        [~baseline, ~event, ~non_event, positive], STAGES[:4], default=STAGES[4])
    frame["delta_c"] = delta_c
    frame["_all_controls_baseline"] = frame.groupby(PAIR_KEYS)[
        "baseline_service_qualified"].transform("all")
    return frame


def _groups(frame):
    yield "overall", "all", frame
    for value, group in frame.groupby("climate", sort=True):
        yield "climate", str(value), group


def _summaries(frame):
    means, qualified_pairs = decompose_means(frame)
    keep = means.group_dimension.isin(("overall", "climate"))
    means = means[keep].copy()
    stage_rows, window_rows = [], []
    for dimension, value, group in _groups(frame):
        for control in CONTROLS:
            selected = group[group.control.eq(control)]
            row = {"group_dimension": dimension, "group_value": value, "control": control,
                   "baseline_qualified_count": int(selected.baseline_service_qualified.sum()),
                   "baseline_qualified_fraction": float(selected.baseline_service_qualified.mean()),
                   "raw_nonpositive_count": int((~_boolean(
                       selected.positive_event_response, "positive_event_response")).sum())}
            for stage in STAGES:
                in_stage = selected.qualification_stage.eq(stage)
                count = int(in_stage.sum())
                row[f"{stage}_count"] = count
                row[f"{stage}_fraction"] = count / len(selected)
                row[f"{stage}_nonpositive_count"] = int((in_stage & ~_boolean(
                    selected.positive_event_response, "positive_event_response")).sum())
            stage_rows.append(row)
            for party in ("baseline", "action"):
                for window in WINDOWS:
                    applicable = (selected.reference_precondition_weighted_profile_hours.gt(0)
                                  if window == "precondition"
                                  else pd.Series(True, index=selected.index))
                    failed = ~selected[f"{party}_{window}_passed"]
                    window_rows.append({
                        "group_dimension": dimension, "group_value": value, "control": control,
                        "party": party, "window": window, "case_count": len(selected),
                        "applicable_count": int(applicable.sum()),
                        "either_metric_failure_count": int(failed.sum()),
                        "either_metric_failure_fraction_all_cases": float(failed.mean()),
                        **{f"{metric}_failure_count": int(selected[
                            f"{party}_{window}_{metric}_failed"].sum()) for metric in METRICS},
                    })
    means = means.merge(pd.DataFrame(stage_rows),
                        on=["group_dimension", "group_value", "control"], validate="one_to_one")
    qualified_pairs = qualified_pairs[qualified_pairs.group_dimension.isin(("overall", "climate"))]
    return means, qualified_pairs, pd.DataFrame(window_rows)


def _baseline_pairs(frame):
    rows = []
    for dimension, value, group in _groups(frame):
        uniform = group[group.control.eq(CONTROLS[0])]
        for control in CONTROLS[1:]:
            alternative = group[group.control.eq(control)]
            paired = alternative.merge(uniform, on=PAIR_KEYS, suffixes=("_alt", "_uniform"),
                                       validate="one_to_one")
            count = len(paired)
            both = paired.baseline_service_qualified_alt & paired.baseline_service_qualified_uniform
            discordant = paired.baseline_service_qualified_alt ^ paired.baseline_service_qualified_uniform
            alt_response = paired.event_mean_power_reduction_kw_alt.clip(lower=0).where(
                paired.qualified_positive_event_response_alt, 0.0)
            uniform_response = paired.event_mean_power_reduction_kw_uniform.clip(lower=0).where(
                paired.qualified_positive_event_response_uniform, 0.0)
            difference = alt_response - uniform_response
            common_contribution = float(difference[both].sum() / count)
            discordant_contribution = float(difference[discordant].sum() / count)
            full_difference = float(difference.mean())
            for kind, subset in (("pairwise_common_baseline", both),
                                 ("all_controls_common_baseline", paired._all_controls_baseline_alt)):
                subset_count = int(subset.sum())
                row = {
                    "group_dimension": dimension, "group_value": value,
                    "alternative_control": control, "subset_kind": kind,
                    "full_pair_count": count, "subset_pair_count": subset_count,
                    "subset_fraction_of_full_cases": subset_count / count,
                    "full_zero_filled_mean_difference_kw": full_difference,
                    "pairwise_both_baseline_count": int(both.sum()),
                    "pairwise_discordant_baseline_count": int(discordant.sum()),
                    "pairwise_both_baseline_difference_contribution_kw": common_contribution,
                    "pairwise_discordant_baseline_difference_contribution_kw": discordant_contribution,
                    "baseline_partition_identity_residual_kw": (
                        full_difference - common_contribution - discordant_contribution),
                    "subset_zero_filled_mean_difference_kw": (
                        float(difference[subset].mean()) if subset_count else None),
                    "subset_difference_contribution_to_full_mean_kw": float(difference[subset].sum() / count),
                    "outside_subset_difference_contribution_to_full_mean_kw": float(difference[~subset].sum() / count),
                }
                for suffix, label, response in (("alt", "alternative", alt_response),
                                                ("uniform", "uniform", uniform_response)):
                    row[f"{label}_raw_signed_mean_kw"] = (
                        float(paired.loc[subset, f"event_mean_power_reduction_kw_{suffix}"].mean())
                        if subset_count else None)
                    row[f"{label}_zero_filled_mean_kw"] = (
                        float(response[subset].mean()) if subset_count else None)
                    for flag in ("action_event_service_qualified", "action_non_event_service_qualified",
                                 "service_qualified", "qualified_positive_event_response"):
                        passed = int(paired.loc[subset, f"{flag}_{suffix}"].sum())
                        row[f"{label}_{flag}_count"] = passed
                        row[f"{label}_{flag}_fraction"] = passed / subset_count if subset_count else None
                rows.append(row)
    return pd.DataFrame(rows)


def analyse(scores: pd.DataFrame):
    """Return full-denominator means, overlaps, and matched baseline subsets."""
    means, qualified_pairs, failures, baseline_pairs = [], [], [], []
    previous = None
    for delta in DELTAS:
        frame = rescore(scores, delta)
        if previous is not None and (previous & ~frame[list(FLAGS)]).any().any():
            raise ValueError("Increasing the additive allowance reduced qualification")
        previous = frame[list(FLAGS)].copy()
        summary, paired, overlap = _summaries(frame)
        for destination, table in ((means, summary), (qualified_pairs, paired),
                                   (failures, overlap), (baseline_pairs, _baseline_pairs(frame))):
            table.insert(0, "delta_c", delta)
            destination.append(table)
    return tuple(pd.concat(tables, ignore_index=True)
                 for tables in (means, failures, baseline_pairs, qualified_pairs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    means, failures, baseline_pairs, qualified_pairs = analyse(scores)
    overall = means[means.group_dimension.eq("overall")]
    strict = overall[overall.delta_c.eq(0)].set_index("control")
    if not overall.case_count.eq(2160).all():
        raise ValueError("Expected 2,160 equally weighted cases per control at every allowance")
    if strict.loc[list(CONTROLS), "qualified_positive_count"].tolist() != [1225, 1655, 1620]:
        raise ValueError("Strict qualified-positive counts differ from the primary analysis")
    errors = {
        "mean_identity_kw": float(means.mean_identity_residual_kw.abs().max()),
        "qualification_pair_identity_kw": float(qualified_pairs.paired_identity_residual_kw.abs().max()),
        "baseline_partition_identity_kw": float(baseline_pairs.baseline_partition_identity_residual_kw.abs().max()),
    }
    if max(errors.values()) > 1e-12:
        raise ValueError("A mean or matched-subset decomposition identity failed")
    files = {"RESPONSE_QUALIFICATION_SENSITIVITY.csv": means,
             "RESPONSE_WINDOW_FAILURE_OVERLAPS.csv": failures,
             "RESPONSE_COMMON_BASELINE_PAIRS.csv": baseline_pairs,
             "RESPONSE_QUALIFIED_PAIRS.csv": qualified_pairs}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in files.items():
        table.to_csv(args.output / name, index=False, float_format="%.17g")
    report = {
        "status": "PASS", "source": args.scores.name,
        "source_sha256": hashlib.sha256(args.scores.read_bytes()).hexdigest(),
        "primary_rows": 6480, "cases_per_control": 2160, "delta_c": DELTAS,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__},
        "allowance_rule": "For baseline and action, both population and fixed-tail mismatch must be finite and <= reference mismatch + delta C + 1e-9 in each applicable window. The additive grid is a sensitivity analysis, not a comfort-standard tolerance.",
        "windows": "Event 2 h; recovery 48 h; preceding 2 h also checked when reference occupied profile-hours > 0, irrespective of the action name.",
        "response_rule": "Retain original event whole-facility electricity reduction / event duration and original positivity (event energy reduction > 1e-9); no trajectory, power, or action changes.",
        "exclusive_stage_order": list(STAGES),
        "stage_interpretation": "Assign the first failing stage. Nonpositive cases can already be counted in a preceding service-failure stage; the final nonpositive stage contains only all-service-pass cases. raw_nonpositive_count separately counts all nonpositive cases. Window failures overlap across parties, windows and metrics.",
        "weighting": "Main and climate means equally weight all 2160 or 240 event-action cases per control; failed-service or nonpositive cases contribute zero. Conditional means and common-baseline fractions use the explicitly reported subset denominator.",
        "common_baseline_subsets": "Pairwise common baseline passes both compared controls; all-controls common baseline passes all three. At each delta the subset is reselected from the same matched contexts. These are conditional descriptive comparisons, not randomized causal effects.",
        "baseline_partition_identity": "Full alternative-minus-uniform zero-filled mean = contribution from pairwise both-baseline-pass cases + contribution from discordant-baseline cases, each divided by the full matched case count; neither-baseline-pass cases contribute zero.",
        "zero_delta_validation": "All five original baseline/event/non-event/service/qualified-positive flags reproduced row by row; original positivity and precondition applicability also reproduced.",
        "max_identity_errors": errors,
        "overall": overall.to_dict("records"),
        "output_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                          for name in files},
    }
    path = args.output / "RESPONSE_QUALIFICATION_SENSITIVITY.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output),
                      "delta_zero_counts": strict.qualified_positive_count.to_dict(),
                      "max_identity_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
