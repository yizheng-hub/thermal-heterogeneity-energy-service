"""Decompose strict event-response means using the retained event qualifications."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = ("uniform_average", "personal_center", "personal_range")
PAIR_KEYS = ["climate", "composition_id", "event_id", "action_id"]


def _boolean(values: pd.Series, name: str) -> pd.Series:
    parsed = values.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"Invalid Boolean values in {name}")
    return parsed.astype(bool)


def _groups(frame):
    yield "overall", "all", frame
    for dimension in ("climate", "action_id"):
        for value, group in frame.groupby(dimension, sort=True):
            yield dimension, str(value), group


def _mean_row(group):
    count = len(group)
    selected = group.loc[group._qualified, "_power"]
    fraction = float(group._qualified.mean())
    conditional = float(selected.mean()) if len(selected) else None
    zero_filled = group._power.clip(lower=0).where(group._qualified, 0.0)
    product = fraction * conditional if conditional is not None else 0.0
    raw_positive = group.loc[group._positive, "_power"]
    return {
        "case_count": count,
        "service_qualified_count": int(group._service.sum()),
        "raw_positive_count": int(group._positive.sum()),
        "qualified_positive_count": len(selected),
        "qualified_positive_fraction": fraction,
        "qualified_positive_conditional_mean_kw": conditional,
        "service_preserving_zero_filled_mean_kw": float(zero_filled.mean()),
        "fraction_times_conditional_mean_kw": product,
        "mean_identity_residual_kw": float(zero_filled.mean()) - product,
        "raw_positive_zero_filled_mean_kw": float(group._power.clip(lower=0).mean()),
        "raw_positive_conditional_mean_kw": float(raw_positive.mean()) if len(raw_positive) else None,
        "raw_signed_mean_kw": float(group._power.mean()),
    }


def analyse(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {*PAIR_KEYS, "control", "event_service_limit_factor",
                "service_qualified", "positive_event_response",
                "qualified_positive_event_response", "event_mean_power_reduction_kw"}
    if missing := required - set(scores):
        raise ValueError(f"Event scores lack {sorted(missing)}")
    main = scores[np.isclose(scores.event_service_limit_factor, 1.0)].copy()
    if set(main.control) != set(CONTROLS) or main.duplicated([*PAIR_KEYS, "control"]).any():
        raise ValueError("Primary control population is incomplete or contains duplicate cases")
    main["_power"] = pd.to_numeric(main.event_mean_power_reduction_kw)
    if not np.isfinite(main._power).all():
        raise ValueError("Event power must be finite")
    for column, target in (("service_qualified", "_service"),
                           ("positive_event_response", "_positive"),
                           ("qualified_positive_event_response", "_qualified")):
        main[target] = _boolean(main[column], column)
    if not main._qualified.eq(main._service & main._positive).all():
        raise ValueError("Retained qualified-positive flags are inconsistent")
    if (main.loc[main._qualified, "_power"] <= 0).any():
        raise ValueError("A qualified-positive case has nonpositive power")

    rows, pairs = [], []
    for dimension, value, group in _groups(main):
        for control in CONTROLS:
            controlled = group[group.control.eq(control)]
            rows.append({"group_dimension": dimension, "group_value": value,
                         "control": control, **_mean_row(controlled)})
        uniform = group[group.control.eq("uniform_average")]
        for control in CONTROLS[1:]:
            alternative = group[group.control.eq(control)]
            paired = alternative.merge(uniform, on=PAIR_KEYS, suffixes=("_alt", "_uniform"),
                                       validate="one_to_one")
            if len(paired) != len(uniform) or len(paired) != len(alternative):
                raise ValueError("Control comparisons do not contain the same event-action contexts")
            alt, base = paired._qualified_alt, paired._qualified_uniform
            only_alt, only_base, both = alt & ~base, base & ~alt, alt & base
            count = len(paired)
            gain = float(paired.loc[only_alt, "_power_alt"].sum() / count)
            loss = float(-paired.loc[only_base, "_power_uniform"].sum() / count)
            common = float((paired.loc[both, "_power_alt"]
                            - paired.loc[both, "_power_uniform"]).sum() / count)
            observed = float((paired._power_alt.where(alt, 0.0)
                              - paired._power_uniform.where(base, 0.0)).mean())
            pairs.append({
                "group_dimension": dimension, "group_value": value,
                "alternative_control": control, "pair_count": count,
                "alternative_only_qualified_count": int(only_alt.sum()),
                "uniform_only_qualified_count": int(only_base.sum()),
                "both_qualified_count": int(both.sum()),
                "neither_qualified_count": int((~alt & ~base).sum()),
                "alternative_only_contribution_kw": gain,
                "uniform_only_contribution_kw": loss,
                "net_exclusive_availability_contribution_kw": gain + loss,
                "both_qualified_power_difference_contribution_kw": common,
                "both_qualified_alternative_conditional_mean_kw": (
                    float(paired.loc[both, "_power_alt"].mean()) if both.any() else None),
                "both_qualified_uniform_conditional_mean_kw": (
                    float(paired.loc[both, "_power_uniform"].mean()) if both.any() else None),
                "observed_zero_filled_mean_difference_kw": observed,
                "sum_of_components_kw": gain + loss + common,
                "paired_identity_residual_kw": observed - gain - loss - common,
            })
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.scores.read_bytes()).hexdigest()
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    means, pairs = analyse(scores)
    overall = means[means.group_dimension.eq("overall")]
    if len(overall) != 3 or not overall.case_count.eq(2160).all():
        raise ValueError("Expected 2,160 primary cases per control")
    if means.mean_identity_residual_kw.abs().max() > 1e-12 or pairs.paired_identity_residual_kw.abs().max() > 1e-12:
        raise ValueError("Mean decomposition identity failed")
    args.output.mkdir(parents=True, exist_ok=True)
    files = {"RESPONSE_MEAN_DECOMPOSITION.csv": means,
             "RESPONSE_MEAN_PAIRED_COMPONENTS.csv": pairs}
    for name, frame in files.items():
        frame.to_csv(args.output / name, index=False, float_format="%.17g")
    report = {
        "status": "PASS", "source": "EVENT_RESPONSE_SCORES.csv", "source_sha256": source_hash,
        "event_service_limit_factor": 1.0, "cases_per_control": 2160,
        "power_definition": "Whole-building event electricity reduction relative to each control's own baseline, divided by the two-hour event",
        "qualification_definition": "Use retained qualified_positive_event_response unchanged: all baseline/action population and fixed-tail service checks pass, and event electricity reduction exceeds the original tolerance",
        "service_windows": "Two-hour event and 48-hour recovery; the two-hour preceding window also enters service checks when precondition_service_evaluated is true (reference occupied profile-hours > 0)",
        "weighting": "Equal weight for every event-action case; nonqualified or nonpositive cases contribute zero",
        "raw_positive_mean_definition": "Mean of max(event power, 0) across all cases, separately from the conditional mean among retained positive cases",
        "identity": "zero-filled mean = qualified-positive fraction * conditional qualified-positive mean; empty qualified groups contribute zero and have undefined conditional means",
        "paired_identity": "Alternative-only power/N minus uniform-only power/N plus the summed power difference in both-qualified contexts/N",
        "interpretation_limit": "The group conditional means describe different qualified subsets; the both-qualified paired component compares the same event-action contexts",
        "overall": overall.to_dict("records"),
        "paired_overall": pairs[pairs.group_dimension.eq("overall")].to_dict("records"),
        "max_mean_identity_error_kw": float(means.mean_identity_residual_kw.abs().max()),
        "max_paired_identity_error_kw": float(pairs.paired_identity_residual_kw.abs().max()),
        "output_sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                          for name in files},
    }
    (args.output / "RESPONSE_MEAN_DECOMPOSITION.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
