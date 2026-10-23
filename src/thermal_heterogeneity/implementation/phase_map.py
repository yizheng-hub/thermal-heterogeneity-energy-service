"""Mean-sufficiency phase classification and anonymous quantile approximations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_approximation(
    profiles: pd.DataFrame,
    *,
    value: str,
    bins: int,
    weight: str = "weight",
) -> pd.DataFrame:
    """Summarize near-equal-count rank bins with their weighted means.

    Stable value sorting retains the input order for ties. Callers reproducing
    the paper's centre/anonymous-ID order must order tied rows by that ID first.
    Positive input weights determine bin means and normalized bin weights;
    unit weights give the paper's arithmetic means and population-count weights.
    This auxiliary summary does not encode fixed-tail membership counts.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    if bins > len(profiles):
        raise ValueError("bins must not exceed the number of profiles")
    values = profiles[value].to_numpy(dtype=float)
    weights = profiles[weight].to_numpy(dtype=float)
    order = np.argsort(values, kind="mergesort")
    groups = np.array_split(order, bins)
    means = [np.average(values[group], weights=weights[group]) for group in groups]
    bin_weights = np.array([weights[group].sum() for group in groups])
    return pd.DataFrame(
        {
            value: means,
            weight: bin_weights / weights.sum(),
            "quantile_bin": np.arange(1, bins + 1),
        }
    )


def classify_cell(
    *,
    mean_available: bool,
    full_available: bool,
    selection_matches: bool,
) -> str:
    if not full_available:
        return "finite_library_unresolved"
    if mean_available and selection_matches:
        return "mean_safe"
    return "distribution_required"


def classify_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["classification"] = [
        classify_cell(
            mean_available=bool(mean),
            full_available=bool(full),
            selection_matches=bool(match),
        )
        for mean, full, match in zip(
            output["mean_available"],
            output["full_available"],
            output["selection_matches"],
            strict=True,
        )
    ]
    return output


def universal_threshold(
    frame: pd.DataFrame,
    *,
    heterogeneity: str = "heterogeneity_sd_c",
    label: str = "distribution_required",
    context: list[str] | None = None,
) -> dict[str, float | bool]:
    y = frame[label].to_numpy(dtype=bool)
    values = frame[heterogeneity].to_numpy(dtype=float)
    candidates = np.unique(values)
    rows = []
    for threshold in candidates:
        predicted = values > threshold
        error = float(np.mean(predicted != y))
        sensitivity = float((predicted & y).sum() / max(1, y.sum()))
        specificity = float((~predicted & ~y).sum() / max(1, (~y).sum()))
        worst = error
        if context:
            temp = frame.assign(_wrong=predicted != y)
            worst = float(temp.groupby(context, sort=True)["_wrong"].mean().max())
        rows.append((error, worst, float(threshold), sensitivity, specificity))
    error, worst, threshold, sensitivity, specificity = min(rows)
    return {
        "threshold_c": threshold,
        "error_rate": error,
        "worst_context_error_rate": worst,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accepted": bool(
            error <= 0.05 and worst <= 0.10 and sensitivity >= 0.9 and specificity >= 0.9
        ),
    }
