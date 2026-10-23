"""Fold-isolated safety margins for anonymous quantile representations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def higher_quantile(values: pd.Series | np.ndarray, quantile: float) -> float:
    array = np.sort(np.asarray(values, dtype=float))
    if len(array) == 0:
        raise ValueError("cannot calibrate an empty margin")
    rank = int(np.ceil(quantile * len(array))) - 1
    return float(array[max(0, min(rank, len(array) - 1))])


def calibrate_margins(
    training: pd.DataFrame,
    *,
    approximate_mean: str,
    approximate_tail: str,
    full_mean: str,
    full_tail: str,
    quantile: float,
) -> tuple[float, float]:
    mean_error = np.maximum(0.0, training[full_mean] - training[approximate_mean])
    tail_error = np.maximum(0.0, training[full_tail] - training[approximate_tail])
    return higher_quantile(mean_error, quantile), higher_quantile(tail_error, quantile)


def certify(
    frame: pd.DataFrame,
    *,
    mean_margin: float,
    tail_margin: float,
    mean_score: str = "mean_c",
    tail_score: str = "fixed_tail_c",
    mean_budget: str = "mean_budget_c",
    tail_budget: str = "fixed_tail_budget_c",
    tolerance: float = 1e-9,
) -> pd.Series:
    return (frame[mean_score] + mean_margin).le(frame[mean_budget] + tolerance) & (
        frame[tail_score] + tail_margin
    ).le(frame[tail_budget] + tolerance)


def certification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    certified = frame["certified_eligible"].astype(bool)
    full = frame["full_service_eligible"].astype(bool)
    return {
        "false_accept": int((certified & ~full).sum()),
        "false_reject": int((~certified & full).sum()),
        "full_qualified": int(full.sum()),
        "retained_full_qualified": int((certified & full).sum()),
        "retention": float((certified & full).sum() / max(1, full.sum())),
    }
