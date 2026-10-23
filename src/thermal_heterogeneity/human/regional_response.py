"""Held-out regional-response evaluation without local-path assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regional_helpers import ranked_probability_score, weighted_log_loss


def balanced_weights(frame: pd.DataFrame, group: str) -> np.ndarray:
    counts = frame.groupby(group, sort=True)[group].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def score_predictions(
    frame: pd.DataFrame,
    *,
    outcome: str,
    probability_columns: list[str],
    class_order: list[str],
    weight_column: str | None = None,
) -> dict[str, float]:
    lookup = {name: index for index, name in enumerate(class_order)}
    y = frame[outcome].map(lookup).to_numpy(dtype=int)
    probability = frame[probability_columns].to_numpy(dtype=float)
    weights = np.ones(len(frame)) if weight_column is None else frame[weight_column].to_numpy(dtype=float)
    return {
        "log_loss": weighted_log_loss(y, probability, weights),
        "ranked_probability_score": ranked_probability_score(y, probability, weights),
    }


def held_out_summary(frame: pd.DataFrame, split: str, score: str) -> pd.DataFrame:
    return (
        frame.groupby([split, "model"], sort=True)[score]
        .mean()
        .rename("score")
        .reset_index()
        .sort_values([split, "model"], kind="mergesort")
        .reset_index(drop=True)
    )
