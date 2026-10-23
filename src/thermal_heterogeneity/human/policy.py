"""Policy-range comparison using nested cluster resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values < low, "below", np.where(values > high, "above", "within"))


def proportions(labels: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels)
    weights = np.ones(len(labels), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("weights must have positive sum")
    return {name: float(weights[labels == name].sum() / denominator) for name in ("below", "within", "above")}


def coverage(frame: pd.DataFrame, value: str, low: str, high: str, weight: str | None = None) -> dict[str, float]:
    labels = classify(frame[value].to_numpy(), frame[low].to_numpy(), frame[high].to_numpy())
    weights = None if weight is None else frame[weight].to_numpy()
    return proportions(labels, weights)


def nested_bootstrap_coverage(
    frame: pd.DataFrame,
    *,
    value: str,
    low: str,
    high: str,
    source: str,
    building: str,
    unit: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Resample source, then buildings, then units with replacement."""

    rng = np.random.default_rng(seed)
    sources = np.array(sorted(frame[source].dropna().unique()))
    rows = []
    for repetition in range(repeats):
        selected = []
        for selected_source in rng.choice(sources, len(sources), replace=True):
            source_frame = frame[frame[source] == selected_source]
            buildings = np.array(sorted(source_frame[building].dropna().unique()))
            for selected_building in rng.choice(buildings, len(buildings), replace=True):
                building_frame = source_frame[source_frame[building] == selected_building]
                units = np.array(sorted(building_frame[unit].dropna().unique()))
                for selected_unit in rng.choice(units, len(units), replace=True):
                    selected.append(building_frame[building_frame[unit] == selected_unit])
        sample = pd.concat(selected, ignore_index=True)
        rows.append({"repetition": repetition, **coverage(sample, value, low, high)})
    return pd.DataFrame(rows)
