"""Weighted summaries and nested variance decomposition for Result 1."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must lie in [0, 1]")
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    threshold = quantile * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), threshold, side="left")])


def interpolate_service_risk(
    distance: np.ndarray, curve_distance: np.ndarray, curve_risk: np.ndarray
) -> np.ndarray:
    return np.interp(
        np.asarray(distance, dtype=float),
        np.asarray(curve_distance, dtype=float),
        np.asarray(curve_risk, dtype=float),
        left=float(curve_risk[0]),
        right=float(curve_risk[-1]),
    )


def nested_variance_decomposition(
    frame: pd.DataFrame,
    value: str,
    levels: tuple[str, ...] = ("country", "city", "building", "unit"),
) -> pd.DataFrame:
    """Sequential law-of-total-variance decomposition with equal record weights.

    Components are increments in explained variance from nested conditional means;
    the final residual is within the deepest level. The components close exactly to
    population variance apart from floating-point error.
    """

    data = frame.dropna(subset=[value, *levels]).copy()
    y = data[value].to_numpy(dtype=float)
    total = float(np.var(y, ddof=0))
    previous = 0.0
    rows: list[dict[str, float | str]] = []
    for index, level in enumerate(levels):
        keys = list(levels[: index + 1])
        conditional = data.groupby(keys, sort=True)[value].transform("mean").to_numpy()
        explained = float(np.var(conditional, ddof=0))
        component = max(0.0, explained - previous)
        rows.append(
            {
                "component": level,
                "variance": component,
                "share": component / total if total > 0 else 0.0,
            }
        )
        previous = explained
    residual = max(0.0, total - previous)
    rows.append(
        {
            "component": "within_unit",
            "variance": residual,
            "share": residual / total if total > 0 else 0.0,
        }
    )
    return pd.DataFrame(rows)


def weighted_nested_variance_decomposition(
    frame: pd.DataFrame,
    *,
    value: str,
    weight: str,
    levels: tuple[str, ...],
) -> pd.DataFrame:
    """Decompose weighted variance across ordered nested levels.

    Each component is the weighted squared difference between a level mean and
    its parent mean.  This is the frozen Result-1 estimator and closes exactly
    (apart from floating-point error) to the weighted population variance.
    """

    data = frame[[value, weight, *levels]].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.loc[data[weight].gt(0)].copy()
    y = data[value].to_numpy(dtype=float)
    w = data[weight].to_numpy(dtype=float)
    total_weight = float(w.sum())
    grand = weighted_mean(y, w)
    means: list[np.ndarray] = []
    for index in range(len(levels)):
        keys = list(levels[: index + 1])
        grouped = data[keys].copy()
        grouped["_w"] = w
        grouped["_wy"] = w * y
        summaries = grouped.groupby(keys, sort=False, dropna=False, as_index=False).agg(
            _w=("_w", "sum"), _wy=("_wy", "sum")
        )
        summaries["_mean"] = summaries["_wy"] / summaries["_w"]
        mapped = data[keys].merge(
            summaries[keys + ["_mean"]], on=keys, how="left", sort=False
        )
        means.append(mapped["_mean"].to_numpy(dtype=float))

    total = float(np.sum(w * (y - grand) ** 2) / total_weight)
    rows: list[dict[str, float | str]] = []
    parent = np.full(len(data), grand, dtype=float)
    for level, mean in zip(levels, means, strict=True):
        variance = float(np.sum(w * (mean - parent) ** 2) / total_weight)
        rows.append(
            {
                "component": level,
                "variance_c2": variance,
                "share": variance / total if total > 0 else 0.0,
            }
        )
        parent = mean
    residual = float(np.sum(w * (y - parent) ** 2) / total_weight)
    rows.append(
        {
            "component": "within_unit",
            "variance_c2": residual,
            "share": residual / total if total > 0 else 0.0,
        }
    )
    result = pd.DataFrame(rows)
    result["total_variance_c2"] = total
    result["closure_error_c2"] = float(result["variance_c2"].sum() - total)
    return result


def archive_scope(frame: pd.DataFrame) -> dict[str, int]:
    aliases = {
        "units": ("unit", "release_unit_id"),
        "buildings": ("building", "release_building_group_id"),
    }
    columns = {}
    for label, candidates in aliases.items():
        columns[label] = next((item for item in candidates if item in frame), None)
        if columns[label] is None:
            raise KeyError(f"missing {label} identifier")
    return {
        "records": int(len(frame)),
        "units": int(frame[columns["units"]].nunique()),
        "buildings": int(frame[columns["buildings"]].nunique()),
        "cities": int(frame[["country", "city"]].drop_duplicates().shape[0]),
        "countries": int(frame["country"].nunique()),
    }
