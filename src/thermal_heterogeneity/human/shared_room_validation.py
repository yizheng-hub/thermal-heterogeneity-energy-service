"""Direct shared-room vote diagnostics for Result 2."""

from __future__ import annotations

import numpy as np
import pandas as pd


VOTE_ORDER = (-1.0, 0.0, 1.0)


def window_statistics(votes: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    votes = np.asarray(votes, dtype=float)
    weights = np.ones(len(votes), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    compressed = abs(float(np.sum(votes * weights)))
    full = float(np.sum(np.abs(votes) * weights))
    return {
        "n_votes": int(len(votes)),
        "compressed_burden": compressed,
        "full_burden": full,
        "jensen_gap": full - compressed,
        "disagreement": bool(np.unique(votes).size > 1),
        "opposing_directions": bool((votes < 0).any() and (votes > 0).any()),
    }


def build_windows(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    vote_column: str,
    weight_column: str | None = None,
) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        values = group[vote_column].to_numpy(dtype=float)
        weights = None if weight_column is None else group[weight_column].to_numpy(dtype=float)
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(zip(group_columns, key, strict=True)) | window_statistics(values, weights))
    return pd.DataFrame(rows)


def threshold_table(windows: pd.DataFrame, thresholds: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        compressed = windows["compressed_burden"] <= threshold
        full = windows["full_burden"] <= threshold
        rows.append(
            {
                "threshold": threshold,
                "n_windows": int(len(windows)),
                "false_accept": int((compressed & ~full).sum()),
                "false_reject": int((~compressed & full).sum()),
            }
        )
    return pd.DataFrame(rows)
