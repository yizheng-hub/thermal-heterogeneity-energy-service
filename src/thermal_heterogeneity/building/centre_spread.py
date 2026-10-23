"""Orthogonal demand-centre and within-population spread effects."""

from __future__ import annotations

import pandas as pd


def decompose_two_by_two(
    frame: pd.DataFrame,
    *,
    keys: list[str],
    centre: str = "delta",
    spread: str = "lambda",
    outcome: str = "value",
    reference_centre: float = 0.0,
    reference_spread: float = 0.0,
) -> pd.DataFrame:
    """Return centre, spread and interaction contrasts from a full factorial grid."""

    rows = []
    grouping = keys if keys else lambda _: 0
    for key, group in frame.groupby(grouping, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(keys, key_tuple, strict=True)) if keys else {}
        indexed = group.set_index([centre, spread])[outcome]
        baseline = float(indexed.loc[(reference_centre, reference_spread)])
        for (centre_value, spread_value), value in indexed.sort_index().items():
            centre_only = float(indexed.loc[(centre_value, reference_spread)]) - baseline
            spread_only = float(indexed.loc[(reference_centre, spread_value)]) - baseline
            total = float(value) - baseline
            rows.append(
                key_values
                | {
                    centre: centre_value,
                    spread: spread_value,
                    "baseline": baseline,
                    "centre_effect": centre_only,
                    "spread_effect": spread_only,
                    "interaction": total - centre_only - spread_only,
                    "total_effect": total,
                }
            )
    return pd.DataFrame(rows)


def direction_counts(frame: pd.DataFrame, value: str, tolerance: float = 1e-12) -> dict[str, int]:
    return {
        "decrease": int(frame[value].lt(-tolerance).sum()),
        "tie": int(frame[value].abs().le(tolerance).sum()),
        "increase": int(frame[value].gt(tolerance).sum()),
    }
