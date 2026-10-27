"""Observation-based seasonal demand adjustments for sensitivity analysis.

The annual experiment treats each demand profile as fixed.  This module adds a
separate sensitivity in which warm- and cool-season demand coordinates move by
an observed within-place contrast.  Centres and both Personal range bounds
always move together, so the estimated interval width and profile ordering are
not changed by the sensitivity.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def estimate_within_place_warm_cool_contrast(
    observations: pd.DataFrame,
    *,
    place_columns: Sequence[str],
    season_column: str = "season",
    centre_column: str = "centre_c",
    warm_labels: Iterable[str] = ("summer", "warm"),
    cool_labels: Iterable[str] = ("winter", "cool"),
    bootstrap_replicates: int = 0,
    seed: int = 0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Estimate a fixed-effect warm-minus-cool contrast within places.

    Each eligible place must contain observations in both season groups.  The
    pooled coefficient is the ordinary least-squares coefficient for a binary
    warm-season indicator after removing place means.  Equivalently, place
    contrasts are weighted by ``n_warm * n_cool / (n_warm + n_cool)``.
    Optional uncertainty resamples whole places rather than individual rows.
    """
    required = {*place_columns, season_column, centre_column}
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Seasonal observations are missing columns: {sorted(missing)}")
    if not place_columns:
        raise ValueError("At least one place column is required")

    warm = {str(value).strip().lower() for value in warm_labels}
    cool = {str(value).strip().lower() for value in cool_labels}
    if not warm or not cool or warm & cool:
        raise ValueError("Warm and cool season labels must be nonempty and disjoint")

    frame = observations[[*place_columns, season_column, centre_column]].copy()
    frame["_season"] = frame[season_column].astype(str).str.strip().str.lower()
    frame["_group"] = np.where(
        frame["_season"].isin(warm), "warm",
        np.where(frame["_season"].isin(cool), "cool", ""),
    )
    frame[centre_column] = pd.to_numeric(frame[centre_column], errors="coerce")
    frame = frame[frame["_group"].ne("") & frame[centre_column].notna()]
    if frame.empty:
        raise ValueError("No finite warm- or cool-season observations remain")

    grouped = (
        frame.groupby([*place_columns, "_group"], dropna=False)[centre_column]
        .agg(["mean", "size"])
        .unstack("_group")
    )
    eligible = grouped.dropna(subset=[("mean", "warm"), ("mean", "cool")]).copy()
    if eligible.empty:
        raise ValueError("No place contains both warm- and cool-season observations")
    effects = eligible.index.to_frame(index=False)
    effects["warm_mean_c"] = eligible[("mean", "warm")].to_numpy(float)
    effects["cool_mean_c"] = eligible[("mean", "cool")].to_numpy(float)
    effects["warm_n"] = eligible[("size", "warm")].to_numpy(int)
    effects["cool_n"] = eligible[("size", "cool")].to_numpy(int)
    effects["warm_minus_cool_c"] = effects["warm_mean_c"] - effects["cool_mean_c"]
    effects["fixed_effect_weight"] = (
        effects["warm_n"] * effects["cool_n"] / (effects["warm_n"] + effects["cool_n"])
    )
    weights = effects["fixed_effect_weight"].to_numpy(float)
    contrasts = effects["warm_minus_cool_c"].to_numpy(float)
    beta = float(np.average(contrasts, weights=weights))

    summary: dict[str, float | int] = {
        "warm_minus_cool_c": beta,
        "eligible_places": int(len(effects)),
        "warm_observations": int(effects["warm_n"].sum()),
        "cool_observations": int(effects["cool_n"].sum()),
    }
    if bootstrap_replicates:
        if bootstrap_replicates < 2:
            raise ValueError("At least two bootstrap replicates are required")
        generator = np.random.default_rng(seed)
        boot = np.empty(bootstrap_replicates)
        for index in range(bootstrap_replicates):
            selected = generator.integers(0, len(effects), len(effects))
            boot[index] = np.average(contrasts[selected], weights=weights[selected])
        summary["bootstrap_p025_c"] = float(np.quantile(boot, 0.025))
        summary["bootstrap_p975_c"] = float(np.quantile(boot, 0.975))
        summary["bootstrap_replicates"] = int(bootstrap_replicates)
        summary["bootstrap_seed"] = int(seed)
    return summary, effects


def warm_cool_month_adjustments(
    months: Sequence[int] | np.ndarray,
    warm_minus_cool_c: float,
    *,
    latitude: float,
    fraction: float = 1.0,
) -> np.ndarray:
    """Return symmetric local warm/cool shifts for calendar months.

    Local summer receives ``+difference/2`` and local winter receives
    ``-difference/2``.  Spring and autumn receive zero.  Southern-hemisphere
    seasons are reversed.  ``fraction`` supports the declared zero, half and
    full observational sensitivities without changing the estimated contrast.
    """
    month = np.asarray(months, dtype=int)
    if month.ndim != 1 or ((month < 1) | (month > 12)).any():
        raise ValueError("Months must be a one-dimensional array in 1..12")
    if not np.isfinite(warm_minus_cool_c) or not np.isfinite(fraction) or fraction < 0:
        raise ValueError("Seasonal contrast and nonnegative fraction must be finite")
    northern = latitude >= 0
    warm_months = {6, 7, 8} if northern else {12, 1, 2}
    cool_months = {12, 1, 2} if northern else {6, 7, 8}
    half = 0.5 * float(warm_minus_cool_c) * float(fraction)
    return np.array([
        half if value in warm_months else -half if value in cool_months else 0.0
        for value in month
    ])


def shift_demand_coordinates(
    profiles: pd.DataFrame,
    adjustment_c: float | Sequence[float] | np.ndarray,
    *,
    centre_column: str = "centre_c",
    lower_column: str = "controller_lower_c",
    upper_column: str = "controller_upper_c",
) -> pd.DataFrame:
    """Shift centres and interval bounds together while preserving widths."""
    required = {centre_column, lower_column, upper_column}
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Demand profiles are missing columns: {sorted(missing)}")
    result = profiles.copy()
    adjustment = np.asarray(adjustment_c, dtype=float)
    if adjustment.ndim == 0:
        adjustment = np.full(len(result), float(adjustment))
    if adjustment.shape != (len(result),) or not np.isfinite(adjustment).all():
        raise ValueError("Seasonal adjustments must be finite and align with profiles")
    before_width = result[upper_column].to_numpy(float) - result[lower_column].to_numpy(float)
    for column in (centre_column, lower_column, upper_column):
        values = result[column].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Demand coordinates must be finite")
        result[column] = values + adjustment
    after_width = result[upper_column].to_numpy(float) - result[lower_column].to_numpy(float)
    if not np.allclose(before_width, after_width, rtol=0, atol=1e-12):
        raise AssertionError("Seasonal adjustment changed Personal range widths")
    result["seasonal_shift_c"] = adjustment
    return result
