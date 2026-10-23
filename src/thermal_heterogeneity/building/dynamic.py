"""Event, recovery and 50-hour flexibility qualification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .annual import SCORE_TOL_C


DYNAMIC_SCORE_COLUMNS = (
    "event_baseline_mean_c",
    "event_baseline_fixed_tail_c",
    "event_perturbed_mean_c",
    "event_perturbed_fixed_tail_c",
    "recovery_baseline_mean_c",
    "recovery_baseline_fixed_tail_c",
    "recovery_perturbed_mean_c",
    "recovery_perturbed_fixed_tail_c",
)


def event_mask(hours: pd.Series, event_start_hour: int = 14, duration_hours: int = 2) -> np.ndarray:
    local_hour = pd.to_datetime(hours).dt.hour.to_numpy()
    return (local_hour >= event_start_hour) & (local_hour < event_start_hour + duration_hours)


def qualify(frame: pd.DataFrame, tolerance: float = SCORE_TOL_C) -> pd.Series:
    checks = []
    for prefix in ("event", "recovery"):
        for trajectory in ("baseline", "perturbed"):
            checks.append(
                frame[f"{prefix}_{trajectory}_mean_c"].le(
                    frame[f"{prefix}_mean_budget_c"] + tolerance
                )
            )
            checks.append(
                frame[f"{prefix}_{trajectory}_fixed_tail_c"].le(
                    frame[f"{prefix}_fixed_tail_budget_c"] + tolerance
                )
            )
    return pd.concat(checks, axis=1).all(axis=1)


def choose_maximum_response(
    frame: pd.DataFrame,
    *,
    response: str = "event_average_reduction_kw",
    action: str = "event_action_id",
    tolerance: float = 1e-9,
) -> pd.Series | None:
    pool = frame[frame["service_eligible"] & frame[response].gt(tolerance)]
    if pool.empty:
        return None
    maximum = float(pool[response].max())
    tied = pool[pool[response].ge(maximum - tolerance)]
    return tied.sort_values(action, kind="mergesort").iloc[0]


def window_endpoints(
    frame: pd.DataFrame,
    *,
    event: np.ndarray,
    recovery: np.ndarray,
    power_column: str,
    step_hours: float,
) -> dict[str, float]:
    power = frame[power_column].to_numpy(dtype=float)
    return {
        "event_average_kw": float(power[event].mean()),
        "event_energy_kwh": float(power[event].sum() * step_hours),
        "recovery_energy_kwh": float(power[recovery].sum() * step_hours),
        "event_plus_recovery_energy_kwh": float(power[event | recovery].sum() * step_hours),
    }


def compare_representations(
    scores: pd.DataFrame,
    *,
    decision_keys: list[str],
    representation: str = "representation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eligibility = []
    for key, group in scores.groupby(decision_keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(decision_keys, key, strict=True))
        pools = {}
        selected = {}
        for label in ("compressed", "full"):
            pool = group[group[representation] == label].copy()
            pool["service_eligible"] = qualify(pool)
            pool["positive_response"] = pool["event_average_reduction_kw"].gt(1e-9)
            pools[label] = pool.set_index("event_action_id", drop=False).sort_index()
            selected[label] = choose_maximum_response(pool)
            eligibility.append(pool.assign(**key_values))
        if not pools["compressed"].index.equals(pools["full"].index):
            raise ValueError(f"dynamic action libraries differ for {key_values}")
        compressed = pools["compressed"]["service_eligible"]
        full = pools["full"]["service_eligible"]
        compressed_selected = selected["compressed"]
        full_selected = selected["full"]
        rows.append(
            key_values
            | {
                "candidate_false_accept_count": int((compressed & ~full).sum()),
                "candidate_false_reject_count": int((~compressed & full).sum()),
                "compressed_selected_action": None
                if compressed_selected is None
                else compressed_selected["event_action_id"],
                "full_selected_action": None
                if full_selected is None
                else full_selected["event_action_id"],
                "compressed_selected_false_accept": bool(
                    compressed_selected is not None
                    and not bool(full.loc[compressed_selected["event_action_id"]])
                ),
                "response_correction_kw": float("nan")
                if compressed_selected is None or full_selected is None
                else float(
                    full_selected["event_average_reduction_kw"]
                    - compressed_selected["event_average_reduction_kw"]
                ),
            }
        )
    return pd.DataFrame(rows), pd.concat(eligibility, ignore_index=True)
