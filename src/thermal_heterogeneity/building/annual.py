"""Annual service qualification and deterministic action selection."""

from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_TOL_C = 1e-9
ENERGY_ABS_TOL_KWH = 1e-6
ENERGY_REL_TOL = 1e-9


def energy_tolerance(a: float, b: float) -> float:
    return max(ENERGY_ABS_TOL_KWH, ENERGY_REL_TOL * max(abs(a), abs(b)))


def qualify(
    frame: pd.DataFrame,
    *,
    mean_score: str = "mean_c",
    tail_score: str = "fixed_tail_c",
    mean_budget: str = "mean_budget_c",
    tail_budget: str = "fixed_tail_budget_c",
    tolerance: float = SCORE_TOL_C,
) -> pd.Series:
    return frame[mean_score].le(frame[mean_budget] + tolerance) & frame[tail_score].le(
        frame[tail_budget] + tolerance
    )


def choose_minimum_energy(
    frame: pd.DataFrame,
    *,
    eligible: str = "service_eligible",
    energy: str = "site_kwh",
    action: str = "physical_action_id",
) -> pd.Series | None:
    pool = frame.loc[frame[eligible]].sort_values([energy, action], kind="mergesort")
    if pool.empty:
        return None
    minimum = float(pool.iloc[0][energy])
    tied = pool[pool[energy].le(minimum + energy_tolerance(minimum, minimum))]
    return tied.sort_values(action, kind="mergesort").iloc[0]


def compare_representations(
    scores: pd.DataFrame,
    *,
    decision_keys: list[str],
    representation: str = "representation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare compressed and full qualification over one common action library."""

    required = {"compressed", "full"}
    eligibility_rows = []
    decision_rows = []
    for key, group in scores.groupby(decision_keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        key_values = dict(zip(decision_keys, key, strict=True))
        pools: dict[str, pd.DataFrame] = {}
        choices: dict[str, pd.Series | None] = {}
        for label in sorted(required):
            pool = group[group[representation] == label].copy()
            if pool.empty:
                raise ValueError(f"missing {label} representation for {key_values}")
            pool["service_eligible"] = qualify(pool)
            pools[label] = pool.set_index("physical_action_id", drop=False).sort_index()
            choices[label] = choose_minimum_energy(pool)
            eligibility_rows.append(pool.assign(**key_values))
        if not pools["compressed"].index.equals(pools["full"].index):
            raise ValueError(f"action libraries differ for {key_values}")
        compressed = pools["compressed"]["service_eligible"]
        full = pools["full"]["service_eligible"]
        comp_choice = choices["compressed"]
        full_choice = choices["full"]
        decision_rows.append(
            key_values
            | {
                "compressed_feasible_candidates": int(compressed.sum()),
                "full_feasible_candidates": int(full.sum()),
                "candidate_false_accept_count": int((compressed & ~full).sum()),
                "candidate_false_reject_count": int((~compressed & full).sum()),
                "compressed_selected_action": None if comp_choice is None else comp_choice["physical_action_id"],
                "full_selected_action": None if full_choice is None else full_choice["physical_action_id"],
                "compressed_selected_false_accept": bool(
                    comp_choice is not None
                    and not bool(full.loc[comp_choice["physical_action_id"]])
                ),
                "selection_changed": bool(
                    comp_choice is not None
                    and full_choice is not None
                    and comp_choice["physical_action_id"] != full_choice["physical_action_id"]
                ),
                "energy_correction_kwh": float("nan")
                if comp_choice is None or full_choice is None
                else float(full_choice["site_kwh"] - comp_choice["site_kwh"]),
            }
        )
    return pd.DataFrame(decision_rows), pd.concat(eligibility_rows, ignore_index=True)
