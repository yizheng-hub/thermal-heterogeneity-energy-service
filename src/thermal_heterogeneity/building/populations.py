"""Deterministic anonymous population assignment helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def population_rng(seed: int, population: str, replicate: int) -> np.random.Generator:
    token = f"{seed}|{population}|{replicate}".encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(derived)


def sample_profiles(
    frame: pd.DataFrame,
    *,
    n: int,
    seed: int,
    population: str,
    replicate: int,
    weight: str | None = None,
) -> pd.DataFrame:
    rng = population_rng(seed, population, replicate)
    probabilities = None
    if weight is not None:
        probabilities = frame[weight].to_numpy(dtype=float)
        probabilities = probabilities / probabilities.sum()
    positions = rng.choice(len(frame), size=n, replace=True, p=probabilities)
    return frame.iloc[positions].reset_index(drop=True)
