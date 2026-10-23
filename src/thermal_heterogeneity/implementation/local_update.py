"""Anonymous local-feedback updating and learning-curve metrics."""

from __future__ import annotations

import hashlib

import numpy as np


def deterministic_rng(seed: int, split: str, held_out: str, repetition: int) -> np.random.Generator:
    token = f"{seed}|{split}|{held_out}|{repetition}".encode()
    derived = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(derived)


def local_intercept_update(
    feedback_classes: np.ndarray,
    evaluation_probability: np.ndarray,
    *,
    prior_strength: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(evaluation_probability, dtype=float)
    classes = probability.shape[1]
    prior = probability.mean(axis=0)
    counts = np.bincount(np.asarray(feedback_classes, dtype=int), minlength=classes)
    posterior = (prior_strength * prior + counts) / (prior_strength + counts.sum())
    ratio = posterior / np.clip(prior, 1e-15, None)
    updated = probability * ratio
    updated /= updated.sum(axis=1, keepdims=True)
    return updated, posterior


def crps_ordinal(y: np.ndarray, probability: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Return the legacy summed ordinal RPS, a dimensionless score.

    This helper keeps its original scale for API compatibility. The paper's
    Level-1 ``_preference_score`` averages over cumulative class boundaries.
    With three classes this helper returns twice the normalized paper RPS.
    """
    probability = np.asarray(probability, dtype=float)
    observed = np.eye(probability.shape[1])[np.asarray(y, dtype=int)]
    row = np.sum((np.cumsum(probability, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]) ** 2, axis=1)
    weights = np.ones(len(row)) if weights is None else np.asarray(weights, dtype=float)
    return float(np.average(row, weights=weights))
