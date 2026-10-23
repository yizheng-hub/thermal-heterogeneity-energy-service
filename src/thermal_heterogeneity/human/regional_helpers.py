"""Pure probability-scoring helpers shared by regional analyses."""

from __future__ import annotations

import numpy as np


def normalize(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-15, None)
    return probability / probability.sum(axis=1, keepdims=True)


def align_probabilities(raw: np.ndarray, fitted: list[str], expected: list[str]) -> np.ndarray:
    output = np.zeros((len(raw), len(expected)), dtype=float)
    index = {name: position for position, name in enumerate(fitted)}
    for column, name in enumerate(expected):
        if name in index:
            output[:, column] = raw[:, index[name]]
    return normalize(output)


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return normalize(np.exp(np.log(np.clip(probability, 1e-15, 1.0)) / temperature))


def weighted_log_loss(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> float:
    chosen = probability[np.arange(len(y)), np.asarray(y, dtype=int)]
    return float(np.average(-np.log(np.clip(chosen, 1e-15, 1.0)), weights=weights))


def ranked_probability_score(y: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> float:
    classes = probability.shape[1]
    observed = np.eye(classes)[np.asarray(y, dtype=int)]
    score = np.sum((np.cumsum(probability, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]) ** 2, axis=1)
    return float(np.average(score, weights=weights))
