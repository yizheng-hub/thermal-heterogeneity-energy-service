"""Three locked demand representations used by same-service comparisons."""

from __future__ import annotations

import pandas as pd


REPRESENTATIONS = ("uniform_mean", "person_centred", "ltpref_interval")


def validate_representations(frame: pd.DataFrame, column: str = "strategy") -> None:
    observed = set(frame[column].dropna().unique())
    unknown = observed - set(REPRESENTATIONS)
    if unknown:
        raise ValueError(f"unknown demand representation(s): {sorted(unknown)}")


def matched_same_service(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    validate_representations(frame)
    counts = frame.groupby(keys, sort=True)["strategy"].nunique()
    complete = counts[counts == len(REPRESENTATIONS)].index
    indexed = frame.set_index(keys)
    return indexed.loc[indexed.index.isin(complete)].reset_index().sort_values(keys + ["strategy"])
