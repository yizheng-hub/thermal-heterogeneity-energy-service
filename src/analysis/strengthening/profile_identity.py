"""Use verified raw-source identity keys in paired sensitivity draws."""

from common import ROOT
from functools import lru_cache
import pandas as pd


@lru_cache(maxsize=1)
def identity_map():
    frame = pd.read_csv(
        ROOT / "audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz", low_memory=False
    )
    pairs = frame[["profile_id", "person_key"]].drop_duplicates()
    if pairs.profile_id.duplicated().any():
        raise ValueError("One simulated profile maps to conflicting raw identities")
    return pairs.set_index("profile_id").person_key.to_dict()


def person_keys(profile_ids):
    mapping = identity_map()
    return [mapping[str(profile)] for profile in profile_ids]


@lru_cache(maxsize=1)
def source_map():
    frame = pd.read_csv(ROOT / "audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz", low_memory=False)
    pairs = frame[["profile_id", "source_key"]].drop_duplicates()
    if pairs.source_key.isna().any() or pairs.profile_id.duplicated().any():
        raise ValueError("Simulated profiles have missing or conflicting source groups")
    return pairs.set_index("profile_id").source_key.to_dict()


def source_keys(profile_ids):
    mapping = source_map()
    return [mapping[str(profile)] for profile in profile_ids]
