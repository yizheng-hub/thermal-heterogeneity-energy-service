"""Memoize unchanged calendars and reference-tail rankings, never service scores."""

from collections import OrderedDict
import hashlib
import numpy as np
import pandas as pd


def enable(module):
    load_original = module.load_compact
    clocks_original = module.compact_timestamps
    tail_original = module.fixed_tail_profile_ids
    score_original = module.score_event
    trajectories = OrderedDict()
    clocks = {}
    tails = {}
    counter = [0]

    def load(path):
        key = str(path)
        if key not in trajectories:
            value = load_original(path)
            value["__clock_key"] = hashlib.sha256(
                b"".join(np.asarray(value[n]).tobytes() for n in ["month", "day", "hour"])
            ).hexdigest()
            value["__path_key"] = key
            trajectories[key] = value
            if len(trajectories) > 32:
                trajectories.popitem(last=False)
        trajectories.move_to_end(key)
        return trajectories[key]

    def clock(compact, year=2001):
        key = (
            compact.get("__clock_key")
            or hashlib.sha256(
                b"".join(np.asarray(compact[n]).tobytes() for n in ["month", "day", "hour"])
            ).hexdigest(),
            year,
        )
        if key not in clocks:
            clocks[key] = clocks_original(compact, year)
        return clocks[key]

    def tail(reference, profiles, tail_fraction=0.1):
        data = (
            profiles[["profile_id", "zone_id", "centre_c"]]
            .sort_values("profile_id")
            .to_csv(index=False)
            .encode()
        )
        key = (
            reference.get("__path_key")
            or hashlib.sha256(reference["temperatures"].tobytes()).hexdigest(),
            hashlib.sha256(data).hexdigest(),
            tail_fraction,
        )
        if key not in tails:
            tails[key] = tail_original(reference, profiles, tail_fraction=tail_fraction)
        return tails[key]

    def score(*args, **kwargs):
        result = score_original(*args, **kwargs)
        counter[0] += 1
        if counter[0] % 400 == 0:
            print("Scored event cases:", counter[0], flush=True)
        return result

    module.load_compact = load
    module.compact_timestamps = clock
    module.fixed_tail_profile_ids = tail
    module.score_event = score
    return {
        "cached_quantities": "unchanged timestamp axes and annual reference tail memberships",
        "service_scores_cached": False,
        "original_scoring_logic_retained": True,
    }
