"""Select a fixed, multi-season demand-response event protocol.

Events are chosen from one declared reference trajectory before controller
comparisons.  Each local meteorological season contributes a typical and a
high-load occupied two-hour window.  Event start times are allowed to vary so
winter morning peaks are not forced into a summer-afternoon template.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


SEASON_BLOCK = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def local_season_name(block: str, latitude: float) -> str:
    """Translate a calendar block to a local meteorological season."""
    northern = latitude >= 0
    names = ({"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
             if northern else
             {"DJF": "summer", "MAM": "autumn", "JJA": "winter", "SON": "spring"})
    if block not in names:
        raise ValueError(f"Unknown meteorological block: {block}")
    return names[block]


def _window_candidates(
    timestamp: pd.DatetimeIndex,
    load_kw: np.ndarray,
    occupied: np.ndarray,
    *,
    window_hours: int,
    recovery_hours: int,
    precondition_hours: int,
    cooling_kw: np.ndarray | None,
    heating_kw: np.ndarray | None,
    schedule_minute: np.ndarray | None = None,
) -> pd.DataFrame:
    if schedule_minute is None:
        schedule_minute = (timestamp.hour * 60 + timestamp.minute).to_numpy(int)
    rows: list[dict] = []
    for start in range(len(timestamp) - window_hours + 1):
        stop = start + window_hours
        recovery_stop = stop + recovery_hours
        if recovery_stop > len(timestamp):
            continue
        times = timestamp[start:stop]
        if not occupied[start:stop].all():
            continue
        if times[-1] - times[0] != pd.Timedelta(hours=window_hours - 1):
            continue
        if times[0].date() != times[-1].date() or times[0].hour < precondition_hours:
            continue
        block = SEASON_BLOCK[int(times[0].month)]
        row = {
            "season_block": block,
            "start_timestamp": times[0],
            "end_timestamp": times[-1] + pd.Timedelta(hours=1),
            "reference_mean_kw": float(load_kw[start:stop].mean()),
            "reference_energy_kwh": float(load_kw[start:stop].sum()),
            "start_minute": int(schedule_minute[start]),
        }
        if cooling_kw is not None and heating_kw is not None:
            cooling = float(cooling_kw[start:stop].sum())
            heating = float(heating_kw[start:stop].sum())
            if cooling + heating <= 1e-9:
                continue
            row["dominant_mode"] = "cooling" if cooling >= heating else "heating"
            row["sign"] = 1.0 if cooling >= heating else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_multi_event_protocol(
    timestamps: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    selection_load_kw: Sequence[float] | np.ndarray,
    occupied: Sequence[bool] | np.ndarray,
    *,
    latitude: float,
    cooling_kw: Sequence[float] | np.ndarray | None = None,
    heating_kw: Sequence[float] | np.ndarray | None = None,
    window_hours: int = 2,
    recovery_hours: int = 48,
    typical_quantile: float = 0.5,
    high_quantile: float = 0.9,
    precondition_hours: int = 2,
    schedule_start_minute: Sequence[int] | np.ndarray | None = None,
    equivalent_dates: bool = False,
) -> pd.DataFrame:
    """Choose typical and high-load events in each meteorological season.

    Candidate windows must be consecutive, occupied, contained within one day
    and leave room for the declared same-day preconditioning period.  Starts
    are separated by at least ``precondition_hours + window_hours +
    recovery_hours`` so one event's preconditioning cannot enter another
    event's recovery window.  Selection is deterministic: candidates
    nearest each season's declared load quantile are considered first, with
    earlier timestamps breaking ties.
    """
    timestamp = pd.DatetimeIndex(timestamps)
    load = np.asarray(selection_load_kw, dtype=float)
    on = np.asarray(occupied, dtype=bool)
    if len(timestamp) != len(load) or load.shape != on.shape or load.ndim != 1:
        raise ValueError("Timestamps, load and occupancy must be aligned one-dimensional arrays")
    if timestamp.has_duplicates or not timestamp.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    if not np.isfinite(load).all() or (load < 0).any():
        raise ValueError("Selection load must be finite and nonnegative")
    if window_hours <= 0 or recovery_hours < 0 or precondition_hours < 0:
        raise ValueError("Window and recovery durations are invalid")
    if not 0 <= typical_quantile < high_quantile <= 1:
        raise ValueError("Typical and high quantiles must satisfy 0 <= typical < high <= 1")
    schedule_minute = (
        (timestamp.hour * 60 + timestamp.minute).to_numpy(int)
        if schedule_start_minute is None
        else np.asarray(schedule_start_minute, dtype=int)
    )
    if schedule_minute.shape != load.shape:
        raise ValueError("Schedule start minutes must align with the annual clock")
    if ((schedule_minute < 0) | (schedule_minute >= 1440)).any():
        raise ValueError("Schedule start minutes must lie within a calendar day")
    cool = None if cooling_kw is None else np.asarray(cooling_kw, dtype=float)
    heat = None if heating_kw is None else np.asarray(heating_kw, dtype=float)
    if (cool is None) != (heat is None):
        raise ValueError("Cooling and heating signals must be supplied together")
    if cool is not None and (cool.shape != load.shape or heat.shape != load.shape):
        raise ValueError("Cooling and heating signals must align with HVAC load")

    candidates = _window_candidates(
        timestamp, load, on, window_hours=window_hours,
        recovery_hours=recovery_hours,
        precondition_hours=precondition_hours, schedule_minute=schedule_minute,
        cooling_kw=cool, heating_kw=heat,
    )
    if candidates.empty:
        raise ValueError("No valid occupied event windows were found")
    selected: list[dict] = []
    separation = pd.Timedelta(
        hours=precondition_hours + window_hours + recovery_hours
    )
    for block in ("DJF", "MAM", "JJA", "SON"):
        seasonal = candidates[candidates.season_block.eq(block)].copy()
        if seasonal.empty:
            raise ValueError(f"No valid event windows were found for {block}")
        for level, quantile in (("high", high_quantile), ("typical", typical_quantile)):
            target = float(seasonal["reference_mean_kw"].quantile(quantile))
            ranked = seasonal.assign(
                _distance=(seasonal["reference_mean_kw"] - target).abs()
            ).sort_values(["_distance", "start_timestamp"], kind="stable")
            choice = None
            for row in ranked.to_dict("records"):
                start = pd.Timestamp(row["start_timestamp"])
                if all(abs(start - pd.Timestamp(old["start_timestamp"])) >= separation
                       for old in selected):
                    choice = row
                    break
            if choice is None:
                raise ValueError(
                    f"No {block} {level} event satisfies the nonoverlapping recovery rule"
                )
            choice.pop("_distance", None)
            choice.update(
                local_season=local_season_name(block, latitude),
                load_level=level,
                target_quantile=float(quantile),
                target_reference_mean_kw=target,
                window_hours=int(window_hours),
                precondition_hours=int(precondition_hours),
                recovery_hours=int(recovery_hours),
            )
            selected.append(choice)
    if equivalent_dates:
        # Exchange a representative date only when another date is equally
        # close to its load quantile at the precision of the hourly meter.
        # Reserve the other original windows and each substitution already made.
        reserved = {row["start_timestamp"] for row in selected}
        for original in sorted(selected, key=lambda row: row["start_timestamp"]):
            start = original["start_timestamp"]
            target = original["target_reference_mean_kw"]
            distance = abs(original["reference_mean_kw"] - target)
            group = candidates.loc[candidates.season_block.eq(original["season_block"])]
            tied = group.loc[(group.reference_mean_kw.sub(target).abs().sub(distance).abs() <= 1e-9)
                             & group.start_timestamp.dt.date.ne(start.date())]
            for _, alternate in tied.sort_values("start_timestamp").iterrows():
                time = alternate.start_timestamp
                if all(abs(time - other) >= separation for other in reserved if other != start):
                    original.update(alternate.to_dict())
                    reserved.add(time)
                    break
    result = pd.DataFrame(selected).sort_values("start_timestamp", kind="stable").reset_index(drop=True)
    result["month"] = result.start_timestamp.dt.month
    result["day"] = result.start_timestamp.dt.day
    result["duration_hours"] = int(window_hours)
    return result


def events_for_idf(protocol: pd.DataFrame) -> list[dict]:
    """Convert a selected protocol to ``event_action_input`` records."""
    required = {"month", "day", "start_minute", "duration_hours", "precondition_hours"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Event protocol is missing fields: {sorted(missing)}")
    records = []
    for row in protocol.to_dict("records"):
        if "sign" not in row or not np.isfinite(row["sign"]):
            raise ValueError("Each event requires a finite heating/cooling action sign")
        records.append({
            "month": int(row["month"]),
            "day": int(row["day"]),
            "sign": float(row["sign"]),
            "start_minute": int(row["start_minute"]),
            "duration_hours": int(row["duration_hours"]),
            "precondition_hours": int(row["precondition_hours"]),
        })
    return records
