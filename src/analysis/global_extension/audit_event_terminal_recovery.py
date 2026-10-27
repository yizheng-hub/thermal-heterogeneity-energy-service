"""Describe residual event effects at and after the 48-hour accounting horizon.

This audit reads completed cached trajectories.  It does not rerun EnergyPlus,
apply a recovery threshold or assert that an action has fully recovered.  Each
event is sampled in its final declared recovery hour and, when present, in the
next six hourly intervals.  Follow-up samples that intersect a later event's
preconditioning, event or recovery window are retained and explicitly flagged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


FOLLOW_UP_HOURS = 6
METRIC_COLUMNS = (
    "operative_temperature_max_abs_difference_c",
    "operative_temperature_mean_abs_difference_c",
    "facility_electricity_kw_difference",
    "hvac_electricity_kw_difference",
    "gas_input_kw_hhv_difference",
)
REQUIRED_COMPACT_ARRAYS = {
    "month",
    "day",
    "hour",
    "zone_ids",
    "temperatures",
    "facility_electric_kwh",
    "hvac_electric_kwh",
    "gas_hhv_kwh",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if values.isna().any():
        raise ValueError(f"{name} contains values that are not Boolean")
    return values.astype(bool)


def _verify_hash(path_value: object, hash_value: object, label: str) -> Path:
    path = Path(str(path_value))
    declared = str(hash_value).strip().lower()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if not declared or declared == "nan" or sha256(path) != declared:
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return path


def _load_compact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    missing = REQUIRED_COMPACT_ARRAYS - set(data)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    return data


def _timestamps(data: Mapping[str, np.ndarray], year: int = 2001) -> pd.DatetimeIndex:
    month = np.asarray(data["month"], dtype=int)
    day = np.asarray(data["day"], dtype=int)
    hour = np.asarray(data["hour"], dtype=int)
    if month.ndim != 1 or month.shape != day.shape or month.shape != hour.shape:
        raise ValueError("Compact month, day and hour arrays must align")
    if ((hour < 1) | (hour > 24)).any():
        raise ValueError("EnergyPlus hour values must lie in 1..24")
    clock = pd.DatetimeIndex([
        pd.Timestamp(year, int(m), int(d)) + pd.Timedelta(hours=int(h) - 1)
        for m, d, h in zip(month, day, hour, strict=True)
    ])
    if clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError("Compact annual clock must be unique and increasing")
    if len(clock) > 1 and not np.all(np.diff(clock.asi8) == pd.Timedelta(hours=1).value):
        raise ValueError("Terminal recovery audit requires hourly trajectories")
    return clock


def _validate_manifests(
    run_manifest: pd.DataFrame, event_manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "action_id",
        "hourly_path",
        "hourly_sha256",
        "baseline_hourly_path",
        "baseline_hourly_sha256",
        "completed",
        "final_engineering_qualified",
    }
    event_required = {
        "run_id",
        "event_id",
        "start_timestamp",
        "duration_hours",
        "precondition_hours",
        "recovery_hours",
    }
    missing_runs = run_required - set(run_manifest)
    missing_events = event_required - set(event_manifest)
    if missing_runs:
        raise ValueError(f"Completed event manifest lacks {sorted(missing_runs)}")
    if missing_events:
        raise ValueError(f"Event evaluation manifest lacks {sorted(missing_events)}")
    runs = run_manifest.copy()
    events = event_manifest.copy()
    if runs.run_id.astype(str).duplicated().any():
        raise ValueError("Completed event manifest duplicates run_id")
    if events.astype({"run_id": str, "event_id": str}).duplicated(
        ["run_id", "event_id"]
    ).any():
        raise ValueError("Event evaluation manifest duplicates a run/event pair")
    completed = _as_bool(runs["completed"], "completed")
    qualified = _as_bool(
        runs["final_engineering_qualified"],
        "final_engineering_qualified",
    )
    if not completed.all() or not qualified.all():
        raise ValueError(
            "Terminal recovery audit requires every input run to be completed "
            "and engineering-qualified"
        )
    run_ids = set(runs.run_id.astype(str))
    event_run_ids = set(events.run_id.astype(str))
    if run_ids != event_run_ids:
        raise ValueError(
            "Completed and planned event run IDs differ: "
            f"completed_only={sorted(run_ids - event_run_ids)}, "
            f"planned_only={sorted(event_run_ids - run_ids)}"
        )
    events["start_timestamp"] = pd.to_datetime(events["start_timestamp"])
    for field in ("duration_hours", "precondition_hours", "recovery_hours"):
        events[field] = pd.to_numeric(events[field], errors="raise").astype(int)
    if (events.duration_hours <= 0).any() or (events.recovery_hours < 0).any():
        raise ValueError("Event and recovery durations are invalid")
    if (events.precondition_hours < 0).any():
        raise ValueError("Preconditioning duration cannot be negative")
    return runs, events


def _later_windows(events: pd.DataFrame, current_start: pd.Timestamp) -> list[tuple]:
    windows: list[tuple] = []
    later = events[events.start_timestamp > current_start]
    for row in later.itertuples(index=False):
        start = pd.Timestamp(row.start_timestamp)
        beginning = start - pd.Timedelta(hours=int(row.precondition_hours))
        end = start + pd.Timedelta(
            hours=int(row.duration_hours) + int(row.recovery_hours)
        )
        windows.append((beginning, end, str(row.event_id)))
    return windows


def _overlap_details(
    start: pd.Timestamp, end: pd.Timestamp, windows: Iterable[tuple]
) -> tuple[bool, list[str]]:
    identifiers = [
        event_id for beginning, ending, event_id in windows
        if start < ending and end > beginning
    ]
    return bool(identifiers), identifiers


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if finite.empty:
        return {
            "n": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "n": int(len(finite)),
        "minimum": float(finite.min()),
        "p05": float(finite.quantile(0.05)),
        "median": float(finite.median()),
        "mean": float(finite.mean()),
        "p95": float(finite.quantile(0.95)),
        "maximum": float(finite.max()),
    }


def _subset_summary(frame: pd.DataFrame) -> dict[str, object]:
    events = frame[["run_id", "event_id"]].drop_duplicates()
    return {
        "event_count": int(len(events)),
        "audit_row_count": int(len(frame)),
        "events_with_later_event_overlap": int(
            events.merge(
                frame.groupby(["run_id", "event_id"], as_index=False)[
                    "later_event_overlaps_post_recovery_audit"
                ].max(),
                on=["run_id", "event_id"],
            )["later_event_overlaps_post_recovery_audit"].sum()
        ),
        "metrics": {column: _distribution(frame[column]) for column in METRIC_COLUMNS},
    }


def _group_summaries(frame: pd.DataFrame, columns: list[str]) -> list[dict]:
    result: list[dict] = []
    grouped = frame.groupby(columns, sort=True, dropna=False)
    for key, group in grouped:
        values = key if isinstance(key, tuple) else (key,)
        item = {column: value for column, value in zip(columns, values, strict=True)}
        final = group[group.audit_hour_index.eq(0)]
        post = group[group.audit_hour_index.gt(0)]
        post_clean = post[~post.overlaps_later_event_window]
        item.update({
            "all_audit_hours": _subset_summary(group),
            "final_recovery_hour": _subset_summary(final),
            "post_recovery_hours": _subset_summary(post),
            "post_recovery_hours_without_later_event_overlap": _subset_summary(post_clean),
        })
        result.append(item)
    return result


def audit_event_terminal_recovery(
    run_manifest: pd.DataFrame,
    event_manifest: pd.DataFrame,
    output_root: Path,
    *,
    follow_up_hours: int = FOLLOW_UP_HOURS,
    run_manifest_path: Path | None = None,
    event_manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Write a descriptive, threshold-free terminal residual audit."""
    if follow_up_hours < 0:
        raise ValueError("follow_up_hours must be nonnegative")
    runs, events = _validate_manifests(run_manifest, event_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    compact_cache: dict[str, dict[str, np.ndarray]] = {}
    clock_cache: dict[str, pd.DatetimeIndex] = {}
    rows: list[dict[str, object]] = []

    def compact(path: Path) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex]:
        key = str(path.resolve())
        if key not in compact_cache:
            compact_cache[key] = _load_compact(path)
            clock_cache[key] = _timestamps(compact_cache[key])
        return compact_cache[key], clock_cache[key]

    for run in runs.to_dict("records"):
        run_id = str(run["run_id"])
        action_path = _verify_hash(
            run["hourly_path"], run["hourly_sha256"], f"Action trajectory for {run_id}"
        )
        baseline_path = _verify_hash(
            run["baseline_hourly_path"],
            run["baseline_hourly_sha256"],
            f"Baseline trajectory for {run_id}",
        )
        action, action_clock = compact(action_path)
        baseline, baseline_clock = compact(baseline_path)
        if not action_clock.equals(baseline_clock):
            raise ValueError(f"Action and baseline clocks differ for {run_id}")
        if not np.array_equal(action["zone_ids"], baseline["zone_ids"]):
            raise ValueError(f"Action and baseline zone identifiers differ for {run_id}")
        for name in REQUIRED_COMPACT_ARRAYS - {"zone_ids"}:
            if len(action[name]) != len(action_clock) or len(baseline[name]) != len(action_clock):
                raise ValueError(f"{name} does not align with the annual clock for {run_id}")
        if action["temperatures"].shape != baseline["temperatures"].shape:
            raise ValueError(f"Action and baseline temperature shapes differ for {run_id}")
        if action["temperatures"].ndim != 2:
            raise ValueError("Operative temperatures must have hour and zone axes")

        by_run = events[events.run_id.astype(str).eq(run_id)].sort_values(
            "start_timestamp", kind="stable"
        )
        clock_index = {timestamp: index for index, timestamp in enumerate(action_clock)}
        for event in by_run.to_dict("records"):
            start = pd.Timestamp(event["start_timestamp"])
            duration = int(event["duration_hours"])
            recovery = int(event["recovery_hours"])
            recovery_end = start + pd.Timedelta(hours=duration + recovery)
            later = _later_windows(by_run, start)
            audit_points = [
                (0, "final_recovery_hour", recovery_end - pd.Timedelta(hours=1))
            ] + [
                (
                    offset,
                    f"post_recovery_hour_{offset}",
                    recovery_end + pd.Timedelta(hours=offset - 1),
                )
                for offset in range(1, follow_up_hours + 1)
            ]
            available = [point for point in audit_points[1:] if point[2] in clock_index]
            overlap_flags = [
                _overlap_details(timestamp, timestamp + pd.Timedelta(hours=1), later)[0]
                for _, _, timestamp in available
            ]
            event_has_overlap = bool(any(overlap_flags))
            for audit_index, label, timestamp in audit_points:
                if timestamp not in clock_index:
                    if audit_index == 0:
                        raise ValueError(
                            f"Final recovery hour is absent for {run_id}/{event['event_id']}"
                        )
                    continue
                index = clock_index[timestamp]
                temperature_difference = np.abs(
                    np.asarray(action["temperatures"][index], dtype=float)
                    - np.asarray(baseline["temperatures"][index], dtype=float)
                )
                if not np.isfinite(temperature_difference).all():
                    raise ValueError(
                        f"Non-finite terminal temperatures for {run_id}/{event['event_id']}"
                    )
                overlaps, overlapping_ids = _overlap_details(
                    timestamp, timestamp + pd.Timedelta(hours=1), later
                )
                row = {
                    "run_id": run_id,
                    "event_id": str(event["event_id"]),
                    "climate": str(run["climate"]),
                    "composition_id": run["composition_id"],
                    "control": str(run["control"]),
                    "action_id": str(run["action_id"]),
                    "action_family": str(run.get("action_family", "")),
                    "permission": str(run.get("permission", "")),
                    "event_start_timestamp": start,
                    "declared_recovery_hours": recovery,
                    "recovery_end_timestamp": recovery_end,
                    "audit_hour_index": audit_index,
                    "audit_hour_label": label,
                    "audit_timestamp": timestamp,
                    "relative_to_recovery_end_hours": audit_index - 1,
                    "requested_post_recovery_hours": follow_up_hours,
                    "available_post_recovery_hours": len(available),
                    "overlaps_later_event_window": overlaps,
                    "overlapping_later_event_ids": ";".join(overlapping_ids),
                    "later_event_overlaps_post_recovery_audit": event_has_overlap,
                    "operative_temperature_max_abs_difference_c": float(
                        temperature_difference.max()
                    ),
                    "operative_temperature_mean_abs_difference_c": float(
                        temperature_difference.mean()
                    ),
                    "facility_electricity_kw_difference": float(
                        action["facility_electric_kwh"][index]
                        - baseline["facility_electric_kwh"][index]
                    ),
                    "hvac_electricity_kw_difference": float(
                        action["hvac_electric_kwh"][index]
                        - baseline["hvac_electric_kwh"][index]
                    ),
                    "gas_input_kw_hhv_difference": float(
                        action["gas_hhv_kwh"][index]
                        - baseline["gas_hhv_kwh"][index]
                    ),
                }
                if not all(np.isfinite(row[column]) for column in METRIC_COLUMNS):
                    raise ValueError(
                        f"Non-finite terminal residual for {run_id}/{event['event_id']}"
                    )
                rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "action_id", "event_start_timestamp",
         "audit_hour_index"],
        kind="stable",
    ).reset_index(drop=True)
    csv_path = output_root / "EVENT_TERMINAL_RECOVERY_RESIDUALS.csv"
    frame.to_csv(csv_path, index=False)
    final_rows = frame[frame.audit_hour_index.eq(0)]
    post_rows = frame[frame.audit_hour_index.gt(0)]
    report: dict[str, object] = {
        "audit_type": "descriptive_terminal_recovery_residuals",
        "threshold_applied": False,
        "full_recovery_claimed": False,
        "interpretation": (
            "Action-minus-baseline residuals at the final declared recovery hour and "
            "the following available hourly intervals; no pass/fail threshold is applied."
        ),
        "declared_recovery_hours": sorted(
            int(value) for value in events.recovery_hours.unique()
        ),
        "requested_post_recovery_hours": int(follow_up_hours),
        "completed_action_runs": int(runs.run_id.nunique()),
        "events": int(frame[["run_id", "event_id"]].drop_duplicates().shape[0]),
        "event_audit_rows": int(len(frame)),
        "final_recovery_rows": int(len(final_rows)),
        "post_recovery_rows": int(len(post_rows)),
        "events_with_later_event_overlap": int(
            final_rows.later_event_overlaps_post_recovery_audit.sum()
        ),
        "groups_by_climate_action_control": _group_summaries(
            frame, ["climate", "action_id", "control"]
        ),
        "marginal_groups": {
            "climate": _group_summaries(frame, ["climate"]),
            "action_id": _group_summaries(frame, ["action_id"]),
            "control": _group_summaries(frame, ["control"]),
        },
        "input_hashes": {
            "run_manifest_sha256": (
                sha256(run_manifest_path) if run_manifest_path is not None else None
            ),
            "event_manifest_sha256": (
                sha256(event_manifest_path) if event_manifest_path is not None else None
            ),
        },
        "output_csv_sha256": sha256(csv_path),
    }
    json_path = output_root / "EVENT_TERMINAL_RECOVERY_AUDIT.json"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--follow-up-hours", type=int, default=FOLLOW_UP_HOURS)
    args = parser.parse_args()
    audit_event_terminal_recovery(
        pd.read_csv(args.run_manifest),
        pd.read_csv(args.event_manifest),
        args.output_root,
        follow_up_hours=args.follow_up_hours,
        run_manifest_path=args.run_manifest,
        event_manifest_path=args.event_manifest,
    )


if __name__ == "__main__":
    main()
