"""Validate and summarise the final multi-event demand-response experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PAIR_KEYS = ["climate", "composition_id", "event_id", "action_id"]
ALTERNATIVE_CONTROLS = ("personal_center", "personal_range")
GROUP_DIMENSIONS = ("action_id", "control", "local_season", "load_level")
RECOVERY_METRICS = {
    "operative_temperature_max_abs_difference_c": "operative_temperature_c",
    "facility_electricity_kw_difference": "facility_electricity_kw",
    "gas_input_kw_hhv_difference": "gas_input_kw_hhv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(series: pd.Series, name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"Invalid Boolean values in {name}")
    return parsed.astype(bool)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{label} lacks {sorted(missing)}")


def validate_inputs(
    scores: pd.DataFrame,
    recovery: pd.DataFrame,
    events: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    expected_runs: int,
    expected_events: int,
    expected_factors: int,
) -> dict[str, object]:
    """Validate the exact event, factor and engineering-qualified populations."""
    _require_columns(
        runs,
        ("run_id", "completed", "final_engineering_qualified"),
        "Completed event manifest",
    )
    _require_columns(
        events,
        (
            "run_id", "event_id", "climate", "composition_id", "control",
            "action_id", "local_season", "load_level",
        ),
        "Event manifest",
    )
    _require_columns(
        scores,
        (
            *PAIR_KEYS, "run_id", "control", "local_season", "load_level",
            "event_service_limit_factor", "service_qualified",
            "qualified_positive_event_response",
            "qualified_positive_full_window_response",
            "event_mean_power_reduction_kw",
            "precondition_event_recovery_net_reduction_kwh",
        ),
        "Event scores",
    )
    _require_columns(
        recovery,
        (
            "run_id", "event_id", "control", "action_id", "audit_hour_index",
            "overlaps_later_event_window", *RECOVERY_METRICS,
        ),
        "Terminal recovery residuals",
    )
    completed = _bool(runs["completed"], "completed")
    qualified = _bool(
        runs["final_engineering_qualified"], "final_engineering_qualified"
    )
    if (
        len(runs) != expected_runs
        or runs.run_id.astype(str).duplicated().any()
        or not completed.all()
        or not qualified.all()
    ):
        raise ValueError("Completed run population failed its final qualification gate")

    event_keys = ["run_id", "event_id"]
    if len(events) != expected_events or events.duplicated(event_keys).any():
        raise ValueError("Event manifest has the wrong population or duplicate keys")
    factors = sorted(pd.to_numeric(scores.event_service_limit_factor).unique())
    score_keys = [*event_keys, "event_service_limit_factor"]
    if len(factors) != expected_factors or scores.duplicated(score_keys).any():
        raise ValueError("Event scores have the wrong factor population or duplicate keys")
    expected_score_rows = expected_events * expected_factors
    if len(scores) != expected_score_rows:
        raise ValueError(
            f"Expected {expected_score_rows} score rows, observed {len(scores)}"
        )
    event_pairs = set(map(tuple, events[event_keys].astype(str).to_numpy()))
    score_pairs = set(map(tuple, scores[event_keys].astype(str).to_numpy()))
    recovery_pairs = set(map(tuple, recovery[event_keys].astype(str).to_numpy()))
    if event_pairs != score_pairs or event_pairs != recovery_pairs:
        raise ValueError("Run/event populations differ across event outputs")
    if set(runs.run_id.astype(str)) != set(events.run_id.astype(str)):
        raise ValueError("Completed runs and event manifest use different run IDs")
    recovery_keys = [*event_keys, "audit_hour_index"]
    if recovery.duplicated(recovery_keys).any():
        raise ValueError("Terminal recovery residuals duplicate an audit hour")
    hours_per_event = recovery.groupby(event_keys).audit_hour_index.nunique()
    if hours_per_event.nunique() != 1:
        raise ValueError("Terminal recovery audit has unequal hours per event")
    expected_hours = set(range(int(hours_per_event.iloc[0])))
    observed_hours = set(pd.to_numeric(recovery.audit_hour_index).astype(int))
    if observed_hours != expected_hours:
        raise ValueError("Terminal recovery audit hours are not consecutive from zero")
    return {
        "completed_runs": len(runs),
        "events": len(events),
        "service_factors": factors,
        "score_rows": len(scores),
        "recovery_rows": len(recovery),
        "recovery_audit_hours_per_event": int(hours_per_event.iloc[0]),
    }


def _response_row(group: pd.DataFrame) -> dict[str, float | int]:
    service = _bool(group.service_qualified, "service_qualified")
    qualified_event = _bool(
        group.qualified_positive_event_response,
        "qualified_positive_event_response",
    )
    qualified_full = _bool(
        group.qualified_positive_full_window_response,
        "qualified_positive_full_window_response",
    )
    event_values = pd.to_numeric(
        group.loc[qualified_event, "event_mean_power_reduction_kw"]
    )
    full_values = pd.to_numeric(
        group.loc[
            qualified_full, "precondition_event_recovery_net_reduction_kwh"
        ]
    )
    count = len(group)
    return {
        "event_count": count,
        "service_qualified_count": int(service.sum()),
        "service_qualified_fraction": float(service.mean()),
        "qualified_positive_event_count": int(qualified_event.sum()),
        "qualified_positive_event_fraction": float(qualified_event.mean()),
        "qualified_event_power_median_kw": float(event_values.median()),
        "qualified_event_power_p05_kw": float(event_values.quantile(0.05)),
        "qualified_event_power_p95_kw": float(event_values.quantile(0.95)),
        "qualified_positive_full_window_count": int(qualified_full.sum()),
        "qualified_positive_full_window_fraction": float(qualified_full.mean()),
        "qualified_full_window_net_median_kwh": float(full_values.median()),
        "qualified_full_window_net_p05_kwh": float(full_values.quantile(0.05)),
        "qualified_full_window_net_p95_kwh": float(full_values.quantile(0.95)),
    }


def response_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Summarise response availability overall and by each declared dimension."""
    rows: list[dict[str, object]] = []
    factor = "event_service_limit_factor"
    for value, group in scores.groupby(factor, sort=True):
        rows.append({
            factor: float(value), "group_dimension": "overall",
            "group_value": "all", **_response_row(group),
        })
    for dimension in GROUP_DIMENSIONS:
        for (value, category), group in scores.groupby(
            [factor, dimension], sort=True, dropna=False
        ):
            rows.append({
                factor: float(value), "group_dimension": dimension,
                "group_value": str(category), **_response_row(group),
            })
    return pd.DataFrame(rows).sort_values(
        [factor, "group_dimension", "group_value"], kind="stable"
    ).reset_index(drop=True)


def _paired_row(group: pd.DataFrame) -> dict[str, float | int]:
    alternative_event = group.alternative_event_qualified.astype(bool)
    uniform_event = group.uniform_event_qualified.astype(bool)
    alternative_full = group.alternative_full_qualified.astype(bool)
    uniform_full = group.uniform_full_qualified.astype(bool)
    difference = pd.to_numeric(group.event_power_difference_kw)
    tolerance = 1e-9
    return {
        "pair_count": len(group),
        "alternative_only_event_qualified": int(
            (alternative_event & ~uniform_event).sum()
        ),
        "uniform_only_event_qualified": int(
            (uniform_event & ~alternative_event).sum()
        ),
        "both_event_qualified": int((alternative_event & uniform_event).sum()),
        "neither_event_qualified": int((~alternative_event & ~uniform_event).sum()),
        "alternative_only_full_window_qualified": int(
            (alternative_full & ~uniform_full).sum()
        ),
        "uniform_only_full_window_qualified": int(
            (uniform_full & ~alternative_full).sum()
        ),
        "both_full_window_qualified": int((alternative_full & uniform_full).sum()),
        "neither_full_window_qualified": int(
            (~alternative_full & ~uniform_full).sum()
        ),
        "alternative_raw_power_greater": int((difference > tolerance).sum()),
        "uniform_raw_power_greater": int((difference < -tolerance).sum()),
        "raw_power_equal": int((difference.abs() <= tolerance).sum()),
        "alternative_minus_uniform_power_median_kw": float(difference.median()),
    }


def paired_control_summary(
    scores: pd.DataFrame, *, main_factor: float = 1.0
) -> pd.DataFrame:
    """Compare each demand-guided control with the same uniform event context."""
    main = scores[np.isclose(scores.event_service_limit_factor, main_factor)].copy()
    controls = set(main.control.astype(str))
    expected_controls = {"uniform_average", *ALTERNATIVE_CONTROLS}
    if controls != expected_controls:
        raise ValueError(f"Main-factor controls differ from {sorted(expected_controls)}")
    uniform = main[main.control.eq("uniform_average")].copy()
    if uniform.duplicated(PAIR_KEYS).any():
        raise ValueError("Uniform main-factor comparison keys are not unique")
    rows: list[dict[str, object]] = []
    for alternative in ALTERNATIVE_CONTROLS:
        candidate = main[main.control.eq(alternative)].copy()
        if candidate.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{alternative} comparison keys are not unique")
        fields = [
            *PAIR_KEYS, "local_season", "load_level",
            "qualified_positive_event_response",
            "qualified_positive_full_window_response",
            "event_mean_power_reduction_kw",
        ]
        paired = candidate[fields].merge(
            uniform[fields], on=PAIR_KEYS, how="inner", validate="one_to_one",
            suffixes=("_alternative", "_uniform"),
        )
        if len(paired) != len(candidate) or len(paired) != len(uniform):
            raise ValueError(f"{alternative} and uniform comparison keys differ")
        for field in ("local_season", "load_level"):
            if not paired[f"{field}_alternative"].equals(
                paired[f"{field}_uniform"]
            ):
                raise ValueError(f"Paired controls disagree on {field}")
            paired[field] = paired[f"{field}_alternative"]
        paired = paired.assign(
            alternative_event_qualified=_bool(
                paired.qualified_positive_event_response_alternative,
                "alternative event qualification",
            ),
            uniform_event_qualified=_bool(
                paired.qualified_positive_event_response_uniform,
                "uniform event qualification",
            ),
            alternative_full_qualified=_bool(
                paired.qualified_positive_full_window_response_alternative,
                "alternative full-window qualification",
            ),
            uniform_full_qualified=_bool(
                paired.qualified_positive_full_window_response_uniform,
                "uniform full-window qualification",
            ),
            event_power_difference_kw=(
                paired.event_mean_power_reduction_kw_alternative
                - paired.event_mean_power_reduction_kw_uniform
            ),
        )
        rows.append({
            "main_service_factor": main_factor, "alternative_control": alternative,
            "group_dimension": "overall", "group_value": "all",
            **_paired_row(paired),
        })
        for dimension in ("action_id", "local_season", "load_level"):
            for value, group in paired.groupby(dimension, sort=True):
                rows.append({
                    "main_service_factor": main_factor,
                    "alternative_control": alternative,
                    "group_dimension": dimension,
                    "group_value": str(value),
                    **_paired_row(group),
                })
    return pd.DataFrame(rows).sort_values(
        ["alternative_control", "group_dimension", "group_value"], kind="stable"
    ).reset_index(drop=True)


def _recovery_row(group: pd.DataFrame) -> dict[str, float | int]:
    overlap = _bool(
        group.overlaps_later_event_window, "overlaps_later_event_window"
    )
    row: dict[str, float | int] = {
        "audit_row_count": len(group),
        "unique_event_count": int(
            group[["run_id", "event_id"]].drop_duplicates().shape[0]
        ),
        "overlap_row_count": int(overlap.sum()),
    }
    for metric, prefix in RECOVERY_METRICS.items():
        values = pd.to_numeric(group[metric]).abs()
        row[f"{prefix}_abs_median"] = float(values.median())
        row[f"{prefix}_abs_p95"] = float(values.quantile(0.95))
        row[f"{prefix}_abs_maximum"] = float(values.max())
    return row


def recovery_summary(
    recovery: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Describe residuals without applying a recovery pass/fail threshold."""
    metadata = events[[
        "run_id", "event_id", "local_season", "load_level"
    ]].copy()
    enriched = recovery.merge(
        metadata, on=["run_id", "event_id"], how="left", validate="many_to_one"
    )
    if enriched[["local_season", "load_level"]].isna().any().any():
        raise ValueError("Recovery residuals could not be joined to event metadata")
    windows = {
        "final_recovery_hour": enriched.audit_hour_index.eq(0),
        "post_recovery_hours_1_to_6": enriched.audit_hour_index.gt(0),
    }
    rows: list[dict[str, object]] = []
    for window, selected in windows.items():
        frame = enriched[selected]
        rows.append({
            "audit_window": window, "group_dimension": "overall",
            "group_value": "all", **_recovery_row(frame),
        })
        for dimension in GROUP_DIMENSIONS:
            for value, group in frame.groupby(dimension, sort=True):
                rows.append({
                    "audit_window": window, "group_dimension": dimension,
                    "group_value": str(value), **_recovery_row(group),
                })
    return pd.DataFrame(rows).sort_values(
        ["audit_window", "group_dimension", "group_value"], kind="stable"
    ).reset_index(drop=True)


def build_summary(
    score_path: Path,
    recovery_path: Path,
    event_path: Path,
    run_path: Path,
    output_root: Path,
    *,
    expected_runs: int = 810,
    expected_events: int = 6480,
    expected_factors: int = 5,
    main_factor: float = 1.0,
) -> dict[str, object]:
    scores = pd.read_csv(score_path)
    recovery = pd.read_csv(recovery_path)
    events = pd.read_csv(event_path)
    runs = pd.read_csv(run_path)
    counts = validate_inputs(
        scores, recovery, events, runs,
        expected_runs=expected_runs,
        expected_events=expected_events,
        expected_factors=expected_factors,
    )
    response = response_summary(scores)
    paired = paired_control_summary(scores, main_factor=main_factor)
    terminal = recovery_summary(recovery, events)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "response": output_root / "EVENT_RESPONSE_GROUP_SUMMARY.csv",
        "paired": output_root / "EVENT_RESPONSE_PAIRED_CONTROL_SUMMARY.csv",
        "recovery": output_root / "EVENT_TERMINAL_RECOVERY_GROUP_SUMMARY.csv",
    }
    response.to_csv(outputs["response"], index=False)
    paired.to_csv(outputs["paired"], index=False)
    terminal.to_csv(outputs["recovery"], index=False)
    overall_response = response[
        response.group_dimension.eq("overall")
        & np.isclose(response.event_service_limit_factor, main_factor)
    ].iloc[0].to_dict()
    overall_pairs = paired[paired.group_dimension.eq("overall")].to_dict("records")
    overall_recovery = terminal[terminal.group_dimension.eq("overall")].to_dict(
        "records"
    )
    report = {
        "status": "PASS",
        "counts": counts,
        "main_service_factor": main_factor,
        "paired_definition": {
            "comparison": (
                "Each demand-guided control is matched to Uniform average using "
                "the same climate, composition, event and action."
            ),
            "keys": PAIR_KEYS,
            "alternatives": list(ALTERNATIVE_CONTROLS),
            "uniform_control": "uniform_average",
            "pairs_per_alternative": int(overall_pairs[0]["pair_count"]),
        },
        "recovery_interpretation": (
            "Descriptive action-minus-baseline residuals at the final hour of the "
            "declared 48-hour recovery window and the next available six hours; "
            "no pass/fail threshold is applied and full recovery is not claimed."
        ),
        "main_factor_overall": overall_response,
        "main_factor_paired_controls": overall_pairs,
        "terminal_recovery_overall": overall_recovery,
        "input_sha256": {
            "scores": sha256(score_path),
            "recovery_residuals": sha256(recovery_path),
            "event_manifest": sha256(event_path),
            "completed_run_manifest": sha256(run_path),
        },
        "output_sha256": {
            name: sha256(path) for name, path in outputs.items()
        },
    }
    report_path = output_root / "EVENT_RESPONSE_SUMMARY.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--recovery-residuals", type=Path, required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--completed-run-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=810)
    parser.add_argument("--expected-events", type=int, default=6480)
    parser.add_argument("--expected-factors", type=int, default=5)
    parser.add_argument("--main-factor", type=float, default=1.0)
    args = parser.parse_args()
    build_summary(
        args.scores.resolve(), args.recovery_residuals.resolve(),
        args.event_manifest.resolve(), args.completed_run_manifest.resolve(),
        args.output_root.resolve(), expected_runs=args.expected_runs,
        expected_events=args.expected_events,
        expected_factors=args.expected_factors, main_factor=args.main_factor,
    )


if __name__ == "__main__":
    main()
