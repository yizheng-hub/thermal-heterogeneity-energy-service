"""Classify annual trajectories without treating every solver warning as a crash."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def classify_summary(summary: dict) -> dict[str, bool | int]:
    """Return primary output validity and the stricter warning-free status."""
    direct = list(summary.get("direct_numerical_warnings") or [])
    isolated_iteration = [
        line for line in direct if "SimHVAC: Maximum iterations" in line
    ]
    other_numerical = [line for line in direct if line not in isolated_iteration]
    recurring = list(summary.get("recurring_warnings") or [])
    relevant_recurring = [
        row for row in recurring
        if row.get("treatment_relevance") != "not_thermostat_hvac"
    ]
    outdoor_air = summary.get("minimum_outdoor_air_delivery_request_ratio", np.nan)
    outdoor_air_ok = bool(np.isfinite(outdoor_air) and float(outdoor_air) >= 0.99)
    primary = bool(
        summary.get("completed", False)
        and int(summary.get("hours", 0)) == 8760
        and int(summary.get("severe", -1)) == 0
        and not other_numerical
        and not relevant_recurring
        and not (summary.get("direct_coil_operating_range_warnings") or [])
        and float(summary.get("hourly_electric_closure_max_kwh", np.inf)) < 1e-8
        and float(summary.get(
            "source_occupancy_ventilation_requirement_max_delta", np.inf
        )) < 1e-10
        and summary.get("same_source_sizing") is True
        and outdoor_air_ok
    )
    strict = bool(primary and not isolated_iteration)
    return {
        "primary_output_valid": primary,
        "strict_warning_free": strict,
        "isolated_hvac_iteration_warnings": len(isolated_iteration),
    }


def classifications(run_manifest: Path) -> dict[str, dict[str, bool | int]]:
    """Read per-run summaries stored beside a run manifest."""
    root = run_manifest.resolve().parent / "runs"
    result = {}
    for folder in root.iterdir():
        summary_path = folder / "RUN_SUMMARY.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result[str(summary["run_id"])] = classify_summary(summary)
    return result
