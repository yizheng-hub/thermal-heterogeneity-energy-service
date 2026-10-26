"""Diagnose engineering exclusions without assuming a thermostat-spread cause."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _failure_reason(qa: dict) -> str:
    if qa.get("hours", 0) != 8760 or qa.get("severe", 0):
        return "incomplete_or_severe"
    if qa.get("direct_numerical_warnings"):
        return "hvac_numerical_convergence"
    if qa.get("direct_coil_operating_range_warnings"):
        return "coil_operating_range"
    recurring = qa.get("all_recurring_warnings", qa.get("recurring_warnings", []))
    if any(item.get("treatment_relevance") not in (None, "not_thermostat_hvac") for item in recurring):
        return "other_treatment_relevant_warning"
    if qa.get("same_sizing") is False or qa.get("same_source_sizing") is False:
        return "sizing_changed"
    if qa.get("physical_fingerprint_verified") is False:
        return "physical_fingerprint"
    oa = qa.get("min_oa_delivery_to_native_scheduled_voz",
                qa.get("minimum_outdoor_air_delivery_request_ratio", 1.0))
    if oa is not None and np.isfinite(oa) and oa < 0.99:
        return "outdoor_air_delivery"
    return "unclassified" if not qa.get("engineering_pass", False) else "qualified"


def diagnose(
    manifest: pd.DataFrame, designs: list[dict], qa_root: Path, contexts: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design_table = pd.DataFrame(designs)
    logical = manifest.merge(
        design_table[["run_id", "context", "composition_id", "layout", "permission",
                      "control", "variant", "targets_c"]],
        on=["run_id", "context", "composition_id", "layout", "permission", "control", "variant"],
        how="left",
        validate="one_to_one",
    )
    if logical.targets_c.isna().any():
        raise ValueError("A logical design lacks its thermostat targets")

    run_rows = []
    for run_id, group in logical.groupby("run_id", sort=True):
        target_sets = {json.dumps(value, sort_keys=True) for value in group.targets_c}
        if len(target_sets) != 1 or group.context.nunique() != 1:
            raise ValueError(f"Physical run {run_id} has inconsistent logical inputs")
        targets = json.loads(next(iter(target_sets)))
        context = group.context.iloc[0]
        floors = contexts[context]["floor_groups"]
        by_floor: dict[str, list[float]] = {}
        for zone, value in targets.items():
            by_floor.setdefault(floors[zone], []).append(float(value))
        qa_path = qa_root / run_id / "ENGINEERING_QA.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}
        values = np.asarray(list(targets.values()), dtype=float)
        run_rows.append(
            {
                "run_id": run_id,
                "context": context,
                "logical_designs": len(group),
                "controls": "|".join(sorted(group.control.unique())),
                "permissions": "|".join(sorted(group.permission.unique())),
                "target_min_c": float(values.min()),
                "target_max_c": float(values.max()),
                "target_range_c": float(values.max() - values.min()),
                "maximum_within_floor_target_range_c": float(
                    max(max(v) - min(v) for v in by_floor.values())
                ),
                "engineering_pass": bool(qa.get("engineering_pass", False)),
                "failure_reason": _failure_reason(qa),
                "direct_numerical_warning_count": len(qa.get("direct_numerical_warnings", [])),
                "direct_coil_warning_count": len(
                    qa.get("direct_coil_operating_range_warnings") or []
                ),
                "baseline_anchor_engineering_pass": bool(
                    contexts[context].get("baseline_anchor_evidence", {}).get("passed", False)
                ),
                "baseline_anchor_direct_numerical_warning_count": len(
                    contexts[context].get("baseline_anchor_evidence", {}).get(
                        "direct_hvac_numerical_warnings", []
                    )
                ),
            }
        )
    runs = pd.DataFrame(run_rows)
    logical_map = logical.drop(columns="targets_c").merge(
        runs[["run_id", "target_range_c", "maximum_within_floor_target_range_c",
              "engineering_pass", "failure_reason"]],
        on="run_id", how="left", validate="many_to_one",
    )
    summary = logical_map.groupby(
        ["context", "permission", "control", "failure_reason"], sort=True
    ).agg(
        logical_designs=("run_id", "size"),
        unique_physical_runs=("run_id", "nunique"),
        target_range_c_median=("target_range_c", "median"),
        target_range_c_maximum=("target_range_c", "max"),
        within_floor_range_c_median=("maximum_within_floor_target_range_c", "median"),
        within_floor_range_c_maximum=("maximum_within_floor_target_range_c", "max"),
    ).reset_index()
    return runs, logical_map, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--designs", type=Path, required=True)
    parser.add_argument("--qa-root", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs, logical, summary = diagnose(
        pd.read_csv(args.manifest),
        json.loads(args.designs.read_text(encoding="utf-8")),
        args.qa_root.resolve(),
        json.loads(args.contexts.read_text(encoding="utf-8")),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output / "ENGINEERING_FAILURES_BY_PHYSICAL_RUN.csv", index=False)
    logical.to_csv(args.output / "ENGINEERING_LOGICAL_TO_PHYSICAL_MAP.csv", index=False)
    summary.to_csv(args.output / "ENGINEERING_FAILURE_SUMMARY.csv", index=False)
    print(runs.groupby(["context", "failure_reason"]).size().to_string())


if __name__ == "__main__":
    main()
