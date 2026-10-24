"""Summarize strict native-zone service feasibility from retained candidate scores.

This analysis uses already executed controls for five demand definitions. It
does not regenerate commands, run EnergyPlus or assess floor-bound controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier"]
CONTROLS = ("uniform", "personal_center", "ltpref_interval")
SUPPORTS = ("primary", "equal_four_variants", "nested_uniform")
FRACTION = 0.50
TOLERANCE_C = 1e-9


def summarize(candidates: pd.DataFrame, anchors: pd.DataFrame):
    """Return all native-zone conditions, retaining unavailable choices."""
    anchors = anchors.set_index(KEYS, verify_integrity=True)
    candidates = candidates.copy()
    candidates["source"] = candidates.source.fillna("")
    if not pd.api.types.is_bool_dtype(candidates.engineering_pass):
        raise ValueError("Engineering status must be an explicit boolean")
    records = []
    for key, group in candidates.groupby(KEYS, sort=True):
        anchor = anchors.loc[key]
        if anchor["mean"] <= 0 or anchor["tail"] <= 0:
            raise ValueError("Strict fractional limits require positive reference mismatches")
        for support in SUPPORTS:
            available = group if support != "equal_four_variants" else group[
                group.source.ne("finite_uniform_grid")]
            for control in CONTROLS:
                pool = available[available.strategy.eq(control)]
                if support == "nested_uniform" and control != "uniform":
                    pool = pd.concat([pool, available[available.strategy.eq("uniform")]])
                valid = pool[pool.engineering_pass]
                eligible = valid[
                    valid["mean"].le(FRACTION * anchor["mean"] + TOLERANCE_C)
                    & valid["tail"].le(FRACTION * anchor["tail"] + TOLERANCE_C)]
                ratio = np.maximum(valid["mean"] / anchor["mean"],
                                   valid["tail"] / anchor["tail"])
                records.append(dict(zip(KEYS, key)) | {
                    "permission": "native_zone", "candidate_support": support,
                    "control": control, "service_fraction": FRACTION,
                    "mean_anchor_c": float(anchor["mean"]),
                    "fixed_tail_anchor_c": float(anchor["tail"]),
                    "mean_budget_c": FRACTION * float(anchor["mean"]),
                    "fixed_tail_budget_c": FRACTION * float(anchor["tail"]),
                    "candidate_rows": len(pool),
                    "engineering_qualified_candidate_rows": len(valid),
                    "strict_qualified_candidate_rows": len(eligible),
                    "selection_status": "strict_service_available" if len(eligible) else
                        "strict_service_unavailable" if len(valid) else "engineering_unavailable",
                    "strict_service_available": bool(len(eligible)),
                    "best_joint_mismatch_as_anchor_fraction": float(ratio.min()) if len(valid) else np.nan,
                    "minimum_strict_qualified_site_kwh": float(eligible.site_kwh.min())
                        if len(eligible) else np.nan,
                })
    detail = pd.DataFrame(records)
    summary_keys = ["candidate_support", "griffiths_coefficient",
                    "interval_width_multiplier", "control"]

    def aggregate(keys):
        return detail.groupby(keys, sort=True).agg(
            native_zone_conditions=("strict_service_available", "size"),
            strict_service_available_conditions=("strict_service_available", "sum"),
            engineering_qualified_candidate_rows=("engineering_qualified_candidate_rows", "sum"),
            strict_qualified_candidate_rows=("strict_qualified_candidate_rows", "sum"),
        ).reset_index()

    return detail, aggregate(summary_keys), aggregate(summary_keys + ["city"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    inputs = [args.input_root / "candidate_scores.csv", args.input_root / "service_anchors.csv"]
    hashes = {path.name: digest(path) for path in inputs}
    candidates, anchors = [pd.read_csv(path) for path in inputs]
    detail, summary, context = summarize(candidates, anchors)
    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in [("STRICT_SERVICE_CONDITIONS.csv", detail),
                        ("STRICT_SERVICE_SUMMARY.csv", summary),
                        ("STRICT_SERVICE_BY_CONTEXT.csv", context)]:
        path = args.output_root / name
        table.to_csv(path, index=False)
        outputs[name] = {"rows": len(table), "sha256": digest(path)}
    if hashes != {path.name: digest(path) for path in inputs}:
        raise ValueError("Source candidate scores or references changed during analysis")
    manifest = {
        "analysis": "strict_native_zone_service_sensitivity",
        "new_physical_runs": 0, "commands_regenerated": False,
        "floor_bound_permissions_assessed": False, "service_fraction": FRACTION,
        "absolute_mismatch_tolerance_c": TOLERANCE_C,
        "input_candidate_rows": len(candidates), "input_reference_rows": len(anchors),
        "input_sha256": hashes, "script_sha256": digest(__file__), "outputs": outputs,
        "scope": "Retained native-zone controls under each stated demand definition; "
                 "candidate-row counts are repeated evaluations, not independent buildings or runs.",
        "support_definitions": {
            "primary": "Finite uniform grid plus four declared uniform variants; four variants per personalized family.",
            "equal_four_variants": "Only the four declared variants of each family.",
            "nested_uniform": "Primary support, with every uniform candidate also available to each personalized family.",
        },
    }
    (args.output_root / "STRICT_SERVICE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
