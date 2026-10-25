"""Describe both demand archives on the same two-layout control design.

Compositions are separate controlled samples, not matched people or estimates
of national population effects. Every designed outcome, including unavailable
comparisons, contributes to the reported denominators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAYOUTS = {"random_mixing", "demand_ordered"}
METRICS = ("delta_site_pct", "delta_annual_operational_co2_kg",
           "delta_interval_coverage_percentage_points", "delta_mean_reference_mismatch_c",
           "delta_fixed_tail_reference_mismatch_c", "delta_modeled_interval_mean_excess_c")


def shared_scope(frame):
    """Remove source-only layouts and single-floor command aliases before counting."""
    keep = frame.layout.isin(LAYOUTS)
    keep &= ~(frame.context.isin(["Seattle_small2019", "PortAngeles_small2019"])
              & frame.permission.eq("floor"))
    return frame.loc[keep].copy()


def summarize(frame, groups):
    records = []
    for key, group in frame.groupby(groups, sort=True, dropna=False):
        row = dict(zip(groups, key))
        selected = group[group.status.eq("matched")]
        row.update(designed_pairs=len(group), matched_pairs=len(selected),
                   unavailable_pairs=int(group.status.ne("matched").sum()))
        for metric in METRICS:
            values = pd.to_numeric(selected[metric]).dropna()
            row.update({metric+"_median": float(values.median()) if len(values) else np.nan,
                        metric+"_minimum": float(values.min()) if len(values) else np.nan,
                        metric+"_maximum": float(values.max()) if len(values) else np.nan,
                        metric+"_positive": int(values.gt(1e-9).sum()),
                        metric+"_negative": int(values.lt(-1e-9).sum()),
                        metric+"_zero": int(values.abs().le(1e-9).sum())})
        records.append(row)
    return pd.DataFrame(records)


def run(primary, chinese, output):
    frames, sources = [], {}
    for cohort, directory in (("ASHRAE", primary), ("CTCD", chinese)):
        path = directory/"CONTROL_COMPARISONS.csv"
        source = pd.read_csv(path)
        frame = shared_scope(source)
        key = ["context", "composition_id", "layout", "permission", "control",
               "comparison_mode", "service_fraction"]
        if len(frame) != 600 or frame.duplicated(key).any():
            raise ValueError(f"{cohort}: all 600 shared-design comparisons are required")
        frame.insert(0, "data_source", cohort)
        frames.append(frame)
        sources[cohort] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output/"COMMON_DESIGN_CONTROL_PAIRS.csv", index=False)
    groups = ["data_source", "comparison_mode", "service_fraction", "alternative"]
    summarize(combined, groups).to_csv(output/"COHORT_CONTROL_SUMMARY.csv", index=False)
    summarize(combined, groups+["context", "permission", "layout"]).to_csv(
        output/"CONTEXT_SPACE_CONTROL_SUMMARY.csv", index=False)
    report = {"status": "PASS", "comparison_rows": len(combined),
              "shared_layouts": sorted(LAYOUTS), "distinct_space_conditions_per_cohort": 60,
              "records_pooled": False, "people_matched_between_archives": False,
              "original_archives_modified": False, "source_sha256": sources,
              "scope": "Within-cohort paired control effects under a common experimental design; differences between cohorts are not attributed causally to nationality, culture or income."}
    (output/"PARALLEL_COMPARISON_QA.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-annual", type=Path, required=True)
    parser.add_argument("--chinese-annual", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.primary_annual, args.chinese_annual, args.output_root), indent=2))


if __name__ == "__main__":
    main()
