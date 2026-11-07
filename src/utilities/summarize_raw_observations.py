"""Regenerate observation summaries and the existing nested-variance analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from thermal_heterogeneity.human.hierarchy import weighted_nested_variance_decomposition  # noqa: E402


def run(demand, output):
    ashrae_path = demand / "ASHRAE_RECORDS.csv.gz"
    ctcd_path = demand / "CTCD_RECORDS.csv.gz"
    ashrae = pd.read_csv(ashrae_path, low_memory=False)
    china = pd.read_csv(ctcd_path, low_memory=False)
    china = china.loc[china.active_conditioning].copy()
    china["country"] = "china"
    combined = pd.concat(
        [ashrae.assign(archive="ASHRAE"), china.assign(archive="CTCD")], ignore_index=True
    )
    output.mkdir(parents=True, exist_ok=True)
    combined.groupby(["archive", "country", "city"], dropna=False).agg(
        observation_records=("reference_g050_c", "size"),
        mean_demand_center_c=("reference_g050_c", "mean"),
        median_demand_center_c=("reference_g050_c", "median"),
        demand_center_p10_c=("reference_g050_c", lambda v: v.quantile(0.1)),
        demand_center_p90_c=("reference_g050_c", lambda v: v.quantile(0.9)),
    ).reset_index().to_csv(output / "CITY_OBSERVATION_SUMMARY.csv", index=False)
    rows = []
    levels = ("country", "city", "building_id", "unit_group")
    for weight in ["record_weight", "unit_weight", "building_weight", "source_weight"]:
        for coefficient in ["033", "050", "067"]:
            values = weighted_nested_variance_decomposition(
                ashrae, value=f"reference_g{coefficient}_c", weight=weight, levels=levels
            )
            values["weighting"] = weight
            values["griffiths_coefficient"] = float(coefficient) / 100
            rows.append(values)
    variance = pd.concat(rows, ignore_index=True)
    if variance.closure_error_c2.abs().max() > 1e-10:
        raise ValueError("Nested-variance components do not close")
    variance.to_csv(output / "ASHRAE_NESTED_VARIANCE.csv", index=False)
    scope = {
        "observation_records": len(combined),
        "ashrae_records": len(ashrae),
        "ctcd_active_conditioning_records": len(china),
        "countries": combined.country.nunique(),
        "ashrae_analysis_units": ashrae.unit_group.nunique(),
        "ashrae_known_subject_units": ashrae.loc[ashrae.subject_id.notna(), "unit_group"].nunique(),
        "counting": "Records, analysis units and assigned simulation profiles are separate quantities.",
        "variance_hierarchy": "ASHRAE source building and analysis-unit fields; city summaries combine declared observational cohorts.",
        "inputs": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ashrae_path, ctcd_path]
        },
    }
    (output / "OBSERVATION_EVIDENCE_AUDIT.json").write_text(json.dumps(scope, indent=2) + "\n")
    print(json.dumps(scope, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.demand_root, args.output_root)
