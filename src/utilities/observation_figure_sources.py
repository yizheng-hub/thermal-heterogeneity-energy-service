"""Rebuild geographic, demand and policy figure sources from raw-rebuilt records."""

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
from thermal_heterogeneity.human.hierarchy import weighted_nested_variance_decomposition  # noqa: E402 - initialize local imports before loading project modules
from thermal_heterogeneity.human.policy import classify  # noqa: E402 - initialize local imports before loading project modules


def map_points(ashrae, china, city_metadata, overrides):
    ashrae = ashrae.rename(columns={"lat": "latitude", "lon": "longitude"}).copy()
    for row in overrides.itertuples(index=False):
        selected = ashrae.country.eq(row.country) & ashrae.city.eq(row.city)
        ashrae.loc[selected, "latitude"] = ashrae.loc[selected, "latitude"].fillna(row.latitude)
        ashrae.loc[selected, "longitude"] = ashrae.loc[selected, "longitude"].fillna(row.longitude)
    china = china.loc[china.active_conditioning].copy()
    china["country"] = "china"
    china["city"] = china.city_normalized
    china = china.merge(city_metadata[["city_normalized", "latitude", "longitude"]],
        on="city_normalized", how="left", validate="many_to_one")
    data = pd.concat([ashrae, china], ignore_index=True)
    if data[["country", "city", "latitude", "longitude", "reference_g050_c"]].isna().any().any():
        raise ValueError("Every mapped observation requires a demand center and retained coordinates")
    points = data.groupby(["country", "city"], sort=True).agg(
        record_count=("reference_g050_c", "size"),
        demand_center_c=("reference_g050_c", "mean"),
        latitude=("latitude", "median"), longitude=("longitude", "median"),
    ).reset_index()
    assert int(points.record_count.sum()) == len(data)
    return points, data


def policy_positions(ashrae, declarations):
    rows = []
    for policy in declarations.sort_values("plot_order").itertuples(index=False):
        data = ashrae.loc[ashrae.country.eq(policy.jurisdiction_key)].copy()
        rule = policy.archive_match_rule
        if rule == "canada_nearest_clo_band":
            light = (data.clo - 0.65).abs().le((data.clo - 1.0).abs())
            low, high = np.where(light, 21, 18), np.where(light, 25, 23)
        elif rule == "india_ecsbc_met_season":
            data = data.loc[data.met.le(1.2)].copy()
            winter = data.season.isin(["winter", "cool/dry"])
            low_met = data.met.le(1.0)
            low = np.where(winter, np.where(low_met, 19, 15), np.where(low_met, 22, 20))
            high = np.where(winter, np.where(low_met, 25, 23), np.where(low_met, 27, 26))
        elif rule == "sweden_seasonal_air_range":
            data = data.loc[data.season.isin(["winter", "summer"])].copy()
            low, high = 20, np.where(data.season.eq("winter"), 24, 26)
        elif rule in ("all_country_fixed_range", "singapore_operational_operative_range"):
            low, high = policy.lower_c, policy.upper_c
        else:
            # One-sided campaigns have a different endpoint from bilateral ranges.
            continue
        positions = classify(data.reference_g050_c.to_numpy(), low, high)
        for position in ("below", "within", "above"):
            rows.append({"policy_id": policy.policy_id, "policy": policy.plot_label,
                "policy_type": policy.layer, "position": position,
                "proportion": float(np.mean(positions == position)), "records": len(data),
                "analysis_units": data.unit_group.nunique(), "griffiths": "g0.50"})
    return pd.DataFrame(rows)


def leave_one_source(ashrae):
    rows = []
    levels = ("country", "city", "building_id", "unit_group")
    for source in sorted(ashrae.source_group.unique()):
        data = ashrae.loc[ashrae.source_group.ne(source)]
        for weight in ("record_weight", "unit_weight", "building_weight", "source_weight"):
            for coefficient in ("033", "050", "067"):
                values = weighted_nested_variance_decomposition(data,
                    value=f"reference_g{coefficient}_c", weight=weight, levels=levels)
                values["omitted_source"] = source
                values["weighting"] = weight
                values["griffiths_coefficient"] = int(coefficient) / 100
                rows.append(values)
    return pd.concat(rows, ignore_index=True)


def run(demand, output):
    if not demand.resolve().is_relative_to(ROOT) or not output.resolve().is_relative_to(ROOT):
        raise ValueError("Observation figure inputs and outputs must stay inside this project")
    paths = [demand / "ASHRAE_RECORDS.csv.gz", demand / "CTCD_RECORDS.csv.gz",
        demand / "ASHRAE_ASSIGNMENTS.csv.gz", demand / "CTCD_ASSIGNMENTS.csv.gz",
        ROOT / "data/metadata/ctcd_city_map_metadata.csv",
        ROOT / "data/metadata/observation_map_coordinate_overrides.csv",
        ROOT / "data/metadata/policy_reference_documents.csv"]
    ashrae, china = [pd.read_csv(p, low_memory=False) for p in paths[:2]]
    points, records = map_points(ashrae, china, pd.read_csv(paths[4]), pd.read_csv(paths[5]))
    policies = policy_positions(ashrae, pd.read_csv(paths[6]))
    profiles = pd.concat([pd.read_csv(p) for p in paths[2:4]], ignore_index=True)
    profiles["range_width_c"] = profiles.upper_c - profiles.lower_c
    counts = profiles.groupby("composition_id").size()
    if len(counts) != 6 or not counts.eq(267).all():
        raise ValueError("Six complete 267-profile compositions are required")
    country = records.groupby("country").agg(record_count=("reference_g050_c", "size"),
        demand_center_mean_c=("reference_g050_c", "mean"),
        demand_center_p10_c=("reference_g050_c", lambda s: s.quantile(.1)),
        demand_center_median_c=("reference_g050_c", "median"),
        demand_center_p90_c=("reference_g050_c", lambda s: s.quantile(.9))).reset_index()
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {"CITY_MAP_POINTS": points, "COUNTRY_DEMAND_SUMMARY": country,
        "DEMAND_COMPOSITION_PROFILES": profiles, "POLICY_RANGE_POSITIONS": policies,
        "LEAVE_ONE_SOURCE_VARIANCE": leave_one_source(ashrae)}.items():
        table.to_csv(output / (name + ".csv"), index=False)
    audit = {"status": "completed", "observation_records": len(records),
        "mapped_locations": len(points), "countries": len(country),
        "composition_profiles": len(profiles), "policy_documents": policies.policy_id.nunique(),
        "leave_one_source_runs_per_specification": ashrae.source_group.nunique(),
        "inputs": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "derived_legacy_tables_read": [],
        "map_counting": "Records from both archives are pooled at shared country-city locations.",
        "policy_counting": "Declared document-specific matching rules; bilateral ranges only."}
    (output / "OBSERVATION_FIGURE_SOURCE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "inputs"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.demand_root.resolve(), args.output_root.resolve())
