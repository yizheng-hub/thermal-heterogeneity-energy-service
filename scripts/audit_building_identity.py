"""Audit original versus inferred building identifiers without exporting raw IDs.

Uses the same checksum-locked inputs and cohort rules as generate_demand_inputs.
This is an identity sensitivity analysis; frozen building simulations are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from generate_demand_inputs import build_cohort, verify_input
from thermal_heterogeneity.human.hierarchy import weighted_nested_variance_decomposition


def numeric_signature(frame: pd.DataFrame) -> str:
    columns = ["observed_temperature_c", "thermal_sensation", "reference_g033_c",
               "reference_g050_c", "reference_g067_c"]
    values = frame[columns].round(6).sort_values(columns).to_numpy(dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def audit(measurements: pd.DataFrame, metadata: pd.DataFrame):
    cohort, scope = build_cohort(measurements, metadata)
    flags = cohort.building_id_inf.astype("string").str.strip().str.lower()
    if flags.isna().any() or not flags.isin(["no", "yes"]).all():
        raise ValueError("Every retained building needs a valid identity-origin flag")
    cohort["identity_origin"] = flags.map({"no": "original", "yes": "inferred"})
    subsets = {"all_database_groups": cohort,
               "original_building_ids": cohort.loc[flags.eq("no")],
               "original_building_and_subject_ids": cohort.loc[flags.eq("no") & cohort.subject_id.notna()]}
    summaries, components = [], []
    levels = ("country", "city", "building_id", "unit_group")
    for name, subset in subsets.items():
        subset = subset.copy()
        weights = {"unit_equal": 1 / subset.groupby("unit_group").record_id.transform("count"),
                   "record_equal": 1.,
                   "building_equal": 1 / subset.groupby("building_id").record_id.transform("count"),
                   "source_equal": 1 / subset.groupby("source_group").record_id.transform("count")}
        summaries.append({"subset": name, "records": len(subset),
                          "analysis_units": subset.unit_group.nunique(),
                          "building_groups": subset.building_id.nunique(),
                          "cities": subset.city.nunique(), "countries": subset.country.nunique()})
        for weighting, weight in weights.items():
            subset["audit_weight"] = weight
            for label, coefficient in (("033", .33), ("050", .50), ("067", .67)):
                result = weighted_nested_variance_decomposition(
                    subset, value=f"reference_g{label}_c", weight="audit_weight", levels=levels)
                result.insert(0, "subset", name)
                result.insert(1, "weighting", weighting)
                result.insert(2, "griffiths_coefficient", coefficient)
                components.append(result)
    return cohort, scope, pd.DataFrame(summaries), pd.concat(components, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--public-profiles", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    hashes = {"measurements": verify_input(args.measurements, "measurements"),
              "metadata": verify_input(args.metadata, "metadata")}
    cohort, scope, summary, components = audit(
        pd.read_csv(args.measurements, low_memory=False), pd.read_csv(args.metadata, low_memory=False))
    original = {}
    for _, group in cohort.groupby("building_id"):
        signature = numeric_signature(group)
        if signature in original:
            raise ValueError("Ambiguous numerical group signature")
        original[signature] = group.identity_origin.iloc[0]
    public = pd.read_csv(args.public_profiles)
    rows = []
    for group_id, group in public.groupby("release_building_group_id"):
        rows.append({"release_building_group_id": group_id,
                     "identity_origin": original.pop(numeric_signature(group)),
                     "records": len(group)})
    if original:
        raise ValueError("Raw/public groups do not match completely")
    origins = pd.DataFrame(rows)
    inventory = pd.read_csv(args.source_inventory)
    pools = inventory.loc[inventory.eligible_primary].merge(
        origins, on="release_building_group_id", validate="one_to_one")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "IDENTITY_SUBSET_SCOPE.csv", index=False)
    components.to_csv(args.output_root / "IDENTITY_VARIANCE_SENSITIVITY.csv", index=False)
    origins.to_csv(args.output_root / "PUBLIC_BUILDING_ID_ORIGINS.csv", index=False)
    pools[["release_building_group_id", "identity_origin", "executable_units"]].to_csv(
        args.output_root / "SOURCE_POOL_ID_ORIGINS.csv", index=False)
    report = {"status": "PASS", "source_sha256": hashes, "cohort": scope,
              "all_group_origins": origins.identity_origin.value_counts().to_dict(),
              "source_pool_group_origins": pools.identity_origin.value_counts().to_dict(),
              "original_record_subject_building_identifiers_exported": False,
              "physics_changed": False, "new_simulations": 0,
              "interpretation": "Original building identifiers support building membership, not room membership or simultaneous presence.",
              "metadata_definition": "https://github.com/CenterForTheBuiltEnvironment/ashrae-db-II/blob/master/README.md"}
    (args.output_root / "BUILDING_IDENTITY_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
