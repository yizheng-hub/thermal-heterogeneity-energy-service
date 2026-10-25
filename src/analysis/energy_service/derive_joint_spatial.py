"""Assemble six declared demand compositions without changing any simulation."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LAYOUTS = ["random_mixing", "demand_ordered"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(inputs, output):
    output.mkdir(parents=True, exist_ok=True)
    sources, spatial, layouts, overlap, selections = {}, [], [], [], []
    largest_coverage_error = 0.0
    largest_within_zone_weight_spread = 0.0
    keys = ["context", "composition_id", "layout"]
    for annual, common, assignments_path in inputs:
        files = [annual / "SPATIAL_PAIRS.csv", annual / "STATIC_ZONE_INTERVAL_COMPATIBILITY.csv",
                 annual / "PROFILE_ANNUAL_SERVICE.csv.gz", common / "COMMON_LAYOUT_PAIRS.csv",
                 common / "COMMON_LAYOUT_SELECTIONS.csv", assignments_path]
        sources.update({p.resolve().as_posix(): sha(p) for p in files})
        permission = pd.read_csv(files[0])
        spatial.append(permission[permission.layout.isin(LAYOUTS)])
        placement = pd.read_csv(files[3])
        layouts.append(placement[placement.layout.eq("demand_ordered")])
        selection = pd.read_csv(files[4])
        selection = selection[selection.layout.isin(LAYOUTS) & selection.permission.eq("native_zone")
                              & selection.weight_view.eq("profile_equal") & selection.service_fraction.eq(1)].copy()
        assignment = pd.read_csv(assignments_path)
        assignment = assignment[assignment.layout.isin(LAYOUTS)]
        static = pd.read_csv(files[1])
        static = static[static.layout.isin(LAYOUTS)]
        zone_rows = []
        for key, group in assignment.groupby(keys + ["zone_id"], sort=True):
            temperatures = np.unique(group[["lower_c", "upper_c"]].to_numpy())
            simultaneous = ((group.lower_c.to_numpy()[:, None] <= temperatures)
                            & (temperatures <= group.upper_c.to_numpy()[:, None])).sum(axis=0)
            zone_rows.append(dict(zip(keys + ["zone_id"], key),
                                  assigned_profile_count=len(group), max_simultaneously_covered_profiles=int(simultaneous.max())))
        verified_static = pd.DataFrame(zone_rows)
        check = static.merge(verified_static, on=keys + ["zone_id"], validate="one_to_one", suffixes=("_saved", "_independent"))
        for metric in ["assigned_profile_count", "max_simultaneously_covered_profiles"]:
            assert check[f"{metric}_saved"].equals(check[f"{metric}_independent"])
        aggregate = verified_static.groupby(keys, as_index=False)[["assigned_profile_count", "max_simultaneously_covered_profiles"]].sum()
        assert aggregate.assigned_profile_count.eq(267).all()
        aggregate["static_equal_profile_coverage_bound"] = aggregate.max_simultaneously_covered_profiles / 267
        profiles = pd.read_csv(files[2])
        profiles = profiles[profiles.layout.isin(LAYOUTS)]
        prof = profiles.merge(assignment[keys + ["profile_id", "zone_id"]],
                              on=keys + ["profile_id"], validate="many_to_one")
        weight_spread = prof.groupby(keys + ["run_id", "zone_id"]).native_person_hour_weight.agg(lambda x: float(x.max() - x.min()))
        largest_within_zone_weight_spread = max(largest_within_zone_weight_spread, float(weight_spread.max()))
        assert weight_spread.le(1e-8).all()
        annual_equal = profiles.groupby(keys + ["run_id"], as_index=False).modeled_interval_coverage.mean()
        annual_equal = annual_equal.rename(columns={"modeled_interval_coverage": "independent_equal_profile_coverage"})
        joined = selection.merge(aggregate, on=keys, validate="many_to_one")
        joined = joined.merge(annual_equal, on=keys + ["run_id"], validate="many_to_one", how="left")
        available = joined.selection_status.eq("selected")
        error = (joined.loc[available, "modeled_interval_coverage"]
                 - joined.loc[available, "independent_equal_profile_coverage"]).abs()
        largest_coverage_error = max(largest_coverage_error, float(error.max()))
        assert error.le(1e-10).all()
        assert (joined.loc[available, "modeled_interval_coverage"]
                <= joined.loc[available, "static_equal_profile_coverage_bound"] + 1e-10).all()
        joined["static_to_annual_coverage_gap_percentage_points"] = 100 * (
            joined.static_equal_profile_coverage_bound - joined.modeled_interval_coverage)
        selections.append(joined)
        overlap.append(aggregate)
    permission = pd.concat(spatial, ignore_index=True)
    placement = pd.concat(layouts, ignore_index=True)
    capacity = pd.concat(overlap, ignore_index=True)
    realization = pd.concat(selections, ignore_index=True)
    assert permission.composition_id.nunique() == 6
    assert len(capacity) == 48 and len(realization) == 144
    assert not realization.duplicated(keys + ["control"]).any()
    for name, table in [("SPATIAL_PAIRS", permission), ("COMMON_LAYOUT_PAIRS", placement),
                        ("STATIC_COVERAGE_BOUND", capacity), ("STATIC_AND_ANNUAL_COVERAGE", realization)]:
        table.to_csv(output / f"{name}.csv", index=False)
    report = {"status": "PASS", "demand_compositions": 6, "layouts": LAYOUTS,
              "weather_contexts": 4, "static_conditions": len(capacity),
              "annual_profile_equal_selection_rows": len(realization),
              "annual_qualified_selection_rows": int(realization.selection_status.eq("selected").sum()),
              "largest_independent_coverage_error": largest_coverage_error,
              "largest_within_zone_profile_occupied_weight_difference": largest_within_zone_weight_spread,
              "overlap_independently_recomputed": True,
              "all_annual_equal_profile_coverages_below_static_bounds": True,
              "selection_contract": "Same random-placement absolute mean/tail limits and fixed profile identities; fraction 1; equal-profile aggregation. Each destination retains its existing native occupancy schedule.",
              "static_bound": "Sum of each existing zone's maximum simultaneous interval-overlap count, divided by 267; unconstrained temperature, no equipment or annual-operation claim.",
              "new_simulations": 0, "source_sha256": sources}
    (output / "JOINT_SPATIAL_QA.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=ROOT / "results/Main_comparison")
    parser.add_argument("--primary-assignments", type=Path,
                        default=ROOT / "data/inputs/energy_service/engineering/PROFILE_ASSIGNMENTS.csv.gz")
    parser.add_argument("--ctcd-root", type=Path, default=ROOT / "output/ctcd_parallel/replay")
    parser.add_argument("--ctcd-assignments", type=Path,
                        default=ROOT / "output/ctcd_parallel/physical/engineering/PROFILE_ASSIGNMENTS.csv.gz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/Joint_main/joint_spatial")
    args = parser.parse_args()
    inputs = [(args.primary_root / "annual", args.primary_root / "common_layout", args.primary_assignments),
              (args.ctcd_root / "annual", args.ctcd_root / "common_layout", args.ctcd_assignments)]
    derive(inputs, args.output_root)


if __name__ == "__main__":
    main()
