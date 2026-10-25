"""Joint control evidence for six fixed compositions on the shared spatial design.

Archive labels retain provenance. Analyses compare controllers within each fixed
composition; they do not estimate a difference between national populations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from service_frontier import run as service_frontier


LAYOUTS = {"random_mixing", "demand_ordered"}


def common_layouts(frame):
    return frame.loc[frame.layout.isin(LAYOUTS)].copy() if "layout" in frame else frame.copy()


def distinct_permissions(frame):
    return frame.loc[~(frame.permission.eq("floor") & frame.context.str.contains("small2019"))].copy()


def combine(paths):
    frames = []
    for label, path in paths:
        frame = common_layouts(pd.read_csv(path, keep_default_na=False, na_values=[""]))
        frame["data_source"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def signed_enduses(pairs, manifests):
    """Sum signed hourly changes before averaging; means retain additive closure."""
    locations = {}
    for manifest in manifests:
        data = pd.read_csv(manifest, keep_default_na=False)
        for row in data.itertuples(index=False):
            if row.path_hourly:
                path = Path(row.path_hourly)
                locations[row.run_id] = path if path.is_absolute() else manifest.parent / path
    cache = {}

    def loads(run):
        if run not in cache:
            with np.load(locations[run], allow_pickle=False) as stored:
                cache[run] = np.array([stored["heating_electric_kwh"], stored["cooling_electric_kwh"],
                    stored["fans_electric_kwh"], stored["gas_hhv_kwh"]])
        return cache[run]

    rows, errors = [], []
    cases = pairs.query("comparison_mode == 'direct' and permission == 'native_zone' and status == 'matched'")
    for row in cases.itertuples(index=False):
        delta = loads(row.alternative_run_id) - loads(row.reference_run_id)
        errors.append(abs(delta.sum() - row.delta_annual_site_kwh))
        for component, values in zip(("heating_electric", "cooling_electric", "fans_electric", "gas_hhv"), delta):
            rows.append({key: getattr(row, key) for key in ("context", "composition_id", "layout", "control", "data_source")}
                | {"component": component, "added_kwh": np.maximum(values, 0).sum(),
                   "avoided_kwh": np.maximum(-values, 0).sum(), "net_kwh": values.sum(),
                   "pair_net_site_kwh": row.delta_annual_site_kwh,
                   "floor_area_m2": 511.16 if "small2019" in row.context else 4982.19})
    error = max(errors, default=0.)
    if error > 1e-6:
        raise ValueError(f"Signed end-use ledger does not close: {error}")
    return pd.DataFrame(rows), error


def cooling_offsets(pairs):
    """Retain every direct native-zone pair, including opposite and null outcomes."""
    cases = pairs.query("comparison_mode == 'direct' and permission == 'native_zone' and status == 'matched'").copy()
    cases["cooling_decreased"] = cases.delta_annual_cooling_electric_kwh.lt(-1e-6)
    cases["cooling_saving_offset"] = (cases.cooling_decreased
        & cases.delta_annual_site_kwh.gt(1e-6) & cases.delta_annual_operational_co2_kg.gt(1e-6))
    cases["range_coverage_improved"] = cases.delta_interval_coverage_percentage_points.gt(1e-9)
    cases["mean_and_tail_mismatch_improved"] = (cases.delta_mean_reference_mismatch_c.lt(-1e-9)
        & cases.delta_fixed_tail_reference_mismatch_c.lt(-1e-9))
    return cases


def run(repository_root, primary_replay_root, ctcd_root, output_root):
    root, primary, ctcd, output = map(Path, (repository_root, primary_replay_root, ctcd_root, output_root))
    output.mkdir(parents=True, exist_ok=True)
    annual = output / "annual"
    annual.mkdir(exist_ok=True)
    composition_frames = []
    for label, path in (("ASHRAE", root / "data/inputs/energy_service/engineering/PROFILE_ASSIGNMENTS.csv.gz"),
        ("CTCD", root / "data/inputs/ctcd_parallel/annual/engineering/PROFILE_ASSIGNMENTS.csv.gz")):
        frame = pd.read_csv(path, float_precision="round_trip")
        frame = frame[frame.context.eq("Miami") & frame.layout.eq("random_mixing")].copy()
        frame = frame[["composition_id", "profile_id", "centre_c", "lower_c", "upper_c"]]
        frame["data_source"] = label
        frame["range_width_c"] = frame.upper_c - frame.lower_c
        composition_frames.append(frame)
    compositions = pd.concat(composition_frames, ignore_index=True)
    if len(compositions) != 1602 or not compositions.groupby("composition_id").size().eq(267).all():
        raise ValueError("Six actual fixed compositions must each contain 267 assigned profiles")
    compositions.to_csv(output / "DEMAND_COMPOSITION_PROFILES.csv", index=False)
    source_hashes = {}
    tables = {}
    primary_annual, ctcd_annual = primary / "energy_service/annual", ctcd / "replay/annual"
    for path in sorted(primary_annual.glob("*.csv")):
        other = ctcd_annual / path.name
        if not other.is_file():
            continue
        frame = combine((("ASHRAE", path), ("CTCD", other)))
        frame.to_csv(annual / path.name, index=False)
        tables[path.stem] = frame
        source_hashes["ASHRAE/annual/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes["CTCD/annual/" + path.name] = hashlib.sha256(other.read_bytes()).hexdigest()
    candidates = tables["ANNUAL_ALL_CANDIDATES"]
    if len(candidates) != 1728 or candidates.composition_id.nunique() != 6:
        raise ValueError("Joint evidence requires exactly six fixed compositions and 1728 logical annual candidates")
    distinct = distinct_permissions(candidates)
    if len(distinct[["context", "composition_id", "layout", "permission"]].drop_duplicates()) != 120:
        raise ValueError("Joint shared design requires 120 distinct spatial conditions")
    service_frontier(annual, output / "service")
    for folder, left, right in (("seasonal", primary / "seasonal", ctcd / "seasonal"),
        ("costs", ctcd / "ashrae_costs", ctcd / "ctcd_costs")):
        target = output / folder
        target.mkdir(exist_ok=True)
        for path in sorted(left.glob("*.csv")):
            other = right / path.name
            if not other.is_file():
                continue
            frame = combine((("ASHRAE", path), ("CTCD", other)))
            if path.name == "implementation_cost_scenarios.csv":
                original = frame.drop(columns="data_source")
                if len(original.drop_duplicates()) != 118:
                    raise ValueError("Implementation cost assumptions differ between sources")
                frame = original.drop_duplicates().reset_index(drop=True)
            frame.to_csv(target / path.name, index=False)
    ledger, residual = signed_enduses(tables["CONTROL_COMPARISONS"], (
        root / "data/inputs/energy_service/engineering/ANNUAL_RUN_MANIFEST.csv",
        root / "data/inputs/ctcd_parallel/annual/engineering/ANNUAL_RUN_MANIFEST.csv"))
    ledger.to_csv(output / "SIGNED_ENDUSE_REALLOCATION.csv", index=False)
    offsets = cooling_offsets(tables["CONTROL_COMPARISONS"])
    offsets.to_csv(output / "COOLING_REDUCTION_OFFSETS.csv", index=False)
    native = tables["CONTROL_COMPARISONS"].query("permission == 'native_zone'")
    metrics = ("delta_site_pct", "delta_annual_operational_co2_kg", "delta_interval_coverage_percentage_points",
               "delta_mean_reference_mismatch_c", "delta_fixed_tail_reference_mismatch_c")
    summaries = []
    for key, group in native.groupby(["comparison_mode", "service_fraction", "alternative"], sort=True):
        selected = group[group.status.eq("matched")]
        record = dict(zip(("comparison_mode", "service_fraction", "control"), key))
        record.update(designed_pairs=len(group), matched_pairs=len(selected))
        for metric in metrics:
            values = selected[metric]
            record.update({metric + "_median": values.median(), metric + "_min": values.min(),
                metric + "_max": values.max(), metric + "_positive": int(values.gt(1e-9).sum()),
                metric + "_negative": int(values.lt(-1e-9).sum()), metric + "_zero": int(values.abs().le(1e-9).sum())})
        summaries.append(record)
    pd.DataFrame(summaries).to_csv(output / "JOINT_NATIVE_CONTROL_SUMMARY.csv", index=False)
    report = {"status": "PASS", "data_sources": ["ASHRAE", "CTCD"], "source_contrast_is_main_estimand": False,
        "fixed_compositions": 6, "layouts": sorted(LAYOUTS), "distinct_spatial_conditions": 120,
        "logical_annual_candidates": 1728, "native_direct_rows": int(tables["ANNUAL_DIRECT"].permission.eq("native_zone").sum()),
        "enduse_hourly_closure_max_error_kwh": residual, "physical_parameters_changed": False,
        "cooling_offset_audit": {"matched_pairs": len(offsets),
            "cooling_decreased": int(offsets.cooling_decreased.sum()),
            "total_energy_and_carbon_increased": int(offsets.cooling_saving_offset.sum()),
            "offset_with_coverage_improvement": int((offsets.cooling_saving_offset & offsets.range_coverage_improved).sum()),
            "offset_with_mean_and_tail_improvement": int((offsets.cooling_saving_offset & offsets.mean_and_tail_mismatch_improved).sum())},
        "source_annual_sha256": source_hashes}
    (output / "JOINT_EVIDENCE_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("repository-root", "primary-replay-root", "ctcd-root", "output-root"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.repository_root, args.primary_replay_root, args.ctcd_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
