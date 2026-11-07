"""Build demand pools and occupant assignments directly from retained raw inputs."""

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
sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
sys.path.insert(0, str(ROOT / "src/analysis/targeted_extension"))
from utilities.generate_demand_inputs import build_cohort, numeric_scenes, verify_input  # noqa: E402
from ctcd_alignment import read_official, align_office, add_intervals  # noqa: E402
from global_matrix import parse_idf  # noqa: E402
from run_global_matrix import ZONES  # noqa: E402


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def zone_inventory(model):
    """Derive floor areas and design occupancy from the physical input model."""
    objects = parse_idf(Path(model).read_text(encoding="utf-8"))
    areas = dict.fromkeys(ZONES, 0.0)
    for obj in objects:
        if obj[0].lower() != "buildingsurface:detailed" or obj[2].lower() != "floor":
            continue
        if obj[4] not in areas:
            continue
        vertices = np.asarray(obj[12:], dtype=float).reshape(-1, 3)
        area = np.linalg.norm(np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)) / 2
        areas[obj[4]] += area
    people = {obj[2]: obj for obj in objects if obj[0].lower() == "people"}
    rows = []
    for zone in ZONES:
        obj = people[zone]
        if obj[4].lower() != "area/person" or areas[zone] <= 0:
            raise ValueError(f"Unsupported occupancy or missing floor surface in {zone}")
        nominal = areas[zone] / float(obj[7])
        rows.append(
            {
                "zone_id": zone,
                "floor_area_m2": areas[zone],
                "design_occupants": nominal,
                "profiles": int(np.round(nominal)),
            }
        )
    result = pd.DataFrame(rows)
    if result.profiles.sum() != 267:
        raise ValueError("Physical model no longer has the declared 267 occupancy slots")
    return result


def sample_assignments(pool, inventory, seeds, *, ashrae):
    rows = []
    zones = np.repeat(inventory.zone_id.to_numpy(), inventory.profiles.to_numpy())
    for seed in seeds:
        probability = pool.source_equal_probability.to_numpy() if ashrae else None
        selected = np.random.default_rng(seed).choice(len(pool), 267, replace=False, p=probability)
        sample = pool.iloc[selected].copy().reset_index(drop=True)
        sample["sample_position"] = np.arange(267)
        if ashrae:
            sample["profile_id"] = [
                "sp_" + hashlib.sha256(f"s{seed}_v{i:03d}".encode()).hexdigest()[:24]
                for i in range(267)
            ]
        sample = sample.iloc[np.random.default_rng(seed + 1000).permutation(267)].copy()
        sample["context"] = "Miami"
        sample["composition_id"] = seed
        sample["layout"] = "random_mixing"
        sample["zone_id"] = zones
        rows.append(sample)
    return pd.concat(rows, ignore_index=True)


def rebuild(data_root, model, output, workers):
    if output.exists():
        raise FileExistsError("Use a new output directory; existing evidence is never overwritten")
    raw_root = data_root / "upstream/ashrae_db_ii"
    measurements = raw_root / "db_measurements_v2.1.0.csv.gz"
    metadata = raw_root / "db_metadata.csv"
    verify_input(measurements, "measurements")
    verify_input(metadata, "metadata")
    scenes_path = data_root / "metadata/numeric_scene_spec_v1.json"
    scenes = numeric_scenes(scenes_path, numeric_only=True)
    measurement = pd.read_csv(measurements, low_memory=False)
    meta = pd.read_csv(metadata, low_memory=False)
    output.mkdir(parents=True)
    inventory = zone_inventory(model)
    inventory.to_csv(output / "ZONE_INVENTORY.csv", index=False, float_format="%.17g")
    cohort, scope = build_cohort(measurement, meta)
    print(f"ASHRAE raw cohort: {len(cohort)} records; replaying numerical intervals", flush=True)
    cohort = add_intervals(cohort, scenes, workers)
    # Keep the original CSV parsing boundary used before sampling.
    cohort.to_csv(
        output / "ASHRAE_RECORDS.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    cohort = pd.read_csv(output / "ASHRAE_RECORDS.csv.gz", low_memory=False)
    eligible = cohort.loc[cohort.main_control_eligible].copy()
    eligible["unit_median_c"] = eligible.groupby("unit_group").reference_g050_c.transform("median")
    eligible["median_distance"] = abs(eligible.reference_g050_c - eligible.unit_median_c)
    pool = (
        eligible.sort_values(["unit_group", "median_distance", "record_id"])
        .drop_duplicates("unit_group")
        .copy()
    )
    pool["source_equal_probability"] = 1 / pool.groupby("source_group").unit_group.transform(
        "count"
    )
    pool["source_equal_probability"] /= pool.source_equal_probability.sum()
    pool["centre_c"] = pool.reference_g050_c
    ashrae = sample_assignments(
        pool.reset_index(drop=True), inventory, (2026102001, 2026102002, 2026102003), ashrae=True
    )
    archive, raw_manifest = read_official(data_root / "upstream/chinese_thermal_comfort")
    selected, stages, overlap = align_office(archive, measurement, meta)
    print(
        f"CTCD aligned cohort: {len(selected)} records; replaying numerical intervals", flush=True
    )
    china = add_intervals(selected, scenes, workers)
    china.to_csv(
        output / "CTCD_RECORDS.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    china = pd.read_csv(output / "CTCD_RECORDS.csv.gz", low_memory=False)
    china_pool = china.loc[china.main_control_eligible & china.active_conditioning].copy()
    ctcd = sample_assignments(
        china_pool, inventory, (2026102501, 2026102502, 2026102503), ashrae=False
    )
    fields = [
        "context",
        "composition_id",
        "layout",
        "profile_id",
        "zone_id",
        "centre_c",
        "lower_c",
        "upper_c",
        "controller_lower_c",
        "controller_upper_c",
    ]
    for name, frame in (("ASHRAE_ASSIGNMENTS", ashrae), ("CTCD_ASSIGNMENTS", ctcd)):
        frame.to_csv(
            output / f"{name}_LINKED.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
            float_format="%.17g",
        )
        frame[fields].to_csv(
            output / f"{name}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
            float_format="%.17g",
        )
    np.savez_compressed(
        output / "ASHRAE_ASSIGNMENTS.npz",
        **{
            c: ashrae[c].to_numpy(dtype=str) if ashrae[c].dtype == object else ashrae[c].to_numpy()
            for c in fields
        },
    )
    pool.to_csv(
        output / "ASHRAE_ANALYSIS_UNIT_POOL.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    china_pool.to_csv(
        output / "CTCD_EXECUTABLE_POOL.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    stages.to_csv(output / "CTCD_FILTER_STAGES.csv", index=False)
    inputs = [measurements, metadata, scenes_path, model]
    inputs += [
        data_root
        / "upstream/chinese_thermal_comfort"
        / f"Chinese Thermal Comfort Dataset Class {k}.csv"
        for k in ("I", "II", "III")
    ]
    report = {
        "status": "completed",
        "input_sha256": {p.relative_to(data_root.parent).as_posix(): file_hash(p) for p in inputs},
        "generator_sha256": file_hash(Path(__file__)),
        "ashrae_scope": scope,
        "ashrae_eligible_records": len(eligible),
        "ashrae_analysis_unit_pool": len(pool),
        "ctcd_aligned_records": len(china),
        "ctcd_active_records": int(china.active_conditioning.sum()),
        "ctcd_executable_pool": len(china_pool),
        "ctcd_raw_inputs": raw_manifest,
        "assignment_rows": len(ashrae) + len(ctcd),
        "ctcd_overlap_rule": overlap,
        "identity_links": "LINKED tables retain source record and available source/building/subject keys. CTCD IDs identify records, not verified people.",
        "derived_inputs_read": [],
        "historical_archive_required": False,
        "outputs": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "PROFILE_REBUILD_AUDIT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "ashrae_analysis_unit_pool",
                    "ctcd_executable_pool",
                    "assignment_rows",
                )
            }
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "data/upstream/frozen_buildings/Miami/baseline.idf"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rebuild(
        args.data_root.resolve(), args.model.resolve(), args.output_root.resolve(), args.workers
    )


if __name__ == "__main__":
    main()
