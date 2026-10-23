"""Export final engineering trajectories and anonymous profiles for publication.

Pending inputs are rejected. No failed trajectory is changed to qualified, and
no source folders, original record IDs or person crosswalks enter this bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


QUALIFIED = {"cached_qualified", "new_qualified"}
FINAL = QUALIFIED | {"cached_rejected", "new_rejected", "execution_failed", "unavailable_baseline"}
PROFILE_FIELDS = ["context", "composition_id", "layout", "profile_id", "zone_id",
                  "centre_c", "lower_c", "upper_c"]
HOURLY_FIELDS = ["temperatures", "people", "zone_ids", "occupied", "month", "day", "hour", "dst",
                 "hvac_electric_kwh", "gas_hhv_kwh", "electricity_hvac_meter_kwh",
                 "heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh",
                 "pumps_electric_kwh", "heating_setpoint_c", "cooling_setpoint_c"]
DESIGN_FIELDS = ["run_id", "context", "composition_id", "layout", "permission", "control", "variant",
                 "kind", "baseline_run_id", "engineering_status", "path_hourly",
                 "normalized_idf_sha256", "permission_alias", "amplitude_c", "baseline_hourly_path"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--controller-inputs", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--assumptions", type=Path, required=True)
    parser.add_argument("--dense-candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stage, out = args.stage_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Choose a new bundle directory")
    sources = {}
    for kind, expected in (("ANNUAL", 1296), ("EVENT", 648)):
        path = stage / "engineering" / f"{kind}_RUN_MANIFEST.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        if len(frame) != expected or not set(frame.engineering_status).issubset(FINAL):
            raise ValueError(f"{kind} collection is incomplete; no draft may be released")
        sources[kind] = frame
    generation = json.loads((stage / "engineering/GENERATION_QA.json").read_text(encoding="utf-8"))
    if generation["native_vectors_reproduced"] != 432 or generation["max_target_error_c"] != 0:
        raise ValueError("The exact native-command reproduction gate has not passed")
    if not generation["physical_fingerprints_unchanged"]:
        raise ValueError("Building physical parameters changed")
    out.mkdir(parents=True)
    engineering = out / "engineering"
    engineering.mkdir()
    hourly = out / "hourly"
    hourly.mkdir()
    copied = set()
    for kind, frame in sources.items():
        for index, row in frame.iterrows():
            if "baseline_hourly_path" in frame:
                frame.loc[index, "baseline_hourly_path"] = (
                    f"../hourly/{row.baseline_run_id}.npz" if row.baseline_run_id else "")
            if row.engineering_status not in QUALIFIED:
                frame.loc[index, "path_hourly"] = ""
                continue
            rid = str(row.run_id)
            source = Path(row.path_hourly)
            if not source.is_absolute():
                source = stage / "engineering" / source
            if rid not in copied:
                with np.load(source, allow_pickle=False) as stored:
                    arrays = {key: stored[key] for key in HOURLY_FIELDS}
                if arrays["temperatures"].shape[0] != 8760:
                    raise ValueError(f"Incomplete trajectory: {rid}")
                np.savez_compressed(hourly / f"{rid}.npz", **arrays)
                copied.add(rid)
            frame.loc[index, "path_hourly"] = f"../hourly/{rid}.npz"
        fields = [key for key in DESIGN_FIELDS if key in frame]
        frame[fields].to_csv(engineering / f"{kind}_RUN_MANIFEST.csv", index=False)
    profile_source = stage / "engineering/PROFILE_ASSIGNMENTS.csv.gz"
    profiles = pd.read_csv(profile_source)
    if list(profiles.columns) != PROFILE_FIELDS or not profiles.profile_id.str.fullmatch(r"sp_[0-9a-f]{24}").all():
        raise ValueError("Unexpected profile fields or nonopaque identifiers")
    shutil.copyfile(profile_source, engineering / profile_source.name)
    for name in ("EVENT_DATES_BEFORE_RUNS.json", "EVENT_GENERATION_QA.json",
                 "BASELINE_DECISION_CONTRACT.json", "ZONE_EXECUTION_GROUPS.csv"):
        shutil.copyfile(stage / "engineering" / name, engineering / name)
    commands = json.loads((stage / "engineering/CONTROL_DESIGNS_BEFORE_RUNS.json").read_text(encoding="utf-8"))
    target_keys = ["context", "composition_id", "layout", "permission", "control", "variant", "targets_c"]
    (engineering / "CONTROL_REFERENCE.json").write_text(json.dumps([
        {key: row[key] for key in target_keys} for row in commands], indent=2) + "\n", encoding="utf-8")
    shutil.copytree(args.controller_inputs / "controller", out / "controller")
    dense = pd.read_csv(args.dense_candidates)
    dense = dense[dense.strategy.eq("uniform") & dense.griffiths_coefficient.eq(.5)
                  & dense.interval_width_multiplier.eq(1) & dense.engineering_pass]
    fields = ["city", "seed", "allocation", "griffiths_coefficient",
              "interval_width_multiplier", "strategy", "engineering_pass", "mean"]
    selected_indices = set(dense.index)
    with args.dense_candidates.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for i, row in enumerate(csv.DictReader(stream)) if i in selected_indices]
    with (out / "controller/DENSE_UNIFORM_SCORES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    factors = out / "factors"
    factors.mkdir()
    shutil.copyfile(args.factors, factors / "regional_factors.csv")
    shutil.copyfile(args.assumptions, factors / "IMPLEMENTATION_ASSUMPTIONS.json")
    shutil.copyfile(stage / "experiments/portfolio/inputs/prototype_areas.csv", factors / "prototype_areas.csv")
    protocol = out / "protocol"
    protocol.mkdir()
    for name in ("ANALYSIS_PROTOCOL.json", "PORTFOLIO_COST_PROTOCOL.json", "COST_SCOPE_BOUNDARY.json"):
        shutil.copyfile(stage / "protocol" / name, protocol / name)
    records = [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
               for p in sorted(out.rglob("*")) if p.is_file()]
    with (out / "MANIFEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    report = {"status": "EXPORTED_FROM_FINAL_ENGINEERING_INPUTS", "input_files": len(records),
              "unique_qualified_hourly_trajectories": len(copied),
              "manifest_sha256": sha256(out / "MANIFEST.csv"),
              "raw_record_or_person_identifiers": False, "historical_source_paths": False,
              "physical_parameters_changed": False, "full_publication_reproduced": False}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
