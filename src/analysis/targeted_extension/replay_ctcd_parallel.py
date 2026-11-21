"""Replay the parallel CTCD annual design from actual retained trajectories.

This entry regenerates all 864 commands, verifies the original 144 random-zone
inputs and engineering decisions, and rescoring leaves the original data intact.
It runs no EnergyPlus simulation and does not change the primary ASHRAE cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from ctcd_controls import ENERGY_SERVICE, QUALIFIED, digest, generate, load_hourly
from ctcd_parallel import LAYOUTS, parallel_profiles
from replay_ctcd_external import verify_contract
from bundle import verify_bundle

DESIGN_KEYS = ["context", "composition_id", "layout", "permission", "control", "variant"]


def verify_engineering(engineering):
    designs = pd.read_csv(engineering / "ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    if len(designs) != 864 or designs.duplicated(DESIGN_KEYS).any():
        raise ValueError("Parallel CTCD design must retain all 864 unique logical candidates")
    if set(designs.layout) != set(LAYOUTS) or set(designs.permission) != {"building", "floor", "native_zone"}:
        raise ValueError("The declared CTCD layouts or permissions changed")
    counts = {}
    for rid, group in designs.groupby("run_id"):
        row = group.iloc[0]
        if group.engineering_status.nunique() != 1 or group.qa_path.nunique() != 1:
            raise ValueError("One physical input has inconsistent engineering decisions")
        status = row.engineering_status
        counts[status] = counts.get(status, 0) + 1
        if status == "execution_failed":
            if row.path_hourly:
                raise ValueError("An incomplete execution cannot enter as a usable trajectory")
            continue
        if status not in QUALIFIED | {"cached_rejected", "new_rejected"}:
            raise ValueError("Parallel engineering collection contains an unfinished status")
        qa_path = Path(row.qa_path)
        qa_path = qa_path if qa_path.is_absolute() else engineering / qa_path
        if digest(qa_path) != row.qa_sha256:
            raise ValueError("A retained engineering QA record changed")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if bool(qa["engineering_pass"]) != (status in QUALIFIED):
            raise ValueError("A rejected original engineering decision was relabelled")
        if not row.path_hourly:
            if status in QUALIFIED:
                raise ValueError("Qualified operation requires its actual hourly data")
            continue
        hourly = Path(row.path_hourly)
        hourly = hourly if hourly.is_absolute() else engineering / hourly
        if digest(hourly) != row.hourly_sha256:
            raise ValueError("A retained hourly trajectory changed")
        load_hourly(hourly)
    return designs, {"logical_candidates": len(designs), "unique_physical_inputs": designs.run_id.nunique(),
                     "unique_status_counts": counts}


def verify_original_native(designs, engineering, original):
    old = pd.read_csv(original / "engineering/ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    current = designs[designs.layout.eq("random_mixing") & designs.permission.eq("native_zone")]
    paired = current.merge(old, on=DESIGN_KEYS, suffixes=("_parallel", "_old"), validate="one_to_one")
    if len(paired) != 144:
        raise ValueError("The original random/native design is incomplete")
    for name in ("run_id", "normalized_idf_sha256", "physical_sha256", "weather_sha256"):
        if not paired[name + "_parallel"].eq(paired[name + "_old"]).all():
            raise ValueError("The original random/native physical input changed")
    for name in ("hourly_sha256", "qa_sha256"):
        if not paired[name + "_parallel"].eq(paired[name + "_old"]).all():
            raise ValueError("The original random/native physical output or QA changed")
    if not np.array_equal(paired.engineering_status_parallel.isin(QUALIFIED),
                          paired.engineering_status_old.isin(QUALIFIED)):
        raise ValueError("Original random/native engineering availability changed")
    new_profiles = pd.read_csv(engineering / "PROFILE_ASSIGNMENTS.csv.gz", float_precision="round_trip")
    old_profiles = pd.read_csv(original / "engineering/PROFILE_ASSIGNMENTS.csv.gz", float_precision="round_trip")
    pd.testing.assert_frame_equal(new_profiles[new_profiles.layout.eq("random_mixing")].reset_index(drop=True),
                                  old_profiles, check_exact=True)
    return {"logical_candidates": 144, "profile_assignment_rows": len(old_profiles),
            "physical_inputs_hourly_arrays_and_engineering_decisions": "UNCHANGED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/inputs/ctcd_parallel"))
    parser.add_argument("--main-input-root", type=Path, default=Path("data/inputs/energy_service"))
    parser.add_argument("--original-input-root", type=Path,
                        help="Defaults to packaged original-native provenance or data/inputs/ctcd_external")
    parser.add_argument("--command-file", type=Path,
                        help="Defaults to the input package's controller/CONTROL_COMMANDS.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, help="Directory containing the supplied CTCD archive manifest and checksum record")
    args = parser.parse_args()
    root, main_input, output = (p.resolve() for p in (
        args.input_root, args.main_input_root, args.output_root))
    bundle_qa = {"mode": "raw_execution_output", "frozen_package_verified": False}
    if (root / "MANIFEST.csv").is_file():
        metadata = args.metadata_root or Path(__file__).resolve().parents[3] / "data/metadata"
        expected = metadata / "ctcd_parallel_bundle_manifest.csv"
        source = json.loads((metadata / "ctcd_parallel_bundle_source.json").read_text(encoding="utf-8"))
        if digest(root / "MANIFEST.csv") != digest(expected) or digest(expected) != source["manifest_sha256"]:
            raise ValueError("Parallel retained package differs from its tracked specification")
        bundle_qa = {"mode": "frozen_retained_package", "verification": verify_bundle(root, root / "MANIFEST.csv")}
    original = (args.original_input_root.resolve() if args.original_input_root else
                root / "provenance/original_random_native" if (root / "provenance/original_random_native").is_dir()
                else Path("data/inputs/ctcd_external").resolve())
    engineering = root / "annual/engineering" if (root / "annual/engineering").is_dir() else root / "engineering"
    command_file = args.command_file.resolve() if args.command_file else root / "controller/CONTROL_COMMANDS.json"
    contexts = json.loads((main_input / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    pool_path = root / "controller/CTCD_EXECUTABLE_PROFILES.csv.gz"
    if not pool_path.is_file():
        pool_path = original / "controller/CTCD_EXECUTABLE_PROFILES.csv.gz"
    pool = pd.read_csv(pool_path)
    old_profiles = pd.read_csv(original / "engineering/PROFILE_ASSIGNMENTS.csv.gz", float_precision="round_trip")
    profiles = parallel_profiles(pool, main_input, contexts, old_profiles)
    retained = pd.read_csv(engineering / "PROFILE_ASSIGNMENTS.csv.gz", float_precision="round_trip")
    pd.testing.assert_frame_equal(profiles, retained, check_exact=True)
    commands = generate(main_input, profile_tables={c: profiles[profiles.context.eq(c)] for c in contexts},
                        permissions=("building", "floor", "native_zone"), expected_commands=864)
    if commands != json.loads(command_file.read_text(encoding="utf-8")):
        raise ValueError("The parallel CTCD commands are not exactly regenerated")
    designs, engineering_qa = verify_engineering(engineering)
    original_qa = verify_original_native(designs, engineering, original)
    output.mkdir(parents=True, exist_ok=False)
    result = subprocess.run([sys.executable, str(ENERGY_SERVICE / "analyse_annual.py"),
                             "--engineering-root", str(engineering),
                             "--factors", str(main_input / "factors/regional_factors.csv"),
                             "--output-root", str(output / "annual"), "--expected-designs", "864",
                             "--expected-profile-groups", "24"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    (output / "ANNUAL_REPLAY.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Parallel annual scoring failed; see ANNUAL_REPLAY.log")
    contract = verify_contract(output / "annual", engineering, expected_anchors=24)
    report = {"status": "PASS", "commands_regenerated": len(commands), "bundle": bundle_qa, "engineering": engineering_qa,
              "original_random_native": original_qa, "service_contract": contract,
              "physical_simulations_run": 0, "physical_parameters_changed": False,
              "original_cohort_changed": False, "scope": "Parallel CTCD annual random/ordered design with building, floor and native-zone permissions",
              "input_sha256": {"commands": digest(command_file),
                  "engineering_manifest": digest(engineering / "ANNUAL_RUN_MANIFEST.csv"),
                  "profile_assignments": digest(engineering / "PROFILE_ASSIGNMENTS.csv.gz")}}
    (output / "CTCD_PARALLEL_REPLAY_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
