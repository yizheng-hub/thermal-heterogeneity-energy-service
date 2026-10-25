"""Preserve the bounded external run results with contained relative paths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(alignment, plan, physical, destination, metadata):
    frame = pd.read_csv(physical / "engineering/ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    if len(frame) != 144 or not frame.engineering_status.isin(
            ["cached_qualified", "cached_rejected", "new_qualified", "new_rejected", "execution_failed"]).all():
        raise ValueError("All 144 declared external candidates need final execution decisions")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("controller", "engineering", "hourly", "qa", "protocol"):
        (destination / name).mkdir()
    for name in ("CTCD_EXECUTABLE_PROFILES.csv.gz", "CTCD_ALIGNED_OFFICE_RECORDS.csv.gz"):
        shutil.copyfile(alignment / name, destination / "controller" / name)
    shutil.copyfile(plan / "CONTROL_COMMANDS.json", destination / "controller/CONTROL_COMMANDS.json")
    for name in ("PROFILE_ASSIGNMENTS.csv.gz", "BASELINE_DECISION_CONTRACT.csv", "FIXED_TAIL_MEMBERSHIPS.csv"):
        shutil.copyfile(plan / name, destination / "engineering" / name)
    for folder, names in ((alignment, ("CTCD_ALIGNMENT_QA.json", "CTCD_FILTER_STAGES.csv", "CTCD_OVERLAP_GROUPS.csv")),
                          (plan, ("EXTERNAL_COHORT_PROTOCOL.json",)),
                          (physical, ("EXECUTION_PROTOCOL_BEFORE_RUNS.json", "EXECUTION_SUMMARY.json"))):
        for name in names:
            shutil.copyfile(folder / name, destination / "protocol" / name)
    for run_id, group in frame.groupby("run_id"):
        row = group.iloc[0]
        for column, check, directory, suffix in (("path_hourly", "hourly_sha256", "hourly", ".npz"),
                                                 ("qa_path", "qa_sha256", "qa", ".json")):
            if not row[column]:
                continue
            source = Path(row[column])
            if not source.is_absolute():
                source = physical / "engineering" / source
            if sha(source) != row[check]:
                raise ValueError("An execution artifact changed before preservation")
            target = destination / directory / (run_id + suffix)
            shutil.copyfile(source, target)
            frame.loc[frame.run_id.eq(run_id), column] = "../" + directory + "/" + target.name
    columns = ["run_id", "context", "composition_id", "layout", "permission", "control", "variant",
               "kind", "engineering_status", "path_hourly", "hourly_sha256", "qa_path", "qa_sha256",
               "normalized_idf_sha256", "physical_sha256", "weather", "weather_sha256",
               "execution_status", "cached_run_id", "permission_alias"]
    frame[columns].to_csv(destination / "engineering/ANNUAL_RUN_MANIFEST.csv", index=False)
    report = {"logical_candidates": len(frame), "unique_physical_inputs": frame.run_id.nunique(),
              "logical_engineering_status": frame.engineering_status.value_counts().to_dict(),
              "source_execution_manifest_sha256": sha(physical / "engineering/ANNUAL_RUN_MANIFEST.csv"),
              "raw_energyplus_sql_eso_and_logs_included": False,
              "physical_trajectory_values_changed": False, "paths": "Contained relative paths only",
              "scope": "Local computational input package; this preservation step does not assert third-party redistribution rights."}
    (destination / "protocol/PACKAGE_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths = [path for path in sorted(destination.rglob("*")) if path.is_file()]
    with (destination / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows({"path": p.relative_to(destination).as_posix(), "bytes": p.stat().st_size,
                          "sha256": sha(p)} for p in paths)
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination / "MANIFEST.csv", metadata / "ctcd_bundle_manifest.csv")
    alignment_qa = json.loads((alignment / "CTCD_ALIGNMENT_QA.json").read_text(encoding="utf-8"))
    execution = json.loads((physical / "EXECUTION_SUMMARY.json").read_text(encoding="utf-8"))
    source = {"scope": "Bounded CTCD external office-composition control comparison",
              "dataset_doi": alignment_qa["dataset_doi"], "source_raw_files": alignment_qa["raw_inputs"],
              "files_excluding_manifest": len(paths), "manifest_sha256": sha(destination / "MANIFEST.csv"),
              "new_annual_runs": execution["new_physical_runs"],
              "exact_retained_inputs": execution["exact_cache_reuses"],
              "source_alignment_qa_sha256": sha(alignment / "CTCD_ALIGNMENT_QA.json"),
              "source_control_protocol_sha256": sha(plan / "EXTERNAL_COHORT_PROTOCOL.json"),
              "raw_source_redistribution_rights_asserted": False}
    (metadata / "ctcd_bundle_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--physical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/metadata")
    args = parser.parse_args()
    print(json.dumps(package(args.alignment_root.resolve(), args.plan_root.resolve(),
                             args.physical_root.resolve(), args.output_root.resolve(),
                             args.metadata_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
