"""Freeze complete CTCD annual/event results as a portable retained-data package.

Only final execution decisions can be packaged. Exact hourly/QA artifacts and
full generated IDFs remain local data; this entry does not publish any files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import pandas as pd

from package_ctcd_external import sha
from run_population_controls import physical_hash, signature

FINAL = {"cached_qualified", "new_qualified", "cached_rejected", "new_rejected", "execution_failed"}
QUALIFIED = {"cached_qualified", "new_qualified"}


def checked_manifest(physical, kind, expected):
    path = physical / "engineering" / f"{kind.upper()}_RUN_MANIFEST.csv"
    summary_path = physical / "EXECUTION_SUMMARY.json"
    if not summary_path.is_file():
        raise ValueError(f"{kind} execution has not produced its final summary")
    frame = pd.read_csv(path, keep_default_na=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(frame) != expected or summary[f"logical_{kind}_inputs"] != expected:
        raise ValueError(f"The full {expected}-row {kind} plan is not finalized")
    keys = ["context", "composition_id", "layout", "permission", "control"]
    keys += ["variant"] if kind == "annual" else ["amplitude_c"]
    if frame.duplicated(keys).any() or frame.run_id.nunique() != summary["unique_inputs"]:
        raise ValueError("The completed logical plan is duplicated or lost physical inputs")
    if not frame.engineering_status.isin(FINAL).all():
        raise ValueError("A physical execution decision is unfinished")
    for _, group in frame.groupby("run_id"):
        row = group.iloc[0]
        if group.engineering_status.nunique() != 1:
            raise ValueError("One physical input has inconsistent final decisions")
        for column, checksum in (("path_hourly", "hourly_sha256"), ("qa_path", "qa_sha256")):
            if not row[column]:
                if row.engineering_status in QUALIFIED:
                    raise ValueError("Qualified trajectories need their actual hourly output and QA")
                continue
            source = Path(row[column])
            source = source if source.is_absolute() else path.parent / source
            if not source.is_file() or sha(source) != row[checksum]:
                raise ValueError("A physical output changed before packaging")
        if row.engineering_status != "execution_failed":
            qa = Path(row.qa_path)
            qa = qa if qa.is_absolute() else path.parent / qa
            if bool(json.loads(qa.read_text(encoding="utf-8"))["engineering_pass"]) != (row.engineering_status in QUALIFIED):
                raise ValueError("A rejected engineering decision was relabelled")
    return frame


def preserve_runs(frame, physical, destination):
    frame = frame.copy()
    for run_id, group in frame.groupby("run_id"):
        row = group.iloc[0]
        for column, checksum, directory, suffix in (
            ("path_hourly", "hourly_sha256", "hourly", ".npz"),
            ("qa_path", "qa_sha256", "qa", ".json"),
        ):
            if not row[column]:
                continue
            source = Path(row[column])
            source = source if source.is_absolute() else physical / "engineering" / source
            target = destination / directory / (run_id + suffix)
            if target.exists() and sha(target) != row[checksum]:
                raise ValueError("Annual/event artifact names collided with different contents")
            if not target.exists():
                shutil.copyfile(source, target)
            frame.loc[frame.run_id.eq(run_id), column] = f"../../{directory}/{target.name}"
    # Cached machine locations are execution provenance, not analysis inputs.
    columns = [c for c in frame if c != "idf_path" and
               (not c.startswith("cached_") or c in {"cached_run_id", "cached_cohort"})]
    return frame[columns]


def package(annual_physical, event_physical, annual_plan, event_plan, annual_analysis,
            original, destination, metadata, repository):
    annual = checked_manifest(annual_physical, "annual", 864)
    events = checked_manifest(event_physical, "event", 432)
    required = {
        annual_plan: ["CONTROL_COMMANDS.json", "PROFILE_ASSIGNMENTS.csv.gz",
                      "BASELINE_DECISION_CONTRACT.csv", "FIXED_TAIL_MEMBERSHIPS.csv",
                      "PARALLEL_COHORT_PROTOCOL.json", "ANNUAL_RUN_PLAN.csv", "PHYSICAL_RUN_PLAN.csv"],
        event_plan: ["EVENT_RUN_PLAN.csv", "PHYSICAL_EVENT_PLAN.csv", "EVENT_DATES_BEFORE_RUNS.json",
                     "EVENT_GENERATION_QA.json", "BASELINE_DECISION_CONTRACT.json"],
        annual_analysis: ["DR_BASELINES.csv"],
    }
    for folder, names in required.items():
        for name in names:
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
    destination.relative_to(repository)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("annual/engineering", "events/engineering", "hourly", "qa", "controller",
                 "protocol", "provenance/original_random_native/engineering",
                 "provenance/original_random_native/controller"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    annual_out = preserve_runs(annual, annual_physical, destination)
    events_out = preserve_runs(events, event_physical, destination)
    for name in ("annual", "events"):
        annual_out.to_csv(destination / name / "engineering/ANNUAL_RUN_MANIFEST.csv", index=False)
        shutil.copyfile(annual_plan / "PROFILE_ASSIGNMENTS.csv.gz",
                        destination / name / "engineering/PROFILE_ASSIGNMENTS.csv.gz")
    events_out.to_csv(destination / "events/engineering/EVENT_RUN_MANIFEST.csv", index=False)
    for name in ("BASELINE_DECISION_CONTRACT.csv", "FIXED_TAIL_MEMBERSHIPS.csv"):
        shutil.copyfile(annual_plan / name, destination / "annual/engineering" / name)
    for name in ("EVENT_DATES_BEFORE_RUNS.json", "EVENT_GENERATION_QA.json", "BASELINE_DECISION_CONTRACT.json"):
        shutil.copyfile(event_plan / name, destination / "events/engineering" / name)
    shutil.copyfile(annual_analysis / "DR_BASELINES.csv", destination / "protocol/DR_BASELINES_BEFORE_EVENTS.csv")
    for name in ("CTCD_EXECUTABLE_PROFILES.csv.gz", "CTCD_ALIGNED_OFFICE_RECORDS.csv.gz"):
        shutil.copyfile(original / "controller" / name, destination / "controller" / name)
    shutil.copyfile(annual_plan / "CONTROL_COMMANDS.json", destination / "controller/CONTROL_COMMANDS.json")
    provenance = destination / "provenance/original_random_native"
    shutil.copyfile(original / "engineering/PROFILE_ASSIGNMENTS.csv.gz",
                    provenance / "engineering/PROFILE_ASSIGNMENTS.csv.gz")
    old_manifest = pd.read_csv(original / "engineering/ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    for i, row in old_manifest.iterrows():
        for column, directory, suffix in (("path_hourly", "hourly", ".npz"), ("qa_path", "qa", ".json")):
            if row[column]:
                old_manifest.loc[i, column] = f"../../../{directory}/{row.run_id}{suffix}"
    old_manifest.to_csv(provenance / "engineering/ANNUAL_RUN_MANIFEST.csv", index=False)
    shutil.copyfile(original / "controller/CONTROL_COMMANDS.json", provenance / "controller/CONTROL_COMMANDS.json")
    # Full generated inputs and portable cache locations support optional engine
    # re-execution. Unmodified pre-run declarations retain the original audit.
    portable_prefix = destination.relative_to(repository).as_posix()
    for kind, plan, physical in (("annual", annual_plan, annual_physical), ("events", event_plan, event_physical)):
        target = destination / "model_inputs" / kind
        shutil.copytree(plan, target)
        original_plans = destination / "protocol" / (kind + "_original_plans")
        original_plans.mkdir()
        checked_idfs = set()
        for name in ("ANNUAL_RUN_PLAN.csv", "PHYSICAL_RUN_PLAN.csv", "EVENT_RUN_PLAN.csv", "PHYSICAL_EVENT_PLAN.csv"):
            if not (target / name).is_file():
                continue
            shutil.copyfile(target / name, original_plans / name)
            table = pd.read_csv(target / name, keep_default_na=False)
            for i, row in table.iterrows():
                if row.run_id not in checked_idfs:
                    model = target / row.idf_path
                    text = model.read_text(encoding="utf-8")
                    if signature(text) != row.normalized_idf_sha256 or physical_hash(text) != row.physical_sha256:
                        raise ValueError("A copied complete IDF no longer matches the executed input")
                    checked_idfs.add(row.run_id)
                if row.execution_status != "exact_retained_cache":
                    continue
                rid = row.run_id
                for column, directory, suffix in (("cached_hourly_path", "hourly", ".npz"),
                                                   ("cached_qa_path", "qa", ".json")):
                    artifact = destination / directory / (rid + suffix)
                    table.loc[i, column] = f"{portable_prefix}/{directory}/{rid}{suffix}" if artifact.is_file() else ""
                table.loc[i, "cached_idf_path"] = f"{portable_prefix}/model_inputs/{kind}/" + row.idf_path
                table.loc[i, "cached_normalized_idf_sha256"] = row.normalized_idf_sha256
            table.to_csv(target / name, index=False)
        for name in ("EXECUTION_PROTOCOL_BEFORE_RUNS.json", "EXECUTION_SUMMARY.json"):
            shutil.copyfile(physical / name, destination / "protocol" / (kind + "_" + name))
    report = {"status": "FROZEN_WITH_FINAL_ENGINEERING_DECISIONS", "annual_logical": len(annual),
              "event_logical": len(events), "annual_unique": annual.run_id.nunique(),
              "event_unique": events.run_id.nunique(), "annual_status": annual.engineering_status.value_counts().to_dict(),
              "event_status": events.engineering_status.value_counts().to_dict(),
              "source_annual_manifest_sha256": sha(annual_physical / "engineering/ANNUAL_RUN_MANIFEST.csv"),
              "source_event_manifest_sha256": sha(event_physical / "engineering/EVENT_RUN_MANIFEST.csv"),
              "large_energyplus_sql_eso_logs_included": False, "hourly_and_qa_values_changed": False,
              "scope": "Parallel CTCD two-layout, three-permission office comparison with own-baseline events; source cohort is kept separate from ASHRAE.",
              "raw_source_redistribution_rights_asserted": False}
    (destination / "protocol/PACKAGE_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    with (destination / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows({"path": p.relative_to(destination).as_posix(), "bytes": p.stat().st_size,
                          "sha256": sha(p)} for p in paths)
    metadata.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(destination / "MANIFEST.csv", metadata / "ctcd_parallel_bundle_manifest.csv")
    source = {"scope": report["scope"], "files_excluding_manifest": len(paths),
              "manifest_sha256": sha(destination / "MANIFEST.csv"), "annual_logical": len(annual),
              "event_logical": len(events), "dataset_doi": "10.1038/s41597-023-02568-3",
              "all_execution_decisions_final": True, "raw_source_redistribution_rights_asserted": False}
    (metadata / "ctcd_parallel_bundle_source.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("annual-physical-root", "event-physical-root", "annual-plan-root", "event-plan-root", "annual-analysis-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--original-input-root", type=Path, default=Path("data/inputs/ctcd_external"))
    parser.add_argument("--output-root", type=Path, default=Path("data/inputs/ctcd_parallel"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/metadata"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(package(*[getattr(args, name).resolve() for name in (
        "annual_physical_root", "event_physical_root", "annual_plan_root", "event_plan_root",
        "annual_analysis_root", "original_input_root", "output_root", "metadata_root", "repository_root")]), indent=2))


if __name__ == "__main__":
    main()
