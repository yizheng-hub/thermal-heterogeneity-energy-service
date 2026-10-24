"""Export and verify recorded engineering decisions; never rerun or retune models."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import zipfile

REPO = Path(__file__).resolve().parents[2]
MANIFEST = "engineering_qa_bundle_manifest.csv"
SOURCE = "engineering_qa_bundle_source.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()
                if key not in {"folder", "source_folder", "qa_path", "hourly_path", "path_hourly"}}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str) and (re.search(r"[A-Za-z]:[\\/]", value)
                                  or re.search(r"/(?:Users|home)/", value)):
        return "<absolute path excluded>"
    return value


def recorded_triggers(qa):
    """Index non-passing recorded fields; this does not reevaluate admission."""
    triggers = []
    for key, value in qa.items():
        if value is False:
            triggers.append(key + "=false")
        elif key in {"severe", "fatal", "unexplained_hvac_recurring_groups"} and value:
            triggers.append(key + ">0")
        elif key.startswith("direct_") and value:
            triggers.append(key)
        elif key in {"warnings", "all_recurring_warnings"}:
            for index, warning in enumerate(value):
                if warning.get("decision") == "requires_review":
                    triggers.append(f"{key}[{index}].decision=requires_review")
    return triggers


def reason_classes(qa, rejected, support_flags=None):
    if not rejected:
        return ["recorded_qualified"]
    classes = []
    if qa.get("direct_numerical_warnings"):
        classes.append("direct_hvac_numerical_warning")
    recurring = qa.get("warnings", []) + qa.get("all_recurring_warnings", [])
    if any(w.get("decision") == "requires_review" for w in recurring):
        classes.append("unresolved_recurring_warning")
    if support_flags and support_flags["direct_coil_range_warning"] == "True":
        classes.append("recorded_direct_coil_range_flag")
    if not classes:
        classes.append("other_recorded_gate_rejection")
    return classes


def build(args):
    root, data, out = args.local_source_root.resolve(), args.paper_data_root.resolve(), args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    annual_path = data / "energy_service/engineering/ANNUAL_RUN_MANIFEST.csv"
    annual = read_csv(annual_path)
    groups = defaultdict(list)
    for row in annual:
        groups[row["run_id"]].append(row)
    if len(groups) != 407:
        raise ValueError("Expected the declared 407 independent annual inputs")
    support = root / "archive/results/three_control_main/balanced_retargeted_sensitivity_20261021_v1"
    rejected_path = support / "NEW_ENGINEERING_REJECTION_CAUSES.csv"
    support_rows = read_csv(rejected_path)
    if len(support_rows) != 153:
        raise ValueError("Expected the separate 153 supporting sensitivity exclusions")
    with gzip.open(support / "ALL_NATIVE_EVALUATED_CANDIDATES.csv.gz", "rt", encoding="utf-8-sig", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    strategies = defaultdict(set)
    for row in candidates:
        strategies[row["folder"]].add(row["strategy"])
    ledgers = []
    sources = []

    def record(scope, run_id, context, controls, status, source, logical_count, flags=None):
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        rejected = "rejected" in status
        if raw.get("engineering_pass") is not (not rejected):
            raise ValueError(f"Recorded admission disagrees with original QA: {scope}/{run_id}")
        cleaned = sanitize(raw)
        relative = f"qa/{scope}/{context}/{run_id}.json"
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, cleaned)
        row = {"experiment_scope": scope, "run_id": run_id, "context": context,
               "control": "|".join(sorted(controls)), "engineering_status": status,
               "recorded_engineering_pass": raw["engineering_pass"],
               "reason_classes": "|".join(reason_classes(raw, rejected, flags)),
               "trigger_fields": "|".join(recorded_triggers(raw)),
               "logical_reference_count": logical_count,
               "qa_relative_path": relative, "original_qa_sha256": sha(source),
               "sanitized_qa_sha256": sha(destination),
               "original_source_locator": source.relative_to(root).as_posix(),
               "recorded_direct_numerical_warning": bool(raw.get("direct_numerical_warnings")),
               "support_recorded_coil_flag": flags["direct_coil_range_warning"] if flags else "not_applicable"}
        ledgers.append(row)
        sources.append({"original_source_locator": row["original_source_locator"],
                        "original_sha256": row["original_qa_sha256"],
                        "sanitized_relative_path": relative,
                        "sanitized_sha256": row["sanitized_qa_sha256"]})

    for run_id, rows in sorted(groups.items()):
        contexts = {r["context"] for r in rows}
        statuses = {r["engineering_status"] for r in rows}
        if len(contexts) != 1 or len(statuses) != 1:
            raise ValueError("Shared physical input has inconsistent context or admission")
        source = root / "upstream_inputs/frozen_buildings/current_run_gates" / run_id / "ENGINEERING_QA.json"
        record("main_annual", run_id, rows[0]["context"], {r["control"] for r in rows},
               rows[0]["engineering_status"], source, len(rows))
    for row in support_rows:
        parts = PureWindowsPath(row["folder"]).parts
        source = root / "archive" / Path(*parts[parts.index("results"):]) / "ENGINEERING_QA.json"
        record("supporting_retargeted_exclusions", parts[-1], row["city"],
               strategies[row["folder"]], "recorded_rejected_attempt", source,
               sum(c["folder"] == row["folder"] for c in candidates), row)
    write_csv(out / "ENGINEERING_RUN_LEDGER.csv", ledgers)
    write_csv(out / "ORIGINAL_QA_SOURCE_MAP.csv", sources)
    write_csv(out / "MAIN_LOGICAL_RUN_REFERENCES.csv", annual)
    context_controls = defaultdict(list)
    for row in annual:
        context_controls[(row["context"], row["control"])].append(row)
    qualification = []
    for (context, control), rows in sorted(context_controls.items()):
        physical = {row["run_id"]: row for row in rows}
        admitted = sum("qualified" in row["engineering_status"] for row in physical.values())
        logical_admitted = sum("qualified" in row["engineering_status"] for row in rows)
        qualification.append({"context": context, "control": control,
            "logical_rows_including_declared_aliases": len(rows),
            "qualified_logical_rows": logical_admitted,
            "rejected_logical_rows": len(rows) - logical_admitted,
            "unique_inputs_within_context_control": len(physical),
            "qualified_unique_inputs": admitted, "rejected_unique_inputs": len(physical) - admitted,
            "counting_note": "Shared inputs occur in multiple controls; do not sum this column to obtain 407 unique main inputs."})
    write_csv(out / "MAIN_QUALIFICATION_BY_CONTEXT_CONTROL.csv", qualification)
    main_rejected = [r for r in ledgers if r["experiment_scope"] == "main_annual"
                     and not r["recorded_engineering_pass"]]
    if len(main_rejected) != 109:
        raise ValueError("Main rejection count differs from the existing manifest")
    summary = {"status": "PASS", "purpose": "Trace existing decisions; no gate changes or new simulations",
               "main_annual_unique_inputs": 407, "main_qualified": 298, "main_rejected": 109,
               "supporting_retargeted_excluded_attempts": 153,
               "main_rejection_contexts": dict(Counter(r["context"] for r in main_rejected)),
               "main_reason_classes": dict(Counter(r["reason_classes"] for r in main_rejected)),
               "support_reason_classes": dict(Counter(r["reason_classes"] for r in ledgers
                   if r["experiment_scope"] == "supporting_retargeted_exclusions")),
               "recorded_source_hashes": {"main_annual_manifest": sha(annual_path),
                   "support_rejection_causes": sha(rejected_path),
                   "support_evaluated_candidates": sha(support / "ALL_NATIVE_EVALUATED_CANDIDATES.csv.gz")},
               "sanitization": "Only absolute-path fields/values removed; original hashes retained",
               "classification": "Reason labels index recorded warnings and the supporting cause flags. They do not replace original engineering decisions.",
               "scope_boundary": "407 main annual inputs and 153 supporting sensitivity exclusions are distinct inventories and must not be summed as an experimental denominator.",
               "not_included": "Native model/weather/executables, raw SQL and full err/end logs remain external. Original QA hashes refer to author-retained originals.",
               "physical_simulations_run": 0, "building_parameters_changed": False}
    write_json(out / "PACKAGE_QA.json", summary)
    manifest_rows = [{"path": f.relative_to(out).as_posix(), "bytes": f.stat().st_size, "sha256": sha(f)}
                     for f in sorted(out.rglob("*")) if f.is_file()]
    write_csv(out / "MANIFEST.csv", manifest_rows)
    archive = out.parent / "engineering_qa_supplement_v1.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(out).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, path.read_bytes())
    metadata = {"archive_filename": archive.name, "sha256": sha(archive),
                "bytes": archive.stat().st_size, "manifest_sha256": sha(out / "MANIFEST.csv"),
                "declared_files": len(manifest_rows), "manifest_files": 1,
                "qa_records": len(ledgers), "source_trajectories_changed": False,
                "scope": "Recorded engineering-QA sidecar; no native inputs or new simulations"}
    if args.metadata_root:
        args.metadata_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / "MANIFEST.csv", args.metadata_root / MANIFEST)
        write_json(args.metadata_root / SOURCE, metadata)
    print(json.dumps(metadata, indent=2))


def verify(root, metadata_root):
    metadata = json.loads((metadata_root / SOURCE).read_text(encoding="utf-8"))
    manifest = metadata_root / MANIFEST
    if sha(manifest) != metadata["manifest_sha256"] or sha(root / "MANIFEST.csv") != sha(manifest):
        raise ValueError("Engineering-QA manifest checksum differs")
    rows = read_csv(manifest)
    expected = {r["path"] for r in rows} | {"MANIFEST.csv"}
    actual = {f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()}
    if actual != expected:
        raise ValueError("Engineering-QA file set differs")
    for row in rows:
        path = root / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise ValueError(f"Engineering-QA file checksum differs: {row['path']}")
    ledger = read_csv(root / "ENGINEERING_RUN_LEDGER.csv")
    for row in ledger:
        if sha(root / row["qa_relative_path"]) != row["sanitized_qa_sha256"]:
            raise ValueError("QA source-map mismatch")
    return {"status": "PASS", "files": len(rows), "qa_records": len(ledger), "physical_simulations_run": 0}


def unpack(args):
    metadata = json.loads((args.metadata_root / SOURCE).read_text(encoding="utf-8"))
    if sha(args.archive) != metadata["sha256"]:
        raise ValueError("Engineering-QA ZIP checksum differs")
    expected = {r["path"] for r in read_csv(args.metadata_root / MANIFEST)} | {"MANIFEST.csv"}
    with zipfile.ZipFile(args.archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Unexpected or duplicate engineering-QA ZIP entries")
        for info in zipped.infolist():
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in info.filename or ":" in info.filename:
                raise ValueError("Unsafe engineering-QA ZIP path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Engineering-QA archive cannot contain symbolic links")
        args.dest.mkdir(parents=True, exist_ok=False)
        zipped.extractall(args.dest)
    print(json.dumps(verify(args.dest.resolve(), args.metadata_root), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--local-source-root", type=Path, required=True)
    builder.add_argument("--paper-data-root", type=Path, required=True)
    builder.add_argument("--output-root", type=Path, required=True)
    builder.add_argument("--metadata-root", type=Path)
    checker = commands.add_parser("verify")
    checker.add_argument("--root", type=Path, required=True)
    extractor = commands.add_parser("unpack")
    extractor.add_argument("--archive", type=Path, required=True)
    extractor.add_argument("--dest", type=Path, required=True)
    for command in (checker, extractor):
        command.add_argument("--metadata-root", type=Path, default=REPO / "data/metadata")
    args = parser.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "verify":
        print(json.dumps(verify(args.root.resolve(), args.metadata_root), indent=2))
    else:
        unpack(args)


if __name__ == "__main__":
    main()
