#!/usr/bin/env python3
"""Validate the frozen optional Level 2 EnergyPlus run boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath


EXPECTED_COLUMNS = [
    "manifest_version",
    "run_id",
    "run_class",
    "city",
    "idf_asset_id",
    "idf_sha256",
    "idf_package_path",
    "epw_asset_id",
    "epw_sha256",
    "epw_package_path",
    "rights_package_id",
    "rights_manifest_path",
    "energyplus_version",
    "command_template",
    "engineering_gate_id",
    "frozen_engineering_gate",
    "frozen_engineering_qa_sha256",
    "historical_source_locator",
    "asset_disposition",
]
EXPECTED_REUSE_COLUMNS = [
    "reuse_id",
    "city",
    "annual_baseline_run_id",
    "baseline_idf_asset_id",
    "baseline_idf_sha256",
    "epw_asset_id",
    "epw_sha256",
    "event_run_ids",
    "event_run_count",
    "reuse_rule",
]
SHA = re.compile(r"^[0-9a-f]{64}$")
VERSION = "22.1.0-ed759b17ee"
FROZEN_GENERATOR_SHA = "c1b0c7a94cfd906568aa9f9f3367120badd299ac2bc7daa73e58a4469ab96c63"
WORKTREE_GENERATOR_SHA = "433e5cc2702df509a6e93d74acbe18abd19d0a76de82506295cf3ee259387070"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _relative(value: str) -> bool:
    path = PurePosixPath(value.replace("\\", "/"))
    return bool(value) and not path.is_absolute() and ".." not in path.parts and not re.match(r"^[A-Za-z]:", value)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(
    manifest_path: Path, baseline_reuse_path: Path, contract_path: Path
) -> dict[str, object]:
    errors: list[str] = []
    if not manifest_path.is_file() or not baseline_reuse_path.is_file() or not contract_path.is_file():
        missing = [str(path) for path in (manifest_path, baseline_reuse_path, contract_path) if not path.is_file()]
        return {"status": "FAIL", "check": "level2_boundary", "errors": [f"missing files: {missing}"]}

    columns, rows = _read_csv(manifest_path)
    reuse_columns, reuse = _read_csv(baseline_reuse_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if columns != EXPECTED_COLUMNS:
        errors.append(f"run manifest columns differ: {columns}")
    if reuse_columns != EXPECTED_REUSE_COLUMNS:
        errors.append(f"baseline reuse columns differ: {reuse_columns}")

    counts = Counter(row.get("run_class") for row in rows)
    if len(rows) != 412 or counts != Counter({"annual": 346, "event": 66}):
        errors.append(f"run counts differ: total={len(rows)}, classes={dict(counts)}")
    run_ids = [row.get("run_id", "") for row in rows]
    if len(set(run_ids)) != len(run_ids):
        errors.append("run_id is not unique")
    source_locators = [row.get("historical_source_locator", "") for row in rows]
    if len(set(source_locators)) != len(source_locators):
        errors.append("historical_source_locator is not unique")

    by_id = {row.get("run_id", ""): row for row in rows}
    for index, row in enumerate(rows, start=2):
        label = f"row {index} ({row.get('run_id', '')})"
        for field in ("idf_sha256", "epw_sha256", "frozen_engineering_qa_sha256"):
            if not SHA.fullmatch(row.get(field, "")):
                errors.append(f"{label}: invalid {field}")
        if row.get("idf_asset_id") != f"idf_{row.get('idf_sha256', '')}":
            errors.append(f"{label}: IDF asset id is not content-addressed")
        for field in ("idf_package_path", "epw_package_path", "rights_manifest_path", "historical_source_locator"):
            if not _relative(row.get(field, "")):
                errors.append(f"{label}: {field} is not a safe relative path")
        if row.get("energyplus_version") != VERSION:
            errors.append(f"{label}: EnergyPlus version differs")
        if row.get("engineering_gate_id") != "LEVEL2_TRAJECTORY_GATE_V1":
            errors.append(f"{label}: engineering gate differs")
        if row.get("frozen_engineering_gate") != "PASS":
            errors.append(f"{label}: frozen engineering gate is not PASS")
        if row.get("asset_disposition") != "LOCAL_USER_SUPPLIED_NOT_DISTRIBUTED":
            errors.append(f"{label}: asset disposition differs")
        expected_command = (
            f"{{energyplus_executable}} -w {{input_package}}/{row.get('epw_package_path')} "
            f"-d {{output_root}}/{row.get('run_id')} {{input_package}}/{row.get('idf_package_path')}"
        )
        if row.get("command_template") != expected_command:
            errors.append(f"{label}: command template differs")

    if len(reuse) != 33 or len({row.get("reuse_id") for row in reuse}) != 33:
        errors.append(f"dynamic baseline reuse count differs: {len(reuse)}")
    referenced_events: list[str] = []
    for index, row in enumerate(reuse, start=2):
        label = f"reuse row {index} ({row.get('reuse_id', '')})"
        baseline = by_id.get(row.get("annual_baseline_run_id", ""))
        if baseline is None or baseline.get("run_class") != "annual":
            errors.append(f"{label}: baseline does not reference an annual run")
            continue
        event_ids = [value for value in row.get("event_run_ids", "").split("|") if value]
        referenced_events.extend(event_ids)
        if row.get("event_run_count") != "2" or len(event_ids) != 2:
            errors.append(f"{label}: each reused baseline must link exactly two event runs")
        if any(by_id.get(value, {}).get("run_class") != "event" for value in event_ids):
            errors.append(f"{label}: one or more linked runs are not event runs")
        if row.get("baseline_idf_asset_id") != baseline.get("idf_asset_id") or row.get("baseline_idf_sha256") != baseline.get("idf_sha256"):
            errors.append(f"{label}: baseline IDF does not match annual run")
        if row.get("epw_asset_id") != baseline.get("epw_asset_id") or row.get("epw_sha256") != baseline.get("epw_sha256"):
            errors.append(f"{label}: baseline EPW does not match annual run")
        if row.get("reuse_rule") != "EVENT_COUNTERFACTUAL_BASELINE_REUSED_NO_EXTRA_ENERGYPLUS_RUN":
            errors.append(f"{label}: reuse rule differs")
    event_set = {run_id for run_id, row in by_id.items() if row.get("run_class") == "event"}
    if len(referenced_events) != 66 or set(referenced_events) != event_set or len(set(referenced_events)) != 66:
        errors.append("event runs are not covered exactly once by the 33 baseline-reuse rows")

    run_counts = contract.get("run_counts", {})
    if run_counts != {"annual": 346, "event": 66, "total": 412, "dynamic_baselines_reused_from_annual": 33}:
        errors.append(f"contract run counts differ: {run_counts}")
    if contract.get("energyplus_version") != VERSION:
        errors.append("contract EnergyPlus version differs")
    provenance = contract.get("historical_annual_generator", {})
    if provenance.get("frozen_package_recorded_sha256") != FROZEN_GENERATOR_SHA:
        errors.append("frozen-package historical generator SHA differs")
    if provenance.get("later_working_tree_sha256") != WORKTREE_GENERATOR_SHA:
        errors.append("later working-tree generator SHA differs")
    if provenance.get("snapshot_available") is not False:
        errors.append("contract must state that the original generator snapshot is unavailable")

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "level2_boundary",
        "manifest_sha256": _file_sha(manifest_path),
        "baseline_reuse_sha256": _file_sha(baseline_reuse_path),
        "contract_sha256": _file_sha(contract_path),
        "run_count": len(rows),
        "run_class_counts": dict(sorted(counts.items())),
        "city_counts": dict(sorted(Counter(row.get("city") for row in rows).items())),
        "unique_idf_assets": len({row.get("idf_asset_id") for row in rows}),
        "unique_epw_assets": len({row.get("epw_asset_id") for row in rows}),
        "dynamic_baseline_reuse_count": len(reuse),
        "historical_generator_sha_boundary": {
            "frozen_package_recorded_sha256": FROZEN_GENERATOR_SHA,
            "later_working_tree_sha256": WORKTREE_GENERATOR_SHA,
            "same": FROZEN_GENERATOR_SHA == WORKTREE_GENERATOR_SHA,
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline-reuse", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest, args.baseline_reuse, args.contract)
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
