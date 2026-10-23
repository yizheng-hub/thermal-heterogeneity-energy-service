#!/usr/bin/env python3
"""Validate the locked 28-class source/licence/privacy decision manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED_COLUMNS = [
    "asset_id",
    "asset_title",
    "results_used",
    "upstream_reference",
    "version_or_hash",
    "licence_status",
    "privacy_or_rights_content",
    "release_class",
    "github_decision",
    "archival_decision",
    "user_download_required",
    "required_transform",
    "blocking_issue",
    "decision_status",
]

EXPECTED_ASSET_IDS = {
    "S01", "S01D", "S02", "S02X", "S03", "S03D", "S04", "S04X", "S04D",
    "S05", "S05X", "S05D", "S06", "S07", "S08A", "S08B", "S08C", "S08D",
    "S09A", "S09B", "S09D", "S10A", "S10B", "S10C", "S10D", "S10E", "S10F", "S11",
}

ALLOWED_RELEASE_CLASSES = {
    "DO_NOT_RELEASE",
    "GITHUB_DERIVED_AGGREGATE_ONLY",
    "GITHUB_DIRECT",
    "USER_DOWNLOAD_REQUIRED",
    "ZENODO_DERIVED_MINIMIZED",
    "ZENODO_LARGE_AUDIT_OPTIONAL",
}

WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)\b[a-z]:[\\/]")
POSIX_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
UNKNOWN = re.compile(r"(?i)\bunknown\b")


def validate_manifest(path: Path, require_no_unknown: bool = False) -> dict[str, object]:
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if fieldnames != REQUIRED_COLUMNS:
        errors.append(f"columns differ from locked schema: {fieldnames}")

    if len(rows) != 28:
        errors.append(f"expected 28 asset classes, found {len(rows)}")

    ids = [row.get("asset_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("asset_id values are not unique")
    if set(ids) != EXPECTED_ASSET_IDS:
        errors.append(
            f"asset set mismatch; missing={sorted(EXPECTED_ASSET_IDS - set(ids))}, "
            f"extra={sorted(set(ids) - EXPECTED_ASSET_IDS)}"
        )

    for line_number, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            value = row.get(column, "").strip()
            if not value:
                errors.append(f"row {line_number}: blank required field {column}")
            if require_no_unknown and UNKNOWN.search(value):
                errors.append(f"row {line_number}: UNKNOWN state in {column}")
            if WINDOWS_ABSOLUTE_PATH.search(value) or POSIX_USER_PATH.search(value):
                errors.append(f"row {line_number}: machine-specific path in {column}")

        release_class = row.get("release_class", "")
        if release_class not in ALLOWED_RELEASE_CLASSES:
            errors.append(f"row {line_number}: invalid release_class {release_class!r}")
        if row.get("user_download_required") not in {"YES", "NO"}:
            errors.append(f"row {line_number}: user_download_required must be YES or NO")
        if row.get("decision_status") != "LOCKED_R1":
            errors.append(f"row {line_number}: decision_status must be LOCKED_R1")
        if release_class == "DO_NOT_RELEASE":
            if row.get("github_decision", "").strip().lower() != "exclude":
                errors.append(f"row {line_number}: DO_NOT_RELEASE must be excluded from GitHub")
            if row.get("archival_decision", "").strip().lower() != "exclude":
                errors.append(f"row {line_number}: DO_NOT_RELEASE must be excluded from archives")
        if release_class == "USER_DOWNLOAD_REQUIRED" and row.get("user_download_required") != "YES":
            errors.append(f"row {line_number}: download-only asset must require user download")

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "source_manifest",
        "asset_class_count": len(rows),
        "no_unknown_required": require_no_unknown,
        "release_class_counts": {
            key: sum(row.get("release_class") == key for row in rows)
            for key in sorted(ALLOWED_RELEASE_CLASSES)
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-no-unknown", action="store_true")
    args = parser.parse_args()
    result = validate_manifest(args.manifest, args.require_no_unknown)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
