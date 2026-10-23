#!/usr/bin/env python3
"""Static release scan for the R0-R1 structure, licences and privacy boundary."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from check_release_layout import run_checks
from validate_source_manifest import validate_manifest


TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_BINARY_SUFFIXES = {".7z", ".docx", ".epw", ".eso", ".idf", ".pdf", ".sql", ".zip", ".zst"}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]")
POSIX_USER_PATH = re.compile(r"^/(?:Users|home)/[^/\s]+/")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_-]?key|password|access[_-]?token|client[_-]?secret)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]"
)
PROHIBITED_PAYLOAD_FIELDS = {
    "subject_id",
    "user_id",
    "analysis_unit_id",
    "source_id",
    "building_id",
    "participant_timestamp",
    "latitude",
    "longitude",
    "room_name",
    "record_id",
    "synthetic_person_id",
    "virtual_profile_id",
    "contributor",
    "age",
    "gender",
    "lat",
    "lon",
}
RELEASE_IDENTIFIER = re.compile(r"^[a-z]{2}_[a-z2-7]{24}$")
IGNORED_LOCAL_PARTS = {".cache", ".pytest_cache", ".pytest_tmp", ".ruff_cache", ".venv", "__pycache__"}


def _ignored_local_path(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if not relative.parts:
        return False
    if relative.parts[0] == "staging":
        return True
    if relative.parts[0] in {"outputs", "results", "intermediate_files"}:
        return relative.as_posix() != relative.parts[0] + "/README.md"
    return bool(
        set(relative.parts) & IGNORED_LOCAL_PARTS
        or any(part.endswith(".egg-info") for part in relative.parts)
    )


def _text_files(root: Path):
    for path in root.rglob("*"):
        if ".git" in path.parts or _ignored_local_path(path, root) or not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"LICENSE", "LICENSE-DOCS-DATA", ".gitignore", ".gitattributes", ".python-version"}:
            yield path


def _scan_common(root: Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or _ignored_local_path(path, root) or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
            errors.append(f"forbidden distributable file type: {relative}")
    for path in _text_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="strict")
        if WINDOWS_ABSOLUTE_PATH.search(text) or POSIX_USER_PATH.search(text):
            errors.append(f"machine-specific absolute path: {relative}")
        if PRIVATE_KEY.search(text) or SECRET_ASSIGNMENT.search(text):
            errors.append(f"possible credential or private key: {relative}")
    return errors


def _scan_payload_headers(root: Path) -> list[str]:
    errors: list[str] = []
    payload_roots = [root / "data" / "derived", root / "data" / "frozen", root / "outputs", root / "staging"]
    for payload_root in payload_roots:
        if not payload_root.exists():
            continue
        for path in payload_root.rglob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                fields = set(csv.DictReader(handle).fieldnames or [])
            prohibited = sorted(fields & PROHIBITED_PAYLOAD_FIELDS)
            if prohibited:
                errors.append(f"prohibited fields in {path.relative_to(root).as_posix()}: {prohibited}")
    return errors


def scan_paths_identifiers(root: Path) -> dict[str, object]:
    """Scan an already-materialized payload root without repository assumptions."""
    errors: list[str] = []
    csv_files = sorted(
        path
        for path in [*root.rglob("*.csv"), *root.rglob("*.csv.gz")]
        if not _ignored_local_path(path, root)
    )
    json_files = sorted(
        path for path in root.rglob("*.json") if not _ignored_local_path(path, root)
    )
    checked_rows = 0
    checked_release_id_columns = 0
    for path in csv_files:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            prohibited = sorted(set(fields) & PROHIBITED_PAYLOAD_FIELDS)
            aggregate = path.relative_to(root).as_posix() in {
                "publication/support/science_audit/demographic_city_season_gender_summary.csv",
                "publication/support/science_audit/demographic_city_season_age_summary.csv",
            }
            if aggregate:
                prohibited = [field for field in prohibited if field != "gender"]
            if prohibited:
                errors.append(f"prohibited fields in {path.relative_to(root).as_posix()}: {prohibited}")
            release_fields = [
                field for field in fields
                if field != "release_id"
                if field.endswith("id") and field.startswith(("release_", "anonymous_"))
            ]
            checked_release_id_columns += len(release_fields)
            for row in reader:
                checked_rows += 1
                if aggregate and (int(row.get("records", 0)) < 20 or int(row.get("analysis_units", 0)) < 10):
                    errors.append(f"demographic aggregate below disclosure threshold: {path.relative_to(root).as_posix()}")
                for field, value in row.items():
                    if not value:
                        continue
                    if WINDOWS_ABSOLUTE_PATH.search(value) or POSIX_USER_PATH.search(value):
                        errors.append(f"machine path in {path.relative_to(root).as_posix()}:{field}")
                        break
                    if field in release_fields and not RELEASE_IDENTIFIER.fullmatch(value):
                        errors.append(f"invalid release ID in {path.relative_to(root).as_posix()}:{field}")
                        break
    for path in json_files:
        text = path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(text) or POSIX_USER_PATH.search(text):
            errors.append(f"machine path in {path.relative_to(root).as_posix()}")
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON in {path.relative_to(root).as_posix()}: {exc}")
    crosswalks = [
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file()
        and not _ignored_local_path(path, root)
        and ("crosswalk" in path.name.lower() or "mapping" in path.name.lower())
    ]
    if crosswalks:
        errors.append(f"crosswalk-like files are forbidden: {crosswalks}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "paths_identifiers",
        "csv_files": len(csv_files),
        "json_files": len(json_files),
        "rows_scanned": checked_rows,
        "release_id_columns": checked_release_id_columns,
        "errors": sorted(set(errors)),
    }


def scan_licence_privacy(root: Path) -> dict[str, object]:
    errors = _scan_common(root)
    errors.extend(_scan_payload_headers(root))

    required_phrases = {
        "LICENSE": ["BSD 3-Clause License"],
        "LICENSE-DOCS-DATA": ["Creative Commons Attribution 4.0 International", "does not cover third-party"],
        "DATA_AND_SOFTWARE_LICENSES.md": ["BSD-3-Clause", "CC BY 4.0", "Third-party"],
        "THIRD_PARTY_NOTICES.md": ["ASHRAE", "Humans as a Sensor", "ENTH", "EnergyPlus", "DOE/PNNL", "New Delhi", "EPA", "CEA", "DERC"],
        "CITATION.cff": ["cff-version: 1.2.0", "license: BSD-3-Clause"],
    }
    for relative, phrases in required_phrases.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing licence/citation file: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase not in text:
                errors.append(f"{relative} missing required phrase: {phrase}")

    citation = (root / "CITATION.cff").read_text(encoding="utf-8") if (root / "CITATION.cff").is_file() else ""
    for prohibited_key in ("doi:", "repository-code:", "url:"):
        if re.search(rf"(?m)^\s*{re.escape(prohibited_key)}", citation):
            errors.append(f"CITATION.cff contains premature locator key: {prohibited_key}")

    manifest_result = validate_manifest(root / "data" / "source_manifest.csv", require_no_unknown=True)
    errors.extend(f"source manifest: {error}" for error in manifest_result["errors"])

    migration_path = root / "protocol" / "SCRIPT_MIGRATION_MAP.json"
    inventory_path = root / "protocol" / "RETAINED_SCRIPT_INVENTORY.csv"
    if migration_path.is_file() and inventory_path.is_file():
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        with inventory_path.open(encoding="utf-8-sig", newline="") as handle:
            inventory = list(csv.DictReader(handle))
        mapped = migration.get("entries", [])
        if len(inventory) != 43 or len(mapped) != 43:
            errors.append(f"43-script accounting failed: inventory={len(inventory)}, map={len(mapped)}")
        if {row["script_path"] for row in inventory} != {row.get("source_script") for row in mapped}:
            errors.append("script inventory and migration map do not contain the same paths")
    else:
        errors.append("script inventory or migration map missing")

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "licence_privacy",
        "asset_class_count": manifest_result.get("asset_class_count"),
        "script_inventory_count": 43 if not any("script" in e.lower() for e in errors) else None,
        "errors": errors,
    }


def scan_portability(root: Path) -> dict[str, object]:
    """Check the public analysis boundary for machine and runtime-dependent state."""

    errors = _scan_common(root)
    science_files = list((root / "src" / "thermal_heterogeneity").rglob("*.py"))
    science_files.extend([root / "configs" / "paper.yaml"])
    forbidden = {
        "sys.path": "dynamic import path mutation",
        "datetime.now": "runtime timestamp in scientific code",
        "datetime.utcnow": "runtime timestamp in scientific code",
        "Timestamp.now": "runtime timestamp in scientific code",
        "time.time": "runtime timestamp in scientific code",
    }
    for path in science_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for token, description in forbidden.items():
            if token in text:
                errors.append(f"{description}: {path.relative_to(root).as_posix()}:{token}")
    config_path = root / "configs" / "paper.yaml"
    if config_path.is_file():
        config_text = config_path.read_text(encoding="utf-8")
        if WINDOWS_ABSOLUTE_PATH.search(config_text) or POSIX_USER_PATH.search(config_text):
            errors.append("paper configuration contains an absolute path")
    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "portability",
        "science_files": len(science_files),
        "errors": sorted(set(errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--mode",
        required=True,
        choices=["structure", "licence-privacy", "paths-identifiers", "portability"],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "structure":
        result = run_checks(root)
    elif args.mode == "licence-privacy":
        result = scan_licence_privacy(root)
    elif args.mode == "paths-identifiers":
        result = scan_paths_identifiers(root)
    else:
        result = scan_portability(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
