"""Compare two Level-1 scientific payloads independently of run location."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(path: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in payload["scientific_files"]}
    return resolved.parent, entries


def compare(first: Path, second: Path) -> dict[str, object]:
    first_root, first_entries = load_manifest(first)
    second_root, second_entries = load_manifest(second)
    errors = []
    first_paths, second_paths = set(first_entries), set(second_entries)
    if first_paths != second_paths:
        errors.append(
            {
                "kind": "file_set",
                "only_first": sorted(first_paths - second_paths),
                "only_second": sorted(second_paths - first_paths),
            }
        )
    csv_checked = 0
    json_checked = 0
    for relative in sorted(first_paths & second_paths):
        first_path, second_path = first_root / relative, second_root / relative
        if relative.endswith(".json"):
            json_checked += 1
            first_hash = canonical_json_sha256(first_path)
            second_hash = canonical_json_sha256(second_path)
        else:
            csv_checked += 1
            first_hash = first_entries[relative]["sha256"]
            second_hash = second_entries[relative]["sha256"]
        if first_hash != second_hash:
            errors.append(
                {
                    "kind": "hash",
                    "path": relative,
                    "first": first_hash,
                    "second": second_hash,
                }
            )
    return {
        "status": "PASS" if not errors else "FAIL",
        "scientific_file_count": len(first_paths & second_paths),
        "scientific_csv_hashes_compared": csv_checked,
        "canonical_json_hashes_compared": json_checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_manifest", type=Path)
    parser.add_argument("second_manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.first_manifest, args.second_manifest)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
