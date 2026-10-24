"""Check that every retained development script has one locked public disposition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ALLOWED = {
    "IMPLEMENTED_R3",
    "IMPLEMENTED_R3_VERIFIER",
    "IMPLEMENTED_R3_HELPER",
    "IMPLEMENTED_R3_LEVEL2_BOUNDARY",
    "PROVENANCE_RECORD_ONLY",
    "EXCLUDED_BY_LOCK",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--expect", type=int, required=True)
    args = parser.parse_args()
    with Path(args.inventory).open(encoding="utf-8-sig", newline="") as handle:
        inventory = list(csv.DictReader(handle))
    mapping = json.loads(Path(args.map).read_text(encoding="utf-8"))
    entries = mapping["entries"]
    if len(inventory) != args.expect or len(entries) != args.expect:
        raise SystemExit(f"expected {args.expect}; inventory={len(inventory)} map={len(entries)}")
    sources = [entry["source_script"] for entry in entries]
    if len(set(sources)) != len(sources):
        raise SystemExit("duplicate source script in migration map")
    inventory_sources = {row["script_path"] for row in inventory}
    if set(sources) != inventory_sources:
        raise SystemExit("inventory and migration map source sets differ")
    bad = [entry["source_script"] for entry in entries if entry["migration_status"] not in ALLOWED]
    if bad:
        raise SystemExit("unfinished migration entries: " + ", ".join(bad))


if __name__ == "__main__":
    main()
