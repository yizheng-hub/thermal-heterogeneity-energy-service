"""Verify contained files and raw reconstruction after copying the project."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

from audit_reproduction_inputs import raw_inputs

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-rebuild", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "SELF_CONTAINED_PROJECT_MANIFEST.json").read_text(encoding="utf-8")
    )
    required = {
        line.strip()
        for line in (ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("output/")
    }
    required.update(p.relative_to(ROOT).as_posix() for p in raw_inputs(ROOT, include_validation=True))
    missing = sorted(required - set(manifest["files"]))
    if missing:
        raise ValueError("The manifest omits required project sources or raw inputs: " + ", ".join(missing))

    def check(item):
        name, info = item
        path = ROOT / name
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"A packaged input resolves outside this project: {name}")
        if not path.is_file() or sha(path) != info["sha256"]:
            raise ValueError(f"A packaged input is missing or changed: {name}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(check, manifest["files"].items()))
    if not Path(sys.executable).resolve().is_relative_to(ROOT) or sys.flags.isolated != 1:
        raise RuntimeError("Use this project's contained isolated Python interpreter")
    report = {
        "status": "contained_files_verified",
        "checked_files": len(manifest["files"]),
        "local_isolated_python": True,
        "external_scientific_inputs": [],
    }
    if args.raw_rebuild:
        raw = args.raw_rebuild.resolve()
        if not raw.is_relative_to(ROOT):
            raise ValueError("The regenerated demand directory must be inside the project")
        expected = json.loads((ROOT / "data/metadata/raw_rebuild_expected_hashes.json").read_text())
        for name, expected_hash in expected["data_products"].items():
            if sha(raw / name) != expected_hash:
                raise ValueError(f"Moved-project raw reconstruction differs: {name}")
        audit = json.loads((raw / "PROFILE_REBUILD_AUDIT.json").read_text())
        if audit["generator_sha256"] != sha(ROOT / "src/utilities/rebuild_profile_assignments.py"):
            raise ValueError(
                "The reconstructed-data provenance does not match the contained generator"
            )
        report["raw_reconstruction"] = {
            "status": "byte_identical",
            "checked_data_products": len(expected["data_products"]),
            "profile_rows": audit["assignment_rows"],
        }
    output = ROOT / "audit/RELOCATION_VERIFICATION.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
