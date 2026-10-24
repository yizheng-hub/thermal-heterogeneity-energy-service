"""Verify or unpack the separate normalized DR date-sensitivity sidecar."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--root", type=Path)
    parser.add_argument("--dest", type=Path)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    record = json.loads((repository / "data/typical_day_dr_bundle_source.json").read_text())
    manifest = repository / "data/typical_day_dr_bundle_manifest.csv"
    if digest(manifest) != record["manifest_sha256"]:
        raise ValueError("Repository manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        expected = {row["path"]: row for row in csv.DictReader(stream)}
    if args.bundle:
        if not args.dest:
            parser.error("--dest is required with --bundle")
        if digest(args.bundle) != record["sha256"]:
            raise ValueError("Sidecar archive checksum mismatch")
        root = args.dest.resolve()
        root.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(args.bundle) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) | {"MANIFEST.csv"}:
                raise ValueError("Archive contains duplicate, missing or undeclared entries")
            for name in names:
                logical = PurePosixPath(name)
                if logical.is_absolute() or ".." in logical.parts or "\\" in name:
                    raise ValueError("Unsafe archive member")
                target = root.joinpath(*logical.parts)
                if not target.resolve().is_relative_to(root):
                    raise ValueError("Archive member leaves destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
    else:
        if args.dest:
            parser.error("--dest is only valid with --bundle")
        root = args.root.resolve()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected) | {"MANIFEST.csv"} or digest(root / "MANIFEST.csv") != record["manifest_sha256"]:
        raise ValueError("Extracted sidecar file set or manifest changed")
    for relative, row in expected.items():
        path = root / relative
        if path.is_symlink() or path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
            raise ValueError(f"Sidecar checksum mismatch: {relative}")
    print(json.dumps({"status": "PASS", "verified_files": len(expected), "hourly_trajectories": 32,
                      "manifest_sha256": record["manifest_sha256"], "physical_simulations_run": 0}, indent=2))


if __name__ == "__main__":
    main()
