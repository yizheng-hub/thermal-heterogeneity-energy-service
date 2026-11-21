"""Verify and unpack the separate source-building robustness input archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import zstandard


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--root", type=Path, help="Verify an already extracted input directory.")
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--metadata-root", type=Path, help="Directory containing the supplied archive manifest and its checksum record")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    metadata = args.metadata_root or repository / "data/metadata"
    record = json.loads((metadata / "source_building_bundle_source.json").read_text())
    manifest = metadata / "source_building_bundle_manifest.csv"
    if sha256(manifest) != record["manifest_sha256"]:
        raise ValueError("Robustness input manifest checksum mismatch")
    if args.bundle and sha256(args.bundle) != record["sha256"]:
        raise ValueError("Robustness input archive checksum mismatch")
    with manifest.open(encoding="utf-8", newline="") as handle:
        expected = {row["path"]: row for row in csv.DictReader(handle)}
    if args.bundle and not args.dest:
        parser.error("--dest is required when unpacking --bundle")
    if args.root and args.dest:
        parser.error("--dest is only used when unpacking --bundle")
    destination = (args.root or args.dest).resolve()
    if args.root:
        actual = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        if actual != set(expected):
            raise ValueError("Extracted robustness input file set differs from its manifest")
        for name, row in expected.items():
            path = destination / name
            if (
                path.is_symlink()
                or path.stat().st_size != int(row["bytes"])
                or sha256(path) != row["sha256"]
            ):
                raise ValueError(f"Robustness input checksum mismatch: {name}")
        result = {
            "status": "PASS",
            "manifest_sha256": record["manifest_sha256"],
            "files_verified": len(actual),
            "hourly_trajectories": record["hourly_trajectories"],
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with args.bundle.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    name = member.name
                    logical = PurePosixPath(name)
                    if (
                        not member.isfile()
                        or name not in expected
                        or name in seen
                        or logical.is_absolute()
                        or ".." in logical.parts
                        or "\\" in name
                    ):
                        raise ValueError(f"Unexpected archive entry: {name}")
                    path = destination.joinpath(*logical.parts)
                    if not path.resolve().is_relative_to(destination):
                        raise ValueError(f"Archive entry leaves destination: {name}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    row = expected[name]
                    if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                        raise ValueError(f"Robustness input checksum mismatch: {name}")
                    seen.add(name)
    if seen != set(expected):
        raise ValueError("Robustness archive is missing declared input files")
    result = {
        "status": "PASS",
        "archive_sha256": record["sha256"],
        "manifest_sha256": record["manifest_sha256"],
        "files_verified": len(seen),
        "hourly_trajectories": record["hourly_trajectories"],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
