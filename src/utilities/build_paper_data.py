"""Combine the three verified minimized input components into one data archive."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import zstandard


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-analysis-root", type=Path, required=True)
    parser.add_argument("--source-building-root", type=Path, required=True)
    parser.add_argument("--energy-service-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    args = parser.parse_args()
    components = {"legacy_analysis": args.legacy_analysis_root,
                  "source_building": args.source_building_root,
                  "energy_service": args.energy_service_root}
    files = []
    counts = {}
    for prefix, root in components.items():
        members = sorted(path for path in root.rglob("*") if path.is_file())
        counts[prefix] = len(members)
        files.extend((f"{prefix}/{path.relative_to(root).as_posix()}", path) for path in members)
    if counts != {"legacy_analysis": 46, "source_building": 425, "energy_service": 406}:
        raise ValueError(f"Input component file counts differ from the frozen release: {counts}")
    files.sort()
    if any(path.is_symlink() for _, path in files):
        raise ValueError("Symlinks are not admitted to the data archive")
    records = [{"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
               for name, path in files]
    args.metadata_root.mkdir(parents=True, exist_ok=True)
    manifest = args.metadata_root / "paper_bundle_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        with zstandard.ZstdCompressor(level=6, threads=0).stream_writer(target) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for (_, path), record in zip(files, records):
                    info = tarfile.TarInfo(record["path"])
                    info.size = record["bytes"]
                    info.mode = 0o644
                    info.mtime = info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    result = {"status": "ARCHIVE_BUILT", "archive": args.output.name,
              "sha256": sha256(args.output), "bytes": args.output.stat().st_size,
              "manifest_sha256": sha256(manifest), "files": len(records),
              "component_files": counts, "hourly_trajectories": 796,
              "analysis_inputs_minimized": True, "physical_input_assets_included": False,
              "raw_human_identifiers_included": False, "public_deposit_url": None}
    (args.metadata_root / "paper_bundle_source.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
