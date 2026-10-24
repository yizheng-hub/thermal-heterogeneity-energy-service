"""Collect exact final plotting inputs from a completed scientific replay."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys


def prepare(root, replay):
    root, replay = Path(root).resolve(), Path(replay).resolve()
    destination = root / "results"
    destination.mkdir(exist_ok=True)
    shutil.copytree(replay / "Supplementary_Data", destination, dirs_exist_ok=True)
    upstream = root / "data/upstream/ashrae_db_ii"
    measurements, metadata = upstream / "db_measurements_v2.1.0.csv.gz", upstream / "db_metadata.csv"
    if measurements.is_file() and metadata.is_file():
        identity = replay.parent / "building_identity"
        subprocess.run([sys.executable, str(root / "src/utilities/audit_building_identity.py"),
                        "--measurements", str(measurements), "--metadata", str(metadata),
                        "--public-profiles", str(root / "data/inputs/legacy_analysis/human/human_profiles_minimized.csv.gz"),
                        "--source-inventory", str(root / "data/inputs/source_building/SOURCE_BUILDING_INVENTORY.csv"),
                        "--output-root", str(identity)], check=True)
        target = destination / "Building_identity"
        target.mkdir(exist_ok=True)
        with (destination / "SOURCE_DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, index = reader.fieldnames, list(reader)
        index = [row for row in index if not row["path"].startswith("Building_identity/")]
        for path in sorted(identity.glob("*.csv")):
            copied = target / path.name
            shutil.copyfile(path, copied)
            with copied.open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.reader(stream)
                columns = len(next(rows))
                count = sum(1 for _ in rows)
            index.append({"path":"Building_identity/"+path.name, "rows":count, "columns":columns,
                          "sha256":hashlib.sha256(copied.read_bytes()).hexdigest()})
        with (destination / "SOURCE_DATA_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(index, key=lambda row:row["path"]))
    copies = {
        root / "data/inputs/legacy_analysis": destination / "legacy_analysis",
        replay / "supporting_evidence/panel_data": destination / "panel_data",
        replay / "supporting_evidence/decision_value": destination / "decision_value",
        root / "src/analysis/support/science_audit": destination / "support/science_audit",
        root / "src/analysis/support/spatial_audit": destination / "support/spatial_audit",
    }
    for source, target in copies.items():
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    engineering = destination / "energy_service/engineering"
    engineering.mkdir(parents=True, exist_ok=True)
    for name in ("PROFILE_ASSIGNMENTS.csv.gz", "CONTROL_REFERENCE.json", "ZONE_EXECUTION_GROUPS.csv"):
        shutil.copyfile(root / "data/inputs/energy_service/engineering" / name, engineering / name)
    captions = destination / "captions"
    captions.mkdir(exist_ok=True)
    for n in range(1, 7):
        source = replay / f"paper/captions_alt/figure_{n:02d}/caption.txt"
        shutil.copyfile(source, captions / f"Figure_{n:02d}.txt")
    for n in range(1, 25):
        source = replay / f"paper/supplementary_figures/Figure_S{n:02d}_caption.txt"
        shutil.copyfile(source, captions / f"Figure_S{n:02d}.txt")
    inventory = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "PLOT_INPUTS.json":
            inventory.append({"path":path.relative_to(destination).as_posix(),
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / "PLOT_INPUTS.json").write_text(
        json.dumps({"files":inventory, "source":"completed scientific replay"}, indent=2)+"\n",
        encoding="utf-8")
    return len(inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parents[2], args.replay)
    print(f"Prepared {count} plotting input files")
