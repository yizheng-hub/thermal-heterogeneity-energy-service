"""Collect exact final plotting inputs from a completed scientific replay."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys


def refresh_plot_inputs(destination):
    """Refresh checksums after figure cells update their caption inputs."""
    destination = Path(destination)
    source_map, source_index = destination / "FIGURE_SOURCE_MAP.csv", destination / "SOURCE_DATA_INDEX.csv"
    if source_map.is_file() and source_index.is_file():
        with source_index.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        with source_map.open(encoding="utf-8-sig", newline="") as stream:
            mapped = csv.reader(stream)
            columns, count = len(next(mapped)), sum(1 for _ in mapped)
        for row in rows:
            if row["path"] == "FIGURE_SOURCE_MAP.csv":
                row.update(rows=count, columns=columns,
                           sha256=hashlib.sha256(source_map.read_bytes()).hexdigest())
        with source_index.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    inventory = []
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "PLOT_INPUTS.json":
            inventory.append({"path":path.relative_to(destination).as_posix(),
                              "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (destination / "PLOT_INPUTS.json").write_text(
        json.dumps({"files":inventory, "source":"completed scientific replay"}, indent=2)+"\n",
        encoding="utf-8")
    return len(inventory)


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
    extension = replay / "targeted_extension"
    if extension.is_dir():
        target = destination / "Targeted_extension"
        shutil.copytree(extension, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.log", "__pycache__"))
        with (destination / "SOURCE_DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, index = reader.fieldnames, list(reader)
        index = [row for row in index if not row["path"].startswith("Targeted_extension/")]
        for path in sorted(target.rglob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.reader(stream)
                columns = len(next(rows))
                count = sum(1 for _ in rows)
            index.append({"path":path.relative_to(destination).as_posix(), "rows":count,
                          "columns":columns, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        with (destination / "SOURCE_DATA_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(index, key=lambda row:row["path"]))
        map_path = destination / "FIGURE_SOURCE_MAP.csv"
        with map_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, mappings = reader.fieldnames, list(reader)
        mappings = [row for row in mappings if not row["relative_path"].startswith("Targeted_extension/")]
        for figure, panel, relative in [
            ("Fig. 3", "e", "service/DIRECT_COMMON_STRATA.csv"),
            ("Fig. 4", "d,e", "spatial/SPATIAL_PAIRS.csv"),
            ("Fig. 5", "b", "response/STAGE_DECISIONS.csv"),
            ("Supplementary Fig. S25", "a,b", "service/NEAR_SERVICE_CONDITION_SUMMARY.csv"),
            ("Supplementary Fig. S25", "a,b", "service/DIRECT_COMMON_STRATA.csv"),
            ("Supplementary Fig. S25", "c,d", "spatial/SPATIAL_PAIRS.csv"),
            ("Supplementary Fig. S25", "d", "spatial/SPATIAL_INTERACTIONS.csv"),
            ("Supplementary Fig. S25", "e,f", "response/STAGE_DECISIONS.csv"),
            ("Supplementary Fig. S25", "e,f", "response/ALL_CONTEXT_STAGE_SUMMARY.csv"),
            ("Supplementary Fig. S25", "g,h", "ranges/DEMAND_SENSITIVITY_SUMMARY.csv"),
        ]:
            path = target / relative
            mappings.append({"current_figure":figure, "panel":panel,
                             "source_key":"Targeted_extension/"+relative,
                             "source_kind":"Source Data", "relative_path":"Targeted_extension/"+relative,
                             "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                             "identifier_note":"Bounded extension from retained trajectories"})
        with map_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mappings)
    external = replay / "ctcd_external"
    if external.is_dir():
        target = destination / "CTCD_external"
        target.mkdir(exist_ok=True)
        for path in sorted(external.rglob("*.csv")):
            copied = target / path.relative_to(external)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copied)
        for path in sorted(external.glob("*.json")):
            shutil.copyfile(path, target / path.name)
        with (destination / "SOURCE_DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, index = reader.fieldnames, list(reader)
        index = [row for row in index if not row["path"].startswith("CTCD_external/")]
        for path in sorted(target.rglob("*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.reader(stream)
                columns = len(next(rows))
                count = sum(1 for _ in rows)
            index.append({"path":path.relative_to(destination).as_posix(), "rows":count,
                          "columns":columns, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        with (destination / "SOURCE_DATA_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(sorted(index, key=lambda row:row["path"]))
        map_path = destination / "FIGURE_SOURCE_MAP.csv"
        with map_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, mappings = reader.fieldnames, list(reader)
        mappings = [row for row in mappings if not row["relative_path"].startswith("CTCD_external/")]
        for figure, panel, relative in [
            ("Fig. 2", "a–f", "annual/ANNUAL_DIRECT.csv"),
            ("Fig. 3", "a,b,d", "annual/CONTROL_COMPARISONS.csv"),
            ("Supplementary Fig. S16", "b", "CTCD_REFERENCE_HISTOGRAM.csv"),
            ("Supplementary Fig. S16", "d", "CTCD_INTERVAL_HISTOGRAM.csv"),
        ]:
            path = target / relative
            mappings.append({"current_figure":figure, "panel":panel,
                             "source_key":"CTCD_external/"+relative,
                             "source_kind":"Source Data", "relative_path":"CTCD_external/"+relative,
                             "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                             "identifier_note":"Separate CTCD draws; random placement and native-zone control"})
        with map_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mappings)
    return refresh_plot_inputs(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parents[2], args.replay)
    print(f"Prepared {count} plotting input files")
