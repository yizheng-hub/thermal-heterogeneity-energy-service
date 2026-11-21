"""Collect exact final plotting inputs from a completed scientific replay."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys


JOINT_PANEL_SOURCES = (
    ("Fig. 1", "g", "DEMAND_COMPOSITION_PROFILES.csv"),
    *(("Fig. 2", panel, "annual/ANNUAL_DIRECT.csv") for panel in "abcd"),
    ("Fig. 2", "e", "annual/CONTROL_COMPARISONS.csv"),
    *(("Fig. 3", panel, "annual/CONTROL_COMPARISONS.csv") for panel in "abd"),
    ("Fig. 3", "c", "SIGNED_ENDUSE_REALLOCATION.csv"),
    ("Fig. 3", "e", "service/DIRECT_COMMON_PAIR_AUDIT.csv"),
    ("Fig. 3", "f", "annual/ANNUAL_SERVICE_SELECTIONS.csv"),
    ("Fig. 3", "g", "seasonal/SEASONAL_SERVICE_BUDGET_DIAGNOSTIC.csv"),
    ("Fig. 3", "h", "seasonal/TEMPORAL_COMPENSATION_PAIRS.csv"),
    ("Fig. 3", "h", "seasonal/PERIOD_TOTALS.csv"),
    *(("Fig. 4", panel, "joint_spatial/SPATIAL_PAIRS.csv") for panel in "abc"),
    *(("Fig. 4", panel, "joint_spatial/COMMON_LAYOUT_PAIRS.csv") for panel in "de"),
    ("Fig. 4", "f", "joint_spatial/STATIC_COVERAGE_BOUND.csv"),
    ("Fig. 4", "f", "joint_spatial/STATIC_AND_ANNUAL_COVERAGE.csv"),
    ("Fig. 5", "a", "joint_portfolio/portfolio_hourly.csv"),
    ("Fig. 5", "b", "joint_events/JOINT_EVENT_CASE_SERVICE_ENERGY.csv"),
    *(("Fig. 5", panel, "joint_events/JOINT_EVENT_PORTFOLIO_RESPONSE.csv") for panel in "cde"),
    ("Fig. 5", "f", "joint_events/JOINT_EVENT_STAGE_COUNTS_BY_CONTROL.csv"),
    *(("Fig. 6", panel, "costs/all_signed_paired_costs.csv") for panel in "abf"),
    ("Fig. 6", "c", "costs/implementation_cost_scenarios.csv"),
    ("Fig. 6", "c", "costs/command_scope_management_boundaries.csv"),
    ("Fig. 6", "d", "costs/bill_implementation_boundaries.csv"),
    ("Fig. 6", "e", "costs/command_scope_management_boundaries.csv"),
    ("Supplementary Fig. S18", "figure inputs", "seasonal/PERIOD_TOTALS.csv"),
    ("Supplementary Fig. S19", "figure inputs", "seasonal/PERIOD_TOTALS.csv"),
    *(("Supplementary Fig. S25", panel, source) for panel in "ab" for source in (
        "service/DIRECT_COMMON_STRATA.csv", "service/NEAR_SERVICE_CONDITION_SUMMARY.csv")),
    *(("Supplementary Fig. S25", panel, source) for panel in "cd" for source in (
        "joint_spatial_diagnostics/SPATIAL_PAIRS.csv", "joint_spatial_diagnostics/SPATIAL_INTERACTIONS.csv")),
    *(("Supplementary Fig. S25", panel, source) for panel in "ef" for source in (
        "joint_events/JOINT_EVENT_STAGE_COUNTS.csv", "joint_events/JOINT_EVENT_STAGE_DECISIONS.csv")),
)

INFORMATION_PANEL_SOURCE = (
    "Fig. 2",
    "f",
    "decision_value/INFORMATION_HIERARCHY_SUMMARY.csv",
)


def refresh_joint_source_map(destination):
    mapping = Path(destination) / "FIGURE_SOURCE_MAP.csv"
    if not mapping.is_file():
        return
    with mapping.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    replaced = {f"Fig. {number}" for number in range(2, 7)}
    rows = [row for row in rows if row["current_figure"] not in replaced
            and row["current_figure"] not in {"Supplementary Fig. S18", "Supplementary Fig. S19"}
            and not (row["current_figure"] == "Fig. 1" and row["panel"] == "g")
            and not (row["current_figure"] == "Supplementary Fig. S25"
                     and row["panel"] in {"a", "b", "c", "d", "e", "f", "a,b", "c,d", "e,f"})]
    for figure, panel, source in JOINT_PANEL_SOURCES:
        relative = "Joint_main/" + source
        path = Path(destination) / relative
        if not path.is_file():
            raise FileNotFoundError(f"A final panel source is missing: {relative}")
        rows.append({"current_figure": figure, "panel": panel, "source_key": relative,
            "source_kind": "Source Data", "relative_path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "identifier_note": "Six fixed compositions; source labels retain provenance"})
    figure, panel, relative = INFORMATION_PANEL_SOURCE
    path = Path(destination) / relative
    if not path.is_file():
        raise FileNotFoundError(f"The Figure 2 information-sufficiency source is missing: {relative}")
    rows.append({
        "current_figure": figure,
        "panel": panel,
        "source_key": relative,
        "source_kind": "Source Data",
        "relative_path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "identifier_note": (
            "Fixed-plan-library review; 606 resolved decisions form the panel denominator, "
            "with 402 unresolved cells reported separately"
        ),
    })
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_joint_sources(parallel, destination):
    """Copy final joint numerical evidence without changing its source tables."""
    parallel, destination = Path(parallel), Path(destination)
    if not (parallel / "joint_main/JOINT_EVIDENCE_QA.json").is_file():
        return
    qa = json.loads((parallel / "joint_main/JOINT_EVIDENCE_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS":
        raise ValueError("Joint evidence has not passed its final checks")
    target = destination / "Joint_main"
    copies = [(parallel / "joint_main", target)] + [
        (parallel / name, target / name) for name in (
            "joint_spatial", "joint_spatial_diagnostics", "joint_events", "joint_portfolio")]
    for source, copied in copies:
        if not source.is_dir():
            raise FileNotFoundError(f"A final joint evidence component is missing: {source}")
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix not in (".csv", ".gz", ".json"):
                continue
            item = copied / path.relative_to(source)
            item.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, item)
    with (destination / "SOURCE_DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, index = reader.fieldnames, list(reader)
    index = [row for row in index if not row["path"].startswith("Joint_main/")]
    for path in sorted(target.rglob("*.csv*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            rows = csv.reader(stream)
            columns = len(next(rows))
            count = sum(1 for _ in rows)
        index.append({"path": path.relative_to(destination).as_posix(), "rows": count, "columns": columns,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (destination / "SOURCE_DATA_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(index, key=lambda row: row["path"]))
    refresh_joint_source_map(destination)


def prepare_information_hierarchy(replay, destination):
    """Copy the derived information hierarchy and index its panel source."""
    replay, destination = Path(replay), Path(destination)
    source = replay / "information_hierarchy"
    required = (
        source / "INFORMATION_HIERARCHY_SUMMARY.csv",
        source / "INFORMATION_HIERARCHY_SUMMARY.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Information-hierarchy replay outputs are missing: "
            + ", ".join(str(path) for path in missing)
        )
    target = destination / "decision_value"
    target.mkdir(parents=True, exist_ok=True)
    for path in required:
        shutil.copyfile(path, target / path.name)

    index_path = destination / "SOURCE_DATA_INDEX.csv"
    with index_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, index = reader.fieldnames, list(reader)
    relative = "decision_value/INFORMATION_HIERARCHY_SUMMARY.csv"
    index = [row for row in index if row["path"] != relative]
    copied = destination / relative
    with copied.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        columns = len(next(rows))
        count = sum(1 for _ in rows)
    index.append({
        "path": relative,
        "rows": count,
        "columns": columns,
        "sha256": hashlib.sha256(copied.read_bytes()).hexdigest(),
    })
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(index, key=lambda row: row["path"]))


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
        root / "results/analysis_results/method_comparisons/thermal_demand": destination / "support/thermal_demand",
        root / "results/analysis_results/method_comparisons/spatial_placement": destination / "support/spatial_placement",
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
    prepare_information_hierarchy(replay, destination)
    prepare_joint_sources(replay / "ctcd_parallel", destination)
    return refresh_plot_inputs(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    args = parser.parse_args()
    count = prepare(Path(__file__).resolve().parents[2], args.replay)
    print(f"Prepared {count} plotting input files")
