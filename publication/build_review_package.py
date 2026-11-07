"""Build source data from selected scientific tables and figure inputs."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "person",
    "subject_id",
    "record_id",
    "profile_id",
    "analysis_unit_id",
    "participant_id",
    "user_id",
    "gender",
    "sex",
    "age",
    "timestamp",
}
MACHINE_PATH = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]|/(?:Users|home)/[^/]+/")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def public_csv(source, preserve_columns=()):
    """Remove local provenance columns without reparsing numerical values."""
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        if FORBIDDEN.intersection(columns):
            raise ValueError("Individual identifiers remain in a supporting table")
        drop = {
            index
            for index, column in enumerate(columns)
            if (column.endswith("_path") or "_path_" in column) and column not in preserve_columns
        }
        for row in reader:
            if len(row) != len(columns):
                raise ValueError("Supporting table has inconsistent CSV columns")
            drop.update(index for index, value in enumerate(row) if MACHINE_PATH.search(value))
    keep = [index for index in range(len(columns)) if index not in drop]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            writer.writerow([row[index] for index in keep])
    return buffer.getvalue().encode("utf-8"), [columns[index] for index in sorted(drop)]


def portable_result(source):
    """Minimize unused local metadata in attachment copies of retained inputs."""
    payload = source.read_bytes()
    if source.name in {
        "ACTION_FULL_COMPLETED.csv",
        "NORMAL_OPERATION_REPLAY_COMPLETED.csv",
    }:
        public, removed = public_csv(source, preserve_columns={"hourly_path"})
        if removed:
            return public, "; removed unused local metadata columns: " + ", ".join(removed)
    elif source.name == "SINGLE_EVENT_ACTION_PLAN.csv":
        buffer = io.StringIO(newline="")
        changed = False
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                event = json.loads(row["event_json"])
                if MACHINE_PATH.search(str(event.get("reference_hourly_path", ""))):
                    del event["reference_hourly_path"]
                    row["event_json"] = json.dumps(event, separators=(",", ":"))
                    changed = True
                writer.writerow(row)
        if changed:
            return (
                buffer.getvalue().encode("utf-8"),
                "; removed unused event_json.reference_hourly_path; event timing and actions retained",
            )
    elif source.name == "STATUS.json":
        metadata = json.loads(payload)
        if MACHINE_PATH.search(str(metadata.get("output_directory", ""))):
            del metadata["output_directory"]
            return (
                (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
                "; removed unused local output_directory metadata",
            )
    return payload, ""


def figure_table_index(publication, figure_files):
    available = {row["result"] for row in figure_files}
    source_paths = {row.get("source", row["result"]): row["result"] for row in figure_files}
    lines = ["FIGURE AND TABLE DATA FILES", ""]
    for folder, pattern, label, expected in (
        ("main", "Figure_*.json", "Figure", 6),
        ("extended", "Figure_E*.json", "Extended Data Figure", 4),
        ("supplementary", "Figure_S*.json", "Supplementary Figure", 6),
    ):
        receipts = sorted((publication / folder).glob(pattern))
        if len(receipts) != expected:
            raise ValueError(f"Expected {expected} complete {folder} figures for the file index")
        for number, receipt in enumerate(receipts, 1):
            supplied = []
            for name in json.loads(receipt.read_text(encoding="utf-8"))["input_sha256"]:
                target = name if name in available else source_paths.get(name, "Data_for_Figures/" + name)
                if target not in available:
                    raise ValueError("Figure input is missing from source data: " + name)
                supplied.append(target)
            lines.extend([f"{label} {number}", *["  " + name for name in sorted(set(supplied))], ""])
    lines.append("Supplementary Tables 1–10")
    lines.extend(f"  Table {number}: Supplementary_tables/Table_S{number:02}.csv" for number in range(1, 11))
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(
    destination,
    revision=None,
    *,
    release_id="0.2.2",
    source_snapshot=None,
    supporting_files=None,
):
    from publication.figure_data import verify
    from publication.build_submission_packages import collect_software_snapshot
    from publication.package_text import documentation_payloads

    in_memory = isinstance(destination, io.BytesIO)
    if not in_memory:
        destination = Path(destination).resolve()
    if not in_memory and destination.exists():
        raise FileExistsError("Choose a new attachment filename; earlier packages are preserved")
    snapshot = collect_software_snapshot(ROOT, release_id)
    if source_snapshot is not None and source_snapshot != snapshot:
        raise ValueError("Working-tree source differs from the matching software snapshot")
    if revision is not None and not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("Use a full base code commit")
    if revision is not None and revision != snapshot.get("base_git_commit"):
        raise ValueError("The supplied base commit differs from the source snapshot")
    if not in_memory:
        destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:

        def add(name, payload, source=None, transformation="byte-identical"):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Archive path must be relative")
            entry = zipfile.ZipInfo(name, (2026, 11, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
            row = {
                "file": name,
                "bytes": len(payload),
                "sha256": digest(payload),
                "transformation": transformation,
            }
            if source is not None:
                row.update(
                    source=source.relative_to(ROOT).as_posix(),
                    source_sha256=digest(source.read_bytes()),
                )
            rows.append(row)

        figure_data = verify()
        packaged_figure_files = []
        for record in figure_data["files"]:
            path = ROOT / record["result"]
            payload, transformation = portable_result(path)
            record = dict(record)
            record.update(
                sha256=digest(payload),
                bytes=len(payload),
                transformation=record["transformation"] + transformation,
            )
            packaged_figure_files.append(record)
            add(record["result"], payload, path, record["transformation"])
        path = ROOT / "Data_for_Figures/FIGURE_DATA_MANIFEST.json"
        packaged_figure_data = dict(figure_data, files=packaged_figure_files)
        add(
            "Data_for_Figures/FIGURE_DATA_MANIFEST.json",
            (json.dumps(packaged_figure_data, indent=2) + "\n").encode("utf-8"),
            path,
            "Package checksums updated for unused local metadata removal",
        )
        add(
            "FIGURE_TABLE_INDEX.txt",
            figure_table_index(ROOT / "output/publication", figure_data["files"]),
        )
        if supporting_files is None:
            inventory = ROOT / "publication/source_data_files.txt"
            if not inventory.is_file():
                inventory = ROOT / "publication/review_data_files.txt"
            names = inventory.read_text().splitlines()
        else:
            names = list(supporting_files)
        names = [name.strip() for name in names if name.strip() and not name.lstrip().startswith("#")]
        if len(names) != len(set(names)):
            raise ValueError("Supporting-table inventory contains duplicate paths")
        for name in names:
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT):
                raise ValueError("Supporting table must remain inside the project")
            payload, drop = public_csv(source)
            add(
                "Supporting_tables/" + name.removeprefix("output/"),
                payload,
                source,
                "CSV text fields preserved; removed local-path columns: " + ", ".join(drop),
            )
        for source in sorted(
            (ROOT / "output/publication/tables").glob("Table_S*.csv")
        ):
            add("Supplementary_tables/" + source.name, source.read_bytes(), source)
        if (
            len(list((ROOT / "output/publication/tables").glob("Table_S*.csv")))
            != 10
        ):
            raise ValueError("Ten final SI tables are required")
        source = ROOT / "LICENSE-DOCS-DATA"
        add(source.name, source.read_bytes(), source)
        texts = documentation_payloads(ROOT, release_id, names={
            "DATA_AND_SOFTWARE_LICENSES.txt", "THIRD_PARTY_NOTICES.txt"
        })
        for name in ("DATA_AND_SOFTWARE_LICENSES.txt", "THIRD_PARTY_NOTICES.txt"):
            source = ROOT / name.replace(".txt", ".md")
            if not source.is_file():
                source = ROOT / name
            add(name, texts[name], source, "Plain-text rights and attribution notice")
        source = ROOT / "data/metadata/source_manifest.csv"
        add("Input_sources/source_manifest.csv", source.read_bytes(), source)
        readme = f"""SOURCE DATA — release {release_id}

Extract the data at the root of the matching code repository.
The repository instructions describe figure, table and simulation commands.

Data_for_Figures/       Inputs used by the figure and table scripts.
Supporting_tables/      Numerical results supporting the article and supplement.
Supplementary_tables/   The ten displayed supplementary tables.
Input_sources/          Upstream source locations and license information.

SOURCE_DATA_MANIFEST.json lists the supplied files and their SHA-256 checksums.
FIGURE_TABLE_INDEX.txt maps each figure and table to its data files.
DATA_AND_SOFTWARE_LICENSES.txt and THIRD_PARTY_NOTICES.txt explain the rights.

Figure and table reproduction uses Data_for_Figures without EnergyPlus.
Full scientific simulation reruns require the separate upstream inputs, runtime
listed in the repository instructions.
"""
        add("README_SOURCE_DATA.txt", readme.encode("utf-8"))
        manifest = {
            "release": release_id,
            "release_id": release_id,
            "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
            "base_git_commit": snapshot.get("base_git_commit"),
            "files": rows,
            "contents": "Figure inputs, supplementary tables and supporting numerical results",
        }
        archive.writestr("SOURCE_DATA_MANIFEST.json", json.dumps(manifest, indent=2).encode())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Attachment CRC verification failed")
        for row in rows:
            if digest(archive.read(row["file"])) != row["sha256"]:
                raise ValueError("Attachment content verification failed")
    if not in_memory:
        print(
            json.dumps(
                {
                    "archive": str(destination),
                    "files": len(rows) + 1,
                    "bytes": destination.stat().st_size,
                    "sha256": digest(destination.read_bytes()),
                    "release_id": release_id,
                    "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
                }
            )
        )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--code-revision", help="Optional base commit for legacy callers")
    parser.add_argument("--release-id", default="0.2.2")
    parser.add_argument("--support-files", type=Path, help="Approved supporting-table inventory")
    parser.add_argument("--source-snapshot-sha256", help="Require the matching software snapshot")
    args = parser.parse_args()
    from publication.build_submission_packages import collect_software_snapshot

    snapshot = collect_software_snapshot(ROOT, args.release_id)
    if (
        args.source_snapshot_sha256 is not None
        and snapshot["source_snapshot_sha256"] != args.source_snapshot_sha256
    ):
        raise ValueError("Working-tree source differs from the matching software snapshot")
    build(
        args.destination.resolve(),
        args.code_revision,
        release_id=args.release_id,
        source_snapshot=snapshot,
        supporting_files=(
            args.support_files.read_text().splitlines() if args.support_files is not None else None
        ),
    )
