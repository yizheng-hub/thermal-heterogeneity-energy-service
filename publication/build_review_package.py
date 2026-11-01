"""Build the review-data attachment from a frozen list and verified figure inputs."""

from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import zipfile

import pandas as pd

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


def build(destination, revision):
    from publication.figure_data import verify

    if destination.exists():
        raise FileExistsError("Choose a new attachment filename; earlier packages are preserved")
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
        for record in figure_data["files"]:
            path = ROOT / record["result"]
            add(record["result"], path.read_bytes(), path, record["transformation"])
        path = ROOT / "Result/FIGURE_DATA_MANIFEST.json"
        add("Result/FIGURE_DATA_MANIFEST.json", path.read_bytes(), path)
        names = (ROOT / "publication/review_data_files.txt").read_text().splitlines()
        for name in names:
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT):
                raise ValueError("Supporting table must remain inside the project")
            frame = pd.read_csv(source, low_memory=False)
            if FORBIDDEN.intersection(frame.columns):
                raise ValueError("Individual identifiers remain in a supporting table")
            drop = [
                column for column in frame.columns if column.endswith("_path") or "_path_" in column
            ]
            for column in frame.select_dtypes(include="object"):
                if frame[column].dropna().astype(str).str.contains(MACHINE_PATH).any():
                    drop.append(column)
            drop = sorted(set(drop))
            public = frame.drop(columns=drop)
            buffer = io.StringIO()
            public.to_csv(buffer, index=False)
            add(
                "Analysis/" + name,
                buffer.getvalue().encode("utf-8"),
                source,
                "CSV serialization; removed local-path columns: " + ", ".join(drop),
            )
        for source in sorted(
            (ROOT / "output/publication_release_acceptance/tables").glob("Table_S*.csv")
        ):
            add("Tables/" + source.name, source.read_bytes(), source)
        if (
            len(list((ROOT / "output/publication_release_acceptance/tables").glob("Table_S*.csv")))
            != 9
        ):
            raise ValueError("Nine final SI tables are required")
        readme = """# Supplementary Data 1

Release 0.2.0 accompanies Supplementary Code 1 at the revision in RELEASE_MANIFEST.json.
Extract Result/ into the source-code root. Run reproduce_publication.py to regenerate
six main figures, sixteen supplementary figures and nine tables. See publication/README.md.
Analysis/ contains the frozen supporting aggregate and simulation-result tables.
Tables/ contains the nine displayed SI tables. SOURCE hashes identify retained originals;
attachment hashes identify the supplied copies. Unused local paths are removed.
Plotting data retain only the required anonymous quantities; raw human records, private
identity crosswalks and licensed runtimes are not redistributed here. Upstream licenses
and acquisition instructions remain in Supplementary Code 1.

These files reproduce publication results and figures; they do not replace the full
weather, model, source-data and runtime inputs for a new EnergyPlus simulation.
Original raw reconstruction and 810 logical annual operations were reproduced in fresh
outputs. Additional branches have separate execution receipts. A combined empty-directory
rerun of all additional physical branches is deferred to the next computing environment.
"""
        add("README.md", readme.encode())
        manifest = {
            "release": "0.2.0",
            "code_revision": revision,
            "files": rows,
            "all_added_physical_branches_fresh_rerun": False,
            "contents": "Minimized figure inputs and supporting analysis tables",
        }
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest, indent=2).encode())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Attachment CRC verification failed")
        for row in rows:
            if digest(archive.read(row["file"])) != row["sha256"]:
                raise ValueError("Attachment content verification failed")
    print(
        json.dumps(
            {
                "archive": str(destination),
                "files": len(rows) + 1,
                "bytes": destination.stat().st_size,
                "sha256": digest(destination.read_bytes()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch("[a-f0-9]{40}", args.code_revision):
        raise ValueError("Use the full verified code commit")
    if not args.destination.resolve().is_relative_to(ROOT):
        raise ValueError("Attachment destination must stay inside the project")
    build(args.destination.resolve(), args.code_revision)
