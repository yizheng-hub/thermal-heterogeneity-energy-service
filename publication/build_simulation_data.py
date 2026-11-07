"""Package the exact simulation inputs and numerical source results."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def input_files():
    from src.utilities.audit_reproduction_inputs import raw_inputs

    return sorted(set(raw_inputs(ROOT, include_validation=True)) | set(
        (ROOT / "data/upstream/global_climate_matrix/prototypes_22_1").glob("*.idf")))


def build_inputs(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    rows = []
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in input_files():
            assert path.is_file() and path.resolve().is_relative_to(ROOT)
            payload = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            rows.append({"file": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
            archive.writestr(name, payload)
        archive.writestr("INPUT_FILES.json", json.dumps(rows, indent=2))
        archive.writestr("README.txt", """SIMULATION INPUTS

Extract at the root of the matching scientific-code repository. Retain data/.
The archive supplies the original observation files, all nine weather files,
reference building model, nine source prototypes and the two numerical inputs
used by the from-raw workflow. INPUT_FILES.json identifies exact file versions.

Install the Python environment and EnergyPlus 22.1.0 as described in the code
repository, then run reproduce_from_raw.py. All scientific inputs are local;
the workflow generates new assignments, models, trajectories and result tables.
Previously generated simulation outputs are not required to start the workflow.

Third-party observation data, weather and prototype models retain their original
terms and attribution. The authors' software/data license does not relicense them.
Original sources are identified in data/metadata in the accompanying repository.
""")
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for row in rows:
            assert hashlib.sha256(archive.read(row["file"])).hexdigest() == row["sha256"]
    return {"archive": str(destination), "input_files": len(rows), "bytes": destination.stat().st_size}


def build_results(destination, figure_archive):
    from publication.build_review_package import public_csv

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(figure_archive) as source:
        files = {n: source.read(n) for n in source.namelist() if n.startswith("Data/")}
    hashes = {hashlib.sha256(b).hexdigest(): n for n, b in files.items()}
    rows = []
    for line in (ROOT / "publication/source_data_files.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        source = ROOT / line.strip()
        payload, _ = public_csv(source)
        digest = hashlib.sha256(payload).hexdigest()
        name = hashes.get(digest)
        if name is None:
            relative = source.relative_to(ROOT).as_posix()
            topic = relative.split("/results/", 1)[-1].rsplit("/", 1)[0]
            topic = "/".join(p for p in topic.split("/") if not p.startswith("run_"))
            name = f"Data/Analysis/{topic}/{source.name}"
            assert name not in files
            files[name] = payload
            hashes[digest] = name
        rows.append({"dataset": source.stem, "file": name, "sha256": digest})
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=["dataset", "file", "sha256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    files["RESULTS_INDEX.csv"] = text.getvalue().encode()
    files["README.txt"] = b"""NUMERICAL RESULTS

Data/Tables, Data/Maps and Data/Hourly contain the figure source data.
Data/Analysis contains additional numerical results supporting the article.
RESULTS_INDEX.csv locates the complete numerical datasets. Identical datasets
are stored once and may have more than one index entry. Units are identified
in CSV headers and statistical definitions are given in the article and SI.

These are computed outputs, not the inputs required to start new simulations.
For new simulations use Simulation_Inputs.zip with the matching code repository.
Author-generated numerical results are CC BY 4.0. Third-party geographic and
observation materials retain their original terms and attribution.
"""
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(archive.read(n) == b for n, b in files.items())
    return {"archive": str(destination), "numerical_datasets": len(rows), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_inputs(args.inputs)))
    print(json.dumps(build_results(args.results, args.figures)))
