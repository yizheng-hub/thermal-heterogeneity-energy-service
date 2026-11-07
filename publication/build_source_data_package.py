"""Export high-resolution figure files and their numerical source data."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    from publication.figure_data import verify

    records = verify()["files"]
    files, index, lookup = {}, [], {}
    metric_names = {
        "single_record/timed/FINAL_METRICS.csv": "TIMED_SINGLE_RECORD_METRICS.csv",
        "single_record/undated/FINAL_METRICS.csv": "UNTIMED_SINGLE_RECORD_METRICS.csv",
        "undated_acceptability/FINAL_METRICS.csv": "UNTIMED_REPEATED_RECORD_METRICS.csv",
    }
    for row in records:
        source = ROOT / row["result"]
        if source.name == "STATUS.json":
            continue
        name = next((v for k, v in metric_names.items() if row["result"].endswith(k)), source.name)
        group = "Hourly" if source.suffix == ".npz" else "Tables"
        if source.suffix == ".geojson" or name in {
            "GLOBAL_CLIMATES.json", "ne_110m_observed_countries.json", "Miami.geometry.json", "MODEL_MANIFEST.json"
        }:
            group = "Maps"
        target = f"Source_Data/{group}/{name}"
        payload = source.read_bytes()
        if source.name == "MODEL_MANIFEST.json":
            payload = (json.dumps({"Miami": {"floor_geometry": "Source_Data/Maps/Miami.geometry.json"}}, indent=2) + "\n").encode()
        elif source.name in {"ACTION_FULL_COMPLETED.csv", "NORMAL_OPERATION_REPLAY_COMPLETED.csv"}:
            text = io.StringIO(newline="")
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
            writer = csv.DictWriter(text, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for item in reader:
                item["hourly_path"] = "Source_Data/Hourly/" + Path(item["hourly_path"]).name
                writer.writerow(item)
            payload = text.getvalue().encode()
        assert target not in files
        files[target] = payload
        key = row["result"].removeprefix("Data_for_Figures/")
        key = key.replace("output/from_raw/", "baseline/").replace("output/strengthening/", "analysis/")
        key = re.sub(r"run_[^/]+/", "", key)
        index.append({"key": key, "file": target, "sha256": hashlib.sha256(payload).hexdigest()})
        lookup[row["result"]] = lookup[row["source"]] = target
    files["Source_Data/index.json"] = (json.dumps(index, indent=2) + "\n").encode()
    files["Source_Data/Maps/natural_earth_110m_land.provenance.json"] = (
        ROOT / "src/thermal_heterogeneity/assets/natural_earth_110m_land.provenance.json").read_bytes()
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["figure", "source_data_file"])
    for folder, prefix, label, count in [
        ("main", "", "Figure", 6), ("extended", "E", "Extended Data Figure", 4),
        ("supplementary", "S", "Fig. S", 6)
    ]:
        for number in range(1, count + 1):
            path = ROOT / f"output/publication/{folder}/Figure_{prefix}{number:02}"
            files[f"Figures/{label.replace(' ', '_').replace('.', '')}_{number}.png"] = path.with_suffix(".png").read_bytes()
            inputs = json.loads(path.with_suffix(".json").read_text())["input_sha256"]
            for item in sorted(inputs):
                writer.writerow([f"{label} {number}", lookup[item]])
    files["Figure_Data_Index.csv"] = text.getvalue().encode()
    files["README.txt"] = b"""SUPPLEMENTARY DATA 1

Figures/ contains 16 high-resolution PNG figures: six main figures, four
Extended Data figures and six supplementary figures.
Source_Data/Tables/ contains the numerical data underlying the figures and
reported statistical summaries. Figure_Data_Index.csv maps figures to files.
Source_Data/Maps/ contains geographic boundaries, locations and floor geometry.
Source_Data/Hourly/ contains trajectories used in the recovery illustration.

These are figure source data and numerical results. The original observation,
weather and model inputs for new simulations are supplied separately through
the data repository. No software or plotting notebook is included here.
Units and statistical definitions are given in column headers, figure captions
and the Methods. Author-generated data and figures are CC BY 4.0; third-party
source materials retain their original terms and attribution.
"""
    return files


def build(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    files = source_files()
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(archive.read(name) == payload for name, payload in files.items())
    return {"files": len(files), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(json.dumps(build(parser.parse_args().destination)))
