"""Export source data and standalone plotting software from the project files."""

from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import tomllib
import zipfile

from publication.build_review_package import build as build_data
from publication.build_submission_packages import _collect_payloads


ROOT = Path(__file__).resolve().parents[1]


def archive(path, files):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for name, payload in sorted(files.items()):
            if name.endswith((".zip", ".7z", ".gz", ".md", ".log", ".err")):
                raise ValueError("Unexpected archive member: " + name)
            stream.writestr(name, payload)
    with zipfile.ZipFile(path) as stream:
        assert stream.testzip() is None
        assert all(stream.read(name) == payload for name, payload in files.items())


def build(data_path, figures_path, release):
    buffer = io.BytesIO()
    build_data(buffer, release_id=release)
    with zipfile.ZipFile(buffer) as stream:
        data = {name: stream.read(name) for name in stream.namelist()
                if name.startswith(("Data_for_Figures/", "Supporting_tables/", "Supplementary_tables/"))
                or name in {"FIGURE_TABLE_INDEX.txt", "LICENSE-DOCS-DATA", "THIRD_PARTY_NOTICES.txt"}}
    registry = list(csv.DictReader((ROOT / "data/metadata/source_manifest.csv").open(encoding="utf-8-sig")))
    columns = ["asset_title", "upstream_reference", "version_or_hash", "licence_status"]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(registry)
    data["Input_sources.csv"] = text.getvalue().encode("utf-8")
    data["README_DATA.txt"] = f"""STUDY DATA — software release {release}

Extract this archive at the root of the matching repository:
https://github.com/yizheng-hub/thermal-heterogeneity-energy-service

Data_for_Figures/ contains the numerical and geographic inputs for the figures.
Supporting_tables/ contains the complete supporting numerical tables.
Supplementary_tables/ contains the ten displayed supplementary tables.
FIGURE_TABLE_INDEX.txt identifies the inputs for each figure and table.
CSV headers identify quantities and units; *_c denotes degrees Celsius, *_kw
kilowatts, *_kwh kilowatt-hours and *_pct percentages. Statistical definitions
and comparison denominators are given in the article and supplementary methods.

The original observation archives, weather and prototype are identified by the
repository's input-source tables and REPRODUCTION_FROM_RAW.txt. Obtain these
third-party inputs under their original terms for new simulations.
Author-generated data are CC BY 4.0; source materials retain their own terms.
""".encode("utf-8")
    manifest = {"software_release": release, "files": [
        {"file": name, "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in sorted(data.items())]}
    data["SOURCE_DATA_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Figure/table index entries must refer to files actually supplied.
    for line in data["FIGURE_TABLE_INDEX.txt"].decode().splitlines():
        if "Supplementary_tables/" in line:
            assert line.split(": ", 1)[1] in data, line
    archive(data_path, data)
    _, source = _collect_payloads(ROOT, release)
    names = [name for name in source if name.startswith("plot/") and name != "plot/reproduce.py"]
    names += ["publication/figure_data.py", "publication/notebook_figures.py",
              "publication/build_tables.py", "reproduce_publication.py", "LICENSE"]
    names += [name for name in source if name.startswith("src/thermal_heterogeneity/")
              and ("assets/" in name or name.endswith("figures.py")
                   or name == "src/thermal_heterogeneity/__init__.py")]
    figures = dict(data)
    figures.update({name: source[name] for name in names})
    for folder, pattern in [("main", "Figure_??.png"), ("extended", "Figure_E??.png"),
                            ("supplementary", "Figure_S??.png")]:
        for path in sorted((ROOT / "output/publication" / folder).glob(pattern)):
            figures[f"Figures/{folder}/{path.name}"] = path.read_bytes()
    assert sum(name.endswith(".png") for name in figures) == 16
    packages = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    figures["requirements.txt"] = "".join(f"{name}=={packages[name]}\n" for name in
        ["numpy", "pandas", "matplotlib", "scipy", "pillow"]).encode()
    figures["README.txt"] = f"""SUPPLEMENTARY SOFTWARE 1

High-resolution PNG figures, their source data and a standalone plotting notebook.
Software release: {release}.

Figures/ contains the six main, four Extended Data and six supplementary figures.
plot/Plot.ipynb has one setup cell followed by one cell per figure.
Data_for_Figures/ contains the plotting inputs; Supporting_tables/ and
Supplementary_tables/ contain the supporting numerical results and displayed tables.

Use Python 3.11, install requirements.txt and Times New Roman regular and bold.
From the extracted directory run:
  python reproduce_publication.py --output-root output/publication
Alternatively open plot/Plot.ipynb with the extracted directory as the working folder.
These figures and tables require neither EnergyPlus nor network access.

Complete analysis and simulation code:
https://github.com/yizheng-hub/thermal-heterogeneity-energy-service
""".encode()
    archive(figures_path, figures)
    return {"data_archive": str(data_path), "figure_archive": str(figures_path),
            "data_members": len(data), "figure_members": len(figures), "figures": 16}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--release", default="0.2.3")
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.figures, args.release)))
