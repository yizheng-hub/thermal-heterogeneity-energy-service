"""Export readable figure source data and a standalone notebook."""

from pathlib import Path
import argparse
import json
import re
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    from publication.figure_data import verify

    data = verify()["files"]
    files, index = {}, []
    metric_names = {
        "single_record/timed/FINAL_METRICS.csv": "TIMED_SINGLE_RECORD_METRICS.csv",
        "single_record/undated/FINAL_METRICS.csv": "UNTIMED_SINGLE_RECORD_METRICS.csv",
        "undated_acceptability/FINAL_METRICS.csv": "UNTIMED_REPEATED_RECORD_METRICS.csv",
    }
    for row in data:
        source = ROOT / row["result"]
        if source.name == "STATUS.json":
            continue
        name = source.name
        for ending, replacement in metric_names.items():
            if row["result"].endswith(ending):
                name = replacement
        group = "Hourly" if source.suffix == ".npz" else "Tables"
        if source.suffix == ".geojson" or name in {
            "GLOBAL_CLIMATES.json", "ne_110m_observed_countries.json", "Miami.geometry.json", "MODEL_MANIFEST.json"
        }:
            group = "Maps"
        target = f"Data/{group}/{name}"
        assert target not in files
        files[target] = source.read_bytes()
        key = row["result"].removeprefix("Data_for_Figures/")
        key = key.replace("output/from_raw/", "baseline/").replace("output/strengthening/", "analysis/")
        key = re.sub(r"run_[^/]+/", "", key)
        index.append({"key": key, "file": target, "sha256": row["sha256"]})
    files["Data/index.json"] = (json.dumps(index, indent=2) + "\n").encode()
    for name in ["reference_figures.py", "supplementary_figures.py", "strengthened_figures.py", "extended_figures.py"]:
        text = (ROOT / "plot" / name).read_text(encoding="utf-8")
        text = text.replace("from thermal_heterogeneity.figures import", "from figure_support import")
        files["Plotting/" + name] = text.encode("utf-8")
    text = (ROOT / "src/thermal_heterogeneity/figures.py").read_text(encoding="utf-8")
    text = text.replace('return Path(str(files(_ASSET_PACKAGE).joinpath(name)))',
                        'return Path(__file__).resolve().parents[1] / "Data/Maps" / name')
    files["Plotting/figure_support.py"] = text.encode()
    files["Data/Maps/natural_earth_110m_land.provenance.json"] = (
        ROOT / "src/thermal_heterogeneity/assets/natural_earth_110m_land.provenance.json").read_bytes()
    files["Plotting/draw.py"] = (ROOT / "publication/standalone_plotting.py").read_bytes()
    notebook = json.loads((ROOT / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    setup = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    display_function = "def show_figure" + "".join(setup["source"]).split("def show_figure", 1)[1]
    text = ('from pathlib import Path\nimport sys\nROOT = Path.cwd().resolve()\n'
            'sys.path.insert(0, str(ROOT))\nfrom Plotting.draw import draw, verify\n'
            'verify()\nOUTPUT = globals().get("PUBLICATION_OUTPUT", ROOT / "Redrawn_Figures")\n'
            + display_function)
    setup["source"] = text.splitlines(keepends=True)
    files["Plot.ipynb"] = (json.dumps(notebook, indent=1) + "\n").encode()
    for folder, prefix, label, count in [
        ("main", "", "Figure_", 6), ("extended", "E", "Extended_Data_Figure_", 4),
        ("supplementary", "S", "Figure_S", 6)
    ]:
        for n in range(1, count + 1):
            files[f"Figures/{label}{n}.png"] = (ROOT / f"output/publication/{folder}/Figure_{prefix}{n:02}.png").read_bytes()
    packages = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    files["requirements.txt"] = "".join(f"{n}=={packages[n]}\n" for n in ["numpy", "pandas", "matplotlib", "scipy", "pillow"]).encode()
    files["README.txt"] = b"""SUPPLEMENTARY SOFTWARE 1

Figures/       Six main, four Extended Data and six supplementary PNG figures.
Data/Tables/   Numerical inputs used to draw the figures.
Data/Maps/     Geographic boundaries, climate locations and floor geometry.
Data/Hourly/   Hourly trajectories used in the recovery illustration.
Plot.ipynb     One setup cell, then one cell per figure.
Plotting/      Shared drawing routines imported by the notebook.

Use Python 3.11 and install requirements.txt. Times New Roman regular and bold
must be installed. Open Plot.ipynb with this directory as the working folder.
All figure data are local; EnergyPlus and network access are not required.
Data/index.json maps numerical inputs to the drawing routines.
The complete scientific code is supplied separately through the linked repository.
"""
    files["LICENSE.txt"] = (ROOT / "LICENSE").read_bytes()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert all(archive.read(n) == b for n, b in files.items())
    return {"files": len(files), "figure_data": len(index), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(json.dumps(build(parser.parse_args().destination)))
