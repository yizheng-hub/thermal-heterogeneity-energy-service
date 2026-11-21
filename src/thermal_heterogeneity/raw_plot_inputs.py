"""Prepare the publication notebook inputs from a completed simulation workflow."""

import csv
import json
from pathlib import Path
import shutil

import pandas as pd

from .figure_inputs import ROOT, floor_geometry, minimize_human_inputs, sha


def prepare(raw, project):
    from analysis.global_extension.implementation_value import build_resource_scenarios

    layout = ROOT / "src/thermal_heterogeneity/assets/plot_input_layout.csv"
    with layout.open(encoding="utf-8", newline="") as stream:
        files = list(csv.DictReader(stream))
    hourly = {Path(row["file"]).name for row in files if row["file"].startswith("Hourly/")}
    index = []
    for row in files:
        key = row["input_key"]
        relative = Path("data/figure_data") / row["file"]
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if key.startswith("output/publication/support/"):
            settings = ROOT / "src/analysis/global_extension/GLOBAL_IMPLEMENTATION_VALUE_ASSUMPTIONS.json"
            build_resource_scenarios(json.loads(settings.read_text())).to_csv(target, index=False)
        else:
            root, name = key.split("/", 1)
            source = {"analysis": raw / "strengthening", "baseline": raw, "src": ROOT / "src"}[root] / name
            if source.name == "Miami.geometry.json":
                source = source.with_name("Miami.idf")
            if not source.is_file() and Path(str(source) + ".gz").is_file():
                source = Path(str(source) + ".gz")
            if not source.is_file():
                matches = sorted(source.parent.glob("run_*/" + source.name))
                if len(matches) != 1:
                    raise FileNotFoundError(f"Expected one completed simulation output for {key}")
                source = matches[0]
            if source.suffix == ".idf":
                target.write_text(json.dumps(floor_geometry(source), indent=2), encoding="utf-8")
            elif source.name == "MODEL_MANIFEST.json":
                model = json.loads(source.read_text())["Miami"]
                if sha(Path(model["prepared_idf"])) != model["prepared_idf_sha256"]:
                    raise ValueError("The illustrated building model does not match the declared geometry")
                geometry = {"Miami": {"floor_geometry": "data/figure_data/Maps/Miami.geometry.json"}}
                target.write_text(json.dumps(geometry), encoding="utf-8")
            elif row["columns"]:
                frame = pd.read_csv(source, low_memory=False)
                frame, _ = minimize_human_inputs(source.name, frame)
                if "hourly_path" in row["columns"].split(","):
                    basenames = frame.hourly_path.map(lambda path: Path(path).name)
                    frame = frame.loc[basenames.isin(hourly)].copy()
                    if frame.empty:
                        raise ValueError(f"Missing illustrated event trajectory in {source.name}")
                    frame["hourly_path"] = "data/figure_data/Hourly/" + basenames.loc[frame.index]
                if "event_json" in frame and source.name == "SINGLE_EVENT_ACTION_PLAN.csv":
                    def window_metadata(value):
                        window = json.loads(value)
                        window.pop("reference_hourly_path", None)
                        return json.dumps(window)
                    frame["event_json"] = frame.event_json.map(window_metadata)
                frame.loc[:, row["columns"].split(",")].to_csv(target, index=False)
            else:
                shutil.copyfile(source, target)
        index.append({"key": key, "file": relative.as_posix(), "sha256": sha(target)})
    (project / "data/figure_data/index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
