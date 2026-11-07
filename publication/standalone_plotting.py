"""Draw the article figures from the accompanying source-data directory."""

from pathlib import Path
import hashlib
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ("Plotting" if (ROOT / "Plotting").is_dir() else "plot")))
import reference_figures as main  # noqa: E402
import supplementary_figures as supplementary  # noqa: E402
import extended_figures as extended_figures  # noqa: E402

INDEX = json.loads((ROOT / "Data/index.json").read_text(encoding="utf-8"))


def key(name):
    value = str(name).replace("\\", "/")
    value = value.split("Data_for_Figures/")[-1]
    value = value.replace("output/from_raw/", "baseline/")
    value = value.replace("output/strengthening/", "analysis/")
    return re.sub(r"run_[^/]+/", "", value)


def resolve(name):
    requested = key(name)
    matches = [row for row in INDEX if row["key"] == requested]
    if not matches:
        matches = [row for row in INDEX if Path(row["key"]).name == Path(requested).name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source file for {requested}; found {len(matches)}")
    return ROOT / matches[0]["file"]


def verify():
    for row in INDEX:
        path = ROOT / row["file"]
        if not path.resolve().is_relative_to(ROOT / "Data"):
            raise ValueError("Data path leaves the source-data directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source data differ: " + row["file"])


def study_init(self, data_root, output, strict_raw=False):
    self.root = data_root.resolve()
    self.output = output.resolve()
    self.strict_raw = strict_raw
    self.used, self.checks = {}, {}
    self.output.mkdir(parents=True, exist_ok=True)
    (self.output / "plot_data").mkdir(exist_ok=True)


def study_path(self, name):
    path = resolve(name)
    self.used[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def source_read(self, name, raw=False):
    prefix = "baseline/" if raw else "analysis/"
    path = resolve(prefix + name)
    self.paths.append(path)
    return pd.read_csv(path, low_memory=False)


def source_latest(self, folder, filename):
    path = resolve("analysis/" + folder + "/" + filename)
    self.paths.append(path)
    return pd.read_csv(path, low_memory=False)


main.Study.__init__ = study_init
main.Study.path = study_path
main.Study.has = lambda self, name: any(Path(r["key"]).name == name for r in INDEX)
supplementary.Sources.read = source_read
supplementary.Sources.latest = source_latest


def draw(number, supplementary=False, extended=False, output=None):
    destination = Path(output) if output is not None else ROOT / "Redrawn_Figures"
    if extended:
        return extended_figures.draw(number, destination / "extended")
    if supplementary:
        module = sys.modules["supplementary_figures"]
        module.main([
            "--figure", str(extended_figures.REMAINING_SI[number]),
            "--published-number", str(number),
            "--raw-root", str(ROOT / "Data"),
            "--analysis-root", str(ROOT / "Data"),
            "--output-root", str(destination / "supplementary"),
        ])
        return destination / "supplementary" / f"Figure_S{number:02}.png"
    return main.draw(number, data_root=ROOT / "Data", output=destination / "main", strict_raw=True)
