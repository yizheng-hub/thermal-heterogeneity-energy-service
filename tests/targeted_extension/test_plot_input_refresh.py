"""Final plot-input hashes must include captions changed by figure cells."""

import hashlib
import csv
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_rehashes_updated_caption_and_excludes_itself(tmp_path):
    module = load_module("plot_input_refresh", ROOT / "src/utilities/prepare_plot_data.py")
    caption = tmp_path / "captions/Figure_03.txt"
    caption.parent.mkdir()
    caption.write_text("Old caption", encoding="utf-8")
    table = tmp_path / "values.csv"
    table.write_text("value\n1\n", encoding="utf-8")
    assert module.refresh_plot_inputs(tmp_path) == 2
    before = json.loads((tmp_path / "PLOT_INPUTS.json").read_text(encoding="utf-8"))
    caption.write_text("Revised 25°C caption", encoding="utf-8")
    assert module.refresh_plot_inputs(tmp_path) == 2
    manifest = tmp_path / "PLOT_INPUTS.json"
    after = json.loads(manifest.read_text(encoding="utf-8"))
    hashes = {row["path"]: row["sha256"] for row in after["files"]}
    assert hashes["captions/Figure_03.txt"] == hashlib.sha256(caption.read_bytes()).hexdigest()
    assert after["files"] != before["files"]
    assert set(hashes) == {"captions/Figure_03.txt", "values.csv"}
    assert hashes["values.csv"] == hashlib.sha256(table.read_bytes()).hexdigest()
    stable = manifest.read_bytes()
    module.refresh_plot_inputs(tmp_path)
    assert manifest.read_bytes() == stable


def test_source_map_index_follows_added_panels_without_rehashing_science(tmp_path):
    module = load_module("plot_map_refresh", ROOT / "src/utilities/prepare_plot_data.py")
    mapping = tmp_path / "FIGURE_SOURCE_MAP.csv"
    mapping.write_text("figure,panel\nFig2,a\nFig3,d\n", encoding="utf-8")
    index = tmp_path / "SOURCE_DATA_INDEX.csv"
    index.write_text("path,rows,columns,sha256\nFIGURE_SOURCE_MAP.csv,1,2,old\nvalues.csv,7,3,locked\n",
                     encoding="utf-8")
    module.refresh_plot_inputs(tmp_path)
    with index.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == {"path":"FIGURE_SOURCE_MAP.csv", "rows":"2", "columns":"2",
                       "sha256":hashlib.sha256(mapping.read_bytes()).hexdigest()}
    assert rows[1]["sha256"] == "locked"


def test_figure_two_requires_information_hierarchy_source(tmp_path):
    runner = load_module("plot_reproduce_required_inputs", ROOT / "plot/reproduce.py")
    with pytest.raises(FileNotFoundError, match="INFORMATION_HIERARCHY_SUMMARY.csv"):
        runner.validate_figure_inputs(tmp_path, {"Figure_02"})
    source = tmp_path / "results/decision_value/INFORMATION_HIERARCHY_SUMMARY.csv"
    source.parent.mkdir(parents=True)
    source.write_text("metric,count\nzone_mean_safe_cells,275\n", encoding="utf-8")
    runner.validate_figure_inputs(tmp_path, {"Figure_02"})


@pytest.mark.parametrize("selection", [None, "3"])
def test_plot_cli_refreshes_after_all_or_single_figure(tmp_path, monkeypatch, selection):
    runner = load_module("plot_reproduce_refresh", ROOT / "plot/reproduce.py")
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    plot = tmp_path / "plot"
    plot.mkdir()
    captions = tmp_path / "results/captions"
    captions.mkdir(parents=True)
    for number in (3, 4):
        (captions / f"Figure_{number:02d}.txt").write_text("old", encoding="utf-8")
    cells = [{"cell_type": "code", "metadata": {"figure": "Setup"},
              "source": ["from pathlib import Path\n",
                         "from utilities.prepare_plot_data import refresh_plot_inputs\n"]}]
    for number in (3, 4):
        cells.append({"cell_type": "code", "metadata": {"figure": f"Figure_{number:02d}"},
                      "source": [f"Path('results/captions/Figure_{number:02d}.txt').write_text('new{number}', encoding='utf-8')\n"]})
    (plot / "Plot.ipynb").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    monkeypatch.setattr(runner, "__file__", str(plot / "reproduce.py"))
    arguments = ["reproduce.py"] + (["--figure", selection] if selection else [])
    monkeypatch.setattr(sys, "argv", arguments)
    previous = Path.cwd()
    runner.main()
    assert Path.cwd() == previous
    inventory = json.loads((tmp_path / "results/PLOT_INPUTS.json").read_text(encoding="utf-8"))
    hashes = {row["path"]: row["sha256"] for row in inventory["files"]}
    assert (captions / "Figure_03.txt").read_text() == "new3"
    assert (captions / "Figure_04.txt").read_text() == ("old" if selection else "new4")
    for relative, checksum in hashes.items():
        assert checksum == hashlib.sha256((tmp_path / "results" / relative).read_bytes()).hexdigest()
