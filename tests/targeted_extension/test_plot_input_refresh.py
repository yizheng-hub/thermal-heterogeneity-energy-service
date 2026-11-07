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


def _write_final_story_bundle(root: Path) -> Path:
    story = root / "results/global_extension/final_story"
    story.mkdir(parents=True)
    outputs = {
        "FINAL_STORY_SUMMARY.json": json.dumps(
            {"design": {"climates": 9, "candidates": 810}}
        ),
        "FINAL_INFORMATION_DIAGNOSTIC.csv": "climate,value\nBeijing,1\n",
        "FINAL_DIRECT_COMPARISONS.csv": "climate,value\nBeijing,1\n",
    }
    for name, content in outputs.items():
        (story / name).write_text(content, encoding="utf-8")
    hashes = {
        name: hashlib.sha256((story / name).read_bytes()).hexdigest()
        for name in outputs
    }
    (story / "FINAL_STORY_HASHES.json").write_text(
        json.dumps({"status": "PASS", "outputs": hashes}), encoding="utf-8"
    )
    return story


def test_figure_two_requires_hashed_global_story_sources(tmp_path):
    runner = load_module("plot_reproduce_required_inputs", ROOT / "plot/reproduce.py")
    with pytest.raises(FileNotFoundError, match="FINAL_STORY_SUMMARY.json"):
        runner.validate_figure_inputs(tmp_path, {"Figure_02"})
    story = _write_final_story_bundle(tmp_path)
    for relative in runner.REQUIRED_FIGURE_INPUTS["Figure_02"]:
        path = tmp_path / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("value\n1\n", encoding="utf-8")
    runner.validate_figure_inputs(tmp_path, {"Figure_02"})
    (story / "FINAL_INFORMATION_DIAGNOSTIC.csv").write_text(
        "climate,value\nBeijing,2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after assembly"):
        runner.validate_figure_inputs(tmp_path, {"Figure_02"})


def test_figure_one_requires_raw_rebuilt_sources_instead_of_legacy_centers(tmp_path):
    runner = load_module("plot_reproduce_figure_one_inputs", ROOT / "plot/reproduce.py")
    for relative in runner.REQUIRED_FIGURE_INPUTS["Figure_01"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="CITY_MAP_POINTS"):
        runner.validate_figure_inputs(tmp_path, {"Figure_01"})
    for relative in (
        "observation_figures/CITY_MAP_POINTS.csv",
        "observation_figures/DEMAND_COMPOSITION_PROFILES.csv",
        "observation_figures/LEAVE_ONE_SOURCE_VARIANCE.csv",
        "observation_figures/POLICY_RANGE_POSITIONS.csv",
        "observations/ASHRAE_NESTED_VARIANCE.csv",
        "shared_room/SHARED_ROOM_SUMMARY.csv",
    ):
        path = tmp_path / "output/from_raw/results" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")
    runner.validate_figure_inputs(tmp_path, {"Figure_01"})


def test_legacy_four_context_figure_source_is_rejected():
    runner = load_module("plot_reproduce_stale_source", ROOT / "plot/reproduce.py")
    notebook = {
        "cells": [{
            "cell_type": "code",
            "metadata": {"figure": "Figure_03"},
            "source": [
                "pd.read_csv(RESULTS / 'final_submission/evidence/old.csv')\n"
            ],
        }]
    }
    with pytest.raises(RuntimeError, match="superseded four-context analysis"):
        runner.validate_notebook_sources(notebook, {"Figure_03"})


def test_publication_legends_must_be_frameless():
    runner = load_module("plot_reproduce_legend_style", ROOT / "plot/reproduce.py")
    notebook = {
        "cells": [{
            "cell_type": "code",
            "metadata": {"figure": "Figure_02"},
            "source": ["axis.legend(loc='upper right')\n"],
        }]
    }
    with pytest.raises(ValueError, match="frameon=False"):
        runner.validate_legend_styles(notebook, {"Figure_02"})
    notebook["cells"][0]["source"] = [
        "axis.legend(loc='upper right', frameon=False, "
        "facecolor='none', edgecolor='none')\n"
    ]
    runner.validate_legend_styles(notebook, {"Figure_02"})


def test_figure_four_building_to_zone_counts_match_final_spatial_evidence():
    source = ROOT / "results/global_extension/final_story/FINAL_SPATIAL_EFFECTS.csv"
    if not source.is_file():
        pytest.skip("Optional retained legacy spatial table is not in the source-only package")
    with source.open(encoding="utf-8", newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["axis"] == "permission"
            and row["reference"] == "building"
            and row["alternative"] == "native_zone"
        ]

    metrics = (
        ("delta_mean_reference_mismatch_c", lambda value: value < 0),
        ("delta_fixed_tail_reference_mismatch_c", lambda value: value < 0),
        ("delta_modeled_interval_person_hour_coverage", lambda value: value > 0),
        ("delta_facility_purchased_pct", lambda value: value > 0),
    )
    expected = {
        "personal_center": [54, 46, 46, 50],
        "personal_range": [50, 53, 37, 52],
    }
    for control, expected_counts in expected.items():
        selected = [row for row in rows if row["control"] == control]
        assert len(selected) == 54
        observed = [
            sum(predicate(float(row[column])) for row in selected)
            for column, predicate in metrics
        ]
        assert observed == expected_counts


@pytest.mark.parametrize("selection", [None, "3"])
def test_notebook_executes_all_or_selected_independent_cells(tmp_path, selection):
    runner = load_module("publication_cell_execution", ROOT / "reproduce_publication.py")
    plot = tmp_path / "plot"
    plot.mkdir()
    captions = tmp_path / "results/captions"
    captions.mkdir(parents=True)
    for number in (3, 4):
        (captions / f"Figure_{number:02d}.txt").write_text("old", encoding="utf-8")
    cells = []
    for number in range(1, 17):
        cells.append({"cell_type": "code", "metadata": {"figure": f"Figure_{number:02d}"},
                      "source": ["from pathlib import Path\n",
                                 "assert 'previous_figure' not in globals()\n",
                                 "previous_figure = True\n",
                                 f"Path('results/captions/Figure_{number:02d}.txt').write_text('new{number}', encoding='utf-8')\n"]})
    (plot / "Plot.ipynb").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    previous = Path.cwd()
    runner.execute_notebook(tmp_path, tmp_path / "output", None if selection is None else {"Figure_03"})
    assert Path.cwd() == previous
    assert (captions / "Figure_03.txt").read_text() == "new3"
    assert (captions / "Figure_04.txt").read_text() == ("old" if selection else "new4")
