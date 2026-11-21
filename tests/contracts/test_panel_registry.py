import csv
from pathlib import Path

import yaml

from thermal_heterogeneity.figure_data import PANEL_IDS, SCHEMATICS, validate_panel_registry


def test_locked_registry_has_exactly_48_panels():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "src/settings" / "paper.yaml").read_text(encoding="utf-8"))
    validate_panel_registry(config["figures"]["panels"])
    assert len(PANEL_IDS) == 48
    assert SCHEMATICS == {"Fig3a", "Fig3c", "Fig4a", "Fig5a", "Fig6a", "Fig6h"}


def test_publication_layout_identifies_each_figure_input():
    root = Path(__file__).resolve().parents[2]
    layout = root / "src/thermal_heterogeneity/assets/plot_input_layout.csv"
    with layout.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [row["input_key"] for row in rows]
    assert len(keys) == len(set(keys))
    assert {row["file"].split("/")[0] for row in rows} == {"Hourly", "Maps", "Tables"}
    assert "baseline/matrix/models/Miami.geometry.json" in keys
    assert {"baseline/demand/ASHRAE_RECORDS.csv", "baseline/demand/CTCD_RECORDS.csv"} <= set(keys)
