import csv
import json
from pathlib import Path

import yaml

from thermal_heterogeneity.figure_data import PANEL_IDS, SCHEMATICS, validate_panel_registry


def test_locked_registry_has_exactly_48_panels():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    validate_panel_registry(config["figures"]["panels"])
    assert len(PANEL_IDS) == 48
    assert SCHEMATICS == {"Fig3a", "Fig3c", "Fig4a", "Fig5a", "Fig6a", "Fig6h"}


def test_analysis_bundle_contains_panel_materialization_support():
    root = Path(__file__).resolve().parents[2]
    with (root / "data" / "analysis_bundle_manifest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    release = json.loads(
        (root / "data" / "analysis_bundle_release.json").read_text(encoding="utf-8")
    )
    assert len(rows) == release["file_count"] == 46
    assert any(
        row["logical_path"] == "panel_support/panel_materialization_support.json"
        for row in rows
    )
