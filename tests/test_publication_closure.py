"""Contracts for portable, single-figure notebook reproduction."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def loader():
    spec = importlib.util.spec_from_file_location(
        "figure_data", ROOT / "publication/figure_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_result_integrity_rejects_changed_payload_and_escape(tmp_path):
    module = loader()
    module.ROOT = tmp_path
    result = tmp_path / "Result"
    result.mkdir()
    payload = result / "scores.csv"
    payload.write_text("energy\n1\n")
    row = {
        "result": "Result/scores.csv",
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    manifest = result / "FIGURE_DATA_MANIFEST.json"
    manifest.write_text(json.dumps({"files": [row]}))
    assert len(module.verify()["files"]) == 1
    payload.write_text("energy\n2\n")
    with pytest.raises(ValueError, match="changed or missing"):
        module.verify()
    row.update(result="Result/../outside.csv", sha256="unused")
    manifest.write_text(json.dumps({"files": [row]}))
    with pytest.raises(ValueError, match="changed or missing"):
        module.verify()


def test_notebook_figures_are_individually_addressable_and_packaged():
    notebook = json.loads((ROOT / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 23
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in cells)
    assert all("".join(c["source"]).count("draw(") == 1 for c in cells[1:])
    allowed = set((ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines())
    for folder in ["publication", "plot"]:
        for path in (ROOT / folder).glob("*.py"):
            assert path.relative_to(ROOT).as_posix() in allowed
    assert "reproduce_publication.py" in allowed


def test_quarter_reassignment_table_excludes_other_move_budgets():
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "build_tables", ROOT / "publication/build_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for control, base in [("personal_center", 1.0), ("personal_range", 2.0)]:
        for budget, value in [(25, base), (50, 100.0), (100, 200.0)]:
            row = {"control": control, "layout": f"within_floor_relocation_{budget}pct"}
            for suffix in ["vav", "alternative"]:
                for metric in ["population_gain_c", "fixed_tail_gain_c", "energy_change_pct"]:
                    row[f"{metric}_{suffix}"] = value
            rows.append(row)
    summary = module.quarter_reassignment_rows(pd.DataFrame(rows))
    assert len(summary) == 4
    assert [row[2] for row in summary] == [1, 1, 1, 1]
    assert [row[3:] for row in summary] == [[1.0] * 3, [1.0] * 3, [2.0] * 3, [2.0] * 3]


def test_publication_support_removes_identity_without_changing_plotted_support():
    import pandas as pd

    module = loader()
    original = pd.DataFrame(
        {
            "person": ["a", "a", "b"],
            "temperature": [20.0, 23.0, 24.0],
            "timestamp": ["private"] * 3,
            "gender": ["private"] * 3,
        }
    )
    public, description = module.minimize_human_inputs(
        "IDENTIFIED_AS_HRAE_RECORDS.csv.gz", original
    )
    assert list(public.columns) == ["records", "lo", "hi"]
    assert public.to_dict("list") == {"records": [2, 1], "lo": [20.0, 24.0], "hi": [23.0, 24.0]}
    assert "no identity" in description
