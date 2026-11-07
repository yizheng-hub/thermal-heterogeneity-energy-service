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
    result = tmp_path / "Data_for_Figures"
    result.mkdir()
    payload = result / "scores.csv"
    payload.write_text("energy\n1\n")
    row = {
        "result": "Data_for_Figures/scores.csv",
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    manifest = result / "FIGURE_DATA_MANIFEST.json"
    manifest.write_text(json.dumps({"files": [row]}))
    assert len(module.verify()["files"]) == 1
    payload.write_text("energy\n2\n")
    with pytest.raises(ValueError, match="changed or missing"):
        module.verify()
    row.update(result="Data_for_Figures/../outside.csv", sha256="unused")
    manifest.write_text(json.dumps({"files": [row]}))
    with pytest.raises(ValueError, match="changed or missing"):
        module.verify()


def test_notebook_figures_are_individually_addressable_and_packaged():
    notebook = json.loads((ROOT / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 16
    assert all(not c.get("outputs") and c.get("execution_count") is None for c in cells)
    import ast
    for cell in cells:
        tree = ast.parse("".join(cell["source"]))
        assert any(isinstance(node, ast.FunctionDef) for node in tree.body)
        assert any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FONT_SIZE" for t in node.targets) for node in tree.body)
        assert not any(isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in {"publication", "plot", "thermal_heterogeneity", "reference_figures", "strengthened_figures"} for node in ast.walk(tree))
    allowed = set((ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines())
    for folder in ["publication", "plot"]:
        for path in (ROOT / folder).glob("*.py"):
            assert path.relative_to(ROOT).as_posix() in allowed
    assert "reproduce_publication.py" in allowed


def test_export_reuses_contained_inputs_and_transformed_geometry(tmp_path):
    module = loader()
    module.ROOT = tmp_path
    records = []
    for source, result, payload in [
        ("output/Miami.idf", "Data_for_Figures/output/Miami.geometry.json", b'{"floors":{}}'),
        ("output/scores.csv", "Data_for_Figures/output/scores.csv", b"score\n1\n"),
    ]:
        path = tmp_path / result
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
            {
                "source": source,
                "result": result,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_sha256": "original-scientific-input-hash",
                "bytes": len(payload),
                "transformation": "retained",
            }
        )
    manifest = tmp_path / "Data_for_Figures/FIGURE_DATA_MANIFEST.json"
    manifest.write_text(json.dumps({"files": records}))
    publication = tmp_path / "publication_output"
    for folder, prefix, count in [("main", "Figure_", 6), ("extended", "Figure_E", 4), ("supplementary", "Figure_S", 6)]:
        (publication / folder).mkdir(parents=True)
        for n in range(1, count + 1):
            row = records[0 if folder == "main" else 1]
            name = row["result"] if folder == "main" else row["source"]
            (publication / folder / f"{prefix}{n:02}.json").write_text(
                json.dumps({"input_sha256": {name: row["sha256"]}})
            )
    (publication / "tables").mkdir()
    (publication / "tables/TABLES.json").write_text(
        json.dumps({"input_sha256": {records[1]["result"]: records[1]["sha256"]}})
    )
    module.export(publication)
    assert module.verify()["files"] == records
    assert not (tmp_path / "Data_for_Figures/Data_for_Figures").exists()
    assert not (tmp_path / "output/Miami.idf").exists()


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


def test_public_figure_generation_does_not_require_private_manuscript(tmp_path):
    spec = importlib.util.spec_from_file_location("publication_runner", ROOT / "reproduce_publication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "plot").mkdir()
    cells = [{"cell_type": "code", "metadata": {"figure": f"Figure_{i:02}"}, "source": [
        "from pathlib import Path\n",
        "assert 'previous_cell' not in globals()\n",
        "previous_cell = True\n",
        f"(PUBLICATION_OUTPUT / 'Figure_{i:02}.txt').write_text('drawn')\n",
    ]} for i in range(1, 17)]
    (tmp_path / "plot/Plot.ipynb").write_text(json.dumps({"cells": cells}))
    output = tmp_path / "output"
    output.mkdir()
    module.execute_notebook(tmp_path, output)
    assert len(list(output.glob("Figure_*.txt"))) == 16
    assert not (tmp_path / "publication/Main_manuscript.md").exists()


def test_internal_cross_references_remain_bold_when_documents_rebuild():
    pytest.importorskip("docx", reason="Optional Word document dependency")
    spec = importlib.util.spec_from_file_location("build_documents", ROOT / "publication/build_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paragraph = module.Document().add_paragraph()
    module.inline(paragraph, "See Fig. 1a,b; Fig. S1; Extended Data Fig. 2e–h; Supplementary Notes 8 and 9; Section 4.2; Supplementary Tables 1–3 [1].", {1: 1})
    references = [run for run in paragraph.runs if module.CROSS_REFERENCE.fullmatch(run.text)]
    assert [run.text for run in references] == ["Fig. 1a,b", "Fig. S1", "Extended Data Fig. 2e–h", "Supplementary Notes 8 and 9", "Section 4.2", "Supplementary Tables 1–3"]
    assert all(run.bold for run in references)
    assert module.CROSS_REFERENCE.search("Equations (2) and (3), w is a weight")[0] == "Equations (2) and (3)"
    assert next(run for run in paragraph.runs if run.font.superscript).text == "1"
    module.typography(paragraph)
    assert paragraph.alignment == module.WD_ALIGN_PARAGRAPH.JUSTIFY
    assert paragraph.paragraph_format.line_spacing == 1.5
    assert all(run.bold for run in references)


def test_published_signed_excess_matches_source_score_minus_limit():
    import pandas as pd

    if not (ROOT / "Data_for_Figures").is_dir():
        pytest.skip("Download and extract Study_Data.zip first")
    source = pd.read_csv(ROOT / "Data_for_Figures/output/from_raw/results/decision_effect_sizes/DECISION_EFFECT_SIZES.csv")
    source = source.loc[source.selection_disagreement.eq(True)].reset_index(drop=True)
    tables = json.loads((ROOT / "output/publication/tables/TABLES.json").read_text(encoding="utf-8"))["tables"]
    table = next(t for t in tables if t["number"] == 3)
    actual = pd.DataFrame(table["rows"], columns=table["columns"])
    assert len(actual) == len(source) == 10
    for column, field in [("Population excess (°C)", "zone_mean_selected_full_population_margin_c"), ("Tail excess (°C)", "zone_mean_selected_full_tail_margin_c")]:
        assert actual[column].tolist() == source[field].map(lambda x: f"{x:.3f}").tolist()
    assert actual["Extra energy (MWh)"].tolist() == (source.delta_purchased_kwh_full_minus_zone_mean / 1000).map(lambda x: f"{x:.3f}").tolist()
