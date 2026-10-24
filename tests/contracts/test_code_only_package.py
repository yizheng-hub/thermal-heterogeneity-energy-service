from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_code_allowlist_excludes_manuscripts_templates_and_generated_binary_products():
    names = [line for line in (ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines()
             if line and not line.startswith("#")]
    forbidden = {".docx", ".pdf", ".png", ".svg", ".npz", ".npy", ".zst", ".zip", ".gz"}
    for name in names:
        path = Path(name)
        assert not {"drafts", "templates", "private", "raw"} & set(path.parts)
        assert path.suffix.lower() not in forbidden
        assert path.name not in {"MAIN_MANUSCRIPT.md", "SUPPLEMENTARY_INFORMATION.md"}


def test_scientific_entry_plan_works_without_manuscripts_or_templates():
    result = subprocess.run([sys.executable, str(ROOT / "reproduce.py"), "--plan"],
                            text=True, encoding="utf-8", capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["manuscripts_required"] is False
    assert plan["data_files"] == 877 and plan["figures"] == 30
    assert plan["typical_day_data_files"] == 44 and plan["typical_day_manifest_files"] == 1
    assert len(plan["steps"]) == 13
    assert all((ROOT / path).is_file() for path in plan["steps"])
    assert not any("document" in path or "compile_references" in path for path in plan["steps"])


def test_local_inputs_and_products_stay_outside_public_source_but_plot_notebook_remains():
    layout = load_module(ROOT / "src/utilities/check_release_layout.py", "source_layout")
    for directory in ["data/inputs", "data/typical_days", "data/upstream", "data/engineering_qa",
                      "output", "results", "intermediate_files", "plot/main", "plot/supplementary"]:
        assert layout._is_ignored_local(Path(directory) / "product.csv")
        assert not layout._is_ignored_local(Path(directory) / "README.md")
    for source in ["plot/Plot.ipynb", "data/metadata/source_manifest.csv", "src/analysis/reproduce_energy_service_publication.py"]:
        assert not layout._is_ignored_local(Path(source))


def test_allowlist_cannot_admit_local_input_data(tmp_path):
    local = tmp_path / "data/inputs/records.csv"
    local.parent.mkdir(parents=True)
    local.write_text("value\n1\n", encoding="utf-8")
    allowlist = tmp_path / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt"
    allowlist.parent.mkdir(parents=True)
    allowlist.write_text("data/inputs/records.csv\nsrc/provenance/PUBLIC_FILE_ALLOWLIST.txt\n", encoding="utf-8")
    layout = load_module(ROOT / "src/utilities/check_release_layout.py", "closed_layout")
    report = layout.run_checks(tmp_path)
    assert report["status"] == "FAIL"
    assert "local-only file must not be allow-listed: data/inputs/records.csv" in report["errors"]


@pytest.mark.parametrize("has_output", [False, True])
def test_plot_notebook_keeps_generated_results_out_of_public_source(tmp_path, has_output, monkeypatch):
    path = tmp_path / "plot/Plot.ipynb"
    path.parent.mkdir()
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": None,
        "source": [], "outputs": [{"output_type": "stream", "text": "fixture"}] if has_output else []}]}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ROOT / "src/utilities"))
    scanner = load_module(ROOT / "src/utilities/release_scan.py", "notebook_scanner")
    errors = scanner._scan_common(tmp_path)
    assert any("notebook contains stored outputs" in item for item in errors) == has_output


@pytest.mark.parametrize("source,expected", [("if x:" + chr(10) + "    pass" + chr(10), False),
                                            ('path = "' + 'Z' + ':' + '/fixture/data"\n', True)])
def test_notebook_path_scan_reads_decoded_source(tmp_path, monkeypatch, source, expected):
    path = tmp_path / "plot/Plot.ipynb"
    path.parent.mkdir()
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "execution_count": None,
        "source": [source], "outputs": []}]}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ROOT / "src/utilities"))
    scanner = load_module(ROOT / "src/utilities/release_scan.py", "decoded_notebook_scanner")
    errors = scanner._scan_common(tmp_path)
    assert any("machine-specific absolute path" in item for item in errors) == expected


@pytest.mark.parametrize("name,needle", [
    ("MAIN_MANUSCRIPT.md", "forbidden manuscript"),
    ("SUPPLEMENTARY_INFORMATION.md", "forbidden manuscript"),
    ("template.docx", "forbidden file type"),
])
def test_code_release_rejects_accidental_manuscript_or_template(tmp_path, name, needle):
    (tmp_path / name).write_text("not distributable", encoding="utf-8")
    layout = load_module(ROOT / "src/utilities/check_release_layout.py", "code_layout")
    report = layout.run_checks(tmp_path)
    assert report["status"] == "FAIL"
    assert any(needle in item and name in item for item in report["errors"])


def test_all_27_figure_assets_assemble_without_any_word_reference_assets(tmp_path):
    previous, stage = tmp_path / "previous", tmp_path / "stage"

    def bundle(base):
        base.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("_word.png", ".png", ".pdf", ".svg"):
            Path(str(base) + suffix).write_bytes(b"fixture graphical product")

    for number in range(1, 7):
        bundle(previous / f"figures/figure_{number:02d}/figure_{number:02d}")
        caption = previous / f"captions_alt/figure_{number:02d}/caption.txt"
        caption.parent.mkdir(parents=True, exist_ok=True)
        caption.write_text(f"Figure {number} | Data and symbols.", encoding="utf-8")
        if number > 1:
            bundle(stage / f"figures/Figure_{number:02d}")
            (stage / f"figures/Figure_{number:02d}_caption.txt").write_text("Figure | Data.")
    for number in range(1, 16):
        bundle(previous / f"supplementary_figures/Figure_S{number:02d}")
        (previous / f"supplementary_figures/Figure_S{number:02d}_caption.txt").write_text("Figure | Data.")
    bundle(stage / "figures/Supplementary_Figure_S21")
    (stage / "figures/Supplementary_Figure_S21_caption.txt").write_text("Figure | Data.")
    assembler = load_module(ROOT / "src/analysis/energy_service_paper/scripts/assemble_publication_assets.py", "code_assets")
    assembler.assemble(previous, stage)
    assert len(list((stage / "supplementary_figures").glob("*_word.png"))) == 21
    assert len(list((stage / "figures").glob("figure_*/*_word.png"))) == 6
    assert not (stage / "reference").exists()
