"""Regenerate each Plot.ipynb figure from data/figure_data without running simulations."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def execute_notebook(root, output, selected=None):
    """Execute each complete figure cell with its own Python namespace."""
    notebook = json.loads((root / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != 16:
        raise ValueError("Expected 16 complete, independently executable figure cells")
    previous = Path.cwd()
    try:
        os.chdir(root)
        for cell in cells:
            label = cell["metadata"]["figure"]
            if selected is not None and label not in selected:
                continue
            namespace = {"__name__": "__main__", "PUBLICATION_OUTPUT": output}
            exec(compile("".join(cell["source"]), f"Plot.ipynb:{label}", "exec"), namespace)
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/publication")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--figure", nargs="+", help="Selected figure IDs: 1–6, E1–E4 or S1–S6")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Publication output must remain inside the project")
    if args.plan:
        print(
            json.dumps(
                {
                    "input": "data/figure_data/FIGURE_DATA_MANIFEST.json",
                    "notebook": "plot/Plot.ipynb",
                    "main_figures": 6,
                    "extended_data_figures": 4,
                    "supplementary_figures": 6,
                    "EnergyPlus_runs": 0,
                    "manuscripts_required": False,
                },
                indent=2,
            )
        )
        return
    from src.thermal_heterogeneity.figure_inputs import verify

    verify()
    output.mkdir(parents=True, exist_ok=True)
    selected = None
    if args.figure:
        selected = {f"Figure_{value[0].upper()}{int(value[1:]):02}" if value[0].isalpha()
                    else f"Figure_{int(value):02}" for value in args.figure}
        allowed = {f"Figure_{i:02}" for i in range(1, 7)} | {f"Figure_E{i:02}" for i in range(1, 5)} | {f"Figure_S{i:02}" for i in range(1, 7)}
        if not selected <= allowed:
            parser.error("Unknown figure ID")
    execute_notebook(ROOT, output, selected)
    if selected is not None:
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/analysis/figure_tables.py"),
            "--raw-root",
            str(ROOT / "data/figure_data/output/from_raw"),
            "--analysis-root",
            str(ROOT / "data/figure_data/output/strengthening"),
            "--publication-root",
            str(output),
            "--resource-scenarios",
            str(
                ROOT
                / "data/figure_data/output/publication/support/implementation/IMPLEMENTATION_RESOURCE_SCENARIOS.csv"
            ),
        ],
        check=True,
    )
    (output / "NOTEBOOK_EXECUTION.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "executed_code_cells": 16,
                "main_figures": 6,
                "extended_data_figures": 4,
                "supplementary_figures": 6,
                "tables": 10,
                "EnergyPlus_runs": 0,
                "input_root": "data/figure_data",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
