"""Regenerate each Plot.ipynb figure from Data_for_Figures without running simulations."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/publication")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--documents",
        action="store_true",
        help="Also build Word files using retained author sources and templates",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Publication output must remain inside the project")
    if args.plan:
        print(
            json.dumps(
                {
                    "input": "Data_for_Figures/FIGURE_DATA_MANIFEST.json",
                    "notebook": "plot/Plot.ipynb",
                    "main_figures": 6,
                    "extended_data_figures": 4,
                    "supplementary_figures": 6,
                    "EnergyPlus_runs": 0,
                    "documents": args.documents,
                },
                indent=2,
            )
        )
        return
    from publication.figure_data import verify

    verify()
    notebook = json.loads((ROOT / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != 17:
        raise ValueError("Expected one setup cell and 16 single-figure cells")
    output.mkdir(parents=True, exist_ok=True)
    namespace = {"__name__": "__main__", "PUBLICATION_OUTPUT": output}
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        for index, cell in enumerate(cells):
            exec(compile("".join(cell["source"]), f"Plot.ipynb:cell{index}", "exec"), namespace)
    finally:
        os.chdir(previous)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "publication/build_tables.py"),
            "--raw-root",
            str(ROOT / "Data_for_Figures/output/from_raw"),
            "--analysis-root",
            str(ROOT / "Data_for_Figures/output/strengthening"),
            "--publication-root",
            str(output),
            "--resource-scenarios",
            str(
                ROOT
                / "Data_for_Figures/output/publication/support/implementation/IMPLEMENTATION_RESOURCE_SCENARIOS.csv"
            ),
        ],
        check=True,
    )
    if args.documents:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "publication/build_documents.py"),
                "--publication-root",
                str(output),
                "--output-root",
                str(output / "documents"),
            ],
            check=True,
        )
    (output / "NOTEBOOK_EXECUTION.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "executed_code_cells": 17,
                "main_figures": 6,
                "extended_data_figures": 4,
                "supplementary_figures": 6,
                "tables": 10,
                "EnergyPlus_runs": 0,
                "input_root": "Data_for_Figures",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
