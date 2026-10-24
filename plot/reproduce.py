"""Execute actual figure cells in Plot.ipynb without storing notebook outputs."""
from pathlib import Path
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", nargs="+", help="Figure IDs: 1 ... 6 or S1 ... S24; default is all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    selected = None if args.figure is None else {
        f"Figure_S{int(value[1:]):02d}" if value.upper().startswith("S")
        else f"Figure_{int(value):02d}" for value in args.figure}
    labels = {cell.get("metadata", {}).get("figure") for cell in notebook["cells"]
              if cell["cell_type"] == "code"}
    if selected is not None and selected - labels:
        parser.error(f"Unknown figure IDs: {sorted(selected - labels)}")
    previous = Path.cwd()
    try:
        os.chdir(root)
        namespace = {"__name__":"notebook_figures", "__file__":str(root / "plot/Plot.ipynb")}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            label = cell.get("metadata", {}).get("figure")
            if label == "Setup" or selected is None or label in selected:
                exec(compile("".join(cell["source"]), f"Plot.ipynb:{label}", "exec"), namespace)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
