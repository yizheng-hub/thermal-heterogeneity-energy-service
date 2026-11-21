"""Regenerate each Plot.ipynb figure from results/figure_data without running simulations."""

from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def crop_png_whitespace(path, padding=10, threshold=5):
    """Remove only the blank outer border while retaining a small safe margin."""
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        difference = ImageChops.difference(white.convert("RGB"), Image.new("RGB", rgba.size, "white"))
        red, green, blue = difference.split()
        maximum = ImageChops.lighter(ImageChops.lighter(red, green), blue)
        mask = maximum.point(lambda value: 255 if value > threshold else 0)
        content = mask.getbbox()
        if content is None:
            return image.size, image.size
        left, top, right, bottom = content
        crop = (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
        old_size = image.size
        if crop == (0, 0, image.width, image.height):
            return old_size, old_size
        cropped = image.crop(crop)
        save_options = {"dpi": image.info["dpi"]} if "dpi" in image.info else {}
        cropped.save(path, **save_options)
        return old_size, cropped.size


def crop_publication_pngs(output, selected=None):
    """Apply the same post-render crop to every requested publication figure."""
    paths = [
        *sorted((output / "main").glob("Figure_[0-9][0-9].png")),
        *sorted((output / "extended").glob("Figure_E[0-9][0-9].png")),
        *sorted((output / "supplementary").glob("Figure_S[0-9][0-9].png")),
    ]
    if selected is not None:
        paths = [path for path in paths if path.stem in selected]
    return {path.stem: crop_png_whitespace(path) for path in paths}


def execute_notebook(root, output, selected=None):
    """Execute each complete figure cell with its own Python namespace."""
    notebook = json.loads((root / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(cells) != 17:
        raise ValueError("Expected 17 independently executable figure cells")
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--raw-run-root", type=Path, help="Draw directly from completed EnergyPlus and analysis results")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--figure", nargs="+", help="Selected figure IDs: 1–6, E1–E4 or S1–S7")
    args = parser.parse_args()
    raw = args.raw_run_root.resolve() if args.raw_run_root else None
    output = (args.output_root or ((raw or ROOT / "output") / "publication")).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Publication output must remain inside the project")
    if args.plan:
        print(
            json.dumps(
                {
                    "input": str(raw) if raw else "results/figure_data/FIGURE_DATA_MANIFEST.json",
                    "notebook": "plot/Plot.ipynb",
                    "main_figures": 6,
                    "extended_data_figures": 4,
                    "supplementary_figures": 7,
                    "EnergyPlus_runs": 0,
                    "manuscripts_required": False,
                },
                indent=2,
            )
        )
        return
    if raw is not None:
        sys.path.insert(0, str(ROOT / "src"))
        from src.thermal_heterogeneity.raw_plot_inputs import prepare

        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as tmp:
            project = Path(tmp)
            for name in ["plot/Plot.ipynb", "src/analysis/figure_tables.py",
                         "src/thermal_heterogeneity/figure_inputs.py", "reproduce_publication.py"]:
                (project / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, project / name)
            if (ROOT / "tools/fonts").is_dir():
                shutil.copytree(ROOT / "tools/fonts", project / "tools/fonts")
            prepare(raw, project)
            command = [sys.executable, "reproduce_publication.py", "--output-root", "rendered"]
            if args.figure:
                command.extend(["--figure", *args.figure])
            subprocess.run(command, cwd=project, check=True)
            shutil.copytree(project / "rendered", output, dirs_exist_ok=True)
        return
    from src.thermal_heterogeneity.figure_inputs import verify

    verify()
    output.mkdir(parents=True, exist_ok=True)
    selected = None
    if args.figure:
        selected = {f"Figure_{value[0].upper()}{int(value[1:]):02}" if value[0].isalpha()
                    else f"Figure_{int(value):02}" for value in args.figure}
        allowed = {f"Figure_{i:02}" for i in range(1, 7)} | {f"Figure_E{i:02}" for i in range(1, 5)} | {f"Figure_S{i:02}" for i in range(1, 8)}
        if not selected <= allowed:
            parser.error("Unknown figure ID")
    execute_notebook(ROOT, output, selected)
    crop_publication_pngs(output, selected)
    if selected is not None:
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/analysis/figure_tables.py"),
            "--from-index",
            "--publication-root",
            str(output),
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
                "supplementary_figures": 7,
                "tables": 12,
                "EnergyPlus_runs": 0,
                "input_root": "results/figure_data",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
