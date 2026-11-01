"""One complete publication figure per notebook cell; inputs come only from Result."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plot"))
import final_figures  # noqa: E402 - initialize local imports before loading project modules
import supplementary_figures  # noqa: E402 - initialize local imports before loading project modules


def draw(number, supplementary=False, output=None):
    destination = Path(output) if output is not None else ROOT / "output/publication"
    folder = "supplementary" if supplementary else "main"
    arguments = [
        "--figure",
        str(number),
        "--raw-root",
        str(ROOT / "Result/output/from_raw"),
        "--analysis-root",
        str(ROOT / "Result/output/strengthening"),
        "--output-root",
        str(destination / folder),
    ]
    (supplementary_figures.main if supplementary else final_figures.main)(arguments)
    name = f"Figure_S{number:02}.png" if supplementary else f"Figure_{number:02}.png"
    return destination / folder / name
