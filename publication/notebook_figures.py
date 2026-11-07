"""One complete publication figure per notebook cell; inputs come only from Data_for_Figures."""

from pathlib import Path
import re
import shutil
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plot"))
import reference_figures  # noqa: E402 - initialize local imports before loading project modules
import supplementary_figures  # noqa: E402 - initialize local imports before loading project modules
import extended_figures  # noqa: E402


def draw(number, supplementary=False, extended=False, output=None):
    destination = Path(output) if output is not None else ROOT / "output/publication"
    if extended:
        extended_figures.draw(number, destination / "extended")
    elif not supplementary:
        reference_figures.draw(number, data_root=ROOT / "Data_for_Figures", output=destination / "main", strict_raw=True)
    else:
        supplementary_figures.main([
            "--figure", str(extended_figures.REMAINING_SI[number]),
            "--published-number", str(number),
            "--raw-root", str(ROOT / "Data_for_Figures/output/from_raw"),
            "--analysis-root", str(ROOT / "Data_for_Figures/output/strengthening"),
            "--output-root", str(destination / "supplementary"),
        ])
    folder = "extended" if extended else ("supplementary" if supplementary else "main")
    key = f"E{number}" if extended else (f"S{number}" if supplementary else str(number))
    name = f"Figure_{key[0]}{number:02}.png" if supplementary or extended else f"Figure_{number:02}.png"
    # Keep the author's viewing/export folders current only for the approved destination.
    if destination.resolve() == (ROOT / "output/publication").resolve():
        viewing = ROOT / "plot" / folder
        viewing.mkdir(exist_ok=True)
        source = destination / folder / name
        for extension in [".png", ".pdf", ".svg", ".json"]:
            shutil.copy2(source.with_suffix(extension), viewing / source.with_suffix(extension).name)
        shutil.copy2(source, viewing / (source.stem + "_word.png"))
        with Image.open(source) as image:
            image.save(viewing / (source.stem + ".tiff"), compression="tiff_lzw", dpi=(300, 300))
        document = "Supplementary_information" if supplementary else "Main_manuscript"
        manuscript = ROOT / "publication" / (document + ".md")
        if manuscript.is_file():
            text = manuscript.read_text(encoding="utf-8")
            caption = re.search(r"\{\{caption:" + key + r"\|(.+?)\}\}", text).group(1)
            (viewing / (source.stem + "_caption.txt")).write_text(caption + "\n", encoding="utf-8")
    return destination / folder / name
