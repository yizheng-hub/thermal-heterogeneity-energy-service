"""Prepare editable Nature Communications vector artwork within 179 by 260 mm."""

from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]


def prepare(publication):
    publication = Path(publication).resolve()
    if not publication.is_relative_to(ROOT):
        raise ValueError("Publication outputs must remain in the project")
    destination = publication / "nc_artwork"
    destination.mkdir(exist_ok=True)
    records = []
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    for folder, prefix, count in [("main", "Figure_", 6), ("extended", "Figure_E", 4), ("supplementary", "Figure_S", 6)]:
        for number in range(1, count + 1):
            source = publication / folder / f"{prefix}{number:02}"
            stem = {"main": "Figure_", "extended": "Extended_Data_Figure_", "supplementary": "Supplementary_Figure_"}[folder] + str(number)
            reader = PdfReader(source.with_suffix(".pdf"))
            if len(reader.pages) != 1:
                raise ValueError("Each figure must occupy one vector page")
            page = reader.pages[0]
            width, height = float(page.mediabox.width), float(page.mediabox.height)
            scale = min(1.0, 179 / (width * 25.4 / 72), 260 / (height * 25.4 / 72))
            page.scale_by(scale)
            writer = PdfWriter()
            writer.add_page(page)
            with (destination / (stem + ".pdf")).open("wb") as stream:
                writer.write(stream)
            svg = ET.parse(source.with_suffix(".svg"))
            root = svg.getroot()
            # Retain the original viewBox, paths and editable text; scale physical dimensions.
            root.set("width", f"{width * scale:.8f}pt")
            root.set("height", f"{height * scale:.8f}pt")
            svg.write(destination / (stem + ".svg"), encoding="utf-8", xml_declaration=True)
            records.append({
                "figure": stem,
                "width_mm": width * scale * 25.4 / 72,
                "height_mm": height * scale * 25.4 / 72,
                "uniform_scale": scale,
                "editable_text": True,
                "source_pdf_sha256": hashlib.sha256(source.with_suffix(".pdf").read_bytes()).hexdigest(),
                "source_svg_sha256": hashlib.sha256(source.with_suffix(".svg").read_bytes()).hexdigest(),
            })
    (destination / "MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"nc_vector_figures": len(records), "maximum_width_mm": 179, "maximum_height_mm": 260}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, default=ROOT / "output/publication")
    prepare(parser.parse_args().publication_root)
