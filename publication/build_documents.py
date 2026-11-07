"""Render the publication sources into the author's retained Word template."""

from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile
import importlib.util
import sys

if importlib.util.find_spec("docx") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/document_libraries"))
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION_START
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 8306 / 1440
TEMPLATE_SHA = "a216cd506f5c05a08e932b84fe3c0f4ba905212a52295b4383791ac61184f1b6"
CITATION = re.compile(r"\[(\d+(?:(?:,|–|-)\s*\d+)*)\]")
REFERENCE_NUMBER = r"\(?[SE]?\d+(?:\.\d+)*[a-z]?\)?"
CROSS_REFERENCE = re.compile(
    r"\b(?:(?:Extended Data|Supplementary)\s+)?(?:"
    r"(?:Figs?\.|Figures?)\s+" + REFERENCE_NUMBER
    + r"(?:\s*(?:[–-]|,|\band\b)\s*(?:" + REFERENCE_NUMBER + r"|[a-z])(?![a-z]))*"
    + r"|(?:Tables?|Sections?|Notes?|Equations?|Data|Software)\s+" + REFERENCE_NUMBER
    + r"(?:\s*(?:[–-]|,|\band\b)\s*" + REFERENCE_NUMBER + r")*)"
)


def citation_ids(value):
    result = []
    for piece in value.split(","):
        bounds = re.split("[–-]", piece.strip())
        result.extend(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def typography(p, size=12, bold=False, first=True):
    f = p.paragraph_format
    f.line_spacing = 1.5
    # Preserve the author's spacing independently of the inherited line grid.
    free_caption_spacing(p)
    f.space_before = Pt(7.8)
    f.space_after = Pt(7.8)
    f.first_line_indent = Pt(24 if first else 0)
    f.widow_control = True
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for run in p.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0, 0, 0)
        if bold:
            run.bold = True
    return p


def inline(p, text, mapping, size=12, italic=False):
    pattern = r"(\*\*.*?\*\*|\*[^*]+\*|\[\d+(?:(?:,|–|-)\s*\d+)*\]|" + CROSS_REFERENCE.pattern + r")"
    for token in re.split(pattern, text):
        if not token:
            continue
        if token.startswith("**"):
            run = p.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*") and token.endswith("*"):
            run = p.add_run(token[1:-1])
            run.italic = True
        elif CITATION.fullmatch(token):
            nums = [mapping[n] for n in citation_ids(token[1:-1])]
            if p.runs:
                p.runs[-1].text = p.runs[-1].text.rstrip()
            run = p.add_run(",".join(str(n) for n in nums))
            run.font.superscript = True
        else:
            run = p.add_run(token)
            if CROSS_REFERENCE.fullmatch(token):
                run.bold = True
        if italic:
            run.italic = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0, 0, 0)


def free_caption_spacing(p):
    """Keep explicit line spacing independent of the inherited template grid."""
    grid = p._p.get_or_add_pPr().find(qn("w:snapToGrid"))
    if grid is None:
        grid = OxmlElement("w:snapToGrid")
        p._p.get_or_add_pPr().append(grid)
    grid.set(qn("w:val"), "0")


def heading(doc, text, level=2):
    style = next((s for s in doc.styles if s.style_id == str(level)), None)
    p = doc.add_paragraph(style=style)
    p.add_run(text)
    typography(p, 16 if level == 1 else 14 if level == 2 else 12, True, False)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    p.paragraph_format.line_spacing = 1.5
    if level == 1:
        p.paragraph_format.space_before = Pt(17)
        p.paragraph_format.space_after = Pt(16.5)
    return p


def native_math(doc, key, number):
    def run(text, roman=False):
        node = OxmlElement("m:r")
        math_prop = OxmlElement("m:rPr")
        style = OxmlElement("m:sty")
        style.set(qn("m:val"), "p" if roman or not any(c.isalpha() for c in text) or text in ["Σ", "Δ"] else "i")
        math_prop.append(style)
        node.append(math_prop)
        prop = OxmlElement("w:rPr")
        size = OxmlElement("w:sz")
        size.set(qn("w:val"), "24")
        prop.append(size)
        node.append(prop)
        t = OxmlElement("m:t")
        t.text = text
        node.append(t)
        return node

    def sub(base, index):
        node = OxmlElement("m:sSub")
        for tag, text in [("m:e", base), ("m:sub", index)]:
            part = OxmlElement(tag)
            if tag == "m:sub" and text in ["baseline,w", "action,w"]:
                label, variable = text.split(",")
                part.append(run(label + ",", True))
                part.append(run(variable))
            else:
                part.append(run(text, tag == "m:sub" and text in ["pop", "total", "facility,electric", "facility,gas,HHV", "baseline,event", "action,event", "event", "net"]))
            node.append(part)
        return node

    def fraction(top, bottom):
        node = OxmlElement("m:f")
        for tag, items in [("m:num", top), ("m:den", bottom)]:
            part = OxmlElement(tag)
            for item in items:
                part.append(item)
            node.append(part)
        return node

    def weighted_error():
        return [
            sub("Σ", "h"),
            run(" "),
            sub("w", "ih"),
            run(" |"),
            sub("T", "z(i),h"),
            run(" − "),
            sub("θ", "i"),
            run("|"),
        ]

    if key == "center":
        items = [
            sub("θ", "i"),
            run(" = "),
            sub("T", "i"),
            run(" − "),
            fraction([sub("S", "i")], [run("g")]),
        ]
    elif key == "population":
        items = [
            sub("M", "pop"),
            run(" = "),
            fraction(
                [sub("Σ", "i"), run(" ")] + weighted_error(),
                [sub("Σ", "i"), run(" "), sub("Σ", "h"), run(" "), sub("w", "ih")],
            ),
        ]
    elif key == "tail":
        items = [
            sub("M", "F"),
            run(" = "),
            fraction([run("1")], [run("|F|")]),
            run(" "),
            sub("Σ", "i∈F"),
            run(" "),
            fraction(weighted_error(), [sub("Σ", "h"), run(" "), sub("w", "ih")]),
        ]
    elif key == "energy":
        items = [
            sub("E", "total"),
            run(" = "),
            sub("E", "facility,electric"),
            run(" + "),
            sub("E", "facility,gas,HHV"),
        ]
    elif key == "power":
        items = [
            sub("P", "event"),
            run(" = "),
            fraction(
                [sub("E", "baseline,event"), run(" − "), sub("E", "action,event")],
                [run("Δ"), sub("t", "event")],
            ),
        ]
    elif key == "net":
        items = [
            run("Δ"),
            sub("E", "net"),
            run(" = "),
            sub("Σ", "w"),
            run(" ("),
            sub("E", "baseline,w"),
            run(" − "),
            sub("E", "action,w"),
            run(")"),
        ]
    else:
        raise ValueError(key)
    p = doc.add_paragraph()
    typography(p, 12, False, False)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together = True
    math = OxmlElement("m:oMath")
    for item in items:
        math.append(item)
    p._p.append(math)
    p.add_run(f"    ({number})")
    typography(p, 12, False, False)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def three_line_table(t):
    """Use 1.5 pt outside rules and a 0.5 pt header rule, with no other lines."""
    properties = t._tbl.tblPr
    for old in list(properties.findall(qn("w:tblBorders"))):
        properties.remove(old)
    borders = OxmlElement("w:tblBorders")
    for name in ["top", "bottom", "left", "right", "insideH", "insideV"]:
        edge = OxmlElement("w:" + name)
        edge.set(qn("w:val"), "single" if name in ["top", "bottom"] else "nil")
        edge.set(qn("w:sz"), "12" if name in ["top", "bottom"] else "0")
        edge.set(qn("w:color"), "000000")
        borders.append(edge)
    properties.append(borders)
    for i, row in enumerate(t.rows):
        for cell in row.cells:
            prop = cell._tc.get_or_add_tcPr()
            for old in list(prop.findall(qn("w:tcBorders"))):
                prop.remove(old)
            if i == 0:
                border = OxmlElement("w:tcBorders")
                bottom = OxmlElement("w:bottom")
                bottom.set(qn("w:val"), "single")
                bottom.set(qn("w:sz"), "4")
                bottom.set(qn("w:color"), "000000")
                border.append(bottom)
                prop.append(border)


def table(doc, spec, mapping):
    p = doc.add_paragraph()
    typography(p, 11, False, False)
    p.add_run(f"Supplementary Table {spec['number']} | {spec['title']}.").bold = True
    typography(p, 11, False, False)
    p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(spec["columns"]))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # Allocate width to readable identifiers as well as multi-line metric headings.
    table_weights = {
        1: [1.55, 1.5, 0.95, 0.95],
        2: [1.1, 1.1, 0.8, 1.3, 0.75, 0.75],
        3: [0.85, 1.0, 1.3, 1.3, 1.3],
        4: [1.0, 0.8, 1.15, 1.15, 1.7],
        5: [1.05, 0.8, 1.0, 1.25, 1.45],
        6: [0.9, 1.05, 0.5, 1.1, 1.15, 1.05],
        7: [1.0, 0.65, 0.9, 1.25, 1.6],
        8: [1.5, 0.7, 0.8, 1.0, 1.0, 1.0],
        9: [1.2, 0.85, 1.1, 1.1, 1.1],
        10: [1.0, 0.9, 3.9],
    }
    lengths = table_weights[spec["number"]]
    if len(lengths) != len(spec["columns"]):
        raise ValueError("Table width specification does not match its columns")
    widths = [WIDTH * v / sum(lengths) for v in lengths]
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    for j, label in enumerate(spec["columns"]):
        t.rows[0].cells[j].text = label
    repeat = OxmlElement("w:tblHeader")
    t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for row in spec["rows"]:
        for cell, text in zip(t.add_row().cells, row):
            cell.text = str(text)
    for i, row in enumerate(t.rows):
        no_split = OxmlElement("w:cantSplit")
        row._tr.get_or_add_trPr().append(no_split)
        for j, cell in enumerate(row.cells):
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cell.paragraphs:
                typography(p, 10 if spec["number"] == 2 else 11, i == 0, False)
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.space_before = Pt(2 if spec["number"] == 2 else 3)
                p.paragraph_format.space_after = Pt(2 if spec["number"] == 2 else 3)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or spec["number"] == 10 else WD_ALIGN_PARAGRAPH.CENTER
                # Keep short tables together; long validation tables repeat their header.
                p.paragraph_format.keep_with_next = (
                    i == 0 or i == len(t.rows) - 1
                    if spec["number"] == 10
                    else len(spec["rows"]) <= 12
                )
    three_line_table(t)
    p = doc.add_paragraph()
    inline(p, spec["note"], mapping, 10 if spec["number"] == 2 else 11)
    typography(p, 10 if spec["number"] == 2 else 11, False, False)
    p.paragraph_format.line_spacing = 1
    p.paragraph_format.keep_together = True


def references(doc, source, mapping):
    original = {}
    for p in source.paragraphs:
        match = re.match(r"^(\d+)\.\s+", p.text)
        if match:
            original[int(match.group(1))] = p
    for old, new in sorted(mapping.items(), key=lambda pair: pair[1]):
        if old == 65:
            p = doc.add_paragraph(
                f"{new}. Meimand, M. & Jazizadeh, F. A personal touch to demand response: An occupant-centric control strategy for HVAC systems using personalized comfort models. Energy and Buildings 303, 113769 (2024). https://doi.org/10.1016/j.enbuild.2023.113769"
            )
        elif old == 66:
            p = doc.add_paragraph(
                f"{new}. Staffell, I., Pfenninger, S. & Johnson, N. A global model of hourly space heating and cooling demand at multiple spatial scales. Nature Energy 8, 1328–1344 (2023). https://doi.org/10.1038/s41560-023-01341-5"
            )
        else:
            if old not in original:
                raise ValueError(f"Unknown verified reference {old}")
            element = deepcopy(original[old]._p)
            for field in list(element.xpath(".//w:instrText | .//w:fldChar")):
                field.getparent().remove(field)
            remove = len(re.match(r"^\d+\.\s+", original[old].text).group())
            for node in element.xpath(".//w:t"):
                n = min(remove, len(node.text or ""))
                node.text = (node.text or "")[n:]
                remove -= n
                if not remove:
                    break
            doc._body._body.insert(-1, element)
            p = doc.paragraphs[-1]
            r = OxmlElement("w:r")
            tx = OxmlElement("w:t")
            tx.set(qn("xml:space"), "preserve")
            tx.text = f"{new}. "
            r.append(tx)
            p._p.insert(1 if p._p.pPr is not None else 0, r)
        text = p.text
        if " et al. " in text:
            title = text.split(" et al. ", 1)[1]
            first = re.match(r"^(\d+\. .+?, (?:[A-ZÀ-Ž]\.\s*){1,4})", text)
            if first is None:
                raise ValueError("Cannot identify first author in reference")
            text = first.group(1).rstrip(" ,") + " et al. " + title
        text = text.replace("Wang, R., Xin, X., Liu, J., Lu, S., Sun, Y. & Zhao, W.", "Wang, R. et al.")
        text = text.replace("User Placement and Optimal Cooling Energy for Co-working Building Spaces", "User placement and optimal cooling energy for co-working building spaces")
        text = text.replace("Investigating the Impact of Space Allocation Strategy on Energy-Comfort Trade-off in Office Buildings", "Investigating the impact of space allocation strategy on energy-comfort trade-off in office buildings")
        if text != p.text:
            p.clear()
            p.add_run(text)
        typography(p, 11, False, False)
        p.paragraph_format.line_spacing = 1
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)


def build(source_name, publication, output):
    template = ROOT / "data/templates/Manuscript.docx"
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA:
        raise ValueError("Author template changed")
    source = Document(ROOT / "data/templates/Original_Main_manuscript.docx")
    text = (ROOT / "publication" / source_name).read_text(encoding="utf-8")
    mapping = {}
    for match in CITATION.finditer(text):
        for n in citation_ids(match.group(1)):
            if n not in mapping:
                mapping[n] = len(mapping) + 1
    tables = {
        str(t["number"]): t
        for t in json.loads((publication / "tables/TABLES.json").read_text(encoding="utf-8"))[
            "tables"
        ]
    }
    doc = Document(template)
    for child in list(doc._element.body):
        if child.tag != qn("w:sectPr"):
            doc._element.body.remove(child)
    # Submission copies retain the author's page geometry and fonts, with Arabic page numbers.
    for section in doc.sections:
        if source_name.startswith("Supplementary"):
            numbering = section._sectPr.find(qn("w:lnNumType"))
            if numbering is not None:
                section._sectPr.remove(numbering)
        footer = section.footer.paragraphs[0]
        footer.clear()
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer.paragraph_format.first_line_indent = Pt(0)
        footer.paragraph_format.line_spacing = 1
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE \\* ARABIC")
        page_run = OxmlElement("w:r")
        page_style = OxmlElement("w:rPr")
        font = OxmlElement("w:rFonts")
        font.set(qn("w:ascii"), "Times New Roman")
        font.set(qn("w:hAnsi"), "Times New Roman")
        page_style.append(font)
        page_size = OxmlElement("w:sz")
        page_size.set(qn("w:val"), "22")
        page_style.append(page_size)
        page_run.append(page_style)
        page_text = OxmlElement("w:t")
        page_text.text = "1"
        page_run.append(page_text)
        field.append(page_run)
        footer._p.append(field)
    lines = text.splitlines()
    if source_name == "Main_manuscript.md":
        plates = [line for line in lines if re.match(r"\{\{(?:figure|caption):[1-6](?:\}\}|\|)", line)]
        lines = [line for line in lines if line not in plates]
        insertion = lines.index("## Extended Data")
        lines[insertion:insertion] = ["## Main figures", ""] + plates + [""]
    equations = 0
    figures = []
    table_numbers = []
    wide_artwork = False
    paper_title = (
        (ROOT / "publication/Main_manuscript.md")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .removeprefix("# ")
    )
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if source_name == "Main_manuscript.md" and line == "## Introduction":
            doc.add_page_break()
            continue
        if line in ["## Main figures", "## Extended Data"]:
            if not wide_artwork:
                section = doc.add_section(WD_SECTION_START.NEW_PAGE)
                section.left_margin = section.right_margin = Mm(15.5)
                section.top_margin = section.bottom_margin = Mm(12)
                section.header_distance = section.footer_distance = Mm(6)
                numbering = section._sectPr.find(qn("w:lnNumType"))
                if numbering is not None:
                    section._sectPr.remove(numbering)
                wide_artwork = True
            continue
        if source_name.startswith("Supplementary") and line in [paper_title, "Supplementary Information"]:
            continue
        if line.startswith("# "):
            heading(doc, line[2:], 1)
            if source_name.startswith("Supplementary"):
                heading(doc, "Supplementary Information", 2)
            for n in [1, 3, 4, 5, 6, 7, 9]:
                p = doc.add_paragraph()
                if 3 <= n <= 7:
                    r = p.add_run(str(n - 2) + " ")
                    r.font.superscript = True
                for r in source.paragraphs[n].runs:
                    text_value = r.text.translate(str.maketrans("abcde", "12345")) if n == 1 and r.font.superscript else r.text
                    new = p.add_run(text_value)
                    if r._r.rPr is not None:
                        new._r.insert(0, deepcopy(r._r.rPr))
                typography(p, 12, False, False)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(3)
                p.paragraph_format.line_spacing = 1
            continue
        if line.startswith("### "):
            heading(doc, line[4:], 3)
            continue
        if line.startswith("## "):
            if line == "## Acknowledgements" and "## Author contributions" not in text:
                heading(doc, "Author contributions", 2)
                p = doc.add_paragraph(
                    source.paragraphs[11].text.removeprefix("Author contributions: ")
                )
                typography(p)
            heading(doc, line[3:], 2)
            continue
        if line == "{{references}}":
            references(doc, source, mapping)
            continue
        if line.startswith("{{equation:"):
            equations += 1
            native_math(doc, line[11:-2], equations)
            continue
        if line.startswith("{{table:S"):
            n = line[9:-2]
            table(doc, tables[n], mapping)
            table_numbers.append(n)
            continue
        if line.startswith("{{figure:"):
            n = line[9:-2]
            suffix = "extended" if n.startswith("E") else ("supplementary" if n.startswith("S") else "main")
            filename = (
                f"Figure_{n[0]}{int(n[1:]):02}.png" if n.startswith(("S", "E")) else f"Figure_{int(n):02}.png"
            )
            path = publication / suffix / filename
            receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if receipt.get("text_collisions", []) or receipt.get("clipped_text", []) or receipt.get("canvas_overflow_text", []):
                raise ValueError(f"Unresolved figure layout {n}")
            p = doc.add_paragraph()
            typography(p, 11, False, False)
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if wide_artwork:
                p.paragraph_format.page_break_before = n != "1"
            picture = p.add_run().add_picture(str(path), width=Mm(179) if wide_artwork else Inches(WIDTH))
            # Reserve room for the retained 11 pt captions on each A4 plate.
            maximum_height = {"4": 200, "6": 200, "E1": 195, "E2": 185, "E3": 200, "E4": 180}.get(n, 210) / 25.4
            if suffix in ["main", "extended"] and picture.height > Inches(maximum_height):
                picture.width = round(picture.width * Inches(maximum_height) / picture.height)
                picture.height = Inches(maximum_height)
            figures.append(n)
            continue
        if line.startswith("{{caption:"):
            n, caption = line[10:-2].split("|", 1)
            title_text, description = caption.split(". ", 1)
            p = doc.add_paragraph()
            caption_size = 11
            typography(p, caption_size, False, False)
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.keep_together = True
            free_caption_spacing(p)
            name = "Extended Data Fig. " if n.startswith("E") else ("Fig. S" if n.startswith("S") else "Fig. ")
            run = p.add_run(name + n.lstrip("SE") + " | " + title_text + ". ")
            run.bold = True
            inline(p, description, mapping, caption_size, True)
            typography(p, caption_size, False, False)
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            continue
        p = doc.add_paragraph()
        inline(p, line, mapping)
        typography(p)
    # Retain the template typography while avoiding isolated spillover pages.
    supplementary_references = False
    for p in doc.paragraphs:
        if source_name == "Main_manuscript.md" and p.text == "Acknowledgements":
            p.paragraph_format.page_break_before = True
        if source_name == "Main_manuscript.md" and p.text == "Introduction":
            p.paragraph_format.page_break_before = True
        if p.text == "Supplementary References":
            supplementary_references = True
        elif supplementary_references:
            p.paragraph_format.space_after = Pt(1.5)
    output.mkdir(parents=True, exist_ok=True)
    target = output / (Path(source_name).stem + ".docx")
    # The template body was replaced; remove its now-unreferenced figure assets.
    used_images = set(doc._element.xpath(".//@r:embed | .//@r:link"))
    for rid, relation in list(doc.part.rels.items()):
        if relation.reltype == RT.IMAGE and rid not in used_images:
            doc.part.drop_rel(rid)
    doc.save(target)
    # Restore preserve-only package parts byte-for-byte after the body edit.
    with zipfile.ZipFile(template) as z:
        preserved = {
            n: z.read(n)
            for n in z.namelist()
            if n
            in ["word/styles.xml", "word/numbering.xml", "word/fontTable.xml", "word/settings.xml"]
            or n.startswith("word/theme/")
        }
    with zipfile.ZipFile(target) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    parts.update(preserved)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, blob in parts.items():
            z.writestr(name, blob)
    audit = {
        "status": "built_pending_page_review",
        "source": source_name,
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "template_sha256": TEMPLATE_SHA,
        "figure_numbers": figures,
        "table_numbers": table_numbers,
        "native_equations": equations,
        "reference_mapping": mapping,
        "preserved_parts": list(preserved),
        "docx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }
    target.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                "document": str(target),
                "figures": len(figures),
                "tables": len(table_numbers),
                "references": len(mapping),
                "status": audit["status"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, default=ROOT / "output/publication")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/publication/documents")
    args = parser.parse_args()
    if not all(p.resolve().is_relative_to(ROOT) for p in [args.publication_root, args.output_root]):
        raise ValueError("Publication inputs and outputs must remain inside the project")
    for filename in ["Main_manuscript.md", "Supplementary_information.md"]:
        build(filename, args.publication_root, args.output_root)
    from prepare_nc_artwork import prepare

    prepare(args.publication_root)
