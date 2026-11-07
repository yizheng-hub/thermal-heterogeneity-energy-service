"""Order supplementary evidence by its first citation in the main manuscript."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import tempfile
from thermal_heterogeneity.figures import clarify_control_caption

OLD_TO_NEW = {1: 16, 2: 2, 3: 1, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
              9: 17, 10: 10, 11: 8, 12: 9, 13: 20, 14: 21, 15: 22,
              16: 23, 17: 12, 18: 13, 19: 14, 20: 15, 21: 24}
TABLE_ORDER = (2, 3, 4, 5, 10, 11, 17, 12, 1, 6, 7, 13, 18, 8, 9, 14, 15, 19, 16)
TABLE_MAP = {old: new for new, old in enumerate(TABLE_ORDER, 1)}
SECTION_MAP = {'1': '7.1', '2': '7.2', '3': '1.1', '4': '1.2', '5': '1.3',
               '6': '1.4', '7': '1.5', '8': '2', '8.1': '2.1', '8.2': '2.2',
               '8.3': '3.1', '8.4': '9', '9': '4', '10': '5', '11': '6.1',
               '12': '8.1', '12.1': '8.2', '12.2': '12.1', '13': '10',
               '13.1': '10.1', '13.2': '3.2', '13.3': '10.2', '13.4': '10.3',
               '13.5': '10.4', '14': '11', '14.1': '12.2', '15': '6.2',
               '15.1': '6.2.1', '15.2': '6.2.2', '16': '6.3', '17': '13',
               '17.1': '13.1', '18': '14'}
SECTION_MAP.update({f'{n}.{k}': f'{SECTION_MAP[str(n)]}.{k}'
                    for n in (9, 10, 11) for k in (1, 2, 3)})
SUFFIXES = ("_word.png", ".png", ".pdf", ".svg")


def figure_references(text):
    pattern = r"((?:Supplementary )?(?:Figures?|Figs?\.)\s+)((?:S\d+[a-z]?(?:[–,][a-z])?)(?:\s*(?:,|and|–)\s*S\d+[a-z]?(?:[–,][a-z])?)*)"
    text = re.sub(pattern, lambda m: m.group(1) + re.sub(
        r"S(\d+)", lambda n: "S" + str(OLD_TO_NEW[int(n.group(1))]), m.group(2)), text)
    for label, mapping in [('Sections?', SECTION_MAP), ('Tables?', TABLE_MAP)]:
        pattern = rf'(Supplementary {label}\s+)(S\d+(?:\.\d+)*(?:\s*(?:,|and|–)\s*S\d+(?:\.\d+)*)*)'
        def mapped(match):
            return match[1] + re.sub(r'S(\d+(?:\.\d+)*)',
                lambda n: 'S' + str(mapping[n[1] if label.startswith('Section') else int(n[1])]), match[2])
        text = re.sub(pattern, mapped, text)
    return text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_bundle(source, target):
    for suffix in SUFFIXES:
        src = Path(str(source) + suffix)
        if suffix == "_word.png" and not src.is_file():
            src = Path(str(source) + ".png")
        shutil.copyfile(src, Path(str(target) + suffix))


def finalize(paper, seasonal, typical):
    order_path = paper / "SUPPLEMENTARY_FIGURE_ORDER.json"
    order = json.loads(order_path.read_text(encoding="utf-8"))
    if len(order) != 21:
        raise ValueError("Finalization requires the freshly assembled 21-figure base")
    folder = paper / "supplementary_figures"
    with tempfile.TemporaryDirectory(prefix="temporal_figures_") as temporary:
        retained = Path(temporary)
        for item in order:
            old = item["display_number"]
            copy_bundle(folder / f"Figure_S{old:02d}", retained / f"Figure_S{old:02d}")
            shutil.copyfile(folder / f"Figure_S{old:02d}_caption.txt", retained / f"Figure_S{old:02d}_caption.txt")
        for item in order:
            old, new = item["display_number"], OLD_TO_NEW[item["display_number"]]
            copy_bundle(retained / f"Figure_S{old:02d}", folder / f"Figure_S{new:02d}")
            caption = (retained / f"Figure_S{old:02d}_caption.txt").read_text(encoding="utf-8")
            body = figure_references(caption.split("|", 1)[1].strip())
            if new == 21:
                body += (" Panel f reports carrier changes in MWh/year; panel g reports "
                         "interactions in the count of service-qualified actions.")
            (folder / f"Figure_S{new:02d}_caption.txt").write_text(f"Supplementary Figure S{new} | {body}\n", encoding="utf-8")
            item["display_number"] = new
    additions = [(18, "seasonal_direct", seasonal / "Supplementary_monthly_direct"),
                 (19, "seasonal_common", seasonal / "Supplementary_monthly_common_service"),
                 (11, "typical_day", typical / "TYPICAL_DAY_DR")]
    for number, key, stem in additions:
        copy_bundle(stem, folder / f"Figure_S{number:02d}")
        raw = Path(str(stem) + "_caption.txt")
        body = clarify_control_caption(raw.read_text(encoding="utf-8").split("|", 1)[-1].strip(),
                                       uses_three_controls=True)
        (folder / f"Figure_S{number:02d}_caption.txt").write_text(f"Supplementary Figure S{number} | {body}\n", encoding="utf-8")
        order.append({"display_number": number, "source_figure_number": key,
                      "caption_source_sha256": digest(raw)})
    for item in order:
        n = item["display_number"]
        item["outputs"] = [{"path": f"supplementary_figures/Figure_S{n:02d}{suffix}",
                            "sha256": digest(folder / f"Figure_S{n:02d}{suffix}")} for suffix in SUFFIXES]
    order.sort(key=lambda item: item["display_number"])
    if [item["display_number"] for item in order] != list(range(1, 25)):
        raise ValueError("Supplementary figures are not a continuous S1-S24 series")
    order_path.write_text(json.dumps(order, indent=2) + "\n", encoding="utf-8")
    (paper / 'SUPPLEMENTARY_READING_ORDER.json').write_text(json.dumps({
        'main_first_cited_figures': list(range(1, 17)),
        'figures_cited_only_inside_supplement': list(range(17, 25)),
        'table_aliases_old_to_new': TABLE_MAP,
        'section_aliases_old_to_new': SECTION_MAP,
        'identity_panels': 'Historical S12a,b are now in displayed S1e,f',
        'annual_severity_panels': 'Historical S12c,d are now displayed S22a,b',
    }, indent=2) + '\n', encoding='utf-8')
    for number in range(1, 7):
        for name in ("caption.txt", "alt_text.txt"):
            path = paper / f"captions_alt/figure_{number:02d}" / name
            path.write_text(figure_references(path.read_text(encoding="utf-8")), encoding="utf-8")
    extended = seasonal / "Figure_03_extended"
    copy_bundle(extended, paper / "figures/figure_03/figure_03")
    copy_bundle(extended, paper / "figures/Figure_03")
    caption = clarify_control_caption(Path(str(extended) + "_caption.txt").read_text(encoding="utf-8"),
                                      uses_three_controls=True)
    for name in ("caption.txt", "alt_text.txt"):
        (paper / "captions_alt/figure_03" / name).write_text(caption, encoding="utf-8")
    (paper / "figures/Figure_03_caption.txt").write_text(caption, encoding="utf-8")
    for path in seasonal.glob("Figure_03*.csv"):
        shutil.copyfile(path, paper / "figures" / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--seasonal-figure-root", type=Path, required=True)
    parser.add_argument("--typical-figure-root", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.paper_root, args.seasonal_figure_root, args.typical_figure_root)
    print("Six main and 24 supplementary figures assembled")
