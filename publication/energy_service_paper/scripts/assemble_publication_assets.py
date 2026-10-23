"""Assemble evidence figures in their continuous supplementary reading order."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

SI_OLD_ORDER = (3, 1, 2, 14, 15, 11, 4, 16, 13, 21, 5, 17, 6, 20, 12, 7, 19, 8, 18, 9, 10)
SECTION_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 15: 8, 16: 9, 17: 10, 18: 11, 8: 12, 9: 13, 10: 14, 11: 15, 12: 16, 13: 17, 14: 18}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def references(text, figure_map):
    def mapped(m, mapping):
        return m.group(1) + re.sub(r"S(\d+)", lambda n: "S" + str(mapping[int(n.group(1))]), m.group(2))
    text = re.sub(r"(Supplementary Sections?\s+)((?:S\d+(?:\.\d+)?)(?:\s*(?:[–,]|and)\s*S?\d+(?:\.\d+)?)*)", lambda m: mapped(m, SECTION_MAP), text)
    return re.sub(r"((?:Supplementary )?(?:Figures?|Figs?\.)\s+)((?:S\d+[a-z]?(?:[–,][a-z])?)(?:\s*(?:,|and|–)\s*S\d+[a-z]?(?:[–,][a-z])?)*)", lambda m: mapped(m, figure_map), text)

def assemble(previous: Path, stage: Path):
    for name in ("supplementary_figures", "captions_alt"):
        (stage / name).mkdir(parents=True, exist_ok=True)
    for src in (previous / "figures/figure_01").iterdir():
        if src.is_file():
            dst = stage / "figures/figure_01" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    figure_map = {old: new for new, old in enumerate(SI_OLD_ORDER, 1)}
    for number in range(1, 7):
        directory = stage / "figures" / f"figure_{number:02d}"
        captiondir = stage / "captions_alt" / f"figure_{number:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        captiondir.mkdir(parents=True, exist_ok=True)
        if number == 1:
            text = (previous / "captions_alt/figure_01/caption.txt").read_text(encoding="utf-8")
        else:
            for suffix in ("_word.png", ".png", ".pdf", ".svg"):
                shutil.copyfile(stage / "figures" / f"Figure_{number:02d}{suffix}", directory / f"figure_{number:02d}{suffix}")
            text = (stage / "figures" / f"Figure_{number:02d}_caption.txt").read_text(encoding="utf-8")
        for name in ("caption.txt", "alt_text.txt"):
            (captiondir / name).write_text(references(text, figure_map), encoding="utf-8")
    former_main = {16: 2, 17: 3, 18: 6, 19: 5, 20: 4}
    manifest = []
    for new, old in enumerate(SI_OLD_ORDER, 1):
        if old <= 15:
            base = previous / "supplementary_figures" / f"Figure_S{old:02d}"
            raw_caption = Path(str(base) + "_caption.txt")
        elif old in former_main:
            m = former_main[old]
            base = previous / "figures" / f"figure_{m:02d}" / f"figure_{m:02d}"
            raw_caption = previous / "captions_alt" / f"figure_{m:02d}" / "caption.txt"
        else:
            base = stage / "figures/Supplementary_Figure_S21"
            raw_caption = Path(str(base) + "_caption.txt")
        outputs = []
        for suffix in ("_word.png", ".png", ".pdf", ".svg"):
            src = Path(str(base) + suffix)
            if suffix == ".png" and not src.is_file():
                src = Path(str(base) + "_600dpi.png")
            dst = stage / "supplementary_figures" / f"Figure_S{new:02d}{suffix}"
            shutil.copyfile(src, dst)
            outputs.append({"path": dst.relative_to(stage).as_posix(), "sha256": digest(dst)})
        body = raw_caption.read_text(encoding="utf-8").split("|", 1)[1].strip()
        caption = f"Supplementary Figure S{new} | " + references(body, figure_map)
        (stage / "supplementary_figures" / f"Figure_S{new:02d}_caption.txt").write_text(caption, encoding="utf-8")
        manifest.append({"display_number": new, "source_figure_number": old, "caption_source_sha256": digest(raw_caption), "outputs": outputs})
    (stage / "SUPPLEMENTARY_FIGURE_ORDER.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.previous_root, args.stage_root)
    print("Six main and 21 supporting figures assembled in continuous order")
