"""Plain-text instructions and document aliases for the distributed code."""

from pathlib import Path
import re


DOCUMENT_ALIASES = {
    "README.md": "README.txt",
    "REPRODUCTION_FROM_RAW.md": "REPRODUCTION_FROM_RAW.txt",
    "DATA_AND_SOFTWARE_LICENSES.md": "DATA_AND_SOFTWARE_LICENSES.txt",
    "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.txt",
    "publication/README.md": "publication/README.txt",
}


def _plain_legal_text(text):
    """Keep license wording and URLs while removing presentation markup."""
    text = text.replace("data/source_manifest.csv", "data/metadata/source_manifest.csv")
    for original, alias in DOCUMENT_ALIASES.items():
        text = text.replace(original, alias)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = text.replace("`", "").replace("**", "")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            line = "; ".join(cells)
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def documentation_payloads(source_root, release_id, names=None):
    """Export the maintained instructions and notices as plain text."""
    root = Path(source_root)
    result = {}
    for original, alias in DOCUMENT_ALIASES.items():
        if names is not None and alias not in names:
            continue
        source = root / original
        distributed = root / alias
        if source.is_file():
            result[alias] = _plain_legal_text(source.read_text(encoding="utf-8-sig")).encode("utf-8")
        else:
            result[alias] = distributed.read_bytes()
    return result
