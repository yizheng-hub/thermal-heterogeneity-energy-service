#!/usr/bin/env python3
"""Render frozen Nature main Figures 1--3 from panel-data v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_heterogeneity.figures import save_figure_bundle
from thermal_heterogeneity.main_figures_1_3 import (
    ALT_TEXT,
    CAPTIONS,
    FIGURE_BUILDERS,
    close_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figures", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    args = parser.parse_args()
    result = {"status": "PASS", "figures": {}}
    for number in args.figures:
        figure = FIGURE_BUILDERS[number](args.panel_data)
        try:
            directory = args.output_root / f"figure_{number:02d}"
            bundle = save_figure_bundle(figure, directory, f"figure_{number:02d}")
            if number in CAPTIONS:
                (directory / "caption_draft.md").write_text(
                    CAPTIONS[number] + "\n", encoding="utf-8"
                )
                (directory / "alt_text.txt").write_text(
                    ALT_TEXT[number] + "\n", encoding="utf-8"
                )
            result["figures"][f"Fig{number}"] = bundle.audit
        finally:
            close_figure(figure)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
