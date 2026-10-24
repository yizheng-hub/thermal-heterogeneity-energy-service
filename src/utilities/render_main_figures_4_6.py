#!/usr/bin/env python3
"""Render frozen main Figures 4--6 from approved panel-data v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_heterogeneity.figures_4_6 import render_main_figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figures", type=int, nargs="+", default=[4, 5, 6], choices=[4, 5, 6])
    args = parser.parse_args()
    results = {}
    for number in args.figures:
        output = args.output_root / f"figure_{number:02d}"
        bundle = render_main_figure(number, args.panel_data, output)
        results[str(number)] = {
            "qa": str(bundle.qa),
            "files": {name: record["sha256"] for name, record in bundle.audit["files"].items()},
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
