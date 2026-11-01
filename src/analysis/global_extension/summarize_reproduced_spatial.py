"""Assemble the original spatial comparison table from a contained raw replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from final_story_summary import spatial_effects, summarise_spatial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    raw, output = args.raw_run_root.resolve(), args.output_root.resolve()
    if not all(path.is_relative_to(root) for path in (raw, output)):
        raise ValueError("Spatial summary inputs and outputs must remain inside this project")
    source = raw / "results/spatial_final"
    paths = [source / name for name in (
        "GLOBAL_PERMISSION_COMPARISONS.csv", "GLOBAL_PLACEMENT_COMPARISONS.csv",
        "GLOBAL_SPATIAL_AUDIT.json",
    )]
    effects = spatial_effects(pd.read_csv(paths[0]), pd.read_csv(paths[1]))
    summary = summarise_spatial(effects, json.loads(paths[2].read_text()))
    output.mkdir(parents=True, exist_ok=True)
    table = output / "FINAL_SPATIAL_EFFECTS.csv"
    effects.to_csv(table, index=False, float_format="%.12g")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    audit = {
        "status": "completed",
        "scope": "Original spatial design reproduction; strengthened placement and HVAC results are separate",
        "summary": summary,
        "input_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
        "output_sha256": {table.name: sha(table)},
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in (
            Path(__file__).resolve(), Path(__file__).with_name("final_story_summary.py"),
        )},
    }
    (output / "REPRODUCED_SPATIAL_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": audit["status"], "rows": len(effects)}))


if __name__ == "__main__":
    main()
