"""Supplementary table generation from regenerated Result tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import load_config, output_root
from .io import sha256, write_json, write_table


def reproduce_si(
    config_path: str | Path,
    analysis: str | Path | None = None,
    output: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    root = output_root(config, analysis)
    destination_root = root if output is None else Path(output).expanduser().resolve()
    destination = destination_root / "si"
    files = []
    for table in sorted(config["supplementary_tables"], key=lambda item: item["table_id"]):
        source = root / "tables" / table["source_table"]
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"empty SI source: {table['table_id']}")
        target = write_table(frame, destination / f"{table['table_id']}.csv", sort_by=list(frame.columns))
        files.append({"table_id": table["table_id"], "path": target.relative_to(root).as_posix(), "sha256": sha256(target), "rows": len(frame)})
    return write_json(root / "qa" / "si_manifest.json", {"tables": files})
