"""Independent file and claim verification entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .io import sha256


def _claim_value(root: Path, claim: dict[str, object]) -> float:
    table = pd.read_csv(root / str(claim["path"]))
    for column, expected in dict(claim.get("where", {})).items():
        table = table[table[column] == expected]
    reducer = claim["reducer"]
    if reducer == "rows":
        return float(len(table))
    column = str(claim["column"])
    if reducer == "sum":
        return float(table[column].sum())
    if reducer == "value":
        if len(table) != 1:
            raise AssertionError(f"claim {claim['claim_id']} expected one row, found {len(table)}")
        return float(table.iloc[0][column])
    raise ValueError(f"unknown claim reducer: {reducer}")


def verify_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.parent if manifest_path.parent.name == "qa" else manifest_path.parent
    failures = []
    entries = payload.get("scientific_files", payload.get("panels", payload.get("tables", [])))
    for entry in entries:
        candidate = root / entry["path"]
        if not candidate.is_file():
            failures.append(f"missing:{entry['path']}")
        elif sha256(candidate) != entry["sha256"]:
            failures.append(f"hash:{entry['path']}")
    contracts = payload.get("contracts", {})
    relative_paths = {str(entry["path"]) for entry in entries}
    panel_count = len(
        [path for path in relative_paths if path.startswith("figure_data/Fig") and path.endswith(".csv")]
    )
    si_count = len(
        [path for path in relative_paths if path.startswith("si/TableS") and path.endswith(".csv")]
    )
    expected_panels = int(contracts.get("figure_panels", panel_count))
    expected_si = int(contracts.get("supplementary_tables", si_count))
    if panel_count != expected_panels:
        failures.append(f"panel_count:{panel_count}!={expected_panels}")
    if si_count != expected_si:
        failures.append(f"si_count:{si_count}!={expected_si}")
    checked_claims = 0
    for claim in payload.get("claim_checks", []):
        observed = _claim_value(root, claim)
        expected = float(claim["expected"])
        tolerance = float(claim.get("tolerance", 0.0))
        checked_claims += 1
        if abs(observed - expected) > tolerance:
            failures.append(f"claim:{claim['claim_id']}:{observed}!={expected}")
    result = {
        "passed": not failures,
        "checked_files": len(entries),
        "checked_claims": checked_claims,
        "checked_figure_panels": panel_count,
        "checked_supplementary_tables": si_count,
        "failures": failures,
    }
    if failures:
        raise AssertionError("manifest verification failed: " + ", ".join(failures))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    verify_manifest(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
