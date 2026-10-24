#!/usr/bin/env python3
"""Build the deterministic, immutable Level-1 analysis archive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from thermal_heterogeneity.release_bundle import build_bundle


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository / "data/metadata" / "analysis_bundle_manifest.csv",
    )
    parser.add_argument(
        "--release-record",
        type=Path,
        default=repository / "data/metadata" / "analysis_bundle_release.json",
    )
    args = parser.parse_args()
    source_date_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    try:
        result = build_bundle(
            args.input,
            args.output,
            args.manifest,
            args.release_record,
            source_date_epoch=source_date_epoch,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
