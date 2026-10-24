#!/usr/bin/env python3
"""Fetch a locked release data tier and fail closed on any hash mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermal_heterogeneity.release_bundle import fetch_bundle


def main() -> int:
    repository = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True, choices=["analysis"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--url", help="Explicit HTTPS, file URL or local path override")
    parser.add_argument(
        "--bundle-record",
        type=Path,
        default=repository / "data/metadata" / "analysis_bundle_release.json",
    )
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=repository / "data/metadata" / "analysis_bundle_manifest.csv",
    )
    args = parser.parse_args()
    try:
        result = fetch_bundle(
            tier=args.tier,
            source_manifest_path=args.manifest,
            bundle_record_path=args.bundle_record,
            analysis_manifest_path=args.analysis_manifest,
            destination=args.dest,
            locator_override=args.url,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
