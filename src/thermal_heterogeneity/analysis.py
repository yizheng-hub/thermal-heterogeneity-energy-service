"""Deterministic orchestration of the complete Level-1 reproduction."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

from .config import load_config, output_root, resolve_asset
from .io import sha256, write_json, write_table
from .level1 import reproduce_all, write_scientific_tables


def _legacy_synthetic_tables(config: dict[str, Any]) -> dict[str, Any]:
    """Retain the small public smoke fixture without weakening the real workflow."""
    import pandas as pd

    from .building.annual import compare_representations as compare_annual
    from .building.centre_spread import decompose_two_by_two
    from .building.dynamic import compare_representations as compare_dynamic
    from .human.hierarchy import archive_scope, nested_variance_decomposition
    from .human.shared_room_validation import threshold_table
    from .implementation.phase_map import classify_decisions
    from .implementation.protected_quantiles import certification_metrics

    def read(asset: str) -> pd.DataFrame:
        return pd.read_csv(resolve_asset(config, asset))
    profiles = read("human_profiles")
    annual, annual_eligibility = compare_annual(
        read("annual_scores"), decision_keys=list(config["science"]["annual_decision_keys"])
    )
    dynamic, dynamic_eligibility = compare_dynamic(
        read("dynamic_scores"), decision_keys=list(config["science"]["dynamic_decision_keys"])
    )
    update = read("local_update")
    return {
        "result1_scope": pd.DataFrame([archive_scope(profiles)]),
        "result1_variance": nested_variance_decomposition(profiles, "reference_center_c"),
        "result2_windows": read("shared_room_windows"),
        "result2_thresholds": threshold_table(
            read("shared_room_windows"),
            tuple(config["science"]["direct_vote_thresholds"]),
        ),
        "result3_centre_effects": decompose_two_by_two(
            read("centre_spread_scores"), keys=["city", "service_fraction"]
        ),
        "result4_spread_effects": annual,
        "result4_interactions": annual_eligibility,
        "result5_dynamic_decisions": dynamic,
        "result5_spread_effects": dynamic_eligibility,
        "result6_phase_map": classify_decisions(read("phase_decisions")),
        "result6_quantile_gate": pd.DataFrame(
            [certification_metrics(read("quantile_certification"))]
        ),
        "result6_local_update": update.groupby(["split", "feedback_n"], sort=True)[
            ["crps", "tail_mae"]
        ].mean().reset_index(),
    }


SCIENTIFIC_DIRECTORIES = ("tables", "ledgers", "figure_data", "si")


def refresh_manifest(root: str | Path, config: dict[str, Any]) -> Path:
    output = Path(root).expanduser().resolve()
    files = []
    for directory in SCIENTIFIC_DIRECTORIES:
        base = output / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".csv", ".json"}:
                files.append(
                    {
                        "path": path.relative_to(output).as_posix(),
                        "sha256": sha256(path),
                    }
                )
    headline = output / "headline_results.json"
    if headline.is_file():
        files.append(
            {
                "path": headline.relative_to(output).as_posix(),
                "sha256": sha256(headline),
            }
        )
    input_files = []
    for asset_id in sorted(config["assets"]):
        path = resolve_asset(config, asset_id)
        input_files.append({"asset_id": asset_id, "sha256": sha256(path)})
    manifest = {
        "schema_version": 2,
        "analysis": "thermal-heterogeneity-level1",
        "seed": int(config["science"]["seed"]),
        "input_files": input_files,
        "scientific_files": sorted(files, key=lambda item: item["path"]),
        "claim_checks": config.get("verification", {}).get("claim_checks", []),
        "contracts": {
            "results": 6,
            "figure_panels": 48,
            "supplementary_tables": 12,
            "energyplus_runs": 0,
        },
    }
    return write_json(output / "manifest.json", manifest)


def reproduce_analysis(
    config_path: str | Path,
    output: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
    workers: int = 1,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be positive")
    config = load_config(config_path, data_root=data_root)
    root = output_root(config, output)
    root.mkdir(parents=True, exist_ok=True)
    if "centre_spread_scores" in config["assets"]:
        tables = _legacy_synthetic_tables(config)
        headline = {}
    else:
        tables, headline = reproduce_all(config)
    write_scientific_tables(tables, root)
    write_json(root / "headline_results.json", headline)
    write_json(
        root / "run_environment.json",
        {
            "workers_requested": workers,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scientific_payload_worker_invariant": True,
            "energyplus_runs": 0,
        },
    )
    return refresh_manifest(root, config)
