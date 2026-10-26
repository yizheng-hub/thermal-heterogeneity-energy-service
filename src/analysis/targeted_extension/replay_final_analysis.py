"""Recompute final evidence tables from retained annual and event trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


SOURCES = {
    "ashrae": "energy_service/engineering/PROFILE_ASSIGNMENTS.csv.gz",
    "ctcd": "ctcd_parallel/annual/engineering/PROFILE_ASSIGNMENTS.csv.gz",
}


def command(script_root: Path, name: str, *arguments: object) -> list[str]:
    return [sys.executable, str(script_root / name), *(str(value) for value in arguments)]


def build_commands(
    script_root: Path,
    retained_root: Path,
    data_root: Path,
    model_root: Path,
    output_root: Path,
) -> list[list[str]]:
    factors = data_root / "energy_service/factors/regional_factors.csv"
    input_root = data_root / "energy_service"
    anchor_manifest = data_root / "energy_service/engineering/ANNUAL_RUN_MANIFEST.csv"
    event_dates = data_root / "energy_service/engineering/EVENT_DATES_BEFORE_RUNS.json"
    commands: list[list[str]] = []
    for source, profile_relative in SOURCES.items():
        profiles = data_root / profile_relative
        retained_annual = retained_root / f"repaired_controls_{source}_full"
        for scope, suffix in (("primary", "full_analysis"), ("strict", "strict_analysis")):
            analysis = output_root / f"repaired_controls_{source}_{suffix}"
            commands.append(
                command(
                    script_root,
                    "analyse_repaired_controls.py",
                    "--run-manifest",
                    retained_annual / "RUN_MANIFEST.csv",
                    "--commands",
                    retained_annual / "LOGICAL_DESIGNS.json",
                    "--profiles",
                    profiles,
                    "--input-root",
                    input_root,
                    "--model-root",
                    model_root,
                    "--factors",
                    factors,
                    "--output",
                    analysis,
                    "--data-source",
                    source.upper(),
                    "--quality-scope",
                    scope,
                )
            )
            spatial_suffix = "spatial" if scope == "primary" else "strict_spatial"
            commands.append(
                command(
                    script_root,
                    "analyse_repaired_spatial.py",
                    "--analysis-dir",
                    analysis,
                    "--output",
                    output_root / f"repaired_controls_{source}_{spatial_suffix}",
                )
            )
            seasonal_suffix = "seasonal" if scope == "primary" else "strict_seasonal"
            commands.append(
                command(
                    script_root,
                    "analyse_seasonal_reallocation.py",
                    "--analysis-dir",
                    analysis,
                    "--run-manifest",
                    retained_annual / "RUN_MANIFEST.csv",
                    "--profiles",
                    profiles,
                    "--anchor-annual-manifest",
                    anchor_manifest,
                    "--factors",
                    factors,
                    "--output",
                    output_root / f"repaired_controls_{source}_{seasonal_suffix}",
                    "--data-source",
                    source.upper(),
                )
            )

    for scope in ("primary", "strict"):
        common = command(
            script_root,
            "analyse_common_baseline_dr.py",
            "--run-manifest",
            retained_root / "common_baseline_dr/RUN_MANIFEST.csv",
            "--annual-manifest",
            anchor_manifest,
            "--event-dates",
            event_dates,
            "--factors",
            factors,
            "--output",
            output_root / f"common_baseline_dr_{scope}_analysis",
            "--quality-scope",
            scope,
        )
        for source, profile_relative in SOURCES.items():
            common.extend(("--profiles", f"{source.upper()}={data_root / profile_relative}"))
        commands.append(common)
        for source, profile_relative in SOURCES.items():
            commands.append(
                command(
                    script_root,
                    "analyse_own_baseline_dr.py",
                    "--data-source",
                    source.upper(),
                    "--event-manifest",
                    retained_root / f"own_baseline_dr_{source}/RUN_MANIFEST.csv",
                    "--annual-manifest",
                    retained_root / f"repaired_controls_{source}_full/RUN_MANIFEST.csv",
                    "--anchor-annual-manifest",
                    anchor_manifest,
                    "--event-dates",
                    event_dates,
                    "--profiles",
                    data_root / profile_relative,
                    "--factors",
                    factors,
                    "--output",
                    output_root / f"own_baseline_dr_{source}_{scope}_analysis",
                    "--quality-scope",
                    scope,
                )
            )
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--information-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    script_root = Path(__file__).resolve().parent
    retained_root = args.retained_root.resolve()
    output_root = args.output_root.resolve()
    commands = build_commands(
        script_root,
        retained_root,
        args.data_root.resolve(),
        args.model_root.resolve(),
        output_root,
    )
    if args.plan:
        print(json.dumps({"steps": len(commands), "commands": commands}, indent=2))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    for item in commands:
        subprocess.run(item, check=True)
    decision_value = output_root.parent / "decision_value"
    decision_value.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        args.information_summary.resolve(),
        decision_value / "INFORMATION_HIERARCHY_SUMMARY.json",
    )
    report = {
        "status": "PASS",
        "retained_trajectory_root": retained_root.name,
        "analysis_steps": len(commands),
        "data_sources": ["ASHRAE", "CTCD"],
        "quality_scopes": ["primary", "strict"],
        "physical_simulations_run": 0,
    }
    (output_root / "FINAL_ANALYSIS_REPLAY_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
