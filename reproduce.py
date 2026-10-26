"""Reproduce the final analyses, source tables and publication figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MAIN_FIGURES = 6
SUPPLEMENTARY_FIGURES = 21


def run(command: list[str], *, root: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=root, env=environment, check=True)


def resolve_final_analysis_root(root: Path, requested: Path | None) -> Path:
    candidates = [] if requested is None else [requested]
    candidates.extend(
        (
            root / "data/inputs/final_analysis/Reviewer_repair",
            root / "results/Reviewer_repair",
        )
    )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else root / candidate
        required = (
            resolved / "repaired_controls_ashrae_full/RUN_MANIFEST.csv",
            resolved / "repaired_controls_ctcd_full/RUN_MANIFEST.csv",
            resolved / "common_baseline_dr/RUN_MANIFEST.csv",
            resolved / "own_baseline_dr_ashrae/RUN_MANIFEST.csv",
            resolved / "own_baseline_dr_ctcd/RUN_MANIFEST.csv",
        )
        if all(path.is_file() for path in required):
            return resolved.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Final retained-trajectory analysis package is missing. "
        f"Supply --final-analysis-root or unpack it at a default location. Searched: {searched}"
    )


def publication_command(root: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(root / "src/analysis/reproduce_energy_service_publication.py"),
        "--data-root",
        str(args.data_root.resolve()),
        "--typical-day-data-root",
        str(args.typical_day_data_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
    ]


def plan(root: Path, args: argparse.Namespace, environment: dict[str, str]) -> dict[str, object]:
    steps: list[str] = []
    base: dict[str, object] = {}
    if args.mode in {"analysis", "full"}:
        command = publication_command(root, args) + ["--plan"]
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
        base = json.loads(completed.stdout)
        steps.extend(base.get("steps", []))
        steps.extend(
            (
                "src/analysis/targeted_extension/run_extension.py",
                "src/analysis/targeted_extension/replay_ctcd_external.py",
                "src/analysis/targeted_extension/replay_parallel_main.py",
                "src/analysis/experiments/decision_value/summarize_information_hierarchy.py",
                "src/utilities/prepare_plot_data.py",
                "src/analysis/targeted_extension/reviewer_sensitivity.py",
                "src/analysis/targeted_extension/replay_final_analysis.py",
                "src/analysis/targeted_extension/assemble_final_evidence.py",
                "src/analysis/energy_service/run_costs.py",
            )
        )
    if args.mode in {"figures", "full"}:
        steps.append("plot/reproduce.py")
    requested_final = args.final_analysis_root
    if requested_final is None:
        local_final = root / "results/Reviewer_repair"
        requested_final = (
            local_final
            if (local_final / "repaired_controls_ashrae_full/RUN_MANIFEST.csv").is_file()
            else Path("data/inputs/final_analysis/Reviewer_repair")
        )
    result = {
        **base,
        "mode": args.mode,
        "steps": steps,
        "figures": MAIN_FIGURES + SUPPLEMENTARY_FIGURES,
        "figure_counts": {
            "main": MAIN_FIGURES,
            "supplementary": SUPPLEMENTARY_FIGURES,
            "total": MAIN_FIGURES + SUPPLEMENTARY_FIGURES,
        },
        "controls": ["Uniform average", "Personal centre", "Personal range"],
        "final_analysis_input": str(requested_final),
        "manuscripts_required": False,
        "word_templates_required": False,
        "manuscript_generated": False,
    }
    if args.mode in {"analysis", "full"}:
        ctcd = json.loads(
            (root / "data/metadata/ctcd_bundle_source.json").read_text(encoding="utf-8")
        )
        parallel = json.loads(
            (root / "data/metadata/ctcd_parallel_bundle_source.json").read_text(
                encoding="utf-8"
            )
        )
        result.update(
            {
                "external_office_composition": (
                    "CTCD: regenerate 144 commands and score retained annual trajectories"
                ),
                "ctcd_data_files": ctcd["files_excluding_manifest"],
                "ctcd_manifest_files": 1,
                "parallel_ctcd_data_files": parallel["files_excluding_manifest"],
                "parallel_ctcd_manifest_files": 1,
                "main_demand_compositions": 6,
                "common_main_layouts": 2,
                "joint_portfolio_offices": 144,
            }
        )
    return result


def reproduce_analysis(
    root: Path, args: argparse.Namespace, environment: dict[str, str]
) -> Path:
    final_analysis_root = resolve_final_analysis_root(root, args.final_analysis_root)
    replay = args.output_root.resolve()
    run(publication_command(root, args), root=root, environment=environment)
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/run_extension.py"),
            "--repository-root",
            str(root),
            "--replay-root",
            str(replay),
            "--output-root",
            str(replay / "targeted_extension"),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/replay_ctcd_external.py"),
            "--input-root",
            str(args.data_root / "ctcd_external"),
            "--main-input-root",
            str(args.data_root / "energy_service"),
            "--output-root",
            str(replay / "ctcd_external"),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/replay_parallel_main.py"),
            "--repository-root",
            str(root),
            "--primary-replay-root",
            str(replay),
            "--output-root",
            str(replay / "ctcd_parallel"),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(
                root
                / "src/analysis/experiments/decision_value/summarize_information_hierarchy.py"
            ),
            "--phase-map",
            str(replay / "supporting_evidence/analysis/tables/result6_phase_map.csv"),
            "--control-comparisons",
            str(replay / "ctcd_parallel/joint_main/annual/CONTROL_COMPARISONS.csv"),
            "--output-root",
            str(replay / "information_hierarchy"),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(root / "src/utilities/prepare_plot_data.py"),
            "--replay",
            str(replay),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/reviewer_sensitivity.py"),
            "--repository-root",
            str(root),
            "--output-root",
            str(replay / "reviewer_sensitivity"),
        ],
        root=root,
        environment=environment,
    )

    final_replay_root = replay / "final_submission_analysis/Reviewer_repair"
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/replay_final_analysis.py"),
            "--retained-root",
            str(final_analysis_root),
            "--data-root",
            str(args.data_root),
            "--model-root",
            str(args.model_root),
            "--information-summary",
            str(replay / "information_hierarchy/INFORMATION_HIERARCHY_SUMMARY.json"),
            "--output-root",
            str(final_replay_root),
        ],
        root=root,
        environment=environment,
    )
    final_submission = root / "results/final_submission"
    evidence = final_submission / "evidence"
    run(
        [
            sys.executable,
            str(root / "src/analysis/targeted_extension/assemble_final_evidence.py"),
            "--results-root",
            str(final_replay_root),
            "--output",
            str(evidence),
        ],
        root=root,
        environment=environment,
    )
    run(
        [
            sys.executable,
            str(root / "src/analysis/energy_service/run_costs.py"),
            "--pairs",
            str(evidence / "FINAL_COST_PAIRS.csv"),
            "--factors",
            str(args.data_root / "energy_service/factors/regional_factors.csv"),
            "--assumptions",
            str(args.data_root / "energy_service/factors/IMPLEMENTATION_ASSUMPTIONS.json"),
            "--zone-groups",
            str(args.data_root / "energy_service/engineering/ZONE_EXECUTION_GROUPS.csv"),
            "--output",
            str(final_submission / "costs"),
        ],
        root=root,
        environment=environment,
    )
    return replay


def reproduce_figures(
    root: Path, args: argparse.Namespace, environment: dict[str, str]
) -> None:
    command = [sys.executable, str(root / "plot/reproduce.py")]
    if args.figure:
        command.extend(("--figure", *args.figure))
    run(command, root=root, environment=environment)


def update_report(root: Path, replay: Path, mode: str) -> None:
    report_path = replay / "SCIENTIFIC_REPLAY_RUN.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["reproduction_mode"] = mode
    report["main_figures"] = MAIN_FIGURES
    report["supplementary_figures"] = SUPPLEMENTARY_FIGURES
    report["final_evidence_qa"] = "results/final_submission/evidence/FINAL_EVIDENCE_QA.json"
    report["final_cost_qa"] = "results/final_submission/costs/cost_QA.json"
    report["manuscript_generated"] = False
    index = root / "results/SOURCE_DATA_INDEX.csv"
    if index.is_file():
        with index.open(encoding="utf-8-sig", newline="") as stream:
            report["final_plot_source_data_csvs"] = sum(1 for _ in csv.DictReader(stream))
    manifest = root / "results/PLOT_INPUTS.json"
    if manifest.is_file():
        report["final_plot_input_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def parse_args(root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "analysis", "figures"), default="full")
    parser.add_argument("--plan", action="store_true", help="Print the read-only execution plan")
    parser.add_argument("--data-root", type=Path, default=root / "data/inputs")
    parser.add_argument("--model-root", type=Path, default=root / "data/upstream/frozen_buildings")
    parser.add_argument(
        "--typical-day-data-root", type=Path, default=root / "data/typical_days"
    )
    parser.add_argument("--output-root", type=Path, default=root / "output/replay")
    parser.add_argument(
        "--final-analysis-root",
        type=Path,
        help="Retained final annual/event trajectories and their analysis tables",
    )
    parser.add_argument(
        "--figure", nargs="+", help="With figures/full mode, redraw selected IDs (1...6, S1...S21)"
    )
    args = parser.parse_args()
    if args.figure and args.mode == "analysis":
        parser.error("--figure is available only with --mode figures or --mode full")
    return args


def main() -> int:
    root = Path(__file__).resolve().parent
    args = parse_args(root)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    if args.plan:
        print(json.dumps(plan(root, args, environment), indent=2))
        return 0
    replay = args.output_root.resolve()
    if args.mode in {"analysis", "full"}:
        replay = reproduce_analysis(root, args, environment)
    if args.mode in {"figures", "full"}:
        reproduce_figures(root, args, environment)
    if args.mode in {"analysis", "full"}:
        update_report(root, replay, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
