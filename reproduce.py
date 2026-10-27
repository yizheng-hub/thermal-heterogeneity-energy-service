"""Verify and reproduce the final nine-climate analysis and figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


MAIN_FIGURES = 6
SUPPLEMENTARY_FIGURES = 21


def run(command: list[str], *, root: Path, environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=root, env=environment, check=True)


def default_input_root(root: Path) -> Path:
    packaged = root / "data/inputs/final_analysis/global_extension"
    local = root / "results/global_extension"
    if (packaged / "final_story/FINAL_STORY_SUMMARY.json").is_file():
        return packaged
    if (local / "final_story/FINAL_STORY_SUMMARY.json").is_file():
        return local
    return packaged


def global_command(
    root: Path,
    args: argparse.Namespace,
    mode: str,
    *,
    include_plan: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(root / "src/analysis/global_extension/reproduce_global_extension.py"),
        "--mode",
        mode,
        "--input-root",
        str(args.global_input_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
    ]
    if include_plan:
        command.append("--plan")
    return command


def figure_command(root: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(root / "plot/reproduce.py")]
    if args.figure:
        command.extend(("--figure", *args.figure))
    return command


def global_plan(
    root: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
    mode: str,
) -> dict[str, object]:
    completed = subprocess.run(
        global_command(root, args, mode, include_plan=True),
        cwd=root,
        env=environment,
        check=True,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    return json.loads(completed.stdout)


def plan(
    root: Path,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, object]:
    global_mode = {
        "verify": "verify",
        "reassemble": "reassemble",
        "analysis": "analysis",
        "full": "reassemble",
        "figures": "verify",
    }[args.mode]
    global_details = global_plan(root, args, environment, global_mode)
    steps = [] if args.mode == "figures" else list(global_details["steps"])
    if args.mode in {"figures", "full"}:
        steps.append("plot/reproduce.py")
    return {
        "status": "PLAN",
        "mode": args.mode,
        "steps": steps,
        "global_extension": global_details,
        "figures": MAIN_FIGURES + SUPPLEMENTARY_FIGURES,
        "figure_counts": {
            "main": MAIN_FIGURES,
            "supplementary": SUPPLEMENTARY_FIGURES,
            "total": MAIN_FIGURES + SUPPLEMENTARY_FIGURES,
        },
        "controls": ["Uniform average", "Personal centre", "Personal range"],
        "manuscripts_required": False,
        "word_templates_required": False,
        "manuscript_generated": False,
    }


def parse_args(root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("verify", "reassemble", "analysis", "figures", "full"),
        default="verify",
        help=(
            "verify retained final tables; reassemble final story and implementation "
            "tables; analysis reserves the full trajectory DAG and currently fails "
            "until its portable input bundle is declared; full reassembles then plots"
        ),
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--global-input-root",
        type=Path,
        default=default_input_root(root),
        help="Checksum-indexed retained global-extension result package",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "results/global_extension",
        help="Destination for reassembled final global result tables",
    )
    parser.add_argument(
        "--figure",
        nargs="+",
        help="With figures/full mode, redraw selected IDs (1...6, S1...S21)",
    )
    args = parser.parse_args()
    if args.figure and args.mode not in {"figures", "full"}:
        parser.error("--figure is available only with --mode figures or --mode full")
    return args


def main() -> int:
    root = Path(__file__).resolve().parent
    args = parse_args(root)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    if args.plan:
        print(json.dumps(plan(root, args, environment), indent=2))
        return 0

    if args.mode in {"verify", "reassemble", "analysis"}:
        run(global_command(root, args, args.mode), root=root, environment=environment)
    elif args.mode == "full":
        run(global_command(root, args, "reassemble"), root=root, environment=environment)
        run(figure_command(root, args), root=root, environment=environment)
    else:
        run(figure_command(root, args), root=root, environment=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
