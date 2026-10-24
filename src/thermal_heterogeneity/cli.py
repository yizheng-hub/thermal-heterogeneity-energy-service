"""Command line interface for the Level-1 public analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import reproduce_analysis
from .figure_data import reproduce_figure_data
from .si import reproduce_si
from .verify import verify_manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="thermal-heterogeneity")
    result.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = result.add_subparsers(dest="command")
    for name in ("analysis", "figures", "si"):
        item = commands.add_parser(name)
        item.add_argument("--config", default="src/settings/paper.yaml")
        item.add_argument("--output")
        item.add_argument("--data-root")
    commands.choices["figures"].add_argument("--analysis")
    commands.choices["analysis"].add_argument("--workers", type=int, default=1)
    commands.choices["si"].add_argument("--analysis")
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command is None:
        parser().print_help()
        return 0
    if args.command == "analysis":
        reproduce_analysis(
            args.config,
            args.output,
            data_root=args.data_root,
            workers=args.workers,
        )
    elif args.command == "figures":
        reproduce_figure_data(
            args.config,
            args.output,
            data_root=args.data_root,
            analysis_root=args.analysis,
        )
    elif args.command == "si":
        reproduce_si(args.config, args.analysis or args.output, args.output if args.analysis else None)
    elif args.command == "verify":
        verify_manifest(Path(args.manifest))
    return 0
