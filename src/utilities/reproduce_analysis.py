"""Reproduce Results 1-6 from Level-1 minimized inputs."""

import argparse

from thermal_heterogeneity.analysis import refresh_manifest, reproduce_analysis
from thermal_heterogeneity.config import load_config
from thermal_heterogeneity.figure_data import reproduce_figure_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/settings/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = reproduce_analysis(
        args.config,
        args.output,
        data_root=args.data_root,
        workers=args.workers,
    )
    output = manifest.parent
    reproduce_figure_data(
        args.config,
        output,
        data_root=args.data_root,
        legacy=True,
    )
    refresh_manifest(output, load_config(args.config, data_root=args.data_root))


if __name__ == "__main__":
    main()
