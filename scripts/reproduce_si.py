"""Generate supplementary tables from regenerated analyses."""

import argparse

from thermal_heterogeneity.analysis import refresh_manifest
from thermal_heterogeneity.config import load_config
from thermal_heterogeneity.si import reproduce_si


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    reproduce_si(args.config, args.analysis, args.output)
    refresh_manifest(
        args.analysis,
        load_config(args.config, data_root=args.data_root),
    )


if __name__ == "__main__":
    main()
