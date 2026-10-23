"""Generate the 48 panel-level figure-data products."""

import argparse

from thermal_heterogeneity.figure_data import reproduce_figure_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--analysis", help="Level-1 run root containing tables/")
    parser.add_argument("--data-root", help="Fetched analysis-bundle directory")
    parser.add_argument("--output", help="Panel-data v2 output directory")
    args = parser.parse_args()
    reproduce_figure_data(
        args.config,
        args.output,
        data_root=args.data_root,
        analysis_root=args.analysis,
    )


if __name__ == "__main__":
    main()
