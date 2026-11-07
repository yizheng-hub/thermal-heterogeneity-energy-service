"""Build the simulation-data deposit and standalone submission attachment."""

import argparse
import json
from pathlib import Path

from publication.build_clean_figure_package import build
from publication.build_simulation_data import build_inputs, build_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.figures)))
    print(json.dumps(build_inputs(args.inputs)))
    print(json.dumps(build_results(args.results, args.figures)))
