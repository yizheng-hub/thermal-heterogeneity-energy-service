"""Independent manifest verification."""

import argparse

from thermal_heterogeneity.verify import verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    verify_manifest(args.manifest)


if __name__ == "__main__":
    main()
