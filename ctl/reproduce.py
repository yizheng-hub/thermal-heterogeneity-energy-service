"""Reproduce scientific analyses, 27 figures and source tables from external inputs."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([
        sys.executable, str(root / "publication/reproduce_energy_service_publication.py"),
        *sys.argv[1:],
    ], cwd=root))
