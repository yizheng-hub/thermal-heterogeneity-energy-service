"""Reproduce scientific analyses, 30 figures and source tables from external inputs."""
from pathlib import Path
import os
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    arguments = list(sys.argv[1:])
    if "--plan" not in arguments:
        for flag, folder in (("--data-root", "data/inputs"),
                             ("--typical-day-data-root", "data/typical_days"),
                             ("--output-root", "output/replay")):
            if flag not in arguments:
                arguments.extend((flag, str(root / folder)))
    status = subprocess.call([
        sys.executable, str(root / "src/analysis/reproduce_energy_service_publication.py"),
        *arguments,
    ], cwd=root, env=environment)
    if status or "--plan" in arguments:
        raise SystemExit(status)
    replay = Path(arguments[arguments.index("--output-root") + 1]).resolve()
    subprocess.run([sys.executable, str(root / "src/utilities/prepare_plot_data.py"),
                    "--replay", str(replay)], cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, str(root / "plot/reproduce.py")],
                   cwd=root, env=environment, check=True)
