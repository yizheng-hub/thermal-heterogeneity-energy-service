"""Inventory the retained raw inputs, source, environment and simulator."""

from __future__ import annotations

from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import sys


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_inputs(root, *, include_validation):
    data = root / "data"
    paths = [
        data / "upstream/ashrae_db_ii/db_measurements_v2.1.0.csv.gz",
        data / "upstream/ashrae_db_ii/db_metadata.csv",
        data / "metadata/numeric_scene_spec_v1.json",
        data / "metadata/global_electricity_carbon_factors_ember_2025.csv",
        data / "upstream/frozen_buildings/Miami/baseline.idf",
    ]
    paths += [
        data / "upstream/chinese_thermal_comfort" / f"Chinese Thermal Comfort Dataset Class {k}.csv"
        for k in ("I", "II", "III")
    ]
    climates = json.loads(
        (root / "src/analysis/global_extension/GLOBAL_CLIMATES.json").read_text()
    )["climates"]
    paths += [data / "upstream/global_climate_matrix/weather" / row["weather"] for row in climates]
    if include_validation:
        paths += [
            data
            / "upstream/shared_room_observations/longitudinal_personal_thermal_comfort_zenodo_5502441/enth_tabular_merged.csv",
            data
            / "upstream/shared_room_observations/humans_as_sensor_2019/2019-11-15_cozie_full_masked.csv",
        ]
    return paths


def inventory(root, runtime, destination, *, include_validation):
    inputs = raw_inputs(root, include_validation=include_validation)
    missing = [str(p.relative_to(root)) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required retained raw inputs are missing: " + ", ".join(missing))
    outside = [str(p) for p in inputs if not p.resolve().is_relative_to(root.resolve())]
    if outside:
        raise ValueError(
            "Required scientific inputs resolve outside the project: " + ", ".join(outside)
        )
    source = [*root.glob("*.py"), root / "pyproject.toml", root / "uv.lock"]
    source += [
        p for p in (root / "src").rglob("*") if p.suffix in {".py", ".json", ".yaml", ".geojson"}
    ]
    source += list((root / "data/metadata/strengthening_protocol").glob("*.yaml"))
    source.append(root / "plot/Plot.ipynb")
    report = {
        "status": "completed",
        "raw_inputs": {p.relative_to(root).as_posix(): sha(p) for p in sorted(inputs)},
        "source": {p.relative_to(root).as_posix(): sha(p) for p in sorted(set(source))},
        "runtime": {
            p.relative_to(runtime.parent).as_posix(): sha(p)
            for p in sorted(runtime.parent.rglob("*"))
            if p.is_file()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                    "PyYAML",
                    "zstandard",
                ]
            },
        },
        "input_roles": "Raw archives, numerical protocol, physical IDF, weather and carbon factors. No historical result or assignment table is an input.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
