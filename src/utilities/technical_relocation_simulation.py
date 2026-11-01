"""Replay three prespecified reference operations using only contained files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
from run_global_matrix import (  # noqa: E402
    ZONES, _write_run_idf, prepare_models, read_climates, run_energyplus,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    runtime = ROOT / "tools/EnergyPlus-22.1.0/energyplus.exe"
    if not output.is_relative_to(ROOT) or not Path(sys.executable).resolve().is_relative_to(ROOT):
        raise ValueError("Use the contained interpreter and an output inside this project")
    climates = [c for c in read_climates(ROOT / "src/analysis/global_extension/GLOBAL_CLIMATES.json")
                if c.climate in {"Montreal", "Delhi", "Brasilia"}]
    models = prepare_models(
        climates, ROOT / "data/upstream/global_climate_matrix/prototypes_22_1",
        ROOT / "data/upstream/global_climate_matrix/weather",
        ROOT / "data/upstream/frozen_buildings/Miami/baseline.idf",
        output / "models", "fixed_reference",
    )
    records = []
    for climate in climates:
        name = climate.climate
        idf = output / "inputs" / f"{name}_reference.idf"
        _write_run_idf(Path(models[name]["prepared_idf"]), dict.fromkeys(ZONES, 25.0), idf)
        hourly = output / "hourly" / f"{name}.npz"
        hourly.parent.mkdir(parents=True, exist_ok=True)
        qa = run_energyplus(
            idf, Path(models[name]["weather_path"]), runtime,
            output / "runs" / name, hourly, ZONES, cleanup_raw=True,
        )
        records.append({"climate": name, "qa": qa})
        print(json.dumps(records[-1]), flush=True)
    report = {
        "purpose": "Technical relocation replay; no new scientific comparison",
        "all_completed": all(r["qa"]["completed"] for r in records),
        "records": records,
    }
    (output / "TECHNICAL_RELOCATION_SIMULATION.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_completed"]:
        raise SystemExit("A contained reference replay did not complete; inspect retained logs")


if __name__ == "__main__":
    main()
