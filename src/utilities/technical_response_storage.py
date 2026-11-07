"""Check complete response reporting and retained metadata using contained inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/strengthening"))
import single_event_simulation as simulation  # noqa: E402 - initialize local imports before loading project modules


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf", type=Path, required=True)
    parser.add_argument("--weather", type=Path, required=True)
    parser.add_argument("--baseline-hourly", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.idf.resolve(), args.weather.resolve(), args.baseline_hourly.resolve()]
    out = args.output_root.resolve()
    if not all(p.is_relative_to(ROOT) for p in [*paths, out, Path(sys.executable).resolve()]):
        raise ValueError("Use only this project's inputs, output directory and Python")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"status": "planned", "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        "comparison": "Identical operation with additional hourly air, humidity and surface reporting",
        "stopping_rule": "Complete 8760-hour arrays; reporting-only differences <=1e-8; retained metadata exists and cached hashes verify."}
    (out / "TECHNICAL_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
    with np.load(paths[2], allow_pickle=False) as saved:
        reference = dict(saved)
    reported = out / "reported.idf"
    reported.write_text(simulation.reported_model(paths[0].read_text()))
    simulation.SIM = out
    qa = simulation.execute(reported, paths[1], reference, "normal_reporting")
    if not qa["completed"]:
        raise RuntimeError("Reporting-only simulation did not complete; outputs are retained")
    with np.load(qa["compact_output"], allow_pickle=False) as saved:
        rerun = dict(saved)
    differences = {key: float(np.max(np.abs(reference[key] - rerun[key])))
        for key in ["temperatures", "people", *simulation.ENERGY_METERS]}
    assert all(np.array_equal(reference[key], rerun[key]) for key in ["month", "day", "hour", "dst", "occupied"])
    assert len(rerun["month"]) == 8760 and max(differences.values()) <= 1e-8
    cached = simulation.execute(reported, paths[1], reference, "normal_reporting")
    metadata = Path(qa["sql_metadata_path"])
    assert metadata.is_file() and sha(metadata) == cached["sql_metadata_sha256"]
    report = {"status": "completed", "hours": len(rerun["month"]),
        "reporting_only_max_absolute_differences": differences,
        "state_variable_count": qa["state_variable_count"],
        "retained_sql_metadata": metadata.relative_to(ROOT).as_posix(),
        "retained_sql_metadata_sha256": sha(metadata), "cached_output_hash_checks": "passed",
        "engineering_and_service_scope": "Technical output preservation check; response capability is evaluated separately."}
    (out / "TECHNICAL_RESPONSE_STORAGE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
