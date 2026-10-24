"""Snapshot the currently frozen extension sources into an isolated release tree.

Run again only after an explicit source revision. Every copied source is hashed;
this operation does not modify the working engineering or analysis directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


SOURCES = {
    "analysis/analyse_annual.py": "analyse_annual.py",
    "analysis/analyse_events.py": "analyse_events.py",
    "analysis/room_interval_feasibility.py": "room_interval_feasibility.py",
    "analysis/thermal_service.py": "thermal_service.py",
    "analysis/nested_permissions.py": "nested_permissions.py",
    "analysis/common_layout_anchor.py": "common_layout_anchor.py",
    "analysis/dense_uniform_service_bound.py": "dense_uniform_service_bound.py",
    "analysis/baseline_decision_contract.py": "baseline_decision_contract.py",
    "experiments/portfolio/run_costs.py": "run_costs.py",
    "experiments/portfolio/run_portfolio.py": "run_portfolio.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.release_root / "src/analysis/energy_service"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for source_name, dest_name in SOURCES.items():
        source = args.stage_root / source_name
        destination = target / dest_name
        shutil.copyfile(source, destination)
        records.append({"source": source_name, "release_path": f"src/analysis/energy_service/{dest_name}",
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    protocol = args.release_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    (protocol / "ENERGY_SERVICE_SOURCE_SNAPSHOT.json").write_text(
        json.dumps({"status": "SOURCE_SNAPSHOT_ONLY", "full_reproduction_complete": False,
                    "sources": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources_copied": len(records), "full_reproduction_complete": False}))


if __name__ == "__main__":
    main()
