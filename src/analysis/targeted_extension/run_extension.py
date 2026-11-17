"""Run the four bounded checks without altering frozen building inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo, replay, output = [p.resolve() for p in
                            (args.repository_root, args.replay_root, args.output_root)]
    output.mkdir(parents=True, exist_ok=True)
    scripts = repo / "src/analysis/targeted_extension"
    locked = [repo / "data/metadata/energy_service_model_contract.json",
              repo / "data/metadata/energy_service_bundle_manifest.csv",
              repo / "data/inputs/energy_service/controller/CONTEXTS.json",
              replay / "energy_service/controls/CONTROL_COMMANDS.json"]
    upstream = repo / "data/upstream/frozen_buildings"
    for context in ("Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"):
        for name in ("baseline.idf", "weather.epw"):
            path = upstream / context / name
            if path.is_file():
                locked.append(path)
    before = {p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.name:
              digest(p) for p in locked}
    commands = [
        ("service", "service_frontier.py", ["--input-root", replay / "energy_service/annual"]),
        ("spatial", "spatial_effects.py", ["--source-root", replay / "energy_service"]),
        ("response", "response_recovery.py", ["--replay-root", replay, "--context-settings",
                                               repo / "data/inputs/energy_service/controller/CONTEXTS.json"]),
        ("ranges", "range_assumptions.py", ["--repository-root", repo, "--replay-root", replay]),
    ]
    for name, script, arguments in commands:
        print(f"Bounded check: {name}", flush=True)
        command = [sys.executable, str(scripts / script), *map(str, arguments),
                   "--output-root", str(output / name)]
        result = subprocess.run(command, cwd=repo, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        (output / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Bounded check {name} failed; see its log")
    after = {key: digest(path) for key, path in zip(before, locked)}
    if before != after:
        raise ValueError("A frozen input changed during the bounded analysis")
    manifest = {
        "status": "PASS", "protocol_sha256": digest(scripts / "PROTOCOL.json"),
        "new_physical_simulations": 0, "frozen_inputs_unchanged": True,
        "frozen_inputs_sha256": before,
        "source_sha256": {p.name: digest(p) for p in sorted(scripts.glob("*.py"))},
        "outputs": {p.relative_to(output).as_posix(): digest(p)
                    for p in sorted(output.rglob("*.csv"))},
        "scope": "Reanalysis of retained trajectories and existing regenerated-control sensitivities; "
                 "not new empirical Personal range validation or global control optimization.",
    }
    (output / "BOUNDED_EXTENSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("status", "new_physical_simulations", "frozen_inputs_unchanged")}))


if __name__ == "__main__":
    main()
