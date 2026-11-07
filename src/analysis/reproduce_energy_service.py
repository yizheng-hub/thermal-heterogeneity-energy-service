"""Recalculate the frozen energy-service extension from minimized hourly inputs.

This entry point recalculates the scientific core. The complete paper workflow
is provided by reproduce_energy_service_publication.py in the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from energy_service.bundle import verify_bundle, verify_commands

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = REPOSITORY_ROOT / "src/analysis/energy_service"
    data, out = args.input_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    verified = verify_bundle(data, data / "MANIFEST.csv")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    steps = []

    def run(name, script, *arguments):
        result = subprocess.run([sys.executable, str(source / script), *map(str, arguments)],
                                text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=source)
        (logs / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        steps.append({"step": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")
        print(name, flush=True)

    (out / "BUNDLE_VERIFICATION.json").write_text(json.dumps(verified, indent=2) + "\n", encoding="utf-8")
    run("01_regenerate_controls", "generate_controls.py", "--input-root", data,
        "--output-root", out / "controls")
    command_check = verify_commands(out / "controls/CONTROL_COMMANDS.json",
                                    data / "engineering/CONTROL_REFERENCE.json")
    (out / "CONTROL_PARITY.json").write_text(json.dumps(command_check, indent=2) + "\n", encoding="utf-8")
    annual, events, costs, portfolio = (out / name for name in ("annual", "events", "costs", "portfolio"))
    run("02_actual_annual_controls", "analyse_annual.py", "--engineering-root", data / "engineering",
        "--factors", data / "factors/regional_factors.csv", "--output-root", annual)
    run("02b_nested_permissions", "nested_permissions.py", "--annual-root", annual,
        "--output-root", out / "nested_permissions")
    run("02c_common_layout_anchor", "common_layout_anchor.py", "--annual-root", annual,
        "--output-root", out / "common_layout")
    run("02d_dense_uniform_bound", "dense_uniform_service_bound.py", "--candidate-scores",
        data / "controller/DENSE_UNIFORM_SCORES.csv", "--common-anchors",
        out / "common_layout/COMMON_LAYOUT_ANCHORS.csv", "--output-root", out / "dense_uniform_bound")
    run("03_own_baseline_events", "analyse_events.py", "--engineering-root", data / "engineering",
        "--annual-root", annual, "--factors", data / "factors/regional_factors.csv",
        "--areas", data / "factors/prototype_areas.csv", "--output-root", events)
    run("04_implementation_costs", "run_costs.py", "--pairs", annual / "CONTROL_COMPARISONS.csv",
        "--factors", data / "factors/regional_factors.csv", "--assumptions",
        data / "factors/IMPLEMENTATION_ASSUMPTIONS.json", "--zone-groups",
        data / "engineering/ZONE_EXECUTION_GROUPS.csv", "--output", costs)
    run("05_synchronized_portfolios", "run_portfolio.py", "--types", events / "PORTFOLIO_TYPES.csv",
        "--hourly", events / "PORTFOLIO_HOURLY.csv", "--design-types", events / "PORTFOLIO_DESIGN_TYPES.csv",
        "--protocol", data / "protocol/PORTFOLIO_COST_PROTOCOL.json", "--output", portfolio)
    files = [{"path": path.relative_to(out).as_posix(),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(out.rglob("*")) if path.is_file()]
    report = {"status": "PASS", "scope": "Controller regeneration and new scientific analysis",
              "figures_and_documents_regenerated": False,
              "physical_simulations_run": 0, "building_physical_parameters_changed": False,
              "input_verification": verified, "command_verification": command_check,
              "steps": steps, "outputs": files}
    (out / "ENERGY_SERVICE_RUN.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
