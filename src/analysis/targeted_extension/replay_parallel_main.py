"""Replay six frozen demand compositions on their shared experimental design.

The existing primary replay is read without modification. CTCD commands and
retained annual/event trajectories are verified before the common numerical
analyses, portfolios and implementation scenarios are recomputed. No physical
simulation, new sampling, or change of building parameters occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def workflow(repository, primary, bundle, output, metadata_root=None):
    analysis = repository / "src/analysis"
    extension = analysis / "targeted_extension"
    energy = analysis / "energy_service"
    main = repository / "data/inputs/energy_service"
    factors = main / "factors/regional_factors.csv"
    contexts = main / "controller/CONTEXTS.json"
    parallel = output / "replay"
    events = extension / "ctcd_parallel_events.py"
    steps = [
        ("annual", extension / "replay_ctcd_parallel.py", [
            "--input-root", bundle, "--main-input-root", main, "--output-root", parallel,
            *(["--metadata-root", metadata_root] if metadata_root else [])]),
        ("common_layout", energy / "common_layout_anchor.py", [
            "--annual-root", parallel / "annual", "--output-root", parallel / "common_layout",
            "--expected-layouts", "2"]),
        ("nested_permissions", energy / "nested_permissions.py", [
            "--annual-root", parallel / "annual", "--output-root", parallel / "nested_permissions",
            "--expected-layouts", "2"]),
        ("spatial", extension / "spatial_effects.py", [
            "--source-root", parallel, "--output-root", parallel / "spatial", "--expected-layouts", "2"]),
        ("service", extension / "service_frontier.py", [
            "--input-root", parallel / "annual", "--output-root", parallel / "service"]),
        ("seasonal", analysis / "seasonal/derive_seasonal_comparison.py", [
            "--engineering-root", bundle / "annual/engineering", "--annual-root", parallel / "annual",
            "--factors", factors, "--analysis-module", energy / "analyse_annual.py",
            "--output", output / "seasonal", "--expected-layouts", "2",
            "--figure-composition", "2026102501"]),
        ("events", energy / "analyse_events.py", [
            "--engineering-root", bundle / "events/engineering", "--annual-root", parallel / "annual",
            "--factors", factors, "--areas", main / "factors/prototype_areas.csv",
            "--expected-baselines", "216", "--expected-profile-groups", "24",
            "--output-root", output / "events"]),
    ]
    for cohort, annual, event_root in (
            ("ASHRAE", primary / "energy_service/annual", primary / "energy_service/events"),
            ("CTCD", parallel / "annual", output / "events")):
        label = cohort.lower()
        steps.extend([
            (label + "_portfolio", events, ["portfolio", "--event-root", event_root,
                "--contexts", contexts, "--cohort", cohort, "--output-root", output / (label + "_portfolio")]),
            (label + "_costs", events, ["costs", "--annual-root", annual, "--factors", factors,
                "--assumptions", main / "factors/IMPLEMENTATION_ASSUMPTIONS.json",
                "--zone-groups", main / "engineering/ZONE_EXECUTION_GROUPS.csv",
                "--cohort", cohort, "--output-root", output / (label + "_costs")]),
        ])
    steps.extend([
        ("joint_spatial_diagnostics", events, ["joint-spatial-diagnostics",
            "--ashrae-spatial", primary / "targeted_extension/spatial",
            "--ctcd-spatial", parallel / "spatial", "--output-root", output / "joint_spatial_diagnostics"]),
        ("joint_spatial", energy / "derive_joint_spatial.py", [
            "--primary-root", primary / "energy_service",
            "--primary-assignments", main / "engineering/PROFILE_ASSIGNMENTS.csv.gz",
            "--ctcd-root", parallel,
            "--ctcd-assignments", bundle / "annual/engineering/PROFILE_ASSIGNMENTS.csv.gz",
            "--output-root", output / "joint_spatial"]),
        ("portfolio_scope_sensitivity", events, ["scope-sensitivity",
            "--event-root", primary / "energy_service/events", "--annual-root", primary / "energy_service/annual",
            "--contexts", contexts, "--output-root", output / "portfolio_scope_sensitivity"]),
        ("joint_portfolio", events, ["joint-portfolio", "--ashrae-events", primary / "energy_service/events",
            "--ctcd-events", output / "events", "--ashrae-portfolio", output / "ashrae_portfolio",
            "--ctcd-portfolio", output / "ctcd_portfolio", "--contexts", contexts,
            "--output-root", output / "joint_portfolio"]),
        ("joint_events", events, ["joint-events", "--ashrae-events", primary / "energy_service/events",
            "--ctcd-events", output / "events", "--joint-portfolio-root", output / "joint_portfolio",
            "--contexts", contexts, "--output-root", output / "joint_events"]),
        ("joint_main", extension / "build_joint_evidence.py", ["--repository-root", repository,
            "--primary-replay-root", primary, "--ctcd-root", output, "--output-root", output / "joint_main"]),
        ("independent_joint_audit", extension / "audit_joint_evidence.py", ["--repository-root", repository,
            "--evidence-root", output / "joint_main", "--output", output / "joint_main/INDEPENDENT_JOINT_EVIDENCE_QA.json"]),
    ])
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--primary-replay-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--metadata-root", type=Path, help="Directory containing the supplied CTCD archive manifest and checksum record")
    args = parser.parse_args()
    repository, primary, output = (p.resolve() for p in (
        args.repository_root, args.primary_replay_root, args.output_root))
    bundle = args.input_root.resolve() if args.input_root else repository / "data/inputs/ctcd_parallel"
    steps = workflow(repository, primary, bundle, output, args.metadata_root)
    if args.plan:
        print(json.dumps({"steps": [name for name, _, _ in steps], "fixed_compositions": 6,
            "common_layouts": 2, "joint_portfolio_offices": 144, "physical_simulations_run": 0,
            "retained_package": str(bundle)}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    executed = []
    for name, script, arguments in steps:
        command = [sys.executable, str(script), *map(str, arguments)]
        result = subprocess.run(command, cwd=repository, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        log = logs / (name + ".log")
        log.write_text(result.stdout, encoding="utf-8")
        executed.append({"step": name, "script": script.relative_to(repository).as_posix(),
                         "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {log}")
        print(name, flush=True)
    files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in (".csv", ".json", ".gz")}
    report = {"status": "PASS", "fixed_compositions": 6, "common_layouts": 2,
              "joint_portfolio_offices": 144, "data_origins_are_provenance": True,
              "physical_simulations_run": 0, "physical_parameters_changed": False,
              "steps": executed, "output_sha256": files}
    (output / "PARALLEL_MAIN_REPLAY_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "output_sha256"}, indent=2))


if __name__ == "__main__":
    main()
