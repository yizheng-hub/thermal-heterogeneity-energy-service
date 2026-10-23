"""Reproduce supporting scientific evidence from minimized inputs and frozen summaries.

Run from any working directory. No local project checkout, original manuscript,
raw human records or EnergyPlus installation is required for this workflow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-building-data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-root", type=Path, help="Reuse an existing verified Level-1 run."
    )
    parser.add_argument("--panel-data", type=Path, help="Reuse panel data from that Level-1 run.")
    parser.add_argument("--energy-service-input-root", type=Path)
    args = parser.parse_args()
    publication = Path(__file__).resolve().parent
    repository = publication.parent
    data = args.data_root.resolve()
    source_building_data = args.source_building_data.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    commands = []

    def run(name: str, script: Path, *arguments: object, python: Path | None = None) -> None:
        command = [str(python or Path(sys.executable)), str(script)] + [str(x) for x in arguments]
        print(name, flush=True)
        result = subprocess.run(
            command,
            cwd=repository,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (logs / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        commands.append({"step": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed. See {logs / (name + '.log')}")

    for record in json.loads(
        (publication / "support" / "MANIFEST.json").read_text(encoding="utf-8")
    ):
        path = publication / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen publication support checksum mismatch: {record['path']}")
    run(
        "00_verify_minimized_bundle",
        repository / "scripts" / "verify_data_manifest.py",
        "--root",
        data,
        "--manifest",
        repository / "data" / "analysis_bundle_manifest.csv",
        "--output",
        output / "bundle_verification.json",
    )
    run(
        "00b_verify_source_building_bundle",
        repository / "scripts" / "unpack_source_building_data.py",
        "--root",
        source_building_data,
        "--report",
        output / "source_building_bundle_verification.json",
    )
    analysis = args.analysis_root.resolve() if args.analysis_root else output / "analysis"
    if not args.analysis_root:
        run(
            "01_reproduce_analysis",
            repository / "scripts" / "reproduce_analysis.py",
            "--config",
            "configs/paper.yaml",
            "--data-root",
            data,
            "--output",
            analysis,
            "--workers",
            "1",
        )
        run(
            "02_reproduce_si",
            repository / "scripts" / "reproduce_si.py",
            "--config",
            "configs/paper.yaml",
            "--data-root",
            data,
            "--analysis",
            analysis,
            "--output",
            analysis,
        )
    run(
        "03_verify_analysis",
        repository / "scripts" / "verify_analysis.py",
        "--manifest",
        analysis / "manifest.json",
    )
    panels = args.panel_data.resolve() if args.panel_data else output / "panel_data"
    if not args.panel_data:
        run(
            "04_reproduce_panel_data",
            repository / "scripts" / "reproduce_figures.py",
            "--config",
            "configs/paper.yaml",
            "--data-root",
            data,
            "--analysis",
            analysis,
            "--output",
            panels,
        )
    decision = publication / "experiments" / "decision_value"
    run(
        "04b_quantile_decision_value",
        decision / "run_decision_value.py",
        "--release-root",
        repository,
        "--data-root",
        data,
        "--output-root",
        output / "decision_value",
    )
    run(
        "04c_assignment_provenance",
        decision / "audit_assignment_provenance.py",
        "--data-root",
        data,
        "--output-root",
        output / "decision_value",
    )
    run(
        "04d_candidate_support",
        publication / "experiments" / "candidate_support" / "reproduce_support.py",
        "--input-root",
        publication / "support" / "candidate_support",
        "--output-root",
        output / "candidate_support",
    )
    run(
        "04e_source_building_robustness",
        publication / "experiments" / "source_building_robustness" / "run_robustness.py",
        "--data-root",
        data,
        "--input-root",
        source_building_data,
        "--output-root",
        output / "source_building_robustness",
    )
    checks = []
    for name in ("MAIN_FIG6G_Q5_REGRET.csv", "TABLE_S17_PROTECTED_QUANTILE_TRANSFER.csv"):
        with (publication / "support" / "decision_value" / name).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            expected = list(csv.DictReader(handle))
        with (output / "decision_value" / name).open(encoding="utf-8-sig", newline="") as handle:
            calculated = list(csv.DictReader(handle))
        if calculated != expected:
            raise ValueError(
                f"Calculated decision-value product differs from frozen reference: {name}"
            )
        checks.append({"path": name, "rows": len(calculated), "identical_values": True})
    (output / "decision_value" / "PUBLIC_REFERENCE_CHECK.json").write_text(
        json.dumps({"used_only_after_calculation": True, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    paper = output / "paper"
    paper.mkdir(parents=True)
    shutil.copyfile(
        publication / "PANEL_UNIT_CORRECTIONS.json", paper / "PANEL_UNIT_CORRECTIONS.json"
    )
    run(
        "05_main_figures",
        publication / "scripts" / "render_revised_main_figures.py",
        "--release-root",
        repository,
        "--panel-data",
        panels,
        "--output-root",
        paper,
        "--decision-value",
        output / "decision_value" / "MAIN_FIG6G_Q5_REGRET.csv",
    )
    run(
        "06_supplementary_figures",
        publication / "scripts" / "render_supplementary.py",
        "--data-root",
        data,
        "--panel-data",
        panels,
        "--output",
        paper / "supplementary_figures",
        "--energy-service-input-root",
        args.energy_service_input_root or data.parent / "energy_service",
        "--science-audit",
        publication / "support" / "science_audit",
        "--spatial-audit",
        publication / "support" / "spatial_audit",
        "--land",
        repository / "src" / "thermal_heterogeneity" / "assets" / "ne_110m_land.geojson",
    )
    cost = publication / "experiments" / "management_cost"
    run(
        "07b_management_cost",
        cost / "reproduce_costs.py",
        "--input",
        data / "building" / "regional_carbon_cost.csv.gz",
        "--output",
        output / "management_cost",
        "--assumptions",
        cost / "ASSUMPTIONS.json",
    )
    tables = output / "Supplementary_Data"
    tables.mkdir()
    for number in range(1, 13):
        shutil.copyfile(analysis / "si" / f"TableS{number}.csv", tables / f"TableS{number}.csv")
    for number, name in (
        (13, "ledger_annual_failure_and_unmatched.csv"),
        (14, "ledger_dynamic_failure_and_unmatched.csv"),
        (15, "ledger_dynamic_nonpositive_response.csv"),
    ):
        shutil.copyfile(analysis / "ledgers" / name, tables / f"TableS{number}.csv")
    shutil.copyfile(output / "management_cost" / "scenario_costs.csv", tables / "TableS16.csv")
    shutil.copyfile(
        output / "decision_value" / "TABLE_S17_PROTECTED_QUANTILE_TRANSFER.csv",
        tables / "TableS17.csv",
    )
    shutil.copyfile(output / "candidate_support" / "TableS18.csv", tables / "TableS18.csv")
    shutil.copyfile(
        output / "source_building_robustness" / "TABLE_S19_SOURCE_BUILDING_ROBUSTNESS.csv",
        tables / "TableS19.csv",
    )
    outputs = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            outputs.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path)})
    result = {
        "status": "PASS",
        "analysis_regenerated": not bool(args.analysis_root),
        "panel_data_regenerated": not bool(args.panel_data),
        "energyplus_runs": 0,
        "building_physics_retuned": False,
        "steps": commands,
        "outputs": outputs,
    }
    paths = [
        line.strip()
        for line in (repository / "protocol" / "PUBLIC_FILE_ALLOWLIST.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.startswith("#")
    ]
    result["source_files"] = [{"path": name, "sha256": sha256(repository / name)} for name in paths]
    (output / "PUBLICATION_RUN.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"outputs", "source_files"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
