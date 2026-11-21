"""Run the prespecified strengthening analyses from a fresh raw-input run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time

ROOT = Path(__file__).resolve().parent
METHODS = ROOT / "src/analysis/strengthening"


PHASES = {
    "validation": [
        ("demand_validation.py",),
        ("undated_acceptability_validation.py",),
        ("temporal_center_validation.py",),
        ("external_preference_validation.py",),
        ("holdout_integrity.py",),
        ("single_record_validation.py",),
        ("group_information_validation.py",),
    ],
    "refinement": [
        ("candidate_runs.py", "plan"),
        ("candidate_runs.py", "pilot"),
        ("candidate_runs.py", "full"),
        ("strict_warning_audit.py", "CANDIDATE_FULL_MANIFEST.csv"),
        ("score_new_controls.py", "candidate"),
        ("score_new_controls.py", "analysis"),
    ],
    "placement": [
        ("strong_baselines_and_placement.py", "baseline", "plan"),
        ("strong_baselines_and_placement.py", "baseline", "pilot"),
        ("strong_baselines_and_placement.py", "baseline", "full"),
        ("strict_warning_audit.py", "BASELINE_FULL_MANIFEST.csv"),
        ("score_new_controls.py", "baseline"),
        ("strong_baselines_and_placement.py", "placement", "plan"),
        ("strong_baselines_and_placement.py", "placement", "pilot"),
        ("strong_baselines_and_placement.py", "placement", "full"),
        ("strict_warning_audit.py", "PLACEMENT_FULL_MANIFEST.csv"),
        ("score_new_controls.py", "placement"),
        ("summarize_constrained_placement.py",),
    ],
    "strong_baselines": [
        ("uniform_command_search.py",),
        ("baseline_energy_search.py", "plan"),
        ("baseline_energy_search.py", "pilot"),
        ("baseline_energy_search.py", "full"),
        ("strict_warning_audit.py", "BASELINE_OFFSETS_FULL_MANIFEST.csv"),
        ("score_new_controls.py", "baseline_offsets"),
        ("budget_frontiers.py",),
    ],
    "hvac": [
        ("hvac_confirmation.py",),
        ("standard_supply_reset.py",),
        ("independent_supply_reset.py",),
        ("outdoor_supply_reset.py",),
        ("retune_heatpump_controls.py",),
        ("placement_hvac_interaction.py",),
        ("placement_hvac_summary.py",),
        ("heatpump_airflow.py",),
        ("spatial_hvac_mechanism.py",),
        ("zone_ventilation_diagnostic.py",),
        ("vrp_request_diagnostic.py",),
        ("summarize_hvac_interaction.py",),
    ],
    "uncertainty": [
        ("paired_uncertainty.py",),
        ("temporal_uncertainty.py",),
        ("paired_uncertainty.py", "--refined"),
        ("paired_uncertainty.py", "--refined", "--pooled"),
        ("paired_uncertainty.py", "--refined", "--pooled", "--study-shared"),
        ("paired_uncertainty.py", "--refined", "--pooled", "--empirical"),
        ("paired_uncertainty.py", "--refined", "--pooled", "--empirical", "--single-record"),
        ("summarize_information_uncertainty.py",),
        ("verify_nominal_information.py",),
        ("aligned_grid_sensitivity.py",),
        ("service_threshold_sensitivity.py",),
        ("population_command_sensitivity.py",),
        ("roster_transfer.py",),
        ("local_command_sensitivity.py",),
        ("command_domain_sensitivity.py",),
        ("placement_uncertainty.py",),
        ("placement_uncertainty.py", "--study-shared"),
        ("placement_uncertainty.py", "--single-record"),
        ("placement_uncertainty.py", "--study-shared", "--hvac"),
        ("placement_uncertainty.py", "--single-record", "--hvac"),
        ("summarize_placement_uncertainty.py",),
        ("spatial_service_bounds.py",),
        ("placement_service_frontiers.py",),
        ("summarize_placement_frontiers.py",),
    ],
    "response": [
        ("dr_capability.py", "plan"),
        ("dr_capability.py", "baseline"),
        ("dr_capability.py", "pilot"),
        ("dr_capability.py", "full"),
        ("dr_capability.py", "score"),
        ("summarize_dr_capability.py",),
        ("dr_selected_uncertainty.py",),
        ("dr_terminal_recovery.py",),
        ("dr_information_review.py",),
        ("dr_information_uncertainty.py",),
        ("event_date_sensitivity.py",),
    ],
}


def phase_commands(raw, output):
    phases = {name: list(commands) for name, commands in PHASES.items()}
    phases["uncertainty"] = [
        (script, "--results-root", str(output / "results"),
         "--matrix-root", str(raw / "matrix"),
         "--memberships", str(output / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv"),
         "--trajectory-root", str(raw),
         "--output-dir", str(output / "results/service_threshold_sensitivity"))
        if script == "service_threshold_sensitivity.py" else (script, *arguments)
        for script, *arguments in phases["uncertainty"]
    ]
    for index, command in enumerate(phases["uncertainty"]):
        script = command[0]
        if script in {"population_command_sensitivity.py", "command_domain_sensitivity.py"}:
            arguments = ["--results-root", str(output / "results"),
                         "--matrix-root", str(raw / "matrix"),
                         "--output-dir", str(output / "results" / Path(script).stem)]
            if script == "population_command_sensitivity.py":
                arguments += ["--demand-root", str(raw / "demand")]
            phases["uncertainty"][index] = (script, *arguments)
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--energyplus", type=Path, default=ROOT / "tools/EnergyPlus-22.1.0/energyplus.exe"
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--phases", nargs="+", choices=list(PHASES), default=list(PHASES))
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    raw = args.raw_run_root.resolve()
    output = (args.output_root or raw / "strengthening").resolve()
    phases = phase_commands(raw, output)
    if not all(path.is_relative_to(ROOT) for path in [raw, output, args.energyplus.resolve()]):
        raise ValueError(
            "The raw run, strengthening output and simulator must all be inside this self-contained project."
        )
    if args.plan:
        print(
            json.dumps(
                {
                    "raw_run": str(raw),
                    "output": str(output),
                    "phases": {k: phases[k] for k in args.phases},
                    "conditional_refinement": "Run one uniform additional0.125C level only when the frozen quarter-grid rule triggers; retain completeness and energy-convergence status separately.",
                    "historical_archive_inputs": [],
                },
                indent=2,
            )
        )
        return
    for name in ["audit", "config", "logs", "results"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    if {"refinement", "placement", "strong_baselines", "hvac", "response"} & set(args.phases):
        sys.path.insert(0, str(ROOT / "src"))
        from thermal_heterogeneity.energyplus_launcher import verify_runtime

        verify_runtime(args.energyplus.resolve(), output / "audit/RUNTIME_API_PREFLIGHT.json")
    env = os.environ.copy()
    env.update(
        HEM_RAW_RUN=str(raw),
        HEM_STRENGTHENING_OUTPUT=str(output),
        HEM_ENERGYPLUS=str(args.energyplus.resolve()),
        HEM_WORKERS=str(args.workers),
    )

    def run(script, *arguments):
        log_arguments = () if script in {"service_threshold_sensitivity.py", "population_command_sensitivity.py", "command_domain_sensitivity.py"} else arguments
        log = output / "logs" / ("_".join([Path(script).stem, *log_arguments]) + ".log")
        with log.open("ab") as stream:
            result = subprocess.run(
                [sys.executable, str(METHODS / script), *arguments],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            state = output / "PHASE_EXECUTION_STATUS.json"
            if state.is_file():
                phases = json.loads(state.read_text())
                for item in phases.values():
                    if item["execution_status"] == "running":
                        item.update(
                            execution_status="failed",
                            failed_command=[script, *arguments],
                            log=str(log),
                        )
                state.write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(
                f"{script} failed. Inspect {log}; no failed stage was silently skipped."
            )

    run("prepare_strengthening_inputs.py")
    previous_execution = output / "PHASE_EXECUTION_STATUS.json"
    if previous_execution.exists():
        history = output / "audit/phase_execution_history"
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous_execution, history / (str(time.time_ns()) + ".json"))
    execution = {}
    independent_failures = []
    for phase in args.phases:
        execution[phase] = {"execution_status": "running", "commands": phases[phase]}
        status_file = output / "PHASE_EXECUTION_STATUS.json"
        status_file.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        print(phase, "running", flush=True)
        for command in phases[phase]:
            try:
                run(*command)
            except RuntimeError as error:
                if phase != "hvac":
                    raise
                independent_failures.append(
                    {"phase": phase, "command": command, "error": str(error)}
                )
        if phase == "refinement":
            audit = json.loads(
                (output / "results/candidate_refinement/REFINEMENT_AUDIT.json").read_text()
            )
            if audit["further_uniform_refinement_required"]:
                for command in [
                    ("candidate_runs.py", "plan_further"),
                    ("candidate_runs.py", "further"),
                    ("strict_warning_audit.py", "CANDIDATE_FURTHER_MANIFEST.csv"),
                    ("score_new_controls.py", "candidate_further"),
                ]:
                    run(*command)
            run("finalize_grid.py")
        if phase in {"refinement", "placement", "strong_baselines", "hvac"}:
            run("archive_sql_outputs.py")
        failed_here = [x for x in independent_failures if x["phase"] == phase]
        execution[phase]["execution_status"] = "partial" if failed_here else "completed"
        execution[phase]["failed_commands"] = failed_here
        execution[phase]["result_status_source"] = (
            "audit/status and each analysis quality report; completion of execution does not imply that every operation was feasible or every engineering condition passed."
        )
        status_file.write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        print(phase, "finished; inspect phase quality reports", flush=True)
    if independent_failures:
        raise SystemExit(
            "Independent HVAC conditions retained failures; other phases completed. Inspect PHASE_EXECUTION_STATUS.json."
        )


if __name__ == "__main__":
    main()
