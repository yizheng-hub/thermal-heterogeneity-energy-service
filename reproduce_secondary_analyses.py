"""Rebuild existing annual and response diagnostics from a fresh raw-input run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
GLOBAL = ROOT / "src/analysis/global_extension"


def commands(raw):
    results = raw / "results"
    annual = results / "global_final"
    matrix = raw / "matrix"
    profiles = matrix / "GLOBAL_PROFILES.csv"
    runs = matrix / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv"
    probes = matrix / "probe_runs/PROBE_RUN_MANIFEST.csv"
    metrics = annual / "GLOBAL_RUN_METRICS.csv"
    anchors = annual / "GLOBAL_25C_SERVICE_ANCHORS.csv"
    response = results / "multi_event_dr_final"
    scores = response / "EVENT_RESPONSE_SCORES.csv"

    def call(script, *args):
        return [sys.executable, str(GLOBAL / script), *map(str, args)]

    return {
        "spatial_summary": call(
            "summarize_reproduced_spatial.py",
            "--raw-run-root",
            raw,
            "--output-root",
            ROOT / "results/global_extension/final_story",
        ),
        "decision_effect_sizes": call(
            "decision_effect_sizes.py",
            "--source",
            annual,
            "--output",
            results / "decision_effect_sizes",
        ),
        "decision_tolerance": call(
            "decision_tolerance_sensitivity.py",
            "--source",
            annual,
            "--output",
            results / "decision_tolerance_sensitivity",
        ),
        "matched_service": call(
            "matched_service_sensitivity.py",
            "--metrics",
            metrics,
            "--output-root",
            results / "matched_service_final",
        ),
        "tail_fractions": call(
            "tail_fraction_sensitivity.py",
            "--metrics",
            metrics,
            "--run-manifest",
            runs,
            "--profiles",
            profiles,
            "--anchor-manifest",
            probes,
            "--output-root",
            results / "tail_fraction_sensitivity",
        ),
        "reference_anchors": call(
            "reference_anchor_sensitivity.py",
            "--metrics",
            metrics,
            "--run-manifest",
            runs,
            "--profiles",
            profiles,
            "--anchor-manifest",
            probes,
            "--output-root",
            results / "reference_anchor_sensitivity",
        ),
        "target_domain": call(
            "target_domain_sensitivity.py",
            "--metrics",
            metrics,
            "--anchors",
            anchors,
            "--run-manifest",
            runs,
            "--commands",
            matrix / "control_generation/controller_input/CONTROL_COMMANDS.json",
            "--controller-root",
            matrix / "control_generation/controller_input",
            "--output-root",
            results / "target_domain_final",
        ),
        "information_compression": call(
            "global_information_compression.py",
            "--metrics",
            metrics,
            "--anchors",
            anchors,
            "--tails",
            annual / "GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
            "--profiles",
            profiles,
            "--output-root",
            results / "information_compression_final",
        ),
        "hvac_end_uses": call(
            "hvac_end_use_mechanism.py",
            "--direct",
            annual / "GLOBAL_DIRECT_COMPARISONS.csv",
            "--runs",
            metrics,
            "--output",
            results / "hvac_end_use_mechanism",
        ),
        "energy_boundary": call(
            "energy_boundary_sensitivity.py",
            "--metrics",
            metrics,
            "--anchors",
            anchors,
            "--seasonal-metrics",
            results
            / "seasonal_facility_energy_final/SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv",
            "--output-root",
            results / "energy_boundary_final",
        ),
        "recovery_residuals": call(
            "audit_event_terminal_recovery.py",
            "--run-manifest",
            raw / "response/EVENT_RUN_MANIFEST_COMPLETED.csv",
            "--event-manifest",
            raw / "response/EVENT_EVALUATION_MANIFEST.csv",
            "--output-root",
            response,
        ),
        "response_summary": call(
            "summarise_multi_event_dr.py",
            "--scores",
            scores,
            "--recovery-residuals",
            response / "EVENT_TERMINAL_RECOVERY_RESIDUALS.csv",
            "--event-manifest",
            raw / "response/EVENT_EVALUATION_MANIFEST.csv",
            "--completed-run-manifest",
            raw / "response/EVENT_RUN_MANIFEST_COMPLETED.csv",
            "--output-root",
            response / "summary",
        ),
        "response_decomposition": call(
            "response_mean_decomposition.py",
            "--scores",
            scores,
            "--output",
            results / "response_mean_decomposition",
        ),
        "response_qualification": call(
            "response_qualification_sensitivity.py",
            "--scores",
            scores,
            "--output",
            results / "response_qualification_sensitivity",
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--analyses", nargs="+")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    raw = args.raw_run_root.resolve()
    if not raw.is_relative_to(ROOT):
        raise ValueError("Secondary analyses must read a raw run inside this project.")
    jobs = commands(raw)
    selected = args.analyses or list(jobs)
    if set(selected) - set(jobs):
        parser.error("Unknown analyses: " + ", ".join(sorted(set(selected) - set(jobs))))
    if args.plan:
        print(json.dumps({k: jobs[k] for k in selected}, indent=2))
        return
    logs = raw / "logs/secondary"
    logs.mkdir(parents=True, exist_ok=True)
    state = raw / "SECONDARY_EXECUTION_STATUS.json"
    status = json.loads(state.read_text()) if state.exists() else {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for name in selected:
        command = jobs[name]
        source = Path(command[1])
        attempt = 1
        log = logs / f"{name}_{attempt}.log"
        while log.exists():
            attempt += 1
            log = log.with_name(f"{name}_{attempt}.log")
        status[name] = {
            "execution_status": "running",
            "command": command,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "log": str(log.relative_to(raw)),
        }
        state.write_text(json.dumps(status, indent=2) + "\n")
        with log.open("wb") as stream:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        status[name].update(
            execution_status="completed" if result.returncode == 0 else "failed",
            returncode=result.returncode,
        )
        state.write_text(json.dumps(status, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} did not complete; inspect {log}")
        print(name, "completed", flush=True)


if __name__ == "__main__":
    main()
