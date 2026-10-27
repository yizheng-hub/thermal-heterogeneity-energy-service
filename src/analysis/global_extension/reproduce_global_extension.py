"""Verify or reassemble the final nine-climate publication evidence.

This is the single reporting-layer entry point for the global extension.  It
does not run EnergyPlus.  ``verify`` checks the retained final tables, their
hashes and their declared denominators.  ``reassemble`` rebuilds the final
story and implementation tables from the retained analysis tables and then
applies the same checks.

The complete trajectory-rescoring DAG is declared in ``--plan`` but is not
exposed as a working mode until a portable, checksum-indexed compact-
trajectory bundle is supplied.  This prevents a final-table replay from being
misrepresented as a full numerical reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve()
REPOSITORY_ROOT = HERE.parents[3]

FINAL_STORY_INPUTS = {
    "annual_metrics": "global_final/GLOBAL_RUN_METRICS.csv",
    "direct": "global_final/GLOBAL_DIRECT_COMPARISONS.csv",
    "information": "global_final/GLOBAL_INFORMATION_DIAGNOSTIC.csv",
    "strict": "global_final/GLOBAL_STRICT_SERVICE_FEASIBILITY.csv",
    "matched": "matched_service_final/MATCHED_REALIZED_SERVICE_CLOSEST_PAIRS.csv",
    "energy_boundary": (
        "energy_boundary_final/GLOBAL_ENERGY_BOUNDARY_SENSITIVITY.json"
    ),
    "seasonal_source_blocks": (
        "seasonal_command/final/SEASONAL_COMMAND_SEASON_METRICS.csv"
    ),
    "seasonal_annual": (
        "seasonal_facility_energy_final/"
        "SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv"
    ),
    "seasonal_blocks": (
        "seasonal_facility_energy_final/"
        "SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv"
    ),
    "seasonal_energy_audit": (
        "seasonal_facility_energy_final/SEASONAL_FACILITY_ENERGY_AUDIT.json"
    ),
    "spatial_permission": "spatial_final/GLOBAL_PERMISSION_COMPARISONS.csv",
    "spatial_placement": "spatial_final/GLOBAL_PLACEMENT_COMPARISONS.csv",
    "spatial_audit": "spatial_final/GLOBAL_SPATIAL_AUDIT.json",
    "event_scores": "multi_event_dr_final/EVENT_RESPONSE_SCORES.csv",
    "event_protocol": "multi_event_dr_final/EVENT_PROTOCOL.csv",
    "personal_range": "personal_range_validation/PERSONAL_RANGE_VALIDATION.json",
}

REASSEMBLY_STEPS = (
    "src/analysis/global_extension/final_story_summary.py",
    "src/analysis/global_extension/implementation_value.py",
    "hash and denominator verification",
)

FULL_ANALYSIS_DAG = (
    "src/analysis/global_extension/analyse_global_results.py",
    "src/analysis/global_extension/matched_service_sensitivity.py",
    "src/analysis/global_extension/energy_boundary_sensitivity.py",
    "src/analysis/global_extension/tail_fraction_sensitivity.py",
    "src/analysis/global_extension/reference_anchor_sensitivity.py",
    "src/analysis/global_extension/target_domain_sensitivity.py",
    "src/analysis/global_extension/seasonal_command_extension.py score",
    "src/analysis/global_extension/seasonal_facility_energy.py",
    "src/analysis/global_extension/global_spatial_extension.py score",
    "src/analysis/global_extension/multi_event_dr_extension.py score",
    "src/analysis/global_extension/audit_event_terminal_recovery.py",
    "src/analysis/global_extension/summarise_multi_event_dr.py",
    "src/analysis/validation/personal_range_validation.py",
    "src/analysis/global_extension/fixed_range_endpoint_sensitivity.py",
    "src/analysis/global_extension/global_information_compression.py",
    "src/analysis/global_extension/implementation_value.py",
    "src/analysis/global_extension/final_story_summary.py",
)

EXPECTED_STORY = {
    "annual_candidates": 810,
    "climates": 9,
    "compositions": 6,
    "seasonal_pairs": 216,
    "spatial_logical_designs": 540,
    "spatial_physical_runs": 516,
    "spatial_effect_rows": 972,
    "event_protocols": 72,
    "event_action_trajectories": 810,
    "event_score_rows": 32400,
    "strict_event_rows": 6480,
}

EXPECTED_IMPLEMENTATION = {
    "direct_comparisons": 108,
    "implementation_scenarios": 216,
    "implementation_resource_categories": 6,
    "value_boundary_rows": 23328,
    "spatial_resource_rows": 972,
}

REQUIRED_STORY_OUTPUTS = {
    "FINAL_DIRECT_COMPARISONS.csv",
    "FINAL_EVENT_RESPONSE.csv",
    "FINAL_EVENT_RESPONSE_SUMMARY.csv",
    "FINAL_INFORMATION_DIAGNOSTIC.csv",
    "FINAL_MATCHED_SERVICE.csv",
    "FINAL_SEASONAL_REALLOCATION.csv",
    "FINAL_SPATIAL_EFFECTS.csv",
    "FINAL_STORY_SUMMARY.json",
    "FINAL_STRICT_SERVICE_FEASIBILITY.csv",
}

REQUIRED_IMPLEMENTATION_OUTPUTS = {
    "IMPLEMENTATION_RESOURCE_SCENARIOS.csv",
    "IMPLEMENTATION_RESOURCE_CATALOG.csv",
    "IMPLEMENTATION_EVIDENCE_SOURCES.csv",
    "GLOBAL_SERVICE_RESOURCE_VALUE_BOUNDARIES.csv",
    "COMMAND_SCOPE_REQUIREMENTS.csv",
    "INFORMATION_BURDEN_SUMMARY.csv",
    "GLOBAL_SPATIAL_IMPLEMENTATION_BOUNDARIES.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Hash mismatch for {label}: expected {expected}, found {actual}"
        )


def verify_story(
    source_root: Path,
    story_root: Path,
) -> dict[str, Any]:
    manifest = read_json(story_root / "FINAL_STORY_HASHES.json")
    if manifest.get("status") != "PASS":
        raise ValueError("Final-story hash manifest is not PASS")
    if set(manifest.get("inputs", {})) != set(FINAL_STORY_INPUTS):
        raise ValueError("Final-story manifest does not declare the complete input set")
    if set(manifest.get("outputs", {})) != REQUIRED_STORY_OUTPUTS:
        raise ValueError("Final-story manifest does not declare the complete output set")
    for name, expected in manifest.get("inputs", {}).items():
        relative = FINAL_STORY_INPUTS.get(name)
        if relative is None:
            raise ValueError(f"Unknown final-story input key: {name}")
        check_hash(source_root / relative, str(expected), f"story input {name}")
    for name, expected in manifest.get("outputs", {}).items():
        check_hash(story_root / name, str(expected), f"story output {name}")

    summary = read_json(story_root / "FINAL_STORY_SUMMARY.json")
    if summary.get("status") != "PASS":
        raise ValueError("Final story is not PASS")
    design = summary.get("design", {})
    seasonal = summary.get("seasonal_reallocation", {})
    spatial = summary.get("placement_and_permissions", {})
    events = summary.get("event_response_and_recovery", {})
    strict = events.get("service_allowance_sensitivity", {}).get("1", {})
    observed = {
        "annual_candidates": design.get("candidates"),
        "climates": design.get("climates"),
        "compositions": design.get("compositions"),
        "seasonal_pairs": seasonal.get("pairs"),
        "spatial_logical_designs": spatial.get("new_logical_designs"),
        "spatial_physical_runs": spatial.get("new_physical_runs"),
        "spatial_effect_rows": spatial.get("matched_effect_rows"),
        "event_protocols": events.get("protocol_events"),
        "event_action_trajectories": events.get("action_trajectories"),
        "event_score_rows": events.get("scored_rows"),
        "strict_event_rows": strict.get("evaluated_rows"),
    }
    if observed != EXPECTED_STORY:
        raise ValueError(
            "Final-story denominator gate failed: "
            f"expected {EXPECTED_STORY}, found {observed}"
        )
    return observed


def verify_implementation(
    source_root: Path,
    story_root: Path,
    implementation_root: Path,
    assumptions_path: Path,
) -> dict[str, Any]:
    audit = read_json(implementation_root / "IMPLEMENTATION_VALUE_AUDIT.json")
    if audit.get("status") != "PASS":
        raise ValueError("Implementation-value audit is not PASS")
    observed = {name: audit.get(name) for name in EXPECTED_IMPLEMENTATION}
    if observed != EXPECTED_IMPLEMENTATION:
        raise ValueError(
            "Implementation denominator gate failed: "
            f"expected {EXPECTED_IMPLEMENTATION}, found {observed}"
        )
    expected_inputs = {
        "direct": story_root / "FINAL_DIRECT_COMPARISONS.csv",
        "assumptions": assumptions_path,
        "information_summary": (
            source_root
            / "information_compression_final/GLOBAL_INFORMATION_COMPRESSION_SUMMARY.json"
        ),
        "permission_comparisons": (
            source_root / "spatial_final/GLOBAL_PERMISSION_COMPARISONS.csv"
        ),
        "placement_comparisons": (
            source_root / "spatial_final/GLOBAL_PLACEMENT_COMPARISONS.csv"
        ),
    }
    if set(audit.get("inputs", {})) != set(expected_inputs):
        raise ValueError("Implementation audit does not declare the complete input set")
    if set(audit.get("outputs", {})) != REQUIRED_IMPLEMENTATION_OUTPUTS:
        raise ValueError("Implementation audit does not declare the complete output set")
    for name, expected in audit.get("inputs", {}).items():
        if name not in expected_inputs:
            raise ValueError(f"Unknown implementation input key: {name}")
        check_hash(expected_inputs[name], str(expected), f"implementation input {name}")
    for name, expected in audit.get("outputs", {}).items():
        check_hash(
            implementation_root / name,
            str(expected),
            f"implementation output {name}",
        )
    return observed


def verify(
    source_root: Path,
    story_root: Path,
    implementation_root: Path,
    assumptions_path: Path,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "story": verify_story(source_root, story_root),
        "implementation": verify_implementation(
            source_root,
            story_root,
            implementation_root,
            assumptions_path,
        ),
    }


def reassembly_commands(
    source_root: Path,
    output_root: Path,
    assumptions_path: Path,
) -> list[list[str]]:
    story_root = output_root / "final_story"
    implementation_root = output_root / "implementation_value_final"
    return [
        [
            sys.executable,
            str(HERE.parent / "final_story_summary.py"),
            "--annual-root",
            str(source_root / "global_final"),
            "--matched-root",
            str(source_root / "matched_service_final"),
            "--energy-boundary-root",
            str(source_root / "energy_boundary_final"),
            "--seasonal-root",
            str(source_root / "seasonal_command/final"),
            "--seasonal-energy-root",
            str(source_root / "seasonal_facility_energy_final"),
            "--spatial-root",
            str(source_root / "spatial_final"),
            "--event-scores",
            str(source_root / "multi_event_dr_final/EVENT_RESPONSE_SCORES.csv"),
            "--event-protocol",
            str(source_root / "multi_event_dr_final/EVENT_PROTOCOL.csv"),
            "--personal-range-audit",
            str(
                source_root
                / "personal_range_validation/PERSONAL_RANGE_VALIDATION.json"
            ),
            "--output-root",
            str(story_root),
        ],
        [
            sys.executable,
            str(HERE.parent / "implementation_value.py"),
            "--direct-comparisons",
            str(story_root / "FINAL_DIRECT_COMPARISONS.csv"),
            "--assumptions",
            str(assumptions_path),
            "--information-summary",
            str(
                source_root
                / "information_compression_final/"
                "GLOBAL_INFORMATION_COMPRESSION_SUMMARY.json"
            ),
            "--permission-comparisons",
            str(source_root / "spatial_final/GLOBAL_PERMISSION_COMPARISONS.csv"),
            "--placement-comparisons",
            str(source_root / "spatial_final/GLOBAL_PLACEMENT_COMPARISONS.csv"),
            "--output-root",
            str(implementation_root),
        ],
    ]


def reassemble(
    source_root: Path,
    output_root: Path,
    assumptions_path: Path,
) -> dict[str, Any]:
    for command in reassembly_commands(source_root, output_root, assumptions_path):
        subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=True,
        )
    return verify(
        source_root,
        output_root / "final_story",
        output_root / "implementation_value_final",
        assumptions_path,
    )


def plan(mode: str, input_root: Path, output_root: Path) -> dict[str, Any]:
    steps: list[str]
    if mode == "verify":
        steps = ["hash and denominator verification"]
    elif mode == "reassemble":
        steps = list(REASSEMBLY_STEPS)
    else:
        steps = list(FULL_ANALYSIS_DAG)
    return {
        "status": "PLAN",
        "mode": mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "implemented_modes": ["verify", "reassemble"],
        "analysis_mode_available": False,
        "analysis_blocker": (
            "A portable checksum-indexed bundle of the retained annual, seasonal, "
            "spatial and event compact trajectories has not yet been declared."
        ),
        "steps": steps,
        "full_analysis_dag": list(FULL_ANALYSIS_DAG),
        "denominator_gates": EXPECTED_STORY | EXPECTED_IMPLEMENTATION,
        "requires_energyplus": False if mode in {"verify", "reassemble"} else None,
        "manuscripts_required": False,
        "word_templates_required": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("verify", "reassemble", "analysis"),
        default="verify",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPOSITORY_ROOT / "data/inputs/final_analysis/global_extension",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "results/global_extension",
    )
    parser.add_argument(
        "--assumptions",
        type=Path,
        default=HERE.parent / "GLOBAL_IMPLEMENTATION_VALUE_ASSUMPTIONS.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    assumptions = args.assumptions.resolve()
    if args.plan:
        print(json.dumps(plan(args.mode, input_root, output_root), indent=2))
        return 0
    if args.mode == "analysis":
        raise RuntimeError(
            "Full trajectory analysis is not enabled: the portable retained-trajectory "
            "input manifest has not been declared. Use --plan to inspect the DAG, or "
            "use --mode verify/reassemble for the final retained tables."
        )
    if args.mode == "verify":
        report = verify(
            input_root,
            input_root / "final_story",
            input_root / "implementation_value_final",
            assumptions,
        )
    else:
        report = reassemble(input_root, output_root, assumptions)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
