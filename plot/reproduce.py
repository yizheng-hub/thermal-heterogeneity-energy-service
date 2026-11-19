"""Execute actual figure cells in Plot.ipynb without storing notebook outputs."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import os


FINAL_STORY_ROOT = "results/global_extension/final_story"
RAW_FIGURE_ONE_INPUTS = (
    "results/observation_figures/CITY_MAP_POINTS.csv",
    "results/observation_figures/DEMAND_COMPOSITION_PROFILES.csv",
    "results/observation_figures/LEAVE_ONE_SOURCE_VARIANCE.csv",
    "results/observation_figures/POLICY_RANGE_POSITIONS.csv",
    "results/observations/ASHRAE_NESTED_VARIANCE.csv",
    "results/shared_room/SHARED_ROOM_SUMMARY.csv",
)
REQUIRED_FIGURE_INPUTS = {
    "Figure_01": (
        "data/metadata/ctcd_city_map_metadata.csv",
        "src/analysis/global_extension/GLOBAL_CLIMATES.json",
    ),
    "Figure_02": (
        f"{FINAL_STORY_ROOT}/FINAL_STORY_SUMMARY.json",
        f"{FINAL_STORY_ROOT}/FINAL_INFORMATION_DIAGNOSTIC.csv",
        f"{FINAL_STORY_ROOT}/FINAL_DIRECT_COMPARISONS.csv",
        "results/global_extension/tail_fraction_sensitivity/"
        "TAIL_FRACTION_INFORMATION_DIAGNOSTIC.csv",
        "results/global_extension/reference_anchor_sensitivity/"
        "REFERENCE_ANCHOR_HEADLINE.csv",
        "results/global_extension/target_domain_final/"
        "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv",
        "results/global_extension/target_domain_20_26/"
        "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv",
        "results/global_extension/target_domain_21_27/"
        "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv",
        "results/global_extension/target_domain_21_26/"
        "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv",
    ),
    "Figure_03": (
        f"{FINAL_STORY_ROOT}/FINAL_STORY_SUMMARY.json",
        f"{FINAL_STORY_ROOT}/FINAL_DIRECT_COMPARISONS.csv",
        f"{FINAL_STORY_ROOT}/FINAL_MATCHED_SERVICE.csv",
        f"{FINAL_STORY_ROOT}/FINAL_SEASONAL_REALLOCATION.csv",
    ),
    "Figure_04": (
        f"{FINAL_STORY_ROOT}/FINAL_STORY_SUMMARY.json",
        f"{FINAL_STORY_ROOT}/FINAL_SPATIAL_EFFECTS.csv",
    ),
    "Figure_05": (
        f"{FINAL_STORY_ROOT}/FINAL_EVENT_RESPONSE.csv",
        "results/global_extension/multi_event_dr_final/summary/"
        "EVENT_RESPONSE_GROUP_SUMMARY.csv",
        "results/global_extension/multi_event_dr_final/summary/"
        "EVENT_RESPONSE_PAIRED_CONTROL_SUMMARY.csv",
        "results/global_extension/multi_event_dr_final/summary/"
        "EVENT_TERMINAL_RECOVERY_GROUP_SUMMARY.csv",
        "results/global_extension/multi_event_dr_final/summary/"
        "EVENT_RESPONSE_SUMMARY.json",
    ),
    "Figure_06": (
        "results/global_extension/implementation_value_final/"
        "GLOBAL_SERVICE_RESOURCE_VALUE_BOUNDARIES.csv",
        "results/global_extension/implementation_value_final/"
        "GLOBAL_SPATIAL_IMPLEMENTATION_BOUNDARIES.csv",
        "results/global_extension/implementation_value_final/"
        "IMPLEMENTATION_RESOURCE_SCENARIOS.csv",
        "results/global_extension/implementation_value_final/"
        "IMPLEMENTATION_RESOURCE_CATALOG.csv",
        "results/global_extension/implementation_value_final/"
        "COMMAND_SCOPE_REQUIREMENTS.csv",
        "results/global_extension/implementation_value_final/"
        "INFORMATION_BURDEN_SUMMARY.csv",
        "results/global_extension/implementation_value_final/"
        "IMPLEMENTATION_VALUE_AUDIT.json",
    ),
}

def validate_legend_styles(notebook, active_labels):
    violations = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        label = cell.get("metadata", {}).get("figure")
        if label not in active_labels:
            continue
        source = "".join(cell.get("source", []))
        tree = ast.parse(source, filename=f"Plot.ipynb:{label}")
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "legend"
            ):
                continue
            frameon = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "frameon"),
                None,
            )
            if not (isinstance(frameon, ast.Constant) and frameon.value is False):
                violations.append(f"{label}: line {node.lineno}")
                continue
            if label and label.startswith("Figure_0"):
                for keyword_name in ("facecolor", "edgecolor"):
                    value = next(
                        (keyword.value for keyword in node.keywords
                         if keyword.arg == keyword_name),
                        None,
                    )
                    if not (
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value.lower() == "none"
                    ):
                        violations.append(
                            f"{label}: line {node.lineno} missing {keyword_name}='none'"
                        )
    if violations:
        raise ValueError(
            "Publication legends must be frameless and transparent; add frameon=False, "
            "facecolor='none' and edgecolor='none': "
            + ", ".join(violations)
        )


def validate_final_story_hashes(root, active_labels):
    if not active_labels.intersection(
        {"Figure_02", "Figure_03", "Figure_04", "Figure_05"}
    ):
        return
    story_root = root / FINAL_STORY_ROOT
    manifest_path = story_root / "FINAL_STORY_HASHES.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Final-story hash manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or not isinstance(manifest.get("outputs"), dict):
        raise ValueError("Final-story hash manifest is not a passing output manifest")
    for name, declared in manifest["outputs"].items():
        path = story_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Manifested final-story output is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != declared:
            raise ValueError(f"Final-story output changed after assembly: {path}")
    summary = json.loads(
        (story_root / "FINAL_STORY_SUMMARY.json").read_text(encoding="utf-8")
    )
    design = summary.get("design", {})
    if design.get("climates") != 9 or design.get("candidates") != 810:
        raise ValueError("Final-story summary is not the complete nine-climate design")


def validate_figure_inputs(root, active_labels, raw_run_root=None):
    missing = []
    for label in active_labels:
        for relative in REQUIRED_FIGURE_INPUTS.get(label, ()):
            if not (root / relative).is_file():
                missing.append((label, relative))
    if missing:
        detail = ", ".join(f"{label}: {relative}" for label, relative in missing)
        raise FileNotFoundError(f"Required figure input is missing: {detail}")
    if "Figure_01" in active_labels:
        raw = raw_run_root or root / "output/from_raw"
        for relative in RAW_FIGURE_ONE_INPUTS:
            if not (raw / relative).is_file():
                raise FileNotFoundError(f"Run the raw demand and observations stages first: {raw / relative}")
    validate_final_story_hashes(root, active_labels)


def main():
    # All drawing code and style settings live in the notebook's figure cells.
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", nargs="+")
    parser.add_argument("--raw-run-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "reproduce_publication.py")]
    if args.figure:
        command.extend(["--figure", *args.figure])
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
