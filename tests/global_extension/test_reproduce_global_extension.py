from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "src/analysis/global_extension/reproduce_global_extension.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "reproduce_global_extension_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def fixture_bundle(tmp_path: Path, module):
    source = tmp_path / "global_extension"
    story = source / "final_story"
    implementation = source / "implementation_value_final"
    assumptions = tmp_path / "assumptions.json"
    write(assumptions, "{}\n")

    input_hashes = {}
    for key, relative in module.FINAL_STORY_INPUTS.items():
        path = source / relative
        write(path, f"{key}\n")
        input_hashes[key] = digest(path)

    summary = {
        "status": "PASS",
        "design": {"candidates": 810, "climates": 9, "compositions": 6},
        "seasonal_reallocation": {"pairs": 216},
        "placement_and_permissions": {
            "new_logical_designs": 540,
            "new_physical_runs": 516,
            "matched_effect_rows": 972,
        },
        "event_response_and_recovery": {
            "protocol_events": 72,
            "action_trajectories": 810,
            "scored_rows": 32400,
            "service_allowance_sensitivity": {"1": {"evaluated_rows": 6480}},
        },
    }
    story_outputs = {}
    for name in module.REQUIRED_STORY_OUTPUTS:
        path = story / name
        if name == "FINAL_STORY_SUMMARY.json":
            write(path, json.dumps(summary))
        else:
            write(path, f"{name}\n")
        story_outputs[name] = digest(path)
    write(
        story / "FINAL_STORY_HASHES.json",
        json.dumps({"status": "PASS", "inputs": input_hashes, "outputs": story_outputs}),
    )

    write(
        source
        / "information_compression_final/GLOBAL_INFORMATION_COMPRESSION_SUMMARY.json",
        "{}\n",
    )
    implementation_inputs = {
        "direct": digest(story / "FINAL_DIRECT_COMPARISONS.csv"),
        "assumptions": digest(assumptions),
        "information_summary": digest(
            source
            / "information_compression_final/GLOBAL_INFORMATION_COMPRESSION_SUMMARY.json"
        ),
        "permission_comparisons": digest(
            source / "spatial_final/GLOBAL_PERMISSION_COMPARISONS.csv"
        ),
        "placement_comparisons": digest(
            source / "spatial_final/GLOBAL_PLACEMENT_COMPARISONS.csv"
        ),
    }
    implementation_outputs = {}
    for name in module.REQUIRED_IMPLEMENTATION_OUTPUTS:
        path = implementation / name
        write(path, f"{name}\n")
        implementation_outputs[name] = digest(path)
    audit = {
        "status": "PASS",
        **module.EXPECTED_IMPLEMENTATION,
        "inputs": implementation_inputs,
        "outputs": implementation_outputs,
    }
    write(
        implementation / "IMPLEMENTATION_VALUE_AUDIT.json",
        json.dumps(audit),
    )
    return source, story, implementation, assumptions


def test_plan_declares_only_honest_working_modes(tmp_path):
    module = load_module()
    report = module.plan("reassemble", tmp_path / "input", tmp_path / "output")
    assert report["implemented_modes"] == ["verify", "reassemble"]
    assert report["analysis_mode_available"] is False
    assert report["steps"] == list(module.REASSEMBLY_STEPS)
    assert "src/analysis/global_extension/analyse_global_results.py" in report[
        "full_analysis_dag"
    ]


def test_analysis_mode_stops_without_a_declared_trajectory_bundle(monkeypatch):
    module = load_module()
    monkeypatch.setattr(sys, "argv", [str(MODULE_PATH), "--mode", "analysis"])
    with pytest.raises(RuntimeError, match="portable retained-trajectory"):
        module.main()


def test_verify_checks_hashes_and_final_denominators(tmp_path):
    module = load_module()
    source, story, implementation, assumptions = fixture_bundle(tmp_path, module)
    report = module.verify(source, story, implementation, assumptions)
    assert report["status"] == "PASS"
    assert report["story"] == module.EXPECTED_STORY
    assert report["implementation"] == module.EXPECTED_IMPLEMENTATION


def test_verify_rejects_a_changed_manifested_output(tmp_path):
    module = load_module()
    source, story, implementation, assumptions = fixture_bundle(tmp_path, module)
    write(story / "FINAL_DIRECT_COMPARISONS.csv", "changed\n")
    with pytest.raises(ValueError, match="Hash mismatch"):
        module.verify(source, story, implementation, assumptions)


def test_verify_rejects_an_old_four_context_denominator(tmp_path):
    module = load_module()
    source, story, implementation, assumptions = fixture_bundle(tmp_path, module)
    summary_path = story / "FINAL_STORY_SUMMARY.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["design"]["candidates"] = 643
    write(summary_path, json.dumps(summary))
    manifest_path = story / "FINAL_STORY_HASHES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["FINAL_STORY_SUMMARY.json"] = digest(summary_path)
    write(manifest_path, json.dumps(manifest))
    with pytest.raises(ValueError, match="denominator gate"):
        module.verify(source, story, implementation, assumptions)


def test_reassembly_commands_use_only_global_final_sources(tmp_path):
    module = load_module()
    commands = module.reassembly_commands(
        tmp_path / "input", tmp_path / "output", tmp_path / "assumptions.json"
    )
    joined = "\n".join(" ".join(command) for command in commands)
    assert "final_story_summary.py" in joined
    assert "implementation_value.py" in joined
    assert "global_final" in joined
