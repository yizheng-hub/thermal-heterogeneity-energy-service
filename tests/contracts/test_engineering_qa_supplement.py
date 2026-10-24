"""Engineering sidecar privacy and original-decision interpretation contracts."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("engineering_sidecar", ROOT / "scripts/engineering_qa_supplement.py")
SIDECAR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIDECAR)


def test_qa_sanitization_drops_paths_but_keeps_original_engineering_evidence(tmp_path):
    source = {"folder": str(tmp_path / "run"), "engineering_pass": False,
              "warnings": [{"message": "Coil operating-range warning", "occurrences": 19,
                            "decision": "requires_review"}],
              "details": {"source_folder": str(tmp_path / "source"), "maximum_error": .02}}
    result = SIDECAR.sanitize(source)
    assert "folder" not in result and "source_folder" not in result["details"]
    assert result["warnings"] == source["warnings"]
    assert result["details"]["maximum_error"] == .02
    assert result["engineering_pass"] is False


def test_qa_reason_labels_index_recorded_failure_without_readmitting_runs():
    numerical = {"engineering_pass": False, "direct_numerical_warnings": ["Solver warning"]}
    assert SIDECAR.reason_classes(numerical, rejected=True) == ["direct_hvac_numerical_warning"]
    unresolved = {"engineering_pass": False,
                  "all_recurring_warnings": [{"decision": "requires_review"}]}
    assert SIDECAR.reason_classes(unresolved, rejected=True) == ["unresolved_recurring_warning"]
    assert SIDECAR.reason_classes(unresolved, rejected=False) == ["recorded_qualified"]
    assert "engineering_pass=false" in SIDECAR.recorded_triggers(unresolved)
