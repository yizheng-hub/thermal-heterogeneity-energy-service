"""A portable package must not certify unfinished or modified physical results."""
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/utilities"))
from package_ctcd_parallel import checked_manifest, preserve_runs, sha  # noqa: E402


def test_unfinished_execution_cannot_be_frozen(tmp_path):
    with pytest.raises(ValueError, match="final summary"):
        checked_manifest(tmp_path, "annual", 864)
    assert not (tmp_path / "MANIFEST.csv").exists()


def test_duplicate_logical_plan_cannot_be_frozen(tmp_path):
    (tmp_path / "engineering").mkdir()
    row = {"context": "city", "composition_id": 1, "layout": "random_mixing",
           "permission": "native_zone", "control": "personal_center", "variant": 0,
           "run_id": "run", "engineering_status": "execution_failed"}
    pd.DataFrame([row, row]).to_csv(tmp_path / "engineering/ANNUAL_RUN_MANIFEST.csv", index=False)
    (tmp_path / "EXECUTION_SUMMARY.json").write_text(json.dumps({"logical_annual_inputs": 2, "unique_inputs": 1}))
    with pytest.raises(ValueError, match="duplicated"):
        checked_manifest(tmp_path, "annual", 2)


def test_rejected_artifact_preservation_keeps_values_and_relative_paths(tmp_path):
    physical, destination = tmp_path / "physical", tmp_path / "package"
    physical.mkdir()
    (destination / "hourly").mkdir(parents=True)
    (destination / "qa").mkdir()
    qa = physical / "decision.json"
    qa.write_text(json.dumps({"engineering_pass": False}))
    source = pd.DataFrame([{"run_id": "rejected", "path_hourly": "", "hourly_sha256": "",
                            "qa_path": str(qa), "qa_sha256": sha(qa),
                            "cached_qa_path": "old_machine_path", "engineering_status": "cached_rejected"}])
    result = preserve_runs(source, physical, destination)
    assert result.iloc[0].qa_path == "../../qa/rejected.json"
    assert result.iloc[0].path_hourly == ""
    assert result.iloc[0].engineering_status == "cached_rejected"
    assert "cached_qa_path" not in result
    assert sha(destination / "qa/rejected.json") == sha(qa)
    assert source.iloc[0].qa_path == str(qa)
