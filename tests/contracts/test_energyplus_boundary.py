from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from thermal_heterogeneity.building.energyplus import (
    LEVEL2_ENGINEERING_GATE_FAILED,
    LEVEL2_INPUT_PACKAGE_INVALID,
    LEVEL2_INPUT_PACKAGE_UNAVAILABLE,
    LEVEL2_READY,
    LEVEL2_RUN_COMPLETE,
    evaluate_engineering_gate,
    prepare_level2_run,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "src/settings" / "energyplus_runs.csv"
REUSE = ROOT / "src/settings" / "energyplus_dynamic_baseline_reuse.csv"
CONTRACT = ROOT / "src/settings" / "energyplus_level2_contract.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_manifest(tmp_path: Path, idf: bytes, epw: bytes) -> tuple[Path, dict[str, str]]:
    idf_sha, epw_sha = _sha(idf), _sha(epw)
    row = {
        "run_id": "annual_synthetic",
        "idf_asset_id": f"idf_{idf_sha}",
        "idf_sha256": idf_sha,
        "idf_package_path": f"idf/idf_{idf_sha}.idf",
        "epw_asset_id": "epw_synthetic",
        "epw_sha256": epw_sha,
        "epw_package_path": "epw/epw_synthetic.epw",
        "rights_package_id": "level2-local-assets-v1",
        "rights_manifest_path": "LEVEL2_RIGHTS.json",
        "energyplus_version": "22.1.0-ed759b17ee",
        "engineering_gate_id": "LEVEL2_TRAJECTORY_GATE_V1",
    }
    path = tmp_path / "runs.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return path, row


def _write_package(root: Path, row: dict[str, str], idf: bytes, epw: bytes) -> None:
    (root / "idf").mkdir(parents=True)
    (root / "epw").mkdir(parents=True)
    (root / row["idf_package_path"]).write_bytes(idf)
    (root / row["epw_package_path"]).write_bytes(epw)
    rights = {
        "package_id": "level2-local-assets-v1",
        "local_use_authorized": True,
        "assets": {
            row["idf_asset_id"]: {"sha256": row["idf_sha256"], "rights_basis": "test fixture"},
            row["epw_asset_id"]: {"sha256": row["epw_sha256"], "rights_basis": "test fixture"},
        },
    }
    (root / "LEVEL2_RIGHTS.json").write_text(json.dumps(rights), encoding="utf-8")


def test_locked_manifest_and_baseline_reuse_contract() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/utilities" / "validate_energyplus_manifest.py"),
            str(MANIFEST),
            "--baseline-reuse",
            str(REUSE),
            "--contract",
            str(CONTRACT),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS"
    assert result["run_class_counts"] == {"annual": 346, "event": 66}
    assert result["dynamic_baseline_reuse_count"] == 33
    assert result["unique_epw_assets"] == 4
    assert result["historical_generator_sha_boundary"]["same"] is False


def test_missing_package_returns_locked_unavailable_status(tmp_path: Path) -> None:
    manifest, _ = _synthetic_manifest(tmp_path, b"idf", b"epw")
    result = prepare_level2_run(manifest, "annual_synthetic", tmp_path / "absent", tmp_path / "out")
    assert result["status"] == LEVEL2_INPUT_PACKAGE_UNAVAILABLE


def test_missing_rights_manifest_returns_locked_unavailable_status(tmp_path: Path) -> None:
    idf, epw = b"expected idf", b"expected epw"
    manifest, row = _synthetic_manifest(tmp_path, idf, epw)
    package = tmp_path / "package"
    (package / "idf").mkdir(parents=True)
    (package / "epw").mkdir(parents=True)
    (package / row["idf_package_path"]).write_bytes(idf)
    (package / row["epw_package_path"]).write_bytes(epw)
    result = prepare_level2_run(manifest, "annual_synthetic", package, tmp_path / "out")
    assert result["status"] == LEVEL2_INPUT_PACKAGE_UNAVAILABLE
    assert "LEVEL2_RIGHTS.json" in result["missing"]


def test_missing_exact_asset_is_not_silently_substituted(tmp_path: Path) -> None:
    manifest, row = _synthetic_manifest(tmp_path, b"expected idf", b"expected epw")
    package = tmp_path / "package"
    package.mkdir()
    (package / "idf").mkdir()
    (package / "epw").mkdir()
    (package / "idf" / "nearby.idf").write_bytes(b"nearby")
    (package / row["epw_package_path"]).write_bytes(b"expected epw")
    (package / "LEVEL2_RIGHTS.json").write_text(
        json.dumps({"package_id": "level2-local-assets-v1", "local_use_authorized": True, "assets": {}}),
        encoding="utf-8",
    )
    result = prepare_level2_run(manifest, "annual_synthetic", package, tmp_path / "out")
    assert result["status"] == LEVEL2_INPUT_PACKAGE_UNAVAILABLE
    assert row["idf_package_path"] in result["missing"]


def test_exact_rights_cleared_assets_prepare_one_command(tmp_path: Path) -> None:
    idf, epw = b"Version,22.1;", b"LOCATION,test"
    manifest, row = _synthetic_manifest(tmp_path, idf, epw)
    package = tmp_path / "package"
    _write_package(package, row, idf, epw)
    result = prepare_level2_run(manifest, "annual_synthetic", package, tmp_path / "out")
    assert result["status"] == LEVEL2_READY
    assert result["argv"][1:5:2] == ["-w", "-d"]
    assert Path(result["argv"][-1]).name == Path(row["idf_package_path"]).name


def test_present_asset_with_wrong_hash_is_invalid(tmp_path: Path) -> None:
    idf, epw = b"expected idf", b"expected epw"
    manifest, row = _synthetic_manifest(tmp_path, idf, epw)
    package = tmp_path / "package"
    _write_package(package, row, idf, epw)
    (package / row["idf_package_path"]).write_bytes(b"tampered")
    result = prepare_level2_run(manifest, "annual_synthetic", package, tmp_path / "out")
    assert result["status"] == LEVEL2_INPUT_PACKAGE_INVALID


def test_engineering_gate_requires_exact_version_completion_and_no_severe(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "eplusout.err").write_text(
        "Program Version,EnergyPlus, Version 22.1.0-ed759b17ee\n", encoding="utf-8"
    )
    (output / "eplusout.end").write_text("EnergyPlus Completed Successfully\n", encoding="utf-8")
    assert evaluate_engineering_gate(output)["status"] == LEVEL2_RUN_COMPLETE
    (output / "eplusout.err").write_text(
        "Program Version,EnergyPlus, Version 22.1.0-ed759b17ee\n** Severe ** failure\n",
        encoding="utf-8",
    )
    assert evaluate_engineering_gate(output)["status"] == LEVEL2_ENGINEERING_GATE_FAILED
