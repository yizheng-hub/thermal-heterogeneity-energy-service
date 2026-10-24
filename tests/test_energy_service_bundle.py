"""Check fail-closed portable input verification on temporary synthetic files."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "src/analysis/energy_service/bundle.py"
SPEC = importlib.util.spec_from_file_location("energy_service_bundle", MODULE)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


def fixture(root):
    payload = root / "small.csv"
    payload.write_text("hour,energy_kwh\n0,1\n", encoding="utf-8")
    manifest = root / "MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow({"path": "small.csv", "bytes": payload.stat().st_size,
                         "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()})
    return payload, manifest


def test_verified_complete_bundle(tmp_path):
    _, manifest = fixture(tmp_path)
    assert bundle.verify_bundle(tmp_path, manifest)["verified_input_files"] == 1


def test_mutated_payload_is_rejected(tmp_path):
    payload, manifest = fixture(tmp_path)
    payload.write_text("hour,energy_kwh\n0,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        bundle.verify_bundle(tmp_path, manifest)


def test_extra_file_cannot_bypass_minimization(tmp_path):
    _, manifest = fixture(tmp_path)
    (tmp_path / "undeclared.csv").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared"):
        bundle.verify_bundle(tmp_path, manifest)


def test_parent_escape_is_rejected(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest.write_text("path,bytes,sha256\n../outside.csv,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contained"):
        bundle.verify_bundle(tmp_path, manifest)


def test_duplicate_manifest_path_is_rejected(tmp_path):
    _, manifest = fixture(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text + text.splitlines()[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        bundle.verify_bundle(tmp_path, manifest)
