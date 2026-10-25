"""Only the two independently locked supplements leave the original inventory."""
import csv
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def unpacker_fixture(tmp_path, monkeypatch):
    spec = spec_from_file_location("paper_input_inventory", ROOT / "src/utilities/unpack_paper_data.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo"
    metadata = repo / "data/metadata"
    metadata.mkdir(parents=True)
    inputs = repo / "data/inputs"
    payload = inputs / "energy_service/control.csv"
    payload.parent.mkdir(parents=True)
    payload.write_text("value\n1\n")
    manifest = metadata / "paper_bundle_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow({"path": "energy_service/control.csv", "bytes": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()})
    (metadata / "paper_bundle_source.json").write_text(json.dumps({
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "hourly_trajectories": 0}))
    monkeypatch.setattr(module, "__file__", str(repo / "src/utilities/unpack_paper_data.py"))
    monkeypatch.setattr(sys, "argv", ["unpack_paper_data.py", "--root", str(inputs)])
    for name in ("ctcd_external", "ctcd_parallel"):
        folder = inputs / name
        folder.mkdir()
        (folder / "separately_locked.csv").write_text("supplement\n1\n")
    return module, inputs


def test_two_separately_locked_packages_do_not_change_the_base_inventory(tmp_path, monkeypatch, capsys):
    module, _ = unpacker_fixture(tmp_path, monkeypatch)
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS" and report["files_verified"] == 1


@pytest.mark.parametrize("extra", ["undeclared.csv", "energy_service/extra.csv", "ctcd_parallel_other/extra.csv"])
def test_unknown_or_similarly_named_extra_file_still_fails(tmp_path, monkeypatch, extra):
    module, inputs = unpacker_fixture(tmp_path, monkeypatch)
    path = inputs / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("undeclared\n1\n")
    with pytest.raises(ValueError, match="file set differs"):
        module.main()
