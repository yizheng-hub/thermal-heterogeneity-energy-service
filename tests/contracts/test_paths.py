from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import sys

import pytest
import yaml

from thermal_heterogeneity.config import load_config, resolve_asset


def test_absolute_asset_path_is_rejected(tmp_path: Path):
    config_path = tmp_path / "paper.yaml"
    config_path.write_text(
        yaml.safe_dump({"assets": {"x": str(tmp_path / "x.csv")}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="relative"):
        resolve_asset(load_config(config_path), "x")


@pytest.mark.parametrize("records,expected", [(20, "PASS"), (19, "FAIL")])
def test_demographic_summary_disclosure_threshold(tmp_path, records, expected):
    root = Path(__file__).resolve().parents[2]
    path = (
        tmp_path
        / "publication"
        / "support"
        / "science_audit"
        / "demographic_city_season_gender_summary.csv"
    )
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gender", "records", "analysis_units"])
        writer.writeheader()
        writer.writerow({"gender": "female", "records": records, "analysis_units": 10})
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "release_scan.py"),
            "--root",
            str(tmp_path),
            "--mode",
            "paths-identifiers",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    )
    assert json.loads(result.stdout)["status"] == expected


@pytest.mark.parametrize("change", ["none", "tampered_payload", "extra_file", "tampered_manifest"])
def test_source_building_input_verification(tmp_path, change):
    repository = Path(__file__).resolve().parents[2]
    script = tmp_path / "repository" / "scripts" / "unpack_source_building_data.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(repository / "scripts" / script.name, script)
    metadata = script.parent.parent / "data"
    metadata.mkdir()
    payload = tmp_path / "input"
    payload.mkdir()
    content = b"frozen input\n"
    (payload / "input.txt").write_bytes(content)
    manifest = metadata / "source_building_bundle_manifest.csv"
    manifest.write_text(
        "path,sha256,bytes\ninput.txt,"
        + hashlib.sha256(content).hexdigest()
        + f",{len(content)}\n",
        encoding="utf-8",
    )
    (metadata / "source_building_bundle_source.json").write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "hourly_trajectories": 0,
            }
        ),
        encoding="utf-8",
    )
    if change == "tampered_payload":
        (payload / "input.txt").write_bytes(b"altered input\n")
    elif change == "extra_file":
        (payload / "extra.txt").write_bytes(content)
    elif change == "tampered_manifest":
        manifest.write_text("path,sha256,bytes\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(payload)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (change == "none")
    if change == "none":
        assert json.loads(result.stdout)["files_verified"] == 1
