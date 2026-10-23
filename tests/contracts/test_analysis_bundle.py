from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from thermal_heterogeneity.release_bundle import (
    build_bundle,
    fetch_bundle,
    sha256_file,
    verify_payload,
)


MANIFEST_COLUMNS = [
    "release_id",
    "logical_path",
    "table_id",
    "results_used",
    "release_tier",
    "row_count",
    "column_count",
    "primary_key",
    "source_asset_ids",
    "source_sha256s",
    "transform",
    "sha256",
    "size_bytes",
]


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    payload = tmp_path / "payload"
    (payload / "tables").mkdir(parents=True)
    csv_path = payload / "tables" / "values.csv"
    csv_path.write_text("id,value\na,1\nb,2\n", encoding="utf-8", newline="\n")
    json_path = payload / "contract.json"
    json_path.write_text('{"locked": true}\n', encoding="utf-8", newline="\n")
    rows = [
        {
            "release_id": "test-release",
            "logical_path": "contract.json",
            "table_id": "contract",
            "results_used": "R6",
            "release_tier": "TEST",
            "row_count": "1",
            "column_count": "0",
            "primary_key": "",
            "source_asset_ids": "test",
            "source_sha256s": "test",
            "transform": "fixture",
            "sha256": sha256_file(json_path),
            "size_bytes": str(json_path.stat().st_size),
        },
        {
            "release_id": "test-release",
            "logical_path": "tables/values.csv",
            "table_id": "values",
            "results_used": "R6",
            "release_tier": "TEST",
            "row_count": "2",
            "column_count": "2",
            "primary_key": "id",
            "source_asset_ids": "test",
            "source_sha256s": "test",
            "transform": "fixture",
            "sha256": sha256_file(csv_path),
            "size_bytes": str(csv_path.stat().st_size),
        },
    ]
    manifest = tmp_path / "analysis_bundle_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_manifest = tmp_path / "source_manifest.csv"
    with source_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["asset_id", "decision_status"])
        writer.writeheader()
        writer.writerow({"asset_id": "fixture", "decision_status": "LOCKED_R1"})
    return payload, manifest, source_manifest


def test_bundle_is_deterministic_and_refetches_from_external_copy(tmp_path: Path):
    payload, manifest, source_manifest = _fixture(tmp_path)
    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    first_record = tmp_path / "first.json"
    second_record = tmp_path / "second.json"
    build_bundle(payload, first, manifest, first_record)
    build_bundle(payload, second, manifest, second_record)
    assert sha256_file(first) == sha256_file(second)
    assert json.loads(first_record.read_text())["bundle_sha256"] == sha256_file(first)

    remote = tmp_path / "outside-release" / "analysis_bundle.tar.zst"
    remote.parent.mkdir()
    shutil.copyfile(first, remote)
    destination = tmp_path / "clean_data"
    result = fetch_bundle(
        tier="analysis",
        source_manifest_path=source_manifest,
        bundle_record_path=first_record,
        analysis_manifest_path=manifest,
        destination=destination,
        locator_override=str(remote),
    )
    assert result["status"] == "PASS"
    assert result["file_count"] == 2
    assert verify_payload(destination, manifest)["status"] == "PASS"


def test_fetch_rejects_a_tampered_bundle(tmp_path: Path):
    payload, manifest, source_manifest = _fixture(tmp_path)
    bundle = tmp_path / "analysis_bundle.tar.zst"
    record = tmp_path / "record.json"
    build_bundle(payload, bundle, manifest, record)
    with bundle.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="size mismatch"):
        fetch_bundle(
            tier="analysis",
            source_manifest_path=source_manifest,
            bundle_record_path=record,
            analysis_manifest_path=manifest,
            destination=tmp_path / "clean_data",
            locator_override=str(bundle),
        )


def test_verifier_rejects_extra_files(tmp_path: Path):
    payload, manifest, _ = _fixture(tmp_path)
    (payload / "extra.txt").write_text("not in manifest", encoding="utf-8")
    result = verify_payload(payload, manifest)
    assert result["status"] == "FAIL"
    assert "extra=['extra.txt']" in result["errors"][0]


def test_builder_rejects_parent_traversal(tmp_path: Path):
    payload, manifest, _ = _fixture(tmp_path)
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("contract.json", "../contract.json"), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid logical path"):
        build_bundle(
            payload,
            tmp_path / "bundle.tar.zst",
            manifest,
            tmp_path / "record.json",
        )
