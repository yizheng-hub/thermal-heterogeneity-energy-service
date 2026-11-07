import csv
import hashlib
import io
import json
import zipfile

import pytest

from publication.build_review_package import portable_result, public_csv


def test_public_csv_preserves_numeric_text_and_removes_local_paths(tmp_path):
    source = tmp_path / "support.csv"
    source.write_text(
        "value,empty,hourly_path,qa,description\n"
        '0.12345678901234567890,,relative/hourly.csv,C:/local/qa.json,"a,b"\n',
        encoding="utf-8",
    )
    payload, removed = public_csv(source)
    rows = list(csv.reader(io.StringIO(payload.decode())))
    assert removed == ["hourly_path", "qa"]
    assert rows == [
        ["value", "empty", "description"],
        ["0.12345678901234567890", "", "a,b"],
    ]


def test_public_csv_rejects_participant_identifiers(tmp_path):
    source = tmp_path / "support.csv"
    source.write_text("participant_id,value\np001,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Individual identifiers"):
        public_csv(source)


def test_portable_result_preserves_plot_links_and_canonical_source(tmp_path):
    source = tmp_path / "ACTION_FULL_COMPLETED.csv"
    original = b"value,hourly_path,qa\n0.12345678901234567890,Data_for_Figures/hourly.csv,C:/local/qa.json\n"
    source.write_bytes(original)
    payload, transformation = portable_result(source)
    assert payload == b"value,hourly_path\n0.12345678901234567890,Data_for_Figures/hourly.csv\n"
    assert "qa" in transformation
    assert source.read_bytes() == original
    status = tmp_path / "STATUS.json"
    status.write_text('{"status":"completed","output_directory":"C:/local/output"}')
    payload, transformation = portable_result(status)
    assert json.loads(payload) == {"status": "completed"}
    assert "output_directory" in transformation
    plan = tmp_path / "SINGLE_EVENT_ACTION_PLAN.csv"
    event = {"start_timestamp": "2001-07-01 14:00:00", "duration_hours": 2,
             "precondition_hours": 2, "recovery_hours": 48, "reference_hourly_path": "C:/local/hourly.npz"}
    with plan.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_json", "value"])
        writer.writerow([json.dumps(event), "0.12345678901234567890"])
    payload, transformation = portable_result(plan)
    row = next(csv.DictReader(io.StringIO(payload.decode())))
    del event["reference_hourly_path"]
    assert json.loads(row["event_json"]) == event
    assert row["value"] == "0.12345678901234567890"


def test_data_manifest_matches_software_snapshot_and_supplied_bytes(tmp_path, monkeypatch):
    from publication import build_review_package, build_submission_packages, figure_data

    root = tmp_path / "source"
    files = {
        "Data_for_Figures/input.csv": b"value\n1\n",
        "Data_for_Figures/FIGURE_DATA_MANIFEST.json": b"{}\n",
        "publication/review_data_files.txt": b"output/support.csv\n",
        "output/support.csv": b"value,qa\n0.12345678901234567890,C:/local/qa.json\n",
        "LICENSE-DOCS-DATA": b"Author-generated material only\n",
        "DATA_AND_SOFTWARE_LICENSES.md": b"Upstream terms retained\n",
        "THIRD_PARTY_NOTICES.md": b"Upstream sources\n",
        "data/metadata/source_manifest.csv": b"asset,upstream\nsource,https://example.org\n",
    }
    files.update({f"output/publication/tables/Table_S{n:02}.csv": b"value\n1\n" for n in range(1, 11)})
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for folder, prefix, count in (("main", "Figure_", 6), ("extended", "Figure_E", 4), ("supplementary", "Figure_S", 6)):
        destination = root / "output/publication" / folder
        destination.mkdir(parents=True)
        for number in range(1, count + 1):
            (destination / f"{prefix}{number:02}.json").write_text(
                json.dumps({"input_sha256": {"Data_for_Figures/input.csv": "fixture"}})
            )
    monkeypatch.setattr(build_review_package, "ROOT", root)
    monkeypatch.setattr(figure_data, "verify", lambda: {
        "files": [{"result": "Data_for_Figures/input.csv", "transformation": "byte-identical"}]
    })
    snapshot = {"release_id": "submission-1", "source_snapshot_sha256": "a" * 64}
    monkeypatch.setattr(build_submission_packages, "collect_software_snapshot", lambda *args: snapshot)
    destination = tmp_path / "data.zip"
    build_review_package.build(destination, release_id="submission-1", source_snapshot=snapshot)
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("SOURCE_DATA_MANIFEST.json"))
        assert "README.md" not in archive.namelist()
        assert not any(name.lower().endswith(".md") for name in archive.namelist())
        assert "CODE_MANIFEST.json" not in archive.namelist()
        assert "Extended Data Figure 4" in archive.read("FIGURE_TABLE_INDEX.txt").decode()
        for line in archive.read("FIGURE_TABLE_INDEX.txt").decode().splitlines():
            if "Supplementary_tables/" in line:
                assert line.split(": ", 1)[1] in archive.namelist()
        assert manifest["source_snapshot_sha256"] == snapshot["source_snapshot_sha256"]
        assert manifest["release_id"] == "submission-1"
        assert archive.read("Supporting_tables/support.csv") == b"value\n0.12345678901234567890\n"
        assert len([name for name in archive.namelist() if name.startswith("Supplementary_tables/")]) == 10
        for row in manifest["files"]:
            assert hashlib.sha256(archive.read(row["file"])).hexdigest() == row["sha256"]
    buffer = io.BytesIO()
    build_review_package.build(buffer, release_id="submission-1", source_snapshot=snapshot)
    with zipfile.ZipFile(buffer) as archive:
        assert json.loads(archive.read("SOURCE_DATA_MANIFEST.json")) == manifest
        assert archive.testzip() is None


def test_data_rejects_stale_or_forged_source_snapshot_before_writing(tmp_path, monkeypatch):
    from publication import build_review_package, build_submission_packages

    current = {"release_id": "submission-1", "source_snapshot_sha256": "a" * 64, "files": []}
    monkeypatch.setattr(build_submission_packages, "collect_software_snapshot", lambda *args: current)
    destination = tmp_path / "data.zip"
    for supplied in (
        {"release_id": "submission-1", "source_snapshot_sha256": "b" * 64, "files": []},
        {"release_id": "submission-1", "source_snapshot_sha256": "a" * 64},
    ):
        with pytest.raises(ValueError, match="differs from the matching software snapshot"):
            build_review_package.build(destination, release_id="submission-1", source_snapshot=supplied)
        assert not destination.exists()
