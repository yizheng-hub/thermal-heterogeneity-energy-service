import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import spatial_hvac_mechanism as module


def test_archived_sql_is_verified_and_read_without_restoring_a_database(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    sql = tmp_path / "physical.sql"
    with sqlite3.connect(":memory:") as connection:
        connection.execute("create table Recorded(Value real)")
        connection.execute("insert into Recorded values(42.25)")
        raw = connection.serialize()
    archive = sql.with_suffix(".sql.zst")
    archive.write_bytes(zstd.ZstdCompressor().compress(raw))
    (tmp_path / "RAW_SQL_LOSSLESS_ARCHIVE.json").write_text(json.dumps({
        "uncompressed_sql_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}))
    connection, resolved = module.database(module.Inputs(), sql)
    assert connection.execute("select Value from Recorded").fetchone()[0] == 42.25
    connection.close()
    assert not sql.exists()
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="SHA differs"):
        module.database(module.Inputs(), sql)


def test_fixed_order_accounting_preserves_a_negative_temperature_response():
    count = 267
    original = pd.DataFrame({"profile_id": np.arange(count),
        "centre_c": [20.] * 133 + [26.] * 133 + [23.],
        "zone_id": ["Warm"] * 133 + ["Cold"] * 133 + ["Warm"]})
    arranged = original.copy()
    arranged["zone_id"] = ["Cold"] * 133 + ["Warm"] * 133 + ["Warm"]
    before = {"zone_ids": np.array(["Cold", "Warm"]), "occupied": np.array([True, True]),
        "people": np.array([[133., 134.], [133., 134.]]),
        "temperatures": np.array([[20., 26.], [20., 26.]])}
    after = before | {"temperatures": np.array([[21., 25.], [21., 25.]])}
    initial = module.score(before, original, set(range(27)))
    regrouped = module.score(before, arranged, set(range(27)))
    final = module.score(after, arranged, set(range(27)))
    placement_step = initial - regrouped
    temperature_step = regrouped - final
    assert (placement_step > 0).all() and (temperature_step < 0).all()
    assert np.allclose(placement_step + temperature_step, initial - final, atol=1e-12)


@pytest.fixture
def registered_case(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path / "strengthening")

    def asset(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    idf, weather, hourly, sql = [asset(name, name.encode()) for name in
                                ["model.idf", "weather.epw", "original.npz", "complete.sql"]]
    runtime = asset("tools/energyplus.exe", b"locked runtime")
    qa_data = {"completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
               "input_idf_sha256": idf["sha256"], "weather_sha256": weather["sha256"],
               "runtime_sha256": runtime["sha256"], "engineering_pass": False}
    qa = asset("RUN_QA.json", json.dumps(qa_data).encode())
    validation_data = {"status": "passed", "hours": 8760, "sql_sha256": sql["sha256"],
                       "qa_sha256": qa["sha256"], "original_hourly_sha256": hourly["sha256"],
                       "input_idf_sha256": idf["sha256"], "weather_sha256": weather["sha256"],
                       "runtime_sha256": runtime["sha256"],
                       "sql_max_absolute_differences": {k: 0. for k in ["temperatures", "people", *module.ENERGY]}}
    validation = asset("VALIDATION.json", json.dumps(validation_data).encode())
    protocol = asset("FROZEN_PROTOCOL.json", b"frozen before results")
    case = dict(climate="Test", composition_id=1, control="personal_center", layout="within_floor_relocation_25pct", system="vav", placement="mixed")
    source = {"source_id": "existing_source", "origin": "existing_exact_cache", "identity": case,
              "input_idf": idf, "weather": weather, "original_hourly": hourly, "sql": sql, "qa": qa,
              "validation": validation, "runtime": {"executable_path": runtime["path"],
              "executable_sha256": runtime["sha256"], "api_sha256": None, "package_manifest_sha256": None}}
    registry = {"schema_version": 1, "project_root": ".", "protocol_path": protocol["path"],
                "protocol_sha256": protocol["sha256"], "sources": [source]}
    path = module.ROOT / "simulations/diagnostic_sql_replay/REGISTERED_SQL_SOURCES.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry))
    expected = [idf["sha256"], hourly["sha256"], weather["sha256"], runtime["sha256"]]
    return case, source, registry, path, expected


def test_registered_source_retains_exact_cache_origin_and_raw_qa(registered_case):
    case, source, _, _, expected = registered_case
    records = module.registered_sql(module.Inputs(), case, *expected)
    assert len(records) == 1 and records[0]["registered_origin"] == "existing_exact_cache"
    assert records[0]["available"] and records[0]["registered_source_id"] == source["source_id"]


def test_same_logical_identity_cannot_override_an_original_hourly_hash_mismatch(registered_case):
    case, _, _, _, expected = registered_case
    expected[1] = "different original hourly hash"
    assert module.registered_sql(module.Inputs(), case, *expected) == []


def test_registered_sql_content_change_is_rejected(registered_case, tmp_path):
    case, _, _, _, expected = registered_case
    (tmp_path / "complete.sql").write_bytes(b"altered SQL")
    with pytest.raises(ValueError, match="SHA differs"):
        module.registered_sql(module.Inputs(), case, *expected)


def test_registered_relative_path_cannot_escape_code(registered_case):
    case, source, registry, path, expected = registered_case
    source["sql"]["path"] = "../foreign.sql"
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="inside this project"):
        module.registered_sql(module.Inputs(), case, *expected)


def test_common_setting_audit_accepts_a_completed_run_with_zero_sql_blockers(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path / "strengthening")
    monkeypatch.setattr(module, "OUT", module.ROOT / "results/spatial_hvac_mechanism")
    previous, out = module.OUT / "previous", module.OUT / "current"
    previous.mkdir(parents=True)
    out.mkdir()
    source = module.ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv"
    source.parent.mkdir()
    source.write_text("retained comparison")
    (previous / "STATUS.json").write_text(json.dumps({"status": "completed"}))
    (previous / "INPUT_HASHES.json").write_text(json.dumps({str(source.resolve()): module.digest(source)}))
    (previous / "CASE_SOURCES.csv").write_text("climate,composition_id,control,layout,system,placement\n")
    (previous / "SQL_BLOCKERS.csv").write_text("\n")
    pd.DataFrame([{"control": "personal_center", "planned_settings": 54,
                   "qualified_settings": 52, "unavailable_settings": 2}]).to_csv(previous / "DENOMINATORS.csv", index=False)
    rows = [{"climate": "Test", "composition_id": 1, "control": "personal_center",
             "layout": "within_floor_relocation_25pct", "system": system, "window": window,
             "cooling_unmet_zone_hours_change": -2. if system == "vav" else 1.}
            for system in ["vav", "pthp"] for window in ["service", "outside_service"]]
    for prefix in ["ENERGY", "TRACKING"]:
        pd.DataFrame(rows).to_csv(previous / (prefix + "_WINDOW_PAIRED_CHANGES.csv"), index=False)
    result = module.audit_locators(module.Inputs(), out, previous)
    assert result["status"] == "completed" and result["previous_blocked_cases"] == 0
    common = pd.read_csv(out / "TRACKING_COMMON_SETTINGS.csv")
    assert len(common) == 2
    assert common.cooling_unmet_zone_hours_change_pthp_minus_vav.eq(3.).all()
