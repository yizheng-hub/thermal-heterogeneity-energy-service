import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
import pytest
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import zone_ventilation_diagnostic as module


def trajectory():
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    return {"month": clock.month.to_numpy(), "day": clock.day.to_numpy(),
            "hour": clock.hour.to_numpy() + 1, "dst": np.zeros(8760, dtype=int),
            "occupied": np.ones(8760, dtype=bool), "zone_ids": np.array(["A", "B"]),
            "people": np.ones((8760, 2))}


def synthetic_sql(path, indexed):
    data = trajectory()
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            create table EnvironmentPeriods(EnvironmentPeriodIndex integer primary key,EnvironmentType integer);
            insert into EnvironmentPeriods values(1,3);
            create table Time(TimeIndex integer primary key,Month integer,Day integer,Hour integer,Dst integer,Interval integer,EnvironmentPeriodIndex integer,WarmupFlag integer);
            create table ReportDataDictionary(ReportDataDictionaryIndex integer primary key,KeyValue text,Name text,ReportingFrequency text,Units text);
            create table ReportData(TimeIndex integer,ReportDataDictionaryIndex integer,Value real);
        """)
        connection.executemany("insert into Time values(?,?,?,?,?,?,?,?)", [
            (i+1, int(data["month"][i]), int(data["day"][i]), int(data["hour"][i]), 0, 60, 1, 0)
            for i in range(8760)])
        connection.executemany("insert into ReportDataDictionary values(?,?,?,?,?)", [
            (1, "A", module.DELIVERED, "Hourly", "m3/s"),
            (2, "B", module.DELIVERED, "Hourly", "m3/s"),
            (3, "A", module.REQUESTED, "Hourly", "m3/s"),
            (4, "B", module.REQUESTED, "Hourly", "m3/s")])
        connection.executemany("insert into ReportData values(?,?,?)", [
            (i+1, identifier, value) for i in range(8760)
            for identifier, value in [(1, 0.5), (2, 2.0), (3, 1.0), (4, 1.0)]])
        if indexed:
            connection.execute("create index retained_flow_index on ReportData(ReportDataDictionaryIndex,TimeIndex)")
    connection.close()
    return data


def test_assessment_uses_zone_people_and_positive_request_without_losing_zero_denominators():
    data = trajectory()
    data["occupied"][4:] = False
    data["people"][1, 0] = 0
    requested = np.ones((8760, 2))
    requested[2, 0] = 0
    requested[:, 1] = 0
    delivered = np.ones((8760, 2))
    delivered[0, 0], delivered[3, 0] = 2.0, 0.5
    rows = module.zone_statistics(data, delivered, requested, "calendar")
    assert rows[0]["assessed_zone_hours"] == 2
    assert rows[0]["occupied_zero_request_hours"] == 1
    assert rows[0]["delivery_ratio_min"] == 0.5
    assert rows[0]["hours_below_99pct"] == 1
    assert rows[0]["delivered_volume_m3"] == 2.5 * 3600
    assert rows[0]["requested_volume_m3"] == 2.0 * 3600
    assert rows[1]["assessed_zone_hours"] == 0
    assert rows[1]["occupied_zero_request_hours"] == 4
    assert np.isnan(rows[1]["delivery_ratio_min"])
    assert rows[1]["ratio_status"] == "no_occupied_positive_request_hours"


@pytest.mark.parametrize("indexed", [False, True])
def test_bulk_query_uses_available_index_and_never_changes_the_original_sql(tmp_path, monkeypatch, indexed):
    monkeypatch.setattr(module, "CODE", tmp_path)
    path = tmp_path / "complete.sql"
    data = synthetic_sql(path, indexed)
    before = path.read_bytes()
    hashes = {}
    history = {str(path): hashlib.sha256(before).hexdigest()}
    with pytest.raises(ValueError, match="input SHA changed"):
        module.open_database(path, {str(path): "different"}, {})
    connection, provenance = module.open_database(path, history, hashes)
    try:
        delivered, requested, calendar, plan = module.extract_flows(connection, data)
        assert delivered.shape == requested.shape == (8760, 2)
        assert np.all(delivered[:, 0] == 0.5) and np.all(requested == 1)
        assert len(calendar) == 64 and len(provenance["sql_sha256"]) == 64
        assert ("retained_flow_index" in plan) == indexed
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("create index forbidden on ReportData(TimeIndex)")
        changed = dict(data)
        changed["dst"] = np.ones(8760, dtype=int)
        with pytest.raises(ValueError, match="calendar"):
            module.extract_flows(connection, changed)
    finally:
        connection.close()
    assert path.read_bytes() == before


def test_archive_requires_both_hashes_and_is_not_restored_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    path = tmp_path / "complete.sql"
    synthetic_sql(path, True)
    raw = path.read_bytes()
    packed = zstd.ZstdCompressor().compress(raw)
    path.unlink()
    archive = path.with_suffix(".sql.zst")
    archive.write_bytes(packed)
    record = path.parent / "RAW_SQL_LOSSLESS_ARCHIVE.json"
    record.write_text(json.dumps({"archive_sha256": hashlib.sha256(packed).hexdigest(),
                                  "uncompressed_sql_sha256": hashlib.sha256(raw).hexdigest()}))
    history = {str(p): module.digest(p) for p in [archive, record]}
    connection, _ = module.open_database(path, history, {})
    connection.close()
    assert not path.exists()
    record.write_text(json.dumps({"archive_sha256": hashlib.sha256(packed).hexdigest(),
                                  "uncompressed_sql_sha256": "incorrect"}))
    history[str(record)] = module.digest(record)
    with pytest.raises(ValueError, match="Uncompressed SQL SHA"):
        module.open_database(path, history, {})


@pytest.mark.parametrize("missing", ["variable", "hour"])
def test_missing_native_data_is_unresolved_instead_of_using_a_building_total(tmp_path, monkeypatch, missing):
    monkeypatch.setattr(module, "CODE", tmp_path)
    path = tmp_path / "complete.sql"
    data = synthetic_sql(path, False)
    with sqlite3.connect(path) as connection:
        if missing == "variable":
            connection.execute("delete from ReportDataDictionary where ReportDataDictionaryIndex=4")
        else:
            connection.execute("delete from ReportData where ReportDataDictionaryIndex=4 and TimeIndex=100")
    connection.close()
    connection, _ = module.open_database(path, {str(path): module.digest(path)}, {})
    try:
        with pytest.raises(ValueError, match="Missing"):
            module.extract_flows(connection, data)
    finally:
        connection.close()


def test_paired_changes_retain_different_assessment_denominators_as_unresolved():
    data = trajectory()
    original = module.zone_statistics(data, np.ones((8760, 2)), np.ones((8760, 2)), "clock")
    altered = [dict(row) for row in original]
    altered[0]["assessment_mask_sha256"] = "different"
    identity = {"climate": "Test", "composition_id": 1, "control": "personal_center",
                "layout": "within_floor_relocation_25pct", "system": "pthp"}
    frame = pd.DataFrame([identity | {"placement": placement} | row
                          for placement, rows in [("mixed", original), ("arranged", altered)] for row in rows])
    paired = module.paired_changes(frame).set_index("zone_id")
    assert paired.loc["A", "pair_status"] == "unresolved_assessment_clock_or_mask"
    assert np.isnan(paired.loc["A", "hours_below_99pct_change"])
    assert paired.loc["B", "hours_below_99pct_change"] == 0


@pytest.mark.parametrize("missing", ["mixed", "arranged"])
def test_paired_changes_retain_a_zone_missing_from_one_placement(missing):
    rows = module.zone_statistics(trajectory(), np.ones((8760, 2)), np.ones((8760, 2)), "clock")
    identity = {"climate": "Test", "composition_id": 1, "control": "personal_center",
                "layout": "within_floor_relocation_25pct", "system": "pthp"}
    frame = pd.DataFrame([identity | {"placement": placement} | row
                          for placement in ["mixed", "arranged"] for row in rows
                          if placement != missing or row["zone_id"] != "A"])
    paired = module.paired_changes(frame).set_index("zone_id")
    assert set(paired.index) == {"A", "B"}
    assert paired.loc["A", "pair_status"] == "unresolved_missing_" + missing
    assert paired.loc["A", "_merge"] == ("right_only" if missing == "mixed" else "left_only")
    assert paired.loc["A", [metric + "_change" for metric in module.METRICS]].isna().all()
    assert paired.loc["B", "pair_status"] == "matched_assessment_hours"
    assert paired.loc["B", "hours_below_99pct_change"] == 0


def test_inventory_keeps_all_105_settings_and_rejects_a_missing_record(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "CODE", tmp_path)
    source = tmp_path / "mechanism"
    source.mkdir()
    (source / "STATUS.json").write_text(json.dumps({"status": "completed", "decomposition": {"paired_settings": 105}}))
    (source / "INPUT_HASHES.json").write_text("{}")
    (source / "DENOMINATORS.csv").write_text("control,planned_settings\npersonal_center,54\npersonal_range,54\n")
    cases = [{"climate": str(n), "composition_id": 1, "control": control,
              "layout": "within_floor_relocation_25pct", "system": system, "placement": placement,
              "hourly_path": "contained.npz"}
             for control, count in [("personal_center", 52), ("personal_range", 53)]
             for n in range(count) for system in ["vav", "pthp"] for placement in ["mixed", "arranged"]]
    frame = pd.DataFrame(cases)
    frame.to_csv(source / "CASE_SOURCES.csv", index=False)
    frame.assign(window="service", resolved_sql_path="contained.sql").to_csv(source / "TRACKING_WINDOW_LEVELS.csv", index=False)
    inventory, _ = module.inventory(source, {})
    assert len(inventory) == 420
    frame.iloc[:-1].to_csv(source / "CASE_SOURCES.csv", index=False)
    with pytest.raises(ValueError, match="420 physical records"):
        module.inventory(source, {})


def test_automatic_mechanism_input_skips_completed_summaries_but_not_failed_attempts(tmp_path):
    complete = tmp_path / "run_001"
    complete.mkdir()
    (complete / "STATUS.json").write_text(json.dumps({"status": "completed", "diagnostics": {"status": "completed"}}))
    summary = tmp_path / "run_002"
    summary.mkdir()
    (summary / "STATUS.json").write_text(json.dumps({"status": "completed", "decomposition": {"status": "completed"}}))
    assert module.completed_mechanism_root(tmp_path) == complete
    failed = tmp_path / "run_003"
    failed.mkdir()
    (failed / "STATUS.json").write_text(json.dumps({"status": "partial", "diagnostics": {"status": "partial"}}))
    with pytest.raises(ValueError, match="latest mechanism attempt"):
        module.completed_mechanism_root(tmp_path)
