from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.enrich_facility_energy import (
    enrich_manifest,
    extract_facility_electric_kwh,
)
from analysis.global_extension.global_matrix import sha256


def _facility_sql(path: Path, values_j: np.ndarray) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("create table Time (TimeIndex integer)")
        connection.executemany(
            "insert into Time values (?)",
            [(index + 1,) for index in range(len(values_j))],
        )
        connection.execute(
            "create table ReportDataDictionary ("
            "ReportDataDictionaryIndex integer, KeyValue text, Name text, "
            "ReportingFrequency text, ScheduleName text, Units text)"
        )
        connection.execute(
            "insert into ReportDataDictionary values (?, ?, ?, ?, ?, ?)",
            (1, "Whole Building", "Electricity:Facility", "Hourly", "", "J"),
        )
        connection.execute(
            "insert into ReportDataDictionary values (?, ?, ?, ?, ?, ?)",
            (2, "Whole Building", "Electricity:HVAC", "Hourly", "", "J"),
        )
        connection.execute(
            "create table ReportData ("
            "ReportDataIndex integer, TimeIndex integer, "
            "ReportDataDictionaryIndex integer, Value real)"
        )
        connection.executemany(
            "insert into ReportData values (?, ?, ?, ?)",
            [
                (index + 1, index + 1, 1, float(value))
                for index, value in enumerate(values_j)
            ],
        )
        connection.executemany(
            "insert into ReportData values (?, ?, ?, ?)",
            [
                (len(values_j) + index + 1, index + 1, 2, float(value) * 0.4)
                for index, value in enumerate(values_j)
            ],
        )
        connection.execute(
            "create table TabularDataWithStrings ("
            "ReportName text, ReportForString text, TableName text, "
            "RowName text, ColumnName text, Units text, Value text)"
        )
        connection.execute(
            "insert into TabularDataWithStrings values (?, ?, ?, ?, ?, ?, ?)",
            (
                "AnnualBuildingUtilityPerformanceSummary",
                "Entire Facility",
                "End Uses",
                "Total End Uses",
                "Natural Gas",
                "GJ",
                "108.0",
            ),
        )


def test_sql_facility_meter_is_extracted_in_hourly_kwh(tmp_path: Path):
    sql = tmp_path / "eplusout.sql"
    expected = np.resize(np.asarray([1.0, 2.0, 3.0]), 8760)
    _facility_sql(sql, expected * 3.6e6)
    assert np.allclose(extract_facility_electric_kwh(sql), expected)


def test_enrichment_atomically_updates_compact_qa_and_manifest(tmp_path: Path):
    hours = 8760
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    compact = run_folder / "annual.npz"
    np.savez_compressed(
        compact,
        month=np.ones(hours, dtype=int),
        hvac_electric_kwh=np.full(hours, 2.0),
        gas_hhv_kwh=np.full(hours, 3.0),
    )
    original_hvac = np.full(hours, 2.0)
    sql = run_folder / "eplusout.sql"
    _facility_sql(sql, np.full(hours, 18.0e6))
    qa = run_folder / "RUN_QA.json"
    qa.write_text(
        json.dumps({"hourly_sha256": sha256(compact), "completed": True}),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "run_id": "r1",
        "run_folder": str(run_folder),
        "hourly_path": str(compact),
        "qa_path": str(qa),
        "hourly_sha256": sha256(compact),
        "completed": True,
    }]).to_csv(manifest, index=False)

    audit = enrich_manifest(manifest)
    updated = pd.read_csv(manifest)
    with np.load(compact, allow_pickle=False) as stored:
        assert np.allclose(stored["facility_electric_kwh"], 5.0)
        assert np.isclose(
            np.asarray(stored["facility_gas_hhv_kwh_annual"]).reshape(-1)[0],
            30_000.0,
        )
        assert np.array_equal(stored["hvac_electric_kwh"], original_hvac)
    updated_hash = sha256(compact)
    assert updated.loc[0, "hourly_sha256"] == updated_hash
    assert bool(updated.loc[0, "facility_energy_enriched"])
    assert json.loads(qa.read_text(encoding="utf-8"))["hourly_sha256"] == updated_hash
    assert audit["compacts_changed"] == 1

    second_audit = enrich_manifest(manifest)
    assert sha256(compact) == updated_hash
    assert second_audit["compacts_changed"] == 0
