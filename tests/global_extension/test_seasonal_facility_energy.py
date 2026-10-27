from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from analysis.global_extension.seasonal_facility_energy import (
    J_PER_KWH,
    enrich_seasonal_facility_energy,
    extract_offset_zero_monthly_gas,
)


MONTHS = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
CONTROLS = ("uniform_average", "personal_center", "personal_range")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_monthly_sql(path: Path, heating_kwh: float, water_kwh: float) -> None:
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE Strings (StringIndex INTEGER PRIMARY KEY, Value TEXT)"
        )
        connection.execute(
            """CREATE TABLE TabularData (
                TabularDataIndex INTEGER PRIMARY KEY,
                ReportNameIndex INTEGER,
                RowNameIndex INTEGER,
                ColumnNameIndex INTEGER,
                UnitsIndex INTEGER,
                Value TEXT
            )"""
        )
        strings = [
            "EnergyConsumptionElectricityNaturalGasMonthly",
            "EndUseEnergyConsumptionNaturalGasMonthly",
            "NATURALGAS:FACILITY",
            "HEATING:NATURALGAS",
            "WATERSYSTEMS:NATURALGAS",
            "J",
            *MONTHS,
        ]
        connection.executemany(
            "INSERT INTO Strings VALUES (?, ?)", enumerate(strings, start=1)
        )
        index = {value: number for number, value in enumerate(strings, start=1)}
        rows = []
        row_id = 1
        for month in MONTHS:
            for report, end_use, energy in (
                (
                    "EnergyConsumptionElectricityNaturalGasMonthly",
                    "NATURALGAS:FACILITY",
                    heating_kwh + water_kwh,
                ),
                (
                    "EndUseEnergyConsumptionNaturalGasMonthly",
                    "HEATING:NATURALGAS",
                    heating_kwh,
                ),
                (
                    "EndUseEnergyConsumptionNaturalGasMonthly",
                    "WATERSYSTEMS:NATURALGAS",
                    water_kwh,
                ),
            ):
                rows.append((
                    row_id, index[report], index[month], index[end_use],
                    index["J"], str(energy * J_PER_KWH),
                ))
                row_id += 1
        connection.executemany(
            "INSERT INTO TabularData VALUES (?, ?, ?, ?, ?, ?)", rows
        )


def _annual_manifest(tmp_path: Path) -> Path:
    rows = []
    for control_index, control in enumerate(CONTROLS):
        run_id = f"annual_{control_index}"
        run_folder = tmp_path / "runs" / run_id
        sql_path = run_folder / "eplusout.sql"
        heating = 10.0 + control_index
        water = 2.0 + control_index / 10
        _write_monthly_sql(sql_path, heating, water)
        input_hash = ("a", "b", "c")[control_index] * 64
        (run_folder / "RUN_QA.json").write_text(
            json.dumps({"completed": True, "input_idf_sha256": input_hash}),
            encoding="utf-8",
        )
        compact = tmp_path / "hourly" / f"{run_id}.npz"
        compact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            compact,
            facility_gas_hhv_kwh_annual=np.asarray(12 * (heating + water)),
        )
        rows.append({
            "run_id": run_id,
            "climate": "Climate A",
            "composition_id": 1,
            "control": control,
            "offset_c": 0.0,
            "run_folder": str(run_folder),
            "input_sha256": input_hash,
            "hourly_path": str(compact),
            "hourly_sha256": _sha256(compact),
        })
    path = tmp_path / "annual.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_monthly_extraction_and_seasonal_facility_reconstruction(tmp_path: Path):
    monthly, extraction = extract_offset_zero_monthly_gas(
        _annual_manifest(tmp_path), expected_runs=3, workers=2
    )

    assert len(monthly) == 36
    assert extraction["all_runs_have_12_months"]
    assert extraction["maximum_absolute_annual_rounding_difference_kwh"] < 1e-9

    seasonal_rows = []
    for control in CONTROLS:
        for block in ("DJF", "MAM", "JJA", "SON"):
            seasonal_rows.append({
                "run_id": f"seasonal_{control}",
                "climate": "Climate A",
                "composition_id": 1,
                "cohort_label": "Cohort A",
                "demand_source": "Source A",
                "control": control,
                "seasonal_contrast_fraction": 1.0,
                "meteorological_block": block,
                "local_season": block,
                "facility_electric_kwh": 100.0,
                "gas_hhv_kwh": 10.0,
                "facility_gas_hhv_kwh": np.nan,
                "facility_purchased_kwh": np.nan,
                "total_purchased_kwh": np.nan,
                "facility_electric_plus_space_heating_gas_kwh": 110.0,
            })
    seasons, annual, audit = enrich_seasonal_facility_energy(
        pd.DataFrame(seasonal_rows), monthly
    )

    uniform = seasons[seasons.control.eq("uniform_average")]
    assert np.allclose(uniform.non_space_heating_facility_gas_kwh, 6.0)
    assert np.allclose(uniform.facility_gas_hhv_kwh, 16.0)
    assert np.allclose(uniform.facility_purchased_kwh, 116.0)
    assert len(annual) == 3
    assert audit["seasonal_runs"] == 3
    assert audit["seasonal_block_rows"] == 12
