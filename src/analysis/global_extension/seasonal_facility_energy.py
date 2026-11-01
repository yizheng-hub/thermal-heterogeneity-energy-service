"""Reconstruct complete seasonal facility gas from retained exact evidence.

Seasonal command compacts contain hourly facility electricity and space-heating
gas.  The fixed prototype also uses natural gas for water systems.  Its monthly
table is retained in the exact offset-zero annual SQL for every
climate/composition/control identity.  This module joins that matched monthly
background load to each seasonal command trajectory.  It never edits an IDF or
runs EnergyPlus.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


MONTH_NUMBER = {
    name: number
    for number, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November",
            "December",
        ),
        start=1,
    )
}
BLOCK_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
IDENTITY = ("climate", "composition_id", "control")
J_PER_KWH = 3_600_000.0


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sql_index(roots: tuple[Path, ...]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("eplusout.sql"):
            result.setdefault(path.parent.name, []).append(path.resolve())
    return result


def _matched_sql(
    row: pd.Series,
    alternatives: dict[str, list[Path]],
) -> Path:
    primary = Path(str(row.run_folder)) / "eplusout.sql"
    if not primary.is_file():
        primary = Path(str(row.run_folder)) / "simulation_metadata.sql"
    candidates = ([primary.resolve()] if primary.is_file() else []) + alternatives.get(
        str(row.run_id), []
    )
    for path in candidates:
        qa_path = path.parent / "RUN_QA.json"
        if not qa_path.is_file():
            continue
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if (
            qa.get("completed") is True
            and str(qa.get("input_idf_sha256", "")) == str(row.input_sha256)
        ):
            return path
    raise FileNotFoundError(
        f"No exact retained SQL matches offset-zero run {row.run_id}"
    )


def _monthly_natural_gas(sql_path: Path) -> pd.DataFrame:
    query = """
        SELECT report.Value, row_name.Value, column_name.Value, units.Value,
               tabular.Value
        FROM TabularData AS tabular
        JOIN Strings AS report
          ON report.StringIndex = tabular.ReportNameIndex
        JOIN Strings AS row_name
          ON row_name.StringIndex = tabular.RowNameIndex
        JOIN Strings AS column_name
          ON column_name.StringIndex = tabular.ColumnNameIndex
        JOIN Strings AS units
          ON units.StringIndex = tabular.UnitsIndex
        WHERE (
            report.Value = 'EnergyConsumptionElectricityNaturalGasMonthly'
            AND column_name.Value = 'NATURALGAS:FACILITY'
        ) OR report.Value = 'EndUseEnergyConsumptionNaturalGasMonthly'
    """
    uri = f"file:{sql_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        raw = connection.execute(query).fetchall()
    if not raw:
        raise ValueError(f"Monthly natural-gas tables are absent from {sql_path}")

    rows: list[dict] = []
    for report, month_name, end_use, units, value in raw:
        if month_name not in MONTH_NUMBER:
            continue
        numeric = float(value)
        if numeric and units != "J":
            raise ValueError(
                f"Monthly natural-gas value lacks joule units in {sql_path}"
            )
        rows.append({
            "report": str(report),
            "month": MONTH_NUMBER[str(month_name)],
            "end_use": str(end_use),
            "energy_kwh": numeric / J_PER_KWH,
        })
    frame = pd.DataFrame(rows)
    facility = frame[
        frame.report.eq("EnergyConsumptionElectricityNaturalGasMonthly")
    ][["month", "energy_kwh"]].rename(
        columns={"energy_kwh": "facility_gas_hhv_kwh"}
    )
    end_uses = frame[
        frame.report.eq("EndUseEnergyConsumptionNaturalGasMonthly")
    ].pivot(index="month", columns="end_use", values="energy_kwh")
    required = {"HEATING:NATURALGAS", "WATERSYSTEMS:NATURALGAS"}
    if set(facility.month) != set(range(1, 13)) or not required.issubset(end_uses):
        raise ValueError(f"Monthly natural-gas table is incomplete in {sql_path}")
    other = end_uses.drop(columns=list(required), errors="ignore").fillna(0.0)
    if not other.empty and other.abs().to_numpy(float).max(initial=0.0) > 1e-9:
        raise ValueError(f"Unexpected non-zero natural-gas end use in {sql_path}")
    result = facility.merge(
        end_uses.reset_index()[
            ["month", "HEATING:NATURALGAS", "WATERSYSTEMS:NATURALGAS"]
        ],
        on="month",
        validate="one_to_one",
    ).rename(columns={
        "HEATING:NATURALGAS": "heating_gas_hhv_kwh",
        "WATERSYSTEMS:NATURALGAS": "water_system_gas_hhv_kwh",
    })
    return result.sort_values("month").reset_index(drop=True)


def extract_offset_zero_monthly_gas(
    annual_manifest_path: Path,
    *,
    sql_search_roots: tuple[Path, ...] = (),
    expected_runs: int = 162,
    workers: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Extract hash-bound monthly facility and end-use gas for offset-zero runs."""
    annual_manifest_path = annual_manifest_path.resolve()
    manifest = pd.read_csv(annual_manifest_path)
    required = {
        "run_id", *IDENTITY, "offset_c", "run_folder", "input_sha256",
        "hourly_path", "hourly_sha256",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Annual manifest lacks {sorted(missing)}")
    selected = manifest[np.isclose(
        pd.to_numeric(manifest.offset_c, errors="raise"), 0.0
    )].copy()
    if len(selected) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} offset-zero runs, found {len(selected)}"
        )
    if selected.duplicated(list(IDENTITY)).any() or selected.run_id.duplicated().any():
        raise ValueError("Offset-zero annual identities are not unique")
    alternatives = _sql_index(sql_search_roots)

    def extract(row: pd.Series) -> tuple[list[dict], dict]:
        compact = Path(str(row.hourly_path)).resolve()
        if not compact.is_file() or sha256(compact) != str(row.hourly_sha256):
            raise ValueError(f"Offset-zero compact hash changed for {row.run_id}")
        with np.load(compact, allow_pickle=False) as saved:
            if "facility_gas_hhv_kwh_annual" not in saved.files:
                raise ValueError(f"Annual facility gas is absent for {row.run_id}")
            annual_gas = float(saved["facility_gas_hhv_kwh_annual"])
        sql_path = _matched_sql(row, alternatives)
        sql_hash = sha256(sql_path)
        monthly = _monthly_natural_gas(sql_path)
        monthly_sum = float(monthly.facility_gas_hhv_kwh.sum())
        rounding_difference = monthly_sum - annual_gas
        if abs(rounding_difference) > max(2.0, annual_gas * 1e-4):
            raise ValueError(
                f"Monthly facility gas does not reproduce annual gas for {row.run_id}"
            )
        records = []
        for item in monthly.to_dict("records"):
            records.append({
                "reference_run_id": str(row.run_id),
                "climate": str(row.climate),
                "composition_id": int(row.composition_id),
                "control": str(row.control),
                **item,
                "reference_input_sha256": str(row.input_sha256),
                "reference_hourly_sha256": str(row.hourly_sha256),
                "reference_sql_sha256": sql_hash,
                "reference_sql_path": str(sql_path),
            })
        return records, {
            "run_id": str(row.run_id),
            "annual_gas_kwh": annual_gas,
            "monthly_sum_kwh": monthly_sum,
            "rounding_difference_kwh": rounding_difference,
        }

    extracted: list[tuple[list[dict], dict]] = []
    rows = [row for _, row in selected.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        extracted.extend(pool.map(extract, rows))
    monthly = pd.DataFrame([
        record for records, _ in extracted for record in records
    ]).sort_values([*IDENTITY, "month"], kind="stable")
    checks = pd.DataFrame([check for _, check in extracted])
    audit = {
        "offset_zero_runs": len(selected),
        "monthly_rows": len(monthly),
        "all_runs_have_12_months": bool(
            monthly.groupby("reference_run_id").size().eq(12).all()
        ),
        "maximum_absolute_annual_rounding_difference_kwh": float(
            checks.rounding_difference_kwh.abs().max()
        ),
        "annual_manifest_sha256": sha256(annual_manifest_path),
    }
    if len(monthly) != 12 * len(selected) or not audit["all_runs_have_12_months"]:
        raise ValueError("Monthly natural-gas extraction is incomplete")
    return monthly.reset_index(drop=True), audit


def enrich_seasonal_facility_energy(
    seasonal_metrics: pd.DataFrame,
    monthly_gas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Join matched monthly water-system gas to seasonal command trajectories."""
    required = {
        "run_id", *IDENTITY, "meteorological_block", "facility_electric_kwh",
        "gas_hhv_kwh",
    }
    missing = required - set(seasonal_metrics)
    if missing:
        raise ValueError(f"Seasonal metrics lack {sorted(missing)}")
    if seasonal_metrics.duplicated(["run_id", "meteorological_block"]).any():
        raise ValueError("Seasonal metrics duplicate a run and block")
    unknown_blocks = sorted(
        set(seasonal_metrics.meteorological_block.astype(str)) - set(BLOCK_MONTHS)
    )
    if unknown_blocks:
        raise ValueError(f"Unknown meteorological blocks: {unknown_blocks}")
    if monthly_gas.duplicated([*IDENTITY, "month"]).any():
        raise ValueError("Monthly gas library duplicates an identity and month")

    lookup = monthly_gas.set_index([*IDENTITY, "month"])
    spread = monthly_gas.groupby(["climate", "month"])[
        "water_system_gas_hhv_kwh"
    ].agg(["min", "max"])
    result = seasonal_metrics.copy()
    backgrounds: list[float] = []
    background_low: list[float] = []
    background_high: list[float] = []
    reference_runs: list[str] = []
    reference_hashes: list[str] = []
    for row in result.itertuples(index=False):
        months = BLOCK_MONTHS[str(row.meteorological_block)]
        key = (str(row.climate), int(row.composition_id), str(row.control))
        selected = lookup.loc[[(*key, month) for month in months]]
        backgrounds.append(float(selected.water_system_gas_hhv_kwh.sum()))
        reference_runs.append(";".join(sorted(set(selected.reference_run_id))))
        reference_hashes.append(";".join(sorted(set(selected.reference_sql_sha256))))
        climate_spread = spread.loc[[(str(row.climate), month) for month in months]]
        background_low.append(float(climate_spread["min"].sum()))
        background_high.append(float(climate_spread["max"].sum()))
    result["non_space_heating_facility_gas_kwh"] = backgrounds
    result["non_space_heating_facility_gas_low_kwh"] = background_low
    result["non_space_heating_facility_gas_high_kwh"] = background_high
    result["facility_gas_reference_run_id"] = reference_runs
    result["facility_gas_reference_sql_sha256"] = reference_hashes
    result["facility_gas_hhv_kwh"] = (
        result.gas_hhv_kwh.astype(float)
        + result.non_space_heating_facility_gas_kwh
    )
    result["facility_gas_hhv_kwh_low"] = (
        result.gas_hhv_kwh.astype(float)
        + result.non_space_heating_facility_gas_low_kwh
    )
    result["facility_gas_hhv_kwh_high"] = (
        result.gas_hhv_kwh.astype(float)
        + result.non_space_heating_facility_gas_high_kwh
    )
    result["facility_electric_plus_space_heating_gas_kwh"] = (
        result.facility_electric_kwh.astype(float) + result.gas_hhv_kwh.astype(float)
    )
    result["facility_purchased_kwh"] = (
        result.facility_electric_kwh.astype(float)
        + result.facility_gas_hhv_kwh
    )
    result["total_purchased_kwh"] = result.facility_purchased_kwh
    result["facility_purchased_kwh_low"] = (
        result.facility_electric_kwh.astype(float)
        + result.facility_gas_hhv_kwh_low
    )
    result["facility_purchased_kwh_high"] = (
        result.facility_electric_kwh.astype(float)
        + result.facility_gas_hhv_kwh_high
    )
    numeric = [
        "facility_gas_hhv_kwh", "facility_purchased_kwh",
        "facility_electric_plus_space_heating_gas_kwh",
    ]
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("Reconstructed seasonal facility energy is non-finite")
    if result.facility_gas_hhv_kwh.lt(result.gas_hhv_kwh - 1e-9).any():
        raise ValueError("Reconstructed facility gas is below space-heating gas")

    annual_identity = [
        name for name in (
            "run_id", "climate", "composition_id", "cohort_label",
            "demand_source", "control", "seasonal_contrast_fraction",
        )
        if name in result
    ]
    annual = result.groupby(annual_identity, as_index=False)[[
        "facility_electric_kwh", "gas_hhv_kwh", "facility_gas_hhv_kwh",
        "facility_electric_plus_space_heating_gas_kwh", "facility_purchased_kwh",
        "non_space_heating_facility_gas_kwh",
        "facility_purchased_kwh_low", "facility_purchased_kwh_high",
    ]].sum()
    audit = {
        "seasonal_runs": int(result.run_id.nunique()),
        "seasonal_block_rows": len(result),
        "annual_energy_rows": len(annual),
        "reconstruction": (
            "Seasonal hourly Heating:NaturalGas plus the matched offset-zero "
            "monthly WaterSystems:NaturalGas end use"
        ),
        "maximum_monthly_background_range_across_offset_zero_controls_kwh": float(
            (spread["max"] - spread["min"]).max()
        ),
        "maximum_block_background_range_kwh": float(
            (result.non_space_heating_facility_gas_high_kwh
             - result.non_space_heating_facility_gas_low_kwh).max()
        ),
    }
    return result, annual, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--seasonal-metrics", type=Path, required=True)
    parser.add_argument("--sql-search-root", type=Path, action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-offset-zero-runs", type=int, default=162)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    monthly, extraction = extract_offset_zero_monthly_gas(
        args.annual_manifest,
        sql_search_roots=tuple(args.sql_search_root),
        expected_runs=args.expected_offset_zero_runs,
        workers=args.workers,
    )
    seasonal_path = args.seasonal_metrics.resolve()
    seasonal, annual, reconstruction = enrich_seasonal_facility_energy(
        pd.read_csv(seasonal_path), monthly
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    monthly_path = output / "OFFSET_ZERO_MONTHLY_NATURAL_GAS.csv"
    seasonal_output = output / "SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv"
    annual_output = output / "SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv"
    monthly.to_csv(monthly_path, index=False)
    seasonal.to_csv(seasonal_output, index=False)
    annual.to_csv(annual_output, index=False)
    audit = {
        "status": "PASS",
        "source_seasonal_metrics_sha256": sha256(seasonal_path),
        "extraction": extraction,
        "reconstruction": reconstruction,
        "outputs": {
            path.name: {"rows": len(frame), "sha256": sha256(path)}
            for path, frame in (
                (monthly_path, monthly),
                (seasonal_output, seasonal),
                (annual_output, annual),
            )
        },
    }
    _write_json(output / "SEASONAL_FACILITY_ENERGY_AUDIT.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
