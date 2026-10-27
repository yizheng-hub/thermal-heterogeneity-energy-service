"""Add hourly facility electricity to completed compact EnergyPlus records.

The command is intentionally post-processing only.  It refuses incomplete
manifests, reads the authoritative hourly ``Electricity:Facility`` meter from
each retained EnergyPlus SQLite file, updates each compact record atomically,
and then refreshes the compact hashes in the run QA and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import numpy as np
import pandas as pd


EXPECTED_HOURS = 8760


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _completed(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    parsed = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError("Manifest contains invalid completion flags")
    return parsed.astype(bool)


def _extract_hourly_meter_kwh(sql_path: Path, meter: str) -> np.ndarray:
    """Read one complete hourly joule meter from EnergyPlus SQL in kWh."""
    if not sql_path.is_file():
        raise FileNotFoundError(f"EnergyPlus SQLite output is missing: {sql_path}")
    uri = f"file:{sql_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        dictionaries = pd.read_sql_query(
            """
            select ReportDataDictionaryIndex as dictionary_index, Units as units
            from ReportDataDictionary
            where upper(Name) = upper(?) and upper(ReportingFrequency) = 'HOURLY'
            """,
            connection,
            params=(meter,),
        )
        if len(dictionaries) != 1:
            raise ValueError(
                f"Expected one hourly {meter} dictionary entry in "
                f"{sql_path}, found {len(dictionaries)}"
            )
        if str(dictionaries.units.iloc[0]).strip().upper() != "J":
            raise ValueError(
                f"{meter} must be reported in joules: {sql_path}"
            )
        values = pd.read_sql_query(
            """
            select rd.Value as value
            from ReportData as rd
            join Time as t on rd.TimeIndex = t.TimeIndex
            where rd.ReportDataDictionaryIndex = ?
            order by rd.TimeIndex
            """,
            connection,
            params=(int(dictionaries.dictionary_index.iloc[0]),),
        ).value.to_numpy(float)
    if values.shape != (EXPECTED_HOURS,) or not np.isfinite(values).all():
        raise ValueError(
            f"Expected {EXPECTED_HOURS} finite hourly {meter} values: {sql_path}"
        )
    if (values < 0).any():
        raise ValueError(f"{meter} contains negative values: {sql_path}")
    return values / 3.6e6


def extract_facility_electric_kwh(sql_path: Path) -> np.ndarray:
    """Read the complete hourly facility-electricity meter from EnergyPlus SQL."""
    return _extract_hourly_meter_kwh(sql_path, "Electricity:Facility")


def extract_hvac_electric_kwh(sql_path: Path) -> np.ndarray:
    """Read the complete hourly HVAC-electricity meter from EnergyPlus SQL."""
    return _extract_hourly_meter_kwh(sql_path, "Electricity:HVAC")


def extract_annual_facility_gas_hhv_kwh(sql_path: Path) -> float:
    """Read whole-building annual natural gas from the EnergyPlus end-use table."""
    uri = f"file:{sql_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        values = pd.read_sql_query(
            """
            select Value as value, Units as units
            from TabularDataWithStrings
            where ReportName = 'AnnualBuildingUtilityPerformanceSummary'
              and TableName = 'End Uses'
              and RowName = 'Total End Uses'
              and ColumnName = 'Natural Gas'
            """,
            connection,
        )
    if len(values) != 1 or str(values.units.iloc[0]).strip().upper() != "GJ":
        raise ValueError(
            f"Expected one annual facility natural-gas total in GJ: {sql_path}"
        )
    value = float(str(values.value.iloc[0]).strip()) * 1_000 / 3.6
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"Invalid annual facility natural-gas total: {sql_path}")
    return value


def _deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ atomically with stable ordering and ZIP metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.asarray(arrays[name]), allow_pickle=False
                )
                metadata = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                metadata.compress_type = zipfile.ZIP_DEFLATED
                metadata.external_attr = 0o600 << 16
                archive.writestr(metadata, buffer.getvalue(), compresslevel=9)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved(value: object, manifest_path: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else manifest_path.parent / path


def _run_folder(row: pd.Series, manifest_path: Path) -> Path:
    if "run_folder" in row and pd.notna(row["run_folder"]):
        return _resolved(row["run_folder"], manifest_path)
    if "qa_path" in row and pd.notna(row["qa_path"]):
        return _resolved(row["qa_path"], manifest_path).parent
    return manifest_path.parent / "runs" / str(row["run_id"])


def _verify_sql_provenance(row: pd.Series, sql_path: Path) -> None:
    qa_path = sql_path.parent / "RUN_QA.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"SQL provenance QA is missing: {qa_path}")
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if not bool(qa.get("completed", False)):
        raise ValueError(f"SQL provenance run is incomplete: {qa_path}")
    declared_input = str(row.get("input_sha256", "")).strip().lower()
    observed_input = str(qa.get("input_idf_sha256", "")).strip().lower()
    if declared_input and declared_input != "nan" and observed_input != declared_input:
        raise ValueError(f"SQL input hash differs from manifest for {row.run_id}")


def enrich_manifest(
    manifest_path: Path,
    output_manifest_path: Path | None = None,
    sql_search_roots: tuple[Path, ...] = (),
) -> dict:
    """Enrich every completed compact from its exact retained EnergyPlus SQL."""
    manifest_path = manifest_path.resolve()
    output_manifest_path = (
        manifest_path if output_manifest_path is None else output_manifest_path.resolve()
    )
    frame = pd.read_csv(manifest_path)
    required = {"run_id", "hourly_path"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Manifest lacks required fields: {sorted(missing)}")
    if frame.run_id.duplicated().any():
        raise ValueError("Manifest contains duplicate run identifiers")
    if "completed" not in frame or not _completed(frame.completed).all():
        raise RuntimeError(
            "Facility-energy enrichment requires a fully completed run manifest"
        )

    alternate_sql: dict[str, list[Path]] = {}
    for root in sql_search_roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"SQL search root is missing: {root}")
        for candidate in root.rglob("eplusout.sql"):
            run_id = candidate.parent.name
            alternate_sql.setdefault(run_id, []).append(candidate)

    sql_paths: dict[str, Path] = {}
    missing_sql: list[str] = []
    for _, row in frame.iterrows():
        direct_path = _run_folder(row, manifest_path) / "eplusout.sql"
        sql_path: Path | None = direct_path if direct_path.is_file() else None
        if sql_path is None:
            for candidate in alternate_sql.get(str(row.run_id), []):
                try:
                    _verify_sql_provenance(row, candidate)
                except (FileNotFoundError, ValueError):
                    continue
                sql_path = candidate
                break
        if sql_path is None:
            missing_sql.append(str(row.run_id))
        else:
            _verify_sql_provenance(row, sql_path)
            sql_paths[str(row.run_id)] = sql_path
    if missing_sql:
        raise FileNotFoundError(
            "Exact EnergyPlus SQL is unavailable for these runs: "
            + ", ".join(sorted(missing_sql))
        )

    changed = 0
    for index, row in frame.iterrows():
        run_folder = _run_folder(row, manifest_path)
        sql_path = sql_paths[str(row.run_id)]
        compact_path = _resolved(row.hourly_path, manifest_path)
        qa_path = (
            _resolved(row.qa_path, manifest_path)
            if "qa_path" in row and pd.notna(row.qa_path)
            else run_folder / "RUN_QA.json"
        )
        if not compact_path.is_file():
            raise FileNotFoundError(f"Compact trajectory is missing: {compact_path}")
        if not qa_path.is_file():
            raise FileNotFoundError(f"Run QA is missing: {qa_path}")
        with np.load(compact_path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        if "month" not in arrays or len(arrays["month"]) != EXPECTED_HOURS:
            raise ValueError(f"Compact trajectory has an invalid clock: {compact_path}")

        facility = extract_facility_electric_kwh(sql_path)
        hvac = extract_hvac_electric_kwh(sql_path)
        facility_gas = extract_annual_facility_gas_hhv_kwh(sql_path)
        existing_facility = arrays.get("facility_electric_kwh")
        existing_hvac = arrays.get("hvac_electric_kwh")
        existing_gas = arrays.get("facility_gas_hhv_kwh_annual")
        gas_matches = (
            existing_gas is not None
            and np.asarray(existing_gas).size == 1
            and float(np.asarray(existing_gas).reshape(-1)[0]) == facility_gas
        )
        if (
            existing_facility is None
            or not np.array_equal(np.asarray(existing_facility), facility)
            or existing_hvac is None
            or not np.array_equal(np.asarray(existing_hvac), hvac)
            or not gas_matches
        ):
            arrays["facility_electric_kwh"] = facility
            arrays["hvac_electric_kwh"] = hvac
            arrays["facility_gas_hhv_kwh_annual"] = np.asarray(facility_gas)
            _deterministic_npz(compact_path, arrays)
            changed += 1

        compact_hash = sha256(compact_path)
        sql_hash = sha256(sql_path)
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if not bool(qa.get("completed", False)):
            raise RuntimeError(f"Run QA is not complete: {qa_path}")
        qa.update({
            "hourly_sha256": compact_hash,
            "facility_energy_enriched": True,
            "facility_energy_meter": "Electricity:Facility",
            "hvac_energy_meter": "Electricity:HVAC",
            "facility_energy_source": "EnergyPlus SQL",
            "facility_gas_source": (
                "AnnualBuildingUtilityPerformanceSummary/End Uses/"
                "Total End Uses/Natural Gas"
            ),
            "facility_energy_sql_sha256": sql_hash,
            "facility_energy_sql_path": str(sql_path.resolve()),
        })
        _atomic_json(qa_path, qa)
        frame.at[index, "hourly_sha256"] = compact_hash
        frame.at[index, "facility_energy_enriched"] = True
        frame.at[index, "facility_energy_source"] = "EnergyPlus SQL"
        frame.at[index, "facility_energy_sql_sha256"] = sql_hash
        frame.at[index, "facility_energy_sql_path"] = str(sql_path.resolve())

    _atomic_csv(output_manifest_path, frame)
    return {
        "status": "PASS",
        "runs": len(frame),
        "compacts_changed": changed,
        "direct_sql_runs": len(frame),
        "manifest": str(output_manifest_path),
        "manifest_sha256": sha256(output_manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--sql-search-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    audit = enrich_manifest(
        args.manifest, args.output_manifest, tuple(args.sql_search_root)
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
