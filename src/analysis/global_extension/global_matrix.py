"""Prepare and audit the controlled global-climate EnergyPlus experiment.

The module separates three operations that must remain explicit:

* apply the existing per-zone thermostat and reporting interface to an
  otherwise unchanged DOE medium-office prototype;
* replace site metadata and design-day sizing with the supplied EPW; and
* declare deterministic probe and control run manifests before simulations.

External DOE prototypes, EPWs and the EnergyPlus runtime are supplied by path.
They are intentionally not bundled with the code-only repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROBE_TARGETS_C = (
    20.0, 20.5, 21.0, 21.5, 22.0, 22.5,
    22.6, 22.7, 22.8, 22.9, 23.0, 23.1, 23.2, 23.3, 23.4,
    23.5, 23.6, 23.7, 23.8, 23.9, 24.0, 24.1, 24.2, 24.3, 24.4,
    24.5, 25.0, 25.5, 26.0, 26.5, 27.0,
)
CONTROLS = ("uniform_average", "personal_center", "personal_range")
OFFSETS_C = (-1.0, -0.5, 0.0, 0.5, 1.0)
TRANSITIONS = (
    ("22.1", "22.2", "Transition-V22-1-0-to-V22-2-0.exe"),
    ("22.2", "23.1", "Transition-V22-2-0-to-V23-1-0.exe"),
    ("23.1", "23.2", "Transition-V23-1-0-to-V23-2-0.exe"),
    ("23.2", "24.1", "Transition-V23-2-0-to-V24-1-0.exe"),
    ("24.1", "24.2", "Transition-V24-1-0-to-V24-2-0.exe"),
    ("24.2", "25.1", "Transition-V24-2-0-to-V25-1-0.exe"),
    ("25.1", "25.2", "Transition-V25-1-0-to-V25-2-0.exe"),
    ("25.2", "26.1", "Transition-V25-2-0-to-V26-1-0.exe"),
    ("26.1", "26.2", "Transition-V26-1-0-to-V26-2-0.exe"),
)


@dataclass(frozen=True)
class ClimateSpec:
    climate: str
    country: str
    continent: str
    prototype_variant: str
    weather: str
    sentinel: bool = False
    latitude: float | None = None
    longitude: float | None = None
    climate_zone: str = ""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_idf(text: str) -> list[list[str]]:
    """Parse IDF objects while preserving duplicate object types and names."""
    stripped = re.sub(r"!.*", "", text)
    return [
        [field.strip() for field in block.split(",")]
        for block in stripped.split(";")
        if block.strip()
    ]


def serialize_idf(objects: Iterable[Sequence[str]]) -> str:
    """Write transition-safe IDF text with one field per line."""
    blocks = []
    for obj in objects:
        fields = [str(value).strip() for value in obj]
        if not fields:
            continue
        blocks.append(",\n  ".join(fields) + ";")
    return "\n\n".join(blocks) + "\n"


def _named(objects: Iterable[list[str]], object_type: str) -> dict[str, list[str]]:
    return {
        obj[1].strip().lower(): obj
        for obj in objects
        if len(obj) > 1 and obj[0].strip().lower() == object_type.lower()
    }


def _interface_object(obj: Sequence[str]) -> bool:
    kind = obj[0].strip().lower()
    name = obj[1].strip().lower() if len(obj) > 1 else ""
    if kind in {"output:variable", "output:meter", "output:sqlite"}:
        return True
    if kind == "thermostatsetpoint:dualsetpoint":
        return True
    if kind == "schedule:compact" and name.endswith(
        (" independent heating", " independent cooling")
    ):
        return True
    if kind == "energymanagementsystem:actuator" and "_independent_" in name:
        return True
    return kind == "energymanagementsystem:program" and name in {
        "oa_based_optimum_start_ems_program",
        "pacu_vav_bot_max_oa_frac",
        "pacu_vav_mid_max_oa_frac",
        "pacu_vav_top_max_oa_frac",
    }


def non_interface_fingerprint(text: str) -> str:
    retained = [obj for obj in parse_idf(text) if not _interface_object(obj)]
    normalized = json.dumps(retained, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def patch_control_interface(source_text: str, reference_text: str) -> str:
    """Transfer only the tested per-zone control/reporting interface.

    ``source_text`` supplies the climate-specific prototype. ``reference_text``
    is the frozen medium-office model containing the independent thermostat
    schedules, Energy Management System links and hourly reporting requests.
    """
    source = parse_idf(source_text)
    reference = parse_idf(reference_text)
    replacements: dict[tuple[str, str], list[str]] = {}
    additions: list[list[str]] = []
    for obj in reference:
        if not _interface_object(obj):
            continue
        kind = obj[0].strip().lower()
        name = obj[1].strip().lower() if len(obj) > 1 else ""
        if kind in {"thermostatsetpoint:dualsetpoint", "energymanagementsystem:program"}:
            replacements[(kind, name)] = obj
        else:
            additions.append(obj)

    output_types = {"output:variable", "output:meter", "output:sqlite"}
    patched: list[list[str]] = []
    seen_replacements: set[tuple[str, str]] = set()
    for obj in source:
        kind = obj[0].strip().lower()
        name = obj[1].strip().lower() if len(obj) > 1 else ""
        if kind in output_types:
            continue
        key = (kind, name)
        if key in replacements:
            patched.append(list(replacements[key]))
            seen_replacements.add(key)
        else:
            patched.append(obj)
    missing = sorted(set(replacements) - seen_replacements)
    if missing:
        raise ValueError(f"Source prototype lacks interface objects: {missing}")
    patched.extend([list(obj) for obj in additions])
    result = serialize_idf(patched)

    # Verify that the operation changed only the declared interface.
    if non_interface_fingerprint(source_text) != non_interface_fingerprint(result):
        raise ValueError("Control-interface patch changed an undeclared physical object")
    schedules = [
        obj for obj in parse_idf(result)
        if obj[0].lower() == "schedule:compact"
        and obj[1].lower().endswith((" independent heating", " independent cooling"))
    ]
    if len(schedules) != 30:
        raise ValueError(f"Expected 30 independent thermostat schedules, found {len(schedules)}")
    return result


def parse_epw_header(path: Path) -> dict:
    """Read location and extreme-week metadata from an EPW."""
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = [next(csv.reader(stream)) for _ in range(8)]
    location = rows[0]
    if not location or location[0].strip().upper() != "LOCATION":
        raise ValueError(f"Invalid EPW LOCATION header: {path}")
    if len(location) < 10:
        raise ValueError(f"Incomplete EPW LOCATION header: {path}")
    periods = rows[2]
    parsed_periods = []
    if periods and periods[0].strip().upper() == "TYPICAL/EXTREME PERIODS":
        count = int(periods[1])
        values = periods[2:]
        for index in range(count):
            start = 4 * index
            if start + 3 >= len(values):
                break
            parsed_periods.append({
                "name": values[start].strip(),
                "kind": values[start + 1].strip(),
                "start": values[start + 2].strip(),
                "end": values[start + 3].strip(),
            })
    return {
        "city": location[1].strip(),
        "state": location[2].strip(),
        "country": location[3].strip(),
        "source": location[4].strip(),
        "station": location[5].strip(),
        "latitude": float(location[6]),
        "longitude": float(location[7]),
        "time_zone": float(location[8]),
        "elevation_m": float(location[9]),
        "periods": parsed_periods,
    }


def _month_day(value: str) -> tuple[int, int]:
    month, day = value.replace(" ", "").split("/", 1)
    return int(month), int(day)


def _computed_extreme_weeks(path: Path) -> tuple[tuple[int, int, int, int], ...]:
    rows = pd.read_csv(path, skiprows=8, header=None, usecols=[1, 2, 6],
                       names=["month", "day", "dry_bulb_c"])
    daily = rows.groupby(["month", "day"], sort=False).dry_bulb_c.mean().reset_index()
    if len(daily) < 14:
        raise ValueError("EPW must contain at least fourteen complete days")
    rolling = daily.dry_bulb_c.rolling(7).mean().dropna()
    indices = (int(rolling.idxmin()), int(rolling.idxmax()))
    result = []
    for end_index in indices:
        start_index = end_index - 6
        start = daily.iloc[start_index]
        end = daily.iloc[end_index]
        result.append((int(start.month), int(start.day), int(end.month), int(end.day)))
    return tuple(result)


def extreme_weeks(path: Path) -> dict[str, tuple[int, int, int, int]]:
    metadata = parse_epw_header(path)
    extremes = [p for p in metadata["periods"] if p["kind"].lower() == "extreme"]

    def select(words: tuple[str, ...]):
        for period in extremes:
            if any(word in period["name"].lower() for word in words):
                sm, sd = _month_day(period["start"])
                em, ed = _month_day(period["end"])
                return sm, sd, em, ed
        return None

    heating = select(("winter", "min", "cold"))
    cooling = select(("summer", "max", "hot"))
    if heating is None or cooling is None:
        computed_heating, computed_cooling = _computed_extreme_weeks(path)
        heating = heating or computed_heating
        cooling = cooling or computed_cooling
    return {"heating": heating, "cooling": cooling}


def replace_site_and_sizing(text: str, epw: Path) -> str:
    """Use the supplied weather location and its coldest/hottest weeks."""
    objects = parse_idf(text)
    excluded = {
        "site:location",
        "sizingperiod:designday",
        "sizingperiod:weatherfiledays",
        "sizingperiod:weatherfileconditiontype",
    }
    retained = [obj for obj in objects if obj[0].strip().lower() not in excluded]
    header = parse_epw_header(epw)
    weeks = extreme_weeks(epw)
    location = [
        "Site:Location", header["city"], header["latitude"], header["longitude"],
        header["time_zone"], header["elevation_m"],
    ]
    heating = [
        "SizingPeriod:WeatherFileDays", "EPW cold extreme week", *weeks["heating"],
        "WinterDesignDay", "No", "Yes",
    ]
    cooling = [
        "SizingPeriod:WeatherFileDays", "EPW hot extreme week", *weeks["cooling"],
        "SummerDesignDay", "No", "Yes",
    ]
    insertion = 1 if retained and retained[0][0].strip().lower() == "version" else 0
    result = retained[:insertion] + [location, heating, cooling] + retained[insertion:]
    return serialize_idf(result)


def idf_version(text: str) -> str:
    versions = [obj[1] for obj in parse_idf(text) if obj[0].strip().lower() == "version"]
    if len(versions) != 1:
        raise ValueError("Expected exactly one IDF Version object")
    return versions[0].strip()


def transition_idf(input_path: Path, output_path: Path, updater_dir: Path,
                   target_version: str = "26.2") -> list[dict]:
    """Transition a copy of an IDF through each official version updater."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(input_path, output_path)
    current = idf_version(output_path.read_text(encoding="utf-8", errors="replace"))
    records = []
    while current != target_version:
        match = next((step for step in TRANSITIONS if step[0] == current), None)
        if match is None:
            raise ValueError(f"No declared transition from EnergyPlus {current}")
        source, destination, executable = match
        program = updater_dir / executable
        if not program.is_file():
            raise FileNotFoundError(program)
        # Each legacy transition creates the same .idfold/.idfnew names. On
        # Windows the process can briefly retain a file handle after exit, so
        # clear those backups before the next version step with a bounded retry.
        for suffix in ("idfnew", "idfold", "VCpErr", "VCperr"):
            artifact = output_path.with_suffix("." + suffix)
            for attempt in range(10):
                try:
                    artifact.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 9:
                        raise
                    time.sleep(0.2)
        completed = subprocess.run(
            [str(program), str(output_path.resolve())], cwd=updater_dir,
            capture_output=True, text=True, timeout=180, check=False,
        )
        records.append({
            "source_version": source,
            "target_version": destination,
            "executable": executable,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        })
        recovered = False
        if completed.returncode != 0:
            # Some Windows updater builds finish conversion, write .idfnew,
            # then fail while deleting the prior backup. Accept only a file
            # whose Version object proves that the requested step completed.
            candidates = (output_path, output_path.with_suffix(".idfnew"))
            valid = next((candidate for candidate in candidates if candidate.is_file()
                          and idf_version(candidate.read_text(
                              encoding="utf-8", errors="replace")) == destination), None)
            if valid is None:
                raise RuntimeError(
                    f"{executable} failed with {completed.returncode}: "
                    f"{completed.stderr[-1000:]}"
                )
            if valid != output_path:
                shutil.copyfile(valid, output_path)
            records[-1]["recovered_from_written_target"] = True
            recovered = True
        current = idf_version(output_path.read_text(encoding="utf-8", errors="replace"))
        if current != destination:
            raise ValueError(f"{executable} produced Version {current}, expected {destination}")
        if not recovered:
            records[-1]["recovered_from_written_target"] = False
    return records


def run_id(prefix: str, payload: dict) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    return f"{prefix}_{digest}"


def probe_manifest(climates: Sequence[ClimateSpec]) -> pd.DataFrame:
    rows = []
    for climate in climates:
        for target in PROBE_TARGETS_C:
            payload = {"climate": climate.climate, "target_c": target, "kind": "probe"}
            rows.append(asdict(climate) | payload | {"run_id": run_id("probe", payload)})
    return pd.DataFrame(rows)


def control_manifest(
    climates: Sequence[ClimateSpec],
    compositions: Sequence[str],
    candidate_scope: str = "sentinel",
) -> pd.DataFrame:
    """Declare direct runs, with five candidates at sentinels or all climates."""
    if candidate_scope not in {"sentinel", "all"}:
        raise ValueError(f"Unknown candidate scope: {candidate_scope}")
    rows = []
    for climate in climates:
        offsets = (
            OFFSETS_C
            if candidate_scope == "all" or climate.sentinel
            else (0.0,)
        )
        for composition in compositions:
            for control in CONTROLS:
                for offset in offsets:
                    payload = {
                        "climate": climate.climate,
                        "composition": str(composition),
                        "control": control,
                        "offset_c": offset,
                        "kind": "annual_control",
                    }
                    rows.append(asdict(climate) | payload | {"run_id": run_id("annual", payload)})
    return pd.DataFrame(rows)


def _column(frame: pd.DataFrame, prefix: str, fragment: str) -> np.ndarray:
    choices = [
        column for column in frame
        if column.split(":", 1)[0].strip().upper() == prefix.strip().upper()
        and fragment.lower() in column.lower()
    ]
    if len(choices) != 1:
        raise ValueError(f"Expected one {prefix}/{fragment} column, found {choices}")
    return frame[choices[0]].to_numpy(float)


def _meter(frame: pd.DataFrame, name: str) -> np.ndarray:
    choices = [column for column in frame if column.split(" [", 1)[0] == name]
    if len(choices) != 1:
        raise ValueError(f"Expected one {name} meter, found {choices}")
    return frame[choices[0]].to_numpy(float) / 3.6e6


def _clock(sql_path: Path) -> pd.DataFrame:
    with sqlite3.connect(sql_path) as connection:
        clock = pd.read_sql_query("select * from Time order by TimeIndex", connection)
    if len(clock) != 8760 or not clock.Interval.eq(60).all():
        raise ValueError("Expected one complete hourly annual clock")
    return clock


def _annual_facility_gas_hhv_kwh(sql_path: Path) -> float:
    """Read annual whole-building natural gas from the EnergyPlus summary."""
    with sqlite3.connect(sql_path) as connection:
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
        raise ValueError("Expected one annual facility natural-gas total in GJ")
    value = float(str(values.value.iloc[0]).strip()) * 1_000 / 3.6
    if not np.isfinite(value) or value < 0:
        raise ValueError("Annual facility natural-gas total must be finite and non-negative")
    return value


def extract_hourly(run_folder: Path, zones: Sequence[str], output_path: Path) -> dict:
    """Extract the compact annual record used by service and energy analysis."""
    frame = pd.read_csv(run_folder / "eplusout.csv")
    clock = _clock(run_folder / "eplusout.sql")
    if len(frame) != 8760:
        raise ValueError("Expected 8,760 hourly output rows")
    temperatures = np.column_stack([
        _column(frame, zone, "Zone Operative Temperature") for zone in zones
    ])
    people = np.column_stack([
        _column(frame, zone, "Zone People Occupant Count") for zone in zones
    ])
    minute = (clock.Hour - 1) * 60 + 30 + clock.Dst * 60
    occupied = (
        clock.DayType.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        & minute.ge(360) & minute.lt(1320)
    ) | (clock.DayType.eq("Saturday") & minute.ge(360) & minute.lt(1080))
    arrays = {
        "zone_ids": np.asarray(zones),
        "temperatures": temperatures,
        "people": people,
        "occupied": occupied.to_numpy(bool),
        "month": clock.Month.to_numpy(int),
        "day": clock.Day.to_numpy(int),
        "hour": clock.Hour.to_numpy(int),
        "dst": clock.Dst.to_numpy(int),
        "facility_electric_kwh": _meter(frame, "Electricity:Facility"),
        "hvac_electric_kwh": _meter(frame, "Electricity:HVAC"),
        "heating_electric_kwh": _meter(frame, "Heating:Electricity"),
        "cooling_electric_kwh": _meter(frame, "Cooling:Electricity"),
        "fans_electric_kwh": _meter(frame, "Fans:Electricity"),
        "gas_hhv_kwh": _meter(frame, "Heating:NaturalGas"),
        "facility_gas_hhv_kwh_annual": np.asarray(
            _annual_facility_gas_hhv_kwh(run_folder / "eplusout.sql")
        ),
    }
    if not all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind != "U"):
        raise ValueError("Non-finite hourly output detected")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    error = (run_folder / "eplusout.err").read_text(errors="replace")
    summary = re.findall(
        r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;",
        error,
    )
    recurring = re.findall(
        r"\*\* Warning \*\* (.+?)\n\s*\*\*\s+~~~\s+\*\*\s+This error occurred (\d+) total times;",
        error,
    )
    recurring_groups = [
        {"message": message.strip(), "occurrences": int(count),
         "source_service_water": "water temperature" in message.lower()}
        for message, count in recurring
    ]
    numerical_tokens = (
        "SimHVAC:", "PsyTwb", "PsyPsat", "Temperature out of range",
        "WetBulb not converged", "CalcDoe2DXCoil:",
    )
    direct_warnings = [
        line.strip() for line in error.splitlines()
        if line.lstrip().startswith("** Warning **")
    ]
    numerical = [
        line for line in direct_warnings if any(token in line for token in numerical_tokens)
    ]
    qa = {
        "hours": len(frame),
        "zones": len(zones),
        "occupied_hours": int(occupied.sum()),
        "reported_warning_count": int(summary[-1][0]) if summary else None,
        "reported_severe_count": int(summary[-1][1]) if summary else None,
        "fatal_occurrences": len(re.findall(r"\*\*\s+Fatal\s+\*\*", error)),
        "direct_numerical_warnings": numerical,
        "unexplained_recurring_groups": [
            group for group in recurring_groups if not group["source_service_water"]
        ],
        "completed": bool(summary),
        "hourly_sha256": sha256(output_path),
    }
    qa["engineering_pass"] = bool(
        qa["completed"] and qa["reported_severe_count"] == 0 and qa["fatal_occurrences"] == 0
        and not qa["direct_numerical_warnings"]
        and not qa["unexplained_recurring_groups"]
    )
    return qa


def extract_probe(run_folder: Path, zones: Sequence[str], target_c: float) -> dict:
    temporary = run_folder / "_probe_hourly.npz"
    qa = extract_hourly(run_folder, zones, temporary)
    with np.load(temporary, allow_pickle=False) as saved:
        occupied = saved["occupied"].astype(bool)
        result = {
            "uniform_target_c": float(target_c),
            "operative_c": saved["temperatures"][occupied],
            "native_people": saved["people"][occupied],
            "purchased_site_kwh": float(
                saved["hvac_electric_kwh"].sum() + saved["gas_hhv_kwh"].sum()
            ),
            "zone_ids": saved["zone_ids"],
            "occupied_source_indices": np.flatnonzero(occupied),
            "qa": qa,
        }
    temporary.unlink()
    return result


def run_energyplus(
    input_idf: Path,
    weather: Path,
    executable: Path,
    run_folder: Path,
    compact_output: Path,
    zones: Sequence[str],
    *,
    cleanup_raw: bool = False,
    timeout_seconds: int = 900,
) -> dict:
    """Run one annual case, extract its compact trajectory and write QA."""
    from thermal_heterogeneity.energyplus_launcher import ROOT as project_root
    if not all(path.resolve().is_relative_to(project_root) for path in (run_folder, compact_output)):
        raise ValueError("Retained simulation outputs must remain inside this project")
    if run_folder.exists() and any(run_folder.iterdir()):
        raise FileExistsError(f"Run folder is not empty: {run_folder}")
    run_folder.mkdir(parents=True, exist_ok=True)
    from thermal_heterogeneity.energyplus_launcher import run_with_scratch, runtime_provenance
    arguments = ["-w", str(weather.resolve()), "-d", str(run_folder.resolve()), "-r", str(input_idf.resolve())]
    started = time.monotonic()
    with (run_folder / "CONSOLE.log").open("wb") as log:
        completed = run_with_scratch(executable, arguments, run_folder, stdout=log,
            timeout_seconds=timeout_seconds)
    error_path = run_folder / "eplusout.err"
    error = error_path.read_text(errors="replace") if error_path.is_file() else ""
    completion = re.findall(
        r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;",
        error,
    )
    qa = {
        "exit_code": completed.returncode,
        "seconds": time.monotonic() - started,
        "completed": completed.returncode == 0 and bool(completion),
        "reported_warning_count": int(completion[-1][0]) if completion else None,
        "reported_severe_count": int(completion[-1][1]) if completion else None,
        "fatal_occurrences": len(re.findall(r"\*\*\s+Fatal\s+\*\*", error)),
        "input_idf_sha256": sha256(input_idf),
        "weather_sha256": sha256(weather),
        "runtime_sha256": sha256(executable),
        "compact_output": str(compact_output.resolve()),
        **runtime_provenance(executable),
        "execution_status": completed.execution_status,
        "execution_audit_path": completed.execution_audit_path,
    }
    if qa["completed"]:
        qa.update(extract_hourly(run_folder, zones, compact_output))
    else:
        qa["engineering_pass"] = False
    qa_path = run_folder / "RUN_QA.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    if cleanup_raw and qa["completed"]:
        # Sizing and monthly gas tables remain inputs to later system and
        # seasonal analyses after hourly arrays have been exported.
        sql_path = run_folder / "eplusout.sql"
        metadata_path = run_folder / "simulation_metadata.sql"
        qa["source_sql_sha256"] = sha256(sql_path)
        with sqlite3.connect(sql_path) as source, sqlite3.connect(metadata_path) as target:
            tables = source.execute(
                "select name, sql from sqlite_master where type='table' and name not like 'sqlite_%'"
            ).fetchall()
            counts = {}
            for name, schema in tables:
                quoted = '"' + name.replace('"', '""') + '"'
                target.execute(schema)
                if name == "ReportData":
                    continue
                cursor = source.execute("select * from " + quoted)
                marks = ",".join("?" for _ in cursor.description)
                count = 0
                while values := cursor.fetchmany(10000):
                    target.executemany("insert into " + quoted + " values(" + marks + ")", values)
                    count += len(values)
                cursor.close()
                if target.execute("select count(*) from " + quoted).fetchone()[0] != count:
                    raise ValueError(f"SQL metadata copy incomplete: {name}")
                counts[name] = count
            target.commit()
        source.close()
        target.close()
        qa["sql_metadata_path"] = str(metadata_path.resolve())
        qa["sql_metadata_sha256"] = sha256(metadata_path)
        qa["sql_metadata_table_counts"] = counts
        qa["sql_retention"] = "All metadata and tabular/sizing results; hourly ReportData values retained in compact NPZ for the declared analysis variables"
        retained = {
            (run_folder / name).resolve()
            for name in ("CONSOLE.log", "RUN_QA.json", "eplusout.err", "eplusout.eio", "eplusout.audit", "simulation_metadata.sql")
        }
        retained.add(compact_output.resolve())
        retained.add(Path(completed.execution_audit_path).resolve())
        for path in run_folder.iterdir():
            if path.is_file() and path.resolve() not in retained:
                for attempt in range(20):
                    try:
                        path.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        if attempt == 19:
                            # Cleanup is a storage optimization. A valid run
                            # must not be reclassified because Windows retains
                            # a short-lived SQLite handle.
                            qa.setdefault("cleanup_deferred", []).append(path.name)
                            break
                        time.sleep(0.25)
        qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


def write_manifests(climate_json: Path, output_root: Path) -> None:
    data = json.loads(climate_json.read_text(encoding="utf-8"))
    climates = [ClimateSpec(**row) for row in data["climates"]]
    compositions = [str(value) for value in data["compositions"]]
    output_root.mkdir(parents=True, exist_ok=True)
    probes = probe_manifest(climates)
    controls = control_manifest(climates, compositions)
    probes.to_csv(output_root / "PROBE_RUN_MANIFEST.csv", index=False)
    controls.to_csv(output_root / "ANNUAL_CONTROL_RUN_MANIFEST.csv", index=False)
    report = {
        "climates": len(climates),
        "compositions": len(compositions),
        "probe_runs": len(probes),
        "annual_control_runs": len(controls),
        "sentinel_climates": [climate.climate for climate in climates if climate.sentinel],
        "probe_targets_c": list(PROBE_TARGETS_C),
        "controls": list(CONTROLS),
        "candidate_offsets_c": list(OFFSETS_C),
    }
    (output_root / "GLOBAL_MATRIX_PROTOCOL.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifests")
    manifest.add_argument("--climates", type=Path, required=True)
    manifest.add_argument("--output-root", type=Path, required=True)

    site = sub.add_parser("site-sizing")
    site.add_argument("--input-idf", type=Path, required=True)
    site.add_argument("--weather", type=Path, required=True)
    site.add_argument("--output-idf", type=Path, required=True)

    patch = sub.add_parser("patch-interface")
    patch.add_argument("--input-idf", type=Path, required=True)
    patch.add_argument("--reference-idf", type=Path, required=True)
    patch.add_argument("--output-idf", type=Path, required=True)

    transition = sub.add_parser("transition")
    transition.add_argument("--input-idf", type=Path, required=True)
    transition.add_argument("--output-idf", type=Path, required=True)
    transition.add_argument("--updater-dir", type=Path, required=True)
    transition.add_argument("--target-version", default="26.2")

    extract = sub.add_parser("extract-hourly")
    extract.add_argument("--run-folder", type=Path, required=True)
    extract.add_argument("--zones", nargs="+", required=True)
    extract.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run")
    run.add_argument("--input-idf", type=Path, required=True)
    run.add_argument("--weather", type=Path, required=True)
    run.add_argument("--executable", type=Path, required=True)
    run.add_argument("--run-folder", type=Path, required=True)
    run.add_argument("--compact-output", type=Path, required=True)
    run.add_argument("--zones", nargs="+", required=True)
    run.add_argument("--cleanup-raw", action="store_true")
    run.add_argument("--timeout-seconds", type=int, default=900)

    args = parser.parse_args()
    if args.command == "manifests":
        write_manifests(args.climates.resolve(), args.output_root.resolve())
    elif args.command == "site-sizing":
        text = replace_site_and_sizing(
            args.input_idf.read_text(encoding="utf-8", errors="replace"),
            args.weather.resolve(),
        )
        args.output_idf.parent.mkdir(parents=True, exist_ok=True)
        args.output_idf.write_text(text, encoding="utf-8")
    elif args.command == "patch-interface":
        text = patch_control_interface(
            args.input_idf.read_text(encoding="utf-8", errors="replace"),
            args.reference_idf.read_text(encoding="utf-8", errors="replace"),
        )
        args.output_idf.parent.mkdir(parents=True, exist_ok=True)
        args.output_idf.write_text(text, encoding="utf-8")
    elif args.command == "transition":
        records = transition_idf(
            args.input_idf.resolve(), args.output_idf.resolve(),
            args.updater_dir.resolve(), args.target_version,
        )
        print(json.dumps(records, indent=2))
    elif args.command == "extract-hourly":
        qa = extract_hourly(args.run_folder.resolve(), args.zones, args.output.resolve())
        print(json.dumps(qa, indent=2))
    elif args.command == "run":
        qa = run_energyplus(
            args.input_idf.resolve(), args.weather.resolve(), args.executable.resolve(),
            args.run_folder.resolve(), args.compact_output.resolve(), args.zones,
            cleanup_raw=args.cleanup_raw, timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
