"""Simulate an unchanged year-start history followed by one response event."""

from common import *
import numpy as np
import pandas as pd
import sqlite3
import subprocess
import re
import time
import zstandard as zstd
from global_matrix import parse_idf, serialize_idf
from hvac_models import idd_fields
from run_global_matrix import (
    ZONES,
    warning_profile,
    relative_engineering_qa,
    set_maximum_hvac_iterations,
)
from lossless_history_storage import retain_sql_metadata, save_states_against
from thermal_heterogeneity.energyplus_launcher import energyplus_command, runtime_provenance, run_with_scratch

SIM = ROOT / "simulations"
STATE_VARIABLES = [
    "Zone Air Temperature",
    "Zone Air Humidity Ratio",
    "Surface Inside Face Temperature",
    "Surface Outside Face Temperature",
]
ENERGY_METERS = {
    "facility_electric_kwh": "Electricity:Facility",
    "hvac_electric_kwh": "Electricity:HVAC",
    "heating_electric_kwh": "Heating:Electricity",
    "cooling_electric_kwh": "Cooling:Electricity",
    "fans_electric_kwh": "Fans:Electricity",
    "gas_hhv_kwh": "Heating:NaturalGas",
}


def reported_model(text, end_date=None):
    objects = parse_idf(text)
    for name in STATE_VARIABLES:
        objects.append(["Output:Variable", "*", name, "Hourly"])
    if end_date is not None:
        fields = idd_fields()["runperiod"]
        periods = [o for o in objects if o[0].lower() == "runperiod"]
        assert len(periods) == 1
        period = periods[0]
        period[fields.index("End Month") + 1] = str(end_date.month)
        period[fields.index("End Day of Month") + 1] = str(end_date.day)
    return serialize_idf(objects)


def sql_matrix(connection, names):
    marks = ",".join("?" for _ in names)
    data = pd.read_sql_query(
        f"""select t.TimeIndex,d.KeyValue,d.Name,r.Value
    from ReportData r join ReportDataDictionary d on d.ReportDataDictionaryIndex=r.ReportDataDictionaryIndex
    join Time t on t.TimeIndex=r.TimeIndex
    join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex
    where e.EnvironmentType=3 and d.ReportingFrequency='Hourly' and d.Name in ({marks})""",
        connection,
        params=names,
    )
    if data.empty:
        raise ValueError("Requested hourly SQL fields are missing: " + str(names))
    data["key"] = data.KeyValue.fillna("") + "|" + data.Name
    return data.pivot(index="TimeIndex", columns="key", values="Value").sort_index()


def extract(sql, baseline):
    with sqlite3.connect(sql) as c:
        clock = pd.read_sql_query(
            """select t.* from Time t join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex
          where e.EnvironmentType=3 order by t.TimeIndex""",
            c,
        )
        if not clock.Interval.eq(60).all() or clock.TimeIndex.duplicated().any():
            raise ValueError("Incomplete or duplicated hourly clock")
        n = len(clock)
        assert 0 < n <= 8760
        for key, col in [("month", "Month"), ("day", "Day"), ("hour", "Hour"), ("dst", "Dst")]:
            assert np.array_equal(
                baseline[key][:n], clock[col].to_numpy()
            ), "Prefix calendar differs from baseline"
        core = sql_matrix(
            c, ["Zone Operative Temperature", "Zone People Occupant Count", *ENERGY_METERS.values()]
        )
        arrays = {k: baseline[k][:n].copy() for k in ["month", "day", "hour", "dst", "occupied"]}
        arrays["zone_ids"] = np.array(ZONES)
        for key, name in [
            ("temperatures", "Zone Operative Temperature"),
            ("people", "Zone People Occupant Count"),
        ]:
            arrays[key] = np.column_stack([core[z.upper() + "|" + name].to_numpy() for z in ZONES])
        for key, meter in ENERGY_METERS.items():
            col = "|" + meter
            if col not in core:
                raise ValueError("Required meter missing from new VAV response run: " + meter)
            arrays[key] = core[col].to_numpy() / 3.6e6
        state = sql_matrix(c, STATE_VARIABLES)
    c.close()
    assert (
        len(core) == n
        and len(state) == n
        and np.isfinite(core.to_numpy()).all()
        and np.isfinite(state.to_numpy()).all()
    )
    assert all(z.upper() + "|Zone Air Temperature" in state for z in ZONES)
    assert all(z.upper() + "|Zone Air Humidity Ratio" in state for z in ZONES)
    assert len([x for x in state if x.endswith("|Surface Inside Face Temperature")]) >= 15
    return arrays, {"keys": state.columns.to_numpy(str), "values": state.to_numpy(float)}


def lossless_sql_archive(path):
    path = Path(path)
    archive = path.with_suffix(".sql.zst")
    expected = digest(path)
    with path.open("rb") as source, archive.open("wb") as destination:
        zstd.ZstdCompressor(level=6).copy_stream(source, destination)
    check = hashlib.sha256()
    with archive.open("rb") as source, zstd.ZstdDecompressor().stream_reader(source) as reader:
        while block := reader.read(1024 * 1024):
            check.update(block)
    assert check.hexdigest() == expected
    cleanup_deferred = False
    for attempt in range(8):
        try:
            path.unlink()
            break
        except PermissionError:
            if attempt == 7:
                cleanup_deferred = True
            else:
                time.sleep(0.25)
    return {
        "uncompressed_sql_sha256": expected,
        "archive": str(archive),
        "archive_sha256": digest(archive),
        "uncompressed_sql_cleanup_deferred": cleanup_deferred,
    }


def execute(idf, weather, baseline_arrays, label, baseline_qa=None):
    folder = SIM / "single_event_runs" / label
    compact = SIM / "single_event_hourly" / (label + ".npz")
    states = SIM / "single_event_states" / (label + ".npz")
    qa_path = folder / "RUN_QA.json"
    if qa_path.exists() and compact.exists() and states.exists():
        qa = json.loads(qa_path.read_text())
        assert qa["input_idf_sha256"] == digest(idf)
        assert qa["weather_sha256"] == digest(weather)
        assert qa["runtime_sha256"] == digest(RUNTIME)
        if qa.get("runtime_api_sha256") is not None:
            assert qa["runtime_api_sha256"] == runtime_provenance(RUNTIME)["runtime_api_sha256"]
            assert qa.get("runtime_package_manifest_sha256") == runtime_provenance(RUNTIME)["runtime_package_manifest_sha256"]
        current_paths = {"compact_output": compact, "state_output": states,
                         "sql_metadata_path": folder / "simulation_metadata.sql"}
        for key, checksum in [("compact_output", "hourly_sha256"),
                              ("state_output", "states_sha256"),
                              ("sql_metadata_path", "sql_metadata_sha256")]:
            retained = current_paths[key].resolve()
            if not retained.is_relative_to(SIM.resolve()):
                raise ValueError("Cached response output points outside this project run; use a fresh output directory")
            if not retained.is_file() or digest(retained) != qa[checksum]:
                raise ValueError("Cached response output is missing or changed: " + key)
            qa[key] = str(retained)
        qa["run_folder"] = str(folder)
        if qa.get("execution_audit_path"):
            qa["execution_audit_path"] = str(folder / Path(qa["execution_audit_path"]).name)
        return qa
    if folder.exists() and any(folder.iterdir()) and not (folder / "eplusout.err").exists():
        raise FileExistsError("Startup or interrupted attempt is preserved; use a distinct attempt label")
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    command = energyplus_command(RUNTIME, ["-w", str(weather), "-d", str(folder), "-r", str(idf)])
    invocation = {
        "command": command,
        "input_sha256": digest(idf),
        "weather_sha256": digest(weather),
        "runtime_sha256": digest(RUNTIME),
        **runtime_provenance(RUNTIME),
    }
    recovered = False
    if (folder / "eplusout.err").exists():
        if not (folder / "INVOCATION.json").exists():
            raise FileExistsError(
                "Prior attempt has no input-hash record; preserve it and use a new label"
            )
        old = json.loads((folder / "INVOCATION.json").read_text())
        assert all(
            old[k] == invocation[k] for k in ["input_sha256", "weather_sha256", "runtime_sha256"]
        )
        for key, value in runtime_provenance(RUNTIME).items():
            if key not in old or old[key] != value:
                raise ValueError("The retained SQL runtime fingerprint is incomplete or different; preserve it and use a distinct attempt label")
        if (
            "EnergyPlus Completed Successfully--"
            not in (folder / "eplusout.err").read_text(errors="replace")
            or not (folder / "eplusout.sql").exists()
        ):
            raise FileExistsError("Incomplete simulation preserved; use a distinct attempt label")
        return_code = 0
        recovered = True
    else:
        write_json(folder / "INVOCATION.json", invocation)
        with (folder / "CONSOLE.log").open("wb") as log:
            result = run_with_scratch(RUNTIME, ["-w", str(weather), "-d", str(folder), "-r", str(idf)],
                folder, stdout=log, timeout_seconds=900)
        return_code = result.returncode
    text = (
        (folder / "eplusout.err").read_text(errors="replace")
        if (folder / "eplusout.err").exists()
        else ""
    )
    counts = re.findall(
        r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;", text
    )
    qa = {
        "completed": return_code == 0 and bool(counts),
        "exit_code": return_code,
        "seconds": None if recovered else time.monotonic() - start,
        "recovered_completed_simulation_after_postprocessing_error": recovered,
        "input_idf_sha256": digest(idf),
        "weather_sha256": digest(weather),
        "runtime_sha256": digest(RUNTIME),
        **runtime_provenance(RUNTIME),
        "execution_status": "recovered_completed_physical_run" if recovered else result.execution_status,
        "execution_audit_path": None if recovered else result.execution_audit_path,
        "reported_warning_count": int(counts[-1][0]) if counts else None,
        "reported_severe_count": int(counts[-1][1]) if counts else None,
        "fatal_occurrences": len(re.findall(r"\*\*\s+Fatal\s+\*\*", text)),
        "compact_output": str(compact),
        "state_output": str(states),
        "run_folder": str(folder),
        "initialization": "Original January1 warmup and uninterrupted normal-operation history; no transplanted air or material states.",
    }
    if qa["completed"]:
        arrays, state = extract(folder / "eplusout.sql", baseline_arrays)
        compact.parent.mkdir(parents=True, exist_ok=True)
        states.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(compact, **arrays)
        if baseline_qa is None:
            np.savez_compressed(states, **state)
        else:
            save_states_against(states, state, baseline_qa["state_output"])
        qa.update(
            {
                "hours": len(arrays["month"]),
                "hourly_sha256": digest(compact),
                "states_sha256": digest(states),
                "state_variable_count": len(state["keys"]),
                "warning_profile": warning_profile(folder / "eplusout.err"),
            }
        )
        qa.update(retain_sql_metadata(folder / "eplusout.sql"))
        if baseline_qa is not None:
            qa["normal_state_reference_sha256"] = baseline_qa["states_sha256"]
        for p in folder.glob("*"):
            if p.is_file() and p.suffix.lower() not in [".json", ".err", ".eio", ".log", ".zst", ".sql"]:
                p.unlink()
    write_json(qa_path, qa)
    return qa


def history_check(baseline, action, baseline_states, action_states, pre_action_hours):
    assert np.array_equal(baseline_states["keys"], action_states["keys"])
    n = pre_action_hours
    assert n > 0
    delta = np.abs(baseline_states["values"][:n] - action_states["values"][:n])
    humidity = np.array(
        [str(k).endswith("|Zone Air Humidity Ratio") for k in baseline_states["keys"]]
    )
    td = float(delta[:, ~humidity].max())
    hd = float(delta[:, humidity].max())
    ed = max(float(abs(baseline[k][:n] - action[k][:n]).max()) for k in ENERGY_METERS)
    pdiff = float(abs(baseline["people"][:n] - action["people"][:n]).max())
    return {
        "pre_action_temperature_state_delta_c": td,
        "pre_action_humidity_ratio_delta": hd,
        "pre_action_energy_delta_kwh": ed,
        "pre_action_occupancy_delta": pdiff,
        "history_matches": bool(td <= 1e-6 and hd <= 1e-9 and ed <= 1e-6 and pdiff <= 1e-10),
        "verified_states": "zone air temperature and humidity,inside/outside surface temperatures,occupancy and energy,over the entire pre-action history",
    }
