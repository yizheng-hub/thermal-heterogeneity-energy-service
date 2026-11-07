"""Retain hash-bound SQL from exact-IDF technical replays, without changing policy."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
from global_matrix import run_energyplus  # noqa: E402 - initialize local imports before loading project modules
from run_global_matrix import ZONES, warning_profile  # noqa: E402 - initialize local imports before loading project modules
from thermal_heterogeneity.energyplus_launcher import runtime_provenance, verify_runtime  # noqa: E402 - initialize local imports before loading project modules

FIELDS = {"temperatures": "Zone Operative Temperature", "people": "Zone People Occupant Count",
          "facility_electric_kwh": "Electricity:Facility", "hvac_electric_kwh": "Electricity:HVAC",
          "heating_electric_kwh": "Heating:Electricity", "cooling_electric_kwh": "Cooling:Electricity",
          "fans_electric_kwh": "Fans:Electricity", "gas_hhv_kwh": "Heating:NaturalGas"}
CLOCK = {"month": "Month", "day": "Day", "hour": "Hour", "dst": "Dst"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Diagnostic replay paths must remain inside this project")
    return path


def asset(path):
    path = inside(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def checked(spec):
    path = inside(spec["path"])
    if sha(path) != spec["sha256"]:
        raise ValueError("Frozen diagnostic input changed: " + str(path))
    return path


def write(path, value):
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def compare(reference, candidate):
    if len(reference["month"]) != 8760 or len(candidate["month"]) != 8760:
        raise ValueError("A complete 8760-hour trajectory is required")
    for key in [*CLOCK, "zone_ids", "occupied"]:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError("Clock, occupancy or zone identity differs: " + key)
    delta = {}
    for key in FIELDS:
        a, b = reference[key], candidate[key]
        if a.shape != b.shape or not np.isfinite(b).all():
            raise ValueError("Incomplete or nonfinite trajectory: " + key)
        delta[key] = float(np.max(np.abs(a-b)))
    if max(delta.values()) > 1e-8:
        raise ValueError("Exact-input trajectory differs beyond the frozen 1e-8 tolerance")
    return delta


def sql_arrays(path, zones):
    connection = sqlite3.connect(":memory:")
    connection.deserialize(path.read_bytes())
    try:
        if connection.execute("pragma quick_check").fetchone()[0] != "ok":
            raise ValueError("Retained SQL integrity check failed")
        clock = pd.read_sql_query("select t.* from Time t join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 order by t.TimeIndex", connection)
        if len(clock) != 8760 or not clock.Interval.eq(60).all():
            raise ValueError("Retained SQL clock is incomplete")
        marks = ",".join("?" for _ in FIELDS)
        dictionary = pd.read_sql_query(f"select ReportDataDictionaryIndex,KeyValue,Name from ReportDataDictionary where ReportingFrequency='Hourly' and Name in ({marks})", connection, params=list(FIELDS.values()))
        identifiers = dictionary.ReportDataDictionaryIndex.tolist()
        if not identifiers:
            raise ValueError("Retained SQL has no required hourly dictionary")
        marks = ",".join("?" for _ in identifiers)
        values = pd.read_sql_query(f"select r.TimeIndex,r.ReportDataDictionaryIndex,r.Value from ReportData r join Time t on t.TimeIndex=r.TimeIndex join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 and r.ReportDataDictionaryIndex in ({marks})", connection, params=identifiers)
        matrix = values.pivot(index="TimeIndex", columns="ReportDataDictionaryIndex", values="Value").reindex(clock.TimeIndex)
        data = {key: clock[column].to_numpy() for key, column in CLOCK.items()}
        data["zone_ids"] = np.asarray(zones)
        minute = (clock.Hour-1)*60+30+clock.Dst*60
        data["occupied"] = ((clock.DayType.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]) & minute.ge(360) & minute.lt(1320)) | (clock.DayType.eq("Saturday") & minute.ge(360) & minute.lt(1080))).to_numpy()
        for key, name in FIELDS.items():
            entries = dictionary[dictionary.Name.eq(name)]
            if key in {"temperatures", "people"}:
                columns = []
                for zone in zones:
                    selected = entries[entries.KeyValue.fillna("").str.upper().eq(str(zone).upper())]
                    if len(selected) != 1:
                        raise ValueError("SQL zone dictionary is missing or ambiguous")
                    columns.append(int(selected.iloc[0].ReportDataDictionaryIndex))
                data[key] = matrix[columns].to_numpy()
            else:
                if len(entries) != 1:
                    raise ValueError("SQL energy dictionary is missing or ambiguous")
                data[key] = matrix[int(entries.iloc[0].ReportDataDictionaryIndex)].to_numpy()/3.6e6
        return data
    finally:
        connection.close()


def sizing(path):
    prefixes = ("Zone Sizing Information,", "System Sizing Information,",
                "Plant Sizing Information,", "Component Sizing Information,")
    return [line.strip() for line in path.read_text(errors="replace").splitlines() if line.strip().startswith(prefixes)]


def registration(case, origin, runtime, sql, qa, validation):
    identity = case["identity"]
    source_id = origin + "_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"source_id": source_id, "origin": origin, "identity": identity,
            "input_idf": case["input_idf"], "weather": case["weather"],
            "runtime": runtime, "original_hourly": case["original_hourly"],
            "sql": asset(sql), "qa": asset(qa), "validation": asset(validation)}


def validate(case, sql, compact, qa_path, validation_path, reference_eio=None, reference_err=None):
    reference = load(checked(case["original_hourly"]))
    qa = json.loads(qa_path.read_text())
    if not qa.get("completed") or qa.get("reported_severe_count") != 0 or qa.get("fatal_occurrences") != 0:
        raise ValueError("Physical execution is incomplete or has severe/fatal errors")
    expected = {"input_idf_sha256": case["input_idf"]["sha256"],
                "weather_sha256": case["weather"]["sha256"], "runtime_sha256": case["runtime_sha256"]}
    if any(qa.get(key) != value for key, value in expected.items()):
        raise ValueError("Retained QA does not match the frozen scientific input/runtime")
    compact_delta = compare(reference, load(compact))
    sql_delta = compare(reference, sql_arrays(sql, reference["zone_ids"]))
    report = {"status": "passed", "scope": "Technical exact-input output acceptance; original engineering qualification is not reclassified",
              "hours": 8760, "compact_max_absolute_differences": compact_delta,
              "sql_max_absolute_differences": sql_delta, "sql_sha256": sha(sql),
              "input_idf_sha256": expected["input_idf_sha256"], "weather_sha256": expected["weather_sha256"],
              "runtime_sha256": expected["runtime_sha256"], "original_hourly_sha256": case["original_hourly"]["sha256"],
              "raw_engineering_pass": qa.get("engineering_pass"), "qa_sha256": sha(qa_path)}
    if reference_eio is not None:
        before, after = sizing(reference_eio), sizing(sql.parent / "eplusout.eio")
        if not before or before != after:
            raise ValueError("Exact-input replay changed the sizing records")
        original, replayed = warning_profile(reference_err), warning_profile(sql.parent / "eplusout.err")
        if {k: v > 0 for k, v in original.items()} != {k: v > 0 for k, v in replayed.items()}:
            raise ValueError("Exact-input replay changed the warning categories")
        report.update(sizing_lines=len(before), sizing_unchanged=True,
                      original_warning_profile=original, replay_warning_profile=replayed,
                      warning_categories_unchanged=True)
    write(validation_path, report)
    return report


def execute(case, batch, runtime):
    folder = batch / case["case_id"]
    inputs = folder / "inputs"
    inputs.mkdir(parents=True)
    idf = inputs / "in.idf"
    shutil.copyfile(checked(case["input_idf"]), idf)
    if sha(idf) != case["input_idf"]["sha256"] or idf.with_suffix(".rvi").exists() or idf.with_suffix(".mvi").exists():
        raise ValueError("A new exact-IDF directory without adjacent controls is required")
    compact, run = folder / "hourly.npz", folder / "run"
    try:
        qualification = json.loads(checked(case["original_qualification"]).read_text())
        if not qualification.get("relative_engineering_pass") or not qualification.get("capacity_sizing_identical_to_reference"):
            raise ValueError("Original operation is not qualified by its fixed engineering protocol")
        qa = run_energyplus(idf, checked(case["weather"]), inside(runtime["executable_path"]), run,
                            compact, ZONES, cleanup_raw=False, timeout_seconds=900)
        if qa.get("runtime_api_sha256") != runtime["api_sha256"] or qa.get("runtime_package_manifest_sha256") != runtime["package_manifest_sha256"]:
            raise ValueError("Replay API runtime differs from the frozen protocol")
        sql = run / "eplusout.sql"
        validation = folder / "TECHNICAL_ACCEPTANCE.json"
        report = validate(case, sql, compact, run / "RUN_QA.json", validation,
                          checked(case["original_eio"]), checked(case["original_err"]))
        engine = json.loads(Path(qa["execution_audit_path"]).read_text())
        if engine.get("status") != "verified_outputs_committed" or engine["retained_sha256"]["eplusout.sql"] != sha(sql):
            raise ValueError("Full SQL is not bound to the verified owned-scratch execution")
        source = registration(case, "diagnostic_exact_idf_replay", runtime, sql, run / "RUN_QA.json", validation)
        outcome = {"case_id": case["case_id"], "status": "passed", "validation": asset(validation),
                   "seconds": qa["seconds"], "max_absolute_difference": max(report["sql_max_absolute_differences"].values())}
        write(folder / "ATTEMPT.json", outcome)
        return outcome, source
    except Exception as error:
        outcome = {"case_id": case["case_id"], "status": "not_accepted", "exception": repr(error),
                   "preservation": "All inputs, outputs, logs and original engineering evidence remain; no registered source is claimed"}
        write(folder / "ATTEMPT.json", outcome)
        return outcome, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not Path(sys.executable).resolve().is_relative_to(ROOT):
        raise ValueError("Use the contained project Python")
    config_path = inside(args.config)
    config = json.loads(config_path.read_text())
    if config["workers"] != 4 or config["timeout_seconds"] != 900 or len(config["replays"]) != 8:
        raise ValueError("The locked protocol requires eight cases, four workers and 900 seconds")
    out = inside(config["output_root"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the earlier diagnostic attempt and use a new protocol output root")
    out.mkdir(parents=True, exist_ok=True)
    runtime = config["runtime"]
    if sha(inside(runtime["executable_path"])) != runtime["executable_sha256"]:
        raise ValueError("Runtime executable changed")
    provenance = runtime_provenance(inside(runtime["executable_path"]))
    if runtime.get("api_sha256") != provenance["runtime_api_sha256"] or runtime.get("package_manifest_sha256") != provenance["runtime_package_manifest_sha256"]:
        raise ValueError("Runtime API or package fingerprint changed")
    protocol = out / "FROZEN_PROTOCOL.json"
    write(protocol, {"status": "frozen_before_results", "configuration": asset(config_path), "plan": config,
          "created_utc": datetime.now(timezone.utc).isoformat(), "utility": asset(Path(__file__)),
          "launcher": asset(ROOT / "src/thermal_heterogeneity/energyplus_launcher.py"),
          "global_matrix": asset(ROOT / "src/analysis/global_extension/global_matrix.py"),
          "python": asset(Path(sys.executable)), "acceptance": "8760 clock/occupancy/zone identity exact; T,people,six energy absolute deltas <=1e-8; unchanged sizing/warning categories; full SQL retained; no service/energy sign requirement"})
    verify_runtime(inside(runtime["executable_path"]), out / "API_PREFLIGHT.json")
    registry = {"schema_version": 1, "project_root": ".", "protocol_path": protocol.relative_to(ROOT).as_posix(),
                "protocol_sha256": sha(protocol), "status": "running", "sources": []}
    registry_path = out / "REGISTERED_SQL_SOURCES.json"
    for case in config["existing_sources"]:
        folder = out / ("existing_" + case["case_id"])
        folder.mkdir()
        sql, compact, qa = checked(case["sql"]), checked(case["cache_hourly"]), checked(case["qa"])
        for spec in [case["input_idf"], case["weather"]]:
            checked(spec)
        validation = folder / "TECHNICAL_ACCEPTANCE.json"
        validate(case, sql, compact, qa, validation)
        cached_qa = json.loads(qa.read_text())
        historical = {"executable_path": runtime["executable_path"], "executable_sha256": case["runtime_sha256"],
                      "api_sha256": cached_qa.get("runtime_api_sha256"),
                      "package_manifest_sha256": cached_qa.get("runtime_package_manifest_sha256"),
                      "runtime_backend": cached_qa.get("runtime_backend", "historical_backend_not_recorded")}
        registry["sources"].append(registration(case, "existing_exact_cache", historical, sql, qa, validation))
        write(registry_path, registry)
    batch = out / "replays"
    batch.mkdir()
    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(execute, case, batch, runtime): case for case in config["replays"]}
        for future in as_completed(pending):
            try:
                outcome, source = future.result()
            except Exception as error:
                outcome = {"case_id": pending[future]["case_id"], "status": "not_accepted", "exception": repr(error)}
                source = None
            outcomes.append(outcome)
            if source is not None:
                registry["sources"].append(source)
            write(out / "REPLAY_RESULTS.json", outcomes)
            write(registry_path, registry)
            print(json.dumps(outcome), flush=True)
    registry["status"] = "completed" if all(r["status"] == "passed" for r in outcomes) else "partial"
    write(registry_path, registry)
    write(out / "SUMMARY.json", {"status": registry["status"], "planned_replays": 8,
          "accepted_replays": sum(r["status"] == "passed" for r in outcomes), "registered_sources": len(registry["sources"]),
          "registry": asset(registry_path), "scope": "Technical acceptance only; original engineering qualification retained"})


if __name__ == "__main__":
    main()
