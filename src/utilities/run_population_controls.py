"""Execute the bounded, hash-locked external-population annual control plan.

Each new IDF runs in an independent EnergyPlus API process. Source physics,
native service/ventilation windows and the existing engineering thresholds are
preserved. Exact retained outputs are identified separately from new runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from audit_ventilation_adaptation import sha, save, series, meter, warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis/energy_service"))
from generate_model_inputs import signature, physical_hash  # noqa: E402


def sizing(folder):
    prefixes = ("Zone Sizing Information,", "System Sizing Information,",
                "Plant Sizing Information,", "Component Sizing Information,")
    return [s.strip() for s in (folder / "eplusout.eio").read_text().splitlines()
            if s.strip().startswith(prefixes)]


def source_clock(folder, small):
    import sqlite3
    with sqlite3.connect(folder / "eplusout.sql") as connection:
        clock = pd.read_sql_query("select * from Time order by TimeIndex", connection)
    if len(clock) != 8760 or not clock.Interval.eq(60).all():
        raise ValueError("Expected one complete hourly annual EnergyPlus clock")
    minute = (clock.Hour - 1) * 60 + 30 + clock.Dst * 60
    weekday = clock.DayType.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    on = weekday & minute.ge(360) & minute.lt(1140 if small else 1320)
    if not small:
        on |= clock.DayType.eq("Saturday") & minute.ge(360) & minute.lt(1080)
    return clock, on.to_numpy(), minute.to_numpy()


def export_and_gate(folder, context, zones, baseline, hourly_path):
    small = len(zones) == 5
    frame = pd.read_csv(folder / "eplusout.csv")
    original = pd.read_csv(baseline / "eplusout.csv")
    clock, occupied, minute = source_clock(folder, small)
    error = (folder / "eplusout.err").read_text()
    final = re.findall(r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;", error)
    if len(frame) != 8760 or not final:
        raise ValueError("Incomplete annual trajectory")
    arrays = {"zone_ids": np.asarray(zones), "occupied": occupied,
              "month": clock.Month.to_numpy(), "day": clock.Day.to_numpy(),
              "hour": clock.Hour.to_numpy(), "dst": clock.Dst.to_numpy()}
    for key, fragment in [("temperatures", "Zone Operative Temperature"),
                          ("people", "Zone People Occupant Count"),
                          ("heating_setpoint_c", "Zone Thermostat Heating Setpoint Temperature"),
                          ("cooling_setpoint_c", "Zone Thermostat Cooling Setpoint Temperature"),
                          ("ventilation_requested_m3_s", "Zone Target Voz Ventilation Flow Rate")]:
        arrays[key] = np.column_stack([series(frame, z.upper(), fragment) for z in zones])
    ratios = []
    if small:
        arrays["ventilation_delivered_m3_s"] = np.column_stack([
            series(frame, z.upper(), "Zone Mechanical Ventilation Standard Density Volume Flow Rate") for z in zones])
        for index in range(len(zones)):
            requested = arrays["ventilation_requested_m3_s"][:, index]
            selected = occupied & (minute >= 420) & (arrays["people"][:, index] > 1e-8) & (requested > 1e-10)
            ratios.extend((arrays["ventilation_delivered_m3_s"][selected, index] / requested[selected]).tolist())
    else:
        ahus = ["PACU_VAV_BOT", "PACU_VAV_MID", "PACU_VAV_TOP"]
        arrays["ahu_ids"] = np.asarray(ahus)
        for key, fragment in [("outdoor_delivered_kg_s", "Air System Outdoor Air Mass Flow Rate"),
                              ("outdoor_requested_kg_s", "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate")]:
            arrays[key] = np.column_stack([series(frame, z, fragment) for z in ahus])
        for index in range(3):
            requested = arrays["outdoor_requested_kg_s"][:, index]
            selected = occupied & (requested > 1e-10)
            ratios.extend((arrays["outdoor_delivered_kg_s"][selected, index] / requested[selected]).tolist())
    if not ratios:
        raise ValueError("No native scheduled ventilation support")
    for key, name in [("electricity_hvac_meter_kwh", "Electricity:HVAC"),
                      ("heating_electric_kwh", "Heating:Electricity"),
                      ("cooling_electric_kwh", "Cooling:Electricity"),
                      ("fans_electric_kwh", "Fans:Electricity"),
                      ("pumps_electric_kwh", "Pumps:Electricity"),
                      ("gas_hhv_kwh", "Heating:NaturalGas")]:
        arrays[key] = meter(frame, name)
    arrays["hvac_electric_kwh"] = sum(arrays[k] for k in ["heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh"])
    losses = [c for c in frame if "Transformer Distribution Electricity Loss Energy" in c]
    arrays["transformer_loss_kwh"] = frame[losses[0]].to_numpy() / 3.6e6 if losses else np.zeros(8760)
    closure = float(np.max(np.abs(arrays["electricity_hvac_meter_kwh"] - arrays["hvac_electric_kwh"] - arrays["transformer_loss_kwh"])))
    invariant_columns = [c for c in frame if "Occupant Count" in c or "Target Voz Ventilation Flow Rate" in c]
    load_delta = float(np.max(np.abs(frame[invariant_columns].to_numpy() - original[invariant_columns].to_numpy())))
    report = warnings(folder)
    if not small:
        # Match the frozen medium gate. The stricter coil-range filter belongs
        # to the source small-office gate, and is not silently generalized.
        report["direct_hvac_numerical_messages"] = [m for m in report["direct_hvac_numerical_messages"] if "CalcDoe2DXCoil:" not in m]
    quality = {"hours": len(frame), "reported_warning_count": int(final[-1][0]),
               "severe": int(final[-1][1]), "fatal": len(re.findall(r"\*\*\s+Fatal\s+\*\*", error)),
               "direct_numerical_warnings": report["direct_hvac_numerical_messages"],
               "all_recurring_warnings": report["recurring_groups"],
               "unexplained_hvac_recurring_groups": len(report["unexplained_recurring_groups"]),
               "hourly_meter_closure_kwh": closure,
               "same_source_sizing": sizing(folder) == sizing(baseline),
               "source_occupancy_ventilation_requirement_max_delta": load_delta,
               "min_native_scheduled_ventilation_delivery_request": min(ratios),
               "ventilation_assessed_zone_or_ahu_hours": len(ratios),
               "controlled_hvac_electric_kwh": float(arrays["hvac_electric_kwh"].sum()),
               "space_heating_gas_kwh_fuel": float(arrays["gas_hhv_kwh"].sum()),
               "source_preheat_06_07_oa_damper_closed": small,
               "physical_parameters_changed": False, "empirical_accuracy_evaluated": False}
    quality["engineering_pass"] = bool(quality["severe"] == quality["fatal"] == 0
        and not quality["direct_numerical_warnings"] and not quality["unexplained_hvac_recurring_groups"]
        and closure < 1e-8 and quality["same_source_sizing"] and load_delta < 1e-10
        and quality["min_native_scheduled_ventilation_delivery_request"] >= .99)
    for values in arrays.values():
        if values.dtype.kind not in "US" and not np.isfinite(values).all():
            raise ValueError("Nonfinite retained hourly output")
    arrays["floor_area_m2"] = np.asarray(float(context["floor_area_m2"]))
    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(hourly_path, **arrays)
    with np.load(hourly_path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or not all(np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise ValueError("Hourly serialization changed output arrays")
    save(folder / "ENGINEERING_QA.json", quality)
    return quality


def retained_result(job, repository, models, source):
    """Reuse an exact recorded decision, including rejected inputs without scores."""
    def location(value):
        path = Path(value)
        return path if path.is_absolute() else repository / path

    rid = job["run_id"]
    cached_folder = models / "current_run_gates" / job["cached_run_id"]
    qa_path = (location(job["cached_qa_path"]) if job.get("cached_qa_path") else
               cached_folder / "ENGINEERING_QA.json")
    if job.get("cached_qa_sha256") and sha(qa_path) != job["cached_qa_sha256"]:
        raise ValueError("Cached engineering QA checksum changed")
    quality = json.loads(qa_path.read_text())
    if job.get("cached_engineering_status"):
        qualified = job["cached_engineering_status"] in {"cached_qualified", "new_qualified"}
        if bool(quality["engineering_pass"]) != qualified:
            raise ValueError("Cached original engineering decision changed")
    original_idf = (location(job["cached_idf_path"]) if job.get("cached_idf_path") else
                    cached_folder / "input.idf")
    if original_idf.is_file():
        if signature(original_idf.read_text()) != job["normalized_idf_sha256"]:
            raise ValueError("Cached model is not identical to the generated model")
    elif job.get("cached_normalized_idf_sha256") != job["normalized_idf_sha256"]:
        raise ValueError("Cached input lacks the matching retained normalized identity")
    hourly = location(job["cached_hourly_path"]) if job.get("cached_hourly_path") else None
    if quality["engineering_pass"] and hourly is None:
        raise ValueError("A qualified cached decision needs its actual hourly trajectory")
    if hourly is not None:
        if job.get("cached_hourly_sha256") and sha(hourly) != job["cached_hourly_sha256"]:
            raise ValueError("Cached hourly checksum changed")
        with np.load(hourly, allow_pickle=False) as saved:
            if saved["temperatures"].shape[0] != 8760:
                raise ValueError("Cached trajectory is incomplete")
    return {"run_id": rid, "context": job["context"],
            "engineering_status": "cached_qualified" if quality["engineering_pass"] else "cached_rejected",
            "cache_reused": True, "cached_run_id": job["cached_run_id"],
            "path_hourly": str(hourly) if hourly is not None else "",
            "hourly_sha256": sha(hourly) if hourly is not None else "",
            "qa_path": str(qa_path), "qa_sha256": sha(qa_path), "physical_input_sha256": sha(source)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-designs", type=int, default=144)
    parser.add_argument("--kind", choices=("annual", "event"), default="annual")
    args = parser.parse_args()
    repository, plan, runtime, output = [p.resolve() for p in
                                        (args.repository_root, args.plan_root, args.runtime_root, args.output_root)]
    models = repository / "data/upstream/frozen_buildings"
    contexts = json.loads((repository / "data/inputs/energy_service/controller/CONTEXTS.json").read_text())
    frozen = json.loads((models / "MODEL_CONTEXTS.json").read_text())
    if sha(runtime / "energyplus.exe") != "8e56b46647e109944d8762df6f35eff2fa149c94dc43a78cbf17c69f56e3ee27":
        raise ValueError("Runtime differs from frozen EnergyPlus22.1")
    logical_name = "ANNUAL_RUN_PLAN.csv" if args.kind == "annual" else "EVENT_RUN_PLAN.csv"
    physical_name = "PHYSICAL_RUN_PLAN.csv" if args.kind == "annual" else "PHYSICAL_EVENT_PLAN.csv"
    manifest_name = "ANNUAL_RUN_MANIFEST.csv" if args.kind == "annual" else "EVENT_RUN_MANIFEST.csv"
    unique = pd.read_csv(plan / physical_name, keep_default_na=False)
    logical = pd.read_csv(plan / logical_name, keep_default_na=False)
    if len(logical) != args.expected_designs or unique.run_id.duplicated().any():
        raise ValueError(f"The declared plan must contain {args.expected_designs} logical {args.kind} inputs and unique physical IDs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "engineering").mkdir(exist_ok=True)
    for name in ("PROFILE_ASSIGNMENTS.csv.gz", "BASELINE_DECISION_CONTRACT.csv", "FIXED_TAIL_MEMBERSHIPS.csv"):
        if (plan / name).is_file():
            shutil.copyfile(plan / name, output / "engineering" / name)
    before = {f"{city}/{name}": sha(models / city / name) for city in frozen for name in ["baseline.idf", "weather.epw"]}
    protocol = {f"logical_{args.kind}_inputs": len(logical), "unique_physical_inputs": len(unique),
                "workers": args.workers, "runtime_sha256": sha(runtime / "energyplus.exe"),
                "physical_plan_sha256": sha(plan / physical_name),
                "logical_plan_sha256": sha(plan / logical_name),
                "script_sha256": sha(__file__), "frozen_input_sha256_before_runs": before,
                "scope": "Three population draws on unchanged building-weather contexts; exact declared layouts, permissions, controls and variants are retained in the hash-locked logical plan. Cached decisions are separate from new physical execution."}
    save(output / "EXECUTION_PROTOCOL_BEFORE_RUNS.json", protocol)
    results_path = output / "RUN_RESULTS.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    def execute(job):
        rid, city = job["run_id"], job["context"]
        source = plan / job["idf_path"]
        weather = models / job["weather"]
        cfg = frozen[city]
        text = source.read_text()
        if signature(text) != job["normalized_idf_sha256"] or physical_hash(text) != job["physical_sha256"]:
            raise ValueError("Generated IDF hash differs from the pre-run plan")
        if physical_hash(text) != physical_hash((models / cfg["baseline_idf"]).read_text()):
            raise ValueError("A generated input changed frozen physics")
        if sha(weather) != cfg["weather_sha256"] or sha(weather) != job["weather_sha256"]:
            raise ValueError("Frozen weather changed")
        if rid in results:
            old = results[rid]
            if old.get("path_hourly") and sha(old["path_hourly"]) != old["hourly_sha256"]:
                raise ValueError("Previously completed hourly output changed")
            return old
        if job["execution_status"] == "exact_retained_cache":
            return retained_result(job, repository, models, source)
        folder = output / "runs" / rid
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "input.idf").write_bytes(source.read_bytes())
        save(folder / "INPUTS.json", job | {"protocol_sha256": sha(output / "EXECUTION_PROTOCOL_BEFORE_RUNS.json")})
        started = time.monotonic()
        item = {"run_id": rid, "context": city, "engineering_status": "execution_failed", "cache_reused": False,
                "physical_input_sha256": sha(source), "source_folder": str(folder)}
        try:
            command = [sys.executable, str(repository / "src/utilities/audit_ventilation_adaptation.py"),
                       "--api-child", "--runtime-root", str(runtime), "--weather", str(weather), "--folder", str(folder)]
            with (folder / "CONSOLE.log").open("wb") as log:
                completed = subprocess.run(command, cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
            error = (folder / "eplusout.err").read_text(errors="replace") if (folder / "eplusout.err").exists() else ""
            run = {"completed": completed.returncode == 0 and "EnergyPlus Completed Successfully" in error,
                   "exit_code": completed.returncode, "runtime_sha256": protocol["runtime_sha256"]}
            save(folder / "RUN_RESULT.json", run)
            if run["completed"]:
                hourly = output / "hourly" / f"{rid}.npz"
                quality = export_and_gate(folder, contexts[city], contexts[city]["zones"],
                                          models / "baseline_evidence" / city, hourly)
                item.update(engineering_status="new_qualified" if quality["engineering_pass"] else "new_rejected",
                            path_hourly=str(hourly), hourly_sha256=sha(hourly),
                            qa_path=str(folder / "ENGINEERING_QA.json"), qa_sha256=sha(folder / "ENGINEERING_QA.json"))
        except Exception:
            item["exception"] = traceback.format_exc()
            save(folder / "EXECUTION_EXCEPTION.json", item)
        item["elapsed_seconds"] = time.monotonic() - started
        save(folder / "EXECUTION_RESULT.json", item)
        return item

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, job) for job in unique.to_dict("records")]
        for future in as_completed(futures):
            item = future.result()
            results[item["run_id"]] = item
            save(results_path, results)
            print(json.dumps({k: item.get(k) for k in ["run_id", "context", "engineering_status", "cache_reused", "elapsed_seconds"]}), flush=True)
    for key in ["engineering_status", "path_hourly", "hourly_sha256", "qa_path", "qa_sha256"]:
        logical[key] = logical.run_id.map(lambda rid: results[rid].get(key, ""))
    if "permission_alias" not in logical:
        logical["permission_alias"] = ""
    logical["engineering_pass"] = logical.engineering_status.isin(["cached_qualified", "new_qualified"])
    logical.to_csv(output / "engineering" / manifest_name, index=False)
    after = {key: sha(models / key) for key in before}
    if after != before:
        raise ValueError("A frozen input changed during execution")
    status = pd.DataFrame(results.values()).groupby(["context", "engineering_status"]).size().to_dict()
    summary = {f"logical_{args.kind}_inputs": len(logical), "unique_inputs": len(unique),
               "new_physical_runs": sum(not r["cache_reused"] for r in results.values()),
               "exact_cache_reuses": sum(r["cache_reused"] for r in results.values()),
               "unique_status_by_context": [{"context": key[0], "status": key[1], "count": int(n)} for key, n in status.items()],
               "logical_status_counts": logical.engineering_status.value_counts().to_dict(),
               "frozen_inputs_unchanged": True, "engineering_thresholds_changed": False}
    save(output / "EXECUTION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
