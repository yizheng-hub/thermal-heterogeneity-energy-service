"""Reproduce four annual probes of the frozen medium-office ventilation adaptation.

The run action preserves all frozen physical inputs and restores only the three
original DOE max-OA EMS programs in one diagnostic variant per context. Analysis
can run without the EnergyPlus runtime when completed outputs are supplied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def objects(text):
    return [[v.strip() for v in part.split(",")]
            for part in re.sub(r"!.*", "", text).split(";") if part.strip()]


def save(path, item):
    path.write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")


def prepare(repository, runtime, output):
    root = repository / "data/upstream/frozen_buildings"
    settings = json.loads((root / "MODEL_CONTEXTS.json").read_text())
    archive = root / "provenance/original_sources/ASHRAE901_OfficeMedium_STD2019.zip"
    executable = runtime / "energyplus.exe"
    if sha(executable) != "8e56b46647e109944d8762df6f35eff2fa149c94dc43a78cbf17c69f56e3ee27":
        raise ValueError("EnergyPlus binary does not match the frozen runtime")
    if sha(archive) != "8431a33c3c27c0c34b1efc31be5f558b2b17e29a8850accc1bbce5f4f0dc5f7c":
        raise ValueError("Original DOE archive changed")
    jobs = []
    with zipfile.ZipFile(archive) as source:
        for context in ["Miami", "NewDelhi"]:
            cfg = settings[context]
            baseline = root / cfg["baseline_idf"]
            weather = root / cfg["weather"]
            if sha(baseline) != cfg["baseline_idf_sha256"] or sha(weather) != cfg["weather_sha256"]:
                raise ValueError("Frozen model/weather input changed")
            frozen = objects(baseline.read_text())
            original = objects(source.read(f"ASHRAE901_OfficeMedium_STD2019_{context}.idf").decode("utf-8-sig"))
            originals = {o[1]: o for o in original if o[0] == "EnergyManagementSystem:Program"
                         and o[1].upper().endswith("_MAX_OA_FRAC")}
            if len(originals) != 3:
                raise ValueError("Expected exactly three native OA EMS programs")
            diagnostics = []
            for level in ["bot", "mid", "top"]:
                for node in [f"PACU_VAV_{level}_OA-PACU_VAV_{level}_CoolCNode",
                             f"PACU_VAV_{level} Supply Equipment Outlet Node",
                             f"PACU_VAV_{level}_OAInlet Node"]:
                    for variable in ["System Node Standard Density Volume Flow Rate", "System Node Mass Flow Rate"]:
                        diagnostics.append(["Output:Variable", node, variable, "Hourly"])
                diagnostics.append(["Output:Variable", f"VAV_{level.upper()} Max OA Fraction", "Schedule Value", "Hourly"])
            for mode in ["current_frozen", "original_oa_ems_restored"]:
                folder = output / context / mode
                folder.mkdir(parents=True, exist_ok=False)
                items = [o.copy() for o in frozen]
                changes = []
                if mode == "original_oa_ems_restored":
                    for i, obj in enumerate(items):
                        if obj[0] == "EnergyManagementSystem:Program" and obj[1] in originals:
                            changes.append({"name": obj[1], "frozen": obj, "native_restored": originals[obj[1]]})
                            items[i] = originals[obj[1]].copy()
                    if len(changes) != 3:
                        raise ValueError("Restoration did not affect exactly three programs")
                text = "\n".join(",".join(o) + ";" for o in items + diagnostics) + "\n"
                path = folder / "input.idf"
                path.write_text(text, encoding="utf-8")
                input_qa = {"context": context, "mode": mode, "frozen_baseline_sha256": sha(baseline),
                            "input_sha256": sha(path), "weather_sha256": sha(weather),
                            "runtime_sha256": sha(executable), "idd_sha256": sha(runtime / "Energy+.idd"),
                            "only_physical_control_changes": changes, "additional_output_objects": diagnostics,
                            "equipment_parameters_changed": False, "thermal_control_changed": False}
                save(folder / "INPUT_QA.json", input_qa)
                jobs.append({"context": context, "mode": mode, "folder": folder,
                             "weather": weather, "executable": executable})
    save(output / "PROTOCOL_BEFORE_RUNS.json", {
        "bounded_run_count": 4, "contexts": ["Miami", "NewDelhi"],
        "modes": ["current_frozen", "original_oa_ems_restored"],
        "questions": ["Does restoring native OA EMS change outdoor-air delivery below its native VRP request?",
                      "Are hourly OA/supply bounds physically consistent in either mode?",
                      "What are the signed annual and winter carrier/operative-temperature differences?"],
        "selection": "Two frozen common25 baseline contexts, no outcome-based parameter search",
        "within_context_run_order": "current_frozen then original_oa_ems_restored",
        "source_archive_sha256": sha(archive), "runtime_sha256": sha(executable),
        "source_models_unchanged": True, "weather_unchanged": True,
        "diagnostic_resolution": "Hourly reported flows; subhourly extrema are not inferred from hourly averages",
        "source_script_sha256": sha(__file__),
    })
    return jobs


def execute_context(jobs):
    results = []
    for job in jobs:
        folder = job["folder"]
        start = time.monotonic()
        command = [sys.executable, str(Path(__file__).resolve()), "--api-child",
                   "--runtime-root", str(job["executable"].parent),
                   "--weather", str(job["weather"]), "--folder", str(folder)]
        with (folder / "CONSOLE.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=folder,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=600)
        error = (folder / "eplusout.err").read_text(errors="replace") if (folder / "eplusout.err").is_file() else ""
        summary = re.findall(r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;", error)
        record = {"context": job["context"], "mode": job["mode"], "seconds": time.monotonic() - start,
                  "exit_code": result.returncode, "completed": result.returncode == 0 and bool(summary),
                  "warning_count": int(summary[-1][0]) if summary else None,
                  "severe_count": int(summary[-1][1]) if summary else None,
                  "fatal_occurrences": len(re.findall(r"\*\*\s+Fatal\s+\*\*", error)),
                  "err_sha256": sha(folder / "eplusout.err") if error else None,
                  "csv_sha256": sha(folder / "eplusout.csv") if (folder / "eplusout.csv").is_file() else None}
        save(folder / "RUN_RESULT.json", record)
        print(json.dumps(record), flush=True)
        results.append(record)
    return results


def clock(folder):
    with sqlite3.connect(folder / "eplusout.sql") as connection:
        times = pd.read_sql_query("select * from Time order by TimeIndex", connection)
    assert len(times) == 8760 and times.Interval.eq(60).all()
    minute = (times.Hour - 1) * 60 + 30 + times.Dst * 60
    service = ((times.DayType.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                & minute.ge(360) & minute.lt(1320))
               | (times.DayType.eq("Saturday") & minute.ge(360) & minute.lt(1080)))
    return times, service.to_numpy()


def series(frame, prefix, fragment):
    choices = [c for c in frame if c.split(":", 1)[0] == prefix and fragment in c]
    assert len(choices) == 1, (prefix, fragment, choices)
    return frame[choices[0]].to_numpy()


def meter(frame, name):
    choices = [c for c in frame if c.split(" [", 1)[0] == name]
    assert len(choices) == 1, (name, choices)
    return frame[choices[0]].to_numpy() / 3.6e6


def warnings(folder):
    error = (folder / "eplusout.err").read_text()
    direct = [line.strip() for line in error.splitlines() if line.lstrip().startswith("** Warning **")]
    recurring = re.findall(r'\*\* Warning \*\* (.+?)\n\s*\*\*\*\*\*\*\*\*\*\*\*\*\*\s+\*\*\s+~~~\s+\*\*\s+This error occurred (\d+) total times;', error)
    groups = [{"message": message, "occurrences": int(n),
               "source_dhw": "water temperature" in message.lower()}
              for message, n in recurring]
    numeric_tokens = ["SimHVAC:", "PsyTwb", "PsyPsat", "Temperature out of range", "WetBulb not converged", "CalcDoe2DXCoil:"]
    return {"direct_messages": direct, "recurring_groups": groups,
            "direct_hvac_numerical_messages": [m for m in direct if any(t in m for t in numeric_tokens)],
            "unexplained_recurring_groups": [g for g in groups if not g["source_dhw"]]}


def analyse(args):
    root, runs = args.repository_root.resolve(), args.output_root.resolve()
    results, flow_rows, energy_rows, temperature_rows, hourly_energy_rows = {}, [], [], [], []
    for context in ["Miami", "NewDelhi"]:
        current_folder = runs / context / "current_frozen"
        times, service = clock(current_folder)
        archived = pd.read_csv(root / "data/upstream/frozen_buildings/baseline_evidence" / context / "eplusout.csv")
        frames, per_mode = {}, {}
        for mode in ["current_frozen", "original_oa_ems_restored"]:
            folder = runs / context / mode
            frame = pd.read_csv(folder / "eplusout.csv")
            assert len(frame) == 8760
            frames[mode] = frame
            active_hours = np.zeros(8760, dtype=bool)
            ahus = []
            for level in ["BOT", "MID", "TOP"]:
                name = f"PACU_VAV_{level}"
                oa = series(frame, name, "Air System Outdoor Air Mass Flow Rate")
                requested = series(frame, name, "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate")
                supply = series(frame, f"{name}_OA-{name}_COOLCNODE", "System Node Mass Flow Rate")
                oa_std = series(frame, f"{name}_OAINLET NODE", "System Node Standard Density Volume Flow Rate")
                supply_std = series(frame, f"{name}_OA-{name}_COOLCNODE", "System Node Standard Density Volume Flow Rate")
                outlet_mass = series(frame, f"{name} SUPPLY EQUIPMENT OUTLET NODE", "System Node Mass Flow Rate")
                ratio = np.divide(oa, requested, out=np.full(8760, np.nan), where=requested > 1e-10)
                fraction = np.divide(oa, supply, out=np.full(8760, np.nan), where=supply > 1e-10)
                fraction_std = np.divide(oa_std, supply_std, out=np.full(8760, np.nan), where=supply_std > 1e-10)
                mask = service & (requested > 1e-10)
                failed = mask & (ratio < .99)
                active_hours |= failed
                assert np.nanmax(np.abs(fraction - fraction_std)) < 1e-10
                ahus.append({"ahu": name, "assessed_service_hours": int(mask.sum()),
                             "delivered_requested_min": float(ratio[mask].min()),
                             "hours_below_99pct_native_vrp_request": int(failed.sum()),
                             "hourly_outdoor_supply_fraction_min": float(np.nanmin(fraction)),
                             "hourly_outdoor_supply_fraction_max": float(np.nanmax(fraction)),
                             "hours_full_fresh_air_99pct": int(np.sum(fraction >= .99)),
                             "hours_outdoor_above_supply": int(np.sum(oa > supply + 1e-10)),
                             "hours_outdoor_above_supply_outlet": int(np.sum(oa > outlet_mass + 1e-10)),
                             "mixed_supply_outlet_mass_difference_max_kg_s": float(np.max(np.abs(supply - outlet_mass))),
                             "outdoor_mass_below_zero": int(np.sum(oa < -1e-10)),
                             "service_oa_deficit_kg": float(np.maximum(requested - oa, 0)[mask].sum() * 3600),
                             "outside_service_hours_with_nonzero_request_no_supply": int(np.sum(~service & (requested > 0) & (supply == 0)))})
                flow_rows.append(pd.DataFrame({"context": context, "mode": mode, "ahu": name,
                                               "hour_index": np.arange(8760), "month": times.Month,
                                               "day": times.Day, "hour_endpoint": times.Hour,
                                               "dst": times.Dst, "source_service_mask": service,
                                               "oa_kg_s": oa, "native_vrp_requested_kg_s": requested,
                                               "mixed_supply_kg_s": supply, "supply_outlet_kg_s": outlet_mass,
                                               "oa_standard_m3_s": oa_std, "supply_standard_m3_s": supply_std,
                                               "delivery_request_ratio": ratio, "oa_supply_fraction": fraction}))
            per_mode[mode] = {"run": json.loads((folder / "RUN_RESULT.json").read_text()),
                              "input": json.loads((folder / "INPUT_QA.json").read_text()),
                              "ahu_hours_denominator": int(service.sum()) * 3,
                              "service_hours_with_any_ahu_below_99pct_request": int(active_hours.sum()),
                              "ahus": ahus, "warnings": warnings(folder)}
        current, native = frames["current_frozen"], frames["original_oa_ems_restored"]
        common = [c for c in current if c in archived and c != "Date/Time"]
        replay = {"shared_numeric_columns": len(common), "hours": len(current),
                  "all_shared_numeric_outputs_exactly_equal": all(np.array_equal(current[c], archived[c]) for c in common),
                  "maximum_absolute_shared_numeric_output_difference": float(max(np.max(np.abs(current[c] - archived[c])) for c in common))}
        energy = []
        for label, mask in [("annual", np.ones(8760, dtype=bool)), ("DJF", times.Month.isin([12, 1, 2]).to_numpy()),
                            ("January", times.Month.eq(1).to_numpy()), ("July", times.Month.eq(7).to_numpy())]:
            for name in ["controlled_hvac_electricity", "Heating:Electricity", "Cooling:Electricity", "Fans:Electricity", "Heating:NaturalGas", "Electricity:Facility", "Pumps:Electricity"]:
                if name == "controlled_hvac_electricity":
                    a, b = [sum(meter(f, n) for n in ["Heating:Electricity", "Cooling:Electricity", "Fans:Electricity"]) for f in [current, native]]
                else:
                    a, b = meter(current, name), meter(native, name)
                ca, cb = float(a[mask].sum()), float(b[mask].sum())
                row = {"context": context, "period": label, "hours": int(mask.sum()), "carrier_end_use": name,
                       "current_frozen_kwh": ca, "restored_native_oa_kwh": cb,
                       "current_minus_restored_native_kwh": ca - cb,
                       "current_relative_to_restored_native_pct": 100 * (ca / cb - 1) if cb > 1e-12 else None}
                energy.append(row)
                energy_rows.append(row)
        temperatures = [c for c in current if "Zone Operative Temperature" in c and "PLENUM" not in c]
        delta = current[temperatures].to_numpy() - native[temperatures].to_numpy()
        people = np.column_stack([series(current, c.split(":", 1)[0], "Zone People Occupant Count") for c in temperatures])
        weight = people * service[:, None]
        for index, name in enumerate(temperatures):
            temperature_rows.append(pd.DataFrame({
                "context": context, "zone": name.split(":", 1)[0],
                "hour_index": np.arange(8760), "month": times.Month, "day": times.Day,
                "hour_endpoint": times.Hour, "dst": times.Dst, "source_service_mask": service,
                "native_people": people[:, index], "current_frozen_operative_c": current[name],
                "restored_native_oa_operative_c": native[name],
                "current_minus_restored_native_c": delta[:, index],
            }))
        hour_energy = pd.DataFrame({"context": context, "hour_index": np.arange(8760),
                                    "month": times.Month, "day": times.Day,
                                    "hour_endpoint": times.Hour, "dst": times.Dst})
        for label, frame in [("current_frozen", current), ("restored_native_oa", native)]:
            for name in ["Heating:Electricity", "Cooling:Electricity", "Fans:Electricity", "Heating:NaturalGas", "Pumps:Electricity"]:
                hour_energy[f"{label}_{name}_kwh"] = meter(frame, name)
        hourly_energy_rows.append(hour_energy)
        results[context] = {"current_replay_against_frozen_archived_baseline": replay,
                            "modes": per_mode, "energy": energy,
                            "operative_temperature_current_minus_restored_native": {
                                "assessed_controlled_zones": len(temperatures), "all_hour_zone_cells": int(delta.size),
                                "mean_absolute_c_all_hour_zone_cells": float(np.abs(delta).mean()),
                                "max_absolute_c_all_hour_zone_cells": float(np.abs(delta).max()),
                                "mean_absolute_c_service_hour_zone_cells": float(np.abs(delta[service]).mean()),
                                "person_weighted_absolute_mean_c_during_source_service_hours": float((np.abs(delta) * weight).sum() / weight.sum()),
                                "person_weighted_signed_mean_c_during_source_service_hours": float((delta * weight).sum() / weight.sum())}}
    pd.concat(flow_rows, ignore_index=True).to_csv(runs / "FLOW_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.concat(temperature_rows, ignore_index=True).to_csv(runs / "TEMPERATURE_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.concat(hourly_energy_rows, ignore_index=True).to_csv(runs / "ENERGY_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(energy_rows).to_csv(runs / "ENERGY_COMPARISONS.csv", index=False)
    summary = {"analysis_script_sha256": sha(__file__), "annual_physical_runs": 4,
               "source_service_mask": "Weekdays local 06:00-22:00, Saturday local06:00-18:00, DST from EnergyPlus Time; this is the frozen analysis service mask, not every nonzero fractional-occupancy hour.",
               "physical_bounds_resolution": "Hourly averaged flow ratios; does not establish timestep maxima or CO2/humidity compliance.",
               "comparison_direction": "Current frozen minus same-model original-OA-EMS-restored, with identical equipment, thermal commands, loads, sizing and weather.",
               "contexts": results,
               "decision": "Keep current frozen simulations: they reproduce archived baselines exactly, satisfy their declared native VRP-request gate, and respect hourly OA/supply physical bounds. Restored original cap fails the same ventilation gate. Correct provenance text to disclose the three disabled max-OA EMS programs as an operating-control adaptation, not an unchanged original controller. Four baseline tests do not measure its influence on every personalized command or establish field calibration.",
               "source_dhw_warning_scope": "Native WaterUse objects have nonzero zone sensible (0.2) and latent (0.05) fractions. They remain thermally coupled internal gains. Original EnergyPlus caps hot flow at total flow when target exceeds available temperature, and computes mixed-water temperature and gains from the available water. Excluding DHW fuel from the controlled-HVAC energy boundary does not remove its zone coupling. Do not claim zero warnings or zero coupling."}
    (runs / "PROBE_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({c: {"replay_exact": results[c]["current_replay_against_frozen_archived_baseline"]["all_shared_numeric_outputs_exactly_equal"],
                          "current_below99_hours": results[c]["modes"]["current_frozen"]["service_hours_with_any_ahu_below_99pct_request"],
                          "native_below99_hours": results[c]["modes"]["original_oa_ems_restored"]["service_hours_with_any_ahu_below_99pct_request"]} for c in results}, indent=2))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run", action="store_true", help="Execute the four predeclared annual probes, then analyze")
    parser.add_argument("--api-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--weather", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.api_child:
        runtime = str(args.runtime_root.resolve())
        directory = os.add_dll_directory(runtime)
        sys.path.insert(0, runtime)
        from pyenergyplus.api import EnergyPlusAPI
        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        code = api.runtime.run_energyplus(state, ["-r", "-w", str(args.weather), "-d", str(args.folder), str(args.folder / "input.idf")])
        api.state_manager.delete_state(state)
        directory.close()
        raise SystemExit(code)
    if args.repository_root is None or args.output_root is None:
        parser.error("--repository-root and --output-root are required")
    if args.run:
        if args.runtime_root is None:
            parser.error("--runtime-root is required with --run")
        args.output_root = args.output_root.resolve()
        args.output_root.mkdir(parents=True, exist_ok=True)
        jobs = prepare(args.repository_root.resolve(), args.runtime_root.resolve(), args.output_root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            groups = [[job for job in jobs if job["context"] == context] for context in ["Miami", "NewDelhi"]]
            results = list(pool.map(execute_context, groups))
        save(args.output_root / "ALL_RUN_RESULTS.json", [row for group in results for row in group])
    analyse(args)


if __name__ == "__main__":
    main()
