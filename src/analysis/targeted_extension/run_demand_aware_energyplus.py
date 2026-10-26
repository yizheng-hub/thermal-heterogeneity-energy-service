"""Run frozen-building trajectories for the repaired demand-aware controllers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from generate_model_inputs import annual_input, physical_hash, signature  # noqa: E402


NUMERICAL_WARNING_TOKENS = (
    "SimHVAC:", "PsyTwb", "PsyPsat", "Temperature out of range", "WetBulb not converged"
)


def set_maximum_hvac_iterations(text: str, maximum: int | None) -> str:
    """Return the IDF with only the HVAC iteration limit changed."""
    if maximum is None:
        return text
    pattern = re.compile(r"(?is)(ConvergenceLimits\s*,\s*[^,;]+\s*,\s*)\d+(\s*;)")
    updated, count = pattern.subn(rf"\g<1>{maximum}\g<2>", text)
    if count != 1:
        raise ValueError(f"Expected one ConvergenceLimits object, found {count}")
    return updated


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def column(frame: pd.DataFrame, object_name: str, fragment: str) -> pd.Series:
    matches = [name for name in frame if name.startswith(object_name.upper() + ":")
               and fragment in name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {object_name}/{fragment} column, found {matches}")
    return frame[matches[0]]


def meter(frame: pd.DataFrame, name: str) -> pd.Series:
    matches = [column_name for column_name in frame if column_name.startswith(name + " [J]")]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} meter, found {matches}")
    return frame[matches[0]] / 3.6e6


def sizing(folder: Path) -> list[str]:
    prefixes = ("Zone Sizing Information,", "System Sizing Information,",
                "Plant Sizing Information,", "Component Sizing Information,")
    return [line.strip() for line in (folder / "eplusout.eio").read_text(
        encoding="utf-8", errors="replace").splitlines() if line.strip().startswith(prefixes)]


def final_counts(error_text: str) -> tuple[int, int]:
    match = re.findall(
        r"EnergyPlus Completed Successfully--\s*(\d+) Warning;\s*(\d+) Severe Errors;",
        error_text,
    )
    if not match:
        return -1, -1
    warnings, severe = match[-1]
    return int(warnings), int(severe)


def recurring_warnings(error_text: str) -> list[dict]:
    parts = error_text.split("===== Recurring Error Summary =====", 1)
    if len(parts) == 1:
        return []
    section = parts[1].split("===== Final Error Summary =====", 1)[0]
    pattern = re.compile(
        r"\*\* Warning \*\*\s*(.*?)\r?\n(?:.*?\r?\n)*?.*?This error occurred\s+(\d+)\s+total times;",
        re.DOTALL,
    )
    rows = []
    for message, occurrences in pattern.findall(section):
        normalized = message.lower()
        if (("target water temperature" in normalized and "hot water temperature" in normalized)
                or ("hot water temperature" in normalized and "cold water temperature" in normalized)):
            relevance = "not_thermostat_hvac"
        elif any(token.lower() in normalized for token in
                 ("heatexchanger:airtoair", "temperature out of range", "simhvac",
                  "wetbulb not converged", "psypsatfntemp")):
            relevance = "treatment_relevant"
        else:
            relevance = "requires_review"
        rows.append({"message": message.strip(), "occurrences": int(occurrences),
                     "treatment_relevance": relevance})
    return rows


def export_hourly(frame: pd.DataFrame, zones: list[str], occupied_source_indices: np.ndarray,
                  destination: Path) -> dict:
    arrays: dict[str, np.ndarray] = {}
    for key, fragment in (("temperatures", "Operative Temperature"),
                          ("people", "Occupant Count"),
                          ("heating_setpoint_c", "Thermostat Heating Setpoint Temperature"),
                          ("cooling_setpoint_c", "Thermostat Cooling Setpoint Temperature"),
                          ("ventilation_requested_m3_s", "Zone Target Voz Ventilation Flow Rate")):
        arrays[key] = np.column_stack([column(frame, zone, fragment).to_numpy(float)
                                       for zone in zones])
    if len(zones) == 5:
        arrays["ventilation_delivered_m3_s"] = np.column_stack([
            column(frame, zone, "Zone Mechanical Ventilation Standard Density Volume Flow Rate").to_numpy(float)
            for zone in zones
        ])
    for key, name in (("electricity_hvac_meter_kwh", "Electricity:HVAC"),
                      ("heating_electric_kwh", "Heating:Electricity"),
                      ("cooling_electric_kwh", "Cooling:Electricity"),
                      ("fans_electric_kwh", "Fans:Electricity"),
                      ("pumps_electric_kwh", "Pumps:Electricity"),
                      ("gas_hhv_kwh", "Heating:NaturalGas")):
        arrays[key] = meter(frame, name).to_numpy(float)
    arrays["hvac_electric_kwh"] = (arrays["heating_electric_kwh"]
                                    + arrays["cooling_electric_kwh"]
                                    + arrays["fans_electric_kwh"])
    arrays["zone_ids"] = np.asarray(zones)
    arrays["occupied"] = np.zeros(len(frame), dtype=bool)
    arrays["occupied"][np.asarray(occupied_source_indices, dtype=int)] = True
    if len(frame) != 8760 or any(not np.isfinite(value).all() for value in arrays.values()
                                  if value.dtype.kind not in "US"):
        raise ValueError("Hourly export is incomplete or non-finite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {"hourly_path": str(destination), "hourly_sha256": sha256(destination),
            "hours": len(frame)}


def engineering_qa(context: str, folder: Path, baseline: Path, zones: list[str],
                   occupied_source_indices: np.ndarray) -> dict:
    error_text = (folder / "eplusout.err").read_text(encoding="utf-8", errors="replace")
    warnings, severe = final_counts(error_text)
    frame = pd.read_csv(folder / "eplusout.csv")
    base = pd.read_csv(baseline / "eplusout.csv")
    direct = [line.strip() for line in error_text.splitlines()
              if line.lstrip().startswith("** Warning **")
              and any(token in line for token in NUMERICAL_WARNING_TOKENS)]
    recurring = recurring_warnings(error_text)
    relevant = [row for row in recurring if row["treatment_relevance"] != "not_thermostat_hvac"]
    heat = meter(frame, "Heating:Electricity")
    cool = meter(frame, "Cooling:Electricity")
    fans = meter(frame, "Fans:Electricity")
    transformer_columns = [name for name in frame if "Transformer Distribution Electricity Loss Energy" in name]
    transformer = (frame[transformer_columns[0]] / 3.6e6 if transformer_columns
                   else pd.Series(np.zeros(len(frame))))
    closure = float(np.max(np.abs(meter(frame, "Electricity:HVAC") - heat - cool - fans - transformer)))
    invariants = [name for name in frame if "Occupant Count" in name
                  or "Target Voz Ventilation Flow Rate" in name]
    invariant_delta = float(np.max(np.abs(frame[invariants].to_numpy() - base[invariants].to_numpy())))
    source_window = np.zeros(len(frame), dtype=bool)
    source_window[np.asarray(occupied_source_indices, dtype=int)] = True
    ratios: list[float] = []
    if len(zones) == 5:
        for zone in zones:
            actual = column(frame, zone, "Zone Mechanical Ventilation Standard Density Volume Flow Rate").to_numpy(float)
            requested = column(frame, zone, "Zone Target Voz Ventilation Flow Rate").to_numpy(float)
            people = column(frame, zone, "Occupant Count").to_numpy(float)
            # The small-office source closes outdoor air during its first
            # preheat hour.  Require delivery only after it becomes positive
            # in the unchanged common-25 source trajectory.
            base_actual = column(base, zone, "Zone Mechanical Ventilation Standard Density Volume Flow Rate").to_numpy(float)
            selected = source_window & (requested > 1e-10) & (people > 1e-8) & (base_actual > 1e-10)
            ratios.extend((actual[selected] / requested[selected]).tolist())
    else:
        for ahu in ("PACU_VAV_BOT", "PACU_VAV_MID", "PACU_VAV_TOP"):
            actual = column(frame, ahu, "Air System Outdoor Air Mass Flow Rate").to_numpy(float)
            requested = column(frame, ahu, "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate").to_numpy(float)
            selected = source_window & (requested > 1e-10)
            ratios.extend((actual[selected] / requested[selected]).tolist())
    coil = [line.strip() for line in error_text.splitlines()
            if line.lstrip().startswith("** Warning **") and "CalcDoe2DXCoil:" in line]
    minimum_ratio = float(min(ratios)) if ratios else float("nan")
    same_sizing = sizing(folder) == sizing(baseline)
    passed = bool(len(frame) == 8760 and severe == 0 and warnings >= 0 and not direct
                  and not relevant and not coil and closure < 1e-8 and invariant_delta < 1e-10
                  and same_sizing and minimum_ratio >= 0.99)
    return {
        "context": context, "completed": True, "hours": len(frame),
        "reported_warning_count": warnings, "severe": severe,
        "direct_numerical_warnings": direct, "recurring_warnings": recurring,
        "direct_coil_operating_range_warnings": coil,
        "hourly_electric_closure_max_kwh": closure,
        "source_occupancy_ventilation_requirement_max_delta": invariant_delta,
        "same_source_sizing": same_sizing,
        "minimum_outdoor_air_delivery_request_ratio": minimum_ratio,
        "controlled_hvac_electric_kwh": float((heat + cool + fans).sum()),
        "space_heating_gas_kwh_hhv": float(meter(frame, "Heating:NaturalGas").sum()),
        "engineering_pass": passed,
    }


def run_experiment(args: argparse.Namespace) -> None:
    model_root = args.model_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    runtime_hash = sha256(runtime)
    models = json.loads((model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    controls = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    commands = json.loads(args.commands.read_text(encoding="utf-8"))
    jobs: dict[str, dict] = {}
    logical = []
    for command in commands:
        context = command["context"]
        model = models[context]
        control = controls[context]
        baseline_idf = model_root / model["baseline_idf"]
        weather = model_root / model["weather"]
        if sha256(baseline_idf) != model["baseline_idf_sha256"] or sha256(weather) != model["weather_sha256"]:
            raise ValueError(f"Frozen input checksum changed for {context}")
        if runtime_hash != model["runtime_sha256"]:
            raise ValueError("EnergyPlus runtime checksum differs from frozen contract")
        source = baseline_idf.read_text(encoding="utf-8")
        text = annual_input(source, command["targets_c"], control["half_deadband_c"])
        if physical_hash(text) != physical_hash(source):
            raise ValueError("Non-control building physics changed")
        text = set_maximum_hvac_iterations(text, args.maximum_hvac_iterations)
        normalized = signature(text)
        run_id = hashlib.sha256((normalized + model["weather_sha256"] + runtime_hash).encode()).hexdigest()[:32]
        logical.append(command | {"source_run_id": command.get("run_id", ""),
                                  "run_id": run_id, "normalized_idf_sha256": normalized})
        jobs.setdefault(run_id, {"run_id": run_id, "context": context, "text": text,
                                 "weather": str(weather), "zones": control["zones"],
                                 "probe_path": str(input_root / control["probes"]),
                                 "targets_c": command["targets_c"]})
    write_json(output_root / "LOGICAL_DESIGNS.json", logical)
    write_json(output_root / "RUN_PROTOCOL.json", {
        "runtime_sha256": runtime_hash, "runtime_version": "22.1.0-ed759b17ee",
        "logical_designs": len(logical), "unique_physical_inputs": len(jobs),
        "common_25_candidate_present": False, "building_parameters_changed": False,
        "maximum_hvac_iterations": args.maximum_hvac_iterations,
        "solver_iteration_limit_changed": args.maximum_hvac_iterations is not None,
    })

    def execute(job: dict) -> dict:
        folder = output_root / "runs" / job["run_id"]
        summary_path = folder / "RUN_SUMMARY.json"
        if summary_path.exists():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            hourly = output_root / "hourly" / f"{job['run_id']}.npz"
            if (
                bool(cached.get("completed"))
                and hourly.exists()
                and cached.get("hourly_sha256") == sha256(hourly)
            ):
                return cached
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.idf").write_text(job["text"], encoding="utf-8")
        write_json(folder / "INPUTS.json", {key: value for key, value in job.items() if key != "text"})
        started = time.time()
        with (folder / "CONSOLE.log").open("wb") as console:
            completed = subprocess.run(
                [str(runtime), "-r", "-w", job["weather"], "-d", str(folder), str(folder / "input.idf")],
                cwd=folder, stdout=console, stderr=subprocess.STDOUT, timeout=900,
            )
        error_path = folder / "eplusout.err"
        error_text = error_path.read_text(encoding="utf-8", errors="replace") if error_path.exists() else ""
        success = completed.returncode == 0 and "EnergyPlus Completed Successfully" in error_text
        result = {"run_id": job["run_id"], "context": job["context"], "completed": success,
                  "exit_code": completed.returncode, "elapsed_seconds": time.time() - started}
        if success:
            baseline = model_root / "baseline_evidence" / job["context"]
            with np.load(job["probe_path"], allow_pickle=False) as probe:
                occupied_source_indices = probe["occupied_source_indices"]
            qa = engineering_qa(job["context"], folder, baseline, job["zones"],
                                occupied_source_indices)
            result.update(qa)
            frame = pd.read_csv(folder / "eplusout.csv")
            result.update(export_hourly(frame, job["zones"], occupied_source_indices,
                                        output_root / "hourly" / f"{job['run_id']}.npz"))
            write_json(folder / "ENGINEERING_QA.json", qa)
            if not args.retain_raw:
                removed = []
                for name in (
                    "eplusout.csv", "eplusout.sql", "eplusout.mtr", "eplusout.rdd",
                    "eplusout.mdd", "eplusout.audit", "eplusout.bnd", "eplusout.eso",
                ):
                    path = folder / name
                    if path.exists():
                        path.unlink()
                        removed.append(name)
                result["removed_regenerable_outputs"] = removed
        else:
            result["engineering_pass"] = False
        write_json(summary_path, result)
        return result

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): run_id for run_id, job in jobs.items()}
        for future in as_completed(futures):
            run_id = futures[future]
            results[run_id] = future.result()
            print(json.dumps({key: results[run_id].get(key) for key in
                              ("run_id", "context", "completed", "engineering_pass", "elapsed_seconds")}), flush=True)
            write_json(output_root / "RUN_RESULTS.json", results)
    frame = pd.DataFrame(logical)
    for field in ("completed", "engineering_pass", "hourly_sha256"):
        frame[field] = frame["run_id"].map(
            lambda run_id: results[run_id].get(field, False)
        )
    frame["hourly_path"] = frame["run_id"].map(
        lambda run_id: f"hourly/{run_id}.npz" if results[run_id].get("completed") else ""
    )
    frame.drop(columns="targets_c").to_csv(output_root / "RUN_MANIFEST.csv", index=False)
    write_json(output_root / "EXECUTION_QA.json", {
        "logical_designs": len(frame), "unique_physical_inputs": len(results),
        "unique_completed": sum(bool(row.get("completed")) for row in results.values()),
        "unique_engineering_pass": sum(bool(row.get("engineering_pass")) for row in results.values()),
        "physical_fingerprints_unchanged": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maximum-hvac-iterations", type=int)
    parser.add_argument("--retain-raw", action="store_true")
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
