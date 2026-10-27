"""Run a finite demand-response action library from one common 25 C baseline.

The physical action library is generated without demand labels.  Building
permission contains uniform event shifts and preconditioning actions;
native-zone permission contains the same actions plus single-group and
one-group-protected shifts.  February and July use the same action template
with opposite thermal direction, preserving the original heating/cooling
event dates and the source thermostat deadband.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve()
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from generate_model_inputs import objects, physical_hash, signature  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from run_demand_aware_energyplus import engineering_qa, export_hourly, sha256, write_json  # noqa: E402


def split_action_block(
    fields: list[str],
    pre_delta: float,
    event_delta: float,
    *,
    pre_start_minute: int = 720,
    event_start_minute: int = 840,
    event_end_minute: int = 960,
) -> list[str]:
    """Split compact schedules around one declared event and preconditioning window."""
    if not 0 <= pre_start_minute <= event_start_minute < event_end_minute <= 1440:
        raise ValueError("Event and preconditioning times must be ordered within one day")
    result: list[str] = []
    previous = 0
    eligible = False
    index = 0
    while index < len(fields):
        token = fields[index]
        if token.lower().startswith("for"):
            result.append(token)
            previous = 0
            eligible = any(day in token.lower() for day in ("weekdays", "saturday"))
            index += 1
            continue
        if not token.lower().startswith("until:"):
            raise ValueError(f"Unsupported compact schedule token: {token}")
        hour, minute = map(int, token.split(":", 1)[1].strip().split(":"))
        end = hour * 60 + minute
        value = float(fields[index + 1])
        cuts = sorted({previous, end} | {
            cut for cut in (pre_start_minute, event_start_minute, event_end_minute)
            if previous < cut < end
        })
        for start, stop in zip(cuts[:-1], cuts[1:]):
            delta = 0.0
            if eligible and start >= pre_start_minute and stop <= event_start_minute:
                delta = pre_delta
            elif eligible and start >= event_start_minute and stop <= event_end_minute:
                delta = event_delta
            result.extend([f"Until: {stop // 60:02d}:{stop % 60:02d}", str(value + delta)])
        previous = end
        index += 2
    return result


def event_action_input(
    text: str,
    events: list[dict],
    precondition_magnitude_c: dict[str, float],
    event_magnitude_c: dict[str, float],
) -> str:
    """Apply zone-specific shifts while leaving all non-schedule physics fixed."""
    parsed = objects(text)
    duals = {item[1]: item for item in parsed if item[0] == "ThermostatSetpoint:DualSetpoint"}
    schedule_zones: dict[str, set[str]] = {}
    for thermostat in [item for item in parsed if item[0] == "ZoneControl:Thermostat"]:
        zone = thermostat[2]
        dual = duals[thermostat[5]]
        for field in (2, 3):
            schedule_zones.setdefault(dual[field], set()).add(zone)
    declared = set(precondition_magnitude_c) | set(event_magnitude_c)
    controlled = {zone for zones in schedule_zones.values() for zone in zones}
    if declared != controlled:
        raise ValueError("Action vector must contain every and only controlled thermal zone")

    for item in parsed:
        if item[0] != "Schedule:Compact" or item[1] not in schedule_zones:
            continue
        zones = schedule_zones[item[1]]
        pre_values = {float(precondition_magnitude_c[zone]) for zone in zones}
        event_values = {float(event_magnitude_c[zone]) for zone in zones}
        if len(pre_values) != 1 or len(event_values) != 1:
            raise ValueError("A shared thermostat schedule received conflicting zone actions")
        if item[3].lower() != "through: 12/31" or sum(
            token.lower().startswith("through:") for token in item
        ) != 1:
            raise ValueError("Expected one original annual thermostat schedule block")
        original = item[4:]
        expanded = item[:3]
        previous = date(2001, 1, 1)
        pre_magnitude = next(iter(pre_values))
        event_magnitude = next(iter(event_values))
        for event in sorted(events, key=lambda value: (value["month"], value["day"])):
            current = date(2001, int(event["month"]), int(event["day"]))
            if previous < current:
                end = current - timedelta(days=1)
                expanded += [f"Through: {end.month}/{end.day}"] + original
            sign = float(event["sign"])
            event_start = int(event.get("start_minute", 840))
            event_end = event_start + int(event.get("duration_hours", 2)) * 60
            pre_start = event_start - int(event.get("precondition_hours", 2)) * 60
            expanded += [f"Through: {current.month}/{current.day}"] + split_action_block(
                original,
                pre_delta=-sign * pre_magnitude,
                event_delta=sign * event_magnitude,
                pre_start_minute=pre_start,
                event_start_minute=event_start,
                event_end_minute=event_end,
            )
            previous = current + timedelta(days=1)
        if previous <= date(2001, 12, 31):
            expanded += ["Through: 12/31"] + original
        item[:] = expanded
    return "\n".join(",".join(item) + ";" for item in parsed)


def action_library(context: str, zones: list[str]) -> list[dict]:
    zero = dict.fromkeys(zones, 0.0)
    rows: list[dict] = []

    def add(permission: str, action_id: str, family: str, pre: dict, event: dict) -> None:
        rows.append({
            "context": context,
            "permission": permission,
            "action_id": action_id,
            "action_family": family,
            "precondition_magnitude_c": {zone: float(pre[zone]) for zone in zones},
            "event_magnitude_c": {zone: float(event[zone]) for zone in zones},
        })

    for permission in ("building", "native_zone"):
        for amplitude in (0.5, 1.0):
            add(permission, f"uniform_{amplitude:.1f}", "uniform_shift", zero,
                dict.fromkeys(zones, amplitude))
        for pre in (0.5, 1.0):
            for event in (0.5, 1.0):
                add(permission, f"pre_{pre:.1f}_uniform_{event:.1f}",
                    "precondition_uniform_shift", dict.fromkeys(zones, pre),
                    dict.fromkeys(zones, event))

    for zone in zones:
        safe = zone.lower().replace(" ", "_")
        for amplitude in (0.5, 1.0):
            single = dict(zero)
            single[zone] = amplitude
            add("native_zone", f"single_{safe}_{amplitude:.1f}",
                "single_group_shift", zero, single)
            protected = dict.fromkeys(zones, amplitude)
            protected[zone] = 0.0
            add("native_zone", f"protect_{safe}_{amplitude:.1f}",
                "one_group_protected", zero, protected)
    return rows


def run(args: argparse.Namespace) -> None:
    input_root = args.input_root.resolve()
    model_root = args.model_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    runtime_hash = sha256(runtime)
    models = json.loads((model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    contexts = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    event_dates = json.loads(
        (input_root / "engineering/EVENT_DATES_BEFORE_RUNS.json").read_text(encoding="utf-8")
    )
    selected = set(args.contexts or contexts)
    actions: list[dict] = []
    jobs: dict[str, dict] = {}
    logical: list[dict] = []
    for context in contexts:
        if context not in selected:
            continue
        model = models[context]
        if runtime_hash != model["runtime_sha256"]:
            raise ValueError("EnergyPlus runtime differs from the frozen model contract")
        baseline_idf = model_root / model["baseline_idf"]
        weather = model_root / model["weather"]
        if sha256(baseline_idf) != model["baseline_idf_sha256"]:
            raise ValueError(f"Frozen model bytes changed for {context}")
        if sha256(weather) != model["weather_sha256"]:
            raise ValueError(f"Frozen weather bytes changed for {context}")
        source = baseline_idf.read_text(encoding="utf-8")
        for action in action_library(context, contexts[context]["zones"]):
            actions.append(action)
            text = event_action_input(
                source,
                event_dates[context],
                action["precondition_magnitude_c"],
                action["event_magnitude_c"],
            )
            if physical_hash(text) != physical_hash(source):
                raise ValueError("A demand-response action changed non-control building physics")
            normalized = signature(text)
            run_id = hashlib.sha256(
                (normalized + model["weather_sha256"] + runtime_hash).encode()
            ).hexdigest()[:32]
            logical.append({
                "context": context,
                "permission": action["permission"],
                "action_id": action["action_id"],
                "action_family": action["action_family"],
                "run_id": run_id,
                "normalized_idf_sha256": normalized,
            })
            jobs.setdefault(run_id, {
                "run_id": run_id,
                "context": context,
                "text": text,
                "weather": str(weather),
                "zones": contexts[context]["zones"],
                "probe_path": str(input_root / contexts[context]["probes"]),
            })
    write_json(output_root / "ACTION_LIBRARY.json", actions)
    write_json(output_root / "LOGICAL_ACTIONS.json", logical)
    write_json(output_root / "RUN_PROTOCOL.json", {
        "runtime_sha256": runtime_hash,
        "runtime_version": "22.1.0-ed759b17ee",
        "contexts": sorted(selected),
        "logical_actions": len(logical),
        "unique_physical_inputs": len(jobs),
        "event_months": [2, 7],
        "precondition_hours": "12:00-14:00",
        "event_hours": "14:00-16:00",
        "recovery_hours": 48,
        "action_library_uses_demand_labels": False,
        "building_parameters_changed": False,
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
        result = {
            "run_id": job["run_id"], "context": job["context"], "completed": success,
            "exit_code": completed.returncode, "elapsed_seconds": time.time() - started,
            "engineering_pass": False,
        }
        if success:
            baseline = model_root / "baseline_evidence" / job["context"]
            with np.load(job["probe_path"], allow_pickle=False) as probe:
                occupied_source_indices = probe["occupied_source_indices"]
            qa = engineering_qa(job["context"], folder, baseline, job["zones"], occupied_source_indices)
            result.update(qa)
            import pandas as pd
            frame = pd.read_csv(folder / "eplusout.csv")
            result.update(export_hourly(
                frame, job["zones"], occupied_source_indices,
                output_root / "hourly" / f"{job['run_id']}.npz",
            ))
            write_json(folder / "ENGINEERING_QA.json", qa)
            if not args.retain_raw:
                for name in (
                    "eplusout.csv", "eplusout.sql", "eplusout.mtr", "eplusout.rdd",
                    "eplusout.mdd", "eplusout.audit", "eplusout.bnd", "eplusout.eso",
                ):
                    path = folder / name
                    if path.exists():
                        path.unlink()
        write_json(summary_path, result)
        return result

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, job): run_id for run_id, job in jobs.items()}
        for future in as_completed(futures):
            run_id = futures[future]
            results[run_id] = future.result()
            print(json.dumps({key: results[run_id].get(key) for key in
                              ("run_id", "context", "completed", "engineering_pass")}), flush=True)
            write_json(output_root / "RUN_RESULTS.json", results)
    import pandas as pd
    manifest = pd.DataFrame(logical)
    for field in ("completed", "engineering_pass", "hourly_sha256"):
        manifest[field] = manifest.run_id.map(
            lambda run_id: results[run_id].get(field, False)
        )
    manifest["hourly_path"] = manifest.run_id.map(
        lambda run_id: f"hourly/{run_id}.npz" if results[run_id].get("completed") else ""
    )
    manifest.to_csv(output_root / "RUN_MANIFEST.csv", index=False)
    write_json(output_root / "EXECUTION_QA.json", {
        "logical_actions": len(manifest),
        "unique_physical_inputs": len(results),
        "unique_completed": sum(bool(value.get("completed")) for value in results.values()),
        "unique_engineering_pass": sum(bool(value.get("engineering_pass")) for value in results.values()),
        "physical_fingerprints_unchanged": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contexts", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retain-raw", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
