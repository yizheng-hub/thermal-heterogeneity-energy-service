"""Run uniform demand-response actions from each controller's own baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from generate_model_inputs import physical_hash, signature  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from common_baseline_dr import event_action_input  # noqa: E402
from run_demand_aware_energyplus import engineering_qa, export_hourly, sha256, write_json  # noqa: E402


KEYS = ["context", "composition_id", "layout", "permission", "control", "variant", "offset_c"]


def run(args: argparse.Namespace) -> None:
    annual_manifest = args.annual_manifest.resolve()
    model_root = args.model_root.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    runtime_hash = sha256(runtime)

    manifest = pd.read_csv(annual_manifest)
    commands = pd.DataFrame(json.loads(args.commands.resolve().read_text(encoding="utf-8")))
    baselines = manifest.merge(
        commands[KEYS], on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    if not baselines["_merge"].eq("both").all():
        raise ValueError("Annual manifest does not match the controller commands")
    baselines = baselines.drop(columns="_merge")
    baselines = baselines[
        baselines.permission.eq("native_zone") & baselines.offset_c.eq(0)
    ].copy()
    models = json.loads((model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    contexts = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    dates = json.loads(
        (input_root / "engineering/EVENT_DATES_BEFORE_RUNS.json").read_text(encoding="utf-8")
    )

    logical: list[dict] = []
    jobs: dict[str, dict] = {}
    for baseline in baselines.itertuples(index=False):
        context = str(baseline.context)
        model = models[context]
        if runtime_hash != model["runtime_sha256"]:
            raise ValueError("EnergyPlus runtime differs from the frozen model contract")
        baseline_idf = annual_manifest.parent / "runs" / str(baseline.run_id) / "input.idf"
        if not baseline_idf.exists():
            raise FileNotFoundError(baseline_idf)
        source = baseline_idf.read_text(encoding="utf-8")
        if signature(source) != str(baseline.normalized_idf_sha256):
            raise ValueError("Annual baseline IDF differs from its manifest identity")
        zones = contexts[context]["zones"]
        weather = model_root / model["weather"]
        if not weather.is_file() or sha256(weather) != model["weather_sha256"]:
            raise ValueError("Weather file differs from the frozen model contract")
        for amplitude in (0.5, 1.0):
            zero = dict.fromkeys(zones, 0.0)
            event = dict.fromkeys(zones, amplitude)
            text = event_action_input(source, dates[context], zero, event)
            if physical_hash(text) != physical_hash(source):
                raise ValueError("An event action changed non-control building physics")
            normalized = signature(text)
            run_id = hashlib.sha256(
                (normalized + model["weather_sha256"] + runtime_hash).encode()
            ).hexdigest()[:32]
            item = {
                "data_source": args.data_source,
                "context": context,
                "composition_id": int(baseline.composition_id),
                "layout": str(baseline.layout),
                "permission": "native_zone",
                "control": str(baseline.control),
                "baseline_run_id": str(baseline.run_id),
                "event_amplitude_c": float(amplitude),
                "run_id": run_id,
                "normalized_idf_sha256": normalized,
                "weather_sha256": model["weather_sha256"],
                "runtime_sha256": runtime_hash,
            }
            logical.append(item)
            jobs.setdefault(run_id, {
                "run_id": run_id,
                "context": context,
                "text": text,
                "weather": str(weather),
                "zones": zones,
                "probe_path": str(input_root / contexts[context]["probes"]),
            })

    write_json(output_root / "LOGICAL_ACTIONS.json", logical)
    write_json(output_root / "RUN_PROTOCOL.json", {
        "data_source": args.data_source,
        "baseline_policy": "direct_offset_zero_native_zone",
        "event_amplitudes_c": [0.5, 1.0],
        "event_months": [2, 7],
        "recovery_hours": 48,
        "logical_actions": len(logical),
        "unique_physical_inputs": len(jobs),
        "building_parameters_changed": False,
        "runtime_sha256": runtime_hash,
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
        started = time.time()
        with (folder / "CONSOLE.log").open("wb") as console:
            completed = subprocess.run(
                [str(runtime), "-r", "-w", job["weather"], "-d", str(folder),
                 str(folder / "input.idf")],
                cwd=folder, stdout=console, stderr=subprocess.STDOUT, timeout=900,
            )
        error_path = folder / "eplusout.err"
        error_text = error_path.read_text(encoding="utf-8", errors="replace") \
            if error_path.exists() else ""
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
            qa = engineering_qa(
                job["context"], folder, baseline, job["zones"], occupied_source_indices
            )
            result.update(qa)
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

    result_table = pd.DataFrame(logical)
    for field in ("completed", "engineering_pass", "hourly_sha256"):
        result_table[field] = result_table.run_id.map(
            lambda run_id: results[run_id].get(field, False)
        )
    result_table["hourly_path"] = result_table.run_id.map(
        lambda run_id: f"hourly/{run_id}.npz" if results[run_id].get("completed") else ""
    )
    result_table.to_csv(output_root / "RUN_MANIFEST.csv", index=False)
    write_json(output_root / "EXECUTION_QA.json", {
        "logical_actions": len(result_table),
        "unique_physical_inputs": len(results),
        "unique_completed": sum(bool(value.get("completed")) for value in results.values()),
        "unique_engineering_pass": sum(
            bool(value.get("engineering_pass")) for value in results.values()
        ),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retain-raw", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
