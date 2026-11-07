from common import *
import argparse
import concurrent.futures as cf
import time
import pandas as pd
import numpy as np
import sqlite3
from global_matrix import run_energyplus
from run_global_matrix import (
    _write_run_idf,
    ZONES,
    warning_profile,
    relative_engineering_qa,
    set_maximum_hvac_iterations,
    _cached_run,
)
from thermal_heterogeneity.energyplus_launcher import runtime_provenance

SIM = Path(json.loads((ROOT / "config/PROJECT.json").read_text())["simulation_root"])
MODELS = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())
OLD = pd.read_csv(MATRIX / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv")
PROBE = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
RUNTIME_HASH = digest(RUNTIME)
RUNTIME_INFO = runtime_provenance(RUNTIME)


def freeze_sizing_lines(path):
    prefixes = (
        "Zone Sizing Information,",
        "System Sizing Information,",
        "Plant Sizing Information,",
        "Component Sizing Information,",
    )
    return [
        x.strip()
        for x in Path(path).read_text(errors="replace").splitlines()
        if x.strip().startswith(prefixes)
    ]


def plan(further=False):
    source = json.loads(
        (MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text()
    )
    rows = []
    for base in source:
        if base["offset_c"] != 0:
            continue
        for offset in (
            [-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875]
            if further
            else [-0.75, -0.25, 0.25, 0.75]
        ):
            row = dict(base)
            row["offset_c"] = offset
            # Quarter-degree offsets must not be rounded back to a tenth-degree grid.
            row["targets_c"] = {
                z: float(np.clip(round(t + offset, 8), 20, 27))
                for z, t in base["targets_c"].items()
            }
            row["boundary_zones"] = sum(t in (20, 27) for t in row["targets_c"].values())
            rows.append(row)
    assert len(rows) == (1296 if further else 648)
    for r in rows:
        targets = r["targets_c"]
        cfg = MODELS[r["context"]]
        payload = json.dumps([r["context"], targets], sort_keys=True).encode()
        physical = "refine_" + hashlib.sha256(payload).hexdigest()[:24]
        path = SIM / "candidate_inputs" / f"{physical}.idf"
        _write_run_idf(Path(cfg["prepared_idf"]), targets, path)
        r["input_idf_path"] = str(path)
        r["input_sha256"] = digest(path)
        r["targets_sha256"] = hashlib.sha256(
            json.dumps(targets, sort_keys=True).encode()
        ).hexdigest()
        r["physical_id"] = physical
        r["climate"] = r["context"]
        r["cohort_label"] = next(x for x in source if x["composition_id"] == r["composition_id"])[
            "composition_id"
        ]
    pd.DataFrame(
        [
            {k: json.dumps(v, sort_keys=True) if k == "targets_c" else v for k, v in r.items()}
            for r in rows
        ]
    ).to_csv(
        ROOT / "config" / ("CANDIDATE_FURTHER_PLAN.csv" if further else "CANDIDATE_PLAN.csv"),
        index=False,
    )
    print(
        "Planned",
        len(rows),
        "logical and",
        len({r["physical_id"] for r in rows}),
        "physical operations",
        flush=True,
    )


def one(row):
    folder = SIM / "candidate_runs" / row.physical_id
    hourly = SIM / "candidate_hourly" / f"{row.physical_id}.npz"
    qa_path = folder / "RUN_QA.json"
    reused = OLD[OLD.climate.eq(row.climate) & OLD.input_sha256.eq(row.input_sha256)]
    if len(reused):
        item = reused.iloc[0].to_dict()
        item["reused_existing_physical_operation"] = True
        item["physical_id"] = row.physical_id
        return item
    if qa_path.exists() and hourly.exists():
        qa = _cached_run({"run_id": row.physical_id, "run_folder": folder, "compact": hourly,
            "input_sha256": row.input_sha256, "weather_sha256": digest(Path(MODELS[row.climate]["weather_path"]))},
            RUNTIME_HASH, RUNTIME_INFO)
    else:
        qa = run_energyplus(
            Path(row.input_idf_path),
            Path(MODELS[row.climate]["weather_path"]),
            RUNTIME,
            folder,
            hourly,
            ZONES,
            cleanup_raw=False,
        )
    anchor = PROBE[PROBE.climate.eq(row.climate) & PROBE.target_c.eq(25)].iloc[0]
    anchor_folder = Path(anchor.hourly_path).parents[1] / "runs" / anchor.run_id
    warnings = warning_profile(folder / "eplusout.err")
    base_warnings = warning_profile(anchor_folder / "eplusout.err")
    relative = relative_engineering_qa(qa, warnings, base_warnings)
    exception_categories = set(relative["new_warning_categories"]) | set(
        relative["critical_warning_count_increases"]
    )
    hvac_iteration_only = exception_categories <= {"hvac_iterations"} and not qa.get(
        "unexplained_recurring_groups"
    )
    sizing_equal = (
        freeze_sizing_lines(folder / "eplusout.eio")
        == freeze_sizing_lines(anchor_folder / "eplusout.eio")
        if qa.get("completed")
        else False
    )
    stability = {}
    passed = bool(
        qa.get("completed")
        and qa.get("reported_severe_count") == 0
        and not qa.get("fatal_occurrences")
        and relative["engineering_pass_relative_to_prototype_baseline"]
        and sizing_equal
    )
    if qa.get("completed") and not relative["engineering_pass_relative_to_prototype_baseline"]:
        path100 = SIM / "candidate_inputs100" / Path(row.input_idf_path).name
        path100.parent.mkdir(exist_ok=True)
        path100.write_text(set_maximum_hvac_iterations(Path(row.input_idf_path).read_text(), 100))
        retry = SIM / "candidate_retry100" / row.physical_id
        h100 = SIM / "candidate_hourly100" / f"{row.physical_id}.npz"
        if (retry / "RUN_QA.json").exists() and h100.exists():
            q100 = _cached_run({"run_id": row.physical_id, "run_folder": retry, "compact": h100,
                "input_sha256": digest(path100), "weather_sha256": digest(Path(MODELS[row.climate]["weather_path"]))},
                RUNTIME_HASH, RUNTIME_INFO)
        else:
            q100 = run_energyplus(
                path100,
                Path(MODELS[row.climate]["weather_path"]),
                RUNTIME,
                retry,
                h100,
                ZONES,
                cleanup_raw=False,
            )
        if q100.get("completed"):
            with np.load(hourly) as a, np.load(h100) as b:
                occ = a["occupied"][:, None] & (a["people"] > 0)
                td = float(np.max(np.abs(a["temperatures"] - b["temperatures"])[occ]))

                def energy(x):
                    return float(
                        x["facility_electric_kwh"].sum() + x["facility_gas_hhv_kwh_annual"]
                    )

                ed = abs(energy(a) - energy(b)) / energy(a)
                pdiff = float(np.max(np.abs(a["people"] - b["people"])))
            categories_equal = {k: v > 0 for k, v in warnings.items()} == {
                k: v > 0 for k, v in warning_profile(retry / "eplusout.err").items()
            }
            stability = {
                "temperature_delta_c": td,
                "relative_energy_delta": ed,
                "people_delta": pdiff,
                "warning_categories_stable": categories_equal,
            }
            passed = bool(
                hvac_iteration_only
                and td <= 0.02
                and ed <= 0.001
                and pdiff == 0
                and categories_equal
                and q100.get("reported_severe_count") == 0
                and sizing_equal
            )
        for f in retry.glob("*"):
            if f.is_file() and f.suffix.lower() not in [
                ".json",
                ".err",
                ".eio",
                ".log",
                ".sql",
                ".zst",
            ]:
                f.unlink()
    # Retain SQL and sizing/error evidence; remove only new bulky raw reporting files.
    for f in folder.glob("*"):
        if f.is_file() and f.suffix.lower() not in [
            ".json",
            ".err",
            ".eio",
            ".log",
            ".sql",
            ".zst",
        ]:
            f.unlink()
    record = {
        "physical_id": row.physical_id,
        "run_id": row.physical_id,
        "climate": row.climate,
        "input_sha256": row.input_sha256,
        "input_idf_path": row.input_idf_path,
        "hourly_path": str(hourly),
        "hourly_sha256": digest(hourly) if hourly.exists() else "",
        "qa_path": str(qa_path),
        "run_folder": str(folder),
        "seconds": qa.get("seconds"),
        "completed": qa.get("completed", False),
        "relative_engineering_pass": passed,
        "raw_relative_engineering_pass": relative[
            "engineering_pass_relative_to_prototype_baseline"
        ],
        "new_warning_categories": json.dumps(relative["new_warning_categories"]),
        "critical_warning_increases": json.dumps(relative["critical_warning_count_increases"]),
        "capacity_sizing_identical_to_reference": sizing_equal,
        "solver_stability": json.dumps(stability),
        "reused_existing_physical_operation": False,
    }
    write_json(folder / "STRENGTHENING_QA.json", record)
    return record


def execute(stage, worker_override=None):
    status = ROOT / "audit/status/original_annual_reproduction.json"
    assert (
        status.exists() and json.loads(status.read_text())["status"] == "completed"
    ), "Original replication must pass first."
    f = pd.read_csv(
        ROOT
        / "config"
        / ("CANDIDATE_FURTHER_PLAN.csv" if stage == "further" else "CANDIDATE_PLAN.csv")
    )
    if stage == "further":
        refinement = json.loads(
            (ROOT / "results/candidate_refinement/REFINEMENT_AUDIT.json").read_text()
        )
        assert refinement["further_uniform_refinement_required"]
    if stage == "pilot":
        pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
        f = f[f.climate.isin(pilot["climates"]) & f.composition_id.isin(pilot["compositions"])]
        assert len(f) == 72
    else:
        audit = json.loads((ROOT / "audit/CANDIDATE_PILOT.json").read_text())
        assert audit["all_complete_and_engineering_pass"], "Pilot quality gate has not passed."
    jobs = f.drop_duplicates("physical_id")
    records = []
    workers = (
        6
        if stage == "pilot"
        else int(
            json.loads((ROOT / "config/PROJECT.json").read_text()).get("confirmation_workers", 6)
        )
    )
    if worker_override is not None:
        workers = worker_override
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, row): row for row in jobs.itertuples()}
        for fut in cf.as_completed(futures):
            row = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "physical_id": row.physical_id,
                    "completed": False,
                    "relative_engineering_pass": False,
                    "exception": repr(e),
                }
            records.append(r)
            pd.DataFrame(records).to_csv(
                ROOT / "results" / f"CANDIDATE_{stage.upper()}_PHYSICAL_PROGRESS.csv", index=False
            )
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "done": len(records),
                        "total": len(jobs),
                        "physical_id": row.physical_id,
                        "passed": r["relative_engineering_pass"],
                    }
                ),
                flush=True,
            )
    physical = pd.DataFrame(records)
    f = f.drop(columns=["run_id", "hourly_path", "hourly_sha256"], errors="ignore").merge(
        physical, on="physical_id", suffixes=("", "_execution"), validate="many_to_one"
    )
    f["cohort_label"] = f.composition_id.map(
        {
            2026102001: "Multinational A",
            2026102002: "Multinational B",
            2026102003: "Multinational C",
            2026102501: "China A",
            2026102502: "China B",
            2026102503: "China C",
        }
    )
    f.to_csv(ROOT / "results" / f"CANDIDATE_{stage.upper()}_MANIFEST.csv", index=False)
    report = {
        "logical_candidates": len(f),
        "physical_operations": len(physical),
        "all_complete_and_engineering_pass": bool(
            f.completed.fillna(False).all() and f.relative_engineering_pass.fillna(False).all()
        ),
        "failed_logical_candidates": int((~f.relative_engineering_pass.fillna(False)).sum()),
        "stage": stage,
        "status": "completed" if f.relative_engineering_pass.fillna(False).all() else "partial",
    }
    write_json(ROOT / "audit" / f"CANDIDATE_{stage.upper()}.json", report)
    record_status("candidate_" + stage, report["status"], report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "pilot", "full", "plan_further", "further"])
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    plan(a.stage == "plan_further") if a.stage in ["plan", "plan_further"] else execute(
        a.stage, a.workers
    )
