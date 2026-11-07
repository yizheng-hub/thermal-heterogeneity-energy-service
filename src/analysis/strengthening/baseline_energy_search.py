"""Give the simple baselines the same stopped offset grid as the demand controls."""

from common import *
from candidate_runs import one, SIM, MODELS
from run_global_matrix import _write_run_idf
import pandas as pd
import numpy as np
import concurrent.futures as cf
import argparse


def plan():
    stop = json.loads((ROOT / "audit/GRID_STOP.json").read_text())
    assert stop["status"] == "completed"
    expected = 2754 if stop["further_refinement_executed"] else 1458
    assert (
        len(pd.read_csv(ROOT / "results/candidate_refinement/STOPPED_GRID_METRICS.csv")) == expected
    ), "Required candidate scores are incomplete"
    base = pd.read_csv(ROOT / "config/BASELINE_PLAN.csv")
    rows = []
    step = 0.125 if stop["further_refinement_executed"] else 0.25
    offsets = np.round(np.arange(-1, 1 + step / 2, step), 8)
    for row in base.to_dict("records"):
        original = json.loads(row["targets_c"])
        for offset in offsets:
            if offset == 0:
                continue
            targets = {z: float(np.clip(round(t + offset, 8), 20, 27)) for z, t in original.items()}
            physical = (
                "refine_"
                + hashlib.sha256(
                    json.dumps([row["climate"], targets], sort_keys=True).encode()
                ).hexdigest()[:24]
            )
            path = SIM / "candidate_inputs" / (physical + ".idf")
            _write_run_idf(Path(MODELS[row["climate"]]["prepared_idf"]), targets, path)
            rows.append(
                {
                    **row,
                    "offset_c": offset,
                    "targets_c": json.dumps(targets, sort_keys=True),
                    "physical_id": physical,
                    "input_idf_path": str(path),
                    "input_sha256": digest(path),
                    "targets_sha256": hashlib.sha256(
                        json.dumps(targets, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "config/BASELINE_OFFSETS_PLAN.csv", index=False)
    write_json(
        ROOT / "audit/BASELINE_OFFSET_SEARCH_PLAN.json",
        {
            "logical_new_candidates": len(frame),
            "physical_operations_before_cache_reuse": frame.physical_id.nunique(),
            "grid_step_c": step,
            "nominal_candidates_already_completed": 216,
            "all54_settings": True,
            "same_control_boundaries": True,
        },
    )


def execute(stage):
    frame = pd.read_csv(ROOT / "config/BASELINE_OFFSETS_PLAN.csv")
    if stage == "pilot":
        pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
        frame = frame[
            frame.climate.isin(pilot["climates"])
            & frame.composition_id.isin(pilot["compositions"])
            & frame.offset_c.isin([-0.25, 0.25])
        ]
        assert len(frame) == 48
    else:
        assert json.loads((ROOT / "audit/BASELINE_OFFSETS_PILOT.json").read_text())[
            "all_complete_and_engineering_pass"
        ]
    jobs = frame.drop_duplicates("physical_id")
    records = []
    with cf.ThreadPoolExecutor(max_workers=6 if stage == "pilot" else 12) as pool:
        futures = {pool.submit(one, row): row for row in jobs.itertuples()}
        for f in cf.as_completed(futures):
            row = futures[f]
            try:
                record = f.result()
            except Exception as error:
                record = {
                    "physical_id": row.physical_id,
                    "completed": False,
                    "relative_engineering_pass": False,
                    "exception": repr(error),
                }
            records.append(record)
            pd.DataFrame(records).to_csv(
                ROOT / "results" / ("BASELINE_OFFSETS_" + stage.upper() + "_PROGRESS.csv"),
                index=False,
            )
            print(stage, len(records), len(jobs), record["relative_engineering_pass"], flush=True)
    result = frame.merge(
        pd.DataFrame(records), on="physical_id", suffixes=("", "_execution"), validate="many_to_one"
    )
    result.to_csv(
        ROOT / "results" / ("BASELINE_OFFSETS_" + stage.upper() + "_MANIFEST.csv"), index=False
    )
    report = {
        "logical_candidates": len(frame),
        "physical_operations_before_cache": len(jobs),
        "all_complete_and_engineering_pass": bool(
            result.completed.fillna(False).all()
            & result.relative_engineering_pass.fillna(False).all()
        ),
    }
    write_json(ROOT / "audit" / ("BASELINE_OFFSETS_" + stage.upper() + ".json"), report)
    record_status(
        "baseline_offset_search_" + stage,
        "completed" if report["all_complete_and_engineering_pass"] else "partial",
        report,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "pilot", "full"])
    a = p.parse_args()
    plan() if a.stage == "plan" else execute(a.stage)
