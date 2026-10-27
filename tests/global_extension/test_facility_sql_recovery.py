from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.facility_sql_recovery import build_recovery_plan
from analysis.global_extension.global_matrix import sha256


def test_recovery_plan_accepts_only_matching_input_provenance(tmp_path: Path):
    inputs = tmp_path / "inputs"
    hourly = tmp_path / "hourly"
    inputs.mkdir()
    hourly.mkdir()
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    rows = []
    search = tmp_path / "search"
    for run_id, matching, completed in (
        ("recover", True, True),
        ("rerun", False, True),
        ("incomplete", True, False),
    ):
        input_path = inputs / f"{run_id}.idf"
        input_path.write_text(run_id, encoding="utf-8")
        compact = hourly / f"{run_id}.npz"
        np.savez_compressed(compact, zone_ids=np.asarray(["Zone A"]))
        folder = tmp_path / "runs" / run_id
        folder.mkdir(parents=True)
        (folder / "RUN_QA.json").write_text(json.dumps({
            "completed": True,
            "input_idf_sha256": sha256(input_path),
            "runtime_sha256": "runtime",
        }), encoding="utf-8")
        candidate = search / run_id
        candidate.mkdir(parents=True)
        (candidate / "eplusout.sql").write_text(run_id, encoding="utf-8")
        (candidate / "RUN_QA.json").write_text(json.dumps({
            "completed": completed,
            "input_idf_sha256": sha256(input_path) if matching else "different",
        }), encoding="utf-8")
        rows.append({
            "run_id": run_id, "climate": "Test",
            "hourly_path": compact, "hourly_sha256": sha256(compact),
            "input_sha256": sha256(input_path), "input_idf_path": input_path,
            "weather_path": weather, "run_folder": folder,
        })
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    plan = build_recovery_plan(manifest, sql_search_roots=(search,))
    status = plan.set_index("run_id").recovery_status.to_dict()
    assert status == {
        "recover": "available_search_root",
        "rerun": "rerun_required",
        "incomplete": "rerun_required",
    }
