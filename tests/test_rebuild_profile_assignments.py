from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utilities.rebuild_profile_assignments import ZONES, sample_assignments, zone_inventory  # noqa: E402


def test_physical_slots_come_from_model_geometry_and_people_objects(tmp_path):
    side = np.sqrt(4982.19 / 15)
    objects = []
    for i, zone in enumerate(ZONES):
        floor = [
            "BuildingSurface:Detailed",
            zone + " floor",
            "Floor",
            "construction",
            zone,
            "Ground",
            "",
            "NoSun",
            "NoWind",
            "",
            "",
            "4",
            "0",
            "0",
            "0",
            str(side),
            "0",
            "0",
            str(side),
            str(side),
            "0",
            "0",
            str(side),
            "0",
        ]
        people = [
            "People",
            zone + " people",
            zone,
            "schedule",
            "Area/Person",
            "",
            "",
            str(side**2 / (18 if i < 12 else 17)),
        ]
        objects.extend([floor, people])
    model = tmp_path / "synthetic.idf"
    model.write_text("\n".join(",".join(o) + ";" for o in objects))
    inventory = zone_inventory(model)
    assert len(inventory) == 15
    assert inventory.profiles.sum() == 267
    assert np.isclose(inventory.floor_area_m2.sum(), 4982.19, atol=0.01)
    assert (inventory.profiles > 0).all()


def test_random_assignment_preserves_sample_order_identity_and_capacity():
    pool = pd.DataFrame(
        {"record_id": np.arange(400), "source_equal_probability": np.full(400, 1 / 400)}
    )
    slots = pd.DataFrame({"zone_id": ["a", "b", "c"], "profiles": [80, 90, 97]})
    first = sample_assignments(pool, slots, [2026102001], ashrae=True)
    second = sample_assignments(pool, slots, [2026102001], ashrae=True)
    pd.testing.assert_frame_equal(first, second)
    assert first.record_id.nunique() == 267
    assert first.profile_id.nunique() == 267
    assert first.groupby("zone_id").size().to_dict() == {"a": 80, "b": 90, "c": 97}
    chosen = np.random.default_rng(2026102001).choice(
        400, 267, replace=False, p=pool.source_equal_probability.to_numpy()
    )
    assert np.array_equal(first.sort_values("sample_position").record_id, chosen)


def test_raw_driver_contains_no_historical_result_or_assignment_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location("raw_driver", ROOT / "reproduce_from_raw.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module.commands(ROOT / "output/new_empty_run", Path("energyplus.exe"), 2)
    text = str(commands)
    assert "HEM_archive" not in text
    assert "ASSIGNMENTS.csv.gz" not in text.replace("ASHRAE_ASSIGNMENTS.csv.gz", "").replace(
        "CTCD_ASSIGNMENTS.csv.gz", ""
    )
    assert "results/global_extension" not in text
    assert set(commands) == {
        "demand",
        "observations",
        "matrix",
        "annual",
        "seasonal",
        "spatial",
        "response",
        "secondary",
        "strengthening",
    }


def test_raw_input_inventory_names_sources_instead_of_cached_assignments(tmp_path):
    from utilities.audit_reproduction_inputs import raw_inputs

    config = tmp_path / "src/analysis/global_extension"
    config.mkdir(parents=True)
    (config / "GLOBAL_CLIMATES.json").write_text('{"climates":[{"weather":"city.epw"}]}')
    inputs = raw_inputs(tmp_path, include_validation=True)
    assert all("ASSIGNMENTS" not in str(p) and "results" not in p.parts for p in inputs)
    assert any(p.name == "city.epw" for p in inputs)
    assert any(p.name == "enth_tabular_merged.csv" for p in inputs)


def test_raw_entry_rejects_external_outputs_before_execution(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "raw_driver_scope_test", ROOT / "reproduce_from_raw.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys, "argv", ["reproduce_from_raw.py", "--output-root", str(tmp_path / "outside"), "--plan"]
    )
    with pytest.raises(ValueError, match="inside this project"):
        module.main()
    assert not (tmp_path / "outside").exists()


def test_cleanup_keeps_sizing_and_monthly_tables_for_downstream_stages(tmp_path, monkeypatch):
    import sqlite3
    import global_matrix as gm

    idf, weather, runtime = [
        tmp_path / name for name in ["input.idf", "weather.epw", "energyplus.exe"]
    ]
    for path in [idf, weather, runtime]:
        path.write_text("test input")
    (tmp_path / "EnergyPlusAPI.dll").write_text("test library")
    run = tmp_path / "run"
    compact = tmp_path / "hourly.npz"

    def execute(command, **kwargs):
        working = Path(kwargs["cwd"])
        with sqlite3.connect(working / "eplusout.sql") as connection:
            connection.execute("create table ComponentSizes (name text, value real)")
            connection.execute("insert into ComponentSizes values ('fixed_capacity',123.5)")
            connection.execute("create table TabularData (name text, value real)")
            connection.execute("insert into TabularData values ('January gas',456.75)")
            connection.execute("create table ReportData (value real)")
            connection.execute("insert into ReportData values (2.0)")
        connection.close()
        (working / "eplusout.err").write_text(
            "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;"
        )
        return 0, "finished"

    def extract(folder, zones, output):
        np.savez_compressed(output, temperatures=np.array([[24.0]]))
        return {"hourly_sha256": gm.sha256(output), "engineering_pass": True}

    import thermal_heterogeneity.energyplus_launcher as launcher
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_run_process", execute)
    monkeypatch.setattr(gm, "extract_hourly", extract)
    qa = gm.run_energyplus(idf, weather, runtime, run, compact, ["A"], cleanup_raw=True)
    assert not (run / "eplusout.sql").exists()
    assert compact.is_file()
    audit = Path(qa["execution_audit_path"])
    assert audit.is_file()
    assert json.loads(audit.read_text())["status"] == "verified_outputs_committed"
    metadata = Path(qa["sql_metadata_path"])
    assert gm.sha256(metadata) == qa["sql_metadata_sha256"]
    with sqlite3.connect(metadata) as connection:
        assert connection.execute("select * from ComponentSizes").fetchall() == [
            ("fixed_capacity", 123.5)
        ]
        assert connection.execute("select * from TabularData").fetchall() == [
            ("January gas", 456.75)
        ]
        assert connection.execute("select count(*) from ReportData").fetchone()[0] == 0


def test_automatic_solver_check_rejects_coil_failures_before_running(tmp_path, monkeypatch):
    import automatic_solver_stability as stability

    failed = pd.DataFrame(
        [
            {
                "run_id": "probe_bad",
                "climate": "test",
                "target_c": 24.0,
                "relative_engineering_pass": False,
                "new_warning_categories": '["coil_operating_range"]',
                "critical_warning_count_increases": '{"coil_operating_range":1}',
            }
        ]
    )
    folder = tmp_path / "probe_runs/runs/probe_bad"
    folder.mkdir(parents=True)
    (folder / "RUN_QA.json").write_text(
        '{"completed":true,"reported_severe_count":0,"fatal_occurrences":0,"unexplained_recurring_groups":[]}'
    )
    monkeypatch.setattr(stability, "reconstruct_probe_manifest", lambda *args, **kwargs: failed)
    with pytest.raises(ValueError, match="beyond solver iteration limits"):
        stability.audit_probes(tmp_path, tmp_path / "energyplus.exe", 1)
