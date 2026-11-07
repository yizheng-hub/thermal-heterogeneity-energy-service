import sys
from pathlib import Path
import numpy as np
import sqlite3
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from lossless_history_storage import retain_sql_metadata, save_states_against, load_states


def test_metadata_preserves_other_tables_without_duplicate_report_values(tmp_path):
    path = tmp_path / "source.sql"
    with sqlite3.connect(path) as c:
        c.execute("create table ReportData(Value real)")
        c.execute("insert into ReportData values(2.1)")
        c.execute("create table Zones(Name text,Area real)")
        c.execute("insert into Zones values('A',12.3)")
    c.close()
    qa = retain_sql_metadata(path)
    with sqlite3.connect(qa["sql_metadata_path"]) as c:
        assert c.execute("select * from Zones").fetchall() == [("A", 12.3)]
        assert c.execute("select count(*) from ReportData").fetchone()[0] == 0
    assert qa["source_table_row_counts"]["ReportData"] == 1 and not path.exists()


def test_state_difference_storage_is_bit_exact(tmp_path):
    a = np.array([[0.0, -0.0, 1.125], [21.23456789, -20.0, 1.0e-10]])
    b = a.copy()
    b[1, 0] = 23.1
    keys = np.array(["x", "y", "z"])
    normal = tmp_path / "normal.npz"
    action = tmp_path / "action.npz"
    np.savez_compressed(normal, keys=keys, values=a)
    save_states_against(action, {"keys": keys, "values": b}, normal)
    restored = load_states(action)
    assert np.array_equal(restored["values"].view(np.uint64), b.view(np.uint64))


def test_metadata_export_recovers_an_interrupted_partial_destination(tmp_path):
    path = tmp_path / "source.sql"
    target = tmp_path / "simulation_metadata.sql"
    for database in [path, target]:
        with sqlite3.connect(database) as connection:
            connection.execute("create table Zones(Name text)")
            connection.execute("insert into Zones values(?)", ("complete" if database == path else "partial",))
        connection.close()
    qa = retain_sql_metadata(path)
    with sqlite3.connect(qa["sql_metadata_path"]) as connection:
        assert connection.execute("select * from Zones").fetchall() == [("complete",)]


def test_completed_response_keeps_its_exported_sql_metadata(tmp_path, monkeypatch):
    import single_event_simulation as module

    for name in ["model.idf", "weather.epw", "energyplus.exe", "EnergyPlusAPI.dll"]:
        (tmp_path / name).write_text("locked input")
    monkeypatch.setattr(module, "SIM", tmp_path)
    monkeypatch.setattr(module, "RUNTIME", tmp_path / "energyplus.exe")
    monkeypatch.setattr(module, "warning_profile", lambda _: {})
    monkeypatch.setattr(module, "extract", lambda *_: (
        {"month": np.array([1])}, {"keys": np.array(["air"]), "values": np.array([[21.]])}))

    def energyplus(*args, **kwargs):
        folder = Path(kwargs["cwd"])
        (folder / "eplusout.err").write_text("EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;")
        with sqlite3.connect(folder / "eplusout.sql") as connection:
            connection.execute("create table Zones(Name text)")
            connection.execute("insert into Zones values('Office')")
        connection.close()
        return 0, "finished"

    import thermal_heterogeneity.energyplus_launcher as launcher
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_run_process", energyplus)
    qa = module.execute(tmp_path / "model.idf", tmp_path / "weather.epw", {}, "test")
    assert qa["completed"]
    assert Path(qa["sql_metadata_path"]).is_file()
    cached = module.execute(tmp_path / "model.idf", tmp_path / "weather.epw", {}, "test")
    assert cached["sql_metadata_sha256"] == qa["sql_metadata_sha256"]
    Path(qa["sql_metadata_path"]).write_text("changed metadata")
    with pytest.raises(ValueError, match="missing or changed: sql_metadata_path"):
        module.execute(tmp_path / "model.idf", tmp_path / "weather.epw", {}, "test")
