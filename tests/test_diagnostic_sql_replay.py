import json
import numpy as np
import pytest
from utilities import replay_diagnostic_sql as module


def arrays():
    result = {key: np.zeros((8760, 1) if key in {"temperatures", "people"} else 8760)
              for key in module.FIELDS}
    result.update({key: np.ones(8760, dtype=int) for key in module.CLOCK})
    result.update(zone_ids=np.array(["ZONE"]), occupied=np.ones(8760, dtype=bool))
    return result


def test_replay_comparison_checks_all_six_energy_meters_and_clock():
    original = arrays()
    changed = {key: value.copy() for key, value in original.items()}
    changed["facility_electric_kwh"][100] = 1e-5
    with pytest.raises(ValueError, match="1e-8"):
        module.compare(original, changed)
    changed = {key: value.copy() for key, value in original.items()}
    changed["dst"][100] = 0
    with pytest.raises(ValueError, match="Clock"):
        module.compare(original, changed)
    changed["dst"][100] = 1
    assert max(module.compare(original, changed).values()) == 0


def test_output_acceptance_preserves_raw_engineering_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    data = arrays()
    reference, compact, sql = tmp_path / "reference.npz", tmp_path / "new.npz", tmp_path / "full.sql"
    np.savez_compressed(reference, **data)
    np.savez_compressed(compact, **data)
    sql.write_bytes(b"retained full physical evidence")
    monkeypatch.setattr(module, "sql_arrays", lambda *_: data)
    case = {"original_hourly": module.asset(reference), "input_idf": {"sha256": "input"},
            "weather": {"sha256": "weather"}, "runtime_sha256": "runtime"}
    qa = tmp_path / "RUN_QA.json"
    qa.write_text(json.dumps({"completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "input_idf_sha256": "input", "weather_sha256": "weather", "runtime_sha256": "runtime", "engineering_pass": False}))
    result = module.validate(case, sql, compact, qa, tmp_path / "validation.json")
    assert result["status"] == "passed" and result["raw_engineering_pass"] is False
    assert sql.exists()


def test_registered_assets_cannot_escape_the_project(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path / "project")
    with pytest.raises(ValueError, match="inside this project"):
        module.inside("../foreign.sql")
