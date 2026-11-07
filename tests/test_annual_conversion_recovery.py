import json
from pathlib import Path
import pytest
from utilities import recover_completed_annual_conversion as module


def test_existing_partial_hourly_output_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    compact = tmp_path / "hourly.npz"
    compact.write_bytes(b"previous partial data")
    with pytest.raises(FileExistsError, match="partial postprocessing"):
        module.recover(folder, tmp_path / "input.idf", tmp_path / "weather.epw", compact)
    assert compact.read_bytes() == b"previous partial data"


def test_changed_completed_physical_output_is_rejected_before_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "ROOT", tmp_path)
    folder = tmp_path / "run"
    folder.mkdir()
    idf, weather = tmp_path / "input.idf", tmp_path / "weather.epw"
    runtime = tmp_path / "tools/EnergyPlus-22.1.0/energyplus.exe"
    runtime.parent.mkdir(parents=True)
    for path in [idf, weather, runtime]:
        path.write_text("locked " + path.name)
    physical = {}
    for name in ["eplusout.eso", "eplusout.sql", "eplusout.err", "eplusout.eio"]:
        path = folder / name
        path.write_text("original " + name)
        physical[name] = module.sha(path)
    provenance = {"runtime_backend": "official_python_c_api_subprocess", "runtime_api_sha256": "dll"}
    monkeypatch.setattr(module, "runtime_provenance", lambda _: provenance)
    execution = {"status": "verified_outputs_committed", "exit_code": 0, **provenance,
                 "runtime_executable_sha256": module.sha(runtime),
                 "scientific_input_sha256": {str(idf): module.sha(idf), str(weather): module.sha(weather)},
                 "retained_sha256": physical}
    (folder / "ENGINE_EXECUTION_test.json").write_text(json.dumps(execution))
    (folder / "eplusout.eso").write_text("changed after completion")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Changed physics must not be converted"))
    with pytest.raises(ValueError, match="physical output changed: eplusout.eso"):
        module.recover(folder, idf, weather, tmp_path / "hourly.npz")

