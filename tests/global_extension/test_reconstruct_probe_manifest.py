import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
ENERGY = MODULE.parent / "energy_service"
sys.path.insert(0, str(MODULE))
sys.path.insert(0, str(ENERGY))

from generate_model_inputs import annual_input  # noqa: E402
from global_matrix import sha256  # noqa: E402
from reconstruct_probe_manifest import reconstruct_probe_manifest  # noqa: E402


MODEL = """\
ConvergenceLimits,1,{iterations},2,8;
ZoneControl:Thermostat,Z1 Thermostat,Z1,Control,ThermostatSetpoint:DualSetpoint,Z1 Dual;
ThermostatSetpoint:DualSetpoint,Z1 Dual,Z1 heat,Z1 cool;
Schedule:Compact,Z1 heat,Temperature,Through: 12/31,For: Weekdays Saturday,Until: 24:00,23.5;
Schedule:Compact,Z1 cool,Temperature,Through: 12/31,For: Weekdays Saturday,Until: 24:00,26.5;
"""


def _write_batch(root: Path, *, iterations: int = 20) -> None:
    models = root / "models"
    probes = root / "probe_runs"
    for path in (models, probes / "runs", probes / "inputs", probes / "hourly"):
        path.mkdir(parents=True, exist_ok=True)
    model = models / "Alpha.idf"
    model.write_text(MODEL.format(iterations=iterations), encoding="utf-8")
    weather_hash = hashlib.sha256(b"weather").hexdigest()
    manifest = {
        "Alpha": {
            "prepared_idf": str(model.resolve()),
            "prepared_idf_sha256": sha256(model),
            "weather_sha256": weather_hash,
            "half_deadband_c": 1.5,
        }
    }
    (models / "MODEL_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = model.read_text(encoding="utf-8")
    for index, target in enumerate((24.0, 25.0)):
        run_id = f"probe_{index}"
        input_idf = probes / "inputs" / f"{run_id}.idf"
        input_idf.write_text(annual_input(source, {"Z1": target}, 1.5), encoding="utf-8")
        compact = probes / "hourly" / f"{run_id}.npz"
        np.savez_compressed(
            compact,
            zone_ids=np.asarray(["Z1"]),
            temperatures=np.asarray([[25.0]]),
            people=np.asarray([[1.0]]),
            occupied=np.asarray([True]),
            hvac_electric_kwh=np.asarray([1.0]),
            gas_hhv_kwh=np.asarray([0.0]),
        )
        run_folder = probes / "runs" / run_id
        run_folder.mkdir()
        qa = {
            "completed": True,
            "reported_severe_count": 0,
            "fatal_occurrences": 0,
            "unexplained_recurring_groups": [],
            "input_idf_sha256": sha256(input_idf),
            "weather_sha256": weather_hash,
            "hourly_sha256": sha256(compact),
        }
        (run_folder / "RUN_QA.json").write_text(json.dumps(qa), encoding="utf-8")
        (run_folder / "eplusout.err").write_text(
            "EnergyPlus Completed Successfully-- 0 Warning; 0 Severe Errors;\n",
            encoding="utf-8",
        )


def test_reconstructs_only_complete_declared_matrix(tmp_path):
    _write_batch(tmp_path)
    result = reconstruct_probe_manifest(
        tmp_path,
        targets_c=(24.0, 25.0),
        expected_climates=1,
        expected_solver_iterations=20,
    )
    assert result[["climate", "target_c"]].to_dict("records") == [
        {"climate": "Alpha", "target_c": 24.0},
        {"climate": "Alpha", "target_c": 25.0},
    ]
    assert result["maximum_hvac_iterations"].eq(20).all()
    assert result["relative_engineering_pass"].all()


def test_rejects_wrong_solver_setting(tmp_path):
    _write_batch(tmp_path, iterations=40)
    with pytest.raises(ValueError, match="uses 40 HVAC iterations"):
        reconstruct_probe_manifest(
            tmp_path,
            targets_c=(24.0, 25.0),
            expected_climates=1,
            expected_solver_iterations=20,
        )


def test_rejects_compact_hash_change(tmp_path):
    _write_batch(tmp_path)
    compact = tmp_path / "probe_runs" / "hourly" / "probe_0.npz"
    compact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Compact trajectory evidence changed"):
        reconstruct_probe_manifest(
            tmp_path,
            targets_c=(24.0, 25.0),
            expected_climates=1,
            expected_solver_iterations=20,
        )
