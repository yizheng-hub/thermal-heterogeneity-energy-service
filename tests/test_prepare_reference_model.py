import hashlib
from pathlib import Path

import pytest

from utilities.prepare_reference_model import parse_objects, prepare_reference_model


SOURCE = Path(__file__).resolve().parents[1] / (
    "data/upstream/global_climate_matrix/prototypes_22_1/Miami.idf"
)


@pytest.mark.skipif(not SOURCE.is_file(), reason="Requires the user-supplied DOE/PNNL prototype.")
def test_reference_is_exact_and_preserves_noncontrol_model_objects(tmp_path):
    original = SOURCE.read_bytes()
    destination = tmp_path / "Miami/baseline.idf"
    prepare_reference_model(SOURCE, destination)
    generated = destination.read_bytes()
    assert hashlib.sha256(generated).hexdigest() == (
        "c0740b1ed872b2a5a0550be6d205a44013752219a4eb05359e4cc4ac0cccf516"
    )
    assert SOURCE.read_bytes() == original
    before, after = parse_objects(original), parse_objects(generated)
    assert len(before) == 943
    assert len(after) == 1025
    assert [obj[:2] for obj in before] == [obj[:2] for obj in after[:len(before)]]
    schedule_names = {f"{mode}SETP_SCH_YES_OPTIMUM{suffix}"
                      for mode in ("HTG", "CLG") for suffix in ("", "_w_SB")}
    program_names = {"OA_BASED_OPTIMUM_START_EMS_Program", "pacu_vav_bot_MAX_OA_FRAC",
                     "pacu_vav_mid_MAX_OA_FRAC", "pacu_vav_top_MAX_OA_FRAC"}
    changes = []
    for old, new in zip(before, after):
        if old != new:
            kind, name = old[:2]
            assert (kind == "ThermostatSetpoint:DualSetpoint"
                    or (kind == "Schedule:Compact" and name in schedule_names)
                    or (kind == "EnergyManagementSystem:Program" and name in program_names))
            changes.append(kind)
    assert changes.count("ThermostatSetpoint:DualSetpoint") == 15
    assert changes.count("Schedule:Compact") == 4
    assert changes.count("EnergyManagementSystem:Program") == 4
    with pytest.raises(FileExistsError):
        prepare_reference_model(SOURCE, destination)
    assert destination.read_bytes() == generated
