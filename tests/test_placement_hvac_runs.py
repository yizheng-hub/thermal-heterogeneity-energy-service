import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
import placement_hvac_interaction as placement


def test_placement_execution_failure_is_retried_without_changing_the_physical_command(tmp_path, monkeypatch):
    model = tmp_path / "model.idf"
    model.write_text("model")
    models = {"Delhi": {"system": "independent_pthp", "fixed_capacity_model": str(model)}}
    targets = {"core": 25.0}
    plan = pd.DataFrame([dict(climate="Delhi", composition_id=1, control="personal_center",
                              targets_c=json.dumps(targets), layout="mixed")])
    monkeypatch.setattr(placement, "OUT", tmp_path)

    def interrupted(model, command):
        raise FileNotFoundError("input unavailable")

    monkeypatch.setattr(placement.hc, "operate", interrupted)
    with pytest.raises(FileNotFoundError, match="input unavailable"):
        placement.execute_phase(plan, models, "paired")
    cache = tmp_path / "independent_pthp_PHYSICAL_CACHE.json"
    assert not cache.exists()

    def compute(model, command):
        assert command["targets_c"] == targets
        assert placement.hc.operation_label(command) == command["scenario_tag"]
        return {"completed": True, "validation_pass": True}

    monkeypatch.setattr(placement.hc, "operate", compute)
    records = placement.execute_phase(plan, models, "paired")
    key = records[0]["operation_key"]
    cache.write_text(json.dumps({key: {"exception": "input unavailable", "validation_pass": False}}))
    assert placement.execute_phase(plan, models, "paired") == records
