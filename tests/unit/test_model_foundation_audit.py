"""An input audit must detect actual physical edits, not only source hashes."""
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "src/utilities/audit_frozen_model_foundation.py"
SPEC = importlib.util.spec_from_file_location("model_foundation_audit", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("old,new,expected", [
    ("Material,Wall,MediumRough,.1,.2,800,900;", "Material,Wall,MediumRough,.01,.2,800,900;", "geometry_envelope_mass"),
    ("ZoneInfiltration:DesignFlowRate,Leak,Zone,Schedule,Flow/ExteriorWallArea,0,,.0005;",
     "ZoneInfiltration:DesignFlowRate,Leak,Zone,Schedule,Flow/ExteriorWallArea,0,,.005;", "infiltration_interzone_airflow"),
])
def test_physical_edit_is_not_hidden_by_named_object_match(old, new, expected):
    changes = AUDIT.object_differences(AUDIT.parse_idf(old), AUDIT.parse_idf(new))
    assert len(changes) == 1
    assert changes[0]["difference"] == "changed"
    assert changes[0]["category"] == expected
    assert changes[0]["source_fields"] != changes[0]["frozen_fields"]


def test_duplicate_output_names_are_compared_as_multiset():
    old = AUDIT.parse_idf("Output:Variable,*,Temperature,Hourly; Output:Variable,*,People,Hourly;")
    new = AUDIT.parse_idf("Output:Variable,*,Temperature,Hourly; Output:Variable,*,Electricity,Hourly;")
    changes = AUDIT.object_differences(old, new)
    assert len(changes) == 1
    assert changes[0]["source_fields"][1] == "People"
    assert changes[0]["frozen_fields"][1] == "Electricity"
