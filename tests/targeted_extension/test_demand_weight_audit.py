import numpy as np
import pandas as pd

from analysis.targeted_extension.demand_weight_audit import audit_profiles


def test_support_points_preserve_native_occupancy(tmp_path):
    np.savez(
        tmp_path / "probe.npz",
        native_people=np.array([[10.0, 2.0], [5.0, 0.0]]),
        zone_ids=np.array(["z1", "z2"]),
    )
    profiles = pd.DataFrame(
        {
            "context": ["Example"] * 3,
            "composition_id": [1] * 3,
            "layout": ["random"] * 3,
            "zone_id": ["z1", "z1", "z2"],
        }
    )
    contexts = {"Example": {"probes": "probe.npz"}}
    result = audit_profiles(profiles, contexts, tmp_path, "test")
    assert result.loc[0, "native_person_hours"] == 17.0
    assert result.loc[0, "recovered_person_hours"] == 17.0
    assert result.loc[0, "weight_conservation_pass"]
