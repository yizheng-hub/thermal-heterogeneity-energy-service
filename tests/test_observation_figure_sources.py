"""Check geographic pooling and document-specific policy denominators."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/utilities"))
from observation_figure_sources import map_points, policy_positions


def test_map_pools_shared_locations_without_losing_active_records():
    ashrae = pd.DataFrame({"country": ["china", "india"], "city": ["beijing", "bangalore"],
        "lat": [40., np.nan], "lon": [116., np.nan], "reference_g050_c": [22., 25.]})
    china = pd.DataFrame({"active_conditioning": [True, False], "city_normalized": ["beijing"] * 2,
        "city": ["Beijing City"] * 2, "reference_g050_c": [26., 99.]})
    coordinates = pd.DataFrame({"city_normalized": ["beijing"], "latitude": [40.], "longitude": [116.]})
    overrides = pd.DataFrame({"country": ["india"], "city": ["bangalore"],
        "latitude": [13.], "longitude": [77.6]})
    points, records = map_points(ashrae, china, coordinates, overrides)
    assert len(points) == 2 and len(records) == 3
    beijing = points[points.city.eq("beijing")].iloc[0]
    assert beijing.record_count == 2 and beijing.demand_center_c == 24.
    with pytest.raises(ValueError, match="retained coordinates"):
        map_points(ashrae, china, coordinates, overrides.iloc[:0])


def test_india_policy_uses_activity_and_season_rules_with_its_own_denominator():
    declarations = pd.DataFrame([{"plot_order": 1, "policy_id": "IND", "plot_label": "India",
        "jurisdiction_key": "india", "archive_match_rule": "india_ecsbc_met_season", "layer": "design"}])
    data = pd.DataFrame({"country": ["india"] * 4, "met": [1., 1.1, 1.2, 1.3],
        "season": ["summer", "winter", "cool/dry", "summer"],
        "reference_g050_c": [22., 24., 14., 99.], "unit_group": list("abcd")})
    result = policy_positions(data, declarations).set_index("position")
    assert result.records.eq(3).all()
    assert result.proportion.to_dict() == {"below": 1/3, "within": 1/3, "above": 1/3}
