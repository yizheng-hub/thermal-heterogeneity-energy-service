from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/validation"))
from shared_room_reproduction import windows  # noqa: E402


def test_one_response_per_person_is_selected_before_temperature_qualification():
    frame = pd.DataFrame(
        {
            "input_row": [0, 1, 2, 3],
            "room": ["r"] * 4,
            "user": ["a", "a", "b", "c"],
            "vote": [1, -1, -1, 0],
            "time": pd.to_datetime(
                ["2001-01-01 14:05", "2001-01-01 14:25", "2001-01-01 14:15", "2001-01-01 14:15"],
                utc=True,
            ),
            "temperature_c": [24, 24, 24.1, 24.2],
        }
    )
    result, kept = windows(frame, 30, 3, 0.5)
    assert len(result) == 1 and result.included.all()
    assert set(kept.input_row) == {0, 2, 3}
    assert result.category.iloc[0] == "Opposed"
    assert result.averaged_burden.iloc[0] == 0
    frame.loc[0, "temperature_c"] = 26
    result, kept = windows(frame, 30, 3, 0.5)
    assert not result.included.any() and kept.empty
    assert not result.temperature_limit_met.any()
