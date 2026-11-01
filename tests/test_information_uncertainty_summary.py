import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
from summarize_information_uncertainty import summarize, ENDPOINTS


def test_descriptive_selection_summary_keeps_setting_denominator_and_rejects_duplicates():
    frame = pd.DataFrame({"climate": ["A", "B"], "composition": ["C", "C"], "scenario": ["S", "S"]})
    for name in ENDPOINTS:
        frame[name] = [0.0, 1.0]
    row = summarize(frame, 2, ["scenario"])[0]
    assert row["evaluated_settings"] == row["planned_settings"] == 2
    assert row["conditional_choice_disagreement_fraction_median"] == 0.5
    assert row["zone_mean_selected_excess_p95_c_min"] == 0
    with pytest.raises(ValueError, match="duplicated"):
        summarize(pd.concat([frame.iloc[:1], frame.iloc[:1]]), 2, ["scenario"])
