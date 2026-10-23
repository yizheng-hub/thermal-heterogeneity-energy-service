"""Analytical event checks with explicit small synthetic test inputs."""

import numpy as np
import pandas as pd

from analyse_events import event_indices, qualify, baseline_qualifies, choose_response, score_window


def test_eight_independent_constraints():
    anchor = {f"{window}_{metric}_c": 2 for window in ("event", "recovery") for metric in ("mean", "fixed_tail")}
    scores = {f"{window}_{trajectory}_{metric}_c": 1 for window in ("event", "recovery") for trajectory in ("baseline", "perturbed") for metric in ("mean", "fixed_tail")}
    assert qualify(scores, .5, anchor)
    for field in scores:
        failed = scores | {field: 1.001}
        assert not qualify(failed, .5, anchor), field
        assert baseline_qualifies(failed, .5, anchor) == ("_perturbed_" in field)


def test_response_ties_and_nonpositive():
    rows = pd.DataFrame([
        {"service_eligible": True, "event_average_reduction_kw": 2., "run_id": "a", "amplitude_c": .5},
        {"service_eligible": True, "event_average_reduction_kw": 2. + 5e-10, "run_id": "b", "amplitude_c": 1},
        {"service_eligible": False, "event_average_reduction_kw": 100., "run_id": "c", "amplitude_c": 1},
    ])
    assert choose_response(rows)["run_id"] == "a"
    rows["event_average_reduction_kw"] = 0
    assert choose_response(rows) is None


def test_dst_window_and_continuous_recovery():
    data = {"month": np.ones(8760, dtype=int), "day": np.ones(8760, dtype=int),
            "hour": np.tile(np.arange(1, 25), 365), "dst": np.ones(8760, dtype=int)}
    data["month"][:24] = 7
    data["day"][:24] = 28
    event, recovery = event_indices(data, {"month": 7, "day": 28})
    assert event.tolist() == [13, 14]
    assert recovery.tolist() == list(range(15, 63))


def test_sliced_service_has_no_night_weight():
    temperature = np.full((8760, 1), 24.)
    temperature[2:4] = 99
    data = {"zone_ids": np.array(["Z"]), "temperatures": temperature,
            "people": np.ones((8760, 1)), "occupied": np.ones(8760, dtype=bool)}
    data["occupied"][2:4] = False
    profiles = pd.DataFrame({"zone_id": ["Z", "Z"], "centre_c": [23., 25.],
                             "lower_c": [23., 24.], "upper_c": [24., 25.]})
    event = score_window(data, profiles, np.array([1]), np.arange(2))
    recovery = score_window(data, profiles, np.array([1]), np.arange(2, 50))
    assert event["summary"]["mean_reference_mismatch_c"] == 1
    assert recovery["summary"]["mean_reference_mismatch_c"] == 1
    assert np.isnan(recovery["hourly"]["mean_reference_mismatch_c"][:2]).all()
    assert recovery["summary"]["occupied_native_person_hours"] == 46


if __name__ == "__main__":
    test_eight_independent_constraints()
    test_response_ties_and_nonpositive()
    test_dst_window_and_continuous_recovery()
    test_sliced_service_has_no_night_weight()
    print("PASS: 4 analytical event tests; all eight inequalities checked independently")
