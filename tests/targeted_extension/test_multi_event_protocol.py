import numpy as np
import pandas as pd

from analysis.targeted_extension.common_baseline_dr import split_action_block
from analysis.targeted_extension.multi_event_protocol import (
    _window_candidates,
    events_for_idf,
    select_multi_event_protocol,
)


def test_dynamic_event_window_splits_at_declared_hours():
    result = split_action_block(
        ["For: Weekdays", "Until: 24:00", "25"],
        -0.5,
        1.0,
        pre_start_minute=360,
        event_start_minute=480,
        event_end_minute=600,
    )
    assert result == [
        "For: Weekdays", "Until: 06:00", "25.0",
        "Until: 08:00", "24.5", "Until: 10:00", "26.0",
        "Until: 24:00", "25.0",
    ]


def test_protocol_selects_two_nonoverlapping_events_per_season():
    time = pd.date_range("2021-01-01", "2021-12-31 23:00", freq="h")
    hour = time.hour.to_numpy()
    weekday = time.weekday.to_numpy() < 5
    occupied = weekday & (hour >= 6) & (hour < 22)
    seasonal = np.sin(2 * np.pi * (time.dayofyear.to_numpy() - 1) / 365)
    daily = np.where((hour >= 8) & (hour < 10), 30.0, 10.0)
    load = np.maximum(0.1, 20 + 8 * seasonal + daily)
    cooling = np.where(seasonal >= 0, load, 0.0)
    heating = np.where(seasonal < 0, load, 0.0)
    protocol = select_multi_event_protocol(
        time, load, occupied, latitude=40, cooling_kw=cooling, heating_kw=heating
    )
    assert len(protocol) == 8
    assert protocol.groupby("season_block").size().eq(2).all()
    assert set(protocol.load_level) == {"high", "typical"}
    starts = protocol.start_timestamp.sort_values().to_numpy()
    assert np.min(np.diff(starts)) >= np.timedelta64(50, "h")
    records = events_for_idf(protocol)
    assert len(records) == 8
    assert {record["duration_hours"] for record in records} == {2}


def test_event_candidate_requires_complete_recovery_window():
    time = pd.date_range("2021-12-28", periods=96, freq="h")
    load = np.ones(len(time))
    occupied = np.zeros(len(time), dtype=bool)
    occupied[-14:-12] = True
    candidates = _window_candidates(
        time,
        load,
        occupied,
        window_hours=2,
        recovery_hours=48,
        precondition_hours=2,
        cooling_kw=load,
        heating_kw=np.zeros(len(time)),
    )
    assert candidates.empty
