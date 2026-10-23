"""Analytic checks for synchrony, participation and service-preserving fallback."""

from datetime import datetime, timedelta

from run_portfolio import aggregate


def fixture():
    protocol = {"portfolio_buildings": 108, "dr_participation_fractions": [0.25, 0.5, 1.0]}
    types, hourly, design = [], [], []
    for i in range(3):
        key = {"climate_context": "test", "scope": "zone", "control": "pc", "event_id": "july", "composition_id": str(i), "layout": "fixed"}
        design.append({"composition_id": str(i), "layout": "fixed"})
        types.append(key | {"floor_area_m2": 100, "baseline_case_id": f"b{i}", "event_case_id": f"e{i}", "baseline_engineering_status": "PASS", "event_engineering_status": "PASS", "baseline_service_qualified": i != 2, "event_service_qualified": True})
        for h in range(50):
            base = 100
            event = base - 10 * (i + 1) if h < 2 else base
            if h == i + 2:
                event += 10 * (i + 1)
            hourly.append(key | {"hour_index": h, "timestamp_local": (datetime(2001, 7, 1, 14) + timedelta(hours=h)).isoformat(), "phase": "event" if h < 2 else "recovery", "timestep_hours": 1, "baseline_electric_kwh": base, "event_electric_kwh": event, "baseline_gas_hhv_kwh": 0, "event_gas_hhv_kwh": 0})
    return types, hourly, design, protocol


def test_analytic_aggregation():
    args = fixture()
    summary, curves, members = aggregate(*args)
    q25, q50, q100 = summary
    assert q25["event_average_reduction_kw"] == 270
    assert q25["event_electric_saving_kwh"] == 540
    assert q25["recovery_extra_electric_kwh"] == 270
    assert q25["event_plus_48h_net_electric_saving_kwh"] == 270
    assert q25["recovery_synchronous_peak_extra_kw"] == 180
    assert q25["sum_individual_recovery_peak_extra_kw_upper_bound"] == 270
    assert q25["event_average_reduction_kw_per_1000m2"] == 25
    assert q25["nominal_participating_buildings"] == 27
    assert q25["active_qualified_positive_response_buildings"] == 18
    assert q25["baseline_service_qualified_buildings"] == 72
    assert not q25["all_baselines_service_qualified"]
    assert q50["event_average_reduction_kw"] == 540
    assert q100["event_average_reduction_kw"] == 1080
    assert curves[0]["baseline_electric_kwh"] == 10800
    assert curves[0]["perturbed_electric_kwh"] == 10800 - 270
    assert len(curves) == 150 and len(members) == 9


def test_calendar_rejected():
    types, hours, design, protocol = fixture()
    hours[1]["timestamp_local"] = hours[0]["timestamp_local"]
    try:
        aggregate(types, hours, design, protocol)
    except ValueError as exc:
        assert "consecutive" in str(exc)
    else:
        raise AssertionError("Nonconsecutive local hours were accepted")


def test_missing_type_rejected():
    types, hours, design, protocol = fixture()
    try:
        aggregate(types[:-1], hours, design, protocol)
    except ValueError as exc:
        assert "Missing" in str(exc)
    else:
        raise AssertionError("Missing composition type was silently deleted")


def test_engineering_invalid_event_falls_back():
    types, hours, design, protocol = fixture()
    types[0]["event_engineering_status"] = "FAIL"
    summary, _, _ = aggregate(types, hours, design, protocol)
    assert summary[0]["event_average_reduction_kw"] == 180
    assert summary[0]["active_qualified_positive_response_buildings"] == 9


if __name__ == "__main__":
    test_analytic_aggregation()
    test_calendar_rejected()
    test_missing_type_rejected()
    test_engineering_invalid_event_falls_back()
    print("PASS: 4 analytical portfolio tests")
