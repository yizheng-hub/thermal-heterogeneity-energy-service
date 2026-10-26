from analysis.targeted_extension.trajectory_quality import classify_summary


def _valid_summary() -> dict:
    return {
        "completed": True,
        "hours": 8760,
        "severe": 0,
        "direct_numerical_warnings": [],
        "recurring_warnings": [],
        "direct_coil_operating_range_warnings": [],
        "hourly_electric_closure_max_kwh": 0.0,
        "source_occupancy_ventilation_requirement_max_delta": 0.0,
        "same_source_sizing": True,
        "minimum_outdoor_air_delivery_request_ratio": 1.0,
    }


def test_isolated_iteration_warning_is_flagged_but_keeps_complete_output():
    summary = _valid_summary()
    summary["direct_numerical_warnings"] = [
        "** Warning ** SimHVAC: Maximum iterations (20) exceeded for all HVAC loops"
    ]
    result = classify_summary(summary)
    assert result["primary_output_valid"] is True
    assert result["strict_warning_free"] is False
    assert result["isolated_hvac_iteration_warnings"] == 1


def test_physical_warning_fails_both_quality_scopes():
    summary = _valid_summary()
    summary["direct_numerical_warnings"] = ["** Warning ** PsyTwbFnTdbWPb failed"]
    result = classify_summary(summary)
    assert result["primary_output_valid"] is False
    assert result["strict_warning_free"] is False
