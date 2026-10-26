from analysis.targeted_extension.engineering_failure_diagnostic import _failure_reason


def test_failure_reason_does_not_infer_target_spread():
    qa = {
        "hours": 8760,
        "severe": 0,
        "direct_numerical_warnings": ["SimHVAC maximum iterations"],
        "engineering_pass": False,
    }
    assert _failure_reason(qa) == "hvac_numerical_convergence"


def test_non_hvac_water_warning_does_not_reject_a_qualified_run():
    qa = {
        "hours": 8760,
        "severe": 0,
        "direct_numerical_warnings": [],
        "all_recurring_warnings": [
            {"treatment_relevance": "not_thermostat_hvac"}
        ],
        "engineering_pass": True,
    }
    assert _failure_reason(qa) == "qualified"
