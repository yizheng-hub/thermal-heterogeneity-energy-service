from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.global_extension.global_matrix import OFFSETS_C, PROBE_TARGETS_C
from analysis.global_extension.analyse_global_results import fixed_tail_from_anchor
from analysis.global_extension.multi_event_dr_extension import (
    DEFAULT_LIMIT_FACTORS,
    MINIMUM_EVENT_DEADBAND_C,
    action_variants,
    directional_protected_zones,
    fixed_tail_profile_ids,
    prepare_reference_manifest,
)
from analysis.global_extension.seasonal_command_extension import (
    build_plan,
    demand_shift,
    fixed_tail_for_seasonal_demand,
)
from analysis.targeted_extension.multi_event_protocol import select_multi_event_protocol


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = (
    ROOT / "data" / "metadata" / "experimental_parameter_provenance.json"
)
FIXED_REFERENCE_PATH = (
    ROOT / "data" / "metadata" / "global_fixed_reference_inputs.provenance.json"
)


def _metadata() -> dict:
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def test_annual_parameter_metadata_matches_executable_constants() -> None:
    parameters = _metadata()["parameters"]
    domain = parameters["annual_probe_target_domain"]["values"]
    assert domain["minimum_c"] == min(PROBE_TARGETS_C)
    assert domain["maximum_c"] == max(PROBE_TARGETS_C)
    assert domain["physical_probe_count"] == len(PROBE_TARGETS_C)
    assert tuple(parameters["annual_candidate_offsets"]["values_c"]) == OFFSETS_C


def test_inherited_deadband_matches_fixed_reference_provenance() -> None:
    metadata = _metadata()["parameters"]["occupied_thermostat_deadband"]
    inherited = json.loads(FIXED_REFERENCE_PATH.read_text(encoding="utf-8"))
    inherited = inherited["prototype"]["thermostat_deadband_evidence"]
    values = metadata["values"]
    assert values["heating_setpoint_c"] == inherited[
        "official_occupied_heating_setpoint_c"
    ]
    assert values["cooling_setpoint_c"] == inherited[
        "official_occupied_cooling_setpoint_c"
    ]
    assert values["experimental_half_deadband_c"] == inherited[
        "experimental_half_deadband_c"
    ]
    assert values["separation_c"] == pytest.approx(
        values["cooling_setpoint_c"] - values["heating_setpoint_c"]
    )


def test_dr_protocol_defaults_match_declared_durations_and_selection_rule() -> None:
    parameters = _metadata()["parameters"]
    signature = inspect.signature(select_multi_event_protocol)
    durations = parameters["dr_preconditioning_and_event_duration"]["values"]
    dates = parameters["dr_event_date_selection"]["values"]
    recovery = parameters["dr_recovery_horizon"]["values"]
    assert signature.parameters["precondition_hours"].default == durations[
        "preconditioning_hours"
    ]
    assert signature.parameters["window_hours"].default == durations["event_hours"]
    assert signature.parameters["recovery_hours"].default == recovery["hours"]
    assert signature.parameters["typical_quantile"].default == dates[
        "facility_electricity_quantiles"
    ][0]
    assert signature.parameters["high_quantile"].default == dates[
        "facility_electricity_quantiles"
    ][1]


def test_common_reference_anchor_selects_only_25_c(tmp_path: Path) -> None:
    anchor = _metadata()["parameters"]["common_reference_anchor"]["values"]
    manifest = pd.DataFrame(
        [
            {"climate": "Test", "target_c": 24.0, "hourly_path": "24.npz",
             "hourly_sha256": "a" * 64},
            {"climate": "Test", "target_c": 25.0, "hourly_path": "25.npz",
             "hourly_sha256": "b" * 64},
            {"climate": "Test", "target_c": 26.0, "hourly_path": "26.npz",
             "hourly_sha256": "c" * 64},
        ]
    )
    climates = tmp_path / "climates.json"
    climates.write_text(
        json.dumps(
            {"climates": [{"climate": "Test", "latitude": 0.0, "longitude": 0.0}]}
        ),
        encoding="utf-8",
    )
    selected = prepare_reference_manifest(manifest, climates)
    assert len(selected) == 1
    assert selected.target_c.iat[0] == anchor["target_c"]


def test_dr_action_library_and_protection_fraction_match_metadata() -> None:
    parameters = _metadata()["parameters"]
    action_values = parameters["dr_temperature_actions"]["values"]
    protection_values = parameters["dr_protected_zone_fraction"]["values"]
    protocol = pd.DataFrame(
        [
            {
                "event_id": "event",
                "month": 7,
                "day": 15,
                "sign": 1.0,
                "start_minute": 14 * 60,
                "duration_hours": 2,
                "precondition_hours": 2,
            }
        ]
    )
    zones = [f"Zone {index:02d}" for index in range(15)]
    profiles = pd.DataFrame(
        {
            "profile_id": [f"profile-{index:02d}" for index in range(15)],
            "zone_id": zones,
            "centre_c": [24.0] * 15,
            "controller_lower_c": [22.0 + index / 100 for index in range(15)],
            "controller_upper_c": [26.0 + index / 100 for index in range(15)],
        }
    )
    variants, audit = action_variants(protocol, profiles, zones)
    assert {item["action_id"] for item in variants} == {
        "uniform_0.5",
        "uniform_1.0",
        "precondition_0.5_uniform_1.0",
        "protected_1.0",
        "precondition_0.5_protected_1.0",
    }
    magnitudes = {
        value
        for item in variants
        for event in item["events"]
        for value in event["event_magnitude_c"].values()
    }
    assert magnitudes == {0.0, *action_values["uniform_event_shift_c"]}
    preconditioning = {
        value
        for item in variants
        for event in item["events"]
        for value in event["precondition_magnitude_c"].values()
    }
    assert preconditioning == {0.0, action_values["preconditioning_shift_c"]}
    default_fraction = inspect.signature(directional_protected_zones).parameters[
        "protected_fraction"
    ].default
    assert default_fraction == protection_values["fraction"]
    protected = audit[audit["protected"]]
    assert protected.zone.nunique() == protection_values[
        "zones_protected_in_15_zone_prototype"
    ]
    assert MINIMUM_EVENT_DEADBAND_C == parameters[
        "dr_minimum_preconditioning_deadband"
    ]["values"]["minimum_c"]


def test_service_factor_and_seasonal_shift_grids_match_metadata() -> None:
    parameters = _metadata()["parameters"]
    assert tuple(parameters["dr_event_service_limit_factors"]["values"]) == (
        DEFAULT_LIMIT_FACTORS
    )
    fractions = inspect.signature(build_plan).parameters["fractions"].default
    assert tuple(parameters["seasonal_demand_shift"]["values"]["contrast_fractions"]) == (
        fractions
    )
    contrast = 2.0
    assert demand_shift(contrast, 1.0, "summer") == pytest.approx(1.0)
    assert demand_shift(contrast, 1.0, "winter") == pytest.approx(-1.0)
    assert demand_shift(contrast, 0.5, "summer") == pytest.approx(0.5)
    assert demand_shift(contrast, 1.0, "spring") == 0.0


def test_fixed_tail_fraction_matches_annual_seasonal_and_event_defaults() -> None:
    fraction = _metadata()["parameters"]["fixed_tail_service_fraction"]["values"][
        "fraction"
    ]
    assert inspect.signature(fixed_tail_from_anchor).parameters["fraction"].default == (
        fraction
    )
    assert inspect.signature(fixed_tail_for_seasonal_demand).parameters[
        "tail_fraction"
    ].default == fraction
    assert inspect.signature(fixed_tail_profile_ids).parameters[
        "tail_fraction"
    ].default == fraction


def test_scenario_classifications_cannot_be_mistaken_for_external_standards() -> None:
    metadata = _metadata()
    allowed = set(metadata["classification_definitions"])
    assert "external_standard" not in allowed
    for name, parameter in metadata["parameters"].items():
        assert parameter["classification"] in allowed, name
        assert parameter["permitted_interpretation"].strip(), name
        assert parameter["prohibited_overstatement"].strip(), name
        assert parameter["additional_experiment_trigger"].strip(), name

    study_defined = {
        "annual_probe_target_domain",
        "annual_candidate_offsets",
        "dr_event_service_limit_factors",
        "fixed_tail_service_fraction",
        "dr_protected_zone_fraction",
        "dr_minimum_preconditioning_deadband",
        "common_reference_anchor",
    }
    for name in study_defined:
        assert metadata["parameters"][name]["classification"] == (
            "study_defined_sensitivity"
        )
