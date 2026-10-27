from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.prepare_figure_inputs import (
    climate_metadata,
    event_effect_source,
    global_effect_source,
    observation_map_source,
    seasonal_command_effect_source,
    seasonal_effect_source,
)


def _climates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "climate": ["Beijing"],
            "country": ["China"],
            "continent": ["Asia"],
            "latitude": [39.8],
            "longitude": [116.47],
            "climate_zone": ["4A"],
        }
    )


def test_climate_metadata_reads_declared_sites(tmp_path: Path) -> None:
    path = tmp_path / "climates.json"
    path.write_text(
        '{"climates":[{"climate":"Beijing","country":"China",'
        '"continent":"Asia","latitude":39.8,"longitude":116.47,'
        '"climate_zone":"4A"}]}',
        encoding="utf-8",
    )
    assert climate_metadata(path, expected_climates=1).climate.tolist() == ["Beijing"]


def test_observation_map_keeps_archives_distinct() -> None:
    map_row = {
        "city": "beijing",
        "country": "china",
        "latitude": 39.8,
        "longitude": 116.5,
        "supported": True,
        "analysis_units": 20,
    }
    warm = pd.DataFrame([map_row])
    cold = pd.DataFrame([{**map_row, "analysis_units": 12}])
    ctcd = pd.DataFrame({"city_normalized": ["beijing", "beijing"]})
    coordinates = pd.DataFrame(
        {"city_normalized": ["beijing"], "latitude": [39.9], "longitude": [116.4]}
    )
    result = observation_map_source(warm, cold, ctcd, coordinates)
    assert set(result.source) == {"Multinational archive", "China archive"}
    assert result.loc[result.source.eq("Multinational archive"), "observation_count"].item() == 20
    assert result.loc[result.source.eq("China archive"), "observation_count"].item() == 2


def test_observation_map_uses_active_china_records_when_flag_is_available() -> None:
    map_row = {
        "city": "beijing", "country": "china", "latitude": 39.8,
        "longitude": 116.5, "supported": True, "analysis_units": 20,
    }
    ctcd = pd.DataFrame({
        "city_normalized": ["beijing", "beijing"],
        "active_conditioning": [True, False],
    })
    coordinates = pd.DataFrame({
        "city_normalized": ["beijing"], "latitude": [39.9],
        "longitude": [116.4],
    })
    result = observation_map_source(
        pd.DataFrame([map_row]), pd.DataFrame([map_row]), ctcd, coordinates
    )
    assert result.loc[result.source.eq("China archive"), "observation_count"].item() == 1


def test_global_and_seasonal_sources_pair_controls() -> None:
    direct = pd.DataFrame([
        {
            "climate": "Beijing", "composition": "China A",
            "layout": "mixed", "permission": "zone",
            "alternative_control": control, "status": "matched",
            "delta_mean_reference_mismatch_c": -0.2,
            "delta_fixed_tail_reference_mismatch_c": -0.3,
            "delta_total_purchased_pct": 1.5,
            "delta_operational_co2_country_kg": 12.0,
        }
        for control in ("personal_center", "personal_range")
    ])
    raw, summary = global_effect_source(
        direct, _climates(), expected_climates=1, expected_compositions=1
    )
    assert raw.population_service_gain_c.eq(0.2).all()
    assert summary.metric.nunique() == 4

    seasonal = pd.DataFrame([
        {
            "climate": "Beijing", "composition": "China A",
            "layout": "mixed", "permission": "zone", "offset_c": 0.0,
            "local_season": season, "control": control,
            "total_purchased_kwh": 100.0 + 10.0 * control_index,
            "operational_co2_country_kg": 50.0 + 6.0 * control_index,
        }
        for season in ("winter", "spring", "summer", "autumn")
        for control_index, control in enumerate(
            ("uniform_average", "personal_center", "personal_range")
        )
    ])
    result = seasonal_effect_source(
        seasonal, _climates(), expected_climates=1, expected_compositions=1
    )
    centre = result[result.control.eq("personal_center")]
    assert centre.energy_change_kwh.eq(10.0).all()
    assert centre.carbon_change_kg.eq(6.0).all()


def test_physical_seasonal_commands_pair_with_each_annual_control() -> None:
    controls = ("uniform_average", "personal_center", "personal_range")
    commands = pd.DataFrame([
        {
            "climate": "Beijing", "cohort_label": "China A", "control": control,
            "seasonal_contrast_fraction": fraction,
            "mean_reference_mismatch_c": 0.8,
            "fixed_tail_reference_mismatch_c": 1.2,
            "modeled_interval_person_hour_coverage": 0.7,
            "total_purchased_kwh": 90.0,
        }
        for control in controls for fraction in (0.5, 1.0)
    ])
    baseline = pd.DataFrame([
        {
            "climate": "Beijing", "composition": "China A", "control": control,
            "offset_c": 0.0, "mean_reference_mismatch_c": 1.0,
            "fixed_tail_reference_mismatch_c": 1.5,
            "modeled_interval_person_hour_coverage": 0.6,
            "total_purchased_kwh": 100.0,
        }
        for control in controls
    ])
    result = seasonal_command_effect_source(
        commands,
        baseline,
        _climates(),
        expected_climates=1,
        expected_compositions=1,
    )
    assert np.allclose(result.change_mean_reference_mismatch_c, -0.2)
    assert result.change_total_purchased_kwh.eq(-10.0).all()
    assert result.energy_change_pct.eq(-10.0).all()


def test_event_source_adds_readable_labels() -> None:
    events = pd.DataFrame(
        {
            "run_id": ["run-1"],
            "event_id": ["event-1"],
            "climate": ["Beijing"],
            "composition_id": ["China A"],
            "cohort_label": ["China A"],
            "control": ["personal_center"],
            "action_id": ["uniform_0.5"],
            "action_family": ["uniform_shift"],
            "season": ["summer"],
            "event_class": ["high"],
            "event_service_limit_factor": [1.0],
            "service_qualified": [True],
            "positive_event_response": [True],
            "positive_full_window_response": [True],
            "qualified_positive_response": [True],
            "qualified_positive_full_window_response": [True],
            "event_mean_power_reduction_kw": [12.0],
            "event_electricity_reduction_kwh": [24.0],
            "recovery_electricity_change_kwh": [8.0],
            "event_plus_recovery_net_reduction_kwh": [16.0],
            "precondition_event_recovery_net_reduction_kwh": [15.0],
        }
    )
    result = event_effect_source(
        events,
        _climates(),
        expected_climates=1,
        expected_runs=1,
        expected_events_per_run=1,
        expected_service_factors=1,
    )
    assert result.control_label.item() == "Personal centre"
    assert result.demand_source.item() == "China archive"
    assert result.action_label.item() == "Uniform shift"
    assert result.event_class_label.item() == "High load"
