from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from analysis.global_extension.final_story_summary import (
    information_heating_association,
    seasonal_reallocation,
    summarise_events,
    summarise_information,
    summarise_matched_service,
    summarise_spatial,
    validate_annual_design,
    validate_complete_seasonal_energy,
)


def _annual_design(rows: int = 810) -> pd.DataFrame:
    data = []
    for climate in range(9):
        for composition in range(6):
            for control in ("uniform_average", "personal_center", "personal_range"):
                for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    data.append(
                        {
                            "climate": f"Climate {climate}",
                            "composition": f"Composition {composition}",
                            "layout": "random_mixing",
                            "permission": "native_zone",
                            "control": control,
                            "offset_c": offset,
                            "run_id": f"{climate}-{composition}-{control}-{offset}",
                            "facility_purchased_kwh": 1_000.0,
                        }
                    )
    return pd.DataFrame(data).iloc[:rows].copy()


def test_annual_gate_accepts_only_complete_nine_climate_library():
    audit = validate_annual_design(_annual_design())
    assert audit["candidates"] == 810
    assert audit["climates"] == 9

    with pytest.raises(ValueError, match="Expected 810 annual candidates"):
        validate_annual_design(_annual_design(378))


def test_information_headline_excludes_full_distribution_unresolved_cells():
    rows = []
    for fraction in (1.0, 0.75, 0.5):
        for climate in range(9):
            for composition in range(6):
                index = climate * 6 + composition
                rows.append(
                    {
                        "climate": f"Climate {climate}",
                        "composition": f"Composition {composition}",
                        "service_fraction": fraction,
                        "full_unresolved": fraction == 1.0 and index < 4,
                        "zone_mean_unresolved": False,
                        "eligibility_disagreement": index < 10,
                        "selection_disagreement": index < 5,
                        "zone_mean_selected_plan_fails_full_review": index < 3,
                        "zone_mean_false_reject_no_selection": False,
                    }
                )
    summary = summarise_information(pd.DataFrame(rows))
    primary = summary["primary_full_distribution_resolvable"]
    assert primary["cells"] == 50
    assert primary["zone_mean_selected_plan_failed_full_review"] == 0
    assert summary["primary_all_cells"]["cells"] == 54


def test_information_heating_context_reports_association_without_thresholding():
    information = pd.DataFrame([
        {
            "climate": climate,
            "composition": composition,
            "service_fraction": 1.0,
            "zone_mean_selected_plan_fails_full_review": failed,
        }
        for climate, composition, failed in (
            ("Cold", "A", True),
            ("Cold", "B", True),
            ("Warm", "A", False),
            ("Warm", "B", False),
        )
    ])
    metrics = pd.DataFrame([
        {
            "climate": climate,
            "composition": composition,
            "control": "uniform_average",
            "offset_c": 0.0,
            "heating_share": share,
        }
        for climate, composition, share in (
            ("Cold", "A", 0.8),
            ("Cold", "B", 0.9),
            ("Warm", "A", 0.1),
            ("Warm", "B", 0.2),
        )
    ])
    summary, enriched = information_heating_association(
        information, metrics, expected_cells=4
    )

    assert len(enriched) == 4
    assert summary["failed_selection_heating_share"]["n"] == 2
    assert np.isclose(summary["failed_selection_heating_share"]["mean"], 0.85)
    assert summary["failed_selection_cells_by_climate"] == {"Cold": 2, "Warm": 0}


def _seasonal_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_rows = []
    season_rows = []
    for climate in range(9):
        for composition in range(6):
            for fraction in (0.5, 1.0):
                for control_index, control in enumerate(
                    ("uniform_average", "personal_center", "personal_range")
                ):
                    annual_rows.append(
                        {
                            "run_id": f"annual-{climate}-{composition}-{fraction}-{control}",
                            "climate": f"Climate {climate}",
                            "composition_id": composition,
                            "seasonal_contrast_fraction": fraction,
                            "cohort_label": f"Composition {composition}",
                            "demand_source": "Archive",
                            "control": control,
                            "facility_purchased_kwh": 1_000.0,
                        }
                    )
                    for season, value in zip(
                        ("winter", "spring", "summer", "autumn"),
                        (10.0, -10.0, 20.0, -20.0),
                        strict=True,
                    ):
                        season_rows.append(
                            {
                                "run_id": f"annual-{climate}-{composition}-{fraction}-{control}",
                                "climate": f"Climate {climate}",
                                "composition_id": composition,
                                "seasonal_contrast_fraction": fraction,
                                "control": control,
                                "local_season": season,
                                "occupied_native_person_hours": 100.0,
                                "mean_reference_mismatch_c": (
                                    1.0 - 0.1 * control_index
                                ),
                                "fixed_tail_reference_mismatch_c": (
                                    2.0 - 0.2 * control_index
                                ),
                                "facility_purchased_kwh": (
                                    250.0 + control_index * value
                                ),
                            }
                        )
    return pd.DataFrame(annual_rows), pd.DataFrame(season_rows)


def test_seasonal_reallocation_reports_mixed_signs_and_cancellation():
    annual, seasons = _seasonal_frames()
    result = seasonal_reallocation(annual, seasons)

    assert len(result) == 216
    assert result.mixed_seasonal_energy_signs.all()
    assert np.allclose(result.seasonal_energy_cancellation_fraction, 1.0)
    assert result.delta_annual_mean_reference_mismatch_c.lt(0).all()
    assert result.delta_annual_fixed_tail_reference_mismatch_c.lt(0).all()


def test_complete_seasonal_energy_is_bound_to_source_and_output_hashes(tmp_path):
    source = tmp_path / "SEASONAL_COMMAND_SEASON_METRICS.csv"
    annual = tmp_path / "SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv"
    seasons = tmp_path / "SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv"
    for path, content in (
        (source, "run_id,value\nsource,1\n"),
        (annual, "run_id,value\nannual,2\n"),
        (seasons, "run_id,value\nseason,3\n"),
    ):
        path.write_text(content, encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    audit = {
        "status": "PASS",
        "source_seasonal_metrics_sha256": digest(source),
        "outputs": {
            annual.name: {"sha256": digest(annual)},
            seasons.name: {"sha256": digest(seasons)},
        },
    }
    validate_complete_seasonal_energy(
        source_seasonal_path=source,
        complete_annual_path=annual,
        complete_seasonal_path=seasons,
        audit=audit,
    )

    seasons.write_text("run_id,value\nseason,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_complete_seasonal_energy(
            source_seasonal_path=source,
            complete_annual_path=annual,
            complete_seasonal_path=seasons,
            audit=audit,
        )


def test_matched_service_requires_complete_facility_energy_and_nine_climates():
    rows = []
    for climate in range(9):
        for composition in range(6):
            for control in ("personal_center", "personal_range"):
                row = {
                    "climate": f"Climate {climate}",
                    "composition": f"Composition {composition}",
                    "alternative_control": control,
                    "energy_scope": "complete_facility_purchased_energy",
                    "delta_energy_pct": 1.0,
                }
                for tolerance in (0.01, 0.025, 0.05, 0.075, 0.10):
                    row[
                        f"within_tolerance_{tolerance:.3f}_c".replace(".", "p")
                    ] = tolerance >= 0.05
                rows.append(row)
    summary = summarise_matched_service(pd.DataFrame(rows))

    assert summary["comparison_rows"] == 108
    assert summary["service_gap_tolerance_c"]["0.05"]["matched_rows"] == 108


def test_event_summary_separates_raw_event_and_full_window_response():
    seasons = ("winter", "spring", "summer", "autumn")
    protocol = pd.DataFrame([
        {
            "event_id": f"event-{index}",
            "climate": "Climate A",
            "local_season": seasons[index // 2],
            "load_level": "high" if index % 2 == 0 else "typical",
        }
        for index in range(8)
    ])
    rows = []
    for run_index in range(2):
        for evaluation_index in range(8):
            for factor in (1.00, 1.05, 1.10, 1.15, 1.20):
                rows.append(
                    {
                        "run_id": f"run-{run_index}",
                        "event_id": f"event-{evaluation_index}",
                        "climate": "Climate A",
                        "composition_id": 1,
                        "cohort_label": "China A",
                        "control": "uniform_average",
                        "action_id": "uniform_0.5",
                        "action_family": "uniform_shift",
                        "local_season": seasons[evaluation_index // 2],
                        "load_level": (
                            "high" if evaluation_index % 2 == 0 else "typical"
                        ),
                        "event_service_limit_factor": factor,
                        "service_qualified": factor >= 1.05,
                        "positive_event_response": True,
                        "positive_full_window_response": True,
                        "qualified_positive_event_response": factor >= 1.05,
                        "qualified_positive_full_window_response": factor >= 1.10,
                        "event_mean_power_reduction_kw": 10.0,
                        "event_electricity_reduction_kwh": 20.0,
                        "recovery_electricity_change_kwh": 5.0,
                        "precondition_event_recovery_net_reduction_kwh": 15.0,
                        "event_facility_mean_power_reduction_kw": 10.0,
                        "event_facility_electricity_reduction_kwh": 20.0,
                        "recovery_facility_electricity_change_kwh": 5.0,
                        "precondition_event_recovery_facility_net_reduction_kwh": 15.0,
                    }
                )
    summary, tidy = summarise_events(
        pd.DataFrame(rows),
        protocol,
        expected_events=8,
        expected_runs=2,
        expected_climates=1,
        expected_actions=1,
        expected_controls=("uniform_average",),
    )

    assert summary["scored_rows"] == 80
    assert summary["service_allowance_sensitivity"]["1"][
        "raw_positive_event_rows"
    ] == 16
    assert summary["service_allowance_sensitivity"]["1"][
        "service_qualified_positive_full_window_rows"
    ] == 0
    assert summary["service_allowance_sensitivity"]["1.05"][
        "event_qualified_power_reduction_kw"
    ]["n"] == 16
    assert summary["service_allowance_sensitivity"]["1.05"][
        "full_window_qualified_event_power_reduction_kw"
    ]["n"] == 0
    assert summary["service_allowance_sensitivity"]["1.1"][
        "service_qualified_positive_full_window_rows"
    ] == 16
    assert summary["service_allowance_sensitivity"]["1.1"][
        "full_window_qualified_net_reduction_kwh"
    ]["n"] == 16
    assert len(tidy) == 40


def test_spatial_summary_accepts_three_ordered_mapping_design():
    effects = pd.DataFrame(
        [
            {
                "effect_type": "placement",
                "reference": "random_mixing",
                "alternative": "demand_ordered_reversed",
                "control": "personal_center",
                "status": "matched",
                "delta_mean_reference_mismatch_c": -0.1,
                "delta_fixed_tail_reference_mismatch_c": -0.2,
                "delta_facility_purchased_kwh": 10.0,
                "delta_facility_purchased_pct": 1.0,
            }
        ]
    )
    audit = {
        "logical_new_designs": 540,
        "distinct_new_physical_runs": 516,
        "logical_metric_rows": 972,
    }

    summary = summarise_spatial(effects, audit)

    assert summary["new_logical_designs"] == 540
    assert summary["new_physical_runs"] == 516
    assert summary["scored_configuration_rows"] == 972
