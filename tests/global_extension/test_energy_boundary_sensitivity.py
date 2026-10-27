from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.global_extension.energy_boundary_sensitivity import (
    analyse_energy_boundary_sensitivity,
    boundary_service_selections,
    direct_boundary_comparisons,
    prepare_metrics,
    seasonal_boundary_comparisons,
    selected_control_boundary_comparisons,
)


OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _metrics() -> pd.DataFrame:
    rows = []
    for control_index, control in enumerate(
        ("uniform_average", "personal_center", "personal_range")
    ):
        for offset in OFFSETS:
            partial = 980.0 + 10.0 * (offset + 1.0) ** 2 + 3.0 * control_index
            residual = 30.0 if offset == 0.0 else 80.0
            if offset == 0.0 and control == "uniform_average":
                partial, residual = 1_000.0, 50.0
            elif offset == 0.0 and control != "uniform_average":
                partial, residual = 990.0 + control_index, 70.0
            rows.append(
                {
                    "climate": "Climate A",
                    "composition": "Composition 1",
                    "layout": "random_mixing",
                    "permission": "building",
                    "control": control,
                    "offset_c": offset,
                    "run_id": f"{control}-{offset:+.1f}",
                    "engineering_pass": True,
                    "mean_reference_mismatch_c": 1.0,
                    "fixed_tail_reference_mismatch_c": 2.0,
                    "facility_electric_kwh": partial - 100.0,
                    "gas_hhv_kwh": 100.0,
                    "facility_gas_hhv_kwh": 100.0 + residual,
                    "facility_electric_plus_space_heating_gas_kwh": partial,
                    "facility_purchased_kwh": partial + residual,
                    "facility_electric_plus_space_heating_gas_operational_co2_country_kg": partial
                    / 2,
                    "operational_co2_country_kg": (partial + residual) / 2,
                }
            )
    return pd.DataFrame(rows)


def _anchors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "climate": "Climate A",
                "composition": "Composition 1",
                "layout": "random_mixing",
                "anchor_mean_mismatch_c": 2.0,
                "anchor_tail_mismatch_c": 4.0,
            }
        ]
    )


def _seasonal_metrics() -> pd.DataFrame:
    data = _metrics()
    data = data[data.offset_c.eq(0.0)].copy()
    data["composition_id"] = 1
    data["cohort_label"] = "Multinational A"
    data["seasonal_contrast_fraction"] = 1.0
    return data


def test_direct_comparison_can_expose_a_boundary_direction_change():
    direct = direct_boundary_comparisons(prepare_metrics(_metrics()))

    assert len(direct) == 2
    assert direct.energy_direction_changed.all()
    assert set(direct.direction_complete_facility) == {"increase"}
    assert set(direct.direction_electricity_plus_space_heating_gas) == {"decrease"}


def test_common_service_selection_uses_the_same_qualified_candidates():
    selections = boundary_service_selections(
        prepare_metrics(_metrics()), _anchors(), fractions=(1.0,)
    )

    assert len(selections) == 3
    assert selections.qualified_candidate_count.eq(5).all()
    assert selections.selection_changed.all()
    assert selections.complete_facility_selected_offset_c.eq(0.0).all()
    assert selections.comparator_selected_offset_c.eq(-1.0).all()
    assert selections.complete_facility_penalty_of_comparator_choice_kwh.gt(0).all()
    assert selections.comparator_penalty_of_complete_choice_kwh.gt(0).all()


def test_selected_control_signs_are_reported_for_both_boundaries():
    selections = boundary_service_selections(
        prepare_metrics(_metrics()), _anchors(), fractions=(1.0,)
    )
    comparisons = selected_control_boundary_comparisons(selections)

    assert len(comparisons) == 2
    assert comparisons.status.eq("matched").all()
    assert set(comparisons.direction_complete_facility) == {"increase"}
    assert set(comparisons.direction_electricity_plus_space_heating_gas) == {"increase"}


def test_selection_changes_are_flagged_when_cross_boundary_penalties_are_near_ties():
    metrics = _metrics()
    for index, row in metrics.iterrows():
        if row.offset_c == -1.0:
            partial, residual = 1_000.0, 50.0
        elif row.offset_c == 0.0:
            partial, residual = 1_000.001, 49.998
        else:
            partial, residual = 1_100.0 + abs(row.offset_c), 50.0
        metrics.loc[index, "gas_hhv_kwh"] = 100.0
        metrics.loc[index, "facility_gas_hhv_kwh"] = 100.0 + residual
        metrics.loc[index, "facility_electric_plus_space_heating_gas_kwh"] = partial
        metrics.loc[index, "facility_purchased_kwh"] = partial + residual

    selections = boundary_service_selections(prepare_metrics(metrics), _anchors(), fractions=(1.0,))

    assert selections.selection_changed.all()
    assert selections.changed_within_0p001_pct.all()


def test_seasonal_direct_comparison_uses_both_boundaries():
    seasonal = seasonal_boundary_comparisons(_seasonal_metrics())

    assert len(seasonal) == 2
    assert seasonal.energy_direction_changed.all()
    assert set(seasonal.direction_complete_facility) == {"increase"}
    assert set(seasonal.direction_electricity_plus_space_heating_gas) == {"decrease"}


def test_command_writes_hashed_audit_outputs(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    anchors = tmp_path / "anchors.csv"
    output = tmp_path / "boundary"
    _metrics().to_csv(metrics, index=False)
    _anchors().to_csv(anchors, index=False)
    seasonal = tmp_path / "seasonal.csv"
    _seasonal_metrics().to_csv(seasonal, index=False)

    audit = analyse_energy_boundary_sensitivity(
        metrics,
        anchors,
        output,
        expected_rows=15,
        expected_climates=1,
        seasonal_metrics_path=seasonal,
    )

    assert audit["status"] == "PASS"
    assert audit["counts"]["direct_pairs"] == 2
    assert audit["direct_energy_directions"]["direction_changes"] == 2
    assert audit["selection_sensitivity"]["changes"] == 9
    assert audit["counts"]["selected_control_pairs"] == 6
    assert audit["service_selected_energy_directions"]["direction_changes"] == 0
    assert audit["seasonal_direct_energy_directions"]["pairs"] == 2
    assert len(list(output.glob("*.csv"))) == 5
    assert all(len(metadata["sha256"]) == 64 for metadata in audit["outputs"].values())
    assert (output / "GLOBAL_ENERGY_BOUNDARY_SENSITIVITY.json").is_file()


def test_final_defaults_reject_a_non_global_library(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    anchors = tmp_path / "anchors.csv"
    _metrics().to_csv(metrics, index=False)
    _anchors().to_csv(anchors, index=False)

    try:
        analyse_energy_boundary_sensitivity(metrics, anchors, tmp_path / "boundary")
    except ValueError as error:
        assert "Expected 810 annual candidates" in str(error)
    else:
        raise AssertionError("The final boundary audit accepted a smoke-test library")
