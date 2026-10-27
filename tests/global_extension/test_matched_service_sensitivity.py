from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.global_extension.matched_service_sensitivity import (
    analyse_matched_service,
    detect_equal_depth_climates,
    enumerate_service_pairs,
    mean_only_diagnostic,
    mean_only_summary,
    prepare_metrics,
    qualifying_pairs,
    resolve_energy_column,
    select_closest_pairs,
    select_two_metric_closest_pairs,
    tolerance_summary,
    two_metric_qualifying_pairs,
    validate_primary_library,
)


OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _metrics(include_facility: bool = True) -> pd.DataFrame:
    rows = []
    for climate in ("Sentinel A", "Sentinel B", "Sparse"):
        offsets = OFFSETS if climate.startswith("Sentinel") else (0.0,)
        for control in ("uniform_average", "personal_center", "personal_range"):
            for offset in offsets:
                adjustment = {
                    "uniform_average": (0.0, 0.0, 0.0),
                    "personal_center": (0.02, 0.03, 10.0),
                    "personal_range": (-0.02, 0.04, 20.0),
                }[control]
                excess_adjustment = {
                    "uniform_average": 0.0,
                    "personal_center": 0.04,
                    "personal_range": 0.08,
                }[control]
                row = {
                    "climate": climate,
                    "composition": "Composition 1",
                    "layout": "random mixing",
                    "permission": "zone",
                    "control": control,
                    "offset_c": offset,
                    "run_id": f"{climate}-{control}-{offset:+.1f}",
                    "engineering_pass": True,
                    "mean_reference_mismatch_c": 1.0 + offset + adjustment[0],
                    "fixed_tail_reference_mismatch_c": 2.0 + offset + adjustment[1],
                    "modeled_interval_mean_excess_c": (1.5 + offset + excess_adjustment),
                    "total_purchased_kwh": 1_000.0 + 50.0 * offset + adjustment[2],
                }
                if include_facility:
                    row["facility_electric_plus_space_heating_gas_kwh"] = (
                        5_000.0 + 50.0 * offset + adjustment[2]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _primary(frame: pd.DataFrame, energy_column: str) -> pd.DataFrame:
    prepared = prepare_metrics(frame)
    climates = detect_equal_depth_climates(prepared)
    return validate_primary_library(prepared, climates, energy_column, 2)


def test_primary_scope_excludes_unequal_candidate_depth():
    prepared = prepare_metrics(_metrics())
    assert detect_equal_depth_climates(prepared) == ["Sentinel A", "Sentinel B"]
    energy_column, scope = resolve_energy_column(prepared)
    assert energy_column == "facility_electric_plus_space_heating_gas_kwh"
    assert scope == "legacy_facility_electricity_plus_space_heating_gas"
    primary = validate_primary_library(prepared, ["Sentinel A", "Sentinel B"], energy_column, 2)
    assert len(primary) == 30
    assert set(primary.climate) == {"Sentinel A", "Sentinel B"}


def test_five_by_five_pairs_and_predeclared_tolerances():
    energy_column, scope = resolve_energy_column(_metrics())
    pairs = enumerate_service_pairs(_primary(_metrics(), energy_column), energy_column, scope)
    closest = select_closest_pairs(pairs)
    all_qualifying = qualifying_pairs(pairs)
    summary = tolerance_summary(closest, pairs)

    assert len(pairs) == 100
    assert len(closest) == 4
    assert len(all_qualifying) == 20
    overall_005 = summary[
        summary.alternative_control.eq("all") & summary.tolerance_c.eq(0.05)
    ].iloc[0]
    assert overall_005.eligible_context_control_groups == 4
    assert overall_005.matched_context_control_groups == 2
    assert overall_005.all_qualifying_pairs == 10
    assert overall_005.all_qualifying_energy_increase == 10
    assert overall_005.energy_increase == 2

    two_metric_closest = select_two_metric_closest_pairs(pairs)
    two_metric_qualifying = two_metric_qualifying_pairs(pairs)
    two_metric_summary = tolerance_summary(
        two_metric_closest, pairs, distance_column="two_metric_linf_gap_c"
    )
    two_metric_005 = two_metric_summary[
        two_metric_summary.alternative_control.eq("all") & two_metric_summary.tolerance_c.eq(0.05)
    ].iloc[0]
    assert len(two_metric_qualifying) == 20
    assert two_metric_005.matched_context_control_groups == 4
    assert two_metric_005.energy_increase == 4
    assert two_metric_005.all_qualifying_energy_increase == 20
    assert not any("coverage" in column for column in pairs.columns)


def test_closest_pair_selection_is_energy_blind():
    frame = _metrics()
    energy_column, scope = resolve_energy_column(frame)
    primary = _primary(frame, energy_column)
    first = select_closest_pairs(enumerate_service_pairs(primary, energy_column, scope))

    changed = primary.copy()
    changed[energy_column] = list(range(1, len(changed) + 1))[::-1]
    second = select_closest_pairs(enumerate_service_pairs(changed, energy_column, scope))
    columns = ["uniform_run_id", "alternative_run_id"]
    assert first[columns].equals(second[columns])


def test_mean_only_diagnostic_exposes_unmatched_tail():
    frame = _metrics()
    energy_column, scope = resolve_energy_column(frame)
    pairs = enumerate_service_pairs(_primary(frame, energy_column), energy_column, scope)
    diagnostic = mean_only_diagnostic(pairs)
    summary = mean_only_summary(diagnostic)
    overall_0025 = summary[
        summary.alternative_control.eq("all") & summary.mean_tolerance_c.eq(0.025)
    ].iloc[0]
    assert overall_0025.mean_matched_context_control_groups == 4
    assert overall_0025.tail_gap_exceeds_same_tolerance == 4


def test_legacy_total_is_retained_only_when_facility_energy_is_absent():
    energy_column, scope = resolve_energy_column(_metrics(include_facility=False))
    assert energy_column == "total_purchased_kwh"
    assert scope == "legacy_conditioning_total"


def test_complete_facility_energy_precedes_partial_facility_boundary():
    frame = _metrics()
    frame["facility_purchased_kwh"] = frame["facility_electric_plus_space_heating_gas_kwh"] + 750.0

    energy_column, scope = resolve_energy_column(frame)

    assert energy_column == "facility_purchased_kwh"
    assert scope == "complete_facility_purchased_energy"


def test_command_writes_tables_and_audit(tmp_path: Path):
    metrics = tmp_path / "GLOBAL_RUN_METRICS.csv"
    _metrics().to_csv(metrics, index=False)
    output = tmp_path / "matched"
    audit = analyse_matched_service(metrics, output, expected_sentinel_climates=2)

    assert audit["status"] == "PASS"
    assert audit["protocol"] == "matched-realized-service-v2"
    assert audit["candidate_library"]["enumerated_pairs"] == 100
    assert audit["matching"]["energy_used_for_matching"] is False
    assert audit["matching"]["coverage_ratio_used_for_matching"] is False
    assert audit["energy"]["column"] == ("facility_electric_plus_space_heating_gas_kwh")
    assert audit["energy"]["legacy_fallback_used"]
    assert not audit["energy"]["complete_facility_energy_used"]
    assert audit["energy"]["label"] == "Legacy partial facility-energy fallback"
    assert "facility gas outside" in audit["energy"]["definition"]
    assert len(list(output.glob("*.csv"))) == 8
    assert (output / "MATCHED_REALIZED_SERVICE_AUDIT.json").is_file()


def test_final_default_rejects_fewer_than_nine_complete_climates(tmp_path: Path):
    metrics = tmp_path / "GLOBAL_RUN_METRICS.csv"
    _metrics().to_csv(metrics, index=False)

    with pytest.raises(ValueError, match="Expected 9 equal-depth sentinel climates"):
        analyse_matched_service(metrics, tmp_path / "matched")
