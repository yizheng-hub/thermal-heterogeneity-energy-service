from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.global_extension.implementation_value import (
    analyse,
    build_information_summary,
    build_resource_catalog,
    build_scope_requirements,
    build_spatial_resource_boundaries,
    build_value_boundaries,
)


def _direct() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "climate": "Climate A",
            "composition": "Multinational A",
            "alternative_control": "personal_center",
            "status": "matched",
            "delta_mean_reference_mismatch_c": -0.4,
            "delta_fixed_tail_reference_mismatch_c": -0.1,
            "delta_modeled_interval_mean_excess_c": -0.2,
            "delta_modeled_interval_person_hour_coverage": 0.05,
            "delta_facility_electric_kwh": 100.0,
            "delta_facility_gas_hhv_kwh": -20.0,
            "delta_facility_purchased_kwh": 80.0,
            "delta_operational_co2_country_kg": 10.0,
        }
    ])


def test_value_boundary_keeps_prices_and_benefits_unassigned():
    direct = _direct().assign(
        target_service_gain_c=0.4,
        fixed_tail_service_gain_c=0.1,
        interval_coverage_gain_percentage_points=5.0,
        control_label="Personal centre",
    )
    scenarios = pd.DataFrame([
        {
            "scenario_id": "one",
            "kind": "named_package",
            "wage_profile": "median",
            "annual_resource_cost_usd": 500.0,
            "scope": "illustrative United States dollar sensitivity",
        }
    ])
    row = build_value_boundaries(direct, scenarios).iloc[0]
    assert row.electricity_price_coefficient_kwh_year == 100.0
    assert row.gas_price_coefficient_kwh_hhv_year == -20.0
    assert row.minimum_external_value_before_local_energy_prices_usd_year == 500.0
    assert row.required_combined_value_if_energy_charge_change_is_zero_usd_year == 500.0
    assert row.required_combined_value_with_local_energy_prices == "not assigned"
    assert not row.local_energy_prices_assigned
    assert not row.annual_dr_revenue_assigned


def test_scope_requirements_match_fixed_medium_office():
    scopes = build_scope_requirements().set_index("command_scope")
    assert scopes.loc["building", "independent_command_groups"] == 1
    assert scopes.loc["floor", "independent_command_groups"] == 3
    assert scopes.loc["native_zone", "independent_command_groups"] == 15
    assert not scopes.per_group_setup_hours_assigned.any()


def test_information_summary_uses_global_double_holdout_result():
    summary = {
        "status": "PASS",
        "climates": 9,
        "demand_compositions": 6,
        "protection_split": "exclude held-out climate and composition",
        "protection_quantile": 0.95,
        "primary_full_resolvable_scenarios": 40,
        "headline_scope": "service fraction 1.0 and full-resolvable scenarios",
        "privacy_interpretation": "anonymous aggregate, not a formal guarantee",
        "primary": {
            "protected five bins": {
                "selections": 36,
                "decision_coverage": 0.9,
                "qualified_action_retention": 0.8,
                "selected_plan_false_acceptances": 0,
                "safe_energy_penalty_median_percent": 0.4,
                "safe_energy_penalty_p95_percent": 1.2,
            }
        },
    }
    row = build_information_summary(summary).iloc[0]
    assert row.climates == 9
    assert row.observed_selected_plan_false_acceptance == 0
    assert row.qualified_action_retention == 0.8
    assert not row.formal_privacy_guarantee


def test_resource_catalog_requires_authoritative_source_records():
    assumptions = {
        "sources": [
            {"reference_id": "doe_controls2024"},
            {"reference_id": "iea_ebc_annex79_2024"},
        ]
    }
    catalog = build_resource_catalog(assumptions).set_index("resource_category")
    assert set(catalog)  # non-empty columns after indexing
    assert not catalog.loc["optional_local_hardware", "required_in_every_deployment"]
    assert not catalog.global_monetary_value_assigned.any()


def test_spatial_resource_boundaries_keep_costs_unassigned():
    permission = pd.DataFrame([{
        "climate": "Climate A",
        "composition": "Multinational A",
        "control": "personal_center",
        "status": "matched",
        "axis": "permission",
        "layout": "random_mixing",
        "reference": "building",
        "alternative": "native_zone",
        "delta_facility_purchased_kwh": 12.0,
    }])
    placement = pd.DataFrame([{
        "climate": "Climate A",
        "composition": "Multinational A",
        "control": "personal_center",
        "status": "matched",
        "axis": "layout",
        "permission": "native_zone",
        "reference": "random_mixing",
        "alternative": "demand_ordered",
        "delta_facility_purchased_kwh": -4.0,
    }])
    result = build_spatial_resource_boundaries(permission, placement)
    command = result[result.resource_category.eq("command_scope")].iloc[0]
    staff = result[result.resource_category.eq("staff_reconfiguration")].iloc[0]
    assert command.incremental_resource_quantity == 14
    assert command.resource_quantity_assigned
    assert not staff.resource_quantity_assigned
    assert not result.annual_implementation_cost_assigned.any()


def test_end_to_end_writes_audited_outputs(tmp_path: Path):
    direct = tmp_path / "direct.csv"
    assumptions = tmp_path / "assumptions.json"
    information = tmp_path / "information.json"
    permission = tmp_path / "permission.csv"
    placement = tmp_path / "placement.csv"
    _direct().to_csv(direct, index=False)
    assumptions.write_text(json.dumps({
        "monetary_scope": "illustrative United States dollar sensitivity",
        "discount_rate": 0.03,
        "lifetime_years": 5,
        "grid": {
            "setup_hours": [0],
            "annual_maintenance_hours": [0],
            "feedback": [[0, 0]],
            "annual_software_usd": [0],
            "added_interface_capital_usd": [0],
        },
        "named_packages": [],
        "wage_profiles": {
            "median": {
                "facility_usd_per_hour": 50,
                "respondent_usd_per_hour": 25,
            }
        },
        "sources": [
            {"reference_id": "bls_facilities2025"},
            {"reference_id": "doe_controls2024"},
            {"reference_id": "iea_ebc_annex79_2024"},
        ],
    }), encoding="utf-8")
    information.write_text(json.dumps({
        "status": "PASS",
        "climates": 9,
        "demand_compositions": 6,
        "protection_split": "exclude held-out climate and composition",
        "protection_quantile": 0.95,
        "primary_full_resolvable_scenarios": 8,
        "headline_scope": "service fraction 1.0 and full-resolvable scenarios",
        "privacy_interpretation": "anonymous aggregate, not a formal guarantee",
        "primary": {
            "protected five bins": {
                "selections": 7,
                "decision_coverage": 0.875,
                "qualified_action_retention": 0.75,
                "selected_plan_false_acceptances": 0,
                "safe_energy_penalty_median_percent": 0.1,
                "safe_energy_penalty_p95_percent": 0.5,
            }
        },
    }), encoding="utf-8")
    pd.DataFrame([{
        "climate": "Climate A",
        "composition": "Multinational A",
        "control": "personal_center",
        "status": "matched",
        "axis": "permission",
        "layout": "random_mixing",
        "reference": "building",
        "alternative": "floor",
    }]).to_csv(permission, index=False)
    pd.DataFrame([{
        "climate": "Climate A",
        "composition": "Multinational A",
        "control": "personal_center",
        "status": "matched",
        "axis": "layout",
        "permission": "native_zone",
        "reference": "random_mixing",
        "alternative": "demand_ordered",
    }]).to_csv(placement, index=False)
    output = tmp_path / "out"
    audit = analyse(
        direct, assumptions, information, output, permission, placement
    )
    assert audit["status"] == "PASS"
    assert audit["direct_comparisons"] == 1
    assert audit["value_boundary_rows"] == 1
    assert audit["spatial_resource_rows"] == 2
    assert (output / "GLOBAL_SERVICE_RESOURCE_VALUE_BOUNDARIES.csv").is_file()
    assert (output / "GLOBAL_SPATIAL_IMPLEMENTATION_BOUNDARIES.csv").is_file()
