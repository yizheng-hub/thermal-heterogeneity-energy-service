"""Analytical signed bill and management-resource checks, without private data."""

import pytest

from run_costs import command_scope_boundaries, price_pairs


def pair(**updates):
    return {"comparison_id": "synthetic", "climate_context": "test", "scope": "native_zone",
            "control": "personal_center", "composition_id": 0, "layout": "test",
            "comparison_mode": "direct", "service_fraction": -1, "status": "matched",
            "delta_electric_kwh": -100, "delta_gas_hhv_kwh": 10} | updates


def factor(currency="USD"):
    return {"city": "test", "currency": currency, "electricity_co2_kg_kwh": .5,
            "gas_co2_kg_kwh_hhv": .2, "electricity_price_currency_kwh": .2,
            "gas_price_currency_kwh_hhv": .1}


def scenario():
    return {"scenario_id": "analytical", "wage_profile": "test", "annuity_factor": .2,
            "annual_operator_cost_usd": 5, "annual_resource_cost_usd": 7}


def test_signed_bill_and_carbon_are_independent_of_site_energy():
    priced, boundary = price_pairs([pair()], [factor()], [scenario()])
    assert boundary[0]["delta_operational_co2_kg"] == -48
    assert boundary[0]["delta_energy_charge_usd_year"] == -19
    assert priced[0]["delta_operator_total_cost_usd_year"] == -14
    assert priced[0]["delta_resource_total_cost_usd_year"] == -12


def test_bill_increase_is_retained():
    _, boundary = price_pairs([pair(delta_electric_kwh=100)], [factor()], [scenario()])
    assert boundary[0]["signed_additional_implementation_allowance_usd_year"] == -21
    assert not boundary[0]["nonnegative_energy_funded_allowance_exists"]


def test_unmatched_cost_is_not_fabricated():
    priced, boundary = price_pairs([pair(status="unmatched")], [factor()], [scenario()])
    assert len(priced) == 1 and not boundary
    assert priced[0]["monetary_status"] == "not_computed"


def test_india_preserves_gas_and_implementation_price_boundary():
    _, boundary = price_pairs([pair()], [factor("INR")], [scenario()])
    assert boundary[0]["delta_bill_intercept_INR_year_pf095"] == -100 * 8.5 / .95
    assert boundary[0]["delta_bill_gas_slope_kwh_hhv"] == 10
    assert not boundary[0]["US_wage_applied"]


def test_command_hours_scale_with_extra_groups_not_physical_ports():
    groups = [{"context": "test", "zone": f"z{i}", "floor": "f"} for i in range(3)]
    cfg = {"wage_profiles": {"test": {"facility_usd_per_hour": 10}}}
    rows = command_scope_boundaries([pair()], [factor()], [scenario()], cfg, groups)
    row = rows[0]
    assert row["declared_independent_command_groups"] == 3
    assert row["existing_physical_hvac_setpoint_ports"] == 6
    assert row["scope_annual_hour_cost_slope_usd_per_group_hour"] == 20
    assert row["scope_setup_hour_cost_slope_usd_per_group_hour"] == 4
    assert row["signed_resource_manager_hour_allowance_per_extra_group_year"] == .6


def test_building_permission_has_no_division_by_zero_hour_threshold():
    groups = [{"context": "test", "zone": "z", "floor": "f"}]
    cfg = {"wage_profiles": {"test": {"facility_usd_per_hour": 10}}}
    row = command_scope_boundaries([pair(scope="building")], [factor()], [scenario()], cfg, groups)[0]
    assert row["extra_groups_over_uniform"] == 0
    assert row["signed_resource_manager_hour_allowance_per_extra_group_year"] is None


def test_direct_and_common_service_modes_cannot_be_pooled():
    with pytest.raises(ValueError, match="direct or equal_service"):
        price_pairs([pair(comparison_mode="pooled")], [factor()], [scenario()])
