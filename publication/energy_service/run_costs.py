"""Price all signed matched Stage8 control comparisons under frozen assumptions.

The paired input must declare its comparison mode and contain actual energy
changes (alternative minus uniform). Direct-service and equal-service pairs
remain distinct. No annual demand-response revenue is invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cost_scenarios(cfg):
    rate, years = cfg["discount_rate"], cfg["lifetime_years"]
    annuity = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    grid = cfg["grid"]
    packages = []
    for index, (setup, maintenance, feedback, software, capital) in enumerate(itertools.product(
        grid["setup_hours"], grid["annual_maintenance_hours"], grid["feedback"],
        grid["annual_software_usd"], grid["added_interface_capital_usd"],
    )):
        packages.append({"name": f"grid_{index:03d}", "kind": "sensitivity_grid", "setup_hours": setup,
            "annual_maintenance_hours": maintenance, "extra_responses_year": feedback[0],
            "seconds_per_response": feedback[1], "annual_software_usd": software,
            "added_interface_capital_usd": capital})
    packages.extend(package | {"kind": "named_package"} for package in cfg["named_packages"])
    scenarios = []
    for package, (name, wage) in itertools.product(packages, cfg["wage_profiles"].items()):
        manager = wage["facility_usd_per_hour"]
        feedback_hours = package["extra_responses_year"] * package["seconds_per_response"] / 3600
        operator = annuity * (package["added_interface_capital_usd"] + manager * package["setup_hours"]) + manager * package["annual_maintenance_hours"] + package["annual_software_usd"]
        scenarios.append(package | {"scenario_id": f"{package['name']}__{name}", "wage_profile": name,
            "annuity_factor": annuity, "annual_operator_cost_usd": operator,
            "respondent_hours_year": feedback_hours,
            "annual_resource_cost_usd": operator + feedback_hours * wage["respondent_usd_per_hour"]})
    return scenarios


def price_pairs(pairs, factors, scenarios):
    factors = {row["city"]: row for row in factors}
    output, boundary = [], []
    for row in pairs:
        if row["comparison_mode"] not in {"direct", "equal_service"}:
            raise ValueError("Declare direct or equal_service instead of combining the two")
        if row["status"] != "matched":
            output.append(row | {"scenario_id": "unmatched", "monetary_status": "not_computed"})
            continue
        factor = factors[row["climate_context"]]
        de, dg = float(row["delta_electric_kwh"]), float(row["delta_gas_hhv_kwh"])
        carbon = de * float(factor["electricity_co2_kg_kwh"]) + dg * float(factor["gas_co2_kg_kwh_hhv"])
        base = row | {"currency": factor["currency"], "delta_operational_co2_kg": carbon,
            "cost_basis": "historical nominal energy-charge proxy plus explicit incremental implementation scenario"}
        if factor["currency"] == "INR":
            intercept = de * 8.50 / 0.95
            boundary.append(base | {"delta_bill_intercept_INR_year_pf095": intercept,
                "delta_bill_gas_slope_kwh_hhv": dg,
                "total_cost_boundary": "delta_bill_intercept + gas_price_INR_per_kWh_HHV * delta_gas + M_INR_year",
                "US_wage_applied": False})
            output.append(base | {"scenario_id": "INR_affine_boundary", "monetary_status": "requires_declared_India_gas_and_implementation_prices"})
            continue
        bill = de * float(factor["electricity_price_currency_kwh"]) + dg * float(factor["gas_price_currency_kwh_hhv"])
        headroom = -bill
        boundary.append(base | {"delta_energy_charge_usd_year": bill,
            "signed_additional_implementation_allowance_usd_year": headroom,
            "nonnegative_energy_funded_allowance_exists": headroom >= -1e-8})
        for scenario in scenarios:
            operator = scenario["annual_operator_cost_usd"]
            resource = scenario["annual_resource_cost_usd"]
            output.append(base | {"scenario_id": scenario["scenario_id"],
                "monetary_status": "priced_scenario", "delta_energy_charge_usd_year": bill,
                "signed_energy_charge_headroom_usd_year": headroom,
                "additional_operator_cost_usd_year": operator,
                "additional_resource_cost_usd_year": resource,
                "delta_operator_total_cost_usd_year": bill + operator,
                "delta_resource_total_cost_usd_year": bill + resource,
                "minimum_additional_annual_external_value_for_resource_breakeven_usd": max(0.0, bill + resource),
                "external_value_interpretation": "required benefit threshold, not observed comfort value or DR revenue"})
    return output, boundary


def command_scope_boundaries(pairs, factors, scenarios, cfg, zone_groups):
    """Price additional *software command-group* hours as an affine boundary."""
    factors = {row["city"]: row for row in factors}
    groups = {}
    for context in {row["context"] for row in zone_groups}:
        selected = [row for row in zone_groups if row["context"] == context]
        groups[context] = {"building": 1, "floor": len({row["floor"] for row in selected}),
                           "native_zone": len({row["zone"] for row in selected})}
    result = []
    for pair in pairs:
        context = pair["climate_context"]
        count = 1 if pair["control"] == "uniform_average" else groups[context][pair["scope"]]
        extra = count - 1
        keep = {key: pair[key] for key in ("comparison_id", "climate_context", "scope", "control", "composition_id", "layout", "comparison_mode", "status", "service_fraction") if key in pair}
        for key in ("delta_mean_reference_mismatch_c", "delta_fixed_tail_reference_mismatch_c", "delta_interval_coverage_percentage_points"):
            if key in pair:
                keep[key] = pair[key]
        base = keep | {"declared_independent_command_groups": count, "extra_groups_over_uniform": extra,
            "existing_physical_hvac_setpoint_ports": 2 * groups[context]["native_zone"],
            "independent_command_channels": 2 * count,
            "resource_units": "annualized manager hours per extra command group per year"}
        if pair["status"] != "matched":
            result.append(base | {"scenario_id": "unmatched", "scope_cost_status": "no_paired_energy_bill"})
            continue
        factor = factors[context]
        de, dg = float(pair["delta_electric_kwh"]), float(pair["delta_gas_hhv_kwh"])
        if factor["currency"] == "INR":
            result.append(base | {"scenario_id": "INR_affine_boundary", "scope_cost_status": "unknown_local_wage_and_fees",
                "delta_bill_intercept_INR_year_pf095": de * 8.5 / .95, "delta_bill_gas_slope_kwh_hhv": dg,
                "scope_cost_equation": "delta_bill_intercept + delta_gas * gas_price_INR + M_fixed_INR + extra_groups * w_manager_INR_hour * (CRF*h_setup_group + h_annual_group)",
                "US_wage_applied": False})
            continue
        bill = de * float(factor["electricity_price_currency_kwh"]) + dg * float(factor["gas_price_currency_kwh_hhv"])
        for scenario in scenarios:
            wage = cfg["wage_profiles"][scenario["wage_profile"]]["facility_usd_per_hour"]
            operator = -bill - scenario["annual_operator_cost_usd"]
            resource = -bill - scenario["annual_resource_cost_usd"]
            result.append(base | {"scenario_id": scenario["scenario_id"], "scope_cost_status": "affine_USD_boundary",
                "manager_wage_usd_hour": wage, "annuity_factor": scenario["annuity_factor"],
                "delta_energy_bill_usd_year": bill, "fixed_operator_package_usd_year": scenario["annual_operator_cost_usd"],
                "fixed_resource_package_usd_year": scenario["annual_resource_cost_usd"],
                "signed_remaining_operator_budget_usd_year": operator, "signed_remaining_resource_budget_usd_year": resource,
                "scope_annual_hour_cost_slope_usd_per_group_hour": extra * wage,
                "scope_setup_hour_cost_slope_usd_per_group_hour": extra * wage * scenario["annuity_factor"],
                "signed_operator_manager_hour_allowance_per_extra_group_year": operator / (extra * wage) if extra else None,
                "signed_resource_manager_hour_allowance_per_extra_group_year": resource / (extra * wage) if extra else None,
                "nonnegative_scope_resource_allowance_exists": resource >= -1e-8 if extra else None,
                "zero_extra_group_rule": "whole-building fixed-fee boundary only" if not extra else "per-group threshold applies"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--factors", required=True, type=Path)
    parser.add_argument("--assumptions", required=True, type=Path)
    parser.add_argument("--zone-groups", required=True, type=Path,
                        help="Frozen context,zone,floor,building_group table; no source machine paths")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.assumptions.read_text(encoding="utf-8"))
    scenarios = cost_scenarios(cfg)
    pairs = read_csv(args.pairs)
    factors = read_csv(args.factors)
    output, boundaries = price_pairs(pairs, factors, scenarios)
    scope_rows = command_scope_boundaries(pairs, factors, scenarios, cfg, read_csv(args.zone_groups))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "implementation_cost_scenarios.csv", scenarios)
    write_csv(args.output / "all_signed_paired_costs.csv", output)
    write_csv(args.output / "bill_implementation_boundaries.csv", boundaries)
    write_csv(args.output / "command_scope_management_boundaries.csv", scope_rows)
    qa = {"scenario_count": len(scenarios), "input_pair_rows": len(pairs), "output_pair_scenario_rows": len(output),
        "unmatched_pair_rows_preserved": sum(row["status"] != "matched" for row in pairs),
        "includes_bill_increases": True, "observed_implementation_cost": False,
        "annual_dr_income_calculated": False,
        "command_scope_boundary_rows": len(scope_rows),
        "unknown_per_group_hours_remain_affine": True,
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.pairs, args.factors, args.assumptions, args.zone_groups)}}
    (args.output / "cost_QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa))


if __name__ == "__main__":
    main()
