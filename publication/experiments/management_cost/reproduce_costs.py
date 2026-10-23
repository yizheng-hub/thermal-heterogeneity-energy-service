"""Reproduce implementation-cost scenarios from the frozen service-paired ledger.

All new monetary results are explicit scenarios. Building parameters and energy
results are read only. This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
from statistics import median


def number(value):
    return None if value in ("", None) else float(value)


def dump_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resource_cost(package, wage, annuity):
    setup = package["setup_hours"] * wage["facility_usd_per_hour"] * annuity
    capital = package["added_interface_capital_usd"] * annuity
    maintenance = package["annual_maintenance_hours"] * wage["facility_usd_per_hour"]
    feedback_hours = package["extra_responses_year"] * package["seconds_per_response"] / 3600
    feedback = feedback_hours * wage["respondent_usd_per_hour"]
    return {
        "annualized_setup_usd": setup,
        "annualized_interface_usd": capital,
        "annual_maintenance_usd": maintenance,
        "respondent_hours_year": feedback_hours,
        "annual_feedback_opportunity_usd": feedback,
        "annual_incremental_operator_cost_usd": setup
        + capital
        + maintenance
        + package["annual_software_usd"],
        "annual_incremental_resource_cost_usd": setup
        + capital
        + maintenance
        + feedback
        + package["annual_software_usd"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "clean_data/building/regional_carbon_cost.csv.gz",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--assumptions", type=Path, default=Path(__file__).with_name("ASSUMPTIONS.json")
    )
    args = parser.parse_args()
    cfg = json.loads(args.assumptions.read_text(encoding="utf-8"))
    open_source = gzip.open if args.input.suffix == ".gz" else open
    with open_source(args.input, "rt", encoding="utf-8-sig", newline="") as stream:
        source = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    rows = [row for row in source if row["strategy"] != "uniform"]
    rate, years = cfg["discount_rate"], cfg["lifetime_years"]
    annuity = rate * (1 + rate) ** years / ((1 + rate) ** years - 1)
    grid = cfg["grid"]
    packages = []
    for index, values in enumerate(
        itertools.product(
            grid["setup_hours"],
            grid["annual_maintenance_hours"],
            grid["feedback"],
            grid["annual_software_usd"],
            grid["added_interface_capital_usd"],
        )
    ):
        setup, maintenance, feedback, software, capital = values
        packages.append(
            {
                "name": f"grid_{index:03d}",
                "setup_hours": setup,
                "annual_maintenance_hours": maintenance,
                "extra_responses_year": feedback[0],
                "seconds_per_response": feedback[1],
                "annual_software_usd": software,
                "added_interface_capital_usd": capital,
                "kind": "sensitivity_grid",
            }
        )
    packages.extend({**package, "kind": "named_package"} for package in cfg["named_packages"])
    scenarios = []
    for package, (wage_name, wage) in itertools.product(packages, cfg["wage_profiles"].items()):
        scenarios.append(
            {
                "scenario_id": f"{package['name']}__{wage_name}",
                "package": package["name"],
                "kind": package["kind"],
                "wage_profile": wage_name,
                **{key: value for key, value in package.items() if key not in ("name", "kind")},
                **wage,
                "annuity_factor": annuity,
                **resource_cost(package, wage, annuity),
            }
        )
    dump_csv(args.output / "scenario_costs.csv", scenarios)
    us_matched = [
        row for row in rows if row["comparison_status"] == "matched" and row["currency"] == "USD"
    ]
    assert len(rows) == 1440 and len(us_matched) == 432
    assert (
        sum(number(row["signed_management_headroom_currency_year"]) > 1e-8 for row in us_matched)
        == 34
    )
    for row in rows:
        if row["comparison_status"] == "matched":
            assert row["status"] == "feasible" and row["status_uniform"] == "feasible"
            assert number(row["actual_mean_c"]) <= number(row["mean_budget_c"]) + 1e-9
            assert number(row["actual_fixed_tail_c"]) <= number(row["fixed_tail_budget_c"]) + 1e-9
    pair_rows, break_rows = [], []
    id_fields = (
        "city",
        "seed",
        "allocation",
        "griffiths_coefficient",
        "interval_width_multiplier",
        "family",
        "budget_basis",
        "tail_basis",
        "service_fraction",
        "strategy",
        "chosen_case",
        "comparison_status",
        "currency",
    )
    for index, row in enumerate(rows):
        base = {"source_nonuniform_row": index, **{key: row[key] for key in id_fields}}
        headroom = number(row["signed_management_headroom_currency_year"])
        matched = row["comparison_status"] == "matched"
        us_priced = matched and row["currency"] == "USD" and headroom is not None
        break_row = {
            **base,
            "signed_energy_bill_headroom_currency_year": headroom,
            "delhi_affine_intercept_INR_year": row[
                "delhi_affine_headroom_intercept_at_pf095_INR_year"
            ],
            "delhi_affine_gas_price_slope_kwh_HHV": row[
                "delhi_affine_headroom_slope_per_INR_kwh_gas"
            ],
            "additional_nonenergy_resource_cost_INR_year": None,
            "nonnegative_break_even_budget_exists": None,
            "break_even_facility_hours_year_median_wage": None,
            "break_even_facility_hours_year_mean_wage": None,
            "break_even_completed_15sec_responses_year_median_wage": None,
            "net_benefit_expression": "unmatched: no same-service monetary comparison",
        }
        if us_priced:
            # Negative headroom has no non-negative break-even spending budget.
            break_row.update(
                {
                    "nonnegative_break_even_budget_exists": headroom >= -1e-8,
                    "break_even_facility_hours_year_median_wage": max(0, headroom)
                    / cfg["wage_profiles"]["median"]["facility_usd_per_hour"]
                    if headroom >= -1e-8
                    else None,
                    "break_even_facility_hours_year_mean_wage": max(0, headroom)
                    / cfg["wage_profiles"]["mean"]["facility_usd_per_hour"]
                    if headroom >= -1e-8
                    else None,
                    "break_even_completed_15sec_responses_year_median_wage": max(0, headroom)
                    / (cfg["wage_profiles"]["median"]["respondent_usd_per_hour"] * 15 / 3600)
                    if headroom >= -1e-8
                    else None,
                    "net_benefit_expression": "headroom_USD - incremental_USD_resource_cost",
                }
            )
            for scenario in scenarios:
                operator_cost = scenario["annual_incremental_operator_cost_usd"]
                resource_cost_value = scenario["annual_incremental_resource_cost_usd"]
                pair_rows.append(
                    {
                        **base,
                        "scenario_id": scenario["scenario_id"],
                        "cost_applicability": "US wage scenario applied to qualified matched pair",
                        "signed_energy_bill_headroom_currency_year": headroom,
                        "annual_incremental_operator_cost_usd": operator_cost,
                        "annual_incremental_resource_cost_usd": resource_cost_value,
                        "annual_operator_net_cost_change_usd": operator_cost - headroom,
                        "annual_resource_net_cost_change_usd": resource_cost_value - headroom,
                        "operator_net_saving": operator_cost - headroom < -1e-8,
                        "resource_net_saving": resource_cost_value - headroom < -1e-8,
                        "delhi_affine_intercept_INR_year": None,
                        "delhi_affine_gas_price_slope_kwh_HHV": None,
                        "net_benefit_expression": "headroom_USD - incremental_USD_resource_cost",
                    }
                )
        else:
            if matched and row["city"] == "NewDelhi":
                break_row["net_benefit_expression"] = (
                    "A_INR + B_kWh_HHV * gas_price_INR_per_kWh_HHV - M_INR_year"
                )
            pair_rows.append(
                {
                    **base,
                    "scenario_id": "not_US_priced",
                    "cost_applicability": "India gas-price and implementation-cost boundary"
                    if matched
                    else "no matched same-service pair",
                    "signed_energy_bill_headroom_currency_year": headroom,
                    "annual_incremental_operator_cost_usd": None,
                    "annual_incremental_resource_cost_usd": None,
                    "annual_operator_net_cost_change_usd": None,
                    "annual_resource_net_cost_change_usd": None,
                    "operator_net_saving": None,
                    "resource_net_saving": None,
                    "delhi_affine_intercept_INR_year": row[
                        "delhi_affine_headroom_intercept_at_pf095_INR_year"
                    ],
                    "delhi_affine_gas_price_slope_kwh_HHV": row[
                        "delhi_affine_headroom_slope_per_INR_kwh_gas"
                    ],
                    "net_benefit_expression": break_row["net_benefit_expression"],
                }
            )
        break_rows.append(break_row)
    assert len(break_rows) == len(rows)
    dump_csv(args.output / "paired_net_costs.csv", pair_rows)
    dump_csv(args.output / "break_even_resources.csv", break_rows)
    grouped = []
    for scenario in scenarios:
        for city in sorted({row["city"] for row in us_matched}):
            group = [
                number(row["signed_management_headroom_currency_year"])
                for row in us_matched
                if row["city"] == city
            ]
            resource_cost_value = scenario["annual_incremental_resource_cost_usd"]
            operator_cost = scenario["annual_incremental_operator_cost_usd"]
            grouped.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "city": city,
                    "matched_rows": len(group),
                    "annual_operator_cost_usd": operator_cost,
                    "annual_resource_cost_usd": resource_cost_value,
                    "operator_net_saving_rows": sum(
                        value - operator_cost > 1e-8 for value in group
                    ),
                    "resource_net_saving_rows": sum(
                        value - resource_cost_value > 1e-8 for value in group
                    ),
                    "median_resource_net_cost_change_usd": median(
                        resource_cost_value - value for value in group
                    ),
                    "minimum_resource_net_cost_change_usd": min(
                        resource_cost_value - value for value in group
                    ),
                    "maximum_resource_net_cost_change_usd": max(
                        resource_cost_value - value for value in group
                    ),
                }
            )
    dump_csv(args.output / "scenario_summary_by_city.csv", grouped)
    positive = [
        number(row["signed_management_headroom_currency_year"])
        for row in us_matched
        if number(row["signed_management_headroom_currency_year"]) > 1e-8
    ]
    named = []
    for scenario in scenarios:
        if scenario["kind"] == "named_package":
            matching = [row for row in grouped if row["scenario_id"] == scenario["scenario_id"]]
            named.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "annual_operator_cost_usd": scenario["annual_incremental_operator_cost_usd"],
                    "annual_resource_cost_usd": scenario["annual_incremental_resource_cost_usd"],
                    "operator_net_saving_rows": sum(
                        row["operator_net_saving_rows"] for row in matching
                    ),
                    "resource_net_saving_rows": sum(
                        row["resource_net_saving_rows"] for row in matching
                    ),
                    "matched_US_rows": len(us_matched),
                }
            )
    city_headroom = {}
    for city in sorted({row["city"] for row in us_matched}):
        group = [
            number(row["signed_management_headroom_currency_year"])
            for row in us_matched
            if row["city"] == city
        ]
        city_headroom[city] = {
            "rows": len(group),
            "positive": sum(value > 1e-8 for value in group),
            "ties": sum(abs(value) <= 1e-8 for value in group),
            "negative": sum(value < -1e-8 for value in group),
            "headroom_min_USD_year": min(group),
            "headroom_median_USD_year": median(group),
            "headroom_max_USD_year": max(group),
            "max_supported_facility_hours_median_wage": max(0, max(group))
            / cfg["wage_profiles"]["median"]["facility_usd_per_hour"],
        }
    summary = {
        "source_sha256": source_hash,
        "assumptions_sha256": hashlib.sha256(args.assumptions.read_bytes()).hexdigest(),
        "source_rows": len(source),
        "nonuniform_rows": len(rows),
        "US_priced_matched_rows": len(us_matched),
        "India_affine_matched_rows": sum(
            row["comparison_status"] == "matched" and row["city"] == "NewDelhi" for row in rows
        ),
        "unmatched_rows_preserved": sum(row["comparison_status"] != "matched" for row in rows),
        "scenario_count": len(scenarios),
        "paired_output_rows": len(pair_rows),
        "annuity_factor": annuity,
        "US_energy_headroom": {
            "positive": len(positive),
            "tie": sum(
                abs(number(row["signed_management_headroom_currency_year"])) <= 1e-8
                for row in us_matched
            ),
            "negative": sum(
                number(row["signed_management_headroom_currency_year"]) < -1e-8
                for row in us_matched
            ),
            "positive_median_USD_year": median(positive),
            "positive_range_USD_year": [min(positive), max(positive)],
        },
        "city_headroom": city_headroom,
        "named_packages": named,
        "scientific_status": "Repeated paired design rows, not a building sample or independent field observations. All new implementation costs are analyst scenarios.",
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == source_hash
    print(
        json.dumps(
            {
                "scenarios": len(scenarios),
                "US_paired_rows": len(us_matched),
                "unmatched_preserved": summary["unmatched_rows_preserved"],
                "output_rows": len(pair_rows),
                "input_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()
