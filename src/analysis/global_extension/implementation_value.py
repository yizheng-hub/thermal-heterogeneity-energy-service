"""Build implementation-value boundaries for the global experiment.

The analysis keeps measured or simulated outcomes separate from illustrative
implementation-resource scenarios.  It does not assign local tariffs, value
thermal service, or invent demand-response revenue.  Instead, it reports the
coefficients needed to make those choices explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis.energy_service.run_costs import cost_scenarios


CONTROL_LABELS = {
    "personal_center": "Personal centre",
    "personal_range": "Personal range",
}
SCOPE_COMMANDS = {"building": 1, "floor": 3, "native_zone": 15}

RESOURCE_CATEGORIES = (
    {
        "resource_category": "demand_acquisition",
        "plain_language_resource": "Demand feedback and profile estimation",
        "quantity_fields": "extra_responses_year; seconds_per_response",
        "required_in_every_deployment": False,
        "optional_resource": True,
        "evidence_reference_ids": "iea_ebc_annex79_2024",
        "boundary": (
            "Existing records can be reused; additional feedback is an explicit "
            "scenario input, not an inferred requirement."
        ),
    },
    {
        "resource_category": "software_configuration",
        "plain_language_resource": "Control software integration and commissioning",
        "quantity_fields": "setup_hours; annual_software_usd",
        "required_in_every_deployment": True,
        "optional_resource": False,
        "evidence_reference_ids": "doe_controls2024; iea_ebc_annex79_2024",
        "boundary": (
            "The sources establish the task category; hours and fees remain "
            "declared sensitivity inputs."
        ),
    },
    {
        "resource_category": "ongoing_maintenance",
        "plain_language_resource": "Ongoing operation, checking and maintenance",
        "quantity_fields": "annual_maintenance_hours",
        "required_in_every_deployment": True,
        "optional_resource": False,
        "evidence_reference_ids": "doe_controls2024",
        "boundary": (
            "Annual hours are varied parametrically; they are not measured at the "
            "nine simulated sites."
        ),
    },
    {
        "resource_category": "staff_reconfiguration",
        "plain_language_resource": "Reassigning occupants to demand-compatible spaces",
        "quantity_fields": "layout comparison; local reassignment burden",
        "required_in_every_deployment": False,
        "optional_resource": True,
        "evidence_reference_ids": "iea_ebc_annex79_2024",
        "boundary": (
            "The spatial experiment measures service and energy effects; local moves, "
            "disruption and organisational cost remain unpriced."
        ),
    },
    {
        "resource_category": "command_scope",
        "plain_language_resource": "Independent building, floor or zone commands",
        "quantity_fields": "1, 3 or 15 independent command groups",
        "required_in_every_deployment": True,
        "optional_resource": False,
        "evidence_reference_ids": "doe_controls2024",
        "boundary": (
            "Command counts are fixed by the simulated prototype; local setup and "
            "management time per command group remain unpriced."
        ),
    },
    {
        "resource_category": "optional_local_hardware",
        "plain_language_resource": "Optional local interface or hardware",
        "quantity_fields": "added_interface_capital_usd",
        "required_in_every_deployment": False,
        "optional_resource": True,
        "evidence_reference_ids": "doe_controls2024",
        "boundary": (
            "No added hardware is required by the simulation. Any dollar amount is an "
            "illustrative United States scenario and does not price physical rezoning."
        ),
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matched_direct(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "climate",
        "composition",
        "alternative_control",
        "status",
        "delta_mean_reference_mismatch_c",
        "delta_fixed_tail_reference_mismatch_c",
        "delta_modeled_interval_mean_excess_c",
        "delta_modeled_interval_person_hour_coverage",
        "delta_facility_electric_kwh",
        "delta_facility_gas_hhv_kwh",
        "delta_facility_purchased_kwh",
        "delta_operational_co2_country_kg",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Direct-comparison table lacks {sorted(missing)}")
    matched = frame[frame.status.eq("matched")].copy()
    if matched.empty:
        raise ValueError("Direct-comparison table has no matched rows")
    if not matched.alternative_control.isin(CONTROL_LABELS).all():
        raise ValueError("Direct comparisons contain an unknown control")
    if matched[["climate", "composition", "alternative_control"]].duplicated().any():
        raise ValueError("Direct comparisons duplicate a context-control row")
    matched["target_service_gain_c"] = matched.apply(
        lambda row: -float(row.delta_mean_reference_mismatch_c)
        if row.alternative_control == "personal_center"
        else -float(row.delta_modeled_interval_mean_excess_c),
        axis=1,
    )
    matched["fixed_tail_service_gain_c"] = (
        -matched.delta_fixed_tail_reference_mismatch_c.astype(float)
    )
    matched["interval_coverage_gain_percentage_points"] = (
        100 * matched.delta_modeled_interval_person_hour_coverage.astype(float)
    )
    matched["control_label"] = matched.alternative_control.map(CONTROL_LABELS)
    return matched


def build_resource_scenarios(assumptions: dict) -> pd.DataFrame:
    if assumptions.get("monetary_scope") != "illustrative United States dollar sensitivity":
        raise ValueError(
            "Global implementation assumptions must declare the illustrative US scope"
        )
    scenarios = pd.DataFrame(cost_scenarios(assumptions))
    if scenarios.empty or scenarios.scenario_id.duplicated().any():
        raise ValueError("Implementation scenarios are empty or duplicated")
    source_ids = {row["reference_id"] for row in assumptions.get("sources", [])}
    if "bls_facilities2025" not in source_ids:
        raise ValueError("Illustrative US wage scenarios require the declared BLS source")
    manager_wages = {
        name: float(values["facility_usd_per_hour"])
        for name, values in assumptions["wage_profiles"].items()
    }
    respondent_wages = {
        name: float(values["respondent_usd_per_hour"])
        for name, values in assumptions["wage_profiles"].items()
    }
    scenarios["annualized_setup_labor_usd"] = scenarios.apply(
        lambda row: row.annuity_factor
        * row.setup_hours
        * manager_wages[row.wage_profile],
        axis=1,
    )
    scenarios["annual_maintenance_labor_usd"] = scenarios.apply(
        lambda row: row.annual_maintenance_hours
        * manager_wages[row.wage_profile],
        axis=1,
    )
    scenarios["annual_feedback_time_usd"] = scenarios.apply(
        lambda row: row.respondent_hours_year
        * respondent_wages[row.wage_profile],
        axis=1,
    )
    scenarios["annualized_optional_interface_usd"] = (
        scenarios.annuity_factor * scenarios.added_interface_capital_usd
    )
    scenarios["scope"] = assumptions["monetary_scope"]
    scenarios["observed_implementation_cost"] = False
    scenarios["climate_specific_cost_estimate"] = False
    return scenarios


def build_resource_catalog(assumptions: dict) -> pd.DataFrame:
    """Describe implementation inputs without assigning global monetary values."""
    sources = {row["reference_id"] for row in assumptions.get("sources", [])}
    required = {
        reference
        for row in RESOURCE_CATEGORIES
        for reference in row["evidence_reference_ids"].split("; ")
    }
    missing = required - sources
    if missing:
        raise ValueError(f"Implementation assumptions lack sources {sorted(missing)}")
    frame = pd.DataFrame(RESOURCE_CATEGORIES)
    frame["global_monetary_value_assigned"] = False
    frame["simulated_outcome"] = False
    return frame


def build_value_boundaries(
    direct: pd.DataFrame, scenarios: pd.DataFrame
) -> pd.DataFrame:
    """Cross simulated resource changes with declared resource scenarios."""
    rows: list[dict] = []
    for comparison in direct.to_dict("records"):
        for scenario in scenarios.to_dict("records"):
            rows.append({
                "climate": comparison["climate"],
                "composition": comparison["composition"],
                "alternative_control": comparison["alternative_control"],
                "control_label": comparison["control_label"],
                "scenario_id": scenario["scenario_id"],
                "scenario_kind": scenario["kind"],
                "wage_profile": scenario["wage_profile"],
                "target_service_gain_c": comparison["target_service_gain_c"],
                "fixed_tail_service_gain_c": comparison["fixed_tail_service_gain_c"],
                "interval_coverage_gain_percentage_points": comparison[
                    "interval_coverage_gain_percentage_points"
                ],
                "delta_facility_electric_kwh_year": comparison[
                    "delta_facility_electric_kwh"
                ],
                "delta_facility_gas_hhv_kwh_year": comparison[
                    "delta_facility_gas_hhv_kwh"
                ],
                "delta_facility_purchased_kwh_year": comparison[
                    "delta_facility_purchased_kwh"
                ],
                "delta_operational_co2_country_kg_year": comparison[
                    "delta_operational_co2_country_kg"
                ],
                "illustrative_annual_resource_cost_usd": scenario[
                    "annual_resource_cost_usd"
                ],
                "minimum_external_value_before_local_energy_prices_usd_year": scenario[
                    "annual_resource_cost_usd"
                ],
                "electricity_price_coefficient_kwh_year": comparison[
                    "delta_facility_electric_kwh"
                ],
                "gas_price_coefficient_kwh_hhv_year": comparison[
                    "delta_facility_gas_hhv_kwh"
                ],
                "value_boundary": (
                    "V_service + V_response >= max(0, C_implementation + "
                    "delta_electricity * local_electricity_price + "
                    "delta_gas * local_gas_price)"
                ),
                "required_combined_value_if_energy_charge_change_is_zero_usd_year": (
                    scenario["annual_resource_cost_usd"]
                ),
                "required_combined_value_with_local_energy_prices": "not assigned",
                "monetary_scope": scenario["scope"],
                "local_energy_prices_assigned": False,
                "thermal_service_value_observed": False,
                "annual_dr_revenue_assigned": False,
            })
    return pd.DataFrame(rows)


def build_scope_requirements() -> pd.DataFrame:
    rows = []
    for scope, commands in SCOPE_COMMANDS.items():
        rows.append({
            "command_scope": scope,
            "independent_command_groups": commands,
            "extra_groups_over_building_command": commands - 1,
            "heating_and_cooling_command_channels": 2 * commands,
            "fixed_medium_office_floors": 3,
            "fixed_medium_office_controlled_zones": 15,
            "per_group_setup_hours_assigned": False,
            "per_group_annual_management_hours_assigned": False,
            "management_boundary": (
                "C_scope = extra_groups * wage * "
                "(annuity_factor * setup_hours_per_group + annual_hours_per_group)"
            ),
        })
    return pd.DataFrame(rows)


def build_spatial_resource_boundaries(
    permission: pd.DataFrame, placement: pd.DataFrame
) -> pd.DataFrame:
    """Attach explicit implementation resources to final spatial comparisons."""
    common = {"climate", "composition", "control", "status", "reference", "alternative"}
    missing_permission = (common | {"axis", "layout"}) - set(permission)
    missing_placement = (common | {"axis", "permission"}) - set(placement)
    if missing_permission:
        raise ValueError(
            f"Permission comparisons lack {sorted(missing_permission)}"
        )
    if missing_placement:
        raise ValueError(
            f"Placement comparisons lack {sorted(missing_placement)}"
        )
    unknown_scopes = (
        set(permission.reference) | set(permission.alternative)
    ) - set(SCOPE_COMMANDS)
    if unknown_scopes:
        raise ValueError(f"Unknown command scopes {sorted(unknown_scopes)}")

    permission_rows = permission.copy()
    permission_rows["resource_category"] = "command_scope"
    permission_rows["reference_resource_quantity"] = permission_rows.reference.map(
        SCOPE_COMMANDS
    )
    permission_rows["alternative_resource_quantity"] = permission_rows.alternative.map(
        SCOPE_COMMANDS
    )
    permission_rows["incremental_resource_quantity"] = (
        permission_rows.alternative_resource_quantity
        - permission_rows.reference_resource_quantity
    )
    permission_rows["resource_unit"] = "independent command groups"
    permission_rows["resource_quantity_assigned"] = True
    permission_rows["staff_reassignments_assigned"] = False

    placement_rows = placement.copy()
    placement_rows["resource_category"] = "staff_reconfiguration"
    placement_rows["reference_resource_quantity"] = pd.NA
    placement_rows["alternative_resource_quantity"] = pd.NA
    placement_rows["incremental_resource_quantity"] = pd.NA
    placement_rows["resource_unit"] = "local occupant reassignments"
    placement_rows["resource_quantity_assigned"] = False
    placement_rows["staff_reassignments_assigned"] = False

    output = pd.concat([permission_rows, placement_rows], ignore_index=True, sort=False)
    output["annual_implementation_cost_assigned"] = False
    output["optional_local_hardware_required_by_simulation"] = False
    return output


def build_information_summary(summary: dict) -> pd.DataFrame:
    if summary.get("status") != "PASS":
        raise ValueError("Global information-compression audit did not pass")
    try:
        row = summary["primary"]["protected five bins"]
    except KeyError as exc:
        raise ValueError("Missing protected five-bin global summary") from exc
    return pd.DataFrame([{
        "representation": "Five anonymous demand bins per zone",
        "protection_rule": summary["protection_split"],
        "protection_quantile": summary["protection_quantile"],
        "climates": summary["climates"],
        "demand_compositions": summary["demand_compositions"],
        "full_information_solvable_cases": summary[
            "primary_full_resolvable_scenarios"
        ],
        "protected_selected_cases": row["selections"],
        "decision_coverage_of_solvable": row["decision_coverage"],
        "qualified_action_retention": row["qualified_action_retention"],
        "observed_selected_plan_false_acceptance": row[
            "selected_plan_false_acceptances"
        ],
        "safe_energy_penalty_median_percent": row[
            "safe_energy_penalty_median_percent"
        ],
        "safe_energy_penalty_p95_percent": row["safe_energy_penalty_p95_percent"],
        "evidence_scope": summary["headline_scope"],
        "formal_privacy_guarantee": False,
        "privacy_interpretation": summary["privacy_interpretation"],
    }])


def analyse(
    direct_path: Path,
    assumptions_path: Path,
    information_summary_path: Path,
    output_root: Path,
    permission_path: Path | None = None,
    placement_path: Path | None = None,
) -> dict:
    direct = _matched_direct(pd.read_csv(direct_path))
    assumptions = json.loads(assumptions_path.read_text(encoding="utf-8"))
    scenarios = build_resource_scenarios(assumptions)
    catalog = build_resource_catalog(assumptions)
    boundaries = build_value_boundaries(direct, scenarios)
    scopes = build_scope_requirements()
    information = build_information_summary(
        json.loads(information_summary_path.read_text(encoding="utf-8"))
    )
    if (permission_path is None) != (placement_path is None):
        raise ValueError("Provide both permission and placement comparisons, or neither")
    spatial = pd.DataFrame()
    if permission_path is not None and placement_path is not None:
        spatial = build_spatial_resource_boundaries(
            pd.read_csv(permission_path), pd.read_csv(placement_path)
        )

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "IMPLEMENTATION_RESOURCE_SCENARIOS.csv": scenarios,
        "IMPLEMENTATION_RESOURCE_CATALOG.csv": catalog,
        "IMPLEMENTATION_EVIDENCE_SOURCES.csv": pd.DataFrame(assumptions["sources"]),
        "GLOBAL_SERVICE_RESOURCE_VALUE_BOUNDARIES.csv": boundaries,
        "COMMAND_SCOPE_REQUIREMENTS.csv": scopes,
        "INFORMATION_BURDEN_SUMMARY.csv": information,
    }
    if not spatial.empty:
        outputs["GLOBAL_SPATIAL_IMPLEMENTATION_BOUNDARIES.csv"] = spatial
    hashes = {}
    for name, frame in outputs.items():
        path = output_root / name
        frame.to_csv(path, index=False)
        hashes[name] = sha256(path)
    audit = {
        "status": "PASS",
        "direct_comparisons": len(direct),
        "implementation_scenarios": len(scenarios),
        "implementation_resource_categories": len(catalog),
        "value_boundary_rows": len(boundaries),
        "spatial_resource_rows": len(spatial),
        "spatial_results_included": not spatial.empty,
        "local_energy_prices_assigned": False,
        "thermal_service_value_observed": False,
        "annual_dr_revenue_assigned": False,
        "observed_implementation_cost": False,
        "five_bin_energy_regret_reused": False,
        "inputs": {
            "direct": sha256(direct_path),
            "assumptions": sha256(assumptions_path),
            "information_summary": sha256(information_summary_path),
        } | (
            {
                "permission_comparisons": sha256(permission_path),
                "placement_comparisons": sha256(placement_path),
            }
            if permission_path is not None and placement_path is not None
            else {}
        ),
        "outputs": hashes,
    }
    audit_path = output_root / "IMPLEMENTATION_VALUE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-comparisons", type=Path, required=True)
    parser.add_argument("--assumptions", type=Path, required=True)
    parser.add_argument("--information-summary", type=Path, required=True)
    parser.add_argument("--permission-comparisons", type=Path)
    parser.add_argument("--placement-comparisons", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    audit = analyse(
        args.direct_comparisons,
        args.assumptions,
        args.information_summary,
        args.output_root,
        args.permission_comparisons,
        args.placement_comparisons,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
