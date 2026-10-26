"""Combine the final repaired analyses into submission-ready source tables.

This module performs no simulation and does not change any qualification rule.
It only joins the declared primary and strict outputs, adds data-source labels,
and records the headline denominators used by the manuscript and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCES = ("ashrae", "ctcd")
SCOPES = ("primary", "strict")
COMMON_LAYOUTS = ("random_mixing", "demand_ordered")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analysis_dir(results_root: Path, source: str, scope: str) -> Path:
    suffix = "full_analysis" if scope == "primary" else "strict_analysis"
    return results_root / f"repaired_controls_{source}_{suffix}"


def read_with_source(path: Path, source: str, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "data_source" not in frame:
        frame.insert(0, "data_source", source.upper())
    frame.insert(1, "quality_scope", scope)
    return frame


def headline_summary(
    annual: pd.DataFrame,
    selections: pd.DataFrame,
    seasonal: pd.DataFrame | None = None,
    permissions: pd.DataFrame | None = None,
    common_dr: pd.DataFrame | None = None,
    own_dr: pd.DataFrame | None = None,
    own_comparisons: pd.DataFrame | None = None,
) -> dict[str, object]:
    direct = annual[
        (annual["quality_scope"] == "primary")
        & (annual["comparison_mode"] == "direct")
        & (annual["permission"] == "native_zone")
        & annual["layout"].isin(COMMON_LAYOUTS)
        & (annual["status"] == "matched")
    ].copy()
    same_service = annual[
        (annual["quality_scope"] == "primary")
        & (annual["comparison_mode"] == "equal_service")
        & (annual["service_fraction"] == 1.0)
        & annual["layout"].isin(COMMON_LAYOUTS)
        & (annual["status"] == "matched")
    ].copy()
    tight = selections[
        (selections["quality_scope"] == "primary")
        & (selections["comparison_mode"] == "equal_service")
        & (selections["service_fraction"] == 0.5)
        & selections["layout"].isin(COMMON_LAYOUTS)
    ].copy()
    tol = 1e-9
    all_service = (
        (direct["delta_mean_reference_mismatch_c"] < -tol)
        & (direct["delta_fixed_tail_reference_mismatch_c"] < -tol)
        & (direct["delta_modeled_interval_person_hour_coverage"] > tol)
    )
    selected = tight[tight["selection_status"] == "selected"].groupby("control").size()
    summary: dict[str, object] = {
        "direct_common_native_zone": {
            "matched_pairs": int(len(direct)),
            "mean_mismatch_improved": int((direct["delta_mean_reference_mismatch_c"] < -tol).sum()),
            "fixed_tail_improved": int((direct["delta_fixed_tail_reference_mismatch_c"] < -tol).sum()),
            "interval_coverage_improved": int(
                (direct["delta_modeled_interval_person_hour_coverage"] > tol).sum()
            ),
            "all_three_service_endpoints_improved": int(all_service.sum()),
            "annual_energy_increased": int((direct["delta_annual_site_kwh"] > tol).sum()),
            "annual_energy_decreased": int((direct["delta_annual_site_kwh"] < -tol).sum()),
        },
        "same_service_common_design": {
            "matched_pairs": int(len(same_service)),
            "median_site_energy_change_percent": float(same_service["delta_site_pct"].median()),
            "annual_energy_increased": int((same_service["delta_annual_site_kwh"] > tol).sum()),
            "annual_energy_decreased": int((same_service["delta_annual_site_kwh"] < -tol).sum()),
            "annual_energy_tied": int((same_service["delta_annual_site_kwh"].abs() <= tol).sum()),
        },
        "tight_service_selected_conditions": {
            control: int(selected.get(control, 0))
            for control in ("uniform_average", "personal_center", "personal_range")
        },
    }
    if seasonal is not None:
        season = seasonal[
            (seasonal["quality_scope"] == "primary")
            & (seasonal["comparison_mode"] == "direct")
            & (seasonal["permission"] == "native_zone")
            & seasonal["layout"].isin(COMMON_LAYOUTS)
        ].copy()
        keys = [
            "data_source",
            "context",
            "composition_id",
            "layout",
            "permission",
            "control",
        ]
        grouped = season.groupby(keys)["delta_site_kwh"]
        signs = grouped.agg(["min", "max", "sum"])
        absolute = grouped.apply(lambda values: values.abs().sum()).rename("absolute_sum")
        signs = signs.join(absolute)
        mixed = signs[(signs["min"] < -tol) & (signs["max"] > tol)].copy()
        cancellation = 1.0 - mixed["sum"].abs() / mixed["absolute_sum"]
        summary["seasonal_reallocation"] = {
            "annual_comparisons": int(len(signs)),
            "mixed_sign_across_seasons": int(len(mixed)),
            "median_cancellation_fraction_in_mixed_cases": float(cancellation.median()),
        }
    if permissions is not None:
        permission = permissions[
            (permissions["quality_scope"] == "primary")
            & (permissions["permission_step"] == "building_to_native_zone")
            & permissions["layout"].isin(COMMON_LAYOUTS)
            & permissions["control"].isin(("personal_center", "personal_range"))
        ]
        summary["building_to_native_zone_permission"] = {
            "matched_pairs": int(len(permission)),
            "mean_mismatch_improved": int(
                (permission["delta_mean_reference_mismatch_c"] < -tol).sum()
            ),
            "fixed_tail_improved": int(
                (permission["delta_fixed_tail_reference_mismatch_c"] < -tol).sum()
            ),
            "interval_coverage_improved": int(
                (permission["delta_modeled_interval_person_hour_coverage"] > tol).sum()
            ),
            "annual_energy_increased": int((permission["delta_annual_site_kwh"] > tol).sum()),
            "annual_energy_decreased": int((permission["delta_annual_site_kwh"] < -tol).sum()),
        }
    if common_dr is not None:
        common = common_dr[common_dr["quality_scope"] == "primary"]
        info: dict[str, object] = {}
        for level, group in common.groupby("information_level"):
            info[str(level)] = {
                "evaluated_conditions": int(len(group)),
                "selected_positive_response": int((group["selected_response_kw"] > tol).sum()),
                "usable_after_full_review": int(
                    (group["full_reviewed_usable_response_kw"] > tol).sum()
                ),
                "false_acceptances": int(
                    (group["selected_review_outcome"] == "false_acceptance").sum()
                ),
            }
        unique_conditions = common.drop_duplicates(
            ["data_source", "context", "composition_id", "layout", "permission", "event_month"]
        )
        info["full_information"] = {
            "evaluated_conditions": int(len(unique_conditions)),
            "selected_positive_response": int(
                (unique_conditions["ideal_full_information_response_kw"] > tol).sum()
            ),
        }
        summary["common_baseline_response"] = info
    if own_dr is not None and own_comparisons is not None:
        own = own_dr[own_dr["quality_scope"] == "primary"].copy()
        qualified = own["qualified_positive_response"].astype(str).str.lower().eq("true")
        own["qualified_positive"] = qualified
        paired = own_comparisons[own_comparisons["quality_scope"] == "primary"].copy()
        uniform_selected = paired["uniform_status"] == "selected"
        demand_selected = paired["demand_guided_status"] == "selected"
        both = paired[uniform_selected & demand_selected]
        response_difference = both["response_difference_kw"]
        selected = own[qualified]
        summary["own_baseline_response"] = {
            "evaluated_control_conditions": int(len(own)),
            "qualified_positive_by_control": {
                str(control): int(group["qualified_positive"].sum())
                for control, group in own.groupby("control")
            },
            "paired_both_positive": int(len(both)),
            "demand_guided_response_higher": int((response_difference > tol).sum()),
            "demand_guided_response_lower": int((response_difference < -tol).sum()),
            "uniform_only_positive": int((uniform_selected & ~demand_selected).sum()),
            "demand_guided_only_positive": int((~uniform_selected & demand_selected).sum()),
            "median_recovery_to_event_energy_ratio": float(
                (selected["recovery_extra_electric_kwh"] / selected["event_electric_saving_kwh"]).median()
            ),
            "net_positive_after_recovery": int(
                (selected["event_plus_recovery_net_electric_saving_kwh"] > tol).sum()
            ),
        }
    return summary


def build(results_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    inputs: list[Path] = []
    annual_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    seasonal_frames: list[pd.DataFrame] = []
    permission_frames: list[pd.DataFrame] = []
    placement_frames: list[pd.DataFrame] = []

    for scope in SCOPES:
        for source in SOURCES:
            directory = analysis_dir(results_root, source, scope)
            annual_path = directory / "REPAIRED_CONTROL_COMPARISONS.csv"
            selections_path = directory / "REPAIRED_ANNUAL_SERVICE_SELECTIONS.csv"
            inputs.extend((annual_path, selections_path))
            annual_frames.append(read_with_source(annual_path, source, scope))
            selection_frames.append(read_with_source(selections_path, source, scope))

            seasonal_directory = results_root / (
                f"repaired_controls_{source}_seasonal"
                if scope == "primary"
                else f"repaired_controls_{source}_strict_seasonal"
            )
            seasonal_path = seasonal_directory / "SEASONAL_CONTROL_DIFFERENCES.csv"
            inputs.append(seasonal_path)
            seasonal_frames.append(read_with_source(seasonal_path, source, scope))

            spatial_directory = results_root / (
                f"repaired_controls_{source}_spatial"
                if scope == "primary"
                else f"repaired_controls_{source}_strict_spatial"
            )
            permission_path = spatial_directory / "DIRECT_PERMISSION_EFFECTS.csv"
            placement_path = spatial_directory / "DIRECT_PLACEMENT_EFFECTS.csv"
            inputs.extend((permission_path, placement_path))
            permission_frames.append(read_with_source(permission_path, source, scope))
            placement_frames.append(read_with_source(placement_path, source, scope))

    annual = pd.concat(annual_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    seasonal = pd.concat(seasonal_frames, ignore_index=True)
    permissions = pd.concat(permission_frames, ignore_index=True)
    placements = pd.concat(placement_frames, ignore_index=True)

    common_frames: list[pd.DataFrame] = []
    own_frames: list[pd.DataFrame] = []
    own_comparison_frames: list[pd.DataFrame] = []
    for scope in SCOPES:
        common_path = (
            results_root
            / f"common_baseline_dr_{scope}_analysis"
            / "COMMON_BASELINE_DR_INFORMATION_COMPARISONS.csv"
        )
        inputs.append(common_path)
        common = pd.read_csv(common_path)
        common.insert(1, "quality_scope", scope)
        common_frames.append(common)
        for source in SOURCES:
            own_directory = results_root / f"own_baseline_dr_{source}_{scope}_analysis"
            own_path = own_directory / "OWN_BASELINE_DR_SELECTIONS.csv"
            comparison_path = own_directory / "OWN_BASELINE_DR_COMPARISONS.csv"
            inputs.extend((own_path, comparison_path))
            own_frames.append(read_with_source(own_path, source, scope))
            own_comparison_frames.append(read_with_source(comparison_path, source, scope))

    common_dr = pd.concat(common_frames, ignore_index=True)
    own_dr = pd.concat(own_frames, ignore_index=True)
    own_comparisons = pd.concat(own_comparison_frames, ignore_index=True)

    information_path = results_root.parent / "decision_value" / "INFORMATION_HIERARCHY_SUMMARY.json"
    information_summary = json.loads(information_path.read_text(encoding="utf-8"))
    fixed_plan_review = information_summary["zone_mean_to_full_profile"]
    inputs.append(information_path)

    cost_pairs = annual[annual["quality_scope"] == "primary"].copy()
    cost_pairs.insert(0, "comparison_id", [f"cmp-{index:06d}" for index in range(len(cost_pairs))])
    cost_pairs["climate_context"] = cost_pairs["context"]
    cost_pairs["scope"] = cost_pairs["permission"]
    cost_pairs["control"] = cost_pairs["alternative"]
    cost_pairs["delta_electric_kwh"] = cost_pairs["delta_annual_hvac_electric_kwh"]
    cost_pairs["delta_gas_hhv_kwh"] = cost_pairs["delta_annual_gas_hhv_kwh"]
    cost_pairs = cost_pairs[
        [
            "comparison_id",
            "data_source",
            "climate_context",
            "scope",
            "control",
            "composition_id",
            "layout",
            "comparison_mode",
            "status",
            "service_fraction",
            "delta_electric_kwh",
            "delta_gas_hhv_kwh",
            "delta_annual_site_kwh",
            "delta_annual_operational_co2_kg",
            "delta_mean_reference_mismatch_c",
            "delta_fixed_tail_reference_mismatch_c",
            "delta_modeled_interval_person_hour_coverage",
        ]
    ]

    outputs = {
        "FINAL_ANNUAL_COMPARISONS.csv": annual,
        "FINAL_ANNUAL_SELECTIONS.csv": selections,
        "FINAL_SEASONAL_DIFFERENCES.csv": seasonal,
        "FINAL_PERMISSION_EFFECTS.csv": permissions,
        "FINAL_PLACEMENT_EFFECTS.csv": placements,
        "FINAL_COMMON_BASELINE_DR.csv": common_dr,
        "FINAL_OWN_BASELINE_DR.csv": own_dr,
        "FINAL_OWN_BASELINE_DR_COMPARISONS.csv": own_comparisons,
        "FINAL_COST_PAIRS.csv": cost_pairs,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output / filename, index=False)

    summary = headline_summary(
        annual,
        selections,
        seasonal=seasonal,
        permissions=permissions,
        common_dr=common_dr,
        own_dr=own_dr,
        own_comparisons=own_comparisons,
    )
    summary["fixed_plan_information_diagnostic"] = fixed_plan_review
    summary["input_sha256"] = {
        path.relative_to(results_root.parent).as_posix(): sha256(path) for path in inputs
    }
    summary["output_rows"] = {name: int(len(frame)) for name, frame in outputs.items()}
    (output / "FINAL_EVIDENCE_QA.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.results_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
