"""Audit annual conclusions against the facility-gas accounting boundary.

The main boundary is complete facility electricity plus complete facility gas.
The comparator retains complete facility electricity but includes only hourly
space-heating gas.  Both are read from the same completed simulations; this
module never edits or reruns a building model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


IDENTITY = ("climate", "composition", "layout", "permission")
CONTROLS = ("uniform_average", "personal_center", "personal_range")
ALTERNATIVES = ("personal_center", "personal_range")
SERVICE_FRACTIONS = (1.0, 0.75, 0.5)
PRIMARY_ENERGY = "facility_purchased_kwh"
COMPARATOR_ENERGY = "facility_electric_plus_space_heating_gas_kwh"
PRIMARY_CARBON = "operational_co2_country_kg"
COMPARATOR_CARBON = "facility_electric_plus_space_heating_gas_operational_co2_country_kg"
SERVICE_TOLERANCE_C = 1e-9
NEAR_TIE_THRESHOLDS_PCT = (0.001, 0.01, 0.1)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _as_bool(values: pd.Series, name: str) -> pd.Series:
    if values.dtype == bool:
        return values
    parsed = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise ValueError(f"{name} contains non-boolean values")
    return parsed.astype(bool)


def prepare_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY,
        "control",
        "offset_c",
        "run_id",
        "engineering_pass",
        "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c",
        "facility_gas_hhv_kwh",
        "gas_hhv_kwh",
        PRIMARY_ENERGY,
        COMPARATOR_ENERGY,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Annual metrics lack required columns: {sorted(missing)}")
    result = frame.copy()
    result["engineering_pass"] = _as_bool(result.engineering_pass, "engineering_pass")
    numeric = [
        "offset_c",
        "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c",
        "facility_gas_hhv_kwh",
        "gas_hhv_kwh",
        PRIMARY_ENERGY,
        COMPARATOR_ENERGY,
    ]
    optional_carbon = [name for name in (PRIMARY_CARBON, COMPARATOR_CARBON) if name in result]
    for name in numeric + optional_carbon:
        result[name] = pd.to_numeric(result[name], errors="raise")
    if not np.isfinite(result[numeric + optional_carbon].to_numpy(float)).all():
        raise ValueError("Energy-boundary inputs must be finite")
    if result[[PRIMARY_ENERGY, COMPARATOR_ENERGY]].le(0).any().any():
        raise ValueError("Purchased-energy boundaries must be positive")
    residual = result.facility_gas_hhv_kwh - result.gas_hhv_kwh
    if residual.lt(-1e-6).any():
        raise ValueError("Facility gas cannot be below hourly space-heating gas")
    result["non_space_heating_facility_gas_kwh"] = residual.clip(lower=0)
    result["boundary_difference_kwh"] = result[PRIMARY_ENERGY] - result[COMPARATOR_ENERGY]
    if not np.allclose(
        result.non_space_heating_facility_gas_kwh,
        result.boundary_difference_kwh,
        rtol=1e-9,
        atol=1e-6,
    ):
        raise ValueError("Facility energy-boundary difference does not equal non-space-heating gas")
    keys = [*IDENTITY, "control", "offset_c"]
    if result.duplicated(keys).any() or result.run_id.duplicated().any():
        raise ValueError("Annual metrics duplicate candidate identities")
    unknown = sorted(set(result.control.astype(str)) - set(CONTROLS))
    if unknown:
        raise ValueError(f"Unknown controls: {unknown}")
    return result


def _direction(delta: float, reference: float) -> str:
    tolerance = max(1e-6, abs(reference) * 1e-12)
    if delta > tolerance:
        return "increase"
    if delta < -tolerance:
        return "decrease"
    return "tie"


def direct_boundary_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare central demand-guided and average operations under both boundaries."""
    central = metrics[np.isclose(metrics.offset_c, 0.0)].copy()
    rows: list[dict] = []
    for key, group in central.groupby(list(IDENTITY), sort=True, dropna=False):
        reference = group[group.control.eq("uniform_average")]
        if len(reference) != 1:
            raise ValueError("Every identity requires one central Uniform average run")
        uniform = reference.iloc[0]
        for control in ALTERNATIVES:
            alternative = group[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError(f"Every identity requires one central {control} run")
            personal = alternative.iloc[0]
            matched = bool(uniform.engineering_pass and personal.engineering_pass)
            row = dict(zip(IDENTITY, key, strict=True)) | {
                "alternative_control": control,
                "status": "matched" if matched else "engineering_unavailable",
                "uniform_run_id": str(uniform.run_id),
                "alternative_run_id": str(personal.run_id),
            }
            for label, column in (
                ("complete_facility", PRIMARY_ENERGY),
                ("electricity_plus_space_heating_gas", COMPARATOR_ENERGY),
            ):
                reference_value = float(uniform[column])
                delta = float(personal[column] - uniform[column]) if matched else np.nan
                row[f"uniform_{label}_kwh"] = reference_value
                row[f"alternative_{label}_kwh"] = float(personal[column])
                row[f"delta_{label}_kwh"] = delta
                row[f"delta_{label}_pct"] = 100.0 * delta / reference_value if matched else np.nan
                row[f"direction_{label}"] = (
                    _direction(delta, reference_value) if matched else "unavailable"
                )
            row["energy_direction_changed"] = bool(
                matched
                and row["direction_complete_facility"]
                != row["direction_electricity_plus_space_heating_gas"]
            )
            row["uniform_non_space_heating_facility_gas_kwh"] = float(
                uniform.non_space_heating_facility_gas_kwh
            )
            row["alternative_non_space_heating_facility_gas_kwh"] = float(
                personal.non_space_heating_facility_gas_kwh
            )
            row["delta_non_space_heating_facility_gas_kwh"] = float(
                personal.non_space_heating_facility_gas_kwh
                - uniform.non_space_heating_facility_gas_kwh
            )
            if {PRIMARY_CARBON, COMPARATOR_CARBON}.issubset(metrics.columns):
                for label, column in (
                    ("complete_facility_carbon", PRIMARY_CARBON),
                    ("electricity_plus_space_heating_gas_carbon", COMPARATOR_CARBON),
                ):
                    reference_value = float(uniform[column])
                    delta = float(personal[column] - uniform[column]) if matched else np.nan
                    row[f"delta_{label}_kg"] = delta
                    row[f"direction_{label}"] = (
                        _direction(delta, reference_value) if matched else "unavailable"
                    )
                row["carbon_direction_changed"] = bool(
                    matched
                    and row["direction_complete_facility_carbon"]
                    != row["direction_electricity_plus_space_heating_gas_carbon"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _choose(candidates: pd.DataFrame, energy_column: str) -> dict | None:
    if candidates.empty:
        return None
    minimum = float(candidates[energy_column].min())
    tolerance = max(1e-6, 1e-9 * abs(minimum))
    return (
        candidates[candidates[energy_column].le(minimum + tolerance)]
        .sort_values(["run_id", "offset_c"], kind="mergesort")
        .iloc[0]
        .to_dict()
    )


def boundary_service_selections(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    fractions: tuple[float, ...] = SERVICE_FRACTIONS,
) -> pd.DataFrame:
    """Select the same service-qualified candidates under both energy boundaries."""
    anchor_keys = ["climate", "composition", "layout"]
    required = {
        *anchor_keys,
        "anchor_mean_mismatch_c",
        "anchor_tail_mismatch_c",
    }
    missing = required - set(anchors)
    if missing:
        raise ValueError(f"Service anchors lack required columns: {sorted(missing)}")
    joined = metrics.merge(
        anchors[list(required)], on=anchor_keys, how="left", validate="many_to_one"
    )
    if joined[["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]].isna().any().any():
        raise ValueError("A candidate lacks its external 25 C service anchor")
    rows: list[dict] = []
    group_keys = [*IDENTITY, "control"]
    for key, group in joined.groupby(group_keys, sort=True, dropna=False):
        for fraction in fractions:
            mean_limit = float(group.anchor_mean_mismatch_c.iloc[0]) * fraction
            tail_limit = float(group.anchor_tail_mismatch_c.iloc[0]) * fraction
            qualified = group[
                group.engineering_pass
                & group.mean_reference_mismatch_c.le(mean_limit + SERVICE_TOLERANCE_C)
                & group.fixed_tail_reference_mismatch_c.le(tail_limit + SERVICE_TOLERANCE_C)
            ]
            complete = _choose(qualified, PRIMARY_ENERGY)
            comparator = _choose(qualified, COMPARATOR_ENERGY)
            complete_id = str(complete["run_id"]) if complete else ""
            comparator_id = str(comparator["run_id"]) if comparator else ""
            changed = bool(complete and comparator and complete_id != comparator_id)
            row = dict(zip(group_keys, key, strict=True)) | {
                "service_fraction": fraction,
                "mean_limit_c": mean_limit,
                "tail_limit_c": tail_limit,
                "qualified_candidate_count": len(qualified),
                "selection_status": "selected" if complete else "no_qualifying_plan",
                "complete_facility_selected_run_id": complete_id,
                "comparator_selected_run_id": comparator_id,
                "selection_changed": changed,
            }
            if complete and comparator:
                row.update(
                    {
                        "complete_facility_selected_offset_c": float(complete["offset_c"]),
                        "comparator_selected_offset_c": float(comparator["offset_c"]),
                        "complete_choice_complete_facility_kwh": float(complete[PRIMARY_ENERGY]),
                        "comparator_choice_complete_facility_kwh": float(
                            comparator[PRIMARY_ENERGY]
                        ),
                        "complete_choice_comparator_kwh": float(complete[COMPARATOR_ENERGY]),
                        "comparator_choice_comparator_kwh": float(comparator[COMPARATOR_ENERGY]),
                        "complete_choice_non_space_heating_gas_kwh": float(
                            complete["non_space_heating_facility_gas_kwh"]
                        ),
                        "comparator_choice_non_space_heating_gas_kwh": float(
                            comparator["non_space_heating_facility_gas_kwh"]
                        ),
                    }
                )
                primary_penalty = (
                    row["comparator_choice_complete_facility_kwh"]
                    - row["complete_choice_complete_facility_kwh"]
                )
                comparator_penalty = (
                    row["complete_choice_comparator_kwh"] - row["comparator_choice_comparator_kwh"]
                )
                row["complete_facility_penalty_of_comparator_choice_kwh"] = primary_penalty
                row["comparator_penalty_of_complete_choice_kwh"] = comparator_penalty
                row["complete_facility_cross_choice_gap_pct"] = (
                    100.0 * primary_penalty / row["complete_choice_complete_facility_kwh"]
                )
                row["comparator_cross_choice_gap_pct"] = (
                    100.0 * comparator_penalty / row["comparator_choice_comparator_kwh"]
                )
                maximum_gap = max(
                    abs(row["complete_facility_cross_choice_gap_pct"]),
                    abs(row["comparator_cross_choice_gap_pct"]),
                )
                row["maximum_cross_boundary_choice_gap_pct"] = maximum_gap
                for threshold in NEAR_TIE_THRESHOLDS_PCT:
                    token = str(threshold).replace(".", "p")
                    row[f"changed_within_{token}_pct"] = bool(
                        changed and maximum_gap <= threshold + 1e-12
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def selected_control_boundary_comparisons(selections: pd.DataFrame) -> pd.DataFrame:
    """Compare selected personal and average operations under each boundary."""
    keys = [*IDENTITY, "service_fraction"]
    rows: list[dict] = []
    for key, group in selections.groupby(keys, sort=True, dropna=False):
        uniform = group[group.control.eq("uniform_average")]
        if len(uniform) != 1:
            raise ValueError("A service cell must contain one Uniform average row")
        for control in ALTERNATIVES:
            alternative = group[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError(f"A service cell must contain one {control} row")
            record = dict(zip(keys, key, strict=True)) | {
                "alternative_control": control,
            }
            reference = uniform.iloc[0]
            personal = alternative.iloc[0]
            if reference.selection_status != "selected" or personal.selection_status != "selected":
                record["status"] = "no_common_selected_plan"
                rows.append(record)
                continue
            record["status"] = "matched"
            for label, column in (
                ("complete_facility", "complete_choice_complete_facility_kwh"),
                (
                    "electricity_plus_space_heating_gas",
                    "comparator_choice_comparator_kwh",
                ),
            ):
                reference_value = float(reference[column])
                personal_value = float(personal[column])
                delta = personal_value - reference_value
                record[f"uniform_{label}_kwh"] = reference_value
                record[f"alternative_{label}_kwh"] = personal_value
                record[f"delta_{label}_kwh"] = delta
                record[f"delta_{label}_pct"] = 100.0 * delta / reference_value
                record[f"direction_{label}"] = _direction(delta, reference_value)
            record["energy_direction_changed"] = bool(
                record["direction_complete_facility"]
                != record["direction_electricity_plus_space_heating_gas"]
            )
            rows.append(record)
    return pd.DataFrame(rows)


def background_gas_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much non-space-heating facility gas varies across candidates."""
    rows: list[dict] = []
    keys = [*IDENTITY, "control"]
    for key, group in metrics.groupby(keys, sort=True, dropna=False):
        residual = group.non_space_heating_facility_gas_kwh.astype(float)
        facility = group[PRIMARY_ENERGY].astype(float)
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "candidate_count": len(group),
                "non_space_heating_gas_min_kwh": float(residual.min()),
                "non_space_heating_gas_median_kwh": float(residual.median()),
                "non_space_heating_gas_max_kwh": float(residual.max()),
                "non_space_heating_gas_range_kwh": float(residual.max() - residual.min()),
                "range_as_pct_of_median_facility_energy": (
                    100.0 * float(residual.max() - residual.min()) / float(facility.median())
                ),
            }
        )
    return pd.DataFrame(rows)


def seasonal_boundary_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair completed seasonal-command operations under both energy boundaries."""
    keys = (
        "climate", "composition_id", "cohort_label", "seasonal_contrast_fraction",
        *(
            ("meteorological_block", "local_season")
            if {"meteorological_block", "local_season"}.issubset(frame)
            else ()
        ),
    )
    required = {
        *keys,
        "control",
        "run_id",
        "facility_gas_hhv_kwh",
        "gas_hhv_kwh",
        PRIMARY_ENERGY,
        COMPARATOR_ENERGY,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Seasonal metrics lack required columns: {sorted(missing)}")
    data = frame.copy()
    numeric = [
        "seasonal_contrast_fraction",
        "facility_gas_hhv_kwh",
        "gas_hhv_kwh",
        PRIMARY_ENERGY,
        COMPARATOR_ENERGY,
    ]
    for name in numeric:
        data[name] = pd.to_numeric(data[name], errors="raise")
    if not np.isfinite(data[numeric].to_numpy(float)).all():
        raise ValueError("Seasonal energy-boundary inputs must be finite")
    run_identity = (
        ["run_id", "meteorological_block"]
        if "meteorological_block" in data
        else ["run_id"]
    )
    if (
        data.duplicated([*keys, "control"]).any()
        or data.duplicated(run_identity).any()
    ):
        raise ValueError("Seasonal metrics duplicate comparison identities")
    residual = data.facility_gas_hhv_kwh - data.gas_hhv_kwh
    if residual.lt(-1e-6).any():
        raise ValueError("Seasonal facility gas cannot be below space-heating gas")
    boundary = data[PRIMARY_ENERGY] - data[COMPARATOR_ENERGY]
    if not np.allclose(residual, boundary, rtol=1e-9, atol=1e-6):
        raise ValueError("Seasonal boundary difference does not equal residual gas")
    data["non_space_heating_facility_gas_kwh"] = residual.clip(lower=0)
    rows: list[dict] = []
    for key, group in data.groupby(list(keys), sort=True, dropna=False):
        uniform = group[group.control.eq("uniform_average")]
        if len(uniform) != 1:
            raise ValueError("Every seasonal identity requires one Uniform average run")
        uniform = uniform.iloc[0]
        for control in ALTERNATIVES:
            alternative = group[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError(f"Every seasonal identity requires one {control} run")
            personal = alternative.iloc[0]
            row = dict(zip(keys, key, strict=True)) | {
                "alternative_control": control,
                "uniform_run_id": str(uniform.run_id),
                "alternative_run_id": str(personal.run_id),
            }
            for label, column in (
                ("complete_facility", PRIMARY_ENERGY),
                ("electricity_plus_space_heating_gas", COMPARATOR_ENERGY),
            ):
                reference = float(uniform[column])
                delta = float(personal[column] - uniform[column])
                row[f"delta_{label}_kwh"] = delta
                row[f"delta_{label}_pct"] = 100.0 * delta / reference
                row[f"direction_{label}"] = _direction(delta, reference)
            row["energy_direction_changed"] = bool(
                row["direction_complete_facility"]
                != row["direction_electricity_plus_space_heating_gas"]
            )
            row["delta_non_space_heating_facility_gas_kwh"] = float(
                personal.non_space_heating_facility_gas_kwh
                - uniform.non_space_heating_facility_gas_kwh
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _direction_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    matched = frame[frame.status.eq("matched")]
    return {
        direction: int(matched[column].eq(direction).sum())
        for direction in ("increase", "decrease", "tie")
    }


def analyse_energy_boundary_sensitivity(
    metrics_path: Path,
    anchors_path: Path,
    output_root: Path,
    *,
    expected_rows: int | None = 810,
    expected_climates: int | None = 9,
    seasonal_metrics_path: Path | None = None,
) -> dict:
    metrics_path = metrics_path.resolve()
    anchors_path = anchors_path.resolve()
    output_root = output_root.resolve()
    metrics = prepare_metrics(pd.read_csv(metrics_path))
    if expected_rows is not None and len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} annual candidates; found {len(metrics)}")
    climate_count = metrics.climate.astype(str).nunique()
    if expected_climates is not None and climate_count != expected_climates:
        raise ValueError(f"Expected {expected_climates} final climates; found {climate_count}")
    anchors = pd.read_csv(anchors_path)
    direct = direct_boundary_comparisons(metrics)
    selections = boundary_service_selections(metrics, anchors)
    selected_control = selected_control_boundary_comparisons(selections)
    background = background_gas_summary(metrics)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "direct": output_root / "GLOBAL_ENERGY_BOUNDARY_DIRECT_COMPARISONS.csv",
        "selections": output_root / "GLOBAL_ENERGY_BOUNDARY_SELECTIONS.csv",
        "selected_control": (
            output_root / "GLOBAL_ENERGY_BOUNDARY_SELECTED_CONTROL_COMPARISONS.csv"
        ),
        "background_gas": output_root / "GLOBAL_ENERGY_BOUNDARY_BACKGROUND_GAS.csv",
    }
    tables = {
        "direct": direct,
        "selections": selections,
        "selected_control": selected_control,
        "background_gas": background,
    }
    seasonal = None
    seasonal_reconstructed = False
    if seasonal_metrics_path is not None:
        seasonal_metrics_path = seasonal_metrics_path.resolve()
        seasonal_metrics = pd.read_csv(seasonal_metrics_path)
        seasonal_reconstructed = {
            "facility_gas_reference_run_id",
            "facility_gas_reference_sql_sha256",
        }.issubset(seasonal_metrics)
        seasonal = seasonal_boundary_comparisons(seasonal_metrics)
        outputs["seasonal_direct"] = output_root / "GLOBAL_ENERGY_BOUNDARY_SEASONAL_COMPARISONS.csv"
        tables["seasonal_direct"] = seasonal
    for name, table in tables.items():
        table.to_csv(outputs[name], index=False, float_format="%.12g")
    matched = direct[direct.status.eq("matched")]
    selected = selections[selections.selection_status.eq("selected")]
    selected_control_matched = selected_control[selected_control.status.eq("matched")]
    by_fraction = {}
    for fraction, group in selected.groupby("service_fraction", sort=True):
        by_fraction[f"{float(fraction):g}"] = {
            "selected_cells": len(group),
            "selection_changes": int(group.selection_changed.sum()),
            "changed_within_0p01_pct": int(group.changed_within_0p01_pct.sum()),
        }
    audit = {
        "status": "PASS",
        "definition": {
            "primary": "Electricity:Facility plus complete annual facility natural gas",
            "comparator": (
                "Electricity:Facility plus hourly Heating:NaturalGas; excludes facility "
                "gas outside the space-heating output variable"
            ),
            "residual": (
                "Complete facility gas minus Heating:NaturalGas. It includes Water "
                "Systems and any other facility gas outside the space-heating variable."
            ),
            "service_qualification_identical_between_boundaries": True,
        },
        "inputs": {
            "metrics": {"path": str(metrics_path), "sha256": sha256(metrics_path)},
            "anchors": {"path": str(anchors_path), "sha256": sha256(anchors_path)},
        },
        "counts": {
            "annual_candidates": len(metrics),
            "direct_pairs": len(direct),
            "matched_direct_pairs": len(matched),
            "service_selection_cells": len(selections),
            "selected_service_cells": len(selected),
            "selected_control_pairs": len(selected_control),
            "matched_selected_control_pairs": len(selected_control_matched),
        },
        "direct_energy_directions": {
            "complete_facility": _direction_counts(direct, "direction_complete_facility"),
            "electricity_plus_space_heating_gas": _direction_counts(
                direct, "direction_electricity_plus_space_heating_gas"
            ),
            "direction_changes": int(matched.energy_direction_changed.sum()),
        },
        "selection_sensitivity": {
            "changes": int(selected.selection_changed.sum()),
            "by_service_fraction": by_fraction,
            "near_tie_thresholds_pct": list(NEAR_TIE_THRESHOLDS_PCT),
            **{
                f"changed_within_{str(value).replace('.', 'p')}_pct": int(
                    selected[f"changed_within_{str(value).replace('.', 'p')}_pct"].sum()
                )
                for value in NEAR_TIE_THRESHOLDS_PCT
            },
        },
        "service_selected_energy_directions": {
            label: {
                f"{float(fraction):g}": {
                    direction: int(group[column].eq(direction).sum())
                    for direction in ("increase", "decrease", "tie")
                }
                for fraction, group in selected_control_matched.groupby(
                    "service_fraction", sort=True
                )
            }
            for label, column in (
                ("complete_facility", "direction_complete_facility"),
                (
                    "electricity_plus_space_heating_gas",
                    "direction_electricity_plus_space_heating_gas",
                ),
            )
        }
        | {"direction_changes": int(selected_control_matched.energy_direction_changed.sum())},
        "background_gas": {
            "median_candidate_group_range_kwh": float(
                background.non_space_heating_gas_range_kwh.median()
            ),
            "maximum_candidate_group_range_kwh": float(
                background.non_space_heating_gas_range_kwh.max()
            ),
            "maximum_range_pct_of_median_facility_energy": float(
                background.range_as_pct_of_median_facility_energy.max()
            ),
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path), "rows": len(tables[name])}
            for name, path in outputs.items()
        },
    }
    if seasonal_metrics_path is not None and seasonal is not None:
        audit["definition"]["seasonal_primary"] = (
            "Electricity:Facility plus seasonal Heating:NaturalGas plus matched "
            "offset-zero monthly WaterSystems:NaturalGas"
            if seasonal_reconstructed
            else "Electricity:Facility plus seasonal facility natural gas"
        )
        audit["inputs"]["seasonal_metrics"] = {
            "path": str(seasonal_metrics_path),
            "sha256": sha256(seasonal_metrics_path),
        }
        audit["seasonal_direct_energy_directions"] = {
            "pairs": len(seasonal),
            "complete_facility": {
                direction: int(seasonal.direction_complete_facility.eq(direction).sum())
                for direction in ("increase", "decrease", "tie")
            },
            "electricity_plus_space_heating_gas": {
                direction: int(
                    seasonal.direction_electricity_plus_space_heating_gas.eq(direction).sum()
                )
                for direction in ("increase", "decrease", "tie")
            },
            "direction_changes": int(seasonal.energy_direction_changed.sum()),
        }
    if {PRIMARY_CARBON, COMPARATOR_CARBON}.issubset(metrics.columns):
        audit["direct_carbon_directions"] = {
            "complete_facility": _direction_counts(direct, "direction_complete_facility_carbon"),
            "electricity_plus_space_heating_gas": _direction_counts(
                direct, "direction_electricity_plus_space_heating_gas_carbon"
            ),
            "direction_changes": int(matched.carbon_direction_changed.sum()),
        }
    audit_path = output_root / "GLOBAL_ENERGY_BOUNDARY_SENSITIVITY.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=810)
    parser.add_argument("--expected-climates", type=int, default=9)
    parser.add_argument("--seasonal-metrics", type=Path)
    args = parser.parse_args()
    audit = analyse_energy_boundary_sensitivity(
        args.metrics,
        args.anchors,
        args.output_root,
        expected_rows=args.expected_rows,
        expected_climates=args.expected_climates,
        seasonal_metrics_path=args.seasonal_metrics,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
