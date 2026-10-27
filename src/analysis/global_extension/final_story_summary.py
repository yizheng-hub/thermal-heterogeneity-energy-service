"""Assemble the final nine-climate evidence into auditable story tables.

This module is deliberately a thin reporting layer.  It never changes a
building model or rescales a simulated result.  Its strict default size gates
prevent an earlier four-context analysis from entering the final figures or
manuscript by mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CONTROLS = ("uniform_average", "personal_center", "personal_range")
ALTERNATIVES = ("personal_center", "personal_range")
TOLERANCES_C = (0.01, 0.025, 0.05, 0.075, 0.10)
LIMIT_FACTORS = (1.00, 1.05, 1.10, 1.15, 1.20)
EXPECTED_CLIMATES = 9
EXPECTED_COMPOSITIONS = 6
EXPECTED_ANNUAL_ROWS = 810
EXPECTED_DIRECT_ROWS = EXPECTED_CLIMATES * EXPECTED_COMPOSITIONS * len(ALTERNATIVES)
EXPECTED_STRICT_ROWS = EXPECTED_CLIMATES * EXPECTED_COMPOSITIONS * len(CONTROLS) * 2
EXPECTED_EVENT_IDS = EXPECTED_CLIMATES * 8
EXPECTED_EVENT_RUNS = EXPECTED_CLIMATES * EXPECTED_COMPOSITIONS * len(CONTROLS) * 5


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"{path.name} lacks {sorted(missing)}")
    return frame


def _read_json(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _bool(values: pd.Series, label: str) -> pd.Series:
    if values.dtype == bool:
        return values
    parsed = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise ValueError(f"{label} contains non-boolean values")
    return parsed.astype(bool)


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _distribution(values: pd.Series) -> dict[str, float | int | None]:
    clean = _finite(values)
    if clean.empty:
        return {"n": 0, "p10": None, "median": None, "p90": None}
    return {
        "n": int(len(clean)),
        "p10": float(clean.quantile(0.10)),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
    }


def _directions(values: pd.Series, tolerance: float = 1e-9) -> dict[str, int]:
    clean = _finite(values)
    return {
        "increase": int(clean.gt(tolerance).sum()),
        "decrease": int(clean.lt(-tolerance).sum()),
        "tie": int(clean.abs().le(tolerance).sum()),
    }


def validate_annual_design(
    metrics: pd.DataFrame,
    *,
    expected_rows: int = EXPECTED_ANNUAL_ROWS,
    expected_climates: int = EXPECTED_CLIMATES,
) -> dict:
    required = {
        "climate",
        "composition",
        "layout",
        "permission",
        "control",
        "offset_c",
        "run_id",
        "facility_purchased_kwh",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Annual metrics lack {sorted(missing)}")
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} annual candidates; found {len(metrics)}")
    climates = int(metrics.climate.astype(str).nunique())
    if climates != expected_climates:
        raise ValueError(f"Expected {expected_climates} climates; found {climates}")
    controls = set(metrics.control.astype(str))
    if controls != set(CONTROLS):
        raise ValueError(f"Unexpected controls: {sorted(controls)}")
    if metrics.composition.astype(str).nunique() != 6:
        raise ValueError("Expected six demand compositions")
    offsets = sorted(pd.to_numeric(metrics.offset_c).round(6).unique())
    if offsets != [-1.0, -0.5, 0.0, 0.5, 1.0]:
        raise ValueError(f"Expected five declared offsets; found {offsets}")
    if metrics.run_id.astype(str).duplicated().any():
        raise ValueError("Annual metrics duplicate a run_id")
    design_keys = ["climate", "composition", "control", "offset_c"]
    if metrics.duplicated(design_keys).any():
        raise ValueError("Annual metrics duplicate a climate/composition/control/offset cell")
    per_control = metrics.groupby(
        ["climate", "composition", "control"], sort=False
    ).size()
    if not per_control.eq(5).all():
        raise ValueError("Every annual climate/composition/control cell requires five offsets")
    if metrics.layout.astype(str).nunique() != 1:
        raise ValueError("The final annual design must use one fixed occupant layout")
    if metrics.permission.astype(str).nunique() != 1:
        raise ValueError("The final annual design must use one fixed command permission")
    return {
        "candidates": len(metrics),
        "climates": climates,
        "compositions": 6,
        "controls": 3,
        "offsets_c": offsets,
        "energy_boundary": "complete facility purchased energy",
    }


def summarise_information(information: pd.DataFrame) -> dict:
    required = {
        "climate",
        "composition",
        "service_fraction",
        "full_unresolved",
        "zone_mean_unresolved",
        "eligibility_disagreement",
        "selection_disagreement",
        "zone_mean_selected_plan_fails_full_review",
        "zone_mean_false_reject_no_selection",
    }
    missing = required - set(information)
    if missing:
        raise ValueError(f"Information diagnostic lacks {sorted(missing)}")
    frame = information.copy()
    for column in required - {"climate", "composition", "service_fraction"}:
        frame[column] = _bool(frame[column], column)
    primary_all = frame[np.isclose(pd.to_numeric(frame.service_fraction), 1.0)].copy()
    if len(primary_all) != 54:
        raise ValueError(
            "The primary information headline requires 54 climate-composition cells; "
            f"found {len(primary_all)}"
        )
    primary = primary_all.loc[~primary_all.full_unresolved].copy()
    if primary.empty:
        raise ValueError("No primary cell is resolvable using the full demand distribution")

    def one(group: pd.DataFrame) -> dict:
        distribution_required = group.eligibility_disagreement | group.selection_disagreement
        return {
            "cells": int(len(group)),
            "full_distribution_resolvable": int((~group.full_unresolved).sum()),
            "distribution_required": int(distribution_required.sum()),
            "eligibility_disagreement": int(group.eligibility_disagreement.sum()),
            "selection_disagreement": int(group.selection_disagreement.sum()),
            "zone_mean_selected_plan_failed_full_review": int(
                group.zone_mean_selected_plan_fails_full_review.sum()
            ),
            "zone_mean_false_rejection": int(
                group.zone_mean_false_reject_no_selection.sum()
            ),
        }

    sensitivities = {
        f"{float(fraction):g}": one(group)
        for fraction, group in frame.groupby("service_fraction", sort=True)
    }
    return {
        "headline_scope": (
            "service fraction 1.0; denominator restricted to cells with at least "
            "one plan qualified by full-demand review"
        ),
        "primary_all_cells": one(primary_all),
        "primary_full_distribution_resolvable": one(primary),
        "service_limit_sensitivity": sensitivities,
    }


def information_heating_association(
    information: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    expected_cells: int = EXPECTED_CLIMATES * EXPECTED_COMPOSITIONS,
) -> tuple[dict, pd.DataFrame]:
    """Describe where zone-mean selection failures occur along heating share."""
    metric_required = {
        "climate", "composition", "control", "offset_c", "heating_share",
    }
    if metric_required - set(metrics):
        raise ValueError(
            f"Annual metrics lack {sorted(metric_required - set(metrics))}"
        )
    keys = ["climate", "composition"]
    baseline = metrics[
        metrics.control.astype(str).eq("uniform_average")
        & np.isclose(pd.to_numeric(metrics.offset_c), 0.0)
    ][keys + ["heating_share"]].copy()
    if len(baseline) != expected_cells or baseline.duplicated(keys).any():
        raise ValueError(
            "Heating-context audit requires one offset-zero Uniform average row "
            "per climate/composition cell"
        )
    baseline["heating_share"] = pd.to_numeric(
        baseline.heating_share, errors="raise"
    )
    if (
        not np.isfinite(baseline.heating_share).all()
        or baseline.heating_share.lt(0).any()
        or baseline.heating_share.gt(1).any()
    ):
        raise ValueError("Uniform-average heating shares must lie within [0, 1]")
    enriched = information.merge(
        baseline, on=keys, how="left", validate="many_to_one"
    )
    if enriched.heating_share.isna().any():
        raise ValueError("An information-diagnostic cell lacks its heating context")
    primary = enriched[
        np.isclose(pd.to_numeric(enriched.service_fraction), 1.0)
    ].copy()
    if len(primary) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} primary information cells; found {len(primary)}"
        )
    failure = _bool(
        primary.zone_mean_selected_plan_fails_full_review,
        "zone_mean_selected_plan_fails_full_review",
    )
    failed = primary.loc[failure, "heating_share"]
    other = primary.loc[~failure, "heating_share"]

    def moments(values: pd.Series) -> dict[str, float | int | None]:
        clean = _finite(values)
        if clean.empty:
            return {"n": 0, "minimum": None, "mean": None, "maximum": None}
        return {
            "n": int(len(clean)),
            "minimum": float(clean.min()),
            "mean": float(clean.mean()),
            "maximum": float(clean.max()),
        }

    all_climates = sorted(primary.climate.astype(str).unique())
    failed_by_climate = (
        primary.loc[failure]
        .groupby("climate", sort=True)
        .size()
        .reindex(all_climates, fill_value=0)
    )
    return (
        {
            "scope": "service fraction 1.0; offset-zero Uniform average heating share",
            "relationship": (
                "association: zone-mean selected-plan failures are concentrated "
                "in contexts with high heating shares"
            ),
            "failed_selection_heating_share": moments(failed),
            "other_cells_heating_share": moments(other),
            "failed_selection_cells_by_climate": {
                str(climate): int(count)
                for climate, count in failed_by_climate.items()
            },
        },
        enriched,
    )


def summarise_direct(direct: pd.DataFrame) -> dict:
    required = {
        "climate",
        "composition",
        "alternative_control",
        "status",
        "delta_mean_reference_mismatch_c",
        "delta_fixed_tail_reference_mismatch_c",
        "delta_modeled_interval_person_hour_coverage",
        "delta_modeled_interval_mean_excess_c",
        "delta_facility_purchased_kwh",
        "delta_facility_purchased_pct",
        "delta_operational_co2_country_kg",
    }
    missing = required - set(direct)
    if missing:
        raise ValueError(f"Direct comparisons lack {sorted(missing)}")
    if len(direct) != EXPECTED_DIRECT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DIRECT_ROWS} direct personal-versus-uniform rows; "
            f"found {len(direct)}"
        )
    keys = ["climate", "composition", "alternative_control"]
    if direct.duplicated(keys).any():
        raise ValueError("Direct comparisons duplicate a climate/composition/control cell")
    if direct.climate.astype(str).nunique() != EXPECTED_CLIMATES:
        raise ValueError("Direct comparisons must cover all nine climates")
    if direct.composition.astype(str).nunique() != EXPECTED_COMPOSITIONS:
        raise ValueError("Direct comparisons must cover all six demand compositions")
    if set(direct.alternative_control.astype(str)) != set(ALTERNATIVES):
        raise ValueError("Direct comparisons must contain both demand-guided controls")
    matched = direct.loc[direct.status.eq("matched")].copy()

    def one(group: pd.DataFrame) -> dict:
        return {
            "rows": int(len(group)),
            "population_mismatch_improved": int(
                pd.to_numeric(group.delta_mean_reference_mismatch_c).lt(0).sum()
            ),
            "least_served_group_mismatch_improved": int(
                pd.to_numeric(group.delta_fixed_tail_reference_mismatch_c).lt(0).sum()
            ),
            "interval_coverage_improved": int(
                pd.to_numeric(group.delta_modeled_interval_person_hour_coverage).gt(0).sum()
            ),
            "interval_excess_improved": int(
                pd.to_numeric(group.delta_modeled_interval_mean_excess_c).lt(0).sum()
            ),
            "facility_energy_direction": _directions(group.delta_facility_purchased_kwh),
            "facility_energy_change_pct": _distribution(
                group.delta_facility_purchased_pct
            ),
            "operational_carbon_direction": _directions(
                group.delta_operational_co2_country_kg
            ),
        }

    by_control = {
        str(control): one(group)
        for control, group in matched.groupby("alternative_control", sort=True)
    }
    return {
        "available_rows": int(len(direct)),
        "matched_rows": int(len(matched)),
        "all_personal_controls": one(matched),
        "by_personal_control": by_control,
    }


def summarise_strict_feasibility(strict: pd.DataFrame) -> dict:
    required = {
        "climate",
        "composition",
        "control",
        "service_fraction",
        "selection_status",
    }
    missing = required - set(strict)
    if missing:
        raise ValueError(f"Strict-service selections lack {sorted(missing)}")
    expected = EXPECTED_STRICT_ROWS
    if len(strict) != expected:
        raise ValueError(f"Expected {expected} strict-service control cells; found {len(strict)}")
    keys = ["climate", "composition", "control", "service_fraction"]
    if strict.duplicated(keys).any():
        raise ValueError("Strict-service selections duplicate a design cell")
    if strict.climate.astype(str).nunique() != EXPECTED_CLIMATES:
        raise ValueError("Strict-service selections must cover all nine climates")
    if strict.composition.astype(str).nunique() != EXPECTED_COMPOSITIONS:
        raise ValueError("Strict-service selections must cover all six demand compositions")
    if set(strict.control.astype(str)) != set(CONTROLS):
        raise ValueError("Strict-service selections contain unexpected controls")
    fractions = sorted(pd.to_numeric(strict.service_fraction).unique())
    if not np.allclose(fractions, (0.5, 0.75)):
        raise ValueError(f"Expected strict-service fractions 0.5 and 0.75; found {fractions}")
    result: dict[str, dict] = {}
    for fraction, group in strict.groupby("service_fraction", sort=True):
        selected = group.selection_status.eq("selected")
        available = group.assign(_selected=selected).pivot_table(
            index=["climate", "composition"],
            columns="control",
            values="_selected",
            aggfunc="first",
        )
        for control in CONTROLS:
            if control not in available:
                available[control] = False
        demand_guided = available[list(ALTERNATIVES)].any(axis=1)
        result[f"{float(fraction):g}"] = {
            "control_cells_selected": {
                control: int((selected & group.control.eq(control)).sum())
                for control in CONTROLS
            },
            "scenario_cells": int(len(available)),
            "demand_guided_available_uniform_unavailable": int(
                (demand_guided & ~available.uniform_average.astype(bool)).sum()
            ),
        }
    return result


def summarise_matched_service(closest: pd.DataFrame) -> dict:
    # The tolerance column tokens are formatted to three decimals (0p010, ...).
    required = {
        "climate",
        "alternative_control",
        "energy_scope",
        "delta_energy_pct",
        *(f"within_tolerance_{value:.3f}_c".replace(".", "p") for value in TOLERANCES_C),
    }
    missing = required - set(closest)
    if missing:
        raise ValueError(f"Matched-service table lacks {sorted(missing)}")
    if len(closest) != EXPECTED_DIRECT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DIRECT_ROWS} matched-service comparison rows; "
            f"found {len(closest)}"
        )
    keys = ["climate", "composition", "alternative_control"]
    if closest.duplicated(keys).any():
        raise ValueError("Matched-service comparisons duplicate a design cell")
    if closest.climate.astype(str).nunique() != EXPECTED_CLIMATES:
        raise ValueError("Matched-service analysis must cover all nine climates")
    if closest.composition.astype(str).nunique() != EXPECTED_COMPOSITIONS:
        raise ValueError("Matched-service analysis must cover all six demand compositions")
    if set(closest.alternative_control.astype(str)) != set(ALTERNATIVES):
        raise ValueError("Matched-service analysis must contain both demand-guided controls")
    scopes = set(closest.energy_scope.astype(str))
    if scopes != {"complete_facility_purchased_energy"}:
        raise ValueError(f"Matched-service analysis uses the wrong energy boundary: {scopes}")
    result: dict[str, dict] = {}
    for tolerance in TOLERANCES_C:
        column = f"within_tolerance_{tolerance:.3f}_c".replace(".", "p")
        eligible = closest.loc[_bool(closest[column], column)].copy()
        result[f"{tolerance:g}"] = {
            "matched_rows": int(len(eligible)),
            "energy_direction": _directions(eligible.delta_energy_pct),
            "energy_change_pct": _distribution(eligible.delta_energy_pct),
        }
    return {
        "comparison_rows": int(len(closest)),
        "energy_boundary": "complete facility purchased energy",
        "service_gap_tolerance_c": result,
    }


def seasonal_reallocation(
    annual: pd.DataFrame,
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["climate", "composition_id", "seasonal_contrast_fraction"]
    annual_required = {
        *keys,
        "run_id",
        "cohort_label",
        "demand_source",
        "control",
        "facility_purchased_kwh",
    }
    season_required = {
        *keys,
        "run_id",
        "control",
        "local_season",
        "occupied_native_person_hours",
        "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c",
        "facility_purchased_kwh",
    }
    if annual_required - set(annual):
        raise ValueError(f"Seasonal annual metrics lack {sorted(annual_required - set(annual))}")
    if season_required - set(seasons):
        raise ValueError(f"Seasonal block metrics lack {sorted(season_required - set(seasons))}")
    if len(annual) != 324:
        raise ValueError(f"Expected 324 seasonal-command annual rows; found {len(annual)}")
    if annual.run_id.astype(str).duplicated().any():
        raise ValueError("Complete seasonal annual metrics duplicate a run_id")
    if len(seasons) != 4 * len(annual):
        raise ValueError(
            f"Expected {4 * len(annual)} complete seasonal block rows; found {len(seasons)}"
        )
    if seasons.duplicated(["run_id", "local_season"]).any():
        raise ValueError("Complete seasonal metrics duplicate a run/season block")
    if set(annual.run_id.astype(str)) != set(seasons.run_id.astype(str)):
        raise ValueError("Complete annual and seasonal energy tables contain different runs")
    annual_energy = annual.set_index("run_id").facility_purchased_kwh.astype(float)
    block_energy = seasons.groupby("run_id").facility_purchased_kwh.sum().astype(float)
    mismatch = (annual_energy.sort_index() - block_energy.sort_index()).abs()
    tolerance = np.maximum(1e-6, annual_energy.sort_index().abs() * 1e-9)
    if (mismatch > tolerance).any():
        raise ValueError("Complete seasonal blocks do not reproduce annual facility energy")
    service_rows: list[dict[str, float | str]] = []
    for run_id, group in seasons.groupby("run_id", sort=False):
        weights = pd.to_numeric(group.occupied_native_person_hours, errors="raise")
        if (weights <= 0).any() or float(weights.sum()) <= 0:
            raise ValueError("Seasonal service aggregation needs positive person-hours")
        service_rows.append({
            "run_id": str(run_id),
            "mean_reference_mismatch_c": float(
                np.average(
                    pd.to_numeric(group.mean_reference_mismatch_c, errors="raise"),
                    weights=weights,
                )
            ),
            "fixed_tail_reference_mismatch_c": float(
                np.average(
                    pd.to_numeric(group.fixed_tail_reference_mismatch_c, errors="raise"),
                    weights=weights,
                )
            ),
        })
    annual_service = pd.DataFrame(service_rows).set_index("run_id")
    rows: list[dict] = []
    for key, group in annual.groupby(keys, sort=True):
        base = group.loc[group.control.eq("uniform_average")]
        if len(base) != 1:
            raise ValueError("A seasonal cell lacks one Uniform average row")
        base_row = base.iloc[0]
        base_service = annual_service.loc[str(base_row.run_id)]
        season_group = seasons
        for name, value in zip(keys, key, strict=True):
            season_group = season_group.loc[season_group[name].eq(value)]
        base_seasons = season_group.loc[season_group.control.eq("uniform_average")].set_index(
            "local_season"
        )
        if set(base_seasons.index.astype(str)) != {"winter", "spring", "summer", "autumn"}:
            raise ValueError("A seasonal cell does not contain four local seasons")
        for control in ALTERNATIVES:
            alternative = group.loc[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError(f"A seasonal cell lacks one {control} row")
            alternative_row = alternative.iloc[0]
            alternative_service = annual_service.loc[str(alternative_row.run_id)]
            alternative_seasons = season_group.loc[
                season_group.control.eq(control)
            ].set_index("local_season")
            record = dict(zip(keys, key, strict=True)) | {
                "cohort_label": base_row.cohort_label,
                "demand_source": base_row.demand_source,
                "alternative_control": control,
                "delta_annual_facility_purchased_kwh": float(
                    alternative_row.facility_purchased_kwh - base_row.facility_purchased_kwh
                ),
                "delta_annual_facility_purchased_pct": float(
                    100
                    * (alternative_row.facility_purchased_kwh - base_row.facility_purchased_kwh)
                    / base_row.facility_purchased_kwh
                ),
                "delta_annual_mean_reference_mismatch_c": float(
                    alternative_service.mean_reference_mismatch_c
                    - base_service.mean_reference_mismatch_c
                ),
                "delta_annual_fixed_tail_reference_mismatch_c": float(
                    alternative_service.fixed_tail_reference_mismatch_c
                    - base_service.fixed_tail_reference_mismatch_c
                ),
            }
            deltas: list[float] = []
            for season in ("winter", "spring", "summer", "autumn"):
                base_value = float(
                    base_seasons.loc[season, "facility_purchased_kwh"]
                )
                alternative_value = float(
                    alternative_seasons.loc[season, "facility_purchased_kwh"]
                )
                delta = alternative_value - base_value
                deltas.append(delta)
                record[f"delta_{season}_facility_purchased_kwh"] = delta
            gross = float(sum(abs(value) for value in deltas))
            net = float(abs(sum(deltas)))
            record["mixed_seasonal_energy_signs"] = bool(
                any(value > 1e-9 for value in deltas)
                and any(value < -1e-9 for value in deltas)
            )
            record["seasonal_energy_cancellation_fraction"] = (
                0.0 if gross <= 1e-9 else max(0.0, min(1.0, 1.0 - net / gross))
            )
            rows.append(record)
    result = pd.DataFrame(rows)
    if len(result) != 216:
        raise AssertionError(f"Expected 216 seasonal personal-control pairs; found {len(result)}")
    return result


def summarise_seasonal(frame: pd.DataFrame) -> dict:
    return {
        "pairs": int(len(frame)),
        "annual_service_direction": {
            "population_mean_mismatch_lower": int(
                frame.delta_annual_mean_reference_mismatch_c.lt(-1e-9).sum()
            ),
            "population_mean_mismatch_higher": int(
                frame.delta_annual_mean_reference_mismatch_c.gt(1e-9).sum()
            ),
            "fixed_tail_mismatch_lower": int(
                frame.delta_annual_fixed_tail_reference_mismatch_c.lt(-1e-9).sum()
            ),
            "fixed_tail_mismatch_higher": int(
                frame.delta_annual_fixed_tail_reference_mismatch_c.gt(1e-9).sum()
            ),
        },
        "annual_complete_facility_energy_direction": _directions(
            frame.delta_annual_facility_purchased_kwh
        ),
        "annual_complete_facility_energy_change_pct": _distribution(
            frame.delta_annual_facility_purchased_pct
        ),
        "mixed_seasonal_energy_signs": int(frame.mixed_seasonal_energy_signs.sum()),
        "seasonal_energy_cancellation_fraction": _distribution(
            frame.seasonal_energy_cancellation_fraction
        ),
        "cancellation_above_0p50": int(
            frame.seasonal_energy_cancellation_fraction.gt(0.50).sum()
        ),
        "cancellation_above_0p75": int(
            frame.seasonal_energy_cancellation_fraction.gt(0.75).sum()
        ),
        "seasonal_energy_boundary": (
            "complete facility purchased energy: facility electricity plus "
            "space-heating gas and matched seasonal water-system gas"
        ),
    }


def validate_complete_seasonal_energy(
    *,
    source_seasonal_path: Path,
    complete_annual_path: Path,
    complete_seasonal_path: Path,
    audit: dict,
) -> None:
    """Bind complete seasonal energy tables to their audited source and hashes."""
    if audit.get("status") != "PASS":
        raise ValueError("Complete seasonal facility-energy audit did not pass")
    if audit.get("source_seasonal_metrics_sha256") != sha256(source_seasonal_path):
        raise ValueError("Complete seasonal energy is not bound to the active source metrics")
    declared = audit.get("outputs")
    if not isinstance(declared, dict):
        raise ValueError("Complete seasonal facility-energy audit lacks output hashes")
    for path in (complete_annual_path, complete_seasonal_path):
        record = declared.get(path.name)
        if not isinstance(record, dict) or record.get("sha256") != sha256(path):
            raise ValueError(f"Complete seasonal energy hash mismatch: {path}")


def spatial_effects(permission: pd.DataFrame, placement: pd.DataFrame) -> pd.DataFrame:
    required = {
        "climate",
        "composition",
        "control",
        "axis",
        "reference",
        "alternative",
        "status",
        "delta_mean_reference_mismatch_c",
        "delta_fixed_tail_reference_mismatch_c",
        "delta_facility_purchased_kwh",
        "delta_facility_purchased_pct",
    }
    for label, frame in (("permission", permission), ("placement", placement)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} comparisons lack {sorted(missing)}")
    result = pd.concat(
        [permission.assign(effect_type="permission"), placement.assign(effect_type="placement")],
        ignore_index=True,
    )
    return result


def summarise_spatial(
    effects: pd.DataFrame,
    spatial_audit: dict,
    *,
    expected_logical: int = 540,
    expected_physical: int = 516,
    expected_scored_rows: int = 972,
) -> dict:
    if spatial_audit.get("logical_new_designs") != expected_logical:
        raise ValueError(
            f"Expected {expected_logical} new spatial logical designs; found "
            f"{spatial_audit.get('logical_new_designs')}"
        )
    physical_runs = int(spatial_audit.get("distinct_new_physical_runs", 0))
    if physical_runs != expected_physical:
        raise ValueError(
            f"Expected {expected_physical} new spatial physical runs; found "
            f"{physical_runs}"
        )
    if not 0 < physical_runs <= expected_logical:
        raise ValueError(
            "Spatial physical-run deduplication produced an invalid run count"
        )
    if spatial_audit.get("logical_metric_rows") != expected_scored_rows:
        raise ValueError(
            f"Expected {expected_scored_rows} scored spatial configuration rows"
        )
    matched = effects.loc[effects.status.eq("matched")].copy()
    result: dict[str, dict] = {}
    for key, group in matched.groupby(
        ["effect_type", "reference", "alternative", "control"], sort=True
    ):
        label = "/".join(map(str, key))
        result[label] = {
            "pairs": int(len(group)),
            "population_mismatch_direction": _directions(
                group.delta_mean_reference_mismatch_c
            ),
            "least_served_group_mismatch_direction": _directions(
                group.delta_fixed_tail_reference_mismatch_c
            ),
            "facility_energy_direction": _directions(group.delta_facility_purchased_kwh),
            "facility_energy_change_pct": _distribution(
                group.delta_facility_purchased_pct
            ),
        }
    return {
        "new_logical_designs": expected_logical,
        "new_physical_runs": physical_runs,
        "scored_configuration_rows": expected_scored_rows,
        "matched_effect_rows": int(len(matched)),
        "by_effect": result,
    }


def summarise_events(
    scores: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    expected_events: int = EXPECTED_EVENT_IDS,
    expected_runs: int = EXPECTED_EVENT_RUNS,
    expected_climates: int = EXPECTED_CLIMATES,
    expected_actions: int = 5,
    expected_controls: tuple[str, ...] = CONTROLS,
) -> tuple[dict, pd.DataFrame]:
    required = {
        "run_id",
        "event_id",
        "climate",
        "composition_id",
        "control",
        "cohort_label",
        "action_id",
        "action_family",
        "local_season",
        "load_level",
        "event_service_limit_factor",
        "service_qualified",
        "positive_event_response",
        "positive_full_window_response",
        "qualified_positive_event_response",
        "qualified_positive_full_window_response",
        "event_mean_power_reduction_kw",
        "event_electricity_reduction_kwh",
        "recovery_electricity_change_kwh",
        "precondition_event_recovery_net_reduction_kwh",
        "event_facility_mean_power_reduction_kw",
        "event_facility_electricity_reduction_kwh",
        "recovery_facility_electricity_change_kwh",
        "precondition_event_recovery_facility_net_reduction_kwh",
    }
    missing = required - set(scores)
    if missing:
        raise ValueError(f"Event scores lack {sorted(missing)}")
    protocol_required = {"event_id", "climate", "local_season", "load_level"}
    if protocol_required - set(protocol):
        raise ValueError(
            f"Event protocol lacks {sorted(protocol_required - set(protocol))}"
        )
    if len(protocol) != expected_events or protocol.event_id.astype(str).duplicated().any():
        raise ValueError("Event protocol must contain one row per declared event")
    if protocol.event_id.astype(str).nunique() != expected_events:
        raise ValueError(
            f"Expected {expected_events} distinct protocol events; found "
            f"{protocol.event_id.astype(str).nunique()}"
        )
    if scores.event_id.astype(str).nunique() != expected_events:
        raise ValueError("Event scores do not cover all declared events")
    if scores.run_id.astype(str).nunique() != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} event-action trajectories; found "
            f"{scores.run_id.astype(str).nunique()}"
        )
    factors = sorted(pd.to_numeric(scores.event_service_limit_factor).unique())
    if not np.allclose(factors, LIMIT_FACTORS):
        raise ValueError(f"Expected event service factors {LIMIT_FACTORS}; found {factors}")
    expected_rows = expected_runs * 8 * len(LIMIT_FACTORS)
    if len(scores) != expected_rows:
        raise ValueError(f"Expected {expected_rows} scored event rows; found {len(scores)}")
    score_keys = ["run_id", "event_id", "event_service_limit_factor"]
    if scores.duplicated(score_keys).any():
        raise ValueError("Event scores duplicate a run/event/service-factor cell")
    per_run = scores.groupby("run_id", sort=False).size()
    if not per_run.eq(8 * len(LIMIT_FACTORS)).all():
        raise ValueError("Every response trajectory requires eight events at five service factors")
    if scores.climate.astype(str).nunique() != expected_climates:
        raise ValueError(
            f"Event scores must cover {expected_climates} climates"
        )
    if set(scores.control.astype(str)) != set(expected_controls):
        raise ValueError("Event scores contain unexpected controls")
    if scores.action_id.astype(str).nunique() != expected_actions:
        raise ValueError(
            f"Event scores must contain {expected_actions} declared response actions"
        )
    protocol_lookup = protocol.set_index(protocol.event_id.astype(str))
    score_event_ids = scores.event_id.astype(str)
    expected_climate = score_event_ids.map(protocol_lookup.climate.astype(str))
    if expected_climate.isna().any() or not expected_climate.eq(
        scores.climate.astype(str)
    ).all():
        raise ValueError("Event scores do not preserve the protocol climate for each event")
    for column in ("local_season", "load_level"):
        expected_value = score_event_ids.map(protocol_lookup[column].astype(str))
        if expected_value.isna().any() or not expected_value.eq(
            scores[column].astype(str)
        ).all():
            raise ValueError(f"Event scores do not preserve protocol {column}")
    frame = scores.copy()
    for column in (
        "positive_event_response",
        "qualified_positive_event_response",
        "qualified_positive_full_window_response",
    ):
        frame[column] = _bool(frame[column], column)
    frame["recovery_fraction_of_event_saving"] = np.where(
        pd.to_numeric(frame.event_facility_electricity_reduction_kwh).gt(1e-9),
        pd.to_numeric(frame.recovery_facility_electricity_change_kwh)
        / pd.to_numeric(frame.event_facility_electricity_reduction_kwh),
        np.nan,
    )
    group_columns = [
        "event_service_limit_factor",
        "climate",
        "local_season",
        "load_level",
        "control",
        "action_id",
        "action_family",
    ]
    rows: list[dict] = []
    for key, group in frame.groupby(group_columns, sort=True, dropna=False):
        event_qualified = group.loc[group.qualified_positive_event_response]
        full_window_qualified = group.loc[
            group.qualified_positive_full_window_response
        ]
        rows.append(
            dict(zip(group_columns, key, strict=True))
            | {
                "evaluated_rows": len(group),
                "distinct_events": int(group.event_id.astype(str).nunique()),
                "raw_positive_event_rows": int(group.positive_event_response.sum()),
                "service_qualified_positive_event_rows": int(
                    group.qualified_positive_event_response.sum()
                ),
                "service_qualified_positive_full_window_rows": int(
                    group.qualified_positive_full_window_response.sum()
                ),
                "event_qualified_median_event_power_reduction_kw": (
                    float(
                        event_qualified.event_facility_mean_power_reduction_kw.median()
                    )
                    if len(event_qualified)
                    else np.nan
                ),
                "full_window_qualified_median_event_power_reduction_kw": (
                    float(
                        full_window_qualified.event_facility_mean_power_reduction_kw.median()
                    )
                    if len(full_window_qualified)
                    else np.nan
                ),
                "full_window_qualified_median_net_reduction_kwh": (
                    float(
                        full_window_qualified.precondition_event_recovery_facility_net_reduction_kwh.median()
                    )
                    if len(full_window_qualified)
                    else np.nan
                ),
                "full_window_qualified_median_recovery_fraction": (
                    float(
                        full_window_qualified.recovery_fraction_of_event_saving.median()
                    )
                    if len(full_window_qualified)
                    else np.nan
                ),
            }
        )
    tidy = pd.DataFrame(rows)
    by_factor = {}
    for factor, group in frame.groupby("event_service_limit_factor", sort=True):
        event_qualified = group.loc[group.qualified_positive_event_response]
        full_window_qualified = group.loc[
            group.qualified_positive_full_window_response
        ]
        by_factor[f"{float(factor):g}"] = {
            "evaluated_rows": int(len(group)),
            "raw_positive_event_rows": int(group.positive_event_response.sum()),
            "service_qualified_positive_event_rows": int(
                group.qualified_positive_event_response.sum()
            ),
            "service_qualified_positive_full_window_rows": int(
                group.qualified_positive_full_window_response.sum()
            ),
            "event_qualified_power_reduction_kw": _distribution(
                event_qualified.event_facility_mean_power_reduction_kw
            ),
            "full_window_qualified_event_power_reduction_kw": _distribution(
                full_window_qualified.event_facility_mean_power_reduction_kw
            ),
            "full_window_qualified_net_reduction_kwh": _distribution(
                full_window_qualified.precondition_event_recovery_facility_net_reduction_kwh
            ),
            "full_window_qualified_recovery_fraction": _distribution(
                full_window_qualified.recovery_fraction_of_event_saving
            ),
        }
    return (
        {
            "protocol_events": expected_events,
            "action_trajectories": expected_runs,
            "scored_rows": int(len(frame)),
            "service_allowance_sensitivity": by_factor,
        },
        tidy,
    )


def _selected_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame]].copy()


def build_final_story_summary(
    *,
    annual_root: Path,
    matched_root: Path,
    energy_boundary_root: Path,
    seasonal_root: Path,
    seasonal_energy_root: Path,
    spatial_root: Path,
    event_scores_path: Path,
    event_protocol_path: Path,
    personal_range_audit_path: Path,
    output_root: Path,
    expected_annual_rows: int = EXPECTED_ANNUAL_ROWS,
    expected_climates: int = EXPECTED_CLIMATES,
    expected_spatial_logical: int = 540,
    expected_spatial_physical: int = 516,
    expected_events: int = EXPECTED_EVENT_IDS,
    expected_event_runs: int = EXPECTED_EVENT_RUNS,
) -> dict:
    roots = {
        "annual": annual_root.resolve(),
        "matched": matched_root.resolve(),
        "energy_boundary": energy_boundary_root.resolve(),
        "seasonal": seasonal_root.resolve(),
        "seasonal_energy": seasonal_energy_root.resolve(),
        "spatial": spatial_root.resolve(),
    }
    paths = {
        "annual_metrics": roots["annual"] / "GLOBAL_RUN_METRICS.csv",
        "direct": roots["annual"] / "GLOBAL_DIRECT_COMPARISONS.csv",
        "information": roots["annual"] / "GLOBAL_INFORMATION_DIAGNOSTIC.csv",
        "strict": roots["annual"] / "GLOBAL_STRICT_SERVICE_FEASIBILITY.csv",
        "matched": roots["matched"] / "MATCHED_REALIZED_SERVICE_CLOSEST_PAIRS.csv",
        "energy_boundary": (
            roots["energy_boundary"] / "GLOBAL_ENERGY_BOUNDARY_SENSITIVITY.json"
        ),
        "seasonal_source_blocks": (
            roots["seasonal"] / "SEASONAL_COMMAND_SEASON_METRICS.csv"
        ),
        "seasonal_annual": (
            roots["seasonal_energy"]
            / "SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv"
        ),
        "seasonal_blocks": (
            roots["seasonal_energy"]
            / "SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv"
        ),
        "seasonal_energy_audit": (
            roots["seasonal_energy"] / "SEASONAL_FACILITY_ENERGY_AUDIT.json"
        ),
        "spatial_permission": roots["spatial"] / "GLOBAL_PERMISSION_COMPARISONS.csv",
        "spatial_placement": roots["spatial"] / "GLOBAL_PLACEMENT_COMPARISONS.csv",
        "spatial_audit": roots["spatial"] / "GLOBAL_SPATIAL_AUDIT.json",
        "event_scores": event_scores_path.resolve(),
        "event_protocol": event_protocol_path.resolve(),
        "personal_range": personal_range_audit_path.resolve(),
    }
    metrics = _read_csv(paths["annual_metrics"], ["facility_purchased_kwh"])
    direct = _read_csv(paths["direct"], ["alternative_control", "status"])
    information = _read_csv(paths["information"], ["service_fraction"])
    strict = _read_csv(paths["strict"], ["selection_status"])
    matched = _read_csv(paths["matched"], ["energy_scope"])
    energy_boundary = _read_json(paths["energy_boundary"])
    source_seasonal_path = paths["seasonal_source_blocks"].resolve()
    if not source_seasonal_path.is_file():
        raise FileNotFoundError(source_seasonal_path)
    seasonal_annual = _read_csv(paths["seasonal_annual"], ["control"])
    seasonal_blocks = _read_csv(paths["seasonal_blocks"], ["local_season"])
    seasonal_energy_audit = _read_json(paths["seasonal_energy_audit"])
    validate_complete_seasonal_energy(
        source_seasonal_path=source_seasonal_path,
        complete_annual_path=paths["seasonal_annual"].resolve(),
        complete_seasonal_path=paths["seasonal_blocks"].resolve(),
        audit=seasonal_energy_audit,
    )
    permission = _read_csv(paths["spatial_permission"], ["axis"])
    placement = _read_csv(paths["spatial_placement"], ["axis"])
    spatial_audit = _read_json(paths["spatial_audit"])
    event_scores = _read_csv(paths["event_scores"], ["event_id"])
    event_protocol = _read_csv(paths["event_protocol"], ["event_id"])
    personal_range = _read_json(paths["personal_range"])

    design = validate_annual_design(
        metrics,
        expected_rows=expected_annual_rows,
        expected_climates=expected_climates,
    )
    if energy_boundary.get("counts", {}).get("annual_candidates") != expected_annual_rows:
        raise ValueError("Energy-boundary audit does not cover the final annual candidate set")
    information_heating, information = information_heating_association(
        information,
        metrics,
        expected_cells=expected_climates * EXPECTED_COMPOSITIONS,
    )
    seasonal = seasonal_reallocation(seasonal_annual, seasonal_blocks)
    effects = spatial_effects(permission, placement)
    event_summary, event_tidy = summarise_events(
        event_scores,
        event_protocol,
        expected_events=expected_events,
        expected_runs=expected_event_runs,
        expected_climates=expected_climates,
    )
    summary = {
        "status": "PASS",
        "design": design,
        "information": summarise_information(information),
        "information_heating_context": information_heating,
        "direct_service_and_resources": summarise_direct(direct),
        "strict_service_feasibility": summarise_strict_feasibility(strict),
        "matched_realised_service": summarise_matched_service(matched),
        "energy_boundary_sensitivity": {
            key: energy_boundary.get(key)
            for key in (
                "definition",
                "counts",
                "direct_energy_directions",
                "selection_sensitivity",
                "service_selected_energy_directions",
                "seasonal_direct_energy_directions",
                "direct_carbon_directions",
                "background_gas",
            )
            if key in energy_boundary
        },
        "seasonal_reallocation": summarise_seasonal(seasonal),
        "placement_and_permissions": summarise_spatial(
            effects,
            spatial_audit,
            expected_logical=expected_spatial_logical,
            expected_physical=expected_spatial_physical,
        ),
        "event_response_and_recovery": event_summary,
        "personal_range_audit": {
            key: personal_range.get(key)
            for key in (
                "status",
                "selected_fixed_half_width_c",
                "selection_rule",
                "variable_width_outperformed_best_fixed_all_outcomes",
                "heldout_width_comparison",
                "records",
            )
            if key in personal_range
        },
    }

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tidy_outputs = {
        "FINAL_INFORMATION_DIAGNOSTIC.csv": information,
        "FINAL_STRICT_SERVICE_FEASIBILITY.csv": strict,
        "FINAL_DIRECT_COMPARISONS.csv": _selected_columns(
            direct,
            (
                "climate",
                "composition",
                "layout",
                "permission",
                "alternative_control",
                "status",
                "delta_mean_reference_mismatch_c",
                "delta_fixed_tail_reference_mismatch_c",
                "delta_modeled_interval_person_hour_coverage",
                "delta_modeled_interval_mean_excess_c",
                "delta_facility_electric_kwh",
                "delta_facility_gas_hhv_kwh",
                "delta_facility_purchased_kwh",
                "delta_facility_purchased_pct",
                "delta_total_purchased_pct",
                "delta_operational_co2_country_kg",
            ),
        ),
        "FINAL_MATCHED_SERVICE.csv": matched,
        "FINAL_SEASONAL_REALLOCATION.csv": seasonal,
        "FINAL_SPATIAL_EFFECTS.csv": effects,
        "FINAL_EVENT_RESPONSE.csv": _selected_columns(
            event_scores,
            (
                "run_id",
                "event_id",
                "climate",
                "composition_id",
                "cohort_label",
                "control",
                "action_id",
                "action_family",
                "permission",
                "season_block",
                "local_season",
                "load_level",
                "event_service_limit_factor",
                "service_qualified",
                "positive_event_response",
                "positive_full_window_response",
                "qualified_positive_event_response",
                "qualified_positive_full_window_response",
                "event_mean_power_reduction_kw",
                "event_electricity_reduction_kwh",
                "recovery_electricity_change_kwh",
                "precondition_event_recovery_net_reduction_kwh",
                "event_facility_mean_power_reduction_kw",
                "event_facility_electricity_reduction_kwh",
                "recovery_facility_electricity_change_kwh",
                "precondition_event_recovery_facility_net_reduction_kwh",
            ),
        ),
        "FINAL_EVENT_RESPONSE_SUMMARY.csv": event_tidy,
    }
    for name, frame in tidy_outputs.items():
        frame.to_csv(output_root / name, index=False, float_format="%.12g")
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    summary["input_sha256"] = input_hashes
    summary["output_sha256"] = {
        name: sha256(output_root / name) for name in sorted(tidy_outputs)
    }
    summary_path = output_root / "FINAL_STORY_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "PASS",
        "inputs": input_hashes,
        "outputs": {
            path.name: sha256(path)
            for path in sorted([summary_path, *(output_root / name for name in tidy_outputs)])
        },
    }
    (output_root / "FINAL_STORY_HASHES.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--matched-root", type=Path, required=True)
    parser.add_argument("--energy-boundary-root", type=Path, required=True)
    parser.add_argument("--seasonal-root", type=Path, required=True)
    parser.add_argument("--seasonal-energy-root", type=Path, required=True)
    parser.add_argument("--spatial-root", type=Path, required=True)
    parser.add_argument("--event-scores", type=Path, required=True)
    parser.add_argument("--event-protocol", type=Path, required=True)
    parser.add_argument("--personal-range-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = build_final_story_summary(
        annual_root=args.annual_root,
        matched_root=args.matched_root,
        energy_boundary_root=args.energy_boundary_root,
        seasonal_root=args.seasonal_root,
        seasonal_energy_root=args.seasonal_energy_root,
        spatial_root=args.spatial_root,
        event_scores_path=args.event_scores,
        event_protocol_path=args.event_protocol,
        personal_range_audit_path=args.personal_range_audit,
        output_root=args.output_root,
    )
    print(json.dumps({"status": summary["status"], "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
