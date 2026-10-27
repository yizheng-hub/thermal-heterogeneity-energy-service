"""Prepare evidence tables for the global main-figure revision.

The functions in this module do not draw figures.  They convert audited global
simulation outputs into tidy, human-readable tables consumed by ``Plot.ipynb``.
Keeping the scientific aggregation here makes each notebook cell responsible
only for selecting styles and drawing one figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_LABELS = {
    "uniform_average": "Uniform average",
    "personal_center": "Personal centre",
    "personal_range": "Personal range",
}

ACTION_LABELS = {
    "uniform_shift": "Uniform shift",
    "precondition_uniform_shift": "Preconditioning + uniform shift",
    "service_protected_shift": "Service-protected shift",
    "precondition_service_protected_shift": (
        "Preconditioning + service-protected shift"
    ),
}


def _require(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def climate_metadata(path: Path, *, expected_climates: int = 9) -> pd.DataFrame:
    """Read the declared global climate matrix with one row per site."""
    content = json.loads(path.read_text(encoding="utf-8"))
    rows = content["climates"] if isinstance(content, dict) else content
    frame = pd.DataFrame(rows)
    required = {
        "climate",
        "country",
        "continent",
        "latitude",
        "longitude",
        "climate_zone",
    }
    _require(frame, required, "climate metadata")
    if frame.climate.duplicated().any():
        raise ValueError("Climate metadata must contain one row per climate")
    if len(frame) != expected_climates:
        raise ValueError(
            f"Expected {expected_climates} declared climates; found {len(frame)}"
        )
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Every climate requires map coordinates")
    return frame.sort_values(["continent", "climate"], kind="mergesort").reset_index(
        drop=True
    )


def observation_map_source(
    warm: pd.DataFrame,
    cold: pd.DataFrame,
    ctcd_records: pd.DataFrame,
    ctcd_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """Build one evidence-bearing map layer for both observational archives.

    The multinational archive retains its existing coarsened coordinates.  CTCD
    counts are joined to a separately versioned city-coordinate table; the
    coordinates are display positions and are never used in the analysis.
    """
    required_archive = {
        "city",
        "country",
        "latitude",
        "longitude",
        "supported",
        "analysis_units",
    }
    _require(warm, required_archive, "warm observation map")
    _require(cold, required_archive, "cold observation map")
    archive = pd.concat(
        [warm.assign(season="warm"), cold.assign(season="cold")],
        ignore_index=True,
    )
    archive["supported"] = archive.supported.astype(str).str.lower().eq("true")
    archive["analysis_units"] = pd.to_numeric(
        archive.analysis_units, errors="coerce"
    ).fillna(0)
    archive = archive[archive.supported].copy()
    # The same analysis unit may appear in both seasonal summaries.  The maximum
    # seasonal count is therefore used for marker area instead of summing twice.
    archive = (
        archive.groupby(
            ["city", "country", "latitude", "longitude"], as_index=False,
            sort=True,
        )
        .agg(observation_count=("analysis_units", "max"))
        .assign(source="Multinational archive", coordinate_role="coarsened display")
    )

    _require(ctcd_records, {"city_normalized"}, "CTCD records")
    if "active_conditioning" in ctcd_records:
        active = ctcd_records.active_conditioning
        if active.dtype != bool:
            active = active.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        if active.isna().any():
            raise ValueError("CTCD active-conditioning flags must be boolean")
        ctcd_records = ctcd_records.loc[active].copy()
    _require(
        ctcd_coordinates,
        {"city_normalized", "latitude", "longitude"},
        "CTCD coordinates",
    )
    ctcd_counts = (
        ctcd_records.groupby("city_normalized", as_index=False)
        .size()
        .rename(columns={"size": "observation_count"})
    )
    ctcd = ctcd_counts.merge(
        ctcd_coordinates,
        on="city_normalized",
        how="left",
        validate="one_to_one",
    )
    if ctcd[["latitude", "longitude"]].isna().any().any():
        missing = sorted(
            ctcd.loc[ctcd.latitude.isna(), "city_normalized"].astype(str).unique()
        )
        raise ValueError(f"CTCD map coordinates are missing for: {missing}")
    ctcd = ctcd.assign(
        city=ctcd.city_normalized.str.replace("_", " ").str.title(),
        country="China",
        source="China archive",
        coordinate_role="display city centre",
    )[
        [
            "city",
            "country",
            "latitude",
            "longitude",
            "observation_count",
            "source",
            "coordinate_role",
        ]
    ]
    result = pd.concat([archive, ctcd], ignore_index=True)
    if (result.observation_count <= 0).any():
        raise ValueError("Every mapped observation site must have positive support")
    return result.sort_values(["source", "country", "city"], kind="mergesort")


def global_effect_source(
    direct: pd.DataFrame,
    climates: pd.DataFrame,
    *,
    expected_climates: int = 9,
    expected_compositions: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw matched comparisons and climate-level display summaries."""
    required = {
        "climate",
        "composition",
        "layout",
        "permission",
        "alternative_control",
        "status",
        "delta_mean_reference_mismatch_c",
        "delta_fixed_tail_reference_mismatch_c",
        "delta_total_purchased_pct",
        "delta_operational_co2_country_kg",
    }
    _require(direct, required, "global direct comparisons")
    _require(
        climates,
        {"climate", "country", "continent", "climate_zone"},
        "climate metadata",
    )
    keys = ["climate", "composition", "alternative_control"]
    expected_rows = expected_climates * expected_compositions * 2
    if len(direct) != expected_rows or direct.duplicated(keys).any():
        raise ValueError(
            "Global direct comparisons must contain one row for each "
            "climate/composition/demand-guided-control cell"
        )
    if direct.climate.astype(str).nunique() != expected_climates:
        raise ValueError("Global direct comparisons do not cover the declared climates")
    if direct.composition.astype(str).nunique() != expected_compositions:
        raise ValueError("Global direct comparisons do not cover the declared compositions")
    matched = direct[direct.status.eq("matched")].copy()
    if matched.empty:
        raise ValueError("No matched global comparisons are available for plotting")
    matched["control"] = matched.alternative_control.map(CONTROL_LABELS)
    if matched.control.isna().any():
        unknown = sorted(matched.loc[matched.control.isna(), "alternative_control"].unique())
        raise ValueError(f"Unknown controls: {unknown}")
    matched["demand_source"] = np.where(
        matched.composition.astype(str).str.startswith("China"),
        "China archive",
        "Multinational archive",
    )
    matched["population_service_gain_c"] = -pd.to_numeric(
        matched.delta_mean_reference_mismatch_c
    )
    matched["least_served_gain_c"] = -pd.to_numeric(
        matched.delta_fixed_tail_reference_mismatch_c
    )
    matched = matched.merge(
        climates[["climate", "country", "continent", "climate_zone"]],
        on="climate",
        how="left",
        validate="many_to_one",
    )
    if matched[["country", "climate_zone"]].isna().any().any():
        raise ValueError("A matched comparison lacks climate metadata")

    metrics = {
        "population_service_gain_c": "Population service gain (°C)",
        "least_served_gain_c": "Least-served group gain (°C)",
        "delta_total_purchased_pct": "Annual energy change (%)",
        "delta_operational_co2_country_kg": "Operational carbon change (kgCO2e)",
    }
    summaries = []
    keys = ["climate", "country", "continent", "climate_zone", "control"]
    for key, group in matched.groupby(keys, sort=True):
        for column, label in metrics.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            summaries.append(
                dict(zip(keys, key, strict=True))
                | {
                    "metric": label,
                    "n": len(values),
                    "p10": values.quantile(0.10),
                    "median": values.median(),
                    "p90": values.quantile(0.90),
                }
            )
    return matched, pd.DataFrame(summaries)


def seasonal_effect_source(
    seasonal: pd.DataFrame,
    climates: pd.DataFrame,
    *,
    expected_climates: int = 9,
    expected_compositions: int = 6,
) -> pd.DataFrame:
    """Pair demand-guided runs with Uniform average within each local season."""
    required = {
        "climate",
        "composition",
        "layout",
        "permission",
        "control",
        "offset_c",
        "local_season",
        "total_purchased_kwh",
        "operational_co2_country_kg",
    }
    _require(seasonal, required, "global seasonal metrics")
    selected = seasonal[np.isclose(pd.to_numeric(seasonal.offset_c), 0.0)].copy()
    expected_rows = expected_climates * expected_compositions * 3 * 4
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} offset-zero seasonal rows; found {len(selected)}"
        )
    keys = [
        "climate",
        "composition",
        "layout",
        "permission",
        "local_season",
    ]
    reference = selected[selected.control.eq("uniform_average")][
        keys + ["total_purchased_kwh", "operational_co2_country_kg"]
    ].rename(
        columns={
            "total_purchased_kwh": "uniform_energy_kwh",
            "operational_co2_country_kg": "uniform_carbon_kg",
        }
    )
    alternatives = selected[selected.control.ne("uniform_average")].merge(
        reference, on=keys, how="left", validate="many_to_one"
    )
    if alternatives[["uniform_energy_kwh", "uniform_carbon_kg"]].isna().any().any():
        raise ValueError("A seasonal demand-guided run lacks its Uniform average pair")
    alternatives["control_label"] = alternatives.control.map(CONTROL_LABELS)
    alternatives["energy_change_kwh"] = (
        alternatives.total_purchased_kwh - alternatives.uniform_energy_kwh
    )
    alternatives["carbon_change_kg"] = (
        alternatives.operational_co2_country_kg - alternatives.uniform_carbon_kg
    )
    alternatives = alternatives.merge(
        climates[["climate", "latitude", "climate_zone"]],
        on="climate",
        how="left",
        validate="many_to_one",
    )
    return alternatives.sort_values(
        ["climate", "control", "local_season", "composition", "layout", "permission"],
        kind="mergesort",
    )

def seasonal_command_effect_source(
    commands: pd.DataFrame,
    annual_baseline: pd.DataFrame,
    climates: pd.DataFrame,
    *,
    expected_climates: int = 9,
    expected_compositions: int = 6,
) -> pd.DataFrame:
    """Compare physical seasonal commands with each control's annual command."""
    command_required = {
        "climate", "cohort_label", "control", "seasonal_contrast_fraction",
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "modeled_interval_person_hour_coverage", "total_purchased_kwh",
    }
    baseline_required = {
        "climate", "composition", "control", "offset_c",
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "modeled_interval_person_hour_coverage", "total_purchased_kwh",
    }
    _require(commands, command_required, "seasonal-command metrics")
    _require(annual_baseline, baseline_required, "annual command baseline")
    _require(
        climates,
        {"climate", "latitude", "country", "continent", "climate_zone"},
        "climate metadata",
    )
    expected_command_rows = expected_climates * expected_compositions * 3 * 2
    if len(commands) != expected_command_rows:
        raise ValueError(
            f"Expected {expected_command_rows} seasonal-command rows; found {len(commands)}"
        )
    command_keys = ["climate", "cohort_label", "control", "seasonal_contrast_fraction"]
    if commands.duplicated(command_keys).any():
        raise ValueError("Seasonal-command metrics duplicate a design cell")
    baseline = annual_baseline[
        np.isclose(pd.to_numeric(annual_baseline.offset_c), 0.0)
    ].copy()
    keys = ["climate", "cohort_label", "control"]
    baseline = baseline.rename(columns={"composition": "cohort_label"})
    metrics = (
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "modeled_interval_person_hour_coverage", "total_purchased_kwh",
    )
    baseline = baseline[keys + list(metrics)].rename(
        columns={name: f"annual_{name}" for name in metrics}
    )
    if baseline.duplicated(keys).any():
        raise ValueError("Annual command baseline duplicates a control/cohort")
    result = commands.merge(baseline, on=keys, how="left", validate="many_to_one")
    annual_columns = [f"annual_{name}" for name in metrics]
    if result[annual_columns].isna().any().any():
        raise ValueError("A seasonal command lacks its annual-command baseline")
    for name in metrics:
        result[f"change_{name}"] = result[name] - result[f"annual_{name}"]
    result["energy_change_pct"] = (
        100 * result.change_total_purchased_kwh / result.annual_total_purchased_kwh
    )
    result["control_label"] = result.control.map(CONTROL_LABELS)
    if result.control_label.isna().any():
        raise ValueError("Seasonal commands contain an unknown control")
    result["demand_source"] = np.where(
        result.cohort_label.astype(str).str.startswith("China"),
        "China archive", "Multinational archive",
    )
    result = result.merge(
        climates[["climate", "latitude", "country", "continent", "climate_zone"]],
        on="climate", how="left", validate="many_to_one",
    )
    if result.climate_zone.isna().any():
        raise ValueError("A seasonal-command row lacks climate metadata")
    return result.sort_values(
        ["climate", "control", "seasonal_contrast_fraction", "cohort_label"],
        kind="mergesort",
    ).reset_index(drop=True)


def event_effect_source(
    events: pd.DataFrame,
    climates: pd.DataFrame,
    *,
    expected_climates: int = 9,
    expected_runs: int = 810,
    expected_events_per_run: int = 8,
    expected_service_factors: int = 5,
) -> pd.DataFrame:
    """Add readable labels to the audited multi-event response table."""
    events = events.rename(columns={
        "local_season": "season",
        "load_level": "event_class",
    }).copy()
    required = {
        "run_id",
        "event_id",
        "climate",
        "composition_id",
        "control",
        "action_id",
        "action_family",
        "season",
        "event_class",
        "event_service_limit_factor",
        "service_qualified",
        "positive_event_response",
        "positive_full_window_response",
        "qualified_positive_response",
        "qualified_positive_full_window_response",
        "event_mean_power_reduction_kw",
        "event_electricity_reduction_kwh",
        "recovery_electricity_change_kwh",
        "precondition_event_recovery_net_reduction_kwh",
    }
    _require(events, required, "multi-event response scores")
    expected_rows = expected_runs * expected_events_per_run * expected_service_factors
    if len(events) != expected_rows:
        raise ValueError(f"Expected {expected_rows} event-score rows; found {len(events)}")
    keys = ["run_id", "event_id", "event_service_limit_factor"]
    if events.duplicated(keys).any():
        raise ValueError("Multi-event scores duplicate a run/event/service-factor cell")
    if events.run_id.astype(str).nunique() != expected_runs:
        raise ValueError(f"Expected {expected_runs} response trajectories")
    if events.climate.astype(str).nunique() != expected_climates:
        raise ValueError("Multi-event scores do not cover the declared climates")
    result = events.copy()
    result["control_label"] = result.control.map(CONTROL_LABELS)
    if result.control_label.isna().any():
        unknown = sorted(result.loc[result.control_label.isna(), "control"].unique())
        raise ValueError(f"Unknown controls: {unknown}")
    if "cohort_label" not in result:
        raise ValueError("Multi-event response scores require readable cohort labels")
    result["action_label"] = result.action_family.map(ACTION_LABELS)
    if result.action_label.isna().any():
        unknown = sorted(result.loc[result.action_label.isna(), "action_family"].unique())
        raise ValueError(f"Unknown response action families: {unknown}")
    result["event_class_label"] = result.event_class.map(
        {"typical": "Typical load", "high": "High load"}
    )
    if result.event_class_label.isna().any():
        raise ValueError("Multi-event response scores contain an unknown event class")
    result["season_label"] = result.season.astype(str).str.title()
    result["demand_source"] = np.where(
        result.cohort_label.astype(str).str.startswith("China"),
        "China archive", "Multinational archive",
    )
    result = result.merge(
        climates[["climate", "country", "continent", "climate_zone"]],
        on="climate",
        how="left",
        validate="many_to_one",
    )
    if result.climate_zone.isna().any():
        raise ValueError("A response row lacks climate metadata")
    return result.sort_values(
        [
            "climate",
            "season",
            "event_class",
            "control",
            "action_family",
            "composition_id",
            "event_service_limit_factor",
        ],
        kind="mergesort",
    )
