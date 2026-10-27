"""Compare controls at closely matched realised thermal service.

The primary analysis uses only climates for which every control has the same
five-offset candidate library. It enumerates every uniform-versus-personal
pair, selects the closest pair without consulting energy, and reports results
over predeclared tolerances for population-mean mismatch, fixed-tail mismatch
and mean excess outside the personal range. The earlier two-metric definition
is retained as a declared comparator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = ("uniform_average", "personal_center", "personal_range")
REFERENCE_CONTROL = "uniform_average"
ALTERNATIVE_CONTROLS = ("personal_center", "personal_range")
IDENTITY = ("climate", "composition", "layout", "permission")
EXPECTED_OFFSETS_C = (-1.0, -0.5, 0.0, 0.5, 1.0)
TOLERANCES_C = (0.01, 0.025, 0.05, 0.075, 0.10)
MEAN_SERVICE = "mean_reference_mismatch_c"
TAIL_SERVICE = "fixed_tail_reference_mismatch_c"
EXCESS_SERVICE = "modeled_interval_mean_excess_c"
FACILITY_ENERGY_COLUMNS = (
    "facility_purchased_kwh",
    # Legacy aliases for complete facility electricity plus complete facility gas.
    "facility_total_purchased_kwh",
    "total_facility_purchased_kwh",
    "whole_building_purchased_kwh",
)
PARTIAL_FACILITY_ENERGY_COLUMNS = ("facility_electric_plus_space_heating_gas_kwh",)
CONDITIONING_ENERGY_COLUMNS = (
    "conditioning_purchased_kwh",
    "total_purchased_kwh",
)
BOUNDARY_EPSILON_C = 1e-12
ENERGY_BOUNDARY_METADATA = {
    "complete_facility_purchased_energy": {
        "label": "Complete whole-building purchased energy",
        "definition": ("Electricity:Facility plus complete annual NaturalGas:Facility"),
    },
    "facility_electricity_plus_space_heating_gas_sensitivity": {
        "label": "Electricity plus space-heating gas sensitivity boundary",
        "definition": (
            "Electricity:Facility plus hourly Heating:NaturalGas; facility gas "
            "outside the space-heating output is excluded"
        ),
    },
    "legacy_facility_electricity_plus_space_heating_gas": {
        "label": "Legacy partial facility-energy fallback",
        "definition": (
            "Older-table fallback using Electricity:Facility plus hourly "
            "Heating:NaturalGas; facility gas outside the space-heating output "
            "is excluded"
        ),
    },
    "conditioning": {
        "label": "Conditioning purchased energy fallback",
        "definition": "Conditioning electricity plus heating gas",
    },
    "legacy_conditioning_total": {
        "label": "Legacy conditioning-energy fallback",
        "definition": "Older-table conditioning electricity plus heating gas total",
    },
    "user_selected": {
        "label": "User-selected energy boundary",
        "definition": "Energy boundary supplied explicitly on the command line",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    parsed = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if parsed.isna().any():
        raise ValueError("engineering_pass contains non-boolean values")
    return parsed.astype(bool)


def _offset_signature(values: pd.Series) -> tuple[float, ...]:
    return tuple(sorted(round(float(value), 9) for value in values))


def _tolerance_column(prefix: str, tolerance: float) -> str:
    token = f"{tolerance:.3f}".replace(".", "p")
    return f"{prefix}_{token}_c"


def resolve_energy_column(frame: pd.DataFrame, requested: str | None = None) -> tuple[str, str]:
    """Choose complete facility energy, with explicitly labelled legacy fallbacks."""
    if requested is not None:
        if requested not in frame:
            raise ValueError(f"Requested energy column is absent: {requested}")
        if requested in FACILITY_ENERGY_COLUMNS:
            return requested, "complete_facility_purchased_energy"
        if requested in PARTIAL_FACILITY_ENERGY_COLUMNS:
            return requested, "facility_electricity_plus_space_heating_gas_sensitivity"
        return requested, "user_selected"
    for column in FACILITY_ENERGY_COLUMNS:
        if column in frame:
            return column, "complete_facility_purchased_energy"
    for column in PARTIAL_FACILITY_ENERGY_COLUMNS:
        if column in frame:
            return column, "legacy_facility_electricity_plus_space_heating_gas"
    for column in CONDITIONING_ENERGY_COLUMNS:
        if column in frame:
            scope = (
                "conditioning"
                if column == "conditioning_purchased_kwh"
                else "legacy_conditioning_total"
            )
            return column, scope
    raise ValueError(
        "No purchased-energy column is available; expected a facility or " "conditioning total"
    )


def prepare_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        *IDENTITY,
        "control",
        "offset_c",
        "run_id",
        MEAN_SERVICE,
        TAIL_SERVICE,
        EXCESS_SERVICE,
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Global metrics lack required columns: {sorted(missing)}")
    result = frame.copy()
    result["control"] = result.control.replace({"ltpref_interval": "personal_range"})
    unknown = sorted(set(result.control) - set(CONTROLS))
    if unknown:
        raise ValueError(f"Global metrics contain unknown controls: {unknown}")
    if result[list(IDENTITY) + ["control", "run_id"]].isna().any().any():
        raise ValueError("Global metrics contain missing comparison identifiers")
    result["offset_c"] = pd.to_numeric(result.offset_c, errors="raise")
    result[MEAN_SERVICE] = pd.to_numeric(result[MEAN_SERVICE], errors="coerce")
    result[TAIL_SERVICE] = pd.to_numeric(result[TAIL_SERVICE], errors="coerce")
    result[EXCESS_SERVICE] = pd.to_numeric(result[EXCESS_SERVICE], errors="coerce")
    if "engineering_pass" in result:
        result["engineering_pass"] = _as_bool(result.engineering_pass)
    else:
        result["engineering_pass"] = True
    keys = [*IDENTITY, "control", "offset_c"]
    if result.duplicated(keys).any():
        raise ValueError("Global metrics duplicate a logical annual candidate")
    if result.run_id.duplicated().any():
        raise ValueError("Global metrics contain duplicate run identifiers")
    return result


def detect_equal_depth_climates(
    frame: pd.DataFrame,
    expected_offsets: tuple[float, ...] = EXPECTED_OFFSETS_C,
) -> list[str]:
    """Find climates whose every identity has all three five-offset libraries."""
    expected = tuple(round(float(value), 9) for value in expected_offsets)
    identity_complete: dict[tuple[object, ...], bool] = {}
    for key, group in frame.groupby(list(IDENTITY), sort=True, dropna=False):
        signatures = {
            control: _offset_signature(part.offset_c)
            for control, part in group.groupby("control", sort=True)
        }
        identity_complete[tuple(key)] = set(signatures) == set(CONTROLS) and all(
            signatures[control] == expected for control in CONTROLS
        )
    climates: list[str] = []
    for climate in sorted(frame.climate.astype(str).unique()):
        flags = [complete for key, complete in identity_complete.items() if str(key[0]) == climate]
        if flags and all(flags):
            climates.append(climate)
    return climates


def validate_primary_library(
    frame: pd.DataFrame,
    sentinel_climates: list[str],
    energy_column: str,
    expected_sentinel_climates: int | None,
) -> pd.DataFrame:
    if expected_sentinel_climates is not None and (
        len(sentinel_climates) != expected_sentinel_climates
    ):
        raise ValueError(
            f"Expected {expected_sentinel_climates} equal-depth sentinel climates; "
            f"found {len(sentinel_climates)}: {sentinel_climates}"
        )
    primary = frame[frame.climate.astype(str).isin(sentinel_climates)].copy()
    numeric = [MEAN_SERVICE, TAIL_SERVICE, EXCESS_SERVICE, energy_column]
    primary[energy_column] = pd.to_numeric(primary[energy_column], errors="coerce")
    if primary.empty:
        raise ValueError("No complete five-offset candidate library is available")
    if not primary.engineering_pass.all():
        failed = primary.loc[~primary.engineering_pass, "run_id"].astype(str).tolist()
        raise ValueError(f"A sentinel candidate failed engineering checks: {failed}")
    if not np.isfinite(primary[numeric].to_numpy(float)).all():
        raise ValueError("Sentinel service and energy values must be finite")
    if primary[energy_column].le(0).any():
        raise ValueError("Sentinel purchased-energy values must be positive")
    return primary


def enumerate_service_pairs(
    primary: pd.DataFrame,
    energy_column: str,
    energy_scope: str,
) -> pd.DataFrame:
    """Enumerate all 5 x 5 reference-alternative pairs in each primary group."""
    rows: list[dict] = []
    for key, group in primary.groupby(list(IDENTITY), sort=True, dropna=False):
        reference = group[group.control.eq(REFERENCE_CONTROL)].sort_values(
            ["offset_c", "run_id"], kind="mergesort"
        )
        for alternative_control in ALTERNATIVE_CONTROLS:
            alternative = group[group.control.eq(alternative_control)].sort_values(
                ["offset_c", "run_id"], kind="mergesort"
            )
            if len(reference) != 5 or len(alternative) != 5:
                raise ValueError("Primary comparison requires exactly five candidates per control")
            for uniform in reference.itertuples(index=False):
                for alternative_row in alternative.itertuples(index=False):
                    uniform_mean = float(getattr(uniform, MEAN_SERVICE))
                    alternative_mean = float(getattr(alternative_row, MEAN_SERVICE))
                    uniform_tail = float(getattr(uniform, TAIL_SERVICE))
                    alternative_tail = float(getattr(alternative_row, TAIL_SERVICE))
                    uniform_excess = float(getattr(uniform, EXCESS_SERVICE))
                    alternative_excess = float(getattr(alternative_row, EXCESS_SERVICE))
                    uniform_energy = float(getattr(uniform, energy_column))
                    alternative_energy = float(getattr(alternative_row, energy_column))
                    delta_energy = alternative_energy - uniform_energy
                    tie_tolerance = max(1e-9, abs(uniform_energy) * 1e-12)
                    direction = (
                        "increase"
                        if delta_energy > tie_tolerance
                        else "decrease"
                        if delta_energy < -tie_tolerance
                        else "tie"
                    )
                    mean_gap = abs(alternative_mean - uniform_mean)
                    tail_gap = abs(alternative_tail - uniform_tail)
                    excess_gap = abs(alternative_excess - uniform_excess)
                    record = dict(zip(IDENTITY, key))
                    record.update(
                        {
                            "alternative_control": alternative_control,
                            "uniform_run_id": str(uniform.run_id),
                            "alternative_run_id": str(alternative_row.run_id),
                            "uniform_offset_c": float(uniform.offset_c),
                            "alternative_offset_c": float(alternative_row.offset_c),
                            "offset_gap_c": abs(
                                float(alternative_row.offset_c) - float(uniform.offset_c)
                            ),
                            "uniform_mean_mismatch_c": uniform_mean,
                            "alternative_mean_mismatch_c": alternative_mean,
                            "mean_gap_c": mean_gap,
                            "uniform_tail_mismatch_c": uniform_tail,
                            "alternative_tail_mismatch_c": alternative_tail,
                            "tail_gap_c": tail_gap,
                            "uniform_interval_mean_excess_c": uniform_excess,
                            "alternative_interval_mean_excess_c": alternative_excess,
                            "interval_mean_excess_gap_c": excess_gap,
                            "two_metric_linf_gap_c": max(mean_gap, tail_gap),
                            "two_metric_l1_gap_c": mean_gap + tail_gap,
                            "service_linf_gap_c": max(mean_gap, tail_gap, excess_gap),
                            "service_l1_gap_c": mean_gap + tail_gap + excess_gap,
                            "energy_scope": energy_scope,
                            "energy_column": energy_column,
                            "uniform_energy_kwh": uniform_energy,
                            "alternative_energy_kwh": alternative_energy,
                            "delta_energy_kwh": delta_energy,
                            "delta_energy_pct": 100.0 * delta_energy / uniform_energy,
                            "energy_direction": direction,
                        }
                    )
                    rows.append(record)
    pairs = pd.DataFrame(rows)
    return pairs.sort_values(
        [
            *IDENTITY,
            "alternative_control",
            "uniform_offset_c",
            "alternative_offset_c",
            "uniform_run_id",
            "alternative_run_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)


def _select_closest_pairs(
    pairs: pd.DataFrame,
    linf_column: str,
    l1_column: str,
    flag_prefix: str,
) -> pd.DataFrame:
    group_keys = [*IDENTITY, "alternative_control"]
    sort_columns = [
        *group_keys,
        linf_column,
        l1_column,
        "offset_gap_c",
        "uniform_offset_c",
        "alternative_offset_c",
        "uniform_run_id",
        "alternative_run_id",
    ]
    selected = (
        pairs.sort_values(sort_columns, kind="mergesort")
        .groupby(group_keys, sort=True, dropna=False)
        .head(1)
        .copy()
    )
    for tolerance in TOLERANCES_C:
        selected[_tolerance_column(flag_prefix, tolerance)] = selected[linf_column].le(
            tolerance + BOUNDARY_EPSILON_C
        )
    return selected.sort_values(group_keys, kind="mergesort").reset_index(drop=True)


def select_closest_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select the closest three-metric service pair without using energy."""
    return _select_closest_pairs(
        pairs, "service_linf_gap_c", "service_l1_gap_c", "within_tolerance"
    )


def select_two_metric_closest_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the earlier mean-plus-tail definition as a comparator."""
    return _select_closest_pairs(
        pairs,
        "two_metric_linf_gap_c",
        "two_metric_l1_gap_c",
        "within_two_metric_tolerance",
    )


def _qualifying_pairs(
    pairs: pd.DataFrame,
    distance_column: str,
    flag_prefix: str,
    minimum_column: str,
) -> pd.DataFrame:
    result = pairs[pairs[distance_column].le(max(TOLERANCES_C) + BOUNDARY_EPSILON_C)].copy()
    for tolerance in TOLERANCES_C:
        result[_tolerance_column(flag_prefix, tolerance)] = result[distance_column].le(
            tolerance + BOUNDARY_EPSILON_C
        )
    result[minimum_column] = [
        next(tolerance for tolerance in TOLERANCES_C if gap <= tolerance + BOUNDARY_EPSILON_C)
        for gap in result[distance_column]
    ]
    return result.reset_index(drop=True)


def qualifying_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return three-metric pairs meeting at least the largest tolerance."""
    return _qualifying_pairs(
        pairs,
        "service_linf_gap_c",
        "within_tolerance",
        "minimum_qualifying_tolerance_c",
    )


def two_metric_qualifying_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return mean-plus-tail comparator pairs meeting the largest tolerance."""
    return _qualifying_pairs(
        pairs,
        "two_metric_linf_gap_c",
        "within_two_metric_tolerance",
        "minimum_two_metric_qualifying_tolerance_c",
    )


def tolerance_summary(
    closest: pd.DataFrame,
    all_pairs: pd.DataFrame,
    distance_column: str = "service_linf_gap_c",
) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("all", closest, all_pairs)] + [
        (
            control,
            closest[closest.alternative_control.eq(control)],
            all_pairs[all_pairs.alternative_control.eq(control)],
        )
        for control in ALTERNATIVE_CONTROLS
    ]
    group_keys = [*IDENTITY, "alternative_control"]
    for scope, closest_part, pair_part in scopes:
        for tolerance in TOLERANCES_C:
            matched = closest_part[closest_part[distance_column].le(tolerance + BOUNDARY_EPSILON_C)]
            qualifying = pair_part[pair_part[distance_column].le(tolerance + BOUNDARY_EPSILON_C)]
            rows.append(
                {
                    "alternative_control": scope,
                    "tolerance_c": tolerance,
                    "eligible_context_control_groups": len(closest_part),
                    "matched_context_control_groups": len(matched),
                    "matched_fraction": (
                        len(matched) / len(closest_part) if len(closest_part) else np.nan
                    ),
                    "all_qualifying_pairs": len(qualifying),
                    "groups_with_any_qualifying_pair": len(
                        qualifying[list(group_keys)].drop_duplicates()
                    ),
                    "all_qualifying_energy_increase": int(
                        qualifying.energy_direction.eq("increase").sum()
                    ),
                    "all_qualifying_energy_decrease": int(
                        qualifying.energy_direction.eq("decrease").sum()
                    ),
                    "all_qualifying_energy_tie": int(qualifying.energy_direction.eq("tie").sum()),
                    "energy_increase": int(matched.energy_direction.eq("increase").sum()),
                    "energy_decrease": int(matched.energy_direction.eq("decrease").sum()),
                    "energy_tie": int(matched.energy_direction.eq("tie").sum()),
                    "median_delta_energy_pct": (
                        float(matched.delta_energy_pct.median()) if len(matched) else np.nan
                    ),
                    "minimum_delta_energy_pct": (
                        float(matched.delta_energy_pct.min()) if len(matched) else np.nan
                    ),
                    "maximum_delta_energy_pct": (
                        float(matched.delta_energy_pct.max()) if len(matched) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def mean_only_diagnostic(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select by mean service alone, then expose the unprotected tail gap."""
    group_keys = [*IDENTITY, "alternative_control"]
    sort_columns = [
        *group_keys,
        "mean_gap_c",
        "offset_gap_c",
        "uniform_offset_c",
        "alternative_offset_c",
        "uniform_run_id",
        "alternative_run_id",
    ]
    selected = (
        pairs.sort_values(sort_columns, kind="mergesort")
        .groupby(group_keys, sort=True, dropna=False)
        .head(1)
        .copy()
    )
    for tolerance in TOLERANCES_C:
        selected[_tolerance_column("mean_within_tolerance", tolerance)] = selected.mean_gap_c.le(
            tolerance + BOUNDARY_EPSILON_C
        )
    return selected.sort_values(group_keys, kind="mergesort").reset_index(drop=True)


def mean_only_summary(diagnostic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    scopes = [("all", diagnostic)] + [
        (control, diagnostic[diagnostic.alternative_control.eq(control)])
        for control in ALTERNATIVE_CONTROLS
    ]
    for scope, part in scopes:
        for tolerance in TOLERANCES_C:
            matched = part[part.mean_gap_c.le(tolerance + BOUNDARY_EPSILON_C)]
            rows.append(
                {
                    "alternative_control": scope,
                    "mean_tolerance_c": tolerance,
                    "eligible_context_control_groups": len(part),
                    "mean_matched_context_control_groups": len(matched),
                    "mean_matched_fraction": len(matched) / len(part) if len(part) else np.nan,
                    "tail_gap_median_c": (
                        float(matched.tail_gap_c.median()) if len(matched) else np.nan
                    ),
                    "tail_gap_p95_c": (
                        float(matched.tail_gap_c.quantile(0.95)) if len(matched) else np.nan
                    ),
                    "tail_gap_exceeds_same_tolerance": int(
                        matched.tail_gap_c.gt(tolerance + BOUNDARY_EPSILON_C).sum()
                    ),
                    "tail_gap_exceeds_0p10_c": int(
                        matched.tail_gap_c.gt(0.10 + BOUNDARY_EPSILON_C).sum()
                    ),
                    "energy_increase": int(matched.energy_direction.eq("increase").sum()),
                    "energy_decrease": int(matched.energy_direction.eq("decrease").sum()),
                    "energy_tie": int(matched.energy_direction.eq("tie").sum()),
                    "median_delta_energy_pct": (
                        float(matched.delta_energy_pct.median()) if len(matched) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def analyse_matched_service(
    metrics_path: Path,
    output_root: Path,
    energy_column: str | None = None,
    expected_sentinel_climates: int | None = 9,
) -> dict:
    metrics_path = metrics_path.resolve()
    output_root = output_root.resolve()
    frame = prepare_metrics(pd.read_csv(metrics_path))
    selected_energy, energy_scope = resolve_energy_column(frame, energy_column)
    energy_metadata = ENERGY_BOUNDARY_METADATA[energy_scope]
    sentinel_climates = detect_equal_depth_climates(frame)
    primary = validate_primary_library(
        frame,
        sentinel_climates,
        selected_energy,
        expected_sentinel_climates,
    )
    pairs = enumerate_service_pairs(primary, selected_energy, energy_scope)
    closest = select_closest_pairs(pairs)
    all_qualifying = qualifying_pairs(pairs)
    summary = tolerance_summary(closest, pairs)
    two_metric_closest = select_two_metric_closest_pairs(pairs)
    two_metric_all_qualifying = two_metric_qualifying_pairs(pairs)
    two_metric_summary = tolerance_summary(
        two_metric_closest, pairs, distance_column="two_metric_linf_gap_c"
    )
    mean_diagnostic = mean_only_diagnostic(pairs)
    mean_summary = mean_only_summary(mean_diagnostic)

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "closest_pairs": output_root / "MATCHED_REALIZED_SERVICE_CLOSEST_PAIRS.csv",
        "all_qualifying_pairs": (output_root / "MATCHED_REALIZED_SERVICE_ALL_QUALIFYING_PAIRS.csv"),
        "tolerance_summary": (output_root / "MATCHED_REALIZED_SERVICE_TOLERANCE_SUMMARY.csv"),
        "two_metric_closest_pairs": (
            output_root / "MATCHED_REALIZED_SERVICE_TWO_METRIC_CLOSEST_PAIRS.csv"
        ),
        "two_metric_all_qualifying_pairs": (
            output_root / "MATCHED_REALIZED_SERVICE_TWO_METRIC_ALL_QUALIFYING_PAIRS.csv"
        ),
        "two_metric_tolerance_summary": (
            output_root / "MATCHED_REALIZED_SERVICE_TWO_METRIC_TOLERANCE_SUMMARY.csv"
        ),
        "mean_only_diagnostic": (output_root / "MATCHED_REALIZED_SERVICE_MEAN_ONLY_DIAGNOSTIC.csv"),
        "mean_only_summary": (output_root / "MATCHED_REALIZED_SERVICE_MEAN_ONLY_SUMMARY.csv"),
    }
    tables = {
        "closest_pairs": closest,
        "all_qualifying_pairs": all_qualifying,
        "tolerance_summary": summary,
        "two_metric_closest_pairs": two_metric_closest,
        "two_metric_all_qualifying_pairs": two_metric_all_qualifying,
        "two_metric_tolerance_summary": two_metric_summary,
        "mean_only_diagnostic": mean_diagnostic,
        "mean_only_summary": mean_summary,
    }
    for name, table in tables.items():
        table.to_csv(outputs[name], index=False, float_format="%.12g")

    audit = {
        "status": "PASS",
        "protocol": "matched-realized-service-v2",
        "primary_scope": "equal-depth five-offset sentinel climates",
        "input": {
            "path": str(metrics_path),
            "sha256": sha256(metrics_path),
            "rows": len(frame),
        },
        "energy": {
            "column": selected_energy,
            "scope": energy_scope,
            "label": energy_metadata["label"],
            "definition": energy_metadata["definition"],
            "complete_facility_energy_used": (energy_scope == "complete_facility_purchased_energy"),
            "legacy_fallback_used": energy_scope.startswith("legacy_"),
        },
        "candidate_library": {
            "sentinel_climates": sentinel_climates,
            "excluded_unequal_depth_climates": sorted(
                set(frame.climate.astype(str)) - set(sentinel_climates)
            ),
            "offsets_c": list(EXPECTED_OFFSETS_C),
            "primary_rows": len(primary),
            "comparison_groups": len(closest),
            "enumerated_pairs": len(pairs),
            "pairs_per_comparison_group": 25,
        },
        "matching": {
            "tolerances_c": list(TOLERANCES_C),
            "primary_distance": (
                "max(abs(mean gap), abs(fixed-tail gap), " "abs(interval-mean-excess gap))"
            ),
            "secondary_distance": (
                "abs(mean gap) + abs(fixed-tail gap) + " "abs(interval-mean-excess gap)"
            ),
            "two_metric_comparator_distance": ("max(abs(mean gap), abs(fixed-tail gap))"),
            "coverage_ratio_used_for_matching": False,
            "tie_break": [
                "offset_gap_c",
                "uniform_offset_c",
                "alternative_offset_c",
                "uniform_run_id",
                "alternative_run_id",
            ],
            "energy_used_for_matching": False,
            "mean_only_diagnostic_uses_tail_for_matching": False,
        },
        "tolerance_results": _json_records(summary),
        "two_metric_comparator_results": _json_records(two_metric_summary),
        "mean_only_results": _json_records(mean_summary),
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "rows": len(tables[name]),
            }
            for name, path in outputs.items()
        },
    }
    audit_path = output_root / "MATCHED_REALIZED_SERVICE_AUDIT.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare building controls at matched realised service"
    )
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--energy-column")
    parser.add_argument("--expected-sentinel-climates", type=int, default=9)
    args = parser.parse_args()
    audit = analyse_matched_service(
        args.metrics,
        args.output_root,
        energy_column=args.energy_column,
        expected_sentinel_climates=args.expected_sentinel_climates,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
