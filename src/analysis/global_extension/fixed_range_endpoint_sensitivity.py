"""Audit service endpoints against fixed personal-range half-widths.

The completed building trajectories and controller commands are not changed.
Only the interval-based service endpoints are rescored using fixed bands around
each profile centre. Centre mismatch, fixed-tail mismatch, energy and carbon
remain the values from the audited annual analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.analyse_global_results import (
    load_hourly,
    normalise_manifest,
    normalise_profiles,
    score_trajectory,
    verify_hourly_hash,
)
from analysis.global_extension.matched_service_sensitivity import (
    analyse_matched_service,
)


HALF_WIDTHS_C = (1.5, 2.0)
CONTROLS = ("uniform_average", "personal_center", "personal_range")
ALTERNATIVES = ("personal_center", "personal_range")
SERVICE_TOLERANCE_C = 1e-9
INTERVAL_ENDPOINTS = (
    "modeled_interval_person_hour_coverage",
    "modeled_interval_mean_excess_c",
    "modeled_interval_excess_native_person_c_hours",
    "profile_coverage_p10",
    "profile_coverage_median",
    "profile_coverage_minimum",
    "profiles_with_no_occupied_hour_interval_violation",
    "assigned_profile_count",
    "all_profiles_without_interval_violation",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolved(value: object, source: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else source.parent / path


def fixed_width_profiles(profiles: pd.DataFrame, half_width_c: float) -> pd.DataFrame:
    """Return profiles with one declared symmetric interval half-width."""
    if not np.isfinite(half_width_c) or half_width_c <= 0:
        raise ValueError("Fixed personal-range half-width must be positive and finite")
    result = profiles.copy()
    result["lower_c"] = result.centre_c.astype(float) - float(half_width_c)
    result["upper_c"] = result.centre_c.astype(float) + float(half_width_c)
    return result


def _validate_inputs(
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    if len(metrics) != 810 or len(manifest) != 810:
        raise ValueError("Fixed-width sensitivity requires all 810 annual candidates")
    if metrics.run_id.astype(str).duplicated().any():
        raise ValueError("Annual metrics duplicate a run_id")
    if set(metrics.run_id.astype(str)) != set(manifest.run_id.astype(str)):
        raise ValueError("Annual metrics and manifest contain different run IDs")
    if not manifest.engineering_pass.all():
        raise ValueError("An annual candidate failed engineering qualification")
    if metrics.climate.astype(str).nunique() != 9:
        raise ValueError("Fixed-width sensitivity requires all nine climates")
    if metrics.composition.astype(str).nunique() != 6:
        raise ValueError("Fixed-width sensitivity requires all six demand compositions")
    if set(metrics.control.astype(str)) != set(CONTROLS):
        raise ValueError("Annual metrics contain unexpected controls")
    declared = set(
        map(
            tuple,
            manifest[["climate", "composition", "layout"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    profile_keys = set(
        map(
            tuple,
            profiles[["climate", "composition", "layout"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if declared != profile_keys:
        raise ValueError("Annual manifest and profile groups do not match")


def direct_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each offset-zero demand-guided run with Uniform average."""
    zero = frame.loc[np.isclose(pd.to_numeric(frame.offset_c), 0.0)].copy()
    keys = ["fixed_half_width_c", "climate", "composition", "layout", "permission"]
    rows: list[dict] = []
    for key, group in zero.groupby(keys, sort=True, dropna=False):
        reference = group.loc[group.control.eq("uniform_average")]
        if len(reference) != 1:
            raise ValueError("A fixed-width direct comparison lacks one Uniform average run")
        uniform = reference.iloc[0]
        for control in ALTERNATIVES:
            alternative = group.loc[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError(f"A fixed-width comparison lacks one {control} run")
            personal = alternative.iloc[0]
            rows.append(
                dict(zip(keys, key, strict=True))
                | {
                    "alternative_control": control,
                    "uniform_run_id": str(uniform.run_id),
                    "alternative_run_id": str(personal.run_id),
                    "delta_mean_reference_mismatch_c": float(
                        personal.mean_reference_mismatch_c
                        - uniform.mean_reference_mismatch_c
                    ),
                    "delta_fixed_tail_reference_mismatch_c": float(
                        personal.fixed_tail_reference_mismatch_c
                        - uniform.fixed_tail_reference_mismatch_c
                    ),
                    "delta_fixed_interval_coverage": float(
                        personal.modeled_interval_person_hour_coverage
                        - uniform.modeled_interval_person_hour_coverage
                    ),
                    "delta_fixed_interval_mean_excess_c": float(
                        personal.modeled_interval_mean_excess_c
                        - uniform.modeled_interval_mean_excess_c
                    ),
                    "delta_facility_purchased_kwh": float(
                        personal.facility_purchased_kwh
                        - uniform.facility_purchased_kwh
                    ),
                    "delta_facility_purchased_pct": float(
                        100.0
                        * (personal.facility_purchased_kwh - uniform.facility_purchased_kwh)
                        / uniform.facility_purchased_kwh
                    ),
                }
            )
    result = pd.DataFrame(rows)
    expected = frame.fixed_half_width_c.nunique() * 108
    if len(result) != expected:
        raise AssertionError(
            f"Expected {expected} fixed-width comparisons; found {len(result)}"
        )
    return result


def summarise_direct(frame: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for (width, control), group in frame.groupby(
        ["fixed_half_width_c", "alternative_control"], sort=True
    ):
        rows.append({
            "fixed_half_width_c": float(width),
            "alternative_control": str(control),
            "comparisons": int(len(group)),
            "population_mean_mismatch_lower": int(
                group.delta_mean_reference_mismatch_c.lt(-SERVICE_TOLERANCE_C).sum()
            ),
            "fixed_tail_mismatch_lower": int(
                group.delta_fixed_tail_reference_mismatch_c.lt(
                    -SERVICE_TOLERANCE_C
                ).sum()
            ),
            "fixed_interval_coverage_higher": int(
                group.delta_fixed_interval_coverage.gt(SERVICE_TOLERANCE_C).sum()
            ),
            "fixed_interval_mean_excess_lower": int(
                group.delta_fixed_interval_mean_excess_c.lt(
                    -SERVICE_TOLERANCE_C
                ).sum()
            ),
            "facility_energy_increase": int(
                group.delta_facility_purchased_kwh.gt(0).sum()
            ),
            "facility_energy_decrease": int(
                group.delta_facility_purchased_kwh.lt(0).sum()
            ),
        })
    return rows


def analyse_fixed_range_endpoints(
    metrics_path: Path,
    run_manifest_path: Path,
    profiles_path: Path,
    validation_path: Path,
    output_root: Path,
    *,
    half_widths_c: tuple[float, ...] = HALF_WIDTHS_C,
) -> dict:
    metrics_path = metrics_path.resolve()
    run_manifest_path = run_manifest_path.resolve()
    profiles_path = profiles_path.resolve()
    validation_path = validation_path.resolve()
    output_root = output_root.resolve()
    widths = tuple(float(value) for value in half_widths_c)
    if widths != tuple(sorted(set(widths))) or any(
        not np.isfinite(value) or value <= 0 for value in widths
    ):
        raise ValueError("Fixed half-widths must be unique, positive and increasing")

    metrics = pd.read_csv(metrics_path)
    manifest = normalise_manifest(pd.read_csv(run_manifest_path))
    profiles = normalise_profiles(pd.read_csv(profiles_path))
    _validate_inputs(metrics, manifest, profiles)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "AUDIT_COMPLETE":
        raise ValueError("Personal-range validation audit is incomplete")

    metric_lookup = metrics.set_index("run_id", verify_integrity=True)
    profile_groups = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(
            ["climate", "composition", "layout"], sort=True
        )
    }
    fixed_groups = {
        width: {
            key: fixed_width_profiles(group, width)
            for key, group in profile_groups.items()
        }
        for width in widths
    }
    rescored_rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        path = _resolved(row.hourly_path, run_manifest_path)
        verify_hourly_hash(path, row.hourly_sha256, f"Annual trajectory {row.run_id}")
        data = load_hourly(path)
        original = metric_lookup.loc[str(row.run_id)].to_dict()
        key = (row.climate, row.composition, row.layout)
        for width in widths:
            score = score_trajectory(
                data,
                fixed_groups[width][key],
                np.asarray([0], dtype=int),
            )["summary"]
            record = dict(original)
            record["run_id"] = str(row.run_id)
            record["fixed_half_width_c"] = width
            for endpoint in INTERVAL_ENDPOINTS:
                record[endpoint] = score[endpoint]
            rescored_rows.append(record)
    rescored = pd.DataFrame(rescored_rows)
    if len(rescored) != len(metrics) * len(widths):
        raise AssertionError("Fixed-width rescore is incomplete")

    output_root.mkdir(parents=True, exist_ok=True)
    metrics_output = output_root / "FIXED_RANGE_ENDPOINT_RUN_METRICS.csv"
    rescored.to_csv(metrics_output, index=False)
    direct = direct_comparisons(rescored)
    direct_output = output_root / "FIXED_RANGE_ENDPOINT_DIRECT_COMPARISONS.csv"
    direct.to_csv(direct_output, index=False)

    matched_results: dict[str, dict] = {}
    matched_output_hashes: dict[str, str] = {}
    for width in widths:
        slug = f"half_width_{width:g}C".replace(".", "p")
        part_path = output_root / f"FIXED_RANGE_ENDPOINT_METRICS_{slug}.csv"
        rescored.loc[
            np.isclose(rescored.fixed_half_width_c, width)
        ].drop(columns="fixed_half_width_c").to_csv(part_path, index=False)
        matched_root = output_root / f"matched_{slug}"
        matched_results[f"{width:g}"] = analyse_matched_service(
            part_path,
            matched_root,
            expected_sentinel_climates=9,
        )
        matched_output_hashes[part_path.name] = sha256(part_path)
        matched_audit_path = matched_root / "MATCHED_REALIZED_SERVICE_AUDIT.json"
        matched_output_hashes[
            str(matched_audit_path.relative_to(output_root))
        ] = sha256(matched_audit_path)

    audit = {
        "status": "PASS",
        "definition": (
            "The same 810 completed annual trajectories were rescored with fixed "
            "symmetric intervals around each profile centre. Controller commands, "
            "centre mismatch, fixed-tail mismatch, energy and carbon were unchanged."
        ),
        "endpoint_dependencies": {
            "information_diagnostic": (
                "Centre mismatch and fixed-tail centre mismatch only; no personal-range "
                "width is used in qualification or selection."
            ),
            "common_service_selection": (
                "Centre mismatch and fixed-tail centre mismatch only; no personal-range "
                "width is used in qualification or selection."
            ),
            "matched_realised_service_primary": (
                "Uses centre mismatch, fixed-tail centre mismatch and interval mean "
                "excess; the interval endpoint depends on the declared width."
            ),
            "matched_realised_service_two_metric_comparator": (
                "Uses centre mismatch and fixed-tail centre mismatch only."
            ),
            "personal_range_controller": (
                "Its commands depend on ranges estimated by the published elicitation "
                "method; fixed-width endpoint rescoring does not redefine that controller."
            ),
        },
        "half_widths_c": list(widths),
        "counts": {
            "physical_trajectories": int(len(metrics)),
            "rescored_rows": int(len(rescored)),
            "direct_comparisons": int(len(direct)),
        },
        "direct_summary": summarise_direct(direct),
        "heldout_validation": {
            key: validation.get(key)
            for key in (
                "selected_fixed_half_width_c",
                "selection_rule",
                "variable_width_outperformed_best_fixed_all_outcomes",
                "heldout_width_comparison",
                "records",
            )
        },
        "heldout_interpretation": (
            "On the available repeated observations, variable widths did not outperform "
            "the best fixed bands as binary classifiers. This test neither establishes "
            "the estimated ranges as measured individual acceptance bands nor invalidates "
            "the independently estimated personal centres."
        ),
        "matched_service": {
            width: {
                "tolerance_results": result["tolerance_results"],
                "two_metric_comparator_results": result[
                    "two_metric_comparator_results"
                ],
            }
            for width, result in matched_results.items()
        },
        "input_sha256": {
            "annual_metrics": sha256(metrics_path),
            "annual_manifest": sha256(run_manifest_path),
            "profiles": sha256(profiles_path),
            "personal_range_validation": sha256(validation_path),
        },
        "output_sha256": {
            metrics_output.name: sha256(metrics_output),
            direct_output.name: sha256(direct_output),
            **matched_output_hashes,
        },
    }
    audit_path = output_root / "FIXED_RANGE_ENDPOINT_SENSITIVITY.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--personal-range-validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--half-widths-c", type=float, nargs="+", default=HALF_WIDTHS_C
    )
    args = parser.parse_args()
    audit = analyse_fixed_range_endpoints(
        args.metrics,
        args.run_manifest,
        args.profiles,
        args.personal_range_validation,
        args.output_root,
        half_widths_c=tuple(args.half_widths_c),
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
