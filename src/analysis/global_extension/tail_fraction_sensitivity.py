"""Test annual conclusions against alternative fixed least-served tail shares.

The analysis reuses completed annual and common-25 C trajectories.  It changes
only the share of profiles fixed from the common reference before controls are
reviewed.  Tail shares are sensitivity endpoints, not vulnerability standards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.analyse_global_results import (
    SERVICE_TOLERANCE_C,
    fixed_plan_information_diagnostic,
    load_hourly,
    normalise_profiles,
    pair_controls,
    score_trajectory,
    select_common_service,
    summarise_information_diagnostic,
    verify_hourly_hash,
    zone_mean_profiles,
)


DEFAULT_TAIL_FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_SERVICE_FRACTIONS = (1.0, 0.75, 0.50)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tail_indices(
    reference_mismatch_c: np.ndarray,
    profile_ids: pd.Series,
    fractions: tuple[float, ...] = DEFAULT_TAIL_FRACTIONS,
) -> dict[float, np.ndarray]:
    """Return deterministic, nested worst-served profile sets."""
    mismatch = np.asarray(reference_mismatch_c, dtype=float)
    ids = profile_ids.astype(str).reset_index(drop=True)
    if mismatch.shape != (len(ids),) or not np.isfinite(mismatch).all():
        raise ValueError("Reference profile mismatch is incomplete")
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("Tail fractions must lie in (0, 1]")
    if tuple(sorted(set(fractions))) != tuple(fractions):
        raise ValueError("Tail fractions must be unique and increasing")
    ranked = pd.DataFrame(
        {
            "position": np.arange(len(ids)),
            "profile_id": ids,
            "distance": mismatch,
        }
    ).sort_values(
        ["distance", "profile_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    return {
        float(fraction): np.sort(
            ranked.head(max(1, int(np.ceil(fraction * len(ids))))).position.to_numpy(
                int
            )
        )
        for fraction in fractions
    }


def _resolved(value: object, source: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else source.parent / path


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    converted = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if converted.isna().any():
        raise ValueError("Boolean provenance contains an unknown value")
    return converted.astype(bool)


def _prepare_inputs(
    metrics_path: Path,
    run_manifest_path: Path,
    profiles_path: Path,
    anchor_manifest_path: Path,
    reference_target_c: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    manifest = pd.read_csv(run_manifest_path)
    profiles = normalise_profiles(pd.read_csv(profiles_path))
    anchors = pd.read_csv(anchor_manifest_path)
    required_metrics = {
        "run_id",
        "climate",
        "composition",
        "layout",
        "permission",
        "control",
        "offset_c",
        "engineering_pass",
        "hourly_path",
        "hourly_sha256",
        "physical_plan_id",
        "mean_reference_mismatch_c",
        "zone_mean_mean_reference_mismatch_c",
        "facility_purchased_kwh",
    }
    missing = required_metrics - set(metrics)
    if missing:
        raise ValueError(f"Annual metrics lack {sorted(missing)}")
    if metrics.run_id.duplicated().any() or manifest.run_id.duplicated().any():
        raise ValueError("Annual inputs duplicate run_id")
    if set(metrics.run_id.astype(str)) != set(manifest.run_id.astype(str)):
        raise ValueError("Annual metrics and run manifest contain different runs")
    if not _as_bool(metrics.engineering_pass).all():
        raise ValueError("Tail sensitivity requires engineering-qualified runs")
    provenance = manifest[["run_id", "hourly_path", "hourly_sha256"]].rename(
        columns={
            "hourly_path": "manifest_hourly_path",
            "hourly_sha256": "manifest_hourly_sha256",
        }
    )
    metrics = metrics.merge(provenance, on="run_id", how="left", validate="one_to_one")
    if not metrics.hourly_sha256.astype(str).eq(
        metrics.manifest_hourly_sha256.astype(str)
    ).all():
        raise ValueError("Annual metric and manifest hourly hashes differ")

    target_column = next(
        (name for name in ("target_c", "uniform_target_c") if name in anchors), None
    )
    if target_column is None:
        raise ValueError("Anchor manifest lacks target temperature")
    if not np.isfinite(reference_target_c):
        raise ValueError("Reference target must be finite")
    anchors = anchors[
        np.isclose(
            pd.to_numeric(anchors[target_column], errors="raise"),
            reference_target_c,
        )
    ].copy()
    if anchors.climate.duplicated().any() or anchors.climate.nunique() == 0:
        raise ValueError(
            f"Expected one common-{reference_target_c:g} C anchor per climate"
        )
    for name in ("hourly_path", "hourly_sha256"):
        if name not in anchors:
            raise ValueError(f"Anchor manifest lacks {name}")
    return metrics, profiles, anchors


def _direction_counts(values: pd.Series) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="raise")
    return {
        "increase": int((numeric > SERVICE_TOLERANCE_C).sum()),
        "decrease": int((numeric < -SERVICE_TOLERANCE_C).sum()),
        "tie": int((numeric.abs() <= SERVICE_TOLERANCE_C).sum()),
    }


def _direct_summary(frame: pd.DataFrame) -> dict:
    result = {}
    for control, group in frame.groupby("alternative_control", sort=True):
        matched = group[group.status.eq("matched")]
        result[str(control)] = {
            "comparisons": int(len(group)),
            "matched": int(len(matched)),
            "population_mean_mismatch_lower": int(
                matched.delta_mean_reference_mismatch_c.lt(-SERVICE_TOLERANCE_C).sum()
            ),
            "fixed_tail_mismatch_lower": int(
                matched.delta_fixed_tail_reference_mismatch_c.lt(
                    -SERVICE_TOLERANCE_C
                ).sum()
            ),
            "both_mismatch_endpoints_lower": int(
                (
                    matched.delta_mean_reference_mismatch_c.lt(
                        -SERVICE_TOLERANCE_C
                    )
                    & matched.delta_fixed_tail_reference_mismatch_c.lt(
                        -SERVICE_TOLERANCE_C
                    )
                ).sum()
            ),
            "facility_energy_direction": _direction_counts(
                matched.delta_facility_purchased_kwh
            ),
        }
    return result


def _common_summary(
    selections: pd.DataFrame, comparisons: pd.DataFrame
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for service_fraction, fraction_pairs in comparisons.groupby(
        "service_fraction", sort=True
    ):
        key = f"{float(service_fraction):g}"
        controls = {}
        for control, group in fraction_pairs.groupby("alternative_control", sort=True):
            matched = group[group.status.eq("matched")]
            selected = selections[
                np.isclose(selections.service_fraction, service_fraction)
                & selections.control.eq(control)
            ]
            controls[str(control)] = {
                "comparisons": int(len(group)),
                "matched": int(len(matched)),
                "control_service_available": int(
                    selected.selection_status.eq("selected").sum()
                ),
                "facility_energy_direction": _direction_counts(
                    matched.delta_facility_purchased_kwh
                ),
            }
        uniform = selections[
            np.isclose(selections.service_fraction, service_fraction)
            & selections.control.eq("uniform_average")
        ]
        result[key] = {
            "uniform_service_available": int(
                uniform.selection_status.eq("selected").sum()
            ),
            "alternatives": controls,
        }
    return result


def _summary_rows(tail_fraction: float, result: dict) -> list[dict]:
    rows = []
    for control, values in result["direct"].items():
        rows.append(
            {
                "tail_fraction": tail_fraction,
                "endpoint": "direct",
                "service_fraction": np.nan,
                "control": control,
                **values,
                **{
                    f"energy_{key}": value
                    for key, value in values["facility_energy_direction"].items()
                },
            }
        )
    for service_fraction, values in result["common_service"].items():
        for control, control_values in values["alternatives"].items():
            rows.append(
                {
                    "tail_fraction": tail_fraction,
                    "endpoint": "common_service",
                    "service_fraction": float(service_fraction),
                    "control": control,
                    "uniform_service_available": values[
                        "uniform_service_available"
                    ],
                    **control_values,
                    **{
                        f"energy_{key}": value
                        for key, value in control_values[
                            "facility_energy_direction"
                        ].items()
                    },
                }
            )
    for service_fraction, values in result["information"][
        "by_service_fraction"
    ].items():
        scoped = values["full_distribution_resolvable"]
        rows.append(
            {
                "tail_fraction": tail_fraction,
                "endpoint": "information",
                "service_fraction": float(service_fraction),
                "control": "zone mean versus full distribution",
                **scoped,
            }
        )
    return rows


def analyse_tail_fraction_sensitivity(
    metrics_path: Path,
    run_manifest_path: Path,
    profiles_path: Path,
    anchor_manifest_path: Path,
    output_root: Path,
    *,
    tail_fractions: tuple[float, ...] = DEFAULT_TAIL_FRACTIONS,
    service_fractions: tuple[float, ...] = DEFAULT_SERVICE_FRACTIONS,
    reference_target_c: float = 25.0,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics, profiles, anchor_manifest = _prepare_inputs(
        metrics_path,
        run_manifest_path,
        profiles_path,
        anchor_manifest_path,
        reference_target_c,
    )
    profile_groups = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(
            ["climate", "composition", "layout"], sort=True
        )
    }
    declared = set(
        map(
            tuple,
            metrics[["climate", "composition", "layout"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    if declared != set(profile_groups):
        raise ValueError("Annual metrics and profile groups do not match")

    anchor_paths = dict(
        zip(anchor_manifest.climate, anchor_manifest.hourly_path, strict=True)
    )
    anchor_hashes = dict(
        zip(anchor_manifest.climate, anchor_manifest.hourly_sha256, strict=True)
    )
    anchor_data: dict[str, dict[str, np.ndarray]] = {}
    tails: dict[tuple, dict[float, np.ndarray]] = {}
    anchor_rows = []
    membership_rows = []
    for key, group in profile_groups.items():
        climate = str(key[0])
        if climate not in anchor_paths:
            raise ValueError(
                f"Missing common-{reference_target_c:g} C anchor for {climate}"
            )
        if climate not in anchor_data:
            path = _resolved(anchor_paths[climate], anchor_manifest_path)
            verify_hourly_hash(path, anchor_hashes[climate], f"Anchor {climate}")
            anchor_data[climate] = load_hourly(path)
        all_profiles = np.arange(len(group), dtype=int)
        anchor_score = score_trajectory(
            anchor_data[climate], group, all_profiles
        )
        group_tails = tail_indices(
            anchor_score["profiles"]["reference_mismatch_c"],
            group.profile_id,
            tail_fractions,
        )
        tails[key] = group_tails
        for fraction, positions in group_tails.items():
            anchor_rows.append(
                {
                    "climate": key[0],
                    "composition": key[1],
                    "layout": key[2],
                    "tail_fraction": fraction,
                    "tail_profile_count": len(positions),
                    "anchor_mean_mismatch_c": anchor_score["summary"][
                        "mean_reference_mismatch_c"
                    ],
                    "anchor_tail_mismatch_c": float(
                        anchor_score["profiles"]["reference_mismatch_c"][
                            positions
                        ].mean()
                    ),
                }
            )
            membership_rows.extend(
                {
                    "climate": key[0],
                    "composition": key[1],
                    "layout": key[2],
                    "tail_fraction": fraction,
                    "profile_id": group.profile_id.iloc[position],
                }
                for position in positions
            )

    score_cache: dict[tuple[str, tuple], tuple[np.ndarray, np.ndarray]] = {}
    rescored_rows = []
    for row in metrics.itertuples(index=False):
        key = (row.climate, row.composition, row.layout)
        group = profile_groups[key]
        path = _resolved(row.hourly_path, metrics_path)
        verify_hourly_hash(path, row.hourly_sha256, f"Annual trajectory {row.run_id}")
        cache_key = (str(row.hourly_sha256), key)
        if cache_key not in score_cache:
            data = load_hourly(path)
            all_profiles = np.arange(len(group), dtype=int)
            full = score_trajectory(data, group, all_profiles)
            mean = score_trajectory(
                data, zone_mean_profiles(group), all_profiles
            )
            score_cache[cache_key] = (
                np.asarray(full["profiles"]["reference_mismatch_c"], dtype=float),
                np.asarray(mean["profiles"]["reference_mismatch_c"], dtype=float),
            )
            if not np.isclose(
                full["summary"]["mean_reference_mismatch_c"],
                float(row.mean_reference_mismatch_c),
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(f"Full-profile mean score drift for {row.run_id}")
            if not np.isclose(
                mean["summary"]["mean_reference_mismatch_c"],
                float(row.zone_mean_mean_reference_mismatch_c),
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(f"Zone-mean score drift for {row.run_id}")
        full_profile, mean_profile = score_cache[cache_key]
        base = row._asdict()
        base.pop("manifest_hourly_path", None)
        base.pop("manifest_hourly_sha256", None)
        for fraction, positions in tails[key].items():
            rescored_rows.append(
                base
                | {
                    "tail_fraction": fraction,
                    "fixed_tail_reference_mismatch_c": float(
                        full_profile[positions].mean()
                    ),
                    "zone_mean_fixed_tail_reference_mismatch_c": float(
                        mean_profile[positions].mean()
                    ),
                }
            )

    rescored = pd.DataFrame(rescored_rows)
    anchors = pd.DataFrame(anchor_rows)
    memberships = pd.DataFrame(membership_rows)
    direct_outputs = []
    selection_outputs = []
    common_outputs = []
    information_outputs = []
    results = {}
    summary_rows = []
    for tail_fraction in tail_fractions:
        subset = rescored[np.isclose(rescored.tail_fraction, tail_fraction)].copy()
        fraction_anchors = anchors[
            np.isclose(anchors.tail_fraction, tail_fraction)
        ].copy()
        direct = subset[np.isclose(pd.to_numeric(subset.offset_c), 0.0)].copy()
        direct["selection_status"] = "selected"
        direct_pairs = pair_controls(direct, "direct")
        direct_pairs["tail_fraction"] = tail_fraction
        selections = select_common_service(
            subset, fraction_anchors, fractions=service_fractions
        )
        selections["tail_fraction"] = tail_fraction
        common_pairs = pair_controls(selections, "common_service")
        common_pairs["tail_fraction"] = tail_fraction
        information = fixed_plan_information_diagnostic(
            subset, fraction_anchors, fractions=service_fractions
        )
        information["tail_fraction"] = tail_fraction
        result = {
            "direct": _direct_summary(direct_pairs),
            "common_service": _common_summary(selections, common_pairs),
            "information": summarise_information_diagnostic(
                information, reference_target_c=reference_target_c
            ),
        }
        results[f"{tail_fraction:g}"] = result
        summary_rows.extend(_summary_rows(tail_fraction, result))
        direct_outputs.append(direct_pairs)
        selection_outputs.append(selections)
        common_outputs.append(common_pairs)
        information_outputs.append(information)

    outputs = {
        "TAIL_FRACTION_RUN_METRICS.csv": rescored,
        "TAIL_FRACTION_ANCHORS.csv": anchors,
        "TAIL_FRACTION_MEMBERSHIPS.csv": memberships,
        "TAIL_FRACTION_DIRECT_COMPARISONS.csv": pd.concat(
            direct_outputs, ignore_index=True
        ),
        "TAIL_FRACTION_COMMON_SERVICE_SELECTIONS.csv": pd.concat(
            selection_outputs, ignore_index=True
        ),
        "TAIL_FRACTION_COMMON_SERVICE_COMPARISONS.csv": pd.concat(
            common_outputs, ignore_index=True
        ),
        "TAIL_FRACTION_INFORMATION_DIAGNOSTIC.csv": pd.concat(
            information_outputs, ignore_index=True
        ),
        "TAIL_FRACTION_SUMMARY.csv": pd.DataFrame(summary_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_root / name, index=False)

    audit = {
        "status": "PASS",
        "interpretation": (
            "Tail shares are predeclared sensitivity endpoints for the fixed "
            f"least-served profiles under the common {reference_target_c:g} C "
            "reference. They are not clinical, physiological or policy "
            "vulnerability standards."
        ),
        "reference_target_c": float(reference_target_c),
        "tail_fractions": list(map(float, tail_fractions)),
        "service_fractions": list(map(float, service_fractions)),
        "counts": {
            "annual_candidates": int(metrics.run_id.nunique()),
            "profile_groups": int(len(profile_groups)),
            "rescored_candidate_rows": int(len(rescored)),
        },
        "results": results,
        "input_sha256": {
            "metrics": sha256(metrics_path),
            "run_manifest": sha256(run_manifest_path),
            "profiles": sha256(profiles_path),
            "anchor_manifest": sha256(anchor_manifest_path),
        },
        "output_sha256": {
            name: sha256(output_root / name) for name in outputs
        },
    }
    (output_root / "TAIL_FRACTION_SENSITIVITY.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tail-fractions",
        type=float,
        nargs="+",
        default=DEFAULT_TAIL_FRACTIONS,
    )
    parser.add_argument("--reference-target-c", type=float, default=25.0)
    args = parser.parse_args()
    audit = analyse_tail_fraction_sensitivity(
        args.metrics.resolve(),
        args.run_manifest.resolve(),
        args.profiles.resolve(),
        args.anchor_manifest.resolve(),
        args.output_root.resolve(),
        tail_fractions=tuple(args.tail_fractions),
        reference_target_c=args.reference_target_c,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
