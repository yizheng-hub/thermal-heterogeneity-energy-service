"""Audit compact demand summaries on the final global physical plan library.

The analysis never changes or reruns a building model.  It scores the same
annual temperature trajectories with four information representations:

* one demand centre per thermal zone;
* at most five near-equal-count centre bins per thermal zone;
* the same five bins plus margins calibrated outside both the held-out climate
  and the held-out demand composition; and
* all assigned demand profiles.

The five-bin summary carries a bin centre, a profile count and a fixed-tail
membership count.  It is an anonymous aggregate, not a formal privacy
guarantee.  Full-profile 25 C anchors define every service limit, so only the
information used to review a fixed physical plan changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.global_extension.analyse_global_results import (  # noqa: E402
    SERVICE_TOLERANCE_C,
    _deduplicate_physical_plans,
    choose_minimum_energy,
    load_hourly,
    normalise_profiles,
    score_trajectory,
    sha256,
    verify_hourly_hash,
)
from src.thermal_heterogeneity.implementation.protected_quantiles import (  # noqa: E402
    calibrate_margins,
)


DEFAULT_FRACTIONS = (1.0, 0.75, 0.5)
FINAL_EXPECTED_ROWS = 810
FINAL_EXPECTED_CLIMATES = 9
FINAL_EXPECTED_COMPOSITIONS = 6
FINAL_CONTROLS = ("uniform_average", "personal_center", "personal_range")
FINAL_OFFSETS_C = (-1.0, -0.5, 0.0, 0.5, 1.0)
REPRESENTATIONS = (
    "full profiles",
    "zone mean",
    "five bins",
    "protected five bins",
)


def _json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _hash_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _flag(series: pd.Series, name: str) -> pd.Series:
    if series.dtype == bool:
        return series
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"Invalid Boolean values in {name}")
    return parsed.astype(bool)


def five_bin_profiles(
    profiles: pd.DataFrame,
    tail_profile_ids: Iterable[str],
    *,
    max_bins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace individual centres by deterministic within-zone bin means."""
    if max_bins < 1:
        raise ValueError("max_bins must be positive")
    result = profiles.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
    result["centre_c"] = pd.to_numeric(result["centre_c"], errors="raise").astype(float)
    tail_ids = set(map(str, tail_profile_ids))
    if not tail_ids:
        raise ValueError("Five-bin scoring requires a nonempty fixed-tail group")
    unknown = tail_ids - set(result.profile_id.astype(str))
    if unknown:
        raise ValueError(f"Fixed-tail profiles are absent from the composition: {sorted(unknown)}")

    result["anonymous_bin"] = -1
    bins: list[dict] = []
    for zone, positions in result.groupby("zone_id", sort=True).groups.items():
        ordered = result.loc[list(positions)].sort_values(
            ["centre_c", "profile_id"], kind="mergesort"
        )
        count = len(ordered)
        bin_count = min(max_bins, count)
        assignments = np.floor(np.arange(count) * bin_count / count).astype(int)
        result.loc[ordered.index, "anonymous_bin"] = assignments
        for bin_index in range(bin_count):
            member_index = ordered.index[assignments == bin_index]
            members = result.loc[member_index]
            centre = float(members.centre_c.mean())
            result.loc[member_index, "centre_c"] = centre
            bins.append({
                "zone_id": str(zone),
                "bin_index": int(bin_index + 1),
                "bin_centre_c": centre,
                "profile_count": int(len(members)),
                "fixed_tail_count": int(
                    members.profile_id.astype(str).isin(tail_ids).sum()
                ),
            })
    if result.anonymous_bin.lt(0).any():
        raise ValueError("A profile was not assigned to an anonymous bin")
    return result, pd.DataFrame(bins)


def calibrate_double_holdout_margins(
    candidates: pd.DataFrame,
    *,
    quantile: float = 0.95,
) -> pd.DataFrame:
    """Calibrate margins while excluding the held-out climate and composition."""
    if not 0 < quantile <= 1:
        raise ValueError("Protection quantile must lie in (0, 1]")
    required = {
        "climate", "composition", "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c", "five_bin_mean_mismatch_c",
        "five_bin_tail_mismatch_c",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"Candidate scores lack {sorted(missing)}")
    rows = []
    scenarios = candidates[["climate", "composition"]].drop_duplicates()
    for climate, composition in scenarios.itertuples(index=False, name=None):
        training = candidates.loc[
            candidates.climate.ne(climate) & candidates.composition.ne(composition)
        ]
        if training.empty:
            raise ValueError("Double-held-out margin has no training candidates")
        mean_margin, tail_margin = calibrate_margins(
            training,
            approximate_mean="five_bin_mean_mismatch_c",
            approximate_tail="five_bin_tail_mismatch_c",
            full_mean="mean_reference_mismatch_c",
            full_tail="fixed_tail_reference_mismatch_c",
            quantile=quantile,
        )
        rows.append({
            "climate": climate,
            "composition": composition,
            "protection_quantile": quantile,
            "training_candidates": int(len(training)),
            "training_climates": int(training.climate.nunique()),
            "training_compositions": int(training.composition.nunique()),
            "mean_margin_c": mean_margin,
            "tail_margin_c": tail_margin,
        })
    return pd.DataFrame(rows).sort_values(
        ["climate", "composition"], kind="mergesort"
    ).reset_index(drop=True)


def _validate_final_scope(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
) -> None:
    required = {
        "climate", "composition", "layout", "permission", "control", "offset_c",
        "run_id", "engineering_pass", "hourly_path", "hourly_sha256",
        "physical_plan_id", "facility_purchased_kwh",
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "zone_mean_mean_reference_mismatch_c",
        "zone_mean_fixed_tail_reference_mismatch_c",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Final annual metrics lack {sorted(missing)}")
    if len(metrics) != FINAL_EXPECTED_ROWS:
        raise ValueError(
            f"Expected {FINAL_EXPECTED_ROWS} annual candidates; found {len(metrics)}"
        )
    if metrics.climate.nunique() != FINAL_EXPECTED_CLIMATES:
        raise ValueError(
            f"Expected {FINAL_EXPECTED_CLIMATES} climates; "
            f"found {metrics.climate.nunique()}"
        )
    if metrics.composition.nunique() != FINAL_EXPECTED_COMPOSITIONS:
        raise ValueError(
            f"Expected {FINAL_EXPECTED_COMPOSITIONS} demand compositions; "
            f"found {metrics.composition.nunique()}"
        )
    if set(metrics.control.astype(str)) != set(FINAL_CONTROLS):
        raise ValueError("Final annual candidates do not contain the three controls")
    offsets = tuple(sorted(pd.to_numeric(metrics.offset_c, errors="raise").unique()))
    if offsets != FINAL_OFFSETS_C:
        raise ValueError(f"Expected offsets {FINAL_OFFSETS_C}; found {offsets}")
    if metrics.run_id.astype(str).duplicated().any():
        raise ValueError("Final annual candidates duplicate a run_id")
    if metrics.physical_plan_id.astype(str).str.strip().isin({"", "nan"}).any():
        raise ValueError("Final annual candidates contain a blank physical plan identity")
    energy = pd.to_numeric(metrics.facility_purchased_kwh, errors="raise")
    if not np.isfinite(energy).all() or energy.le(0).any():
        raise ValueError("Final annual candidates require positive finite facility energy")
    scenarios = metrics[["climate", "composition", "layout", "permission"]]
    if len(scenarios.drop_duplicates()) != 54:
        raise ValueError("Expected 54 climate-composition information scenarios")
    pairs = metrics[["climate", "composition"]].drop_duplicates()
    if len(pairs) != FINAL_EXPECTED_CLIMATES * FINAL_EXPECTED_COMPOSITIONS:
        raise ValueError("The final audit requires the full 9-by-6 scenario matrix")
    sizes = metrics.groupby(
        ["climate", "composition", "layout", "permission"], sort=True
    ).size()
    if not sizes.eq(15).all():
        raise ValueError("Every information scenario must contain 15 declared plan aliases")
    plan_grid = metrics.groupby(
        ["climate", "composition", "layout", "permission"], sort=True
    ).apply(
        lambda group: set(zip(
            group.control.astype(str),
            pd.to_numeric(group.offset_c).astype(float),
            strict=True,
        )),
        include_groups=False,
    )
    expected_grid = set(zip(
        np.repeat(FINAL_CONTROLS, len(FINAL_OFFSETS_C)),
        np.tile(FINAL_OFFSETS_C, len(FINAL_CONTROLS)),
        strict=True,
    ))
    if not plan_grid.map(lambda value: value == expected_grid).all():
        raise ValueError("Every information scenario must contain the final 3-by-5 plan grid")
    if profiles.duplicated(["climate", "composition", "layout", "profile_id"]).any():
        raise ValueError("A profile identity is repeated within an assigned composition")
    profile_sizes = profiles.groupby(
        ["climate", "composition", "layout"], sort=True
    ).size()
    if len(profile_sizes) != 54 or not profile_sizes.eq(267).all():
        raise ValueError("Expected 267 assigned profiles in each of 54 compositions")


def score_five_bin_candidates(
    metrics: pd.DataFrame,
    profiles: pd.DataFrame,
    tails: pd.DataFrame,
    *,
    metrics_source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every completed physical trajectory with five centre bins per zone."""
    profiles_by_key = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(
            ["climate", "composition", "layout"], sort=True
        )
    }
    tails_by_key = {
        key: group.profile_id.astype(str).tolist()
        for key, group in tails.groupby(
            ["climate", "composition", "layout"], sort=True
        )
    }
    if set(profiles_by_key) != set(tails_by_key):
        raise ValueError("Profile groups and fixed-tail memberships differ")

    binned: dict[tuple, pd.DataFrame] = {}
    bin_rows = []
    tail_positions: dict[tuple, np.ndarray] = {}
    for key, group in profiles_by_key.items():
        binned_group, bin_table = five_bin_profiles(group, tails_by_key[key])
        binned[key] = binned_group
        tail_set = set(tails_by_key[key])
        tail_positions[key] = np.flatnonzero(
            group.profile_id.astype(str).isin(tail_set).to_numpy()
        )
        if len(tail_positions[key]) != 27:
            raise ValueError(f"Expected 27 fixed-tail profiles for {key}")
        prefix = dict(zip(["climate", "composition", "layout"], key, strict=True))
        bin_rows.extend(prefix | row for row in bin_table.to_dict("records"))

    hash_cache: dict[str, str] = {}
    score_cache: dict[tuple[str, tuple], tuple[float, float]] = {}
    rows = []
    for row in metrics.itertuples(index=False):
        key = (row.climate, row.composition, row.layout)
        path = Path(str(row.hourly_path))
        if not path.is_absolute():
            path = metrics_source.parent / path
        resolved_path = str(path.resolve())
        if resolved_path not in hash_cache:
            hash_cache[resolved_path] = verify_hourly_hash(
                path, row.hourly_sha256, f"Annual trajectory {row.run_id}"
            )
        elif str(row.hourly_sha256).strip().lower() != hash_cache[resolved_path]:
            raise ValueError(
                f"Annual trajectory {row.run_id} declares a different SHA-256 "
                "for a reused hourly path"
            )
        score_key = (resolved_path, key)
        if score_key not in score_cache:
            score = score_trajectory(
                load_hourly(path), binned[key], tail_positions[key]
            )["summary"]
            score_cache[score_key] = (
                float(score["mean_reference_mismatch_c"]),
                float(score["fixed_tail_reference_mismatch_c"]),
            )
        mean_score, tail_score = score_cache[score_key]
        item = row._asdict()
        item.update({
            "five_bin_mean_mismatch_c": mean_score,
            "five_bin_tail_mismatch_c": tail_score,
        })
        rows.append(item)
    result = pd.DataFrame(rows)
    engineering = _flag(result.engineering_pass, "engineering_pass")
    if not engineering.all():
        failed = result.loc[~engineering, "run_id"].astype(str).tolist()
        raise ValueError(f"Final information audit contains unqualified runs: {failed[:5]}")
    return result, pd.DataFrame(bin_rows)


def validate_support_tables(
    profiles: pd.DataFrame,
    anchors: pd.DataFrame,
    tails: pd.DataFrame,
) -> None:
    """Require anchors and fixed-tail memberships for the same final 54 cells."""
    keys = ["climate", "composition", "layout"]
    expected = set(map(tuple, profiles[keys].drop_duplicates().to_numpy()))
    anchor_required = {*keys, "anchor_mean_mismatch_c", "anchor_tail_mismatch_c"}
    tail_required = {*keys, "profile_id"}
    if missing := anchor_required - set(anchors):
        raise ValueError(f"Service anchors lack {sorted(missing)}")
    if missing := tail_required - set(tails):
        raise ValueError(f"Fixed-tail memberships lack {sorted(missing)}")
    if anchors.duplicated(keys).any():
        raise ValueError("Service anchors duplicate a climate-composition cell")
    if set(map(tuple, anchors[keys].to_numpy())) != expected:
        raise ValueError("Service anchors do not match the final 54 profile cells")
    if set(map(tuple, tails[keys].drop_duplicates().to_numpy())) != expected:
        raise ValueError("Fixed-tail memberships do not match the final 54 profile cells")
    tail_sizes = tails.groupby(keys, sort=True).size()
    if not tail_sizes.eq(27).all():
        raise ValueError("Every final profile cell must contain 27 fixed-tail members")
    if tails.duplicated([*keys, "profile_id"]).any():
        raise ValueError("A fixed-tail profile is repeated within a profile cell")


def deduplicate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact aliases and verify their five-bin scores agree."""
    for _, aliases in candidates.groupby("physical_plan_id", sort=True):
        for column in ("five_bin_mean_mismatch_c", "five_bin_tail_mismatch_c"):
            values = aliases[column].to_numpy(float)
            scale = max(1.0, float(np.max(np.abs(values))))
            if float(np.ptp(values)) > 1e-9 * scale:
                raise ValueError("Physical aliases have inconsistent five-bin scores")
    return _deduplicate_physical_plans(candidates)


def select_information_decisions(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    margins: pd.DataFrame,
    *,
    fractions: Iterable[float] = DEFAULT_FRACTIONS,
) -> pd.DataFrame:
    """Select the least-energy qualified plan under each information level."""
    keys = ["climate", "composition", "layout"]
    joined = candidates.merge(
        anchors[keys + ["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]],
        on=keys,
        how="left",
        validate="many_to_one",
    ).merge(
        margins[["climate", "composition", "mean_margin_c", "tail_margin_c"]],
        on=["climate", "composition"],
        how="left",
        validate="many_to_one",
    )
    if joined[[
        "anchor_mean_mismatch_c", "anchor_tail_mismatch_c",
        "mean_margin_c", "tail_margin_c",
    ]].isna().any().any():
        raise ValueError("Decision audit has missing service anchors or margins")

    score_columns = {
        "full profiles": (
            "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c", 0.0, 0.0,
        ),
        "zone mean": (
            "zone_mean_mean_reference_mismatch_c",
            "zone_mean_fixed_tail_reference_mismatch_c", 0.0, 0.0,
        ),
        "five bins": (
            "five_bin_mean_mismatch_c", "five_bin_tail_mismatch_c", 0.0, 0.0,
        ),
        "protected five bins": (
            "five_bin_mean_mismatch_c", "five_bin_tail_mismatch_c",
            "mean_margin_c", "tail_margin_c",
        ),
    }
    rows = []
    group_keys = ["climate", "composition", "layout", "permission"]
    for key, group in joined.groupby(group_keys, sort=True, dropna=False):
        if group.physical_plan_id.astype(str).duplicated().any():
            raise ValueError("Decision input must contain one row per physical plan")
        plans = group.copy()
        for fraction in map(float, fractions):
            mean_limit = fraction * float(plans.anchor_mean_mismatch_c.iloc[0])
            tail_limit = fraction * float(plans.anchor_tail_mismatch_c.iloc[0])
            eligibility: dict[str, set[str]] = {}
            choices: dict[str, dict | None] = {}
            for representation, (mean_col, tail_col, mean_extra, tail_extra) in (
                score_columns.items()
            ):
                mean_margin = (
                    plans[mean_extra].to_numpy(float)
                    if isinstance(mean_extra, str)
                    else float(mean_extra)
                )
                tail_margin = (
                    plans[tail_extra].to_numpy(float)
                    if isinstance(tail_extra, str)
                    else float(tail_extra)
                )
                mask = (
                    plans[mean_col].to_numpy(float) + mean_margin
                    <= mean_limit + SERVICE_TOLERANCE_C
                ) & (
                    plans[tail_col].to_numpy(float) + tail_margin
                    <= tail_limit + SERVICE_TOLERANCE_C
                )
                qualified = plans.loc[mask]
                eligibility[representation] = set(
                    qualified.physical_plan_id.astype(str)
                )
                choices[representation] = choose_minimum_energy(qualified)
            full_ids = eligibility["full profiles"]
            full_choice = choices["full profiles"]
            full_id = str(full_choice["physical_plan_id"]) if full_choice else ""
            full_energy = (
                float(full_choice["facility_purchased_kwh"])
                if full_choice else float("nan")
            )
            prefix = dict(zip(group_keys, key, strict=True)) | {
                "service_fraction": fraction,
                "mean_limit_c": mean_limit,
                "tail_limit_c": tail_limit,
                "physical_plan_count": int(len(plans)),
                "declared_alias_count": int(plans.physical_alias_count.sum()),
                "full_resolvable": bool(full_choice),
                "full_selected_plan_id": full_id,
                "full_selected_energy_kwh": full_energy,
            }
            for representation in REPRESENTATIONS:
                choice = choices[representation]
                selected_id = str(choice["physical_plan_id"]) if choice else ""
                selected_energy = (
                    float(choice["facility_purchased_kwh"])
                    if choice else float("nan")
                )
                represented_ids = eligibility[representation]
                safe_selection = bool(choice and selected_id in full_ids)
                rows.append(prefix | {
                    "representation": representation,
                    "qualified_plan_count": int(len(represented_ids)),
                    "full_qualified_plan_count": int(len(full_ids)),
                    "qualified_under_both_count": int(len(represented_ids & full_ids)),
                    "candidate_false_accept_count": int(len(represented_ids - full_ids)),
                    "candidate_false_reject_count": int(len(full_ids - represented_ids)),
                    "selection_status": "selected" if choice else "no qualifying plan",
                    "selected_plan_id": selected_id,
                    "selected_energy_kwh": selected_energy,
                    "selected_plan_passes_full_review": safe_selection,
                    "selected_plan_false_accept": bool(choice and not safe_selection),
                    "false_reject_no_selection": bool(full_choice and not choice),
                    "exact_selection_agreement": bool(
                        choice and full_choice and selected_id == full_id
                    ),
                    "safe_energy_penalty_percent": (
                        100.0 * (selected_energy - full_energy) / full_energy
                        if safe_selection and full_choice and full_energy > 0
                        else float("nan")
                    ),
                })
    return pd.DataFrame(rows)


def summarise_decisions(decisions: pd.DataFrame) -> dict:
    """Summarise primary full-resolvable decisions and threshold sensitivities."""
    def block(frame: pd.DataFrame) -> dict:
        output = {}
        for representation, group in frame.groupby("representation", sort=False):
            full_qualified = int(group.full_qualified_plan_count.sum())
            retained = int(group.qualified_under_both_count.sum())
            penalties = group.safe_energy_penalty_percent.dropna().to_numpy(float)
            output[str(representation)] = {
                "scenarios": int(len(group)),
                "selections": int(group.selection_status.eq("selected").sum()),
                "decision_coverage": float(
                    group.selection_status.eq("selected").mean() if len(group) else 0.0
                ),
                "selected_plan_false_acceptances": int(
                    group.selected_plan_false_accept.sum()
                ),
                "safe_selections": int(
                    group.selected_plan_passes_full_review.sum()
                ),
                "safe_decision_coverage": float(
                    group.selected_plan_passes_full_review.mean()
                    if len(group) else 0.0
                ),
                "false_reject_no_selection": int(group.false_reject_no_selection.sum()),
                "exact_selection_agreements": int(group.exact_selection_agreement.sum()),
                "exact_selection_agreement_rate": float(
                    group.exact_selection_agreement.mean() if len(group) else 0.0
                ),
                "candidate_false_acceptances": int(
                    group.candidate_false_accept_count.sum()
                ),
                "candidate_false_rejections": int(
                    group.candidate_false_reject_count.sum()
                ),
                "retained_full_qualified": retained,
                "full_qualified": full_qualified,
                "qualified_action_retention": float(
                    retained / full_qualified if full_qualified else 0.0
                ),
                "safe_energy_penalty_median_percent": (
                    float(np.median(penalties)) if penalties.size else None
                ),
                "safe_energy_penalty_p95_percent": (
                    float(np.quantile(penalties, 0.95)) if penalties.size else None
                ),
            }
        return output

    primary_all = decisions.loc[np.isclose(decisions.service_fraction, 1.0)]
    primary = primary_all.loc[primary_all.full_resolvable]
    if len(primary_all[["climate", "composition"]].drop_duplicates()) != 54:
        raise ValueError("Primary information summary must contain 54 scenarios")
    sensitivity = {}
    for fraction, group in decisions.groupby("service_fraction", sort=False):
        sensitivity[f"{float(fraction):g}"] = {
            "all_scenarios": block(group),
            "full_resolvable": block(group.loc[group.full_resolvable]),
        }
    return {
        "headline_scope": (
            "service_fraction=1.0 and scenarios resolvable under full profiles; "
            "0.75 and 0.5 are threshold sensitivities"
        ),
        "primary_full_resolvable_scenarios": int(
            len(primary[["climate", "composition"]].drop_duplicates())
        ),
        "primary": block(primary),
        "sensitivity_by_service_fraction": sensitivity,
        "information_values_per_scenario": {
            "zone mean": {
                "centre_values": 15,
                "group_counts": 15,
                "fixed_tail_counts": 15,
                "total_values": 45,
            },
            "five bins": {
                "centre_values": 75,
                "group_counts": 75,
                "fixed_tail_counts": 75,
                "total_values": 225,
            },
            "protected five bins": {
                "centre_values": 75,
                "group_counts": 75,
                "fixed_tail_counts": 75,
                "cross-scenario protection margins": 2,
                "total_values": 227,
            },
            "full profiles": {
                "centre_values": 267,
                "individual_tail_flags": 267,
                "total_values": 534,
            },
        },
        "privacy_interpretation": (
            "The bin tables contain anonymous aggregate centres and counts; "
            "they do not provide a formal privacy guarantee."
        ),
    }


def run_audit(
    metrics_path: Path,
    anchors_path: Path,
    tails_path: Path,
    profiles_path: Path,
    output_root: Path,
    *,
    protection_quantile: float = 0.95,
) -> dict:
    """Execute and save the zero-simulation information-compression audit."""
    inputs = {
        "metrics": metrics_path.resolve(),
        "anchors": anchors_path.resolve(),
        "tails": tails_path.resolve(),
        "profiles": profiles_path.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} input: {path}")
    metrics = pd.read_csv(inputs["metrics"])
    metrics["engineering_pass"] = _flag(metrics.engineering_pass, "engineering_pass")
    profiles = normalise_profiles(pd.read_csv(inputs["profiles"]))
    anchors = pd.read_csv(inputs["anchors"])
    tails = pd.read_csv(inputs["tails"])
    _validate_final_scope(metrics, profiles)
    validate_support_tables(profiles, anchors, tails)
    scored, bins = score_five_bin_candidates(
        metrics, profiles, tails, metrics_source=inputs["metrics"]
    )
    bin_groups = bins.groupby(["climate", "composition", "layout"], sort=True)
    if len(bin_groups) != 54 or not bin_groups.size().eq(75).all():
        raise ValueError("Expected five anonymous bins in each of 15 zones")
    if not bin_groups.profile_count.sum().eq(267).all():
        raise ValueError("Anonymous bin counts do not preserve 267 profiles")
    if not bin_groups.fixed_tail_count.sum().eq(27).all():
        raise ValueError("Anonymous bin tail counts do not preserve the fixed 10% group")
    candidates = []
    for _, group in scored.groupby(
        ["climate", "composition", "layout", "permission"], sort=True
    ):
        candidates.append(deduplicate_candidates(group))
    candidates = pd.concat(candidates, ignore_index=True)
    margins = calibrate_double_holdout_margins(
        candidates, quantile=protection_quantile
    )
    if len(margins) != 54:
        raise ValueError("Expected one double-held-out margin for each scenario")
    decisions = select_information_decisions(candidates, anchors, margins)
    if len(decisions) != 54 * len(DEFAULT_FRACTIONS) * len(REPRESENTATIONS):
        raise ValueError("Information decision table is incomplete")
    summary = summarise_decisions(decisions)
    summary.update({
        "status": "PASS",
        "annual_candidate_aliases": int(len(metrics)),
        "unique_physical_plans": int(len(candidates)),
        "climates": int(metrics.climate.nunique()),
        "demand_compositions": int(metrics.composition.nunique()),
        "design_coverage": {
            "climate_names": sorted(metrics.climate.astype(str).unique()),
            "demand_composition_names": sorted(
                metrics.composition.astype(str).unique()
            ),
            "climate_composition_cells": 54,
            "declared_plan_aliases_per_cell": 15,
            "full_cartesian_matrix": True,
        },
        "protection_split": (
            "for each scenario, exclude both its climate and its demand composition"
        ),
        "protection_quantile": protection_quantile,
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
    })

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "GLOBAL_INFORMATION_COMPRESSION_BINS.csv": bins,
        "GLOBAL_INFORMATION_COMPRESSION_CANDIDATES.csv": candidates,
        "GLOBAL_INFORMATION_COMPRESSION_MARGINS.csv": margins,
        "GLOBAL_INFORMATION_COMPRESSION_DECISIONS.csv": decisions,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_root / name, index=False, lineterminator="\n")
    summary_path = output_root / "GLOBAL_INFORMATION_COMPRESSION_SUMMARY.json"
    summary_path.write_text(_json_text(summary), encoding="utf-8")
    output_hashes = {
        name: sha256(output_root / name) for name in outputs
    } | {summary_path.name: sha256(summary_path)}
    hashes = {
        "input_sha256": summary["input_sha256"],
        "output_sha256": output_hashes,
        "in_memory_table_sha256": {
            name: _hash_frame(frame) for name, frame in outputs.items()
        },
    }
    (output_root / "GLOBAL_INFORMATION_COMPRESSION_HASHES.json").write_text(
        _json_text(hashes), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--tails", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protection-quantile", type=float, default=0.95)
    args = parser.parse_args()
    result = run_audit(
        args.metrics,
        args.anchors,
        args.tails,
        args.profiles,
        args.output_root,
        protection_quantile=args.protection_quantile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
