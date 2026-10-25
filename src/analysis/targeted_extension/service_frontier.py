"""Finite-candidate service/energy analysis; no trajectories are interpolated.

The input is the retained annual analysis directory. Common requirements are
upper bounds on mean and fixed-tail reference mismatch, not identical realized
service. Every engineering-qualified trajectory, including common 25 C, stays
in its declared control family. Duplicate floor/building permissions are omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITION = ["context", "composition_id", "layout", "permission"]
PROFILE = CONDITION[:3]
CONTROLS = ["uniform_average", "personal_center", "ltpref_interval"]
SERVICE = ["mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"]
ENERGY = "annual_site_kwh"
CARBON = "annual_operational_co2_kg"
TOL = 1e-9
NEAR_SERVICE_TOLERANCES_C = (0.02, 0.05, 0.10)
PRIMARY_NEAR_SERVICE_TOLERANCE_C = 0.05


def energy_choice(pool):
    """Use the retained deterministic energy/tie selection rule."""
    chosen = None
    for row in pool.sort_values(["run_id", "variant"]).to_dict("records"):
        if chosen is None:
            chosen = row
        else:
            threshold = max(1e-6, 1e-9 * max(abs(row[ENERGY]), abs(chosen[ENERGY])))
            if row[ENERGY] < chosen[ENERGY] - threshold:
                chosen = row
    return chosen


def pareto_mask(values):
    """True for non-dominated rows when every supplied objective is minimized."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto objectives must be finite")
    return np.array([
        not np.any(np.all(values <= point + TOL, axis=1)
                   & np.any(values < point - TOL, axis=1))
        for point in values
    ])


def canonical_candidates(candidates):
    """Retain actual qualified candidates and remove only duplicate identities."""
    candidates = candidates.copy()
    candidates["permission_alias"] = candidates.permission_alias.fillna("")
    candidates = candidates[candidates.permission_alias.eq("")]
    candidates = candidates[candidates.engineering_status.isin({"cached_qualified", "new_qualified"})]
    if not np.isfinite(candidates[SERVICE + [ENERGY, CARBON]].to_numpy()).all():
        raise ValueError("Qualified candidate scores must be finite")
    for _, group in candidates.groupby(CONDITION + ["control", "run_id"], sort=True):
        if np.ptp(group[SERVICE + [ENERGY, CARBON]].to_numpy(), axis=0).max() > 1e-8:
            raise ValueError("The same physical run has inconsistent scores within a condition")
    candidates = candidates.sort_values(CONDITION + ["control", "run_id", "variant"])
    return candidates.drop_duplicates(CONDITION + ["control", "run_id"]).reset_index(drop=True)


def paired_fields(reference, alternative, anchor_run):
    """Record both service values, not just whether common limits were passed."""
    item = {"reference_status": "selected" if reference else "service_unavailable",
            "alternative_status": "selected" if alternative else "service_unavailable",
            "status": "matched" if reference and alternative else "unmatched"}
    for name, chosen in (("reference", reference), ("alternative", alternative)):
        item[name + "_run_id"] = chosen["run_id"] if chosen else ""
        item[name + "_common25"] = bool(chosen and chosen["run_id"] == anchor_run)
        for metric in SERVICE + [ENERGY, CARBON]:
            item[name + "_" + metric] = chosen[metric] if chosen else np.nan
    if reference and alternative:
        item["same_physical_run"] = reference["run_id"] == alternative["run_id"]
        item["both_common25"] = item["reference_common25"] and item["alternative_common25"]
        item["delta_site_pct"] = 100 * (alternative[ENERGY] / reference[ENERGY] - 1)
        item["delta_carbon_pct"] = 100 * (alternative[CARBON] / reference[CARBON] - 1)
        for metric in SERVICE:
            item["delta_" + metric] = alternative[metric] - reference[metric]
    else:
        item.update(same_physical_run=False, both_common25=False,
                    delta_site_pct=np.nan, delta_carbon_pct=np.nan)
        for metric in SERVICE:
            item["delta_" + metric] = np.nan
    return item


def matched_direct_common(candidates, selections, comparisons, anchors):
    """Audit all budget-one pairs and the same direct/common matched subset."""
    selected = selections[selections.service_fraction.eq(1.0)]
    direct = comparisons[comparisons.comparison_mode.eq("direct")]
    rows = []
    for key, group in candidates.groupby(CONDITION, sort=True):
        condition = dict(zip(CONDITION, key))
        selected_group = selected
        direct_group = direct
        anchor = anchors
        for col, value in condition.items():
            selected_group = selected_group[selected_group[col].eq(value)]
            direct_group = direct_group[direct_group[col].eq(value)]
            if col in PROFILE:
                anchor = anchor[anchor[col].eq(value)]
        anchor_run = anchor.iloc[0].anchor_run_id
        picks = {}
        for control in CONTROLS:
            row = selected_group[selected_group.control.eq(control)]
            if len(row) != 1:
                raise ValueError("Expected one common-requirement selection per control")
            record = row.iloc[0].to_dict()
            picks[control] = record if record["selection_status"] == "selected" else None
        for control in CONTROLS[1:]:
            common = paired_fields(picks[CONTROLS[0]], picks[control], anchor_run)
            prior = direct_group[direct_group.control.eq(control)]
            if len(prior) != 1:
                raise ValueError("Expected one direct control comparison")
            prior = prior.iloc[0]
            rows.append(condition | {"control": control, "service_fraction": 1.0,
                        "direct_status": prior.status,
                        "direct_delta_site_pct": prior.delta_site_pct,
                        "direct_alternative_run_id": prior.alternative_run_id,
                        "direct_reference_run_id": prior.reference_run_id,
                        "both_direct_common_matched": prior.status == "matched" and common["status"] == "matched",
                        "mean_budget_c": picks[CONTROLS[0]]["mean_budget_c"],
                        "fixed_tail_budget_c": picks[CONTROLS[0]]["fixed_tail_budget_c"]} | common)
    return pd.DataFrame(rows)


def observed_requirement_frontiers(candidates, anchors):
    """Evaluate exact observed service-coordinate pairs as exploratory limits.

    Each node is an actual trajectory's two service scores, not a synthetic
    cross-product or an interpolated control. Selections are always real runs.
    """
    output = []
    for key, group in candidates.groupby(CONDITION, sort=True):
        condition = dict(zip(CONDITION, key))
        anchor = anchors
        for col in PROFILE:
            anchor = anchor[anchor[col].eq(condition[col])]
        anchor_run = anchor.iloc[0].anchor_run_id
        nodes = group.sort_values(SERVICE + ["run_id"]).drop_duplicates(SERVICE)
        for node_id, node in enumerate(nodes.to_dict("records")):
            picks = {}
            for control in CONTROLS:
                pool = group[group.control.eq(control)]
                eligible = pool[(pool[SERVICE[0]] <= node[SERVICE[0]] + TOL)
                                & (pool[SERVICE[1]] <= node[SERVICE[1]] + TOL)]
                picks[control] = energy_choice(eligible)
            for control in CONTROLS[1:]:
                output.append(condition | {"node_id": node_id, "control": control,
                              "mean_budget_c": node[SERVICE[0]],
                              "fixed_tail_budget_c": node[SERVICE[1]],
                              "node_source_run_id": node["run_id"]}
                              | paired_fields(picks[CONTROLS[0]], picks[control], anchor_run))
    return pd.DataFrame(output)


def near_service_pairs(candidates):
    """All real pairings within declared two-score tolerances; retain no match."""
    output = []
    for key, group in candidates.groupby(CONDITION, sort=True):
        condition = dict(zip(CONDITION, key))
        reference = group[group.control.eq(CONTROLS[0])]
        for control in CONTROLS[1:]:
            alternative = group[group.control.eq(control)]
            for tolerance in NEAR_SERVICE_TOLERANCES_C:
                matches = []
                for ref in reference.to_dict("records"):
                    for alt in alternative.to_dict("records"):
                        if all(abs(alt[s] - ref[s]) <= tolerance + TOL for s in SERVICE):
                            matches.append(condition | {"control": control, "tolerance_c": tolerance,
                                           "tolerance_role": "primary" if tolerance == PRIMARY_NEAR_SERVICE_TOLERANCE_C else "sensitivity",
                                           "status": "matched", "reference_run_id": ref["run_id"],
                                           "alternative_run_id": alt["run_id"],
                                           "same_physical_run": ref["run_id"] == alt["run_id"],
                                           "delta_site_pct": 100 * (alt[ENERGY] / ref[ENERGY] - 1),
                                           "delta_carbon_pct": 100 * (alt[CARBON] / ref[CARBON] - 1),
                                           "delta_mean_reference_mismatch_c": alt[SERVICE[0]] - ref[SERVICE[0]],
                                           "delta_fixed_tail_reference_mismatch_c": alt[SERVICE[1]] - ref[SERVICE[1]]})
                output.extend(matches or [condition | {"control": control, "tolerance_c": tolerance,
                              "tolerance_role": "primary" if tolerance == PRIMARY_NEAR_SERVICE_TOLERANCE_C else "sensitivity",
                              "status": "no_match", "same_physical_run": False}])
    return pd.DataFrame(output)


def near_condition_summary(pairs):
    rows = []
    for key, group in pairs.groupby(CONDITION + ["control", "tolerance_c", "tolerance_role"], sort=True):
        matched = group[group.status.eq("matched")]
        distinct = matched[~matched.same_physical_run]
        record = dict(zip(CONDITION + ["control", "tolerance_c", "tolerance_role"], key)) | {
            "candidate_pairs": len(matched), "distinct_run_candidate_pairs": len(distinct),
            "status": "distinct_run_matches" if len(distinct) else "shared_run_only" if len(matched) else "no_match",
        }
        for prefix, subset in (("all", matched), ("distinct_run", distinct)):
            record[prefix + "_median_delta_site_pct"] = subset.delta_site_pct.median()
            record[prefix + "_median_absolute_delta_site_pct"] = subset.delta_site_pct.abs().median()
            record[prefix + "_minimum_delta_site_pct"] = subset.delta_site_pct.min()
            record[prefix + "_maximum_delta_site_pct"] = subset.delta_site_pct.max()
        rows.append(record)
    return pd.DataFrame(rows)


def verify_existing_selections(candidates, selections):
    """Independently reselect every original budget without modifying its pool."""
    grouped = {key: group for key, group in candidates.groupby(CONDITION + ["control"], sort=True)}
    for selection in selections.to_dict("records"):
        pool = grouped[tuple(selection[col] for col in CONDITION + ["control"])]
        pool = pool[pool.engineering_status.isin({"cached_qualified", "new_qualified"})]
        pool = pool[(pool[SERVICE[0]] <= selection["mean_budget_c"] + TOL)
                    & (pool[SERVICE[1]] <= selection["fixed_tail_budget_c"] + TOL)]
        chosen = energy_choice(pool)
        if bool(chosen) != (selection["selection_status"] == "selected"):
            raise ValueError("Existing selection availability differs from independent reselection")
        if chosen and chosen["run_id"] != selection["run_id"]:
            raise ValueError("Existing selected trajectory differs from independent reselection")


def summarize_pairs(frame, by, metric="delta_site_pct"):
    """Descriptive medians, with condition counts and no-match denominators."""
    records = []
    for key, group in frame.groupby(by, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        valid = group[group.status.eq("matched")]
        nonidentical = valid[~valid.same_physical_run]
        record = dict(zip(by, key)) | {"records": len(group), "matched_records": len(valid),
                  "conditions": len(group[CONDITION].drop_duplicates()),
                  "matched_conditions": len(valid[CONDITION].drop_duplicates()),
                  "nonidentical_records": len(nonidentical),
                  "nonidentical_conditions": len(nonidentical[CONDITION].drop_duplicates())}
        for label, subset in (("all", valid), ("nonidentical", nonidentical)):
            # First summarize within a condition to avoid weighting contexts by
            # how many observed nodes or approximately matching runs they have.
            medians = subset.groupby(CONDITION, sort=True)[metric].median()
            record[label + "_condition_median_delta_site_pct"] = medians.median()
            record[label + "_minimum_delta_site_pct"] = subset[metric].min()
            record[label + "_maximum_delta_site_pct"] = subset[metric].max()
            record[label + "_negative_records"] = int((subset[metric] < -1e-6).sum())
            record[label + "_zero_records"] = int((subset[metric].abs() <= 1e-6).sum())
            record[label + "_positive_records"] = int((subset[metric] > 1e-6).sum())
        records.append(record)
    return pd.DataFrame(records)


def run(input_root, output_root):
    names = ["ANNUAL_ALL_CANDIDATES.csv", "ANNUAL_SERVICE_SELECTIONS.csv",
             "CONTROL_COMPARISONS.csv", "ANNUAL_ANCHORS.csv"]
    raw, selections, comparisons, anchors = [pd.read_csv(input_root / name) for name in names]
    verify_existing_selections(raw, selections)
    candidates = canonical_candidates(raw)
    candidates["pareto_in_family"] = False
    candidates["pareto_across_families"] = False
    for _, group in candidates.groupby(CONDITION + ["control"], sort=True):
        candidates.loc[group.index, "pareto_in_family"] = pareto_mask(group[[ENERGY] + SERVICE])
    for _, group in candidates.groupby(CONDITION, sort=True):
        candidates.loc[group.index, "pareto_across_families"] = pareto_mask(group[[ENERGY] + SERVICE])
    audit = matched_direct_common(candidates, selections, comparisons, anchors)
    nodes = observed_requirement_frontiers(candidates, anchors)
    near = near_service_pairs(candidates)
    summary = summarize_pairs(audit, ["control"])
    node_summary = summarize_pairs(nodes, ["context", "control"])
    near_summary = summarize_pairs(near, ["context", "control", "tolerance_c"])
    near_conditions = near_condition_summary(near)
    tolerance_summary = []
    for tolerance in NEAR_SERVICE_TOLERANCES_C:
        cells = near_conditions[near_conditions.tolerance_c.eq(tolerance)
                                & near_conditions.status.eq("distinct_run_matches")]
        pairs = near[near.tolerance_c.eq(tolerance) & near.status.eq("matched")
                     & ~near.same_physical_run]
        tolerance_summary.append({
            "tolerance_c": tolerance,
            "tolerance_role": "primary" if tolerance == PRIMARY_NEAR_SERVICE_TOLERANCE_C else "sensitivity",
            "designed_control_conditions": len(near_conditions[near_conditions.tolerance_c.eq(tolerance)]),
            "distinct_run_matched_control_conditions": len(cells),
            "distinct_run_matched_spatial_conditions": len(cells[CONDITION].drop_duplicates()),
            "distinct_run_candidate_pairs": len(pairs),
            "negative_pairs": int(pairs.delta_site_pct.lt(-1e-6).sum()),
            "zero_pairs": int(pairs.delta_site_pct.abs().le(1e-6).sum()),
            "positive_pairs": int(pairs.delta_site_pct.gt(1e-6).sum()),
            "minimum_delta_site_pct": pairs.delta_site_pct.min(),
            "maximum_delta_site_pct": pairs.delta_site_pct.max(),
            "control_condition_median_signed_delta_site_pct": cells.distinct_run_median_delta_site_pct.median(),
            "control_condition_median_absolute_delta_site_pct": cells.distinct_run_median_absolute_delta_site_pct.median(),
        })
    pooled_runs = candidates.groupby(CONDITION + ["run_id"], sort=True).agg(
        control_families=("control", lambda values: "|".join(sorted(set(values)))),
        pareto_across_families=("pareto_across_families", "first"),
    ).reset_index()
    pooled_front = pooled_runs[pooled_runs.pareto_across_families].copy()
    cohort = audit[audit.both_direct_common_matched]
    strata = []
    for label, mask in (("all", np.ones(len(cohort), dtype=bool)),
                        ("same_physical", cohort.same_physical_run),
                        ("nonidentical_physical", ~cohort.same_physical_run),
                        ("both_common25", cohort.both_common25),
                        ("different_with_either_common25", ~cohort.same_physical_run
                         & (cohort.reference_common25 | cohort.alternative_common25)),
                        ("different_neither_common25", ~cohort.same_physical_run
                         & ~cohort.reference_common25 & ~cohort.alternative_common25)):
        subset = cohort[mask]
        strata.append({"stratum": label, "pairs": len(subset),
                       "median_absolute_direct_delta_site_pct": subset.direct_delta_site_pct.abs().median(),
                       "median_absolute_common_delta_site_pct": subset.delta_site_pct.abs().median(),
                       "median_signed_common_delta_site_pct": subset.delta_site_pct.median(),
                       "common_delta_site_pct_min": subset.delta_site_pct.min(),
                       "common_delta_site_pct_max": subset.delta_site_pct.max(),
                       "mean_service_difference_abs_median_c": subset["delta_" + SERVICE[0]].abs().median(),
                       "tail_service_difference_abs_median_c": subset["delta_" + SERVICE[1]].abs().median()})
    output_root.mkdir(parents=True, exist_ok=True)
    frames = {"CANDIDATE_PARETO_POINTS.csv": candidates,
              "DIRECT_COMMON_PAIR_AUDIT.csv": audit,
              "DIRECT_COMMON_STRATA.csv": pd.DataFrame(strata),
              "COMMON_PAIR_SUMMARY.csv": summary,
              "OBSERVED_SERVICE_FRONTIERS.csv": nodes,
              "OBSERVED_SERVICE_FRONTIER_SUMMARY.csv": node_summary,
              "NEAR_SERVICE_PAIRS.csv": near,
              "NEAR_SERVICE_CONDITION_SUMMARY.csv": near_conditions,
              "NEAR_SERVICE_SUMMARY.csv": near_summary,
              "NEAR_SERVICE_TOLERANCE_SUMMARY.csv": pd.DataFrame(tolerance_summary),
              "POOLED_PARETO_RUNS.csv": pooled_front}
    for name, frame in frames.items():
        frame.to_csv(output_root / name, index=False)
    manifest = {
        "status": "PASS", "physical_parameters_changed": False, "new_physical_simulations": 0,
        "input_sha256": {name: hashlib.sha256((input_root / name).read_bytes()).hexdigest() for name in names},
        "qualified_logical_candidates_including_aliases": int(raw.engineering_status.isin({"cached_qualified", "new_qualified"}).sum()),
        "qualified_unique_condition_family_runs": len(candidates),
        "distinct_spatial_conditions": len(candidates[CONDITION].drop_duplicates()),
        "paired_direct_common_budget_one": len(cohort),
        "same_physical_pairs_in_paired_subset": int(cohort.same_physical_run.sum()),
        "both_common25_pairs_in_paired_subset": int(cohort.both_common25.sum()),
        "near_service_tolerances_c": list(NEAR_SERVICE_TOLERANCES_C),
        "primary_near_service_tolerance_c": PRIMARY_NEAR_SERVICE_TOLERANCE_C,
        "near_service_condition_status_counts": near_conditions.groupby(
            ["tolerance_c", "status"], sort=True).size().rename("count").reset_index().to_dict("records"),
        "independently_reselected_original_budget_rows": len(selections),
        "pooled_front_condition_run_identities": len(pooled_front),
        "pooled_front_family_membership_counts": pooled_front.control_families.value_counts().sort_index().to_dict(),
        "definitions": {
            "common_requirements": "Upper bounds on mean and fixed-tail reference mismatch; realized scores need not match.",
            "same_physical": "Identical retained run_id within the same people/placement/permission condition.",
            "pareto": "Non-dominated in annual purchased site energy, mean mismatch, fixed-tail mismatch within the finite observed library.",
            "frontier_nodes": "Exploratory service limits equal to an actual candidate's two scores; no interpolation or synthetic trajectory.",
            "near_service": "All candidate pairings satisfying both absolute service-score differences at a declared tolerance, including same-run matches.",
            "near_service_tolerances": "Predeclared approximate-service matching windows: 0.05 C primary, 0.02 and 0.10 C sensitivity. These do not represent model precision, measured comfort or health thresholds.",
            "summary": "Medians of condition-level median differences avoid counting all candidate pairs as independent experiments.",
            "service": "Reconstructed reference-temperature mismatch, not measured personal satisfaction.",
        },
        "limitations": ["Finite candidate library; not a global control optimum.",
                        "Observed service-node analysis is exploratory, not independent validation.",
                        "Approximate two-score equality is not identical individual service.",
                        "Missing engineering trajectories remain unavailable; no replacement trajectory is invented.",
                        "Site-energy minimization does not imply the same choice minimizes carbon."],
        "output_sha256": {name: hashlib.sha256((output_root / name).read_bytes()).hexdigest() for name in frames},
    }
    (output_root / "SERVICE_FRONTIER_QA.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Retained energy_service/annual directory")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
