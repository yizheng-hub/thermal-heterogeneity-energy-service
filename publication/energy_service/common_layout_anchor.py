"""Re-select fixed physical candidates under one absolute contract across layouts.

The random-mixing common-25 reference supplies the mean budget and fixed 27
profile identities for every layout. This post-result SI sensitivity introduces
no physical runs and does not alter the primary analysis or DR baselines.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_annual import BUDGETS, KEYS, QUALIFIED, SERVICE_TOL, choose_energy, digest


VIEWS = ("native_occupancy", "profile_equal")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    paths = {name: args.annual_root / name for name in (
        "ANNUAL_ALL_CANDIDATES.csv", "ANNUAL_ANCHORS.csv", "FIXED_TAIL_MEMBERSHIPS.csv",
        "PROFILE_ANNUAL_SERVICE.csv.gz", "ANNUAL_SERVICE_SELECTIONS.csv", "DR_BASELINES.csv")}
    initial_hashes = {name: digest(path) for name, path in paths.items()}
    candidates = pd.read_csv(paths["ANNUAL_ALL_CANDIDATES.csv"])
    anchors = pd.read_csv(paths["ANNUAL_ANCHORS.csv"])
    tails = pd.read_csv(paths["FIXED_TAIL_MEMBERSHIPS.csv"])
    profiles = pd.read_csv(paths["PROFILE_ANNUAL_SERVICE.csv.gz"])
    original_random = anchors[anchors.layout.eq("random_mixing")]
    if len(original_random) != 12:
        raise ValueError("Expected 12 random-layout composition/context anchors")
    profile_groups = {}
    for key, group in profiles.groupby(KEYS[:3] + ["run_id"], sort=True):
        if len(group) != group.profile_id.nunique() or len(group) != 267:
            raise ValueError("Each physical/profile group must have exactly 267 profiles")
        profile_groups[key] = group
    common, common_tails, identities = {}, {}, {}
    for reference in original_random.itertuples(index=False):
        key = (reference.context, reference.composition_id)
        group = profile_groups[key + ("random_mixing", reference.anchor_run_id)]
        relevant = tails[(tails.context == key[0]) & (tails.composition_id == key[1])
                         & tails.layout.eq("random_mixing")]
        fixed = set(relevant.profile_id)
        if len(fixed) != 27:
            raise ValueError("The random-reference tail must contain 27 fixed identities")
        tail_value = float(group[group.profile_id.isin(fixed)].reference_mismatch_c.mean())
        native_mean = float(np.average(group.reference_mismatch_c, weights=group.native_person_hour_weight))
        if abs(native_mean - reference.anchor_mean_c) > 1e-8 or abs(tail_value - reference.anchor_fixed_tail_c) > 1e-8:
            raise ValueError("Per-profile anchor values do not reproduce the primary random-layout anchor")
        common_tails[key] = fixed
        identities[key] = set(group.profile_id)
        common[key] = {
            "context": key[0], "composition_id": key[1], "anchor_layout": "random_mixing",
            "anchor_run_id": reference.anchor_run_id, "fixed_tail_profile_ids": "|".join(sorted(fixed)),
            "native_mean_anchor_c": native_mean,
            "profile_equal_mean_anchor_c": float(group.reference_mismatch_c.mean()),
            "fixed_tail_anchor_c": tail_value,
        }
    score_cache, rows = {}, []
    for design in candidates.to_dict("records"):
        key = tuple(design[name] for name in KEYS[:3]) + (design["run_id"],)
        cohort = key[:2]
        for view in VIEWS:
            row = {name: design[name] for name in KEYS + ["variant", "run_id", "engineering_status"]}
            row.update(weight_view=view, anchor_layout="random_mixing")
            for name in ("annual_site_kwh", "annual_electric_kwh", "annual_gas_hhv_kwh", "annual_operational_co2_kg"):
                row[name] = design.get(name, np.nan)
            if design["engineering_status"] in QUALIFIED:
                if (key, view) not in score_cache:
                    group = profile_groups[key]
                    if set(group.profile_id) != identities[cohort]:
                        raise ValueError("A layout changed the people rather than their placement")
                    weights = group.native_person_hour_weight if view == "native_occupancy" else np.ones(len(group))
                    score_cache[key, view] = {
                        "mean_service_c": float(np.average(group.reference_mismatch_c, weights=weights)),
                        "fixed_tail_service_c": float(group[group.profile_id.isin(common_tails[cohort])].reference_mismatch_c.mean()),
                        "modeled_interval_coverage": float(np.average(group.modeled_interval_coverage, weights=weights)),
                        "modeled_interval_mean_excess_c": float(np.average(group.modeled_interval_mean_excess_c, weights=weights)),
                    }
                row.update(score_cache[key, view])
            rows.append(row)
    rescored = pd.DataFrame(rows)
    selections = []
    for key, group in rescored.groupby(KEYS + ["weight_view"], sort=True):
        anchor = common[key[:2]]
        anchor_name = "native_mean_anchor_c" if key[-1] == "native_occupancy" else "profile_equal_mean_anchor_c"
        admitted = group[group.engineering_status.isin(QUALIFIED)]
        for fraction in BUDGETS:
            mean_budget = fraction * anchor[anchor_name]
            tail_budget = fraction * anchor["fixed_tail_anchor_c"]
            feasible = admitted[(admitted.mean_service_c <= mean_budget + SERVICE_TOL)
                                & (admitted.fixed_tail_service_c <= tail_budget + SERVICE_TOL)]
            chosen = choose_energy(feasible)
            item = dict(zip(KEYS + ["weight_view"], key))
            if chosen:
                item.update(chosen)
            item.update(service_fraction=fraction, mean_budget_c=mean_budget,
                        fixed_tail_budget_c=tail_budget, anchor_layout="random_mixing",
                        service_qualified_candidates=len(feasible), engineering_qualified_candidates=len(admitted),
                        selection_status="selected" if chosen else "service_unavailable")
            selections.append(item)
    selected = pd.DataFrame(selections)
    original = pd.read_csv(paths["ANNUAL_SERVICE_SELECTIONS.csv"])
    original = original[original.layout.eq("random_mixing")]
    reproduced = selected[selected.layout.eq("random_mixing") & selected.weight_view.eq("native_occupancy")]
    matched_primary = reproduced.merge(original, on=KEYS + ["service_fraction"], suffixes=("_new", "_old"), validate="one_to_one")
    if len(matched_primary) != 432 or not matched_primary.selection_status_new.eq(matched_primary.selection_status_old).all():
        raise ValueError("Common-layout diagnostic changed the original random-layout selection states")
    chosen_primary = matched_primary[matched_primary.selection_status_new.eq("selected")]
    if not chosen_primary.run_id_new.eq(chosen_primary.run_id_old).all():
        raise ValueError("Common-layout diagnostic changed a reproduced random-layout chosen plan")
    pair_keys = [name for name in KEYS if name != "layout"] + ["weight_view", "service_fraction"]
    reference_index = {tuple(row[name] for name in pair_keys): row
                       for row in selected[selected.layout.eq("random_mixing")].to_dict("records")}
    pairs = []
    for row in selected[~selected.layout.eq("random_mixing")].to_dict("records"):
        reference = reference_index[tuple(row[name] for name in pair_keys)]
        if row["mean_budget_c"] != reference["mean_budget_c"] or row["fixed_tail_budget_c"] != reference["fixed_tail_budget_c"]:
            raise ValueError("Common absolute layout budgets changed across layouts")
        matched = row["selection_status"] == reference["selection_status"] == "selected"
        item = {name: row[name] for name in KEYS + ["weight_view", "service_fraction"]}
        item.update(reference_layout="random_mixing", status="matched" if matched else "unmatched",
                    reference_status=reference["selection_status"], alternative_status=row["selection_status"],
                    reference_run_id=reference.get("run_id", ""), alternative_run_id=row.get("run_id", ""),
                    mean_budget_c=row["mean_budget_c"], fixed_tail_budget_c=row["fixed_tail_budget_c"])
        for name in ("annual_site_kwh", "annual_operational_co2_kg", "mean_service_c", "fixed_tail_service_c",
                     "modeled_interval_coverage", "modeled_interval_mean_excess_c"):
            item["delta_" + name] = row[name] - reference[name] if matched else np.nan
        item["delta_site_pct"] = 100 * item["delta_annual_site_kwh"] / reference["annual_site_kwh"] if matched else np.nan
        pairs.append(item)
    pd.DataFrame(list(common.values())).to_csv(out / "COMMON_LAYOUT_ANCHORS.csv", index=False)
    rescored.to_csv(out / "COMMON_LAYOUT_CANDIDATES.csv", index=False)
    selected.to_csv(out / "COMMON_LAYOUT_SELECTIONS.csv", index=False)
    paired = pd.DataFrame(pairs)
    paired.to_csv(out / "COMMON_LAYOUT_PAIRS.csv", index=False)
    if len(rescored) != 2592 or len(selected) != 2592 or len(paired) != 1728:
        raise ValueError("The full two-weight-view common-layout diagnostic grid is incomplete")
    if initial_hashes != {name: digest(path) for name, path in paths.items()}:
        raise ValueError("A primary product or DR baseline changed during the SI diagnostic")
    report = {
        "status": "PASS", "post_primary_results_SI_sensitivity": True, "new_physical_runs": 0,
        "common_random_layout_anchors": len(common), "fixed_tail_identities": 27,
        "weight_views": list(VIEWS), "candidate_rows": len(rescored), "selection_rows": len(selected),
        "layout_pair_rows": len(paired), "primary_analysis_and_DR_baselines_unchanged": True,
        "primary_random_layout_selection_rows_reproduced": len(matched_primary),
        "selected_counts": selected.groupby(["weight_view", "service_fraction", "layout", "selection_status"]).size().rename("count").reset_index().to_dict("records"),
        "input_sha256": initial_hashes,
        "output_sha256": {path.name: digest(path) for path in sorted(out.glob("*.csv"))},
        "interpretation": "One absolute mean/tail service contract and the same random-reference tail identities across all layouts; model-derived intervals remain a separate diagnostic.",
    }
    (out / "COMMON_LAYOUT_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report[name] for name in ("status", "common_random_layout_anchors", "candidate_rows", "selection_rows", "layout_pair_rows", "new_physical_runs")}))


if __name__ == "__main__":
    main()
