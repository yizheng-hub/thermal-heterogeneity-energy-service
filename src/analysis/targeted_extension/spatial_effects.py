"""Paired placement/permission contrasts on frozen, common-tail annual scores.

Equal-profile weighting fixes aggregate person weights; each destination keeps
its native occupancy schedule. Native-occupancy weighting is retained as a
separate sensitivity. The same
random-layout reference supplies all service budgets and tail identities.
No contrast across layouts uses the layout-specific primary service anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["context", "composition_id", "layout", "permission", "control", "weight_view"]
BLOCK = ["context", "composition_id", "control", "weight_view", "comparison_mode", "service_fraction"]
METRICS = ["annual_site_kwh", "annual_electric_kwh", "annual_gas_hhv_kwh",
           "annual_operational_co2_kg", "mean_service_c", "fixed_tail_service_c",
           "modeled_interval_coverage", "modeled_interval_mean_excess_c"]
QUALIFIED = {"cached_qualified", "new_qualified"}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remove_aliases(frame, aliases):
    """Drop known permission aliases, never deduplicate by numerical outcomes."""
    keys = ["context", "permission"]
    if not len(aliases):
        return frame.copy()
    marked = frame.merge(aliases[keys].drop_duplicates().assign(_alias=True), on=keys,
                         how="left", validate="many_to_one")
    return marked.loc[marked._alias.isna(), frame.columns].copy()


def assert_same_contract(cells):
    if cells[0]["comparison_mode"] == "direct":
        return
    for name in ["mean_budget_c", "fixed_tail_budget_c"]:
        values = [float(row[name]) for row in cells]
        if not np.isfinite(values).all() or max(values) - min(values) > 1e-9:
            raise ValueError("A spatial contrast changed the absolute service contract")


def pair_contrasts(frame, axes=("layout", "permission")):
    """Keep every designed pair, including failed/unavailable selections."""
    rows = []
    for axis in axes:
        reference = {"layout": "random_mixing", "permission": "building"}[axis]
        match_keys = BLOCK + ["permission" if axis == "layout" else "layout"]
        for _, group in frame.groupby(match_keys, sort=True, dropna=False):
            baseline = group[group[axis].eq(reference)]
            if len(baseline) != 1:
                raise ValueError("A spatial design cell lost or duplicated its reference")
            ref = baseline.iloc[0].to_dict()
            for alt in group[~group[axis].eq(reference)].to_dict("records"):
                assert_same_contract([ref, alt])
                matched = ref["selection_status"] == alt["selection_status"] == "selected"
                row = {name: alt[name] for name in match_keys}
                row.update(axis=axis, reference=reference, alternative=alt[axis],
                           reference_status=ref["selection_status"],
                           alternative_status=alt["selection_status"],
                           status="matched" if matched else "unmatched",
                           reference_run_id=ref.get("run_id"), alternative_run_id=alt.get("run_id"),
                           same_physical_run=bool(matched and ref["run_id"] == alt["run_id"]))
                for metric in METRICS:
                    row["delta_" + metric] = alt[metric] - ref[metric] if matched else np.nan
                row["delta_site_pct"] = 100 * row["delta_annual_site_kwh"] / ref["annual_site_kwh"] if matched else np.nan
                row["delta_coverage_pp"] = 100 * row["delta_modeled_interval_coverage"]
                rows.append(row)
    return pd.DataFrame(rows)


def interaction_contrasts(frame):
    """Four-corner differences; do not treat the correlated corners as repeats."""
    rows = []
    for _, group in frame.groupby(BLOCK, sort=True, dropna=False):
        cells = {(row["layout"], row["permission"]): row for row in group.to_dict("records")}
        for layout in sorted(set(group.layout) - {"random_mixing"}):
            for permission in sorted(set(group.permission) - {"building"}):
                corners = [("random_mixing", "building"), (layout, "building"),
                           ("random_mixing", permission), (layout, permission)]
                selected = [cells[key] for key in corners]
                assert_same_contract(selected)
                qualified = [int(cell["selection_status"] == "selected") for cell in selected]
                matched = all(qualified)
                row = {name: selected[0][name] for name in BLOCK}
                row.update(layout=layout, permission=permission,
                           status="matched" if matched else "unmatched",
                           selected_corners=sum(qualified),
                           qualification_interaction=qualified[3] - qualified[2] - qualified[1] + qualified[0])
                for number, cell in enumerate(selected):
                    row[f"corner_{number}_status"] = cell["selection_status"]
                    row[f"corner_{number}_run_id"] = cell.get("run_id")
                for metric in METRICS:
                    row["interaction_" + metric] = (selected[3][metric] - selected[2][metric]
                        - selected[1][metric] + selected[0][metric]) if matched else np.nan
                row["interaction_site_pct"] = (100 * row["interaction_annual_site_kwh"]
                    / selected[0]["annual_site_kwh"]) if matched else np.nan
                row["interaction_coverage_pp"] = 100 * row["interaction_modeled_interval_coverage"]
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame, groups, metrics):
    rows = []
    for key, group in frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row.update(designed_pairs=len(group), matched_pairs=int(group.status.eq("matched").sum()),
                   unmatched_pairs=int(group.status.ne("matched").sum()),
                   composition_context_blocks=group[["context", "composition_id"]].drop_duplicates().shape[0])
        if "reference_status" in group:
            selected_ref = group.reference_status.eq("selected")
            selected_alt = group.alternative_status.eq("selected")
            row.update(reference_only=int((selected_ref & ~selected_alt).sum()),
                       alternative_only=int((selected_alt & ~selected_ref).sum()),
                       neither_selected=int((~selected_ref & ~selected_alt).sum()),
                       same_physical_run=int(group.same_physical_run.sum()))
        for metric in metrics:
            values = group.loc[group.status.eq("matched"), metric].dropna()
            tolerance = 1e-6 if metric.endswith("kwh") else 1e-8
            row.update({metric + "_n": len(values),
                        metric + "_median": float(values.median()) if len(values) else np.nan,
                        metric + "_min": float(values.min()) if len(values) else np.nan,
                        metric + "_max": float(values.max()) if len(values) else np.nan,
                        metric + "_negative": int((values < -tolerance).sum()),
                        metric + "_zero": int((values.abs() <= tolerance).sum()),
                        metric + "_positive": int((values > tolerance).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def verify_people_and_tails(annual_root, common_root, expected_layouts=3):
    profiles = pd.read_csv(annual_root / "PROFILE_ANNUAL_SERVICE.csv.gz",
                           usecols=["context", "composition_id", "layout", "profile_id"])
    profiles = profiles.drop_duplicates()
    anchors = pd.read_csv(common_root / "COMMON_LAYOUT_ANCHORS.csv")
    for anchor in anchors.itertuples(index=False):
        block = profiles[(profiles.context == anchor.context) & (profiles.composition_id == anchor.composition_id)]
        identities = [set(group.profile_id) for _, group in block.groupby("layout")]
        if len(identities) != expected_layouts or len(identities[0]) != 267 or any(ids != identities[0] for ids in identities):
            raise ValueError("Placement comparisons do not preserve the same 267 identities")
        fixed = set(anchor.fixed_tail_profile_ids.split("|"))
        if len(fixed) != 27 or not fixed.issubset(identities[0]):
            raise ValueError("Common-tail identities are incomplete or absent from the cohort")
    return len(anchors)


def nested_audit(nested, aliases):
    """Describe inclusion as a construction check, not an estimated benefit."""
    independent = remove_aliases(nested, aliases)
    frame = pair_contrasts(independent, axes=("permission",))
    matched = frame[frame.status.eq("matched")]
    lost = frame.reference_status.eq("selected") & frame.alternative_status.ne("selected")
    tolerance = np.maximum(1e-6, matched.delta_annual_site_kwh.abs() * 1e-9)
    if lost.any() or (matched.delta_annual_site_kwh > tolerance).any():
        raise ValueError("Inherited candidate-set inclusion failed")
    return frame


def run(source, output, expected_layouts=3):
    annual, common, nested = source / "annual", source / "common_layout", source / "nested_permissions"
    paths = [annual / "ANNUAL_DIRECT.csv", annual / "PROFILE_ANNUAL_SERVICE.csv.gz",
             common / "COMMON_LAYOUT_CANDIDATES.csv", common / "COMMON_LAYOUT_SELECTIONS.csv",
             common / "COMMON_LAYOUT_ANCHORS.csv", nested / "NESTED_PERMISSION_SELECTIONS.csv"]
    before = {str(path.relative_to(source)): sha256(path) for path in paths}
    direct = pd.read_csv(paths[0])
    aliases = direct.loc[direct.permission_alias.notna(), ["context", "permission", "permission_alias"]].drop_duplicates()
    candidates = pd.read_csv(paths[2])
    selected = pd.read_csv(paths[3])
    blocks = verify_people_and_tails(annual, common, expected_layouts)
    direct_common = candidates[candidates.variant.eq(0)].copy()
    direct_common["comparison_mode"] = "direct"
    direct_common["service_fraction"] = -1.0
    direct_common["selection_status"] = np.where(direct_common.engineering_status.isin(QUALIFIED), "selected", "engineering_rejected")
    direct_common["mean_budget_c"] = np.nan
    direct_common["fixed_tail_budget_c"] = np.nan
    selected["comparison_mode"] = "equal_service"
    records = remove_aliases(pd.concat([direct_common, selected], ignore_index=True), aliases)
    if records.duplicated(KEYS + ["comparison_mode", "service_fraction"]).any():
        raise ValueError("Duplicate independent selection cell")
    pairs, interactions = pair_contrasts(records), interaction_contrasts(records)
    nested_records = pd.read_csv(paths[5]).rename(columns={
        "mean_reference_mismatch_c": "mean_service_c", "fixed_tail_reference_mismatch_c": "fixed_tail_service_c",
        "modeled_interval_person_hour_coverage": "modeled_interval_coverage"})
    # Nested supports retain layout-specific anchors/tails; only permissions
    # within a layout can be compared, never cross-layout placement effects.
    nested_records["weight_view"] = "native_occupancy"
    nested_pairs = nested_audit(nested_records, aliases)
    summary_groups = ["weight_view", "comparison_mode", "service_fraction", "control", "axis", "reference", "alternative", "permission", "layout"]
    pair_metrics = ["delta_site_pct", "delta_annual_site_kwh", "delta_annual_electric_kwh",
                    "delta_annual_gas_hhv_kwh", "delta_annual_operational_co2_kg",
                    "delta_coverage_pp", "delta_mean_service_c", "delta_fixed_tail_service_c"]
    pair_summary = summarize(pairs, summary_groups, pair_metrics)
    context_summary = summarize(pairs, ["context"] + summary_groups,
                                pair_metrics)
    interaction_summary = summarize(interactions, ["weight_view", "comparison_mode", "service_fraction", "control", "layout", "permission"],
                                     ["interaction_site_pct", "interaction_coverage_pp", "interaction_mean_service_c", "interaction_fixed_tail_service_c"])
    qualification = records.groupby(["weight_view", "comparison_mode", "service_fraction", "context", "control", "layout", "permission", "selection_status"], dropna=False).size().rename("count").reset_index()
    output.mkdir(parents=True, exist_ok=True)
    tables = {"SPATIAL_SELECTIONS": records, "SPATIAL_PAIRS": pairs, "SPATIAL_PAIR_SUMMARY": pair_summary,
              "SPATIAL_CONTEXT_SUMMARY": context_summary, "SPATIAL_INTERACTIONS": interactions,
              "SPATIAL_INTERACTION_SUMMARY": interaction_summary, "SPATIAL_QUALIFICATION_COUNTS": qualification,
              "NESTED_INCLUSION_AUDIT": nested_pairs, "EXCLUDED_PERMISSION_ALIASES": aliases}
    for name, table in tables.items():
        table.to_csv(output / (name + ".csv"), index=False)
    after = {str(path.relative_to(source)): sha256(path) for path in paths}
    if before != after:
        raise ValueError("A frozen input changed during analysis")
    report = {
        "status": "PASS", "physical_runs_added": 0, "composition_context_blocks": blocks,
        "source_cohort_compositions": int(records.composition_id.nunique()),
        "building_weather_contexts": int(records.context.nunique()),
        "assigned_profiles_per_composition": 267, "fixed_reference_tail_profiles": 27,
        "independent_spatial_conditions": int(records[["context", "composition_id", "layout", "permission"]].drop_duplicates().shape[0]),
        "rows": {name: len(table) for name, table in tables.items()},
        "primary_weight_view": "profile_equal", "sensitivity_weight_view": "native_occupancy",
        "input_sha256": before,
        "output_sha256": {path.name: sha256(path) for path in sorted(output.glob("*.csv"))},
        "interpretation": {
            "direct": "Variant-zero controller output; both trajectories must pass engineering screening; no common-service promise.",
            "equal_service": "Separate energy-minimizing candidates subject to identical absolute mean/tail limits; achieved services need not be identical.",
            "placement": "Same profile identities, frozen equipment/weather/gains; primary equal-profile view fixes aggregate person weights. Each destination retains its native occupancy schedule, so temporal exposure weights are not held identical when a person moves.",
            "permissions": "Commands applied to unchanged native thermal zones; no independent personal room or new HVAC equipment is simulated.",
            "interaction": "Four-corner within-composition difference; only complete qualified quadruplets have continuous effects; qualification interaction retains all cells.",
            "nested": "Energy non-increase is a mathematical finite-set inclusion property, not independent empirical evidence of savings.",
            "independence": "Three compositions are repeated across contexts/layouts/controls. Contrasts, budget levels and weights are not independent samples; no population p-values are estimated.",
        },
    }
    (output / "SPATIAL_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=root / "output/replay/energy_service")
    parser.add_argument("--output-root", type=Path, default=root / "output/targeted_extension/spatial")
    parser.add_argument("--expected-layouts", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    report = run(args.source_root.resolve(), args.output_root.resolve(), args.expected_layouts)
    print(json.dumps({key: report[key] for key in ["status", "physical_runs_added", "independent_spatial_conditions", "rows"]}))


if __name__ == "__main__":
    main()
