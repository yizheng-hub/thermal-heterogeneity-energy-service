"""Candidate-permission sensitivity using only already simulated physical runs.

Finer existing-zone permissions can execute coarser vectors unchanged. This
check isolates that set inclusion from the finite four-candidate control rule.
The minimum-site-energy ordering is mathematical, not a new energy finding.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from analyse_annual import BUDGETS, KEYS, PROFILE_KEYS, QUALIFIED, SERVICE_TOL, choose_energy, digest, pair_outputs


PARENTS = {"building": ["building"], "floor": ["building", "floor"],
           "native_zone": ["building", "floor", "native_zone"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-layouts", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    source = args.annual_root
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    candidates_path = source / "ANNUAL_ALL_CANDIDATES.csv"
    anchors_path = source / "ANNUAL_ANCHORS.csv"
    candidates = pd.read_csv(candidates_path)
    declared_layouts = {"random_mixing", "demand_ordered"}
    if args.expected_layouts == 3:
        declared_layouts.add("source_structured")
    if set(candidates.layout) != declared_layouts or len(candidates) != 432 * args.expected_layouts:
        raise ValueError("Annual candidates do not match the declared layout design")
    anchors = {tuple(row[k] for k in PROFILE_KEYS): row for row in pd.read_csv(anchors_path).to_dict("records")}
    selections = []
    inherited_candidates = 0
    for key, group in candidates.groupby(PROFILE_KEYS + ["control"], sort=True):
        qualified = group[group.engineering_status.isin(QUALIFIED)]
        anchor = anchors[key[:3]]
        for permission, parents in PARENTS.items():
            pool = qualified[qualified.permission.isin(parents)].sort_values(
                ["run_id", "permission", "variant"]
            ).drop_duplicates("run_id")
            ids = set(pool.run_id)
            for parent in parents:
                if not set(qualified[qualified.permission.eq(parent)].run_id).issubset(ids):
                    raise ValueError("A qualified coarser command disappeared from the inherited support")
            inherited_candidates += len(pool)
            for fraction in BUDGETS:
                mean_budget = fraction * anchor["anchor_mean_c"]
                tail_budget = fraction * anchor["anchor_fixed_tail_c"]
                feasible = pool[(pool.mean_reference_mismatch_c <= mean_budget+SERVICE_TOL)
                                & (pool.fixed_tail_reference_mismatch_c <= tail_budget+SERVICE_TOL)]
                chosen = choose_energy(feasible)
                row = dict(zip(PROFILE_KEYS + ["control"], key))
                if chosen:
                    row.update(chosen)
                row.update(permission=permission, candidate_support="inherited_permissions",
                           origin_permission=chosen["permission"] if chosen else "",
                           comparison_mode="equal_service", service_fraction=fraction,
                           mean_budget_c=mean_budget, fixed_tail_budget_c=tail_budget,
                           engineering_qualified_candidates=len(pool),
                           service_qualified_candidates=len(feasible),
                           selection_status="selected" if chosen else "service_unavailable")
                selections.append(row)
    table = pd.DataFrame(selections)
    if len(table) != 432 * args.expected_layouts:
        raise ValueError("Nested-permission diagnostic lost logical design cells")
    table.to_csv(out / "NESTED_PERMISSION_SELECTIONS.csv", index=False)
    pairs = pair_outputs(table, "permission", ["building", "floor", "native_zone"],
                        out / "NESTED_PERMISSION_PAIRS.csv", "space")
    reference_index = {tuple(row[k] for k in PROFILE_KEYS+["control", "service_fraction"]): row
                       for row in table[table.permission.eq("building")].to_dict("records")}
    increases = []
    for row in pairs:
        if row["reference_status"] == "selected" and row["alternative_status"] != "selected":
            raise ValueError("A feasible coarse plan has no finer-permission inherited choice")
        if row["status"] == "matched":
            ref = reference_index[tuple(row[k] for k in PROFILE_KEYS+["control", "service_fraction"])]
            tolerance = max(1e-6, 1e-9*abs(ref["annual_site_kwh"]))
            if row["delta_annual_site_kwh"] > tolerance:
                increases.append(row["comparison_id"])
    if increases:
        raise ValueError(f"Inherited minimum-energy set inclusion failed: {increases}")
    report = {
        "status": "PASS", "physical_runs_added": 0,
        "logical_selection_rows": len(table), "deduplicated_candidate_memberships": inherited_candidates,
        "paired_spatial_rows": len(pairs), "site_energy_set_inclusion_violations": len(increases),
        "not_a_general_ordering_of": ["operational carbon", "energy bill", "DR response", "actual controller output"],
        "interpretation": "Existing coarser commands are executable with finer permissions; minimum site energy cannot increase within the inherited finite set, using the declared energy-tie tolerance",
        "input_sha256": {"candidates": digest(candidates_path), "anchors": digest(anchors_path)},
        "output_sha256": {path.name: digest(path) for path in out.glob("*.csv")},
    }
    (out / "NESTED_PERMISSION_QA.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
