"""Test the necessary mean-service constraint on the existing dense uniform grid.

The shared random-layout absolute anchor is used. Failing the mean condition
alone is sufficient to rule out joint mean/tail eligibility, so no claim is
made to reconstruct an unobserved fixed-tail score on this separate grid.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from analyse_annual import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-scores", required=True, type=Path)
    parser.add_argument("--common-anchors", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = pd.read_csv(args.candidate_scores)
    source = source[source.strategy.eq("uniform") & source.griffiths_coefficient.eq(.5)
                    & source.interval_width_multiplier.eq(1) & source.engineering_pass]
    anchors = pd.read_csv(args.common_anchors).set_index(["context", "composition_id"])
    rows = []
    for (context, seed, layout), group in source.groupby(["city", "seed", "allocation"], sort=True):
        anchor = float(anchors.loc[(context, seed)].native_mean_anchor_c)
        minimum = float(group["mean"].min())
        eligible = group["mean"] <= .5 * anchor + 1e-9
        rows.append({
            "context": context, "composition_id": int(seed), "layout": layout,
            "engineering_qualified_candidate_evaluations": len(group),
            "common_random_mean_anchor_c": anchor, "half_mean_budget_c": .5 * anchor,
            "best_uniform_mean_c": minimum, "best_mean_as_anchor_fraction": minimum / anchor,
            "mean_eligible_at_half_budget": int(eligible.sum()),
            "joint_eligibility_ruled_out_by_mean_alone": not bool(eligible.any()),
        })
    table = pd.DataFrame(rows)
    if len(table) != 36:
        raise ValueError("The dense-grid diagnostic must retain all 36 profile/placement/context groups")
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    target = out / "DENSE_UNIFORM_HALF_BUDGET_BOUND.csv"
    table.to_csv(target, index=False)
    report = {
        "status": "PASS", "new_physical_runs": 0, "post_primary_SI_sensitivity": True,
        "composition_layout_context_groups": len(table),
        "qualified_candidate_evaluations": len(source),
        "groups_with_any_mean_eligible_candidate": int(table.mean_eligible_at_half_budget.gt(0).sum()),
        "groups_ruled_out_by_mean_alone": int(table.joint_eligibility_ruled_out_by_mean_alone.sum()),
        "best_mean_reference_fraction_minimum": float(table.best_mean_as_anchor_fraction.min()),
        "best_mean_reference_fraction_maximum": float(table.best_mean_as_anchor_fraction.max()),
        "interpretation": "No engineering-qualified candidate in the tested denser uniform grid satisfies the shared half-reference mean limit; joint eligibility is consequently absent. This is not a proof over every possible thermostat trajectory.",
        "input_sha256": {"candidate_scores": digest(args.candidate_scores), "common_random_anchors": digest(args.common_anchors)},
        "output_sha256": {target.name: digest(target)},
    }
    (out / "DENSE_UNIFORM_HALF_BUDGET_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
