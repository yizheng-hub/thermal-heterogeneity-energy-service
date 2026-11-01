from common import *
from pipeline import command
import time
import pandas as pd
import numpy as np
from analyse_global_results import fixed_plan_information_diagnostic


def wait_file(path):
    while not path.exists():
        time.sleep(5)


def final_grid():
    quarter = pd.read_csv(ROOT / "results/candidate_refinement/REFINED_RUN_METRICS.csv")
    further_path = ROOT / "results/CANDIDATE_FURTHER_METRICS.csv"
    full = quarter
    report = {
        "quarter_degree_logical_candidates": len(quarter),
        "expected_quarter_degree_logical_candidates": 1458,
        "further_refinement_executed": further_path.exists(),
    }
    anchors = pd.read_csv(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv")
    if further_path.exists():
        further = pd.read_csv(further_path)
        full = pd.concat([quarter, further], ignore_index=True)
        a = fixed_plan_information_diagnostic(quarter, anchors, fractions=(1.0,))
        b = fixed_plan_information_diagnostic(full, anchors, fractions=(1.0,))
        keys = ["climate", "composition", "layout", "permission"]
        compare = b.merge(a, on=keys, suffixes=("_eighth", "_quarter"), validate="one_to_one")
        change = (
            compare.full_selected_facility_purchased_kwh_eighth
            / compare.full_selected_facility_purchased_kwh_quarter
            - 1
        )
        compare["relative_full_selected_energy_change"] = change
        compare.to_csv(
            ROOT / "results/candidate_refinement/EIGHTH_GRID_CONVERGENCE.csv", index=False
        )
        report.update(
            {
                "further_logical_candidates_scored": len(further),
                "maximum_selected_energy_change": float(abs(change).max()),
                "energy_tolerance_0p1pct_reached": bool(abs(change).max() <= 0.001),
                "stop_reason": "predeclared single additional level exhausted",
            }
        )
    else:
        earlier = json.loads(
            (ROOT / "results/candidate_refinement/REFINEMENT_AUDIT.json").read_text()
        )
        report.update(
            {
                "energy_tolerance_0p1pct_reached": not earlier[
                    "further_uniform_refinement_required"
                ],
                "stop_reason": "quality or completeness blocks required further refinement"
                if earlier["further_uniform_refinement_required"]
                else "quarter-degree grid met declared energy tolerance",
            }
        )
    full.to_csv(ROOT / "results/candidate_refinement/STOPPED_GRID_METRICS.csv", index=False)
    d = fixed_plan_information_diagnostic(full, anchors, fractions=(1.0,))
    d.to_csv(
        ROOT / "results/candidate_refinement/STOPPED_GRID_INFORMATION_DIAGNOSTIC.csv", index=False
    )
    complete = len(quarter) == 1458 and (
        (report["further_refinement_executed"] and len(further) == 1296)
        or (not report["further_refinement_executed"] and report["energy_tolerance_0p1pct_reached"])
    )
    report["all_required_candidates_scored"] = complete
    report["expected_further_logical_candidates"] = (
        1296 if report["further_refinement_executed"] else 0
    )
    report.update(
        {
            "status": "completed" if complete else "partial",
            "settings": len(d),
            "logical_candidates": len(full),
            "qualifying_set_changes": int(d.eligibility_disagreement.sum()),
            "choice_changes": int(d.selection_disagreement.sum()),
            "interpretation": "Finite evaluated library; convergence reported separately from global optimality.",
        }
    )
    write_json(ROOT / "audit/GRID_STOP.json", report)
    record_status("uniform_candidate_refinement", report["status"], report)
