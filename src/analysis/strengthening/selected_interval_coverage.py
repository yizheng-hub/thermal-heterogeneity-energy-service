"""Export annual time outside individual reference ranges for the Fig. 2 selections."""

import argparse
from pathlib import Path

import pandas as pd


FILENAME = "ANNUAL_SELECTED_INTERVAL_COVERAGE.csv"
SETTINGS = ["climate", "composition"]
PLAN = ["physical_plan_id", "run_id", "control", "permission"]


def calculate_interval_coverage(selections, library):
    reference = selections.loc[selections.factor.eq(1)].copy()
    if reference.duplicated(SETTINGS).any() or not reference.feasibility.eq("both_feasible").all():
        raise ValueError("Expected one pair of feasible reference selections per setting")
    result = reference.set_index(SETTINGS)[["choices_differ"]].copy()
    for side in ("zone_average", "individual"):
        names = {f"{side}_{field}": field for field in PLAN}
        selected = reference[[*SETTINGS, *names]].rename(columns=names)
        rows = selected.merge(library, how="left", on=SETTINGS + PLAN)
        coverage = rows.modeled_interval_person_hour_coverage
        if len(rows) != len(selected) or not coverage.between(0, 1).all():
            raise ValueError(f"Expected one scored operation for each {side} selection")
        rows["outside_person_hour_fraction"] = 1 - coverage
        rows["outside_person_hours"] = rows.occupied_native_person_hours * rows.outside_person_hour_fraction
        metrics = [*PLAN, "assigned_profile_count", "occupied_native_person_hours", "outside_person_hours",
                   "outside_person_hour_fraction", "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"]
        result = result.join(rows.set_index(SETTINGS)[metrics].add_prefix(f"{side}_"), validate="one_to_one")
    result["individual_minus_zone_average_outside_percentage_points"] = 100 * (
        result.individual_outside_person_hour_fraction - result.zone_average_outside_person_hour_fraction
    )
    return result.sort_index()


def export_interval_coverage(selections, library, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    calculate_interval_coverage(selections, library).to_csv(output, float_format="%.9f", lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[3] / "results",
                        help="Results directory extracted from Numerical_Results.zip")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tables = args.results_root / "figure_data/Tables"
    selections = pd.read_csv(tables / "PAIRED_SERVICE_THRESHOLD_SELECTIONS.csv")
    library = pd.read_csv(args.results_root / "analysis_results/budget_frontiers/ALL_OPERATION_LIBRARY.csv")
    export_interval_coverage(selections, library, args.output or tables / FILENAME)
