"""Describe every frozen service/move cell without treating settings as samples."""

from common import *
import pandas as pd


KEYS = ["arm", "system", "budget_type", "reference_fraction", "population_budget_c",
        "fixed_tail_budget_c", "maximum_relocation_fraction"]


def summarize(frame):
    # Relative budgets differ numerically by cohort; aggregate only their ratio.
    frame = frame.copy()
    relative = frame.budget_type.eq("reference_relative")
    frame.loc[relative, ["population_budget_c", "fixed_tail_budget_c"]] = float("nan")
    rows = []
    for identity, group in frame.groupby(KEYS, dropna=False, sort=True):
        if len(group) != 54 or group.duplicated(["climate", "composition"]).any():
            raise ValueError("Each descriptive service/move cell requires all54 distinct planned settings")
        paired = group[group.zero_movement_status.eq("feasible") & group.status.eq("feasible")]
        feasible = group[group.status.eq("feasible")]
        row = dict(zip(KEYS, identity)) | {"planned_settings": 54,
            "feasible_settings": len(feasible), "newly_feasible_settings": int(group.newly_feasible_with_movement.sum()),
            "paired_feasible_settings": len(paired),
            "selected_operation_moves_profiles": int(feasible.selected_moved_profiles.gt(0).sum()),
            "selected_actual_relocation_fraction_median": feasible.selected_moved_profiles.median() / 267,
            "paired_energy_change_pct_median": paired.energy_change_from_zero_pct.median(),
            "paired_energy_decrease_settings": int(paired.energy_change_from_zero_kwh.lt(-1e-6).sum())}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    out = ROOT / "results/placement_service_frontiers"
    audit = json.loads((out / "AUDIT.json").read_text())
    if audit["status"] != "completed":
        raise ValueError("Complete the placement frontiers before summarizing")
    paths = [out / name for name in ["SHARED_MENU_FRONTIERS.csv", "VAV_STRONG_FRONTIERS.csv"]]
    for path in paths:
        if digest(path) != audit["output_sha256"][path.name]:
            raise ValueError("Placement frontier result checksum differs")
    result = summarize(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True))
    if len(result) != 496:
        raise ValueError("Incomplete frozen-budget descriptive summary")
    target = out / "DESCRIPTIVE_BUDGET_SUMMARY.csv"
    result.to_csv(target, index=False)
    write_json(out / "DESCRIPTIVE_SUMMARY_AUDIT.json", {"status": "completed", "summary_cells": len(result),
        "source_sha256": {p.name: digest(p) for p in paths}, "executed_source_sha256": digest(__file__),
        "output_sha256": digest(target), "inference": "Descriptive finite-design counts and paired contrasts; no population confidence intervals"})
    print("Completed", len(result), "descriptive budget cells", flush=True)


if __name__ == "__main__":
    main()
