"""Compare new capacity-preserving within-floor arrangements with their fixed baseline."""

from common import *
import numpy as np
import pandas as pd


def main():
    new = pd.read_csv(ROOT / "results/PLACEMENT_METRICS.csv")
    old = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    old = old[old.offset_c.eq(0) & old.control.isin(["personal_center", "personal_range"])]
    keys = ["climate", "composition", "control"]
    paired = new.merge(old, on=keys, suffixes=("_arranged", "_mixed"), validate="many_to_one")
    paired["maximum_relocation_fraction"] = (
        paired.layout_arranged.str.extract(r"_(\d+)pct$").iloc[:, 0].astype(float) / 100
    )
    paired["actual_relocation_fraction"] = paired.moved_profiles / 267
    paired["population_gain_c"] = (
        paired.mean_reference_mismatch_c_mixed - paired.mean_reference_mismatch_c_arranged
    )
    paired["fixed_tail_gain_c"] = (
        paired.fixed_tail_reference_mismatch_c_mixed
        - paired.fixed_tail_reference_mismatch_c_arranged
    )
    paired["energy_change_pct"] = 100 * (
        paired.facility_purchased_kwh_arranged / paired.facility_purchased_kwh_mixed - 1
    )
    rows = []
    for (control, budget), g in paired.groupby(["control", "maximum_relocation_fraction"]):
        rows.append(
            {
                "control": control,
                "maximum_relocation_fraction": budget,
                "settings": len(g),
                "median_actual_relocation_fraction": g.actual_relocation_fraction.median(),
                "population_gain_median_c": g.population_gain_c.median(),
                "population_gain_min_c": g.population_gain_c.min(),
                "population_gain_max_c": g.population_gain_c.max(),
                "fixed_tail_gain_median_c": g.fixed_tail_gain_c.median(),
                "population_improved_settings": int(g.population_gain_c.gt(1e-9).sum()),
                "fixed_tail_improved_settings": int(g.fixed_tail_gain_c.gt(1e-9).sum()),
                "energy_change_median_pct": g.energy_change_pct.median(),
                "energy_increased_settings": int(g.energy_change_pct.gt(1e-9).sum()),
            }
        )
    out = ROOT / "results/constrained_placement"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "PLACEMENT_BUDGET_SUMMARY.csv", index=False)
    report = {
        "status": "completed",
        "paired_physical_comparisons": len(paired),
        "constraints": "At most declared relocation fraction; identical zone capacities; no movement across floors; fixed original demand estimates.",
        "comparisons": "Same demand-guided control and same physical building; actual relocated fraction is reported separately from permitted budget.",
        "interpretation": "A simulated organization constraint; no real team assignment data or measured relocation cost is assumed.",
    }
    write_json(out / "AUDIT.json", report)
    record_status("constrained_placement_effects", "completed", report)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
