"""Compare placement gains across systems in exactly matched qualified settings."""
from common import *
import pandas as pd

KEYS = ["climate", "composition_id", "control", "layout"]
ENDPOINTS = ["population_gain_c", "fixed_tail_gain_c", "energy_change_pct"]
ENERGY_COMPONENTS = ["heating_electric", "cooling_electric", "fans_electric", "facility_electric", "facility_gas_hhv"]


def matched_effects(vav, alternative):
    base = vav.rename(columns={"layout_arranged": "layout"})
    if base.duplicated(KEYS).any() or alternative.duplicated(KEYS).any():
        raise ValueError("Placement system comparisons contain duplicate settings")
    endpoints = list(ENDPOINTS)
    for component in ENERGY_COMPONENTS:
        arranged, mixed = component + "_kwh_arranged", component + "_kwh_mixed"
        if all(arranged in frame and mixed in frame for frame in [base, alternative]):
            endpoint = component + "_change_mwh"
            for frame in [base, alternative]:
                frame[endpoint] = (frame[arranged] - frame[mixed]) / 1000
            endpoints.append(endpoint)
    valid = alternative.engineering_pass_arranged.fillna(False) & alternative.engineering_pass_mixed.fillna(False)
    qualified = alternative.loc[valid, KEYS + endpoints]
    comparison = base[KEYS + endpoints + ["actual_relocation_fraction"]].merge(
        qualified, on=KEYS, how="inner", suffixes=("_vav", "_alternative"), validate="one_to_one")
    if len(comparison) != len(qualified):
        raise ValueError("Alternative-system placement has no matching original-system setting")
    rows = []
    for (control, layout), planned in base.groupby(["control", "layout"]):
        pairs = comparison[comparison.control.eq(control) & comparison.layout.eq(layout)]
        row = {"control": control, "layout": layout, "planned_settings": len(planned),
               "matched_qualified_settings": len(pairs), "unavailable_settings": len(planned) - len(pairs),
               "comparison_scope": "Within-system placement contrasts, restricted to identical qualified settings; transferred commands"}
        for endpoint in endpoints:
            row[endpoint + "_vav_median"] = pairs[endpoint + "_vav"].median()
            row[endpoint + "_alternative_median"] = pairs[endpoint + "_alternative"].median()
            row[endpoint + "_paired_system_difference_median"] = (
                pairs[endpoint + "_alternative"] - pairs[endpoint + "_vav"]).median()
        row["actual_relocation_fraction_median"] = pairs.actual_relocation_fraction.median()
        row["vav_energy_increase_cases"] = int(pairs.energy_change_pct_vav.gt(0).sum())
        row["alternative_energy_decrease_cases"] = int(pairs.energy_change_pct_alternative.lt(0).sum())
        row["vav_population_gain_cases"] = int(pairs.population_gain_c_vav.gt(0).sum())
        row["alternative_population_gain_cases"] = int(pairs.population_gain_c_alternative.gt(0).sum())
        rows.append(row)
    return comparison, pd.DataFrame(rows)


def main():
    root = ROOT / "results/placement_hvac_interaction"
    vav = pd.read_csv(ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv")
    status = []
    for system in ["independent_pthp", "vav_coordinated_outdoor_reset"]:
        source = root / (system + "_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv")
        if not source.is_file():
            status.append({"system": system, "status": "not_completed"})
            continue
        comparison, summary = matched_effects(vav, pd.read_csv(source))
        comparison.to_csv(root / (system + "_MATCHED_CROSS_SYSTEM_EFFECTS.csv"), index=False)
        summary.to_csv(root / (system + "_MATCHED_CROSS_SYSTEM_SUMMARY.csv"), index=False)
        grouping = ["control", "layout", "climate"]
        planned = vav.rename(columns={"layout_arranged": "layout"}).groupby(grouping).size().rename("planned_settings")
        columns = [endpoint + suffix for endpoint in ENDPOINTS for suffix in ["_vav", "_alternative"]]
        local = comparison.groupby(grouping)[columns].median().reindex(planned.index)
        local["planned_settings"] = planned
        local["matched_qualified_settings"] = comparison.groupby(grouping).size().reindex(planned.index, fill_value=0)
        local["unavailable_settings"] = local.planned_settings - local.matched_qualified_settings
        local.to_csv(root / (system + "_MATCHED_CROSS_SYSTEM_BY_CLIMATE.csv"))
        status.append({"system": system, "status": "completed_with_unavailable_operations" if summary.unavailable_settings.sum() else "completed",
                       "planned_comparisons": int(summary.planned_settings.sum()), "matched_comparisons": len(comparison)})
    write_json(root / "MATCHED_CROSS_SYSTEM_AUDIT.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
