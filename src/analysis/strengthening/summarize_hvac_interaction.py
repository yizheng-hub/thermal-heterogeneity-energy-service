"""Report paired policy differences within each independently tested HVAC condition."""

from common import *
import pandas as pd
import numpy as np


def policy_pairs(frame, condition):
    good = frame[frame.engineering_pass.fillna(False)].copy()
    uniform = good[good.control.eq("uniform_average")]
    pairs = []
    for control in ["personal_center", "personal_range"]:
        p = good[good.control.eq(control)].merge(
            uniform,
            on=["climate", "composition_id"],
            suffixes=("_demand", "_average"),
            validate="one_to_one",
        )
        p["condition"] = condition
        p["compared_control"] = control
        p["population_gain_c"] = p.population_mismatch_c_average - p.population_mismatch_c_demand
        p["fixed_tail_gain_c"] = p.fixed_tail_mismatch_c_average - p.fixed_tail_mismatch_c_demand
        p["delta_energy_kwh"] = p.facility_purchased_kwh_demand - p.facility_purchased_kwh_average
        p["delta_energy_pct"] = 100 * (
            p.facility_purchased_kwh_demand / p.facility_purchased_kwh_average - 1
        )
        pairs.append(p)
    return pd.concat(pairs, ignore_index=True)


def main():
    original = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    original = original[original.offset_c.eq(0)].copy()
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    labels = (
        profiles[["composition_id", "cohort_label"]]
        .drop_duplicates()
        .set_index("cohort_label")
        .composition_id.to_dict()
    )
    original["composition_id"] = original.composition.map(labels)
    original = original.rename(
        columns={
            "mean_reference_mismatch_c": "population_mismatch_c",
            "fixed_tail_reference_mismatch_c": "fixed_tail_mismatch_c",
        }
    )
    frames = [policy_pairs(original, "original_vav")]
    status = [
        {"condition": "original_vav", "status": "completed", "operation_scores": len(original)}
    ]
    specifications = [
        ("heatpump_transferred", "hvac_confirmation", "independent_pthp"),
        ("vav_weather_reset", "vav_outdoor_reset", "vav_coordinated_outdoor_reset"),
        ("vav_zone_feedback_reset", "vav_feedback_reset_independent", "vav_coordinated_standard"),
        ("heatpump_own_response", "hvac_retuned", "independent_pthp"),
    ]
    for condition, folder, system in specifications:
        path = ROOT / "results" / folder / "SYSTEM_OPERATION_SCORES.csv"
        if not path.exists():
            status.append({"condition": condition, "status": "not_completed"})
            continue
        frame = pd.read_csv(path)
        frame = frame[frame.system.eq(system)]
        complete = len(frame) == 162 and frame.engineering_pass.fillna(False).all()
        status.append(
            {
                "condition": condition,
                "status": "completed" if complete else "partial",
                "operation_scores": len(frame),
            }
        )
        frames.append(policy_pairs(frame, condition))
    pairs = pd.concat(frames, ignore_index=True)
    summary = []
    for (condition, control), g in pairs.groupby(["condition", "compared_control"]):
        both = g.population_gain_c.gt(1e-9) & g.fixed_tail_gain_c.gt(1e-9)
        summary.append(
            {
                "condition": condition,
                "control": control,
                "paired_settings": len(g),
                "population_gain_median_c": g.population_gain_c.median(),
                "fixed_tail_gain_median_c": g.fixed_tail_gain_c.median(),
                "energy_change_median_pct": g.delta_energy_pct.median(),
                "energy_change_median_kwh": g.delta_energy_kwh.median(),
                "population_improved_settings": int(g.population_gain_c.gt(1e-9).sum()),
                "both_service_endpoints_improved_settings": int(both.sum()),
                "energy_increased_settings": int(g.delta_energy_pct.gt(1e-9).sum()),
                "both_service_improved_and_energy_increased": int(
                    (both & g.delta_energy_pct.gt(1e-9)).sum()
                ),
                "both_service_improved_and_energy_decreased": int(
                    (both & g.delta_energy_pct.lt(-1e-9)).sum()
                ),
            }
        )
    out = ROOT / "results/hvac_interaction"
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "WITHIN_SYSTEM_POLICY_DIFFERENCES.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "HVAC_POLICY_SUMMARY.csv", index=False)
    original_pairs = pairs[pairs.condition.eq("original_vav")]
    interactions = pairs[~pairs.condition.eq("original_vav")].merge(
        original_pairs,
        on=["climate", "composition_id", "compared_control"],
        suffixes=("_alternative", "_vav"),
        validate="many_to_one",
    )
    interactions["energy_effect_difference_percentage_points"] = (
        interactions.delta_energy_pct_alternative - interactions.delta_energy_pct_vav
    )
    interactions["energy_direction_changed"] = np.sign(
        interactions.delta_energy_pct_alternative
    ) != np.sign(interactions.delta_energy_pct_vav)
    interactions.to_csv(out / "PAIRED_HVAC_INTERACTIONS.csv", index=False)
    write_json(
        out / "AUDIT.json",
        {
            "status": "completed" if all(s["status"] == "completed" for s in status) else "partial",
            "conditions": status,
            "interpretation": "Strategy differences are measured within each system. Transferred commands and own-response retuning are separate conditions; equal commands do not imply equal delivered service.",
        },
    )
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
