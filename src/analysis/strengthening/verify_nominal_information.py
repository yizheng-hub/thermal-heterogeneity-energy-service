"""Reproduce pooled nominal qualifications with the explicit hourly scorer."""

from common import *
import numpy as np
import pandas as pd
from analyse_global_results import (
    normalise_profiles, load_hourly, zone_mean_profiles, score_trajectory,
    choose_minimum_energy,
)


def main():
    out = ROOT / "results/nominal_information"
    audit_path = out / "AUDIT.json"
    table = out / "NOMINAL_QUALIFICATION_COUNTS.csv"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        inputs = json.loads((out / "INPUT_SHA256.json").read_text())
        if audit["status"] != "completed" or audit["source_sha256"] != digest(__file__):
            raise ValueError("Nominal qualification cache has a different source or incomplete execution")
        if any(digest(CODE / name) != expected for name, expected in inputs.items()):
            raise ValueError("Nominal qualification cache input changed")
        if digest(table) != audit["output_sha256"]:
            raise ValueError("Nominal qualification cache output changed")
        print("Nominal qualifications: verified completed cache", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the incomplete nominal qualification attempt before retrying")
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "library": ROOT / "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv",
        "draws": ROOT / "results/paired_uncertainty_pooled_identity/PAIRED_SELECTION_DRAWS.csv.gz",
        "profiles": MATRIX / "GLOBAL_PROFILES.csv",
        "tails": ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
        "references": MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv",
    }
    inputs = {p.relative_to(CODE).as_posix(): digest(p) for p in paths.values()}
    library = pd.read_csv(paths["library"])
    profiles = normalise_profiles(pd.read_csv(paths["profiles"]))
    members = pd.read_csv(paths["tails"])
    references = pd.read_csv(paths["references"])
    references = references[references.target_c.eq(25)].set_index("climate")
    draws = pd.read_csv(paths["draws"])
    draws = draws[draws.scenario.eq("sourceSD0_personSD0")]
    rows, maximum_delta, filled = [], 0.0, 0

    def direct(path, group, tail):
        path = Path(path).resolve()
        if not path.is_relative_to(CODE):
            raise ValueError("Nominal qualification trajectory is outside this project")
        inputs[path.relative_to(CODE).as_posix()] = digest(path)
        return score_trajectory(load_hourly(path), group, tail)["summary"]

    for (climate, composition), group in profiles.groupby(["climate", "composition"], sort=True):
        group = group.sort_values("profile_id").reset_index(drop=True)
        means = zone_mean_profiles(group)
        ids = set(members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id)
        tail = np.flatnonzero(group.profile_id.isin(ids))
        if len(group) != 267 or len(tail) != 27:
            raise ValueError("Demand identities or original tail membership changed")
        reference = direct(references.loc[climate, "hourly_path"], group, tail)
        population = reference["mean_reference_mismatch_c"]
        least = reference["fixed_tail_reference_mismatch_c"]
        candidates = library[library.climate.eq(climate) & library.composition.eq(composition)].sort_values(
            ["run_id", "control", "offset_c"], kind="stable").drop_duplicates("physical_plan_id").copy()
        if not candidates.engineering_pass.eq(True).all():
            raise ValueError("Nominal library contains engineering-unqualified operations")
        for index in candidates[candidates.zone_mean_mean_reference_mismatch_c.isna()].index:
            score = direct(candidates.loc[index, "hourly_path"], means, tail)
            for endpoint in ("mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"):
                candidates.loc[index, "zone_mean_" + endpoint] = score[endpoint]
            filled += 1
        full_mask = candidates.mean_reference_mismatch_c.le(population + 1e-9) & candidates.fixed_tail_reference_mismatch_c.le(least + 1e-9)
        mean_mask = candidates.zone_mean_mean_reference_mismatch_c.le(population + 1e-9) & candidates.zone_mean_fixed_tail_reference_mismatch_c.le(least + 1e-9)
        full = choose_minimum_energy(candidates[full_mask])
        mean = choose_minimum_energy(candidates[mean_mask])
        expected = draws[draws.climate.eq(climate) & draws.composition.eq(composition)]
        if len(expected) != 256 or expected.full_choice.nunique() != 1 or expected.zone_mean_choice.nunique() != 1:
            raise ValueError("Zero-error paired selections are inconsistent")
        if full["physical_plan_id"] != expected.full_choice.iloc[0] or mean["physical_plan_id"] != expected.zone_mean_choice.iloc[0]:
            raise ValueError("Explicit hourly and empirical-distribution selections differ")
        for selected in (full, mean):
            for prefix, demand in (("", group), ("zone_mean_", means)):
                actual = direct(selected["hourly_path"], demand, tail)
                for endpoint in ("mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"):
                    difference = abs(actual[endpoint] - selected[prefix + endpoint])
                    maximum_delta = max(maximum_delta, difference)
                    if difference > 1e-8:
                        raise ValueError("Explicit hourly and stored service scores differ")
        excess = max(0.0, mean["mean_reference_mismatch_c"] - population, mean["fixed_tail_reference_mismatch_c"] - least)
        if not np.allclose(expected.zone_mean_selected_full_service_excess_c, excess, rtol=0, atol=1e-8):
            raise ValueError("Nominal selected-operation exceedance differs")
        rows.append({
            "climate": climate, "composition": composition, "candidate_operations": len(candidates),
            "full_qualified_operations": int(full_mask.sum()), "zone_mean_qualified_operations": int(mean_mask.sum()),
            "qualified_by_both": int((full_mask & mean_mask).sum()),
            "qualified_only_by_full": int((full_mask & ~mean_mask).sum()),
            "qualified_only_by_mean": int((~full_mask & mean_mask).sum()),
            "qualified_operation_sets_differ": bool((full_mask != mean_mask).any()),
            "choices_differ": full["physical_plan_id"] != mean["physical_plan_id"],
            "full_choice": full["physical_plan_id"], "zone_mean_choice": mean["physical_plan_id"],
            "zone_mean_selected_full_service_excess_c": excess,
            "energy_difference_kwh": full["facility_purchased_kwh"] - mean["facility_purchased_kwh"],
            "population_budget_c": population, "fixed_tail_budget_c": least,
        })
        print("NOMINAL_INFORMATION", climate, composition, flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 54:
        raise ValueError("Nominal qualification design is incomplete")
    frame.to_csv(table, index=False)
    for name, expected in inputs.items():
        if digest(CODE / name) != expected:
            raise ValueError("Nominal qualification input changed during execution")
    write_json(out / "INPUT_SHA256.json", inputs)
    write_json(audit_path, {
        "status": "completed", "settings": len(frame),
        "qualified_set_changes": int(frame.qualified_operation_sets_differ.sum()),
        "choice_changes": int(frame.choices_differ.sum()),
        "maximum_direct_service_score_difference_c": maximum_delta,
        "additional_spatial_zone_mean_scores": filled, "EnergyPlus_runs": 0,
        "source_sha256": digest(__file__), "output_sha256": digest(table),
    })


if __name__ == "__main__":
    main()
