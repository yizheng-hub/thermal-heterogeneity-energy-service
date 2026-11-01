from common import *
import pandas as pd
import numpy as np
import argparse
from service_distribution import ServiceDistribution
from analyse_global_results import (
    normalise_profiles,
    load_hourly,
    energy_metrics,
    zone_mean_profiles,
    fixed_plan_information_diagnostic,
)


def score_manifest(name):
    path = (
        ROOT
        / "results"
        / (
            "CANDIDATE_FURTHER_MANIFEST.csv"
            if name == "candidate_further"
            else name.upper() + "_FULL_MANIFEST.csv"
        )
    )
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    review = path.with_name(path.stem + "_STRICT_WARNING_REVIEW.csv")
    if review.exists():
        strict = pd.read_csv(review).set_index("physical_id").strict_warning_review_pass
        manifest["relative_engineering_pass"] &= manifest.physical_id.map(strict).fillna(False)
    if name == "placement":
        p = pd.read_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv")
    else:
        p = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    p = normalise_profiles(p)
    memberships = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    original_profiles = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    controls = {
        2026102001: "Multinational A",
        2026102002: "Multinational B",
        2026102003: "Multinational C",
        2026102501: "China A",
        2026102502: "China B",
        2026102503: "China C",
    }
    rows = []
    for row in manifest.itertuples():
        composition = controls[row.composition_id]
        if not row.completed or not row.relative_engineering_pass:
            continue
        group = (
            p[p.climate.eq(row.climate) & p.composition.eq(composition) & p.layout.eq(row.layout)]
            .sort_values("profile_id")
            .reset_index(drop=True)
        )
        assert len(group) == 267
        tailids = set(
            memberships[
                memberships.climate.eq(row.climate) & memberships.composition.eq(composition)
            ].profile_id
        )
        tail = np.flatnonzero(group.profile_id.isin(tailids))
        assert len(tail) == 27
        data = load_hourly(Path(row.hourly_path))
        curve = ServiceDistribution(data, group)
        individual = curve.individual(group.centre_c.to_numpy())
        mean, least = curve.score(group.centre_c.to_numpy(), tail)
        zm, zt = curve.score(zone_mean_profiles(group).centre_c.to_numpy(), tail)
        coverage = curve.coverage(group.lower_c.to_numpy(), group.upper_c.to_numpy())
        lowerloss = curve.individual(group.lower_c.to_numpy())
        upperloss = curve.individual(group.upper_c.to_numpy())
        excess = 0.5 * (lowerloss + upperloss - (group.upper_c - group.lower_c).to_numpy())
        score = {
            "climate": row.climate,
            "composition": composition,
            "composition_id": row.composition_id,
            "layout": row.layout,
            "permission": row.permission,
            "control": row.control,
            "offset_c": row.offset_c,
            "run_id": row.run_id,
            "physical_plan_id": row.input_sha256,
            "input_sha256": row.input_sha256,
            "targets_sha256": row.targets_sha256,
            "hourly_path": row.hourly_path,
            "hourly_sha256": row.hourly_sha256,
            "input_idf_path": row.input_idf_path,
            "qa_path": row.qa_path,
            "engineering_pass": True,
            "assigned_profile_count": len(group),
            "occupied_native_person_hours": float(curve.person_hours.sum()),
            "mean_reference_mismatch_c": float(mean),
            "fixed_tail_reference_mismatch_c": float(least),
            "zone_mean_mean_reference_mismatch_c": float(zm),
            "zone_mean_fixed_tail_reference_mismatch_c": float(zt),
            "modeled_interval_person_hour_coverage": float(coverage @ curve.weights),
            "modeled_interval_mean_excess_c": float(excess @ curve.weights),
            "post_operation_worst10_reference_mismatch_c": float(
                np.partition(individual, -27)[-27:].mean()
            ),
            **energy_metrics(data),
        }
        if name == "placement":
            old = original_profiles[
                original_profiles.climate.eq(row.climate)
                & original_profiles.composition.eq(composition)
            ].sort_values("profile_id")
            score["moved_profiles"] = int(
                np.sum(group.zone_id.to_numpy() != old.zone_id.to_numpy())
            )
        rows.append(score)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "results" / (name.upper() + "_METRICS.csv"), index=False)
    record_status(
        name + "_scoring",
        "completed" if len(result) == len(manifest) else "partial",
        {
            "planned": len(manifest),
            "scored": len(result),
            "failed_operations_retained_in": str(path),
        },
    )
    print(name, "scored", len(result), "of", len(manifest), flush=True)


def analysis():
    old = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    new = pd.read_csv(ROOT / "results/CANDIDATE_METRICS.csv")
    anchors = pd.read_csv(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv")
    out = ROOT / "results/candidate_refinement"
    out.mkdir(parents=True, exist_ok=True)
    union = pd.concat([old, new], ignore_index=True)
    union.to_csv(out / "REFINED_RUN_METRICS.csv", index=False)
    decisions = fixed_plan_information_diagnostic(union, anchors, fractions=(1.0,))
    decisions.to_csv(out / "REFINED_INFORMATION_DIAGNOSTIC.csv", index=False)
    original = pd.read_csv(ROOT / "results/base/annual/GLOBAL_INFORMATION_DIAGNOSTIC.csv")
    original = original[original.service_fraction.eq(1)]
    key = ["climate", "composition", "layout", "permission"]
    convergence = decisions.merge(
        original, on=key, suffixes=("_refined", "_original"), validate="one_to_one"
    )
    for representation in ["full", "zone_mean"]:
        name = representation + "_selected_facility_purchased_kwh"
        convergence[representation + "_relative_selected_energy_change"] = (
            convergence[name + "_refined"] / convergence[name + "_original"] - 1
        )
    convergence.to_csv(out / "GRID_ENERGY_CONVERGENCE.csv", index=False)
    maximum = float(convergence.full_relative_selected_energy_change.abs().max())
    report = {
        "original_logical_candidates": len(old),
        "new_logical_candidates_scored": len(new),
        "expected_new_candidates": 648,
        "settings": len(decisions),
        "qualifying_set_changes": int(decisions.eligibility_disagreement.sum()),
        "choice_changes": int(decisions.selection_disagreement.sum()),
        "maximum_absolute_relative_full_selected_energy_change": maximum,
        "further_uniform_refinement_required": maximum > 0.001,
        "status": "completed" if len(new) == 648 else "partial",
        "finite_library_only": True,
    }
    write_json(out / "REFINEMENT_AUDIT.json", report)
    record_status("candidate_refinement_analysis", report["status"], report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "candidate",
            "candidate_further",
            "baseline",
            "baseline_offsets",
            "placement",
            "analysis",
        ],
    )
    args = parser.parse_args()
    analysis() if args.task == "analysis" else score_manifest(args.task)
