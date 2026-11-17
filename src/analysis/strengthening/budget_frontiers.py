from common import *
import pandas as pd
import numpy as np
from itertools import product
from analyse_global_results import choose_minimum_energy


def select(frame, population, tail, representation):
    prefix = "zone_mean_" if representation == "zone_mean" else ""
    mask = frame.engineering_pass.fillna(False) & frame[prefix + "mean_reference_mismatch_c"].le(population + 1e-9) & frame[
        prefix + "fixed_tail_reference_mismatch_c"
    ].le(tail + 1e-9)
    candidates = frame[mask]
    if candidates.empty:
        return {
            "status": "infeasible",
            "qualifying_plans": 0,
            "selected_energy_kwh": None,
            "finite_library_minimum_energy_kwh": None,
            "selected_physical_plan": None,
        }
    best = pd.Series(choose_minimum_energy(candidates))
    return {
        "status": "feasible",
        "qualifying_plans": len(candidates),
        "selected_energy_kwh": float(best.facility_purchased_kwh),
        "selected_physical_plan": best.physical_plan_id,
        "finite_library_minimum_energy_kwh": float(candidates.facility_purchased_kwh.min()),
        "selected_population_mismatch_c": float(best.mean_reference_mismatch_c),
        "selected_fixed_tail_mismatch_c": float(best.fixed_tail_reference_mismatch_c),
        "selected_control": best.control,
        "selected_actual_permission": best.permission,
    }


def search_coverage():
    grid = json.loads((ROOT / "audit/GRID_STOP.json").read_text())
    further = grid["further_refinement_executed"]
    planned = {"CANDIDATE": 648, "BASELINE": 216,
               "BASELINE_OFFSETS": 3456 if further else 1728, "UNIFORM_SEARCH": 1674}
    if further:
        planned["CANDIDATE_FURTHER"] = 1296
    rows = []
    for name, expected in planned.items():
        filename = "CANDIDATE_FURTHER_MANIFEST.csv" if name == "CANDIDATE_FURTHER" else name + "_FULL_MANIFEST.csv"
        path = ROOT / "results" / filename
        metrics = ROOT / "results" / (name + "_METRICS.csv")
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        passed = int((frame.completed.fillna(False) & frame.relative_engineering_pass.fillna(False)).sum()) if len(frame) else 0
        scored = len(pd.read_csv(metrics)) if metrics.exists() else 0
        rows.append({"candidate_family": name, "expected_logical_operations": expected,
            "returned_operations": len(frame), "engineering_passed_operations": passed,
            "scored_operations": scored, "complete": len(frame) == expected and passed == expected and scored == expected})
    return rows


def align_spatial_operations(spatial, profiles, run_manifests):
    frame = spatial.copy()
    labels = profiles[["composition_id", "cohort_label"]].drop_duplicates().set_index("composition_id").cohort_label
    frame["composition_id"] = pd.to_numeric(frame.composition, errors="raise").astype(int)
    frame["composition"] = frame.composition_id.map(labels)
    lookup = pd.concat(run_manifests, ignore_index=True).drop_duplicates("run_id").set_index("run_id")
    fields = ["input_sha256", "input_idf_path", "qa_path", "weather_path", "hourly_path", "hourly_sha256"]
    for field in fields:
        frame[field] = frame.run_id.map(lookup[field])
    if frame[["composition", *fields]].isna().any().any():
        raise ValueError("Spatial cohorts or physical-operation metadata are missing from the common library")
    frame["physical_plan_id"] = frame.input_sha256
    return frame


def main():
    out = ROOT / "results/budget_frontiers"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    new = pd.read_csv(ROOT / "results/CANDIDATE_METRICS.csv")
    baseline = pd.read_csv(ROOT / "results/BASELINE_METRICS.csv")
    anchor = pd.read_csv(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv")
    spatial = pd.read_csv(ROOT / "results/base/spatial/GLOBAL_SPATIAL_METRICS.csv")
    spatial = spatial[spatial.layout.eq("random_mixing")]
    spatial["offset_c"] = 0.0
    mapping = pd.read_csv(RAW_RUN / "spatial/GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv")
    annual_runs = pd.read_csv(MATRIX / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv")
    spatial = align_spatial_operations(spatial, pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"), [annual_runs, mapping])
    # Spatial tables carry the same full-service scores but not zone-mean scores.
    # Do not fabricate the latter: the nested-permission frontier uses full demands.
    full = pd.concat([original, new, baseline, spatial], ignore_index=True)
    uniform_search = ROOT / "results/UNIFORM_SEARCH_METRICS.csv"
    if not uniform_search.is_file():
        raise FileNotFoundError(
            "Complete uniform-command benchmark must be scored before common-library frontiers"
        )
    full = pd.concat([full, pd.read_csv(uniform_search)], ignore_index=True)
    baseline_offsets = ROOT / "results/BASELINE_OFFSETS_METRICS.csv"
    if baseline_offsets.exists():
        full = pd.concat([full, pd.read_csv(baseline_offsets)], ignore_index=True)
    further = ROOT / "results/CANDIDATE_FURTHER_METRICS.csv"
    if further.exists():
        full = pd.concat([full, pd.read_csv(further)], ignore_index=True)
    full.loc[full.control.eq("uniform_average"), "permission"] = "building"
    full.to_csv(out / "ALL_OPERATION_LIBRARY.csv", index=False)
    rows = []
    nested = []
    pareto = []
    strategy = []
    for (climate, composition), group in full.groupby(["climate", "composition"], sort=True):
        limits = anchor[anchor.climate.eq(climate) & anchor.composition.eq(composition)].iloc[0]
        pm = float(limits.anchor_mean_mismatch_c)
        pt = float(limits.anchor_tail_mismatch_c)
        libraries = {}
        previous = set()
        for scope, allowed in [
            ("building", ["building"]),
            ("floor", ["building", "floor"]),
            ("native_zone", ["building", "floor", "native_zone"]),
        ]:
            library = (
                group[group.permission.isin(allowed) & group.engineering_pass.fillna(False)]
                .sort_values("facility_purchased_kwh")
                .drop_duplicates("physical_plan_id")
            )
            identities = set(library.physical_plan_id)
            assert previous <= identities
            nested.append(
                {
                    "climate": climate,
                    "composition": composition,
                    "permission": scope,
                    "candidate_operations": len(library),
                    "contains_previous_scope": True,
                    "lower_scope_operation_count": len(previous),
                }
            )
            previous = identities
            libraries[scope] = library
        budgets = [
            ("reference_relative", factor * pm, factor * pt, factor)
            for factor in [0.5, 0.75, 0.9, 1.0, 1.05, 1.1]
        ]
        budgets += [
            ("absolute_mismatch", p, t, None)
            for p, t in product([0.5, 1, 1.5, 2, 2.5], [1, 2, 3, 4, 5])
        ]
        for control, family in group.groupby("control"):
            if control not in [
                "uniform_average",
                "personal_center",
                "personal_range",
                "building_mean",
                "building_median",
                "native_zone_mean",
                "native_zone_median",
                "uniform_command_search",
            ]:
                continue
            scope = (
                "building"
                if control
                in ["uniform_average", "building_mean", "building_median", "uniform_command_search"]
                else "native_zone"
            )
            family = family[family.permission.eq(scope)].drop_duplicates("physical_plan_id")
            for btype, pop, tail, factor in budgets:
                strategy.append(
                    {
                        "climate": climate,
                        "composition": composition,
                        "control": control,
                        "budget_type": btype,
                        "population_budget_c": pop,
                        "fixed_tail_budget_c": tail,
                        "reference_factor": factor,
                        **select(family, pop, tail, "full"),
                    }
                )
        for btype, pop, tail, factor in budgets:
            preceding_energy = None
            for scope, library in libraries.items():
                selected = select(library, pop, tail, "full")
                energy = selected["finite_library_minimum_energy_kwh"]
                if preceding_energy is not None:
                    assert energy is not None and energy <= preceding_energy + 1e-6
                preceding_energy = energy
                rows.append(
                    {
                        "climate": climate,
                        "composition": composition,
                        "budget_type": btype,
                        "reference_factor": factor,
                        "population_budget_c": pop,
                        "fixed_tail_budget_c": tail,
                        "permission": scope,
                        "representation": "full",
                        **selected,
                    }
                )
        library = libraries["native_zone"]
        values = library[
            [
                "mean_reference_mismatch_c",
                "fixed_tail_reference_mismatch_c",
                "facility_purchased_kwh",
            ]
        ].to_numpy()
        for i, row in enumerate(library.itertuples()):
            dominates = np.all(values <= values[i] + 1e-10, axis=1) & np.any(
                values < values[i] - 1e-10, axis=1
            )
            pareto.append(
                {
                    "climate": climate,
                    "composition": composition,
                    "physical_plan_id": row.physical_plan_id,
                    "finite_library_nondominated": not bool(dominates.any()),
                    "population_mismatch_c": row.mean_reference_mismatch_c,
                    "fixed_tail_mismatch_c": row.fixed_tail_reference_mismatch_c,
                    "energy_kwh": row.facility_purchased_kwh,
                    "control": row.control,
                    "permission": row.permission,
                }
            )
    pd.DataFrame(rows).to_csv(out / "COMMON_BUDGET_MINIMUM_ENERGY.csv", index=False)
    pd.DataFrame(strategy).to_csv(out / "STRATEGY_BUDGET_CHOICES.csv", index=False)
    pd.DataFrame(nested).to_csv(out / "NESTED_PERMISSION_CHECKS.csv", index=False)
    pd.DataFrame(pareto).to_csv(out / "FINITE_LIBRARY_PARETO.csv", index=False)
    coverage = search_coverage()
    pd.DataFrame(coverage).to_csv(out / "CANDIDATE_SEARCH_COVERAGE.csv", index=False)
    audit = {
        "status": "completed" if all(row["complete"] for row in coverage) else "partial",
        "candidate_search_coverage": coverage,
        "settings": 54,
        "budget_pairs_per_setting": 31,
        "permission_levels": 3,
        "frontier_cells": len(rows),
        "infeasible_cells": sum(r["status"] == "infeasible" for r in rows),
        "nested_scope_monotonicity": "passed for every setting and budget",
        "interpretation": "Best among retained physical operations only; not a global optimum or real comfort standard",
        "zone_mean_role": "Primary/refined information diagnostics are separate; no uncomputed zone-mean score filled in for spatial rows.",
    }
    write_json(out / "BUDGET_FRONTIER_AUDIT.json", audit)
    record_status("service_budget_frontiers", audit["status"], audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
