"""Recompute protected-quantile choices and their energy/CO2 opportunity cost.

No expected result is read until the independent reference check at the end.
Frozen candidate scores, carrier totals, budgets, folds and rules are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def stats(values):
    a = pd.Series(values, dtype=float).dropna()
    if a.empty:
        return {"n": 0}
    return {
        "n": len(a),
        "min": float(a.min()),
        "median": float(a.median()),
        "p90": float(a.quantile(0.90)),
        "p95": float(a.quantile(0.95)),
        "max": float(a.max()),
    }


def summarize(group):
    comparable = group[group.status.eq("matched_full_qualified")]
    return {
        "design_cases": len(group),
        "full_available": int(group.full_available.sum()),
        "matched_full_qualified": len(comparable),
        "missed_despite_full_available": int(
            group.status.eq("missed_despite_full_available").sum()
        ),
        "neither_available": int(group.status.eq("no_full_or_certified_selection").sum()),
        "selected_false_accept": int(group.selected_false_accept.sum()),
        "coverage_of_solvable": None
        if not group.full_available.any()
        else len(comparable) / group.full_available.sum(),
        "same_selected_plan": int(comparable.same_selected_plan.sum()),
        "energy_positive_count": int(comparable.energy_regret_kwh.gt(1e-6).sum()),
        "carbon_positive_count": int(comparable.operational_co2_delta_kg.gt(1e-6).sum()),
        "carbon_negative_count": int(comparable.operational_co2_delta_kg.lt(-1e-6).sum()),
        "energy_regret_kwh": stats(comparable.energy_regret_kwh),
        "energy_regret_percent": stats(comparable.energy_regret_percent),
        "operational_co2_delta_kg": stats(comparable.operational_co2_delta_kg),
        "operational_co2_delta_percent": stats(comparable.operational_co2_delta_percent),
    }


def main():
    parser = argparse.ArgumentParser()
    project = Path(__file__).resolve().parents[3]
    parser.add_argument("--data-root", type=Path, default=project / "clean_data")
    parser.add_argument("--release-root", type=Path, default=project)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.release_root.resolve() / "src"))
    from thermal_heterogeneity.building.annual import choose_minimum_energy, qualify
    from thermal_heterogeneity.implementation.protected_quantiles import calibrate_margins

    paths = {
        "representation_scores": args.data_root / "implementation/representation_scores.csv.gz",
        "annual_actions": args.data_root / "building/annual_action_library.csv.gz",
    }
    source = pd.read_csv(paths["representation_scores"])
    actions = pd.read_csv(paths["annual_actions"]).rename(columns={"context_id": "city"})
    assert not actions.duplicated(["city", "physical_action_id"]).any()
    keys = ["population_scenario", "city", "seed", "lambda"]
    fullcols = keys + ["physical_action_id", "mean_c", "fixed_tail_c"]
    full = source[source.representation.eq("full")][fullcols].rename(
        columns={"mean_c": "full_mean_c", "fixed_tail_c": "full_tail_c"}
    )
    scores = source[source.representation.ne("full")].merge(
        full, on=keys + ["physical_action_id"], validate="many_to_one"
    )
    scores = scores.merge(
        actions[["city", "physical_action_id", "operational_co2_kg"]],
        on=["city", "physical_action_id"],
        validate="many_to_one",
    )
    assert scores.operational_co2_kg.notna().all()
    rules = [
        ("quantile5", "pooled_q95", 0.95),
        ("quantile5", "pooled_q90", 0.90),
        ("quantile3", "pooled_q90", 0.90),
    ]
    folds = []
    for city, seed in sorted(
        scores[["city", "seed"]].drop_duplicates().itertuples(index=False, name=None)
    ):
        mask = scores.city.eq(city) & scores.seed.eq(seed)
        folds.append(("leave_one_context_seed_out", f"{city}__{seed}", ~mask, mask))
    for population in sorted(scores.population_scenario.unique()):
        mask = scores.population_scenario.eq(population)
        folds.append(("leave_one_population_out", population, ~mask, mask))
    assignments = pd.read_csv(args.data_root / "population/population_assignments_minimized.csv.gz")
    profiles = pd.read_csv(args.data_root / "human/human_profiles_minimized.csv.gz")
    provenance = assignments.merge(
        profiles[["release_profile_id", "release_unit_id", "release_building_group_id"]],
        left_on="release_archive_profile_id",
        right_on="release_profile_id",
        validate="many_to_one",
    )
    disjoint_proof = []
    for training, heldout in [
        ("india_archive_stratum", "us_archive_stratum"),
        ("us_archive_stratum", "india_archive_stratum"),
    ]:
        tr = provenance[provenance.scenario_id.eq(training)]
        te = provenance[provenance.scenario_id.eq(heldout)]
        shared_units = set(tr.release_unit_id) & set(te.release_unit_id)
        shared_buildings = set(tr.release_building_group_id) & set(te.release_building_group_id)
        assert not shared_units and not shared_buildings
        disjoint_proof.append(
            dict(
                training=training,
                heldout=heldout,
                train_units=tr.release_unit_id.nunique(),
                test_units=te.release_unit_id.nunique(),
                shared_units=len(shared_units),
                shared_source_buildings=len(shared_buildings),
            )
        )
        folds.append(
            (
                "cross_country_person_disjoint_transfer",
                f"{training}__to__{heldout}",
                scores.population_scenario.eq(training),
                scores.population_scenario.eq(heldout),
            )
        )
    decisions, candidates, margins = [], [], []
    for scheme, fold_id, train_mask, test_mask in folds:
        train, test = scores[train_mask], scores[test_mask]
        for representation, strategy, q in rules:
            localtrain = train[train.representation.eq(representation)]
            mm, tm = calibrate_margins(
                localtrain,
                approximate_mean="mean_c",
                approximate_tail="fixed_tail_c",
                full_mean="full_mean_c",
                full_tail="full_tail_c",
                quantile=q,
            )
            margins.append(
                dict(
                    split_scheme=scheme,
                    fold_id=fold_id,
                    representation=representation,
                    strategy=strategy,
                    mean_margin_c=mm,
                    tail_margin_c=tm,
                    train_score_rows=len(localtrain),
                )
            )
            for key, group in test[test.representation.eq(representation)].groupby(keys, sort=True):
                base = dict(zip(keys, key)) | dict(
                    split_scheme=scheme,
                    fold_id=fold_id,
                    representation=representation,
                    strategy=strategy,
                )
                for fraction in [0.5, 0.75, 0.9, 1.0]:
                    pool = group.copy()
                    pool["mean_budget_c"] = pool.common25_mean_anchor_c * fraction
                    pool["fixed_tail_budget_c"] = pool.common25_fixed_tail_anchor_c * fraction
                    pool["full_service_eligible"] = qualify(
                        pool, mean_score="full_mean_c", tail_score="full_tail_c"
                    )
                    pool["protected_mean_c"] = pool.mean_c + mm
                    pool["protected_tail_c"] = pool.fixed_tail_c + tm
                    pool["certified_eligible"] = qualify(
                        pool, mean_score="protected_mean_c", tail_score="protected_tail_c"
                    )
                    fullchoice = choose_minimum_energy(pool, eligible="full_service_eligible")
                    choice = choose_minimum_energy(pool, eligible="certified_eligible")
                    row = base | dict(
                        service_fraction=fraction,
                        mean_margin_c=mm,
                        tail_margin_c=tm,
                        mean_budget_c=float(pool.mean_budget_c.iloc[0]),
                        fixed_tail_budget_c=float(pool.fixed_tail_budget_c.iloc[0]),
                        full_available=fullchoice is not None,
                        certified_available=choice is not None,
                        full_candidate_count=int(pool.full_service_eligible.sum()),
                        certified_candidate_count=int(pool.certified_eligible.sum()),
                        candidate_false_accept_count=int(
                            (pool.certified_eligible & ~pool.full_service_eligible).sum()
                        ),
                        candidate_retained_count=int(
                            (pool.certified_eligible & pool.full_service_eligible).sum()
                        ),
                        selected_false_accept=False,
                        same_selected_plan=False,
                    )
                    for label, selected in [("full", fullchoice), ("protected", choice)]:
                        row[f"{label}_selected_action_id"] = (
                            None if selected is None else selected.physical_action_id
                        )
                        row[f"{label}_site_kwh"] = (
                            np.nan if selected is None else float(selected.site_kwh)
                        )
                        row[f"{label}_operational_co2_kg"] = (
                            np.nan if selected is None else float(selected.operational_co2_kg)
                        )
                    if choice is not None and not bool(choice.full_service_eligible):
                        row["status"] = "selected_false_accept"
                        row["selected_false_accept"] = True
                    elif choice is not None and fullchoice is not None:
                        row["status"] = "matched_full_qualified"
                        row["same_selected_plan"] = (
                            choice.physical_action_id == fullchoice.physical_action_id
                        )
                    elif fullchoice is not None:
                        row["status"] = "missed_despite_full_available"
                    else:
                        row["status"] = "no_full_or_certified_selection"
                    valid = row["status"] == "matched_full_qualified"
                    row["energy_regret_kwh"] = (
                        row["protected_site_kwh"] - row["full_site_kwh"] if valid else np.nan
                    )
                    row["energy_regret_percent"] = (
                        100 * row["energy_regret_kwh"] / row["full_site_kwh"] if valid else np.nan
                    )
                    row["operational_co2_delta_kg"] = (
                        row["protected_operational_co2_kg"] - row["full_operational_co2_kg"]
                        if valid
                        else np.nan
                    )
                    row["operational_co2_delta_percent"] = (
                        100 * row["operational_co2_delta_kg"] / row["full_operational_co2_kg"]
                        if valid
                        else np.nan
                    )
                    decisions.append(row)
                    keep = [
                        "physical_action_id",
                        "mean_c",
                        "fixed_tail_c",
                        "full_mean_c",
                        "full_tail_c",
                        "protected_mean_c",
                        "protected_tail_c",
                        "mean_budget_c",
                        "fixed_tail_budget_c",
                        "full_service_eligible",
                        "certified_eligible",
                        "site_kwh",
                        "operational_co2_kg",
                    ]
                    candidates.append(pool[keep].assign(**base, service_fraction=fraction))
    decisions = pd.DataFrame(decisions)
    candidates = pd.concat(candidates, ignore_index=True)
    decisions.to_csv(out / "DECISIONS.csv", index=False)
    candidates.to_csv(
        out / "CANDIDATE_EVALUATIONS.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(margins).to_csv(out / "FOLD_MARGINS.csv", index=False)
    group_keys = ["split_scheme", "representation", "strategy"]
    summary, bygroup = [], []
    for key, group in decisions.groupby(group_keys, sort=True):
        item = dict(zip(group_keys, key)) | summarize(group)
        cand = candidates
        for name, value in zip(group_keys, key):
            cand = cand[cand[name].eq(value)]
        fullpass, certified = cand.full_service_eligible, cand.certified_eligible
        item.update(
            candidate_evaluations=len(cand),
            full_qualified_evaluations=int(fullpass.sum()),
            retained_qualified_evaluations=int((fullpass & certified).sum()),
            candidate_false_accept_evaluations=int((~fullpass & certified).sum()),
            candidate_retention=float((fullpass & certified).sum() / fullpass.sum()),
        )
        summary.append(item)
        for dimension in ["city", "population_scenario", "lambda"]:
            for value, subset in group.groupby(dimension, sort=True):
                bygroup.append(
                    dict(zip(group_keys, key))
                    | dict(dimension=dimension, value=str(value))
                    | summarize(subset)
                )
    q5 = decisions[decisions.representation.eq("quantile5") & decisions.strategy.eq("pooled_q95")]
    table_rows = []
    groups = [
        (
            "Primary context-seed holdout",
            q5[q5.split_scheme.eq("leave_one_context_seed_out")],
            "Other context-seed groups",
            "All three archive labels",
            "All test units reused across contexts",
        )
    ]
    for label, group in q5[q5.split_scheme.eq("leave_one_population_out")].groupby(
        "population_scenario"
    ):
        heldout_units = set(provenance.loc[provenance.scenario_id.eq(label), "release_unit_id"])
        training_units = set(provenance.loc[~provenance.scenario_id.eq(label), "release_unit_id"])
        groups.append(
            (
                "Composition-label holdout",
                group,
                "Other two archive labels",
                label,
                f"{len(heldout_units & training_units)} shared units",
            )
        )
    for fold_id, group in q5[q5.split_scheme.eq("cross_country_person_disjoint_transfer")].groupby(
        "fold_id"
    ):
        training, test = fold_id.split("__to__")
        groups.append(
            (
                "Archive-unit-disjoint country transfer",
                group,
                training,
                test,
                "0 shared units and 0 source buildings",
            )
        )
    for name, group, training, test, overlap in groups:
        row = dict(
            split=name, calibration_group=training, evaluation_group=test, unit_overlap=overlap
        ) | summarize(group)
        cand = candidates[
            candidates.representation.eq("quantile5")
            & candidates.strategy.eq("pooled_q95")
            & candidates.split_scheme.eq(group.split_scheme.iloc[0])
        ]
        if group.population_scenario.nunique() == 1:
            cand = cand[cand.population_scenario.eq(group.population_scenario.iloc[0])]
        fullpass, certified = cand.full_service_eligible, cand.certified_eligible
        row.update(
            candidate_evaluations=len(cand),
            full_qualified_evaluations=int(fullpass.sum()),
            retained_qualified_evaluations=int((fullpass & certified).sum()),
            candidate_false_accept_evaluations=int((~fullpass & certified).sum()),
            candidate_retention=float((fullpass & certified).sum() / fullpass.sum()),
        )
        table_rows.append(row)
    pd.json_normalize(table_rows).to_csv(
        out / "TABLE_S17_PROTECTED_QUANTILE_TRANSFER.csv", index=False
    )
    (out / "SUMMARY.json").write_text(
        json.dumps(
            {
                "building_physics_changes": 0,
                "new_EnergyPlus_runs": 0,
                "decision_row_count": len(decisions),
                "candidate_evaluation_row_count": len(candidates),
                "rules": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "GROUP_SUMMARIES.json").write_text(json.dumps(bygroup, indent=2), encoding="utf-8")
    pd.json_normalize(bygroup).to_csv(out / "GROUP_SUMMARIES.csv", index=False)
    primary = decisions[
        decisions.representation.eq("quantile5")
        & decisions.strategy.eq("pooled_q95")
        & decisions.split_scheme.eq("leave_one_context_seed_out")
    ]
    primary.sort_values("energy_regret_percent", ascending=False, na_position="last").to_csv(
        out / "PRIMARY_ORDERED_CASES.csv", index=False
    )
    primary[
        [
            "population_scenario",
            "city",
            "seed",
            "lambda",
            "service_fraction",
            "status",
            "full_available",
            "certified_available",
            "same_selected_plan",
            "full_site_kwh",
            "energy_regret_kwh",
            "energy_regret_percent",
            "operational_co2_delta_kg",
            "operational_co2_delta_percent",
        ]
    ].to_csv(out / "MAIN_FIG6G_Q5_REGRET.csv", index=False)
    assert len(primary) == 1008 and int(primary.full_available.sum()) == 606
    assert int(primary.selected_false_accept.sum()) == 0
    assert primary.energy_regret_kwh.dropna().min() >= -1e-6
    reference = pd.read_csv(args.data_root / "expected/quantile_decisions_reference.csv.gz")
    checkkeys = keys + ["service_fraction", "representation", "strategy", "split_scheme"]
    joined = decisions.merge(
        reference, on=checkkeys, suffixes=("_new", "_reference"), validate="one_to_one"
    )
    assert len(joined) == 6048
    for label in ["full_available", "certified_available"]:
        assert (joined[f"{label}_new"] == joined[f"{label}_reference"]).all()
    assert (
        joined.protected_selected_action_id.fillna("")
        == joined.certified_selected_action_id.fillna("")
    ).all()
    assert np.allclose(
        joined.energy_regret_kwh.fillna(0),
        joined.conservative_energy_penalty_kwh.fillna(0),
        atol=1e-6,
        rtol=0,
    )
    paths["assignments"] = args.data_root / "population/population_assignments_minimized.csv.gz"
    paths["profiles"] = args.data_root / "human/human_profiles_minimized.csv.gz"
    hashes = {
        path.relative_to(args.data_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths.values()
    }
    (out / "INPUT_MANIFEST.json").write_text(
        json.dumps(
            dict(
                calculation_inputs=hashes,
                reference_only_after_calculation={
                    "expected/quantile_decisions_reference.csv.gz": hashlib.sha256(
                        (
                            args.data_root / "expected/quantile_decisions_reference.csv.gz"
                        ).read_bytes()
                    ).hexdigest()
                },
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_package_functions=[
                    "building.annual.qualify",
                    "building.annual.choose_minimum_energy",
                    "implementation.protected_quantiles.calibrate_margins",
                ],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "QA.json").write_text(
        json.dumps(
            dict(
                passed=True,
                reference_decisions_verified=6048,
                no_new_physical_runs=True,
                no_physics_changes=True,
                input_sha256=hashes,
                person_disjoint_proof=disjoint_proof,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                row
                for row in summary
                if row["representation"] == "quantile5" and row["strategy"] == "pooled_q95"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
