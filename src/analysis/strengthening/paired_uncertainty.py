from common import *
import pandas as pd
import numpy as np
from service_distribution import ServiceDistribution, PairedWorseningHours
from scipy.special import ndtr
from profile_identity import person_keys, source_keys
from analyse_global_results import normalise_profiles, load_hourly, zone_mean_profiles

OUT = ROOT / "results/paired_uncertainty"
DRAWS = 256


def keyed_normal(key):
    seed = int(hashlib.sha256(("2026103103|" + str(key)).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).normal(size=DRAWS)


def shared_normal(profile_ids, archive, study_shared=False):
    if study_shared:
        return np.column_stack([keyed_normal("study_source|" + key) for key in source_keys(profile_ids)])
    return keyed_normal("source|" + archive)[:, None]


def choose(eligible, energies, ids):
    minimum = np.where(eligible, energies[None, :], np.inf).min(axis=1)
    tolerance = np.maximum(1e-6, 1e-9 * np.where(np.isfinite(minimum), abs(minimum), 0))
    tied = eligible & (energies[None, :] <= minimum[:, None] + tolerance[:, None])
    ordered = np.argsort(np.asarray(ids).astype(str), kind="mergesort")
    result = np.full(eligible.shape[0], -1, dtype=int)
    for i in ordered:
        mask = (result < 0) & tied[:, i]
        result[mask] = i
    return result


def empirical_temporal_shifts(profile_ids, single_record=False):
    if single_record:
        pool = pd.read_csv(ROOT / "results/demand_validation/single_record/timed/CONTINUOUS_PROXY_ERRORS.csv")
        pool = pool[pool.partition.eq("development") & pool.center_model.eq("raw_center")].rename(
            columns={"center_model": "model", "signed_later_proxy_shift_c": "signed_unit_center_shift_c"}
        )
    else:
        pool = pd.read_csv(
            ROOT / "results/demand_validation/temporal_centers/DEVELOPMENT_EMPIRICAL_TEMPORAL_SHIFT_POOL.csv"
        )
    assert pool.partition.eq("development").all() and pool.model.eq("raw_center").all()
    groups = [g for _, g in pool.groupby("source", sort=True)]
    selected = np.random.default_rng(2026103103).integers(0, len(groups), DRAWS)
    result = np.zeros((DRAWS, len(profile_ids)))
    for j, profile in enumerate(person_keys(profile_ids)):
        uniform = ndtr(keyed_normal("observed_temporal_proxy|" + str(profile)))
        for k, g in enumerate(groups):
            mask = selected == k
            values = g.signed_unit_center_shift_c.to_numpy()
            bias = float(values.mean())
            residual = values - bias
            index = np.minimum((uniform[mask] * len(values)).astype(int), len(values) - 1)
            result[mask, j] = bias + residual[index]
    return result


def run(refined=False, pooled=False, empirical=False, single_record=False, study_shared=False):
    global OUT
    if refined:
        OUT = ROOT / "results/paired_uncertainty_refined"
    if pooled:
        OUT = ROOT / "results/paired_uncertainty_pooled"
    if empirical:
        OUT = ROOT / "results/paired_uncertainty_pooled_temporal"
    if single_record:
        assert empirical and pooled and refined
        OUT = ROOT / "results/paired_uncertainty_pooled_single_record_temporal"
    if study_shared:
        assert refined and pooled and not empirical and not single_record
        OUT = ROOT / "results/paired_uncertainty_pooled_study_shared"
    OUT = OUT.with_name(OUT.name + "_identity")
    assert (ROOT / "audit/status/original_annual_reproduction.json").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    metric_path = (
        "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv"
        if pooled
        else (
            "results/candidate_refinement/STOPPED_GRID_METRICS.csv"
            if refined
            else "results/base/annual/GLOBAL_RUN_METRICS.csv"
        )
    )
    metrics = pd.read_csv(ROOT / metric_path)
    if refined:
        grid = json.loads((ROOT / "audit/GRID_STOP.json").read_text())
        expected = 2754 if grid["further_refinement_executed"] else 1458
        assert (
            len(pd.read_csv(ROOT / "results/candidate_refinement/STOPPED_GRID_METRICS.csv"))
            == expected
        ), "Required candidate scores are incomplete"
    profiles = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    if empirical:
        profiles = profiles[profiles.composition.str.contains("Multinational")]
    memberships = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    anchors = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    anchors = anchors[anchors.target_c.eq(25)].set_index("climate")
    results = []
    selection = []
    distrows = []
    scenarios = (
        [(s, None) for s in [0, 0.5, 1.0]]
        if empirical
        else [(s, p) for s in [0, 0.25, 0.5] for p in [0, 0.25, 0.5]]
    )
    for (climate, composition), g in profiles.groupby(["climate", "composition"], sort=True):
        g = g.sort_values("profile_id").reset_index(drop=True)
        group = metrics[metrics.climate.eq(climate) & metrics.composition.eq(composition)].copy()
        group = (
            group.sort_values(["run_id", "control", "offset_c"], kind="mergesort")
            .drop_duplicates("physical_plan_id")
            .reset_index(drop=True)
        )
        tailids = set(
            memberships[
                memberships.climate.eq(climate) & memberships.composition.eq(composition)
            ].profile_id.astype(str)
        )
        tail = np.flatnonzero(g.profile_id.astype(str).isin(tailids))
        assert len(tail) == 27
        reference = load_hourly(Path(anchors.loc[climate, "hourly_path"]))
        refcurve = ServiceDistribution(reference, g)
        curves = []
        worsening = []
        for row in group.itertuples():
            operation = load_hourly(Path(row.hourly_path))
            curves.append(ServiceDistribution(operation, g))
            worsening.append(PairedWorseningHours(operation, reference, g))
        nominal = g.centre_c.to_numpy()
        source = "ASHRAE" if "Multinational" in composition else "CTCD"
        individual = np.column_stack(
            [keyed_normal("person|" + str(p)) for p in person_keys(g.profile_id)]
        )
        shared = shared_normal(g.profile_id, source, study_shared)
        temporal_shift = empirical_temporal_shifts(g.profile_id, single_record) if empirical else None
        originalmean, originaltail = refcurve.score(nominal, tail)
        energy = group.facility_purchased_kwh.to_numpy()
        ids = group.physical_plan_id.astype(str).to_numpy()
        for source_sd, person_sd in scenarios:
            centers = (
                nominal[None, :] + source_sd * temporal_shift
                if empirical
                else nominal[None, :] + source_sd * shared + person_sd * individual
            )
            means = centers.copy()
            for zone in g.zone_id.unique():
                ix = np.flatnonzero(g.zone_id.eq(zone))
                means[:, ix] = centers[:, ix].mean(axis=1)[:, None]
            rm, rt = refcurve.score(centers, tail)
            scores = [curve.score(centers, tail) for curve in curves]
            mean = np.column_stack([x[0] for x in scores])
            tails = np.column_stack([x[1] for x in scores])
            mean_scores = [curve.score(means, tail) for curve in curves]
            zm = np.column_stack([x[0] for x in mean_scores])
            zt = np.column_stack([x[1] for x in mean_scores])
            paired = (mean <= rm[:, None] + 1e-9) & (tails <= rt[:, None] + 1e-9)
            fixed = (mean <= originalmean + 1e-9) & (tails <= originaltail + 1e-9)
            zone = (zm <= rm[:, None] + 1e-9) & (zt <= rt[:, None] + 1e-9)
            fullchoice = choose(paired, energy, group.run_id.to_numpy())
            meanchoice = choose(zone, energy, group.run_id.to_numpy())
            both = (fullchoice >= 0) & (meanchoice >= 0)
            extra = np.full(DRAWS, np.nan)
            extra[both] = energy[fullchoice[both]] - energy[meanchoice[both]]
            selected_mean_excess = np.full(DRAWS, np.nan)
            selected_population_loss = np.full(DRAWS, np.nan)
            selected_tail_loss = np.full(DRAWS, np.nan)
            draws = np.flatnonzero(both)
            selected_mean_excess[draws] = np.maximum(
                np.maximum(
                    mean[draws, meanchoice[draws]] - rm[draws],
                    tails[draws, meanchoice[draws]] - rt[draws],
                ),
                0,
            )
            selected_population_loss[draws] = (
                mean[draws, meanchoice[draws]] - mean[draws, fullchoice[draws]]
            )
            selected_tail_loss[draws] = (
                tails[draws, meanchoice[draws]] - tails[draws, fullchoice[draws]]
            )
            label = (
                f"observed_temporal_proxy_shift_fraction{source_sd:g}"
                if empirical
                else f"{'studyGroupSD' if study_shared else 'sourceSD'}{source_sd:g}_personSD{person_sd:g}"
            )
            for j, row in enumerate(group.itertuples()):
                excess = np.maximum(np.maximum(mean[:, j] - rm, tails[:, j] - rt), 0)
                person = curves[j].individual(centers)
                refperson = refcurve.individual(centers)
                affected = (person > refperson + 1e-9).sum(axis=1)
                coverage = curves[j].coverage(centers - 1, centers + 1)
                outside_hours = (1 - coverage) @ curves[j].person_hours
                results.append(
                    {
                        "climate": climate,
                        "composition": composition,
                        "scenario": label,
                        "physical_plan_id": row.physical_plan_id,
                        "source_sd_c": None if empirical else source_sd,
                        "person_sd_c": person_sd,
                        "observed_temporal_shift_fraction": source_sd if empirical else None,
                        "conditional_paired_exceedance_probability": float((~paired[:, j]).mean()),
                        "conditional_fixed_limit_exceedance_probability": float(
                            (~fixed[:, j]).mean()
                        ),
                        "excess_median_c": float(np.median(excess)),
                        "excess_p95_c": float(np.quantile(excess, 0.95)),
                        "affected_profiles_mean": float(affected.mean()),
                        "affected_profiles_p95": float(np.quantile(affected, 0.95)),
                        "person_hours_worse_than_paired_reference_mean": float(
                            worsening[j].person_hours(centers).mean()
                        ),
                        "person_hours_outside_center_plusminus1c_mean": float(outside_hours.mean()),
                        "extra_metric_budget": "±1C center band descriptive only; not real acceptability",
                        "choice_probability": float((fullchoice == j).mean()),
                        "post_operation_worst10_mismatch_mean_c": float(
                            np.partition(person, -27, axis=1)[:, -27:].mean()
                        ),
                    }
                )
            selection.append(
                {
                    "climate": climate,
                    "composition": composition,
                    "scenario": label,
                    "conditional_full_feasible_fraction": float((fullchoice >= 0).mean()),
                    "conditional_zone_mean_feasible_fraction": float((meanchoice >= 0).mean()),
                    "conditional_choice_disagreement_fraction": float(
                        (fullchoice[both] != meanchoice[both]).mean()
                    )
                    if both.any()
                    else None,
                    "extra_selected_energy_median_kwh": float(np.nanmedian(extra))
                    if both.any()
                    else None,
                    "extra_selected_energy_p05_kwh": float(np.nanquantile(extra, 0.05))
                    if both.any()
                    else None,
                    "extra_selected_energy_p95_kwh": float(np.nanquantile(extra, 0.95))
                    if both.any()
                    else None,
                    "paired_draws_with_choices": int(both.sum()),
                }
            )
            selection[-1].update(
                {
                    "zone_mean_selection_full_service_exceedance_fraction": float(
                        (selected_mean_excess[both] > 1e-9).mean()
                    )
                    if both.any()
                    else None,
                    "zone_mean_selected_excess_p95_c": float(
                        np.nanquantile(selected_mean_excess, 0.95)
                    )
                    if both.any()
                    else None,
                }
            )
            for d in range(DRAWS):
                distrows.append(
                    {
                        "climate": climate,
                        "composition": composition,
                        "scenario": label,
                        "draw": d,
                        "full_choice": ids[fullchoice[d]] if fullchoice[d] >= 0 else "infeasible",
                        "zone_mean_choice": ids[meanchoice[d]]
                        if meanchoice[d] >= 0
                        else "infeasible",
                        "extra_energy_kwh": extra[d],
                        "zone_mean_selected_full_service_excess_c": selected_mean_excess[d],
                        "zone_mean_selected_population_loss_vs_full_choice_c": selected_population_loss[
                            d
                        ],
                        "zone_mean_selected_fixed_tail_loss_vs_full_choice_c": selected_tail_loss[
                            d
                        ],
                    }
                )
        pd.DataFrame(results).to_csv(OUT / "OPERATION_EXCEEDANCE_AND_CHOICE.csv", index=False)
        pd.DataFrame(selection).to_csv(OUT / "SETTING_SELECTION_STABILITY.csv", index=False)
        print(climate, composition, "completed", flush=True)
    pd.DataFrame(distrows).to_csv(OUT / "PAIRED_SELECTION_DRAWS.csv.gz", index=False)
    audit = {
        "status": "partial",
        "library": metric_path,
        "completed": "Fixed-operation rescoring and common-library reselection under declared correlated sensitivity scenarios,256 paired draws each; both candidate-relative and fixed-numeric-limit analyses.",
        "interpretation": "Scenario-conditional frequencies, not empirically calibrated probabilities. Raw identities are linked; latent estimation error is not separately identified from temporal variation and response noise.",
        "shared_error_grouping": "empirical source-pool transport" if empirical else (
            "study/contributor" if study_shared else "archive"),
        "correlation": "One draw per retained source_key replaces the archive term; personal pairing unchanged."
        if study_shared else "Archive-wide draw shared across people; personal draw keyed by qualified source/building/subject or raw record identity, retained across repeated representations, compositions, climates, operations and25C reference",
        "not_completed": [
            "calibrated measurement-error propagation",
            "control regeneration under uncertainty",
        ],
        "temporal_sensitivity": "separate completed calendar-block analysis in temporal_uncertainty; not an empirical adaptation model",
        "tail": "original27 members unchanged; post-operation worst27 is separate output",
        "operations": len(results),
        "settings": len(selection) // len(scenarios),
        "scenarios": len(scenarios),
        "draws": DRAWS,
        "empirical_calibration_records_per_person": 1 if single_record else "multiple early records",
        "empirical_temporal_transport": "Development units only; equal weight to identified calibration sources; shared source-specific mean temporal proxy shift plus within-source residual sampled per linked raw identity, retained across operations and climates. ASHRAE-derived compositions only; transport stress, not person-specific error calibration."
        if empirical
        else None,
        "person_hours": "model native occupancy equally divided among assigned profiles; center-band endpoint is descriptive",
    }
    write_json(OUT / "UNCERTAINTY_AUDIT.json", audit)
    record_status(OUT.name, "partial", audit, audit["not_completed"])


if __name__ == "__main__":
    run("--refined" in sys.argv, "--pooled" in sys.argv, "--empirical" in sys.argv, "--single-record" in sys.argv, "--study-shared" in sys.argv)
