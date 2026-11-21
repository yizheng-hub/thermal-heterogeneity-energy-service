"""Evaluate independently sampled occupants on the building, floor and zone operation libraries."""

import numpy as np
import pandas as pd

from common import MATRIX, RAW_RUN, ROOT
from population_command_sensitivity import scores, select
from service_distribution import ServiceDistribution
from service_threshold_sensitivity import SCORE_COLUMNS, Trajectories


def main():
    results = ROOT / "results"
    primary = pd.read_csv(results / "service_threshold_sensitivity/PAIRED_OPERATION_SERVICE_SCORES.csv")
    library = pd.read_csv(results / "budget_frontiers/ALL_OPERATION_LIBRARY.csv", low_memory=False)
    paths = library[["hourly_sha256", "hourly_path"]].drop_duplicates("hourly_sha256")
    primary = primary.merge(paths, on="hourly_sha256", validate="many_to_one", how="left")
    if primary.hourly_path.isna().any():
        raise ValueError("Every operation must have a traceable hourly trajectory")
    sampled = pd.read_csv(results / "population_command_sensitivity/SAMPLED_COMPOSITIONS.csv.gz")
    rosters = {name: group.sort_values(["zone_id", "profile_id"]).reset_index(drop=True)
               for name, group in sampled.groupby("composition", sort=True)}
    template = next(iter(rosters.values()))
    if any(len(group) != 267 or not group.zone_id.equals(template.zone_id) for group in rosters.values()):
        raise ValueError("All compositions must occupy the same zone slots")
    centers = np.stack([group.centre_c for group in rosters.values()])
    zone_means = centers.copy()
    for zone in template.zone_id.unique():
        ix = template.zone_id.eq(zone).to_numpy()
        zone_means[:, ix] = centers[:, ix].mean(axis=1, keepdims=True)
    probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    anchors = probes[probes.target_c.eq(25)].set_index("climate", verify_integrity=True)
    members = pd.read_csv(results / "base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    store = Trajectories(RAW_RUN)
    output = ROOT / "results/roster_transfer"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for climate, local in primary.groupby("climate", sort=True):
        anchor = anchors.loc[climate]
        ref = store.load(anchor.hourly_path, anchor.hourly_sha256)
        central = ServiceDistribution(ref, template)
        individual = central.individual(centers)
        tails = []
        for i, (name, group) in enumerate(rosters.items()):
            if group.sample_kind.iloc[0] == "original":
                ids = members.loc[members.climate.eq(climate) & members.composition.eq(name), "profile_id"]
                tail = np.flatnonzero(group.profile_id.isin(ids))
            else:
                tail = np.lexsort((group.profile_id.to_numpy(str), -individual[i]))[:27]
            if len(tail) != 27:
                raise ValueError("The reference group must contain 27 unique profiles")
            tails.append(tail)
        tails = np.stack(tails)
        limits = np.column_stack([individual @ central.weights,
                                  np.take_along_axis(individual, tails, axis=1).mean(axis=1)])
        cached = {}
        for operation in local.drop_duplicates("hourly_sha256").itertuples():
            data = store.load(operation.hourly_path, operation.hourly_sha256)
            np.testing.assert_array_equal(data["zone_ids"], ref["zone_ids"])
            np.testing.assert_array_equal(data["occupied"], ref["occupied"])
            np.testing.assert_allclose(data["people"], ref["people"], rtol=0, atol=1e-9)
            calculated_energy = data["facility_electric_kwh"].sum() + np.sum(data["facility_gas_hhv_kwh_annual"])
            np.testing.assert_allclose(calculated_energy, operation.facility_purchased_kwh, rtol=0, atol=1e-6)
            cached[operation.hourly_sha256] = scores(ServiceDistribution(data, template), centers, zone_means, tails)
        for menu, candidates in local.groupby("composition", sort=True):
            candidates = candidates.sort_values("run_id").reset_index(drop=True)
            values = np.stack([cached[h] for h in candidates.hourly_sha256], axis=1)
            original_ix = list(rosters).index(menu)
            np.testing.assert_allclose(values[original_ix], candidates[SCORE_COLUMNS], rtol=0, atol=1e-8)
            for i, (name, group) in enumerate(rosters.items()):
                source = group.source.iloc[0]
                if group.sample_kind.iloc[0] != "new" or source != rosters[menu].source.iloc[0]:
                    continue
                matrix = values[i]
                full = (matrix[:, :2] <= limits[i] + 1e-9).all(axis=1)
                mean = (matrix[:, 2:] <= limits[i] + 1e-9).all(axis=1)
                row = dict(climate=climate, source=source, draw=group.draw.iloc[0], menu=menu,
                           individual_feasible=full.any(), zone_average_feasible=mean.any(),
                           qualifying_sets_differ=bool(np.any(full != mean)))
                if full.any() and mean.any():
                    row.update(select(candidates, matrix, limits[i]))
                rows.append(row)
        print(climate, "completed", flush=True)
    comparisons = pd.DataFrame(rows)
    if len(comparisons) != 1620 or comparisons.duplicated(["climate", "source", "draw", "menu"]).any():
        raise ValueError("Expected three same-source menus for each of 540 weather–roster combinations")
    comparisons.to_csv(output / "ROSTER_MENU_SELECTIONS.csv", index=False)
    matched = comparisons.individual_feasible & comparisons.zone_average_feasible
    comparisons["paired"] = matched
    comparisons["changed"] = comparisons.choices_differ.eq(True)
    summary = comparisons.groupby(["climate", "source", "menu"], as_index=False).agg(
        rosters=("draw", "size"), paired_feasible=("paired", "sum"), different_choices=("changed", "sum"),
        different_sets=("qualifying_sets_differ", "sum"))
    summary.to_csv(output / "ROSTER_MENU_SUMMARY.csv", index=False)


if __name__ == "__main__":
    main()
