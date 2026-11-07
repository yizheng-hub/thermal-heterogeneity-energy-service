from common import *
import pandas as pd
import numpy as np
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles, load_hourly


def main():
    out = ROOT / "results/spatial_bounds"
    out.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    constrained = pd.read_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv")
    profiles = normalise_profiles(pd.concat([original, constrained], ignore_index=True))
    anchors = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    anchors = anchors[anchors.target_c.eq(25)].set_index("climate")
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    rows = []
    zones = []
    for (climate, composition, layout), g in profiles.groupby(
        ["climate", "composition", "layout"], sort=True
    ):
        g = g.sort_values("profile_id").reset_index(drop=True)
        curve = ServiceDistribution(load_hourly(Path(anchors.loc[climate, "hourly_path"])), g)
        tailids = set(
            members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id
        )
        is_tail = g.profile_id.isin(tailids).to_numpy()
        assert is_tail.sum() == 27
        pop_total = 0.0
        tail_min = 0.0
        tail_at_pop = 0.0
        interval_total = 0.0
        for zone, local in g.groupby("zone_id", sort=True):
            ix = local.index.to_numpy()
            center = local.centre_c.to_numpy()
            median = float(np.median(center))
            mismatch = abs(center - median)
            weight = curve.person_hours[ix]
            pop_total += float(mismatch @ weight)
            local_tail = is_tail[ix]
            if local_tail.any():
                tail_center = center[local_tail]
                tail_min += float(abs(tail_center - np.median(tail_center)).sum())
                tail_at_pop += float(abs(tail_center - median).sum())
            endpoints = np.unique(np.r_[local.lower_c.to_numpy(), local.upper_c.to_numpy()])
            excess = np.maximum(
                local.lower_c.to_numpy()[None, :] - endpoints[:, None], 0
            ) + np.maximum(endpoints[:, None] - local.upper_c.to_numpy()[None, :], 0)
            interval_total += float((excess @ weight).min())
            zones.append(
                {
                    "climate": climate,
                    "composition": composition,
                    "layout": layout,
                    "zone": zone,
                    "profiles": len(local),
                    "center_mean_c": float(center.mean()),
                    "center_median_c": median,
                    "center_sd_c": float(center.std()),
                    "center_iqr_c": float(np.quantile(center, 0.75) - np.quantile(center, 0.25)),
                    "center_span_c": float(center.max() - center.min()),
                    "interval_width_median_c": float((local.upper_c - local.lower_c).median()),
                    "unweighted_zone_minimum_center_mismatch_c": float(mismatch.mean()),
                }
            )
        rows.append(
            {
                "climate": climate,
                "composition": composition,
                "layout": layout,
                "ideal_population_l1_lower_bound_c": pop_total / curve.person_hours.sum(),
                "ideal_fixed_tail_l1_separate_lower_bound_c": tail_min / 27,
                "fixed_tail_mismatch_at_population_optimal_shared_temperatures_c": tail_at_pop / 27,
                "ideal_interval_excess_lower_bound_c": interval_total / curve.person_hours.sum(),
                "interpretation": "Unconstrained ideal one-temperature-per-zone model; not guaranteed HVAC attainability. Population and tail minima are separate optima.",
            }
        )
    pd.DataFrame(rows).to_csv(out / "IDEAL_SHARED_TEMPERATURE_BOUNDS.csv", index=False)
    pd.DataFrame(zones).to_csv(out / "WITHIN_ZONE_DEMAND_DISTRIBUTIONS.csv", index=False)
    actual = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    placement = ROOT / "results/PLACEMENT_METRICS.csv"
    if placement.exists():
        actual = pd.concat([actual, pd.read_csv(placement)], ignore_index=True)
    comparison = actual.merge(
        pd.DataFrame(rows), on=["climate", "composition", "layout"], validate="many_to_one"
    )
    comparison["population_gap_above_ideal_c"] = (
        comparison.mean_reference_mismatch_c - comparison.ideal_population_l1_lower_bound_c
    )
    assert comparison.population_gap_above_ideal_c.ge(-1e-8).all()
    comparison.to_csv(out / "PHYSICAL_SERVICE_VS_IDEAL_BOUND.csv", index=False)
    record_status(
        "spatial_ideal_bounds",
        "completed",
        {
            "groups": len(rows),
            "zone_distributions": len(zones),
            "physical_scores_checked_above_bound": len(comparison),
            "scope": "Ideal independent zone temperatures; actual hardware results reported separately.",
        },
    )
    print("BOUNDS", len(rows), len(zones), flush=True)


if __name__ == "__main__":
    main()
