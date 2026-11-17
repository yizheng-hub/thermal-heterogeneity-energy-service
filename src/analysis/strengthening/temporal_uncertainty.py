from common import *
import pandas as pd
import numpy as np
from service_distribution import ServiceDistribution
from paired_uncertainty import keyed_normal, choose, DRAWS
from profile_identity import person_keys
from analyse_global_results import normalise_profiles, load_hourly


def season_metrics(data, g, centers, tail):
    # Each quarter-specific latent demand is constant within its whole time block.
    total = np.zeros((DRAWS, len(g)))
    hours = np.zeros(len(g))
    mean_total = np.zeros_like(total)
    blocks = ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    for b, months in enumerate(blocks):
        curve = ServiceDistribution(data, g, np.isin(data["month"], months))
        local = centers[b]
        zmean = local.copy()
        for zone in g.zone_id.unique():
            ix = np.flatnonzero(g.zone_id.eq(zone))
            zmean[:, ix] = local[:, ix].mean(axis=1)[:, None]
        total += curve.individual(local) * curve.person_hours
        mean_total += curve.individual(zmean) * curve.person_hours
        hours += curve.person_hours
    person = total / hours
    zperson = mean_total / hours
    weights = hours / hours.sum()
    return (
        person @ weights,
        person[:, tail].mean(axis=1),
        zperson @ weights,
        zperson[:, tail].mean(axis=1),
    )


def main():
    out = ROOT / "results/paired_uncertainty_identity"
    out.mkdir(parents=True, exist_ok=True)
    p = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    metrics = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    ref = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    ref = ref[ref.target_c.eq(25)].set_index("climate")
    rows = []
    ops = []
    for (climate, composition), g in p.groupby(["climate", "composition"], sort=True):
        g = g.sort_values("profile_id").reset_index(drop=True)
        tailids = set(
            members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id
        )
        tail = np.flatnonzero(g.profile_id.isin(tailids))
        assert len(tail) == 27
        group = (
            metrics[metrics.climate.eq(climate) & metrics.composition.eq(composition)]
            .sort_values(["facility_purchased_kwh", "physical_plan_id"])
            .drop_duplicates("physical_plan_id")
            .reset_index(drop=True)
        )
        source = "ASHRAE" if "Multinational" in composition else "CTCD"
        person = np.column_stack(
            [keyed_normal("person|" + str(i)) for i in person_keys(g.profile_id)]
        )
        shared = keyed_normal("source|" + source)[:, None]
        centers = np.array(
            [
                g.centre_c.to_numpy()[None, :]
                + 0.25 * shared
                + 0.25 * person
                + 0.25 * keyed_normal("calendar_block|" + source + "|" + str(b))[:, None]
                for b in range(4)
            ]
        )
        rm, rt, _, _ = season_metrics(
            load_hourly(Path(ref.loc[climate, "hourly_path"])), g, centers, tail
        )
        scores = [
            season_metrics(load_hourly(Path(row.hourly_path)), g, centers, tail)
            for row in group.itertuples()
        ]
        full = np.column_stack([x[0] for x in scores])
        ft = np.column_stack([x[1] for x in scores])
        zone = np.column_stack([x[2] for x in scores])
        zt = np.column_stack([x[3] for x in scores])
        eligible = (full <= rm[:, None] + 1e-9) & (ft <= rt[:, None] + 1e-9)
        meaneligible = (zone <= rm[:, None] + 1e-9) & (zt <= rt[:, None] + 1e-9)
        ids = group.physical_plan_id.to_numpy()
        energy = group.facility_purchased_kwh.to_numpy()
        a = choose(eligible, energy, ids)
        b = choose(meaneligible, energy, ids)
        valid = (a >= 0) & (b >= 0)
        rows.append(
            {
                "climate": climate,
                "composition": composition,
                "scenario": "source0.25_person0.25_calendar_block0.25",
                "conditional_full_feasible_fraction": float((a >= 0).mean()),
                "conditional_zone_feasible_fraction": float((b >= 0).mean()),
                "conditional_choice_disagreement_fraction": float((a[valid] != b[valid]).mean())
                if valid.any()
                else None,
                "extra_selected_energy_median_kwh": float(
                    np.median(energy[a[valid]] - energy[b[valid]])
                )
                if valid.any()
                else None,
            }
        )
        for j, row in enumerate(group.itertuples()):
            excess = np.maximum(np.maximum(full[:, j] - rm, ft[:, j] - rt), 0)
            ops.append(
                {
                    "climate": climate,
                    "composition": composition,
                    "physical_plan_id": row.physical_plan_id,
                    "conditional_exceedance_probability": float((~eligible[:, j]).mean()),
                    "excess_median_c": float(np.median(excess)),
                    "excess_p95_c": float(np.quantile(excess, 0.95)),
                    "choice_probability": float((a == j).mean()),
                }
            )
        pd.DataFrame(rows).to_csv(out / "TEMPORAL_BLOCK_SELECTION_STABILITY.csv", index=False)
        pd.DataFrame(ops).to_csv(out / "TEMPORAL_BLOCK_OPERATION_EXCEEDANCE.csv", index=False)
        print("TEMPORAL", climate, composition, flush=True)
    write_json(
        out / "TEMPORAL_BLOCK_AUDIT.json",
        {
            "status": "completed",
            "draws": DRAWS,
            "settings": len(rows),
            "interpretation": "uncalibrated0.25C archive-wide,person-constant andarchive-specific quarter-block sensitivity terms; not independent hourly noise or measured adaptation",
            "pairing": "Same personal and calendar draws for every operation,climate and25C reference; original tail retained; no command regenerated",
        },
    )
    record_status(
        "temporal_block_uncertainty",
        "completed",
        ["TEMPORAL_BLOCK_SELECTION_STABILITY.csv", "TEMPORAL_BLOCK_OPERATION_EXCEEDANCE.csv"],
    )


if __name__ == "__main__":
    main()
