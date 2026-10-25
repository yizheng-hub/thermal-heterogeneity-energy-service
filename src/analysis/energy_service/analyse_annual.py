"""Compare actual controls and spatial permissions on frozen hourly inputs.

This analysis starts at qualified physical trajectories. It never substitutes
an average-demand review for an actual controller or changes building physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from room_interval_feasibility import room_interval_feasibility
from thermal_service import score_service


KEYS = ["context", "composition_id", "layout", "permission", "control"]
PROFILE_KEYS = KEYS[:3]
QUALIFIED = {"cached_qualified", "new_qualified"}
FINAL = QUALIFIED | {"cached_rejected", "new_rejected", "execution_failed"}
BUDGETS = (0.50, 0.75, 0.90, 1.00)
SERVICE_TOL = 1e-9
PAIR_METRICS = [
    "annual_electric_kwh", "annual_gas_hhv_kwh", "annual_site_kwh",
    "annual_operational_co2_kg", "annual_heating_electric_kwh",
    "annual_cooling_electric_kwh", "annual_fans_electric_kwh",
    "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
    "profile_reference_mismatch_p90_c", "modeled_interval_person_hour_coverage",
    "modeled_interval_mean_excess_c", "modeled_interval_excess_native_person_c_hours",
    "profile_coverage_p10", "profile_coverage_minimum",
    "profile_equal_reference_mismatch_c", "profile_equal_modeled_interval_coverage",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_hourly(path):
    with np.load(path, allow_pickle=False) as stored:
        data = dict(stored)
    if data["temperatures"].shape != data["people"].shape:
        raise ValueError(f"Temperature and native occupancy shapes differ: {path}")
    if data["temperatures"].shape[0] != 8760:
        raise ValueError(f"Expected a complete 8760-hour physical run: {path}")
    for key in ("hvac_electric_kwh", "gas_hhv_kwh"):
        if data[key].shape != (8760,) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid purchased carrier {key}: {path}")
    return data


def resolve_hourly(root, row):
    path = Path(row.path_hourly)
    return path if path.is_absolute() else root / path


def score(data, profiles, tail):
    positions = {str(z).strip().upper(): i for i, z in enumerate(data["zone_ids"])}
    zone = np.array([positions[str(z).strip().upper()] for z in profiles.zone_id])
    # Preserve the locked native source service window. Native counts outside
    # that window can be nonzero and must not silently change the denominator.
    weights = data["people"] * data["occupied"].astype(bool)[:, None]
    return score_service(
        data["temperatures"], weights, zone,
        profiles.centre_c.to_numpy(), profiles.lower_c.to_numpy(),
        profiles.upper_c.to_numpy(), tail,
    )


def choose_energy(candidates):
    best = None
    for row in candidates.sort_values(["run_id", "variant"]).to_dict("records"):
        if best is None:
            best = row
            continue
        a, b = row["annual_site_kwh"], best["annual_site_kwh"]
        tolerance = max(1e-6, 1e-9 * max(abs(a), abs(b)))
        if a < b - tolerance:
            best = row
    return best


def pair_outputs(table, reference_axis, references, output, comparison):
    key = [k for k in KEYS if k != reference_axis] + ["comparison_mode", "service_fraction"]
    baseline = table[table[reference_axis].eq(references[0])]
    index = {tuple(row[k] for k in key): row for row in baseline.to_dict("records")}
    rows = []
    for row in table[table[reference_axis].isin(references[1:])].to_dict("records"):
        ref = index.get(tuple(row[k] for k in key))
        both = ref is not None and row["selection_status"] == ref["selection_status"] == "selected"
        item = {k: row[k] for k in KEYS + ["comparison_mode", "service_fraction"]}
        item.update(comparison_axis=reference_axis, reference=references[0],
                    alternative=row[reference_axis], status="matched" if both else "unmatched",
                    alternative_status=row["selection_status"],
                    reference_status=ref["selection_status"] if ref else "missing_design",
                    alternative_run_id=row.get("run_id", ""),
                    reference_run_id=ref.get("run_id", "") if ref else "")
        item["comparison_id"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:24]
        for metric in PAIR_METRICS:
            item["delta_" + metric] = row[metric] - ref[metric] if both else np.nan
        if both:
            item["delta_site_pct"] = 100 * item["delta_annual_site_kwh"] / ref["annual_site_kwh"]
            item["delta_interval_coverage_percentage_points"] = 100 * item["delta_modeled_interval_person_hour_coverage"]
        else:
            item["delta_site_pct"] = np.nan
            item["delta_interval_coverage_percentage_points"] = np.nan
        if comparison == "control":
            item.update(climate_context=row["context"], scope=row["permission"],
                        delta_electric_kwh=item["delta_annual_electric_kwh"],
                        delta_gas_hhv_kwh=item["delta_annual_gas_hhv_kwh"])
        rows.append(item)
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def add_hourly_reallocation(path, designs, root):
    """Decompose paired yearly differences into added and released hourly loads.

    These are whole-building temporal differences, never metered energy
    assigned to individual people. They include all hours, as does annual use.
    """
    locations = {row.run_id: resolve_hourly(root, row) for row in designs.itertuples(index=False)
                 if row.engineering_status in QUALIFIED}
    curves = {}

    def carriers(run_id):
        if run_id not in curves:
            with np.load(locations[run_id], allow_pickle=False) as stored:
                curves[run_id] = (stored["hvac_electric_kwh"], stored["gas_hhv_kwh"])
        return curves[run_id]

    rows = pd.read_csv(path)
    values = []
    for row in rows.itertuples(index=False):
        item = {}
        for name in ("electric", "gas_hhv", "purchased_site"):
            for direction in ("added", "released"):
                item[f"hourly_{direction}_{name}_kwh"] = np.nan
        if row.status == "matched":
            alt_e, alt_g = carriers(row.alternative_run_id)
            ref_e, ref_g = carriers(row.reference_run_id)
            for name, delta in (("electric", alt_e-ref_e), ("gas_hhv", alt_g-ref_g),
                                ("purchased_site", alt_e+alt_g-ref_e-ref_g)):
                item[f"hourly_added_{name}_kwh"] = float(np.maximum(delta, 0).sum())
                item[f"hourly_released_{name}_kwh"] = float(np.maximum(-delta, 0).sum())
            net = item["hourly_added_purchased_site_kwh"]-item["hourly_released_purchased_site_kwh"]
            if not np.isclose(net, row.delta_annual_site_kwh, rtol=1e-9, atol=1e-6):
                raise ValueError("Added minus released hourly energy does not close to annual difference")
        values.append(item)
    pd.concat([rows, pd.DataFrame(values)], axis=1).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-root", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-designs", type=int, default=1296)
    parser.add_argument("--expected-profile-groups", type=int, default=36)
    args = parser.parse_args()
    root, out = args.engineering_root.resolve(), args.output_root
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "ANNUAL_RUN_MANIFEST.csv"
    profiles_path = root / "PROFILE_ASSIGNMENTS.csv.gz"
    designs = pd.read_csv(manifest_path, keep_default_na=False)
    profiles = pd.read_csv(profiles_path)
    factors = pd.read_csv(args.factors).set_index("city")
    if len(designs) != args.expected_designs or designs.duplicated(KEYS + ["variant"]).any():
        raise ValueError(f"The declared {args.expected_designs} logical control designs are incomplete or duplicated")
    if not set(designs.engineering_status).issubset(FINAL):
        raise ValueError("Annual engineering collection still contains unfinished designs")
    groups = {}
    for key, group in profiles.groupby(PROFILE_KEYS, sort=True):
        group = group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        if len(group) != 267 or group.profile_id.nunique() != 267:
            raise ValueError(f"Expected 267 unique assigned profiles in {key}")
        groups[key] = group
    if len(groups) != args.expected_profile_groups:
        raise ValueError(f"Expected {args.expected_profile_groups} declared profile/context/placement groups")
    hashes = {}
    cache = {}

    def trajectory(row):
        path = resolve_hourly(root, row)
        if str(path) not in hashes:
            hashes[str(path)] = digest(path)
        return load_hourly(path)

    anchors, tails, static = [], [], []
    for key, group in groups.items():
        selected = designs[(designs[PROFILE_KEYS] == pd.Series(key, index=PROFILE_KEYS)).all(axis=1)]
        fallback = selected[selected.variant.eq(3)]
        if fallback.run_id.nunique() != 1 or not fallback.engineering_status.isin(QUALIFIED).all():
            raise ValueError(f"Common25 fallback not identical and qualified: {key}")
        data = trajectory(next(fallback.itertuples(index=False)))
        first = score(data, group, np.arange(267))
        distances = first["profiles"]["reference_mismatch_c"]
        ranking = pd.DataFrame({"profile_id": group.profile_id, "distance": distances}).sort_values(
            ["distance", "profile_id"], ascending=[False, True], kind="mergesort"
        )
        fixed = ranking.head(27).profile_id.tolist()
        tail = np.flatnonzero(group.profile_id.isin(fixed))
        anchor = dict(zip(PROFILE_KEYS, key)) | {
            "anchor_run_id": fallback.run_id.iloc[0],
            "anchor_mean_c": first["summary"]["mean_reference_mismatch_c"],
            "anchor_fixed_tail_c": float(distances[tail].mean()),
            "fixed_tail_profile_ids": "|".join(fixed),
        }
        anchors.append(anchor)
        for position in tail:
            tails.append(dict(zip(PROFILE_KEYS, key)) | {
                "profile_id": group.profile_id.iloc[position],
                "anchor_profile_mismatch_c": float(distances[position]),
            })
        for row in room_interval_feasibility(group.zone_id, group.lower_c, group.upper_c):
            static.append(dict(zip(PROFILE_KEYS, key)) | row)
        cache[key] = {"profiles": group, "tail": tail, "anchor": anchor}
    pd.DataFrame(anchors).to_csv(out / "ANNUAL_ANCHORS.csv", index=False)
    pd.DataFrame(tails).to_csv(out / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    pd.DataFrame(static).to_csv(out / "STATIC_ZONE_INTERVAL_COMPATIBILITY.csv", index=False)

    score_cache, candidates, per_profile = {}, [], []
    for d in designs.itertuples(index=False):
        row = {k: getattr(d, k) for k in KEYS + ["variant", "run_id", "engineering_status", "permission_alias"]}
        row["selection_status"] = "selected" if d.engineering_status in QUALIFIED else "engineering_rejected"
        key = tuple(row[k] for k in PROFILE_KEYS)
        if d.engineering_status in QUALIFIED:
            score_key = key + (d.run_id,)
            if score_key not in score_cache:
                data = trajectory(d)
                group, tail = cache[key]["profiles"], cache[key]["tail"]
                calculated = score(data, group, tail)
                values = dict(calculated["summary"])
                values["profile_reference_mismatch_p90_c"] = float(np.quantile(calculated["profiles"]["reference_mismatch_c"], .9))
                values["profile_equal_reference_mismatch_c"] = float(calculated["profiles"]["reference_mismatch_c"].mean())
                values["profile_equal_modeled_interval_coverage"] = float(calculated["profiles"]["modeled_interval_coverage"].mean())
                for name, array in [("electric", "hvac_electric_kwh"), ("gas_hhv", "gas_hhv_kwh"),
                                    ("heating_electric", "heating_electric_kwh"),
                                    ("cooling_electric", "cooling_electric_kwh"),
                                    ("fans_electric", "fans_electric_kwh")]:
                    values[f"annual_{name}_kwh"] = float(data[array].sum())
                values["annual_site_kwh"] = values["annual_electric_kwh"] + values["annual_gas_hhv_kwh"]
                factor = factors.loc[d.context]
                values["annual_operational_co2_kg"] = (
                    values["annual_electric_kwh"] * factor.electricity_co2_kg_kwh
                    + values["annual_gas_hhv_kwh"] * factor.gas_co2_kg_kwh_hhv
                )
                score_cache[score_key] = values
                for i, profile_id in enumerate(group.profile_id):
                    p = dict(zip(PROFILE_KEYS, key)) | {"run_id": d.run_id, "profile_id": profile_id}
                    p.update({name: value[i] for name, value in calculated["profiles"].items()})
                    per_profile.append(p)
            row.update(score_cache[score_key])
        candidates.append(row)
    all_candidates = pd.DataFrame(candidates)
    all_candidates.to_csv(out / "ANNUAL_ALL_CANDIDATES.csv", index=False)
    pd.DataFrame(per_profile).to_csv(out / "PROFILE_ANNUAL_SERVICE.csv.gz", index=False,
                                   compression={"method": "gzip", "mtime": 0})
    direct = all_candidates[all_candidates.variant.eq(0)].copy()
    direct["comparison_mode"], direct["service_fraction"] = "direct", -1.0
    direct.to_csv(out / "ANNUAL_DIRECT.csv", index=False)

    selections = []
    for key, group in all_candidates.groupby(KEYS, sort=True):
        base = cache[key[:3]]["anchor"]
        for fraction in BUDGETS:
            mean_limit, tail_limit = fraction * base["anchor_mean_c"], fraction * base["anchor_fixed_tail_c"]
            valid = group[group.engineering_status.isin(QUALIFIED)]
            pool = valid[(valid.mean_reference_mismatch_c <= mean_limit + SERVICE_TOL)
                         & (valid.fixed_tail_reference_mismatch_c <= tail_limit + SERVICE_TOL)]
            chosen = choose_energy(pool)
            item = dict(zip(KEYS, key)) | {
                "comparison_mode": "equal_service", "service_fraction": fraction,
                "mean_budget_c": mean_limit, "fixed_tail_budget_c": tail_limit,
                "engineering_qualified_candidates": int(len(valid)),
                "service_qualified_candidates": int(len(pool)),
                "selection_status": "selected" if chosen else "service_unavailable" if len(valid) else "engineering_unavailable",
            }
            if chosen:
                item.update(chosen)
            selections.append(item)
    selected = pd.DataFrame(selections)
    selected.to_csv(out / "ANNUAL_SERVICE_SELECTIONS.csv", index=False)
    selected[selected.service_fraction.eq(1.0)].to_csv(out / "DR_BASELINES.csv", index=False)
    both = pd.concat([direct, selected], ignore_index=True)
    pair_outputs(both, "control", ["uniform_average", "personal_center", "ltpref_interval"],
                 out / "CONTROL_COMPARISONS.csv", "control")
    add_hourly_reallocation(out / "CONTROL_COMPARISONS.csv", designs, root)
    pair_outputs(both, "permission", ["building", "floor", "native_zone"],
                 out / "SPATIAL_PAIRS.csv", "space")
    pair_outputs(both, "layout", ["random_mixing", "demand_ordered", "source_structured"],
                 out / "LAYOUT_PAIRS.csv", "layout")
    summary = {
        "status": "PASS",
        "logical_annual_candidates": len(all_candidates),
        "unique_physical_inputs": int(designs.run_id.nunique()),
        "engineering_state_counts": designs.engineering_status.value_counts().to_dict(),
        "direct_control_rows": len(direct), "common_service_selection_rows": len(selected),
        "selected_by_fraction": selected.groupby(["service_fraction", "selection_status"]).size().rename("count").reset_index().to_dict("records"),
        "scored_profile_trajectory_pairs": len(score_cache),
        "all_profiles_within_modeled_intervals": int(all_candidates.all_profiles_without_interval_violation.fillna(False).sum()),
        "physical_parameters_changed": False,
        "service_interpretation": "Reference-centre mismatch and model-derived interval coverage, not measured satisfaction",
        "source_sha256": {"manifest": digest(manifest_path), "profiles": digest(profiles_path), "factors": digest(args.factors)},
        "hourly_input_sha256": hashes,
        "output_sha256": {path.name: digest(path) for path in sorted(out.glob("*.csv*"))},
    }
    (out / "ANALYSIS_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
