"""Independently check fixed profile identities and signed energy accounting.

The accounting check reads actual reference and alternative hourly trajectories.
Added and avoided energy are reconstructed from annual differences and absolute
hourly differences, independently of the plotting ledger's positive-part sums.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_FIELDS = ["profile_id", "centre_c", "lower_c", "upper_c"]
PAIR_KEYS = ["context", "composition_id", "layout", "control", "data_source"]
LAYOUTS = {"random_mixing", "demand_ordered"}
COMPONENTS = {"heating_electric": ("heating_electric_kwh",),
              "cooling_electric": ("cooling_electric_kwh",),
              "fans_electric": ("fans_electric_kwh",),
              "gas_hhv": ("gas_hhv_kwh",)}


def verify_compositions(assignments, exported):
    canonical, source_ids = [], {}
    blocks = 0
    for source, frame in assignments.items():
        frame = frame[frame.layout.isin(LAYOUTS)]
        if frame.composition_id.nunique() != 3 or frame.context.nunique() != 4 or set(frame.layout) != LAYOUTS:
            raise ValueError("Each source must retain three draws, four contexts and two layouts")
        source_ids[source] = set(frame.composition_id)
        for composition, draw in frame.groupby("composition_id", sort=True):
            reference = None
            for _, group in draw.groupby(["context", "layout"], sort=True):
                if len(group) != 267 or group.profile_id.nunique() != 267:
                    raise ValueError("A fixed composition no longer contains its 267 distinct records")
                current = group[PROFILE_FIELDS].sort_values("profile_id").reset_index(drop=True)
                if reference is None:
                    reference = current
                elif not current.equals(reference):
                    raise ValueError("Profile identities or temperatures changed between placement or weather contexts")
                blocks += 1
            canonical.append(reference.assign(composition_id=composition, data_source=source))
    if source_ids["ASHRAE"] & source_ids["CTCD"]:
        raise ValueError("Composition IDs must distinguish all six declared draws")
    expected = pd.concat(canonical, ignore_index=True)
    keys = ["data_source", "composition_id", "profile_id"]
    columns = keys + ["centre_c", "lower_c", "upper_c"]
    left = expected[columns].sort_values(keys).reset_index(drop=True)
    right = exported[columns].sort_values(keys).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("The exported six-composition table changed a sampled record or temperature")
    width_error = float(np.max(np.abs(exported.range_width_c - (exported.upper_c - exported.lower_c))))
    if width_error > 1e-10:
        raise ValueError("Exported range widths differ from the retained bounds")
    return {"fixed_compositions": 6, "distinct_profile_rows_in_display": len(expected),
            "context_layout_blocks_verified": blocks, "width_max_error_c": width_error,
            "identities_and_temperatures_unchanged": True}


def signed_identity(reference, alternative):
    """Recover both signed totals from the net and the L1 hourly change."""
    net = float(np.sum(alternative) - np.sum(reference))
    absolute = float(np.linalg.norm(alternative - reference, ord=1))
    return {"net_kwh": net, "added_kwh": (absolute + net) / 2,
            "avoided_kwh": (absolute - net) / 2}


def verify_signed_energy(pairs, ledger, manifests, factors):
    locations = {}
    for manifest in manifests:
        frame = pd.read_csv(manifest, keep_default_na=False)
        for row in frame.itertuples(index=False):
            if row.path_hourly:
                path = Path(row.path_hourly)
                locations[row.run_id] = path if path.is_absolute() else manifest.parent / path

    @lru_cache(maxsize=32)
    def raw(run):
        with np.load(locations[run], allow_pickle=False) as stored:
            names = {field for fields in COMPONENTS.values() for field in fields}
            names.update(("hvac_electric_kwh", "pumps_electric_kwh"))
            result = {name: stored[name] for name in names}
        if any(value.shape != (8760,) or not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Signed accounting requires complete finite annual carrier arrays")
        return result

    direct = pairs[(pairs.comparison_mode == "direct") & (pairs.permission == "native_zone")
                   & (pairs.status == "matched")]
    if ledger.duplicated(PAIR_KEYS + ["component"]).any() or len(ledger) != 4 * len(direct):
        raise ValueError("Signed ledger must retain four components for every matched direct pair")
    if direct.empty or not np.isfinite(ledger[["added_kwh", "avoided_kwh", "net_kwh"]]).all().all():
        raise ValueError("A signed ledger must contain finite matched energy outcomes")
    indexed = ledger.set_index(PAIR_KEYS + ["component"])
    errors = {"signed_component_kwh": 0., "annual_site_kwh": 0., "annual_carbon_kg": 0.,
              "hourly_space_electric_submeter_kwh": 0., "hourly_excluded_pump_change_kwh": 0.,
              "component_net_sum_kwh": 0.}
    for row in direct.itertuples(index=False):
        reference, alternative = raw(row.reference_run_id), raw(row.alternative_run_id)
        site, carriers, signed_sum = {}, {}, 0.
        errors["hourly_excluded_pump_change_kwh"] = max(errors["hourly_excluded_pump_change_kwh"],
            float(np.max(np.abs(alternative["pumps_electric_kwh"] - reference["pumps_electric_kwh"]))))
        for label, data in (("reference", reference), ("alternative", alternative)):
            electric = data["heating_electric_kwh"] + data["cooling_electric_kwh"] + data["fans_electric_kwh"]
            errors["hourly_space_electric_submeter_kwh"] = max(errors["hourly_space_electric_submeter_kwh"],
                float(np.max(np.abs(electric - data["hvac_electric_kwh"]))))
            carriers[label] = (float(data["hvac_electric_kwh"].sum()), float(data["gas_hhv_kwh"].sum()))
            site[label] = sum(carriers[label])
        for component, fields in COMPONENTS.items():
            before = sum(reference[field] for field in fields)
            after = sum(alternative[field] for field in fields)
            independent = signed_identity(before, after)
            saved = indexed.loc[tuple(getattr(row, key) for key in PAIR_KEYS) + (component,)]
            errors["signed_component_kwh"] = max(errors["signed_component_kwh"], *(
                abs(saved[name] - value) for name, value in independent.items()))
            signed_sum += float(saved.net_kwh)
        delta = site["alternative"] - site["reference"]
        factor = factors.loc[row.context]
        carbon = ((carriers["alternative"][0] - carriers["reference"][0]) * factor.electricity_co2_kg_kwh
                  + (carriers["alternative"][1] - carriers["reference"][1]) * factor.gas_co2_kg_kwh_hhv)
        errors["annual_site_kwh"] = max(errors["annual_site_kwh"], abs(delta - row.delta_annual_site_kwh))
        errors["annual_carbon_kg"] = max(errors["annual_carbon_kg"], abs(carbon - row.delta_annual_operational_co2_kg))
        errors["component_net_sum_kwh"] = max(errors["component_net_sum_kwh"], abs(signed_sum - delta))
    if any(value > 1e-6 for value in errors.values()):
        raise ValueError(f"Independent signed-energy/carbon accounting failed: {errors}")
    return {"actual_reference_alternative_pairs": len(direct), "signed_component_rows": len(ledger),
            "annual_hours_per_trajectory": 8760, "absolute_identity_tolerance": 1e-6,
            "carrier_boundary": "Space heating, cooling and fans; separately reported DHW pumps are verified unchanged hour by hour",
            "maximum_absolute_errors": errors, "independent_identity": "added/avoided = (L1 hourly change +/- annual net)/2"}


def run(repository, evidence, output):
    main = repository / "data/inputs/energy_service"
    parallel = repository / "data/inputs/ctcd_parallel/annual/engineering"
    paths = {"ASHRAE": main / "engineering/PROFILE_ASSIGNMENTS.csv.gz",
             "CTCD": parallel / "PROFILE_ASSIGNMENTS.csv.gz"}
    tables = {label: pd.read_csv(path, float_precision="round_trip") for label, path in paths.items()}
    profiles = pd.read_csv(evidence / "DEMAND_COMPOSITION_PROFILES.csv", float_precision="round_trip")
    composition = verify_compositions(tables, profiles)
    pairs_path, ledger_path = evidence / "annual/CONTROL_COMPARISONS.csv", evidence / "SIGNED_ENDUSE_REALLOCATION.csv"
    energy = verify_signed_energy(pd.read_csv(pairs_path), pd.read_csv(ledger_path),
        (main / "engineering/ANNUAL_RUN_MANIFEST.csv", parallel / "ANNUAL_RUN_MANIFEST.csv"),
        pd.read_csv(main / "factors/regional_factors.csv").set_index("city"))
    paths.update(compositions=evidence / "DEMAND_COMPOSITION_PROFILES.csv", pairs=pairs_path, ledger=ledger_path)
    report = {"status": "PASS", "composition_identity": composition, "signed_accounting": energy,
              "source_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()},
              "new_simulations": 0, "source_is_not_a_main_comparison": True}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.repository_root.resolve(), args.evidence_root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
