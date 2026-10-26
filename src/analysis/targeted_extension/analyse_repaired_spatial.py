"""Separate profile placement from the number of independently commanded spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyse_repaired_controls import choose_energy  # noqa: E402


METRICS = [
    "annual_site_kwh", "annual_operational_co2_kg",
    "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
    "modeled_interval_person_hour_coverage", "modeled_interval_mean_excess_c",
]
TOL = 1e-9


def paired_difference(
    frame: pd.DataFrame, keys: list[str], contrast: str, reference: str, alternative: str,
) -> pd.DataFrame:
    indexed = frame.set_index([*keys, contrast])
    rows = []
    groups = frame[keys].drop_duplicates().itertuples(index=False, name=None)
    for key in groups:
        reference_key = (*key, reference)
        alternative_key = (*key, alternative)
        if reference_key not in indexed.index or alternative_key not in indexed.index:
            continue
        ref = indexed.loc[reference_key]
        alt = indexed.loc[alternative_key]
        if isinstance(ref, pd.DataFrame) or isinstance(alt, pd.DataFrame):
            raise ValueError("A spatial contrast is not one-to-one")
        row = dict(zip(keys, key, strict=True)) | {
            "reference": reference,
            "alternative": alternative,
            "reference_run_id": ref.run_id,
            "alternative_run_id": alt.run_id,
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = float(alt[metric] - ref[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.analysis_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    direct = pd.read_csv(source / "REPAIRED_ANNUAL_DIRECT.csv")
    direct = direct[direct.selection_status.eq("selected")].copy()
    candidate = pd.read_csv(source / "REPAIRED_ANNUAL_CANDIDATES.csv")
    anchors = pd.read_csv(source / "COMMON25_ANCHORS.csv")
    anchor_map = anchors.set_index(["context", "composition_id", "layout"])

    permission_keys = ["data_source", "context", "composition_id", "layout", "control"]
    permission = pd.concat([
        paired_difference(
            direct, permission_keys, "permission", reference, alternative
        ).assign(permission_step=step)
        for reference, alternative, step in (
            ("building", "floor", "building_to_floor"),
            ("floor", "native_zone", "floor_to_native_zone"),
            ("building", "native_zone", "building_to_native_zone"),
        )
    ], ignore_index=True)
    permission.to_csv(output / "DIRECT_PERMISSION_EFFECTS.csv", index=False)

    layout_keys = ["data_source", "context", "composition_id", "permission", "control"]
    placement = paired_difference(
        direct, layout_keys, "layout", "random_mixing", "demand_ordered"
    )
    placement.to_csv(output / "DIRECT_PLACEMENT_EFFECTS.csv", index=False)

    nested_rows = []
    group_keys = ["data_source", "context", "composition_id", "layout", "control"]
    for key, group in candidate.groupby(group_keys, sort=True):
        anchor = anchor_map.loc[(key[1], key[2], key[3])]
        for fraction in (1.0, 0.75, 0.5):
            qualified = group[
                group.engineering_qualified
                & group.mean_reference_mismatch_c.le(
                    fraction * anchor.anchor_mean_c + TOL
                )
                & group.fixed_tail_reference_mismatch_c.le(
                    fraction * anchor.anchor_fixed_tail_c + TOL
                )
            ]
            building = choose_energy(qualified[qualified.permission.eq("building")])
            nested_floor = choose_energy(
                qualified[qualified.permission.isin({"building", "floor"})]
            )
            nested = choose_energy(
                qualified[qualified.permission.isin({"building", "floor", "native_zone"})]
            )
            row = dict(zip(group_keys, key, strict=True)) | {
                "service_fraction": fraction,
                "building_status": "selected" if building else "unavailable",
                "nested_zone_status": "selected" if nested else "unavailable",
                "building_run_id": building["run_id"] if building else "",
                "nested_floor_status": "selected" if nested_floor else "unavailable",
                "nested_floor_run_id": nested_floor["run_id"] if nested_floor else "",
                "nested_zone_run_id": nested["run_id"] if nested else "",
                "building_energy_kwh": building["annual_site_kwh"] if building else np.nan,
                "nested_floor_energy_kwh": (
                    nested_floor["annual_site_kwh"] if nested_floor else np.nan
                ),
                "nested_zone_energy_kwh": nested["annual_site_kwh"] if nested else np.nan,
                "nested_floor_energy_change_kwh": (
                    nested_floor["annual_site_kwh"] - building["annual_site_kwh"]
                    if building and nested_floor else np.nan
                ),
                "nested_zone_energy_change_kwh": (
                    nested["annual_site_kwh"] - building["annual_site_kwh"]
                    if building and nested else np.nan
                ),
                "nested_selected_building_action": bool(
                    nested and nested["permission"] == "building"
                ),
                "nested_floor_selected_building_action": bool(
                    nested_floor and nested_floor["permission"] == "building"
                ),
            }
            nested_rows.append(row)
    nested_table = pd.DataFrame(nested_rows)
    nested_table.to_csv(output / "NESTED_PERMISSION_SELECTIONS.csv", index=False)
    violation = nested_table[
        nested_table.building_status.eq("selected")
        & nested_table.nested_zone_status.eq("selected")
        & nested_table.nested_zone_energy_change_kwh.gt(TOL)
    ]
    floor_violation = nested_table[
        nested_table.building_status.eq("selected")
        & nested_table.nested_floor_status.eq("selected")
        & nested_table.nested_floor_energy_change_kwh.gt(TOL)
    ]
    zone_step_violation = nested_table[
        nested_table.nested_floor_status.eq("selected")
        & nested_table.nested_zone_status.eq("selected")
        & nested_table.nested_zone_energy_kwh.gt(
            nested_table.nested_floor_energy_kwh + TOL
        )
    ]
    if not violation.empty or not floor_violation.empty or not zone_step_violation.empty:
        raise ValueError("A nested permission library increased minimum energy")
    report = {
        "permission_pairs": len(permission),
        "placement_pairs": len(placement),
        "nested_service_selections": len(nested_table),
        "nested_permission_energy_monotonicity_violations": len(violation),
        "nested_floor_energy_monotonicity_violations": len(floor_violation),
        "nested_zone_step_energy_monotonicity_violations": len(zone_step_violation),
        "candidate_support_is_nested": True,
    }
    (output / "REPAIRED_SPATIAL_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
