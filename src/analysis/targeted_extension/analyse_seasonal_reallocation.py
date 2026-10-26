"""Resolve annual controller comparisons into seasonal energy and service changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyse_repaired_controls import load_hourly, resolve_hourly_path, score  # noqa: E402


PROFILE_KEYS = ["context", "composition_id", "layout"]
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
TOL = 1e-9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--anchor-annual-manifest", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-source", required=True)
    args = parser.parse_args()

    analysis = args.analysis_dir.resolve()
    run_manifest_root = args.run_manifest.resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = pd.read_csv(args.profiles.resolve())
    factors = pd.read_csv(args.factors.resolve()).set_index("city")
    tails = pd.read_csv(analysis / "FIXED_TAIL_MEMBERSHIPS.csv")
    direct = pd.read_csv(analysis / "REPAIRED_ANNUAL_DIRECT.csv")
    selected = pd.read_csv(analysis / "REPAIRED_ANNUAL_SERVICE_SELECTIONS.csv")
    selected = selected[selected.selection_status.eq("selected")]
    choices = pd.concat(
        [direct[direct.selection_status.eq("selected")], selected], ignore_index=True
    )
    choices["choice_order_internal"] = np.arange(len(choices))
    choices = choices.sort_values("run_id", kind="mergesort")

    anchor_manifest = pd.read_csv(args.anchor_annual_manifest.resolve(), keep_default_na=False)
    anchor_paths = {}
    for context, group in anchor_manifest[anchor_manifest.variant.eq(3)].groupby("context"):
        qualified = group[group.engineering_status.isin({"cached_qualified", "new_qualified"})]
        if qualified.run_id.nunique() != 1:
            raise ValueError(f"Expected one common-25 anchor for {context}")
        anchor_paths[context] = (
            args.anchor_annual_manifest.resolve().parent / Path(qualified.iloc[0].path_hourly)
        ).resolve()

    profile_groups = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(PROFILE_KEYS, sort=True)
    }
    tail_groups = {
        key: set(group.profile_id)
        for key, group in tails.groupby(PROFILE_KEYS, sort=True)
    }
    anchors = {context: load_hourly(path) for context, path in anchor_paths.items()}
    active_path: Path | None = None
    active_data: dict[str, np.ndarray] | None = None

    rows = []
    for choice in choices.itertuples(index=False):
        key = (choice.context, choice.composition_id, choice.layout)
        group = profile_groups[key]
        tail = np.flatnonzero(group.profile_id.isin(tail_groups[key]))
        path = resolve_hourly_path(choice.hourly_path, run_manifest_root).resolve()
        if path != active_path:
            active_path = path
            active_data = load_hourly(path)
        if active_data is None:
            raise RuntimeError("Seasonal trajectory was not loaded")
        data = active_data
        anchor = anchors[str(choice.context)]
        factor = factors.loc[str(choice.context)]
        for season_order, (season, months) in enumerate(SEASONS.items()):
            indices = np.flatnonzero(np.isin(anchor["month"], months))
            seasonal_data = {
                name: value[indices] if isinstance(value, np.ndarray) and value.shape[:1] == (8760,)
                else value
                for name, value in data.items()
            }
            anchor_data = {
                name: value[indices] if isinstance(value, np.ndarray) and value.shape[:1] == (8760,)
                else value
                for name, value in anchor.items()
            }
            result = score(seasonal_data, group, tail)["summary"]
            anchor_score = score(anchor_data, group, tail)["summary"]
            electric = float(data["hvac_electric_kwh"][indices].sum())
            gas = float(data["gas_hhv_kwh"][indices].sum())
            fraction = float(choice.service_fraction)
            service_qualified = True
            if choice.comparison_mode == "equal_service":
                service_qualified = bool(
                    result["mean_reference_mismatch_c"]
                    <= fraction * anchor_score["mean_reference_mismatch_c"] + TOL
                    and result["fixed_tail_reference_mismatch_c"]
                    <= fraction * anchor_score["fixed_tail_reference_mismatch_c"] + TOL
                )
            rows.append({
                "data_source": args.data_source,
                "context": choice.context,
                "composition_id": choice.composition_id,
                "layout": choice.layout,
                "permission": choice.permission,
                "control": choice.control,
                "comparison_mode": choice.comparison_mode,
                "service_fraction": fraction,
                "run_id": choice.run_id,
                "choice_order_internal": choice.choice_order_internal,
                "_season_order": season_order,
                "season": season,
                "site_kwh": electric + gas,
                "hvac_electric_kwh": electric,
                "gas_hhv_kwh": gas,
                "operational_co2_kg": (
                    electric * factor.electricity_co2_kg_kwh
                    + gas * factor.gas_co2_kg_kwh_hhv
                ),
                "mean_reference_mismatch_c": result["mean_reference_mismatch_c"],
                "fixed_tail_reference_mismatch_c": result["fixed_tail_reference_mismatch_c"],
                "seasonal_service_qualified": service_qualified,
            })
    totals = pd.DataFrame(rows)
    totals = totals.sort_values(
        ["choice_order_internal", "_season_order"], kind="mergesort"
    ).drop(columns=["choice_order_internal", "_season_order"])
    totals.to_csv(output / "SEASONAL_CONTROL_TOTALS.csv", index=False)

    identities = [
        "data_source", *PROFILE_KEYS, "permission", "comparison_mode",
        "service_fraction", "season",
    ]
    reference = totals[totals.control.eq("uniform_average")].set_index(identities)
    differences = []
    for row in totals[totals.control.ne("uniform_average")].to_dict("records"):
        key = tuple(row[name] for name in identities)
        if key not in reference.index:
            continue
        ref = reference.loc[key]
        differences.append({name: row[name] for name in identities} | {
            "control": row["control"],
            "delta_site_kwh": row["site_kwh"] - ref.site_kwh,
            "delta_hvac_electric_kwh": row["hvac_electric_kwh"] - ref.hvac_electric_kwh,
            "delta_gas_hhv_kwh": row["gas_hhv_kwh"] - ref.gas_hhv_kwh,
            "delta_operational_co2_kg": row["operational_co2_kg"] - ref.operational_co2_kg,
            "alternative_seasonal_service_qualified": row["seasonal_service_qualified"],
            "uniform_seasonal_service_qualified": bool(ref.seasonal_service_qualified),
        })
    difference_table = pd.DataFrame(differences)
    difference_table.to_csv(output / "SEASONAL_CONTROL_DIFFERENCES.csv", index=False)
    report = {
        "data_source": args.data_source,
        "seasonal_comparisons": len(difference_table),
        "equal_service_seasonal_checks": int(
            difference_table.comparison_mode.eq("equal_service").sum()
        ),
        "alternative_equal_service_failures": int((
            difference_table.comparison_mode.eq("equal_service")
            & ~difference_table.alternative_seasonal_service_qualified
        ).sum()),
        "uniform_equal_service_failures": int((
            difference_table.comparison_mode.eq("equal_service")
            & ~difference_table.uniform_seasonal_service_qualified
        ).sum()),
    }
    (output / "SEASONAL_REALLOCATION_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
