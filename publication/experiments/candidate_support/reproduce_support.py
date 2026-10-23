"""Reselect frozen control candidates under three declared support rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier"]
METHODS = ["uniform", "personal_center", "ltpref_interval"]


def reproduce(input_root: Path, output_root: Path) -> None:
    candidates = pd.read_csv(input_root / "candidate_scores.csv")
    candidates["source"] = candidates.source.fillna("")
    anchors = pd.read_csv(input_root / "service_anchors.csv").set_index(KEYS)
    tariffs = pd.read_csv(input_root / "regional_factors.csv").set_index("city")
    light_package = 8 * 51.28 * (0.03 * 1.03**5 / (1.03**5 - 1)) + 4 * 51.28
    rows = []
    for key, group in candidates.groupby(KEYS, sort=True):
        anchor = anchors.loc[key]
        for support in ("primary", "equal_four_variants", "nested_uniform"):
            available = (
                group
                if support != "equal_four_variants"
                else group[group.source.ne("finite_uniform_grid")]
            )
            for budget in (0.5, 0.75, 0.9, 1.0):
                chosen = {}
                for method in METHODS:
                    options = available[available.strategy.eq(method)]
                    if support == "nested_uniform" and method != "uniform":
                        options = pd.concat(
                            [options, available[available.strategy.eq("uniform")]],
                            ignore_index=True,
                        )
                    options = options[options.engineering_pass]
                    ratio = np.maximum(
                        options["mean"] / anchor["mean"], options["tail"] / anchor["tail"]
                    )
                    feasible = options[ratio.le(budget + 1e-9)]
                    chosen[method] = (
                        None if feasible.empty else feasible.sort_values("site_kwh").iloc[0]
                    )
                for method in METHODS:
                    value, reference = chosen[method], chosen["uniform"]
                    factor = tariffs.loc[key[0]]
                    headroom = None
                    if value is not None and reference is not None and factor.currency == "USD":
                        headroom = -(
                            (value.electric_kwh - reference.electric_kwh)
                            * factor.electricity_price_currency_kwh
                            + (value.gas_kwh_fuel - reference.gas_kwh_fuel)
                            * factor.gas_price_currency_kwh_hhv
                        )
                    rows.append(
                        dict(zip(KEYS, key))
                        | {
                            "support": support,
                            "service_fraction": budget,
                            "strategy": method,
                            "chosen_site_kwh": None if value is None else float(value.site_kwh),
                            "chosen_electric_kwh": None
                            if value is None
                            else float(value.electric_kwh),
                            "chosen_gas_kwh_hhv": None
                            if value is None
                            else float(value.gas_kwh_fuel),
                            "change_pct": None
                            if value is None or reference is None
                            else 100 * (value.site_kwh / reference.site_kwh - 1),
                            "usd_energy_bill_headroom": headroom,
                        }
                    )
    selections = pd.DataFrame(rows)
    summaries = []
    for support, local in selections.groupby("support", sort=True):
        alternatives = local[local.strategy.ne("uniform")]
        matched = alternatives[alternatives.change_pct.notna()]
        priced = matched[matched.usd_energy_bill_headroom.notna()]
        coefficients = alternatives[alternatives.interval_width_multiplier.eq(1)].pivot(
            index=["city", "seed", "allocation", "service_fraction", "strategy"],
            columns="griffiths_coefficient",
            values="change_pct",
        )
        coefficients = coefficients[coefficients[0.5].notna()]
        directions = np.select(
            [
                coefficients.isna().any(axis=1),
                coefficients.lt(-1e-6).all(axis=1),
                coefficients.gt(1e-6).all(axis=1),
                coefficients.abs().le(1e-6).all(axis=1),
            ],
            ["match_lost", "stable_decrease", "stable_increase", "all_tied"],
            default="direction_unstable",
        )
        counts = pd.Series(directions).value_counts()
        summaries.append(
            {
                "candidate_support": support,
                "default_matched_comparison_definitions": len(coefficients),
                **{
                    name: int(counts.get(name, 0))
                    for name in (
                        "stable_increase",
                        "stable_decrease",
                        "all_tied",
                        "direction_unstable",
                        "match_lost",
                    )
                },
                "all_definition_matched_alternative_rows": len(matched),
                "all_definition_increase": int(matched.change_pct.gt(1e-8).sum()),
                "all_definition_decrease": int(matched.change_pct.lt(-1e-8).sum()),
                "all_definition_tie": int(matched.change_pct.abs().le(1e-8).sum()),
                "usd_matched_rows": len(priced),
                "usd_positive_bill_headroom": int(priced.usd_energy_bill_headroom.gt(1e-8).sum()),
                "usd_tied_bill_headroom": int(priced.usd_energy_bill_headroom.abs().le(1e-8).sum()),
                "usd_negative_bill_headroom": int(priced.usd_energy_bill_headroom.lt(-1e-8).sum()),
                "light_package_usd_per_year": light_package,
                "usd_positive_after_light_package": int(
                    priced.usd_energy_bill_headroom.gt(light_package).sum()
                ),
            }
        )
    nested = selections[
        selections.support.eq("nested_uniform")
        & selections.strategy.ne("uniform")
        & selections.change_pct.notna()
    ]
    if nested.change_pct.gt(1e-8).any():
        raise AssertionError("Nested minimum-energy selection exceeds its uniform option")
    output_root.mkdir(parents=True, exist_ok=True)
    selections.to_csv(output_root / "support_selections.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_root / "TableS18.csv", index=False)
    qa = {
        "status": "PASS",
        "new_physical_runs": 0,
        "frozen_candidates": len(candidates),
        "service_anchors": len(anchors),
        "selection_rows": len(selections),
        "nested_support_energy_order_verified": True,
        "summaries": summaries,
    }
    (output_root / "QA.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.input_root, args.output_root)
