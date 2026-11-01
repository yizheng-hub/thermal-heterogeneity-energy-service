"""Decompose direct control comparisons into HVAC end-use changes.

The analysis joins the immutable run identifiers in the direct-comparison table
to the annual run ledger.  It reports signed changes in terminal heating,
cooling, fans, space-heating gas and whole-building purchased energy.  No
EnergyPlus output is altered or reinterpreted as a different HVAC topology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


END_USES = {
    "terminal_heating_electric_kwh": "heating_electric_kwh",
    "cooling_electric_kwh": "cooling_electric_kwh",
    "fans_electric_kwh": "fans_electric_kwh",
    "space_heating_gas_kwh": "gas_hhv_kwh",
    "whole_building_purchased_kwh": "facility_purchased_kwh",
}


def analyse(direct: pd.DataFrame, runs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_direct = {
        "alternative_run_id", "reference_run_id", "alternative_control",
        "climate", "composition", "delta_facility_purchased_kwh",
    }
    required_runs = {"run_id", *END_USES.values()}
    if missing := required_direct - set(direct):
        raise ValueError(f"Direct comparisons are missing columns: {sorted(missing)}")
    if missing := required_runs - set(runs):
        raise ValueError(f"Run metrics are missing columns: {sorted(missing)}")

    if runs.run_id.duplicated().any():
        raise ValueError("Run identifiers must be unique")
    ledger = runs.set_index("run_id")
    rows: list[dict] = []
    for row in direct.itertuples(index=False):
        if row.alternative_run_id not in ledger.index or row.reference_run_id not in ledger.index:
            raise ValueError("A direct comparison refers to an unknown annual run")
        alternative = ledger.loc[row.alternative_run_id]
        reference = ledger.loc[row.reference_run_id]
        result = {
            "climate": row.climate,
            "composition": row.composition,
            "alternative_control": row.alternative_control,
            "alternative_run_id": row.alternative_run_id,
            "reference_run_id": row.reference_run_id,
        }
        for label, column in END_USES.items():
            result[f"delta_{label}"] = float(alternative[column] - reference[column])
        reported = float(row.delta_facility_purchased_kwh)
        calculated = result["delta_whole_building_purchased_kwh"]
        if not np.isclose(reported, calculated, atol=1e-6, rtol=0):
            raise ValueError("Joined whole-building energy change does not match the comparison ledger")
        rows.append(result)

    detail = pd.DataFrame(rows)
    increased = detail[detail.delta_whole_building_purchased_kwh > 0].copy()
    summary = {
        "status": "COMPLETE",
        "scope": "Direct Personal centre/range versus Uniform average comparisons in the fixed VAV office system",
        "comparisons": int(len(detail)),
        "whole_building_energy_increases": int(len(increased)),
        "whole_building_energy_decreases": int((detail.delta_whole_building_purchased_kwh < 0).sum()),
        "energy_increase_cases": {},
    }
    for label in END_USES:
        column = f"delta_{label}"
        values = increased[column]
        summary["energy_increase_cases"][label] = {
            "median_delta_kwh": float(values.median()),
            "positive_cases": int((values > 0).sum()),
            "negative_cases": int((values < 0).sum()),
            "correlation_with_whole_building_change": float(
                values.corr(increased.delta_whole_building_purchased_kwh)
            ),
        }
    return detail, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    detail, summary = analyse(pd.read_csv(args.direct), pd.read_csv(args.runs))
    args.output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output / "HVAC_END_USE_DIRECT_COMPARISONS.csv", index=False)
    (args.output / "HVAC_END_USE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
