"""Reassess July response actions while changing only the event-service allowance.

Annual baseline selection, baseline qualification and the 48-hour recovery
limits remain fixed.  The output is a discrete sensitivity curve over the
simulated actions; it does not interpolate untested control actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TOL = 1e-9
DEFAULT_ALLOWANCES_C = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)
DECISION_KEYS = [
    "data_source",
    "context",
    "composition_id",
    "layout",
    "permission",
    "control",
]


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    text = values.astype(str).str.lower()
    if not text.isin(("true", "false", "1", "0")).all():
        raise ValueError("Expected an explicit Boolean column")
    return text.isin(("true", "1"))


def prepare_actions(gates: pd.DataFrame) -> pd.DataFrame:
    """Return one row per positive July action with fixed non-event gates."""
    table = gates[
        gates.event_month.eq(7)
        & gates.service_fraction.eq(1)
        & gates.permission.isin(("building", "native_zone"))
    ].copy()
    if table.empty:
        raise ValueError("No July, fraction-1 event candidates were found")

    for column in (
        "engineering_qualified",
        "raw_positive",
        "baseline_qualified",
        "event_qualified",
        "recovery_qualified",
    ):
        table[column] = _boolean(table[column])

    table["recovery_perturbed_pass_fixed"] = (
        table.recovery_perturbed_mean_c.le(table.recovery_mean_budget_c + TOL)
        & table.recovery_perturbed_fixed_tail_c.le(
            table.recovery_fixed_tail_budget_c + TOL
        )
    )
    table["non_event_gates_pass"] = (
        table.engineering_qualified
        & table.raw_positive
        & table.baseline_qualified
        & table.recovery_perturbed_pass_fixed
    )
    table["event_mean_allowance_needed_c"] = np.maximum(
        table.event_perturbed_mean_c - table.event_mean_budget_c, 0.0
    )
    table["event_tail_allowance_needed_c"] = np.maximum(
        table.event_perturbed_fixed_tail_c - table.event_fixed_tail_budget_c, 0.0
    )
    table["minimum_event_allowance_c"] = table[
        ["event_mean_allowance_needed_c", "event_tail_allowance_needed_c"]
    ].max(axis=1)
    table.loc[~table.non_event_gates_pass, "minimum_event_allowance_c"] = np.inf

    table["fixed_exclusion"] = np.select(
        [
            ~table.engineering_qualified,
            ~table.raw_positive,
            ~table.baseline_qualified,
            ~table.recovery_perturbed_pass_fixed,
        ],
        ["engineering", "nonpositive_response", "baseline_service", "recovery_service"],
        default="event_allowance",
    )
    original = table.non_event_gates_pass & table.minimum_event_allowance_c.le(TOL)
    if not original.eq(table.recovery_qualified).all():
        raise ValueError("Zero allowance does not reproduce retained recovery qualification")
    return table


def calculate_frontier(
    actions: pd.DataFrame, allowances_c: tuple[float, ...] = DEFAULT_ALLOWANCES_C
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the largest tested response for every decision and allowance."""
    allowances = tuple(sorted(set(float(value) for value in allowances_c)))
    if not allowances or allowances[0] < 0 or 0.0 not in allowances:
        raise ValueError("Allowances must be nonnegative and include zero")

    selections: list[dict] = []
    for key, group in actions.groupby(DECISION_KEYS, sort=True, dropna=False):
        for allowance in allowances:
            eligible = group[
                group.non_event_gates_pass
                & group.minimum_event_allowance_c.le(allowance + TOL)
            ]
            row = dict(zip(DECISION_KEYS, key, strict=True)) | {
                "event_allowance_c": allowance,
                "tested_actions": len(group),
                "qualified_actions": len(eligible),
                "selected": not eligible.empty,
                "selected_response_kw": 0.0,
                "selected_run_id": "",
                "selected_amplitude_c": np.nan,
            }
            if not eligible.empty:
                maximum = eligible.event_average_reduction_kw.max()
                selected = eligible[
                    eligible.event_average_reduction_kw.ge(maximum - TOL)
                ].sort_values(["run_id", "amplitude_c"], kind="mergesort").iloc[0]
                row.update(
                    selected_response_kw=float(selected.event_average_reduction_kw),
                    selected_run_id=selected.run_id,
                    selected_amplitude_c=float(selected.amplitude_c),
                )
            selections.append(row)
    decision_table = pd.DataFrame(selections)

    summaries = []
    group_keys = ["data_source", "context", "permission", "control", "event_allowance_c"]
    for key, group in decision_table.groupby(group_keys, sort=True, dropna=False):
        summaries.append(
            dict(zip(group_keys, key, strict=True))
            | {
                "controller_decisions": len(group),
                "decisions_with_response": int(group.selected.sum()),
                "diagnostic_sum_response_kw_not_stock_capacity": float(
                    group.selected_response_kw.sum()
                ),
                "response_kw_median": float(group.selected_response_kw.median()),
                "response_kw_p95": float(group.selected_response_kw.quantile(0.95)),
            }
        )
    summary = pd.DataFrame(summaries)

    ordered = decision_table.sort_values(DECISION_KEYS + ["event_allowance_c"])
    changes = ordered.groupby(DECISION_KEYS, dropna=False).selected_response_kw.diff().fillna(0)
    if changes.lt(-TOL).any():
        raise ValueError("A larger allowance reduced the selected response")
    return decision_table, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allowances", nargs="*", type=float, default=DEFAULT_ALLOWANCES_C)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    actions = prepare_actions(pd.read_csv(args.gates.resolve()))
    decisions, summary = calculate_frontier(actions, tuple(args.allowances))
    actions.replace({np.inf: np.nan}).to_csv(output / "DR_EVENT_ACTION_ALLOWANCE.csv", index=False)
    decisions.to_csv(output / "DR_EVENT_BUDGET_FRONTIER_DECISIONS.csv", index=False)
    summary.to_csv(output / "DR_EVENT_BUDGET_FRONTIER_SUMMARY.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
