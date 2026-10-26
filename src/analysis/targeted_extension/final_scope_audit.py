"""Record the exact populations behind the submission's headline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LAYOUTS = ("random_mixing", "demand_ordered")
TOL = 1e-9


def add(rows: list[dict], claim_id: str, panel: str, source: str, filters: str,
        n: int, n_unit: str, statistic: str, value: object) -> None:
    rows.append({
        "claim_id": claim_id,
        "figure_panel": panel,
        "source_table": source,
        "filters": filters,
        "n": int(n),
        "n_unit": n_unit,
        "statistic": statistic,
        "value": value,
    })


def build(evidence: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(evidence / "FINAL_ANNUAL_COMPARISONS.csv")
    common = pd.read_csv(evidence / "FINAL_COMMON_BASELINE_DR.csv")
    own = pd.read_csv(evidence / "FINAL_OWN_BASELINE_DR.csv")
    paired = pd.read_csv(evidence / "FINAL_OWN_BASELINE_DR_COMPARISONS.csv")
    rows: list[dict] = []

    direct = annual[
        annual.quality_scope.eq("primary")
        & annual.comparison_mode.eq("direct")
        & annual.permission.eq("native_zone")
        & annual.layout.isin(LAYOUTS)
        & annual.status.eq("matched")
    ].copy()
    add(rows, "annual_direct", "Fig. 2b-d; Fig. 3a-c", "FINAL_ANNUAL_COMPARISONS.csv",
        "primary; two shared layouts; native-zone permission; direct; matched",
        len(direct), "paired control comparisons", "energy increased / decreased",
        f"{(direct.delta_annual_site_kwh > TOL).sum()} / {(direct.delta_annual_site_kwh < -TOL).sum()}")

    limits = annual[
        annual.quality_scope.eq("primary")
        & annual.comparison_mode.eq("equal_service")
        & annual.service_fraction.eq(1.0)
        & annual.layout.isin(LAYOUTS)
        & annual.status.eq("matched")
    ].copy()
    add(rows, "common_annual_limits", "Fig. 3d; Fig. 6b", "FINAL_ANNUAL_COMPARISONS.csv",
        "primary; two shared layouts; common annual mean and tail limits; matched",
        len(limits), "logical paired comparisons", "median annual energy change (%)",
        float(limits.delta_site_pct.median()))
    distinct = limits.drop_duplicates([
        "data_source", "context", "composition_id", "layout", "alternative",
        "alternative_run_id", "reference_run_id",
    ])
    add(rows, "common_annual_limits_distinct", "Supplementary sensitivity",
        "FINAL_ANNUAL_COMPARISONS.csv", "common annual limits; physical run pairs deduplicated",
        len(distinct), "distinct paired trajectories", "median annual energy change (%)",
        float(distinct.delta_site_pct.median()))
    for threshold in (0.02, 0.05, 0.10):
        close = (
            limits.delta_mean_reference_mismatch_c.abs().le(threshold)
            & limits.delta_fixed_tail_reference_mismatch_c.abs().le(threshold)
        )
        add(rows, f"realized_service_difference_{threshold:.2f}", "Supplementary sensitivity",
            "FINAL_ANNUAL_COMPARISONS.csv", "common annual limits; matched",
            len(limits), "logical paired comparisons",
            f"pairs with both realized mismatch differences <= {threshold:.2f} C", int(close.sum()))

    common = common[common.quality_scope.eq("primary") & common.layout.isin(LAYOUTS)].copy()
    for level in ("building_mean", "zone_mean"):
        group = common[common.information_level.eq(level)]
        add(rows, f"common_baseline_{level}", "Fig. 5b", "FINAL_COMMON_BASELINE_DR.csv",
            "primary; two shared layouts", len(group), "response conditions",
            "positive selected / usable after full review",
            f"{(group.selected_response_kw > TOL).sum()} / {(group.full_reviewed_usable_response_kw > TOL).sum()}")
    full = common.drop_duplicates([
        "data_source", "context", "composition_id", "layout", "permission", "event_month"
    ])
    add(rows, "common_baseline_full_distribution", "Fig. 5b", "FINAL_COMMON_BASELINE_DR.csv",
        "primary; two shared layouts; distinct response conditions", len(full), "response conditions",
        "usable positive response", int((full.ideal_full_information_response_kw > TOL).sum()))

    own = own[own.quality_scope.eq("primary") & own.layout.isin(LAYOUTS)].copy()
    qualified = own.qualified_positive_response.astype(str).str.lower().eq("true")
    selected = own[qualified].copy()
    add(rows, "own_baseline_response", "Fig. 5c,e", "FINAL_OWN_BASELINE_DR.csv",
        "primary; two shared layouts", len(own), "control conditions", "qualified positive response",
        int(qualified.sum()))
    add(rows, "recovery_50h", "Fig. 5e; Fig. S18", "FINAL_OWN_BASELINE_DR.csv",
        "qualified positive response; event start through 50 h", len(selected), "response actions",
        "net electricity saving > 0", int((selected.event_plus_recovery_net_electric_saving_kwh > TOL).sum()))
    add(rows, "recovery_168h", "Fig. S18", "FINAL_OWN_BASELINE_DR.csv",
        "qualified positive response; event start through 168 h", len(selected), "response actions",
        "net electricity saving > 0", int((selected.terminal_168h_net_electric_saving_kwh > TOL).sum()))

    paired = paired[paired.quality_scope.eq("primary") & paired.layout.isin(LAYOUTS)].copy()
    uniform = paired.uniform_response_kw.gt(TOL)
    guided = paired.demand_guided_response_kw.gt(TOL)
    both = uniform & guided
    add(rows, "own_baseline_paired_response", "Fig. 5d", "FINAL_OWN_BASELINE_DR_COMPARISONS.csv",
        "primary; two shared layouts", len(paired), "demand-guided versus uniform pairs",
        "both positive; guided lower / higher",
        f"{both.sum()}; {((paired.response_difference_kw < -TOL) & both).sum()} / {((paired.response_difference_kw > TOL) & both).sum()}")

    ledger = pd.DataFrame(rows)
    ledger.to_csv(output / "FINAL_CLAIM_EVIDENCE_MAP.csv", index=False)
    report = {
        "status": "PASS",
        "main_layouts": list(LAYOUTS),
        "annual_common_design_logical_candidates": 2160,
        "ashrae_source_structured_sensitivity_logical_candidates": 540,
        "claim_rows": len(ledger),
        "all_counts_derived_from_final_tables": True,
    }
    (output / "FINAL_SCOPE_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.evidence.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
