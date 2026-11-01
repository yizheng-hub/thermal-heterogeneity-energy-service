"""Reselect the fixed annual candidate library under absolute service margins.

The frozen delta grid is a sensitivity scan, not a comfort-standard allowance.
Only the two common service limits change; trajectories and scores are reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import decision_effect_sizes as audit
except ImportError:  # direct script execution
    import decision_effect_sizes as audit


original = audit.original
DELTAS_C = (0.0, 0.01, 0.025, 0.05, 0.10)
SCORES = {
    "full": ("mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"),
    "zone_mean": (
        "zone_mean_mean_reference_mismatch_c", "zone_mean_fixed_tail_reference_mismatch_c",
    ),
}


def scan(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return every setting, every physical-plan qualification and grid totals."""
    plan_columns = [
        *audit.SCENARIO_KEYS, "physical_plan_id", "run_id", "control", "offset_c",
        "physical_alias_count", "engineering_pass", "facility_purchased_kwh",
        *(name for pair in SCORES.values() for name in pair),
    ]
    plans = pd.concat([
        original._deduplicate_physical_plans(group)[plan_columns]
        for _, group in metrics.groupby(audit.SCENARIO_KEYS, sort=True, dropna=False)
    ], ignore_index=True)
    decisions, qualifications, totals = [], [], []
    baseline_choices = None
    for delta in DELTAS_C:
        relaxed = anchors.copy()
        relaxed["anchor_mean_mismatch_c"] += delta
        relaxed["anchor_tail_mismatch_c"] += delta
        # Reuse the original all-library eligibility, alias and selection rules.
        detail = original.fixed_plan_information_diagnostic(
            metrics, relaxed, fractions=(1.0,)
        ).sort_values(audit.SCENARIO_KEYS).reset_index(drop=True)
        detail.insert(0, "delta_c", delta)
        resolved = ~detail.full_unresolved & ~detail.zone_mean_unresolved
        changed = resolved & detail.selection_disagreement
        detail["paired_resolved"] = resolved
        detail["selected_physical_plan_changed"] = changed
        detail["availability_disagreement"] = detail.full_unresolved != detail.zone_mean_unresolved
        energy_delta = (
            detail.full_selected_facility_purchased_kwh
            - detail.zone_mean_selected_facility_purchased_kwh
        )
        detail["delta_purchased_kwh_full_minus_zone_mean"] = energy_delta.where(resolved)
        detail["delta_purchased_pct_full_minus_zone_mean"] = (
            100 * energy_delta / detail.zone_mean_selected_facility_purchased_kwh
        ).where(resolved)
        choice_columns = [f"{prefix}_selected_plan_id" for prefix in SCORES]
        if baseline_choices is None:
            baseline_choices = detail[choice_columns].copy()
        for prefix in SCORES:
            column = f"{prefix}_selected_plan_id"
            detail[f"{prefix}_choice_changed_from_delta_zero"] = (
                detail[column] != baseline_choices[column]
            )

        limits = detail[[
            *audit.SCENARIO_KEYS, "delta_c", "mean_limit_c", "tail_limit_c", *choice_columns,
        ]]
        qualification = plans.merge(limits, on=audit.SCENARIO_KEYS, validate="many_to_one")
        for prefix, (population, tail) in SCORES.items():
            qualified = (
                qualification.engineering_pass.astype(bool)
                & qualification[population].le(
                    qualification.mean_limit_c + original.SERVICE_TOLERANCE_C
                )
                & qualification[tail].le(
                    qualification.tail_limit_c + original.SERVICE_TOLERANCE_C
                )
            )
            qualification[f"{prefix}_qualified"] = qualified
            qualification[f"{prefix}_selected"] = (
                qualification.physical_plan_id == qualification[f"{prefix}_selected_plan_id"]
            )
            counts = qualification.groupby(audit.SCENARIO_KEYS, sort=True)[
                f"{prefix}_qualified"
            ].sum().to_numpy()
            if not np.array_equal(counts, detail[f"{prefix}_qualified_plan_count"]):
                raise ValueError("Qualification membership disagrees with original selection")

        total = {
            "delta_c": delta, "settings": len(detail), "candidate_rows": len(metrics),
            "physical_plans": len(plans),
            "eligibility_set_changed_settings": int(detail.eligibility_disagreement.sum()),
            "selected_physical_plan_changed_settings": int(changed.sum()),
            "selection_disagreement_settings": int(detail.selection_disagreement.sum()),
            "availability_disagreement_settings": int(detail.availability_disagreement.sum()),
            "paired_resolved_settings": int(resolved.sum()),
            "zone_mean_choices_failing_full_review": int(
                detail.zone_mean_selected_plan_fails_full_review.sum()
            ),
        }
        for prefix in SCORES:
            total[f"{prefix}_unresolved_settings"] = int(detail[f"{prefix}_unresolved"].sum())
            total[f"{prefix}_qualified_physical_plans"] = int(
                detail[f"{prefix}_qualified_plan_count"].sum()
            )
            total[f"{prefix}_choices_changed_from_delta_zero"] = int(
                detail[f"{prefix}_choice_changed_from_delta_zero"].sum()
            )
        for name in ("qualified_under_both_count", "zone_mean_only_qualified_count",
                     "full_only_qualified_count"):
            total[name] = int(detail[name].sum())
        for subset, mask in (("changed", changed), ("all_paired", resolved)):
            for unit in ("kwh", "pct"):
                values = detail.loc[mask, f"delta_purchased_{unit}_full_minus_zone_mean"]
                for statistic in ("min", "median", "max", "mean"):
                    total[f"{subset}_energy_delta_{unit}_{statistic}"] = (
                        float(getattr(values, statistic)()) if len(values) else None
                    )
        decisions.append(detail)
        qualifications.append(qualification)
        totals.append(total)
    return (pd.concat(decisions, ignore_index=True),
            pd.concat(qualifications, ignore_index=True), pd.DataFrame(totals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/global_extension/global_final"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/global_extension/decision_tolerance_sensitivity"))
    args = parser.parse_args()
    paths = {key: args.source / name for key, name in audit.INPUT_NAMES.items()}
    source_audit_path = args.source / "GLOBAL_ANALYSIS_AUDIT.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    hashes = {key: original.sha256(path) for key, path in paths.items()}
    for key, path in paths.items():
        if hashes[key] != source_audit["output_sha256"][path.name]:
            raise ValueError(f"Input differs from original analysis audit: {path.name}")
    inputs = {key: pd.read_csv(path) for key, path in paths.items()}
    effects, replay, baseline_summary = audit.analyse(**inputs)
    detail, qualification, totals = scan(inputs["metrics"], inputs["anchors"])
    zero = detail.loc[detail.delta_c.eq(0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(zero[replay.columns], replay, check_exact=True)
    zero_changed = zero.loc[zero.selected_physical_plan_changed].reset_index(drop=True)
    for column in ("delta_purchased_kwh_full_minus_zone_mean",
                   "delta_purchased_pct_full_minus_zone_mean"):
        pd.testing.assert_series_equal(zero_changed[column], effects[column], check_exact=True)
    if (len(inputs["metrics"]), len(zero), int(zero.eligibility_disagreement.sum()),
            int(zero.selection_disagreement.sum())) != (810, 54, 23, 10):
        raise ValueError("Delta-zero analysis does not reproduce the frozen 810/54/23/10 contract")

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "DECISION_TOLERANCE_SETTINGS.csv": detail,
        "DECISION_TOLERANCE_QUALIFICATIONS.csv": qualification,
        "DECISION_TOLERANCE_SUMMARY.csv": totals,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False, float_format="%.17g")
    summary = {
        "status": "PASS", "delta_grid_c": list(DELTAS_C),
        "definition": "Absolute delta added to both common full-profile 25-C annual service limits, identically for both demand representations; all 810 candidate rows are re-evaluated and reselected at every delta.",
        "interpretation": "The frozen grid is an analysis sensitivity specification, not a comfort-standard tolerance or numerical score tolerance.",
        "contract": baseline_summary["contract"] | {
            "service_limits": "Population limit = original full-profile 25-C population mismatch + delta; fixed-tail limit = original full-profile 25-C fixed-tail mismatch + delta. The existing 1e-9 C numerical tolerance is then applied.",
            "candidate_library": "The same 810 rows, physical aliases, engineering status, trajectories, energy values and precomputed service scores at every delta.",
            "qualification_membership": "One row per setting, delta and canonical physical plan; full_qualified and zone_mean_qualified identify both complete sets.",
            "selected_physical_plan_changed": "Different selected physical IDs with qualifying choices under both representations; availability disagreement is reported separately.",
            "energy_summaries": "Individual/full selected energy minus zone-mean selected energy; changed statistics use only different physical choices with both reviews resolved, all_paired statistics include unchanged choices. No unresolved case is assigned an energy difference.",
        },
        "baseline_exact_replay": {"settings": 54, "eligibility_set_changes": 23,
                                  "selected_physical_plan_changes": 10,
                                  "all_recorded_decision_fields_equal": True,
                                  "changed_choice_energy_deltas_equal": True},
        "grid_summary": totals.astype(object).where(pd.notna(totals), None).to_dict(orient="records"),
        "input_sha256": hashes | {"original_analysis_audit": original.sha256(source_audit_path)},
        "source_code_sha256": {
            "selection": original.sha256(Path(original.__file__)),
            "baseline_audit": original.sha256(Path(audit.__file__)),
            "tolerance_sensitivity": original.sha256(Path(__file__)),
            "thermal_service": original.sha256(original.ENERGY_SERVICE / "thermal_service.py"),
        },
        "output_sha256": {name: original.sha256(args.output / name) for name in outputs},
    }
    (args.output / "DECISION_TOLERANCE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
