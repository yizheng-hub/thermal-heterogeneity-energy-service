"""Report CTCD engineering denominators and warning differences from baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

from audit_ventilation_adaptation import save, sha, warnings


NUMERICAL = ("SimHVAC:", "PsyTwb", "PsyPsat", "Temperature out of range",
             "WetBulb not converged", "CalcDoe2DXCoil:")


def nonnumerical_messages(folder):
    messages = warnings(folder)["direct_messages"]
    return {re.sub(r"\([\d.+Ee-]+ C\)", "(value C)", message)
            for message in messages if not any(token in message for token in NUMERICAL)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    repository, runs = args.repository_root.resolve(), args.run_root.resolve()
    engineering = runs / "engineering"
    manifest = pd.read_csv(engineering / "ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    results = json.loads((runs / "RUN_RESULTS.json").read_text())
    if len(manifest) != 144 or len(results) != manifest.run_id.nunique():
        raise ValueError("The fixed annual plan has not completed")
    manifest["qualified"] = manifest.engineering_status.isin(["new_qualified", "cached_qualified"])
    manifest["rejected"] = manifest.engineering_status.isin(["new_rejected", "cached_rejected"])
    manifest["execution_failed"] = manifest.engineering_status.eq("execution_failed")
    rows = []
    for key, group in manifest.groupby(["context", "control", "variant"], sort=True):
        rows.append(dict(zip(["context", "control", "variant"], key)) | {
            "logical_draw_candidate_denominator": len(group), "distinct_input_count_within_group": group.run_id.nunique(),
            "qualified_draw_candidates": int(group.qualified.sum()),
            "rejected_draw_candidates": int(group.rejected.sum()),
            "execution_failed_draw_candidates": int(group.execution_failed.sum()),
            "scope": "Each cell contains three predeclared population draws; repeated physical inputs across cells are not independent simulations."})
    pd.DataFrame(rows).to_csv(engineering / "QA_DENOMINATORS_BY_CONTEXT_CONTROL_VARIANT.csv", index=False)
    records, comparison = [], []
    for rid, item in sorted(results.items()):
        city = item["context"]
        qa = json.loads(Path(item["qa_path"]).read_text()) if item.get("qa_path") else None
        source = Path(item["source_folder"]) if item.get("source_folder") else Path(item["qa_path"]).parent
        baseline = repository / "data/upstream/frozen_buildings/baseline_evidence" / city
        additional = sorted(nonnumerical_messages(source) - nonnumerical_messages(baseline))
        recurring = warnings(source)["recurring_groups"]
        baseline_recurring = {g["message"] for g in warnings(baseline)["recurring_groups"]}
        additional_nonnumeric_recurring = [g for g in recurring if not any(token in g["message"] for token in NUMERICAL)
                                           and g["message"] not in baseline_recurring]
        comparison.append({"run_id": rid, "context": city, "engineering_status": item["engineering_status"],
                           "cache_reused": item["cache_reused"],
                           "additional_nonnumerical_direct_warning_groups": len(additional),
                           "additional_nonnumerical_recurring_warning_groups": len(additional_nonnumeric_recurring),
                           "additional_direct_messages": additional,
                           "additional_recurring_groups": additional_nonnumeric_recurring})
        if qa is not None:
            records.append({"run_id": rid, "context": city, "engineering_status": item["engineering_status"],
                            "cache_reused": item["cache_reused"], "qa_sha256": sha(item["qa_path"]),
                            "direct_numerical_warnings": qa.get("direct_numerical_warnings", []),
                            "severe": qa.get("severe", 0), "fatal": qa.get("fatal", 0),
                            "unexplained_hvac_recurring_groups": qa.get("unexplained_hvac_recurring_groups", 0),
                            "source_occupancy_ventilation_requirement_max_delta": qa.get("source_occupancy_ventilation_requirement_max_delta"),
                            "min_native_scheduled_ventilation_delivery_request": qa.get("min_native_scheduled_ventilation_delivery_request"),
                            "same_source_sizing": qa.get("same_source_sizing", qa.get("same_sizing")),
                            "hourly_meter_closure_kwh": qa.get("hourly_meter_closure_kwh", qa.get("hourly_electric_closure_max_kwh"))})
    save(engineering / "WARNING_BASELINE_COMPARISON.json", comparison)
    save(engineering / "UNIQUE_RUN_ENGINEERING_DETAILS.json", records)
    summary = {"logical_candidate_denominator": len(manifest), "unique_physical_input_denominator": len(results),
               "new_physical_runs": sum(not item["cache_reused"] for item in results.values()),
               "exact_cache_reuses": sum(item["cache_reused"] for item in results.values()),
               "logical_status_counts": manifest.engineering_status.value_counts().to_dict(),
               "distinct_input_status_counts": pd.Series([item["engineering_status"] for item in results.values()]).value_counts().to_dict(),
               "additional_nonnumerical_direct_warning_groups_total": sum(c["additional_nonnumerical_direct_warning_groups"] for c in comparison),
               "additional_nonnumerical_recurring_warning_groups_total": sum(c["additional_nonnumerical_recurring_warning_groups"] for c in comparison),
               "warning_comparison_scope": "Compare actual warning families with each corresponding original baseline. Numeric target-hot-water shortfall magnitudes and repeated occurrence counts can differ; these are not additional warning families. New numerical warnings remain explicit and reject affected candidates under the unchanged gate.",
               "no_extra_simulations_in_this_audit": True, "engineering_thresholds_changed": False}
    save(engineering / "ENGINEERING_DENOMINATOR_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
