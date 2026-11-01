"""Compile fixed-library selection sensitivities with explicit design denominators."""

from common import *
import numpy as np
import pandas as pd


BRANCHES = [
    ("original_family", "paired_uncertainty_identity", 54, 9),
    ("refined_family", "paired_uncertainty_refined_identity", 54, 9),
    ("pooled_archive_shared", "paired_uncertainty_pooled_identity", 54, 9),
    ("pooled_study_shared", "paired_uncertainty_pooled_study_shared_identity", 54, 9),
    ("pooled_temporal_multiple_records", "paired_uncertainty_pooled_temporal_identity", 27, 3),
    ("pooled_temporal_single_record", "paired_uncertainty_pooled_single_record_temporal_identity", 27, 3),
]
ENDPOINTS = ["conditional_choice_disagreement_fraction", "zone_mean_selection_full_service_exceedance_fraction",
             "conditional_full_feasible_fraction", "conditional_zone_mean_feasible_fraction",
             "extra_selected_energy_median_kwh", "zone_mean_selected_excess_p95_c"]


def summarize(frame, expected_settings, grouping):
    rows = []
    for identity, group in frame.groupby(grouping, sort=True, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        if len(group) != expected_settings or group.duplicated(["climate", "composition"]).any():
            raise ValueError("Incomplete or duplicated fixed-design selection settings")
        row = dict(zip(grouping, identity)) | {"planned_settings": expected_settings, "evaluated_settings": len(group)}
        for endpoint in ENDPOINTS:
            values = group[endpoint].dropna()
            row[endpoint + "_available_settings"] = len(values)
            for name in ["min", "median", "max"]:
                row[endpoint + "_" + name] = getattr(values, name)()
        rows.append(row)
    return rows


def nominal_choices(library, draws):
    nominal = draws[draws.scenario.eq("sourceSD0_personSD0")]
    if len(nominal) != 54 * 256:
        raise ValueError("Incomplete nominal pooled paired draws")
    for _, group in nominal.groupby(["climate", "composition"]):
        if len(group) != 256 or group.full_choice.nunique() != 1 or group.zone_mean_choice.nunique() != 1:
            raise ValueError("A zero-error nominal selection is inconsistent across repeated draws")
    nominal = nominal[nominal.draw.eq(0)].copy()
    source = library.sort_values(["run_id", "control", "offset_c"], kind="stable").drop_duplicates(
        ["climate", "composition", "physical_plan_id"])
    metadata = ["climate", "composition", "physical_plan_id", "control", "permission", "offset_c",
                "facility_purchased_kwh", "facility_electric_kwh", "facility_gas_hhv_kwh"]
    for label, field in [("full", "full_choice"), ("zone_mean", "zone_mean_choice")]:
        renamed = source[metadata].rename(columns={key: label + "_" + key for key in metadata[2:]})
        nominal = nominal.merge(renamed, left_on=["climate", "composition", field],
            right_on=["climate", "composition", label + "_physical_plan_id"], validate="one_to_one")
    if len(nominal) != 54:
        raise ValueError("A nominal choice has no matching physical-operation metadata")
    nominal["choices_differ"] = nominal.full_choice.ne(nominal.zone_mean_choice)
    nominal["full_service_requirement_exceeded"] = nominal.zone_mean_selected_full_service_excess_c.gt(1e-9)
    if not np.allclose(nominal.extra_energy_kwh,
                       nominal.full_facility_purchased_kwh - nominal.zone_mean_facility_purchased_kwh, atol=1e-6, rtol=0):
        raise ValueError("Nominal selected-energy difference does not close")
    return nominal


def main():
    out = ROOT / "results/information_uncertainty_summary"
    out.mkdir(parents=True, exist_ok=True)
    inputs, frames, reports = {}, [], []
    for label, folder, settings, scenarios in BRANCHES:
        root = ROOT / "results" / folder
        audit_path = root / "UNCERTAINTY_AUDIT.json"
        path = root / "SETTING_SELECTION_STABILITY.csv"
        audit = json.loads(audit_path.read_text())
        frame = pd.read_csv(path)
        if audit["settings"] != settings or audit["scenarios"] != scenarios or len(frame) != settings * scenarios:
            raise ValueError("Incomplete declared uncertainty branch: " + label)
        frame["library_and_scenario_design"] = label
        frames.append(frame)
        reports.append({"branch": label, "execution": "completed", "planned_settings": settings,
            "scenarios": scenarios, "draws_per_scenario": audit["draws"],
            "calibrated_latent_measurement_error": "not identified by these data",
            "source_audit_status": audit["status"]})
        inputs.update({p.relative_to(CODE).as_posix(): digest(p) for p in [audit_path, path]})
    summaries = []
    for (label, _, settings, _), frame in zip(BRANCHES, frames):
        summaries += summarize(frame, settings, ["library_and_scenario_design", "scenario"])
    pd.DataFrame(summaries).to_csv(out / "DESCRIPTIVE_SCENARIO_SELECTION_SUMMARY.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(out / "SETTING_SELECTION_STABILITY_ALL_BRANCHES.csv", index=False)
    library_path = ROOT / "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv"
    draws_path = ROOT / "results/paired_uncertainty_pooled_identity/PAIRED_SELECTION_DRAWS.csv.gz"
    nominal = nominal_choices(pd.read_csv(library_path), pd.read_csv(draws_path))
    nominal.to_csv(out / "NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv", index=False)
    inputs.update({p.relative_to(CODE).as_posix(): digest(p) for p in [library_path, draws_path]})
    write_json(out / "INPUT_SHA256.json", inputs)
    report = {"status": "completed", "branches": reports, "nominal_settings": len(nominal),
        "nominal_changed_choices": int(nominal.choices_differ.sum()),
        "nominal_mean_choices_exceeding_full_service_requirements": int(nominal.full_service_requirement_exceeded.sum()),
        "nominal_full_choice_command_scope": nominal.full_permission.value_counts().to_dict(),
        "scope": "Fixed-design per-setting conditional frequencies and finite-library choices; no independent-sample or global-population inference",
        "executed_source_sha256": digest(__file__), "output_sha256": {p.name: digest(p) for p in out.glob("*.csv")}}
    write_json(out / "AUDIT.json", report)
    record_status("information_uncertainty_summary", "completed", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
