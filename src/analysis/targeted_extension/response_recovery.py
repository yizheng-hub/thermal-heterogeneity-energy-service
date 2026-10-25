"""Separate tested response power from baseline, event and recovery qualification.

Reuses retained 2 h + 48 h trajectories. Original peak-day designs and the
one-composition, random-layout July date check remain separate scopes. Sums of
logical scenario responses are diagnostic totals, never city/grid capacity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOL = 1e-9
QUALIFIED = {"cached_qualified", "new_qualified"}
KEYS = ["source_scope", "context", "composition_id", "layout", "permission", "control",
        "date_class", "event_month", "event_day", "service_fraction"]
STAGES = ["raw_positive", "baseline_qualified", "event_qualified", "recovery_qualified"]
METRICS = ["event_average_reduction_kw", "event_electric_saving_kwh",
           "recovery_extra_electric_kwh", "recovery_peak_hour_extra_kw",
           "event_plus48h_net_electric_saving_kwh", "event_plus48h_gas_change_kwh_hhv",
           "event_gas_change_kwh_hhv", "recovery_gas_change_kwh_hhv"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def explicit_bool(values):
    text = values.astype(str).str.lower()
    if not text.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Expected explicit Boolean decisions")
    return text.isin(["true", "1"])


def energy_accounting(hours):
    """Independent signed electricity and gas accounting for exactly 50 hours."""
    hours = hours.sort_values("hour_index")
    if hours.hour_index.tolist() != list(range(50)):
        raise ValueError("Each action must contain exactly 2 event + 48 recovery hours")
    de = hours.perturbed_electric_kwh.to_numpy() - hours.baseline_electric_kwh.to_numpy()
    dg = hours.perturbed_gas_hhv_kwh.to_numpy() - hours.baseline_gas_hhv_kwh.to_numpy()
    if not np.isfinite(de).all() or not np.isfinite(dg).all():
        raise ValueError("Missing hourly energy carriers")
    saving, rebound = float(-de[:2].sum()), float(de[2:].sum())
    return {"event_average_reduction_kw": saving / 2,
            "event_electric_saving_kwh": saving,
            "recovery_extra_electric_kwh": rebound,
            "recovery_peak_hour_extra_kw": float(max(0., de[2:].max())),
            "event_plus48h_net_electric_saving_kwh": saving - rebound,
            "event_plus48h_gas_change_kwh_hhv": float(dg.sum()),
            "event_gas_change_kwh_hhv": float(dg[:2].sum()),
            "recovery_gas_change_kwh_hhv": float(dg[2:].sum())}


def classify_gates(table):
    """Baseline gate includes the baseline during both event and recovery."""
    table = table.copy()
    table["engineering_qualified"] = table.engineering_status.isin(QUALIFIED)
    table["baseline_available"] = explicit_bool(table.baseline_available)
    for window in ["event", "recovery"]:
        for trajectory in ["baseline", "perturbed"]:
            passed = pd.Series(True, index=table.index)
            for metric in ["mean", "fixed_tail"]:
                score = pd.to_numeric(table[f"{window}_{trajectory}_{metric}_c"], errors="coerce")
                budget = pd.to_numeric(table[f"{window}_{metric}_budget_c"], errors="coerce")
                check = score.le(budget + TOL)
                table[f"pass_{window}_{trajectory}_{metric}"] = check
                passed &= check
            table[f"pass_{window}_{trajectory}"] = passed
    table["baseline_gate"] = (table.baseline_available & table.pass_event_baseline &
                              table.pass_recovery_baseline)
    table["raw_positive"] = (table.engineering_qualified & table.baseline_available &
                             table.event_average_reduction_kw.gt(TOL))
    table["baseline_qualified"] = table.raw_positive & table.baseline_gate
    table["event_qualified"] = table.baseline_qualified & table.pass_event_perturbed
    table["recovery_qualified"] = table.event_qualified & table.pass_recovery_perturbed
    if "baseline_service_qualified" in table:
        if not table.baseline_gate.eq(explicit_bool(table.baseline_service_qualified)).all():
            raise ValueError("Recomputed baseline service decision differs from retained result")
    if "service_eligible" in table:
        final = (table.engineering_qualified & table.baseline_gate &
                 table.pass_event_perturbed & table.pass_recovery_perturbed)
        if not final.eq(explicit_bool(table.service_eligible)).all():
            raise ValueError("Recomputed eight-constraint eligibility differs from retained result")
    table["first_exclusion"] = np.select(
        [~table.engineering_qualified, ~table.baseline_available, ~table.raw_positive,
         ~table.baseline_gate, ~table.pass_event_perturbed, ~table.pass_recovery_perturbed],
        ["engineering", "baseline_unavailable", "nonpositive_response", "baseline_service",
         "event_service", "recovery_service"], default="retained")
    return table


def mark_spatial_aliases(table, contexts):
    """Retain logical aliases, but exclude structurally identical floor commands from primary counts."""
    table = table.copy()
    if not set(table.context).issubset(contexts):
        raise ValueError("Context settings do not cover every event context")
    aliases = {name for name, item in contexts.items() if item["floor_binding_equals_building"]}
    table["distinct_spatial_condition"] = ~(table.context.isin(aliases) & table.permission.eq("floor"))
    key_columns = [key for key in KEYS if key != "permission"] + ["amplitude_c"]
    fields = ["run_id", "baseline_run_id", *STAGES, *METRICS]
    reference = table[table.permission.eq("building")]
    for row in table[~table.distinct_spatial_condition].to_dict("records"):
        match = reference[(reference[key_columns] == pd.Series(row)[key_columns]).all(axis=1)]
        if len(match) != 1:
            raise ValueError("Floor alias lacks exactly one corresponding building command")
        for field in fields:
            left, right = row[field], match.iloc[0][field]
            if pd.isna(left) and pd.isna(right):
                continue
            if left != right:
                raise ValueError(f"Structurally identical floor/building aliases differ in {field}")
    return table


def select_stages(table):
    records = []
    for key, group in table.groupby(KEYS, sort=True, dropna=False):
        if group.baseline_run_id.nunique() != 1 or group.baseline_gate.nunique() != 1:
            raise ValueError("Amplitude candidates do not share an identical own-family baseline")
        previous_kw = None
        for stage in STAGES:
            candidates = group[group[stage]]
            baseline_ok = bool(group.baseline_gate.iloc[0])
            row = dict(zip(KEYS, key)) | {"stage": stage,
                "logical_candidates": len(group), "engineering_candidates": int(group.engineering_qualified.sum()),
                "positive_candidates_retained": len(candidates), "selected": not candidates.empty,
                "baseline_run_id": group.baseline_run_id.iloc[0], "baseline_service_qualified": baseline_ok,
                "response_kw_with_no_action_zero": 0.,
                "service_qualified_available_kw": 0. if baseline_ok else np.nan,
                "selected_run_id": "", "selected_amplitude_c": np.nan}
            if not candidates.empty:
                maximum = float(candidates.event_average_reduction_kw.max())
                selected = candidates[candidates.event_average_reduction_kw.ge(maximum - TOL)].sort_values(
                    ["run_id", "amplitude_c"], kind="mergesort").iloc[0]
                row.update({metric: float(selected[metric]) for metric in METRICS})
                row.update(selected_run_id=selected.run_id, selected_amplitude_c=float(selected.amplitude_c),
                           response_kw_with_no_action_zero=float(selected.event_average_reduction_kw))
                if baseline_ok:
                    row["service_qualified_available_kw"] = float(selected.event_average_reduction_kw)
                row["recovery_fraction_of_event_saving"] = (selected.recovery_extra_electric_kwh /
                                                             selected.event_electric_saving_kwh)
            current_kw = row["response_kw_with_no_action_zero"]
            row["response_kw_lost_from_previous_stage"] = 0. if previous_kw is None else previous_kw - current_kw
            if previous_kw is not None and current_kw > previous_kw + TOL:
                raise ValueError("Nested service filters cannot raise maximum positive response")
            records.append(row)
            previous_kw = current_kw
    return pd.DataFrame(records)


def summarise(decisions, group_columns):
    records = []
    for key, group in decisions.groupby(group_columns, sort=True, dropna=False):
        selected = group[group.selected]
        row = dict(zip(group_columns, key)) | {
            "logical_controller_decisions": len(group),
            "logical_candidate_budget_windows": int(group.logical_candidates.sum()),
            "engineering_candidate_budget_windows": int(group.engineering_candidates.sum()),
            "retained_positive_candidate_budget_windows": int(group.positive_candidates_retained.sum()),
            "decisions_with_positive_selection": len(selected),
            "baseline_service_qualified_decisions": int(group.baseline_service_qualified.sum()),
            "selected_unique_physical_runs": int(selected.selected_run_id.nunique()),
            "diagnostic_sum_response_kw_not_stock_capacity": float(group.response_kw_with_no_action_zero.sum()),
            "diagnostic_sum_kw_lost_from_previous_stage": float(group.response_kw_lost_from_previous_stage.sum()),
            "selected_net_electric_increase_count": int(selected.event_plus48h_net_electric_saving_kwh.lt(-TOL).sum())
                if len(selected) else 0,
            "selected_gas_increase_count": int(selected.event_plus48h_gas_change_kwh_hhv.gt(TOL).sum())
                if len(selected) else 0,
        }
        for metric in METRICS + ["recovery_fraction_of_event_saving"]:
            series = selected[metric] if metric in selected else pd.Series(dtype=float)
            row.update({f"{metric}_{label}": float(series.quantile(q)) if len(series) else np.nan
                        for label, q in [("p05", .05), ("median", .5), ("p95", .95)]})
        records.append(row)
    return pd.DataFrame(records)


def pair_controls(decisions):
    key_columns = [key for key in KEYS if key != "control"] + ["stage"]
    uniform = decisions[decisions.control.eq("uniform_average")]
    output = []
    for row in decisions[~decisions.control.eq("uniform_average")].to_dict("records"):
        matches = uniform[(uniform[key_columns] == pd.Series(row)[key_columns]).all(axis=1)]
        if len(matches) != 1:
            raise ValueError("Each personal control requires exactly one matched uniform controller")
        reference = matches.iloc[0]
        matched = row["selected"] and bool(reference.selected)
        output.append({key: row[key] for key in KEYS + ["stage"]} | {
            "personal_selected": row["selected"], "uniform_selected": bool(reference.selected),
            "both_selected": matched,
            "both_baselines_service_qualified": row["baseline_service_qualified"] and bool(reference.baseline_service_qualified),
            "delta_response_kw_with_no_action_zero": row["response_kw_with_no_action_zero"] - reference.response_kw_with_no_action_zero,
            "delta_service_qualified_available_kw": row["service_qualified_available_kw"] - reference.service_qualified_available_kw,
            **{f"delta_{metric}": row.get(metric, np.nan) - reference.get(metric, np.nan) if matched else np.nan
               for metric in METRICS}})
    return pd.DataFrame(output)


def read_scope(candidates_path, hourly_path, scope):
    table = pd.read_csv(candidates_path, keep_default_na=False)
    hours = pd.read_csv(hourly_path)
    table["source_scope"] = scope
    if scope == "main_peak_dates":
        table["date_class"] = "original_maximum"
        hours = hours.rename(columns={"hour_from_event_start": "hour_index", "event_electric_kwh": "perturbed_electric_kwh",
                                      "event_gas_hhv_kwh": "perturbed_gas_hhv_kwh"})
        join = ["context", "composition_id", "layout", "permission", "control", "run_id", "baseline_run_id",
                "amplitude_c", "event_month", "event_day"]
    else:
        join = ["context", "control", "date_class", "run_id", "baseline_run_id", "amplitude_c", "event_day"]
    metrics = []
    for key, group in hours.groupby(join, sort=True, dropna=False):
        metrics.append(dict(zip(join, key)) | energy_accounting(group))
    independent = pd.DataFrame(metrics)
    table = table.merge(independent, on=join, how="left", suffixes=("_retained", ""), validate="many_to_one")
    qualified = table.engineering_status.isin(QUALIFIED)
    if table.loc[qualified, "event_electric_saving_kwh"].isna().any():
        raise ValueError("Engineering-qualified candidate lacks full hourly support")
    max_error = 0.
    for metric in METRICS:
        previous = metric + "_retained"
        if previous not in table:
            continue
        original = pd.to_numeric(table[previous], errors="coerce")
        # The retained recovery peak can be negative; this extension records positive rebound only.
        if metric == "recovery_peak_hour_extra_kw":
            original = original.clip(lower=0)
        error = abs(original - table[metric]).loc[qualified].max()
        max_error = max(max_error, float(error))
        if error > 1e-8:
            raise ValueError(f"Independent hourly accounting differs in {metric}: {error}")
    return classify_gates(table), {"source_scope": scope, "candidate_budget_windows": len(table),
        "unique_context_run_event_windows": len(independent), "hourly_rows": len(hours),
        "max_hourly_vs_retained_energy_difference": max_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--context-settings", type=Path, required=True,
                        help="Frozen controller/CONTEXTS.json defining floor/building command equivalence")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    replay, out = args.replay_root.resolve(), args.output_root.resolve()
    paths = [(replay / "energy_service/events", "main_peak_dates"),
             (replay / "typical_day_dr", "additional_july_dates")]
    source_hash, tables, checks = {}, [], []
    for root, scope in paths:
        candidate, hourly = root / "EVENT_ALL_CANDIDATES.csv", root / "EVENT_HOURLY.csv.gz"
        for path in [candidate, hourly]:
            source_hash[path.relative_to(replay).as_posix()] = digest(path)
        table, check = read_scope(candidate, hourly, scope)
        tables.append(table)
        checks.append(check)
    settings = args.context_settings.resolve()
    settings_hash = digest(settings)
    all_candidates = mark_spatial_aliases(pd.concat(tables, ignore_index=True),
                                          json.loads(settings.read_text(encoding="utf-8")))
    logical_decisions = select_stages(all_candidates)
    decisions = select_stages(all_candidates[all_candidates.distinct_spatial_condition])
    grouping = ["source_scope", "context", "permission", "control", "date_class", "event_month",
                "service_fraction", "stage"]
    summary = summarise(decisions, grouping)
    all_contexts = summarise(decisions, ["source_scope", "date_class", "event_month", "service_fraction", "stage"])
    logical_contexts = summarise(logical_decisions, ["source_scope", "date_class", "event_month", "service_fraction", "stage"])
    paired = pair_controls(decisions)
    # Peak subset uses exactly the same composition, layout and permissions as the extra dates.
    date_check = decisions[decisions.composition_id.eq(2026102001) &
        decisions.layout.eq("random_mixing") & decisions.permission.eq("native_zone") & decisions.event_month.eq(7)].copy()
    date_summary = summarise(date_check, ["context", "date_class", "control", "service_fraction", "stage"])
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"CANDIDATE_GATES.csv": all_candidates, "STAGE_DECISIONS.csv": decisions,
               "STAGE_SUMMARY.csv": summary, "ALL_CONTEXT_STAGE_SUMMARY.csv": all_contexts,
               "LOGICAL_STAGE_DECISIONS.csv": logical_decisions,
               "LOGICAL_ALL_CONTEXT_STAGE_SUMMARY.csv": logical_contexts,
               "CONTROL_PAIRS.csv": paired, "THREE_DATE_STAGE_COMPARISON.csv": date_summary}
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    if source_hash != {name: digest(replay / name) for name in source_hash}:
        raise ValueError("Frozen input analysis products changed during the extension")
    if digest(settings) != settings_hash:
        raise ValueError("Frozen context settings changed during analysis")
    report = {"status": "PASS", "physical_simulations_run": 0, "physical_parameters_changed": False,
        "own_family_baselines_only": True, "event_hours": 2, "recovery_hours": 48,
        "service_thresholds_and_tail_memberships_unchanged": True,
        "baseline_gate": "Own baseline meets mean and fixed-tail limits in both event and recovery",
        "stage_order": STAGES,
        "primary_scope": "Distinct spatial conditions: structurally identical single-floor floor/building commands counted once; complete logical aliases retained in LOGICAL tables and CANDIDATE_GATES",
        "context_settings_sha256": settings_hash,
        "excluded_alias_candidate_budget_windows": int((~all_candidates.distinct_spatial_condition).sum()),
        "stage_order_interpretation": "Sequential attribution depends on declared order; intersected filters are mathematical, actual retained/lost power and energy are empirical outcomes",
        "scope": "Main original peak dates: compositions and layouts are declared scenarios. Additional July median/P90: one composition, random layout, native-zone permissions. No headline replay duplication.",
        "denominators": "Service fractions are repeated evaluations, not independent experiments; unique run identifiers are reused physical trajectories, not independent buildings or cities",
        "response_interpretation": "Best positive response among tested 0.5 and 1 C actions, not maximum or annual city capacity",
        "signed_energy": "Electric savings positive; recovery extra electricity and gas change positive for increases. Net savings = event savings - signed recovery extra electricity.",
        "availability": "Ineligible baseline has NaN qualified availability. No-action zero is separately reported; eligible baseline without tested action has zero availability.",
        "carbon": "No hourly marginal carbon inference is made by this extension",
        "input_sha256": source_hash, "checks": checks,
        "rows": {name: len(frame) for name, frame in outputs.items()},
        "output_sha256": {name: digest(out / name) for name in outputs}}
    (out / "RESPONSE_RECOVERY_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["status", "physical_simulations_run", "rows", "checks"]}, indent=2))


if __name__ == "__main__":
    main()
