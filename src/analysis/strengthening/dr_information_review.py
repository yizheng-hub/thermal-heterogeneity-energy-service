"""Nominal event-information review, gated by complete explicit full replay."""

from common import *
import common
import importlib.util
from functools import lru_cache
import numbers
import numpy as np
import pandas as pd
import multi_event_dr_extension as original_dr
from analyse_global_results import zone_mean_profiles
import analyse_global_results


ACTION_KEYS = ["climate", "composition_id", "control", "event_id", "action_id"]
EVENT_KEYS = ACTION_KEYS[:-1]
CELL_KEYS = EVENT_KEYS + ["normal_energy_reference_factor"]
EXPECTED_ACTIONS, EXPECTED_CELLS = 6480, 3888
METRICS = ("population_mean_mismatch_c", "fixed_tail_mean_mismatch_c")
POWER = "event_facility_mean_power_reduction_kw"
NET = "precondition_event_recovery_facility_net_reduction_kwh"
STAGES = ["baseline_window", "action_precondition", "action_event", "action_recovery"]
CHOICE_COLUMNS = ["climate", "composition_id", "control", "normal_energy_reference_factor", "event_id",
    "baseline_status", "selection_information", "status", "selected_action", "baseline_facility_mean_kw",
    "absolute_event_facility_mean_kw", "event_power_reduction_kw", "preconditioning_electricity_change_kwh",
    "recovery_electricity_change_kwh", "net_full_window_electricity_reduction_kwh", "positive_net_full_window",
    "positive_net_selected_action", "positive_net_event_power_kw", "positive_net_full_window_kwh"]


def source_hashes():
    return {Path(p).resolve().relative_to(CODE).as_posix(): digest(p)
            for p in [__file__, common.__file__, original_dr.__file__, analyse_global_results.__file__]}


def explicit_scorer():
    # A separate module has never been patched by exact_dr_cache.enable.
    spec = importlib.util.spec_from_file_location("_dr_information_explicit", original_dr.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bindings(inputs, sources):
    if source_hashes() != sources:
        raise ValueError("Response-information executable sources changed")
    for name, expected in inputs.items():
        path = (CODE / name).resolve()
        if not path.is_relative_to(CODE) or not path.is_file() or digest(path) != expected:
            raise ValueError("Response-information input changed: " + name)


def validate_context_binding(plan, done, normal):
    qa = json.loads(done.qa)
    if not normal.normal_replay_matches_original or not json.loads(done.history_check)["history_matches"]:
        raise ValueError("Frozen normal or event history is not verified")
    if (qa["normal_state_reference_sha256"] != normal.states_sha256
            or qa["input_idf_sha256"] != plan.input_sha256
            or qa["hourly_sha256"] != done.hourly_sha256):
        raise ValueError("Completed action does not match the planned input and normal state reference")
    return qa


def verified_cache(out, primary_assets=None):
    receipt = out / "AUDIT.json"
    if not receipt.is_file():
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Preserve the incomplete response-information attempt before retrying")
        return False
    audit = json.loads(receipt.read_text())
    if audit["status"] != "completed":
        raise ValueError("Response-information cache is incomplete")
    if primary_assets is not None and audit.get("primary_assets") != primary_assets:
        raise ValueError("Response-information current primary inputs differ from the completed cache")
    verify_bindings(json.loads((out / "INPUT_SHA256.json").read_text()), audit["source_sha256"])
    for name, expected in audit["output_sha256"].items():
        if not (out / name).is_file() or digest(out / name) != expected:
            raise ValueError("Response-information output changed: " + name)
    return True


def assess_service(row):
    """Use immutable full-reference scores; event/recovery always require finiteness."""
    result = dict(row)
    pre = bool(row["reference_precondition_weighted_profile_hours"] > 0)
    windows = ["event", "recovery"] + (["precondition"] if pre else [])
    passes = {}
    for trajectory in ["baseline", "action"]:
        for window in ["precondition", "event", "recovery"]:
            checked = window != "precondition" or pre
            observed = np.array([row[f"{trajectory}_{window}_{metric}"] for metric in METRICS]) if checked else np.array([])
            reference = np.array([row[f"reference_{window}_{metric}"] for metric in METRICS]) if checked else np.array([])
            differences = observed - reference
            finite = np.isfinite(observed).all() and np.isfinite(reference).all()
            passes[trajectory, window] = not checked or (finite and (observed <= reference + 1e-9).all())
            result[f"{trajectory}_{window}_excess_c"] = max(0.0, *differences) if checked and finite else np.nan
    result["baseline_service_qualified"] = all(passes["baseline", w] for w in windows)
    result["baseline_window_excess_c"] = max(result[f"baseline_{w}_excess_c"] for w in windows) if all(
        np.isfinite(result[f"baseline_{w}_excess_c"]) for w in windows) else np.nan
    result["action_non_event_service_qualified"] = all(passes["action", w] for w in windows if w != "event")
    result["action_event_service_qualified"] = passes["action", "event"]
    result["service_qualified"] = all(passes.values())
    result["positive_event_response"] = row["event_facility_electricity_reduction_kwh"] > 1e-9
    result["positive_full_window_response"] = row[NET] > 1e-9
    result["qualified_positive_event_response"] = result["service_qualified"] and result["positive_event_response"]
    result["qualified_positive_full_window_response"] = result["service_qualified"] and result["positive_full_window_response"]
    result["qualified_positive_response"] = result["qualified_positive_full_window_response"]
    return result


def replay_actions(data, context, scorer):
    completed = data["completed"].set_index("run_id", verify_integrity=True)
    rows = []
    for plan in data["plan"].itertuples(index=False):
        done = completed.loc[plan.run_id]
        identity = {key: getattr(plan, key) for key in ACTION_KEYS} | {"run_id": plan.run_id}
        if not done.completed or not done.final_engineering_qualified:
            rows.append(identity | {"status": "engineering_failure"})
            continue
        baseline, action, reference, profiles, tail, event = context(plan)
        previous = scorer.fixed_tail_profile_ids
        try:
            scorer.fixed_tail_profile_ids = lambda reference, group, tail_fraction=.1: tail
            scored = scorer.score_event(baseline, action, reference, profiles, event, limit_factors=(1.0,))[0]
        finally:
            scorer.fixed_tail_profile_ids = previous
        clock = scorer.compact_timestamps(baseline)
        start = pd.Timestamp(event["start_timestamp"])
        mask = (clock >= start) & (clock < start + pd.Timedelta(hours=int(event["duration_hours"])))
        baseline_kw = float(baseline["facility_electric_kwh"][mask].mean())
        row = identity | {"normal_id": plan.normal_id, "status": "evaluated",
            "baseline_facility_mean_kw": baseline_kw,
            "absolute_action_facility_mean_kw": baseline_kw - scored[POWER]} | scored
        assessed = assess_service(row)
        for name in ["baseline_service_qualified", "action_non_event_service_qualified", "action_event_service_qualified",
                     "service_qualified", "qualified_positive_event_response", "qualified_positive_full_window_response"]:
            if assessed[name] != row[name]:
                raise ValueError("Explicit replay and stage checks differ: " + name)
        rows.append(assessed)
    result = pd.DataFrame(rows)
    needed = ["service_qualified", "baseline_service_qualified", "qualified_positive_event_response",
              *[stage + "_excess_c" for stage in STAGES]]
    return result.reindex(columns=[*result.columns, *[c for c in needed if c not in result]])


def select_choices(scores, budgets, protocol):
    rows = []
    groups = {key: group for key, group in scores.groupby(EVENT_KEYS, sort=False)}
    for normal in budgets.itertuples(index=False):
        for event in protocol[protocol.climate.eq(normal.climate)].itertuples(index=False):
            row = {key: getattr(normal, key) for key in ["climate", "composition_id", "control", "normal_energy_reference_factor"]}
            row.update(event_id=event.event_id, baseline_status=normal.status,
                       selection_information="perfect future response and recovery from finite action library")
            if normal.status != "feasible":
                rows.append(row | {"status": normal.status})
                continue
            group = groups.get((normal.climate, normal.composition_id, normal.control, event.event_id), pd.DataFrame())
            if len(group) != 5 or not group.status.eq("evaluated").all():
                rows.append(row | {"status": "unresolved_engineering_operation"})
                continue
            valid = group[group.qualified_positive_event_response.fillna(False).astype(bool)]
            if valid.empty:
                rows.append(row | {"status": "infeasible_service_preserving_positive_response"})
                continue
            peak = valid.sort_values([POWER, "action_id"], ascending=[False, True]).iloc[0]
            positive_net = valid[valid[NET].gt(0)].sort_values([POWER, "action_id"], ascending=[False, True])
            net = positive_net.iloc[0] if len(positive_net) else None
            rows.append(row | {"status": "selected", "selected_action": peak.action_id,
                "baseline_facility_mean_kw": peak.baseline_facility_mean_kw,
                "absolute_event_facility_mean_kw": peak.absolute_action_facility_mean_kw,
                "event_power_reduction_kw": peak[POWER],
                "preconditioning_electricity_change_kwh": peak.preconditioning_facility_electricity_change_kwh,
                "recovery_electricity_change_kwh": peak.recovery_facility_electricity_change_kwh,
                "net_full_window_electricity_reduction_kwh": peak[NET],
                "positive_net_full_window": bool(peak[NET] > 0),
                "positive_net_selected_action": net.action_id if net is not None else None,
                "positive_net_event_power_kw": net[POWER] if net is not None else None,
                "positive_net_full_window_kwh": net[NET] if net is not None else None})
    return pd.DataFrame(rows, columns=CHOICE_COLUMNS)


def verify_table(actual, expected, keys):
    if not set(expected.columns) <= set(actual.columns):
        raise ValueError("Full replay columns differ")
    left = actual.set_index(keys, verify_integrity=True)
    right = expected.set_index(keys, verify_integrity=True)
    if set(left.index) != set(right.index):
        raise ValueError("Full replay inventory differs")
    maximum = 0.0
    for key, row in right.iterrows():
        for column, expected_value in row.items():
            value = left.loc[key, column] if column in left else np.nan
            if pd.api.types.is_scalar(value) and pd.api.types.is_scalar(expected_value) and pd.isna(value) and pd.isna(expected_value):
                continue
            if isinstance(expected_value, numbers.Number) and not isinstance(expected_value, (bool, np.bool_)):
                equal = np.isfinite(value) and np.isfinite(expected_value) and abs(value - expected_value) <= 1e-8
                if equal:
                    maximum = max(maximum, abs(float(value) - float(expected_value)))
            else:
                equal = str(value) == str(expected_value)
            if not equal:
                raise ValueError(f"Full replay differs: {key} {column}: {value} != {expected_value}")
    return maximum


def mean_actions(full, data, context, scorer):
    lookup = full.set_index(ACTION_KEYS, verify_integrity=True)
    rows = []
    for plan in data["plan"].itertuples(index=False):
        row = lookup.loc[tuple(getattr(plan, key) for key in ACTION_KEYS)].to_dict()
        row.update({key: getattr(plan, key) for key in ACTION_KEYS})
        if row["status"] == "evaluated":
            baseline, action, reference, profiles, tail, event = context(plan)
            means = zone_mean_profiles(profiles)
            for name, compact in [("baseline", baseline), ("action", action)]:
                masks = scorer._window_masks(compact, pd.Timestamp(event["start_timestamp"]),
                    int(event["duration_hours"]), int(event["recovery_hours"]), int(event.get("precondition_hours", 0)))
                for window, mask in zip(["precondition", "event", "recovery"], masks):
                    values = scorer.service_metrics(compact, means, mask, tail)
                    row.update({f"{name}_{window}_{metric}": value for metric, value in values.items()})
            # The reference_* fields above still contain the full replay scores.
            row = assess_service(row)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_representations(full, mean, full_choices, mean_choices):
    action = full[ACTION_KEYS + ["status", "service_qualified", "qualified_positive_event_response"]].merge(
        mean[ACTION_KEYS + ["service_qualified", "qualified_positive_event_response"]],
        on=ACTION_KEYS, suffixes=("_full", "_mean"), validate="one_to_one")
    for flag in ["service_qualified", "qualified_positive_event_response"]:
        evaluated = action.status.eq("evaluated")
        action[flag + "_mean_only"] = np.where(evaluated,
            action[flag + "_mean"].eq(True) & ~action[flag + "_full"].eq(True), np.nan)
        action[flag + "_full_only"] = np.where(evaluated,
            action[flag + "_full"].eq(True) & ~action[flag + "_mean"].eq(True), np.nan)
    full_groups = dict(tuple(full.groupby(EVENT_KEYS, sort=False)))
    mean_groups = dict(tuple(mean.groupby(EVENT_KEYS, sort=False)))
    full_by_action = full.set_index(ACTION_KEYS, verify_integrity=True)
    paired = full_choices.merge(mean_choices, on=CELL_KEYS, suffixes=("_full", "_mean"), validate="one_to_one")
    rows = []
    for cell in paired.to_dict("records"):
        key = tuple(cell[k] for k in EVENT_KEYS)
        fg = full_groups.get(key, full.iloc[:0])
        mg = mean_groups.get(key, mean.iloc[:0])
        support = cell["baseline_status_full"] == "feasible" and len(fg) == 5 and fg.status.eq("evaluated").all()
        base_full = bool(fg.baseline_service_qualified.eq(True).all()) if support else None
        base_mean = bool(mg.baseline_service_qualified.eq(True).all()) if support else None
        common_baseline = bool(support and base_full and base_mean)
        sets = {}
        for label, g in [("full", fg), ("mean", mg)]:
            sets[label] = set(g.loc[g.service_qualified.eq(True), "action_id"]) if support else None
            sets[label + "_positive"] = set(g.loc[g.qualified_positive_event_response.eq(True), "action_id"]) if support else None
        for policy, selected_column, power_column, net_column in [
            ("maximum_power", "selected_action", "event_power_reduction_kw", "net_full_window_electricity_reduction_kwh"),
            ("positive_net", "positive_net_selected_action", "positive_net_event_power_kw", "positive_net_full_window_kwh")]:
            row = {k: cell[k] for k in CELL_KEYS} | {"policy": policy,
                "normal_eligible": cell["baseline_status_full"] == "feasible",
                "normal_eligible_engineering_resolved_support": bool(support),
                "full_baseline_windows_qualified": base_full, "mean_baseline_windows_qualified": base_mean,
                "common_baseline_window_support": common_baseline,
                "service_action_sets_differ": sets["full"] != sets["mean"] if support else None,
                "positive_action_sets_differ": sets["full_positive"] != sets["mean_positive"] if support else None,
                "full_service_actions": json.dumps(sorted(sets["full"])) if support else None,
                "mean_service_actions": json.dumps(sorted(sets["mean"])) if support else None,
                "full_positive_actions": json.dumps(sorted(sets["full_positive"])) if support else None,
                "mean_positive_actions": json.dumps(sorted(sets["mean_positive"])) if support else None}
            for label in ["full", "mean"]:
                selected = cell.get(selected_column + "_" + label)
                selected = selected if pd.notna(selected) else None
                row[label + "_status"] = cell["status_" + label]
                row[label + "_policy_status"] = ("selected" if selected else "no_available_positive_net_response"
                    if support and policy == "positive_net" else cell["status_" + label])
                row[label + "_selected_action"] = selected
                row[label + "_available_power_kw"] = cell.get(power_column + "_" + label) if support and selected else (0.0 if support else np.nan)
                row[label + "_selected_net_electricity_kwh"] = cell.get(net_column + "_" + label) if selected else np.nan
            row["mean_minus_full_available_power_kw"] = row["mean_available_power_kw"] - row["full_available_power_kw"]
            selected = row["mean_selected_action"]
            checked = full_by_action.loc[(*key, selected)] if selected else None
            row["mean_selected_full_service_qualified"] = bool(checked.service_qualified) if selected else None
            row["mean_selected_fails_full_review"] = not bool(checked.service_qualified) if selected else None
            for stage in STAGES:
                row["mean_selected_full_" + stage + "_excess_c"] = checked[stage + "_excess_c"] if selected else np.nan
            rows.append(row)
    return action, pd.DataFrame(rows)


def summarize(pairs, grouping):
    rows = []
    for identity, group in pairs.groupby(grouping, dropna=False, sort=True):
        for subset, column in [("all_normal_available", "normal_eligible_engineering_resolved_support"),
                               ("common_baseline_windows", "common_baseline_window_support")]:
            supported = group[group[column]]
            rows.append(dict(zip(grouping, identity)) | {"support": subset, "planned_events": len(group),
                "normal_eligible_events": int(group.normal_eligible.sum()),
                "engineering_resolved_normal_events": int(group.normal_eligible_engineering_resolved_support.sum()),
                "supported_events": len(supported),
                "full_positive_response_events": int(supported.full_available_power_kw.gt(0).sum()),
                "mean_positive_response_events": int(supported.mean_available_power_kw.gt(0).sum()),
                "mean_selected_fails_full_review_events": int(supported.mean_selected_fails_full_review.eq(True).sum()),
                "service_action_set_difference_events": int(supported.service_action_sets_differ.eq(True).sum()),
                "full_available_power_mean_kw": supported.full_available_power_kw.mean(),
                "mean_available_power_mean_kw": supported.mean_available_power_kw.mean(),
                "paired_mean_minus_full_power_mean_kw": supported.mean_minus_full_available_power_kw.mean(),
                "full_selected_net_electricity_mean_kwh": supported.full_selected_net_electricity_kwh.mean(),
                "mean_selected_net_electricity_mean_kwh": supported.mean_selected_net_electricity_kwh.mean()})
    return pd.DataFrame(rows)


def review(out, data, context, scorer, inputs, sources, primary_assets=None):
    if len(data["plan"]) != EXPECTED_ACTIONS or len(data["choices"]) != EXPECTED_CELLS:
        raise ValueError("Incomplete declared response-information inventory")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the incomplete response-information attempt before retrying")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "EXECUTION_PLAN.json", {"action_rows": EXPECTED_ACTIONS, "budget_event_cells": EXPECTED_CELLS,
        "source_sha256": sources, "sequence": "Full explicit replay and receipt before any zone-mean score"})
    full = replay_actions(data, context, scorer)
    full.to_csv(out / "FULL_REPLAY_ACTION_SCORES.csv", index=False)
    score_delta = verify_table(full, data["scores"], ACTION_KEYS)
    full_choices = select_choices(full, data["budgets"], data["protocol"])
    full_choices.to_csv(out / "FULL_REPLAY_CHOICES.csv", index=False)
    choice_delta = verify_table(full_choices, data["choices"], CELL_KEYS)
    agreement = full_choices[CELL_KEYS].assign(all_original_fields_agree=True,
        maximum_power_policy_agrees=True, positive_net_policy_agrees=True)
    agreement.to_csv(out / "FULL_REPLAY_CHOICE_AGREEMENT.csv", index=False)
    verify_bindings(inputs, sources)
    write_json(out / "INPUT_SHA256.json", inputs)
    full_outputs = ["FULL_REPLAY_ACTION_SCORES.csv", "FULL_REPLAY_CHOICES.csv", "FULL_REPLAY_CHOICE_AGREEMENT.csv"]
    write_json(out / "FULL_REPLAY_RECEIPT.json", {"status": "completed", "action_rows": len(full),
        "explicit_evaluated_actions": int(full.status.eq("evaluated").sum()), "budget_event_cells": len(full_choices),
        "policy_cell_agreements": 2 * len(full_choices), "maximum_numeric_score_difference": score_delta,
        "maximum_numeric_choice_difference": choice_delta, "zone_mean_scores_computed": 0,
        "source_sha256": sources, "input_sha256": inputs,
        "output_sha256": {name: digest(out / name) for name in full_outputs}})
    # No call using averaged centers occurs before the persisted receipt above.
    mean = mean_actions(full, data, context, scorer)
    mean.to_csv(out / "MEAN_REVIEW_ACTION_SCORES.csv", index=False)
    mean_choices = select_choices(mean, data["budgets"], data["protocol"])
    mean_choices.to_csv(out / "MEAN_REVIEW_CHOICES.csv", index=False)
    disagreement, pairs = compare_representations(full, mean, full_choices, mean_choices)
    disagreement.to_csv(out / "ACTION_QUALIFICATION_DISAGREEMENTS.csv", index=False)
    pairs.to_csv(out / "PAIRED_EVENT_INFORMATION_REVIEW.csv", index=False)
    grouping = ["control", "normal_energy_reference_factor", "policy"]
    for suffix, extra in [("ALL_SETTINGS", []), ("BY_CLIMATE", ["climate"]), ("BY_COMPOSITION", ["composition_id"])]:
        summarize(pairs, grouping + extra).to_csv(out / ("INFORMATION_SUMMARY_" + suffix + ".csv"), index=False)
    verify_bindings(inputs, sources)
    write_json(out / "AUDIT.json", {"status": "completed", "EnergyPlus_runs": 0,
        "action_rows_per_representation": len(full), "budget_event_cells_per_representation": len(full_choices),
        "paired_policy_cells": len(pairs), "source_sha256": sources, "primary_assets": primary_assets,
        "service_tolerance_c": 1e-9, "positive_event_threshold_kwh": 1e-9,
        "positive_net_subset_threshold_kwh_strict": 0,
        "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        "common_baseline_subset": "Normal-eligible and engineering-resolved; all five actions' original baseline windows pass in both representations",
        "repeated_evaluations": "Menus and annual energy budgets can share physical operations; they are not independent buildings",
        "scope": "Nominal fixed-normal fixed-action-menu information review; no mean-selected uncertainty or placement response experiment"})


def main(base=None, out=None, protocol_path=None):
    out = out or ROOT / "results/dr_information_review"
    base = base or ROOT / "results/dr_capability"
    paths = {name: base / filename for name, filename in [
        ("capability", "CAPABILITY_AUDIT.json"), ("plan_audit", "PLAN_AUDIT.json"),
        ("plan", "SINGLE_EVENT_ACTION_PLAN.csv"), ("completed", "ACTION_FULL_COMPLETED.csv"),
        ("normals", "NORMAL_OPERATION_REPLAY_COMPLETED.csv"), ("budgets", "NORMAL_OPERATION_BUDGET_STATUS.csv"),
        ("scores", "ISOLATED_EVENT_ACTION_SCORES.csv"), ("choices", "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")]}
    paths.update(protocol=protocol_path or RAW_RUN / "response/EVENT_PROTOCOL.csv", profiles=MATRIX / "GLOBAL_PROFILES.csv",
        probes=MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv",
        tails=ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
        amendment=CODE / "data/metadata/strengthening_protocol/response_information.yaml")
    inputs = {}

    def tracked(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(CODE):
            raise ValueError("Response-information input is outside this project")
        name = path.relative_to(CODE).as_posix()
        if name not in inputs:
            inputs[name] = digest(path)
        if expected is not None and inputs[name] != expected:
            raise ValueError("Declared response-information input hash differs: " + name)
        return path

    primary_assets = {name: {"path": str(tracked(path).relative_to(CODE).as_posix()),
                             "sha256": inputs[Path(path).resolve().relative_to(CODE).as_posix()]}
                      for name, path in paths.items()}
    if verified_cache(out, primary_assets):
        print("Response information: verified completed cache", flush=True)
        return
    capability = json.loads(paths["capability"].read_text())
    if capability["status"] not in {"completed", "partial"} or capability["logical_event_actions"] != EXPECTED_ACTIONS or capability["budget_event_cells"] != EXPECTED_CELLS:
        raise ValueError("Complete the formal capability scores and choices first")
    plan_audit = json.loads(paths["plan_audit"].read_text())
    tracked(paths["protocol"], plan_audit["event_protocol_sha256"])
    data = {name: pd.read_csv(paths[name]) for name in ["plan", "completed", "budgets", "protocol", "scores", "choices"]}
    normals = pd.read_csv(paths["normals"]).set_index("normal_id", verify_integrity=True)
    completed = data["completed"].set_index("run_id", verify_integrity=True)
    probes = pd.read_csv(paths["probes"])
    probes = probes[probes.target_c.eq(25)].set_index("climate", verify_integrity=True)
    profiles, members = pd.read_csv(paths["profiles"]), pd.read_csv(paths["tails"])

    @lru_cache(maxsize=32)
    def load(path, expected):
        path = tracked(path, expected)
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)

    def context(plan):
        done, normal, probe = completed.loc[plan.run_id], normals.loc[plan.normal_id], probes.loc[plan.climate]
        qa = validate_context_binding(plan, done, normal)
        tracked(normal.states_path, normal.states_sha256)
        tracked(plan.input_idf_path, plan.input_sha256)
        tracked(plan.weather_path, qa["weather_sha256"])
        action = load(str(done.hourly_path), str(done.hourly_sha256))
        n = len(action["month"])

        def truncate(values):
            return {k: v[:n] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == 8760 else v
                    for k, v in values.items()}

        baseline = truncate(load(str(normal.hourly_path), str(normal.hourly_sha256)))
        reference = truncate(load(str(probe.hourly_path), str(probe.hourly_sha256)))
        g = profiles[profiles.context.eq(plan.climate) & profiles.composition_id.eq(plan.composition_id)]
        tail = set(members[members.climate.eq(plan.climate) & members.composition.eq(plan.composition)].profile_id.astype(str))
        if len(g) != 267 or g.profile_id.duplicated().any() or len(tail) != 27 or not tail <= set(g.profile_id.astype(str)):
            raise ValueError("Frozen profile identities or annual27tail changed")
        if set(g.zone_id.astype(str)) != set(baseline["zone_ids"].astype(str)):
            raise ValueError("Frozen native-zone assignments changed")
        return baseline, action, reference, g, tail, json.loads(plan.event_json)

    sources = source_hashes()
    scorer = explicit_scorer()
    verify_bindings(inputs, sources)
    review(out, data, context, scorer, inputs, sources, primary_assets)


if __name__ == "__main__":
    main()
