"""Paired full-service review of frozen full and zone-mean event selections."""

from common import *
from functools import lru_cache
import inspect
import numpy as np
import pandas as pd
import dr_information_review as information
from dr_selected_uncertainty import window_score
from paired_uncertainty import keyed_normal, shared_normal, empirical_temporal_shifts, DRAWS
from profile_identity import person_keys
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles, load_hourly
import multi_event_dr_extension as dr


INPUTS = {}
SOURCES = {}
METRICS = information.METRICS
CONTEXT_KEYS = ["climate", "composition_id", "normal_id", "run_id", "event_id", "event_json"]


def tracked(path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(CODE):
        raise ValueError("Frozen-selection input is outside the independent project")
    name = path.relative_to(CODE).as_posix()
    if name not in INPUTS:
        INPUTS[name] = digest(path)
    if expected is not None and INPUTS[name] != expected:
        raise ValueError("Frozen-selection input hash differs: " + name)
    return path


def read(path):
    return pd.read_csv(tracked(path), float_precision="round_trip")


def verify_bindings():
    for name, expected in {**INPUTS, **SOURCES}.items():
        if digest(CODE / name) != expected:
            raise ValueError("Frozen information uncertainty source or input changed: " + name)


def primary_assets(paths):
    return {name: {"path": str(tracked(path).relative_to(CODE).as_posix()),
                   "sha256": INPUTS[Path(path).resolve().relative_to(CODE).as_posix()]}
            for name, path in paths.items()}


def expected_window_pass(saved, name):
    window = name.split("_", 1)[1]
    values = np.array([saved[name + "_" + metric] for metric in METRICS])
    limits = np.array([saved["reference_" + window + "_" + metric] for metric in METRICS])
    return bool(np.isfinite(values).all() and np.isfinite(limits).all() and (values <= limits + 1e-9).all())


@lru_cache(maxsize=8)
def hourly(path, expected):
    return load_hourly(tracked(path, expected), require_facility=False)


def demand_draws(g):
    nominal = g.centre_c.to_numpy()
    source = "ASHRAE" if "Multinational" in g.composition.iloc[0] else "CTCD"
    individual = np.column_stack([keyed_normal("person|" + key) for key in person_keys(g.profile_id)])
    result = []
    for prefix, shared in [("sourceSD", keyed_normal("source|" + source)[:, None]),
                           ("studyGroupSD", shared_normal(g.profile_id, source, study_shared=True))]:
        result.extend((f"{prefix}{s:g}_personSD{p:g}", nominal[None, :] + s * shared + p * individual)
                      for s in [0, .25, .5] for p in [0, .25, .5])
    if source == "ASHRAE":
        for prefix, single in [("observed_temporal_proxy", False), ("single_record_temporal_proxy", True)]:
            shifts = empirical_temporal_shifts(g.profile_id, single_record=single)
            result.extend((f"{prefix}_shift_fraction{f:g}", nominal[None, :] + f * shifts) for f in [.5, 1.])
    return result


def checked_scores(candidate, reference, centers, tail, annual=False):
    score = (lambda curve: curve.score(centers, tail)) if annual else (
        lambda curve: window_score(curve, centers, tail))
    population, least = score(candidate)
    rp, rt = score(reference)
    observed = np.column_stack([population, least])
    limits = np.column_stack([rp, rt])
    if not np.isfinite(observed).all() or not np.isfinite(limits).all():
        raise ValueError("Nonfinite service value in an assessed window")
    passed = (observed <= limits + 1e-9).all(axis=1)
    excess = np.maximum(observed - limits, 0).max(axis=1)
    return passed, excess, observed, limits


def frozen_power(selected, resolved, mask, power):
    if not resolved:
        return np.full(DRAWS, np.nan)
    if not selected:
        return np.zeros(DRAWS)
    return np.asarray(mask, dtype=float) * power


def choice_inventory(pairs, plan):
    lookup = plan.set_index(information.ACTION_KEYS, verify_integrity=True)
    contexts, mapping = {}, []
    for cell in pairs.to_dict("records"):
        for representation in ["full", "mean"]:
            action = cell[representation + "_selected_action"]
            if pd.isna(action):
                continue
            key = tuple(cell[k] for k in information.EVENT_KEYS) + (action,)
            operation = lookup.loc[key].to_dict() | dict(zip(information.ACTION_KEYS, key))
            identity = {k: operation[k] for k in CONTEXT_KEYS}
            context_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]
            contexts[context_id] = operation | {"context_id": context_id}
            mapping.append({k: cell[k] for k in information.CELL_KEYS} | {
                "representation": representation, "action_id": action, "context_id": context_id})
    return pd.DataFrame(contexts.values()), pd.DataFrame(mapping)


def verify_old_full(summary, windows, mapping, old_summary, old_windows):
    aliases = mapping[mapping.representation.eq("full")].drop_duplicates(
        information.EVENT_KEYS + ["action_id"])
    alias_keys = ["climate", "composition_id", "control", "event_id", "action_id", "context_id"]
    new = aliases[alias_keys].merge(summary, on="context_id", validate="many_to_many", suffixes=("", "_context"))
    new_windows = aliases[alias_keys].merge(windows, on="context_id", validate="many_to_many", suffixes=("", "_context"))
    keys = ["climate", "composition", "control", "event_id", "action_id", "scenario"]
    checks = []
    for label, expected, actual, extra, fields in [
        ("operation", old_summary, new, [], ["conditional_any_service_violation_frequency",
            "maximum_exceedance_median_c", "maximum_exceedance_p95_c",
            "event_power_kw_fixed_operation", "net_electricity_kwh_fixed_operation"]),
        ("window", old_windows, new_windows, ["window"], ["conditional_violation_frequency",
            "exceedance_median_c", "exceedance_p95_c"])]:
        index = keys + extra
        a, b = expected.set_index(index).sort_index(), actual.set_index(index).sort_index()
        if not a.index.equals(b.index):
            raise ValueError("Original full-selected uncertainty inventory differs: " + label)
        for field in fields:
            error = np.max(np.abs(a[field].to_numpy() - b[field].to_numpy()))
            tolerance = 0 if "frequency" in field else 1e-10
            if not np.isfinite(error) or error > tolerance:
                raise ValueError("Original full-selected uncertainty closure failed: " + field)
            checks.append({"level": label, "metric": field, "rows": len(a), "maximum_difference": error})
    return pd.DataFrame(checks)


def paired_summaries(pairs, mapping, summaries, masks):
    choice_map = mapping.set_index(information.CELL_KEYS + ["representation"], verify_integrity=True)
    all_scenarios = sorted(summaries.scenario.unique())
    draws, grouped = [], []
    for (control, budget), cells in pairs.groupby(["control", "normal_energy_reference_factor"], sort=True):
        for subset, flag in [("all_normal_available", "normal_eligible_engineering_resolved_support"),
                             ("common_baseline_windows", "common_baseline_window_support")]:
            for stratum, groups in [("all", [("all", cells)]),
                                   ("climate", cells.groupby("climate", sort=True)),
                                   ("composition", cells.groupby("composition_id", sort=True))]:
                for value, group in groups:
                    resolved = group[group[flag]]
                    scenarios = all_scenarios if len(resolved) else []
                    # Ineligible groups retain a missing summary, not a fictitious zero capacity.
                    if not scenarios:
                        grouped.append({"control": control, "budget": budget, "support": subset,
                            "stratum": stratum, "stratum_value": value, "scenario": "not_applicable",
                            "planned_events": len(group), "supported_events": len(resolved)})
                    for scenario in scenarios:
                        current = resolved[resolved.composition_id.isin(
                            resolved.composition_id if "temporal_proxy" not in scenario else
                            summaries.loc[summaries.scenario.eq(scenario), "composition_id"].unique())]
                        powers = {rep: [] for rep in ["full", "mean"]}
                        claimed = []
                        for cell in current.to_dict("records"):
                            key = tuple(cell[k] for k in information.CELL_KEYS)
                            claimed.append(cell["mean_available_power_kw"])
                            for rep in powers:
                                selected = pd.notna(cell[rep + "_selected_action"])
                                context = choice_map.loc[key + (rep,), "context_id"] if selected else None
                                powers[rep].append(frozen_power(selected, True,
                                    masks[context, scenario] if selected else None, cell[rep + "_available_power_kw"]))
                        if not len(current):
                            continue
                        full, mean = (np.mean(powers[rep], axis=0) for rep in ["full", "mean"])
                        difference = mean - full
                        row = {"control": control, "budget": budget, "support": subset,
                            "stratum": stratum, "stratum_value": value, "scenario": scenario,
                            "planned_events": len(group), "supported_events": len(current),
                            "nominal_mean_claimed_power_kw": float(np.mean(claimed)),
                            "full_selection_full_checked_power_mean_kw": float(full.mean()),
                            "mean_selection_full_checked_power_mean_kw": float(mean.mean()),
                            "paired_mean_minus_full_power_mean_kw": float(difference.mean()),
                            "paired_mean_minus_full_power_p05_kw": float(np.quantile(difference, .05)),
                            "paired_mean_minus_full_power_p95_kw": float(np.quantile(difference, .95))}
                        grouped.append(row)
                        if stratum == "all":
                            draws.extend({k: row[k] for k in ["control", "budget", "support", "scenario", "supported_events"]} | {
                                "draw": i, "full_selection_full_checked_power_kw": full[i],
                                "mean_selection_full_checked_power_kw": mean[i],
                                "paired_mean_minus_full_power_kw": difference[i]} for i in range(DRAWS))
    return pd.DataFrame(grouped), pd.DataFrame(draws)


def main():
    INPUTS.clear()
    SOURCES.clear()
    for function in [main, information.verified_cache, window_score, ServiceDistribution, normalise_profiles,
                     load_hourly, keyed_normal, shared_normal, empirical_temporal_shifts, person_keys, dr._window_masks, digest]:
        path = Path(inspect.getfile(function)).resolve()
        SOURCES[path.relative_to(CODE).as_posix()] = digest(path)
    out = ROOT / "results/dr_information_uncertainty"
    info, capability = ROOT / "results/dr_information_review", ROOT / "results/dr_capability"
    info_paths = {name: capability / filename for name, filename in [
        ("capability", "CAPABILITY_AUDIT.json"), ("plan_audit", "PLAN_AUDIT.json"),
        ("plan", "SINGLE_EVENT_ACTION_PLAN.csv"), ("completed", "ACTION_FULL_COMPLETED.csv"),
        ("normals", "NORMAL_OPERATION_REPLAY_COMPLETED.csv"), ("budgets", "NORMAL_OPERATION_BUDGET_STATUS.csv"),
        ("scores", "ISOLATED_EVENT_ACTION_SCORES.csv"), ("choices", "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")]}
    info_paths.update(protocol=RAW_RUN / "response/EVENT_PROTOCOL.csv", profiles=MATRIX / "GLOBAL_PROFILES.csv",
        probes=MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv", tails=ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
        amendment=CODE / "data/metadata/strengthening_protocol/response_information.yaml")
    current_info_assets = primary_assets(info_paths)
    current_assets = primary_assets(info_paths | {"information_receipt": info / "AUDIT.json",
        "information_pairs": info / "PAIRED_EVENT_INFORMATION_REVIEW.csv",
        "full_replay": info / "FULL_REPLAY_ACTION_SCORES.csv",
        "old_uncertainty": ROOT / "results/dr_capability_uncertainty_identity/AUDIT.json",
        "amendment36": CODE / "data/metadata/strengthening_protocol/frozen_selection_uncertainty.yaml"})
    if (out / "AUDIT.json").is_file():
        audit = json.loads((out / "AUDIT.json").read_text())
        if (audit["status"] != "completed" or audit["source_sha256"] != SOURCES
                or audit.get("primary_assets") != current_assets):
            raise ValueError("Frozen information uncertainty cache source or current primary inputs differ")
        for name, sha in audit["input_sha256"].items():
            tracked(CODE / name, sha)
        for name, sha in audit["output_sha256"].items():
            if digest(out / name) != sha:
                raise ValueError("Frozen information uncertainty cache output differs")
        print("Frozen information uncertainty: verified completed cache", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the incomplete frozen information uncertainty attempt")
    if not information.verified_cache(info, current_info_assets):
        raise ValueError("Complete the nominal information review before paired uncertainty")
    for path, sha in json.loads((info / "INPUT_SHA256.json").read_text()).items():
        tracked(CODE / path, sha)
    tracked(info / "AUDIT.json")
    tracked(CODE / "data/metadata/strengthening_protocol/frozen_selection_uncertainty.yaml")
    pairs = read(info / "PAIRED_EVENT_INFORMATION_REVIEW.csv")
    pairs = pairs[pairs.policy.eq("maximum_power")].copy()
    plan = read(capability / "SINGLE_EVENT_ACTION_PLAN.csv")
    contexts, mapping = choice_inventory(pairs, plan)
    logical_counts = {rep: len(group.drop_duplicates(information.ACTION_KEYS))
                      for rep, group in mapping.groupby("representation")}
    union_count = len(mapping.drop_duplicates(information.ACTION_KEYS))
    if (len(pairs) != 3888 or len(contexts) != 441 or union_count != 1276
            or logical_counts != {"full": 336, "mean": 1270}):
        raise ValueError("Frozen maximum-power selection inventory is incomplete")
    profiles = normalise_profiles(read(MATRIX / "GLOBAL_PROFILES.csv"))
    members = read(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    probes = read(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    probes = probes[probes.target_c.eq(25)].set_index("climate", verify_integrity=True)
    normals = read(capability / "NORMAL_OPERATION_REPLAY_COMPLETED.csv").set_index("normal_id", verify_integrity=True)
    actions = read(capability / "ACTION_FULL_COMPLETED.csv").set_index("run_id", verify_integrity=True)
    full_scores = read(info / "FULL_REPLAY_ACTION_SCORES.csv").set_index(information.ACTION_KEYS, verify_integrity=True)
    for relative in ["audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz",
                     "results/demand_validation/temporal_centers/DEVELOPMENT_EMPIRICAL_TEMPORAL_SHIFT_POOL.csv",
                     "results/demand_validation/single_record/timed/CONTINUOUS_PROXY_ERRORS.csv"]:
        tracked(ROOT / relative)
    old = ROOT / "results/dr_capability_uncertainty_identity"
    old_audit = json.loads(tracked(old / "AUDIT.json").read_text())
    old_summary, old_windows = (read(old / name) for name in [
        "FROZEN_ACTION_SERVICE_VIOLATIONS.csv", "SERVICE_WINDOW_VIOLATIONS.csv"])
    for name, sha in old_audit["output_sha256"].items():
        tracked(old / name, sha)
    out.mkdir(parents=True, exist_ok=True)
    contexts.to_csv(out / "SCORING_CONTEXTS.csv", index=False)
    mapping.to_csv(out / "FROZEN_CHOICE_MAPPING.csv", index=False)
    pairs.to_csv(out / "ALL_NOMINAL_CHOICE_STATUS.csv", index=False)
    write_json(out / "EXECUTION_PLAN.json", {"additional_EnergyPlus_runs": 0, "contexts": len(contexts),
        "logical_selected_counts": logical_counts, "logical_union": union_count,
        "policy": "maximum_power", "draws": DRAWS, "source_sha256": SOURCES,
        "sequence": "Persist complete nominal full-service reproduction receipt before nonzero draws"})
    curves, tails, nominal_rows = {}, {}, []
    annual = {}
    for (climate, comp), group in contexts.groupby(["climate", "composition_id"], sort=True):
        g = profiles[profiles.climate.eq(climate) & profiles.composition_id.eq(comp)].sort_values("profile_id").reset_index(drop=True)
        tail = np.flatnonzero(g.profile_id.isin(members[members.climate.eq(climate) & members.composition.eq(g.composition.iloc[0])].profile_id))
        if len(g) != 267 or len(tail) != 27:
            raise ValueError("Frozen demand identities or tail changed")
        reference = hourly(str(probes.loc[climate, "hourly_path"]), str(probes.loc[climate, "hourly_sha256"]))
        refannual = ServiceDistribution(reference, g)
        centers = g.centre_c.to_numpy()[None, :]
        for operation in group.itertuples():
            normal, action = normals.loc[operation.normal_id], actions.loc[operation.run_id]
            information.validate_context_binding(operation, action, normal)
            tracked(normal.states_path, normal.states_sha256)
            baseline = hourly(str(normal.hourly_path), str(normal.hourly_sha256))
            trajectory = hourly(str(action.hourly_path), str(action.hourly_sha256))
            event = json.loads(operation.event_json)
            masks = dr._window_masks(reference, pd.Timestamp(event["start_timestamp"]), int(event["duration_hours"]),
                int(event["recovery_hours"]), int(event.get("precondition_hours", 0)))
            akey = (climate, comp, operation.normal_id)
            if akey not in annual:
                annual[akey] = ServiceDistribution(baseline, g)
            current = {"annual": (annual[akey], refannual)}
            saved = full_scores.loc[tuple(getattr(operation, k) for k in information.ACTION_KEYS)]
            for name, mask in zip(["precondition", "event", "recovery"], masks):
                occupied = float((reference["people"] * (reference["occupied"] & mask)[:, None]).sum())
                if occupied == 0:
                    if name != "precondition" or saved.reference_precondition_weighted_profile_hours != 0:
                        raise ValueError("Required event or recovery window has no occupied support")
                    nominal_rows.extend({"context_id": operation.context_id, "window": prefix + name,
                        "status": "not_applicable", "maximum_score_difference_c": np.nan, "passed": np.nan}
                        for prefix in ["baseline_", "action_"])
                    continue
                ref = ServiceDistribution(reference, g, mask)
                current["baseline_" + name] = (ServiceDistribution(baseline, g, mask), ref)
                current["action_" + name] = (ServiceDistribution(trajectory, g, mask), ref)
            all_windows = True
            for name, (candidate, ref) in current.items():
                passed, excess, values, limits = checked_scores(candidate, ref, centers, tail, name == "annual")
                error = 0.
                if name != "annual":
                    window = name.split("_", 1)[1]
                    expected = np.array([saved[name + "_" + metric] for metric in METRICS])
                    expected_ref = np.array([saved["reference_" + window + "_" + metric] for metric in METRICS])
                    error = float(max(np.max(abs(values[0] - expected)), np.max(abs(limits[0] - expected_ref))))
                    if bool(passed[0]) != expected_window_pass(saved, name):
                        raise ValueError("Frozen nominal per-window qualification differs: " + name)
                    all_windows &= bool(passed[0])
                elif not passed.all():
                    raise ValueError("Frozen annual normal nominal service differs")
                if error > 1e-10:
                    raise ValueError("Frozen union-selected nominal score differs")
                nominal_rows.append({"context_id": operation.context_id, "window": name, "status": "assessed",
                    "maximum_score_difference_c": error, "passed": bool(passed[0])})
            if all_windows != bool(saved.service_qualified):
                raise ValueError("Frozen union-selected nominal qualification differs")
            curves[operation.context_id], tails[operation.context_id] = current, tail
    pd.DataFrame(nominal_rows).to_csv(out / "NOMINAL_REPRODUCTION.csv", index=False)
    verify_bindings()
    write_json(out / "NOMINAL_REPRODUCTION_RECEIPT.json", {"status": "completed", "contexts": len(curves),
        "mean_selected_failures_retained": True, "nonzero_draws_computed": 0,
        "input_sha256": INPUTS, "source_sha256": SOURCES,
        "maximum_score_difference_c": pd.DataFrame(nominal_rows).maximum_score_difference_c.max()})
    print("NOMINAL_UNION_REPRODUCTION_COMPLETE", len(curves), flush=True)
    summary, window_rows, window_masks, pass_masks, mask_index = [], [], [], {}, []
    for (climate, comp), group in contexts.groupby(["climate", "composition_id"], sort=True):
        g = profiles[profiles.climate.eq(climate) & profiles.composition_id.eq(comp)].sort_values("profile_id").reset_index(drop=True)
        scenarios = demand_draws(g)
        for operation in group.itertuples():
            saved = full_scores.loc[tuple(getattr(operation, k) for k in information.ACTION_KEYS)]
            for label, centers in scenarios:
                passed_all, maximum = np.ones(DRAWS, bool), np.zeros(DRAWS)
                base = {"context_id": operation.context_id, "climate": climate, "composition_id": comp,
                    "composition": g.composition.iloc[0], "scenario": label}
                for name, (candidate, ref) in curves[operation.context_id].items():
                    passed, excess, _, _ = checked_scores(candidate, ref, centers, tails[operation.context_id], name == "annual")
                    passed_all &= passed
                    maximum = np.maximum(maximum, excess)
                    window_rows.append(base | {"window": name, "conditional_violation_frequency": float((~passed).mean()),
                        "exceedance_median_c": float(np.median(excess)), "exceedance_p95_c": float(np.quantile(excess, .95)),
                        "window_mask_row": len(window_masks)})
                    window_masks.append(passed)
                pass_masks[operation.context_id, label] = passed_all
                mask_index.append(base | {"mask_row": len(mask_index)})
                summary.append(base | {"conditional_any_service_violation_frequency": float((~passed_all).mean()),
                    "maximum_exceedance_median_c": float(np.median(maximum)),
                    "maximum_exceedance_p95_c": float(np.quantile(maximum, .95)),
                    "event_power_kw_fixed_operation": saved[information.POWER],
                    "net_electricity_kwh_fixed_operation": saved[information.NET]})
        print("FROZEN_INFORMATION_PAIRED", climate, comp, len(group), flush=True)
    summary, window_rows = pd.DataFrame(summary), pd.DataFrame(window_rows)
    closure = verify_old_full(summary, window_rows, mapping, old_summary, old_windows)
    closure.to_csv(out / "ORIGINAL_FULL_UNCERTAINTY_CLOSURE.csv", index=False)
    summary.to_csv(out / "SELECTED_OPERATION_SERVICE_UNCERTAINTY.csv", index=False)
    window_rows.to_csv(out / "SELECTED_WINDOW_SERVICE_UNCERTAINTY.csv", index=False)
    pd.DataFrame(mask_index).to_csv(out / "PASS_MASK_INDEX.csv", index=False)
    np.savez_compressed(out / "PAIRED_PASS_MASKS.npz", passed=np.array(list(pass_masks.values()), dtype=bool),
        window_passed=np.array(window_masks, dtype=bool))
    aggregate, draws = paired_summaries(pairs, mapping, summary, pass_masks)
    aggregate.to_csv(out / "PAIRED_FROZEN_POWER_SUMMARY.csv", index=False)
    draws.to_csv(out / "PAIRED_FROZEN_POWER_DRAWS.csv", index=False)
    verify_bindings()
    write_json(out / "AUDIT.json", {"status": "completed", "additional_EnergyPlus_runs": 0,
        "contexts": len(contexts), "scenario_rows": len(summary), "draws": DRAWS,
        "scope": "A fixed maximum-power selections; both representations checked against paired full demand; no reselection",
        "primary_assets": current_assets,
        "source_sha256": SOURCES, "input_sha256": INPUTS,
        "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
        "interpretation": "Scenario-conditional frozen-operation availability, not robust optimization or real discomfort probabilities"})
    record_status("dr_information_paired_uncertainty", "completed", {"contexts": len(contexts), "scenario_rows": len(summary)})


if __name__ == "__main__":
    main()
