"""Replay supplementary DR date checks from the normalized hourly sidecar.

Run from the merged repository. All physical outputs are retained inputs;
this command runs no EnergyPlus simulation and changes no physical parameter.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "src/analysis/energy_service"))
from analyse_events import event_indices, score_window, validate_event_pair  # noqa: E402
from bundle import verify_bundle  # noqa: E402


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, item):
    path.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def contained(root, relative):
    logical = PurePosixPath(relative)
    if logical.is_absolute() or ".." in logical.parts or "\\" in relative:
        raise ValueError("Input locators must be contained relative paths")
    path = root.joinpath(*logical.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Input locator escapes its declared root")
    return path


@lru_cache(maxsize=48)
def read(path):
    with np.load(path, allow_pickle=False) as stored:
        data = dict(stored)
    if data["temperatures"].shape != data["people"].shape or data["temperatures"].shape[0] != 8760:
        raise ValueError("Incomplete retained annual trajectory")
    return data


def verify_inputs(data, annual, supplement):
    record = load(REPOSITORY / "data/metadata/typical_day_dr_bundle_source.json")
    manifest = REPOSITORY / "data/metadata/typical_day_dr_bundle_manifest.csv"
    if digest(manifest) != record["manifest_sha256"] or digest(supplement / "MANIFEST.csv") != record["manifest_sha256"]:
        raise ValueError("Supplementary input manifest does not match the repository record")
    verified = verify_bundle(supplement, supplement / "MANIFEST.csv")
    contract = load(supplement / "protocol/INPUT_CONTRACT.json")
    if digest(data / "MANIFEST.csv") != contract["paper_manifest_sha256"]:
        raise ValueError("The existing paper input package changed")
    for relative, expected in contract["paper_files_sha256"].items():
        if digest(contained(data, relative)) != expected:
            raise ValueError(f"Paper input changed: {relative}")
    analysis = load(annual / "ANALYSIS_MANIFEST.json")
    if analysis.get("status") != "PASS" or analysis.get("physical_parameters_changed"):
        raise ValueError("Fresh annual replay has not passed")
    for relative, expected in contract["annual_outputs_sha256"].items():
        if digest(contained(annual, relative)) != expected or analysis["output_sha256"][relative] != expected:
            raise ValueError(f"Fresh annual replay differs from the frozen baseline decision: {relative}")
    return {"status": "PASS", "supplement_files": verified["verified_input_files"],
            "supplement_manifest_sha256": record["manifest_sha256"],
            "paper_manifest_sha256": contract["paper_manifest_sha256"],
            "verified_paper_input_files": len(contract["paper_files_sha256"]),
            "fresh_annual_outputs_sha256": contract["annual_outputs_sha256"],
            "physical_simulations_run": 0}


def profile_groups(data, annual):
    profiles = pd.read_csv(data / "engineering/PROFILE_ASSIGNMENTS.csv.gz")
    profiles = profiles[profiles.composition_id.eq(2026102001) & profiles.layout.eq("random_mixing")]
    tails = pd.read_csv(annual / "FIXED_TAIL_MEMBERSHIPS.csv")
    tails = tails[tails.composition_id.eq(2026102001) & tails.layout.eq("random_mixing")]
    anchors = pd.read_csv(annual / "ANNUAL_ANCHORS.csv")
    anchors = anchors[anchors.composition_id.eq(2026102001) & anchors.layout.eq("random_mixing")]
    groups = {}
    for context in sorted(profiles.context.unique()):
        group = profiles[profiles.context.eq(context)].sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        tail = np.flatnonzero(group.profile_id.isin(tails[tails.context.eq(context)].profile_id))
        if len(group) != 267 or len(tail) != 27:
            raise ValueError("Profile placement or fixed annual tail identities changed")
        groups[context] = group, tail, anchors[anchors.context.eq(context)].iloc[0].anchor_run_id
    return groups


def verify_dates(data, supplement, groups):
    dates = load(supplement / "protocol/DATES_BEFORE_RUNS.json")
    original = load(data / "engineering/EVENT_DATES_BEFORE_RUNS.json")
    for context, (group, tail, anchor_id) in groups.items():
        clock = pd.read_csv(supplement / "source_clocks" / f"{context}.csv")
        common = read(str(data / "hourly" / f"{anchor_id}.npz"))
        if len(clock) != 8760 or not clock.Interval.eq(60).all():
            raise ValueError("Source calendar is incomplete")
        for key, column in [("month", "Month"), ("day", "Day"), ("hour", "Hour"), ("dst", "Dst")]:
            if not np.array_equal(common[key], clock[column]):
                raise ValueError("Source calendar differs from the retained annual anchor")
        civil = common["hour"] - 1 + common["dst"]
        rows = []
        for day in sorted(set(common["day"][common["month"] == 7])):
            mask = (common["month"] == 7) & (common["day"] == day) & (civil >= 14) & (civil < 16)
            if mask.sum() != 2 or not common["occupied"][mask].all():
                continue
            if not clock.loc[mask, "DayType"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).all():
                continue
            energy = float(common["hvac_electric_kwh"][mask].sum())
            if energy > 0:
                rows.append({"day": int(day), "energy": energy})
        pool = pd.DataFrame(rows)
        original_day = next(event["day"] for event in original[context] if event["month"] == 7)
        maximum = pool.sort_values(["energy", "day"], ascending=[False, True]).iloc[0].day
        if original_day != maximum:
            raise ValueError("Original headline date is not the source maximum")
        chosen_days = []
        for date_class, quantile in [("median", .5), ("high90_nonmaximum", .9)]:
            target = float(pool.energy.quantile(quantile, interpolation="linear"))
            eligible = pool[pool.day.ne(original_day)].copy()
            if quantile == .9:
                eligible = eligible[eligible.day.ne(maximum) & ~eligible.day.isin(chosen_days)]
            eligible["distance"] = abs(eligible.energy - target)
            frozen = next(event for event in dates[context] if event["date_class"] == date_class)
            locked = eligible[eligible.day.eq(frozen["day"])]
            if len(locked) != 1:
                raise ValueError("Prelocked date is no longer eligible under its source calendar")
            locked = locked.iloc[0]
            # CSV round trips can reverse floating-point median ties. The
            # pre-run date is authoritative within the locked numerical tolerance.
            if locked.distance > eligible.distance.min() + 1e-12 or abs(locked.energy - frozen["common25_event_electric_kwh"]) > 1e-12:
                raise ValueError("Frozen extra-day rule does not reproduce from source inputs")
            chosen_days.append(int(frozen["day"]))
    return {"status": "PASS", "contexts": len(groups), "added_dates": 8,
            "quantiles_recomputed_from_frozen_common25": True, "day_type_from_source_clock": True,
            "prelocked_dates_authoritative_with_nearest_distance_tolerance_kwh": 1e-12}


def normalized_designs(data, annual, supplement):
    designs = load(supplement / "protocol/DESIGNS.json")
    results = load(supplement / "protocol/RESULTS.json")
    baselines = pd.read_csv(annual / "DR_BASELINES.csv", keep_default_na=False)
    baselines = baselines[baselines.composition_id.eq(2026102001) & baselines.layout.eq("random_mixing") & baselines.permission.eq("native_zone")]
    expected = {(row.context, row.control): row.run_id for row in baselines.itertuples(index=False)}
    if len(designs) != 48 or len(results) != 32 or len(expected) != 12:
        raise ValueError("Incomplete supplementary design support")
    for design in designs:
        if design["baseline_run_id"] != expected[design["context"], design["control"]]:
            raise ValueError("Supplementary design does not use its own freshly reproduced annual baseline")
        design["baseline_hourly_path"] = str(contained(data, design["baseline_hourly_file"]))
    for rid, result in results.items():
        path = contained(supplement, result["hourly_file"])
        if digest(path) != result["hourly_sha256"] or not result["retained_engineering_gate"]["engineering_pass"]:
            raise ValueError("Retained physical output or engineering decision changed")
        result["path_hourly"] = str(path)
    return designs, results


def headline_designs(data, annual, supplement):
    annual_rows = pd.read_csv(annual / "DR_BASELINES.csv", keep_default_na=False)
    annual_rows = annual_rows[annual_rows.composition_id.eq(2026102001) & annual_rows.layout.eq("random_mixing") & annual_rows.permission.eq("native_zone")]
    baselines = {(row.context, row.control): row for row in annual_rows.itertuples(index=False)}
    rows = pd.read_csv(data / "engineering/EVENT_RUN_MANIFEST.csv", keep_default_na=False)
    rows = rows[rows.composition_id.eq(2026102001) & rows.layout.eq("random_mixing") & rows.permission.eq("native_zone")]
    dates = load(data / "engineering/EVENT_DATES_BEFORE_RUNS.json")
    designs, results = [], {}
    for row in rows.to_dict("records"):
        own = baselines[row["context"], row["control"]]
        if row["baseline_run_id"] != own.run_id:
            raise ValueError("Original headline event does not match its own annual baseline")
        event = dict(next(event for event in dates[row["context"]] if event["month"] == 7))
        clock = pd.read_csv(supplement / "source_clocks" / f"{row['context']}.csv")
        event["source_day_type"] = clock[(clock.Month == 7) & (clock.Day == event["day"])].DayType.iloc[0]
        design = {key: row[key] for key in ["context", "composition_id", "layout", "permission", "control", "variant", "run_id", "baseline_run_id", "amplitude_c"]}
        design.update(event=event, date_class="original_maximum", all_events=dates[row["context"]],
                      baseline_hourly_path=str(data / "hourly" / f"{own.run_id}.npz"))
        designs.append(design)
        results[row["run_id"]] = {"engineering_status": "new_qualified" if row["engineering_status"] in ["cached_qualified", "new_qualified"] else row["engineering_status"],
                                   "path_hourly": str(data / "hourly" / f"{row['run_id']}.npz")}
    if len(designs) != 24:
        raise ValueError("Expected 24 original headline logical amplitude cases")
    return designs, results



def analyse_designs(designs, results, groups, data, out):
    candidates, hourly, anchor_rows, command_proofs = [], [], {}, []
    for design in designs:
        result, event = results[design["run_id"]], design["event"]
        group, tail, anchor_id = groups[design["context"]]
        base = read(design["baseline_hourly_path"])
        common = read(str(data / "hourly" / f"{anchor_id}.npz"))
        indices = event_indices(base, event)
        assert all(np.array_equal(base[key], common[key]) for key in ["month", "day", "hour", "dst", "occupied"])
        base_score = [score_window(base, group, tail, ix) for ix in indices]
        anchor_score = [score_window(common, group, tail, ix) for ix in indices]
        qualified = result["engineering_status"] == "new_qualified"
        available = bool(result.get("path_hourly"))
        perturbed = read(result["path_hourly"]) if available else None
        item = {k: design[k] for k in ["context", "composition_id", "layout", "permission", "control", "variant", "run_id", "baseline_run_id", "amplitude_c", "date_class"]}
        item.update(event_month=7, event_day=int(event["day"]), source_day_type=event["source_day_type"],
                    engineering_status=result["engineering_status"], baseline_available=True)
        common_limits = {}
        for prefix, scored_base, scored_anchor in zip(["event", "recovery"], base_score, anchor_score):
            for label, scored in [("baseline", scored_base), ("common25", scored_anchor)]:
                for short, metric in [("mean", "mean_reference_mismatch_c"), ("fixed_tail", "fixed_tail_reference_mismatch_c")]:
                    item[f"{prefix}_{label}_{short}_c"] = scored["summary"][metric]
            for short in ["mean", "fixed_tail"]:
                common_limits[f"{prefix}_{short}_c"] = item[f"{prefix}_common25_{short}_c"]
        anchor_key = design["context"], design["date_class"]
        anchor_rows[anchor_key] = {"context": design["context"], "date_class": design["date_class"], "event_day": event["day"],
                                   "anchor_run_id": anchor_id, "fixed_annual_tail_profiles": 27} | common_limits
        if available:
            validate_event_pair(base, perturbed, design.get("all_events", [event]), design["amplitude_c"])
            horizon = np.concatenate(indices)
            # Full annual comparisons enforce the intervention and unchanged native source clocks/loads.
            expected = np.zeros(8760)
            for command_event in design.get("all_events", [event]):
                command_ix, _ = event_indices(base, command_event)
                expected[command_ix] = command_event["sign"] * design["amplitude_c"]
            command_error = max(float(np.max(abs(perturbed[key] - base[key] - expected[:, None])))
                                for key in ["heating_setpoint_c", "cooling_setpoint_c"])
            assert command_error < 1e-8
            command_proofs.append({"run_id": design["run_id"], "control": design["control"], "command_hour_count": int(np.count_nonzero(expected)),
                                   "full_annual_max_setpoint_error_c": command_error, "non_event_commands_unchanged": True,
                                   "people_max_delta": float(np.max(abs(perturbed["people"] - base["people"]))),
                                   "source_clock_occupied_mask_identical": True})
            pert_score = [score_window(perturbed, group, tail, ix) for ix in indices]
            for prefix, scored in zip(["event", "recovery"], pert_score):
                for short, metric in [("mean", "mean_reference_mismatch_c"), ("fixed_tail", "fixed_tail_reference_mismatch_c")]:
                    item[f"{prefix}_perturbed_{short}_c"] = scored["summary"][metric]
                for metric in ["modeled_interval_person_hour_coverage", "modeled_interval_mean_excess_c", "occupied_native_person_hours"]:
                    item[f"{prefix}_perturbed_{metric}"] = scored["summary"][metric]
            de = perturbed["hvac_electric_kwh"][horizon] - base["hvac_electric_kwh"][horizon]
            dg = perturbed["gas_hhv_kwh"][horizon] - base["gas_hhv_kwh"][horizon]
            item.update(event_average_reduction_kw=float(-de[:2].mean()), event_electric_saving_kwh=float(-de[:2].sum()),
                        recovery_extra_electric_kwh=float(de[2:].sum()), recovery_peak_hour_extra_kw=float(de[2:].max()),
                        event_plus48h_net_electric_saving_kwh=float(-de.sum()), event_plus48h_gas_change_kwh_hhv=float(dg.sum()),
                        event_baseline_average_hvac_kw=float(base["hvac_electric_kwh"][indices[0]].mean()))
            for h, ix in enumerate(horizon):
                phase, pos = (0, h) if h < 2 else (1, h - 2)
                row = {k: item[k] for k in ["context", "control", "date_class", "event_day", "run_id", "baseline_run_id", "amplitude_c", "engineering_status"]}
                row.update(hour_index=h, source_month=int(base["month"][ix]), source_day=int(base["day"][ix]),
                           source_hour_ending=int(base["hour"][ix]), source_dst=int(base["dst"][ix]), source_occupied=bool(base["occupied"][ix]),
                           phase="event" if h < 2 else "recovery", baseline_electric_kwh=float(base["hvac_electric_kwh"][ix]),
                           perturbed_electric_kwh=float(perturbed["hvac_electric_kwh"][ix]), baseline_gas_hhv_kwh=float(base["gas_hhv_kwh"][ix]),
                           perturbed_gas_hhv_kwh=float(perturbed["gas_hhv_kwh"][ix]))
                for label, scored in [("baseline", base_score), ("perturbed", pert_score)]:
                    for metric in ["mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c", "modeled_interval_coverage"]:
                        row[f"{label}_{metric}"] = float(scored[phase]["hourly"][metric][pos])
                hourly.append(row)
        for fraction in [.5, .75, .9, 1.]:
            row = item | {"service_fraction": fraction}
            decisions, baseline_decisions = [], []
            for prefix in ["event", "recovery"]:
                for metric in ["mean", "fixed_tail"]:
                    budget = fraction * common_limits[f"{prefix}_{metric}_c"]
                    row[f"{prefix}_{metric}_budget_c"] = budget
                    for label in ["baseline", "perturbed"]:
                        passed = (label == "baseline" or available) and row[f"{prefix}_{label}_{metric}_c"] <= budget + 1e-9
                        row[f"pass_{prefix}_{label}_{metric}"] = bool(passed)
                        decisions.append(passed)
                        if label == "baseline":
                            baseline_decisions.append(passed)
            row["baseline_service_qualified"] = bool(all(baseline_decisions))
            row["service_eligible"] = bool(qualified and all(decisions))
            row["positive_response"] = bool(available and row["event_average_reduction_kw"] > 1e-9)
            candidates.append(row)
    table = pd.DataFrame(candidates)
    table.to_csv(out / "EVENT_ALL_CANDIDATES.csv", index=False)
    pd.DataFrame(hourly).to_csv(out / "EVENT_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(anchor_rows.values()).to_csv(out / "EVENT_SERVICE_ANCHORS.csv", index=False)
    pd.DataFrame(command_proofs).to_csv(out / "FULL_ANNUAL_COMMAND_PROOFS.csv", index=False)
    selections = []
    for key, group in table.groupby(["context", "date_class", "control", "service_fraction"], sort=True):
        pool = group[group.service_eligible & group.positive_response]
        baseline_ok = bool(group.baseline_service_qualified.iloc[0])
        row = dict(zip(["context", "date_class", "control", "service_fraction"], key))
        row.update(event_day=int(group.event_day.iloc[0]), source_day_type=group.source_day_type.iloc[0], baseline_run_id=group.baseline_run_id.iloc[0],
                   baseline_service_qualified=baseline_ok, engineering_qualified_candidates=int(group.engineering_status.eq("new_qualified").sum()),
                   service_qualified_candidates=int(group.service_eligible.sum()), positive_qualified_candidates=len(pool),
                   service_qualified_available_response_kw=0. if baseline_ok else np.nan, executed_response_kw_with_baseline_fallback=0.,
                   selection_status="engineering_unavailable" if not group.engineering_status.eq("new_qualified").any() else "service_unavailable" if not group.service_eligible.any() else "eligible_no_positive_response", selected_run_id="", selected_amplitude_c=np.nan)
        if not pool.empty:
            best = pool.event_average_reduction_kw.max()
            chosen = pool[pool.event_average_reduction_kw.ge(best - 1e-9)].sort_values(["run_id", "amplitude_c"], kind="mergesort").iloc[0].to_dict()
            row.update(chosen)
            row.update(selection_status="selected", selected_run_id=chosen["run_id"], selected_amplitude_c=chosen["amplitude_c"],
                       service_qualified_available_response_kw=chosen["event_average_reduction_kw"], executed_response_kw_with_baseline_fallback=chosen["event_average_reduction_kw"])
        selections.append(row)
    selected = pd.DataFrame(selections)
    selected.to_csv(out / "EVENT_SELECTIONS.csv", index=False)
    return table, selected, len(hourly), len(command_proofs)



def independent_checks(designs, results, groups, candidates, selected):
    maximum_error, maximum_before_event = 0., 0.
    for design in designs:
        group, tail, anchor_id = groups[design["context"]]
        base, perturbed = read(design["baseline_hourly_path"]), read(results[design["run_id"]]["path_hourly"])
        ix_event, ix_recovery = event_indices(base, design["event"])
        for key in ["temperatures", "hvac_electric_kwh", "gas_hhv_kwh"]:
            error = float(np.max(abs(perturbed[key][:ix_event[0]] - base[key][:ix_event[0]])))
            maximum_before_event = max(maximum_before_event, error)
            if error != 0:
                raise ValueError("The supplementary perturbation changed its pre-event annual history")
        rows = candidates[candidates.run_id.eq(design["run_id"]) & candidates.control.eq(design["control"])]
        if len(rows) != 4:
            raise ValueError("Expected four budget rows per supplementary case")
        positions = {str(zone).upper(): i for i, zone in enumerate(base["zone_ids"])}
        zones = np.array([positions[str(zone).upper()] for zone in group.zone_id])
        counts = np.bincount(zones, minlength=len(positions))
        for prefix, ix in [("event", ix_event), ("recovery", ix_recovery)]:
            for label, data in [("baseline", base), ("perturbed", perturbed)]:
                weights = data["people"][ix][:, zones] * data["occupied"][ix, None] / counts[zones]
                distance = abs(data["temperatures"][ix][:, zones] - group.centre_c.to_numpy())
                profile = (weights * distance).sum(axis=0) / weights.sum(axis=0)
                metrics = {"mean": float((weights * distance).sum() / weights.sum()), "fixed_tail": float(profile[tail].mean())}
                for metric, value in metrics.items():
                    error = float(np.max(abs(rows[f"{prefix}_{label}_{metric}_c"] - value)))
                    maximum_error = max(maximum_error, error)
                    if error > 1e-12:
                        raise ValueError("Independent service endpoint does not match published scorer")
        for row in rows.to_dict("records"):
            passes = []
            for prefix in ["event", "recovery"]:
                for label in ["baseline", "perturbed"]:
                    for metric in ["mean", "fixed_tail"]:
                        passed = row[f"{prefix}_{label}_{metric}_c"] <= row["service_fraction"] * row[f"{prefix}_common25_{metric}_c"] + 1e-9
                        if passed != row[f"pass_{prefix}_{label}_{metric}"]:
                            raise ValueError("Constraint decision differs from independent comparison")
                        passes.append(passed)
            if all(passes) != row["service_eligible"]:
                raise ValueError("Joint eligibility changed")
        response = float((base["hvac_electric_kwh"][ix_event] - perturbed["hvac_electric_kwh"][ix_event]).mean())
        if np.max(abs(rows.event_average_reduction_kw - response)) > 1e-12:
            raise ValueError("Response does not close to the retained physical electricity difference")
    for row in selected.to_dict("records"):
        subset = candidates[candidates.context.eq(row["context"]) & candidates.date_class.eq(row["date_class"]) & candidates.control.eq(row["control"]) & candidates.service_fraction.eq(row["service_fraction"])]
        pool = subset[subset.service_eligible & subset.event_average_reduction_kw.gt(1e-9)]
        if pool.empty:
            if row["selected_run_id"] != "" or row["executed_response_kw_with_baseline_fallback"] != 0:
                raise ValueError("Unavailable tested action was silently selected")
        else:
            best = pool.event_average_reduction_kw.max()
            chosen = pool[pool.event_average_reduction_kw.ge(best - 1e-9)].sort_values(["run_id", "amplitude_c"], kind="mergesort").iloc[0]
            if row["selected_run_id"] != chosen.run_id:
                raise ValueError("Independent amplitude selection differs")
    return {"status": "PASS", "directly_rescored_logical_cases": len(designs),
            "independently_verified_eight_constraint_rows": len(candidates), "independently_verified_selection_rows": len(selected),
            "max_direct_endpoint_error_c": maximum_error, "max_before_event_trajectory_error": maximum_before_event,
            "before_event_trajectories_bitwise_identical": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-data-root", type=Path, required=True, help="Extracted paper root containing energy_service/, or the energy-service directory")
    parser.add_argument("--annual-root", type=Path, required=True, help="Fresh main replay energy_service/annual directory")
    parser.add_argument("--supplement-root", type=Path, required=True, help="Verified extracted normalized sidecar")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    data, annual, supplement, out = (path.resolve() for path in [args.paper_data_root, args.annual_root, args.supplement_root, args.output_root])
    if (data / "energy_service/engineering/ANNUAL_RUN_MANIFEST.csv").is_file():
        data = data / "energy_service"
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    inputs = verify_inputs(data, annual, supplement)
    groups = profile_groups(data, annual)
    dates = verify_dates(data, supplement, groups)
    designs, results = normalized_designs(data, annual, supplement)
    out.mkdir(parents=True)
    save(out / "INPUT_VERIFICATION.json", inputs)
    save(out / "DATE_SELECTION_QA.json", dates)
    candidates, selected, hourly_count, proof_count = analyse_designs(designs, results, groups, data, out)
    peak_designs, peak_results = headline_designs(data, annual, supplement)
    headline_out = out / "headline_replay"
    headline_out.mkdir()
    headline_candidates, headline, peak_hourly_count, _ = analyse_designs(peak_designs, peak_results, groups, data, headline_out)
    fields = ["context", "date_class", "event_day", "control", "service_fraction", "baseline_run_id", "baseline_service_qualified",
              "selection_status", "selected_amplitude_c", "service_qualified_available_response_kw", "executed_response_kw_with_baseline_fallback",
              "event_average_reduction_kw", "event_plus48h_net_electric_saving_kwh", "event_plus48h_gas_change_kwh_hhv"]
    comparison = pd.concat([headline.reindex(columns=fields), selected.reindex(columns=fields)], ignore_index=True)
    comparison.to_csv(out / "THREE_DATE_COMPARISON.csv", index=False)
    if (len(candidates), len(selected), hourly_count, proof_count, len(comparison), len(headline_candidates), peak_hourly_count) != (192, 96, 2400, 48, 144, 96, 1200):
        raise ValueError("Incomplete locked supplementary or original-peak support")
    independent = independent_checks(designs, results, groups, candidates, selected)
    save(out / "INDEPENDENT_QA.json", independent)
    report = {"status": "PASS", "scope": "Independent hourly-output analysis replay only", "physical_simulations_run": 0,
              "physical_parameters_changed": False, "logical_event_cases": len(designs), "retained_new_physical_trajectories": len(results),
              "candidate_budget_rows": len(candidates), "selection_rows": len(selected), "new_hourly_rows": hourly_count,
              "original_peak_candidates_recomputed": len(headline_candidates), "original_peak_hourly_rows": peak_hourly_count,
              "three_date_rows": len(comparison), "event_hours_per_case": 2, "recovery_hours_per_case": 48,
              "annual_tail_profiles_fixed": 27, "no_baseline_or_tail_reselection": True,
              "source_clocks_used_without_invented_display_year": True, "source_manifest_sha256": inputs["supplement_manifest_sha256"],
              "published_service_module_sha256": digest(REPOSITORY / "src/analysis/energy_service/thermal_service.py"),
              "replay_script_sha256": digest(__file__), "independent_checks": independent,
              "outputs": {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}}
    save(out / "REPLAY_REPORT.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ["outputs", "independent_checks"]}, indent=2))


if __name__ == "__main__":
    main()
