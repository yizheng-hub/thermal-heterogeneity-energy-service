"""Evaluate tested DR actions on each actual controller's own annual baseline.

Service uses the common-25 window budgets and the annual fixed 27 profile IDs.
Only 2-h and 48-h slices enter scoring. No maximum-capacity or city claim is made.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_annual import KEYS, PROFILE_KEYS, QUALIFIED, BUDGETS, digest, load_hourly
from thermal_service import score_service
from baseline_decision_contract import verify_baseline


FINAL = QUALIFIED | {"cached_rejected", "new_rejected", "execution_failed", "unavailable_baseline"}
SCORE_TOL = 1e-9
RESPONSE_TOL = 1e-9
SERVICE_METRICS = (
    "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
    "modeled_interval_person_hour_coverage", "modeled_interval_mean_excess_c",
    "occupied_native_person_hours",
)
PAIR_METRICS = (
    "event_average_reduction_kw", "event_electric_saving_kwh",
    "recovery_extra_electric_kwh", "recovery_peak_hour_extra_kw",
    "event_plus48h_net_electric_saving_kwh", "event_plus48h_gas_change_kwh_hhv",
    "event_plus48h_average_factor_co2_change_kg", "event_plus48h_energy_charge_change_currency",
)


def event_indices(data, event):
    civil = data["hour"] - 1 + data["dst"]
    indices = np.flatnonzero((data["month"] == event["month"]) &
                             (data["day"] == event["day"]) & (civil >= 14) & (civil < 16))
    if len(indices) != 2 or indices[1] != indices[0] + 1 or indices[-1] + 49 > 8760:
        raise ValueError("Expected exactly two consecutive event hours and 48 recovery hours")
    return indices, np.arange(indices[-1] + 1, indices[-1] + 49)


def score_window(data, profiles, tail, index):
    position = {str(z).strip().upper(): i for i, z in enumerate(data["zone_ids"])}
    zone = np.array([position[str(z).strip().upper()] for z in profiles.zone_id])
    weight = data["people"][index] * data["occupied"][index].astype(bool)[:, None]
    return score_service(data["temperatures"][index], weight, zone,
                         profiles.centre_c.to_numpy(), profiles.lower_c.to_numpy(),
                         profiles.upper_c.to_numpy(), tail)


def qualify(scores, fraction, anchor):
    decisions = []
    for prefix in ("event", "recovery"):
        for trajectory in ("baseline", "perturbed"):
            for metric in ("mean", "fixed_tail"):
                decisions.append(scores[f"{prefix}_{trajectory}_{metric}_c"] <=
                                 fraction * anchor[f"{prefix}_{metric}_c"] + SCORE_TOL)
    return all(decisions)


def baseline_qualifies(scores, fraction, anchor):
    return all(scores[f"{prefix}_baseline_{metric}_c"] <=
               fraction * anchor[f"{prefix}_{metric}_c"] + SCORE_TOL
               for prefix in ("event", "recovery") for metric in ("mean", "fixed_tail"))


def choose_response(rows):
    pool = rows[rows.service_eligible & rows.event_average_reduction_kw.gt(RESPONSE_TOL)]
    if pool.empty:
        return None
    best = float(pool.event_average_reduction_kw.max())
    return pool[pool.event_average_reduction_kw.ge(best - RESPONSE_TOL)].sort_values(
        ["run_id", "amplitude_c"], kind="mergesort").iloc[0].to_dict()


def local_timestamp(data, index):
    # Year 2001 is a display-only non-leap model calendar, not a measured year.
    return [(pd.Timestamp(2001, int(data["month"][i]), int(data["day"][i])) +
             pd.Timedelta(hours=int(data["hour"][i]) - 1 + int(data["dst"][i]))).isoformat()
            for i in index]


def validate_event_pair(base, perturbed, dates, amplitude):
    for field in ("month", "day", "hour", "dst", "zone_ids", "occupied"):
        if not np.array_equal(base[field], perturbed[field]):
            raise ValueError(f"Event and own baseline differ in {field}")
    if not np.allclose(base["people"], perturbed["people"], rtol=0, atol=1e-10):
        raise ValueError("Native occupancy changed in event trajectory")
    expected = np.zeros(8760)
    for event in dates:
        indices, _ = event_indices(base, event)
        expected[indices] = float(event["sign"]) * amplitude
    for field in ("heating_setpoint_c", "cooling_setpoint_c"):
        delta = perturbed[field] - base[field]
        if not np.allclose(delta, expected[:, None], rtol=0, atol=1e-8):
            raise ValueError(f"Event schedule changed outside the four locked event hours: {field}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engineering-root", required=True, type=Path)
    parser.add_argument("--annual-root", required=True, type=Path)
    parser.add_argument("--factors", required=True, type=Path)
    parser.add_argument("--areas", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.engineering_root.resolve(), args.output_root
    inputs = {"events": root / "EVENT_RUN_MANIFEST.csv", "annual_runs": root / "ANNUAL_RUN_MANIFEST.csv",
              "profiles": root / "PROFILE_ASSIGNMENTS.csv.gz", "event_dates": root / "EVENT_DATES_BEFORE_RUNS.json",
              "anchors": args.annual_root / "ANNUAL_ANCHORS.csv", "tail_ids": args.annual_root / "FIXED_TAIL_MEMBERSHIPS.csv",
              "baselines": args.annual_root / "DR_BASELINES.csv", "factors": args.factors, "areas": args.areas,
              "annual_manifest": args.annual_root / "ANALYSIS_MANIFEST.json",
              "event_generation": root / "EVENT_GENERATION_QA.json",
              "baseline_decisions": root / "BASELINE_DECISION_CONTRACT.json"}
    source_hash = {key: digest(path) for key, path in inputs.items()}
    annual_manifest = json.loads(inputs["annual_manifest"].read_text(encoding="utf-8"))
    if annual_manifest.get("status") != "PASS" or annual_manifest["physical_parameters_changed"]:
        raise ValueError("Annual source analysis has not passed")
    for name in ("ANNUAL_ANCHORS.csv", "FIXED_TAIL_MEMBERSHIPS.csv", "DR_BASELINES.csv"):
        if annual_manifest["output_sha256"][name] != digest(args.annual_root / name):
            raise ValueError(f"Annual source product hash changed: {name}")
    generation = json.loads(inputs["event_generation"].read_text(encoding="utf-8"))
    decision_contract = json.loads(inputs["baseline_decisions"].read_text(encoding="utf-8"))
    if decision_contract["source_event_generation_sha256"] != source_hash["event_generation"]:
        raise ValueError("Baseline decision sidecar does not bind the original event generation evidence")
    baseline_verification = verify_baseline(inputs["baselines"], generation, decision_contract)
    if (generation["physical_refits"] != 0 or not generation["per_strategy_baselines"] or
        not generation["event_dates_from_frozen_common25"] or
        generation["event_dates_sha256"] != source_hash["event_dates"]):
        raise ValueError("Event generation does not match the frozen own-baseline/date contract")
    designs = pd.read_csv(inputs["events"], keep_default_na=False)
    if len(designs) != 648 or designs.duplicated(KEYS + ["amplitude_c"]).any():
        raise ValueError("Expected 648 complete logical event designs")
    if not set(designs.engineering_status).issubset(FINAL):
        raise ValueError("Event engineering collection contains unfinished/export-failed designs")
    if set(designs.amplitude_c) != {.5, 1.}:
        raise ValueError("Only the two locked event amplitudes may enter")
    annual = pd.read_csv(inputs["annual_runs"], keep_default_na=False)
    run_rows = {row.run_id: row for row in annual.itertuples(index=False) if row.engineering_status in QUALIFIED}
    profiles = pd.read_csv(inputs["profiles"])
    anchors = pd.read_csv(inputs["anchors"])
    tails = pd.read_csv(inputs["tail_ids"])
    baselines = pd.read_csv(inputs["baselines"], keep_default_na=False)
    dates = json.loads(inputs["event_dates"].read_text(encoding="utf-8"))
    if any(len(value) != 2 or {event["month"] for event in value} != {2, 7} for value in dates.values()):
        raise ValueError("Expected one fixed February date and one fixed July date per context")
    if any(event["sign"] != (-1 if event["month"] == 2 else 1) for value in dates.values() for event in value):
        raise ValueError("Heating/cooling event signs differ from the locked protocol")
    factors = pd.read_csv(args.factors).set_index("city")
    areas = pd.read_csv(args.areas).set_index("climate_context").floor_area_m2.to_dict()
    baseline_rows = {tuple(getattr(row, key) for key in KEYS): row for row in baselines.itertuples(index=False)}
    if len(baseline_rows) != 324 or len(baselines) != 324:
        raise ValueError("Expected one own-family budget-1 baseline per logical controller")
    hashes = {}

    @lru_cache(maxsize=8)
    def trajectory(path_string):
        path = Path(path_string)
        if not path.is_absolute():
            path = root / path
        if str(path) not in hashes:
            hashes[str(path)] = digest(path)
        return load_hourly(path)

    def annual_data(run_id):
        return trajectory(str(run_rows[run_id].path_hourly))

    groups = {}
    for key, group in profiles.groupby(PROFILE_KEYS, sort=True):
        group = group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        relevant = tails[(tails[PROFILE_KEYS] == pd.Series(key, index=PROFILE_KEYS)).all(axis=1)]
        fixed = np.flatnonzero(group.profile_id.isin(relevant.profile_id))
        anchor_row = anchors[(anchors[PROFILE_KEYS] == pd.Series(key, index=PROFILE_KEYS)).all(axis=1)]
        if len(group) != 267 or len(fixed) != 27 or len(anchor_row) != 1:
            raise ValueError(f"Profiles, annual tail identities or common25 anchor inconsistent: {key}")
        groups[key] = (group, fixed, anchor_row.iloc[0].anchor_run_id)
    if len(groups) != 36:
        raise ValueError("Incomplete actual profile placement groups")

    score_cache, anchor_cache = {}, {}

    def windows(run_id, profile_key, event, path=None):
        key = (run_id, profile_key, event["month"])
        if key not in score_cache:
            data = annual_data(run_id) if path is None else trajectory(str(path))
            group, fixed, _ = groups[profile_key]
            indices = event_indices(data, event)
            scored = [score_window(data, group, fixed, index) for index in indices]
            score_cache[key] = (scored, np.concatenate(indices))
        return score_cache[key]

    def budget_anchor(profile_key, event):
        key = profile_key + (event["month"],)
        if key not in anchor_cache:
            scored, _ = windows(groups[profile_key][2], profile_key, event)
            anchor_cache[key] = {f"{prefix}_{metric}_c": item["summary"][full]
                for prefix, item in zip(("event", "recovery"), scored)
                for metric, full in (("mean", "mean_reference_mismatch_c"), ("fixed_tail", "fixed_tail_reference_mismatch_c"))}
        return anchor_cache[key]

    candidates, hourly, baseline_scores = [], [], {}
    for design in designs.itertuples(index=False):
        key = tuple(getattr(design, name) for name in KEYS)
        profile_key = key[:3]
        source = baseline_rows[key]
        baseline_available = source.selection_status == "selected" and source.run_id in run_rows
        if baseline_available and design.baseline_run_id != source.run_id:
            raise ValueError("Event design does not use its own annual control-family baseline")
        valid = design.engineering_status in QUALIFIED
        if valid and not baseline_available:
            raise ValueError("Qualified event cannot substitute a different baseline")
        base = annual_data(source.run_id) if baseline_available else None
        perturbed = trajectory(str(design.path_hourly)) if valid else None
        if valid:
            validate_event_pair(base, perturbed, dates[design.context], float(design.amplitude_c))
        for event in dates[design.context]:
            if event["month"] not in (2, 7):
                raise ValueError("Only locked February and July event dates are supported")
            common = budget_anchor(profile_key, event)
            item = {name: getattr(design, name) for name in KEYS + ["variant", "amplitude_c", "run_id", "baseline_run_id", "engineering_status"]}
            item.update(event_month=event["month"], event_day=event["day"], event_sign=event["sign"],
                        baseline_available=baseline_available, physical_status="qualified" if valid else design.engineering_status)
            scored_base = None
            if baseline_available:
                scored_base, horizon = windows(source.run_id, profile_key, event)
                for prefix, score in zip(("event", "recovery"), scored_base):
                    for metric in SERVICE_METRICS:
                        item[f"{prefix}_baseline_{metric}"] = score["summary"][metric]
                    item[f"{prefix}_baseline_mean_c"] = score["summary"]["mean_reference_mismatch_c"]
                    item[f"{prefix}_baseline_fixed_tail_c"] = score["summary"]["fixed_tail_reference_mismatch_c"]
                baseline_scores[key + (event["month"],)] = (dict(item), horizon, scored_base)
            if valid:
                scored_perturbed, horizon = windows(design.run_id, profile_key, event, design.path_hourly)
                for prefix, score in zip(("event", "recovery"), scored_perturbed):
                    for metric in SERVICE_METRICS:
                        item[f"{prefix}_perturbed_{metric}"] = score["summary"][metric]
                    item[f"{prefix}_perturbed_mean_c"] = score["summary"]["mean_reference_mismatch_c"]
                    item[f"{prefix}_perturbed_fixed_tail_c"] = score["summary"]["fixed_tail_reference_mismatch_c"]
                de = perturbed["hvac_electric_kwh"][horizon] - base["hvac_electric_kwh"][horizon]
                dg = perturbed["gas_hhv_kwh"][horizon] - base["gas_hhv_kwh"][horizon]
                for prefix, indices in (("event", horizon[:2]), ("recovery", horizon[2:])):
                    for label, data in (("baseline", base), ("perturbed", perturbed)):
                        item[f"{prefix}_{label}_electric_kwh"] = float(data["hvac_electric_kwh"][indices].sum())
                        item[f"{prefix}_{label}_gas_hhv_kwh"] = float(data["gas_hhv_kwh"][indices].sum())
                item["event_baseline_average_hvac_kw"] = item["event_baseline_electric_kwh"] / 2
                item["event_perturbed_average_hvac_kw"] = item["event_perturbed_electric_kwh"] / 2
                item.update(event_average_reduction_kw=float(-de[:2].mean()), event_electric_saving_kwh=float(-de[:2].sum()),
                    recovery_extra_electric_kwh=float(de[2:].sum()), recovery_peak_hour_extra_kw=float(de[2:].max()),
                    recovery_positive_rebound_peak_kw=float(max(0, de[2:].max())), event_plus48h_net_electric_saving_kwh=float(-de.sum()),
                    event_plus48h_gas_change_kwh_hhv=float(dg.sum()), event_gas_change_kwh_hhv=float(dg[:2].sum()),
                    recovery_gas_change_kwh_hhv=float(dg[2:].sum()))
                factor = factors.loc[design.context]
                item["event_plus48h_average_factor_co2_change_kg"] = float(de.sum() * factor.electricity_co2_kg_kwh + dg.sum() * factor.gas_co2_kg_kwh_hhv)
                item["currency"] = factor.currency
                item["event_plus48h_energy_charge_change_currency"] = float(de.sum() * factor.electricity_price_currency_kwh + dg.sum() * factor.gas_price_currency_kwh_hhv) if factor.currency == "USD" else np.nan
                item["delhi_energy_charge_intercept_INR_pf095"] = float(de.sum() * 8.5 / .95) if factor.currency == "INR" else np.nan
                item["delhi_gas_price_slope_kwh_hhv"] = float(dg.sum()) if factor.currency == "INR" else np.nan
                time = local_timestamp(base, horizon)
                for h, index in enumerate(horizon):
                    row = {name: item[name] for name in KEYS + ["amplitude_c", "run_id", "baseline_run_id", "event_month", "event_day"]}
                    row.update(hour_from_event_start=h, timestamp_local=time[h], source_occupied_mask=bool(base["occupied"][index]),
                        baseline_electric_kwh=float(base["hvac_electric_kwh"][index]), event_electric_kwh=float(perturbed["hvac_electric_kwh"][index]),
                        baseline_gas_hhv_kwh=float(base["gas_hhv_kwh"][index]), event_gas_hhv_kwh=float(perturbed["gas_hhv_kwh"][index]))
                    window, position = (0, h) if h < 2 else (1, h - 2)
                    for label, score in (("baseline", scored_base), ("perturbed", scored_perturbed)):
                        row.update({f"{label}_{metric}": float(score[window]["hourly"][metric][position]) for metric in ("mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c", "modeled_interval_coverage", "modeled_interval_mean_excess_c")})
                    hourly.append(row)
            for fraction in BUDGETS:
                row = item | {"service_fraction": fraction}
                row.update({f"{prefix}_{metric}_budget_c": fraction * common[f"{prefix}_{metric}_c"] for prefix in ("event", "recovery") for metric in ("mean", "fixed_tail")})
                row["baseline_service_qualified"] = bool(baseline_available and baseline_qualifies(item, fraction, common))
                row["service_eligible"] = bool(valid and qualify(item, fraction, common))
                row["positive_response"] = bool(valid and item["event_average_reduction_kw"] > RESPONSE_TOL)
                candidates.append(row)
    all_candidates = pd.DataFrame(candidates)
    selected = []
    for key, group in all_candidates.groupby(KEYS + ["event_month", "service_fraction"], sort=True):
        chosen = choose_response(group)
        available = bool(group.baseline_available.iloc[0])
        engineering = group[group.engineering_status.isin(QUALIFIED)]
        eligible = engineering[engineering.service_eligible]
        state = "selected" if chosen else "unavailable_baseline" if not available else "engineering_unavailable" if engineering.empty else "service_unavailable" if eligible.empty else "eligible_no_positive_response"
        row = dict(zip(KEYS + ["event_month", "service_fraction"], key)) | {
            "selection_status": state, "baseline_available": available,
            "baseline_run_id": group.baseline_run_id.iloc[0], "baseline_service_qualified": bool(group.baseline_service_qualified.iloc[0]),
            "engineering_qualified_candidates": len(engineering), "service_qualified_candidates": len(eligible),
            "positive_qualified_candidates": int(eligible.positive_response.sum()),
            "executed_response_kw_with_baseline_fallback": 0.0, "selected_run_id": "", "selected_amplitude_c": np.nan,
            "service_qualified_available_response_kw": 0.0 if bool(group.baseline_service_qualified.iloc[0]) else np.nan,
        }
        if chosen:
            row.update(chosen)
            row.update(selected_run_id=chosen["run_id"], selected_amplitude_c=chosen["amplitude_c"],
                       executed_response_kw_with_baseline_fallback=chosen["event_average_reduction_kw"],
                       service_qualified_available_response_kw=chosen["event_average_reduction_kw"])
        selected.append(row)
    selections = pd.DataFrame(selected)
    references = {tuple(row[key] for key in PROFILE_KEYS + ["permission", "event_month", "service_fraction"]): row for row in selections[selections.control.eq("uniform_average")].to_dict("records")}
    paired = []
    for row in selections[~selections.control.eq("uniform_average")].to_dict("records"):
        ref = references[tuple(row[key] for key in PROFILE_KEYS + ["permission", "event_month", "service_fraction"])]
        matched = row["selection_status"] == ref["selection_status"] == "selected"
        item = {key: row[key] for key in KEYS + ["event_month", "service_fraction"]}
        item.update(status="matched" if matched else "unmatched", alternative_status=row["selection_status"],
            reference_status=ref["selection_status"], alternative_run_id=row["selected_run_id"], reference_run_id=ref["selected_run_id"],
            both_baselines_service_qualified=bool(row["baseline_service_qualified"] and ref["baseline_service_qualified"]),
            delta_executed_response_kw_with_baseline_fallback=row["executed_response_kw_with_baseline_fallback"] - ref["executed_response_kw_with_baseline_fallback"])
        item["delta_service_qualified_available_response_kw"] = row["service_qualified_available_response_kw"] - ref["service_qualified_available_response_kw"]
        for metric in PAIR_METRICS:
            item["delta_" + metric] = row.get(metric, np.nan) - ref.get(metric, np.nan) if matched else np.nan
        for prefix in ("event", "recovery"):
            for trajectory_label in ("baseline", "perturbed"):
                for metric in SERVICE_METRICS:
                    name = f"{prefix}_{trajectory_label}_{metric}"
                    item["delta_" + name] = row.get(name, np.nan) - ref.get(name, np.nan) if matched else np.nan
        paired.append(item)

    types, adapter_hours = [], []
    for row in selections[selections.service_fraction.eq(1.)].to_dict("records"):
        key = tuple(row[name] for name in KEYS)
        month = int(row["event_month"])
        identity = {"climate_context": row["context"], "scope": row["permission"], "control": row["control"],
                    "event_id": f"month_{month:02d}", "composition_id": row["composition_id"], "layout": row["layout"]}
        active = row["selection_status"] == "selected"
        types.append(identity | {"baseline_case_id": row["baseline_run_id"], "event_case_id": row["selected_run_id"] if active else row["baseline_run_id"],
            "floor_area_m2": areas[row["context"]], "baseline_engineering_status": "PASS" if row["baseline_available"] else "UNAVAILABLE",
            "event_engineering_status": "PASS" if row["baseline_available"] else "UNAVAILABLE", "baseline_service_qualified": row["baseline_service_qualified"],
            "event_service_qualified": bool(active or row["baseline_service_qualified"]), "selection_status": row["selection_status"],
            "tested_positive_response_selected": active})
        if not row["baseline_available"]:
            continue
        base = annual_data(row["baseline_run_id"])
        _, horizon, _ = baseline_scores[key + (month,)]
        if active:
            matches = designs[(designs.run_id == row["selected_run_id"]) & (designs.context == row["context"])]
            perturbed = trajectory(str(matches.iloc[0].path_hourly))
        else:
            perturbed = base
        timestamps = local_timestamp(base, horizon)
        for h, index in enumerate(horizon):
            adapter_hours.append(identity | {"hour_index": h, "timestamp_local": timestamps[h], "phase": "event" if h < 2 else "recovery", "timestep_hours": 1,
                "baseline_electric_kwh": float(base["hvac_electric_kwh"][index]), "event_electric_kwh": float(perturbed["hvac_electric_kwh"][index]),
                "baseline_gas_hhv_kwh": float(base["gas_hhv_kwh"][index]), "event_gas_hhv_kwh": float(perturbed["gas_hhv_kwh"][index])})
    out.mkdir(parents=True, exist_ok=True)
    all_candidates.to_csv(out / "EVENT_ALL_CANDIDATES.csv", index=False)
    selections.to_csv(out / "EVENT_SELECTIONS.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "EVENT_COMPARISONS.csv", index=False)
    pd.DataFrame(hourly).to_csv(out / "EVENT_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(types).to_csv(out / "PORTFOLIO_TYPES.csv", index=False)
    pd.DataFrame(adapter_hours).to_csv(out / "PORTFOLIO_HOURLY.csv", index=False)
    anchor_output = [dict(zip(PROFILE_KEYS + ["event_month"], key)) | {"anchor_run_id": groups[key[:3]][2]} | value for key, value in sorted(anchor_cache.items())]
    pd.DataFrame(anchor_output).to_csv(out / "EVENT_SERVICE_ANCHORS.csv", index=False)
    profiles[["composition_id", "layout"]].drop_duplicates().sort_values(["composition_id", "layout"]).to_csv(out / "PORTFOLIO_DESIGN_TYPES.csv", index=False)
    if len(all_candidates) != 5184 or len(selections) != 2592 or len(paired) != 1728 or len(types) != 648:
        raise ValueError("Incomplete predeclared event outcome grid")
    if source_hash != {key: digest(path) for key, path in inputs.items()}:
        raise ValueError("Event source inputs changed during analysis")
    report = {"status": "PASS", "logical_event_designs": len(designs), "candidate_budget_windows": len(all_candidates),
        "selection_rows": len(selections), "control_comparison_rows": len(paired), "hourly_candidate_rows": len(hourly),
        "headline_portfolio_types": len(types), "headline_portfolio_hourly_rows": len(adapter_hours),
        "selection_status_counts": selections.selection_status.value_counts().to_dict(), "physical_parameters_changed": False,
        "own_control_family_baselines_only": True, "fixed_annual_tail_profile_count": 27,
        "baseline_reproduction": baseline_verification,
        "service_slicing": "2 event rows and 48 consecutive recovery rows, source occupied hours only",
        "event_budget_sensitivity": "All four event/recovery fractions use each family's fixed annual-budget-1 baseline; these are not separately annual-budget-optimized baselines",
        "response_zero_rule": "A service-qualified baseline without a tested positive action has available response zero; an ineligible baseline has NaN service-qualified availability, separately from physical no-DR fallback zero",
        "display_timestamp_basis": "Year 2001 is a display-only non-leap calendar. Original model month/day and hour-1+DST define windows; no measured year or displayed weekday is inferred",
        "carbon_interpretation": "Annual-average-factor operating accounting, not hourly marginal avoided emissions",
        "response_interpretation": "Best qualified positive response among two predeclared amplitudes, not maximum building capacity",
        "source_sha256": source_hash, "hourly_input_sha256": hashes,
        "output_sha256": {path.name: digest(path) for path in sorted(out.glob("*.csv*"))}}
    (out / "EVENT_ANALYSIS_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
