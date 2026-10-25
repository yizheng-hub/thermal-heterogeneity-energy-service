"""Prepare own-baseline events and matched portfolios for parallel office cohorts.

The event evaluator, physical executor, tariffs and implementation calculations
are the same as in the primary experiment. Only cohort compositions and the
declared two-layout inventory differ; neither source supplies local city stock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

ENERGY_SERVICE = Path(__file__).resolve().parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from analyse_annual import KEYS, QUALIFIED, digest  # noqa: E402
from baseline_decision_contract import SERVICE, decision_sha256, rows  # noqa: E402
from generate_model_inputs import annual_input, event_input, physical_hash, signature  # noqa: E402
from run_portfolio import aggregate, read_csv, write_csv  # noqa: E402
from run_costs import command_scope_boundaries, cost_scenarios, price_pairs, write_csv as write_cost_csv  # noqa: E402
from response_recovery import mark_spatial_aliases, read_scope, select_stages, summarise  # noqa: E402

LAYOUTS = {"random_mixing", "demand_ordered"}
EXPECTED_BASELINES = 216


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def resolved(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def event_cache(primary_engineering, model_root, models):
    """Reuse only an identical normalized input, weather, and retained QA."""
    frame = pd.read_csv(primary_engineering / "EVENT_RUN_MANIFEST.csv", keep_default_na=False)
    cache = {}
    for row in frame[frame.engineering_status.isin(QUALIFIED)].drop_duplicates("run_id").to_dict("records"):
        source = model_root / "current_run_gates" / row["run_id"]
        qa_path, idf = source / "ENGINEERING_QA.json", source / "input.idf"
        hourly = resolved(primary_engineering, row["path_hourly"])
        quality = json.loads(qa_path.read_text(encoding="utf-8"))
        if not quality["engineering_pass"] or signature(idf.read_text()) != row["normalized_idf_sha256"]:
            raise ValueError("A primary cached event has inconsistent input or engineering evidence")
        key = (row["context"], row["normalized_idf_sha256"], models[row["context"]]["weather_sha256"])
        cache.setdefault(key, {"cached_run_id": row["run_id"],
            "cached_engineering_status": row["engineering_status"],
            "cached_normalized_idf_sha256": row["normalized_idf_sha256"],
            "cached_hourly_path": str(hourly), "cached_hourly_sha256": digest(hourly),
            "cached_qa_path": str(qa_path.resolve()), "cached_qa_sha256": digest(qa_path)})
    return cache


def prepare(annual_engineering, annual_root, commands_path, model_root, primary_root, output):
    annual_engineering, annual_root, model_root, primary_root = [Path(p).resolve() for p in
        (annual_engineering, annual_root, model_root, primary_root)]
    annual_manifest = json.loads((annual_root / "ANALYSIS_MANIFEST.json").read_text())
    if annual_manifest["status"] != "PASS" or annual_manifest["physical_parameters_changed"]:
        raise ValueError("Annual analysis must pass unchanged engineering rules first")
    baselines_path = annual_root / "DR_BASELINES.csv"
    baselines = rows(baselines_path, EXPECTED_BASELINES)
    if {row["layout"] for row in baselines} != LAYOUTS:
        raise ValueError("The shared design has random and demand-ordered layouts only")
    if any(float(row["service_fraction"]) != 1 for row in baselines):
        raise ValueError("Events must use each family's annual budget-1 operation")
    for name in ("DR_BASELINES.csv", "ANNUAL_ANCHORS.csv", "FIXED_TAIL_MEMBERSHIPS.csv"):
        if annual_manifest["output_sha256"][name] != digest(annual_root / name):
            raise ValueError(f"Annual decision evidence changed: {name}")
    original_dates = primary_root / "engineering/EVENT_DATES_BEFORE_RUNS.json"
    original_generation = json.loads((primary_root / "engineering/EVENT_GENERATION_QA.json").read_text())
    if (not original_generation["event_dates_from_frozen_common25"]
            or original_generation["event_dates_sha256"] != digest(original_dates)):
        raise ValueError("Primary dates do not bind the original common-25 operation")
    dates = json.loads(original_dates.read_text())
    models = json.loads((model_root / "MODEL_CONTEXTS.json").read_text())
    if set(dates) != set(models) or any(
            len(events) != 2 or {event["month"] for event in events} != {2, 7}
            or any(event["sign"] != (-1 if event["month"] == 2 else 1) for event in events)
            for events in dates.values()):
        raise ValueError("The locked design requires February/July dates with original event signs")
    commands = json.loads(Path(commands_path).read_text())
    command_keys = KEYS + ["variant"]
    lookup = {tuple(row[key] for key in command_keys): row["targets_c"] for row in commands}
    if len(commands) != 864 or len(lookup) != 864:
        raise ValueError("Expected all 864 declared parallel annual commands")
    annual = pd.read_csv(annual_engineering / "ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    if len(annual) != 864 or annual.duplicated(command_keys).any():
        raise ValueError("Expected the complete parallel annual manifest")
    selected = {tuple(row[key] for key in command_keys): row for row in annual.to_dict("records")}
    texts, physics = {}, {}
    for context, config in models.items():
        source, weather = model_root / config["baseline_idf"], model_root / config["weather"]
        if digest(source) != config["baseline_idf_sha256"] or digest(weather) != config["weather_sha256"]:
            raise ValueError("Frozen source model or weather changed")
        texts[context] = source.read_text()
        physics[context] = physical_hash(texts[context])
    cache = event_cache(primary_root / "engineering", model_root, models)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "idfs").mkdir()
    logical, unique = [], {}
    for baseline in baselines:
        context = baseline["context"]
        for amplitude in (.5, 1.):
            item = {key: baseline[key] for key in KEYS + ["variant"]}
            item.update(kind="event", amplitude_c=amplitude, baseline_run_id=baseline["run_id"],
                        permission_alias="floor_equals_building" if baseline["permission"] == "floor"
                        and "small2019" in context else "")
            if baseline["selection_status"] != "selected":
                key = json.dumps(item, sort_keys=True)
                item.update(run_id="ctcd_event_unavailable_" + hashlib.sha256(key.encode()).hexdigest()[:24],
                            engineering_status="unavailable_baseline", path_hourly="", normalized_idf_sha256="")
                logical.append(item)
                continue
            annual_key = tuple(baseline[key] for key in command_keys)
            retained = selected[annual_key]
            if retained["run_id"] != baseline["run_id"] or retained["engineering_status"] not in QUALIFIED:
                raise ValueError("The selected own-family baseline is not an engineering-qualified annual input")
            text = annual_input(texts[context], lookup[annual_key], models[context]["half_deadband_c"])
            if signature(text) != retained["normalized_idf_sha256"]:
                raise ValueError("Own-baseline commands do not reproduce the retained annual input")
            text = event_input(text, dates[context], amplitude)
            normal = signature(text)
            if physical_hash(text) != physics[context]:
                raise ValueError("An event input changed a non-control physical object")
            config = models[context]
            run_id = "ctcd_event_" + hashlib.sha256(
                f"{context}|{normal}|{config['weather_sha256']}".encode()).hexdigest()[:24]
            cached = cache.get((context, normal, config["weather_sha256"]), {})
            item.update(run_id=run_id, normalized_idf_sha256=normal, physical_sha256=physics[context],
                weather=config["weather"], weather_sha256=config["weather_sha256"],
                idf_path=f"idfs/{run_id}.idf", execution_status="exact_retained_cache" if cached else "pending_new_run",
                **cached)
            logical.append(item)
            if run_id not in unique:
                unique[run_id] = item
                (output / item["idf_path"]).write_text(text, encoding="utf-8")
    frame = pd.DataFrame(logical).fillna("")
    frame.to_csv(output / "EVENT_RUN_PLAN.csv", index=False)
    pd.DataFrame(unique.values()).fillna("").to_csv(output / "PHYSICAL_EVENT_PLAN.csv", index=False)
    # The evaluator resolves annual trajectories from this independent event root.
    for column in ("path_hourly", "qa_path"):
        if column in annual:
            annual[column] = annual[column].map(lambda value: str(resolved(annual_engineering, value)) if value else "")
    annual.to_csv(output / "ANNUAL_RUN_MANIFEST.csv", index=False)
    shutil.copyfile(annual_engineering / "PROFILE_ASSIGNMENTS.csv.gz", output / "PROFILE_ASSIGNMENTS.csv.gz")
    shutil.copyfile(original_dates, output / "EVENT_DATES_BEFORE_RUNS.json")
    generation = {"logical_event_designs": len(frame), "unique_annual_event_commands": len(unique),
        "baseline_selection_sha256": digest(baselines_path), "event_dates_sha256": digest(original_dates),
        "event_dates_from_frozen_common25": True, "per_strategy_baselines": True, "physical_refits": 0,
        "expected_baselines": EXPECTED_BASELINES, "expected_profile_groups": 24,
        "cohort": "CTCD", "layouts": sorted(LAYOUTS),
        "logical_status_counts_before_runs": frame.execution_status.value_counts().to_dict(),
        "date_rule": "The original primary common-25 peak occupied weekday in February and July, shared across cohorts",
        "source_sha256": {"annual_manifest": digest(annual_engineering / "ANNUAL_RUN_MANIFEST.csv"),
            "annual_analysis": digest(annual_root / "ANALYSIS_MANIFEST.json"), "commands": digest(commands_path)},
        "scope": "CTCD parallel office event/recovery on own annual-budget-1 baselines; random and demand-ordered placement; existing building/floor/zone permissions"}
    write_json(output / "EVENT_GENERATION_QA.json", generation)
    contract = {"source_baseline_csv_sha256": digest(baselines_path),
        "source_event_generation_sha256": digest(output / "EVENT_GENERATION_QA.json"),
        "derived_after_generation_from_verified_original": True,
        "decision_sha256": decision_sha256(baselines), "decision_rows": len(baselines),
        "service_tolerance": 1e-9,
        "service_references": [{name: float(row[name]) for name in SERVICE} for row in baselines]}
    write_json(output / "BASELINE_DECISION_CONTRACT.json", contract)
    if len(frame) != EXPECTED_BASELINES * 2 or frame.duplicated(KEYS + ["amplitude_c"]).any():
        raise ValueError("Incomplete or duplicated parallel event design")
    return generation


def portfolio(event_root, contexts_path, cohort, output):
    """Both cohorts use six types and the original twelve replicas per type."""
    event_root, output = Path(event_root), Path(output)
    contexts = json.loads(Path(contexts_path).read_text())
    types = [row for row in read_csv(event_root / "PORTFOLIO_TYPES.csv")
             if row["layout"] in LAYOUTS and not (row["scope"] == "floor"
                 and contexts[row["climate_context"]]["floor_binding_equals_building"])]
    hourly = [row for row in read_csv(event_root / "PORTFOLIO_HOURLY.csv")
              if row["layout"] in LAYOUTS and not (row["scope"] == "floor"
                 and contexts[row["climate_context"]]["floor_binding_equals_building"])]
    design_types = [row for row in read_csv(event_root / "PORTFOLIO_DESIGN_TYPES.csv") if row["layout"] in LAYOUTS]
    if len(design_types) != 6:
        raise ValueError("Parallel portfolios require exactly three draws and two layouts")
    protocol = {"portfolio_buildings": 72, "dr_participation_fractions": [.25, .5, 1.],
                "replicas_per_type": 12, "types": 6, "cohort": cohort,
                "scope": "Matched standardized office portfolio; no actual-city stock estimate"}
    summary, curves, components = aggregate(types, hourly, design_types, protocol)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("portfolio_summary.csv", summary), ("portfolio_hourly.csv", curves),
                          ("portfolio_components.csv", components)):
        write_csv(output / name, [row | {"data_source": cohort} for row in records])
    write_json(output / "PORTFOLIO_PROTOCOL.json", protocol)
    report = {"status": "PASS", "cohort": cohort, "actual_city_prediction": False,
        "summary_rows": len(summary), "hourly_rows": len(curves), "type_count": 6,
        "source_sha256": {name: digest(event_root / name) for name in
            ("PORTFOLIO_TYPES.csv", "PORTFOLIO_HOURLY.csv", "PORTFOLIO_DESIGN_TYPES.csv")}}
    write_json(output / "PORTFOLIO_QA.json", report)
    return report


def analyse(plan_root, execution_root, annual_root, factors, areas, output):
    """Bind completed event execution to the predeclared plan, then reuse scoring."""
    plan_root, execution_root, output = [Path(path).resolve() for path in (plan_root, execution_root, output)]
    plan = pd.read_csv(plan_root / "EVENT_RUN_PLAN.csv", keep_default_na=False)
    source = execution_root / "engineering/EVENT_RUN_MANIFEST.csv"
    executed = pd.read_csv(source, keep_default_na=False)
    keys = KEYS + ["amplitude_c"]
    paired = plan.merge(executed, on=keys, suffixes=("_planned", "_executed"), validate="one_to_one")
    if len(plan) != 432 or len(paired) != len(plan) or len(executed) != len(plan):
        raise ValueError("Event execution does not cover all predeclared parallel inputs")
    for column in ("run_id", "baseline_run_id", "normalized_idf_sha256"):
        if not paired[column + "_planned"].equals(paired[column + "_executed"]):
            raise ValueError(f"Event execution relabelled a predeclared input: {column}")
    final_states = QUALIFIED | {"cached_rejected", "new_rejected", "execution_failed", "unavailable_baseline"}
    if not set(executed.engineering_status).issubset(final_states):
        raise ValueError("Event execution is incomplete")
    for column in ("path_hourly", "qa_path"):
        if column in executed:
            executed[column] = executed[column].map(lambda value: str(resolved(source.parent, value)) if value else "")
    executed.to_csv(plan_root / "EVENT_RUN_MANIFEST.csv", index=False)
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(ENERGY_SERVICE / "analyse_events.py"),
        "--engineering-root", str(plan_root), "--annual-root", str(Path(annual_root).resolve()),
        "--factors", str(Path(factors).resolve()), "--areas", str(Path(areas).resolve()),
        "--expected-baselines", str(EXPECTED_BASELINES), "--expected-profile-groups", "24",
        "--output-root", str(output)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8")
    (output / "ANALYSIS.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("Original event analysis failed; see ANALYSIS.log")
    return json.loads((output / "EVENT_ANALYSIS_MANIFEST.json").read_text())


def costs(annual_root, factors_path, assumptions_path, zone_groups_path, cohort, output):
    """Apply the unchanged signed price/resource equations on matched scope."""
    source = Path(annual_root) / "CONTROL_COMPARISONS.csv"
    zone_groups = read_csv(zone_groups_path)
    floor_counts = {context: len({row["floor"] for row in zone_groups if row["context"] == context})
                    for context in {row["context"] for row in zone_groups}}
    pairs = [row for row in read_csv(source) if row["layout"] in LAYOUTS
             and not (row["scope"] == "floor" and floor_counts[row["climate_context"]] == 1)]
    if len(pairs) != 600:
        raise ValueError("The shared two-layout design requires all 600 paired outcome rows")
    assumptions = json.loads(Path(assumptions_path).read_text())
    scenarios = cost_scenarios(assumptions)
    factors = read_csv(factors_path)
    priced, boundaries = price_pairs(pairs, factors, scenarios)
    commands = command_scope_boundaries(pairs, factors, scenarios, assumptions, zone_groups)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("implementation_cost_scenarios.csv", scenarios),
                          ("all_signed_paired_costs.csv", priced),
                          ("bill_implementation_boundaries.csv", boundaries),
                          ("command_scope_management_boundaries.csv", commands)):
        write_cost_csv(output / name, [row | {"data_source": cohort} for row in records])
    report = {"status": "PASS", "cohort": cohort, "input_pair_rows": len(pairs),
        "unmatched_rows_preserved": sum(row["status"] != "matched" for row in pairs),
        "cost_scenarios": len(scenarios), "physical_replicas_not_used_for_annual_costs": True,
        "annual_dr_revenue_invented": False, "observed_implementation_cost": False,
        "source_sha256": {Path(path).name: digest(path) for path in
            (source, factors_path, assumptions_path, zone_groups_path)}}
    write_json(output / "COST_QA.json", report)
    return report


def scope_sensitivity(event_root, annual_root, contexts_path, output):
    """Separate portfolio size from its primary-cohort layout composition."""
    event_root, output = Path(event_root), Path(output)
    contexts = json.loads(Path(contexts_path).read_text())
    def not_alias(row):
        return not (row["scope"] == "floor"
                    and contexts[row["climate_context"]]["floor_binding_equals_building"])
    types = [row for row in read_csv(event_root / "PORTFOLIO_TYPES.csv") if not_alias(row)]
    hours = [row for row in read_csv(event_root / "PORTFOLIO_HOURLY.csv") if not_alias(row)]
    inventory = read_csv(event_root / "PORTFOLIO_DESIGN_TYPES.csv")
    if {row["layout"] for row in inventory} != LAYOUTS | {"source_structured"} or len(inventory) != 9:
        raise ValueError("This decomposition requires the original three-layout primary experiment")
    definitions = [("three_layouts_108", LAYOUTS | {"source_structured"}),
        ("shared_layouts_72", LAYOUTS), ("source_structured_36", {"source_structured"}),
        ("random_mixing_36", {"random_mixing"}), ("demand_ordered_36", {"demand_ordered"})]
    summaries, curves = [], []
    for label, layouts in definitions:
        design = [row for row in inventory if row["layout"] in layouts]
        selected_types = [row for row in types if row["layout"] in layouts]
        selected_hours = [row for row in hours if row["layout"] in layouts]
        table, hourly, _ = aggregate(selected_types, selected_hours, design,
            {"portfolio_buildings": len(design) * 12, "dr_participation_fractions": [.25, .5, 1.]})
        for row in table:
            row.update(portfolio_scope=label, data_source="ASHRAE")
        for row in hourly:
            row.update(portfolio_scope=label, data_source="ASHRAE")
        summaries.extend(table)
        curves.extend(hourly)
    summary = pd.DataFrame(summaries)
    hourly = pd.DataFrame(curves)
    group = ["portfolio_scope", "climate_context", "scope", "event_id", "dr_participation_fraction"]
    reference = summary[summary.control.eq("uniform_average")][group + ["event_average_reduction_kw"]]
    reference = reference.rename(columns={"event_average_reduction_kw": "uniform_response_kw"})
    summary = summary.merge(reference, on=group, validate="many_to_one")
    summary["response_change_from_uniform_pct"] = (
        100 * (summary.event_average_reduction_kw - summary.uniform_response_kw)
        / summary.uniform_response_kw.where(summary.uniform_response_kw.abs().gt(1e-9)))
    clock_group = ["portfolio_scope", "climate_context", "scope", "control", "event_id", "dr_participation_fraction"]
    baseline = hourly[hourly.hour_index.lt(2)].groupby(clock_group).baseline_electric_kwh.mean()
    summary = summary.merge(baseline.rename("own_baseline_event_average_kw").reset_index(),
                            on=clock_group, validate="one_to_one")
    summary["event_reduction_fraction_of_own_baseline_pct"] = (
        100 * summary.event_average_reduction_kw
        / summary.own_baseline_event_average_kw.where(summary.own_baseline_event_average_kw.gt(1e-9)))
    key = ["climate_context", "scope", "control", "event_id", "dr_participation_fraction", "hour_index"]
    values = ["baseline_electric_kwh", "perturbed_electric_kwh", "electric_change_kwh",
              "baseline_gas_hhv_kwh", "perturbed_gas_hhv_kwh", "gas_change_hhv_kwh"]
    blocks = {name: hourly[hourly.portfolio_scope.eq(name)].set_index(key)[values].sort_index()
              for name in ("three_layouts_108", "shared_layouts_72", "source_structured_36")}
    residual = blocks["three_layouts_108"] - blocks["shared_layouts_72"] - blocks["source_structured_36"]
    closure = float(residual.abs().to_numpy().max())
    if closure > 1e-9:
        raise ValueError("Original portfolio hours do not close as shared plus source-structured offices")
    case_keys = ["climate_context", "scope", "control", "event_id", "composition_id", "layout"]
    cases = pd.DataFrame(types)
    raw_hours = pd.DataFrame(hours)
    event_power = raw_hours[raw_hours.hour_index.astype(int).lt(2)].copy()
    for column in ("baseline_electric_kwh", "event_electric_kwh"):
        event_power[column] = event_power[column].astype(float)
    event_power = event_power.groupby(case_keys)[["baseline_electric_kwh", "event_electric_kwh"]].mean().reset_index()
    event_power["implemented_response_kw_per_office"] = event_power.baseline_electric_kwh - event_power.event_electric_kwh
    cases = cases.merge(event_power, on=case_keys, validate="one_to_one")
    annual = pd.read_csv(Path(annual_root) / "DR_BASELINES.csv", keep_default_na=False)
    annual = annual.rename(columns={"context": "climate_context", "permission": "scope",
                                     "run_id": "annual_baseline_run_id", "variant": "annual_baseline_variant"})
    cases["composition_id"] = cases.composition_id.astype(int)
    annual_keys = ["climate_context", "scope", "control", "composition_id", "layout"]
    cases = cases.merge(annual[annual_keys + ["annual_baseline_run_id", "annual_baseline_variant",
        "annual_site_kwh", "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"]],
        on=annual_keys, validate="many_to_one")
    if not cases.baseline_case_id.equals(cases.annual_baseline_run_id):
        raise ValueError("A component does not use its own selected annual baseline")
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "PORTFOLIO_SCOPE_SENSITIVITY.csv", index=False)
    cases.to_csv(output / "PORTFOLIO_LAYOUT_CASES.csv", index=False)
    hourly.to_csv(output / "PORTFOLIO_SCOPE_HOURLY.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    candidates = pd.read_csv(event_root / "EVENT_ALL_CANDIDATES.csv")
    candidates = candidates[candidates.service_fraction.eq(1)].copy()
    candidates = candidates[~(candidates.permission.eq("floor") & candidates.context.map(
        {name: config["floor_binding_equals_building"] for name, config in contexts.items()}))]
    for period in ("event", "recovery"):
        for metric in ("mean", "fixed_tail"):
            for trajectory in ("baseline", "perturbed"):
                candidates[f"{period}_{trajectory}_{metric}_remaining_margin_c"] = (
                    candidates[f"{period}_{metric}_budget_c"]
                    - candidates[f"{period}_{trajectory}_{metric}_c"])
    candidates.to_csv(output / "EVENT_SERVICE_MARGIN_BY_LAYOUT.csv", index=False)
    report = {"status": "PASS", "physical_simulations_added": 0,
        "scope_rows": len(summary), "component_rows": len(cases),
        "original_event_candidate_rows_at_reference_limits": len(candidates),
        "original_equals_shared_plus_source_hourly_max_residual_kwh": closure,
        "replicas_per_type": 12, "scale_rule": "Report both kW and kW/1000m2, and change versus uniform within each identical layout inventory",
        "source_sha256": {name: digest(event_root / name) for name in
            ("PORTFOLIO_TYPES.csv", "PORTFOLIO_HOURLY.csv", "PORTFOLIO_DESIGN_TYPES.csv")}}
    write_json(output / "PORTFOLIO_SCOPE_QA.json", report)
    return report


def joint_portfolio(ashrae_events, ctcd_events, ashrae_portfolio, ctcd_portfolio, contexts_path, output):
    """Combine all six predeclared compositions at synchronous physical hours."""
    contexts = json.loads(Path(contexts_path).read_text())
    types, hours, design, source = [], [], [], {}
    for cohort, root in (("ASHRAE", Path(ashrae_events)), ("CTCD", Path(ctcd_events))):
        def selected(row):
            return row["layout"] in LAYOUTS and not (row["scope"] == "floor"
                and contexts[row["climate_context"]]["floor_binding_equals_building"])
        types.extend(row | {"data_source": cohort} for row in read_csv(root / "PORTFOLIO_TYPES.csv") if selected(row))
        hours.extend(row | {"data_source": cohort} for row in read_csv(root / "PORTFOLIO_HOURLY.csv") if selected(row))
        design.extend(row for row in read_csv(root / "PORTFOLIO_DESIGN_TYPES.csv") if row["layout"] in LAYOUTS)
        source[cohort] = {name: digest(root / name) for name in
            ("PORTFOLIO_TYPES.csv", "PORTFOLIO_HOURLY.csv", "PORTFOLIO_DESIGN_TYPES.csv")}
    if len(design) != 12 or len({(row["composition_id"], row["layout"]) for row in design}) != 12:
        raise ValueError("The joint inventory must contain six distinct compositions and two layouts")
    protocol = {"portfolio_buildings": 144, "dr_participation_fractions": [.25, .5, 1.],
        "replicas_per_type": 12, "types": 12, "demand_compositions": 6,
        "scope": "Joint standardized portfolio of predeclared demand scenarios; no actual-city stock estimate"}
    summary, curves, components = aggregate(types, hours, design, protocol)
    key = ["climate_context", "scope", "control", "event_id", "dr_participation_fraction", "hour_index"]
    values = ["baseline_electric_kwh", "perturbed_electric_kwh", "electric_change_kwh",
              "baseline_gas_hhv_kwh", "perturbed_gas_hhv_kwh", "gas_change_hhv_kwh"]
    retained = [pd.read_csv(Path(root) / "portfolio_hourly.csv").set_index(key)[values].sort_index()
                for root in (ashrae_portfolio, ctcd_portfolio)]
    joint = pd.DataFrame(curves).set_index(key)[values].sort_index()
    if not joint.index.equals(retained[0].index) or not joint.index.equals(retained[1].index):
        raise ValueError("Source portfolios have different dates or physical output support")
    residual = float((joint - retained[0] - retained[1]).abs().to_numpy().max())
    if residual > 1e-9:
        raise ValueError("Joint 144-office hours do not close against both 72-office contributions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in (("portfolio_summary.csv", summary), ("portfolio_hourly.csv", curves),
                          ("portfolio_components.csv", components)):
        write_csv(output / name, [row | {"data_source": "joint_six_compositions"} for row in records])
    write_json(output / "PORTFOLIO_PROTOCOL.json", protocol)
    report = {"status": "PASS", "physical_simulations_added": 0, "actual_city_prediction": False,
        "portfolio_buildings": 144, "compositions": 6, "types": 12,
        "summary_rows": len(summary), "hourly_rows": len(curves),
        "joint_equals_two_source_portfolios_hourly_max_residual_kwh": residual,
        "synchronous_recovery_peak_recomputed": True, "source_sha256": source}
    write_json(output / "PORTFOLIO_QA.json", report)
    return report


def joint_events(ashrae_events, ctcd_events, joint_portfolio_root, contexts_path, output):
    """Recompute sequential service gates for the six fixed demand compositions."""
    contexts = json.loads(Path(contexts_path).read_text())
    candidates, selections, types, hours, checks, source = [], [], [], [], [], {}
    for cohort, root in (("ASHRAE", Path(ashrae_events)), ("CTCD", Path(ctcd_events))):
        table, check = read_scope(root / "EVENT_ALL_CANDIDATES.csv", root / "EVENT_HOURLY.csv.gz",
                                  "main_peak_dates")
        table = mark_spatial_aliases(table, contexts)
        table = table[table.layout.isin(LAYOUTS) & table.distinct_spatial_condition].copy()
        table["data_source"] = cohort
        candidates.append(table)
        cases = pd.read_csv(root / "EVENT_SELECTIONS.csv")
        cases = cases[cases.layout.isin(LAYOUTS) & ~(cases.permission.eq("floor") & cases.context.map(
            {name: config["floor_binding_equals_building"] for name, config in contexts.items()}))].copy()
        cases["data_source"] = cohort
        selections.append(cases)
        for name, collection in (("PORTFOLIO_TYPES.csv", types), ("PORTFOLIO_HOURLY.csv", hours)):
            collection.extend(row for row in read_csv(root / name) if row["layout"] in LAYOUTS
                and not (row["scope"] == "floor" and contexts[row["climate_context"]]["floor_binding_equals_building"]))
        checks.append(check | {"data_source": cohort})
        source[cohort] = {name: digest(root / name) for name in
            ("EVENT_ALL_CANDIDATES.csv", "EVENT_HOURLY.csv.gz", "EVENT_SELECTIONS.csv",
             "PORTFOLIO_TYPES.csv", "PORTFOLIO_HOURLY.csv")}
    table = pd.concat(candidates, ignore_index=True)
    cases = pd.concat(selections, ignore_index=True)
    identity = KEYS + ["event_month", "service_fraction"]
    if len(table) != 5760 or len(cases) != 2880 or cases.duplicated(identity).any():
        raise ValueError("The joint event scope must contain six compositions and 120 distinct spatial conditions")
    stage = select_stages(table)
    final = stage[stage.stage.eq("recovery_qualified")]
    verify = cases.merge(final, on=identity, suffixes=("_original", "_independent"), validate="one_to_one")
    if (not verify.selection_status.eq("selected").equals(verify.selected)
            or not verify.selected_run_id_original.fillna("").equals(verify.selected_run_id_independent.fillna(""))
            or (verify.executed_response_kw_with_baseline_fallback - verify.response_kw_with_no_action_zero).abs().max() > 1e-9):
        raise ValueError("Sequential gate recomputation changed an original selected response")
    baseline_columns = [name for name in table if "_baseline_" in name or name.endswith("_budget_c")]
    baseline = table.drop_duplicates(identity)[identity + baseline_columns]
    cases = cases.merge(baseline, on=identity, suffixes=("", "_candidate_baseline"), validate="one_to_one")
    for column in baseline_columns:
        retained = column + "_candidate_baseline"
        if retained in cases:
            cases[column] = cases[column].combine_first(cases.pop(retained))
    for frame in (table, cases):
        for window in ("event", "recovery"):
            for trajectory in ("baseline", "perturbed"):
                for metric in ("mean", "fixed_tail"):
                    budget = pd.to_numeric(frame[f"{window}_{metric}_budget_c"], errors="coerce")
                    column = f"{window}_{trajectory}_{metric}_c"
                    if column not in frame:
                        column = f"{window}_{trajectory}_{metric}_reference_mismatch_c"
                    score = pd.to_numeric(frame[column], errors="coerce")
                    frame[f"{window}_{trajectory}_{metric}_ratio"] = score / budget.where(budget.gt(0))
    selected = cases.selection_status.eq("selected")
    cases["recovery_fraction_of_event_saving"] = (cases.recovery_extra_electric_kwh
        / cases.event_electric_saving_kwh.where(selected & cases.event_electric_saving_kwh.gt(1e-9)))
    cases["net_electric_saving_with_no_action_zero_kwh"] = cases.event_plus48h_net_electric_saving_kwh.where(
        selected, 0.).where(cases.baseline_service_qualified)
    cases["recovery_extra_with_no_action_zero_kwh"] = cases.recovery_extra_electric_kwh.where(
        selected, 0.).where(cases.baseline_service_qualified)
    stages = summarise(stage, ["event_month", "service_fraction", "stage"])
    by_control = summarise(stage, ["event_month", "service_fraction", "control", "stage"])
    portfolio_rows = pd.read_csv(Path(joint_portfolio_root) / "portfolio_summary.csv")
    portfolio_rows["result_scope"] = "joint_144_offices"
    portfolio_rows["composition_id"] = "all_six"
    compositions = sorted(cases.composition_id.unique())
    if len(compositions) != 6:
        raise ValueError("Expected all six predeclared demand compositions")
    grouped = [portfolio_rows]
    for index, composition in enumerate(compositions, 1):
        design = [{"composition_id": str(composition), "layout": layout} for layout in sorted(LAYOUTS)]
        protocol = {"portfolio_buildings": 24, "dr_participation_fractions": [.25, .5, 1.],
                    "replicas_per_type": 12, "types": 2}
        records, _, _ = aggregate([row for row in types if int(row["composition_id"]) == composition],
            [row for row in hours if int(row["composition_id"]) == composition], design, protocol)
        grouped.append(pd.DataFrame(records).assign(result_scope="one_composition_24_offices",
            composition_id=composition, composition_label=f"Composition {index}"))
    responses = pd.concat(grouped, ignore_index=True)
    full = responses[responses.dr_participation_fraction.eq(1)]
    response_keys = ["climate_context", "scope", "control", "event_id"]
    additive = ["event_average_reduction_kw", "event_electric_saving_kwh", "recovery_extra_electric_kwh",
                "event_plus_48h_net_electric_saving_kwh", "event_plus_48h_gas_change_kwh_hhv"]
    total = full[full.result_scope.eq("joint_144_offices")].set_index(response_keys)[additive].sort_index()
    parts = full[full.result_scope.eq("one_composition_24_offices")].groupby(response_keys)[additive].sum().sort_index()
    residual = float((total - parts).abs().to_numpy().max())
    if residual > 1e-9:
        raise ValueError("Per-composition response accounting does not close against the joint inventory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("JOINT_EVENT_CANDIDATE_GATES.csv", table),
                        ("JOINT_EVENT_CASE_SERVICE_ENERGY.csv", cases),
                        ("JOINT_EVENT_PORTFOLIO_RESPONSE.csv", responses),
                        ("JOINT_EVENT_STAGE_DECISIONS.csv", stage),
                        ("JOINT_EVENT_STAGE_COUNTS.csv", stages),
                        ("JOINT_EVENT_STAGE_COUNTS_BY_CONTROL.csv", by_control)):
        frame.to_csv(output / name, index=False)
    july = stages[stages.event_month.eq(7) & stages.service_fraction.eq(1)]
    recovered = cases[selected & cases.event_month.eq(7) & cases.service_fraction.eq(1)]
    report = {"status": "PASS", "physical_simulations_added": 0, "compositions": 6,
        "layouts": sorted(LAYOUTS), "distinct_spatial_conditions": 120,
        "candidate_budget_windows": len(table), "controller_budget_windows": len(cases),
        "portfolio_response_rows": len(responses), "portfolio_buildings": 144,
        "source_labels_are_provenance_only": True, "hourly_accounting_checks": checks,
        "composition_sums_vs_joint_max_residual_kwh": residual,
        "july_fraction1_selected_decisions_by_stage": dict(zip(july.stage, july.decisions_with_positive_selection)),
        "july_fraction1_positive_selected_cases": len(recovered),
        "july_fraction1_median_individual_recovery_fraction": float(recovered.recovery_fraction_of_event_saving.median()),
        "source_sha256": source, "joint_portfolio_qa_sha256": digest(Path(joint_portfolio_root) / "PORTFOLIO_QA.json")}
    write_json(output / "JOINT_EVENT_QA.json", report)
    return report


def joint_spatial_diagnostics(ashrae_spatial, ctcd_spatial, output):
    """Pool the existing paired spatial diagnostics without changing their weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, counts = {}, {}
    for name in ("SPATIAL_PAIRS.csv", "SPATIAL_INTERACTIONS.csv"):
        parts = []
        for cohort, root in (("ASHRAE", Path(ashrae_spatial)), ("CTCD", Path(ctcd_spatial))):
            path = root / name
            frame = pd.read_csv(path)
            if name == "SPATIAL_PAIRS.csv":
                keep = ((frame.axis.eq("layout") & frame.alternative.eq("demand_ordered"))
                        | (frame.axis.eq("permission") & frame.layout.isin(LAYOUTS)))
            else:
                keep = frame.layout.eq("demand_ordered")
            frame = frame[keep].copy()
            frame["data_source"] = cohort
            parts.append(frame)
            source[f"{cohort}/{name}"] = digest(path)
        pooled = pd.concat(parts, ignore_index=True)
        expected = 3960 if name == "SPATIAL_PAIRS.csv" else 1080
        if len(pooled) != expected or pooled.composition_id.nunique() != 6:
            raise ValueError("Spatial diagnostics lost a declared composition or retained a source-only layout")
        pooled.to_csv(output / name, index=False)
        counts[name] = len(pooled)
    report = {"status": "PASS", "physical_simulations_added": 0, "demand_compositions": 6,
        "layouts": sorted(LAYOUTS), "weight_views": ["profile_equal", "native_occupancy"],
        "scope": "Original paired and four-corner spatial contrasts; same random-layout service anchor and fixed tail",
        "rows": counts, "source_sha256": source}
    write_json(output / "JOINT_SPATIAL_DIAGNOSTICS_QA.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("prepare")
    for name in ("annual-engineering", "annual-root", "commands", "model-root", "primary-root", "output-root"):
        plan.add_argument("--" + name, required=True, type=Path)
    port = sub.add_parser("portfolio")
    for name in ("event-root", "contexts", "output-root"):
        port.add_argument("--" + name, required=True, type=Path)
    port.add_argument("--cohort", required=True, choices=("ASHRAE", "CTCD"))
    price = sub.add_parser("costs")
    for name in ("annual-root", "factors", "assumptions", "zone-groups", "output-root"):
        price.add_argument("--" + name, required=True, type=Path)
    price.add_argument("--cohort", required=True, choices=("ASHRAE", "CTCD"))
    evaluate = sub.add_parser("analyse")
    for name in ("plan-root", "execution-root", "annual-root", "factors", "areas", "output-root"):
        evaluate.add_argument("--" + name, required=True, type=Path)
    scope = sub.add_parser("scope-sensitivity")
    for name in ("event-root", "annual-root", "contexts", "output-root"):
        scope.add_argument("--" + name, required=True, type=Path)
    joint = sub.add_parser("joint-portfolio")
    for name in ("ashrae-events", "ctcd-events", "ashrae-portfolio", "ctcd-portfolio", "contexts", "output-root"):
        joint.add_argument("--" + name, required=True, type=Path)
    events = sub.add_parser("joint-events")
    for name in ("ashrae-events", "ctcd-events", "joint-portfolio-root", "contexts", "output-root"):
        events.add_argument("--" + name, required=True, type=Path)
    spatial = sub.add_parser("joint-spatial-diagnostics")
    for name in ("ashrae-spatial", "ctcd-spatial", "output-root"):
        spatial.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.annual_engineering, args.annual_root, args.commands,
                         args.model_root, args.primary_root, args.output_root)
    elif args.command == "portfolio":
        result = portfolio(args.event_root, args.contexts, args.cohort, args.output_root)
    elif args.command == "costs":
        result = costs(args.annual_root, args.factors, args.assumptions, args.zone_groups,
                       args.cohort, args.output_root)
    elif args.command == "analyse":
        result = analyse(args.plan_root, args.execution_root, args.annual_root,
                         args.factors, args.areas, args.output_root)
    elif args.command == "scope-sensitivity":
        result = scope_sensitivity(args.event_root, args.annual_root, args.contexts, args.output_root)
    elif args.command == "joint-events":
        result = joint_events(args.ashrae_events, args.ctcd_events, args.joint_portfolio_root,
                              args.contexts, args.output_root)
    elif args.command == "joint-spatial-diagnostics":
        result = joint_spatial_diagnostics(args.ashrae_spatial, args.ctcd_spatial, args.output_root)
    else:
        result = joint_portfolio(args.ashrae_events, args.ctcd_events, args.ashrae_portfolio,
                                  args.ctcd_portfolio, args.contexts, args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
