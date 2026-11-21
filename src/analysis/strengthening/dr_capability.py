"""Common-service, bounded-normal-energy, isolated-event response comparisons."""

from common import *
import pandas as pd
import numpy as np
import concurrent.futures as cf
import argparse
from contextlib import contextmanager
import multi_event_dr_extension as dr
from analyse_global_results import choose_minimum_energy, energy_metrics, load_hourly
from single_event_simulation import execute, reported_model, history_check, SIM, ENERGY_METERS
from run_global_matrix import warning_profile, set_maximum_hvac_iterations
from exact_dr_cache import enable
from lossless_history_storage import load_states

OUT = ROOT / "results/dr_capability"
LABELS = {
    2026102001: "Multinational A",
    2026102002: "Multinational B",
    2026102003: "Multinational C",
    2026102501: "China A",
    2026102502: "China B",
    2026102503: "China C",
}
NORMAL_LIBRARIES = {
    "building_search": "Building command search",
    "zone_summary": "Building and zone mean/median commands",
    "zone_full": "Pooled building, floor and full-demand zone commands",
}


@contextmanager
def response_command_lock():
    """Serialize commands that share the same physical response trajectories."""
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    api.CreateMutexW.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReleaseMutex.argtypes = [wintypes.HANDLE]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    name = "Local\\HEM_DR_" + hashlib.sha256(str(SIM.resolve()).casefold().encode()).hexdigest()[:24]
    handle = api.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        result = api.WaitForSingleObject(handle, 0xFFFFFFFF)
        if result not in [0, 0x80]:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = True
        yield
    finally:
        if acquired:
            api.ReleaseMutex(handle)
        api.CloseHandle(handle)


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def baseline_library(group, control):
    selected = group.permission.eq("building")
    if control in {"zone_summary", "zone_full"}:
        selected |= group.permission.eq("native_zone") & group.control.isin(
            ["native_zone_mean", "native_zone_median"])
    if control == "zone_full":
        selected |= group.permission.isin(["floor", "native_zone"])
    if control not in NORMAL_LIBRARIES:
        raise ValueError("Unknown normal-operation library: " + control)
    return group[selected].sort_values("run_id").drop_duplicates("physical_plan_id")


def plan(protocol_path=None):
    coverage = json.loads((ROOT / "results/budget_frontiers/BUDGET_FRONTIER_AUDIT.json").read_text())
    if coverage["status"] != "completed":
        raise ValueError("Complete the declared strong-baseline search and quality review before freezing response baselines")
    stop = json.loads((ROOT / "audit/GRID_STOP.json").read_text())
    assert stop["status"] == "completed"
    expected = 2754 if stop["further_refinement_executed"] else 1458
    assert (
        len(pd.read_csv(ROOT / "results/candidate_refinement/STOPPED_GRID_METRICS.csv")) == expected
    ), "Required candidate scores are incomplete"
    assert (
        json.loads((ROOT / "audit/status/original_response_reproduction.json").read_text())[
            "status"
        ]
        == "completed"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(ROOT / "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv")
    annual = pd.read_csv(MATRIX / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv")
    manifests = [annual, pd.read_csv(ROOT / "results/CANDIDATE_FULL_MANIFEST.csv")]
    for name in [
        "BASELINE_FULL_MANIFEST.csv",
        "BASELINE_OFFSETS_FULL_MANIFEST.csv",
        "UNIFORM_SEARCH_FULL_MANIFEST.csv",
    ]:
        manifests.append(pd.read_csv(ROOT / "results" / name))
    manifests.append(pd.read_csv(RAW_RUN / "spatial/GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv"))
    further = ROOT / "results/CANDIDATE_FURTHER_MANIFEST.csv"
    if further.exists():
        manifests.append(pd.read_csv(further))
    manifest = pd.concat(manifests, ignore_index=True).drop_duplicates("run_id").set_index("run_id")
    anchors = pd.read_csv(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv")
    probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    probes = probes[probes.target_c.eq(25)].set_index("climate")
    protocol_path = protocol_path or RAW_RUN / "response/EVENT_PROTOCOL.csv"
    protocol = pd.read_csv(protocol_path)
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    status = []
    normal_rows = []
    logical_actions = []
    normal_jobs = {}
    library_members = []
    idfroot = SIM / "dr_capability_inputs"
    idfroot.mkdir(exist_ok=True)
    for (climate, composition), g in metrics.groupby(["climate", "composition"], sort=True):
        comp = next(k for k, v in LABELS.items() if v == composition)
        a = anchors[anchors.climate.eq(climate) & anchors.composition.eq(composition)].iloc[0]
        ref = load_hourly(Path(probes.loc[climate, "hourly_path"]))
        reference_energy = energy_metrics(ref)["facility_purchased_kwh"]
        p = profiles[profiles.context.eq(climate) & profiles.composition_id.eq(comp)]
        previous_library = set()
        for control in NORMAL_LIBRARIES:
            family = baseline_library(g, control)
            physical = set(family.physical_plan_id)
            assert previous_library <= physical
            previous_library = physical
            library_members.extend({"climate": climate, "composition_id": comp,
                "normal_library": control, "physical_plan_id": row.physical_plan_id,
                "run_id": row.run_id, "operation_origin": row.control,
                "normal_command_permission": row.permission} for row in family.itertuples())
            eligible = family[
                family.engineering_pass
                & family.mean_reference_mismatch_c.le(a.anchor_mean_mismatch_c + 1e-9)
                & family.fixed_tail_reference_mismatch_c.le(a.anchor_tail_mismatch_c + 1e-9)
            ]
            chosen = choose_minimum_energy(eligible)
            for budget in [0.95, 1.0, 1.05]:
                feasible = (
                    chosen is not None
                    and chosen["facility_purchased_kwh"] <= budget * reference_energy + 1e-6
                )
                status.append(
                    {
                        "climate": climate,
                        "composition_id": comp,
                        "composition": composition,
                        "control": control,
                        "service_reference_factor": 1.0,
                        "normal_energy_reference_factor": budget,
                        "reference_energy_kwh": reference_energy,
                        "selected_normal_energy_kwh": chosen["facility_purchased_kwh"]
                        if chosen
                        else None,
                        "selected_run_id": chosen["run_id"] if chosen else None,
                        "baseline_library": NORMAL_LIBRARIES[control],
                        "selected_operation_origin": chosen["control"] if chosen else None,
                        "selected_operation_permission": chosen["permission"] if chosen else None,
                        "status": "feasible"
                        if feasible
                        else (
                            "infeasible_service_budget"
                            if chosen is None
                            else "infeasible_normal_energy_budget"
                        ),
                    }
                )
            if chosen is None or chosen["facility_purchased_kwh"] > 1.05 * reference_energy + 1e-6:
                continue
            actual = manifest.loc[chosen["run_id"]]
            source_path = Path(actual.input_idf_path)
            assert digest(source_path) == chosen["input_sha256"]
            source = source_path.read_text()
            weather = (
                Path(actual.weather_path)
                if pd.notna(actual.get("weather_path"))
                else Path(
                    json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())[climate][
                        "weather_path"
                    ]
                )
            )
            normal_id = (
                "normal_"
                + hashlib.sha256((chosen["input_sha256"] + digest(weather)).encode()).hexdigest()[
                    :24
                ]
            )
            normal_idf = idfroot / (normal_id + ".idf")
            normal_idf.write_text(reported_model(source))
            base = {
                "climate": climate,
                "composition_id": comp,
                "composition": composition,
                "control": control,
                "normal_id": normal_id,
                "input_idf_path": str(normal_idf),
                "source_input_idf_path": str(source_path),
                "source_hourly_path": str(actual.hourly_path),
                "source_qa_path": str(actual.qa_path),
                "source_run_id": chosen["run_id"],
                "selected_offset_c": float(chosen["offset_c"]),
                "weather_path": str(weather),
                "selected_normal_energy_kwh": chosen["facility_purchased_kwh"],
                "reference_energy_kwh": reference_energy,
            }
            base.update(
                selected_operation_origin=chosen["control"],
                selected_operation_permission=chosen["permission"],
                baseline_library=NORMAL_LIBRARIES[control],
            )
            normal_rows.append(base)
            normal_jobs[normal_id] = base
            local_protocol = protocol[protocol.climate.eq(climate)]
            assert len(local_protocol) == 8
            for event in local_protocol.to_dict("records"):
                variants, _ = dr.action_variants(
                    pd.DataFrame([event]), p, list(ref["zone_ids"]), protected_fraction=0.2
                )
                for variant in variants:
                    event_action = variant["events"][0]
                    text = dr.event_specific_action_input(source, [event_action])
                    end = pd.Timestamp(event["start_timestamp"]) + pd.Timedelta(
                        hours=int(event["duration_hours"]) + int(event["recovery_hours"])
                    )
                    text = reported_model(text)
                    key = hashlib.sha256((text + digest(weather) + normal_id).encode()).hexdigest()[
                        :24
                    ]
                    run_id = "isolated_" + key
                    path = idfroot / (run_id + ".idf")
                    if not path.exists():
                        path.write_text(text)
                    else:
                        assert path.read_text() == text
                    logical_actions.append(
                        {
                            **base,
                            "run_id": run_id,
                            "input_idf_path": str(path),
                            "event_id": event["event_id"],
                            "action_id": variant["action_id"],
                            "action_family": variant["action_family"],
                            "permission": variant["permission"],
                            "event_json": json.dumps(event, default=str),
                            "action_json": json.dumps(event_action, default=str),
                            "input_sha256": digest(path),
                            "initialization": "same year-start normal-operation history; exactly one event action",
                            "physical_run_end_day": "2001-12-31",
                            "accounting_window_end": str(end),
                        }
                    )
    pd.DataFrame(status).to_csv(OUT / "NORMAL_OPERATION_BUDGET_STATUS.csv", index=False)
    membership = pd.DataFrame(library_members)
    membership.to_csv(OUT / "NORMAL_OPERATION_LIBRARY_MEMBERSHIP.csv", index=False)
    pd.DataFrame(normal_rows).to_csv(OUT / "NORMAL_OPERATION_LOGICAL_PLAN.csv", index=False)
    pd.DataFrame(normal_jobs.values()).to_csv(OUT / "NORMAL_OPERATION_REPLAY_PLAN.csv", index=False)
    pd.DataFrame(logical_actions).to_csv(OUT / "SINGLE_EVENT_ACTION_PLAN.csv", index=False)
    write_json(
        OUT / "PLAN_AUDIT.json",
        {
            "annual_settings": 54,
            "normal_budget_cells": len(status),
            "normal_budget_feasible_cells": sum(x["status"] == "feasible" for x in status),
            "logical_selected_baselines": len(normal_rows),
            "distinct_normal_replays": len(normal_jobs),
            "logical_event_actions": len(logical_actions),
            "distinct_event_actions": len({x["run_id"] for x in logical_actions}),
            "baseline_selection_uses_event_results": False,
            "perfect_foresight_action_selection": "conditional upper bound",
            "event_protocol_sha256": digest(protocol_path),
            "library_membership_sha256": digest(OUT / "NORMAL_OPERATION_LIBRARY_MEMBERSHIP.csv"),
            "normal_libraries": NORMAL_LIBRARIES,
            "comparison": "Nested normal-operation libraries under common individual service and normal-energy budgets. The same five event actions use shared native-zone execution capability and demand-based protection rules in every library.",
        },
    )
    print("DR_PLANNED", len(normal_jobs), len(logical_actions), flush=True)


def baseline_one(row):
    original = arrays(row.source_hourly_path)
    qa = execute(Path(row.input_idf_path), Path(row.weather_path), original, row.normal_id)
    if not qa["completed"]:
        return {**row._asdict(), "completed": False, "qa": json.dumps(qa)}
    replay = arrays(qa["compact_output"])
    differences = {
        key: float(abs(original[key] - replay[key]).max())
        for key in ["temperatures", "people", *ENERGY_METERS]
    }
    equal = all(value <= 1e-6 for value in differences.values())
    return {
        **row._asdict(),
        "completed": True,
        "normal_replay_matches_original": equal,
        "hourly_path": qa["compact_output"],
        "states_path": qa["state_output"],
        "hourly_sha256": qa["hourly_sha256"],
        "states_sha256": qa["states_sha256"],
        "history_replay_differences": json.dumps(differences),
        "qa": json.dumps(qa),
    }


def baseline_execute():
    plan = pd.read_csv(OUT / "NORMAL_OPERATION_REPLAY_PLAN.csv")
    records = []
    workers = int(
        json.loads((ROOT / "config/PROJECT.json").read_text()).get("confirmation_workers", 6)
    )
    workers = min(workers, 4)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future = {pool.submit(baseline_one, row): row for row in plan.itertuples(index=False)}
        for f in cf.as_completed(future):
            row = future[f]
            try:
                r = f.result()
            except Exception as e:
                r = {
                    **row._asdict(),
                    "completed": False,
                    "normal_replay_matches_original": False,
                    "exception": repr(e),
                }
            records.append(r)
            pd.DataFrame(records).to_csv(OUT / "NORMAL_OPERATION_REPLAY_PROGRESS.csv", index=False)
            print(
                "NORMAL_REPLAY",
                len(records),
                len(plan),
                r.get("normal_replay_matches_original"),
                flush=True,
            )
    pd.DataFrame(records).to_csv(OUT / "NORMAL_OPERATION_REPLAY_COMPLETED.csv", index=False)
    audit = {
        "planned": len(plan),
        "completed": sum(r.get("completed", False) for r in records),
        "all_reporting_only_replays_equal": all(
            r.get("normal_replay_matches_original", False) for r in records
        ),
    }
    write_json(OUT / "NORMAL_REPLAY_AUDIT.json", audit)


def action_one(row, normal):
    baseline = arrays(normal.hourly_path)
    baseline_states = load_states(normal.states_path)
    primary_input = Path(row.input_idf_path)
    qa = execute(
        primary_input,
        Path(row.weather_path),
        baseline,
        row.run_id,
        baseline_qa=json.loads(normal.qa),
    )
    result = {
        "run_id": row.run_id,
        "completed": qa["completed"],
        "final_engineering_qualified": False,
        "qa": json.dumps(qa),
    }
    if not qa["completed"]:
        return result
    action = arrays(qa["compact_output"])
    states = load_states(qa["state_output"])
    event = json.loads(row.event_json)
    clock = dr.compact_timestamps(baseline)
    pre = pd.Timestamp(event["start_timestamp"]) - pd.Timedelta(
        hours=int(event["precondition_hours"])
    )
    check = history_check(baseline, action, baseline_states, states, int((clock < pre).sum()))
    if not check["history_matches"]:
        # If shortening the run changes any pre-event state, retain the attempt
        # and use a full-year simulation of this single event instead.
        from global_matrix import parse_idf, serialize_idf
        from hvac_models import idd_fields

        objs = parse_idf(Path(row.input_idf_path).read_text())
        fields = idd_fields()["runperiod"]
        for o in objs:
            if o[0].lower() == "runperiod":
                o[fields.index("End Month") + 1] = "12"
                o[fields.index("End Day of Month") + 1] = "31"
        path = Path(row.input_idf_path).with_name(row.run_id + "_full_year.idf")
        path.write_text(serialize_idf(objs))
        qa = execute(
            path,
            Path(row.weather_path),
            baseline,
            row.run_id + "_full_year",
            baseline_qa=json.loads(normal.qa),
        )
        primary_input = path
        action = arrays(qa["compact_output"])
        states = load_states(qa["state_output"])
        check = history_check(baseline, action, baseline_states, states, int((clock < pre).sum()))
        result["full_year_history_fallback"] = True
    warnings = qa["warning_profile"]
    coil = warnings.get("coil_operating_range", 0) > 0
    numerical = warnings.get("hvac_iterations", 0) > 0
    stable = True
    stability = {}
    if numerical and not coil:
        path = primary_input.with_name(primary_input.stem + "_100.idf")
        path.write_text(set_maximum_hvac_iterations(primary_input.read_text(), 100))
        higher = execute(
            path,
            Path(row.weather_path),
            baseline,
            row.run_id + "_100",
            baseline_qa=json.loads(normal.qa),
        )
        stable = False
        if higher["completed"]:
            b = arrays(higher["compact_output"])
            n = min(len(action["month"]), len(b["month"]))
            temperature = float(abs(action["temperatures"][:n] - b["temperatures"][:n]).max())
            energy = sum(action[k][:n].sum() for k in ["facility_electric_kwh", "gas_hhv_kwh"])
            delta = abs(
                energy - sum(b[k][:n].sum() for k in ["facility_electric_kwh", "gas_hhv_kwh"])
            ) / max(abs(energy), 1e-9)
            stable = bool(
                temperature <= 0.02
                and delta <= 0.001
                and higher["reported_severe_count"] == 0
                and higher["fatal_occurrences"] == 0
                and higher["warning_profile"].get("coil_operating_range", 0) == 0
            )
            stability = {
                "temperature_delta_c": temperature,
                "period_purchased_energy_relative_delta": float(delta),
            }

    def capacity_lines(p):
        return [
            x.strip()
            for x in Path(p).read_text(errors="replace").splitlines()
            if x.strip().startswith(
                (
                    "Component Sizing Information,",
                    "Zone Sizing Information,",
                    "System Sizing Information,",
                )
            )
        ]

    normal_folder = Path(json.loads(normal.qa)["run_folder"])
    capacity_same = capacity_lines(Path(qa["run_folder"]) / "eplusout.eio") == capacity_lines(
        normal_folder / "eplusout.eio"
    )
    passed = bool(
        check["history_matches"]
        and not coil
        and stable
        and qa["reported_severe_count"] == 0
        and qa["fatal_occurrences"] == 0
        and capacity_same
    )
    result.update(
        {
            "hourly_path": qa["compact_output"],
            "states_path": qa["state_output"],
            "hourly_sha256": qa["hourly_sha256"],
            "history_check": json.dumps(check),
            "solver_stability": json.dumps(stability),
            "capacity_equal_to_frozen_normal_operation": capacity_same,
            "coil_operating_range_warning": coil,
            "final_engineering_qualified": passed,
            "qa": json.dumps(qa),
        }
    )
    return result


def action_execute(stage):
    audit = json.loads((OUT / "NORMAL_REPLAY_AUDIT.json").read_text())
    assert audit["all_reporting_only_replays_equal"]
    jobs = pd.read_csv(OUT / "SINGLE_EVENT_ACTION_PLAN.csv")
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    if stage == "pilot":
        jobs = jobs[
            jobs.climate.isin(pilot["climates"]) & jobs.composition_id.isin(pilot["compositions"])
        ]
    else:
        assert json.loads((OUT / "ACTION_PILOT_AUDIT.json").read_text())[
            "all_complete_and_engineering_pass"
        ]
    jobs = jobs.drop_duplicates("run_id")
    normal = pd.read_csv(OUT / "NORMAL_OPERATION_REPLAY_COMPLETED.csv").set_index("normal_id")
    workers = (
        6
        if stage == "pilot"
        else int(
            json.loads((ROOT / "config/PROJECT.json").read_text()).get("confirmation_workers", 6)
        )
    )
    workers = min(workers, 4)
    records = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future = {
            pool.submit(action_one, row, normal.loc[row.normal_id]): row
            for row in jobs.itertuples(index=False)
        }
        for f in cf.as_completed(future):
            row = future[f]
            try:
                r = f.result()
            except Exception as e:
                r = {
                    "run_id": row.run_id,
                    "completed": False,
                    "final_engineering_qualified": False,
                    "exception": repr(e),
                }
            records.append(r)
            pd.DataFrame(records).to_csv(
                OUT / ("ACTION_" + stage.upper() + "_PROGRESS.csv"), index=False
            )
            print(
                "ISOLATED_EVENT",
                stage,
                len(records),
                len(jobs),
                r["final_engineering_qualified"],
                flush=True,
            )
    pd.DataFrame(records).to_csv(OUT / ("ACTION_" + stage.upper() + "_COMPLETED.csv"), index=False)
    report = {
        "physical_operations": len(jobs),
        "completed": sum(r["completed"] for r in records),
        "all_complete_and_engineering_pass": all(
            r["completed"] and r["final_engineering_qualified"] for r in records
        ),
    }
    write_json(OUT / ("ACTION_" + stage.upper() + "_AUDIT.json"), report)
    record_status(
        OUT.name + "_" + stage,
        "completed" if report["all_complete_and_engineering_pass"] else "partial",
        report,
    )


def score(protocol_path=None):
    planned = pd.read_csv(OUT / "SINGLE_EVENT_ACTION_PLAN.csv")
    completed = pd.read_csv(OUT / "ACTION_FULL_COMPLETED.csv").set_index("run_id")
    normal = pd.read_csv(OUT / "NORMAL_OPERATION_REPLAY_COMPLETED.csv").set_index("normal_id")
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    probes = probes[probes.target_c.eq(25)].set_index("climate")
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    enable(dr)
    base_tail = dr.fixed_tail_profile_ids

    def annual_tail(reference, g, tail_fraction=0.1):
        return reference["__annual_tail_ids"]

    dr.fixed_tail_profile_ids = annual_tail
    rows = []
    for row in planned.itertuples(index=False):
        done = completed.loc[row.run_id]
        if not done.completed or not done.final_engineering_qualified:
            rows.append(
                {
                    "climate": row.climate,
                    "composition_id": row.composition_id,
                    "control": row.control,
                    "event_id": row.event_id,
                    "action_id": row.action_id,
                    "status": "engineering_failure",
                    "run_id": row.run_id,
                }
            )
            continue
        action = arrays(done.hourly_path)
        n = len(action["month"])
        baseline_full = arrays(normal.loc[row.normal_id, "hourly_path"])
        ref_full = arrays(probes.loc[row.climate, "hourly_path"])

        def truncate(d):
            return {
                k: v[:n] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == 8760 else v
                for k, v in d.items()
            }

        baseline = truncate(baseline_full)
        reference = truncate(ref_full)
        reference["__annual_tail_ids"] = set(
            members[
                members.climate.eq(row.climate) & members.composition.eq(row.composition)
            ].profile_id.astype(str)
        )
        g = profiles[
            profiles.context.eq(row.climate) & profiles.composition_id.eq(row.composition_id)
        ]
        event = json.loads(row.event_json)
        scored = dr.score_event(baseline, action, reference, g, event, limit_factors=(1.0,))[0]
        clock = dr.compact_timestamps(baseline)
        start = pd.Timestamp(event["start_timestamp"])
        mask = (clock >= start) & (clock < start + pd.Timedelta(hours=int(event["duration_hours"])))
        baseline_kw = float(baseline["facility_electric_kwh"][mask].mean())
        rows.append(
            {
                "climate": row.climate,
                "composition_id": row.composition_id,
                "control": row.control,
                "normal_id": row.normal_id,
                "event_id": row.event_id,
                "action_id": row.action_id,
                "run_id": row.run_id,
                "status": "evaluated",
                "baseline_facility_mean_kw": baseline_kw,
                "absolute_action_facility_mean_kw": baseline_kw
                - scored["event_facility_mean_power_reduction_kw"],
                **scored,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "ISOLATED_EVENT_ACTION_SCORES.csv", index=False)
    budget = pd.read_csv(OUT / "NORMAL_OPERATION_BUDGET_STATUS.csv")
    protocol_path = protocol_path or RAW_RUN / "response/EVENT_PROTOCOL.csv"
    if digest(protocol_path) != json.loads((OUT / "PLAN_AUDIT.json").read_text())["event_protocol_sha256"]:
        raise ValueError("The frozen event protocol changed after normal-operation selection")
    protocol = pd.read_csv(protocol_path)
    choices = []
    for row in budget.itertuples(index=False):
        for event in protocol[protocol.climate.eq(row.climate)].itertuples(index=False):
            item = {
                "climate": row.climate,
                "composition_id": row.composition_id,
                "control": row.control,
                "normal_energy_reference_factor": row.normal_energy_reference_factor,
                "event_id": event.event_id,
                "baseline_status": row.status,
                "selection_information": "perfect future response and recovery from finite action library",
            }
            if row.status != "feasible":
                choices.append({**item, "status": row.status})
                continue
            g = frame[
                frame.climate.eq(row.climate)
                & frame.composition_id.eq(row.composition_id)
                & frame.control.eq(row.control)
                & frame.event_id.eq(event.event_id)
            ]
            if len(g) != 5 or g.status.eq("engineering_failure").any():
                choices.append({**item, "status": "unresolved_engineering_operation"})
                continue
            valid = g[g.qualified_positive_event_response.fillna(False).astype(bool)]
            if valid.empty:
                choices.append(
                    {**item, "status": "infeasible_service_preserving_positive_response"}
                )
                continue
            selected = valid.sort_values(
                ["event_facility_mean_power_reduction_kw", "action_id"], ascending=[False, True]
            ).iloc[0]
            positive_net = valid[valid.precondition_event_recovery_facility_net_reduction_kwh.gt(0)]
            net_choice = (
                positive_net.sort_values(
                    ["event_facility_mean_power_reduction_kw", "action_id"], ascending=[False, True]
                ).iloc[0]
                if len(positive_net)
                else None
            )
            choices.append(
                {
                    **item,
                    "status": "selected",
                    "selected_action": selected.action_id,
                    "baseline_facility_mean_kw": selected.baseline_facility_mean_kw,
                    "absolute_event_facility_mean_kw": selected.absolute_action_facility_mean_kw,
                    "event_power_reduction_kw": selected.event_facility_mean_power_reduction_kw,
                    "preconditioning_electricity_change_kwh": selected.preconditioning_facility_electricity_change_kwh,
                    "recovery_electricity_change_kwh": selected.recovery_facility_electricity_change_kwh,
                    "net_full_window_electricity_reduction_kwh": selected.precondition_event_recovery_facility_net_reduction_kwh,
                    "positive_net_full_window": bool(
                        selected.precondition_event_recovery_facility_net_reduction_kwh > 0
                    ),
                    "positive_net_selected_action": net_choice.action_id
                    if net_choice is not None
                    else None,
                    "positive_net_event_power_kw": float(
                        net_choice.event_facility_mean_power_reduction_kw
                    )
                    if net_choice is not None
                    else None,
                    "positive_net_full_window_kwh": float(
                        net_choice.precondition_event_recovery_facility_net_reduction_kwh
                    )
                    if net_choice is not None
                    else None,
                }
            )
    choice_frame = pd.DataFrame(choices, columns=[
        "climate", "composition_id", "control", "normal_energy_reference_factor", "event_id",
        "baseline_status", "selection_information", "status", "selected_action",
        "baseline_facility_mean_kw", "absolute_event_facility_mean_kw", "event_power_reduction_kw",
        "preconditioning_electricity_change_kwh", "recovery_electricity_change_kwh",
        "net_full_window_electricity_reduction_kwh", "positive_net_full_window",
        "positive_net_selected_action", "positive_net_event_power_kw", "positive_net_full_window_kwh",
    ])
    choice_frame.to_csv(OUT / "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv", index=False)
    paired = []
    keys = ["climate", "composition_id", "normal_energy_reference_factor", "event_id"]
    for control, reference_control in [("zone_summary", "building_search"),
                                       ("zone_full", "building_search"), ("zone_full", "zone_summary")]:
        reference_choices = choice_frame[choice_frame.control.eq(reference_control)]
        comparison = choice_frame[choice_frame.control.eq(control)].merge(
            reference_choices, on=keys, suffixes=("_expanded", "_reference"), validate="one_to_one"
        )
        comparison["compared_library"] = control
        comparison["reference_library"] = reference_control
        joint = comparison.status_expanded.eq("selected") & comparison.status_reference.eq("selected")
        comparison["both_have_selected_action"] = joint
        comparison["paired_power_difference_kw"] = np.where(
            joint,
            comparison.event_power_reduction_kw_expanded
            - comparison.event_power_reduction_kw_reference,
            np.nan,
        )
        paired.append(comparison)
    if paired:
        pd.concat(paired, ignore_index=True).to_csv(
            OUT / "PAIRED_COMMON_BUDGET_POWER_DIFFERENCES.csv", index=False
        )
    report = {
        "status": "completed" if frame.status.eq("evaluated").all() else "partial",
        "logical_event_actions": len(frame),
        "budget_event_cells": len(choices),
        "selected_event_cells": sum(x["status"] == "selected" for x in choices),
        "interpretation": "Isolated-event,perfect-foresight finite-library bound under common annual model-service and normal-energy budgets; not a market-certified or jointly dispatched annual capacity.",
    }
    write_json(OUT / "CAPABILITY_AUDIT.json", report)
    record_status("common_budget_" + OUT.name, report["status"], report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["plan", "baseline", "pilot", "full", "score"])
    p.add_argument("--equivalent-dates", action="store_true",
                   help="Use equally representative event dates with the same normal operations")
    a = p.parse_args()
    calendar = None
    if a.equivalent_dates:
        OUT = ROOT / "results/equivalent_date_capability"
        calendar = ROOT / "results/event_date_sensitivity/ALTERNATE_EVENT_PROTOCOL.csv"
    with response_command_lock():
        if a.stage == "plan":
            plan(calendar)
        elif a.stage == "baseline":
            baseline_execute()
        elif a.stage in ["pilot", "full"]:
            action_execute(a.stage)
        else:
            score(calendar)
