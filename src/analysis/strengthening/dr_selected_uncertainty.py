"""Rescore frozen selected response actions with paired raw-identity demand shifts."""

from common import *
from paired_uncertainty import keyed_normal, shared_normal, empirical_temporal_shifts, DRAWS
from profile_identity import person_keys
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles, load_hourly
import multi_event_dr_extension as dr
import numpy as np
import pandas as pd
import inspect


CONSUMED_INPUTS = {}
EXECUTED_SOURCES = {}


def tracked(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root) for root in (CODE, ROOT, MATRIX)):
        raise ValueError("Response sensitivity input resolves outside the contained project")
    if path not in CONSUMED_INPUTS:
        CONSUMED_INPUTS[path] = digest(path)
    return path


def read_csv(path):
    return pd.read_csv(tracked(path))


def hourly(path, **kwargs):
    return load_hourly(tracked(path), **kwargs)


def window_score(curve, centers, tail):
    individual = curve.individual(centers)
    weights = curve.person_hours[tail]
    return individual @ curve.weights, individual[..., tail] @ (weights / weights.sum())


def asset_key(path):
    return (path.relative_to(CODE).as_posix() if path.is_relative_to(CODE)
            else "analysis/" + path.relative_to(ROOT).as_posix())


def completed_cache(out):
    if not out.exists() or not any(out.iterdir()):
        return False
    receipt = out / "AUDIT.json"
    if not receipt.is_file():
        raise FileExistsError("Preserve the incomplete response sensitivity attempt before retrying")
    audit = json.loads(receipt.read_text())
    expected_outputs = {"FROZEN_ACTION_SERVICE_VIOLATIONS.csv", "SERVICE_WINDOW_VIOLATIONS.csv"}
    if (audit.get("status") != "completed" or
            set(audit.get("output_sha256", {})) != expected_outputs or
            not audit.get("input_sha256") or not audit.get("executed_source_sha256")):
        raise ValueError("Response sensitivity cache has no complete provenance receipt")
    if audit["executed_source_sha256"] != {asset_key(p): sha for p, sha in EXECUTED_SOURCES.items()}:
        raise ValueError("Response sensitivity executable sources changed; preserve the prior attempt")
    for key in ("input_sha256", "executed_source_sha256"):
        for name, expected in audit[key].items():
            path = (ROOT / name.removeprefix("analysis/") if name.startswith("analysis/")
                    else CODE / name).resolve()
            if (not any(path.is_relative_to(root) for root in (CODE, ROOT, MATRIX)) or
                    not path.is_file() or digest(path) != expected):
                raise ValueError("Response sensitivity cache input or source changed; preserve the prior attempt")
    if any(not (out / name).is_file() or digest(out / name) != expected
           for name, expected in audit["output_sha256"].items()):
        raise ValueError("Response sensitivity cache output changed; preserve the prior attempt")
    print("Selected response sensitivity: verified completed cache", flush=True)
    return True


def write_results(rows, window_rows, selected_count):
    for path, expected in {**CONSUMED_INPUTS, **EXECUTED_SOURCES}.items():
        if digest(path) != expected:
            raise ValueError("Response sensitivity input or source changed during execution")
    out = ROOT / "results/dr_capability_uncertainty_identity"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the prior response sensitivity attempt before writing new results")
    out.mkdir(parents=True, exist_ok=True)
    columns = ["climate", "composition", "control", "event_id", "action_id", "scenario"]
    pd.DataFrame(rows, columns=columns + ["conditional_any_service_violation_frequency",
        "maximum_exceedance_median_c", "maximum_exceedance_p95_c",
        "event_power_kw_fixed_operation", "net_electricity_kwh_fixed_operation"]).to_csv(
        out / "FROZEN_ACTION_SERVICE_VIOLATIONS.csv", index=False)
    pd.DataFrame(window_rows, columns=columns + ["window", "conditional_violation_frequency",
        "exceedance_median_c", "exceedance_p95_c"]).to_csv(out / "SERVICE_WINDOW_VIOLATIONS.csv", index=False)
    report = {
        "status": "completed", "selected_operations": selected_count, "rows": len(rows),
        "draws": DRAWS, "operation_draw_evaluations": len(rows) * DRAWS,
        "shared_error_groups": {"sourceSD": "archive", "studyGroupSD": "study/contributor",
            "combination": "Alternative shared terms at the same SD; not added to each other"},
        "scope": "A: fixed-operation rescoring. Normal operation and response action remain frozen; no adaptive controller or uncertain-demand action reselection is claimed.",
        "reference": "Same paired demand draw for each candidate, normal baseline and25C reference; original annual fixed tail retained.",
        "probability": "Scenario-conditional violation frequencies; not calibrated probabilities of real discomfort.",
        "input_sha256": {asset_key(p): sha for p, sha in CONSUMED_INPUTS.items()},
        "executed_source_sha256": {asset_key(p): sha for p, sha in EXECUTED_SOURCES.items()},
        "output_sha256": {p.name: digest(p) for p in out.glob("*.csv")},
        "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(out / "AUDIT.json", report)
    record_status("dr_selected_paired_uncertainty", "completed", report)


def main():
    CONSUMED_INPUTS.clear()
    EXECUTED_SOURCES.clear()
    for function in (main, load_hourly, normalise_profiles, ServiceDistribution,
                     keyed_normal, shared_normal, person_keys, empirical_temporal_shifts, dr._window_masks, digest):
        path = Path(inspect.getfile(function)).resolve()
        EXECUTED_SOURCES[path] = digest(path)
    if completed_cache(ROOT / "results/dr_capability_uncertainty_identity"):
        return
    folder = ROOT / "results/dr_capability"
    choices = read_csv(folder / "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")
    selected = choices[choices.status.eq("selected")].drop_duplicates(
        ["climate", "composition_id", "control", "event_id", "selected_action"]
    )
    if selected.empty:
        write_results([], [], 0)
        return
    planned = read_csv(folder / "SINGLE_EVENT_ACTION_PLAN.csv")
    actions = read_csv(folder / "ACTION_FULL_COMPLETED.csv").set_index("run_id")
    normals = read_csv(folder / "NORMAL_OPERATION_REPLAY_COMPLETED.csv").set_index("normal_id")
    profiles = normalise_profiles(read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    members = read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    probes = read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    probes = probes[probes.target_c.eq(25)].set_index("climate")
    rows = []
    window_rows = []
    for relative in ("audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz",
                     "results/demand_validation/temporal_centers/DEVELOPMENT_EMPIRICAL_TEMPORAL_SHIFT_POOL.csv",
                     "results/demand_validation/single_record/timed/CONTINUOUS_PROXY_ERRORS.csv"):
        path = ROOT / relative
        if path.is_file():
            tracked(path)
    for (climate, composition), g in profiles.groupby(["climate", "composition"], sort=True):
        g = g.sort_values("profile_id").reset_index(drop=True)
        comp = int(g.composition_id.iloc[0])
        nominal = g.centre_c.to_numpy()
        ids = set(
            members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id
        )
        tail = np.flatnonzero(g.profile_id.isin(ids))
        assert len(tail) == 27
        source = "ASHRAE" if "Multinational" in composition else "CTCD"
        individual = np.column_stack(
            [keyed_normal("person|" + key) for key in person_keys(g.profile_id)]
        )
        shared = keyed_normal("source|" + source)[:, None]
        scenarios = [
            (f"sourceSD{s:g}_personSD{p:g}", nominal[None, :] + s * shared + p * individual)
            for s in [0, 0.25, 0.5]
            for p in [0, 0.25, 0.5]
        ]
        grouped = shared_normal(g.profile_id, source, study_shared=True)
        scenarios += [(f"studyGroupSD{s:g}_personSD{p:g}", nominal[None, :] + s * grouped + p * individual)
            for s in [0, 0.25, 0.5] for p in [0, 0.25, 0.5]]
        if source == "ASHRAE":
            shift = empirical_temporal_shifts(g.profile_id)
            scenarios += [
                (f"observed_temporal_proxy_shift_fraction{f:g}", nominal[None, :] + f * shift)
                for f in [0.5, 1.0]
            ]
            single_shift = empirical_temporal_shifts(g.profile_id, single_record=True)
            scenarios += [
                (f"single_record_temporal_proxy_shift_fraction{fraction:g}", nominal[None, :] + fraction * single_shift)
                for fraction in [0.5, 1.0]
            ]
        reference = hourly(Path(probes.loc[climate, "hourly_path"]))
        reference_annual = ServiceDistribution(reference, g)
        for row in selected[
            selected.climate.eq(climate) & selected.composition_id.eq(comp)
        ].itertuples():
            plan = planned[
                planned.climate.eq(climate)
                & planned.composition_id.eq(comp)
                & planned.control.eq(row.control)
                & planned.event_id.eq(row.event_id)
                & planned.action_id.eq(row.selected_action)
            ].iloc[0]
            baseline = hourly(Path(normals.loc[plan.normal_id, "hourly_path"]), require_facility=False)
            action = hourly(Path(actions.loc[plan.run_id, "hourly_path"]), require_facility=False)
            event = json.loads(plan.event_json)
            start = pd.Timestamp(event["start_timestamp"])
            masks = dr._window_masks(
                reference,
                start,
                int(event["duration_hours"]),
                int(event["recovery_hours"]),
                int(event.get("precondition_hours", 0)),
            )
            curves = {"annual": (ServiceDistribution(baseline, g), reference_annual)}
            for name, mask in zip(["precondition", "event", "recovery"], masks):
                if not (reference["people"] * (reference["occupied"] & mask)[:, None]).sum():
                    continue
                ref = ServiceDistribution(reference, g, mask)
                curves["baseline_" + name] = (ServiceDistribution(baseline, g, mask), ref)
                curves["action_" + name] = (ServiceDistribution(action, g, mask), ref)
            for label, centers in scenarios:
                all_pass = np.ones(DRAWS, dtype=bool)
                maximum = np.zeros(DRAWS)
                for name, (candidate, ref) in curves.items():
                    score = (
                        (lambda curve: curve.score(centers, tail))
                        if name == "annual"
                        else (lambda curve: window_score(curve, centers, tail))
                    )
                    population, least = score(candidate)
                    rp, rt = score(ref)
                    excess = np.maximum(np.maximum(population - rp, least - rt), 0)
                    passed = excess <= 1e-9
                    all_pass &= passed
                    maximum = np.maximum(maximum, excess)
                    window_rows.append(
                        {
                            "climate": climate,
                            "composition": composition,
                            "control": row.control,
                            "event_id": row.event_id,
                            "action_id": row.selected_action,
                            "scenario": label,
                            "window": name,
                            "conditional_violation_frequency": float((~passed).mean()),
                            "exceedance_median_c": float(np.median(excess)),
                            "exceedance_p95_c": float(np.quantile(excess, 0.95)),
                        }
                    )
                if label == "sourceSD0_personSD0" and not all_pass.all():
                    raise ValueError(
                        "Nominal frozen selected action does not pass reproduced service checks"
                    )
                rows.append(
                    {
                        "climate": climate,
                        "composition": composition,
                        "control": row.control,
                        "event_id": row.event_id,
                        "action_id": row.selected_action,
                        "scenario": label,
                        "conditional_any_service_violation_frequency": float((~all_pass).mean()),
                        "maximum_exceedance_median_c": float(np.median(maximum)),
                        "maximum_exceedance_p95_c": float(np.quantile(maximum, 0.95)),
                        "event_power_kw_fixed_operation": row.event_power_reduction_kw,
                        "net_electricity_kwh_fixed_operation": row.net_full_window_electricity_reduction_kwh,
                    }
                )
        print("DR_PAIRED_SERVICE", climate, composition, flush=True)
    write_results(rows, window_rows, len(selected))


if __name__ == "__main__":
    main()
