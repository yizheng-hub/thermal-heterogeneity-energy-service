"""Describe residuals after the new isolated responses, without changing selection."""

from common import *
import numpy as np
import pandas as pd
from audit_event_terminal_recovery import _timestamps


def residuals(action, normal, event):
    clock = _timestamps(action)
    if not clock.equals(_timestamps(normal)) or not np.array_equal(action["zone_ids"], normal["zone_ids"]):
        raise ValueError("Response and normal-operation clocks or zone axes differ")
    if len(clock) != 8760 or action["temperatures"].shape != normal["temperatures"].shape:
        raise ValueError("Recovery diagnostic requires matched annual zone trajectories")
    for name in ("occupied", "dst"):
        if not np.array_equal(action[name], normal[name]):
            raise ValueError("Response and normal-operation occupancy or daylight-saving status differs")
    if (action["people"].shape != normal["people"].shape or
            not np.allclose(action["people"], normal["people"], rtol=0, atol=1e-10)):
        raise ValueError("Response and normal-operation physical people loads differ")
    end = pd.Timestamp(event["start_timestamp"]) + pd.Timedelta(
        hours=int(event["duration_hours"]) + int(event["recovery_hours"]))
    rows = []
    for offset in range(7):
        stamp = end + pd.Timedelta(hours=offset - 1)
        index = clock.get_indexer([stamp])[0]
        if index < 0:
            if offset == 0:
                raise ValueError("Final recovery hour is absent")
            continue
        difference = np.abs(action["temperatures"][index] - normal["temperatures"][index])
        row = {"audit_hour_index": offset, "audit_timestamp": str(stamp),
               "operative_temperature_max_abs_difference_c": float(difference.max()),
               "operative_temperature_zone_mean_abs_difference_c": float(difference.mean())}
        for array, label in (("facility_electric_kwh", "facility_electricity"),
                             ("hvac_electric_kwh", "hvac_electricity"),
                             ("gas_hhv_kwh", "space_heating_gas_hhv")):
            if action[array].shape != (8760,) or normal[array].shape != (8760,):
                raise ValueError("Recovery energy arrays do not match the hourly clock")
            row[label + "_interval_mean_kw_difference"] = float(action[array][index] - normal[array][index])
        if not all(np.isfinite(value) for name, value in row.items() if name not in {"audit_timestamp"}):
            raise ValueError("Non-finite recovery residual")
        rows.append(row)
    return rows


def main():
    folder = ROOT / "results/dr_capability"
    out = ROOT / "results/dr_terminal_recovery"
    receipt = out / "AUDIT.json"
    source_paths = [Path(__file__).resolve(), GLOBAL / "audit_event_terminal_recovery.py",
                    CODE / "src/analysis/strengthening/common.py"]
    sources = {p.relative_to(CODE).as_posix(): digest(p) for p in source_paths}
    if receipt.is_file():
        report = json.loads(receipt.read_text())
        if report.get("status") != "completed":
            raise ValueError("Incomplete recovery diagnostic receipt")
        if report["source_sha256"] != sources:
            raise ValueError("Recovery executable sources changed; preserve the prior attempt")
        for name, sha in {**report["input_sha256"], **report["source_sha256"], **report["output_sha256"]}.items():
            if digest(CODE / name) != sha:
                raise ValueError("Recovery diagnostic cache changed; preserve the prior attempt")
        print("New-response recovery: verified completed cache", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the incomplete recovery diagnostic before retrying")
    inputs = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(CODE):
            raise ValueError("Recovery diagnostic input is outside this project")
        name = path.relative_to(CODE).as_posix()
        if name not in inputs:
            inputs[name] = digest(path)
        if expected is not None and inputs[name] != expected:
            raise ValueError("Recovery trajectory differs from its completed manifest")
        return path

    plan = pd.read_csv(track(folder / "SINGLE_EVENT_ACTION_PLAN.csv"))
    actions = pd.read_csv(track(folder / "ACTION_FULL_COMPLETED.csv")).set_index("run_id")
    normals = pd.read_csv(track(folder / "NORMAL_OPERATION_REPLAY_COMPLETED.csv")).set_index("normal_id")
    if not actions.index.is_unique or not normals.index.is_unique:
        raise ValueError("Completed response manifests duplicate physical identifiers")
    if not set(plan.run_id).issubset(actions.index) or not set(plan.normal_id).issubset(normals.index):
        raise ValueError("Response plan is missing completed-operation records")
    protocol = CODE / "data/metadata/strengthening_protocol/response_recovery.yaml"
    track(protocol)
    aliases, rows, followup = [], [], []
    normal_cache = {}
    keys = ["normal_id", "run_id", "event_id"]
    for identifiers, group in plan.groupby(keys, sort=True):
        normal_id, run_id, event_id = identifiers
        if group.event_json.nunique() != 1:
            raise ValueError("Physical recovery aliases have different event windows")
        done, baseline = actions.loc[run_id], normals.loc[normal_id]
        eligible = bool(done.completed and done.final_engineering_qualified and
                        baseline.completed and baseline.normal_replay_matches_original)
        aliases.extend([{**item, "recovery_diagnostic_status": "evaluated" if eligible else "engineering_unavailable"}
                        for item in group[[*keys, "climate", "composition_id", "control", "action_id"]].to_dict("records")])
        if not eligible:
            continue
        action_path = track(done.hourly_path, done.hourly_sha256)
        normal_path = track(baseline.hourly_path, baseline.hourly_sha256)
        for item, idf in ((done, group.input_idf_path.iloc[0]), (baseline, baseline.input_idf_path)):
            qa = json.loads(item.qa)
            track(idf, qa["input_idf_sha256"])
            track(RUNTIME, qa["runtime_sha256"])
            track(RUNTIME.with_name("energyplusapi.dll"), qa["runtime_api_sha256"])
            track(group.weather_path.iloc[0], qa["weather_sha256"])
        if normal_id not in normal_cache:
            normal_cache.clear()
            with np.load(normal_path, allow_pickle=False) as saved:
                normal_cache[normal_id] = dict(saved)
        with np.load(action_path, allow_pickle=False) as saved:
            values = residuals(dict(saved), normal_cache[normal_id], json.loads(group.event_json.iloc[0]))
        identity = dict(zip(keys, identifiers))
        rows.extend([{**identity, **value} for value in values])
        later = [value for value in values if value["audit_hour_index"] > 0]
        summary = {**identity, "available_post_recovery_hours": len(later), "requested_post_recovery_hours": 6}
        for name in ("facility_electricity", "hvac_electricity", "space_heating_gas_hhv"):
            values_kw = [value[name + "_interval_mean_kw_difference"] for value in later]
            summary[name + "_post_recovery_change_kwh"] = float(sum(values_kw)) if values_kw else np.nan
            summary[name + "_post_recovery_max_abs_kw_difference"] = max(map(abs, values_kw)) if values_kw else np.nan
        followup.append(summary)
    for name, sha in {**inputs, **sources}.items():
        if digest(CODE / name) != sha:
            raise ValueError("Recovery diagnostic input changed during execution")
    out.mkdir(parents=True, exist_ok=True)
    tables = {"PHYSICAL_PAIR_HOURLY_RESIDUALS.csv": rows, "LOGICAL_EVALUATION_MAPPING.csv": aliases,
              "PHYSICAL_PAIR_POST_RECOVERY_SUMMARY.csv": followup}
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(out / name, index=False)
    write_json(receipt, {"status": "completed", "logical_evaluations": len(aliases),
        "evaluated_physical_pairs": len(followup), "hourly_residual_rows": len(rows),
        "missing_six_hour_followup_pairs": sum(row["available_post_recovery_hours"] < 6 for row in followup),
        "threshold_applied": False, "input_sha256": inputs, "source_sha256": sources,
        "output_sha256": {p.relative_to(CODE).as_posix(): digest(p) for p in out.glob("*.csv")},
        "temperature_mean": "Unweighted across modeled occupied-zone identifiers; hourly averages",
        "gas_boundary": "Hourly space-heating gas HHV meter, separate from facility electricity",
        "scope": "Existing terminal diagnostic applied to new operations; no change to 48-hour accounting or action selection"})


if __name__ == "__main__":
    main()
