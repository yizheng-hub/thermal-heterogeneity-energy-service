"""Compare central VAV intake with its own native VRP request, not zone Voz."""
from common import *
import argparse
import io
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import zone_ventilation_diagnostic as zone

ACTUAL = "Air System Outdoor Air Mass Flow Rate"
REQUESTED = "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate"
FIELDS = ["airloop", "assessed_hours", "occupied_zero_request_hours", "ratio_status",
          "actual_to_vrp_request_ratio_min", "actual_to_vrp_request_ratio_p01",
          "hours_below99pct_controller_request", "actual_intake_mass_kg", "controller_requested_mass_kg",
          "maximum_absolute_mass_flow_difference_kg_s"]


def loop_statistics(connection, data):
    clock = pd.read_sql_query("select t.TimeIndex,t.Month,t.Day,t.Hour,t.Dst,t.Interval from Time t join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 order by t.TimeIndex", connection)
    if len(clock) != 8760 or not clock.Interval.eq(60).all():
        raise ValueError("Native VRP check requires a complete hourly annual clock")
    for name, field in [("Month", "month"), ("Day", "day"), ("Hour", "hour"), ("Dst", "dst")]:
        if not np.array_equal(clock[name], data[field]):
            raise ValueError("SQL and retained occupancy calendars differ")
    if data["occupied"].shape != (8760,) or data["occupied"].dtype.kind != "b":
        raise ValueError("The original study occupancy mask is required")
    dictionary = pd.read_sql_query("select ReportDataDictionaryIndex,KeyValue,Name,Units from ReportDataDictionary where ReportingFrequency='Hourly' and Name in (?,?)", connection, params=[ACTUAL, REQUESTED])
    dictionary["key"] = dictionary.KeyValue.fillna("").str.upper()
    keys = sorted(dictionary[dictionary.Name.eq(ACTUAL)].key.unique())
    if len(keys) != 3 or set(keys) != set(dictionary[dictionary.Name.eq(REQUESTED)].key):
        raise ValueError("Expected matched actual/request variables for all three VAV airloops")
    indices = []
    for name in [ACTUAL, REQUESTED]:
        for key in keys:
            selected = dictionary[dictionary.Name.eq(name) & dictionary.key.eq(key)]
            if len(selected) != 1 or selected.iloc[0].Units != "kg/s":
                raise ValueError("Missing, duplicated or incorrect-unit VRP flow")
            indices.append(int(selected.iloc[0].ReportDataDictionaryIndex))
    placeholders = ",".join("?" for _ in indices)
    values = pd.read_sql_query(f"select r.TimeIndex,r.ReportDataDictionaryIndex,r.Value from ReportData r join Time t on t.TimeIndex=r.TimeIndex join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 and r.ReportDataDictionaryIndex in ({placeholders})", connection, params=indices)
    values = values.pivot(index="TimeIndex", columns="ReportDataDictionaryIndex", values="Value").reindex(index=clock.TimeIndex, columns=indices).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Incomplete or invalid native VRP flows")
    rows = []
    for index, key in enumerate(keys):
        actual, requested = values[:, index], values[:, index + 3]
        assessed = data["occupied"] & (requested > 0)
        ratio = actual[assessed] / requested[assessed]
        rows.append({"airloop": key, "assessed_hours": int(assessed.sum()),
            "occupied_zero_request_hours": int((data["occupied"] & ~assessed).sum()),
            "ratio_status": "available" if len(ratio) else "no_positive_occupied_controller_request",
            "actual_to_vrp_request_ratio_min": float(ratio.min()) if len(ratio) else np.nan,
            "actual_to_vrp_request_ratio_p01": float(np.quantile(ratio, .01)) if len(ratio) else np.nan,
            "hours_below99pct_controller_request": int((ratio < .99).sum()),
            "actual_intake_mass_kg": float(actual[assessed].sum() * 3600),
            "controller_requested_mass_kg": float(requested[assessed].sum() * 3600),
            "maximum_absolute_mass_flow_difference_kg_s": float(abs(actual[assessed] - requested[assessed]).max()) if len(ratio) else np.nan})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    mechanism = (args.mechanism_root or zone.completed_mechanism_root(ROOT / "results/spatial_hvac_mechanism")).resolve()
    out = (args.output_root or ROOT / "results/vrp_request_diagnostic" /
           datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    if not mechanism.is_relative_to(CODE) or not out.is_relative_to(CODE):
        raise ValueError("Native VRP inputs and outputs must remain inside this project")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the prior native VRP check before retrying")
    hashes = {}
    for path in [Path(__file__).resolve(), Path(zone.__file__).resolve(),
                 CODE / "src/analysis/strengthening/common.py",
                 CODE / "data/metadata/strengthening_protocol/ventilation_request.yaml"]:
        zone.read_input(path, hashes)
    cases, history = zone.inventory(mechanism, hashes)
    frozen = dict(hashes)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "FROZEN_INPUTS.json", {"metadata_and_sources": frozen,
        "scope": "Native central VRP request fulfillment; no change to qualification"})
    rows, statuses = [], []
    for number, case in enumerate(cases.to_dict("records"), 1):
        identity = {name: case[name] for name in zone.CASES}
        if case["system"] == "pthp":
            statuses.append({**identity, "status": "architecture_not_applicable"})
            continue
        connection = None
        try:
            path = Path(case["hourly_path"]).resolve()
            with np.load(io.BytesIO(zone.read_input(path, hashes, history[str(path)])), allow_pickle=False) as saved:
                data = {name: saved[name] for name in ["month", "day", "hour", "dst", "occupied"]}
            connection, provenance = zone.open_database(case["resolved_sql_path"], history, hashes)
            rows.extend({**identity, **row} for row in loop_statistics(connection, data))
            statuses.append({**identity, "status": "completed", **provenance})
        except Exception as error:
            statuses.append({**identity, "status": "unresolved", "reason": repr(error)})
        finally:
            if connection is not None:
                connection.close()
        if number % 20 == 0:
            pd.DataFrame(statuses).to_csv(out / "CASE_STATUS.csv", index=False)
            print("VRP_REQUEST_CHECK", number, "/420 inventory cases", flush=True)
    pd.DataFrame(statuses).to_csv(out / "CASE_STATUS.csv", index=False)
    pd.DataFrame(rows, columns=zone.CASES + FIELDS).to_csv(out / "AIRLOOP_REQUEST_FULFILLMENT.csv", index=False)
    for name, expected in frozen.items():
        zone.read_input(CODE / name, hashes, expected)
    write_json(out / "INPUT_SHA256.json", hashes)
    counts = pd.DataFrame(statuses).status.value_counts().to_dict()
    write_json(out / "AUDIT.json", {"status": "completed" if counts.get("completed", 0) == 210 and counts.get("architecture_not_applicable", 0) == 210 else "partial",
        "mechanism_source": mechanism.relative_to(CODE).as_posix(), "case_status_counts": counts,
        "planned_inventory_cases": 420, "applicable_VAV_cases": 210, "airloop_rows": len(rows),
        "actual_variable": ACTUAL, "request_variable": REQUESTED, "units": "Hourly average kg/s",
        "scope": "Actual native VRP system request, not a terminal direct-OA constraint or subhourly/IAQ compliance certificate",
        "additional_EnergyPlus_runs": 0, "engineering_eligibility_changed": False,
        "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()}})


if __name__ == "__main__":
    main()
