"""Amendment 32: diagnose retained hourly native zone ventilation, without simulation."""
from common import *
import common
import argparse
import io
import sqlite3
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import zstandard as zstd

KEYS = ["climate", "composition_id", "control", "layout"]
CASES = KEYS + ["system", "placement"]
DELIVERED = "Zone Mechanical Ventilation Standard Density Volume Flow Rate"
REQUESTED = "Zone Target Voz Ventilation Flow Rate"
METRICS = ["assessed_zone_hours", "occupied_zero_request_hours", "delivery_ratio_min",
           "delivery_ratio_p01", "hours_below_99pct", "delivered_volume_m3", "requested_volume_m3"]


def read_input(path, hashes, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(CODE):
        raise ValueError("Ventilation inputs must remain inside this project")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    key = path.relative_to(CODE).as_posix()
    if (expected is not None and actual != expected) or (key in hashes and hashes[key] != actual):
        raise ValueError("Ventilation input SHA changed: " + str(path))
    hashes[key] = actual
    return raw


def inventory(mechanism, hashes):
    status = json.loads(read_input(mechanism / "STATUS.json", hashes))
    if status.get("status") != "completed" or status["decomposition"]["paired_settings"] != 105:
        raise ValueError("Use the completed 105-setting mechanism analysis")
    cases = pd.read_csv(io.BytesIO(read_input(mechanism / "CASE_SOURCES.csv", hashes)))
    resolved = pd.read_csv(io.BytesIO(read_input(mechanism / "TRACKING_WINDOW_LEVELS.csv", hashes)))
    resolved = resolved[resolved.window.eq("service")]
    frame = cases[CASES + ["hourly_path"]].merge(
        resolved[CASES + ["resolved_sql_path"]], on=CASES, validate="one_to_one")
    if len(frame) != 420 or len(cases) != 420 or len(resolved) != 420:
        raise ValueError("The declared 420 physical records must remain in the inventory")
    if frame[CASES].isna().any().any():
        raise ValueError("The mechanism inventory contains missing setting identifiers")
    if set(frame.system) != {"vav", "pthp"} or set(frame.placement) != {"mixed", "arranged"}:
        raise ValueError("Unexpected architecture or placement")
    if not frame.layout.eq("within_floor_relocation_25pct").all():
        raise ValueError("This diagnostic is restricted to the frozen quarter-budget settings")
    if not frame.groupby(KEYS).size().eq(4).all():
        raise ValueError("Each setting requires both placements and architectures")
    counts = frame[KEYS].drop_duplicates().groupby("control").size().to_dict()
    if counts != {"personal_center": 52, "personal_range": 53}:
        raise ValueError("The original 52/53 common qualified setting denominator changed")
    history = json.loads(read_input(mechanism / "INPUT_HASHES.json", hashes))
    read_input(mechanism / "DENOMINATORS.csv", hashes)
    return frame.sort_values(CASES), history


def open_database(path, history, hashes):
    """Hash exactly the bytes deserialized; original SQL and archives are never opened writable."""
    path = Path(path).resolve()
    if path.is_file():
        raw = read_input(path, hashes, history[str(path)])
        storage, disk_bytes = "raw_sql_in_memory", len(raw)
    else:
        record = path.parent / "RAW_SQL_LOSSLESS_ARCHIVE.json"
        info = json.loads(read_input(record, hashes, history[str(record)]))
        archive = path.with_suffix(".sql.zst")
        packed = read_input(archive, hashes, history[str(archive)])
        if hashlib.sha256(packed).hexdigest() != info["archive_sha256"]:
            raise ValueError("Lossless SQL archive SHA differs from its receipt")
        with zstd.ZstdDecompressor().stream_reader(packed) as stream:
            raw = stream.read()
        if hashlib.sha256(raw).hexdigest() != info["uncompressed_sql_sha256"]:
            raise ValueError("Uncompressed SQL SHA differs from its receipt")
        storage, disk_bytes = "verified_zst_in_memory", len(packed)
    connection = sqlite3.connect(":memory:")
    connection.deserialize(raw)
    connection.execute("PRAGMA query_only=ON")
    return connection, {"sql_storage": storage, "disk_bytes": disk_bytes,
                        "sql_bytes": len(raw), "sql_sha256": hashlib.sha256(raw).hexdigest()}


def extract_flows(connection, data):
    zones = [str(z).upper() for z in data["zone_ids"]]
    if len(zones) != len(set(zones)) or np.asarray(data["people"]).shape != (8760, len(zones)):
        raise ValueError("Incomplete or duplicated NPZ zone identities")
    if np.asarray(data["occupied"]).shape != (8760,) or data["occupied"].dtype.kind != "b":
        raise ValueError("The retained study occupancy mask must contain 8760 Boolean hours")
    if not np.isfinite(data["people"]).all() or (data["people"] < 0).any():
        raise ValueError("Non-finite or negative native zone occupancy")
    clock = pd.read_sql_query("select t.TimeIndex,t.Month,t.Day,t.Hour,t.Dst,t.Interval from Time t join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 order by t.TimeIndex", connection)
    if len(clock) != 8760 or not clock.Interval.eq(60).all():
        raise ValueError("A complete hourly annual SQL clock is required")
    for column, field in [("Month", "month"), ("Day", "day"), ("Hour", "hour"), ("Dst", "dst")]:
        if not np.array_equal(clock[column], data[field]):
            raise ValueError("SQL calendar differs from the retained NPZ: " + field)
    dictionary = pd.read_sql_query("select ReportDataDictionaryIndex,KeyValue,Name,Units from ReportDataDictionary where ReportingFrequency='Hourly' and Name in (?,?)", connection, params=[DELIVERED, REQUESTED])
    dictionary["zone"] = dictionary.KeyValue.fillna("").str.upper()
    indices = []
    for name in [DELIVERED, REQUESTED]:
        for zone in zones:
            match = dictionary[dictionary.Name.eq(name) & dictionary.zone.eq(zone)]
            if len(match) != 1 or match.iloc[0].Units != "m3/s":
                raise ValueError("Missing, duplicated or incorrect-unit native zone flow: " + name + " / " + zone)
            indices.append(int(match.iloc[0].ReportDataDictionaryIndex))
    marks = ",".join("?" for _ in indices)
    query = f"select r.TimeIndex,r.ReportDataDictionaryIndex,r.Value from ReportData r join Time t on t.TimeIndex=r.TimeIndex join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 and r.ReportDataDictionaryIndex in ({marks})"
    plan = list(connection.execute("EXPLAIN QUERY PLAN " + query, indices))
    # One bulk query per case. Existing indexes survive deserialization; no index is created.
    values = pd.read_sql_query(query, connection, params=indices)
    values = values.pivot(index="TimeIndex", columns="ReportDataDictionaryIndex", values="Value").reindex(index=clock.TimeIndex, columns=indices).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing, non-finite or negative native zone-flow records")
    calendar = hashlib.sha256(np.asarray(clock[["Month", "Day", "Hour", "Dst"]], dtype="<i8").tobytes()).hexdigest()
    return values[:, :len(zones)], values[:, len(zones):], calendar, " | ".join(row[3] for row in plan)


def zone_statistics(data, delivered, requested, calendar):
    rows = []
    for j, zone in enumerate(data["zone_ids"]):
        occupied = data["occupied"] & (data["people"][:, j] > 0)
        active = occupied & (requested[:, j] > 0)
        ratios = delivered[active, j] / requested[active, j]
        if not np.isfinite(ratios).all():
            raise ValueError("Non-finite delivered/requested zone-flow ratios")
        rows.append({"zone_id": str(zone).upper(), "assessed_zone_hours": int(active.sum()),
            "occupied_zero_request_hours": int((occupied & ~active).sum()),
            "ratio_status": "available" if len(ratios) else "no_occupied_positive_request_hours",
            "delivery_ratio_min": float(ratios.min()) if len(ratios) else np.nan,
            "delivery_ratio_p01": float(np.quantile(ratios, 0.01)) if len(ratios) else np.nan,
            "hours_below_99pct": int((ratios < 0.99).sum()),
            "delivered_volume_m3": float(delivered[active, j].sum() * 3600),
            "requested_volume_m3": float(requested[active, j].sum() * 3600),
            "assessment_mask_sha256": hashlib.sha256(np.packbits(active).tobytes()).hexdigest(),
            "calendar_sha256": calendar})
    return rows


def paired_changes(levels):
    keys = KEYS + ["system", "zone_id"]
    paired = levels[levels.placement.eq("mixed")].merge(
        levels[levels.placement.eq("arranged")], on=keys, how="outer", indicator=True,
        validate="one_to_one", suffixes=("_mixed", "_arranged"))
    same = paired._merge.eq("both") & paired.calendar_sha256_mixed.eq(paired.calendar_sha256_arranged) & paired.assessment_mask_sha256_mixed.eq(paired.assessment_mask_sha256_arranged)
    paired["pair_status"] = np.select(
        [paired._merge.eq("left_only"), paired._merge.eq("right_only"), same],
        ["unresolved_missing_arranged", "unresolved_missing_mixed", "matched_assessment_hours"],
        default="unresolved_assessment_clock_or_mask")
    for metric in METRICS:
        paired[metric + "_change"] = (paired[metric + "_arranged"] - paired[metric + "_mixed"]).where(same)
    return paired


def completed_mechanism_root(parent):
    for folder in sorted(parent.glob("run_*"), reverse=True):
        status_path = folder / "STATUS.json"
        if not status_path.is_file():
            raise ValueError("A newer mechanism attempt has no status; select an audited input explicitly")
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed":
            raise ValueError("Complete the latest mechanism attempt before ventilation diagnostics")
        if "diagnostics" not in status:
            continue  # Completed decomposition-only or locator summaries do not contain flow data.
        if status["diagnostics"].get("status") != "completed":
            raise ValueError("The latest mechanism diagnostic inventory is incomplete")
        return folder
    raise FileNotFoundError("Complete the spatial/HVAC mechanism diagnostic before ventilation analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    mechanism = (args.mechanism_root or completed_mechanism_root(ROOT / "results/spatial_hvac_mechanism")).resolve()
    out = (args.output_root or ROOT / "results/zone_ventilation_diagnostic" /
           datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")).resolve()
    if not mechanism.is_relative_to(CODE) or not out.is_relative_to(CODE):
        raise ValueError("Diagnostic inputs and outputs must remain inside this project")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the previous ventilation diagnostic attempt")
    hashes = {}
    for path in [Path(__file__).resolve(), Path(common.__file__).resolve(),
                 CODE / "data/metadata/strengthening_protocol/analysis_plan_amendment_32_zone_ventilation.yaml"]:
        read_input(path, hashes)
    cases, history = inventory(mechanism, hashes)
    frozen = dict(hashes)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "FROZEN_INPUTS.json", {"status": "frozen_before_ratios", "metadata_and_sources": frozen,
        "trajectory_and_sql_hashes": "Bound to the complete mechanism INPUT_HASHES.json recorded above",
        "planned_records": 420, "common_settings": 105, "engineering_eligibility_changes": False})
    levels, statuses = [], []
    for number, case in enumerate(cases.to_dict("records"), 1):
        status = {k: case[k] for k in CASES}
        connection = None
        started = time.perf_counter()
        try:
            path = Path(case["hourly_path"]).resolve()
            with np.load(io.BytesIO(read_input(path, hashes, history[str(path)])), allow_pickle=False) as saved:
                data = {key: saved[key] for key in ["month", "day", "hour", "dst", "occupied", "zone_ids", "people"]}
            connection, provenance = open_database(case["resolved_sql_path"], history, hashes)
            delivered, requested, calendar, plan = extract_flows(connection, data)
            levels.extend(status | row for row in zone_statistics(data, delivered, requested, calendar))
            status.update(status="completed", **provenance, query_plan=plan)
        except Exception as error:
            status.update(status="unresolved", reason=repr(error))
        finally:
            if connection is not None:
                connection.close()
        status["seconds"] = time.perf_counter() - started
        statuses.append(status)
        if number % 10 == 0:
            pd.DataFrame(statuses).to_csv(out / "CASE_STATUS.csv", index=False)
            print("ZONE_VENTILATION", number, "/420", flush=True)
    columns = CASES + ["zone_id", *METRICS, "ratio_status", "assessment_mask_sha256", "calendar_sha256"]
    levels = pd.DataFrame(levels, columns=columns)
    paired = paired_changes(levels)
    levels.to_csv(out / "ZONE_FLOW_DIAGNOSTICS.csv", index=False)
    paired.to_csv(out / "PAIRED_PLACEMENT_CHANGES.csv", index=False)
    pd.DataFrame(statuses).to_csv(out / "CASE_STATUS.csv", index=False)
    for key, expected in frozen.items():
        read_input(CODE / key, hashes, expected)
    write_json(out / "INPUT_SHA256.json", hashes)
    unresolved = sum(row["status"] == "unresolved" for row in statuses)
    unmatched = int(paired.pair_status.ne("matched_assessment_hours").sum())
    write_json(out / "AUDIT.json", {"status": "completed" if not unresolved and not unmatched else "partial",
        "mechanism_source": mechanism.relative_to(CODE).as_posix(),
        "planned_records": 420, "completed_records": 420 - unresolved, "unresolved_records": unresolved,
        "common_settings": 105, "planned_within_system_setting_pairs": 210,
        "zone_rows": len(levels), "paired_zone_rows": len(paired), "unresolved_assessment_zone_pairs": unmatched,
        "zones_without_assessed_hours": int(levels.assessed_zone_hours.eq(0).sum()),
        "primary_variables": {"delivered": DELIVERED, "requested": REQUESTED},
        "assessment": "Study occupied mask AND positive native zone occupancy AND positive native requested flow",
        "sign": "Paired changes are arranged minus mixed; no engineering eligibility is changed",
        "scope": "Native hourly mechanical outdoor-air allocation; not subhourly compliance, IAQ/health validation or matched thermal delivery",
        "thermal_gap": "Sample dictionaries did not retain zone or coil delivered heating/cooling; no thermal-load inference or new physics",
        "environment": {"python": sys.version, "sqlite": sqlite3.sqlite_version, "numpy": np.__version__, "pandas": pd.__version__, "zstandard": zstd.__version__},
        "output_sha256": {p.name: digest(p) for p in [*out.glob("*.csv"), out / "FROZEN_INPUTS.json", out / "INPUT_SHA256.json"]}})


if __name__ == "__main__":
    main()
