"""Account for fixed-order placement effects using retained physical trajectories.

This analysis never launches a simulator or restores a database to disk. Each
invocation creates a separate result directory, preserving previous analyses.
"""
from common import *
import common
import argparse
from datetime import datetime, timezone
import inspect
import sqlite3
import numpy as np
import pandas as pd
import zstandard as zstd
from analyse_global_results import normalise_profiles
from service_distribution import ServiceDistribution

OUT = ROOT / "results/spatial_hvac_mechanism"
KEYS = ["climate", "composition_id", "control", "layout"]
ENDPOINTS = ["population", "fixed_tail"]
ENERGY = ["heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh",
          "hvac_electric_kwh", "facility_electric_kwh", "gas_hhv_kwh"]
VARIABLES = ["Zone Heating Setpoint Not Met While Occupied Time",
             "Zone Cooling Setpoint Not Met While Occupied Time",
             "Zone Mean Air Temperature", "Zone Air Temperature",
             "Zone Thermostat Heating Setpoint Temperature",
             "Zone Thermostat Cooling Setpoint Temperature"]


class Inputs:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(CODE):
            raise ValueError("Mechanism inputs must remain inside this project")
        value = digest(path)
        if str(path) in self.hashes and self.hashes[str(path)] != value:
            raise ValueError("Input changed during analysis: " + str(path))
        if expected is not None and pd.notna(expected) and value != expected:
            raise ValueError("Input SHA differs from retained provenance: " + str(path))
        self.hashes[str(path)] = value
        return path

    def csv(self, path):
        return pd.read_csv(self.check(path))

    def json(self, path):
        return json.loads(self.check(path).read_text())

    def hourly(self, path, expected=None):
        with np.load(self.check(path, expected), allow_pickle=False) as saved:
            return dict(saved)


def summarise(frame, identifiers, metrics):
    rows = []
    for identity, group in frame.groupby(identifiers, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        row = dict(zip(identifiers, identity)) | {"paired_settings": len(group)}
        for column in metrics:
            values = group[column].dropna()
            row.update({column + "_n": len(values), column + "_mean": values.mean(),
                        column + "_median": values.median(),
                        column + "_positive_n": int((values > 1e-10).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def score(data, group, tail_ids):
    tail = np.flatnonzero(group.profile_id.isin(tail_ids))
    if len(group) != 267 or len(tail) != 27:
        raise ValueError("Incomplete profile or original fixed-tail identities")
    return np.asarray(ServiceDistribution(data, group).score(group.centre_c.to_numpy(), tail))


def decompose(inputs, out):
    source = ROOT / "results/placement_hvac_interaction"
    alternative = inputs.csv(source / "independent_pthp_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv")
    original = inputs.csv(ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv")
    original = original.rename(columns={"layout_arranged": "layout"})
    original = original[original.layout.eq("within_floor_relocation_25pct")]
    alternative = alternative[alternative.layout.eq("within_floor_relocation_25pct")]
    if original.duplicated(KEYS).any() or alternative.duplicated(KEYS).any():
        raise ValueError("Duplicated logical settings")
    paired = original.merge(alternative, on=KEYS, suffixes=("_vav", "_pthp"),
                            validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Architecture tables have different planned settings")
    passed = np.ones(len(paired), dtype=bool)
    for system in ["vav", "pthp"]:
        for layout in ["arranged", "mixed"]:
            passed &= paired[f"engineering_pass_{layout}_{system}"].eq(True).to_numpy()
    unavailable = paired.loc[~passed, KEYS].copy()
    unavailable["reason"] = "At least one paired operation did not pass its existing engineering gate"
    unavailable.to_csv(out / "UNAVAILABLE_SETTINGS.csv", index=False)
    denominator = paired.groupby("control").size().rename("planned_settings").reset_index()
    denominator["qualified_settings"] = denominator.control.map(paired.loc[passed].groupby("control").size()).fillna(0).astype(int)
    denominator["unavailable_settings"] = denominator.planned_settings - denominator.qualified_settings
    denominator.to_csv(out / "DENOMINATORS.csv", index=False)
    profiles = normalise_profiles(inputs.csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv"))
    mixed_profiles = normalise_profiles(inputs.csv(MATRIX / "GLOBAL_PROFILES.csv"))
    members = inputs.csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    logical = []
    for phase in ["pilot", "confirmation"]:
        logical += inputs.json(source / f"independent_pthp_{phase}_LOGICAL_RESULTS.json")
    logical = {tuple(row[k] for k in KEYS): row for row in logical}
    mixed = []
    for phase in ["PILOT", "CONFIRMATION"]:
        mixed += inputs.json(ROOT / f"results/hvac_confirmation/{phase}_OPERATIONS.json")
    mixed = {(r["climate"], r["composition_id"], r["control"]): r for r in mixed
             if r.get("system") == "independent_pthp" and r.get("composition_id") is not None}
    rows, cases = [], []
    for _, row in paired.loc[passed].iterrows():
        identity = {k: row[k] for k in KEYS}
        arranged_group = profiles[profiles.climate.eq(row.climate) & profiles.composition_id.eq(row.composition_id) & profiles.layout.eq(row.layout)].sort_values("profile_id").reset_index(drop=True)
        mixed_group = mixed_profiles[mixed_profiles.climate.eq(row.climate) & mixed_profiles.composition_id.eq(row.composition_id)].sort_values("profile_id").reset_index(drop=True)
        if not arranged_group.profile_id.equals(mixed_group.profile_id):
            raise ValueError("Profile identities changed")
        for column in ["centre_c", "lower_c", "upper_c"]:
            if not np.allclose(arranged_group[column], mixed_group[column], atol=1e-12, rtol=0):
                raise ValueError("Profile preferences changed")
        if not arranged_group.zone_id.value_counts().sort_index().equals(mixed_group.zone_id.value_counts().sort_index()):
            raise ValueError("Zone profile capacities changed")
        tail_ids = set(members[members.climate.eq(row.climate) & members.composition.eq(arranged_group.composition.iloc[0])].profile_id)
        cross_people = None
        for system in ["vav", "pthp"]:
            data, paths = {}, {}
            for layout in ["mixed", "arranged"]:
                path = Path(row[f"hourly_path_{layout}_{system}"])
                checksum = row.get(f"hourly_sha256_{layout}_{system}", row.get(f"hourly_sha256_{layout}"))
                if system == "pthp":
                    record = mixed[(row.climate, row.composition_id, row.control)] if layout == "mixed" else logical[tuple(row[k] for k in KEYS)]
                    checksum = record["hourly_sha256"]
                    sql = record["sql"]
                else:
                    folder = Path(row.qa_path).parent if layout == "arranged" else path.parents[1] / "runs" / row.run_id_mixed
                    sql = str(folder / "eplusout.sql")
                data[layout] = inputs.hourly(path, checksum)
                paths[layout] = path
                cases.append(identity | {"system": system, "placement": layout,
                                         "hourly_path": str(path), "sql_path": sql})
            for field in ["occupied", "zone_ids", "month", "day", "hour", "dst"]:
                if not np.array_equal(data["mixed"][field], data["arranged"][field]):
                    raise ValueError("Paired calendar or zone identity changed: " + field)
            people_difference = float(abs(data["mixed"]["people"] - data["arranged"]["people"]).max())
            if people_difference > 1e-9:
                raise ValueError("Native occupancy changed")
            if cross_people is not None and not np.allclose(cross_people, data["mixed"]["people"], atol=1e-9, rtol=0):
                raise ValueError("Native occupancy differs across architectures")
            cross_people = data["mixed"]["people"]
            initial = score(data["mixed"], mixed_group, tail_ids)
            regrouped = score(data["mixed"], arranged_group, tail_ids)
            final = score(data["arranged"], arranged_group, tail_ids)
            result = identity | {"system": system, "native_people_max_difference": people_difference}
            for i, endpoint in enumerate(ENDPOINTS):
                values = {"initial": initial[i], "regrouped_original_temperature": regrouped[i], "final": final[i],
                          "assignment_step": initial[i] - regrouped[i], "trajectory_step": regrouped[i] - final[i],
                          "total_step": initial[i] - final[i]}
                values["closure_error"] = values["assignment_step"] + values["trajectory_step"] - values["total_step"]
                expected = row[f"{'population' if i == 0 else 'fixed_tail'}_gain_c_{system}"]
                values["source_gain_disagreement"] = values["total_step"] - expected
                if abs(values["closure_error"]) > 1e-10 or abs(values["source_gain_disagreement"]) > 1e-9:
                    raise ValueError("Decomposition did not reproduce the retained comparison")
                result.update({endpoint + "_" + k + "_c": float(v) for k, v in values.items()})
            rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "DECOMPOSITION.csv", index=False)
    metrics = [c for c in frame if c.endswith("_c")]
    summary = summarise(frame, ["system", "control"], metrics).merge(denominator, on="control", validate="many_to_one")
    summary.to_csv(out / "DECOMPOSITION_SUMMARY.csv", index=False)
    pd.DataFrame(cases).to_csv(out / "CASE_SOURCES.csv", index=False)
    return cases, {"status": "completed", "paired_settings": int(passed.sum()), "system_contrasts": len(frame),
                   "planned_settings": len(paired), "unavailable_settings": len(unavailable),
                   "max_closure_error_c": float(frame.filter(like="closure_error").abs().max().max()),
                   "max_source_gain_disagreement_c": float(frame.filter(like="source_gain_disagreement").abs().max().max())}


def database(inputs, path):
    path = Path(path)
    if not path.is_file() and not path.with_suffix(".sql.zst").is_file():
        configuration = ROOT / "config/PROJECT.json"
        simulation_root = Path(inputs.json(configuration)["simulation_root"]) if configuration.exists() else ROOT / "simulations"
        for parent in path.parents:
            if parent.name in {"candidate_runs", "hvac_runs_v2"}:
                relocated = simulation_root / path.relative_to(parent.parent)
                if relocated.is_file() or relocated.with_suffix(".sql.zst").is_file():
                    path = relocated
                break
    archive_record = path.parent / "RAW_SQL_LOSSLESS_ARCHIVE.json"
    record = inputs.json(archive_record) if archive_record.exists() else None
    if path.is_file():
        raw = inputs.check(path, record["uncompressed_sql_sha256"] if record else None).read_bytes()
    elif path.with_suffix(".sql.zst").is_file():
        if record is None:
            raise ValueError("Lossless SQL archive has no retained SHA record")
        archive = inputs.check(path.with_suffix(".sql.zst"), record["archive_sha256"])
        with archive.open("rb") as stream, zstd.ZstdDecompressor().stream_reader(stream) as reader:
            raw = reader.read()
        if hashlib.sha256(raw).hexdigest() != record["uncompressed_sql_sha256"]:
            raise ValueError("Decompressed SQL differs from retained SHA")
    else:
        raise FileNotFoundError("Hourly SQL and lossless archive unavailable: " + str(path))
    connection = sqlite3.connect(":memory:")
    connection.deserialize(raw)
    return connection, str(path)


def tracking(inputs, case, data):
    connection, resolved_sql = database(inputs, case["sql_path"])
    try:
        clock = pd.read_sql_query("select t.TimeIndex,t.Month,t.Day,t.Hour,t.Dst,t.Interval from Time t join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 order by t.TimeIndex", connection)
        if len(clock) != len(data["occupied"]) or not clock.Interval.eq(60).all():
            raise ValueError("SQL clock differs from the retained hourly trajectory")
        for column, field in [("Month", "month"), ("Day", "day"), ("Hour", "hour"), ("Dst", "dst")]:
            if not np.array_equal(clock[column], data[field]):
                raise ValueError("SQL calendar differs from compact output")
        marks = ",".join("?" for _ in VARIABLES)
        dictionary = pd.read_sql_query(f"select ReportDataDictionaryIndex,KeyValue,Name from ReportDataDictionary where ReportingFrequency='Hourly' and Name in ({marks})", connection, params=VARIABLES)
        if dictionary.empty:
            frame = pd.DataFrame(columns=["TimeIndex", "KeyValue", "Name", "Value"])
        else:
            identifiers = dictionary.ReportDataDictionaryIndex.tolist()
            marks = ",".join("?" for _ in identifiers)
            frame = pd.read_sql_query(f"select r.TimeIndex,r.ReportDataDictionaryIndex,r.Value from ReportData r join Time t on t.TimeIndex=r.TimeIndex join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex where e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0 and r.ReportDataDictionaryIndex in ({marks})", connection, params=identifiers)
            frame = frame.merge(dictionary, on="ReportDataDictionaryIndex", validate="many_to_one")
    finally:
        connection.close()
    zones = [str(z).upper() for z in data["zone_ids"]]
    matrices, missing = {}, []
    for name in VARIABLES:
        selected = frame[frame.Name.eq(name)].copy()
        if selected.empty:
            missing.append(name)
            continue
        selected["KeyValue"] = selected.KeyValue.str.upper()
        pivot = selected.pivot(index="TimeIndex", columns="KeyValue", values="Value").reindex(index=clock.TimeIndex, columns=zones)
        if pivot.isna().any().any():
            missing.append(name + " (incomplete clock/zones)")
        else:
            matrices[name] = pivot.to_numpy()
    if "Zone Mean Air Temperature" in matrices or "Zone Air Temperature" in matrices:
        missing = [name for name in missing if not name.startswith(("Zone Mean Air Temperature", "Zone Air Temperature"))]
    rows = []
    for window, mask in [("service", data["occupied"].astype(bool)), ("outside_service", ~data["occupied"].astype(bool))]:
        row = {k: case[k] for k in KEYS + ["system", "placement"]}
        row.update(window=window, window_hours=int(mask.sum()), missing_variables=";".join(missing), resolved_sql_path=resolved_sql)
        for mode in ["Heating", "Cooling"]:
            values = matrices.get("Zone " + mode + " Setpoint Not Met While Occupied Time")
            row[mode.lower() + "_unmet_zone_hours"] = float(values[mask].sum()) if values is not None else np.nan
            row[mode.lower() + "_hourly_max_zone_duration_sum"] = float(values[mask].max(axis=1).sum()) if values is not None else np.nan
        air = matrices.get("Zone Mean Air Temperature", matrices.get("Zone Air Temperature"))
        heat = matrices.get("Zone Thermostat Heating Setpoint Temperature")
        cool = matrices.get("Zone Thermostat Cooling Setpoint Temperature")
        row["air_temperature_variable"] = "Zone Mean Air Temperature" if "Zone Mean Air Temperature" in matrices else "Zone Air Temperature" if air is not None else "missing"
        for mode, target in [("heating", heat), ("cooling", cool)]:
            error = None if air is None or target is None else np.maximum(target-air if mode == "heating" else air-target, 0)
            row[mode + "_hourly_mean_air_tracking_degree_zone_hours"] = float(error[mask].sum()) if error is not None else np.nan
        rows.append(row)
    return rows


def diagnostics(inputs, out, cases):
    energy, targets, blockers = [], [], []
    for number, case in enumerate(cases, 1):
        data = inputs.hourly(case["hourly_path"])
        for window, mask in [("service", data["occupied"].astype(bool)), ("outside_service", ~data["occupied"].astype(bool))]:
            row = {k: case[k] for k in KEYS + ["system", "placement"]}
            row.update(window=window, window_hours=int(mask.sum()))
            row.update({name: float(data[name][mask].sum()) for name in ENERGY})
            joint = mask & (data["heating_electric_kwh"] > 1e-8) & (data["cooling_electric_kwh"] > 1e-8)
            row.update(joint_heating_cooling_hour_bins=int(joint.sum()),
                       joint_heating_electric_kwh=float(data["heating_electric_kwh"][joint].sum()),
                       joint_cooling_electric_kwh=float(data["cooling_electric_kwh"][joint].sum()))
            energy.append(row)
        try:
            targets.extend(tracking(inputs, case, data))
        except FileNotFoundError as error:
            try:
                candidates = locate_sql(inputs, case)
                available = [r for r in candidates if r["available"]]
                if available:
                    targets.extend(tracking(inputs, case | {"sql_path": available[0]["candidate_sql_path"]}, data))
                    error = None
            except Exception as located_error:
                error = located_error
            if error is not None:
                blockers.append({k: case[k] for k in KEYS + ["system", "placement", "sql_path"]} | {"reason": repr(error)})
        except Exception as error:
            blockers.append({k: case[k] for k in KEYS + ["system", "placement", "sql_path"]} | {"reason": repr(error)})
        if number % 10 == 0 or number == len(cases):
            pd.DataFrame(energy).to_csv(out / "ENERGY_WINDOW_LEVELS.csv", index=False)
            pd.DataFrame(targets).to_csv(out / "TRACKING_WINDOW_LEVELS.csv", index=False)
            pd.DataFrame(blockers).to_csv(out / "SQL_BLOCKERS.csv", index=False)
            write_json(out / "STATUS.json", {"decomposition": "completed", "diagnostics": "running",
                       "processed_cases": number, "planned_cases": len(cases), "sql_blocked_cases": len(blockers)})
            print(f"DIAGNOSTICS {number}/{len(cases)}; SQL blockers={len(blockers)}", flush=True)
    for prefix, rows in [("ENERGY", energy), ("TRACKING", targets)]:
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        metrics = [c for c in frame if c not in KEYS + ["system", "placement", "window", "missing_variables", "air_temperature_variable"] and pd.api.types.is_numeric_dtype(frame[c])]
        join = KEYS + ["system", "window"]
        paired = frame[frame.placement.eq("mixed")].merge(frame[frame.placement.eq("arranged")], on=join, suffixes=("_mixed", "_arranged"), validate="one_to_one")
        for metric in metrics:
            paired[metric + "_change"] = paired[metric + "_arranged"] - paired[metric + "_mixed"]
        paired.to_csv(out / (prefix + "_WINDOW_PAIRED_CHANGES.csv"), index=False)
        summarise(paired, ["system", "control", "window"], [m + "_change" for m in metrics]).to_csv(out / (prefix + "_WINDOW_SUMMARY.csv"), index=False)
    missing_cases = sum(bool(r["missing_variables"]) for r in targets[::2])
    return {"status": "completed" if not blockers and not missing_cases else "partial", "planned_cases": len(cases),
            "sql_completed_cases": len(cases)-len(blockers), "sql_blocked_cases": len(blockers),
            "missing_variable_cases": missing_cases}


def registry_asset(inputs, spec):
    value = Path(spec["path"])
    if value.is_absolute():
        raise ValueError("Registered SQL assets must use project-relative paths")
    return inputs.check((CODE / value).resolve(), spec["sha256"])


def original_environment(inputs, case, expected_input, expected_hourly):
    environments = set()
    for _, table in inputs.locator_tables:
        matches = table[table.input_sha256.eq(expected_input) & table.hourly_sha256.eq(expected_hourly)]
        for value in matches.qa_path.drop_duplicates():
            qa_path = Path(value).resolve()
            if qa_path.is_relative_to(CODE) and qa_path.is_file():
                qa = inputs.json(qa_path)
                if qa.get("input_idf_sha256") == expected_input:
                    environments.add((qa["weather_sha256"], qa["runtime_sha256"]))
    if len(environments) > 1:
        raise ValueError("Original physical manifests disagree about weather/runtime")
    if environments:
        return next(iter(environments))
    # Some legacy annual folders retained only NPZ. Their model and execution
    # manifests independently freeze the weather bytes and runtime executable.
    model = inputs.json(MATRIX / "models/MODEL_MANIFEST.json")[case["climate"]]
    inputs.check(model["weather_path"], model["weather_sha256"])
    protocol = inputs.json(MATRIX / "GLOBAL_EXECUTION_PROTOCOL.json")
    return model["weather_sha256"], protocol["runtime_sha256"]


def registered_sql(inputs, case, expected_input, expected_hourly, expected_weather, expected_runtime):
    path = ROOT / "simulations/diagnostic_sql_replay/REGISTERED_SQL_SOURCES.json"
    if not path.is_file():
        return []
    registry = inputs.json(path)
    if registry.get("schema_version") != 1 or registry.get("project_root") != ".":
        raise ValueError("Unsupported registered SQL source schema")
    registry_asset(inputs, {"path": registry["protocol_path"], "sha256": registry["protocol_sha256"]})
    records = []
    for source in registry["sources"]:
        runtime = source["runtime"]
        if (source["input_idf"]["sha256"], source["original_hourly"]["sha256"],
            source["weather"]["sha256"], runtime["executable_sha256"]) != (
                expected_input, expected_hourly, expected_weather, expected_runtime):
            continue
        assets = {name: registry_asset(inputs, source[name]) for name in
                  ["input_idf", "weather", "original_hourly", "sql", "qa", "validation"]}
        executable = registry_asset(inputs, {"path": runtime["executable_path"], "sha256": expected_runtime})
        qa, validation = inputs.json(assets["qa"]), inputs.json(assets["validation"])
        if not qa.get("completed") or qa.get("reported_severe_count") != 0 or qa.get("fatal_occurrences") != 0:
            raise ValueError("Registered physical QA is incomplete or has severe/fatal errors")
        for key, expected in [("input_idf_sha256", expected_input), ("weather_sha256", expected_weather),
                              ("runtime_sha256", expected_runtime)]:
            if qa.get(key) != expected or validation.get(key) != expected:
                raise ValueError("Registered QA/validation disagrees with the original physical environment")
        if runtime.get("api_sha256") is not None:
            inputs.check(executable.parent / "EnergyPlusAPI.dll", runtime["api_sha256"])
            if qa.get("runtime_api_sha256") != runtime["api_sha256"]:
                raise ValueError("Registered API provenance differs from the executed QA")
        if runtime.get("package_manifest_sha256") is not None:
            from thermal_heterogeneity.energyplus_launcher import runtime_provenance
            if runtime_provenance(executable)["runtime_package_manifest_sha256"] != runtime["package_manifest_sha256"]:
                raise ValueError("Registered runtime package differs from this project")
        required = {"temperatures", "people", *ENERGY}
        deltas = validation.get("sql_max_absolute_differences", {})
        if (validation.get("status") != "passed" or validation.get("hours") != 8760
                or validation.get("sql_sha256") != source["sql"]["sha256"]
                or validation.get("qa_sha256") != source["qa"]["sha256"]
                or validation.get("original_hourly_sha256") != expected_hourly
                or not required.issubset(deltas)
                or any(not np.isfinite(deltas[k]) or deltas[k] < 0 or deltas[k] > 1e-8 for k in required)):
            raise ValueError("Registered SQL lacks the frozen exact-trajectory acceptance evidence")
        records.append({k: case[k] for k in KEYS + ["system", "placement"]} |
                       {"manifest": str(path), "candidate_sql_path": str(assets["sql"]),
                        "expected_input_sha256": expected_input, "expected_hourly_sha256": expected_hourly,
                        "expected_weather_sha256": expected_weather, "expected_runtime_sha256": expected_runtime,
                        "registered_source_id": source["source_id"], "registered_origin": source["origin"],
                        "qa_input_sha_matches": True, "raw_sql_exists": True, "archive_exists": False,
                        "available": True, "reused_existing_physical_operation": True})
    return records


def locate_sql(inputs, case):
    """Use full physical manifests, binding both input and compact-output SHA."""
    if not hasattr(inputs, "locator_tables"):
        inputs.locator_source = inputs.csv(ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv").rename(columns={"layout_arranged": "layout"}).set_index(KEYS)
        paths = [ROOT / "results/PLACEMENT_FULL_MANIFEST.csv",
                 MATRIX / "annual_runs_full/ANNUAL_CONTROL_RUN_MANIFEST.csv",
                 MATRIX / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv"]
        configuration = ROOT / "config/PROJECT.json"
        inputs.simulation_root = Path(inputs.json(configuration)["simulation_root"]) if configuration.exists() else ROOT / "simulations"
        for ancestor in [inputs.simulation_root, *inputs.simulation_root.parents]:
            if (ancestor / "SELF_CONTAINED_PROJECT_MANIFEST.json").is_file():
                paths.append(ancestor / "output/from_raw/matrix/annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv")
                break
        inputs.locator_tables = [(path, inputs.csv(path)) for path in dict.fromkeys(paths) if path.is_file()]
    if case["system"] != "vav":
        return []
    source = inputs.locator_source.loc[tuple(case[k] for k in KEYS)]
    role = case["placement"]
    expected_input, expected_hourly = source["input_sha256_" + role], source["hourly_sha256_" + role]
    registry_path = ROOT / "simulations/diagnostic_sql_replay/REGISTERED_SQL_SOURCES.json"
    records = []
    if registry_path.is_file():
        weather, runtime = original_environment(inputs, case, expected_input, expected_hourly)
        records = registered_sql(inputs, case, expected_input, expected_hourly, weather, runtime)
    for manifest, table in inputs.locator_tables:
        matches = table[table.input_sha256.eq(expected_input) & table.hourly_sha256.eq(expected_hourly)]
        for _, matched in matches.drop_duplicates("qa_path").iterrows():
            path = Path(matched.qa_path).parent / "eplusout.sql"
            paths = [path]
            for parent in path.parents:
                if parent.name in {"candidate_runs", "hvac_runs_v2"}:
                    paths.append(inputs.simulation_root / path.relative_to(parent.parent))
                    break
            for candidate in dict.fromkeys(paths):
                qa = candidate.parent / "RUN_QA.json"
                qa_input_matches = inputs.json(qa).get("input_idf_sha256") == expected_input if qa.is_file() else False
                raw, archive = candidate.is_file(), candidate.with_suffix(".sql.zst").is_file()
                records.append({k: case[k] for k in KEYS + ["system", "placement"]} |
                               {"manifest": str(manifest), "candidate_sql_path": str(candidate),
                                "expected_input_sha256": expected_input, "expected_hourly_sha256": expected_hourly,
                                "qa_input_sha_matches": qa_input_matches, "raw_sql_exists": raw,
                                "archive_exists": archive, "available": bool((raw or archive) and qa_input_matches),
                                "reused_existing_physical_operation": matched.get("reused_existing_physical_operation", "not_recorded")})
    return records


def audit_locators(inputs, out, previous):
    previous = Path(previous).resolve()
    if not previous.is_relative_to(OUT.resolve()):
        raise ValueError("Locator audit must inspect an existing mechanism result directory")
    previous_status = inputs.json(previous / "STATUS.json")
    if previous_status.get("status") not in {"partial", "completed"}:
        raise ValueError("Wait for the previous serial analysis to finish")
    history = inputs.json(previous / "INPUT_HASHES.json")
    original_source = ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv"
    inputs.check(original_source, history.get(str(original_source.resolve())))
    cases = inputs.csv(previous / "CASE_SOURCES.csv")
    try:
        blocked = inputs.csv(previous / "SQL_BLOCKERS.csv")
    except pd.errors.EmptyDataError:
        blocked = pd.DataFrame(columns=KEYS + ["system", "placement", "sql_path"])
    rows, unresolved, recovered = [], [], []
    for case in blocked.to_dict("records"):
        candidates = locate_sql(inputs, case)
        rows.extend(candidates)
        available = [r for r in candidates if r["available"]]
        if available:
            descriptor = cases.merge(pd.DataFrame([case])[KEYS + ["system", "placement"]], on=KEYS + ["system", "placement"], validate="one_to_one").iloc[0].to_dict()
            descriptor["sql_path"] = available[0]["candidate_sql_path"]
            try:
                expected = history.get(str(Path(descriptor["hourly_path"]).resolve()))
                recovered.extend(tracking(inputs, descriptor, inputs.hourly(descriptor["hourly_path"], expected)))
                continue
            except Exception as error:
                case["locator_analysis_error"] = repr(error)
        unresolved.append(case | {"manifest_candidates_checked": len(candidates)})
    pd.DataFrame(rows).to_csv(out / "LOCATOR_AUDIT.csv", index=False)
    pd.DataFrame(unresolved).to_csv(out / "UNRESOLVED_SQL.csv", index=False)
    denominator = inputs.csv(previous / "DENOMINATORS.csv")
    for prefix in ["ENERGY", "TRACKING"]:
        frame = inputs.csv(previous / (prefix + "_WINDOW_PAIRED_CHANGES.csv"))
        if prefix == "TRACKING" and recovered:
            levels = pd.concat([inputs.csv(previous / "TRACKING_WINDOW_LEVELS.csv"), pd.DataFrame(recovered)], ignore_index=True)
            join = KEYS + ["system", "window"]
            frame = levels[levels.placement.eq("mixed")].merge(levels[levels.placement.eq("arranged")], on=join, suffixes=("_mixed", "_arranged"), validate="one_to_one")
            for name in inputs.csv(previous / "TRACKING_WINDOW_PAIRED_CHANGES.csv"):
                if name.endswith("_change"):
                    metric = name[:-7]
                    frame[name] = frame[metric + "_arranged"] - frame[metric + "_mixed"]
            levels.to_csv(out / "TRACKING_WINDOW_LEVELS_WITH_RESOLVED_LOCATORS.csv", index=False)
        metrics = [c for c in frame if c.endswith("_change") and c != "window_hours_change"]
        join = KEYS + ["window"]
        common = frame[frame.system.eq("vav")][join + metrics].merge(frame[frame.system.eq("pthp")][join + metrics], on=join, suffixes=("_vav", "_pthp"), validate="one_to_one")
        for metric in metrics:
            common[metric + "_pthp_minus_vav"] = common[metric + "_pthp"] - common[metric + "_vav"]
        common.to_csv(out / (prefix + "_COMMON_SETTINGS.csv"), index=False)
        summarise(common, ["control", "window"], [c for c in common if c not in join]).merge(denominator, on="control", validate="many_to_one").to_csv(out / (prefix + "_COMMON_SUMMARY.csv"), index=False)
    write_json(out / "DEFINITIONS.json", {"scope": "Locator audit of preserved blockers and diagnostics restricted to identical settings across systems",
               "previous_analysis": str(previous), "locator": "Full placement/810-run manifests must match both input IDF and hourly-output SHA; available candidates also require QA input SHA match. Archives are verified again when read.",
               "sign": "Diagnostic changes are arranged minus mixed. Cross-system differences are PTHP change minus VAV change.",
               "denominators": "Use paired_settings and metric-specific n; original 54/control denominators and engineering-qualified 52/53 remain explicit. Missing SQL is not engineering infeasibility."})
    return {"previous_analysis": str(previous), "previous_blocked_cases": len(blocked),
            "resolved_cases": len(recovered)//2, "unresolved_cases": len(unresolved),
            "status": "partial" if unresolved else "completed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition-only", action="store_true")
    parser.add_argument("--audit-locators", type=Path)
    args = parser.parse_args()
    out = OUT / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True)
    (out / "EXECUTED_ANALYSIS_SOURCE.py").write_bytes(Path(__file__).read_bytes())
    inputs = Inputs()
    for path in [__file__, common.__file__, inspect.getfile(ServiceDistribution), inspect.getfile(normalise_profiles)]:
        inputs.check(path)
    definitions = {
        "scope": "Original VAV and independent PTHP; all common qualified 25% settings; no simulation or target retuning",
        "decomposition": "L(original T,original assignment) -> L(original T,new assignment) -> L(new T,new assignment). Positive steps reduce mismatch. Fixed ordering is an exact accounting identity, not a unique causal mediation decomposition.",
        "tail": "Original annual fixed 27-person identities; unweighted mean of individual mismatch; population uses native occupied-person-hour weights through ServiceDistribution",
        "denominator": "Retain each control's complete planned denominator and separately list engineering-unavailable settings",
        "window": "Service=data['occupied']; outside_service is its exact complement. Energy quantities are sums of retained hourly kWh.",
        "unmet_zone_hours": "Sum original hourly EnergyPlus While Occupied duration over conditioned zones and selected service-window hours; the native EnergyPlus occupancy definition remains in the variable",
        "hourly_max_zone_duration_sum": "Sum hourly max of zone unmet durations; not a subhourly cross-zone union",
        "air_tracking": "Sum positive deviations of hourly mean AIR temperature from hourly thermostat targets, degree-zone-hours; no operative-temperature substitution and no claim of exact subhourly tracking",
        "joint_heating_cooling": "Hour bins with aggregate cooling and electric heating both >1e-8 kWh; not proof of simultaneous subhourly operation or same-zone operation",
        "sql": "Raw SQL read into memory; zstd requires retained compressed and uncompressed SHA matches; original databases are never modified or restored to disk",
        "summary": "Means and medians are paired across settings, not independent population samples; medians of steps need not add. Diagnostics may have fewer paired settings when SQL/variables are unavailable."}
    write_json(out / "DEFINITIONS.json", definitions)
    status = {"status": "running", "output_directory": str(out)}
    try:
        if args.audit_locators is not None:
            status["locator_audit"] = audit_locators(inputs, out, args.audit_locators)
            status["status"] = status["locator_audit"]["status"]
            return
        cases, status["decomposition"] = decompose(inputs, out)
        write_json(out / "INPUT_HASHES.json", inputs.hashes)
        write_json(out / "STATUS.json", status)
        print("DECOMPOSITION_COMPLETED " + str(out), flush=True)
        if not args.decomposition_only:
            status["diagnostics"] = diagnostics(inputs, out, cases)
        status["status"] = "completed" if args.decomposition_only or status["diagnostics"]["status"] == "completed" else "partial"
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", exception=repr(error))
        raise
    finally:
        write_json(out / "INPUT_HASHES.json", inputs.hashes)
        write_json(out / "STATUS.json", status)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
