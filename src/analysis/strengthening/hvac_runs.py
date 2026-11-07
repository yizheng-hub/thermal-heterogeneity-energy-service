from common import *
import pandas as pd
import numpy as np
import sqlite3
import concurrent.futures as cf
import argparse
import re
import time
import zstandard as zstd
from global_matrix import parse_idf, serialize_idf, run_energyplus, extract_hourly
from run_global_matrix import ZONES
from generate_model_inputs import annual_input
from hvac_models import idd_fields, SIM

HARDWARE = ("coil:", "fan:", "airloophvac", "airterminal:", "zonehvac:")


def normalized(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        re.sub(r"(?i)^(design size|user-specified)\s+", "", str(name)).split("[")[0].lower(),
    )


def freeze_hardware(idf, sql):
    with sqlite3.connect(sql) as conn:
        sizes = pd.read_sql_query(
            "select CompType,CompName,Description,Value,Units from ComponentSizes", conn
        )
    schema = idd_fields()
    objects = parse_idf(Path(idf).read_text())
    frozen = []
    missing = []
    for o in objects:
        if not o[0].lower().startswith(HARDWARE):
            continue
        for j, value in enumerate(o[1:], start=1):
            if value.lower() not in ["autosize", "autocalculate"]:
                continue
            fields = schema.get(o[0].lower(), [])
            field = fields[j - 1] if j - 1 < len(fields) else f"unknown_field_{j}"
            if (
                o[0].lower() == "coil:heating:dx:singlespeed"
                and field == "Resistive Defrost Heater Capacity"
            ):
                if o[fields.index("Defrost Strategy") + 1].lower() == "reversecycle":
                    o[j] = "0"
                    frozen.append(
                        {
                            "type": o[0],
                            "name": o[1],
                            "field": field,
                            "value": 0.0,
                            "sql_descriptions": "inactive resistive heater under reverse-cycle defrost; structural zero",
                        }
                    )
                    continue
            if (
                o[0].lower() == "airterminal:singleduct:vav:reheat"
                and normalized(field) == "maximumhotwaterorsteamflowrate"
            ):
                coil_index = fields.index("Reheat Coil Object Type") + 1
                if o[coil_index].lower() == "coil:heating:electric":
                    o[j] = "0"
                    frozen.append(
                        {
                            "type": o[0],
                            "name": o[1],
                            "field": field,
                            "value": 0.0,
                            "sql_descriptions": "inactive water-flow field for electric reheat; structural zero",
                        }
                    )
                    continue
            choices = sizes[
                sizes.CompType.str.lower().eq(o[0].lower())
                & sizes.CompName.str.lower().eq(o[1].lower())
            ]
            target = normalized(field)
            aliases = {
                "coolingoutdoorairflowrate": "outdoorairflowrateduringcoolingoperation",
                "heatingoutdoorairflowrate": "outdoorairflowrateduringheatingoperation",
                "noloadoutdoorairflowrate": "outdoorairflowratewhennocoolingorheatingisneeded",
            }
            target = aliases.get(target, target)
            matched = choices[choices.Description.map(normalized).eq(target)]
            # A parent unit's maximum fan flow is the max of its declared heating/cooling design flows.
            if not len(matched) and target == "maximumsupplyairflowrate":
                matched = choices[
                    choices.Description.map(normalized).isin(
                        ["coolingsupplyairflowrate", "heatingsupplyairflowrate"]
                    )
                ]
            if len(matched) and (
                matched.Value.nunique() == 1 or target == "maximumsupplyairflowrate"
            ):
                number = float(matched.Value.max())
                o[j] = format(number, ".12g")
                frozen.append(
                    {
                        "type": o[0],
                        "name": o[1],
                        "field": field,
                        "value": number,
                        "sql_descriptions": ";".join(matched.Description),
                    }
                )
            else:
                missing.append(
                    {
                        "type": o[0],
                        "name": o[1],
                        "field": field,
                        "available_descriptions": choices.Description.tolist(),
                    }
                )
    return serialize_idf(objects), frozen, missing


def add_reporting(text):
    objects = parse_idf(text)
    requested = [
        "Zone Heating Setpoint Not Met While Occupied Time",
        "Zone Cooling Setpoint Not Met While Occupied Time",
        "Zone Thermostat Heating Setpoint Temperature",
        "Zone Thermostat Cooling Setpoint Temperature",
        "System Node Mass Flow Rate",
        "System Node Standard Density Volume Flow Rate",
        "Air System Outdoor Air Mass Flow Rate",
        "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate",
    ]
    existing = {(o[1].lower(), o[2].lower()) for o in objects if o[0].lower() == "output:variable"}
    for name in requested:
        if ("*", name.lower()) not in existing:
            objects.append(["Output:Variable", "*", name, "Hourly"])
    for obj in list(objects):
        if obj[0].lower() == "designspecification:outdoorair" and len(obj) > 7 and obj[7]:
            if (obj[7].lower(), "schedule value") not in existing:
                objects.append(["Output:Variable", obj[7], "Schedule Value", "Hourly"])
                existing.add((obj[7].lower(), "schedule value"))
    return serialize_idf(objects)


def run_case(system, climate, idf, label):
    p = SIM / "hvac_runs_v2" / system / climate / label
    hourly = SIM / "hvac_hourly_v2" / system / climate / (label + ".npz")
    metadata = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())[climate]
    if (p / "RUN_QA.json").exists() and hourly.exists():
        qa = json.loads((p / "RUN_QA.json").read_text())
        assert qa["input_idf_sha256"] == digest(idf)
        sql = p / "eplusout.sql"
        if not sql.exists() and sql.with_suffix(".sql.zst").exists():
            info = json.loads((p / "RAW_SQL_LOSSLESS_ARCHIVE.json").read_text())
            archive = sql.with_suffix(".sql.zst")
            assert digest(archive) == info["archive_sha256"]
            temporary = p / "eplusout.restoring.sql"
            with archive.open("rb") as source, temporary.open("wb") as target:
                zstd.ZstdDecompressor().copy_stream(source, target)
            assert digest(temporary) == info["uncompressed_sql_sha256"]
            temporary.replace(sql)
    else:
        start = time.monotonic()
        try:
            qa = run_energyplus(
                idf, Path(metadata["weather_path"]), RUNTIME, p, hourly, ZONES, cleanup_raw=False
            )
        except ValueError as error:
            if "Expected one Heating:NaturalGas meter, found []" not in str(error):
                raise
            with sqlite3.connect(p / "eplusout.sql") as connection:
                tab = pd.read_sql_query(
                    "select Value,Units from TabularDataWithStrings where ReportName='AnnualBuildingUtilityPerformanceSummary' and TableName='End Uses' and RowName='Heating' and ColumnName='Natural Gas'",
                    connection,
                )
            if len(tab) != 1 or float(tab.Value.iloc[0]) != 0:
                raise ValueError("Missing gas-heating meter is not a proven structural zero")
            frame = pd.read_csv(p / "eplusout.csv")
            raw_hash = digest(p / "eplusout.csv")
            frame["Heating:NaturalGas [J](Hourly)"] = 0.0
            frame.to_csv(p / "eplusout.csv", index=False, float_format="%.17g")
            write_json(
                p / "STRUCTURAL_ZERO_METER.json",
                {
                    "meter": "Heating:NaturalGas",
                    "basis": "SQL annual Heating/NaturalGas end-use equals0; no gas space-heating equipment in PTHP architecture",
                    "sql_value": float(tab.Value.iloc[0]),
                    "raw_csv_sha256": raw_hash,
                    "derived_csv_sha256": digest(p / "eplusout.csv"),
                },
            )
            qa = extract_hourly(p, ZONES, hourly)
            qa.update(
                {
                    "seconds": time.monotonic() - start,
                    "input_idf_sha256": digest(idf),
                    "runtime_sha256": digest(RUNTIME),
                    "weather_sha256": digest(Path(metadata["weather_path"])),
                    "structural_zero_heating_gas": True,
                }
            )
            write_json(p / "RUN_QA.json", qa)
    # SQL, EIO and ERR remain. Only bulky temporary outputs from this new run are removed.
    if qa.get("completed"):
        for f in p.glob("*"):
            if f.is_file() and f.suffix.lower() not in [
                ".json",
                ".err",
                ".eio",
                ".log",
                ".sql",
                ".zst",
            ]:
                f.unlink()
    return {
        "system": system,
        "climate": climate,
        "label": label,
        "hourly_path": str(hourly),
        "hourly_sha256": digest(hourly) if hourly.exists() else None,
        "idf": str(idf),
        "sql": str(p / "eplusout.sql"),
        "qa": qa,
        "completed": qa.get("completed", False),
    }


def design_one(row):
    folder = Path(row.prepared_idf).parent
    path = folder / "common25_sizing.idf"
    path.write_text(add_reporting(Path(row.prepared_idf).read_text()))
    first = run_case(row.system, row.climate, path, "common25_sizing")
    if not first["completed"]:
        return first
    capacity_sql = first["sql"]
    capacity_reference = "system-specific common25 design"
    if row.system == "vav_coordinated":
        probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
        anchor = probes[probes.climate.eq(row.climate) & probes.target_c.eq(25)].iloc[0]
        original_sql = Path(anchor.hourly_path).parents[1] / "runs" / anchor.run_id / "eplusout.sql"
        if not original_sql.exists():
            cfg = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())[row.climate]
            original_idf = folder / "original_vav_common25.idf"
            original_idf.write_text(add_reporting(Path(cfg["prepared_idf"]).read_text()))
            original_run = run_case("vav_original", row.climate, original_idf, "common25_capacity")
            if not original_run["completed"]:
                return {**first, "capacity_reference_failed": True}
            original_sql = Path(original_run["sql"])
        capacity_sql = str(original_sql)
        capacity_reference = "original VAV common25 hardware, unchanged by supply reset"
    frozen, values, unresolved = freeze_hardware(path, capacity_sql)
    pd.DataFrame(values).to_csv(folder / "FROZEN_CAPACITIES.csv", index=False)
    write_json(folder / "UNRESOLVED_CAPACITIES.json", unresolved)
    ready = folder / "fixed_capacity_model.idf"
    ready.write_text(frozen)
    return {
        **first,
        "capacity_reference": capacity_reference,
        "capacity_sql": capacity_sql,
        "fixed_capacity_model": str(ready),
        "fixed_capacity_sha256": digest(ready),
        "fixed_hardware_fields": len(values),
        "unresolved_hardware_fields": len(unresolved),
        "capacity_freeze_pass": not unresolved,
    }


def design(workers=1):
    records = []
    models = pd.read_csv(ROOT / "config/HVAC_PREPARATION.csv")
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(design_one, row) for row in models.itertuples()]
        for f in cf.as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                r = {"completed": False, "exception": repr(e)}
            records.append(r)
            write_json(ROOT / "results/HVAC_DESIGN_PROGRESS.json", records)
            print(
                "HVAC_DESIGN",
                r.get("system"),
                r.get("climate"),
                r.get("completed"),
                r.get("unresolved_hardware_fields"),
                flush=True,
            )
    write_json(ROOT / "results/HVAC_DESIGN_RESULTS.json", records)
    record_status(
        "hvac_design",
        "completed" if all(r.get("capacity_freeze_pass") for r in records) else "partial",
        records,
        [
            "Architecture comparisons remain pending until fixed capacity and ventilation validation pass."
        ],
    )


def variable_frame(sql, name):
    with sqlite3.connect(sql) as connection:
        # Index only newly generated SQL; all scientific values remain unchanged.
        assert Path(sql).resolve().is_relative_to(SIM.resolve())
        connection.execute(
            "create index if not exists hem_validation_rdd_time on ReportData(ReportDataDictionaryIndex,TimeIndex)"
        )
        dictionary = pd.read_sql_query(
            "select ReportDataDictionaryIndex,KeyValue from ReportDataDictionary where ReportingFrequency='Hourly' and lower(Name)=lower(?)",
            connection,
            params=(name,),
        )
        if dictionary.empty:
            return pd.DataFrame()
        ids = dictionary.ReportDataDictionaryIndex.tolist()
        marks = ",".join("?" for _ in ids)
        frame = pd.read_sql_query(
            f"""select r.TimeIndex,r.ReportDataDictionaryIndex,r.Value
        from ReportData r join Time t on t.TimeIndex=r.TimeIndex
        join EnvironmentPeriods e on e.EnvironmentPeriodIndex=t.EnvironmentPeriodIndex
        where r.ReportDataDictionaryIndex in ({marks}) and e.EnvironmentType=3 and coalesce(t.WarmupFlag,0)=0""",
            connection,
            params=ids,
        )
        frame = frame.merge(dictionary, on="ReportDataDictionaryIndex", validate="many_to_one")
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot(index="TimeIndex", columns="KeyValue", values="Value").sort_index()


def inspect_operation(case):
    if not case["completed"]:
        return {"completed": False}
    sql = case["sql"]
    objects = parse_idf(Path(case["idf"]).read_text())
    unmet = {}
    for mode in ["Heating", "Cooling"]:
        frame = variable_frame(sql, "Zone " + mode + " Setpoint Not Met While Occupied Time")
        unmet[mode.lower() + "_unmet_zone_hours"] = (
            float(frame.sum().sum()) if not frame.empty else None
        )
        unmet[mode.lower() + "_unmet_any_zone_hours"] = (
            float(frame.max(axis=1).sum()) if not frame.empty else None
        )
    with np.load(case["hourly_path"]) as saved:
        occupied = saved["occupied"] & (saved["people"].sum(axis=1) > 1e-9)
        people = saved["people"].copy()
        residual = (
            saved["hvac_electric_kwh"]
            - saved["heating_electric_kwh"]
            - saved["cooling_electric_kwh"]
            - saved["fans_electric_kwh"]
        )
    transformer = variable_frame(sql, "Transformer:ExteriorEquipment:Electricity")
    if transformer.empty:
        raise ValueError("Declared transformer meter missing from electrical-balance check")
    transformer_kwh = transformer.sum(axis=1).to_numpy() / 3.6e6
    closure = float(np.max(abs(residual - transformer_kwh)))
    with sqlite3.connect(sql) as connection:
        areas = (
            pd.read_sql_query("select ZoneName,FloorArea from Zones", connection)
            .set_index("ZoneName")
            .FloorArea.to_dict()
        )
    area = {str(k).lower(): float(v) for k, v in areas.items()}
    schedules = variable_frame(sql, "Schedule Value")
    expected = np.zeros(8760)
    for obj in objects:
        if obj[0].lower() != "designspecification:outdoorair":
            continue
        zone = obj[1].replace("SZ DSOA ", "")
        if zone not in ZONES:
            continue
        if obj[2].lower() != "flow/area":
            raise ValueError("Ventilation QA requires the declared Flow/Area specification")
        flow = float(obj[4]) * area[zone.lower()]
        fraction = np.ones(8760)
        if len(obj) > 7 and obj[7]:
            if obj[7].upper() not in schedules:
                raise ValueError("Outdoor-air fractional schedule was not retained in SQL")
            fraction = schedules[obj[7].upper()].to_numpy()
        expected += flow * fraction
    volume = variable_frame(sql, "System Node Standard Density Volume Flow Rate")
    nodes = [o[3].upper() for o in objects if o[0].lower() == "outdoorair:mixer"]
    if not nodes or any(n not in volume for n in nodes):
        raise ValueError("Outdoor-air inlet node flows unavailable")
    actual = volume[nodes].sum(axis=1).to_numpy()
    assert len(actual) == 8760
    active = occupied & (expected > 1e-9)
    ratios = actual[active] / expected[active]
    return {
        "completed": True,
        **unmet,
        "hourly_hvac_end_use_closure_max_kwh": closure,
        "metered_transformer_electricity_kwh": float(transformer_kwh.sum()),
        "balance_terms": "Heating+Cooling+Fans+metered transformer loss = Electricity:HVAC",
        "occupied_outdoor_air_delivery_min_fraction": float(ratios.min()),
        "occupied_outdoor_air_delivery_p01_fraction": float(np.quantile(ratios, 0.01)),
        "occupied_outdoor_air_delivery_mean_fraction": float(ratios.mean()),
        "hours_below99pct_ventilation_request": int((ratios < 0.99).sum()),
        "ventilation_request_rule": "same source Flow/Area specifications and fractional schedules",
        "capacity_sha256": digest(Path(case["idf"]).parent / "FROZEN_CAPACITIES.csv")
        if (Path(case["idf"]).parent / "FROZEN_CAPACITIES.csv").exists()
        else None,
    }


def pilot(workers=1):
    design_rows = json.loads((ROOT / "results/HVAC_DESIGN_RESULTS.json").read_text())
    assert len(design_rows) == 6 and all(r.get("capacity_freeze_pass") for r in design_rows)
    commands = json.loads(
        (MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text()
    )
    compositions = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())["compositions"]
    records = []
    for row in design_rows:
        fixed = Path(row["fixed_capacity_model"])
        base_text = add_reporting(fixed.read_text())
        reference_path = fixed.parent / "fixed_capacity_common25_reported.idf"
        reference_path.write_text(base_text)
        reference = run_case(
            row["system"], row["climate"], reference_path, "fixed_capacity_common25_reported"
        )
        try:
            qa = inspect_operation(reference)
        except Exception as error:
            qa = {"validation_error": repr(error)}
        reference["engineering_validation"] = qa
        records.append(reference)
        write_json(ROOT / "results/HVAC_PILOT_PROGRESS.json", records)
        print("HVAC_FIXED_REFERENCE", row["system"], row["climate"], qa, flush=True)
        # Temperature commands are held fixed for this hardware-mechanism contrast.
        # Retuned system-specific controllers are a separate unperformed problem.
        selected = [
            c
            for c in commands
            if c["context"] == row["climate"]
            and c["composition_id"] in compositions
            and c["offset_c"] == 0
        ]
        for c in selected:
            label = f"composition_{c['composition_id']}_{c['control']}"
            path = fixed.parent / (label + ".idf")
            path.write_text(annual_input(base_text, c["targets_c"], 1.5))
            case = run_case(row["system"], row["climate"], path, label)
            try:
                case["engineering_validation"] = inspect_operation(case)
            except Exception as error:
                case["engineering_validation"] = {"validation_error": repr(error)}
            case.update(
                {
                    "composition_id": c["composition_id"],
                    "control": c["control"],
                    "reference_capacity_sha256": digest(fixed.parent / "FROZEN_CAPACITIES.csv"),
                    "command_comparison": "identical frozen commands transferred between hardware configurations",
                }
            )
            records.append(case)
            write_json(ROOT / "results/HVAC_PILOT_PROGRESS.json", records)
            print(
                "HVAC_PILOT",
                row["system"],
                row["climate"],
                c["control"],
                case["completed"],
                flush=True,
            )
    write_json(ROOT / "results/HVAC_PILOT_RESULTS.json", records)
    passed = all(
        r.get("completed") and not r.get("engineering_validation", {}).get("validation_error")
        for r in records
    )
    record_status(
        "hvac_pilot",
        "completed" if passed else "partial",
        records,
        ["Full54-setting confirmation and system-specific target reoptimization are pending."],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["design", "pilot"])
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    design(a.workers) if a.stage == "design" else pilot(a.workers)
