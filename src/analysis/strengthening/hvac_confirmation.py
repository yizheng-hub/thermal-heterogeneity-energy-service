"""Confirm fixed-command system contrasts after a weather-stratified technical gate."""

from hvac_runs import *
from hvac_models import heatpump, coordination, invariant_objects
from run_global_matrix import (
    warning_profile,
    set_maximum_hvac_iterations,
    compare_solver_warning_categories,
)
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles, load_hourly, energy_metrics
import subprocess

OUT = ROOT / "results/hvac_confirmation"


def prepare_model(climate, system):
    model = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())[climate]
    source = parse_idf(Path(model["prepared_idf"]).read_text())
    converted = (coordination if system.startswith("vav_coordinated") else heatpump)(source)
    assert invariant_objects(source) == invariant_objects(converted)
    folder = SIM / "hvac_models_v3" / system / climate
    folder.mkdir(parents=True, exist_ok=True)
    if system == "independent_pthp":
        (folder / "in.idf").write_text(serialize_idf(converted))
        (folder / "Energy+.idd").write_bytes((RUNTIME.parent / "Energy+.idd").read_bytes())
        result = subprocess.run(
            [str(RUNTIME.parent / "ExpandObjects.exe")], cwd=folder, capture_output=True, timeout=60
        )
        (folder / "EXPAND.log").write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0
        converted = parse_idf((folder / "expanded.idf").read_text())
        assert sorted(x.lower() for x in invariant_objects(source)) == sorted(
            x.lower() for x in invariant_objects(converted)
        )
        i = (
            idd_fields()["coil:cooling:dx:singlespeed"].index(
                "Minimum Outdoor Dry-Bulb Temperature for Compressor Operation"
            )
            + 1
        )
        for o in converted:
            if o[0].lower() == "coil:cooling:dx:singlespeed":
                o[i] = "0"
    sizing = folder / "common25_sizing_v3.idf"
    sizing.write_text(add_reporting(serialize_idf(converted)))
    first = run_case(system, climate, sizing, "common25_sizing_v3")
    assert first["completed"], "Sizing trajectory incomplete"
    capacity_sql = first["sql"]
    if system.startswith("vav_coordinated"):
        probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
        anchor = probes[probes.climate.eq(climate) & probes.target_c.eq(25)].iloc[0]
        capacity_sql = str(
            Path(anchor.hourly_path).parents[1] / "runs" / anchor.run_id / "eplusout.sql"
        )
        if not Path(capacity_sql).exists():
            capacity_sql = str(Path(capacity_sql).with_name("simulation_metadata.sql"))
        assert Path(capacity_sql).exists()
    frozen, values, missing = freeze_hardware(sizing, capacity_sql)
    assert not missing, missing
    pd.DataFrame(values).to_csv(folder / "FROZEN_CAPACITIES.csv", index=False)
    path = folder / "fixed_capacity_v3.idf"
    path.write_text(frozen)
    return {
        "system": system,
        "climate": climate,
        "fixed_capacity_model": str(path),
        "capacity_sha256": digest(folder / "FROZEN_CAPACITIES.csv"),
        "capacity_fields": len(values),
        "cooling_compressor_cutout_c": 0 if system == "independent_pthp" else None,
        "geometry_load_ventilation_objects_unchanged": True,
    }


def operation_label(command):
    if command is None:
        return "fixed_reference_v3"
    label = f"composition_{command['composition_id']}_{command['control']}_v3"
    if command.get("scenario_tag"):
        label += "_" + command["scenario_tag"]
    # ReadVarsESO truncates temporary filenames at the first period.
    return label.replace(".", "p")


def operate(model, command=None):
    fixed = Path(model["fixed_capacity_model"])
    base = add_reporting(fixed.read_text())
    text = base if command is None else annual_input(base, command["targets_c"], 1.5)
    label = operation_label(command)
    path = fixed.parent / (label + ".idf")
    path.write_text(text)
    first = run_case(model["system"], model["climate"], path, label)
    if not first["completed"]:
        return {
            **model,
            "label": label,
            "completed": False,
            "validation_pass": False,
            "first": first,
            "composition_id": command["composition_id"] if command else None,
            "control": command["control"] if command else "reference",
        }
    profile = warning_profile(Path(first["sql"]).parent / "eplusout.err")
    stability = {}
    passed_numerics = True
    if any(profile.values()):
        high = path.with_name(path.stem + "_100.idf")
        high.write_text(set_maximum_hvac_iterations(text, 100))
        retry = run_case(model["system"], model["climate"], high, label + "_100")
        passed_numerics = False
        if retry["completed"]:
            with np.load(first["hourly_path"]) as a, np.load(retry["hourly_path"]) as b:
                dt = float(abs(a["temperatures"] - b["temperatures"]).max())
                ea = float(a["facility_electric_kwh"].sum() + a["gas_hhv_kwh"].sum())
                eb = float(b["facility_electric_kwh"].sum() + b["gas_hhv_kwh"].sum())
                de = abs(ea - eb) / max(abs(ea), 1e-9)
                dp = float(abs(a["people"] - b["people"]).max())
            higher = warning_profile(Path(retry["sql"]).parent / "eplusout.err")
            category = compare_solver_warning_categories(profile, higher)
            passed_numerics = bool(
                dt <= 0.02
                and de <= 0.001
                and dp <= 1e-10
                and category["warning_categories_stable"]
                and not higher["coil_operating_range"]
            )
            stability = {
                "temperature_delta_c": dt,
                "relative_purchased_energy_delta": de,
                "people_delta": dp,
                **category,
                "retry": retry,
            }
    qa = inspect_operation(first)
    passed = bool(
        passed_numerics
        and not profile["coil_operating_range"]
        and first["qa"]["reported_severe_count"] == 0
        and first["qa"]["fatal_occurrences"] == 0
        and qa["hourly_hvac_end_use_closure_max_kwh"] <= 1e-6
        and qa["hours_below99pct_ventilation_request"] == 0
    )
    return {
        **model,
        **first,
        "engineering_validation": qa,
        "solver_stability": stability,
        "warning_profile": profile,
        "validation_pass": passed,
        "composition_id": command["composition_id"] if command else None,
        "control": command["control"] if command else "reference",
        "contrast": "same frozen commands; system-specific capacity fixed before comparisons",
    }


def batch(jobs, function, filename):
    records = []
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        fs = {pool.submit(function, *job): job for job in jobs}
        for f in cf.as_completed(fs):
            try:
                r = f.result()
            except Exception as e:
                job = fs[f]
                r = {"validation_pass": False, "exception": repr(e), "job": str(job)}
                if isinstance(job[0], dict):
                    r.update({k: job[0].get(k) for k in ["system", "climate", "capacity_sha256"]})
                    c = job[1] if len(job) > 1 else None
                    r.update(
                        composition_id=c["composition_id"] if c else None,
                        control=c["control"] if c else "reference",
                    )
            records.append(r)
            write_json(OUT / filename, records)
            print(
                filename,
                len(records),
                len(jobs),
                r.get("validation_pass", r.get("capacity_fields")),
                r.get("exception"),
                flush=True,
            )
    return records


def score(records):
    profiles = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    rows = []
    for r in records:
        if r.get("control") == "reference" or r.get("composition_id") is None:
            continue
        item = {
            k: r[k] for k in ["system", "climate", "composition_id", "control", "capacity_sha256"]
        }
        item["engineering_pass"] = r.get("validation_pass", False)
        if not item["engineering_pass"]:
            rows.append(item)
            continue
        g = (
            profiles[
                profiles.climate.eq(r["climate"]) & profiles.composition_id.eq(r["composition_id"])
            ]
            .sort_values("profile_id")
            .reset_index(drop=True)
        )
        composition = g.composition.iloc[0]
        ids = set(
            members[
                members.climate.eq(r["climate"]) & members.composition.eq(composition)
            ].profile_id
        )
        tail = np.flatnonzero(g.profile_id.isin(ids))
        assert len(tail) == 27
        data = load_hourly(Path(r["hourly_path"]))
        curve = ServiceDistribution(data, g)
        population, least = curve.score(g.centre_c.to_numpy(), tail)
        rows.append(
            {
                **item,
                "population_mismatch_c": population,
                "fixed_tail_mismatch_c": least,
                **energy_metrics(data),
                **r["engineering_validation"],
                "hourly_path": r["hourly_path"],
                "hourly_sha256": r["hourly_sha256"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "SYSTEM_OPERATION_SCORES.csv", index=False)
    if "population_mismatch_c" not in frame:
        return
    uniform = frame[frame.control.eq("uniform_average")]
    comparisons = []
    for control in ["personal_center", "personal_range"]:
        p = frame[frame.control.eq(control)].merge(
            uniform,
            on=["system", "climate", "composition_id"],
            suffixes=("_demand", "_average"),
            validate="one_to_one",
        )
        p["compared_control"] = control
        p["delta_purchased_energy_pct"] = 100 * (
            p.facility_purchased_kwh_demand / p.facility_purchased_kwh_average - 1
        )
        p["population_mismatch_reduction_c"] = (
            p.population_mismatch_c_average - p.population_mismatch_c_demand
        )
        p["fixed_tail_mismatch_reduction_c"] = (
            p.fixed_tail_mismatch_c_average - p.fixed_tail_mismatch_c_demand
        )
        comparisons.append(p)
    pd.concat(comparisons, ignore_index=True).to_csv(
        OUT / "WITHIN_SYSTEM_STRATEGY_COMPARISONS.csv", index=False
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    commands = json.loads(
        (MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text()
    )
    pilot_models = batch(
        [(c, s) for c in pilot["climates"] for s in ["vav_coordinated", "independent_pthp"]],
        prepare_model,
        "PILOT_MODEL_CAPACITIES.json",
    )
    assert all("fixed_capacity_model" in m for m in pilot_models)
    jobs = []
    for m in pilot_models:
        jobs.append((m, None))
        jobs.extend(
            (m, c)
            for c in commands
            if c["context"] == m["climate"]
            and c["composition_id"] in pilot["compositions"]
            and c["offset_c"] == 0
        )
    tested = batch(jobs, operate, "PILOT_OPERATIONS.json")
    validated = []
    for s in ["vav_coordinated", "independent_pthp"]:
        subset = [r for r in tested if r.get("system") == s]
        if len(subset) == 21 and all(r.get("validation_pass") for r in subset):
            validated.append(s)
    write_json(
        OUT / "CONFIRMATION_GATE.json",
        {
            "validated_systems": validated,
            "pilot_operations": len(tested),
            "pilot_validated": sum(r.get("validation_pass", False) for r in tested),
        },
    )
    all_climates = list(json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text()))
    models = pilot_models + batch(
        [(c, s) for c in all_climates if c not in pilot["climates"] for s in validated],
        prepare_model,
        "FURTHER_MODEL_CAPACITIES.json",
    )
    jobs = []
    for m in models:
        if m["system"] not in validated:
            continue
        if m["climate"] not in pilot["climates"]:
            jobs.append((m, None))
        jobs.extend(
            (m, c)
            for c in commands
            if c["context"] == m["climate"]
            and c["offset_c"] == 0
            and not (
                c["context"] in pilot["climates"] and c["composition_id"] in pilot["compositions"]
            )
        )
    confirmation = batch(jobs, operate, "CONFIRMATION_OPERATIONS.json")
    complete = tested + confirmation
    score(complete)
    report = {
        "status": "completed"
        if len(validated) == 2 and all(r.get("validation_pass") for r in confirmation)
        else "partial",
        "validated_architectures": validated,
        "evaluated_operations": len(complete),
        "validated_operations": sum(r.get("validation_pass", False) for r in complete),
        "pending": "System-specific target retuning was not performed; contrasts transfer identical original commands.",
    }
    write_json(OUT / "HVAC_CONFIRMATION_AUDIT.json", report)
    record_status("hvac_confirmation", report["status"], report)
