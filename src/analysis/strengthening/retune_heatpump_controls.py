"""Separate hardware transfer from response-calibrated control in the heat pump."""

from common import *
import hvac_confirmation as hc
from demand_aware_controls import generate_controls
from hvac_runs import parse_idf, annual_input, ZONES
import numpy as np
import pandas as pd

OUT = ROOT / "results/heatpump_control_response"


def command_key(c):
    return hashlib.sha256(
        json.dumps([c["context"], c["targets_c"]], sort_keys=True).encode()
    ).hexdigest()


def run_commands(models, commands, label):
    by_climate = {m["climate"]: m for m in models}
    unique = {}
    for c in commands:
        unique.setdefault(command_key(c), c)
    physical = hc.batch(
        [
            (by_climate[c["context"]], {**c, "scenario_tag": "own_response"})
            for c in unique.values()
        ],
        hc.operate,
        label + "_PHYSICAL.json",
    )
    assert len(physical) == len(unique)
    index = {}
    for c in unique.values():
        match = [
            r
            for r in physical
            if r.get("climate") == c["context"]
            and r.get("composition_id") == c["composition_id"]
            and r.get("control") == c["control"]
        ]
        assert len(match) == 1, (c, match)
        index[command_key(c)] = match[0]
    logical = []
    for c in commands:
        original = index[command_key(c)]
        logical.append(
            {
                **original,
                "composition_id": c["composition_id"],
                "control": c["control"],
                "physical_source_composition": original["composition_id"],
                "physical_source_control": original["control"],
                "command_generation": "own-system uniform-probe response,identical objective and0.1C search domain",
                "command_key": command_key(c),
            }
        )
    write_json(OUT / (label + "_LOGICAL.json"), logical)
    return logical


def complete_probe_climates(records, targets, climates):
    rows = []
    for climate in sorted(climates):
        local = [r for r in records if r.get("climate") == climate]
        observed = [r.get("probe_target_c") for r in local]
        complete = len(local) == len(targets) and set(observed) == set(targets)
        passed = sum(bool(r.get("validation_pass")) for r in local)
        rows.append({"climate": climate, "expected_probes": len(targets),
            "returned_probes": len(local), "passed_probes": passed,
            "retuning_available": complete and passed == len(targets)})
    return rows


def probe_models(models, targets, references, label, require_all=True):
    records = []
    jobs = []
    for m in models:
        for target in targets:
            if target == 25:
                r = references[m["climate"]]
                assert r["validation_pass"]
                # Check numerical schedule equality, ignoring formatting only.
                before = parse_idf(Path(m["fixed_capacity_model"]).read_text())
                after = parse_idf(
                    annual_input(
                        Path(m["fixed_capacity_model"]).read_text(), {z: 25.0 for z in ZONES}, 1.5
                    )
                )

                def canonical(objects):
                    normalized = []
                    for obj in objects:
                        row = []
                        for value in obj:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                            row.append(value)
                        normalized.append(row)
                    return normalized

                assert canonical(before) == canonical(after)
                records.append({**r, "probe_target_c": 25.0, "reference_reused": True})
            else:
                c = {
                    "context": m["climate"],
                    "composition_id": 0,
                    "control": f"probe_{target:g}",
                    "targets_c": {z: float(target) for z in ZONES},
                    "scenario_tag": "own_response",
                }
                jobs.append((m, c))
    executed = hc.batch(jobs, hc.operate, label + "_PHYSICAL.json")
    for r in executed:
        r["probe_target_c"] = float(r["control"].split("_")[-1])
    records += executed
    write_json(OUT / (label + "_ALL.json"), records)
    if not all(r.get("validation_pass") for r in executed):
        write_json(
            OUT / (label + "_GATE.json"),
            {
                "status": "partial",
                "planned": len(jobs),
                "evaluated": len(executed),
                "passed": sum(r.get("validation_pass", False) for r in executed),
            },
        )
        if require_all:
            raise RuntimeError(
                "Own-response probe grid failed technical quality; every operation is retained in "
                + label + "_PHYSICAL.json"
            )
    return records


def save_responses(records):
    folder = OUT / "controller_input/controller"
    folder.mkdir(parents=True, exist_ok=True)
    configs = {}
    for climate in sorted({r["climate"] for r in records}):
        local = sorted(
            [r for r in records if r["climate"] == climate], key=lambda r: r["probe_target_c"]
        )
        arrays = [hc.load_hourly(Path(r["hourly_path"])) for r in local]
        ref = arrays[0]
        occupied = ref["occupied"]
        people = ref["people"][occupied]
        assert all(np.array_equal(a["people"][occupied], people) for a in arrays)
        path = folder / (climate + "_probes.npz")
        np.savez_compressed(
            path,
            uniform_target_c=np.array([r["probe_target_c"] for r in local]),
            operative_c=np.stack([a["temperatures"][occupied] for a in arrays]),
            native_people=people,
            purchased_site_kwh=np.array(
                [hc.energy_metrics(a)["facility_purchased_kwh"] for a in arrays]
            ),
            zone_ids=np.array(ZONES),
        )
        configs[climate] = {
            "zones": ZONES,
            "floor_groups": {},
            "floor_binding_equals_building": False,
            "probes": str(path.relative_to(OUT / "controller_input")),
        }
    write_json(folder / "CONTEXTS.json", configs)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hc.OUT = OUT
    gate = json.loads((ROOT / "results/hvac_confirmation/CONFIRMATION_GATE.json").read_text())
    assert "independent_pthp" in gate["validated_systems"]
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    models = json.loads(
        (ROOT / "results/hvac_confirmation/PILOT_MODEL_CAPACITIES.json").read_text()
    ) + json.loads((ROOT / "results/hvac_confirmation/FURTHER_MODEL_CAPACITIES.json").read_text())
    models = [m for m in models if m.get("system") == "independent_pthp"]
    assert len(models) == 9
    cases = json.loads(
        (ROOT / "results/hvac_confirmation/PILOT_OPERATIONS.json").read_text()
    ) + json.loads((ROOT / "results/hvac_confirmation/CONFIRMATION_OPERATIONS.json").read_text())
    confirmed = [r for r in cases if r.get("system") == "independent_pthp"]
    assert len(confirmed) == 171 and all(
        r.get("validation_pass") for r in confirmed
    ), "All 171 fixed-command heat-pump operations must meet the equipment criteria."
    references = {
        r["climate"]: r
        for r in cases
        if r.get("system") == "independent_pthp" and r.get("control") == "reference"
    }
    targets = sorted(pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv").target_c.unique())
    assert min(targets) == 20 and max(targets) == 27 and 25 in targets
    pilot_models = [m for m in models if m["climate"] in pilot["climates"]]
    probes = probe_models(pilot_models, targets, references, "PILOT_OWN_RESPONSE_PROBES")
    save_responses(probes)
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    commands, _ = generate_controls(
        OUT / "controller_input",
        profile_table=profiles,
        permissions=("native_zone",),
        contexts_to_include=set(pilot["climates"]),
        groups_to_include={(int(c), "random_mixing") for c in pilot["compositions"]},
        offsets_c=(0.0,),
    )
    assert len(commands) == 18
    tested = run_commands(pilot_models, commands, "PILOT_RETUNED_CONTROLS")
    assert all(
        r.get("validation_pass") for r in tested
    ), "The calibration commands must meet the heat-pump equipment criteria."
    probes += probe_models(
        [m for m in models if m["climate"] not in pilot["climates"]],
        targets,
        references,
        "FURTHER_OWN_RESPONSE_PROBES",
        require_all=False,
    )
    quality = complete_probe_climates(probes, targets, [m["climate"] for m in models])
    write_json(OUT / "CLIMATE_PROBE_QUALITY.json", quality)
    available = {r["climate"] for r in quality if r["retuning_available"]}
    save_responses([r for r in probes if r["climate"] in available])
    commands, diagnostic = generate_controls(
        OUT / "controller_input",
        profile_table=profiles,
        permissions=("native_zone",),
        offsets_c=(0.0,),
        contexts_to_include=available,
    )
    assert len(commands) == 18 * len(available)
    unavailable = [{"climate": m["climate"], "composition_id": int(composition),
        "control": control, "status": "reference_grid_does_not_meet_coverage_criteria"}
        for m in models if m["climate"] not in available
        for composition in profiles.composition_id.unique()
        for control in ("uniform_average", "personal_center", "personal_range")]
    pd.DataFrame(unavailable, columns=["climate", "composition_id", "control", "status"]).to_csv(
        OUT / "HEATPUMP_CONTROL_GRID_COVERAGE.csv", index=False)
    write_json(OUT / "OWN_SYSTEM_CONTROL_COMMANDS.json", commands)
    write_json(OUT / "OWN_SYSTEM_TARGET_DIAGNOSTICS.json", diagnostic)
    extra = [
        c
        for c in commands
        if not (c["context"] in pilot["climates"] and c["composition_id"] in pilot["compositions"])
    ]
    complete = tested + run_commands(models, extra, "CONFIRMATION_RETUNED_CONTROLS")
    hc.score(complete)
    report = {
        "status": "completed" if len(available) == 9 and all(r.get("validation_pass") for r in complete) else "partial",
        "planned_logical_control_operations": 162,
        "unavailable_logical_control_operations": len(unavailable),
        "climate_probe_quality": quality,
        "logical_control_operations": len(complete),
        "validated_operations": sum(r.get("validation_pass", False) for r in complete),
        "uniform_probe_operations": len(probes),
        "probe_targets": targets,
        "purpose": "Own-response calibrated strategy contrasts complement identical-command hardware-transfer contrasts; capacities,loads and outdoor-air specifications remain frozen.",
    }
    write_json(OUT / "RETUNED_HVAC_AUDIT.json", report)
    record_status("own_system_heatpump_control_confirmation", report["status"], report)
