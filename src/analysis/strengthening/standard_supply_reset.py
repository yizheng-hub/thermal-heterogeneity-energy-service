"""Evaluate the documented narrower warmest-zone supply reset on fixed VAV hardware."""

import hvac_confirmation as hc
from common import *


def standard_coordination(objects):
    result = hc.coordination_original(objects)
    count = 0
    for obj in result:
        if obj[0].lower() == "setpointmanager:warmest":
            assert obj[5] == "18.0"
            obj[5] = "15.56"
            count += 1
    assert count == 3
    return result


def main(
    system="vav_coordinated_standard", transform=standard_coordination, folder="vav_standard_reset"
):
    hc.coordination_original = hc.coordination
    hc.coordination = transform
    hc.OUT = ROOT / "results" / folder
    hc.OUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    commands = json.loads(
        (MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text()
    )
    models = hc.batch(
        [(c, system) for c in pilot["climates"]], hc.prepare_model, "PILOT_CAPACITIES.json"
    )
    assert all("fixed_capacity_model" in model for model in models)
    jobs = []
    for model in models:
        jobs.append((model, None))
        jobs.extend(
            (model, c)
            for c in commands
            if c["context"] == model["climate"]
            and c["composition_id"] in pilot["compositions"]
            and c["offset_c"] == 0
        )
    tested = hc.batch(jobs, hc.operate, "PILOT_OPERATIONS.json")
    gate = len(tested) == 21 and all(r.get("validation_pass") for r in tested)
    write_json(
        hc.OUT / "TECHNICAL_GATE.json",
        {
            "passed": gate,
            "operations": len(tested),
            "validated": sum(r.get("validation_pass", False) for r in tested),
        },
    )
    confirmation = []
    if gate:
        climates = list(json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text()))
        models += hc.batch(
            [(c, system) for c in climates if c not in pilot["climates"]],
            hc.prepare_model,
            "CONFIRMATION_CAPACITIES.json",
        )
        jobs = []
        for model in models:
            if model["climate"] not in pilot["climates"]:
                jobs.append((model, None))
            jobs.extend(
                (model, c)
                for c in commands
                if c["context"] == model["climate"]
                and c["offset_c"] == 0
                and not (
                    c["context"] in pilot["climates"]
                    and c["composition_id"] in pilot["compositions"]
                )
            )
        confirmation = hc.batch(jobs, hc.operate, "CONFIRMATION_OPERATIONS.json")
    complete = tested + confirmation
    hc.score(complete)
    report = {
        "status": "completed"
        if gate and all(r.get("validation_pass") for r in confirmation)
        else "partial",
        "system": system,
        "operations": len(complete),
        "validated": sum(r.get("validation_pass", False) for r in complete),
        "supply_reset_min_c": 12.8,
        "supply_reset_max_c": 15.56,
        "commands": "Original VAV demand commands transferred unchanged; same physical hardware and fixed capacity.",
    }
    write_json(hc.OUT / "AUDIT.json", report)
    record_status("vav_standard_supply_reset", report["status"], report)


if __name__ == "__main__":
    main()
