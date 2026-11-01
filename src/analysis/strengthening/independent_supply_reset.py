"""Confirm the fixed warmest-zone reset separately in each declared climate."""
from common import *
import hvac_confirmation as hc
from standard_supply_reset import standard_coordination
import pandas as pd


def eligible_climates(records, climates, compositions):
    expected = {(None, "reference")} | {(composition, control)
        for composition in compositions for control in ("uniform_average", "personal_center", "personal_range")}
    return {climate for climate in climates
        if len(local := [r for r in records if r.get("climate") == climate]) == 7
        and {(r.get("composition_id"), r.get("control")) for r in local} == expected
        and all(r.get("validation_pass") for r in local)}


def main():
    source = ROOT / "results/vav_standard_reset"
    out = ROOT / "results/vav_feedback_reset_independent"
    out.mkdir(parents=True, exist_ok=True)
    hc.OUT = out
    hc.coordination_original = hc.coordination
    hc.coordination = standard_coordination
    system = "vav_coordinated_standard"
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    climates = list(json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text()))
    commands = [c for c in json.loads((MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text()) if c["offset_c"] == 0]
    assert len(commands) == 162 and len(climates) == 9
    models = json.loads((source / "PILOT_CAPACITIES.json").read_text())
    tested = json.loads((source / "PILOT_OPERATIONS.json").read_text())
    assert len(models) == 3 and len(tested) == 21
    added = hc.batch([(c, system) for c in climates if c not in pilot["climates"]],
        hc.prepare_model, "ADDITIONAL_CAPACITIES.json")
    models += [m for m in added if "fixed_capacity_model" in m]
    jobs = []
    for model in models:
        if model["climate"] in pilot["climates"]:
            continue
        jobs.append((model, None))
        jobs.extend((model, c) for c in commands if c["context"] == model["climate"]
            and c["composition_id"] in pilot["compositions"])
    tested += hc.batch(jobs, hc.operate, "ADDITIONAL_CLIMATE_PILOTS.json")
    available = eligible_climates(tested, climates, pilot["compositions"])
    write_json(out / "CLIMATE_TECHNICAL_GATES.json", [
        {"climate": c, "eligible": c in available,
         "pilot_operations": sum(r.get("climate") == c for r in tested),
         "pilot_passes": sum(r.get("climate") == c and r.get("validation_pass", False) for r in tested)}
        for c in climates])
    jobs = [(m, c) for m in models if m["climate"] in available
        for c in commands if c["context"] == m["climate"] and c["composition_id"] not in pilot["compositions"]]
    complete = tested + hc.batch(jobs, hc.operate, "REMAINING_COHORT_OPERATIONS.json")
    write_json(out / "ALL_ATTEMPTED_OPERATIONS.json", complete)
    hc.score(complete)
    lookup = {(r.get("climate"), r.get("composition_id"), r.get("control")): r for r in complete}
    inventory = []
    for climate in climates:
        expected = [(None, "reference")] + [(c["composition_id"], c["control"]) for c in commands if c["context"] == climate]
        for composition, control in expected:
            row = lookup.get((climate, composition, control))
            inventory.append({"climate": climate, "composition_id": composition, "control": control,
                "attempted": row is not None, "engineering_pass": bool(row and row.get("validation_pass")),
                "status": "passed" if row and row.get("validation_pass") else (
                    "attempted_engineering_not_passed" if row else "not_run_after_climate_quality_gate")})
    assert len(inventory) == 171
    pd.DataFrame(inventory).to_csv(out / "DECLARED_OPERATION_INVENTORY.csv", index=False)
    report = {"status": "completed" if all(r["engineering_pass"] for r in inventory) else "partial",
        "protocol_execution_completed": True, "planned_logical_operations": len(inventory),
        "attempted_logical_operations": sum(r["attempted"] for r in inventory),
        "engineering_passed_operations": sum(r["engineering_pass"] for r in inventory),
        "eligible_climates": sorted(available), "supply_reset_min_c": 12.8, "supply_reset_max_c": 15.56,
        "scope": "Fixed zone-feedback reset; unchanged transferred demand commands. All unsuccessful and unavailable conditions remain in the declared inventory."}
    write_json(out / "AUDIT.json", report)
    record_status("independent_climate_supply_reset", report["status"], report)


if __name__ == "__main__":
    main()
