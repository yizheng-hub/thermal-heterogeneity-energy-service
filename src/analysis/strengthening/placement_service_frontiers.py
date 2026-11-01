"""Reselect joint commands and placements under common service and move budgets."""

from common import *
from itertools import product
from analyse_global_results import normalise_profiles, load_hourly, energy_metrics
from service_distribution import ServiceDistribution
import numpy as np
import pandas as pd


SYSTEM_FOLDERS = {"independent_pthp": "hvac_confirmation",
                  "vav_coordinated_outdoor_reset": "vav_outdoor_reset"}
CARRIERS = ["facility_purchased_kwh", "facility_electric_kwh", "facility_gas_hhv_kwh"]
MOVE_BUDGETS = [0.0, 0.25, 0.5, 1.0]


def service_budgets(population, tail):
    rows = [("reference_relative", factor, factor * population, factor * tail)
            for factor in [0.5, 0.75, 0.9, 1.0, 1.05, 1.1]]
    return rows + [("absolute_mismatch", None, p, t)
                   for p, t in product([0.5, 1, 1.5, 2, 2.5], [1, 2, 3, 4, 5])]


def select_operation(frame, population, tail, move_budget):
    eligible = frame[frame.engineering_pass.eq(True)
        & frame.moved_profiles.le(267 * move_budget + 1e-9)
        & frame.population_mismatch_c.le(population + 1e-9)
        & frame.fixed_tail_mismatch_c.le(tail + 1e-9)]
    if eligible.empty:
        return {"status": "infeasible", "qualifying_logical_operations": 0,
                "finite_minimum_energy_kwh": None, "selected_energy_kwh": None}
    minimum = float(eligible.facility_purchased_kwh.min())
    tolerance = max(1e-6, 1e-9 * abs(minimum))
    selected = eligible[eligible.facility_purchased_kwh.le(minimum + tolerance)].sort_values(
        ["moved_profiles", "operation_id"], kind="stable").iloc[0]
    return {"status": "feasible", "qualifying_logical_operations": len(eligible),
        "finite_minimum_energy_kwh": minimum, "selected_energy_kwh": float(selected.facility_purchased_kwh),
        **{"selected_" + key: selected[key] for key in ["operation_id", "control", "layout", "moved_profiles",
            "population_mismatch_c", "fixed_tail_mismatch_c", "facility_electric_kwh", "facility_gas_hhv_kwh"]}}


def frontiers(inventory, anchors, arm):
    rows = []
    for (system, climate, composition), group in inventory.groupby(["system", "climate", "composition"], sort=True):
        reference = anchors[anchors.climate.eq(climate) & anchors.composition.eq(composition)]
        if len(reference) != 1:
            raise ValueError("Missing or duplicated common VAV service reference")
        anchor = reference.iloc[0]
        for kind, factor, population, tail in service_budgets(anchor.anchor_mean_mismatch_c, anchor.anchor_tail_mismatch_c):
            baseline = select_operation(group, population, tail, 0.0)
            previous = None
            for budget in MOVE_BUDGETS:
                choice = select_operation(group, population, tail, budget)
                exact = choice["finite_minimum_energy_kwh"]
                if previous is not None and (exact is None or exact > previous + 1e-6):
                    raise ValueError("Nested relocation library worsened the exact finite energy minimum")
                previous = exact
                row = {"arm": arm, "system": system, "climate": climate, "composition": composition,
                    "budget_type": kind, "reference_fraction": factor, "population_budget_c": population,
                    "fixed_tail_budget_c": tail, "maximum_relocation_fraction": budget,
                    "zero_movement_status": baseline["status"],
                    "newly_feasible_with_movement": baseline["status"] == "infeasible" and choice["status"] == "feasible",
                    **choice}
                if baseline["status"] == choice["status"] == "feasible":
                    zero = baseline["selected_energy_kwh"]
                    row.update(energy_change_from_zero_kwh=choice["selected_energy_kwh"] - zero,
                        energy_change_from_zero_pct=100 * (choice["selected_energy_kwh"] / zero - 1),
                        finite_minimum_change_from_zero_kwh=exact - baseline["finite_minimum_energy_kwh"])
                    for carrier in CARRIERS[1:]:
                        key = "selected_" + carrier
                        row[carrier + "_change_from_zero"] = choice[key] - baseline[key]
                rows.append(row)
    return pd.DataFrame(rows)


def normalize(frame, system, prefix="", arranged=False):
    suffix = "_arranged" if arranged else ""
    def values(name):
        return frame[name + suffix]
    out = frame[["climate", "composition", "composition_id", "control"]].copy()
    out["system"] = system
    out["layout"] = values("layout") if arranged else "random_mixing"
    out["moved_profiles"] = frame.moved_profiles if arranged else 0
    out["engineering_pass"] = values("engineering_pass")
    out["population_mismatch_c"] = values(prefix + "mean_reference_mismatch_c") if not prefix else values("population_mismatch_c")
    out["fixed_tail_mismatch_c"] = values("fixed_tail_reference_mismatch_c") if not prefix else values("fixed_tail_mismatch_c")
    for carrier in CARRIERS:
        out[carrier] = values(carrier)
    out["hourly_path"] = values("hourly_path")
    identities = (out.climate + ":" + out.composition_id.astype(str) + ":" + out.control + ":" + out.layout)
    out["operation_id"] = system + ":" + identities
    return out


def contained_input(path, inputs):
    path = Path(path).resolve()
    if not path.is_relative_to(CODE):
        raise ValueError("Placement frontier input is outside this project")
    inputs[path.relative_to(CODE).as_posix()] = digest(path)
    return path


def build_inventory(inputs):
    def read(relative):
        return pd.read_csv(contained_input(ROOT / relative, inputs))
    profiles = normalise_profiles(pd.read_csv(contained_input(MATRIX / "GLOBAL_PROFILES.csv", inputs)))
    labels = profiles[["composition_id", "composition"]].drop_duplicates()
    composition_ids = labels.set_index("composition").composition_id
    members = read("results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    original = read("results/base/annual/GLOBAL_RUN_METRICS.csv")
    original = original[original.offset_c.eq(0)]
    original = original.assign(composition_id=original.composition.map(composition_ids))
    vav = read("results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv")
    reference = read("results/UNIFORM_SEARCH_METRICS.csv")
    reference = reference[reference.uniform_target_c.eq(25)].copy()
    reference["control"] = "reference_25c"
    frames = [normalize(original, "original_vav"), normalize(reference, "original_vav"),
              normalize(vav, "original_vav", arranged=True)]
    keys = ["climate", "composition_id", "control", "layout"]
    moves = vav.rename(columns={"layout_arranged": "layout"})[keys + ["moved_profiles"]]
    for system, folder in SYSTEM_FOLDERS.items():
        mixed = read(f"results/{folder}/SYSTEM_OPERATION_SCORES.csv")
        mixed = mixed[mixed.system.eq(system)].merge(labels, on="composition_id", validate="many_to_one")
        arranged = read(f"results/placement_hvac_interaction/{system}_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv")
        arranged = arranged.merge(moves, on=keys, validate="one_to_one").merge(labels, on="composition_id", validate="many_to_one")
        arranged = arranged.rename(columns={"layout": "layout_arranged"})
        frames += [normalize(mixed, system, prefix="alternative"),
                   normalize(arranged, system, prefix="alternative", arranged=True)]
        records = []
        for name in ["PILOT_OPERATIONS.json", "CONFIRMATION_OPERATIONS.json"]:
            path = contained_input(ROOT / "results" / folder / name, inputs)
            records += [r for r in json.loads(path.read_text()) if r["system"] == system and r.get("control") == "reference"]
        if len(records) != 9 or len({r["climate"] for r in records}) != 9:
            raise ValueError("The alternative-system actual25C reference coverage is incomplete")
        for record in records:
            path = contained_input(record["hourly_path"], inputs)
            if digest(path) != record["hourly_sha256"]:
                raise ValueError("Reference trajectory checksum differs")
            data = load_hourly(path)
            for composition, group in profiles[profiles.climate.eq(record["climate"])].groupby("composition", sort=True):
                group = group.sort_values("profile_id").reset_index(drop=True)
                ids = set(members[members.climate.eq(record["climate"]) & members.composition.eq(composition)].profile_id)
                tail = np.flatnonzero(group.profile_id.isin(ids))
                if len(group) != 267 or len(tail) != 27:
                    raise ValueError("Reference demand or fixed-tail membership changed")
                population, least = ServiceDistribution(data, group).score(group.centre_c.to_numpy(), tail)
                row = {"system": system, "climate": record["climate"], "composition": composition,
                    "composition_id": int(group.composition_id.iloc[0]), "control": "reference_25c", "layout": "random_mixing",
                    "moved_profiles": 0, "engineering_pass": record["validation_pass"],
                    "population_mismatch_c": population, "fixed_tail_mismatch_c": least,
                    "hourly_path": str(path), **energy_metrics(data)}
                row["operation_id"] = f"{system}:{row['climate']}:{row['composition_id']}:reference_25c:random_mixing"
                frames.append(pd.DataFrame([row]))
    inventory = pd.concat(frames, ignore_index=True)
    if inventory.duplicated(["system", "climate", "composition", "operation_id"]).any():
        raise ValueError("Duplicated joint command-placement operation")
    counts = inventory.groupby(["system", "climate", "composition"]).size()
    if len(counts) != 162 or not counts.eq(10).all():
        raise ValueError("The shared menu must contain10 planned joint operations per system-setting")
    strong = read("results/budget_frontiers/ALL_OPERATION_LIBRARY.csv")
    if len(strong) != 8586 or not strong.layout.eq("random_mixing").all():
        raise ValueError("The complete8586-operation strong mixed library is required")
    strong["composition_id"] = strong.composition.map(composition_ids)
    strong = normalize(strong, "original_vav").assign(
        operation_id=lambda f: "original_vav:strong:" + strong.run_id.astype(str) + ":" + strong.index.astype(str))
    strong = pd.concat([strong, inventory[inventory.system.eq("original_vav") & inventory.moved_profiles.gt(0)]], ignore_index=True)
    return inventory, strong


def validate_inventory(frame, inputs):
    valid = frame[frame.engineering_pass.eq(True)]
    values = valid[CARRIERS + ["population_mismatch_c", "fixed_tail_mismatch_c", "moved_profiles"]].to_numpy(float)
    if not np.isfinite(values).all() or (values < -1e-9).any():
        raise ValueError("Qualified operation has missing, negative or nonfinite service/energy values")
    if not np.allclose(valid.facility_purchased_kwh, valid.facility_electric_kwh + valid.facility_gas_hhv_kwh, rtol=1e-10, atol=1e-6):
        raise ValueError("Purchased-energy carriers do not close")
    for path in valid.hourly_path.unique():
        contained_input(path, inputs)


def verified_cache(out):
    if not (out / "AUDIT.json").is_file():
        return False
    audit = json.loads((out / "AUDIT.json").read_text())
    if audit.get("status") != "completed":
        return False
    for name, expected in json.loads((out / "INPUT_SHA256.json").read_text()).items():
        if digest(CODE / name) != expected:
            raise ValueError("Completed placement-frontier input changed: " + name)
    for name, expected in audit["output_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Completed placement-frontier output changed: " + name)
    print("Verified completed contained placement frontiers", out, flush=True)
    return True


def main():
    out = ROOT / "results/placement_service_frontiers"
    out.mkdir(parents=True, exist_ok=True)
    if verified_cache(out):
        return
    if any(out.iterdir()):
        raise FileExistsError("Preserve the previous placement-frontier run before recomputing")
    inputs = {}
    protocol = contained_input(ROOT / "analysis_plan_amendment_27_placement_service_frontiers.yaml", inputs)
    for name in [__file__, __import__("service_distribution").__file__, __import__("analyse_global_results").__file__]:
        contained_input(name, inputs)
    write_json(out / "EXECUTION_PLAN.json", {"protocol_sha256": digest(protocol), "planned_frontier_cells": 26784,
               "additional_EnergyPlus_runs": 0, "started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()})
    inventory, strong = build_inventory(inputs)
    validate_inventory(pd.concat([inventory, strong], ignore_index=True), inputs)
    anchors_path = contained_input(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv", inputs)
    anchors = pd.read_csv(anchors_path)
    shared = frontiers(inventory, anchors, "shared_command_menu")
    full = frontiers(strong, anchors, "vav_strong_mixed_library")
    if len(shared) != 20088 or len(full) != 6696:
        raise ValueError("Incomplete placement/service frontier coverage")
    for name, frame in [("SHARED_OPERATION_INVENTORY", inventory), ("VAV_STRONG_OPERATION_INVENTORY", strong),
                        ("SHARED_MENU_FRONTIERS", shared), ("VAV_STRONG_FRONTIERS", full)]:
        if "hourly_path" in frame:
            frame = frame.copy()
            frame["hourly_path"] = frame.hourly_path.map(lambda p: Path(p).resolve().relative_to(CODE).as_posix() if pd.notna(p) else p)
        frame.to_csv(out / (name + ".csv"), index=False)
    for path, expected in inputs.items():
        if digest(CODE / path) != expected:
            raise ValueError("Placement frontier input changed during execution: " + path)
    write_json(out / "INPUT_SHA256.json", inputs)
    audit = {"status": "completed", "frontier_cells": len(shared) + len(full), "settings": 54,
        "shared_planned_operations": len(inventory), "shared_engineering_unavailable": int((~inventory.engineering_pass.eq(True)).sum()),
        "strong_planned_operations": len(strong), "nested_exact_energy_minimum_check": "passed",
        "additional_EnergyPlus_runs": 0, "stochastic_draws": 0,
        "output_sha256": {p.name: digest(p) for p in sorted(out.glob("*.csv"))},
        "interpretation": "Finite frozen joint command-placement menus under nominal common model-service budgets; hardware menus and larger within-VAV search are separate arms"}
    write_json(out / "AUDIT.json", audit)
    record_status("placement_service_frontiers", "completed", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
