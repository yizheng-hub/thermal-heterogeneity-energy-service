"""Paired Beijing office placements at two rated cooling airflows for the top-floor core."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import MATRIX, ROOT, digest
from hvac_confirmation import operate
from hvac_models import idd_fields, parse_idf, serialize_idf
from analyse_global_results import energy_metrics, load_hourly, normalise_profiles
from service_distribution import ServiceDistribution

OUT = ROOT / "results/heatpump_airflow"
MIXED = "within_floor_relocation_0pct"


def equipment(source, name, ratio):
    objects = parse_idf(Path(source["fixed_capacity_model"]).read_text())
    equipment = {(obj[0].lower(), obj[1].lower()): obj for obj in objects if len(obj) > 1}
    coil = equipment["coil:cooling:dx:singlespeed", "core_top pthp cooling coil"]
    unit = equipment["zonehvac:packagedterminalheatpump", "core_top pthp"]
    fan = equipment["fan:onoff", "core_top pthp supply fan"]
    fields = idd_fields()

    def index(obj, field):
        return fields[obj[0].lower()].index(field) + 1

    capacity = float(coil[index(coil, "Gross Rated Total Cooling Capacity")])
    flow = capacity * ratio
    assert float(coil[index(coil, "Rated Air Flow Rate")]) < flow
    for obj, field in [(coil, "Rated Air Flow Rate"), (unit, "Cooling Supply Air Flow Rate")]:
        obj[index(obj, field)] = f"{flow:.12g}"
    fan[index(fan, "Maximum Flow Rate")] = f"{max(flow, float(fan[index(fan, 'Maximum Flow Rate')])):.12g}"
    folder = ROOT / "simulations/airflow_models" / name
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / "heatpump.idf"
    model.write_text(serialize_idf(objects))
    capacities = pd.read_csv(Path(source["fixed_capacity_model"]).parent / "FROZEN_CAPACITIES.csv")
    for obj, field in [(coil, "Rated Air Flow Rate"), (unit, "Cooling Supply Air Flow Rate"), (fan, "Maximum Flow Rate")]:
        slot = capacities.type.eq(obj[0]) & capacities.name.eq(obj[1]) & capacities.field.eq(field)
        if slot.sum() != 1:
            raise ValueError(f"Expected one sizing entry for {obj[1]} {field}")
        capacities.loc[slot, "value"] = float(obj[index(obj, field)])
    capacity_table = folder / "FROZEN_CAPACITIES.csv"
    capacities.to_csv(capacity_table, index=False)
    return dict(source, system=name, fixed_capacity_model=str(model), capacity_sha256=digest(capacity_table))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = ROOT / "results/hvac_confirmation/FURTHER_MODEL_CAPACITIES.json"
    source, = [m for m in json.loads(models.read_text()) if m["climate"] == "Beijing" and m["system"] == "independent_pthp"]
    mixed = json.loads((MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text())
    commands = [dict(c, layout=MIXED) for c in mixed if c["context"] == "Beijing" and c["offset_c"] == 0
                and c["control"] in {"personal_center", "personal_range"}]
    placements = pd.read_csv(ROOT / "config/PLACEMENT_PLAN.csv")
    for entry in placements.loc[placements.climate.eq("Beijing")].to_dict("records"):
        commands.append(dict(entry, targets_c=json.loads(entry["targets_c"])))
    keys = [(c["composition_id"], c["control"], c["layout"]) for c in commands]
    if len(set(keys)) != 48 or len(commands) != 48:
        raise ValueError("Expected twelve mixed and thirty-six rearranged Beijing operations")
    profiles = normalise_profiles(pd.read_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv"))
    groups = profiles.loc[profiles.climate.eq("Beijing")].groupby(["composition_id", "layout"])
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    members = members.loc[members.climate.eq("Beijing")]
    original = pd.read_csv(ROOT / "results/placement_hvac_interaction/independent_pthp_PLACEMENT_SCORES.csv")
    mixed_scores = pd.read_csv(ROOT / "results/hvac_confirmation/SYSTEM_OPERATION_SCORES.csv")
    mixed_scores = mixed_scores.loc[mixed_scores.system.eq("independent_pthp")].assign(layout=MIXED)
    original = pd.concat([original,mixed_scores],ignore_index=True)
    original = original.loc[original.climate.eq("Beijing") & original.control.isin(["personal_center","personal_range"])]
    id_fields = ["composition_id", "control", "layout"]
    original = original.set_index(id_fields, verify_integrity=True)
    output = []
    # Midpoint and upper bound of the EnergyPlus 22.1 rated DX airflow range, in (m3/s)/W.
    for name, ratio in [("pthp_midflow", (0.00004027 + 0.00006041) / 2), ("pthp_highflow", 0.00006041)]:
        model = equipment(source, name, ratio)
        def simulate(command):
            key = tuple(command[k] for k in id_fields)
            tag = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:18]
            result = operate(model, dict(command, scenario_tag="placement_" + tag))
            if not result.get("completed"):
                raise RuntimeError(f"EnergyPlus did not complete {name} {key}")
            group = groups.get_group((key[0],key[2])).sort_values("profile_id").reset_index(drop=True)
            tail = np.flatnonzero(group.profile_id.isin(members.loc[members.composition.eq(group.composition.iloc[0]),"profile_id"]))
            if len(group) != 267 or len(tail) != 27:
                raise ValueError("Occupant references did not match the original comparisons")
            row = dict(zip(id_fields, key), airflow_case=name, airflow_per_capacity_m3_s_w=ratio,
                       original_engineering_pass=original.loc[key,"engineering_pass"],
                       engineering_pass=result["validation_pass"], **result["warning_profile"])
            for measure in ["population_mismatch_c", "fixed_tail_mismatch_c", "facility_purchased_kwh"]:
                row["original_" + measure] = original.loc[key, measure]
            row["hours_below_ventilation_requirement"] = result["engineering_validation"]["hours_below99pct_ventilation_request"]
            if result["validation_pass"]:
                data = load_hourly(Path(result["hourly_path"]))
                row["population_mismatch_c"],row["fixed_group_mismatch_c"] = ServiceDistribution(data,group).score(group.centre_c.to_numpy(),tail)
                row["purchased_energy_kwh"] = energy_metrics(data)["facility_purchased_kwh"]
            print(name,*key,row["engineering_pass"],flush=True)
            return row
        with ThreadPoolExecutor(max_workers=4) as workers:
            output.extend(workers.map(simulate,commands))
    rows = pd.DataFrame(output)
    rows.to_csv(OUT / "BEIJING_AIRFLOW_OPERATIONS.csv",index=False)
    pair_keys = ["airflow_case","composition_id","control"]
    endpoints = ["engineering_pass","population_mismatch_c","fixed_group_mismatch_c","purchased_energy_kwh"]
    paired = rows.loc[rows.layout.ne(MIXED)].merge(rows.loc[rows.layout.eq(MIXED),pair_keys+endpoints],
                on=pair_keys,suffixes=("_arranged","_mixed"),validate="many_to_one")
    for quantity in ["population_mismatch_c","fixed_group_mismatch_c"]:
        paired[quantity+"_reduction"] = paired[quantity+"_mixed"] - paired[quantity+"_arranged"]
    paired["energy_change_pct"] = 100 * (paired.purchased_energy_kwh_arranged / paired.purchased_energy_kwh_mixed - 1)
    paired.to_csv(OUT / "BEIJING_AIRFLOW_PLACEMENT_EFFECTS.csv",index=False)


if __name__ == "__main__":
    main()
