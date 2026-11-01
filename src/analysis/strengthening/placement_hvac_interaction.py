"""Cross fixed constrained arrangements with validated alternative HVAC conditions."""

import hvac_confirmation as hc
from common import *
from analyse_global_results import normalise_profiles, load_hourly, energy_metrics
from service_distribution import ServiceDistribution
import pandas as pd
import numpy as np
import time
import concurrent.futures as cf


OUT = ROOT / "results/placement_hvac_interaction"


def execute_phase(plan, models, phase):
    unique = {}
    mapping = []
    system = next(iter(models.values()))["system"]
    cache_path = OUT / (system + "_PHYSICAL_CACHE.json")
    results = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for row in plan.to_dict("records"):
        targets = json.loads(row["targets_c"])
        model = models[row["climate"]]
        key = hashlib.sha256(
            json.dumps(
                [system, row["climate"], digest(model["fixed_capacity_model"]), targets],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:24]
        command = {
            "context": row["climate"],
            "composition_id": int(row["composition_id"]),
            "control": row["control"],
            "targets_c": targets,
            "scenario_tag": "placement_" + key,
        }
        if key not in results:
            unique.setdefault(key, (model, command))
        mapping.append({**row, "operation_key": key, "system": system})
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(hc.operate, *job): key for key, job in unique.items()}
        for future in cf.as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as error:
                results[key] = {"validation_pass": False, "exception": repr(error)}
            write_json(OUT / (phase + "_PHYSICAL_PROGRESS.json"), results)
            write_json(cache_path, results)
            print(phase, len(results), len(unique), results[key].get("validation_pass"), flush=True)
    records = [
        {
            **row,
            **results[row["operation_key"]],
            "composition_id": row["composition_id"],
            "control": row["control"],
            "layout": row["layout"],
        }
        for row in mapping
    ]
    write_json(OUT / (phase + "_LOGICAL_RESULTS.json"), records)
    return records


def score(records, source):
    profiles = normalise_profiles(pd.read_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv"))
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    rows = []
    for row in records:
        item = {
            k: row[k]
            for k in ["system", "climate", "composition_id", "control", "layout", "operation_key"]
            if k in row
        }
        item["engineering_pass"] = row.get("validation_pass", False)
        if item["engineering_pass"]:
            group = (
                profiles[
                    profiles.climate.eq(row["climate"])
                    & profiles.composition_id.eq(row["composition_id"])
                    & profiles.layout.eq(row["layout"])
                ]
                .sort_values("profile_id")
                .reset_index(drop=True)
            )
            assert len(group) == 267
            ids = set(
                members[
                    members.climate.eq(row["climate"])
                    & members.composition.eq(group.composition.iloc[0])
                ].profile_id
            )
            tail = np.flatnonzero(group.profile_id.isin(ids))
            assert len(tail) == 27
            data = load_hourly(Path(row["hourly_path"]))
            population, least = ServiceDistribution(data, group).score(
                group.centre_c.to_numpy(), tail
            )
            item.update(
                population_mismatch_c=population,
                fixed_tail_mismatch_c=least,
                hourly_path=row["hourly_path"],
                **energy_metrics(data),
            )
        rows.append(item)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / (source + "_PLACEMENT_SCORES.csv"), index=False)
    base_folder = "hvac_confirmation" if source == "independent_pthp" else "vav_outdoor_reset"
    mixed = pd.read_csv(ROOT / "results" / base_folder / "SYSTEM_OPERATION_SCORES.csv")
    comparison = frame.merge(
        mixed,
        on=["system", "climate", "composition_id", "control"],
        suffixes=("_arranged", "_mixed"),
        validate="many_to_one",
    )
    if "population_mismatch_c_arranged" in comparison:
        comparison["population_gain_c"] = (
            comparison.population_mismatch_c_mixed - comparison.population_mismatch_c_arranged
        )
        comparison["fixed_tail_gain_c"] = (
            comparison.fixed_tail_mismatch_c_mixed - comparison.fixed_tail_mismatch_c_arranged
        )
        comparison["energy_change_pct"] = 100 * (
            comparison.facility_purchased_kwh_arranged / comparison.facility_purchased_kwh_mixed - 1
        )
    comparison.to_csv(OUT / (source + "_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv"), index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(ROOT / "config/PLACEMENT_PLAN.csv")
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    reports = []
    for system, folder, audit in [
        ("independent_pthp", "hvac_confirmation", "HVAC_CONFIRMATION_AUDIT.json"),
        ("vav_coordinated_outdoor_reset", "vav_outdoor_reset", "AUDIT.json"),
    ]:
        root = ROOT / "results" / folder
        while not (root / audit).exists():
            time.sleep(10)
        confirmation = json.loads((root / audit).read_text())
        if system == "independent_pthp":
            operations = json.loads((root / "PILOT_OPERATIONS.json").read_text()) + json.loads(
                (root / "CONFIRMATION_OPERATIONS.json").read_text()
            )
            own = [row for row in operations if row.get("system") == system]
            eligible = len(own) == 171 and all(row.get("validation_pass") for row in own)
        else:
            eligible = confirmation["status"] == "completed"
        if not eligible:
            reports.append(
                {
                    "system": system,
                    "status": "not_completed",
                    "reason": "Mixed-placement architecture confirmation did not pass.",
                }
            )
            continue
        models = {}
        names = (
            ["PILOT_MODEL_CAPACITIES.json", "FURTHER_MODEL_CAPACITIES.json"]
            if system == "independent_pthp"
            else ["PILOT_CAPACITIES.json", "CONFIRMATION_CAPACITIES.json"]
        )
        for name in names:
            for model in json.loads((root / name).read_text()):
                if model.get("system") == system:
                    models[model["climate"]] = model
        assert len(models) == 9
        small = plan[
            plan.climate.isin(pilot["climates"]) & plan.composition_id.isin(pilot["compositions"])
        ]
        tested = execute_phase(small, models, system + "_pilot")
        gate = len(tested) == 36 and all(r.get("validation_pass") for r in tested)
        write_json(
            OUT / (system + "_GATE.json"), {"passed": gate, "logical_operations": len(tested)}
        )
        if gate:
            mask = plan.climate.isin(pilot["climates"]) & plan.composition_id.isin(
                pilot["compositions"]
            )
            tested += execute_phase(plan[~mask], models, system + "_confirmation")
        score(tested, system)
        reports.append(
            {
                "system": system,
                "status": "completed"
                if gate and all(r.get("validation_pass") for r in tested)
                else "partial",
                "logical_operations": len(tested),
                "engineering_passes": sum(r.get("validation_pass", False) for r in tested),
            }
        )
        write_json(OUT / "AUDIT.json", reports)
    write_json(OUT / "AUDIT.json", reports)


if __name__ == "__main__":
    main()
