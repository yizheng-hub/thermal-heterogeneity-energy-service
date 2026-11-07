from common import *
import pandas as pd
import numpy as np
import concurrent.futures as cf
import argparse
from run_global_matrix import _write_run_idf, FLOOR_GROUPS, ZONES
from demand_aware_controls import generate_controls, choose_grid_value

SIM = ROOT / "simulations"


def relocation(group, fraction):
    g = group.sort_values("profile_id").copy().reset_index(drop=True)
    original = g.zone_id.to_numpy().copy()
    assigned = original.copy()
    c = g.centre_c.to_numpy()
    floor = np.array([FLOOR_GROUPS[z] for z in original])
    moved = np.zeros(len(g), dtype=bool)
    limit = int(np.floor(len(g) * fraction))
    for _ in range(limit // 2):
        median = {z: np.median(c[assigned == z]) for z in np.unique(assigned)}
        target = np.array([median[z] for z in assigned])
        old = abs(c - target)
        gain = (
            old[:, None]
            + old[None, :]
            - abs(c[:, None] - target[None, :])
            - abs(c[None, :] - target[:, None])
        )
        allowed = (
            (floor[:, None] == floor[None, :])
            & (assigned[:, None] != assigned[None, :])
            & ~moved[:, None]
            & ~moved[None, :]
        )
        gain[~allowed] = -np.inf
        i, j = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[i, j] <= 1e-10:
            break
        assigned[i], assigned[j] = assigned[j], assigned[i]
        moved[i] = moved[j] = True
    g["zone_id"] = assigned
    g["layout"] = f"within_floor_relocation_{int(100*fraction)}pct"
    assert (
        pd.Series(original)
        .value_counts()
        .sort_index()
        .equals(pd.Series(assigned).value_counts().sort_index())
    )
    assert all(FLOOR_GROUPS[a] == FLOOR_GROUPS[b] for a, b in zip(original, assigned))
    assert np.sum(assigned != original) <= limit
    lower = np.mean([abs(x - np.median(c[assigned == z])) for x, z in zip(c, assigned)])
    return g, {
        "relocation_budget": fraction,
        "profiles_moved": int(np.sum(assigned != original)),
        "profile_count": len(g),
        "ideal_shared_temperature_l1_lower_bound_c": float(lower),
    }


def save_designs(designs, name):
    MODELS = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())
    rows = []
    for d in designs:
        targets = d["targets_c"]
        climate = d["context"]
        physical = (
            name
            + "_"
            + hashlib.sha256(json.dumps([climate, targets], sort_keys=True).encode()).hexdigest()[
                :24
            ]
        )
        path = SIM / (name + "_inputs") / (physical + ".idf")
        _write_run_idf(Path(MODELS[climate]["prepared_idf"]), targets, path)
        rows.append(
            {
                **d,
                "climate": climate,
                "physical_id": physical,
                "input_idf_path": str(path),
                "input_sha256": digest(path),
                "targets_sha256": hashlib.sha256(
                    json.dumps(targets, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    f = pd.DataFrame(rows)
    f["targets_c"] = f.targets_c.map(lambda x: json.dumps(x, sort_keys=True))
    f.to_csv(ROOT / "config" / (name.upper() + "_PLAN.csv"), index=False)
    print(name, len(f), "logical,", f.physical_id.nunique(), "physical", flush=True)


def prepare_baselines():
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    designs = []
    controller = MATRIX / "control_generation/controller_input"
    configs = json.loads((controller / "controller/CONTEXTS.json").read_text())
    for climate, climate_profiles in profiles.groupby("context"):
        cfg = configs[climate]
        with np.load(controller / cfg["probes"]) as saved:
            probes = dict(saved)
        probe_targets = probes["uniform_target_c"].astype(float)
        grid = np.round(np.arange(20, 27.0001, 0.1), 8)
        prediction = {
            z: np.array(
                [
                    np.interp(grid, probe_targets, probes["operative_c"][:, h, j])
                    for h in range(probes["operative_c"].shape[1])
                ]
            ).T
            for j, z in enumerate(ZONES)
        }
        weights = {z: probes["native_people"][:, j] for j, z in enumerate(ZONES)}
        for composition, g in climate_profiles.groupby("composition_id"):
            for scope in ["building", "native_zone"]:
                for statistic in ["mean", "median"]:
                    if scope == "building":
                        center = float(getattr(g.centre_c, statistic)())
                        loss = sum(
                            np.einsum("gh,h->g", abs(prediction[z] - center), weights[z])
                            for z in ZONES
                        ) / sum(weights[z].sum() for z in ZONES)
                        target = float(choose_grid_value(grid, loss))
                        targets = {z: target for z in ZONES}
                    else:
                        targets = {}
                        for z, v in g.groupby("zone_id"):
                            center = float(getattr(v.centre_c, statistic)())
                            loss = (
                                np.einsum("gh,h->g", abs(prediction[z] - center), weights[z])
                                / weights[z].sum()
                            )
                            targets[z] = float(choose_grid_value(grid, loss))
                    designs.append(
                        {
                            "context": climate,
                            "composition_id": int(composition),
                            "layout": "random_mixing",
                            "permission": scope,
                            "control": scope + "_" + statistic,
                            "offset_c": 0.0,
                            "targets_c": targets,
                        }
                    )
    assert len(designs) == 216
    save_designs(designs, "baseline")


def prepare_placement():
    p = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    maps = []
    audits = []
    representative = p[p.context.eq(sorted(p.context.unique())[0])]
    for composition, g in representative.groupby("composition_id"):
        for frac in [0, 0.25, 0.5, 1]:
            local, audit = relocation(g, frac)
            audit["composition_id"] = composition
            audits.append(audit)
            for climate in sorted(p.context.unique()):
                c = local.copy()
                c["context"] = climate
                maps.append(c)
    new = pd.concat(maps, ignore_index=True)
    new.to_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv", index=False)
    pd.DataFrame(audits).to_csv(
        ROOT / "results/CONSTRAINED_PLACEMENT_LOWER_BOUNDS.csv", index=False
    )
    new = new[~new.layout.eq("within_floor_relocation_0pct")]
    designs, _ = generate_controls(
        MATRIX / "control_generation/controller_input",
        profile_table=new,
        permissions=("native_zone",),
        offsets_c=(0.0,),
    )
    designs = [d for d in designs if d["control"] != "uniform_average"]
    assert len(designs) == 324
    save_designs(designs, "placement")
    write_json(
        ROOT / "audit/PLACEMENT_DESIGN_SCOPE.json",
        {
            "grouping": "frozen profile estimates; deterministic capacity-preserving within-floor swaps",
            "budgets": [0, 0.25, 0.5, 1],
            "objective": "reduce ideal shared-temperature L1 lower bound",
            "evaluation": "new physically simulated zone temperatures with frozen profiles; linked individual longitudinal held-out validation remains unavailable",
            "interpretation": "mechanism test under simulated arrangement constraints; not a measured team arrangement or realized relocation benefit",
        },
    )


def execute(name, stage):
    from candidate_runs import one
    f = pd.read_csv(ROOT / "config" / (name.upper() + "_PLAN.csv"))
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    if stage == "pilot":
        f = f[f.climate.isin(pilot["climates"]) & f.composition_id.isin(pilot["compositions"])]
    if stage == "full":
        a = json.loads((ROOT / "audit" / f"{name.upper()}_PILOT.json").read_text())
        assert a["all_complete_and_engineering_pass"]
    records = []
    jobs = f.drop_duplicates("physical_id")
    workers = (
        6
        if stage == "pilot"
        else int(
            json.loads((ROOT / "config/PROJECT.json").read_text()).get("confirmation_workers", 6)
        )
    )
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        future = {pool.submit(one, row): row for row in jobs.itertuples()}
        for fu in cf.as_completed(future):
            row = future[fu]
            try:
                r = fu.result()
            except Exception as e:
                r = {
                    "physical_id": row.physical_id,
                    "completed": False,
                    "relative_engineering_pass": False,
                    "exception": repr(e),
                }
            records.append(r)
            pd.DataFrame(records).to_csv(
                ROOT / "results" / f"{name.upper()}_{stage.upper()}_PHYSICAL_PROGRESS.csv",
                index=False,
            )
            print(
                name, stage, len(records), len(jobs), r.get("relative_engineering_pass"), flush=True
            )
    physical = pd.DataFrame(records)
    result = f.merge(
        physical, on="physical_id", suffixes=("", "_execution"), validate="many_to_one"
    )
    result.to_csv(ROOT / "results" / f"{name.upper()}_{stage.upper()}_MANIFEST.csv", index=False)
    audit = {
        "logical_candidates": len(f),
        "physical_operations": len(jobs),
        "all_complete_and_engineering_pass": bool(
            result.completed.fillna(False).all()
            and result.relative_engineering_pass.fillna(False).all()
        ),
    }
    write_json(ROOT / "audit" / f"{name.upper()}_{stage.upper()}.json", audit)
    record_status(
        name + "_" + stage,
        "completed" if audit["all_complete_and_engineering_pass"] else "partial",
        audit,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["baseline", "placement"])
    p.add_argument("stage", choices=["plan", "pilot", "full"])
    a = p.parse_args()
    if a.stage == "plan":
        prepare_baselines() if a.task == "baseline" else prepare_placement()
    else:
        execute(a.task, a.stage)
