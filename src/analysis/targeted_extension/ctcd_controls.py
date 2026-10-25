"""Generate external CTCD commands and actual annual IDFs on frozen buildings.

The external observations are normalized separately using the same numerical
demand reconstruction as the main cohort. This entry changes only thermostat
commands. Cache reuse requires identical normalized IDF and weather hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENERGY_SERVICE = Path(__file__).resolve().parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from analyse_annual import BUDGETS, QUALIFIED, digest, load_hourly, score  # noqa: E402
from generate_controls import generate  # noqa: E402
from generate_model_inputs import annual_input, physical_hash, signature  # noqa: E402


PROFILE_FIELDS = ["context", "composition_id", "layout", "profile_id", "zone_id",
                  "centre_c", "lower_c", "upper_c", "controller_lower_c", "controller_upper_c"]
DRAW_SEEDS = (2026102501, 2026102502, 2026102503)


def assign_external_profiles(pool, input_root, contexts):
    """Sample without replacement within each draw, then use native capacities."""
    if pool.profile_id.duplicated().any() or len(pool) < 267:
        raise ValueError("External executable pool needs at least 267 unique anonymous profiles")
    rows = []
    for seed in DRAW_SEEDS:
        chosen = np.random.default_rng(seed).choice(len(pool), size=267, replace=False)
        sample = pool.iloc[chosen].iloc[np.random.default_rng(seed + 1000).permutation(267)].copy()
        for context, config in contexts.items():
            with np.load(input_root / config["profiles"], allow_pickle=False) as stored:
                native = pd.DataFrame(dict(stored))
            native = native[native.layout.eq("random_mixing")]
            native = native[native.composition_id.eq(native.composition_id.min())]
            capacity = native.groupby("zone_id").size()
            zones = np.repeat(config["zones"], [capacity[zone] for zone in config["zones"]])
            assigned = sample.copy()
            assigned["context"], assigned["composition_id"] = context, seed
            assigned["layout"], assigned["zone_id"] = "random_mixing", zones
            rows.append(assigned[PROFILE_FIELDS])
    return pd.concat(rows, ignore_index=True)


def validate_profiles(profiles, input_root, contexts):
    """Require three declared 267-profile draws and unchanged zone capacities."""
    missing = set(PROFILE_FIELDS) - set(profiles)
    if missing:
        raise ValueError(f"External profile fields missing: {sorted(missing)}")
    if set(profiles.context) != set(contexts) or set(profiles.layout) != {"random_mixing"}:
        raise ValueError("Use exactly the four frozen contexts and random mixing")
    if profiles.composition_id.nunique() != 3:
        raise ValueError("The bounded external comparison requires exactly three draws")
    if not np.isfinite(profiles[PROFILE_FIELDS[5:]].to_numpy()).all():
        raise ValueError("External demand coordinates must be finite")
    if (profiles.lower_c >= profiles.upper_c).any():
        raise ValueError("Only executable positive-width personal ranges can be assigned")
    if (profiles.controller_lower_c > profiles.controller_upper_c).any():
        raise ValueError("Controller interval bounds are reversed")
    for context, config in contexts.items():
        with np.load(input_root / config["profiles"], allow_pickle=False) as stored:
            original = pd.DataFrame(dict(stored))
        native = original[original.layout.eq("random_mixing")]
        native = native[native.composition_id.eq(native.composition_id.min())]
        capacity = native.groupby("zone_id").size().sort_index()
        for _, group in profiles[profiles.context.eq(context)].groupby("composition_id"):
            if len(group) != 267 or group.profile_id.nunique() != 267:
                raise ValueError("Each draw/context must contain 267 unique profiles")
            if not group.groupby("zone_id").size().sort_index().equals(capacity):
                raise ValueError("Assigned profile capacities differ from the frozen main experiment")
    for _, group in profiles.groupby("composition_id"):
        coordinates = [g.set_index("profile_id")[PROFILE_FIELDS[5:]].sort_index()
                       for _, g in group.groupby("context")]
        if any(not coordinates[0].equals(frame) for frame in coordinates[1:]):
            raise ValueError("The same draw must preserve identities and demand across contexts")


def cache_index(engineering_root):
    """Index only qualified actual retained runs by context and exact input."""
    frame = pd.read_csv(engineering_root / "ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    result = {}
    for row in frame[frame.engineering_status.isin(QUALIFIED)].to_dict("records"):
        path = Path(row["path_hourly"])
        path = path if path.is_absolute() else engineering_root / path
        if not path.is_file():
            raise FileNotFoundError(path)
        result.setdefault((row["context"], row["normalized_idf_sha256"]),
                          {"run_id": row["run_id"], "hourly": path.resolve(),
                           "engineering_status": row["engineering_status"]})
    return result


def prepare(profiles_path, input_root, model_root, output_root):
    supplied = pd.read_csv(profiles_path)
    contexts = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    models = json.loads((model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    profiles = (supplied if "context" in supplied else
                assign_external_profiles(supplied, input_root, contexts))
    validate_profiles(profiles, input_root, contexts)
    tables = {context: profiles[profiles.context.eq(context)].copy() for context in contexts}
    commands = generate(input_root, profile_tables=tables, permissions=("native_zone",),
                        expected_commands=144)
    cache = cache_index(input_root / "engineering")
    source_text, source_physics = {}, {}
    for context, config in models.items():
        baseline, weather = (model_root / config[key] for key in ("baseline_idf", "weather"))
        if digest(baseline) != config["baseline_idf_sha256"] or digest(weather) != config["weather_sha256"]:
            raise ValueError(f"Frozen model/weather checksum changed: {context}")
        source_text[context] = baseline.read_text(encoding="utf-8")
        source_physics[context] = physical_hash(source_text[context])
    output_root.mkdir(parents=True, exist_ok=False)
    idf_root = output_root / "idfs"
    idf_root.mkdir()
    profiles[PROFILE_FIELDS].to_csv(output_root / "PROFILE_ASSIGNMENTS.csv.gz", index=False,
                                   compression={"method": "gzip", "mtime": 0})
    commands_path = output_root / "CONTROL_COMMANDS.json"
    commands_path.write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    plans, unique, anchors = [], {}, {}
    for command in commands:
        context = command["context"]
        config = models[context]
        text = annual_input(source_text[context], command["targets_c"], config["half_deadband_c"])
        physics = physical_hash(text)
        if physics != source_physics[context]:
            raise ValueError("External demand changed a non-thermostat physical input")
        normalized = signature(text)
        # Context and weather prevent cross-weather reuse of an identical small IDF.
        identity = context + ":" + config["weather_sha256"] + ":" + normalized
        run_id = "ctcd_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        cached = cache.get((context, normalized))
        record = {key: command[key] for key in (
            "context", "composition_id", "layout", "permission", "control", "variant")}
        record.update(run_id=run_id, normalized_idf_sha256=normalized, physical_sha256=physics,
                      idf_path=f"idfs/{run_id}.idf", weather=config["weather"],
                      weather_sha256=config["weather_sha256"], kind="annual",
                      execution_status="exact_retained_cache" if cached else "pending_new_run",
                      cached_run_id=cached["run_id"] if cached else "",
                      cached_hourly_path=str(cached["hourly"]) if cached else "",
                      cached_engineering_status=cached["engineering_status"] if cached else "",
                      cached_qa_path=str(model_root / "current_run_gates" / cached["run_id"]
                                         / "ENGINEERING_QA.json") if cached else "",
                      permission_alias="")
        plans.append(record)
        if run_id not in unique:
            unique[run_id] = record
            (idf_root / f"{run_id}.idf").write_text(text, encoding="utf-8")
        if command["variant"] == 3:
            if not cached:
                raise ValueError("The predeclared common25 service anchor must have a qualified retained run")
            anchors[context, command["composition_id"], command["layout"]] = cached
    plan = pd.DataFrame(plans)
    plan.to_csv(output_root / "ANNUAL_RUN_PLAN.csv", index=False)
    pd.DataFrame(unique.values()).to_csv(output_root / "PHYSICAL_RUN_PLAN.csv", index=False)
    contracts, tails = [], []
    for key, cached in sorted(anchors.items()):
        group = profiles[(profiles.context.eq(key[0])) & profiles.composition_id.eq(key[1])]
        group = group.sort_values("profile_id", kind="stable").reset_index(drop=True)
        data = load_hourly(cached["hourly"])
        first = score(data, group, np.arange(267))
        distances = first["profiles"]["reference_mismatch_c"]
        ranking = pd.DataFrame({"profile_id": group.profile_id, "distance": distances}).sort_values(
            ["distance", "profile_id"], ascending=[False, True], kind="stable")
        identities = ranking.head(27).profile_id
        indices = np.flatnonzero(group.profile_id.isin(identities))
        base = {"context": key[0], "composition_id": key[1], "layout": key[2],
                "anchor_cached_run_id": cached["run_id"], "anchor_hourly_sha256": digest(cached["hourly"]),
                "anchor_mean_c": first["summary"]["mean_reference_mismatch_c"],
                "anchor_fixed_tail_c": float(distances[indices].mean())}
        for fraction in BUDGETS:
            contracts.append(base | {"service_fraction": fraction,
                                     "mean_budget_c": fraction * base["anchor_mean_c"],
                                     "fixed_tail_budget_c": fraction * base["anchor_fixed_tail_c"]})
        for index in indices:
            tails.append({"context": key[0], "composition_id": key[1], "layout": key[2],
                          "profile_id": group.profile_id.iloc[index],
                          "anchor_reference_mismatch_c": distances[index]})
    pd.DataFrame(contracts).to_csv(output_root / "BASELINE_DECISION_CONTRACT.csv", index=False)
    pd.DataFrame(tails).to_csv(output_root / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    physical = pd.DataFrame(unique.values())
    report = {"status": "READY_FOR_ACTUAL_ANNUAL_RUNS", "logical_commands": len(plan),
              "unique_physical_inputs": len(physical),
              "exact_retained_physical_inputs": int(physical.execution_status.eq("exact_retained_cache").sum()),
              "pending_new_physical_inputs": int(physical.execution_status.eq("pending_new_run").sum()),
              "physical_simulations_run_by_this_entry": 0, "fixed_building_physics_unchanged": True,
              "command_sha256": digest(commands_path), "profiles_input_sha256": digest(profiles_path),
              "source_manifest_sha256": digest(input_root / "engineering/ANNUAL_RUN_MANIFEST.csv"),
              "model_contexts_sha256": digest(model_root / "MODEL_CONTEXTS.json"),
              "controller_contexts_sha256": digest(input_root / "controller/CONTEXTS.json"),
              "probe_sha256": {context: digest(input_root / cfg["probes"])
                                for context, cfg in contexts.items()},
              "scope": "CTCD external office composition only; three draws, four frozen building-weather contexts, random placement, native-zone permissions, three controls and four declared candidates",
              "population_comparison": "New compositions are not paired individuals from the original cohort; each draw retains its own common25 anchor and fixed 27-profile tail across controls.",
              "probe_reuse": "Existing physical probe arrays are used only to regenerate new commands. Only identical normalized annual IDFs in the qualified retained manifest are cached; a nearby temperature is never a cache match.",
              "physics": "Native People schedules, equipment and all gains remain unchanged; external demand profiles only enter control and service scoring.",
              "draw_seeds": list(DRAW_SEEDS), "sampling": "267 executable profiles without replacement within each draw; overlap between different draws is retained and is not additional independent observations.",
              "service_contract": "Same mean/fixed-tail mismatch endpoints and fractions as the main analysis; absolute budgets and tail identities frozen before new annual runs."}
    (output_root / "EXTERNAL_COHORT_PROTOCOL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.profiles.resolve(), args.input_root.resolve(),
                             args.model_root.resolve(), args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
