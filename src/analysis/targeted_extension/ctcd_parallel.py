"""Prepare the parallel CTCD annual placement/permission design on frozen offices.

The existing three draws and random/native commands remain unchanged. This
entry adds demand ordering and building/floor permissions, without inventing
source-building identities or changing physical occupants and equipment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctcd_controls import (
    BUDGETS, PROFILE_FIELDS, QUALIFIED, annual_input, assign_external_profiles,
    digest, generate, load_hourly, physical_hash, score, signature, validate_profiles,
)

LAYOUTS = ("random_mixing", "demand_ordered")
PERMISSIONS = ("building", "floor", "native_zone")


def parallel_profiles(pool, input_root, contexts, retained):
    random = assign_external_profiles(pool, input_root, contexts)
    pd.testing.assert_frame_equal(random, retained, check_exact=True)
    ordered = []
    for (context, _), group in random.groupby(["context", "composition_id"], sort=False):
        row = group.sort_values(["centre_c", "profile_id"], kind="stable").copy()
        capacities = group.groupby("zone_id").size()
        row["zone_id"] = np.repeat(contexts[context]["zones"],
                                    [capacities[z] for z in contexts[context]["zones"]])
        row["layout"] = "demand_ordered"
        ordered.append(row)
    result = pd.concat([random] + ordered, ignore_index=True)[PROFILE_FIELDS]
    for layout in LAYOUTS:
        check = result[result.layout.eq(layout)].assign(layout="random_mixing")
        validate_profiles(check, input_root, contexts)
    return result


def retained_cache(main_input, external_input, models_root):
    """Retain original successful and rejected decisions for exact input hashes."""
    result = {}
    sources = []
    # External first preserves the original CTCD physical results, including its
    # rejected runs. Primary-cohort cache only supplies additional exact inputs.
    for root, external in ((external_input, True), (main_input, False)):
        manifest = root / "engineering/ANNUAL_RUN_MANIFEST.csv"
        rows = pd.read_csv(manifest, keep_default_na=False).drop_duplicates("run_id")
        sources.append({"path": str(manifest.resolve()), "sha256": digest(manifest)})
        for row in rows.to_dict("records"):
            if row["engineering_status"] not in QUALIFIED | {"cached_rejected", "new_rejected"}:
                continue
            rid = row["run_id"]
            quality = (root / "engineering" / row["qa_path"] if external else
                       models_root / "current_run_gates" / rid / "ENGINEERING_QA.json")
            if not quality.is_file():
                raise FileNotFoundError(quality)
            qa = json.loads(quality.read_text(encoding="utf-8"))
            if bool(qa["engineering_pass"]) != (row["engineering_status"] in QUALIFIED):
                raise ValueError("Stored engineering decision disagrees with retained QA")
            hourly = (root / "engineering" / row["path_hourly"]).resolve()
            available = bool(row["path_hourly"]) and hourly.is_file()
            if row["engineering_status"] in QUALIFIED and not available:
                raise FileNotFoundError(hourly)
            if external:
                if digest(quality) != row["qa_sha256"]:
                    raise ValueError("External QA checksum changed")
                if available and digest(hourly) != row["hourly_sha256"]:
                    raise ValueError("External hourly checksum changed")
            source_idf = (models_root / "current_run_gates" / rid / "input.idf")
            item = {"cached_run_id": rid,
                    "cached_hourly_path": str(hourly) if available else "",
                    "cached_hourly_sha256": digest(hourly) if available else "",
                    "cached_qa_path": str(quality.resolve()), "cached_qa_sha256": digest(quality),
                    "cached_idf_path": str(source_idf.resolve()) if source_idf.is_file() else "",
                    "cached_normalized_idf_sha256": row["normalized_idf_sha256"],
                    "cached_engineering_status": row["engineering_status"],
                    "cached_cohort": "CTCD" if external else "ASHRAE"}
            result.setdefault((row["context"], row["normalized_idf_sha256"]), item)
    return result, sources


def prepare(main_input, external_input, models_root, output):
    contexts = json.loads((main_input / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    models = json.loads((models_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    pool_path = external_input / "controller/CTCD_EXECUTABLE_PROFILES.csv.gz"
    old_profiles = pd.read_csv(external_input / "engineering/PROFILE_ASSIGNMENTS.csv.gz",
                               float_precision="round_trip")
    profiles = parallel_profiles(pd.read_csv(pool_path), main_input, contexts, old_profiles)
    commands = generate(main_input, profile_tables={c: profiles[profiles.context.eq(c)] for c in contexts},
                        permissions=PERMISSIONS, expected_commands=864)
    old_commands = json.loads((external_input / "controller/CONTROL_COMMANDS.json").read_text(encoding="utf-8"))
    random_native = [row for row in commands if row["layout"] == "random_mixing"
                     and row["permission"] == "native_zone"]
    if random_native != old_commands:
        raise ValueError("Original CTCD random/native commands changed")
    cache, cache_sources = retained_cache(main_input, external_input, models_root)
    texts = {}
    for context, cfg in models.items():
        model, weather = (models_root / cfg[key] for key in ("baseline_idf", "weather"))
        if digest(model) != cfg["baseline_idf_sha256"] or digest(weather) != cfg["weather_sha256"]:
            raise ValueError("Frozen physical model/weather changed")
        texts[context] = model.read_text(encoding="utf-8")
    output.mkdir(parents=True, exist_ok=False)
    (output / "idfs").mkdir()
    profiles.to_csv(output / "PROFILE_ASSIGNMENTS.csv.gz", index=False,
                    compression={"method": "gzip", "mtime": 0})
    (output / "CONTROL_COMMANDS.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    records, unique, anchors = [], {}, {}
    for command in commands:
        context, cfg = command["context"], models[command["context"]]
        text = annual_input(texts[context], command["targets_c"], cfg["half_deadband_c"])
        physics, normalized = physical_hash(text), signature(text)
        if physics != physical_hash(texts[context]):
            raise ValueError("A new command changed a non-thermostat physical field")
        identity = context + ":" + cfg["weather_sha256"] + ":" + normalized
        rid = "ctcd_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        cached = cache.get((context, normalized))
        record = {key: command[key] for key in (
            "context", "composition_id", "layout", "permission", "control", "variant", "permission_alias")}
        record.update(run_id=rid, normalized_idf_sha256=normalized, physical_sha256=physics,
                      idf_path=f"idfs/{rid}.idf", weather=cfg["weather"],
                      weather_sha256=cfg["weather_sha256"], kind="annual",
                      execution_status="exact_retained_cache" if cached else "pending_new_run")
        record.update(cached or dict.fromkeys([
            "cached_run_id", "cached_hourly_path", "cached_hourly_sha256", "cached_qa_path",
            "cached_qa_sha256", "cached_idf_path", "cached_normalized_idf_sha256",
            "cached_engineering_status", "cached_cohort"], ""))
        records.append(record)
        if rid not in unique:
            unique[rid] = record
            (output / record["idf_path"]).write_text(text, encoding="utf-8")
        if command["variant"] == 3:
            if cached is None or cached["cached_engineering_status"] not in QUALIFIED:
                raise ValueError("Every common25 anchor needs an existing qualified physical run")
            anchors[context, command["composition_id"], command["layout"]] = cached
    plan, physical = pd.DataFrame(records), pd.DataFrame(unique.values())
    plan.to_csv(output / "ANNUAL_RUN_PLAN.csv", index=False)
    physical.to_csv(output / "PHYSICAL_RUN_PLAN.csv", index=False)
    contracts, tails = [], []
    for (context, draw, layout), cached in sorted(anchors.items()):
        group = profiles[profiles.context.eq(context) & profiles.composition_id.eq(draw)
                         & profiles.layout.eq(layout)].sort_values("profile_id", kind="stable").reset_index(drop=True)
        hourly = Path(cached["cached_hourly_path"])
        result = score(load_hourly(hourly), group, np.arange(267))
        distances = result["profiles"]["reference_mismatch_c"]
        ranked = pd.DataFrame({"profile_id": group.profile_id, "distance": distances}).sort_values(
            ["distance", "profile_id"], ascending=[False, True], kind="stable")
        indices = np.flatnonzero(group.profile_id.isin(ranked.head(27).profile_id))
        base = {"context": context, "composition_id": draw, "layout": layout,
                "anchor_cached_run_id": cached["cached_run_id"], "anchor_hourly_sha256": digest(hourly),
                "anchor_mean_c": result["summary"]["mean_reference_mismatch_c"],
                "anchor_fixed_tail_c": float(distances[indices].mean())}
        for fraction in BUDGETS:
            contracts.append(base | {"service_fraction": fraction,
                                     "mean_budget_c": fraction * base["anchor_mean_c"],
                                     "fixed_tail_budget_c": fraction * base["anchor_fixed_tail_c"]})
        tails.extend({"context": context, "composition_id": draw, "layout": layout,
                      "profile_id": group.profile_id.iloc[i], "anchor_reference_mismatch_c": distances[i]}
                     for i in indices)
    pd.DataFrame(contracts).to_csv(output / "BASELINE_DECISION_CONTRACT.csv", index=False)
    pd.DataFrame(tails).to_csv(output / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)
    cached = physical.execution_status.eq("exact_retained_cache")
    report = {"status": "READY_FOR_ACTUAL_ANNUAL_RUNS", "logical_commands": len(plan),
              "distinct_spatial_conditions": 60, "profile_context_layout_groups": len(anchors),
              "unique_physical_inputs": len(physical), "exact_retained_physical_inputs": int(cached.sum()),
              "pending_new_physical_inputs": int((~cached).sum()),
              "new_inputs_by_context": physical[~cached].context.value_counts().to_dict(),
              "cached_cohort_counts": physical[cached].cached_cohort.value_counts().to_dict(),
              "cached_decision_counts": physical[cached].cached_engineering_status.value_counts().to_dict(),
              "random_native_commands_unchanged": random_native == old_commands,
              "random_native_profiles_unchanged": True, "physical_parameters_changed": False,
              "layouts": list(LAYOUTS), "permissions": list(PERMISSIONS),
              "scope": "CTCD parallel main-cohort annual design: random and demand-ordered placements, building/floor/native-zone permissions; same three draws, three control families and four variants.",
              "source_structure": "No source-structured CTCD layout is invented without comparable source-building identifiers.",
              "service_contract": "Every context/draw/layout has its pre-run common25 mean and fixed 27-profile tail; budgets and identities remain fixed across controls and permissions.",
              "cache_sources": cache_sources, "profiles_input_sha256": digest(pool_path),
              "model_contexts_sha256": digest(models_root / "MODEL_CONTEXTS.json"),
              "command_sha256": digest(output / "CONTROL_COMMANDS.json"),
              "logical_plan_sha256": digest(output / "ANNUAL_RUN_PLAN.csv"),
              "physical_plan_sha256": digest(output / "PHYSICAL_RUN_PLAN.csv")}
    (output / "PARALLEL_COHORT_PROTOCOL.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-input-root", type=Path, default=Path("data/inputs/energy_service"))
    parser.add_argument("--external-input-root", type=Path, default=Path("data/inputs/ctcd_external"))
    parser.add_argument("--model-root", type=Path, default=Path("data/upstream/frozen_buildings"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.main_input_root.resolve(), args.external_input_root.resolve(),
                             args.model_root.resolve(), args.output_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
