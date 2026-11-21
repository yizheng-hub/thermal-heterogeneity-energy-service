"""Test paired command choices at midpoints adjacent to each selected operation."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from candidate_runs import MODELS, SIM, ZONES, _write_run_idf, one
from common import MATRIX, RAW_RUN, ROOT, digest
from service_threshold_sensitivity import (
    KEYS, SCORE_COLUMNS, evaluate_thresholds, load_hourly, normalise_profiles,
    score_representations, summarise,
)
from strict_warning_audit import audit


STEP = 0.0625


def selected_commands(selected, results):
    requested = set(selected.individual_run_id) | set(selected.zone_average_run_id)
    commands = {}

    def add(run_id, targets):
        if run_id not in requested:
            return
        vector = {name: float(targets[name]) for name in ZONES}
        if run_id in commands and vector != commands[run_id]:
            raise ValueError(f"Conflicting occupied targets for {run_id}")
        commands[run_id] = vector

    for row in pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv").itertuples():
        add(row.run_id, dict.fromkeys(ZONES, row.target_c))
    for path in results.glob("*MANIFEST.csv"):
        fields = pd.read_csv(path, nrows=0).columns
        if {"run_id", "targets_c"}.issubset(fields):
            rows = pd.read_csv(path, usecols=["run_id", "targets_c"]).dropna()
            for row in rows[rows.run_id.isin(requested)].itertuples():
                add(row.run_id, json.loads(row.targets_c))
    annual = pd.read_csv(MATRIX / "annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv")
    initial = json.loads((MATRIX / "control_generation/controller_input/CONTROL_COMMANDS.json").read_text())
    targets = {(c["context"], c["composition_id"], c["control"], c["offset_c"]): c["targets_c"] for c in initial}
    for row in annual[annual.run_id.isin(requested)].itertuples():
        add(row.run_id, targets[row.climate, row.composition_id, row.control, row.offset_c])
    spatial = pd.read_csv(RAW_RUN / "spatial/GLOBAL_SPATIAL_LOGICAL_PLAN.csv")
    for row in spatial[spatial.run_id.isin(requested)].itertuples():
        add(row.run_id, json.loads(row.targets_json))
    if set(commands) != requested:
        raise ValueError(f"Missing target vectors for {sorted(requested - set(commands))}")
    return commands


def main():
    results = ROOT / "results"
    out = results / "local_command_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    source = results / "service_threshold_sensitivity"
    scores = pd.read_csv(source / "PAIRED_OPERATION_SERVICE_SCORES.csv")
    limits = pd.read_csv(source / "REFERENCE_SERVICE_LIMITS.csv")
    primary = evaluate_thresholds(scores, limits, [1])
    commands = selected_commands(primary, results)
    profiles = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    base_inputs = SIM / "candidate_inputs"
    pairs, jobs = [], {}
    checked = set()
    for row in primary.itertuples():
        climate = row.climate
        model = Path(MODELS[climate]["prepared_idf"])
        for rule in ("individual", "zone_average"):
            run_id = getattr(row, rule + "_run_id")
            original = getattr(row, rule + "_physical_plan_id")
            targets = commands[run_id]
            if original not in checked:
                exact = base_inputs / f"local_parent_{original}.idf"
                _write_run_idf(model, targets, exact)
                if digest(exact) != original:
                    raise ValueError(f"Parent targets do not regenerate the selected operation {run_id}")
                checked.add(original)
            for sign in (-1, 1):
                shifted = {z: min(27, max(20, t + sign * STEP)) for z, t in targets.items()}
                path = base_inputs / f"local_{climate}_{run_id}_{sign}.idf"
                _write_run_idf(model, shifted, path)
                fingerprint = digest(path)
                key = climate, fingerprint
                jobs.setdefault(key, SimpleNamespace(climate=climate, physical_id="local_" + climate + "_" + fingerprint[:24],
                                input_sha256=fingerprint, input_idf_path=str(path)))
                scope = getattr(row, rule + "_permission") if len(set(shifted.values())) > 1 else "building"
                pairs.append(dict(climate=climate, composition=row.composition, parent=rule,
                                  displacement_c=sign * STEP, physical_plan_id=fingerprint,
                                  permission=scope))
    plan = pd.DataFrame(pairs)
    plan.to_csv(out / "LOCAL_COMMAND_PAIRS.csv", index=False)
    settings = plan[KEYS].drop_duplicates()
    if len(settings) != 54 or len(plan) != 216:
        raise ValueError("Both neighbours of both selections are required in each setting")
    workers = json.loads((ROOT / "config/PROJECT.json").read_text())["confirmation_workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = pd.DataFrame(list(pool.map(one, jobs.values())))
    manifest = "LOCAL_COMMAND_MANIFEST.csv"
    records.to_csv(results / manifest, index=False)
    audited = audit(manifest)
    if audited["status"] != "completed":
        raise ValueError("Local command engineering checks require review before service assessment")
    hourly = records.set_index(["climate", "input_sha256"], verify_integrity=True)
    extra = []
    for (climate, composition), setting in plan.groupby(KEYS):
        group = profiles[profiles.climate.eq(climate) & profiles.composition.eq(composition)]
        group = group.sort_values("profile_id").reset_index(drop=True)
        ref = limits[limits.climate.eq(climate) & limits.composition.eq(composition)].iloc[0]
        tail = np.flatnonzero(group.profile_id.astype(str).isin(ref.fixed_profile_ids.split(";")))
        if len(group) != 267 or len(tail) != 27:
            raise ValueError("Local comparisons require the original occupants and fixed group")
        for item in setting.drop_duplicates("physical_plan_id").itertuples():
            found = hourly.loc[(climate, item.physical_plan_id)]
            if digest(found.hourly_path) != found.hourly_sha256:
                raise ValueError("Hourly file changed after engineering checks")
            data = load_hourly(Path(found.hourly_path))
            values = score_representations(data, group, tail)
            energy = float(data["facility_electric_kwh"].sum() + np.sum(data["facility_gas_hhv_kwh_annual"]))
            extra.append(dict(climate=climate, composition=composition, run_id=found.run_id,
                              physical_plan_id=item.physical_plan_id, control="adjacent_command",
                              permission=item.permission, facility_purchased_kwh=energy,
                              **dict(zip(SCORE_COLUMNS, values))))
    extra = pd.DataFrame(extra)
    extra.to_csv(out / "LOCAL_COMMAND_SCORES.csv", index=False)
    identifiers = [*KEYS, "physical_plan_id"]
    joined = scores.merge(extra, on=identifiers, suffixes=("_old", "_new"))
    for field in [*SCORE_COLUMNS, "facility_purchased_kwh"]:
        np.testing.assert_allclose(joined[field + "_old"], joined[field + "_new"], rtol=0, atol=1e-6)
    enlarged = pd.concat([scores, extra], ignore_index=True).drop_duplicates(identifiers)
    additions = evaluate_thresholds(enlarged, limits, [1])
    paired = primary.merge(additions, on=KEYS, suffixes=("_primary", "_local"), validate="one_to_one")
    paired.to_csv(out / "PAIRED_LOCAL_SELECTIONS.csv", index=False)
    summaries = []
    for name, selection in [("primary", primary), ("with_neighbours", additions)]:
        summaries.append(summarise(selection).assign(
            library=name, settings_above_one_percent=selection.additional_purchased_energy_pct.gt(1.0).sum()))
    pd.concat(summaries).to_csv(out / "LOCAL_COMMAND_SUMMARY.csv", index=False)


if __name__ == "__main__":
    main()
