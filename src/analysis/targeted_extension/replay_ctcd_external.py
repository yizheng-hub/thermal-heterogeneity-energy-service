"""Replay the bounded CTCD external-composition comparison without EnergyPlus.

Retained actual hourly runs and engineering decisions are checksum verified.
Optional upstream/model-input checks replay the raw-data and IDF construction.
The external sample never replaces or pairs individuals with the main cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from ctcd_controls import assign_external_profiles, generate, digest, annual_input, signature
from ctcd_controls import ENERGY_SERVICE
from ctcd_alignment import plot_distributions

sys.path.insert(0, str(ENERGY_SERVICE))
from analyse_annual import QUALIFIED, load_hourly  # noqa: E402
from bundle import verify_bundle  # noqa: E402

KEYS = ["context", "composition_id", "layout"]


def validate_retained(root):
    """Keep rejected/failed rows and check each recorded physical trajectory."""
    engineering = root / "engineering"
    table = pd.read_csv(engineering / "ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    if len(table) != 144 or table.duplicated(KEYS + ["permission", "control", "variant"]).any():
        raise ValueError("External matrix is incomplete or duplicated")
    counts = {}
    for run_id, group in table.groupby("run_id"):
        row = group.iloc[0]
        if group.engineering_status.nunique() != 1 or group.path_hourly.nunique() != 1:
            raise ValueError("One physical input has inconsistent retained decisions")
        status = row.engineering_status
        counts[status] = counts.get(status, 0) + 1
        if status == "execution_failed":
            if row.path_hourly:
                raise ValueError("A failed execution must not enter as a usable trajectory")
            continue
        if status not in QUALIFIED | {"cached_rejected", "new_rejected"}:
            raise ValueError(f"Unfinished or unknown status for {run_id}")
        path = engineering / row.path_hourly
        if Path(row.path_hourly).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Retained hourly paths must be contained relative package paths")
        if digest(path) != row.hourly_sha256:
            raise ValueError("Retained external hourly checksum changed")
        load_hourly(path)
        qa_path = engineering / row.qa_path
        if Path(row.qa_path).is_absolute() or not qa_path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Engineering QA paths must be contained relative package paths")
        if digest(qa_path) != row.qa_sha256:
            raise ValueError("Retained engineering decision checksum changed")
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if bool(qa["engineering_pass"]) != (status in QUALIFIED):
            raise ValueError("Engineering decisions do not agree with retained status")
    return {"logical_candidates": len(table), "unique_physical_inputs": table.run_id.nunique(),
            "unique_status_counts": counts}


def verify_contract(annual_root, engineering_root, expected_anchors=12):
    """The new runs cannot redefine their common25 budgets or tail identities."""
    expected = pd.read_csv(engineering_root / "BASELINE_DECISION_CONTRACT.csv")
    actual = pd.read_csv(annual_root / "ANNUAL_ANCHORS.csv")
    paired = expected.merge(actual, on=KEYS, suffixes=("_before", "_after"), validate="many_to_one")
    if len(paired) != expected_anchors * 4 or len(actual) != expected_anchors:
        raise ValueError(f"Expected {expected_anchors} anchors and four budgets for each")
    for column in ("anchor_mean_c", "anchor_fixed_tail_c"):
        if not np.allclose(paired[column + "_before"], paired[column + "_after"], rtol=0., atol=1e-9):
            raise ValueError("The predeclared absolute service anchor changed")
    old_tail = pd.read_csv(engineering_root / "FIXED_TAIL_MEMBERSHIPS.csv")
    new_tail = pd.read_csv(annual_root / "FIXED_TAIL_MEMBERSHIPS.csv")
    for key, group in old_tail.groupby(KEYS):
        matches = new_tail[(new_tail[KEYS] == pd.Series(key, index=KEYS)).all(axis=1)]
        if len(group) != 27 or set(group.profile_id) != set(matches.profile_id):
            raise ValueError("The predeclared 27-profile tail identities changed")
    selections = pd.read_csv(annual_root / "ANNUAL_SERVICE_SELECTIONS.csv")
    paired = selections.merge(expected, on=KEYS + ["service_fraction"],
                              suffixes=("_selected", "_before"), validate="many_to_one")
    for column in ("mean_budget_c", "fixed_tail_budget_c"):
        if not np.allclose(paired[column + "_selected"], paired[column + "_before"], rtol=0., atol=1e-9):
            raise ValueError("A control used a different predeclared service budget")
    return {"anchors": expected_anchors, "budget_contracts": expected_anchors * 4, "fixed_tail_memberships": len(old_tail),
            "absolute_tolerance_c": 1e-9, "status": "PASS"}


def summarize(annual_root, output_root):
    pairs = pd.read_csv(annual_root / "CONTROL_COMPARISONS.csv")
    candidates = pd.read_csv(annual_root / "ANNUAL_ALL_CANDIDATES.csv")
    carbon = candidates.dropna(subset=["annual_operational_co2_kg"]).drop_duplicates("run_id").set_index(
        "run_id").annual_operational_co2_kg
    reference = pairs.reference_run_id.map(carbon)
    pairs["carbon_change_pct"] = np.where(reference.gt(0),
        100 * pairs.delta_annual_operational_co2_kg / reference, np.nan)
    pairs["data_source"] = "CTCD"
    pairs["placement"] = "random_mixing"
    pairs["permission_scope"] = "native_zone"
    pairs[(pairs.comparison_mode.eq("direct")) | pairs.service_fraction.eq(1.)].to_csv(
        output_root / "CTCD_MAIN_FIGURE_POINTS.csv", index=False)
    rows = []
    for key, group in pairs.groupby(["context", "comparison_mode", "service_fraction", "alternative"]):
        matched = group[group.status.eq("matched")]
        item = dict(zip(["context", "comparison_mode", "service_fraction", "control"], key))
        item.update(declared_draws=len(group), matched_draws=len(matched),
                    unmatched_draws=int(group.status.ne("matched").sum()))
        for column, name in (("delta_site_pct", "energy_change_pct"),
                             ("carbon_change_pct", "carbon_change_pct"),
                             ("delta_interval_coverage_percentage_points", "coverage_change_pp"),
                             ("delta_mean_reference_mismatch_c", "population_mismatch_change_c"),
                             ("delta_fixed_tail_reference_mismatch_c", "fixed_tail_mismatch_change_c")):
            item[name + "_median"] = float(matched[column].median()) if len(matched) else np.nan
            item[name + "_minimum"] = float(matched[column].min()) if len(matched) else np.nan
            item[name + "_maximum"] = float(matched[column].max()) if len(matched) else np.nan
        item["energy_increase_draws"] = int(matched.delta_site_pct.gt(1e-9).sum())
        item["energy_decrease_draws"] = int(matched.delta_site_pct.lt(-1e-9).sum())
        item["energy_tie_draws"] = int(matched.delta_site_pct.abs().le(1e-9).sum())
        item["coverage_improved_draws"] = int(matched.delta_interval_coverage_percentage_points.gt(1e-9).sum())
        item["carbon_change_kg_median"] = float(matched.delta_annual_operational_co2_kg.median()) if len(matched) else np.nan
        item["data_source"] = "CTCD"
        item["placement"], item["permission_scope"] = "random_mixing", "native_zone"
        rows.append(item)
    pd.DataFrame(rows).to_csv(output_root / "CTCD_PAIR_SUMMARY.csv", index=False)
    selected = pd.read_csv(annual_root / "ANNUAL_SERVICE_SELECTIONS.csv")
    selected.groupby(["context", "control", "service_fraction", "selection_status"], dropna=False).size().rename(
        "draws").reset_index().to_csv(output_root / "CTCD_SERVICE_FEASIBILITY.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/inputs/ctcd_external"))
    parser.add_argument("--main-input-root", type=Path, default=Path("data/inputs/energy_service"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, help="Optional locally supplied frozen complete models for exact IDF reconstruction")
    parser.add_argument("--raw-root", type=Path, help="Optional official three-class CTCD files for complete normalization replay")
    parser.add_argument("--ashrae-root", type=Path, default=Path("data/upstream/ashrae_db_ii"))
    parser.add_argument("--numeric-scenes", type=Path, default=Path("data/metadata/numeric_scene_spec_v1.json"))
    args = parser.parse_args()
    root, original, output = args.input_root.resolve(), args.main_input_root.resolve(), args.output_root.resolve()
    metadata = Path(__file__).resolve().parents[3] / "data/metadata"
    expected_manifest = metadata / "ctcd_bundle_manifest.csv"
    source = json.loads((metadata / "ctcd_bundle_source.json").read_text(encoding="utf-8"))
    if (digest(root / "MANIFEST.csv") != digest(expected_manifest)
            or digest(expected_manifest) != source["manifest_sha256"]):
        raise ValueError("External package manifest differs from the tracked independent specification")
    verified = verify_bundle(root, root / "MANIFEST.csv")
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()

    def execute(name, command):
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8")
        (logs / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see its replay log")

    pool = root / "controller/CTCD_EXECUTABLE_PROFILES.csv.gz"
    if args.raw_root:
        destination = output / "upstream"
        execute("raw_alignment", [sys.executable, str(Path(__file__).with_name("ctcd_alignment.py")),
                                   "--input-root", str(args.raw_root.resolve()),
                                   "--ashrae-root", str(args.ashrae_root.resolve()),
                                   "--numeric-scenes", str(args.numeric_scenes.resolve()),
                                   "--output-root", str(destination)])
        if digest(destination / pool.name) != digest(pool):
            raise ValueError("Raw-data normalization does not reproduce the frozen external pool")
    contexts = json.loads((original / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    profiles = assign_external_profiles(pd.read_csv(pool), original, contexts)
    retained = pd.read_csv(root / "engineering/PROFILE_ASSIGNMENTS.csv.gz", float_precision="round_trip")
    if not profiles.equals(retained):
        raise ValueError("External draw, allocation or numerical demand coordinates changed")
    commands = generate(original, profile_tables={c: profiles[profiles.context.eq(c)] for c in contexts},
                        permissions=("native_zone",), expected_commands=144)
    frozen_commands = json.loads((root / "controller/CONTROL_COMMANDS.json").read_text(encoding="utf-8"))
    if commands != frozen_commands:
        raise ValueError("Reconstructed external controller commands differ")
    (output / "CONTROL_COMMANDS.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    if args.model_root:
        models = json.loads((args.model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
        table = pd.read_csv(root / "engineering/ANNUAL_RUN_MANIFEST.csv")
        texts = {}
        for context, model in models.items():
            path = args.model_root / model["baseline_idf"]
            if digest(path) != model["baseline_idf_sha256"] or digest(args.model_root / model["weather"]) != model["weather_sha256"]:
                raise ValueError("Locally supplied frozen IDF or weather changed")
            texts[context] = path.read_text(encoding="utf-8")
        command_keys = KEYS + ["permission", "control", "variant"]
        expected = {tuple(row[k] for k in command_keys): row for row in table.to_dict("records")}
        for command in commands:
            context = command["context"]
            row = expected[tuple(command[k] for k in command_keys)]
            text = annual_input(texts[context], command["targets_c"], models[context]["half_deadband_c"])
            if signature(text) != row["normalized_idf_sha256"]:
                raise ValueError("External physical input is not exactly reconstructed")
    physical = validate_retained(root)
    annual = output / "annual"
    execute("annual_scoring", [sys.executable, str(ENERGY_SERVICE / "analyse_annual.py"),
                               "--engineering-root", str(root / "engineering"),
                               "--factors", str(original / "factors/regional_factors.csv"),
                               "--output-root", str(annual), "--expected-designs", "144",
                               "--expected-profile-groups", "12"])
    contract = verify_contract(annual, root / "engineering")
    summarize(annual, output)
    plot_distributions(pd.read_csv(root / "controller/CTCD_ALIGNED_OFFICE_RECORDS.csv.gz"), output)
    report = {"status": "PASS", "scope": "Separate CTCD office-composition robustness comparison",
              "bundle": verified, "physical_inputs": physical, "service_contract": contract,
              "commands_regenerated": 144, "raw_alignment_replayed": bool(args.raw_root),
              "normalized_model_inputs_reconstructed": bool(args.model_root),
              "energyplus_runs_by_this_entry": 0, "physical_parameters_changed": False,
              "original_cohort_changed": False,
              "comparison": "Controls are paired within each declared external draw/context; CTCD versus main-cohort differences are source-composition descriptions, not matched-individual effects."}
    (output / "CTCD_REPLAY_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
