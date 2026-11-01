"""Compare aligned retained offset grids; no new physical operations."""

from common import *
import common
import analyse_global_results as scoring
import numpy as np
import pandas as pd


LEVELS = {"half": 0.5, "quarter": 0.25, "eighth": 0.125}
EXPECTED_SETTINGS = 54
MEAN_COLUMNS = ["zone_mean_mean_reference_mismatch_c", "zone_mean_fixed_tail_reference_mismatch_c"]
SPATIAL_SOURCES = {"new_spatial_run", "reused_annual_baseline", "reused_uniform_physical_run"}
DERIVED_COLUMNS = ["climate", "composition", "physical_plan_id", "hourly_path", *MEAN_COLUMNS]


def source_hashes():
    return {Path(p).relative_to(CODE).as_posix(): digest(p)
            for p in [__file__, common.__file__, scoring.__file__]}


def verified_cache(out):
    audit_path = out / "AUDIT.json"
    if not audit_path.is_file():
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Preserve the incomplete aligned-grid attempt before retrying")
        return False
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "completed" or audit["source_sha256"] != source_hashes():
        raise ValueError("Aligned-grid cache source changed or execution is incomplete")
    inputs = json.loads((out / "INPUT_SHA256.json").read_text())
    for name, expected in inputs.items():
        path = (CODE / name).resolve()
        if not path.is_relative_to(CODE) or not path.is_file() or digest(path) != expected:
            raise ValueError("Aligned-grid cache input changed: " + name)
    for name, expected in audit["output_sha256"].items():
        if not (out / name).is_file() or digest(out / name) != expected:
            raise ValueError("Aligned-grid cache output changed: " + name)
    return True


def level_operations(group, step):
    offsets = pd.to_numeric(group.offset_c, errors="raise")
    constant = group.control.eq("uniform_command_search") | group.source_kind.isin(SPATIAL_SOURCES)
    aligned = offsets.between(-1 - 1e-8, 1 + 1e-8) & np.isclose(
        offsets / step, np.round(offsets / step), rtol=0, atol=1e-8)
    # Filter aliases before choosing the canonical physical operation.
    return group[constant | aligned].sort_values(
        ["run_id", "control", "offset_c"], kind="stable").drop_duplicates("physical_plan_id").copy()


def prepare_levels(group, score_missing):
    levels = {name: level_operations(group, step) for name, step in LEVELS.items()}
    missing = {}
    for frame in levels.values():
        for _, row in frame[frame[MEAN_COLUMNS].isna().any(axis=1)].iterrows():
            missing.setdefault(row.physical_plan_id, row)
    derived = []
    for physical, row in missing.items():
        scores = score_missing(row)
        values = {"zone_mean_" + key: scores[key]
                  for key in ["mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"]}
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Explicit zone-mean score is nonfinite")
        for frame in levels.values():
            aliases = frame.physical_plan_id.eq(physical)
            for column, value in values.items():
                stored = frame.loc[aliases, column].dropna()
                if not np.allclose(stored, value, rtol=0, atol=1e-8):
                    raise ValueError("Stored and explicit zone-mean scores disagree")
                frame.loc[aliases & frame[column].isna(), column] = value
        derived.append({key: row[key] for key in DERIVED_COLUMNS[:4]} | values)
    return levels, derived


def evaluate_levels(levels, population, tail):
    rows, changes, previous, previous_ids = [], [], None, set()
    for level, frame in levels.items():
        ids = set(frame.physical_plan_id)
        if not previous_ids <= ids:
            raise ValueError("Aligned physical libraries are not nested")
        previous_ids = ids
        masks = {}
        row = {"level": level, "offset_step_c": LEVELS[level], "candidate_operations": len(frame),
               "population_budget_c": population, "fixed_tail_budget_c": tail}
        for prefix, score_prefix in [("full", ""), ("zone_mean", "zone_mean_")]:
            masks[prefix] = (frame.engineering_pass.eq(True)
                & frame[score_prefix + "mean_reference_mismatch_c"].le(population + 1e-9)
                & frame[score_prefix + "fixed_tail_reference_mismatch_c"].le(tail + 1e-9))
            qualified = frame[masks[prefix]]
            chosen = scoring.choose_minimum_energy(qualified)
            row[prefix + "_qualified_operations"] = len(qualified)
            row[prefix + "_status"] = "selected" if chosen is not None else "infeasible"
            row[prefix + "_choice"] = chosen["physical_plan_id"] if chosen else None
            row[prefix + "_finite_minimum_energy_kwh"] = float(qualified.facility_purchased_kwh.min()) if chosen else None
            for field, column in [("energy_kwh", "facility_purchased_kwh"), ("control", "control"),
                                  ("permission", "permission"), ("offset_c", "offset_c"),
                                  ("full_population_mismatch_c", "mean_reference_mismatch_c"),
                                  ("full_fixed_tail_mismatch_c", "fixed_tail_reference_mismatch_c")]:
                row[prefix + "_selected_" + field] = chosen[column] if chosen else None
            row[prefix + "_selected_full_service_excess_c"] = max(
                0.0, chosen["mean_reference_mismatch_c"] - population,
                chosen["fixed_tail_reference_mismatch_c"] - tail) if chosen else None
        full, mean = masks["full"], masks["zone_mean"]
        row.update(qualified_by_both=int((full & mean).sum()), qualified_only_by_full=int((full & ~mean).sum()),
                   qualified_only_by_mean=int((~full & mean).sum()),
                   qualified_operation_sets_differ=bool((full != mean).any()))
        both = row["full_choice"] is not None and row["zone_mean_choice"] is not None
        row["choices_differ"] = row["full_choice"] != row["zone_mean_choice"] if both else None
        row["energy_difference_kwh"] = row["full_selected_energy_kwh"] - row["zone_mean_selected_energy_kwh"] if both else None
        rows.append(row)
        if previous is not None:
            change = {"from_level": previous["level"], "to_level": level}
            for prefix in ["full", "zone_mean"]:
                old_ok = previous[prefix + "_choice"] is not None
                new_ok = row[prefix + "_choice"] is not None
                paired = old_ok and new_ok
                if old_ok and (not new_ok or row[prefix + "_finite_minimum_energy_kwh"] >
                               previous[prefix + "_finite_minimum_energy_kwh"] + 1e-6):
                    raise ValueError("Exact finite-library minimum increased in a nested grid")
                delta = row[prefix + "_selected_energy_kwh"] - previous[prefix + "_selected_energy_kwh"] if paired else None
                change[prefix + "_feasibility_changed"] = old_ok != new_ok
                change[prefix + "_physical_choice_changed"] = row[prefix + "_choice"] != previous[prefix + "_choice"] if paired else None
                change[prefix + "_selected_energy_change_kwh"] = delta
                change[prefix + "_selected_energy_change_pct"] = 100 * delta / previous[prefix + "_selected_energy_kwh"] if paired and previous[prefix + "_selected_energy_kwh"] != 0 else None
                change[prefix + "_selected_energies_tied"] = abs(delta) <= max(
                    1e-6, 1e-9 * abs(min(row[prefix + "_selected_energy_kwh"], previous[prefix + "_selected_energy_kwh"]))) if paired else None
                change[prefix + "_full_service_excess_change_c"] = row[prefix + "_selected_full_service_excess_c"] - previous[prefix + "_selected_full_service_excess_c"] if paired else None
            changes.append(change)
        previous = row
    return rows, changes


def validate_fine(choices, expected):
    fine = choices[choices.level.eq("eighth")].set_index(["climate", "composition"])
    expected = expected.set_index(["climate", "composition"])
    if fine.index.has_duplicates or expected.index.has_duplicates or set(fine.index) != set(expected.index):
        raise ValueError("Fine nominal settings differ")
    for key, row in expected.iterrows():
        for column, value in row.items():
            actual = fine.loc[key, column]
            if pd.isna(value) and pd.isna(actual):
                continue
            tolerance = 1e-6 if column == "energy_difference_kwh" else 1e-8
            equal = np.isclose(actual, value, rtol=0, atol=tolerance) if isinstance(value, (float, np.floating)) else actual == value
            if not equal:
                raise ValueError(f"Fine nominal reproduction differs: {key} {column}")


def main():
    out = ROOT / "results/aligned_grid_sensitivity"
    if verified_cache(out):
        print("Aligned grids: verified completed cache", flush=True)
        return
    paths = {"library": ROOT / "results/budget_frontiers/ALL_OPERATION_LIBRARY.csv",
             "anchors": ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv",
             "profiles": MATRIX / "GLOBAL_PROFILES.csv",
             "tails": ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
             "nominal": ROOT / "results/nominal_information/NOMINAL_QUALIFICATION_COUNTS.csv",
             "nominal_audit": ROOT / "results/nominal_information/AUDIT.json",
             "nominal_inputs": ROOT / "results/nominal_information/INPUT_SHA256.json",
             "protocol": CODE / "data/metadata/strengthening_protocol/analysis_plan_amendment_31_aligned_grid_sensitivity.yaml"}
    inputs = {p.relative_to(CODE).as_posix(): digest(p) for p in paths.values()}
    sources = source_hashes()
    audit = json.loads(paths["nominal_audit"].read_text())
    nominal_inputs = json.loads(paths["nominal_inputs"].read_text())
    if audit["status"] != "completed" or audit["output_sha256"] != digest(paths["nominal"]):
        raise ValueError("Complete and verify the public nominal qualifications first")
    for key in ["library", "profiles", "tails"]:
        name = paths[key].relative_to(CODE).as_posix()
        if nominal_inputs.get(name) != inputs[name]:
            raise ValueError("Aligned grid and nominal qualification inputs differ: " + key)
    library, anchors, members, expected = [pd.read_csv(paths[key]) for key in ["library", "anchors", "tails", "nominal"]]
    profiles = scoring.normalise_profiles(pd.read_csv(paths["profiles"]))
    keys = ["climate", "composition"]
    settings = set(map(tuple, library[keys].drop_duplicates().to_numpy()))
    if len(settings) != EXPECTED_SETTINGS or settings != set(map(tuple, expected[keys].to_numpy())) or not library.layout.eq("random_mixing").all():
        raise ValueError("Aligned grid requires every frozen mixed-layout setting")
    rows, changes, derived = [], [], []
    for identity, group in library.groupby(keys, sort=True):
        climate, composition = identity
        if group.loc[group.control.eq("uniform_command_search"), "offset_c"].nunique() != 31:
            raise ValueError("Every grid must retain all31 uniform-command targets")
        demand = profiles[profiles.climate.eq(climate) & profiles.composition.eq(composition)].sort_values("profile_id").reset_index(drop=True)
        ids = set(members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id)
        tail = np.flatnonzero(demand.profile_id.isin(ids))
        if len(demand) != 267 or len(ids) != 27 or len(tail) != 27:
            raise ValueError("Demand profiles or original fixed tail changed")
        means = scoring.zone_mean_profiles(demand)

        def direct(row):
            path = Path(row.hourly_path).resolve()
            if not path.is_relative_to(CODE):
                raise ValueError("Aligned-grid trajectory is outside this project")
            observed = digest(path)
            if observed != row.hourly_sha256:
                raise ValueError("Aligned-grid trajectory differs from its scored library hash")
            inputs[path.relative_to(CODE).as_posix()] = observed
            return scoring.score_trajectory(scoring.load_hourly(path), means, tail)["summary"]

        levels, added = prepare_levels(group, direct)
        reference = anchors[anchors.climate.eq(climate) & anchors.composition.eq(composition)]
        if len(reference) != 1:
            raise ValueError("A setting must have exactly one full-demand25C anchor")
        result, paired = evaluate_levels(levels, float(reference.iloc[0].anchor_mean_mismatch_c), float(reference.iloc[0].anchor_tail_mismatch_c))
        rows.extend({"climate": climate, "composition": composition} | row for row in result)
        changes.extend({"climate": climate, "composition": composition} | row for row in paired)
        derived.extend(added)
        print("ALIGNED_GRID", climate, composition, flush=True)
    choices, transitions = pd.DataFrame(rows), pd.DataFrame(changes)
    validate_fine(choices, expected)
    summaries = []
    for level in LEVELS:
        frame = choices[choices.level.eq(level)]
        summaries.append({"level": level, "settings": len(frame),
            "qualified_set_changes": int(frame.qualified_operation_sets_differ.sum()),
            "choice_changes": int(frame.choices_differ.fillna(False).sum()),
            "paired_choices": int((frame.full_status.eq("selected") & frame.zone_mean_status.eq("selected")).sum()),
            "full_infeasible_settings": int(frame.full_status.eq("infeasible").sum()),
            "zone_mean_infeasible_settings": int(frame.zone_mean_status.eq("infeasible").sum()),
            "mean_selected_full_service_exceedances": int(frame.zone_mean_selected_full_service_excess_c.gt(1e-9).sum()),
            "mean_selected_full_service_excess_max_c": frame.zone_mean_selected_full_service_excess_c.max(),
            "selected_energy_difference_median_kwh": frame.energy_difference_kwh.median()})
    if len(choices) != EXPECTED_SETTINGS * 3 or len(transitions) != EXPECTED_SETTINGS * 2:
        raise ValueError("Aligned-grid design is incomplete")
    if source_hashes() != sources or any(digest(CODE / name) != value for name, value in inputs.items()):
        raise ValueError("Aligned-grid source or input changed during execution")
    out.mkdir(parents=True, exist_ok=True)
    outputs = {"ALIGNED_GRID_CHOICES.csv": choices, "ALIGNED_GRID_LEVEL_CHANGES.csv": transitions,
               "ALIGNED_GRID_SUMMARY.csv": pd.DataFrame(summaries),
               "DERIVED_ZONE_MEAN_SCORES.csv": pd.DataFrame(derived, columns=DERIVED_COLUMNS)}
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    write_json(out / "INPUT_SHA256.json", inputs)
    write_json(out / "AUDIT.json", {"status": "completed", "settings": EXPECTED_SETTINGS,
        "levels": LEVELS, "setting_level_rows": len(choices), "paired_level_rows": len(transitions),
        "explicit_missing_mean_scores": len(derived), "fine_nominal_reproduction": "passed for every setting and public field",
        "exact_finite_minimum_monotonicity": "passed within1e-6kWh numerical tolerance for both representations",
        "EnergyPlus_runs": 0, "source_sha256": sources,
        "output_sha256": {name: digest(out / name) for name in [*outputs, "INPUT_SHA256.json"]},
        "interpretation": "Aligned completed finite libraries; fine outcomes already known; not continuous convergence or a global optimum"})


if __name__ == "__main__":
    main()
