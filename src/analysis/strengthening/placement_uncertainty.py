"""Evaluate frozen constrained arrangements under paired demand shifts."""

from common import *
from paired_uncertainty import keyed_normal, shared_normal, empirical_temporal_shifts, DRAWS
from profile_identity import person_keys
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles, load_hourly
import numpy as np
import pandas as pd


def hvac_operations(root, single_record=False):
    keys = ["climate", "composition_id", "control", "layout"]
    vav = pd.read_csv(root / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv").rename(
        columns={"layout_arranged": "layout"})
    frames = {"original_vav": vav}
    for system in ["independent_pthp", "vav_coordinated_outdoor_reset"]:
        frames[system] = pd.read_csv(root / f"results/placement_hvac_interaction/{system}_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv").merge(
            vav[keys + ["composition", "moved_profiles"]], on=keys, validate="one_to_one")
    operations, baselines, denominators = [], [], []
    for system, frame in frames.items():
        if frame.duplicated(keys).any() or len(frame) != 324:
            raise ValueError("Incomplete or duplicated planned HVAC placement settings")
        frame = frame.copy()
        frame["system"] = system
        valid = frame.engineering_pass_arranged.eq(True) & frame.engineering_pass_mixed.eq(True)
        for (control, layout), planned in frame.groupby(["control", "layout"]):
            if len(planned) != 54:
                raise ValueError("Each HVAC/control/relocation budget must have 54 planned settings")
            available = int(valid.loc[planned.index].sum())
            eligible = planned.composition.str.contains("Multinational") if single_record else pd.Series(True, index=planned.index)
            denominators.append({"system": system, "control": control, "layout": layout,
                                 "planned_settings": 54, "available_settings": available,
                                 "unavailable_settings": 54 - available,
                                 "scenario_eligible_planned_settings": int(eligible.sum()),
                                 "scenario_eligible_available_settings": int((eligible & valid.loc[planned.index]).sum())})
        for _, mixed in frame.groupby(keys[:3]):
            columns = ["hourly_path_mixed", "facility_purchased_kwh_mixed", "engineering_pass_mixed"]
            columns += [column for column in ["hourly_sha256_mixed", "hourly_sha256"] if column in mixed]
            for column in columns:
                if mixed[column].nunique(dropna=False) != 1:
                    raise ValueError("Conflicting repeated mixed baseline: " + column)
        mixed = frame.drop_duplicates(keys[:3])
        for control, planned in mixed.groupby("control"):
            eligible = planned.composition.str.contains("Multinational") if single_record else pd.Series(True, index=planned.index)
            available = planned.engineering_pass_mixed.eq(True)
            denominators.append({"system": system, "control": control, "layout": "within_floor_relocation_0pct",
                                 "planned_settings": len(planned), "available_settings": int(available.sum()),
                                 "unavailable_settings": int((~available).sum()),
                                 "scenario_eligible_planned_settings": int(eligible.sum()),
                                 "scenario_eligible_available_settings": int((eligible & available).sum())})
        mixed = mixed[mixed.engineering_pass_mixed.eq(True)]
        for suffix, source, destination in [("mixed", mixed, baselines), ("arranged", frame.loc[valid], operations)]:
            selected = source[["system", "climate", "composition", "control", "layout", "moved_profiles",
                               "hourly_path_" + suffix, "facility_purchased_kwh_" + suffix,
                               "population_gain_c", "fixed_tail_gain_c", "energy_change_pct"]].copy()
            selected = selected.rename(columns={"hourly_path_" + suffix: "hourly_path",
                                                "facility_purchased_kwh_" + suffix: "facility_purchased_kwh"})
            destination.append(selected)
    return pd.concat(operations, ignore_index=True), pd.concat(baselines, ignore_index=True), pd.DataFrame(denominators)


def matched_hvac_draws(frame):
    keys = ["climate", "composition", "control", "layout", "scenario", "draw"]
    endpoints = ["population_gain_c", "fixed_tail_gain_c"]
    base = frame[frame.system.eq("original_vav")][keys + endpoints]
    rows = []
    for system, alternative in frame[~frame.system.eq("original_vav")].groupby("system"):
        paired = base.merge(alternative[keys + endpoints], on=keys, validate="one_to_one",
                            suffixes=("_vav", "_alternative"))
        if len(paired) != len(alternative):
            raise ValueError("Alternative HVAC draws have no identical qualified VAV pairing")
        for identity, group in paired.groupby(keys[:-1]):
            row = dict(zip(keys[:-1], identity)) | {"system": system, "paired_draws": len(group)}
            if sorted(group.draw) != list(range(DRAWS)):
                raise ValueError("Incomplete paired HVAC draw indices")
            for endpoint in endpoints:
                difference = group[endpoint + "_alternative"] - group[endpoint + "_vav"]
                row[endpoint + "_paired_system_difference_median"] = difference.median()
                row[endpoint + "_paired_system_difference_p05"] = difference.quantile(0.05)
                row[endpoint + "_paired_system_difference_p95"] = difference.quantile(0.95)
                row[endpoint + "_conditional_fraction_greater_than_vav"] = difference.gt(1e-9).mean()
            rows.append(row)
    return rows


def same_physical_calendar(a, b):
    for field in ["occupied", "zone_ids", "month", "day", "hour", "dst"]:
        if not np.array_equal(a[field], b[field]):
            raise ValueError("HVAC placement calendar or zone changed: " + field)
    if not np.allclose(a["people"], b["people"], atol=1e-9, rtol=0):
        raise ValueError("HVAC placement native occupancy changed")


def contained_hourly(path, inputs):
    path = Path(path).resolve()
    if not path.is_relative_to(CODE):
        raise ValueError("HVAC uncertainty input is outside this project")
    actual = digest(path)
    if str(path) in inputs and inputs[str(path)] != actual:
        raise ValueError("HVAC uncertainty trajectory changed during execution")
    inputs[str(path)] = actual
    return load_hourly(path)


def executed_sources():
    return {"placement_uncertainty": Path(__file__).resolve()} | {
        name: Path(__import__(name).__file__).resolve() for name in
        ["paired_uncertainty", "profile_identity", "service_distribution", "analyse_global_results"]}


def verified_hvac_cache(out):
    if not (out / "AUDIT.json").is_file():
        return False
    report = json.loads((out / "AUDIT.json").read_text())
    if report.get("status") != "completed":
        return False
    for path, expected in json.loads((out / "INPUT_SHA256.json").read_text()).items():
        if digest(CODE / path) != expected:
            raise ValueError("Completed HVAC uncertainty input changed: " + path)
    for name, expected in report["executed_source_sha256"].items():
        if digest(executed_sources()[name]) != expected:
            raise ValueError("Completed HVAC uncertainty code differs: " + name)
    for name, expected in report["output_sha256"].items():
        if digest(out / name) != expected:
            raise ValueError("Completed HVAC uncertainty result changed: " + name)
    print("Verified completed contained HVAC uncertainty", out, flush=True)
    return True


def main():
    single_record = "--single-record" in __import__("sys").argv
    study_shared = "--study-shared" in __import__("sys").argv
    hvac = "--hvac" in __import__("sys").argv
    assert not (single_record and study_shared)
    if hvac and not (single_record or study_shared):
        raise ValueError("HVAC propagation requires a declared study-shared or single-record scenario")
    out = ROOT / ("results/placement_uncertainty_single_record_identity" if single_record else "results/placement_uncertainty_identity")
    if study_shared:
        out = ROOT / "results/placement_uncertainty_study_shared_identity"
    if hvac:
        out = out.with_name(out.name + "_hvac")
    out.mkdir(exist_ok=True)
    if hvac and verified_hvac_cache(out):
        return
    if hvac and any(out.iterdir()):
        raise FileExistsError("Preserve incomplete HVAC uncertainty output before rerunning")
    inputs = {}
    if hvac:
        paths = [MATRIX / "GLOBAL_PROFILES.csv", ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv",
                 ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
                 ROOT / "audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz",
                 ROOT / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv",
                 *sorted((ROOT / "results/placement_hvac_interaction").glob("*_WITHIN_SYSTEM_PLACEMENT_EFFECTS.csv")),
                 *executed_sources().values()]
        if single_record:
            paths.append(ROOT / "results/demand_validation/single_record/timed/CONTINUOUS_PROXY_ERRORS.csv")
        for number, suffix in [(25, "shared_error_groups"), (26, "placement_hvac_uncertainty")]:
            name = suffix + ".yaml"
            path = CODE / "data/metadata/strengthening_protocol" / name
            if not path.is_file():
                path = Path(__file__).resolve().parent / name
            paths.append(path)
        for path in paths:
            if not path.resolve().is_relative_to(CODE):
                raise ValueError("HVAC uncertainty input or executed method is outside this project")
            inputs[str(path.resolve())] = digest(path)
    metrics = pd.read_csv(ROOT / "results/PLACEMENT_METRICS.csv")
    original = pd.read_csv(ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv")
    original = original[
        original.offset_c.eq(0) & original.control.isin(["personal_center", "personal_range"])
    ]
    if hvac:
        metrics, original, denominators = hvac_operations(ROOT, single_record)
        denominators.to_csv(out / "PLANNED_AVAILABLE_SETTINGS.csv", index=False)
    else:
        original["system"] = "original_vav"
        metrics["system"] = "original_vav"
    native = normalise_profiles(pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv"))
    arranged = normalise_profiles(pd.read_csv(ROOT / "config/CONSTRAINED_PLACEMENT_PROFILES.csv"))
    members = pd.read_csv(ROOT / "results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    summaries = []
    drawrows = []
    cross_system = []
    nominal_errors = []
    draw_path = out / "PAIRED_CONSTRAINED_PLACEMENT_DRAWS.csv.gz"
    wrote_draws = False
    for (climate, composition), p in native.groupby(["climate", "composition"], sort=True):
        p = p.sort_values("profile_id").reset_index(drop=True)
        ids = set(
            members[members.climate.eq(climate) & members.composition.eq(composition)].profile_id
        )
        tail = np.flatnonzero(p.profile_id.isin(ids))
        assert len(tail) == 27
        nominal = p.centre_c.to_numpy()
        source = "ASHRAE" if "Multinational" in composition else "CTCD"
        if single_record and source != "ASHRAE":
            continue
        shared = shared_normal(p.profile_id, source, study_shared)
        individual = np.column_stack(
            [keyed_normal("person|" + str(x)) for x in person_keys(p.profile_id)]
        )
        scenarios = [
            (f"{'studyGroupSD' if study_shared else 'sourceSD'}{s:g}_personSD{i:g}", nominal[None, :] + s * shared + i * individual)
            for s in [0, 0.25, 0.5]
            for i in [0, 0.25, 0.5]
        ]
        if single_record:
            scenarios = []
        if source == "ASHRAE" and not study_shared:
            shift = empirical_temporal_shifts(p.profile_id, single_record)
            scenarios += [
                (
                    f"observed_temporal_proxy_shift_fraction{fraction:g}",
                    nominal[None, :] + fraction * shift,
                )
                for fraction in [0.5, 1.0]
            ]
        physical_calendar = None
        for (system, control), baselines in original.groupby(["system", "control"], sort=True):
            baseline = baselines[
                baselines.climate.eq(climate)
                & baselines.composition.eq(composition)
            ].iloc[0]
            baseline_data = contained_hourly(baseline.hourly_path, inputs) if hvac else load_hourly(Path(baseline.hourly_path))
            if hvac and physical_calendar is not None:
                same_physical_calendar(physical_calendar, baseline_data)
            physical_calendar = baseline_data
            curve = ServiceDistribution(baseline_data, p)
            candidates = metrics[
                metrics.climate.eq(climate)
                & metrics.composition.eq(composition)
                & metrics.control.eq(control)
                & metrics.system.eq(system)
            ]
            zero = {
                **baseline.to_dict(),
                "layout": "within_floor_relocation_0pct",
                "moved_profiles": 0,
                **({"population_gain_c": 0., "fixed_tail_gain_c": 0., "energy_change_pct": 0.} if hvac else {}),
            }
            candidates = pd.concat([pd.DataFrame([zero]), candidates], ignore_index=True)
            for row in candidates.itertuples():
                group = (
                    arranged[
                        arranged.climate.eq(climate)
                        & arranged.composition.eq(composition)
                        & arranged.layout.eq(row.layout)
                    ]
                    .sort_values("profile_id")
                    .reset_index(drop=True)
                )
                assert np.array_equal(group.profile_id, p.profile_id)
                if hvac:
                    for column in ["centre_c", "lower_c", "upper_c"]:
                        np.testing.assert_allclose(group[column], p[column], atol=1e-12, rtol=0)
                    if not group.zone_id.value_counts().sort_index().equals(p.zone_id.value_counts().sort_index()):
                        raise ValueError("HVAC placement zone capacities changed")
                alternative_data = contained_hourly(row.hourly_path, inputs) if hvac else load_hourly(Path(row.hourly_path))
                if hvac:
                    same_physical_calendar(baseline_data, alternative_data)
                alternative = ServiceDistribution(alternative_data, group)
                if hvac:
                    actual = np.r_[np.asarray(curve.score(nominal, tail)) - np.asarray(alternative.score(nominal, tail)),
                                   100 * (row.facility_purchased_kwh / baseline.facility_purchased_kwh - 1)]
                    expected = [row.population_gain_c, row.fixed_tail_gain_c, row.energy_change_pct]
                    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
                    nominal_errors.append(float(np.max(np.abs(actual - expected))))
                for label, centers in scenarios:
                    a, b = curve.score(centers, tail)
                    c, d = alternative.score(centers, tail)
                    population = a - c
                    least = b - d
                    summaries.append(
                        {
                            **({"system": system} if hvac else {}),
                            "climate": climate,
                            "composition": composition,
                            "control": control,
                            "layout": row.layout,
                            "moved_profiles": row.moved_profiles,
                            "scenario": label,
                            "population_gain_median_c": float(np.median(population)),
                            "population_gain_p05_c": float(np.quantile(population, 0.05)),
                            "population_gain_p95_c": float(np.quantile(population, 0.95)),
                            "fixed_tail_gain_median_c": float(np.median(least)),
                            "fixed_tail_gain_p05_c": float(np.quantile(least, 0.05)),
                            "fixed_tail_gain_p95_c": float(np.quantile(least, 0.95)),
                            "conditional_draw_fraction_population_gain": float(
                                (population > 1e-9).mean()
                            ),
                            "conditional_draw_fraction_tail_gain": float((least > 1e-9).mean()),
                            **({"conditional_draw_fraction_both_service_gains": float(((population > 1e-9) & (least > 1e-9)).mean())} if hvac else {}),
                            "energy_change_pct_fixed_operations": 100
                            * (row.facility_purchased_kwh / baseline.facility_purchased_kwh - 1),
                        }
                    )
                    for draw in range(DRAWS):
                        drawrows.append(
                            {
                                **({"system": system} if hvac else {}),
                                "climate": climate,
                                "composition": composition,
                                "control": control,
                                "layout": row.layout,
                                "scenario": label,
                                "draw": draw,
                                "population_gain_c": population[draw],
                                "fixed_tail_gain_c": least[draw],
                            }
                        )
        if hvac:
            draws = pd.DataFrame(drawrows)
            cross_system.extend(matched_hvac_draws(draws))
            draws.to_csv(draw_path, mode="a" if wrote_draws else "w", header=not wrote_draws, index=False)
            wrote_draws = True
            drawrows.clear()
        print("PLACEMENT_UNCERTAINTY", climate, composition, flush=True)
    pd.DataFrame(summaries).to_csv(out / "PAIRED_CONSTRAINED_PLACEMENT_SUMMARY.csv", index=False)
    if hvac:
        pd.DataFrame(cross_system).to_csv(out / "MATCHED_CROSS_SYSTEM_DRAW_DIFFERENCES.csv", index=False)
        expected_summaries = int(denominators.scenario_eligible_available_settings.sum()) * (2 if single_record else 9)
        if len(summaries) != expected_summaries:
            raise ValueError("HVAC uncertainty did not cover every available declared setting/scenario")
        for path, expected in inputs.items():
            if digest(Path(path)) != expected:
                raise ValueError("HVAC uncertainty input changed during analysis: " + path)
        write_json(out / "INPUT_SHA256.json", {str(Path(path).relative_to(CODE)): value for path, value in inputs.items()})
    else:
        pd.DataFrame(drawrows).to_csv(draw_path, index=False)
    report = {
        "status": "completed",
        "comparisons": len(summaries),
        "draws_per_scenario": DRAWS,
        "hardware_scope": "three fixed HVAC architectures" if hvac else "original VAV",
        **({"planned_setting_operation_pairs": int(denominators.planned_settings.sum()),
            "available_setting_operation_pairs": int(denominators.available_settings.sum()),
            "scenario_eligible_setting_operation_pairs": int(denominators.scenario_eligible_available_settings.sum()),
            "maximum_nominal_reproduction_error": max(nominal_errors),
            "paired_cross_system_summaries": len(cross_system),
            "input_file_count": len(inputs),
            "executed_source_sha256": {name: digest(path) for name, path in executed_sources().items()},
            "output_sha256": {path.name: digest(path) for path in [*out.glob("*.csv*"), out / "INPUT_SHA256.json"]},
            "mechanism_scope": "Total fixed-trajectory placement gains; intermediate-state decomposition uncertainty was not recomputed"} if hvac else {}),
        "shared_error_grouping": "empirical source-pool temporal-proxy transport" if single_record else ("study/contributor" if study_shared else "archive"),
        "empirical_calibration_records_per_person": 1 if single_record else (None if study_shared else "multiple early records for the separate empirical scenarios"),
        "interpretation": "Fixed commands and training arrangements,rescored against paired demand shifts. Fractions are scenario-conditional,not confidence in actual relocation benefits. Empirical temporal proxy stress applies only to ASHRAE-derived compositions.",
    }
    write_json(out / "AUDIT.json", report)
    task = "placement_paired_uncertainty_study_shared" if study_shared else (
        "placement_paired_uncertainty_single_record" if single_record else "placement_paired_uncertainty")
    if hvac:
        task += "_hvac"
    record_status(task, "completed", report)


if __name__ == "__main__":
    main()
