"""Audit the retained personal-range inputs and finite demand sensitivities.

The published elicitation framework is retained as the method assumption. This
script checks its numerical application and existing sensitivity outputs; it
does not recruit people or estimate empirical interval validity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def range_geometry(profiles: pd.DataFrame, intervals: pd.DataFrame) -> dict:
    joined = intervals.merge(profiles[["release_profile_id", "reference_g050_c"]],
                             on="release_profile_id", validate="one_to_one")
    if len(joined) != len(intervals):
        raise ValueError("A range has no matching reconstructed reference")
    low, high = joined.interval_lower_c, joined.interval_upper_c
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise ValueError("Nonfinite personal range")
    if not ((low >= 16 - 1e-10) & (high <= 30 + 1e-10) & (low < high)).all():
        raise ValueError("Invalid executable personal range")
    error = np.abs(joined.interval_centre_c - joined.reference_g050_c)
    if error.max() > 1e-10:
        raise ValueError("Personal range and Griffiths reference are misaligned")
    width = high - low
    return {
        "archive_records": len(profiles), "executable_ranges": len(joined),
        "maximum_centre_alignment_error_c": float(error.max()),
        "range_width_p05_c": float(width.quantile(.05)),
        "range_width_median_c": float(width.median()),
        "range_width_p95_c": float(width.quantile(.95)),
        "ranges_at_least_10_c_wide": int(width.ge(10).sum()),
        "lower_bound_clipped_records": int(np.isclose(low, 16).sum()),
        "upper_bound_clipped_records": int(np.isclose(high, 30).sum()),
        "reference_centres_outside_executable_domain": int(
            ((joined.interval_centre_c < 16) | (joined.interval_centre_c > 30)).sum()),
        "all_ranges_flagged_model_generated": bool(joined.virtual_model_derived.all()),
    }


def sensitivity_transitions(selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["city", "seed", "allocation", "service_fraction", "strategy", "support"]
    base = selections[selections.griffiths_coefficient.eq(.5)
                      & selections.interval_width_multiplier.eq(1)].copy()
    if base.duplicated(keys).any():
        raise ValueError("Duplicate primary comparison")
    base = base[keys + ["change_pct"]].rename(columns={"change_pct": "primary_change_pct"})
    alternatives = selections[selections.strategy.ne("uniform")].merge(
        base, on=keys, validate="many_to_one")
    alternatives["primary_matched"] = alternatives.primary_change_pct.notna()
    alternatives["variant_matched"] = alternatives.change_pct.notna()
    alternatives["both_matched"] = alternatives.primary_matched & alternatives.variant_matched

    def sign(values):
        return np.where(np.abs(values) <= 1e-6, 0, np.sign(values))

    alternatives["strict_sign_reversal"] = (
        alternatives.both_matched &
        (sign(alternatives.change_pct) * sign(alternatives.primary_change_pct) == -1))
    alternatives["direction_changed_including_ties"] = (
        alternatives.both_matched &
        (sign(alternatives.change_pct) != sign(alternatives.primary_change_pct)))
    rows = []
    for key, group in alternatives.groupby(
            ["support", "griffiths_coefficient", "interval_width_multiplier", "strategy"],
            sort=True):
        valid = group[group.both_matched]
        rows.append(dict(zip(["candidate_support", "griffiths_coefficient",
                              "interval_width_multiplier", "control"], key)) | {
            "comparison_definitions": len(group),
            "primary_matched": int(group.primary_matched.sum()),
            "variant_matched": int(group.variant_matched.sum()),
            "both_matched": len(valid),
            "primary_match_lost": int((group.primary_matched & ~group.variant_matched).sum()),
            "variant_match_gained": int((~group.primary_matched & group.variant_matched).sum()),
            "strict_sign_reversals": int(valid.strict_sign_reversal.sum()),
            "direction_changes_including_ties": int(valid.direction_changed_including_ties.sum()),
            "variant_energy_change_median_pct": float(valid.change_pct.median()),
            "variant_energy_change_min_pct": float(valid.change_pct.min()),
            "variant_energy_change_max_pct": float(valid.change_pct.max()),
        })
    return alternatives, pd.DataFrame(rows)


def fixed_reference_baselines(controller_root: Path) -> pd.DataFrame:
    """Report exact retained 24/25/26 C centres, never interpolate a missing run."""
    contexts = json.loads((controller_root / "CONTEXTS.json").read_text(encoding="utf-8"))
    rows = []
    for context, setting in contexts.items():
        with np.load(controller_root.parent / setting["probes"], allow_pickle=False) as data:
            targets, energy = data["uniform_target_c"], data["purchased_site_kwh"]
        reference = float(energy[np.isclose(targets, 25)][0])
        for centre in (24., 25., 26.):
            matches = np.flatnonzero(np.isclose(targets, centre, atol=1e-8, rtol=0))
            value = float(energy[matches[0]]) if len(matches) else np.nan
            rows.append({"context": context, "target_centre_c": centre,
                         "nominal_occupied_heating_setpoint_c": centre - setting["half_deadband_c"],
                         "nominal_occupied_cooling_setpoint_c": centre + setting["half_deadband_c"],
                         "status": "retained_qualified_probe" if len(matches) else "no_retained_qualified_probe",
                         "annual_purchased_site_kwh": value,
                         "annual_site_kwh_per_m2": value / setting["floor_area_m2"],
                         "site_change_from_common25_pct": 100 * (value / reference - 1)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo, output = args.repository_root.resolve(), args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    human = repo / "data/inputs/legacy_analysis/human"
    paths = [human / "human_profiles_minimized.csv.gz", human / "ltpref_intervals_minimized.csv.gz"]
    profiles, intervals = [pd.read_csv(path) for path in paths]
    geometry = range_geometry(profiles, intervals)
    script = repo / "src/analysis/experiments/candidate_support/reproduce_support.py"
    spec = importlib.util.spec_from_file_location("retained_candidate_sensitivity", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reproduce(repo / "src/analysis/support/candidate_support", output / "selection_replay")
    selections = pd.read_csv(output / "selection_replay/support_selections.csv")
    details, summaries = sensitivity_transitions(selections)
    details.to_csv(output / "DEMAND_SENSITIVITY_TRANSITIONS.csv", index=False)
    summaries.to_csv(output / "DEMAND_SENSITIVITY_SUMMARY.csv", index=False)
    replay = args.replay_root.resolve() if args.replay_root else repo / "output/replay"
    strict_path = replay / "strict_service/STRICT_SERVICE_SUMMARY.csv"
    strict = pd.read_csv(strict_path)
    strict.to_csv(output / "STRICT_SERVICE_DEMAND_DEFINITIONS.csv", index=False)
    controller_root = repo / "data/inputs/energy_service/controller"
    fixed_reference_baselines(controller_root).to_csv(output / "FIXED_REFERENCE_BASELINES.csv", index=False)
    provenance_paths = paths + [script, strict_path] + [
        repo / "src/analysis/support/candidate_support" / name
        for name in ("candidate_scores.csv", "service_anchors.csv", "regional_factors.csv")]
    provenance_paths += [controller_root / "CONTEXTS.json", *sorted(controller_root.glob("*_probes.npz"))]
    report = {
        "analysis": "retained_personal_range_method_and_sensitivity",
        "method_doi": "10.1016/j.buildenv.2026.114586",
        "method_status": "accepted elicitation framework; existing numerical adaptation retained",
        "new_empirical_validation": False, "new_physical_runs": 0,
        "geometry": geometry, "selection_rows": len(selections),
        "control_definitions": {
            "uniform_average": "demand-derived building-average target, not a permanently fixed 25 C command",
            "common25": "shared reference and fallback centre; prototype nominal occupied heating/cooling separation retained",
            "fixed24_25_26": "exact retained qualified probes only; missing New Delhi 24 C is not interpolated",
            "policy_ranges": "archive-demand comparison; not all policy statements were simulated as thermostat rules",
            "thermostat": "native air-temperature thermostat, finite equipment capacity and original availability/setback schedules",
            "centre_mapping": "study control rule translates nominal occupied plateaus together; not a personal perceptual threshold",
            "setpoint_table_scope": "nominal occupied plateaus only; preconditioning, transition, weekend and setback schedules remain distinct",
            "service_temperature": "operative-temperature scoring is distinct from the thermostat's air-temperature signal",
        },
        "input_sha256": {p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.name:
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in provenance_paths},
        "interpretation": "Sensitivity to demand definitions, not empirical tolerance validation. "
                          "Repeated candidate evaluations are not independent people or buildings.",
        "outputs": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(output.rglob("*.csv"))},
    }
    (output / "RANGE_METHOD_AUDIT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    main()
