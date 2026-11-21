"""Paired service-limit sensitivity for the final pooled operation library.

Retained candidate scores are completed from hashed annual trajectories. Both
demand representations use the same candidates, fixed reference group and limits.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "src/analysis/global_extension"))
from analyse_global_results import fixed_tail_from_anchor, load_hourly, normalise_profiles
from paired_uncertainty import choose
from service_distribution import ServiceDistribution

FACTORS = (0.50, 0.75, 0.90, 1.00, 1.05, 1.10)
TOLERANCE = 1e-9
KEYS = ["climate", "composition"]
SCORE_COLUMNS = [
    "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
    "zone_mean_mean_reference_mismatch_c", "zone_mean_fixed_tail_reference_mismatch_c",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_path(path):
    path = Path(path)
    return path.relative_to(CODE).as_posix() if path.is_relative_to(CODE) else str(path)


def score_representations(data, profiles, tail):
    centers = profiles.centre_c.to_numpy(float)
    means = profiles.groupby("zone_id", sort=False).centre_c.transform("mean").to_numpy()
    curve = ServiceDistribution(data, profiles)
    return np.array([*curve.score(centers, tail), *curve.score(means, tail)])


class Trajectories:
    def __init__(self, root):
        self.paths = defaultdict(list)
        for path in Path(root).rglob("*.npz"):
            self.paths[path.name].append(path)
        self.hashes = {}
        self.used = {}

    def load(self, declared_path, expected_hash):
        paths = self.paths[Path(declared_path).name]
        for path in sorted(paths):
            if path not in self.hashes:
                self.hashes[path] = digest(path)
            if self.hashes[path] == expected_hash:
                self.used[str(path)] = expected_hash
                return load_hourly(path)
        raise FileNotFoundError(f"No retained trajectory matches {declared_path} and {expected_hash}")


def complete_scores(library, profiles, memberships, references, anchors, trajectories):
    rows, reference_rows = [], []
    recomputed = 0
    for key, raw in library.groupby(KEYS, sort=True):
        climate, composition = key
        group = profiles[profiles.climate.eq(climate) & profiles.composition.eq(composition)]
        group = group.sort_values("profile_id").reset_index(drop=True)
        fixed = memberships[memberships.climate.eq(climate) & memberships.composition.eq(composition)]
        tail_ids = set(fixed.profile_id.astype(str))
        tail = np.flatnonzero(group.profile_id.astype(str).isin(tail_ids))
        if len(group) != 267 or len(tail) != 27 or len(tail_ids) != 27:
            raise ValueError(f"Profile or fixed-group support differs for {key}")
        anchor = references.loc[climate]
        reference = trajectories.load(anchor.hourly_path, anchor.hourly_sha256)
        reference_tail, _ = fixed_tail_from_anchor(reference, group)
        np.testing.assert_array_equal(tail, reference_tail)
        reference_scores = score_representations(reference, group, tail)
        declared = anchors[(anchors.climate.eq(climate)) & anchors.composition.eq(composition)].iloc[0]
        np.testing.assert_allclose(reference_scores[:2],
                                   [declared.anchor_mean_mismatch_c, declared.anchor_tail_mismatch_c],
                                   rtol=0, atol=1e-9)
        reference_rows.append(dict(climate=climate, composition=composition,
                                   reference_population_c=reference_scores[0],
                                   reference_fixed_group_c=reference_scores[1],
                                   fixed_profile_ids=";".join(sorted(tail_ids))))
        ordered = raw.sort_values(["run_id", "control", "offset_c"], kind="mergesort")
        for _, aliases in ordered.groupby("physical_plan_id", sort=False):
            row = aliases.iloc[0].copy()
            for column in [*SCORE_COLUMNS, "facility_purchased_kwh"]:
                values = aliases[column].dropna().to_numpy(float)
                if len(values):
                    np.testing.assert_allclose(values, values[0], rtol=0,
                                               atol=1e-6 if column.endswith("kwh") else 1e-8)
                    row[column] = values[0]
            row["score_source"] = "retained_candidate_scores"
            if row[SCORE_COLUMNS].isna().any():
                data = trajectories.load(row.hourly_path, row.hourly_sha256)
                np.testing.assert_array_equal(data["zone_ids"], reference["zone_ids"])
                np.testing.assert_array_equal(data["occupied"], reference["occupied"])
                np.testing.assert_allclose(data["people"], reference["people"], rtol=0, atol=1e-9)
                scores = score_representations(data, group, tail)
                np.testing.assert_allclose(scores[:2], row[SCORE_COLUMNS[:2]].to_numpy(float),
                                           rtol=0, atol=1e-8)
                row[SCORE_COLUMNS] = scores
                row["score_source"] = "retained_trajectory_rescoring"
                recomputed += 1
            keep = [*KEYS, "physical_plan_id", "run_id", "control", "permission", "offset_c",
                    "facility_purchased_kwh", *SCORE_COLUMNS, "hourly_sha256", "score_source"]
            rows.append(row[keep].to_dict())
        print(f"Scored {climate}, {composition}: {len(ordered)} logical candidates", flush=True)
    scores = pd.DataFrame(rows)
    if not np.isfinite(scores[[*SCORE_COLUMNS, "facility_purchased_kwh"]].to_numpy()).all():
        raise ValueError("Incomplete paired scores")
    return scores, pd.DataFrame(reference_rows), recomputed


def evaluate_thresholds(scores, references, factors=FACTORS):
    rows = []
    for key, group in scores.groupby(KEYS, sort=True):
        group = group.reset_index(drop=True)
        reference = references[references.climate.eq(key[0]) & references.composition.eq(key[1])].iloc[0]
        values = group[SCORE_COLUMNS].to_numpy(float)
        energy = group.facility_purchased_kwh.to_numpy(float)
        for factor in factors:
            limits = factor * np.array([reference.reference_population_c, reference.reference_fixed_group_c])
            individual = (values[:, :2] <= limits + TOLERANCE).all(axis=1)
            zone = (values[:, 2:] <= limits + TOLERANCE).all(axis=1)
            a, b = choose(np.stack([individual, zone]), energy, group.run_id.to_numpy())
            both = a >= 0 and b >= 0
            state = "both_feasible" if both else ("individual_only" if a >= 0 else
                    "zone_average_only" if b >= 0 else "neither_feasible")
            row = dict(climate=key[0], composition=key[1], factor=factor,
                       population_limit_c=limits[0], fixed_group_limit_c=limits[1],
                       candidate_physical_operations=len(group), individual_qualifying_operations=int(individual.sum()),
                       zone_average_qualifying_operations=int(zone.sum()), feasibility=state,
                       qualifying_sets_differ=bool(np.any(individual != zone)), choices_differ=None,
                       zone_selected_population_exceedance_c=None, zone_selected_fixed_group_exceedance_c=None,
                       zone_selected_individual_exceedance_c=None, additional_purchased_energy_kwh=None,
                       additional_purchased_energy_pct=None)
            for prefix, index in [("individual", a), ("zone_average", b)]:
                for column in ["physical_plan_id", "run_id", "control", "permission", "facility_purchased_kwh"]:
                    row[f"{prefix}_{column}"] = group.iloc[index][column] if index >= 0 else None
            if b >= 0:
                exceedance = np.maximum(values[b, :2] - limits, 0)
                row.update(zone_selected_population_exceedance_c=exceedance[0],
                           zone_selected_fixed_group_exceedance_c=exceedance[1],
                           zone_selected_individual_exceedance_c=exceedance.max())
            if both:
                row.update(choices_differ=bool(a != b),
                           additional_purchased_energy_kwh=energy[a] - energy[b],
                           additional_purchased_energy_pct=100 * (energy[a] - energy[b]) / energy[b])
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(selections):
    rows = []
    for factor, group in selections.groupby("factor", sort=True):
        both = group[group.feasibility.eq("both_feasible")]
        changed = both[both.choices_differ.eq(True)]
        zone = group[group.zone_average_physical_plan_id.notna()]
        row = dict(factor=factor, settings=len(group),
                   qualifying_set_differences=int(group.qualifying_sets_differ.sum()),
                   different_choices=int(changed.shape[0]),
                   different_choices_pct_of_both_feasible=100 * len(changed) / len(both) if len(both) else None,
                   zone_choices_exceeding_individual_limits=int((zone.zone_selected_individual_exceedance_c > TOLERANCE).sum()))
        for state in ["both_feasible", "individual_only", "zone_average_only", "neither_feasible"]:
            row[state] = int(group.feasibility.eq(state).sum())
        for label, subset in [("both_feasible", both), ("changed_choice", changed), ("zone_feasible", zone)]:
            columns = ["zone_selected_individual_exceedance_c"] if label == "zone_feasible" else [
                "zone_selected_individual_exceedance_c", "additional_purchased_energy_kwh", "additional_purchased_energy_pct"]
            for column in columns:
                for statistic in ["min", "median", "max"]:
                    row[f"{label}_{column}_{statistic}"] = getattr(subset[column], statistic)() if len(subset) else None
        rows.append(row)
    return pd.DataFrame(rows)


def verify_reference(selections, expected):
    reference = selections[selections.factor.eq(1)].set_index(KEYS).sort_index()
    expected = expected.set_index(KEYS).sort_index()
    if len(reference) != 54 or not reference.feasibility.eq("both_feasible").all():
        raise AssertionError("Reference requirements must have 54 paired feasible settings")
    if reference.qualifying_sets_differ.sum() != 54 or reference.choices_differ.sum() != 24:
        raise AssertionError("Reference qualifying sets or choices do not reproduce the manuscript")
    for actual, original in [("individual_physical_plan_id", "full_physical_plan_id"),
                             ("zone_average_physical_plan_id", "zone_mean_physical_plan_id")]:
        if not reference[actual].equals(expected[original].rename(actual)):
            raise AssertionError(f"Reference physical choices differ: {actual}")
    np.testing.assert_allclose(reference.additional_purchased_energy_kwh, expected.extra_energy_kwh, rtol=0, atol=1e-6)
    np.testing.assert_allclose(reference.zone_selected_individual_exceedance_c,
                               expected.zone_mean_selected_full_service_excess_c, rtol=0, atol=1e-8)


def run(args):
    result_root, matrix = args.results_root, args.matrix_root
    inputs = dict(library=result_root / "budget_frontiers/ALL_OPERATION_LIBRARY.csv",
                  profiles=matrix / "GLOBAL_PROFILES.csv", references=matrix / "probe_runs/PROBE_RUN_MANIFEST.csv",
                  memberships=args.memberships, anchors=result_root / "base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv",
                  nominal=result_root / "information_uncertainty_summary/NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv")
    library = pd.read_csv(inputs["library"])
    if len(library) != 8586 or not library.engineering_pass.eq(True).all():
        raise ValueError("Expected the complete 8,586-entry qualified pooled library")
    profiles = normalise_profiles(pd.read_csv(inputs["profiles"]))
    references = pd.read_csv(inputs["references"])
    references = references[references.target_c.eq(25)].set_index("climate")
    trajectories = Trajectories(args.trajectory_root)
    scores, limits, recomputed = complete_scores(library, profiles, pd.read_csv(inputs["memberships"]),
                                                references, pd.read_csv(inputs["anchors"]), trajectories)
    reference = evaluate_thresholds(scores, limits, [1.0])
    verify_reference(reference, pd.read_csv(inputs["nominal"]))
    selections = evaluate_thresholds(scores, limits)
    summary = summarise(selections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in [("PAIRED_OPERATION_SERVICE_SCORES.csv", scores),
                            ("REFERENCE_SERVICE_LIMITS.csv", limits),
                            ("PAIRED_SERVICE_THRESHOLD_SELECTIONS.csv", selections),
                            ("SERVICE_THRESHOLD_SUMMARY.csv", summary)]:
        frame.to_csv(args.output_dir / filename, index=False)
    manifest = dict(factors=FACTORS, numerical_service_tolerance_c=TOLERANCE,
                    logical_candidates=len(library), physical_candidates_within_settings=len(scores),
                    setting_count=len(limits), rescored_setting_operations=recomputed,
                    fixed_group_size=27, candidate_deduplication="physical_plan_id within climate and composition; aliases sorted by run_id, control, offset_c",
                    selection="minimum purchased energy; max(1e-6 kWh,1e-9*abs(minimum)) tie tolerance; stable run_id order",
                    energy_difference_denominator="zone-average-selected annual facility purchased energy",
                    infeasible_comparisons="energy differences and choice disagreement are missing when either representation has no qualifying operation",
                    input_files={key: dict(path=source_path(path), sha256=digest(path)) for key, path in inputs.items()},
                    trajectory_files=[dict(path=source_path(path), sha256=sha) for path, sha in sorted(trajectories.used.items())],
                    scoring_sources={source_path(path): digest(path) for path in [
                        Path(__file__).with_name("service_distribution.py"),
                        Path(__file__).with_name("paired_uncertainty.py"),
                        CODE / "src/analysis/global_extension/analyse_global_results.py",
                        CODE / "src/analysis/energy_service/thermal_service.py"]},
                    analysis_source_sha256=digest(__file__))
    (args.output_dir / "SERVICE_THRESHOLD_METHODS.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(summary.to_string(index=False), flush=True)
    return scores, limits, selections, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=CODE / "output/from_raw/strengthening/results")
    parser.add_argument("--matrix-root", type=Path, default=CODE / "output/from_raw/matrix")
    parser.add_argument("--memberships", type=Path, default=CODE / "output/from_raw/strengthening/results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv")
    parser.add_argument("--trajectory-root", type=Path, default=CODE / "output/from_raw")
    parser.add_argument("--output-dir", type=Path, default=CODE / "output/from_raw/strengthening/results/service_threshold_sensitivity")
    run(parser.parse_args())
