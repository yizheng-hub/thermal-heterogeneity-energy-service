"""Compare demand information using one baseline and one physical DR action library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from thermal_service import score_service  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from analyse_repaired_controls import resolve_hourly_path  # noqa: E402
from trajectory_quality import classifications  # noqa: E402


PROFILE_KEYS = ["context", "composition_id", "layout"]
INFORMATION_LEVELS = ("building_mean", "zone_mean", "full_distribution")
TOL = 1e-9


def load_hourly(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return dict(stored)


def phase_indices(calendar: dict[str, np.ndarray], event: dict) -> dict[str, np.ndarray]:
    civil = calendar["hour"] - 1 + calendar["dst"]
    same_day = (calendar["month"] == int(event["month"])) & (calendar["day"] == int(event["day"]))
    precondition = np.flatnonzero(same_day & (civil >= 12) & (civil < 14))
    response = np.flatnonzero(same_day & (civil >= 14) & (civil < 16))
    if (len(precondition) != 2 or len(response) != 2
            or response[0] != precondition[-1] + 1 or response[-1] + 49 > len(civil)):
        raise ValueError("Expected consecutive two-hour precondition and event windows")
    return {
        "precondition": precondition,
        "event": response,
        "recovery": np.arange(response[-1] + 1, response[-1] + 49),
    }


def information_centres(profiles: pd.DataFrame, level: str) -> np.ndarray:
    if level == "full_distribution":
        return profiles.centre_c.to_numpy(float)
    if level == "zone_mean":
        return profiles.groupby("zone_id").centre_c.transform("mean").to_numpy(float)
    if level == "building_mean":
        return np.full(len(profiles), float(profiles.centre_c.mean()))
    raise ValueError(f"Unknown information level: {level}")


def service_score(
    data: dict[str, np.ndarray],
    profiles: pd.DataFrame,
    tail: np.ndarray,
    indices: np.ndarray,
    information_level: str,
) -> dict[str, float]:
    positions = {str(zone).upper(): index for index, zone in enumerate(data["zone_ids"])}
    profile_zones = np.asarray(
        [positions[str(zone).upper()] for zone in profiles.zone_id], dtype=int
    )
    centres = information_centres(profiles, information_level)
    people = data["people"][indices] * data["occupied"][indices].astype(bool)[:, None]
    result = score_service(
        data["temperatures"][indices], people, profile_zones,
        centres, centres, centres, tail,
    )
    return {
        "mean_c": float(result["summary"]["mean_reference_mismatch_c"]),
        "fixed_tail_c": float(result["summary"]["fixed_tail_reference_mismatch_c"]),
    }


def choose_action(group: pd.DataFrame) -> dict | None:
    eligible = group[
        group.predicted_service_qualified
        & group.engineering_qualified
        & group.event_average_reduction_kw.gt(TOL)
    ]
    if eligible.empty:
        return None
    best = float(eligible.event_average_reduction_kw.max())
    return eligible[eligible.event_average_reduction_kw.ge(best - TOL)].sort_values(
        ["action_id", "run_id"], kind="mergesort"
    ).iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--event-dates", type=Path, required=True)
    parser.add_argument("--profiles", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-scope", choices=("primary", "strict"), default="primary")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = args.run_manifest.resolve()
    runs = pd.read_csv(run_manifest)
    quality = classifications(args.run_manifest)
    runs["primary_output_valid"] = runs.run_id.map(
        lambda run_id: bool(quality[str(run_id)]["primary_output_valid"])
    )
    runs["strict_warning_free"] = runs.run_id.map(
        lambda run_id: bool(quality[str(run_id)]["strict_warning_free"])
    )
    runs["isolated_hvac_iteration_warnings"] = runs.run_id.map(
        lambda run_id: int(quality[str(run_id)]["isolated_hvac_iteration_warnings"])
    )
    runs["analysis_qualified"] = runs[
        "primary_output_valid" if args.quality_scope == "primary" else "strict_warning_free"
    ]
    annual = pd.read_csv(args.annual_manifest.resolve(), keep_default_na=False)
    dates = json.loads(args.event_dates.resolve().read_text(encoding="utf-8"))
    factors = pd.read_csv(args.factors.resolve()).set_index("city")
    sources: dict[str, pd.DataFrame] = {}
    for specification in args.profiles:
        if "=" not in specification:
            parser.error("--profiles must use NAME=PATH")
        name, path = specification.split("=", 1)
        sources[name] = pd.read_csv(Path(path).resolve())

    baseline_paths: dict[str, Path] = {}
    for context, group in annual[annual.variant.eq(3)].groupby("context", sort=True):
        qualified = group[group.engineering_status.isin({"cached_qualified", "new_qualified"})]
        if qualified.run_id.nunique() != 1:
            raise ValueError(f"Expected one qualified common-25 physical baseline for {context}")
        relative = Path(qualified.iloc[0].path_hourly)
        baseline_paths[context] = (args.annual_manifest.resolve().parent / relative).resolve()

    trajectory_cache: dict[str, dict[str, np.ndarray]] = {}
    def trajectory(path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in trajectory_cache:
            trajectory_cache[key] = load_hourly(path.resolve())
        return trajectory_cache[key]

    candidates: list[dict] = []
    anchors_out: list[dict] = []
    for source, profiles in sources.items():
        for profile_key, raw_group in profiles.groupby(PROFILE_KEYS, sort=True):
            context = str(profile_key[0])
            group = raw_group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
            baseline = trajectory(baseline_paths[context])
            positions = {str(zone).upper(): index for index, zone in enumerate(baseline["zone_ids"])}
            profile_zones = np.asarray([positions[str(zone).upper()] for zone in group.zone_id], dtype=int)
            annual_people = baseline["people"] * baseline["occupied"].astype(bool)[:, None]
            full = score_service(
                baseline["temperatures"], annual_people, profile_zones,
                group.centre_c.to_numpy(float), group.centre_c.to_numpy(float),
                group.centre_c.to_numpy(float), np.arange(len(group), dtype=int),
            )
            distance = full["profiles"]["reference_mismatch_c"]
            tail_order = pd.DataFrame({
                "profile_id": group.profile_id, "distance": distance,
            }).sort_values(["distance", "profile_id"], ascending=[False, True], kind="mergesort")
            fixed_ids = set(tail_order.head(max(1, int(np.ceil(0.10 * len(group))))).profile_id)
            tail = np.flatnonzero(group.profile_id.isin(fixed_ids))

            for event in dates[context]:
                phases = phase_indices(baseline, event)
                anchors = {
                    phase: service_score(
                        baseline, group, tail, indices, "full_distribution"
                    )
                    for phase, indices in phases.items()
                }
                anchors_out.extend(
                    dict(zip(PROFILE_KEYS, profile_key, strict=True)) | {
                        "data_source": source,
                        "event_month": int(event["month"]),
                        "event_day": int(event["day"]),
                        "phase": phase,
                        "anchor_mean_c": values["mean_c"],
                        "anchor_fixed_tail_c": values["fixed_tail_c"],
                        "fixed_tail_profiles": len(tail),
                    }
                    for phase, values in anchors.items()
                )

                context_runs = runs[runs.context.eq(context)]
                for permission in ("building", "native_zone"):
                    applicable = context_runs[context_runs.permission.eq(permission)]
                    for action in applicable.itertuples(index=False):
                        event_data = trajectory(
                            resolve_hourly_path(action.hourly_path, run_manifest.parent)
                        ) if action.analysis_qualified else None
                        base_item = dict(zip(PROFILE_KEYS, profile_key, strict=True)) | {
                            "data_source": source,
                            "permission": permission,
                            "event_month": int(event["month"]),
                            "event_day": int(event["day"]),
                            "action_id": action.action_id,
                            "action_family": action.action_family,
                            "run_id": action.run_id,
                            "engineering_qualified": bool(action.analysis_qualified),
                        }
                        if event_data is not None:
                            pre = phases["precondition"]
                            response = phases["event"]
                            recovery = phases["recovery"]
                            delta_e = event_data["hvac_electric_kwh"] - baseline["hvac_electric_kwh"]
                            delta_g = event_data["gas_hhv_kwh"] - baseline["gas_hhv_kwh"]
                            horizon = np.concatenate([pre, response, recovery])
                            factor = factors.loc[context]
                            base_item.update({
                                "event_average_reduction_kw": float(-delta_e[response].mean()),
                                "event_electric_saving_kwh": float(-delta_e[response].sum()),
                                "precondition_extra_electric_kwh": float(delta_e[pre].sum()),
                                "recovery_extra_electric_kwh": float(delta_e[recovery].sum()),
                                "precondition_peak_extra_kw": float(delta_e[pre].max()),
                                "recovery_peak_extra_kw": float(delta_e[recovery].max()),
                                "full_window_net_electric_saving_kwh": float(-delta_e[horizon].sum()),
                                "full_window_gas_change_kwh_hhv": float(delta_g[horizon].sum()),
                                "full_window_operational_co2_change_kg": float(
                                    delta_e[horizon].sum() * factor.electricity_co2_kg_kwh
                                    + delta_g[horizon].sum() * factor.gas_co2_kg_kwh_hhv
                                ),
                            })
                        for level in INFORMATION_LEVELS:
                            item = base_item | {"information_level": level}
                            if event_data is None:
                                item.update(
                                    predicted_service_qualified=False,
                                    full_distribution_service_qualified=False,
                                    review_outcome="engineering_unavailable",
                                )
                            else:
                                predicted_pass = True
                                full_pass = True
                                for phase, indices in phases.items():
                                    predicted = service_score(event_data, group, tail, indices, level)
                                    reviewed = service_score(
                                        event_data, group, tail, indices, "full_distribution"
                                    )
                                    for metric in ("mean_c", "fixed_tail_c"):
                                        item[f"{phase}_predicted_{metric}"] = predicted[metric]
                                        item[f"{phase}_full_{metric}"] = reviewed[metric]
                                        limit = anchors[phase][metric]
                                        item[f"{phase}_{metric}_budget"] = limit
                                        predicted_pass &= predicted[metric] <= limit + TOL
                                        full_pass &= reviewed[metric] <= limit + TOL
                                item["predicted_service_qualified"] = bool(predicted_pass)
                                item["full_distribution_service_qualified"] = bool(full_pass)
                                item["review_outcome"] = (
                                    "qualified" if predicted_pass and full_pass
                                    else "false_acceptance" if predicted_pass and not full_pass
                                    else "false_rejection" if not predicted_pass and full_pass
                                    else "rejected"
                                )
                            candidates.append(item)

    candidate_table = pd.DataFrame(candidates)
    candidate_table.to_csv(
        output / "COMMON_BASELINE_DR_CANDIDATES.csv.gz", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    pd.DataFrame(anchors_out).to_csv(output / "COMMON_BASELINE_DR_ANCHORS.csv", index=False)

    selection_keys = [
        "data_source", *PROFILE_KEYS, "permission", "event_month", "information_level",
    ]
    selections = []
    for key, group in candidate_table.groupby(selection_keys, sort=True):
        selected = choose_action(group)
        row = dict(zip(selection_keys, key, strict=True)) | {
            "candidate_actions": len(group),
            "engineering_qualified_actions": int(group.engineering_qualified.sum()),
            "predicted_qualified_actions": int(group.predicted_service_qualified.sum()),
            "full_qualified_actions": int(group.full_distribution_service_qualified.sum()),
            "selection_status": "selected" if selected else "no_predicted_qualified_positive_action",
            "selected_action_id": "",
            "selected_run_id": "",
            "selected_response_kw": 0.0,
            "full_reviewed_usable_response_kw": 0.0,
            "selected_passes_full_review": False,
            "selected_review_outcome": "none",
        }
        if selected:
            row.update(selected)
            row.update({
                "selected_action_id": selected["action_id"],
                "selected_run_id": selected["run_id"],
                "selected_response_kw": selected["event_average_reduction_kw"],
                "selected_passes_full_review": bool(
                    selected["full_distribution_service_qualified"]
                ),
                "selected_review_outcome": selected["review_outcome"],
            })
            row["full_reviewed_usable_response_kw"] = (
                row["selected_response_kw"] if row["selected_passes_full_review"] else 0.0
            )
        selections.append(row)
    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(output / "COMMON_BASELINE_DR_SELECTIONS.csv", index=False)

    comparisons = []
    for key, group in selection_table.groupby(
        ["data_source", *PROFILE_KEYS, "permission", "event_month"], sort=True
    ):
        indexed = group.set_index("information_level")
        full = indexed.loc["full_distribution"]
        for level in ("building_mean", "zone_mean"):
            row = indexed.loc[level]
            comparisons.append(
                dict(zip(
                    ["data_source", *PROFILE_KEYS, "permission", "event_month"],
                    key, strict=True,
                )) | {
                    "information_level": level,
                    "selected_response_kw": row.selected_response_kw,
                    "full_reviewed_usable_response_kw": (
                        row.full_reviewed_usable_response_kw
                    ),
                    "ideal_full_information_response_kw": full.selected_response_kw,
                    "response_gap_kw": row.selected_response_kw - full.selected_response_kw,
                    "selected_passes_full_review": bool(row.selected_passes_full_review),
                    "selected_review_outcome": row.selected_review_outcome,
                    "selected_action_id": row.selected_action_id,
                    "ideal_action_id": full.selected_action_id,
                }
            )
    pd.DataFrame(comparisons).to_csv(
        output / "COMMON_BASELINE_DR_INFORMATION_COMPARISONS.csv", index=False
    )

    full = selection_table[selection_table.information_level.eq("full_distribution")]
    pivot = full.pivot_table(
        index=["data_source", *PROFILE_KEYS, "event_month"],
        columns="permission", values="selected_response_kw", aggfunc="first",
    ).dropna()
    violations = int((pivot.native_zone + TOL < pivot.building).sum()) if len(pivot) else 0
    if violations:
        raise ValueError("Full-information capacity fell after adding a nested action library")
    report = {
        "status": "PASS",
        "data_sources": sorted(sources),
        "profile_groups": int(sum(
            frame[PROFILE_KEYS].drop_duplicates().shape[0] for frame in sources.values()
        )),
        "candidate_rows": len(candidate_table),
        "selection_rows": len(selection_table),
        "engineering_qualified_physical_runs": int(
            runs.loc[runs.analysis_qualified, "run_id"].nunique()
        ),
        "full_information_permission_monotonicity_violations": violations,
        "shared_physical_action_library": True,
        "common_25_baseline": True,
        "fixed_full_distribution_service_budgets": True,
        "fixed_full_distribution_tail_identity": True,
        "interval_width_used_for_primary_information_comparison": False,
        "quality_scope": args.quality_scope,
        "isolated_hvac_iteration_warnings": int(
            runs.drop_duplicates("run_id").isolated_hvac_iteration_warnings.sum()
        ),
    }
    (output / "COMMON_BASELINE_DR_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
