"""Evaluate grid response from each controller's own direct annual baseline."""

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
from analyse_common_baseline_dr import load_hourly, phase_indices  # noqa: E402
from analyse_repaired_controls import resolve_hourly_path  # noqa: E402
from trajectory_quality import classifications  # noqa: E402


PROFILE_KEYS = ["context", "composition_id", "layout"]
TOL = 1e-9


def score_window(
    data: dict[str, np.ndarray], profiles: pd.DataFrame, tail: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    positions = {str(zone).upper(): index for index, zone in enumerate(data["zone_ids"])}
    profile_zones = np.asarray(
        [positions[str(zone).upper()] for zone in profiles.zone_id], dtype=int
    )
    centres = profiles.centre_c.to_numpy(float)
    people = data["people"][indices] * data["occupied"][indices].astype(bool)[:, None]
    result = score_service(
        data["temperatures"][indices], people, profile_zones,
        centres, centres, centres, tail,
    )
    return {
        "mean_c": float(result["summary"]["mean_reference_mismatch_c"]),
        "fixed_tail_c": float(result["summary"]["fixed_tail_reference_mismatch_c"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--annual-manifest", type=Path, required=True)
    parser.add_argument("--anchor-annual-manifest", type=Path, required=True)
    parser.add_argument("--event-dates", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-scope", choices=("primary", "strict"), default="primary")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    event_manifest = args.event_manifest.resolve()
    annual_manifest = args.annual_manifest.resolve()
    events = pd.read_csv(event_manifest)
    annual = pd.read_csv(annual_manifest)
    anchor_manifest = pd.read_csv(args.anchor_annual_manifest.resolve(), keep_default_na=False)
    profiles = pd.read_csv(args.profiles.resolve())
    dates = json.loads(args.event_dates.resolve().read_text(encoding="utf-8"))
    factors = pd.read_csv(args.factors.resolve()).set_index("city")

    event_quality = classifications(args.event_manifest)
    annual_quality = classifications(args.annual_manifest)
    quality_field = "primary_output_valid" if args.quality_scope == "primary" else "strict_warning_free"
    events["analysis_qualified"] = events.run_id.map(
        lambda run_id: bool(event_quality[str(run_id)][quality_field])
    )
    annual["analysis_qualified"] = annual.run_id.map(
        lambda run_id: bool(annual_quality[str(run_id)][quality_field])
    )
    annual_by_id = annual.drop_duplicates("run_id").set_index("run_id")

    anchor_paths = {}
    for context, group in anchor_manifest[anchor_manifest.variant.eq(3)].groupby("context"):
        qualified = group[group.engineering_status.isin({"cached_qualified", "new_qualified"})]
        if qualified.run_id.nunique() != 1:
            raise ValueError(f"Expected one common-25 anchor for {context}")
        anchor_paths[context] = (
            args.anchor_annual_manifest.resolve().parent / Path(qualified.iloc[0].path_hourly)
        ).resolve()

    cache: dict[str, dict[str, np.ndarray]] = {}
    def trajectory(path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in cache:
            cache[key] = load_hourly(path.resolve())
        return cache[key]

    rows = []
    anchors_out = []
    for profile_key, raw_group in profiles.groupby(PROFILE_KEYS, sort=True):
        context = str(profile_key[0])
        group = raw_group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        anchor_data = trajectory(anchor_paths[context])
        positions = {str(zone).upper(): index for index, zone in enumerate(anchor_data["zone_ids"])}
        profile_zones = np.asarray(
            [positions[str(zone).upper()] for zone in group.zone_id], dtype=int
        )
        annual_people = anchor_data["people"] * anchor_data["occupied"].astype(bool)[:, None]
        centres = group.centre_c.to_numpy(float)
        full = score_service(
            anchor_data["temperatures"], annual_people, profile_zones,
            centres, centres, centres, np.arange(len(group), dtype=int),
        )
        distance = full["profiles"]["reference_mismatch_c"]
        order = pd.DataFrame({"profile_id": group.profile_id, "distance": distance}).sort_values(
            ["distance", "profile_id"], ascending=[False, True], kind="mergesort"
        )
        fixed_ids = set(order.head(max(1, int(np.ceil(0.10 * len(group))))).profile_id)
        tail = np.flatnonzero(group.profile_id.isin(fixed_ids))

        logical = events[
            events.context.eq(context)
            & events.composition_id.eq(profile_key[1])
            & events.layout.eq(profile_key[2])
        ]
        for event_date in dates[context]:
            phases = phase_indices(anchor_data, event_date)
            budgets = {
                phase: score_window(anchor_data, group, tail, indices)
                for phase, indices in phases.items() if phase != "precondition"
            }
            anchors_out.extend(
                dict(zip(PROFILE_KEYS, profile_key, strict=True)) | {
                    "data_source": args.data_source,
                    "event_month": int(event_date["month"]),
                    "phase": phase,
                    "anchor_mean_c": score["mean_c"],
                    "anchor_fixed_tail_c": score["fixed_tail_c"],
                }
                for phase, score in budgets.items()
            )
            for action in logical.itertuples(index=False):
                baseline_row = annual_by_id.loc[str(action.baseline_run_id)]
                baseline_ok = bool(baseline_row.analysis_qualified)
                event_ok = bool(action.analysis_qualified)
                item = dict(zip(PROFILE_KEYS, profile_key, strict=True)) | {
                    "data_source": args.data_source,
                    "control": action.control,
                    "permission": action.permission,
                    "event_month": int(event_date["month"]),
                    "event_day": int(event_date["day"]),
                    "event_amplitude_c": float(action.event_amplitude_c),
                    "baseline_run_id": str(action.baseline_run_id),
                    "event_run_id": str(action.run_id),
                    "baseline_output_valid": baseline_ok,
                    "event_output_valid": event_ok,
                    "baseline_service_qualified": False,
                    "event_service_qualified": False,
                }
                if baseline_ok and event_ok:
                    baseline_data = trajectory(resolve_hourly_path(
                        baseline_row.hourly_path, annual_manifest.parent
                    ))
                    event_data = trajectory(resolve_hourly_path(
                        action.hourly_path, event_manifest.parent
                    ))
                    delta_e = event_data["hvac_electric_kwh"] - baseline_data["hvac_electric_kwh"]
                    delta_g = event_data["gas_hhv_kwh"] - baseline_data["gas_hhv_kwh"]
                    response = phases["event"]
                    recovery = phases["recovery"]
                    horizon = np.concatenate([response, recovery])
                    terminal = np.arange(response[0], min(response[0] + 168, len(delta_e)))
                    factor = factors.loc[context]
                    baseline_pass = True
                    perturbed_pass = True
                    for phase in ("event", "recovery"):
                        base_score = score_window(
                            baseline_data, group, tail, phases[phase]
                        )
                        perturbed_score = score_window(
                            event_data, group, tail, phases[phase]
                        )
                        for metric in ("mean_c", "fixed_tail_c"):
                            item[f"{phase}_baseline_{metric}"] = base_score[metric]
                            item[f"{phase}_perturbed_{metric}"] = perturbed_score[metric]
                            item[f"{phase}_{metric}_budget"] = budgets[phase][metric]
                            baseline_pass &= base_score[metric] <= budgets[phase][metric] + TOL
                            perturbed_pass &= perturbed_score[metric] <= budgets[phase][metric] + TOL
                    item.update({
                        "baseline_service_qualified": bool(baseline_pass),
                        "event_service_qualified": bool(perturbed_pass),
                        "event_average_reduction_kw": float(-delta_e[response].mean()),
                        "event_electric_saving_kwh": float(-delta_e[response].sum()),
                        "recovery_extra_electric_kwh": float(delta_e[recovery].sum()),
                        "event_plus_recovery_net_electric_saving_kwh": float(-delta_e[horizon].sum()),
                        "event_plus_recovery_gas_change_kwh_hhv": float(delta_g[horizon].sum()),
                        "terminal_168h_net_electric_saving_kwh": float(-delta_e[terminal].sum()),
                        "event_plus_recovery_operational_co2_change_kg": float(
                            delta_e[horizon].sum() * factor.electricity_co2_kg_kwh
                            + delta_g[horizon].sum() * factor.gas_co2_kg_kwh_hhv
                        ),
                    })
                item["qualified_positive_response"] = bool(
                    item["baseline_output_valid"] and item["event_output_valid"]
                    and item["baseline_service_qualified"] and item["event_service_qualified"]
                    and item.get("event_average_reduction_kw", 0.0) > TOL
                )
                rows.append(item)

    candidates = pd.DataFrame(rows)
    candidates.to_csv(output / "OWN_BASELINE_DR_CANDIDATES.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(anchors_out).to_csv(output / "OWN_BASELINE_DR_ANCHORS.csv", index=False)

    selections = []
    keys = ["data_source", *PROFILE_KEYS, "permission", "control", "event_month"]
    for key, group in candidates.groupby(keys, sort=True):
        eligible = group[group.qualified_positive_response]
        selected = None
        if not eligible.empty:
            maximum = float(eligible.event_average_reduction_kw.max())
            selected = eligible[eligible.event_average_reduction_kw.ge(maximum - TOL)].sort_values(
                ["event_amplitude_c", "event_run_id"], kind="mergesort"
            ).iloc[0].to_dict()
        row = dict(zip(keys, key, strict=True)) | {
            "selection_status": "selected" if selected else "no_qualified_positive_response",
            "selected_response_kw": 0.0,
        }
        if selected:
            row.update(selected)
            row["selected_response_kw"] = selected["event_average_reduction_kw"]
        selections.append(row)
    selection_table = pd.DataFrame(selections)
    selection_table.to_csv(output / "OWN_BASELINE_DR_SELECTIONS.csv", index=False)

    comparisons = []
    comparison_keys = ["data_source", *PROFILE_KEYS, "permission", "event_month"]
    for key, group in selection_table.groupby(comparison_keys, sort=True):
        indexed = group.set_index("control")
        if "uniform_average" not in indexed.index:
            continue
        reference = indexed.loc["uniform_average"]
        for control in ("personal_center", "personal_range"):
            if control not in indexed.index:
                continue
            alternative = indexed.loc[control]
            comparisons.append(dict(zip(comparison_keys, key, strict=True)) | {
                "control": control,
                "uniform_response_kw": float(reference.selected_response_kw),
                "demand_guided_response_kw": float(alternative.selected_response_kw),
                "response_difference_kw": float(
                    alternative.selected_response_kw - reference.selected_response_kw
                ),
                "uniform_status": reference.selection_status,
                "demand_guided_status": alternative.selection_status,
            })
    pd.DataFrame(comparisons).to_csv(output / "OWN_BASELINE_DR_COMPARISONS.csv", index=False)
    report = {
        "data_source": args.data_source,
        "quality_scope": args.quality_scope,
        "candidate_rows": len(candidates),
        "selection_rows": len(selection_table),
        "qualified_positive_candidates": int(candidates.qualified_positive_response.sum()),
        "event_and_recovery_service_checked": True,
        "common_25_external_service_anchor": True,
        "annual_controller_baselines_are_distinct": True,
    }
    (output / "OWN_BASELINE_DR_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
