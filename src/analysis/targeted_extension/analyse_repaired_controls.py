"""Score repaired controllers against an external common-25 service reference."""

from __future__ import annotations

import argparse
import hashlib
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
from run_demand_aware_energyplus import export_hourly  # noqa: E402
from trajectory_quality import classifications  # noqa: E402


KEYS = ["context", "composition_id", "layout", "permission", "control"]
PROFILE_KEYS = KEYS[:3]
CONTROLS = ("uniform_average", "personal_center", "personal_range")
TOL = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_hourly_path(value: str, manifest_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return manifest_root / path


def load_hourly(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return dict(stored)


def score(data: dict[str, np.ndarray], profiles: pd.DataFrame, tail: np.ndarray) -> dict:
    positions = {str(zone).upper(): i for i, zone in enumerate(data["zone_ids"])}
    zones = np.asarray([positions[str(zone).upper()] for zone in profiles.zone_id], dtype=int)
    people = data["people"] * data["occupied"].astype(bool)[:, None]
    return score_service(
        data["temperatures"], people, zones,
        profiles.centre_c.to_numpy(float), profiles.lower_c.to_numpy(float),
        profiles.upper_c.to_numpy(float), tail,
    )


def choose_energy(candidates: pd.DataFrame) -> dict | None:
    if candidates.empty:
        return None
    minimum = float(candidates.annual_site_kwh.min())
    eligible = candidates[
        candidates.annual_site_kwh.le(minimum + max(1e-6, 1e-9 * abs(minimum)))
    ]
    return eligible.sort_values(["run_id", "variant"], kind="mergesort").iloc[0].to_dict()


def comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "annual_site_kwh", "annual_hvac_electric_kwh", "annual_gas_hhv_kwh",
        "annual_heating_electric_kwh", "annual_cooling_electric_kwh",
        "annual_fans_electric_kwh", "annual_operational_co2_kg",
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "modeled_interval_person_hour_coverage", "modeled_interval_mean_excess_c",
    ]
    identities = [
        "context", "composition_id", "layout", "permission",
        "comparison_mode", "service_fraction",
    ]
    reference = rows[rows.control.eq("uniform_average")].set_index(identities)
    output = []
    for item in rows[rows.control.ne("uniform_average")].to_dict("records"):
        key = tuple(item[name] for name in identities)
        ref = reference.loc[key] if key in reference.index else None
        matched = (ref is not None and item["selection_status"] == "selected"
                   and ref.selection_status == "selected")
        row = {name: item[name] for name in identities} | {
            "alternative": item["control"],
            "status": "matched" if matched else "unmatched",
            "alternative_run_id": item.get("run_id", ""),
            "reference_run_id": ref.run_id if matched else "",
        }
        for metric in metrics:
            row[f"delta_{metric}"] = item.get(metric, np.nan) - ref[metric] if matched else np.nan
        row["delta_site_pct"] = (
            100 * row["delta_annual_site_kwh"] / ref.annual_site_kwh if matched else np.nan
        )
        output.append(row)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-source", default="ASHRAE")
    parser.add_argument("--quality-scope", choices=("primary", "strict"), default="primary")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = args.run_manifest.resolve()
    manifest = pd.read_csv(run_manifest)
    quality = classifications(args.run_manifest)
    manifest["primary_output_valid"] = manifest.run_id.map(
        lambda run_id: bool(quality[str(run_id)]["primary_output_valid"])
    )
    manifest["strict_warning_free"] = manifest.run_id.map(
        lambda run_id: bool(quality[str(run_id)]["strict_warning_free"])
    )
    manifest["isolated_hvac_iteration_warnings"] = manifest.run_id.map(
        lambda run_id: int(quality[str(run_id)]["isolated_hvac_iteration_warnings"])
    )
    commands = pd.DataFrame(json.loads(args.commands.read_text(encoding="utf-8")))
    profiles = pd.read_csv(args.profiles.resolve())
    factors = pd.read_csv(args.factors.resolve()).set_index("city")
    model_contexts = json.loads(
        (args.model_root.resolve() / "MODEL_CONTEXTS.json").read_text(encoding="utf-8")
    )
    input_contexts = json.loads(
        (args.input_root.resolve() / "controller/CONTEXTS.json").read_text(encoding="utf-8")
    )
    command_keys = commands[[*KEYS, "variant", "offset_c"]].copy()
    manifest_keys = manifest[[*KEYS, "variant", "offset_c"]].copy()
    if len(command_keys) != len(manifest_keys) or set(map(tuple, command_keys.to_numpy())) != set(
        map(tuple, manifest_keys.to_numpy())
    ):
        raise ValueError("Run manifest does not match the declared controller commands")
    designs = manifest.merge(
        commands[[*KEYS, "variant", "offset_c"]],
        on=[*KEYS, "variant", "offset_c"], how="left", validate="one_to_one",
        indicator=True,
    )
    if not designs["_merge"].eq("both").all():
        raise ValueError("Run manifest does not match the declared controller commands")
    designs = designs.drop(columns="_merge")
    designs["engineering_qualified"] = designs[
        "primary_output_valid" if args.quality_scope == "primary" else "strict_warning_free"
    ].astype(bool)

    declared_groups = designs[PROFILE_KEYS].drop_duplicates()
    profile_groups = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(PROFILE_KEYS, sort=True)
        if ((declared_groups == pd.Series(key, index=PROFILE_KEYS)).all(axis=1)).any()
    }
    if len(profile_groups) != len(declared_groups):
        raise ValueError("A controller group lacks its demand profiles")

    anchor_trajectories = {}
    for context in designs.context.unique():
        destination = output / "anchor_hourly" / f"{context}.npz"
        if not destination.exists():
            frame = pd.read_csv(
                args.model_root.resolve() / "baseline_evidence" / context / "eplusout.csv"
            )
            with np.load(
                args.input_root.resolve() / input_contexts[context]["probes"], allow_pickle=False
            ) as probe:
                occupied = probe["occupied_source_indices"]
            export_hourly(frame, input_contexts[context]["zones"], occupied, destination)
        anchor_trajectories[context] = load_hourly(destination)

    anchors: dict[tuple, dict] = {}
    tails = []
    for key, group in profile_groups.items():
        temporary = score(
            anchor_trajectories[key[0]], group, np.arange(len(group), dtype=int)
        )
        distance = temporary["profiles"]["reference_mismatch_c"]
        order = pd.DataFrame({"profile_id": group.profile_id, "distance": distance}).sort_values(
            ["distance", "profile_id"], ascending=[False, True], kind="mergesort"
        )
        fixed_ids = order.head(max(1, int(np.ceil(0.10 * len(group))))).profile_id
        tail = np.flatnonzero(group.profile_id.isin(fixed_ids))
        anchor_score = score(anchor_trajectories[key[0]], group, tail)
        anchors[key] = {
            "profiles": group,
            "tail": tail,
            "anchor_mean_c": anchor_score["summary"]["mean_reference_mismatch_c"],
            "anchor_fixed_tail_c": anchor_score["summary"]["fixed_tail_reference_mismatch_c"],
        }
        tails.extend(
            dict(zip(PROFILE_KEYS, key, strict=True)) | {"profile_id": profile_id}
            for profile_id in fixed_ids
        )
    pd.DataFrame([
        dict(zip(PROFILE_KEYS, key, strict=True)) | {
            "anchor_mean_c": value["anchor_mean_c"],
            "anchor_fixed_tail_c": value["anchor_fixed_tail_c"],
        }
        for key, value in anchors.items()
    ]).to_csv(output / "COMMON25_ANCHORS.csv", index=False)
    pd.DataFrame(tails).to_csv(output / "FIXED_TAIL_MEMBERSHIPS.csv", index=False)

    cache: dict[tuple, dict] = {}
    candidates = []
    for row in designs.itertuples(index=False):
        item = {
            name: getattr(row, name)
            for name in [*KEYS, "variant", "offset_c", "run_id", "hourly_path"]
        }
        item["engineering_qualified"] = bool(row.engineering_qualified)
        if row.engineering_qualified:
            key = (row.context, row.composition_id, row.layout)
            score_key = (*key, row.run_id)
            if score_key not in cache:
                data = load_hourly(resolve_hourly_path(row.hourly_path, run_manifest.parent))
                result = score(data, anchors[key]["profiles"], anchors[key]["tail"])
                values = dict(result["summary"])
                for name in (
                    "hvac_electric_kwh", "gas_hhv_kwh", "heating_electric_kwh",
                    "cooling_electric_kwh", "fans_electric_kwh",
                ):
                    values[f"annual_{name}"] = float(data[name].sum())
                values["annual_site_kwh"] = (
                    values["annual_hvac_electric_kwh"] + values["annual_gas_hhv_kwh"]
                )
                factor = factors.loc[row.context]
                values["annual_operational_co2_kg"] = (
                    values["annual_hvac_electric_kwh"] * factor.electricity_co2_kg_kwh
                    + values["annual_gas_hhv_kwh"] * factor.gas_co2_kg_kwh_hhv
                )
                cache[score_key] = values
            item.update(cache[score_key])
        candidates.append(item)
    candidate_table = pd.DataFrame(candidates)
    candidate_table.insert(0, "data_source", args.data_source)
    candidate_table.to_csv(output / "REPAIRED_ANNUAL_CANDIDATES.csv", index=False)

    direct = candidate_table[candidate_table.offset_c.eq(0)].copy()
    direct["comparison_mode"] = "direct"
    direct["service_fraction"] = -1.0
    direct["selection_status"] = np.where(
        direct.engineering_qualified, "selected", "engineering_unavailable"
    )
    selections = []
    for key, group in candidate_table.groupby(KEYS, sort=True):
        anchor = anchors[key[:3]]
        for fraction in (1.0, 0.75, 0.50):
            qualified = group[
                group.engineering_qualified
                & group.mean_reference_mismatch_c.le(fraction * anchor["anchor_mean_c"] + TOL)
                & group.fixed_tail_reference_mismatch_c.le(
                    fraction * anchor["anchor_fixed_tail_c"] + TOL
                )
            ]
            selected = choose_energy(qualified)
            item = dict(zip(KEYS, key, strict=True)) | {
                "data_source": args.data_source,
                "comparison_mode": "equal_service",
                "service_fraction": fraction,
                "engineering_qualified_candidates": int(group.engineering_qualified.sum()),
                "service_qualified_candidates": len(qualified),
                "selection_status": "selected" if selected else "service_unavailable",
            }
            if selected:
                item.update(selected)
            selections.append(item)
    selection_table = pd.DataFrame(selections)
    direct.to_csv(output / "REPAIRED_ANNUAL_DIRECT.csv", index=False)
    selection_table.to_csv(output / "REPAIRED_ANNUAL_SERVICE_SELECTIONS.csv", index=False)
    comparison_table = comparisons(pd.concat([direct, selection_table], ignore_index=True))
    comparison_table.to_csv(output / "REPAIRED_CONTROL_COMPARISONS.csv", index=False)
    qa = {
        "logical_candidates": len(candidate_table),
        "unique_physical_runs": int(designs.run_id.nunique()),
        "engineering_qualified_runs": int(
            designs.loc[designs.engineering_qualified, "run_id"].nunique()
        ),
        "direct_matched_pairs": int(
            comparison_table.query("comparison_mode == 'direct' and status == 'matched'").shape[0]
        ),
        "equal_service_matched_pairs": int(
            comparison_table.query("comparison_mode == 'equal_service' and status == 'matched'").shape[0]
        ),
        "quality_scope": args.quality_scope,
        "isolated_hvac_iteration_warnings": int(
            designs.drop_duplicates("run_id").isolated_hvac_iteration_warnings.sum()
        ),
        "input_sha256": {
            "run_manifest": sha256(run_manifest),
            "commands": sha256(args.commands.resolve()),
            "profiles": sha256(args.profiles.resolve()),
            "factors": sha256(args.factors.resolve()),
        },
    }
    (output / "ANALYSIS_QA.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
