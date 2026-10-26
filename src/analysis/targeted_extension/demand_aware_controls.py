"""Generate repaired demand-aware thermostat targets from frozen probes.

Uniform average uses one building-wide demand centre. Personal centre uses
the complete distribution of assigned individual centres, and Personal range
uses the complete distribution of model-derived intervals. The common 25 C
operation is not a controller candidate; it remains an external service
reference in the downstream comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTROLS = ("uniform_average", "personal_center", "personal_range")
OFFSETS_C = (-1.0, -0.5, 0.0, 0.5, 1.0)


def profile_mismatch_curves(
    predicted_temperature: np.ndarray,
    native_people: np.ndarray,
    centres_c: np.ndarray,
) -> np.ndarray:
    """Annual mean absolute mismatch for every grid point and demand profile."""
    predicted = np.asarray(predicted_temperature, dtype=float)
    people = np.asarray(native_people, dtype=float)
    centres = np.asarray(centres_c, dtype=float)
    if predicted.ndim != 2 or people.shape != (predicted.shape[1],):
        raise ValueError("Probe prediction and native occupancy shapes differ")
    if centres.ndim != 1 or centres.size == 0 or people.sum() <= 0:
        raise ValueError("Each occupied binding requires demand centres")
    return np.einsum(
        "ghp,h->gp",
        np.abs(predicted[:, :, None] - centres[None, None, :]),
        people,
        optimize=True,
    ) / people.sum()


def profile_interval_curves(
    predicted_temperature: np.ndarray,
    native_people: np.ndarray,
    lower_c: np.ndarray,
    upper_c: np.ndarray,
) -> np.ndarray:
    """Annual mean distance outside every model-derived profile interval."""
    predicted = np.asarray(predicted_temperature, dtype=float)
    people = np.asarray(native_people, dtype=float)
    lower = np.asarray(lower_c, dtype=float)
    upper = np.asarray(upper_c, dtype=float)
    if predicted.ndim != 2 or people.shape != (predicted.shape[1],):
        raise ValueError("Probe prediction and native occupancy shapes differ")
    if lower.ndim != 1 or lower.shape != upper.shape or lower.size == 0:
        raise ValueError("Each occupied binding requires personal intervals")
    if (lower > upper).any() or people.sum() <= 0:
        raise ValueError("Personal interval bounds or occupancy are invalid")
    distance = np.maximum(lower[None, None, :] - predicted[:, :, None], 0.0)
    distance += np.maximum(predicted[:, :, None] - upper[None, None, :], 0.0)
    return np.einsum("ghp,h->gp", distance, people, optimize=True) / people.sum()


def choose_grid_value(grid: np.ndarray, loss: np.ndarray) -> float:
    """Choose a deterministic minimizer, centring only a contiguous plateau."""
    minimum = float(np.min(loss))
    eligible = np.flatnonzero(loss <= minimum + 1e-12)
    if eligible.size == 0:
        raise ValueError("The objective has no finite minimum")
    contiguous = np.all(np.diff(eligible) == 1)
    if contiguous:
        return float((grid[eligible[0]] + grid[eligible[-1]]) / 2.0)
    return float(grid[eligible[0]])


def _binding_objectives(
    members: list[str],
    centre_curves: dict[str, np.ndarray],
    interval_curves: dict[str, np.ndarray],
    native_person_hours: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    centre = np.concatenate([centre_curves[zone] for zone in members], axis=1)
    interval = np.concatenate([interval_curves[zone] for zone in members], axis=1)
    weights = np.concatenate([
        np.full(centre_curves[zone].shape[1], native_person_hours[zone]
                / centre_curves[zone].shape[1])
        for zone in members
    ])
    return (
        np.average(centre, axis=1, weights=weights),
        np.average(interval, axis=1, weights=weights),
    )


def generate_controls(
    input_root: Path,
    *,
    profile_table: pd.DataFrame | None = None,
    permissions: tuple[str, ...] = ("building", "floor", "native_zone"),
    contexts_to_include: set[str] | None = None,
    groups_to_include: set[tuple[int, str]] | None = None,
    offsets_c: tuple[float, ...] = OFFSETS_C,
) -> tuple[list[dict], dict]:
    contexts = json.loads((input_root / "controller/CONTEXTS.json").read_text(encoding="utf-8"))
    designs: list[dict] = []
    diagnostics: dict[str, dict] = {}

    for context, config in contexts.items():
        if contexts_to_include is not None and context not in contexts_to_include:
            continue
        with np.load(input_root / config["probes"], allow_pickle=False) as saved:
            probes = dict(saved)
        if profile_table is None:
            with np.load(input_root / config["profiles"], allow_pickle=False) as saved:
                profiles = pd.DataFrame(dict(saved))
        else:
            profiles = profile_table[profile_table.context.eq(context)].copy()
            if profiles.empty:
                raise ValueError(f"No supplied demand profiles for {context}")

        probe_targets = probes["uniform_target_c"].astype(float)
        grid = np.round(np.arange(probe_targets.min(), probe_targets.max() + 0.00001, 0.1), 8)
        zones = list(config["zones"])
        prediction: dict[str, np.ndarray] = {}
        people: dict[str, np.ndarray] = {}
        for zone_index, zone in enumerate(zones):
            raw = probes["operative_c"][:, :, zone_index]
            prediction[zone] = np.asarray([
                np.interp(grid, probe_targets, raw[:, hour]) for hour in range(raw.shape[1])
            ]).T
            people[zone] = probes["native_people"][:, zone_index].astype(float)

        context_diag = {"grid_min_c": float(grid.min()), "grid_max_c": float(grid.max()),
                        "groups": {}}
        for (composition_id, layout), group in profiles.groupby(["composition_id", "layout"], sort=True):
            key = (int(composition_id), str(layout))
            if groups_to_include is not None and key not in groups_to_include:
                continue
            global_mean = float(group["centre_c"].mean())
            uniform_loss = np.zeros(grid.size)
            total_person_hours = 0.0
            centre_curves: dict[str, np.ndarray] = {}
            interval_curves: dict[str, np.ndarray] = {}
            native_person_hours: dict[str, float] = {}
            for zone in zones:
                local = group[group["zone_id"] == zone]
                if local.empty:
                    raise ValueError(f"{context}/{key}: no profiles assigned to {zone}")
                hours = float(people[zone].sum())
                native_person_hours[zone] = hours
                centre_curves[zone] = profile_mismatch_curves(
                    prediction[zone], people[zone], local["centre_c"].to_numpy(float)
                )
                interval_curves[zone] = profile_interval_curves(
                    prediction[zone], people[zone],
                    local["controller_lower_c"].to_numpy(float),
                    local["controller_upper_c"].to_numpy(float)
                )
                uniform_loss += np.einsum(
                    "gh,h->g", np.abs(prediction[zone] - global_mean), people[zone], optimize=True
                )
                total_person_hours += hours
            uniform_loss /= total_person_hours
            uniform_target = choose_grid_value(grid, uniform_loss)

            group_targets: dict[str, dict[str, float]] = {}
            group_objectives: dict[str, dict[str, float]] = {}
            for permission in permissions:
                bindings = {
                    zone: ("building" if permission == "building" else
                           config["floor_groups"][zone] if permission == "floor" else zone)
                    for zone in zones
                }
                control_targets: dict[str, dict[str, float]] = {
                    "uniform_average": dict.fromkeys(zones, uniform_target),
                    "personal_center": {},
                    "personal_range": {},
                }
                control_objectives: dict[str, list[float]] = {
                    "uniform_average": [float(uniform_loss.min())],
                    "personal_center": [],
                    "personal_range": [],
                }
                for binding in dict.fromkeys(bindings.values()):
                    members = [zone for zone in zones if bindings[zone] == binding]
                    centre, interval = _binding_objectives(
                        members, centre_curves, interval_curves, native_person_hours
                    )
                    centre_target = choose_grid_value(grid, centre)
                    range_target = choose_grid_value(grid, interval)
                    control_targets["personal_center"].update(
                        dict.fromkeys(members, centre_target)
                    )
                    control_targets["personal_range"].update(
                        dict.fromkeys(members, range_target)
                    )
                    control_objectives["personal_center"].append(float(centre.min()))
                    control_objectives["personal_range"].append(float(interval.min()))

                for control in CONTROLS:
                    base_targets = control_targets[control]
                    for variant, offset in enumerate(offsets_c):
                        targets = {
                            zone: float(np.clip(np.round(value + offset, 1), grid.min(), grid.max()))
                            for zone, value in base_targets.items()
                        }
                        designs.append({
                            "context": context,
                            "composition_id": int(composition_id),
                            "layout": str(layout),
                            "permission": permission,
                            "control": control,
                            "variant": int(variant),
                            "offset_c": float(offset),
                            "targets_c": targets,
                            "permission_alias": "building" if permission == "floor"
                            and config["floor_binding_equals_building"] else "",
                        })
                    group_targets[f"{permission}/{control}"] = base_targets
                    group_objectives[f"{permission}/{control}"] = {
                        "minimum_loss_c": float(np.mean(control_objectives[control]))
                    }
            context_diag["groups"][f"{composition_id}/{layout}"] = {
                "global_mean_c": global_mean,
                "base_targets_c": group_targets,
                "objectives": group_objectives,
            }
        diagnostics[context] = context_diag
    return designs, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contexts", nargs="*")
    parser.add_argument("--permission", choices=("building", "floor", "native_zone"))
    parser.add_argument("--composition-id", type=int)
    parser.add_argument("--layout")
    parser.add_argument("--profiles-csv", type=Path)
    args = parser.parse_args()
    groups = None
    if args.composition_id is not None or args.layout is not None:
        if args.composition_id is None or args.layout is None:
            parser.error("--composition-id and --layout must be supplied together")
        groups = {(args.composition_id, args.layout)}
    designs, diagnostics = generate_controls(
        args.input_root.resolve(),
        profile_table=pd.read_csv(args.profiles_csv.resolve()) if args.profiles_csv else None,
        permissions=(args.permission,) if args.permission else ("building", "floor", "native_zone"),
        contexts_to_include=set(args.contexts) if args.contexts else None,
        groups_to_include=groups,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "CONTROL_COMMANDS.json").write_text(
        json.dumps(designs, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "TARGET_DIAGNOSTICS.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "logical_commands": len(designs),
        "controls": list(CONTROLS),
        "offsets_c": list(OFFSETS_C),
        "personal_center_objective": "population_mean_absolute_mismatch",
        "personal_range_objective": "population_mean_distance_outside_interval",
        "minimum_loss_tie_break": "midpoint_of_contiguous_minimum_plateau_else_lowest_minimizer",
        "common_25_candidate_present": False,
    }
    (args.output_root / "GENERATION_QA.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
