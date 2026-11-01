"""Plan and run season-specific demand-command sensitivity cases.

Warm-minus-cool demand shifts are estimated separately from the multinational
and Chinese observation cohorts.  Each source therefore retains its own
observed seasonal contrast.  For every climate, cohort and controller, target
temperatures are optimized independently for the four local meteorological
seasons from the existing uniform-probe response.  The half and full observed
contrasts are physical annual reruns; the zero-contrast case is the existing
offset-zero annual operation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ANALYSIS = HERE.parents[1]
for directory in (
    HERE.parent,
    ANALYSIS / "energy_service",
    ANALYSIS / "seasonal",
    ANALYSIS / "targeted_extension",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from demand_aware_controls import (  # noqa: E402
    _binding_objectives,
    choose_grid_value,
    profile_interval_curves,
    profile_mismatch_curves,
)
from audit_seasonal_solver_stability import (  # noqa: E402
    apply_seasonal_solver_stability_evidence,
    audit_seasonal_solver_stability,
    read_seasonal_solver_stability_evidence,
)
from analyse_global_results import (  # noqa: E402
    DEFAULT_GAS_CO2_KG_KWH_HHV,
    _carbon_table,
    energy_metrics,
    load_hourly,
    profile_zone_positions,
    verify_hourly_hash,
)
from generate_model_inputs import objects, physical_hash, signature  # noqa: E402
from global_matrix import extract_hourly, run_energyplus, sha256  # noqa: E402
from run_global_matrix import (  # noqa: E402
    relative_engineering_qa,
    solver_configuration,
    warning_profile,
)
from seasonal_adaptation import (  # noqa: E402
    estimate_within_place_warm_cool_contrast,
)
from thermal_service import score_service  # noqa: E402


CONTROLS = ("uniform_average", "personal_center", "personal_range")
SOURCE_ALIASES = {
    "ASHRAE": "ASHRAE",
    "Multinational archive": "ASHRAE",
    "CTCD": "CTCD",
    "China archive": "CTCD",
}
BLOCKS = ("DJF", "MAM", "JJA", "SON")
BLOCK_ENDS = (("DJF", "2/28"), ("MAM", "5/31"), ("JJA", "8/31"),
              ("SON", "11/30"), ("DJF", "12/31"))
NORTH = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
SOUTH = {"DJF": "summer", "MAM": "autumn", "JJA": "winter", "SON": "spring"}
MONTH_BLOCK = {
    1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA",
    7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON", 12: "DJF",
}


def derive_source_contrasts(
    ashrae: pd.DataFrame,
    ctcd: pd.DataFrame,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 20261027,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate source-specific warm-minus-cool contrasts within cities."""
    ashrae_frame = ashrae.rename(columns={"demand_centre_c": "centre_c"}).copy()
    ctcd_frame = ctcd.copy()
    for flag in ("main_control_eligible", "active_conditioning"):
        if flag in ctcd_frame:
            values = ctcd_frame[flag]
            if values.dtype != bool:
                values = values.astype(str).str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            ctcd_frame = ctcd_frame.loc[values.fillna(False)].copy()
    specifications = (
        ("ASHRAE", ashrae_frame, ("country", "city"),
         ("summer", "hot/wet"), ("winter", "cool/dry")),
        ("CTCD", ctcd_frame, ("city_normalized",),
         ("summer season",), ("winter season",)),
    )
    summaries: list[dict] = []
    effects: list[pd.DataFrame] = []
    for source, frame, places, warm, cool in specifications:
        summary, local = estimate_within_place_warm_cool_contrast(
            frame,
            place_columns=places,
            warm_labels=warm,
            cool_labels=cool,
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        summaries.append({"demand_source": source, **summary})
        local.insert(0, "demand_source", source)
        effects.append(local)
    return pd.DataFrame(summaries), pd.concat(effects, ignore_index=True)


def local_season(block: str, latitude: float) -> str:
    if block not in BLOCKS or not np.isfinite(latitude):
        raise ValueError("A meteorological block and finite latitude are required")
    return (NORTH if latitude >= 0 else SOUTH)[block]


def demand_shift(contrast_c: float, fraction: float, season: str) -> float:
    if not np.isfinite(contrast_c) or not np.isfinite(fraction) or fraction < 0:
        raise ValueError("Contrast and nonnegative fraction must be finite")
    if season == "summer":
        return 0.5 * contrast_c * fraction
    if season == "winter":
        return -0.5 * contrast_c * fraction
    if season in {"spring", "autumn"}:
        return 0.0
    raise ValueError(f"Unknown local season: {season}")


def _probe_months(probes: Mapping[str, np.ndarray]) -> np.ndarray:
    indices = np.asarray(probes["occupied_source_indices"], dtype=int)
    if indices.ndim != 1 or (indices < 0).any() or (indices >= 8760).any():
        raise ValueError("Probe source indices must lie in a non-leap 8,760-hour year")
    clock = pd.date_range("2001-01-01", periods=8760, freq="h")
    return clock.month.to_numpy(int)[indices]


def optimise_seasonal_targets(
    probes: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    *,
    latitude: float,
    contrast_c: float,
    fraction: float,
) -> tuple[dict[str, dict[str, dict[str, float]]], list[dict]]:
    """Optimize three native-zone controls independently in each season."""
    required = {
        "zone_id", "centre_c", "controller_lower_c", "controller_upper_c"
    }
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Profiles lack fields: {sorted(missing)}")
    targets = np.asarray(probes["uniform_target_c"], dtype=float)
    grid = np.round(np.arange(targets.min(), targets.max() + 0.00001, 0.1), 8)
    operative = np.asarray(probes["operative_c"], dtype=float)
    native_people = np.asarray(probes["native_people"], dtype=float)
    zones = np.asarray(probes["zone_ids"]).astype(str).tolist()
    months = _probe_months(probes)
    if operative.shape[1:] != native_people.shape or operative.shape[1] != len(months):
        raise ValueError("Probe temperatures, occupancy and clock do not align")
    if operative.shape[2] != len(zones):
        raise ValueError("Probe zone axis does not match zone identifiers")

    result: dict[str, dict[str, dict[str, float]]] = {}
    diagnostics: list[dict] = []
    for block in BLOCKS:
        season = local_season(block, latitude)
        shift = demand_shift(contrast_c, fraction, season)
        local_profiles = profiles.copy()
        for column in ("centre_c", "controller_lower_c", "controller_upper_c"):
            local_profiles[column] = pd.to_numeric(local_profiles[column]) + shift
        selected = np.asarray([MONTH_BLOCK[int(month)] == block for month in months])
        if not selected.any():
            raise ValueError(f"No occupied probe hours remain in {block}")

        prediction: dict[str, np.ndarray] = {}
        people: dict[str, np.ndarray] = {}
        centre_curves: dict[str, np.ndarray] = {}
        interval_curves: dict[str, np.ndarray] = {}
        native_person_hours: dict[str, float] = {}
        total_person_hours = 0.0
        global_mean = float(local_profiles.centre_c.mean())
        uniform_loss = np.zeros(len(grid))
        for zone_index, zone in enumerate(zones):
            raw = operative[:, selected, zone_index]
            prediction[zone] = np.asarray([
                np.interp(grid, targets, raw[:, hour]) for hour in range(raw.shape[1])
            ]).T
            people[zone] = native_people[selected, zone_index]
            local = local_profiles[local_profiles.zone_id.astype(str).eq(zone)]
            if local.empty:
                raise ValueError(f"No demand profiles are assigned to {zone}")
            hours = float(people[zone].sum())
            if hours <= 0:
                raise ValueError(f"No occupied weight remains for {zone}/{block}")
            native_person_hours[zone] = hours
            centre_curves[zone] = profile_mismatch_curves(
                prediction[zone], people[zone], local.centre_c.to_numpy(float)
            )
            interval_curves[zone] = profile_interval_curves(
                prediction[zone], people[zone],
                local.controller_lower_c.to_numpy(float),
                local.controller_upper_c.to_numpy(float),
            )
            uniform_loss += np.einsum(
                "gh,h->g",
                np.abs(prediction[zone] - global_mean),
                people[zone],
                optimize=True,
            )
            total_person_hours += hours
        uniform_loss /= total_person_hours
        block_targets = {
            "uniform_average": dict.fromkeys(
                zones, choose_grid_value(grid, uniform_loss)
            ),
            "personal_center": {},
            "personal_range": {},
        }
        for zone in zones:
            centre, interval = _binding_objectives(
                [zone], centre_curves, interval_curves, native_person_hours
            )
            block_targets["personal_center"][zone] = choose_grid_value(grid, centre)
            block_targets["personal_range"][zone] = choose_grid_value(grid, interval)
        result[block] = block_targets
        grid_min = float(grid.min())
        grid_max = float(grid.max())
        for control, zone_targets in block_targets.items():
            values = np.asarray(list(zone_targets.values()), dtype=float)
            at_min = np.isclose(values, grid_min, rtol=0, atol=1e-9)
            at_max = np.isclose(values, grid_max, rtol=0, atol=1e-9)
            at_boundary = at_min | at_max
            diagnostics.append({
                "meteorological_block": block,
                "local_season": season,
                "control": control,
                "demand_shift_c": shift,
                "occupied_probe_hours": int(selected.sum()),
                "native_person_hours": total_person_hours,
                "grid_min_c": grid_min,
                "grid_max_c": grid_max,
                "zone_count": int(len(values)),
                "target_min_c": float(values.min()),
                "target_max_c": float(values.max()),
                "zones_at_grid_min": int(at_min.sum()),
                "zones_at_grid_max": int(at_max.sum()),
                "zones_at_grid_boundary": int(at_boundary.sum()),
                "share_at_grid_boundary": float(at_boundary.mean()),
            })
    return result, diagnostics


def _shift_schedule_fields(fields: Sequence[str], target: float, old: float) -> list[str]:
    shifted = list(fields)
    eligible = False
    for index in range(len(shifted)):
        token = shifted[index]
        if token.lower().startswith("for:"):
            eligible = any(
                day in token.split(":", 1)[1].lower().split()
                for day in ("weekdays", "saturday")
            )
        elif (
            eligible
            and index > 0
            and shifted[index - 1].lower().startswith("until:")
            and abs(float(token) - old) < 1e-7
        ):
            shifted[index] = str(float(token) + float(target) - 25.0)
    return shifted


def seasonal_annual_input(
    text: str,
    targets_by_block: Mapping[str, Mapping[str, float]],
    half_deadband_c: float,
) -> str:
    """Write four seasonal thermostat commands and preserve all other physics."""
    if set(targets_by_block) != set(BLOCKS):
        raise ValueError("Seasonal targets must contain DJF, MAM, JJA and SON")
    parsed = objects(text)
    duals = {
        item[1]: item for item in parsed
        if item[0] == "ThermostatSetpoint:DualSetpoint"
    }
    schedule_links: dict[str, list[tuple[str, float]]] = {}
    for thermostat in [item for item in parsed if item[0] == "ZoneControl:Thermostat"]:
        zone = thermostat[2]
        dual = duals[thermostat[5]]
        for field in (2, 3):
            old = 25 + (1 if field == 3 else -1) * half_deadband_c
            schedule_links.setdefault(dual[field], []).append((zone, old))
    controlled_zones = {zone for links in schedule_links.values() for zone, _ in links}
    for block, values in targets_by_block.items():
        if set(values) != controlled_zones:
            raise ValueError(f"{block} targets do not match controlled zones")

    for item in parsed:
        if item[0] != "Schedule:Compact" or item[1] not in schedule_links:
            continue
        if item[3].lower() != "through: 12/31" or sum(
            token.lower().startswith("through:") for token in item
        ) != 1:
            raise ValueError("Expected one original annual thermostat block")
        original = item[4:]
        expanded = item[:3]
        linked = schedule_links[item[1]]
        old_values = {old for _, old in linked}
        if len(old_values) != 1:
            raise ValueError("A thermostat schedule mixes heating and cooling baselines")
        old = next(iter(old_values))
        for block, end_date in BLOCK_ENDS:
            block_values = {float(targets_by_block[block][zone]) for zone, _ in linked}
            if len(block_values) != 1:
                raise ValueError("A shared thermostat schedule received conflicting targets")
            expanded += [f"Through: {end_date}"] + _shift_schedule_fields(
                original, next(iter(block_values)), old
            )
        item[:] = expanded
    return "\n".join(",".join(item) + ";" for item in parsed)


def _hash_payload(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _climate_rows(path: Path) -> pd.DataFrame:
    content = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(content["climates"] if isinstance(content, dict) else content)


def build_plan(
    models_path: Path,
    profiles_path: Path,
    probe_root: Path,
    climates_path: Path,
    contrasts_path: Path,
    output_root: Path,
    *,
    fractions: Sequence[float] = (0.5, 1.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the deterministic 324-run seasonal-command extension."""
    models = json.loads(models_path.read_text(encoding="utf-8"))
    profiles = pd.read_csv(profiles_path)
    climates = _climate_rows(climates_path).set_index("climate")
    contrasts = pd.read_csv(contrasts_path).set_index("demand_source")
    fractions = tuple(float(value) for value in fractions)
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("Physical seasonal-command fractions must lie in (0, 1]")
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = output_root / "inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    for (climate, composition), group in profiles.groupby(
        ["context", "composition_id"], sort=True
    ):
        climate = str(climate)
        source_values = group.demand_source.astype(str).unique()
        if len(source_values) != 1:
            raise ValueError(f"{climate}/{composition} mixes demand sources")
        source_label = source_values[0]
        source = SOURCE_ALIASES.get(source_label)
        if source is None or source not in contrasts.index:
            raise ValueError(f"Missing seasonal contrast for {source_label}")
        contrast = float(contrasts.loc[source, "warm_minus_cool_c"])
        latitude = float(climates.loc[climate, "latitude"])
        probe_path = probe_root / "probes" / f"{climate}_probes.npz"
        with np.load(probe_path, allow_pickle=False) as stored:
            probes = dict(stored)
        model = models[climate]
        source_idf = Path(model["prepared_idf"])
        source_text = source_idf.read_text(encoding="utf-8", errors="replace")
        for fraction in fractions:
            fraction = float(fraction)
            targets, target_diagnostics = optimise_seasonal_targets(
                probes,
                group,
                latitude=latitude,
                contrast_c=contrast,
                fraction=fraction,
            )
            for item in target_diagnostics:
                diagnostics.append({
                    "climate": climate,
                    "composition_id": int(composition),
                    "cohort_label": group.cohort_label.iloc[0],
                    "demand_source": source_label,
                    "contrast_source": source,
                    "warm_minus_cool_c": contrast,
                    "seasonal_contrast_fraction": fraction,
                    **item,
                })
            for control in CONTROLS:
                by_block = {block: targets[block][control] for block in BLOCKS}
                text = seasonal_annual_input(
                    source_text, by_block, float(model["half_deadband_c"])
                )
                if physical_hash(text) != physical_hash(source_text):
                    raise ValueError("Seasonal command changed non-control building physics")
                payload = {
                    "climate": climate,
                    "composition_id": int(composition),
                    "control": control,
                    "seasonal_contrast_fraction": fraction,
                    "warm_minus_cool_c": contrast,
                    "targets": by_block,
                    "prepared_idf_sha256": model["prepared_idf_sha256"],
                    "weather_sha256": model["weather_sha256"],
                }
                run_id = "seasonal_" + _hash_payload(payload)
                input_path = input_root / f"{run_id}.idf"
                input_path.write_text(text, encoding="utf-8")
                rows.append({
                    "run_id": run_id,
                    "climate": climate,
                    "composition_id": int(composition),
                    "cohort_label": group.cohort_label.iloc[0],
                    "demand_source": source_label,
                    "contrast_source": source,
                    "control": control,
                    "seasonal_contrast_fraction": fraction,
                    "warm_minus_cool_c": contrast,
                    "input_idf_path": str(input_path.resolve()),
                    "weather_path": str(Path(model["weather_path"]).resolve()),
                    "targets_json": json.dumps(by_block, sort_keys=True),
                    "normalized_idf_sha256": signature(text),
                    "physical_fingerprint_unchanged": True,
                })
    frame = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "seasonal_contrast_fraction"]
    )
    expected = len(models) * profiles.composition_id.nunique() * len(CONTROLS) * len(fractions)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} seasonal runs, found {len(frame)}")
    frame.to_csv(output_root / "SEASONAL_COMMAND_RUN_MANIFEST.csv", index=False)
    diagnostic_frame = pd.DataFrame(diagnostics).sort_values(
        [
            "climate", "composition_id", "seasonal_contrast_fraction",
            "meteorological_block", "control",
        ]
    )
    diagnostic_frame.to_csv(output_root / "SEASONAL_TARGET_DIAGNOSTICS.csv", index=False)
    return frame, diagnostic_frame


def require_complete_seasonal_execution(
    result: pd.DataFrame, expected_runs: int
) -> None:
    """Reject an incomplete or engineering-failed seasonal batch after audit output."""
    required = {"run_id", "completed", "relative_engineering_pass"}
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"Seasonal execution audit lacks {sorted(missing)}")

    def flags(column: str) -> pd.Series:
        values = result[column]
        if values.dtype == bool:
            return values
        parsed = values.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if parsed.isna().any():
            raise RuntimeError(f"Seasonal execution has invalid {column} flags")
        return parsed.astype(bool)

    completed = flags("completed")
    qualified = flags("relative_engineering_pass")
    invalid = result.loc[~completed | ~qualified, "run_id"].astype(str).tolist()
    if len(result) != expected_runs or result.run_id.duplicated().any() or invalid:
        raise RuntimeError(
            "Seasonal-command execution failed its completeness/engineering gate: "
            f"expected={expected_runs}, observed={len(result)}, failed={invalid}"
        )


def execute_plan(
    runs: pd.DataFrame,
    probe_manifest: pd.DataFrame,
    probe_run_root: Path,
    output_root: Path,
    runtime: Path,
    *,
    workers: int = 8,
    solver_stability_manifest: Path | None = None,
) -> pd.DataFrame:
    """Run seasonal-command cases with baseline-relative engineering QA."""
    references = probe_manifest[np.isclose(
        pd.to_numeric(probe_manifest.target_c), 25.0
    )].set_index("climate")
    if references.index.duplicated().any():
        raise ValueError("Expected one 25 C probe per climate")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_hash = sha256(runtime)

    def execute(row: dict) -> dict:
        run_id = str(row["run_id"])
        folder = output_root / "runs" / run_id
        compact = output_root / "hourly" / f"{run_id}.npz"
        qa_path = folder / "RUN_QA.json"
        input_path = Path(row["input_idf_path"])
        weather_path = Path(row["weather_path"])
        if qa_path.is_file() and compact.is_file():
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            if (
                qa.get("hourly_sha256") == sha256(compact)
                and qa.get("runtime_sha256") == runtime_hash
                and qa.get("input_idf_sha256") == sha256(input_path)
                and qa.get("weather_sha256") == sha256(weather_path)
            ):
                if "relative_engineering_pass" not in qa:
                    reference_id = str(references.loc[row["climate"], "run_id"])
                    baseline_error = probe_run_root / "runs" / reference_id / "eplusout.err"
                    relative = relative_engineering_qa(
                        qa,
                        warning_profile(folder / "eplusout.err"),
                        warning_profile(baseline_error),
                    )
                    qa.update(relative)
                    qa["relative_engineering_pass"] = relative[
                        "engineering_pass_relative_to_prototype_baseline"
                    ]
                    qa_path.write_text(
                        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
                    )
                return row | qa | {
                    "hourly_path": str(compact.resolve()),
                    "qa_path": str(qa_path.resolve()),
                }
        qa = run_energyplus(
            input_path, weather_path, runtime,
            folder, compact,
            list(json.loads(row["targets_json"])["DJF"]),
            cleanup_raw=True,
        )
        reference_id = str(references.loc[row["climate"], "run_id"])
        baseline_error = probe_run_root / "runs" / reference_id / "eplusout.err"
        relative = relative_engineering_qa(
            qa,
            warning_profile(folder / "eplusout.err"),
            warning_profile(baseline_error),
        )
        qa.update(relative)
        qa["relative_engineering_pass"] = relative[
            "engineering_pass_relative_to_prototype_baseline"
        ]
        qa.update({
            "run_id": run_id,
            "climate": row["climate"],
            "composition_id": row["composition_id"],
            "control": row["control"],
            "seasonal_contrast_fraction": row["seasonal_contrast_fraction"],
            "hourly_path": str(compact.resolve()),
            "qa_path": str(qa_path.resolve()),
        })
        qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        return row | qa

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(execute, row) for row in runs.to_dict("records")]
        for future in as_completed(pending):
            completed.append(future.result())
    stability, stability_hash = read_seasonal_solver_stability_evidence(solver_stability_manifest)
    used = set()
    for row in completed:
        run_id = str(row["run_id"])
        folder = output_root / "runs" / run_id
        qa_path = folder / "RUN_QA.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        input_path = Path(row["input_idf_path"])
        compact = Path(row["hourly_path"])
        if qa["input_idf_sha256"] != sha256(input_path) or qa["hourly_sha256"] != sha256(compact):
            raise ValueError(f"Seasonal primary evidence changed: {run_id}")
        reference_id = str(references.loc[row["climate"], "run_id"])
        relative = relative_engineering_qa(
            qa, warning_profile(folder / "eplusout.err"),
            warning_profile(probe_run_root / "runs" / reference_id / "eplusout.err"),
        )
        passed, audit = apply_seasonal_solver_stability_evidence(
            relative, qa, stability.get(run_id),
            current_maximum_hvac_iterations=int(solver_configuration(input_path.read_text())["maximum_hvac_iterations"]),
            active_input_sha256=sha256(input_path), active_hourly_sha256=sha256(compact),
            active_error_sha256=sha256(folder / "eplusout.err"),
        )
        if audit["seasonal_solver_stability_exception_applied"]:
            used.add(run_id)
        qa.update(audit)
        qa["relative_engineering_pass"] = passed
        qa["seasonal_solver_stability_evidence_sha256"] = stability_hash if run_id in used else None
        qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        row.update(qa)
    if set(stability) != used:
        raise ValueError("Seasonal stability evidence includes stale or unused runs")
    result = pd.DataFrame(completed).sort_values(
        ["climate", "composition_id", "control", "seasonal_contrast_fraction"]
    )
    result.to_csv(output_root / "SEASONAL_COMMAND_RUN_MANIFEST_COMPLETED.csv", index=False)
    require_complete_seasonal_execution(result, len(runs))
    return result


def recover_existing_plan(
    runs: pd.DataFrame,
    probe_manifest: pd.DataFrame,
    probe_run_root: Path,
    output_root: Path,
    runtime: Path,
    *,
    workers: int = 4,
    cleanup_raw: bool = True,
    stability_manifest: Path | None = None,
) -> pd.DataFrame:
    """Recover compact records from completed raw seasonal EnergyPlus outputs."""
    if runs.run_id.duplicated().any():
        raise ValueError("Seasonal recovery manifest duplicates run_id")
    references = probe_manifest[np.isclose(
        pd.to_numeric(probe_manifest.target_c), 25.0
    )].set_index("climate")
    if references.index.duplicated().any():
        raise ValueError("Expected one 25 C probe per climate")
    runtime = runtime.resolve()
    runtime_hash = sha256(runtime)
    stability, stability_hash = read_seasonal_solver_stability_evidence(
        stability_manifest
    )
    used_stability_rows: set[str] = set()

    required_raw = ("eplusout.csv", "eplusout.sql", "eplusout.err")
    for row in runs.to_dict("records"):
        folder = output_root / "runs" / str(row["run_id"])
        missing = [name for name in required_raw if not (folder / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Raw seasonal run {row['run_id']} lacks {missing}"
            )
        error = (folder / "eplusout.err").read_text(errors="replace")
        if "EnergyPlus Completed Successfully" not in error:
            raise RuntimeError(f"Seasonal run did not complete: {row['run_id']}")

    def recover(row: dict) -> dict:
        run_id = str(row["run_id"])
        folder = output_root / "runs" / run_id
        compact = output_root / "hourly" / f"{run_id}.npz"
        qa_path = folder / "RUN_QA.json"
        input_path = Path(row["input_idf_path"])
        weather_path = Path(row["weather_path"])
        zones = list(json.loads(row["targets_json"])["DJF"].keys())
        input_hash = sha256(input_path)
        weather_hash = sha256(weather_path)
        sql_hash = sha256(folder / "eplusout.sql")
        error_hash = sha256(folder / "eplusout.err")
        existing: dict = {}
        if qa_path.is_file() and compact.is_file():
            existing = json.loads(qa_path.read_text(encoding="utf-8"))
        reusable = bool(
            existing
            and existing.get("input_idf_sha256") == input_hash
            and existing.get("weather_sha256") == weather_hash
            and existing.get("runtime_sha256") == runtime_hash
            and existing.get("raw_sql_sha256") == sql_hash
            and existing.get("hourly_sha256") == sha256(compact)
        )
        if reusable:
            qa = existing.copy()
            for name in list(qa):
                if name.startswith("seasonal_solver_stability_"):
                    qa.pop(name)
            qa.pop("raw_relative_engineering_pass", None)
        else:
            qa = {
                "exit_code": 0,
                "seconds": None,
                "input_idf_sha256": input_hash,
                "weather_sha256": weather_hash,
                "runtime_sha256": runtime_hash,
                "compact_output": str(compact.resolve()),
                "recovered_from_existing_raw_output": True,
                "raw_sql_sha256": sql_hash,
            }
            qa.update(extract_hourly(folder, zones, compact))
        reference_id = str(references.loc[row["climate"], "run_id"])
        baseline_error = probe_run_root / "runs" / reference_id / "eplusout.err"
        relative = relative_engineering_qa(
            qa,
            warning_profile(folder / "eplusout.err"),
            warning_profile(baseline_error),
        )
        qa.update(relative)
        evidence = stability.get(run_id)
        maximum_iterations = -1
        if evidence is not None:
            maximum_iterations = int(solver_configuration(
                input_path.read_text(encoding="utf-8", errors="replace")
            )["maximum_hvac_iterations"])
            expected_identity = {
                "climate": str(row["climate"]),
                "composition_id": int(row["composition_id"]),
                "control": str(row["control"]),
            }
            for name, value in expected_identity.items():
                if str(evidence[name]) != str(value):
                    raise ValueError(
                        f"Seasonal stability evidence changed {name} for {run_id}"
                    )
            if not np.isclose(
                float(evidence["seasonal_contrast_fraction"]),
                float(row["seasonal_contrast_fraction"]),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Seasonal stability evidence changed the contrast fraction "
                    f"for {run_id}"
                )
            used_stability_rows.add(run_id)
        passed, stability_audit = apply_seasonal_solver_stability_evidence(
            relative,
            qa,
            evidence,
            current_maximum_hvac_iterations=maximum_iterations,
            active_input_sha256=input_hash,
            active_hourly_sha256=sha256(compact),
            active_error_sha256=error_hash,
        )
        qa.update(stability_audit)
        qa["relative_engineering_pass"] = passed
        qa["seasonal_solver_stability_evidence_sha256"] = (
            stability_hash
            if stability_audit["seasonal_solver_stability_exception_applied"]
            else None
        )
        qa.update({
            "run_id": run_id,
            "climate": row["climate"],
            "composition_id": row["composition_id"],
            "control": row["control"],
            "seasonal_contrast_fraction": row["seasonal_contrast_fraction"],
            "hourly_path": str(compact.resolve()),
            "qa_path": str(qa_path.resolve()),
        })
        temporary = qa_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        temporary.replace(qa_path)
        return row | qa

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(recover, row) for row in runs.to_dict("records")]
        for future in as_completed(pending):
            completed.append(future.result())
    result = pd.DataFrame(completed).sort_values(
        ["climate", "composition_id", "control", "seasonal_contrast_fraction"]
    )
    unused_stability_rows = set(stability) - used_stability_rows
    if unused_stability_rows:
        raise ValueError(
            "Seasonal solver-stability evidence is stale or does not correspond "
            f"to an active run: {sorted(unused_stability_rows)}"
        )
    recovery_audit_path = output_root / "SEASONAL_COMMAND_RECOVERY_AUDIT.csv"
    temporary_audit = recovery_audit_path.with_suffix(".csv.tmp")
    result.to_csv(temporary_audit, index=False)
    temporary_audit.replace(recovery_audit_path)
    require_complete_seasonal_execution(result, len(runs))
    completed_path = output_root / "SEASONAL_COMMAND_RUN_MANIFEST_COMPLETED.csv"
    temporary_manifest = completed_path.with_suffix(".csv.tmp")
    result.to_csv(temporary_manifest, index=False)
    temporary_manifest.replace(completed_path)

    if cleanup_raw:
        retained_names = {
            "CONSOLE.log", "RUN_QA.json", "eplusout.err", "eplusout.eio",
            "eplusout.audit",
        }
        for run_id in result.run_id.astype(str):
            folder = output_root / "runs" / run_id
            for path in folder.iterdir():
                if path.is_file() and path.name not in retained_names:
                    path.unlink()
    return result


def _shifted_profiles(profiles: pd.DataFrame, shift_c: float) -> pd.DataFrame:
    shifted = profiles.copy()
    for column in ("centre_c", "controller_lower_c", "controller_upper_c"):
        shifted[column] = pd.to_numeric(shifted[column]) + shift_c
    return shifted


def _block_score(
    data: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    mask: np.ndarray,
    fixed_tail: np.ndarray,
) -> dict:
    weights = np.asarray(data["people"])[mask] * np.asarray(data["occupied"])[
        mask, None
    ].astype(bool)
    return score_service(
        np.asarray(data["temperatures"])[mask],
        weights,
        profile_zone_positions(data, profiles),
        profiles.centre_c.to_numpy(float),
        profiles.controller_lower_c.to_numpy(float),
        profiles.controller_upper_c.to_numpy(float),
        fixed_tail,
    )


def fixed_tail_for_seasonal_demand(
    reference: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    *,
    latitude: float,
    contrast_c: float,
    fraction: float,
    tail_fraction: float = 0.10,
) -> np.ndarray:
    """Fix the annual least-served group from the seasonal-demand 25 C anchor."""
    if not 0 < tail_fraction <= 1:
        raise ValueError("Tail fraction must lie in (0, 1]")
    numerator = np.zeros(len(profiles))
    denominator = np.zeros(len(profiles))
    months = np.asarray(reference["month"], dtype=int)
    dummy_tail = np.arange(len(profiles), dtype=int)
    for block in BLOCKS:
        season = local_season(block, latitude)
        shifted = _shifted_profiles(
            profiles, demand_shift(contrast_c, fraction, season)
        )
        mask = np.asarray([MONTH_BLOCK[int(month)] == block for month in months])
        scored = _block_score(reference, shifted, mask, dummy_tail)["profiles"]
        weight = scored["native_person_hour_weight"]
        numerator += scored["reference_mismatch_c"] * weight
        denominator += weight
    mismatch = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )
    if not np.isfinite(mismatch).all():
        raise ValueError("Seasonal reference did not serve every demand profile")
    ranked = pd.DataFrame({
        "position": np.arange(len(profiles)),
        "profile_id": profiles.profile_id.astype(str),
        "mismatch": mismatch,
    }).sort_values(["mismatch", "profile_id"], ascending=[False, True], kind="stable")
    count = max(1, int(np.ceil(tail_fraction * len(profiles))))
    return np.sort(ranked.head(count).position.to_numpy(int))


def score_seasonal_trajectory(
    data: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    *,
    latitude: float,
    contrast_c: float,
    fraction: float,
    fixed_tail: np.ndarray,
) -> tuple[dict, list[dict]]:
    """Score one physical trajectory against season-shifted demand coordinates."""
    months = np.asarray(data["month"], dtype=int)
    total_weight = 0.0
    mismatch_sum = 0.0
    coverage_sum = 0.0
    excess_sum = 0.0
    profile_numerator = np.zeros(len(profiles))
    profile_denominator = np.zeros(len(profiles))
    seasonal: list[dict] = []
    for block in BLOCKS:
        season = local_season(block, latitude)
        shift = demand_shift(contrast_c, fraction, season)
        shifted = _shifted_profiles(profiles, shift)
        mask = np.asarray([MONTH_BLOCK[int(month)] == block for month in months])
        scored = _block_score(data, shifted, mask, fixed_tail)
        summary = scored["summary"]
        per_profile = scored["profiles"]
        weight = float(summary["occupied_native_person_hours"])
        total_weight += weight
        mismatch_sum += weight * float(summary["mean_reference_mismatch_c"])
        coverage_sum += weight * float(summary["modeled_interval_person_hour_coverage"])
        excess_sum += weight * float(summary["modeled_interval_mean_excess_c"])
        pweight = per_profile["native_person_hour_weight"]
        profile_numerator += per_profile["reference_mismatch_c"] * pweight
        profile_denominator += pweight
        seasonal.append({
            "meteorological_block": block,
            "local_season": season,
            "demand_shift_c": shift,
            **summary,
            **energy_metrics(data, mask=mask),
        })
    profile_mismatch = np.divide(
        profile_numerator,
        profile_denominator,
        out=np.full_like(profile_numerator, np.nan),
        where=profile_denominator > 0,
    )
    if total_weight <= 0 or not np.isfinite(profile_mismatch).all():
        raise ValueError("Seasonal scoring lacks complete occupied demand support")
    annual = {
        "occupied_native_person_hours": total_weight,
        "mean_reference_mismatch_c": mismatch_sum / total_weight,
        "fixed_tail_reference_mismatch_c": float(profile_mismatch[fixed_tail].mean()),
        "modeled_interval_person_hour_coverage": coverage_sum / total_weight,
        "modeled_interval_mean_excess_c": excess_sum / total_weight,
        **energy_metrics(data),
    }
    return annual, seasonal


def score_completed_plan(
    completed_path: Path,
    profiles_path: Path,
    climates_path: Path,
    contrasts_path: Path,
    reference_manifest_path: Path,
    output_root: Path,
    carbon_path: Path | None = None,
    gas_factor: float = DEFAULT_GAS_CO2_KG_KWH_HHV,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score all engineering-qualified seasonal-command trajectories."""
    runs = pd.read_csv(completed_path)
    if "hourly_sha256" not in runs:
        raise ValueError("Seasonal run manifest lacks hourly_sha256")
    profiles = pd.read_csv(profiles_path)
    climates = _climate_rows(climates_path).set_index("climate")
    contrasts = pd.read_csv(contrasts_path).set_index("demand_source")
    fractions = pd.to_numeric(runs.seasonal_contrast_fraction).dropna().unique()
    profile_groups = profiles[["context", "composition_id"]].drop_duplicates()
    expected_runs = len(profile_groups) * len(CONTROLS) * len(fractions)
    require_complete_seasonal_execution(runs, expected_runs)
    references = pd.read_csv(reference_manifest_path)
    references = references[np.isclose(pd.to_numeric(references.target_c), 25.0)]
    if references.climate.duplicated().any():
        raise ValueError("Expected one 25 C trajectory per climate")
    if "hourly_sha256" not in references:
        raise ValueError("Seasonal reference manifest lacks hourly_sha256")
    reference_paths = references.set_index("climate").hourly_path.to_dict()
    reference_hashes = references.set_index("climate").hourly_sha256.to_dict()
    carbon_path = (
        HERE.parents[3] / "data" / "metadata"
        / "global_electricity_carbon_factors_ember_2025.csv"
        if carbon_path is None else carbon_path
    )
    carbon = _carbon_table(carbon_path).set_index("climate")
    output_root.mkdir(parents=True, exist_ok=True)
    annual_rows: list[dict] = []
    seasonal_rows: list[dict] = []
    tail_rows: list[dict] = []
    reference_cache: dict[str, dict[str, np.ndarray]] = {}
    tail_cache: dict[tuple[str, int, float], np.ndarray] = {}
    for row in runs.to_dict("records"):
        climate = str(row["climate"])
        composition = int(row["composition_id"])
        fraction = float(row["seasonal_contrast_fraction"])
        group = profiles[
            profiles.context.astype(str).eq(climate)
            & profiles.composition_id.astype(int).eq(composition)
        ].sort_values("profile_id", kind="stable").reset_index(drop=True)
        if group.empty:
            raise ValueError(f"Missing profiles for {climate}/{composition}")
        source_label = group.demand_source.astype(str).unique()
        if len(source_label) != 1:
            raise ValueError(f"Mixed demand sources for {climate}/{composition}")
        contrast_source = SOURCE_ALIASES.get(source_label[0])
        if contrast_source is None:
            raise ValueError(f"Unknown demand source: {source_label[0]}")
        contrast = float(contrasts.loc[contrast_source, "warm_minus_cool_c"])
        if climate not in reference_cache:
            reference_path = Path(reference_paths[climate])
            verify_hourly_hash(
                reference_path, reference_hashes[climate],
                f"Seasonal reference trajectory for {climate}",
            )
            reference_cache[climate] = load_hourly(reference_path)
        key = (climate, composition, fraction)
        if key not in tail_cache:
            tail_cache[key] = fixed_tail_for_seasonal_demand(
                reference_cache[climate], group,
                latitude=float(climates.loc[climate, "latitude"]),
                contrast_c=contrast, fraction=fraction,
            )
            for position in tail_cache[key]:
                tail_rows.append({
                    "climate": climate,
                    "composition_id": composition,
                    "seasonal_contrast_fraction": fraction,
                    "profile_id": group.profile_id.iloc[position],
                })
        trajectory_path = Path(row["hourly_path"])
        verify_hourly_hash(
            trajectory_path, row["hourly_sha256"],
            f"Seasonal trajectory {row['run_id']}",
        )
        data = load_hourly(trajectory_path, require_facility=False)
        annual, seasons = score_seasonal_trajectory(
            data, group,
            latitude=float(climates.loc[climate, "latitude"]),
            contrast_c=contrast,
            fraction=fraction,
            fixed_tail=tail_cache[key],
        )
        factors = carbon.loc[climate]

        def add_annual_carbon(metrics: dict) -> dict:
            return metrics | {
                "conditioning_operational_co2_country_kg": (
                    metrics["hvac_electric_kwh"]
                    * factors.electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
                "conditioning_operational_co2_common_factor_kg": (
                    metrics["hvac_electric_kwh"]
                    * factors.common_electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
                "operational_co2_country_kg": (
                    metrics["facility_electric_kwh"]
                    * factors.electricity_co2_kg_kwh
                    + metrics["facility_gas_hhv_kwh"] * gas_factor
                ),
                "operational_co2_common_factor_kg": (
                    metrics["facility_electric_kwh"]
                    * factors.common_electricity_co2_kg_kwh
                    + metrics["facility_gas_hhv_kwh"] * gas_factor
                ),
            }

        def add_seasonal_carbon(metrics: dict) -> dict:
            return metrics | {
                "conditioning_operational_co2_country_kg": (
                    metrics["hvac_electric_kwh"]
                    * factors.electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
                "conditioning_operational_co2_common_factor_kg": (
                    metrics["hvac_electric_kwh"]
                    * factors.common_electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
                "facility_electric_plus_space_heating_gas_operational_co2_country_kg": (
                    metrics["facility_electric_kwh"]
                    * factors.electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
                "facility_electric_plus_space_heating_gas_operational_co2_common_factor_kg": (
                    metrics["facility_electric_kwh"]
                    * factors.common_electricity_co2_kg_kwh
                    + metrics["gas_hhv_kwh"] * gas_factor
                ),
            }

        annual = add_annual_carbon(annual)
        seasons = [add_seasonal_carbon(item) for item in seasons]
        identifiers = {
            "run_id": row["run_id"], "climate": climate,
            "composition_id": composition, "cohort_label": row["cohort_label"],
            "demand_source": source_label[0], "control": row["control"],
            "seasonal_contrast_fraction": fraction,
            "warm_minus_cool_c": contrast,
        }
        annual_rows.append(identifiers | annual)
        seasonal_rows.extend(identifiers | item for item in seasons)
    annual_frame = pd.DataFrame(annual_rows)
    seasonal_frame = pd.DataFrame(seasonal_rows)
    annual_frame.to_csv(output_root / "SEASONAL_COMMAND_ANNUAL_METRICS.csv", index=False)
    seasonal_frame.to_csv(output_root / "SEASONAL_COMMAND_SEASON_METRICS.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(
        output_root / "SEASONAL_COMMAND_FIXED_TAIL_MEMBERSHIPS.csv", index=False
    )
    return annual_frame, seasonal_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    contrast = sub.add_parser("contrasts")
    contrast.add_argument("--ashrae", type=Path, required=True)
    contrast.add_argument("--ctcd", type=Path, required=True)
    contrast.add_argument("--output-root", type=Path, required=True)
    contrast.add_argument("--bootstrap-replicates", type=int, default=2000)
    contrast.add_argument("--seed", type=int, default=20261027)

    plan = sub.add_parser("plan")
    plan.add_argument("--models", type=Path, required=True)
    plan.add_argument("--profiles", type=Path, required=True)
    plan.add_argument("--probe-root", type=Path, required=True)
    plan.add_argument("--climates", type=Path, required=True)
    plan.add_argument("--contrasts", type=Path, required=True)
    plan.add_argument("--fractions", nargs="+", type=float, default=[0.5, 1.0])
    plan.add_argument("--output-root", type=Path, required=True)

    execute = sub.add_parser("execute")
    execute.add_argument("--run-manifest", type=Path, required=True)
    execute.add_argument("--probe-manifest", type=Path, required=True)
    execute.add_argument("--probe-run-root", type=Path, required=True)
    execute.add_argument("--runtime", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--workers", type=int, default=8)

    recover = sub.add_parser("recover")
    recover.add_argument("--run-manifest", type=Path, required=True)
    recover.add_argument("--probe-manifest", type=Path, required=True)
    recover.add_argument("--probe-run-root", type=Path, required=True)
    recover.add_argument("--runtime", type=Path, required=True)
    recover.add_argument("--output-root", type=Path, required=True)
    recover.add_argument("--workers", type=int, default=4)
    recover.add_argument("--retain-raw", action="store_true")
    recover.add_argument("--stability-manifest", type=Path)

    stability = sub.add_parser("audit-stability")
    stability.add_argument("--run-manifest", type=Path, required=True)
    stability.add_argument("--runtime", type=Path, required=True)
    stability.add_argument("--primary-root", type=Path, required=True)
    stability.add_argument("--output-root", type=Path, required=True)
    stability.add_argument("--first-iterations", type=int, default=40)
    stability.add_argument("--higher-iterations", type=int, default=100)
    stability.add_argument("--workers", type=int, default=4)
    stability.add_argument("--temperature-tolerance-c", type=float, default=0.01)
    stability.add_argument("--energy-relative-tolerance", type=float, default=0.001)

    score = sub.add_parser("score")
    score.add_argument("--completed-manifest", type=Path, required=True)
    score.add_argument("--profiles", type=Path, required=True)
    score.add_argument("--climates", type=Path, required=True)
    score.add_argument("--contrasts", type=Path, required=True)
    score.add_argument("--reference-manifest", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument(
        "--carbon-factors", type=Path,
        default=HERE.parents[3] / "data" / "metadata"
        / "global_electricity_carbon_factors_ember_2025.csv",
    )
    score.add_argument(
        "--gas-co2-kg-kwh-hhv", type=float,
        default=DEFAULT_GAS_CO2_KG_KWH_HHV,
    )
    args = parser.parse_args()

    if args.command == "contrasts":
        summary, effects = derive_source_contrasts(
            pd.read_csv(args.ashrae),
            pd.read_csv(args.ctcd),
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_root / "SEASONAL_SOURCE_CONTRASTS.csv", index=False)
        effects.to_csv(args.output_root / "SEASONAL_PLACE_EFFECTS.csv", index=False)
    elif args.command == "plan":
        build_plan(
            args.models.resolve(), args.profiles.resolve(), args.probe_root.resolve(),
            args.climates.resolve(), args.contrasts.resolve(), args.output_root.resolve(),
            fractions=args.fractions,
        )
    elif args.command == "execute":
        execute_plan(
            pd.read_csv(args.run_manifest), pd.read_csv(args.probe_manifest),
            args.probe_run_root.resolve(), args.output_root.resolve(),
            args.runtime.resolve(), workers=args.workers,
        )
    elif args.command == "recover":
        recover_existing_plan(
            pd.read_csv(args.run_manifest), pd.read_csv(args.probe_manifest),
            args.probe_run_root.resolve(), args.output_root.resolve(),
            args.runtime.resolve(), workers=args.workers,
            cleanup_raw=not args.retain_raw,
            stability_manifest=(
                args.stability_manifest.resolve()
                if args.stability_manifest is not None else None
            ),
        )
    elif args.command == "audit-stability":
        result = audit_seasonal_solver_stability(
            pd.read_csv(args.run_manifest),
            args.runtime.resolve(),
            args.primary_root.resolve(),
            args.output_root.resolve(),
            first_iterations=args.first_iterations,
            higher_iterations=args.higher_iterations,
            workers=args.workers,
            temperature_tolerance_c=args.temperature_tolerance_c,
            energy_relative_tolerance=args.energy_relative_tolerance,
        )
        verified = (
            result.trajectory_stable_within_declared_tolerance
            & result.higher_iteration_completed
            & result.warning_profile_nonworsening
            & result.higher_iteration_severe_count.eq(0)
            & result.higher_iteration_fatal_occurrences.eq(0)
        )
        if not verified.all():
            raise RuntimeError(
                "Seasonal solver-stability audit did not verify every selected run"
            )
    elif args.command == "score":
        score_completed_plan(
            args.completed_manifest.resolve(), args.profiles.resolve(),
            args.climates.resolve(), args.contrasts.resolve(),
            args.reference_manifest.resolve(), args.output_root.resolve(),
            args.carbon_factors.resolve(), args.gas_co2_kg_kwh_hhv,
        )


if __name__ == "__main__":
    main()
