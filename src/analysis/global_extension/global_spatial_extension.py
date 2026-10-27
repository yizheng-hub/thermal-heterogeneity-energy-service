"""Run and score the fixed-prototype global placement and permission extension.

The extension adds only physical trajectories that are absent from the annual
global matrix.  Existing random-placement/native-zone trajectories are reused.
Uniform-average trajectories are also reused because that controller has one
building-wide target under every permission and is unchanged by placement.
Exact duplicate thermostat commands are executed once and referenced by every
logical design that uses them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
GLOBAL = HERE.parent
TARGETED = HERE.parents[1] / "targeted_extension"
sys.path.insert(0, str(GLOBAL))
sys.path.insert(0, str(TARGETED))

from analyse_global_results import (  # noqa: E402
    DEFAULT_GAS_CO2_KG_KWH_HHV,
    _carbon_table,
    energy_metrics,
    fixed_tail_from_anchor,
    load_hourly,
    normalise_profiles,
    score_trajectory,
    verify_hourly_hash,
)
from demand_aware_controls import generate_controls  # noqa: E402
from run_global_matrix import (  # noqa: E402
    COHORT_LABELS,
    MAXIMUM_HVAC_ITERATIONS,
    ZONES,
    _execute_jobs,
    _write_run_idf,
    apply_annual_solver_stability_exception,
    read_annual_solver_stability_evidence,
    relative_engineering_qa,
    sha256,
    warning_profile,
)


PERSONAL_CONTROLS = ("personal_center", "personal_range")
ALL_CONTROLS = ("uniform_average", *PERSONAL_CONTROLS)
DEMAND_ORDERED_ZONE_ORDERS = {
    "demand_ordered": tuple(ZONES),
    "demand_ordered_reversed": tuple(reversed(ZONES)),
    "demand_ordered_shift5": tuple((*ZONES[5:], *ZONES[:5])),
}
DEMAND_ORDERED_LAYOUTS = tuple(DEMAND_ORDERED_ZONE_ORDERS)
NEW_LOGICAL_RUNS = 9 * 6 * len(PERSONAL_CONTROLS) * (
    2 + len(DEMAND_ORDERED_LAYOUTS)
)
NEW_PHYSICAL_RUNS = 516
METRICS = (
    "mean_reference_mismatch_c",
    "fixed_tail_reference_mismatch_c",
    "modeled_interval_person_hour_coverage",
    "modeled_interval_mean_excess_c",
    "hvac_electric_kwh",
    "facility_electric_kwh",
    "gas_hhv_kwh",
    "facility_gas_hhv_kwh",
    "conditioning_purchased_kwh",
    "facility_electric_plus_space_heating_gas_kwh",
    "facility_purchased_kwh",
    "conditioning_operational_co2_country_kg",
    "operational_co2_country_kg",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _bool_series(values: pd.Series, label: str) -> pd.Series:
    if values.dtype == bool:
        return values
    result = (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if result.isna().any():
        raise ValueError(f"{label} contains a non-boolean value")
    return result.astype(bool)


def demand_ordered_profiles(
    random_profiles: pd.DataFrame,
    *,
    layout: str = "demand_ordered",
) -> pd.DataFrame:
    """Place the same centre-ordered identities using one declared zone map."""
    if layout not in DEMAND_ORDERED_ZONE_ORDERS:
        raise ValueError(f"Unknown demand-ordered layout: {layout}")
    zone_order = DEMAND_ORDERED_ZONE_ORDERS[layout]
    required = {
        "context",
        "composition_id",
        "layout",
        "profile_id",
        "zone_id",
        "centre_c",
        "controller_lower_c",
        "controller_upper_c",
    }
    missing = required - set(random_profiles)
    if missing:
        raise ValueError(f"Profiles lack {sorted(missing)}")
    source = random_profiles[random_profiles.layout.eq("random_mixing")].copy()
    if source.empty:
        raise ValueError("No random-placement profiles are available")
    if source.duplicated(["context", "composition_id", "profile_id"]).any():
        raise ValueError("Profile identities must be unique within a composition")

    groups: list[pd.DataFrame] = []
    for key, group in source.groupby(["context", "composition_id"], sort=True, dropna=False):
        if len(group) != 267 or set(group.zone_id) != set(ZONES):
            raise ValueError(f"{key} does not contain 267 profiles in all native zones")
        counts = group.zone_id.value_counts().to_dict()
        # Mergesort preserves the existing identity order when centres tie.
        ordered = group.sort_values("centre_c", kind="mergesort").reset_index(drop=True)
        assignment = np.concatenate(
            [np.repeat(zone, int(counts[zone])) for zone in zone_order]
        )
        if len(assignment) != len(ordered):
            raise AssertionError("Zone-slot reconstruction changed composition size")
        ordered["zone_id"] = assignment
        ordered["layout"] = layout
        if ordered.zone_id.value_counts().to_dict() != {
            zone: int(counts[zone]) for zone in ZONES
        }:
            raise AssertionError("Demand ordering changed native-zone slot counts")
        groups.append(ordered)
    result = pd.concat(groups, ignore_index=True)
    return result.sort_values(
        ["context", "composition_id", "zone_id", "profile_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _fixed_tail_positions(
    profiles: pd.DataFrame,
    fixed_tail_profile_ids: tuple[object, ...],
) -> np.ndarray:
    """Map one frozen profile-ID set into a layout-specific score table."""
    profile_ids = profiles.profile_id.astype(str)
    if profile_ids.duplicated().any():
        raise ValueError("A spatial profile table duplicates profile identities")
    positions = {
        profile_id: position
        for position, profile_id in enumerate(profile_ids)
    }
    requested = tuple(map(str, fixed_tail_profile_ids))
    missing = sorted(set(requested) - set(positions))
    if missing:
        raise ValueError(f"Frozen random-anchor tail identities are missing: {missing}")
    return np.asarray([positions[profile_id] for profile_id in requested], dtype=int)


def select_extension_designs(designs: list[dict]) -> list[dict]:
    """Retain only the three missing personal-control design families."""
    retained = []
    for row in designs:
        if row["control"] not in PERSONAL_CONTROLS or not np.isclose(float(row["offset_c"]), 0.0):
            continue
        layout = str(row["layout"])
        permission = str(row["permission"])
        if (layout == "random_mixing" and permission in {"building", "floor"}) or (
            layout in DEMAND_ORDERED_LAYOUTS and permission == "native_zone"
        ):
            retained.append(row)
    keys = [
        (
            row["context"],
            int(row["composition_id"]),
            row["layout"],
            row["permission"],
            row["control"],
            float(row["offset_c"]),
        )
        for row in retained
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Spatial extension duplicates a logical design")
    return sorted(
        retained,
        key=lambda row: (
            row["context"],
            int(row["composition_id"]),
            row["layout"],
            row["permission"],
            row["control"],
            float(row["offset_c"]),
        ),
    )


def _physical_run_id(
    *,
    climate: str,
    targets: dict[str, float],
    model_sha256: str,
    weather_sha256: str,
    runtime_sha256: str,
) -> str:
    payload = {
        "kind": "global_spatial_extension",
        "climate": climate,
        "targets_c": {key: float(targets[key]) for key in sorted(targets)},
        "model_sha256": model_sha256,
        "weather_sha256": weather_sha256,
        "runtime_sha256": runtime_sha256,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    return f"spatial_{digest}"


def build_plan(
    controller_root: Path,
    profiles_path: Path,
    model_manifest_path: Path,
    runtime: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the 540 logical designs and their deduplicated physical plan."""
    random_profiles = pd.read_csv(profiles_path)
    ordered = [
        demand_ordered_profiles(random_profiles, layout=layout)
        for layout in DEMAND_ORDERED_LAYOUTS
    ]
    profiles = pd.concat(
        [random_profiles[random_profiles.layout.eq("random_mixing")], *ordered],
        ignore_index=True,
    )
    expected_groups = 9 * 6 * (1 + len(DEMAND_ORDERED_LAYOUTS))
    groups = profiles[["context", "composition_id", "layout"]].drop_duplicates()
    if len(groups) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} climate/composition/layout groups; found {len(groups)}"
        )

    designs, diagnostics = generate_controls(
        controller_root,
        profile_table=profiles,
        permissions=("building", "floor", "native_zone"),
        offsets_c=(0.0,),
    )
    retained = select_extension_designs(designs)
    if len(retained) != NEW_LOGICAL_RUNS:
        raise ValueError(f"Expected {NEW_LOGICAL_RUNS} missing logical runs; found {len(retained)}")
    models = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if set(models) != set(profiles.context.astype(str).unique()):
        raise ValueError("Prepared models and spatial profile climates differ")
    for climate, config in models.items():
        model = Path(config["prepared_idf"])
        weather = Path(config["weather_path"])
        if sha256(model) != str(config["prepared_idf_sha256"]):
            raise ValueError(f"Prepared model hash differs from its manifest: {climate}")
        if sha256(weather) != str(config["weather_sha256"]):
            raise ValueError(f"Weather hash differs from its manifest: {climate}")
        configured_iterations = int(
            config.get("solver_configuration", {}).get("maximum_hvac_iterations", -1)
        )
        if configured_iterations != MAXIMUM_HVAC_ITERATIONS:
            raise ValueError(f"{climate} does not use {MAXIMUM_HVAC_ITERATIONS} HVAC iterations")
    runtime_hash = sha256(runtime)
    logical_rows: list[dict] = []
    physical: dict[str, dict] = {}
    input_root = output_root / "inputs"
    for design in retained:
        climate = str(design["context"])
        if climate not in models:
            raise ValueError(f"No prepared fixed prototype for {climate}")
        config = models[climate]
        targets = {str(key): float(value) for key, value in design["targets_c"].items()}
        if set(targets) != set(ZONES):
            raise ValueError(f"{climate} design lacks a complete native-zone target vector")
        run_id = _physical_run_id(
            climate=climate,
            targets=targets,
            model_sha256=str(config["prepared_idf_sha256"]),
            weather_sha256=str(config["weather_sha256"]),
            runtime_sha256=runtime_hash,
        )
        input_path = input_root / f"{run_id}.idf"
        if not input_path.is_file():
            _write_run_idf(Path(config["prepared_idf"]), targets, input_path)
        record = {
            "run_id": run_id,
            "climate": climate,
            "composition_id": int(design["composition_id"]),
            "cohort_label": COHORT_LABELS[int(design["composition_id"])],
            "layout": str(design["layout"]),
            "permission": str(design["permission"]),
            "control": str(design["control"]),
            "offset_c": 0.0,
            "targets_json": json.dumps(targets, sort_keys=True),
            "targets_sha256": hashlib.sha256(
                json.dumps(targets, sort_keys=True).encode()
            ).hexdigest(),
            "input_idf_path": str(input_path.resolve()),
            "input_sha256": sha256(input_path),
            "weather_path": str(Path(config["weather_path"]).resolve()),
            "weather_sha256": str(config["weather_sha256"]),
            "runtime_sha256": runtime_hash,
        }
        logical_rows.append(record)
        physical.setdefault(
            run_id,
            {
                "run_id": run_id,
                "climate": climate,
                # One representative logical identity makes the physical
                # manifest directly compatible with the existing 40-to-100
                # iteration audit. Qualification remains tied to run_id and
                # exact input/hourly hashes, not to these descriptive fields.
                "composition_id": int(design["composition_id"]),
                "control": str(design["control"]),
                "offset_c": 0.0,
                "layout": str(design["layout"]),
                "permission": str(design["permission"]),
                "input_idf_path": record["input_idf_path"],
                "input_sha256": record["input_sha256"],
                "weather_path": record["weather_path"],
                "weather_sha256": record["weather_sha256"],
                "runtime_sha256": runtime_hash,
            },
        )
    logical = pd.DataFrame(logical_rows).sort_values(
        ["climate", "composition_id", "layout", "permission", "control"],
        kind="mergesort",
    )
    execution = pd.DataFrame(physical.values()).sort_values(["climate", "run_id"], kind="mergesort")
    if len(execution) != NEW_PHYSICAL_RUNS:
        raise ValueError(
            f"Expected {NEW_PHYSICAL_RUNS} distinct physical runs; found {len(execution)}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(output_root / "GLOBAL_SPATIAL_PROFILES.csv", index=False)
    logical.to_csv(output_root / "GLOBAL_SPATIAL_LOGICAL_PLAN.csv", index=False)
    execution.to_csv(output_root / "GLOBAL_SPATIAL_EXECUTION_PLAN.csv", index=False)
    _write_json(output_root / "GLOBAL_SPATIAL_TARGET_DIAGNOSTICS.json", diagnostics)
    _write_json(
        output_root / "GLOBAL_SPATIAL_PLAN_QA.json",
        {
            "status": "PASS",
            "logical_runs": len(logical),
            "distinct_physical_runs": len(execution),
            "exact_duplicate_commands_reused": len(logical) - len(execution),
            "climates": int(logical.climate.nunique()),
            "compositions": int(logical.composition_id.nunique()),
            "controls": sorted(logical.control.unique()),
            "layouts": sorted(profiles.layout.unique()),
            "permissions": sorted(logical.permission.unique()),
            "profiles_sha256": sha256(output_root / "GLOBAL_SPATIAL_PROFILES.csv"),
            "logical_plan_sha256": sha256(output_root / "GLOBAL_SPATIAL_LOGICAL_PLAN.csv"),
            "execution_plan_sha256": sha256(output_root / "GLOBAL_SPATIAL_EXECUTION_PLAN.csv"),
            "model_manifest_sha256": sha256(model_manifest_path),
            "runtime_sha256": runtime_hash,
        },
    )
    return logical, execution


def execute_plan(
    execution_plan: pd.DataFrame,
    probe_manifest: pd.DataFrame,
    probe_root: Path,
    runtime: Path,
    output_root: Path,
    *,
    workers: int = 8,
    solver_stability_manifest: Path | None = None,
) -> pd.DataFrame:
    """Execute each distinct thermostat vector and retain every QA outcome."""
    required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "offset_c",
        "input_idf_path",
        "input_sha256",
        "weather_path",
        "weather_sha256",
        "runtime_sha256",
    }
    missing = required - set(execution_plan)
    if missing:
        raise ValueError(f"Execution plan lacks {sorted(missing)}")
    if execution_plan.run_id.duplicated().any():
        raise ValueError("Execution plan duplicates a physical run")
    runtime_hash = sha256(runtime)
    if not execution_plan.runtime_sha256.eq(runtime_hash).all():
        raise ValueError("Execution plan was created for a different runtime")
    for row in execution_plan.itertuples(index=False):
        if sha256(Path(row.input_idf_path)) != str(row.input_sha256):
            raise ValueError(f"Input IDF hash differs from the plan: {row.run_id}")
        if sha256(Path(row.weather_path)) != str(row.weather_sha256):
            raise ValueError(f"Weather hash differs from the plan: {row.run_id}")

    references = probe_manifest[np.isclose(pd.to_numeric(probe_manifest.target_c), 25.0)].copy()
    if references.climate.duplicated().any() or references.climate.nunique() != 9:
        raise ValueError("Expected one 25 C probe for each of nine climates")
    baseline_errors = {
        str(row.climate): probe_root / "runs" / str(row.run_id) / "eplusout.err"
        for row in references.itertuples(index=False)
    }
    stability, stability_sha256 = read_annual_solver_stability_evidence(solver_stability_manifest)
    jobs = []
    for row in execution_plan.to_dict("records"):
        run_id = str(row["run_id"])
        jobs.append(
            {
                "run_id": run_id,
                "climate": str(row["climate"]),
                "composition_id": int(row["composition_id"]),
                "control": str(row["control"]),
                "offset_c": float(row["offset_c"]),
                "idf": Path(row["input_idf_path"]),
                "weather": Path(row["weather_path"]),
                "run_folder": output_root / "runs" / run_id,
                "compact": output_root / "hourly" / f"{run_id}.npz",
                "input_sha256": str(row["input_sha256"]),
                "weather_sha256": str(row["weather_sha256"]),
            }
        )
    records = _execute_jobs(jobs, runtime, workers)
    rows: list[dict] = []
    for row in records:
        climate = str(row["climate"])
        qa = row["run_qa"]
        relative = relative_engineering_qa(
            qa,
            warning_profile(row["run_folder"] / "eplusout.err"),
            warning_profile(baseline_errors[climate]),
        )
        evidence = stability.get(str(row["run_id"]))
        final_pass, stability_audit = apply_annual_solver_stability_exception(
            relative,
            qa,
            evidence,
            current_maximum_hvac_iterations=MAXIMUM_HVAC_ITERATIONS,
            active_input_sha256=str(row["input_sha256"]),
            active_hourly_sha256=str(qa.get("hourly_sha256")),
        )
        rows.append(
            {
                "run_id": row["run_id"],
                "climate": climate,
                "composition_id": int(row["composition_id"]),
                "control": str(row["control"]),
                "offset_c": float(row["offset_c"]),
                "completed": bool(qa.get("completed")),
                "final_engineering_qualified": final_pass,
                "relative_engineering_pass": final_pass,
                **stability_audit,
                "maximum_hvac_iterations": MAXIMUM_HVAC_ITERATIONS,
                "new_warning_categories": json.dumps(relative["new_warning_categories"]),
                "warning_count_increases": json.dumps(relative["warning_count_increases"]),
                "critical_warning_count_increases": json.dumps(
                    relative["critical_warning_count_increases"]
                ),
                "baseline_warning_profile": json.dumps(relative["baseline_warning_profile"]),
                "treatment_warning_profile": json.dumps(relative["treatment_warning_profile"]),
                "input_idf_path": str(row["idf"].resolve()),
                "input_sha256": row["input_sha256"],
                "weather_path": str(row["weather"].resolve()),
                "weather_sha256": row["weather_sha256"],
                "qa_path": str((row["run_folder"] / "RUN_QA.json").resolve()),
                "run_folder": str(row["run_folder"].resolve()),
                "hourly_path": str(row["compact"].resolve()),
                "hourly_sha256": qa.get("hourly_sha256"),
            }
        )
    result = pd.DataFrame(rows).sort_values(["climate", "run_id"], kind="mergesort")
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / "GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv", index=False)
    _write_json(
        output_root / "GLOBAL_SPATIAL_EXECUTION_QA.json",
        {
            "status": (
                "PASS"
                if result.completed.all() and result.final_engineering_qualified.all()
                else "REVIEW"
                if result.completed.all()
                else "FAIL"
            ),
            "physical_runs": len(result),
            "completed": int(result.completed.sum()),
            "final_engineering_qualified": int(
                result.final_engineering_qualified.sum()
            ),
            "final_engineering_not_qualified": int(
                (~result.final_engineering_qualified).sum()
            ),
            "relative_engineering_pass": int(result.relative_engineering_pass.sum()),
            "relative_engineering_fail": int((~result.relative_engineering_pass).sum()),
            "solver_stability_manifest_sha256": stability_sha256,
            "manifest_sha256": sha256(output_root / "GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv"),
        },
    )
    return result


def _annual_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "offset_c",
        "relative_engineering_pass",
        "hourly_path",
        "hourly_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Annual manifest lacks {sorted(missing)}")
    result = frame[np.isclose(pd.to_numeric(frame.offset_c), 0.0)].copy()
    if result.duplicated(["climate", "composition_id", "control"]).any():
        raise ValueError("Annual manifest duplicates an offset-zero baseline")
    expected = 9 * 6 * 3
    if len(result) != expected:
        raise ValueError(f"Expected {expected} offset-zero annual baselines; found {len(result)}")
    result["relative_engineering_pass"] = _bool_series(
        result.relative_engineering_pass, "Annual engineering qualification"
    )
    if "final_engineering_qualified" in result:
        result["final_engineering_qualified"] = _bool_series(
            result.final_engineering_qualified,
            "Final annual engineering qualification",
        )
        if not result.final_engineering_qualified.equals(
            result.relative_engineering_pass
        ):
            raise ValueError(
                "Final annual qualification disagrees with its compatibility alias"
            )
    else:
        # The frozen annual manifest predates the explicit final-field name;
        # its relative_engineering_pass already includes the audited solver result.
        result["final_engineering_qualified"] = result.relative_engineering_pass
    return result


def _pair_effects(
    metrics: pd.DataFrame,
    *,
    axis: str,
    reference: str,
    alternatives: tuple[str, ...],
) -> pd.DataFrame:
    keys = ["climate", "composition", "control"]
    fixed = "layout" if axis == "permission" else "permission"
    rows: list[dict] = []
    for key, group in metrics.groupby(keys + [fixed], sort=True, dropna=False):
        base = group[group[axis].eq(reference)]
        if len(base) != 1:
            raise ValueError(f"A spatial comparison lacks one {reference} reference")
        base_row = base.iloc[0]
        for alternative in alternatives:
            selected = group[group[axis].eq(alternative)]
            if len(selected) != 1:
                raise ValueError(f"A spatial comparison lacks one {alternative} alternative")
            item = selected.iloc[0]
            matched = bool(base_row.engineering_pass and item.engineering_pass)
            record = dict(zip(keys + [fixed], key, strict=True)) | {
                "axis": axis,
                "reference": reference,
                "alternative": alternative,
                "status": "matched" if matched else "engineering_unavailable",
                "reference_run_id": base_row.run_id,
                "alternative_run_id": item.run_id,
                "same_physical_run": bool(base_row.run_id == item.run_id),
            }
            for metric in METRICS:
                record[f"delta_{metric}"] = (
                    float(item[metric]) - float(base_row[metric]) if matched else np.nan
                )
            if matched:
                record["delta_facility_purchased_pct"] = (
                    100
                    * record["delta_facility_purchased_kwh"]
                    / float(base_row.facility_purchased_kwh)
                )
                record["delta_conditioning_purchased_pct"] = (
                    100
                    * record["delta_conditioning_purchased_kwh"]
                    / float(base_row.conditioning_purchased_kwh)
                )
            else:
                record["delta_facility_purchased_pct"] = np.nan
                record["delta_conditioning_purchased_pct"] = np.nan
            rows.append(record)
    return pd.DataFrame(rows)


def score_plan(
    logical_plan_path: Path,
    completed_manifest_path: Path,
    annual_manifest_path: Path,
    profiles_path: Path,
    reference_manifest_path: Path,
    carbon_path: Path,
    output_root: Path,
    *,
    gas_factor: float = DEFAULT_GAS_CO2_KG_KWH_HHV,
) -> dict:
    """Score logical placement/permission cells and their paired effects."""
    logical = pd.read_csv(logical_plan_path)
    completed = pd.read_csv(completed_manifest_path)
    completed["completed"] = _bool_series(completed.completed, "Spatial execution completion")
    if not completed.completed.all():
        raise ValueError("Spatial scoring requires every physical run to complete")
    if "final_engineering_qualified" not in completed:
        raise ValueError(
            "Spatial scoring requires the explicit final engineering qualification"
        )
    completed["final_engineering_qualified"] = _bool_series(
        completed.final_engineering_qualified,
        "Final spatial engineering qualification",
    )
    completed["relative_engineering_pass"] = _bool_series(
        completed.relative_engineering_pass, "Spatial engineering qualification"
    )
    if not completed.final_engineering_qualified.equals(
        completed.relative_engineering_pass
    ):
        raise ValueError(
            "Final spatial qualification disagrees with its compatibility alias"
        )
    if completed.run_id.duplicated().any():
        raise ValueError("Completed spatial manifest duplicates a physical run")
    new = logical.merge(
        completed[
            [
                "run_id",
                "final_engineering_qualified",
                "relative_engineering_pass",
                "hourly_path",
                "hourly_sha256",
            ]
        ],
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    if new.hourly_path.isna().any():
        raise ValueError("A logical spatial design lacks a completed physical run")
    annual = _annual_baselines(pd.read_csv(annual_manifest_path))
    # Drop the human-readable cohort label so normalisation uses the stable
    # integer composition identifier shared by the run manifests.
    profiles = normalise_profiles(
        pd.read_csv(profiles_path).drop(columns=["cohort_label"], errors="ignore")
    )
    carbon = _carbon_table(carbon_path).set_index("climate")

    references = pd.read_csv(reference_manifest_path)
    references = references[np.isclose(pd.to_numeric(references.target_c), 25.0)]
    if references.climate.duplicated().any() or references.climate.nunique() != 9:
        raise ValueError("Expected one 25 C reference trajectory per climate")
    reference_lookup = references.set_index("climate")

    annual_lookup = annual.set_index(["climate", "composition_id", "control"])
    new_lookup = new.set_index(["climate", "composition_id", "layout", "permission", "control"])
    configurations = (
        ("random_mixing", "building"),
        ("random_mixing", "floor"),
        ("random_mixing", "native_zone"),
        *((layout, "native_zone") for layout in DEMAND_ORDERED_LAYOUTS),
    )
    data_cache: dict[str, dict[str, np.ndarray]] = {}
    reference_cache: dict[str, dict[str, np.ndarray]] = {}
    tail_cache: dict[tuple[str, object, str], np.ndarray] = {}
    tail_rows: list[dict] = []
    metric_rows: list[dict] = []

    def trajectory(path_value: object, hash_value: object) -> dict[str, np.ndarray]:
        path = Path(str(path_value))
        key = str(path.resolve())
        verify_hourly_hash(path, hash_value, "Spatial trajectory")
        if key not in data_cache:
            data_cache[key] = load_hourly(path)
        return data_cache[key]

    for climate in sorted(profiles.climate.astype(str).unique()):
        reference_row = reference_lookup.loc[climate]
        reference_path = Path(str(reference_row.hourly_path))
        verify_hourly_hash(
            reference_path,
            reference_row.hourly_sha256,
            f"25 C reference trajectory for {climate}",
        )
        reference_cache[climate] = load_hourly(reference_path, require_facility=False)
        for composition in sorted(
            profiles.loc[profiles.climate.eq(climate), "composition"].unique()
        ):
            random_anchor_profiles = (
                profiles[
                    profiles.climate.eq(climate)
                    & profiles.composition.eq(composition)
                    & profiles.layout.eq("random_mixing")
                ]
                .sort_values("profile_id", kind="mergesort")
                .reset_index(drop=True)
            )
            if len(random_anchor_profiles) != 267:
                raise ValueError(
                    f"Expected 267 random-placement profiles for {climate}/{composition}"
                )
            anchor_tail, _ = fixed_tail_from_anchor(
                reference_cache[climate], random_anchor_profiles
            )
            fixed_tail_profile_ids = tuple(
                random_anchor_profiles.profile_id.iloc[int(position)]
                for position in anchor_tail
            )
            if len(fixed_tail_profile_ids) != 27 or len(set(fixed_tail_profile_ids)) != 27:
                raise ValueError(
                    f"Expected 27 unique random-anchor tail profiles for "
                    f"{climate}/{composition}"
                )
            for layout, permission in configurations:
                local_profiles = (
                    profiles[
                        profiles.climate.eq(climate)
                        & profiles.composition.eq(composition)
                        & profiles.layout.eq(layout)
                    ]
                    .sort_values("profile_id", kind="mergesort")
                    .reset_index(drop=True)
                )
                if len(local_profiles) != 267:
                    raise ValueError(f"Expected 267 profiles for {climate}/{composition}/{layout}")
                tail_key = (climate, composition, layout)
                if tail_key not in tail_cache:
                    tail_cache[tail_key] = _fixed_tail_positions(
                        local_profiles,
                        fixed_tail_profile_ids,
                    )
                    tail_rows.extend(
                        {
                            "climate": climate,
                            "composition": composition,
                            "layout": layout,
                            "anchor_layout": "random_mixing",
                            "profile_id": str(profile_id),
                        }
                        for profile_id in fixed_tail_profile_ids
                    )
                for control in ALL_CONTROLS:
                    baseline = annual_lookup.loc[(climate, composition, control)]
                    if layout == "random_mixing" and permission == "native_zone":
                        source = baseline
                        source_kind = "reused_annual_baseline"
                    elif control == "uniform_average":
                        source = baseline
                        source_kind = "reused_uniform_physical_run"
                    else:
                        source = new_lookup.loc[(climate, composition, layout, permission, control)]
                        source_kind = "new_spatial_run"
                    engineering_pass = bool(source.final_engineering_qualified)
                    data = trajectory(source.hourly_path, source.hourly_sha256)
                    service = score_trajectory(data, local_profiles, tail_cache[tail_key])[
                        "summary"
                    ]
                    energy = energy_metrics(data)
                    factors = carbon.loc[climate]
                    metric_rows.append(
                        {
                            "climate": climate,
                            "composition": composition,
                            "layout": layout,
                            "permission": permission,
                            "control": control,
                            "run_id": source.run_id,
                            "source_kind": source_kind,
                            "engineering_pass": engineering_pass,
                            **service,
                            **energy,
                            "conditioning_operational_co2_country_kg": (
                                energy["hvac_electric_kwh"] * factors.electricity_co2_kg_kwh
                                + energy["gas_hhv_kwh"] * gas_factor
                            ),
                            "operational_co2_country_kg": (
                                energy["facility_electric_kwh"] * factors.electricity_co2_kg_kwh
                                + energy["facility_gas_hhv_kwh"] * gas_factor
                            ),
                        }
                    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["climate", "composition", "layout", "permission", "control"],
        kind="mergesort",
    )
    expected_metrics = 9 * 6 * len(configurations) * len(ALL_CONTROLS)
    if len(metrics) != expected_metrics:
        raise AssertionError(
            f"Expected {expected_metrics} logical spatial metrics; found {len(metrics)}"
        )
    permission = _pair_effects(
        metrics[metrics.layout.eq("random_mixing")],
        axis="permission",
        reference="building",
        alternatives=("floor", "native_zone"),
    )
    # Add the incremental floor-to-native comparison without changing the
    # declared building-reference comparisons above.
    floor_to_native = _pair_effects(
        metrics[metrics.layout.eq("random_mixing")],
        axis="permission",
        reference="floor",
        alternatives=("native_zone",),
    )
    permission = pd.concat([permission, floor_to_native], ignore_index=True)
    placement = _pair_effects(
        metrics[metrics.permission.eq("native_zone")],
        axis="layout",
        reference="random_mixing",
        alternatives=DEMAND_ORDERED_LAYOUTS,
    )
    expected_permission_pairs = 9 * 6 * len(ALL_CONTROLS) * 3
    expected_placement_pairs = (
        9 * 6 * len(ALL_CONTROLS) * len(DEMAND_ORDERED_LAYOUTS)
    )
    if len(permission) != expected_permission_pairs:
        raise AssertionError(
            f"Expected {expected_permission_pairs} permission pairs; found {len(permission)}"
        )
    if len(placement) != expected_placement_pairs:
        raise AssertionError(
            f"Expected {expected_placement_pairs} placement pairs; found {len(placement)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_root / "GLOBAL_SPATIAL_METRICS.csv", index=False)
    permission.to_csv(output_root / "GLOBAL_PERMISSION_COMPARISONS.csv", index=False)
    placement.to_csv(output_root / "GLOBAL_PLACEMENT_COMPARISONS.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(
        output_root / "GLOBAL_SPATIAL_FIXED_TAIL_MEMBERSHIPS.csv", index=False
    )
    report = {
        "status": "PASS",
        "logical_new_designs": len(logical),
        "distinct_new_physical_runs": int(completed.run_id.nunique()),
        "logical_metric_rows": len(metrics),
        "permission_pairs": len(permission),
        "placement_pairs": len(placement),
        "matched_permission_pairs": int(permission.status.eq("matched").sum()),
        "matched_placement_pairs": int(placement.status.eq("matched").sum()),
        "reused_physical_rows": int(metrics.source_kind.str.startswith("reused").sum()),
        "new_physical_rows": int(metrics.source_kind.eq("new_spatial_run").sum()),
        "fixed_tail_profiles_per_context": 27,
        "fixed_tail_anchor_layout": "random_mixing",
        "demand_ordered_layouts": list(DEMAND_ORDERED_LAYOUTS),
        "demand_ordered_zone_orders": {
            layout: list(order)
            for layout, order in DEMAND_ORDERED_ZONE_ORDERS.items()
        },
        "input_sha256": {
            "logical_plan": sha256(logical_plan_path),
            "completed_manifest": sha256(completed_manifest_path),
            "annual_manifest": sha256(annual_manifest_path),
            "profiles": sha256(profiles_path),
            "reference_manifest": sha256(reference_manifest_path),
            "carbon": sha256(carbon_path),
        },
        "output_sha256": {path.name: sha256(path) for path in sorted(output_root.glob("*.csv"))},
    }
    _write_json(output_root / "GLOBAL_SPATIAL_AUDIT.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--controller-root", type=Path, required=True)
    plan.add_argument("--profiles", type=Path, required=True)
    plan.add_argument("--model-manifest", type=Path, required=True)
    plan.add_argument("--runtime", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)

    execute = sub.add_parser("execute")
    execute.add_argument("--execution-plan", type=Path, required=True)
    execute.add_argument("--probe-manifest", type=Path, required=True)
    execute.add_argument("--probe-root", type=Path, required=True)
    execute.add_argument("--runtime", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--workers", type=int, default=8)
    execute.add_argument("--solver-stability-manifest", type=Path)

    score = sub.add_parser("score")
    score.add_argument("--logical-plan", type=Path, required=True)
    score.add_argument("--completed-manifest", type=Path, required=True)
    score.add_argument("--annual-manifest", type=Path, required=True)
    score.add_argument("--profiles", type=Path, required=True)
    score.add_argument("--reference-manifest", type=Path, required=True)
    score.add_argument("--carbon-factors", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument(
        "--gas-co2-kg-kwh-hhv",
        type=float,
        default=DEFAULT_GAS_CO2_KG_KWH_HHV,
    )
    args = parser.parse_args()

    if args.command == "plan":
        logical, execution = build_plan(
            args.controller_root.resolve(),
            args.profiles.resolve(),
            args.model_manifest.resolve(),
            args.runtime.resolve(),
            args.output_root.resolve(),
        )
        print(
            json.dumps(
                {
                    "logical_runs": len(logical),
                    "distinct_physical_runs": len(execution),
                }
            )
        )
    elif args.command == "execute":
        result = execute_plan(
            pd.read_csv(args.execution_plan),
            pd.read_csv(args.probe_manifest),
            args.probe_root.resolve(),
            args.runtime.resolve(),
            args.output_root.resolve(),
            workers=args.workers,
            solver_stability_manifest=(
                args.solver_stability_manifest.resolve() if args.solver_stability_manifest else None
            ),
        )
        print(
            json.dumps(
                {
                    "physical_runs": len(result),
                    "relative_engineering_pass": int(result.relative_engineering_pass.sum()),
                }
            )
        )
    elif args.command == "score":
        report = score_plan(
            args.logical_plan.resolve(),
            args.completed_manifest.resolve(),
            args.annual_manifest.resolve(),
            args.profiles.resolve(),
            args.reference_manifest.resolve(),
            args.carbon_factors.resolve(),
            args.output_root.resolve(),
            gas_factor=args.gas_co2_kg_kwh_hhv,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
