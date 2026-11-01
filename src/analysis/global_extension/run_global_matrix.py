"""Run the standardized global-climate energy-service experiment.

The runner keeps the DOE prototype, weather file, demand profiles, generated
thermostat commands and compact hourly evidence linked by hashes.  Raw
EnergyPlus files are removed after successful extraction; the compact records,
error log and quality-assurance record remain available for review.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
GLOBAL = HERE.parent
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
TARGETED = HERE.parents[1] / "targeted_extension"
sys.path.insert(0, str(GLOBAL))
sys.path.insert(0, str(ENERGY_SERVICE))
sys.path.insert(0, str(TARGETED))

from demand_aware_controls import generate_controls  # noqa: E402
from generate_model_inputs import annual_input, physical_hash, signature  # noqa: E402
from global_matrix import (  # noqa: E402
    OFFSETS_C,
    PROBE_TARGETS_C,
    ClimateSpec,
    non_interface_fingerprint,
    patch_control_interface,
    replace_site_and_sizing,
    run_energyplus,
    sha256,
)


ZONES = (
    "Core_bottom",
    "Perimeter_bot_ZN_1",
    "Perimeter_bot_ZN_2",
    "Perimeter_bot_ZN_3",
    "Perimeter_bot_ZN_4",
    "Core_mid",
    "Perimeter_mid_ZN_1",
    "Perimeter_mid_ZN_2",
    "Perimeter_mid_ZN_3",
    "Perimeter_mid_ZN_4",
    "Core_top",
    "Perimeter_top_ZN_1",
    "Perimeter_top_ZN_2",
    "Perimeter_top_ZN_3",
    "Perimeter_top_ZN_4",
)
FLOOR_GROUPS = {
    zone: ("bottom" if "bottom" in zone.lower() or "_bot_" in zone.lower()
           else "middle" if "_mid" in zone.lower() else "top")
    for zone in ZONES
}
HALF_DEADBAND_C = 1.5
FLOOR_AREA_M2 = 4982.19
MAXIMUM_HVAC_ITERATIONS = 40
INHERITED_DRIFT_WARNING_CATEGORIES = frozenset({
    "wet_bulb", "psychrometric_temperature",
})
CRITICAL_WARNING_CATEGORIES = frozenset({
    "hvac_iterations", "coil_operating_range",
})
COHORT_LABELS = {
    2026102001: "Multinational A",
    2026102002: "Multinational B",
    2026102003: "Multinational C",
    2026102501: "China A",
    2026102502: "China B",
    2026102503: "China C",
}
CONTROL_IDENTITY = ("climate", "composition_id", "control", "offset_c")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _hash_payload(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def read_climates(path: Path) -> list[ClimateSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("climate", "country", "continent", "prototype_variant", "weather")
    climates = []
    for row in payload["climates"]:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Climate row lacks required fields: {missing}")
        climate = ClimateSpec(
            **{key: row[key] for key in required},
            sentinel=bool(row.get("sentinel", False)),
            latitude=row.get("latitude"),
            longitude=row.get("longitude"),
            climate_zone=str(row.get("climate_zone", "")),
        )
        if climate.latitude is not None and not -90 <= float(climate.latitude) <= 90:
            raise ValueError(f"Invalid latitude for {climate.climate}")
        if climate.longitude is not None and not -180 <= float(climate.longitude) <= 180:
            raise ValueError(f"Invalid longitude for {climate.climate}")
        climates.append(climate)
    if len(climates) != len({row.climate for row in climates}):
        raise ValueError("Climate names must be unique")
    if not climates or not any(row.sentinel for row in climates):
        raise ValueError("The matrix needs climates and at least one sentinel")
    return climates


def set_maximum_hvac_iterations(text: str, maximum: int) -> str:
    """Set one declared solver limit while preserving every other IDF field."""
    if maximum <= 0:
        raise ValueError("Maximum HVAC iterations must be positive")
    pattern = re.compile(
        r"(?is)(ConvergenceLimits\s*,\s*[^,;]+\s*,\s*)[^,;]+"
    )
    updated, count = pattern.subn(rf"\g<1>{int(maximum)}", text)
    if count != 1:
        raise ValueError(f"Expected one ConvergenceLimits object, found {count}")
    return updated


def solver_configuration(text: str) -> dict[str, int | None]:
    """Read the convergence settings that are part of the run contract."""
    blocks = [
        [field.strip() for field in block.split(",")]
        for block in re.sub(r"!.*", "", text).split(";")
        if block.strip()
    ]
    convergence = [row for row in blocks if row[0].lower() == "convergencelimits"]
    if len(convergence) != 1 or len(convergence[0]) < 3:
        raise ValueError("Expected one complete ConvergenceLimits object")
    row = convergence[0]
    return {
        "minimum_system_timestep_minutes": int(float(row[1])),
        "maximum_hvac_iterations": int(float(row[2])),
        "minimum_plant_iterations": (
            int(float(row[3])) if len(row) > 3 and row[3] else None
        ),
        "maximum_plant_iterations": (
            int(float(row[4])) if len(row) > 4 and row[4] else None
        ),
    }


def _prototype_path(root: Path, climate: ClimateSpec) -> Path:
    choices = (
        root / f"{climate.climate}.idf",
        root / f"{climate.prototype_variant}.idf",
        root / f"ASHRAE901_OfficeMedium_STD2019_{climate.prototype_variant}.idf",
    )
    found = next((path for path in choices if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"No prototype for {climate.climate}: {choices}")
    return found


def build_profiles(
    climates: Iterable[ClimateSpec],
    ashrae_npz: Path,
    ctcd_csv: Path,
) -> pd.DataFrame:
    """Create six equally sized demand compositions for every climate."""
    required = {
        "composition_id", "layout", "profile_id", "zone_id", "centre_c",
        "controller_lower_c", "controller_upper_c",
    }
    with np.load(ashrae_npz, allow_pickle=False) as saved:
        multinational = pd.DataFrame({name: saved[name] for name in saved.files})
    multinational = multinational[multinational["layout"].eq("random_mixing")].copy()
    china = pd.read_csv(ctcd_csv)
    missing_assignment = (required | {"context"}) - set(china)
    if missing_assignment:
        raise ValueError(
            "China profile assignment file lacks "
            f"{sorted(missing_assignment)}; pass PROFILE_ASSIGNMENTS.csv.gz, "
            "not the unassigned profile archive"
        )
    china = china[
        china["context"].eq("Miami") & china["layout"].eq("random_mixing")
    ].copy()
    for name, frame in (("multinational", multinational), ("China", china)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} profiles lack {sorted(missing)}")
    source = pd.concat(
        [multinational[list(required)], china[list(required)]], ignore_index=True
    )
    found = set(source["composition_id"].astype(int))
    if found != set(COHORT_LABELS):
        raise ValueError(f"Expected six declared compositions, found {sorted(found)}")
    counts = source.groupby("composition_id").size()
    if not counts.eq(267).all():
        raise ValueError(f"Every composition must contain 267 profiles: {counts.to_dict()}")
    if source.duplicated(["composition_id", "profile_id"]).any():
        raise ValueError("Profile identifiers must be unique within each composition")
    if set(source["zone_id"]) != set(ZONES):
        raise ValueError("Demand assignments and medium-office zones differ")
    numeric = source[["centre_c", "controller_lower_c", "controller_upper_c"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Demand coordinates must be finite")
    if numeric.controller_lower_c.gt(numeric.controller_upper_c).any():
        raise ValueError("A Personal range has reversed bounds")
    source[["centre_c", "controller_lower_c", "controller_upper_c"]] = numeric
    source["cohort_label"] = source["composition_id"].map(COHORT_LABELS)
    source["demand_source"] = np.where(
        source["composition_id"].astype(int).lt(2026102500),
        "Multinational archive", "China archive",
    )
    frames = []
    for climate in climates:
        local = source.copy()
        local.insert(0, "context", climate.climate)
        frames.append(local)
    result = pd.concat(frames, ignore_index=True)
    columns = [
        "context", "composition_id", "cohort_label", "demand_source", "layout",
        "profile_id", "zone_id", "centre_c", "controller_lower_c",
        "controller_upper_c",
    ]
    return result[columns].sort_values(
        ["context", "composition_id", "zone_id", "profile_id"]
    ).reset_index(drop=True)


def prepare_models(
    climates: Iterable[ClimateSpec],
    prototype_root: Path,
    weather_root: Path,
    reference_idf: Path,
    model_root: Path,
    prototype_mode: str = "climate_matched",
) -> dict[str, dict]:
    """Prepare either climate-matched prototypes or one fixed reference model."""
    if prototype_mode not in {"climate_matched", "fixed_reference"}:
        raise ValueError(f"Unknown prototype mode: {prototype_mode}")
    reference = reference_idf.read_text(encoding="utf-8", errors="replace")
    manifest: dict[str, dict] = {}
    model_root.mkdir(parents=True, exist_ok=True)
    for climate in climates:
        prototype = (
            reference_idf
            if prototype_mode == "fixed_reference"
            else _prototype_path(prototype_root, climate)
        )
        weather = weather_root / climate.weather
        if not weather.is_file():
            raise FileNotFoundError(weather)
        source = prototype.read_text(encoding="utf-8", errors="replace")
        controlled = (
            source
            if prototype_mode == "fixed_reference"
            else patch_control_interface(source, reference)
        )
        prepared = set_maximum_hvac_iterations(
            replace_site_and_sizing(controlled, weather),
            MAXIMUM_HVAC_ITERATIONS,
        )
        solver = solver_configuration(prepared)
        if solver["maximum_hvac_iterations"] != MAXIMUM_HVAC_ITERATIONS:
            raise AssertionError("Prepared model did not retain the declared solver limit")
        path = model_root / f"{climate.climate}.idf"
        path.write_text(prepared, encoding="utf-8")
        manifest[climate.climate] = {
            "country": climate.country,
            "continent": climate.continent,
            "sentinel": climate.sentinel,
            "prototype_mode": prototype_mode,
            "prototype_variant": (
                "Fixed reference medium office"
                if prototype_mode == "fixed_reference"
                else climate.prototype_variant
            ),
            "prototype_path": str(prototype.resolve()),
            "prototype_sha256": sha256(prototype),
            "weather_path": str(weather.resolve()),
            "weather_sha256": sha256(weather),
            "prepared_idf": str(path.resolve()),
            "prepared_idf_sha256": sha256(path),
            "prototype_non_interface_sha256": non_interface_fingerprint(source),
            "controlled_non_interface_before_site_sha256": non_interface_fingerprint(
                controlled
            ),
            "physical_hash": physical_hash(prepared),
            "solver_configuration": solver,
            "half_deadband_c": HALF_DEADBAND_C,
            "floor_area_m2": FLOOR_AREA_M2,
        }
    _write_json(model_root / "MODEL_MANIFEST.json", manifest)
    return manifest


def warning_profile(error_path: Path) -> dict[str, int]:
    """Count treatment-relevant warning groups in an EnergyPlus error log."""
    text = error_path.read_text(encoding="utf-8", errors="replace")
    categories = {
        "hvac_iterations": r"SimHVAC: (?:Maximum|Exceeding Maximum) iterations",
        "wet_bulb": r"WetBulb not converged",
        "psychrometric_temperature": r"Temperature out of range .*PsyPsat",
        "coil_operating_range": r"CalcDoe2DXCoil:",
    }
    result: dict[str, int] = {}
    recurring = text.split("===== Recurring Error Summary =====", 1)
    recurring_text = recurring[1] if len(recurring) == 2 else ""
    for name, pattern in categories.items():
        direct = len(re.findall(rf"\*\* Warning \*\*[^\n]*{pattern}", text, re.I))
        counts = [int(value) for value in re.findall(
            rf"\*\* Warning \*\*[^\n]*{pattern}.*?occurred\s+(\d+)\s+total times",
            recurring_text,
            re.I | re.S,
        )]
        # Direct messages contain the first occurrences and the recurring
        # summary.  Prefer the explicit annual count when it is available.
        result[name] = max(counts) if counts else direct
    return result


def compare_solver_warning_categories(
    first: dict[str, int],
    second: dict[str, int],
) -> dict:
    """Compare warning categories while retaining count drift as a diagnostic."""
    for label, profile in (("first", first), ("second", second)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in profile.values()
        ):
            raise ValueError(f"{label} solver-warning profile has an invalid count")
    first_active = {name for name, value in first.items() if int(value) > 0}
    second_active = {name for name, value in second.items() if int(value) > 0}
    new_categories = sorted(second_active - first_active)
    drift = {
        name: {
            "first": int(first.get(name, 0)),
            "second": int(second.get(name, 0)),
            "delta": int(second.get(name, 0)) - int(first.get(name, 0)),
        }
        for name in sorted(set(first) | set(second))
        if int(first.get(name, 0)) != int(second.get(name, 0))
    }
    return {
        "warning_categories_stable": not new_categories,
        "higher_iteration_new_warning_categories": new_categories,
        "warning_count_drift": drift,
        "warning_profile_nonworsening": all(
            int(second.get(name, 0)) <= int(first.get(name, 0))
            for name in set(first) | set(second)
        ),
    }


def relative_engineering_qa(run_qa: dict, treatment: dict[str, int],
                            baseline: dict[str, int]) -> dict:
    new_categories = sorted(
        name for name, value in treatment.items()
        if (
            value > 0
            and baseline.get(name, 0) == 0
            and name not in INHERITED_DRIFT_WARNING_CATEGORIES
        )
    )
    increases = {
        name: {"baseline": baseline.get(name, 0), "treatment": value}
        for name, value in treatment.items()
        if value > baseline.get(name, 0)
    }
    critical_increases = {
        name: values for name, values in increases.items()
        if name in CRITICAL_WARNING_CATEGORIES
    }
    inherited_environmental_drift = {
        name: {"baseline": baseline.get(name, 0), "treatment": value}
        for name, value in treatment.items()
        if (
            name in INHERITED_DRIFT_WARNING_CATEGORIES
            and value != baseline.get(name, 0)
        )
    }
    passed = bool(
        run_qa.get("completed")
        and run_qa.get("reported_severe_count") == 0
        and run_qa.get("fatal_occurrences") == 0
        and not new_categories
        and not critical_increases
        and not run_qa.get("unexplained_recurring_groups")
    )
    return {
        "engineering_pass_relative_to_prototype_baseline": passed,
        "baseline_warning_profile": baseline,
        "treatment_warning_profile": treatment,
        "new_warning_categories": new_categories,
        "warning_count_increases": increases,
        "critical_warning_count_increases": critical_increases,
        "inherited_environmental_warning_count_drift": inherited_environmental_drift,
    }


def read_solver_stability_evidence(path: Path | None) -> tuple[dict[tuple[str, float], dict], str | None]:
    """Load probe-specific higher-iteration evidence used by the probe gate."""
    if path is None:
        return {}, None
    frame = pd.read_csv(path)
    required = {
        "climate", "target_c", "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference",
        "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "trajectory_stable_within_declared_tolerance",
        "declared_temperature_tolerance_c",
        "declared_energy_relative_tolerance",
        "first_treatment_warning_profile", "second_treatment_warning_profile",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Solver-stability evidence lacks {sorted(missing)}")
    if frame.duplicated(["climate", "target_c"]).any():
        raise ValueError("Solver-stability evidence duplicates a climate/target key")
    stable = frame["trajectory_stable_within_declared_tolerance"].astype(
        str
    ).str.lower().map({"true": True, "false": False})
    if stable.isna().any():
        raise ValueError("Solver-stability evidence has an invalid stability flag")
    frame = frame.assign(trajectory_stable_within_declared_tolerance=stable)
    numeric = frame[[
        "target_c", "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference",
        "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "declared_temperature_tolerance_c",
        "declared_energy_relative_tolerance",
    ]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Solver-stability evidence contains non-finite values")
    frame[numeric.columns] = numeric
    differences = frame[[
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference",
        "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
    ]]
    if (differences < 0).any().any():
        raise ValueError("Solver-stability differences cannot be negative")
    if (
        (frame.declared_temperature_tolerance_c <= 0).any()
        or (frame.declared_energy_relative_tolerance <= 0).any()
    ):
        raise ValueError("Solver-stability tolerances must be positive")
    if (frame.second_maximum_hvac_iterations <= frame.first_maximum_hvac_iterations).any():
        raise ValueError("Higher-iteration evidence must use a larger solver limit")
    energy_difference = frame[[
        "hvac_electricity_relative_difference",
        "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
    ]]
    if (
        frame.operative_temperature_max_abs_difference_c
        > frame.declared_temperature_tolerance_c
    ).any() or energy_difference.gt(
        frame.declared_energy_relative_tolerance, axis="index"
    ).any().any():
        raise ValueError("Solver-stability evidence contradicts its declared tolerances")
    if not frame.trajectory_stable_within_declared_tolerance.all():
        raise ValueError("Every supplied solver-stability row must pass its declared limits")
    if not frame.first_treatment_warning_profile.eq(
        frame.second_treatment_warning_profile
    ).all():
        raise ValueError(
            "Higher-iteration evidence changed the treatment warning profile"
        )
    for value in frame.first_treatment_warning_profile:
        try:
            profile = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Solver-stability warning profile is not valid JSON") from exc
        if not isinstance(profile, dict):
            raise ValueError("Solver-stability warning profile must be a JSON object")
    evidence = {
        (str(row["climate"]), round(float(row["target_c"]), 6)): row
        for row in frame.to_dict("records")
    }
    return evidence, sha256(path)


def apply_solver_stability_exception(
    relative: dict,
    run_qa: dict,
    evidence: dict | None,
    *,
    current_maximum_hvac_iterations: int,
) -> tuple[bool, dict]:
    """Accept only a probe whose sole gate failure has verified stable outputs."""
    raw_pass = bool(relative["engineering_pass_relative_to_prototype_baseline"])
    audit = {
        "raw_relative_engineering_pass": raw_pass,
        "solver_stability_exception_applied": False,
    }
    if raw_pass or evidence is None:
        return raw_pass, audit
    only_hvac = bool(
        set(relative["new_warning_categories"]).issubset({"hvac_iterations"})
        and set(relative["critical_warning_count_increases"]).issubset(
            {"hvac_iterations"}
        )
        and relative["critical_warning_count_increases"]
    )
    clean_run = bool(
        run_qa.get("completed")
        and run_qa.get("reported_severe_count") == 0
        and run_qa.get("fatal_occurrences") == 0
        and not run_qa.get("unexplained_recurring_groups")
    )
    if not only_hvac or not clean_run:
        return False, audit
    try:
        evidence_profile = json.loads(evidence["first_treatment_warning_profile"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Solver-stability warning profile is not valid JSON") from exc
    if evidence_profile != relative["treatment_warning_profile"]:
        raise ValueError(
            "Solver-stability evidence does not match the active warning profile"
        )
    first_limit = int(evidence["first_maximum_hvac_iterations"])
    if first_limit != int(current_maximum_hvac_iterations):
        raise ValueError(
            "Solver-stability evidence does not match the active HVAC iteration limit"
        )
    audit.update({
        "solver_stability_exception_applied": True,
        "solver_stability_higher_iteration_limit": int(
            evidence["second_maximum_hvac_iterations"]
        ),
        "solver_stability_temperature_difference_c": float(
            evidence["operative_temperature_max_abs_difference_c"]
        ),
        "solver_stability_energy_relative_difference": float(
            evidence["total_purchased_energy_relative_difference"]
        ),
    })
    return True, audit


def read_annual_solver_stability_evidence(
    path: Path | None,
) -> tuple[dict[str, dict], str | None]:
    """Load exact annual-run 40-vs-100 stability evidence."""
    if path is None:
        return {}, None
    frame = pd.read_csv(path)
    required = {
        "run_id", "climate", "composition_id", "control", "offset_c",
        "first_maximum_hvac_iterations", "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "trajectory_stable_within_declared_tolerance",
        "declared_temperature_tolerance_c", "declared_energy_relative_tolerance",
        "higher_iteration_completed", "higher_iteration_severe_count",
        "higher_iteration_fatal_occurrences",
        "higher_iteration_relative_engineering_pass",
        "warning_profile_nonworsening",
        "first_treatment_warning_profile", "second_treatment_warning_profile",
        "first_input_sha256", "first_hourly_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Annual solver-stability evidence lacks {sorted(missing)}")
    if frame.run_id.duplicated().any():
        raise ValueError("Annual solver-stability evidence duplicates a run identifier")
    boolean_columns = [
        "trajectory_stable_within_declared_tolerance",
        "higher_iteration_completed",
        "higher_iteration_relative_engineering_pass",
        "warning_profile_nonworsening",
    ]
    boolean_columns.extend(
        column for column in ("warning_categories_stable", "solver_stability_pass")
        if column in frame
    )
    for column in boolean_columns:
        parsed = frame[column].astype(str).str.lower().map(
            {"true": True, "false": False}
        )
        if parsed.isna().any():
            raise ValueError(f"Annual solver-stability evidence has invalid {column}")
        frame[column] = parsed
    numeric_columns = [
        "composition_id", "offset_c", "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "declared_temperature_tolerance_c", "declared_energy_relative_tolerance",
        "higher_iteration_severe_count", "higher_iteration_fatal_occurrences",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Annual solver-stability evidence contains non-finite values")
    frame[numeric_columns] = numeric
    differences = frame[[
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
    ]]
    if (differences < 0).any().any():
        raise ValueError("Annual solver-stability differences cannot be negative")
    if (
        (frame.declared_temperature_tolerance_c <= 0).any()
        or (frame.declared_energy_relative_tolerance <= 0).any()
        or (frame.second_maximum_hvac_iterations <= frame.first_maximum_hvac_iterations).any()
    ):
        raise ValueError("Annual solver-stability limits are invalid")
    energy = frame[[
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
    ]]
    contradicts = bool(
        (
            frame.operative_temperature_max_abs_difference_c
            > frame.declared_temperature_tolerance_c
        ).any()
        or energy.gt(
            frame.declared_energy_relative_tolerance, axis="index"
        ).any().any()
    )
    if contradicts or not frame.trajectory_stable_within_declared_tolerance.all():
        raise ValueError("Annual solver-stability evidence fails its declared limits")
    category_stable: list[bool] = []
    new_categories: list[str] = []
    count_drift: list[str] = []
    stability_pass: list[bool] = []
    for item in frame.to_dict("records"):
        try:
            first = json.loads(item["first_treatment_warning_profile"])
            second = json.loads(item["second_treatment_warning_profile"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Annual solver-stability warning profile is not valid JSON"
            ) from exc
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError("Annual solver-stability warning profiles must be objects")
        comparison = compare_solver_warning_categories(first, second)
        computed_pass = bool(
            item["trajectory_stable_within_declared_tolerance"]
            and item["higher_iteration_completed"]
            and int(item["higher_iteration_severe_count"]) == 0
            and int(item["higher_iteration_fatal_occurrences"]) == 0
            and comparison["warning_categories_stable"]
        )
        if (
            "warning_categories_stable" in item
            and bool(item["warning_categories_stable"])
            != comparison["warning_categories_stable"]
        ):
            raise ValueError("Annual warning-category stability flag is inconsistent")
        if (
            "solver_stability_pass" in item
            and bool(item["solver_stability_pass"]) != computed_pass
        ):
            raise ValueError("Annual solver-stability pass flag is inconsistent")
        for column, computed in (
            (
                "higher_iteration_new_warning_categories",
                comparison["higher_iteration_new_warning_categories"],
            ),
            ("warning_count_drift", comparison["warning_count_drift"]),
        ):
            if column in item:
                try:
                    declared = json.loads(item[column])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Annual solver-stability {column} is not valid JSON"
                    ) from exc
                if declared != computed:
                    raise ValueError(
                        f"Annual solver-stability {column} is inconsistent"
                    )
        category_stable.append(bool(comparison["warning_categories_stable"]))
        new_categories.append(json.dumps(
            comparison["higher_iteration_new_warning_categories"], sort_keys=True
        ))
        count_drift.append(json.dumps(
            comparison["warning_count_drift"], sort_keys=True
        ))
        stability_pass.append(computed_pass)
        for column in ("first_input_sha256", "first_hourly_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(item[column]).lower()):
                raise ValueError(f"Annual solver-stability {column} is invalid")
    frame["warning_categories_stable"] = category_stable
    frame["higher_iteration_new_warning_categories"] = new_categories
    frame["warning_count_drift"] = count_drift
    frame["solver_stability_pass"] = stability_pass
    if not frame.solver_stability_pass.all():
        raise ValueError(
            "Annual higher-iteration evidence fails the category-and-trajectory rule"
        )
    evidence = {
        str(row["run_id"]): row for row in frame.to_dict("records")
    }
    return evidence, sha256(path)


def apply_annual_solver_stability_exception(
    relative: dict,
    run_qa: dict,
    evidence: dict | None,
    *,
    current_maximum_hvac_iterations: int,
    active_input_sha256: str,
    active_hourly_sha256: str,
) -> tuple[bool, dict]:
    """Qualify one exact annual run using matched higher-iteration evidence."""
    raw_pass = bool(relative["engineering_pass_relative_to_prototype_baseline"])
    audit = {
        "raw_relative_engineering_pass": raw_pass,
        "annual_solver_stability_exception_applied": False,
        "annual_solver_stability_pass": None,
    }
    if raw_pass or evidence is None:
        return raw_pass, audit
    only_hvac = bool(
        set(relative["new_warning_categories"]) == {"hvac_iterations"}
        and set(relative["critical_warning_count_increases"]).issubset(
            {"hvac_iterations"}
        )
    )
    clean_run = bool(
        run_qa.get("completed")
        and run_qa.get("reported_severe_count") == 0
        and run_qa.get("fatal_occurrences") == 0
        and not run_qa.get("unexplained_recurring_groups")
    )
    if not only_hvac or not clean_run:
        return False, audit
    try:
        evidence_profile = json.loads(evidence["first_treatment_warning_profile"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Annual solver-stability warning profile is not valid JSON"
        ) from exc
    if evidence_profile != relative["treatment_warning_profile"]:
        raise ValueError(
            "Annual solver-stability evidence does not match the active warning profile"
        )
    if int(evidence["first_maximum_hvac_iterations"]) != int(
        current_maximum_hvac_iterations
    ):
        raise ValueError(
            "Annual solver-stability evidence does not match the active iteration limit"
        )
    if str(evidence["first_input_sha256"]) != str(active_input_sha256):
        raise ValueError(
            "Annual solver-stability evidence does not match the active input"
        )
    if str(evidence["first_hourly_sha256"]) != str(active_hourly_sha256):
        raise ValueError(
            "Annual solver-stability evidence does not match the active trajectory"
        )
    if not bool(evidence.get("solver_stability_pass", False)):
        raise ValueError("Annual solver-stability evidence is not qualified")
    audit.update({
        "annual_solver_stability_exception_applied": True,
        "annual_solver_stability_higher_iteration_limit": int(
            evidence["second_maximum_hvac_iterations"]
        ),
        "annual_solver_stability_temperature_difference_c": float(
            evidence["operative_temperature_max_abs_difference_c"]
        ),
        "annual_solver_stability_energy_relative_difference": float(
            evidence["total_purchased_energy_relative_difference"]
        ),
        "annual_solver_stability_higher_iteration_relative_pass": bool(
            evidence.get("higher_iteration_relative_engineering_pass", False)
        ),
        "annual_solver_stability_warning_categories_stable": bool(
            evidence["warning_categories_stable"]
        ),
        "annual_solver_stability_warning_count_drift": str(
            evidence["warning_count_drift"]
        ),
        "annual_solver_stability_pass": bool(evidence["solver_stability_pass"]),
    })
    return True, audit


def _cached_run(job: dict, runtime_sha256: str, runtime_info: dict | None = None) -> dict | None:
    run_folder = job["run_folder"]
    compact = job["compact"]
    qa_path = run_folder / "RUN_QA.json"
    if not qa_path.is_file() or not compact.is_file():
        return None
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    expected = {
        "input_idf_sha256": job["input_sha256"],
        "weather_sha256": job["weather_sha256"],
        "runtime_sha256": runtime_sha256,
        "hourly_sha256": sha256(compact),
    }
    if qa.get("completed") and all(qa.get(key) == value for key, value in expected.items()):
        if runtime_info and qa.get("runtime_api_sha256") is not None:
            if any(qa.get(key) != runtime_info[key] for key in ("runtime_api_sha256", "runtime_package_manifest_sha256")):
                raise ValueError("Cached API runtime fingerprint differs from the declared environment")
        qa["cached_artifact_runtime_backend"] = qa.get("runtime_backend", "historical_executable")
        qa["compact_output"] = str(compact.resolve())
        for key in ("sql_metadata_path", "execution_audit_path"):
            if qa.get(key):
                qa[key] = str(run_folder / Path(qa[key]).name)
        if qa.get("sql_metadata_path") and qa.get("sql_metadata_sha256"):
            metadata = Path(qa["sql_metadata_path"])
            if not metadata.is_file() or sha256(metadata) != qa["sql_metadata_sha256"]:
                raise ValueError("Cached simulation metadata is missing or changed")
        return qa
    raise ValueError(
        f"Cached run provenance mismatch for {job['run_id']}; "
        "the existing folder was not reused"
    )


def _execute_jobs(jobs: list[dict], executable: Path, workers: int) -> list[dict]:
    runtime_hash = sha256(executable)
    from thermal_heterogeneity.energyplus_launcher import runtime_provenance
    runtime_info = runtime_provenance(executable)

    def execute(job: dict) -> dict:
        cached = _cached_run(job, runtime_hash, runtime_info)
        if cached is None:
            qa = run_energyplus(
                job["idf"], job["weather"], executable, job["run_folder"],
                job["compact"], ZONES, cleanup_raw=True,
            )
        else:
            qa = cached
        return job | {"run_qa": qa}

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(execute, job): job["run_id"] for job in jobs}
        for future in as_completed(pending):
            result = future.result()
            completed.append(result)
            print(json.dumps({
                "run_id": result["run_id"],
                "climate": result["climate"],
                "completed": result["run_qa"].get("completed"),
            }), flush=True)
    return completed


def _write_run_idf(model: Path, targets: dict[str, float], destination: Path) -> None:
    source = model.read_text(encoding="utf-8", errors="replace")
    text = annual_input(source, targets, HALF_DEADBAND_C)
    if physical_hash(text) != physical_hash(source):
        raise ValueError("Thermostat command changed non-control building physics")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def run_probes(climates: list[ClimateSpec], model_manifest: dict[str, dict],
               output_root: Path, executable: Path, workers: int,
               solver_stability_manifest: Path | None = None) -> dict[str, Path]:
    stability, stability_sha256 = read_solver_stability_evidence(
        solver_stability_manifest
    )
    jobs: list[dict] = []
    runtime_hash = sha256(executable)
    for climate in climates:
        cfg = model_manifest[climate.climate]
        model = Path(cfg["prepared_idf"])
        weather = Path(cfg["weather_path"])
        for target in PROBE_TARGETS_C:
            payload = {
                "kind": "probe", "climate": climate.climate, "target_c": target,
                "model_sha256": cfg["prepared_idf_sha256"],
                "weather_sha256": cfg["weather_sha256"],
                "runtime_sha256": runtime_hash,
            }
            run_id = "probe_" + _hash_payload(payload)
            idf = output_root / "inputs" / f"{run_id}.idf"
            if not idf.is_file():
                _write_run_idf(model, dict.fromkeys(ZONES, float(target)), idf)
            jobs.append({
                "run_id": run_id, "climate": climate.climate, "target_c": float(target),
                "idf": idf, "weather": weather,
                "run_folder": output_root / "runs" / run_id,
                "compact": output_root / "hourly" / f"{run_id}.npz",
                "input_sha256": sha256(idf),
                "weather_sha256": cfg["weather_sha256"],
            })
    records = _execute_jobs(jobs, executable, workers)
    probe_paths: dict[str, Path] = {}
    qa_rows = []
    for climate in climates:
        rows = sorted(
            (row for row in records if row["climate"] == climate.climate),
            key=lambda row: row["target_c"],
        )
        baseline = next(row for row in rows if row["target_c"] == 25.0)
        baseline_profile = warning_profile(baseline["run_folder"] / "eplusout.err")
        temperatures = []
        purchased = []
        native_people = None
        occupied_indices = None
        for row in rows:
            profile = warning_profile(row["run_folder"] / "eplusout.err")
            relative = relative_engineering_qa(row["run_qa"], profile, baseline_profile)
            stability_row = stability.get(
                (row["climate"], round(float(row["target_c"]), 6))
            )
            final_pass, stability_audit = apply_solver_stability_exception(
                relative,
                row["run_qa"],
                stability_row,
                current_maximum_hvac_iterations=int(
                    model_manifest[climate.climate]["solver_configuration"][
                        "maximum_hvac_iterations"
                    ]
                ),
            )
            row.update(relative)
            row["relative_engineering_pass"] = final_pass
            with np.load(row["compact"], allow_pickle=False) as saved:
                occupied = saved["occupied"].astype(bool)
                if occupied_indices is None:
                    occupied_indices = np.flatnonzero(occupied)
                    native_people = saved["people"][occupied]
                elif not np.array_equal(native_people, saved["people"][occupied]):
                    raise ValueError(f"Occupancy changed across {climate.climate} probes")
                temperatures.append(saved["temperatures"][occupied])
                purchased.append(float(
                    saved["hvac_electric_kwh"].sum() + saved["gas_hhv_kwh"].sum()
                ))
            qa_rows.append({
                "run_id": row["run_id"], "climate": row["climate"],
                "target_c": row["target_c"], "completed": row["run_qa"].get("completed"),
                "maximum_hvac_iterations": int(
                    model_manifest[climate.climate]["solver_configuration"][
                        "maximum_hvac_iterations"
                    ]
                ),
                "relative_engineering_pass": final_pass,
                **stability_audit,
                "solver_stability_evidence_sha256": (
                    stability_sha256 if stability_audit[
                        "solver_stability_exception_applied"
                    ] else None
                ),
                "new_warning_categories": json.dumps(relative["new_warning_categories"]),
                "warning_count_increases": json.dumps(relative["warning_count_increases"]),
                "critical_warning_count_increases": json.dumps(
                    relative["critical_warning_count_increases"]
                ),
                "inherited_environmental_warning_count_drift": json.dumps(
                    relative["inherited_environmental_warning_count_drift"]
                ),
                "baseline_warning_profile": json.dumps(
                    relative["baseline_warning_profile"]
                ),
                "treatment_warning_profile": json.dumps(
                    relative["treatment_warning_profile"]
                ),
                "input_sha256": row["input_sha256"],
                "hourly_path": str(row["compact"].resolve()),
                "hourly_sha256": row["run_qa"].get("hourly_sha256"),
            })
        if any(not row["relative_engineering_pass"] for row in qa_rows
               if row["climate"] == climate.climate):
            pd.DataFrame(qa_rows).to_csv(
                output_root / "PROBE_RUN_MANIFEST.csv", index=False
            )
            failed = [
                row["target_c"] for row in qa_rows
                if row["climate"] == climate.climate
                and not row["relative_engineering_pass"]
            ]
            raise RuntimeError(
                f"Probe engineering gate failed for {climate.climate} at {failed}; "
                "flagged probes are not removed or interpolated"
            )
        destination = output_root / "probes" / f"{climate.climate}_probes.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            uniform_target_c=np.asarray([row["target_c"] for row in rows]),
            operative_c=np.asarray(temperatures),
            native_people=native_people,
            purchased_site_kwh=np.asarray(purchased),
            zone_ids=np.asarray(ZONES),
            occupied_source_indices=occupied_indices,
        )
        probe_paths[climate.climate] = destination
    pd.DataFrame(qa_rows).to_csv(output_root / "PROBE_RUN_MANIFEST.csv", index=False)
    return probe_paths


def _controller_root(climates: list[ClimateSpec], probe_paths: dict[str, Path],
                     profiles: pd.DataFrame, output_root: Path) -> Path:
    root = output_root / "controller_input"
    controller = root / "controller"
    controller.mkdir(parents=True, exist_ok=True)
    contexts = {}
    for climate in climates:
        destination = controller / f"{climate.climate}_probes.npz"
        if destination.resolve() != probe_paths[climate.climate].resolve():
            destination.write_bytes(probe_paths[climate.climate].read_bytes())
        contexts[climate.climate] = {
            "zones": list(ZONES), "floor_groups": FLOOR_GROUPS,
            "floor_binding_equals_building": False,
            "half_deadband_c": HALF_DEADBAND_C, "floor_area_m2": FLOOR_AREA_M2,
            "probes": f"controller/{destination.name}",
            "profiles": "unused_when_profiles_csv_is_supplied.npz",
        }
    _write_json(controller / "CONTEXTS.json", contexts)
    profiles.to_csv(root / "GLOBAL_PROFILES.csv", index=False)
    return root


def generate_control_designs(
    climates: list[ClimateSpec],
    controller_root: Path,
    candidate_scope: str = "sentinel",
) -> list[dict]:
    profiles = pd.read_csv(controller_root / "GLOBAL_PROFILES.csv")
    designs, diagnostics = generate_controls(
        controller_root,
        profile_table=profiles,
        permissions=("native_zone",),
        offsets_c=OFFSETS_C,
    )
    retained = retain_control_designs(climates, designs, candidate_scope)
    sentinel = {row.climate for row in climates if row.sentinel}
    expanded_climates = len(climates) if candidate_scope == "all" else len(sentinel)
    expected = len(climates) * 6 * 3 + expanded_climates * 6 * 3 * 4
    if len(retained) != expected:
        raise ValueError(f"Expected {expected} control designs, found {len(retained)}")
    _write_json(controller_root / "CONTROL_COMMANDS.json", retained)
    _write_json(controller_root / "TARGET_DIAGNOSTICS.json", diagnostics)
    return retained


def retain_control_designs(
    climates: list[ClimateSpec],
    designs: list[dict],
    candidate_scope: str = "sentinel",
) -> list[dict]:
    """Keep direct operation globally and the requested five-candidate scope."""
    if candidate_scope not in {"sentinel", "all"}:
        raise ValueError(f"Unknown candidate scope: {candidate_scope}")
    if candidate_scope == "all":
        return list(designs)
    sentinel = {row.climate for row in climates if row.sentinel}
    return [
        row for row in designs
        if float(row["offset_c"]) == 0.0 or row["context"] in sentinel
    ]


def _control_identity(row: dict | pd.Series) -> tuple[str, int, str, float]:
    climate = row["context"] if "context" in row else row["climate"]
    return (
        str(climate),
        int(row["composition_id"]),
        str(row["control"]),
        round(float(row["offset_c"]), 9),
    )


def select_missing_control_designs(
    designs: list[dict], existing_manifest: Path
) -> tuple[pd.DataFrame, list[dict]]:
    """Verify exact completed evidence and return only absent control designs."""
    frame = pd.read_csv(existing_manifest)
    required = {
        *CONTROL_IDENTITY, "completed", "relative_engineering_pass",
        "input_sha256", "targets_sha256", "input_idf_path",
        "hourly_sha256", "hourly_path",
    }
    missing_columns = required - set(frame)
    if missing_columns:
        raise ValueError(
            f"Existing annual manifest lacks {sorted(missing_columns)}"
        )
    if frame.duplicated(list(CONTROL_IDENTITY)).any():
        raise ValueError("Existing annual manifest duplicates a control identity")
    completed = frame["completed"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    passed = frame["relative_engineering_pass"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if completed.isna().any() or passed.isna().any():
        raise ValueError("Existing annual manifest has invalid QA flags")
    if not completed.all() or not passed.all():
        raise ValueError("Every reused annual run must be complete and QA-qualified")
    design_by_identity = {_control_identity(row): row for row in designs}
    if len(design_by_identity) != len(designs):
        raise ValueError("Generated control designs contain duplicate identities")
    existing_identities: set[tuple[str, int, str, float]] = set()
    for row in frame.to_dict("records"):
        identity = _control_identity(row)
        design = design_by_identity.get(identity)
        if design is None:
            raise ValueError(f"Existing annual run is outside the active design: {identity}")
        if str(row["targets_sha256"]) != _hash_payload(design["targets_c"]):
            raise ValueError(f"Existing targets differ from the active design: {identity}")
        for path_column, hash_column in (
            ("input_idf_path", "input_sha256"),
            ("hourly_path", "hourly_sha256"),
        ):
            path = Path(str(row[path_column]))
            if not path.is_file() or sha256(path) != str(row[hash_column]):
                raise ValueError(
                    f"Existing {path_column} evidence is missing or changed: {identity}"
                )
        existing_identities.add(identity)
    missing = [
        design for design in designs
        if _control_identity(design) not in existing_identities
    ]
    return frame, missing


def merge_control_manifests(
    existing: pd.DataFrame,
    extension_manifest: Path,
    designs: list[dict],
    destination: Path,
    annual_solver_stability_manifests: tuple[Path, ...] = (),
) -> pd.DataFrame:
    """Combine reused and new exact evidence into one complete candidate library."""
    extension = pd.read_csv(extension_manifest)
    combined = (
        existing.copy()
        if extension.empty
        else pd.concat([existing, extension], ignore_index=True, sort=False)
    )
    if combined.duplicated(list(CONTROL_IDENTITY)).any():
        raise ValueError("Combined annual manifest duplicates a control identity")
    expected = {_control_identity(row) for row in designs}
    found = {_control_identity(row) for row in combined.to_dict("records")}
    if found != expected:
        raise ValueError(
            f"Combined annual manifest differs from the active design: "
            f"missing={len(expected - found)}, extra={len(found - expected)}"
        )
    combined = normalize_combined_annual_stability_evidence(
        combined, annual_solver_stability_manifests
    )
    combined = combined.sort_values(list(CONTROL_IDENTITY), kind="stable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    return combined


def _manifest_boolean(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    parsed = str(value).strip().lower()
    if parsed in {"true", "1"}:
        return True
    if parsed in {"false", "0", "nan", "none", ""}:
        return False
    raise ValueError(f"Combined annual manifest has invalid {name}: {value!r}")


def normalize_combined_annual_stability_evidence(
    frame: pd.DataFrame,
    evidence_paths: tuple[Path, ...],
) -> pd.DataFrame:
    """Normalize legacy and current solver evidence after a manifest merge.

    Older accepted manifests pre-date the explicit category-stability and
    solver-pass columns.  This function rereads the exact, hash-bound evidence
    and writes the current fields without changing a trajectory or result.
    """
    result = frame.copy()
    applied_name = "annual_solver_stability_exception_applied"
    if applied_name not in result:
        result[applied_name] = False
    applied = result[applied_name].map(
        lambda value: _manifest_boolean(value, applied_name)
    )
    for column in (
        "annual_solver_stability_warning_categories_stable",
        "annual_solver_stability_warning_count_drift",
        "annual_solver_stability_pass",
    ):
        if column not in result:
            result[column] = pd.NA
    if not applied.any():
        return result

    evidence: dict[str, dict] = {}
    evidence_hashes: dict[str, str] = {}
    for path in evidence_paths:
        rows, evidence_hash = read_annual_solver_stability_evidence(path.resolve())
        if evidence_hash is None:
            raise ValueError("Annual solver-stability evidence hash is unavailable")
        overlap = set(evidence).intersection(rows)
        if overlap:
            raise ValueError(
                "Annual solver-stability manifests duplicate run identifiers: "
                f"{sorted(overlap)}"
            )
        evidence.update(rows)
        evidence_hashes.update(dict.fromkeys(rows, evidence_hash))

    active_ids = set(result.loc[applied, "run_id"].astype(str))
    if set(evidence) != active_ids:
        raise ValueError(
            "Annual solver-stability evidence does not exactly cover merged "
            f"exceptions: missing={sorted(active_ids - set(evidence))}, "
            f"extra={sorted(set(evidence) - active_ids)}"
        )

    for index in result.index[applied]:
        row = result.loc[index]
        run_id = str(row.run_id)
        item = evidence[run_id]
        identity = {
            "climate": str(row.climate),
            "composition_id": int(row.composition_id),
            "control": str(row.control),
            "offset_c": float(row.offset_c),
        }
        evidence_identity = {
            "climate": str(item["climate"]),
            "composition_id": int(item["composition_id"]),
            "control": str(item["control"]),
            "offset_c": float(item["offset_c"]),
        }
        if identity != evidence_identity:
            raise ValueError(
                f"Annual solver-stability identity changed for {run_id}"
            )
        if str(row.input_sha256) != str(item["first_input_sha256"]):
            raise ValueError(
                f"Annual solver-stability input hash changed for {run_id}"
            )
        declared_hash = row.get("annual_solver_stability_evidence_sha256")
        expected_hash = evidence_hashes[run_id]
        if pd.notna(declared_hash) and str(declared_hash) != expected_hash:
            raise ValueError(
                f"Annual solver-stability evidence hash changed for {run_id}"
            )
        if str(row.hourly_sha256) != str(item["first_hourly_sha256"]):
            enriched = _manifest_boolean(
                row.get("facility_energy_enriched", False),
                "facility_energy_enriched",
            )
            compact = Path(str(row.hourly_path))
            qa_path = Path(str(row.qa_path))
            if not enriched or not compact.is_file() or not qa_path.is_file():
                raise ValueError(
                    f"Annual solver-stability trajectory hash changed for {run_id}"
                )
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            if (
                sha256(compact) != str(row.hourly_sha256)
                or str(qa.get("hourly_sha256", "")) != str(row.hourly_sha256)
                or qa.get("facility_energy_enriched") is not True
            ):
                raise ValueError(
                    f"Facility-energy enrichment provenance changed for {run_id}"
                )
        try:
            warning_profile = json.loads(str(row.treatment_warning_profile))
            evidence_profile = json.loads(
                str(item["first_treatment_warning_profile"])
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Annual solver-stability warning profile is invalid for {run_id}"
            ) from exc
        if warning_profile != evidence_profile:
            raise ValueError(
                f"Annual solver-stability warning profile changed for {run_id}"
            )
        if not bool(item["solver_stability_pass"]):
            raise ValueError(
                f"Annual solver-stability evidence is not qualified for {run_id}"
            )
        result.at[index, "annual_solver_stability_evidence_sha256"] = expected_hash
        result.at[
            index, "annual_solver_stability_warning_categories_stable"
        ] = bool(item["warning_categories_stable"])
        result.at[index, "annual_solver_stability_warning_count_drift"] = str(
            item["warning_count_drift"]
        )
        result.at[index, "annual_solver_stability_pass"] = True
    return result


def run_controls(designs: list[dict], model_manifest: dict[str, dict],
                 probe_output_root: Path, output_root: Path, executable: Path,
                 workers: int,
                 annual_solver_stability_manifest: Path | None = None) -> None:
    runtime_hash = sha256(executable)
    stability, stability_sha256 = read_annual_solver_stability_evidence(
        annual_solver_stability_manifest
    )
    used_stability_rows: set[str] = set()
    baseline_profiles = {}
    for climate, cfg in model_manifest.items():
        manifest = pd.read_csv(probe_output_root / "PROBE_RUN_MANIFEST.csv")
        baseline_id = manifest[
            manifest["climate"].eq(climate) & manifest["target_c"].eq(25.0)
        ]["run_id"].iloc[0]
        baseline_profiles[climate] = warning_profile(
            probe_output_root / "runs" / baseline_id / "eplusout.err"
        )
    jobs = []
    for design in designs:
        climate = design["context"]
        cfg = model_manifest[climate]
        payload = {
            "kind": "control", "climate": climate,
            "composition_id": int(design["composition_id"]),
            "control": design["control"], "offset_c": float(design["offset_c"]),
            "targets_c": design["targets_c"],
            "model_sha256": cfg["prepared_idf_sha256"],
            "weather_sha256": cfg["weather_sha256"],
            "runtime_sha256": runtime_hash,
        }
        run_id = "annual_" + _hash_payload(payload)
        idf = output_root / "inputs" / f"{run_id}.idf"
        if not idf.is_file():
            _write_run_idf(Path(cfg["prepared_idf"]), design["targets_c"], idf)
        jobs.append({
            "run_id": run_id, "climate": climate,
            "composition_id": int(design["composition_id"]),
            "cohort_label": COHORT_LABELS[int(design["composition_id"])],
            "control": design["control"], "offset_c": float(design["offset_c"]),
            "idf": idf, "weather": Path(cfg["weather_path"]),
            "run_folder": output_root / "runs" / run_id,
            "compact": output_root / "hourly" / f"{run_id}.npz",
            "input_sha256": sha256(idf),
            "weather_sha256": cfg["weather_sha256"],
            "targets_sha256": _hash_payload(design["targets_c"]),
        })
    records = _execute_jobs(jobs, executable, workers)
    rows = []
    for row in records:
        profile = warning_profile(row["run_folder"] / "eplusout.err")
        relative = relative_engineering_qa(
            row["run_qa"], profile, baseline_profiles[row["climate"]]
        )
        stability_row = stability.get(str(row["run_id"]))
        if stability_row is not None:
            identity = {
                "climate": str(row["climate"]),
                "composition_id": int(row["composition_id"]),
                "control": str(row["control"]),
                "offset_c": float(row["offset_c"]),
            }
            evidence_identity = {
                "climate": str(stability_row["climate"]),
                "composition_id": int(stability_row["composition_id"]),
                "control": str(stability_row["control"]),
                "offset_c": float(stability_row["offset_c"]),
            }
            if identity != evidence_identity:
                raise ValueError(
                    "Annual solver-stability evidence does not match the active run identity"
                )
        final_pass, stability_audit = apply_annual_solver_stability_exception(
            relative,
            row["run_qa"],
            stability_row,
            current_maximum_hvac_iterations=int(
                model_manifest[row["climate"]]["solver_configuration"][
                    "maximum_hvac_iterations"
                ]
            ),
            active_input_sha256=row["input_sha256"],
            active_hourly_sha256=row["run_qa"].get("hourly_sha256", ""),
        )
        if stability_audit["annual_solver_stability_exception_applied"]:
            used_stability_rows.add(str(row["run_id"]))
        rows.append({
            "run_id": row["run_id"], "climate": row["climate"],
            "composition_id": row["composition_id"], "cohort_label": row["cohort_label"],
            "control": row["control"], "offset_c": row["offset_c"],
            "maximum_hvac_iterations": int(
                model_manifest[row["climate"]]["solver_configuration"][
                    "maximum_hvac_iterations"
                ]
            ),
            "completed": row["run_qa"].get("completed"),
            "relative_engineering_pass": final_pass,
            "new_warning_categories": json.dumps(relative["new_warning_categories"]),
            "warning_count_increases": json.dumps(relative["warning_count_increases"]),
            "critical_warning_count_increases": json.dumps(
                relative["critical_warning_count_increases"]
            ),
            "inherited_environmental_warning_count_drift": json.dumps(
                relative["inherited_environmental_warning_count_drift"]
            ),
            "baseline_warning_profile": json.dumps(
                relative["baseline_warning_profile"]
            ),
            "treatment_warning_profile": json.dumps(
                relative["treatment_warning_profile"]
            ),
            "input_sha256": row["input_sha256"],
            "targets_sha256": row["targets_sha256"],
            "input_idf_path": str(row["idf"].resolve()),
            "weather_path": str(row["weather"].resolve()),
            "qa_path": str((row["run_folder"] / "RUN_QA.json").resolve()),
            "run_folder": str(row["run_folder"].resolve()),
            "hourly_path": str(row["compact"].resolve()),
            "hourly_sha256": row["run_qa"].get("hourly_sha256"),
            **stability_audit,
            "annual_solver_stability_evidence_sha256": (
                stability_sha256
                if stability_audit["annual_solver_stability_exception_applied"]
                else None
            ),
        })
    unused_stability_rows = set(stability) - used_stability_rows
    if unused_stability_rows:
        raise ValueError(
            "Annual solver-stability evidence is stale or does not correspond to "
            f"an active HVAC-only failure: {sorted(unused_stability_rows)}"
        )
    frame = pd.DataFrame(rows).sort_values(
        ["climate", "composition_id", "control", "offset_c"]
    )
    frame.to_csv(output_root / "ANNUAL_CONTROL_RUN_MANIFEST.csv", index=False)
    execution_qa = {
        "logical_runs": len(frame),
        "completed_runs": int(frame["completed"].sum()),
        "raw_relative_engineering_passes": int(
            frame["raw_relative_engineering_pass"].sum()
        ),
        "relative_engineering_passes": int(frame["relative_engineering_pass"].sum()),
        "annual_solver_stability_exceptions": int(
            frame["annual_solver_stability_exception_applied"].sum()
        ),
        "all_completed": bool(frame["completed"].all()),
        "all_passed_relative_engineering_gate": bool(
            frame["relative_engineering_pass"].all()
        ),
        "annual_solver_stability_evidence_sha256": stability_sha256,
        "building_parameters_changed": False,
    }
    _write_json(output_root / "EXECUTION_QA.json", execution_qa)
    if not execution_qa["all_completed"] or not execution_qa[
        "all_passed_relative_engineering_gate"
    ]:
        failed = frame.loc[
            ~frame["completed"] | ~frame["relative_engineering_pass"],
            ["climate", "composition_id", "control", "offset_c", "run_id"],
        ].to_dict("records")
        raise RuntimeError(
            "Annual-control engineering gate failed; manifests were retained and "
            f"failed runs were not used: {failed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--climates", type=Path,
        default=GLOBAL / "GLOBAL_CLIMATES.json",
    )
    parser.add_argument("--prototype-root", type=Path, required=True)
    parser.add_argument("--weather-root", type=Path, required=True)
    parser.add_argument("--reference-idf", type=Path, required=True)
    parser.add_argument(
        "--prototype-mode",
        choices=("fixed_reference", "climate_matched"),
        default="fixed_reference",
        help="Hold one medium-office model fixed across climates, or use matched DOE variants.",
    )
    parser.add_argument("--ashrae-profiles", type=Path, required=True)
    parser.add_argument("--ctcd-profiles", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--solver-stability-manifest", type=Path,
        help=(
            "Optional probe-specific comparison against a higher HVAC iteration "
            "limit. It can qualify only otherwise-clean probes whose sole gate "
            "failure is an HVAC maximum-iteration warning."
        ),
    )
    parser.add_argument(
        "--annual-solver-stability-manifest", type=Path,
        help=(
            "Optional run-specific comparison against higher-iteration annual "
            "trajectories. It can qualify only completed annual runs whose sole "
            "raw gate failure is an HVAC maximum-iteration warning."
        ),
    )
    parser.add_argument(
        "--reused-annual-solver-stability-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Exact higher-iteration evidence for exceptions already present in "
            "--extend-from-annual-manifest. Repeat for multiple source manifests."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--automatic-solver-stability", action="store_true",
                        help="Generate fresh40-versus100 iteration evidence for otherwise-clean HVAC-iteration-only gate failures.")
    parser.add_argument(
        "--candidate-scope",
        choices=("sentinel", "all"),
        default="sentinel",
        help=(
            "Generate five offsets only in sentinel climates or in every climate. "
            "Existing exact runs are reused by their provenance hashes."
        ),
    )
    parser.add_argument(
        "--extend-from-annual-manifest",
        type=Path,
        help=(
            "Reuse an exact completed annual manifest and execute only candidate "
            "identities that are absent from it. The extension and combined "
            "manifests are written separately."
        ),
    )
    args = parser.parse_args()

    climates = read_climates(args.climates.resolve())
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    profiles = build_profiles(
        climates, args.ashrae_profiles.resolve(), args.ctcd_profiles.resolve()
    )
    profiles.to_csv(root / "GLOBAL_PROFILES.csv", index=False)
    model_manifest = prepare_models(
        climates, args.prototype_root.resolve(), args.weather_root.resolve(),
        args.reference_idf.resolve(), root / "models", args.prototype_mode,
    )
    try:
        probes = run_probes(
            climates, model_manifest, root / "probe_runs", args.runtime.resolve(), args.workers,
            args.solver_stability_manifest.resolve() if args.solver_stability_manifest is not None else None,
        )
    except RuntimeError:
        if not args.automatic_solver_stability or args.solver_stability_manifest is not None:
            raise
        from automatic_solver_stability import audit_probes
        evidence = audit_probes(root, args.runtime.resolve(), args.workers)
        probes = run_probes(climates, model_manifest, root / "probe_runs", args.runtime.resolve(), args.workers, evidence)
    if args.automatic_solver_stability:
        from automatic_solver_stability import ensure_reference_metadata
        ensure_reference_metadata(root, args.runtime.resolve(), args.workers)
    controller = _controller_root(climates, probes, profiles, root / "control_generation")
    designs = generate_control_designs(climates, controller, args.candidate_scope)
    existing_annual = None
    designs_to_run = designs
    annual_output_root = root / "annual_runs"
    if args.extend_from_annual_manifest is not None:
        if args.candidate_scope != "all":
            raise ValueError(
                "Annual extension requires --candidate-scope all"
            )
        existing_annual, designs_to_run = select_missing_control_designs(
            designs, args.extend_from_annual_manifest.resolve()
        )
        annual_output_root = root / "annual_candidate_extension"
    try:
        run_controls(
            designs_to_run, model_manifest, root / "probe_runs", annual_output_root,
            args.runtime.resolve(), args.workers,
            args.annual_solver_stability_manifest.resolve() if args.annual_solver_stability_manifest is not None else None,
        )
    except RuntimeError:
        if not args.automatic_solver_stability or args.annual_solver_stability_manifest is not None:
            raise
        from automatic_solver_stability import audit_annual
        args.annual_solver_stability_manifest = audit_annual(
            annual_output_root / "ANNUAL_CONTROL_RUN_MANIFEST.csv", args.runtime.resolve(),
            root / "solver_stability/annual", args.workers,
        )
        run_controls(designs_to_run, model_manifest, root / "probe_runs", annual_output_root,
                     args.runtime.resolve(), args.workers, args.annual_solver_stability_manifest)
    annual_manifest = annual_output_root / "ANNUAL_CONTROL_RUN_MANIFEST.csv"
    if existing_annual is not None:
        extension_manifest = (
            annual_output_root / "ANNUAL_CONTROL_EXTENSION_RUN_MANIFEST.csv"
        )
        annual_manifest.replace(extension_manifest)
        annual_manifest = root / "annual_runs_full" / "ANNUAL_CONTROL_RUN_MANIFEST.csv"
        combined = merge_control_manifests(
            existing_annual,
            extension_manifest,
            designs,
            annual_manifest,
            annual_solver_stability_manifests=tuple(
                path.resolve()
                for path in (
                    [*args.reused_annual_solver_stability_manifest]
                    + (
                        [args.annual_solver_stability_manifest]
                        if args.annual_solver_stability_manifest is not None
                        else []
                    )
                )
            ),
        )
        _write_json(root / "annual_runs_full" / "EXECUTION_QA.json", {
            "logical_runs": len(combined),
            "reused_runs": len(existing_annual),
            "new_runs": len(designs_to_run),
            "completed_runs": int(
                combined["completed"].astype(str).str.lower().eq("true").sum()
            ),
            "relative_engineering_passes": int(
                combined["relative_engineering_pass"]
                .astype(str).str.lower().eq("true").sum()
            ),
            "all_completed": bool(
                combined["completed"].astype(str).str.lower().eq("true").all()
            ),
            "all_passed_relative_engineering_gate": bool(
                combined["relative_engineering_pass"]
                .astype(str).str.lower().eq("true").all()
            ),
            "building_parameters_changed": False,
            "source_manifest": str(args.extend_from_annual_manifest.resolve()),
            "source_manifest_sha256": sha256(
                args.extend_from_annual_manifest.resolve()
            ),
            "extension_manifest": str(extension_manifest.resolve()),
            "extension_manifest_sha256": sha256(extension_manifest),
            "merged_solver_stability_evidence": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256(path.resolve()),
                }
                for path in (
                    [*args.reused_annual_solver_stability_manifest]
                    + (
                        [args.annual_solver_stability_manifest]
                        if args.annual_solver_stability_manifest is not None
                        else []
                    )
                )
            ],
        })
    _write_json(root / "GLOBAL_EXECUTION_PROTOCOL.json", {
        "climates": len(climates), "compositions_per_climate": 6,
        "probe_runs": len(climates) * len(PROBE_TARGETS_C),
        "annual_control_runs": len(designs),
        "reused_annual_control_runs": (
            len(existing_annual) if existing_annual is not None else 0
        ),
        "new_annual_control_runs": len(designs_to_run),
        "candidate_scope": args.candidate_scope,
        "candidate_offsets_c": list(OFFSETS_C),
        "sentinel_climates": [row.climate for row in climates if row.sentinel],
        "runtime_sha256": sha256(args.runtime.resolve()),
        "climate_spec_sha256": sha256(args.climates.resolve()),
        "profiles_sha256": sha256(root / "GLOBAL_PROFILES.csv"),
        "model_manifest_sha256": sha256(root / "models" / "MODEL_MANIFEST.json"),
        "probe_manifest_sha256": sha256(
            root / "probe_runs" / "PROBE_RUN_MANIFEST.csv"
        ),
        "control_commands_sha256": sha256(
            controller / "CONTROL_COMMANDS.json"
        ),
        "target_diagnostics_sha256": sha256(
            controller / "TARGET_DIAGNOSTICS.json"
        ),
        "annual_manifest_sha256": sha256(
            annual_manifest
        ),
        "prototype_mode": args.prototype_mode,
        "solver_configuration": {
            "maximum_hvac_iterations": MAXIMUM_HVAC_ITERATIONS,
            "minimum_plant_iterations": "prototype_default",
        },
        "solver_stability_evidence": (
            {
                "path": str(args.solver_stability_manifest.resolve()),
                "sha256": sha256(args.solver_stability_manifest.resolve()),
            }
            if args.solver_stability_manifest is not None else None
        ),
        "annual_solver_stability_evidence": (
            {
                "path": str(args.annual_solver_stability_manifest.resolve()),
                "sha256": sha256(args.annual_solver_stability_manifest.resolve()),
            }
            if args.annual_solver_stability_manifest is not None else None
        ),
    })


if __name__ == "__main__":
    main()
