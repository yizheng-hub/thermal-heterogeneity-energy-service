"""Plan, run and score a multi-season demand-response extension.

The protocol fixes two occupied two-hour events in every meteorological
season from a declared 25 °C reference trajectory.  Controller comparisons
then start from each controller's own annual operation.  Cooling events alter
only the cooling boundary and heating events alter only the heating boundary;
the inactive side of the deadband is unchanged.  A compact action library
tests uniform shedding, preconditioning and a zone-protected action, while
the scoring stage follows electricity and thermal service through the event
and the following 48 hours.

The global-climate run manifests provide the reference trajectories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
TARGETED = HERE.parents[1] / "targeted_extension"
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(TARGETED))
sys.path.insert(0, str(ENERGY_SERVICE))

from common_baseline_dr import split_action_block  # noqa: E402
from generate_model_inputs import objects, physical_hash, signature  # noqa: E402
from multi_event_protocol import select_multi_event_protocol  # noqa: E402

try:  # package import in tests and installed use
    from .enrich_facility_energy import extract_facility_electric_kwh
    from .global_matrix import run_energyplus, sha256
    from .run_global_matrix import (
        compare_solver_warning_categories,
        relative_engineering_qa,
        solver_configuration,
        warning_profile,
    )
except ImportError:  # direct script execution
    from enrich_facility_energy import extract_facility_electric_kwh
    from global_matrix import run_energyplus, sha256  # type: ignore[no-redef]
    from run_global_matrix import (  # type: ignore[no-redef]
        compare_solver_warning_categories,
        relative_engineering_qa,
        solver_configuration,
        warning_profile,
    )


DEFAULT_LIMIT_FACTORS = (1.0, 1.05, 1.10, 1.15, 1.20)
MINIMUM_EVENT_DEADBAND_C = 1.0
CONTROLS = {"uniform_average", "personal_center", "personal_range"}
REQUIRED_PROFILE_COLUMNS = {
    "profile_id", "zone_id", "centre_c", "controller_lower_c", "controller_upper_c"
}


def _profile_climate_column(profiles: pd.DataFrame) -> str:
    if "climate" in profiles:
        return "climate"
    if "context" in profiles:
        return "context"
    raise ValueError("Profile table requires a climate or context column")


def _array(saved: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in saved:
            return np.asarray(saved[name])
    raise KeyError(f"Compact trajectory lacks all alternatives: {names}")


def load_compact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        result = {name: np.asarray(saved[name]) for name in saved.files}
    required = {
        "facility_electric_kwh", "hvac_electric_kwh", "heating_electric_kwh",
        "cooling_electric_kwh", "fans_electric_kwh", "gas_hhv_kwh",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    return result


def compact_timestamps(compact: Mapping[str, np.ndarray], year: int = 2001) -> pd.DatetimeIndex:
    """Rebuild hourly local standard-time stamps from compact EnergyPlus output."""
    month = _array(compact, "month").astype(int)
    day = _array(compact, "day").astype(int)
    hour = _array(compact, "hour").astype(int)
    if not (month.shape == day.shape == hour.shape) or month.ndim != 1:
        raise ValueError("Compact month, day and hour arrays must align")
    if ((hour < 1) | (hour > 24)).any():
        raise ValueError("EnergyPlus hour must be in 1..24")
    values = [
        pd.Timestamp(year=year, month=int(m), day=int(d)) + pd.Timedelta(hours=int(h) - 1)
        for m, d, h in zip(month, day, hour, strict=True)
    ]
    result = pd.DatetimeIndex(values)
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("Compact annual clock is not unique and increasing")
    return result


def prepare_reference_manifest(
    probe_manifest: pd.DataFrame, climates_path: Path
) -> pd.DataFrame:
    """Select qualified 25 C anchors and attach declared climate coordinates."""
    required = {"climate", "target_c", "hourly_path", "hourly_sha256"}
    missing = required - set(probe_manifest)
    if missing:
        raise ValueError(f"Probe manifest lacks fields: {sorted(missing)}")
    selected = probe_manifest[
        np.isclose(pd.to_numeric(probe_manifest.target_c), 25.0)
    ].copy()
    for field in ("completed", "relative_engineering_pass"):
        if field in selected:
            values = selected[field]
            if values.dtype != bool:
                values = values.astype(str).str.strip().str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            selected = selected[values.fillna(False)]
    content = json.loads(climates_path.read_text(encoding="utf-8"))
    climate = pd.DataFrame(content["climates"] if isinstance(content, dict) else content)
    climate = climate[["climate", "latitude", "longitude"]]
    result = selected.merge(climate, on="climate", how="left", validate="one_to_one")
    if result[["latitude", "longitude"]].isna().any().any():
        raise ValueError("A 25 C anchor lacks declared climate coordinates")
    if result.climate.duplicated().any() or len(result) != len(climate):
        raise ValueError("Expected one qualified 25 C anchor for every climate")
    return result.sort_values("climate", kind="stable").reset_index(drop=True)


def select_protocols(reference_manifest: pd.DataFrame, reference_runs: Path | None = None) -> pd.DataFrame:
    """Select eight events per climate from declared reference trajectories."""
    required = {"climate", "latitude", "hourly_path"}
    missing = required - set(reference_manifest)
    if missing:
        raise ValueError(f"Reference manifest lacks fields: {sorted(missing)}")
    if reference_manifest.climate.duplicated().any():
        raise ValueError("Reference manifest must contain one trajectory per climate")
    rows: list[pd.DataFrame] = []
    for source in reference_manifest.to_dict("records"):
        path = Path(source["hourly_path"])
        declared = str(source.get("hourly_sha256", "")).strip().lower()
        if not declared or declared == "nan" or sha256(path) != declared:
            raise ValueError(
                f"Reference trajectory hash mismatch for {source['climate']}"
            )
        compact = load_compact(path)
        electricity = _array(compact, "facility_electric_kwh").astype(float)
        if reference_runs is not None:
            meter = extract_facility_electric_kwh(reference_runs / source["run_id"] / "eplusout.sql")
            if not np.allclose(electricity, meter, rtol=0, atol=1e-9):
                raise ValueError(f"Reference electricity meters disagree for {source['climate']}")
            # Ranking near a seasonal quantile needs the native meter precision.
            electricity = meter
        cooling = _array(compact, "cooling_electric_kwh").astype(float)
        heating = _array(compact, "heating_electric_kwh").astype(float)
        heating += _array(compact, "gas_hhv_kwh").astype(float)
        occupied = _array(compact, "occupied").astype(bool)
        hour = _array(compact, "hour").astype(int)
        dst = _array(compact, "dst").astype(int)
        schedule_start_minute = ((hour - 1) * 60 + dst * 60) % 1440
        protocol = select_multi_event_protocol(
            compact_timestamps(compact), electricity, occupied,
            latitude=float(source["latitude"]), cooling_kw=cooling, heating_kw=heating,
            schedule_start_minute=schedule_start_minute,
        )
        protocol["selection_load_scope"] = "facility electricity"
        protocol.insert(0, "climate", str(source["climate"]))
        protocol["reference_run_id"] = str(source.get("run_id", ""))
        protocol["reference_hourly_path"] = str(Path(source["hourly_path"]).resolve())
        protocol["event_id"] = [
            hashlib.sha256(
                (
                    f"{source['climate']}|{row.start_timestamp.isoformat()}|"
                    f"pre{int(row.precondition_hours)}h|"
                    f"event{int(row.duration_hours)}h|"
                    f"recovery{int(row.recovery_hours)}h"
                ).encode()
            ).hexdigest()[:20]
            for row in protocol.itertuples(index=False)
        ]
        rows.append(protocol)
    result = pd.concat(rows, ignore_index=True)
    if not result.groupby("climate").size().eq(8).all():
        raise AssertionError("Each climate must contribute exactly eight events")
    return result


def directional_protected_zones(
    profiles: pd.DataFrame,
    zones: Sequence[str],
    sign: float,
    *,
    protected_fraction: float = 0.2,
) -> tuple[list[str], pd.DataFrame]:
    """Choose zones most exposed to the declared direction of temperature change.

    A positive event sign relaxes cooling and makes rooms warmer, so zones with
    the lowest mean upper demand bound are protected.  A negative sign relaxes
    heating and makes rooms cooler, so zones with the highest mean lower demand
    bound are protected.  This ranking is fixed before event simulation.
    """
    missing = REQUIRED_PROFILE_COLUMNS - set(profiles)
    if missing:
        raise ValueError(f"Profiles lack fields: {sorted(missing)}")
    if sign == 0 or not np.isfinite(sign):
        raise ValueError("Event sign must be finite and nonzero")
    if not 0 < protected_fraction <= 1:
        raise ValueError("Protected fraction must lie in (0, 1]")
    zones = [str(zone) for zone in zones]
    unknown = set(profiles.zone_id.astype(str)) - set(zones)
    if unknown:
        raise ValueError(f"Profiles contain unknown zones: {sorted(unknown)}")
    grouped = profiles.groupby("zone_id", sort=False).agg(
        mean_lower_c=("controller_lower_c", "mean"),
        mean_upper_c=("controller_upper_c", "mean"),
        profiles=("profile_id", "size"),
    ).reindex(zones)
    if grouped.isna().any().any():
        raise ValueError("Every controlled zone requires demand profiles")
    grouped = grouped.reset_index().rename(columns={"zone_id": "zone"})
    if sign > 0:
        grouped["directional_vulnerability"] = -grouped["mean_upper_c"]
    else:
        grouped["directional_vulnerability"] = grouped["mean_lower_c"]
    count = max(1, int(np.ceil(len(zones) * protected_fraction)))
    ranked = grouped.sort_values(
        ["directional_vulnerability", "zone"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    protected = ranked.head(count).zone.astype(str).tolist()
    ranked["protected"] = ranked.zone.isin(protected)
    return protected, ranked


def action_variants(
    protocol: pd.DataFrame,
    profiles: pd.DataFrame,
    zones: Sequence[str],
    *,
    protected_fraction: float = 0.2,
) -> tuple[list[dict], pd.DataFrame]:
    """Return a bounded action library and its zone-protection audit."""
    required = {
        "event_id", "month", "day", "sign", "start_minute", "duration_hours",
        "precondition_hours",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(f"Event protocol lacks fields: {sorted(missing)}")
    zones = [str(zone) for zone in zones]
    specifications = (
        ("uniform_0.5", "uniform_shift", "building", 0.0, 0.5, False),
        ("uniform_1.0", "uniform_shift", "building", 0.0, 1.0, False),
        ("precondition_0.5_uniform_1.0", "precondition_uniform_shift",
         "building", 0.5, 1.0, False),
        ("protected_1.0", "service_protected_shift", "native_zone", 0.0, 1.0, True),
        ("precondition_0.5_protected_1.0", "precondition_service_protected_shift",
         "native_zone", 0.5, 1.0, True),
    )
    variants: list[dict] = []
    audit_rows: list[pd.DataFrame] = []
    for action_id, family, permission, pre, amplitude, protect in specifications:
        events: list[dict] = []
        for event in protocol.to_dict("records"):
            sign = float(event["sign"])
            protected: list[str] = []
            if protect:
                protected, ranking = directional_protected_zones(
                    profiles, zones, sign, protected_fraction=protected_fraction
                )
                ranking.insert(0, "event_id", str(event["event_id"]))
                ranking.insert(1, "action_id", action_id)
                audit_rows.append(ranking)
            event_magnitude = {
                zone: (0.0 if zone in protected else float(amplitude)) for zone in zones
            }
            precondition = {
                zone: (0.0 if zone in protected else float(pre)) for zone in zones
            }
            events.append({
                "event_id": str(event["event_id"]),
                "month": int(event["month"]),
                "day": int(event["day"]),
                "sign": sign,
                "start_minute": int(event["start_minute"]),
                "duration_hours": int(event["duration_hours"]),
                "precondition_hours": int(event["precondition_hours"]),
                "protected_zones": protected,
                "precondition_magnitude_c": precondition,
                "event_magnitude_c": event_magnitude,
            })
        variants.append({
            "action_id": action_id,
            "action_family": family,
            "permission": permission,
            "events": events,
        })
    audit = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    return variants, audit


def _eligible_day_values(schedule: list[str]) -> dict[str, np.ndarray]:
    """Expand weekday and Saturday values from one annual compact schedule."""
    result = {
        "weekday": np.full(1440, np.nan),
        "saturday": np.full(1440, np.nan),
    }
    active: list[str] = []
    previous = 0
    index = 3
    while index < len(schedule):
        token = schedule[index]
        lowered = token.lower()
        if lowered.startswith("through:"):
            index += 1
            continue
        if lowered.startswith("for"):
            active = []
            previous = 0
            if "weekdays" in lowered:
                active.append("weekday")
            if "saturday" in lowered:
                active.append("saturday")
            index += 1
            continue
        if not lowered.startswith("until:") or index + 1 >= len(schedule):
            raise ValueError(f"Unsupported compact schedule token: {token}")
        hour, minute = map(int, token.split(":", 1)[1].strip().split(":"))
        end = hour * 60 + minute
        value = float(schedule[index + 1])
        for day in active:
            result[day][previous:end] = value
        previous = end
        index += 2
    return result


def _validate_preconditioning_deadband(
    parsed: Sequence[list[str]],
    duals: Mapping[str, list[str]],
    events: Sequence[dict],
) -> None:
    schedules = {
        item[1]: item for item in parsed if item[0] == "Schedule:Compact"
    }
    maximum_preconditioning = max(
        (
            max(map(float, event["precondition_magnitude_c"].values()), default=0.0)
            for event in events
        ),
        default=0.0,
    )
    for dual in duals.values():
        heating = _eligible_day_values(schedules[dual[2]])
        cooling = _eligible_day_values(schedules[dual[3]])
        for day in ("weekday", "saturday"):
            if np.isnan(heating[day]).any() or np.isnan(cooling[day]).any():
                raise ValueError(f"Incomplete {day} thermostat schedule")
            minimum = float(np.min(cooling[day] - heating[day]))
            if minimum - maximum_preconditioning < MINIMUM_EVENT_DEADBAND_C:
                raise ValueError(
                    "Preconditioning would reduce the thermostat deadband below "
                    f"{MINIMUM_EVENT_DEADBAND_C:.1f} °C"
                )


def event_specific_action_input(text: str, events: Sequence[dict]) -> str:
    """Apply each event only to its active heating or cooling boundary."""
    parsed = objects(text)
    duals = {item[1]: item for item in parsed if item[0] == "ThermostatSetpoint:DualSetpoint"}
    schedule_zones: dict[str, set[str]] = {}
    schedule_role: dict[str, str] = {}
    for thermostat in [item for item in parsed if item[0] == "ZoneControl:Thermostat"]:
        zone = thermostat[2]
        dual = duals[thermostat[5]]
        for field, role in ((2, "heating"), (3, "cooling")):
            old_role = schedule_role.setdefault(dual[field], role)
            if old_role != role:
                raise ValueError("One schedule cannot be both a heating and cooling boundary")
            schedule_zones.setdefault(dual[field], set()).add(zone)
    controlled = {zone for linked in schedule_zones.values() for zone in linked}
    for event in events:
        for field in ("precondition_magnitude_c", "event_magnitude_c"):
            if set(event[field]) != controlled:
                raise ValueError("Every event vector must contain every and only controlled zones")
    _validate_preconditioning_deadband(parsed, duals, events)

    ordered = sorted(events, key=lambda value: (int(value["month"]), int(value["day"])))
    dates = [(int(event["month"]), int(event["day"])) for event in ordered]
    if len(dates) != len(set(dates)):
        raise ValueError("At most one response event may occur on a calendar day")
    for item in parsed:
        if item[0] != "Schedule:Compact" or item[1] not in schedule_zones:
            continue
        linked = schedule_zones[item[1]]
        if item[3].lower() != "through: 12/31" or sum(
            token.lower().startswith("through:") for token in item
        ) != 1:
            raise ValueError("Expected one original annual thermostat schedule block")
        original = item[4:]
        expanded = item[:3]
        previous = date(2001, 1, 1)
        for event in ordered:
            current = date(2001, int(event["month"]), int(event["day"]))
            if previous < current:
                end = current - timedelta(days=1)
                expanded += [f"Through: {end.month}/{end.day}"] + original
            pre_values = {float(event["precondition_magnitude_c"][zone]) for zone in linked}
            action_values = {float(event["event_magnitude_c"][zone]) for zone in linked}
            if len(pre_values) != 1 or len(action_values) != 1:
                raise ValueError("A shared thermostat schedule received conflicting zone actions")
            sign = float(event["sign"])
            active = (
                sign > 0 and schedule_role[item[1]] == "cooling"
            ) or (
                sign < 0 and schedule_role[item[1]] == "heating"
            )
            event_start = int(event["start_minute"])
            event_end = event_start + int(event["duration_hours"]) * 60
            pre_start = event_start - int(event["precondition_hours"]) * 60
            expanded += [f"Through: {current.month}/{current.day}"] + split_action_block(
                original,
                pre_delta=(
                    -sign * next(iter(pre_values)) if active else 0.0
                ),
                event_delta=(
                    sign * next(iter(action_values)) if active else 0.0
                ),
                pre_start_minute=pre_start,
                event_start_minute=event_start,
                event_end_minute=event_end,
            )
            previous = current + timedelta(days=1)
        if previous <= date(2001, 12, 31):
            expanded += ["Through: 12/31"] + original
        item[:] = expanded
    return "\n".join(",".join(item) + ";" for item in parsed)


def build_event_plan(
    annual_manifest: pd.DataFrame,
    protocols: pd.DataFrame,
    profiles: pd.DataFrame,
    output_root: Path,
    *,
    protected_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Create event IDFs and logical event rows from global annual outputs."""
    annual_required = {
        "climate", "composition_id", "cohort_label", "control", "offset_c", "run_id",
        "input_idf_path", "hourly_path", "weather_path",
        "relative_engineering_pass",
    }
    missing = annual_required - set(annual_manifest)
    if missing:
        raise ValueError(f"Annual manifest lacks fields: {sorted(missing)}")
    profile_climate = _profile_climate_column(profiles)
    profile_required = REQUIRED_PROFILE_COLUMNS | {profile_climate, "composition_id"}
    missing = profile_required - set(profiles)
    if missing:
        raise ValueError(f"Profile table lacks fields: {sorted(missing)}")
    selected = annual_manifest[np.isclose(annual_manifest.offset_c.astype(float), 0.0)].copy()
    qualification = selected.relative_engineering_pass
    if qualification.dtype != bool:
        qualification = qualification.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
    if qualification.isna().any() or not qualification.all():
        failed = selected.loc[
            ~qualification.fillna(False), ["climate", "composition_id", "control"]
        ]
        raise ValueError(
            "Every offset-zero event baseline must pass engineering QA:\n"
            f"{failed.to_string(index=False)}"
        )
    keys = ["climate", "composition_id", "control"]
    if selected.duplicated(keys).any():
        raise ValueError("Annual manifest duplicates an offset-zero event baseline")
    found_controls = selected.groupby(["climate", "composition_id"])["control"].agg(set)
    if not found_controls.map(lambda value: value == CONTROLS).all():
        raise ValueError("Every climate/cohort needs all three offset-zero controls")
    output_root.mkdir(parents=True, exist_ok=True)
    idf_root = output_root / "idf"
    idf_root.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    event_rows: list[dict] = []
    action_records: list[dict] = []
    protection: list[pd.DataFrame] = []
    for baseline in selected.to_dict("records"):
        climate = str(baseline["climate"])
        composition = baseline["composition_id"]
        protocol = protocols[protocols.climate.astype(str).eq(climate)].copy()
        local_profiles = profiles[
            profiles[profile_climate].astype(str).eq(climate)
            & profiles.composition_id.astype(str).eq(str(composition))
        ].copy()
        if len(protocol) != 8 or local_profiles.empty:
            raise ValueError(f"Incomplete protocol or profiles for {climate}/{composition}")
        baseline_compact = load_compact(Path(baseline["hourly_path"]))
        baseline_hourly_path = Path(str(baseline["hourly_path"]))
        baseline_qa_path = Path(str(baseline.get("qa_path", "")))
        if not baseline_hourly_path.is_file() or not baseline_qa_path.is_file():
            raise FileNotFoundError(
                f"Annual baseline artifacts are incomplete for {climate}/{composition}"
            )
        baseline_hourly_sha256 = sha256(baseline_hourly_path)
        baseline_qa_sha256 = sha256(baseline_qa_path)
        zones = _array(baseline_compact, "zone_ids").astype(str).tolist()
        variants, audit = action_variants(
            protocol, local_profiles, zones, protected_fraction=protected_fraction
        )
        if not audit.empty:
            audit.insert(0, "climate", climate)
            audit.insert(1, "composition_id", composition)
            audit.insert(2, "control", str(baseline["control"]))
            protection.append(audit)
        source_path = Path(baseline["input_idf_path"])
        source = source_path.read_text(encoding="utf-8", errors="replace")
        for variant in variants:
            text = event_specific_action_input(source, variant["events"])
            if physical_hash(text) != physical_hash(source):
                raise ValueError("Response action changed non-thermostat building physics")
            digest = hashlib.sha256(
                (signature(text) + str(baseline["run_id"])).encode()
            ).hexdigest()[:24]
            run_id = f"event_{digest}"
            input_path = idf_root / f"{run_id}.idf"
            input_path.write_text(text, encoding="utf-8")
            run_rows.append({
                "run_id": run_id,
                "climate": climate,
                "composition_id": composition,
                "cohort_label": str(baseline["cohort_label"]),
                "control": str(baseline["control"]),
                "baseline_run_id": str(baseline["run_id"]),
                "baseline_hourly_path": str(baseline_hourly_path.resolve()),
                "baseline_hourly_sha256": baseline_hourly_sha256,
                "baseline_qa_path": str(baseline_qa_path.resolve()),
                "baseline_qa_sha256": baseline_qa_sha256,
                "baseline_warning_profile": str(
                    baseline.get("treatment_warning_profile", "{}")
                ),
                "action_id": variant["action_id"],
                "action_family": variant["action_family"],
                "permission": variant["permission"],
                "input_idf_path": str(input_path.resolve()),
                "weather_path": str(Path(baseline["weather_path"]).resolve()),
                "zones_json": json.dumps(zones),
                "normalized_idf_sha256": signature(text),
                "physical_fingerprint_unchanged": True,
            })
            action_records.append({
                "run_id": run_id,
                "climate": climate,
                "composition_id": composition,
                "control": str(baseline["control"]),
                **variant,
            })
            by_id = {str(event["event_id"]): event for event in variant["events"]}
            for event in protocol.to_dict("records"):
                action_event = by_id[str(event["event_id"])]
                event_rows.append({
                    "run_id": run_id,
                    "climate": climate,
                    "composition_id": composition,
                    "cohort_label": str(baseline["cohort_label"]),
                    "control": str(baseline["control"]),
                    "baseline_run_id": str(baseline["run_id"]),
                    "action_id": variant["action_id"],
                    "action_family": variant["action_family"],
                    "permission": variant["permission"],
                    "event_id": str(event["event_id"]),
                    "season_block": event["season_block"],
                    "local_season": event["local_season"],
                    "load_level": event["load_level"],
                    "start_timestamp": pd.Timestamp(event["start_timestamp"]),
                    "duration_hours": int(event["duration_hours"]),
                    "precondition_hours": int(event["precondition_hours"]),
                    "recovery_hours": int(event["recovery_hours"]),
                    "sign": float(event["sign"]),
                    "protected_zones_json": json.dumps(action_event["protected_zones"]),
                })
    runs = pd.DataFrame(run_rows).drop_duplicates("run_id").reset_index(drop=True)
    events = pd.DataFrame(event_rows).reset_index(drop=True)
    if len(events) != 8 * len(runs):
        raise AssertionError("Every planned run must contain exactly eight events")
    runs.to_csv(output_root / "EVENT_RUN_MANIFEST.csv", index=False)
    events.to_csv(output_root / "EVENT_EVALUATION_MANIFEST.csv", index=False)
    (output_root / "ACTION_SPECS.json").write_text(
        json.dumps(action_records, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if protection:
        pd.concat(protection, ignore_index=True).to_csv(
            output_root / "PROTECTED_ZONE_AUDIT.csv", index=False
        )
    return runs, events, action_records


def _warning_counter(qa: Mapping[str, object]) -> Counter[str]:
    result: Counter[str] = Counter()
    for message in qa.get("direct_numerical_warnings", []) or []:
        text = str(message)
        text = text.split(", at ", 1)[0]
        if "(" in text:
            text = text.split("(", 1)[0].rstrip()
        result[text] += 1
    return result


def baseline_relative_qa(
    baseline_qa: Mapping[str, object], action_qa: Mapping[str, object],
    *,
    baseline_profile: Mapping[str, int] | None = None,
    action_profile: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Apply the global engineering gate relative to an annual baseline."""
    if (baseline_profile is None) != (action_profile is None):
        raise ValueError("Baseline and action warning profiles must be supplied together")
    if baseline_profile is not None and action_profile is not None:
        checked = relative_engineering_qa(
            dict(action_qa), dict(action_profile), dict(baseline_profile)
        )
        return {
            "baseline_relative_engineering_pass": checked[
                "engineering_pass_relative_to_prototype_baseline"
            ],
            "new_numerical_warning_categories": checked["new_warning_categories"],
            "increased_numerical_warning_counts": checked["warning_count_increases"],
            "critical_warning_count_increases": checked[
                "critical_warning_count_increases"
            ],
            "inherited_environmental_warning_count_drift": checked[
                "inherited_environmental_warning_count_drift"
            ],
            "baseline_warning_profile": dict(baseline_profile),
            "action_warning_profile": dict(action_profile),
        }
    baseline = _warning_counter(baseline_qa)
    action = _warning_counter(action_qa)
    new_categories = sorted(set(action) - set(baseline))
    increased = {
        category: {"baseline": baseline[category], "action": action[category]}
        for category in action
        if action[category] > baseline[category]
    }
    passed = bool(
        action_qa.get("completed")
        and int(action_qa.get("reported_severe_count") or 0) == 0
        and int(action_qa.get("fatal_occurrences") or 0) == 0
        and not new_categories
        and not increased
        and not action_qa.get("unexplained_recurring_groups")
    )
    return {
        "baseline_relative_engineering_pass": passed,
        "new_numerical_warning_categories": new_categories,
        "increased_numerical_warning_counts": increased,
        "baseline_numerical_warning_counts": dict(baseline),
        "action_numerical_warning_counts": dict(action),
    }


def read_event_solver_stability_evidence(
    path: Path | None,
) -> tuple[dict[str, dict], str | None]:
    """Load exact event-action 40-vs-100 solver-stability evidence."""
    if path is None:
        return {}, None
    frame = pd.read_csv(path)
    required = {
        "run_id", "climate", "composition_id", "control", "action_id",
        "first_maximum_hvac_iterations", "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "facility_electricity_compared",
        "facility_electricity_relative_difference",
        "trajectory_stable_within_declared_tolerance",
        "declared_temperature_tolerance_c", "declared_energy_relative_tolerance",
        "higher_iteration_completed", "higher_iteration_severe_count",
        "higher_iteration_fatal_occurrences",
        "higher_iteration_relative_engineering_pass",
        "warning_profile_nonworsening",
        "first_action_warning_profile", "second_action_warning_profile",
        "first_input_sha256", "first_hourly_sha256",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Event solver-stability evidence lacks {sorted(missing)}")
    if frame.run_id.duplicated().any():
        raise ValueError("Event solver-stability evidence duplicates a run identifier")
    boolean_columns = [
        "facility_electricity_compared",
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
        parsed = frame[column].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if parsed.isna().any():
            raise ValueError(f"Event solver-stability evidence has invalid {column}")
        frame[column] = parsed.astype(bool)
    numeric_columns = [
        "composition_id", "first_maximum_hvac_iterations",
        "second_maximum_hvac_iterations",
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "facility_electricity_relative_difference",
        "declared_temperature_tolerance_c", "declared_energy_relative_tolerance",
        "higher_iteration_severe_count", "higher_iteration_fatal_occurrences",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Event solver-stability evidence contains non-finite values")
    frame[numeric_columns] = numeric
    difference_columns = [
        "operative_temperature_max_abs_difference_c",
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
        "facility_electricity_relative_difference",
    ]
    if (frame[difference_columns] < 0).any().any():
        raise ValueError("Event solver-stability differences cannot be negative")
    if (
        (frame.declared_temperature_tolerance_c <= 0).any()
        or (frame.declared_energy_relative_tolerance <= 0).any()
        or (
            frame.second_maximum_hvac_iterations
            <= frame.first_maximum_hvac_iterations
        ).any()
    ):
        raise ValueError("Event solver-stability limits are invalid")
    energy = frame[[
        "hvac_electricity_relative_difference", "gas_hhv_relative_difference",
        "total_purchased_energy_relative_difference",
    ]]
    facility_failed = (
        frame.facility_electricity_compared
        & (
            frame.facility_electricity_relative_difference
            > frame.declared_energy_relative_tolerance
        )
    )
    if (
        (
            frame.operative_temperature_max_abs_difference_c
            > frame.declared_temperature_tolerance_c
        ).any()
        or energy.gt(
            frame.declared_energy_relative_tolerance, axis="index"
        ).any().any()
        or facility_failed.any()
        or not frame.trajectory_stable_within_declared_tolerance.all()
    ):
        raise ValueError("Event solver-stability evidence fails its declared limits")
    category_stable: list[bool] = []
    new_categories: list[str] = []
    count_drift: list[str] = []
    stability_pass: list[bool] = []
    for item in frame.to_dict("records"):
        try:
            first = json.loads(item["first_action_warning_profile"])
            second = json.loads(item["second_action_warning_profile"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Event solver-stability warning profile is not valid JSON"
            ) from exc
        if not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError(
                "Event solver-stability warning profiles must be objects"
            )
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
            raise ValueError("Event warning-category stability flag is inconsistent")
        if (
            "solver_stability_pass" in item
            and bool(item["solver_stability_pass"]) != computed_pass
        ):
            raise ValueError("Event solver-stability pass flag is inconsistent")
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
                        f"Event solver-stability {column} is not valid JSON"
                    ) from exc
                if declared != computed:
                    raise ValueError(
                        f"Event solver-stability {column} is inconsistent"
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
                raise ValueError(f"Event solver-stability {column} is invalid")
    frame["warning_categories_stable"] = category_stable
    frame["higher_iteration_new_warning_categories"] = new_categories
    frame["warning_count_drift"] = count_drift
    frame["solver_stability_pass"] = stability_pass
    if not frame.solver_stability_pass.all():
        raise ValueError(
            "Event higher-iteration evidence fails the category-and-trajectory rule"
        )
    return (
        {str(row["run_id"]): row for row in frame.to_dict("records")},
        sha256(path),
    )


def apply_event_solver_stability_exception(
    relative: Mapping[str, object],
    run_qa: Mapping[str, object],
    evidence: Mapping[str, object] | None,
    *,
    current_maximum_hvac_iterations: int,
    active_input_sha256: str,
    active_hourly_sha256: str,
) -> tuple[bool, dict[str, object]]:
    """Qualify one exact event action using matched higher-iteration evidence."""
    raw_pass = bool(relative["baseline_relative_engineering_pass"])
    audit: dict[str, object] = {
        "raw_baseline_relative_engineering_pass": raw_pass,
        "event_solver_stability_exception_applied": False,
        "event_solver_stability_numerical_stability_pass": None,
    }
    if raw_pass or evidence is None:
        return raw_pass, audit
    new_categories = set(relative["new_numerical_warning_categories"])
    critical = set(relative["critical_warning_count_increases"])
    only_hvac = bool(
        new_categories.issubset({"hvac_iterations"})
        and critical == {"hvac_iterations"}
    )
    clean_run = bool(
        run_qa.get("completed")
        and int(run_qa.get("reported_severe_count") or 0) == 0
        and int(run_qa.get("fatal_occurrences") or 0) == 0
        and not run_qa.get("unexplained_recurring_groups")
    )
    if not only_hvac or not clean_run:
        return False, audit
    try:
        evidence_profile = json.loads(str(evidence["first_action_warning_profile"]))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Event solver-stability warning profile is not valid JSON"
        ) from exc
    if evidence_profile != relative["action_warning_profile"]:
        raise ValueError(
            "Event solver-stability evidence does not match the active warning profile"
        )
    if int(evidence["first_maximum_hvac_iterations"]) != int(
        current_maximum_hvac_iterations
    ):
        raise ValueError(
            "Event solver-stability evidence does not match the active iteration limit"
        )
    if str(evidence["first_input_sha256"]) != str(active_input_sha256):
        raise ValueError(
            "Event solver-stability evidence does not match the active input"
        )
    if str(evidence["first_hourly_sha256"]) != str(active_hourly_sha256):
        raise ValueError(
            "Event solver-stability evidence does not match the active trajectory"
        )
    if not bool(evidence.get("solver_stability_pass", False)):
        raise ValueError("Event solver-stability evidence is not qualified")
    audit.update({
        "event_solver_stability_exception_applied": True,
        "event_solver_stability_higher_iteration_limit": int(
            evidence["second_maximum_hvac_iterations"]
        ),
        "event_solver_stability_temperature_difference_c": float(
            evidence["operative_temperature_max_abs_difference_c"]
        ),
        "event_solver_stability_energy_relative_difference": float(
            evidence["total_purchased_energy_relative_difference"]
        ),
        "event_solver_stability_facility_electricity_relative_difference": float(
            evidence["facility_electricity_relative_difference"]
        ),
        "event_solver_stability_higher_iteration_baseline_warning_pass": bool(
            evidence["higher_iteration_relative_engineering_pass"]
        ),
        "event_solver_stability_warning_profile_nonworsening": bool(
            evidence["warning_profile_nonworsening"]
        ),
        "event_solver_stability_warning_categories_stable": bool(
            evidence["warning_categories_stable"]
        ),
        "event_solver_stability_numerical_stability_pass": bool(
            evidence["solver_stability_pass"]
        ),
        "event_solver_stability_warning_count_drift": str(
            evidence["warning_count_drift"]
        ),
    })
    return True, audit


def require_complete_event_execution(result: pd.DataFrame, expected_runs: int) -> None:
    """Reject an incomplete or engineering-failed response batch after QA output."""
    required = {"run_id", "completed", "final_engineering_qualified"}
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"Event execution audit lacks {sorted(missing)}")

    def flags(column: str) -> pd.Series:
        values = result[column]
        if values.dtype == bool:
            return values
        parsed = values.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if parsed.isna().any():
            raise RuntimeError(f"Event execution has invalid {column} flags")
        return parsed.astype(bool)

    completed = flags("completed")
    qualified = flags("final_engineering_qualified")
    invalid = result.loc[~completed | ~qualified, "run_id"].astype(str).tolist()
    if len(result) != expected_runs or result.run_id.duplicated().any() or invalid:
        raise RuntimeError(
            "Multi-event execution failed its completeness/engineering gate: "
            f"expected={expected_runs}, observed={len(result)}, failed={invalid}"
        )


def attach_planned_event_metadata(
    executed: pd.DataFrame, planned: pd.DataFrame
) -> pd.DataFrame:
    """Restore the planned identifiers required by event scoring."""
    fields = (
        "baseline_hourly_path", "baseline_hourly_sha256",
        "baseline_qa_path", "baseline_qa_sha256",
        "action_family", "permission", "cohort_label",
    )
    required = {"run_id", *fields}
    missing = required - set(planned)
    if missing:
        raise ValueError(f"Planned event manifest lacks {sorted(missing)}")
    if planned.run_id.duplicated().any():
        raise ValueError("Planned event manifest duplicates run_id")
    result = executed.copy()
    result["run_id"] = result.run_id.astype(str)
    source = planned.assign(run_id=planned.run_id.astype(str)).set_index("run_id")
    unknown = sorted(set(result.run_id) - set(source.index))
    if unknown:
        raise ValueError(f"Executed event runs are absent from the plan: {unknown}")
    for field in fields:
        result[field] = result.run_id.map(source[field])
    if result[list(fields)].isna().any().any():
        raise ValueError("Completed event manifest lost required planned metadata")
    return result


def baseline_artifact_hashes(row: Mapping[str, object]) -> dict[str, str]:
    """Verify that a response run still points to its planned annual baseline."""
    result: dict[str, str] = {}
    for kind in ("hourly", "qa"):
        path_field = f"baseline_{kind}_path"
        hash_field = f"baseline_{kind}_sha256"
        path = Path(str(row.get(path_field, "")))
        declared = str(row.get(hash_field, "")).strip()
        if not path.is_file():
            raise FileNotFoundError(f"Baseline {kind} artifact is missing: {path}")
        if not declared or declared.lower() == "nan":
            raise ValueError(f"Event plan lacks {hash_field}")
        actual = sha256(path)
        if actual != declared:
            raise RuntimeError(
                f"Baseline {kind} artifact changed after event planning: {path}"
            )
        result[hash_field] = actual
    return result


def execute_event_plan(
    runs: pd.DataFrame,
    output_root: Path,
    runtime: Path,
    *,
    workers: int = 4,
    cleanup_raw: bool = True,
    solver_stability_manifest: Path | None = None,
) -> pd.DataFrame:
    """Execute planned annual event trajectories with cache-aware QA."""
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = runtime.resolve()
    runtime_hash = sha256(runtime)
    stability, stability_sha256 = read_event_solver_stability_evidence(
        solver_stability_manifest
    )
    if runs.run_id.duplicated().any():
        raise ValueError("Event execution plan duplicates run_id")
    plan_by_id = {
        str(row["run_id"]): row for row in runs.to_dict("records")
    }

    def execute(row: dict) -> dict:
        run_id = str(row["run_id"])
        folder = output_root / "runs" / run_id
        compact = output_root / "hourly" / f"{run_id}.npz"
        summary_path = output_root / "summaries" / f"{run_id}.json"
        input_path = Path(row["input_idf_path"])
        weather_path = Path(row["weather_path"])
        baseline_hashes = baseline_artifact_hashes(row)
        if summary_path.is_file() and compact.is_file():
            cached = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                cached.get("hourly_sha256") == sha256(compact)
                and "baseline_relative_engineering_pass" in cached
                and cached.get("runtime_sha256") == runtime_hash
                and cached.get("input_idf_sha256") == sha256(input_path)
                and cached.get("weather_sha256") == sha256(weather_path)
                and cached.get("baseline_hourly_sha256")
                == baseline_hashes["baseline_hourly_sha256"]
                and cached.get("baseline_qa_sha256")
                == baseline_hashes["baseline_qa_sha256"]
            ):
                return cached
        qa = run_energyplus(
            input_path, weather_path, runtime,
            folder, compact, json.loads(row["zones_json"]), cleanup_raw=cleanup_raw,
        )
        baseline_path = Path(str(row.get("baseline_qa_path", "")))
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"Baseline QA is required for inherited-warning review: {baseline_path}"
            )
        baseline_qa = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_profile = json.loads(str(row["baseline_warning_profile"]))
        action_profile = warning_profile(folder / "eplusout.err")
        qa.update(baseline_relative_qa(
            baseline_qa,
            qa,
            baseline_profile=baseline_profile,
            action_profile=action_profile,
        ))
        qa.update({
            "run_id": run_id,
            "climate": row["climate"],
            "composition_id": row["composition_id"],
            "control": row["control"],
            "action_id": row["action_id"],
            "hourly_path": str(compact.resolve()),
            **baseline_hashes,
        })
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
        return qa

    completed: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, row) for row in runs.to_dict("records")]
        for future in as_completed(futures):
            completed.append(future.result())
    used_stability_rows: set[str] = set()
    eligible_failure_rows: set[str] = set()
    for qa in completed:
        run_id = str(qa["run_id"])
        plan = plan_by_id[run_id]
        raw_pass = bool(
            qa.get(
                "raw_baseline_relative_engineering_pass",
                qa["baseline_relative_engineering_pass"],
            )
        )
        relative = {
            "baseline_relative_engineering_pass": raw_pass,
            "new_numerical_warning_categories": list(
                qa["new_numerical_warning_categories"]
            ),
            "critical_warning_count_increases": dict(
                qa["critical_warning_count_increases"]
            ),
            "baseline_warning_profile": dict(qa["baseline_warning_profile"]),
            "action_warning_profile": dict(qa["action_warning_profile"]),
        }
        if (
            not raw_pass
            and set(relative["new_numerical_warning_categories"]).issubset(
                {"hvac_iterations"}
            )
            and set(relative["critical_warning_count_increases"])
            == {"hvac_iterations"}
            and qa.get("completed")
            and int(qa.get("reported_severe_count") or 0) == 0
            and int(qa.get("fatal_occurrences") or 0) == 0
            and not qa.get("unexplained_recurring_groups")
        ):
            eligible_failure_rows.add(run_id)
        configuration = solver_configuration(
            Path(plan["input_idf_path"]).read_text(
                encoding="utf-8", errors="replace"
            )
        )
        maximum_iterations = configuration["maximum_hvac_iterations"]
        if maximum_iterations is None:
            raise ValueError(
                f"Event input lacks an explicit HVAC iteration limit: {run_id}"
            )
        final_pass, stability_audit = apply_event_solver_stability_exception(
            relative,
            qa,
            stability.get(run_id),
            current_maximum_hvac_iterations=int(maximum_iterations),
            active_input_sha256=str(qa["input_idf_sha256"]),
            active_hourly_sha256=str(qa["hourly_sha256"]),
        )
        if stability_audit["event_solver_stability_exception_applied"]:
            used_stability_rows.add(run_id)
        qa.pop("event_solver_stability_higher_iteration_relative_pass", None)
        qa.update(stability_audit)
        qa["baseline_relative_engineering_pass"] = final_pass
        qa["final_engineering_qualified"] = final_pass
        qa["event_solver_stability_evidence_sha256"] = (
            stability_sha256
            if stability_audit["event_solver_stability_exception_applied"]
            else None
        )
        summary_path = output_root / "summaries" / f"{run_id}.json"
        summary_path.write_text(
            json.dumps(qa, indent=2) + "\n", encoding="utf-8"
        )
    if solver_stability_manifest is not None:
        missing = eligible_failure_rows - set(stability)
        unused = set(stability) - used_stability_rows
        if missing:
            raise ValueError(
                "Event solver-stability evidence is missing for active HVAC-only "
                f"failures: {sorted(missing)}"
            )
        if unused:
            raise ValueError(
                "Event solver-stability evidence is stale or does not correspond "
                f"to an active HVAC-only failure: {sorted(unused)}"
            )
    result = pd.DataFrame(completed)
    result.to_csv(output_root / "EVENT_EXECUTION_QA.csv", index=False)
    require_complete_event_execution(result, len(runs))
    return result


def fixed_tail_profile_ids(
    reference: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    *,
    tail_fraction: float = 0.1,
) -> set[str]:
    """Fix the least-served profile group from the annual 25 °C reference."""
    if not 0 < tail_fraction <= 1:
        raise ValueError("Tail fraction must lie in (0, 1]")
    missing = REQUIRED_PROFILE_COLUMNS - set(profiles)
    if missing:
        raise ValueError(f"Profiles lack fields: {sorted(missing)}")
    zones = _array(reference, "zone_ids").astype(str)
    temperatures = _array(reference, "temperatures").astype(float)
    people = _array(reference, "people").astype(float)
    occupied = _array(reference, "occupied").astype(bool)
    scores: list[tuple[str, float]] = []
    for zone_index, zone in enumerate(zones):
        local = profiles[profiles.zone_id.astype(str).eq(zone)]
        if local.empty:
            raise ValueError(f"No profiles are assigned to {zone}")
        weight = people[occupied, zone_index]
        denominator = float(weight.sum())
        if denominator <= 0:
            raise ValueError(f"Reference trajectory has no occupied weight for {zone}")
        for profile in local.to_dict("records"):
            mismatch = np.abs(temperatures[occupied, zone_index] - float(profile["centre_c"]))
            scores.append((str(profile["profile_id"]), float(np.dot(mismatch, weight) / denominator)))
    scores.sort(key=lambda value: (-value[1], value[0]))
    count = max(1, int(np.ceil(len(scores) * tail_fraction)))
    return {profile_id for profile_id, _ in scores[:count]}


def service_metrics(
    compact: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    selected: np.ndarray,
    fixed_tail: set[str],
) -> dict[str, float]:
    """Calculate population, fixed-tail and Personal range service metrics."""
    selected = np.asarray(selected, dtype=bool)
    temperatures = _array(compact, "temperatures").astype(float)
    people = _array(compact, "people").astype(float)
    zones = _array(compact, "zone_ids").astype(str)
    if selected.shape != (len(temperatures),):
        raise ValueError("Service window must align with the compact trajectory")
    selected &= _array(compact, "occupied").astype(bool)
    totals = Counter()
    tail_totals = Counter()
    for zone_index, zone in enumerate(zones):
        local = profiles[profiles.zone_id.astype(str).eq(zone)].copy()
        if local.empty:
            raise ValueError(f"No profiles are assigned to {zone}")
        time_weight = people[selected, zone_index].astype(float)
        if not len(time_weight) or time_weight.sum() <= 0:
            continue
        person_weight = time_weight[:, None] / len(local)
        temperature = temperatures[selected, zone_index][:, None]
        centre = local.centre_c.to_numpy(float)[None, :]
        lower = local.controller_lower_c.to_numpy(float)[None, :]
        upper = local.controller_upper_c.to_numpy(float)[None, :]
        mismatch = np.abs(temperature - centre)
        excess = np.maximum(lower - temperature, 0.0) + np.maximum(temperature - upper, 0.0)
        within = (temperature >= lower) & (temperature <= upper)
        expanded_weight = np.broadcast_to(person_weight, mismatch.shape)
        totals["weight"] += float(expanded_weight.sum())
        totals["mismatch"] += float((mismatch * expanded_weight).sum())
        totals["excess"] += float((excess * expanded_weight).sum())
        totals["within"] += float((within * expanded_weight).sum())
        tail_mask = local.profile_id.astype(str).isin(fixed_tail).to_numpy()
        if tail_mask.any():
            tail_weight = expanded_weight[:, tail_mask]
            tail_totals["weight"] += float(tail_weight.sum())
            tail_totals["mismatch"] += float((mismatch[:, tail_mask] * tail_weight).sum())
    if totals["weight"] <= 0 or tail_totals["weight"] <= 0:
        return {
            "population_mean_mismatch_c": float("nan"),
            "fixed_tail_mean_mismatch_c": float("nan"),
            "interval_excess_c": float("nan"),
            "interval_coverage": float("nan"),
            "weighted_profile_hours": float(totals["weight"]),
        }
    return {
        "population_mean_mismatch_c": totals["mismatch"] / totals["weight"],
        "fixed_tail_mean_mismatch_c": tail_totals["mismatch"] / tail_totals["weight"],
        "interval_excess_c": totals["excess"] / totals["weight"],
        "interval_coverage": totals["within"] / totals["weight"],
        "weighted_profile_hours": float(totals["weight"]),
    }


def _window_masks(
    compact: Mapping[str, np.ndarray], start: pd.Timestamp, duration_hours: int,
    recovery_hours: int, precondition_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clock = compact_timestamps(compact)
    precondition_start = start - pd.Timedelta(hours=precondition_hours)
    event_end = start + pd.Timedelta(hours=duration_hours)
    recovery_end = event_end + pd.Timedelta(hours=recovery_hours)
    precondition = (clock >= precondition_start) & (clock < start)
    event = (clock >= start) & (clock < event_end)
    recovery = (clock >= event_end) & (clock < recovery_end)
    return np.asarray(precondition), np.asarray(event), np.asarray(recovery)


def score_event(
    baseline: Mapping[str, np.ndarray],
    action: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    profiles: pd.DataFrame,
    event: Mapping[str, object],
    *,
    limit_factors: Iterable[float] = DEFAULT_LIMIT_FACTORS,
) -> list[dict]:
    """Score one event with a strict baseline/recovery and event-only allowance."""
    start = pd.Timestamp(event["start_timestamp"])
    duration = int(event["duration_hours"])
    recovery_hours = int(event["recovery_hours"])
    precondition_hours = int(event.get("precondition_hours", 0))
    clocks = [compact_timestamps(value) for value in (baseline, action, reference)]
    if not (clocks[0].equals(clocks[1]) and clocks[0].equals(clocks[2])):
        raise ValueError("Baseline, action and reference clocks differ")
    baseline_pre, baseline_event, baseline_recovery = _window_masks(
        baseline, start, duration, recovery_hours, precondition_hours
    )
    action_pre, action_event, action_recovery = _window_masks(
        action, start, duration, recovery_hours, precondition_hours
    )
    reference_pre, reference_event, reference_recovery = _window_masks(
        reference, start, duration, recovery_hours, precondition_hours
    )
    fixed_tail = fixed_tail_profile_ids(reference, profiles)
    service: dict[str, float] = {}
    for trajectory_name, compact, event_mask, recovery_mask in (
        ("baseline", baseline, baseline_event, baseline_recovery),
        ("action", action, action_event, action_recovery),
        ("reference", reference, reference_event, reference_recovery),
    ):
        pre_mask = {
            "baseline": baseline_pre,
            "action": action_pre,
            "reference": reference_pre,
        }[trajectory_name]
        for window_name, mask in (
            ("precondition", pre_mask),
            ("event", event_mask),
            ("recovery", recovery_mask),
        ):
            values = service_metrics(compact, profiles, mask, fixed_tail)
            service.update({
                f"{trajectory_name}_{window_name}_{name}": value
                for name, value in values.items()
            })
    base_hvac_electric = _array(
        baseline, "hvac_electric_kwh", "electricity_hvac_meter_kwh"
    )
    action_hvac_electric = _array(
        action, "hvac_electric_kwh", "electricity_hvac_meter_kwh"
    )
    base_facility_electric = _array(baseline, "facility_electric_kwh")
    action_facility_electric = _array(action, "facility_electric_kwh")
    base_gas = _array(baseline, "gas_hhv_kwh")
    action_gas = _array(action, "gas_hhv_kwh")
    hvac_preconditioning_change = float(
        (action_hvac_electric[action_pre] - base_hvac_electric[baseline_pre]).sum()
    )
    hvac_event_reduction = float(
        (base_hvac_electric[baseline_event] - action_hvac_electric[action_event]).sum()
    )
    hvac_recovery_change = float(
        (action_hvac_electric[action_recovery] - base_hvac_electric[baseline_recovery]).sum()
    )
    facility_preconditioning_change = float(
        (action_facility_electric[action_pre] - base_facility_electric[baseline_pre]).sum()
    )
    facility_event_reduction = float(
        (
            base_facility_electric[baseline_event]
            - action_facility_electric[action_event]
        ).sum()
    )
    facility_recovery_change = float(
        (
            action_facility_electric[action_recovery]
            - base_facility_electric[baseline_recovery]
        ).sum()
    )
    end_use_changes: dict[str, float] = {}
    for label, field in (
        ("cooling", "cooling_electric_kwh"),
        ("heating", "heating_electric_kwh"),
        ("fans", "fans_electric_kwh"),
    ):
        baseline_values = _array(baseline, field)
        action_values = _array(action, field)
        preconditioning_change = float(
            (action_values[action_pre] - baseline_values[baseline_pre]).sum()
        )
        event_reduction = float(
            (baseline_values[baseline_event] - action_values[action_event]).sum()
        )
        recovery_change = float(
            (action_values[action_recovery] - baseline_values[baseline_recovery]).sum()
        )
        end_use_changes.update({
            f"event_{label}_electricity_reduction_kwh": event_reduction,
            f"preconditioning_{label}_electricity_change_kwh": (
                preconditioning_change
            ),
            f"recovery_{label}_electricity_change_kwh": recovery_change,
            f"precondition_event_recovery_{label}_electricity_net_reduction_kwh": (
                event_reduction - preconditioning_change - recovery_change
            ),
        })
    gas_event_reduction = float((base_gas[baseline_event] - action_gas[action_event]).sum())
    gas_preconditioning_change = float(
        (action_gas[action_pre] - base_gas[baseline_pre]).sum()
    )
    gas_recovery_change = float((action_gas[action_recovery] - base_gas[baseline_recovery]).sum())
    common = {
        **{key: event[key] for key in event if key not in {"start_timestamp"}},
        "start_timestamp": start,
        "event_hvac_electricity_reduction_kwh": hvac_event_reduction,
        "event_hvac_mean_power_reduction_kw": hvac_event_reduction / duration,
        "preconditioning_hvac_electricity_change_kwh": hvac_preconditioning_change,
        "recovery_hvac_electricity_change_kwh": hvac_recovery_change,
        "precondition_event_recovery_hvac_net_reduction_kwh": (
            hvac_event_reduction - hvac_preconditioning_change - hvac_recovery_change
        ),
        "event_facility_electricity_reduction_kwh": facility_event_reduction,
        "event_facility_mean_power_reduction_kw": facility_event_reduction / duration,
        "preconditioning_facility_electricity_change_kwh": (
            facility_preconditioning_change
        ),
        "recovery_facility_electricity_change_kwh": facility_recovery_change,
        "precondition_event_recovery_facility_net_reduction_kwh": (
            facility_event_reduction
            - facility_preconditioning_change
            - facility_recovery_change
        ),
        **end_use_changes,
        # Generic response fields use the grid-facing facility meter.
        "event_electricity_reduction_kwh": facility_event_reduction,
        "event_mean_power_reduction_kw": facility_event_reduction / duration,
        "preconditioning_electricity_change_kwh": facility_preconditioning_change,
        "recovery_electricity_change_kwh": facility_recovery_change,
        "event_plus_recovery_net_reduction_kwh": (
            facility_event_reduction - facility_recovery_change
        ),
        "precondition_event_recovery_net_reduction_kwh": (
            facility_event_reduction
            - facility_preconditioning_change
            - facility_recovery_change
        ),
        "event_gas_reduction_kwh_hhv": gas_event_reduction,
        "preconditioning_gas_change_kwh_hhv": gas_preconditioning_change,
        "recovery_gas_change_kwh_hhv": gas_recovery_change,
        "precondition_event_recovery_net_gas_reduction_kwh_hhv": (
            gas_event_reduction - gas_preconditioning_change - gas_recovery_change
        ),
        "event_conditioning_purchased_energy_reduction_kwh": (
            hvac_event_reduction + gas_event_reduction
        ),
        "precondition_event_recovery_conditioning_purchased_energy_net_reduction_kwh": (
            hvac_event_reduction + gas_event_reduction
            - hvac_preconditioning_change - gas_preconditioning_change
            - hvac_recovery_change - gas_recovery_change
        ),
        "event_facility_electric_plus_space_heating_gas_reduction_kwh": (
            facility_event_reduction + gas_event_reduction
        ),
        "precondition_event_recovery_facility_electric_plus_space_heating_gas_net_reduction_kwh": (
            facility_event_reduction + gas_event_reduction
            - facility_preconditioning_change - gas_preconditioning_change
            - facility_recovery_change - gas_recovery_change
        ),
        **service,
    }
    rows: list[dict] = []
    tolerance = 1e-9
    full_window_reduction = (
        facility_event_reduction
        - facility_preconditioning_change
        - facility_recovery_change
    )
    metrics = ("population_mean_mismatch_c", "fixed_tail_mean_mismatch_c")
    for factor in limit_factors:
        factor = float(factor)
        if factor <= 0 or not np.isfinite(factor):
            raise ValueError("Service-limit factors must be finite and positive")
        precondition_service_evaluated = bool(
            service["reference_precondition_weighted_profile_hours"] > 0
        )
        baseline_windows = ["event", "recovery"]
        if precondition_service_evaluated:
            baseline_windows.insert(0, "precondition")
        baseline_checks = []
        for window in baseline_windows:
            for metric in metrics:
                observed = service[f"baseline_{window}_{metric}"]
                reference_value = service[f"reference_{window}_{metric}"]
                baseline_checks.append(bool(
                    np.isfinite(observed) and np.isfinite(reference_value)
                    and observed <= reference_value + tolerance
                ))
        action_non_event_checks = []
        action_non_event_windows = ["recovery"]
        if precondition_service_evaluated:
            action_non_event_windows.insert(0, "precondition")
        for window in action_non_event_windows:
            for metric in metrics:
                observed = service[f"action_{window}_{metric}"]
                reference_value = service[f"reference_{window}_{metric}"]
                action_non_event_checks.append(bool(
                    np.isfinite(observed) and np.isfinite(reference_value)
                    and observed <= reference_value + tolerance
                ))
        action_event_checks = []
        for metric in metrics:
            observed = service[f"action_event_{metric}"]
            reference_value = service[f"reference_event_{metric}"]
            action_event_checks.append(bool(
                np.isfinite(observed) and np.isfinite(reference_value)
                and observed <= factor * reference_value + tolerance
            ))
        checks = baseline_checks + action_non_event_checks + action_event_checks
        rows.append(common | {
            "event_service_limit_factor": factor,
            "non_event_service_limit_factor": 1.0,
            "baseline_service_qualified": all(baseline_checks),
            "action_non_event_service_qualified": all(action_non_event_checks),
            "action_event_service_qualified": all(action_event_checks),
            "service_qualified": all(checks),
            "qualification_checks": len(checks),
            "strict_qualification_checks": (
                len(baseline_checks) + len(action_non_event_checks)
            ),
            "event_allowance_checks": len(action_event_checks),
            "precondition_service_evaluated": precondition_service_evaluated,
            "positive_event_response": facility_event_reduction > tolerance,
            "positive_full_window_response": full_window_reduction > tolerance,
            "qualified_positive_event_response": (
                all(checks) and facility_event_reduction > tolerance
            ),
            "qualified_positive_full_window_response": (
                all(checks) and full_window_reduction > tolerance
            ),
            "qualified_positive_response": (
                all(checks) and full_window_reduction > tolerance
            ),
        })
    return rows


def score_event_manifest(
    run_manifest: pd.DataFrame,
    event_manifest: pd.DataFrame,
    profiles: pd.DataFrame,
    reference_manifest: pd.DataFrame,
    *,
    limit_factors: Iterable[float] = DEFAULT_LIMIT_FACTORS,
) -> pd.DataFrame:
    """Score all completed action runs into one tidy event table."""
    planned_run_ids = set(event_manifest.run_id.astype(str))
    completed_run_ids = set(run_manifest.run_id.astype(str))
    if planned_run_ids != completed_run_ids:
        missing = sorted(planned_run_ids - completed_run_ids)
        unexpected = sorted(completed_run_ids - planned_run_ids)
        raise ValueError(
            "Completed and planned event runs differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    require_complete_event_execution(run_manifest, len(planned_run_ids))
    if "hourly_sha256" not in run_manifest:
        raise ValueError("Completed event manifest lacks hourly_sha256")
    if "hourly_sha256" not in reference_manifest:
        raise ValueError("Reference manifest lacks hourly_sha256")
    reference_paths = reference_manifest.set_index("climate").hourly_path.to_dict()
    reference_hashes = (
        reference_manifest.set_index("climate").hourly_sha256.to_dict()
    )
    profile_climate = _profile_climate_column(profiles)
    rows: list[dict] = []
    for run in run_manifest.to_dict("records"):
        action_path = Path(str(run.get("hourly_path", "")))
        if not action_path.is_file():
            raise FileNotFoundError(f"Completed event trajectory is missing: {action_path}")
        declared_action_hash = str(run.get("hourly_sha256", "")).strip().lower()
        if (
            not declared_action_hash
            or declared_action_hash == "nan"
            or sha256(action_path) != declared_action_hash
        ):
            raise ValueError(f"Action trajectory hash mismatch: {action_path}")
        baseline_artifact_hashes(run)
        baseline = load_compact(Path(run["baseline_hourly_path"]))
        action = load_compact(action_path)
        reference_path = Path(reference_paths[str(run["climate"])])
        declared_reference_hash = str(
            reference_hashes[str(run["climate"])]
        ).strip().lower()
        if (
            not declared_reference_hash
            or declared_reference_hash == "nan"
            or sha256(reference_path) != declared_reference_hash
        ):
            raise ValueError(f"Reference trajectory hash mismatch: {reference_path}")
        reference = load_compact(reference_path)
        local_profiles = profiles[
            profiles[profile_climate].astype(str).eq(str(run["climate"]))
            & profiles.composition_id.astype(str).eq(str(run["composition_id"]))
        ]
        events = event_manifest[event_manifest.run_id.astype(str).eq(str(run["run_id"]))]
        for event in events.to_dict("records"):
            scored = score_event(
                baseline, action, reference, local_profiles, event,
                limit_factors=limit_factors,
            )
            for row in scored:
                row.update({
                    "run_id": run["run_id"],
                    "climate": run["climate"],
                    "composition_id": run["composition_id"],
                    "cohort_label": run["cohort_label"],
                    "control": run["control"],
                    "action_id": run["action_id"],
                    "action_family": run["action_family"],
                    "permission": run["permission"],
                })
            rows.extend(scored)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    reference = sub.add_parser("reference")
    reference.add_argument("--probe-manifest", type=Path, required=True)
    reference.add_argument("--climates", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)

    protocol = sub.add_parser("protocol")
    protocol.add_argument("--reference-manifest", type=Path, required=True)
    protocol.add_argument("--reference-runs", type=Path, help="Reference run folders with native hourly SQL meters")
    protocol.add_argument("--output", type=Path, required=True)

    plan = sub.add_parser("plan")
    plan.add_argument("--annual-manifest", type=Path, required=True)
    plan.add_argument("--protocol", type=Path, required=True)
    plan.add_argument("--profiles", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--protected-fraction", type=float, default=0.2)

    execute = sub.add_parser("execute")
    execute.add_argument("--run-manifest", type=Path, required=True)
    execute.add_argument("--runtime", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--workers", type=int, default=4)
    execute.add_argument("--retain-raw", action="store_true")
    execute.add_argument("--solver-stability-manifest", type=Path)

    score = sub.add_parser("score")
    score.add_argument("--run-manifest", type=Path, required=True)
    score.add_argument("--event-manifest", type=Path, required=True)
    score.add_argument("--profiles", type=Path, required=True)
    score.add_argument("--reference-manifest", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--limit-factors", nargs="*", type=float,
                       default=list(DEFAULT_LIMIT_FACTORS))
    args = parser.parse_args()

    if args.command == "reference":
        result = prepare_reference_manifest(
            pd.read_csv(args.probe_manifest), args.climates
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    elif args.command == "protocol":
        result = select_protocols(pd.read_csv(args.reference_manifest), args.reference_runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    elif args.command == "plan":
        build_event_plan(
            pd.read_csv(args.annual_manifest), pd.read_csv(args.protocol),
            pd.read_csv(args.profiles), args.output_root,
            protected_fraction=args.protected_fraction,
        )
    elif args.command == "execute":
        runs = pd.read_csv(args.run_manifest)
        result = execute_event_plan(
            runs, args.output_root, args.runtime, workers=args.workers,
            cleanup_raw=not args.retain_raw,
            solver_stability_manifest=(
                args.solver_stability_manifest.resolve()
                if args.solver_stability_manifest else None
            ),
        )
        result = attach_planned_event_metadata(result, runs)
        result.to_csv(args.output_root / "EVENT_RUN_MANIFEST_COMPLETED.csv", index=False)
    elif args.command == "score":
        result = score_event_manifest(
            pd.read_csv(args.run_manifest), pd.read_csv(args.event_manifest),
            pd.read_csv(args.profiles), pd.read_csv(args.reference_manifest),
            limit_factors=args.limit_factors,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
