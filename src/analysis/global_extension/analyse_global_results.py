"""Analyse service, energy and carbon across the global-climate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ENERGY_SERVICE = HERE.parents[1] / "energy_service"
sys.path.insert(0, str(ENERGY_SERVICE))
from thermal_service import score_service  # noqa: E402


CONTROLS = ("uniform_average", "personal_center", "personal_range")
CONTROL_ALIASES = {"ltpref_interval": "personal_range"}
SERVICE_FRACTIONS = (1.0, 0.75, 0.5)
SERVICE_TOLERANCE_C = 1e-9
DEFAULT_GAS_CO2_KG_KWH_HHV = 0.1810482350537686


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _column(frame: pd.DataFrame, choices: tuple[str, ...], label: str) -> str:
    found = [name for name in choices if name in frame.columns]
    if not found:
        raise ValueError(f"Missing {label}; expected one of {choices}")
    return found[0]


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    mapping = {"true": True, "false": False, "1": True, "0": False}
    converted = values.astype(str).str.strip().str.lower().map(mapping)
    if converted.isna().any():
        raise ValueError("Engineering qualification contains non-boolean values")
    return converted.astype(bool)


def normalise_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the common schema used by the global analysis."""
    renamed = frame.rename(columns={
        _column(frame, ("climate", "context", "site"), "climate column"): "climate",
        _column(frame, ("cohort_label", "composition", "composition_id"),
                "composition column"):
            "composition",
        _column(frame, ("hourly_path", "compact_output", "path_hourly"),
                "compact hourly path"): "hourly_path",
    }).copy()
    qualification = _column(
        renamed,
        ("engineering_pass", "relative_engineering_pass", "engineering_qualified",
         "primary_output_valid"),
        "engineering qualification",
    )
    renamed["engineering_pass"] = _as_bool(renamed[qualification])
    if "layout" not in renamed:
        renamed["layout"] = "random_mixing"
    if "permission" not in renamed:
        renamed["permission"] = "native_zone"
    if "offset_c" not in renamed:
        renamed["offset_c"] = 0.0
    if "run_id" not in renamed:
        raise ValueError("Run manifest lacks run_id")
    if "control" not in renamed:
        raise ValueError("Run manifest lacks control")
    renamed["control"] = renamed.control.replace(CONTROL_ALIASES)
    if not set(renamed.control).issubset(CONTROLS):
        raise ValueError("Run manifest contains an unknown control")
    keys = ["climate", "composition", "layout", "permission", "control", "offset_c"]
    if renamed.duplicated(keys).any():
        raise ValueError("Run manifest duplicates a logical annual candidate")
    return renamed


def normalise_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={
        _column(frame, ("climate", "context", "site"), "profile climate column"):
            "climate",
        _column(frame, ("cohort_label", "composition", "composition_id"),
                "profile composition column"):
            "composition",
        _column(frame, ("controller_lower_c", "lower_c"), "lower interval bound"):
            "lower_c",
        _column(frame, ("controller_upper_c", "upper_c"), "upper interval bound"):
            "upper_c",
    }).copy()
    if "layout" not in renamed:
        renamed["layout"] = "random_mixing"
    required = {"profile_id", "zone_id", "centre_c", "lower_c", "upper_c"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(f"Profile table lacks {sorted(missing)}")
    numeric = ["centre_c", "lower_c", "upper_c"]
    renamed[numeric] = renamed[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(renamed[numeric].to_numpy()).all():
        raise ValueError("Profile temperatures must be finite")
    if renamed.lower_c.gt(renamed.upper_c).any():
        raise ValueError("A personal range has reversed bounds")
    return renamed


def zone_mean_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Replace each profile centre with its assigned zone's arithmetic mean.

    Profiles share native zone occupancy equally in :func:`score_service`, so
    the arithmetic mean is the matching zone-level representation.  Personal
    range bounds are retained because the fixed-plan information diagnostic
    changes only the centre information used for plan review.
    """
    result = profiles.copy()
    result["centre_c"] = result.groupby("zone_id", sort=False)["centre_c"].transform(
        "mean"
    )
    return result


def physical_plan_identity_column(frame: pd.DataFrame) -> str:
    """Return a stable manifest hash that identifies an exact physical plan."""
    for name in ("input_sha256", "targets_sha256"):
        if name not in frame:
            continue
        values = frame[name].astype(str).str.strip().str.lower()
        if values.eq("").any() or values.eq("nan").any():
            raise ValueError(f"Run manifest contains a blank {name}")
        return name
    raise ValueError(
        "Run manifest needs input_sha256 or targets_sha256 to identify physical plans"
    )


def load_hourly(
    path: Path, *, require_facility: bool = True
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        data = dict(stored)
    required = {
        "zone_ids", "temperatures", "people", "occupied", "month",
        "hvac_electric_kwh", "heating_electric_kwh", "cooling_electric_kwh",
        "fans_electric_kwh", "gas_hhv_kwh",
    }
    if require_facility:
        required |= {"facility_electric_kwh", "facility_gas_hhv_kwh_annual"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    hours = len(data["month"])
    if hours != 8760:
        raise ValueError(f"Expected 8,760 hourly rows: {path}")
    if data["temperatures"].shape != data["people"].shape:
        raise ValueError(f"Temperature and occupancy shapes differ: {path}")
    if data["temperatures"].shape != (hours, len(data["zone_ids"])):
        raise ValueError(f"Zone axis differs from zone identifiers: {path}")
    for name in required - {"zone_ids"}:
        values = np.asarray(data[name])
        if values.dtype.kind not in "USb" and not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name}: {path}")
    return data


def verify_hourly_hash(path: Path, expected: object, label: str) -> str:
    """Fail closed when a compact trajectory differs from its manifest hash."""
    declared = str(expected).strip().lower()
    if not declared or declared == "nan":
        raise ValueError(f"{label} lacks a declared hourly SHA-256")
    observed = sha256(path)
    if observed != declared:
        raise ValueError(
            f"{label} hourly SHA-256 mismatch: expected {declared}, observed {observed}"
        )
    return observed


def profile_zone_positions(data: dict[str, np.ndarray], profiles: pd.DataFrame) -> np.ndarray:
    positions = {str(zone).strip().upper(): index
                 for index, zone in enumerate(data["zone_ids"])}
    missing = sorted({str(zone) for zone in profiles.zone_id
                      if str(zone).strip().upper() not in positions})
    if missing:
        raise ValueError(f"Assigned zones are absent from the trajectory: {missing}")
    return np.asarray(
        [positions[str(zone).strip().upper()] for zone in profiles.zone_id], dtype=int
    )


def score_trajectory(
    data: dict[str, np.ndarray],
    profiles: pd.DataFrame,
    fixed_tail: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    if mask is None:
        mask = np.ones(len(data["occupied"]), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(data["occupied"]),):
        raise ValueError("Service mask differs from the trajectory clock")
    weights = data["people"][mask] * data["occupied"][mask, None].astype(bool)
    return score_service(
        data["temperatures"][mask],
        weights,
        profile_zone_positions(data, profiles),
        profiles.centre_c.to_numpy(float),
        profiles.lower_c.to_numpy(float),
        profiles.upper_c.to_numpy(float),
        fixed_tail,
    )


def fixed_tail_from_anchor(
    data: dict[str, np.ndarray], profiles: pd.DataFrame, fraction: float = 0.10
) -> tuple[np.ndarray, dict]:
    if not 0 < fraction <= 1:
        raise ValueError("Tail fraction must lie in (0, 1]")
    provisional = score_trajectory(data, profiles, np.arange(len(profiles), dtype=int))
    distances = provisional["profiles"]["reference_mismatch_c"]
    ranked = pd.DataFrame({
        "position": np.arange(len(profiles)),
        "profile_id": profiles.profile_id.astype(str),
        "distance": distances,
    }).sort_values(
        ["distance", "profile_id"], ascending=[False, True], kind="mergesort"
    )
    count = max(1, int(np.ceil(fraction * len(profiles))))
    tail = np.sort(ranked.head(count).position.to_numpy(int))
    return tail, score_trajectory(data, profiles, tail)


def energy_metrics(data: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(data["month"]), dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    hvac_electricity = float(np.asarray(data["hvac_electric_kwh"])[mask].sum())
    has_facility = {
        "facility_electric_kwh", "facility_gas_hhv_kwh_annual"
    }.issubset(data)
    facility_electricity = (
        float(np.asarray(data["facility_electric_kwh"])[mask].sum())
        if has_facility else float("nan")
    )
    annual_facility_gas_values = (
        np.asarray(data["facility_gas_hhv_kwh_annual"], dtype=float).reshape(-1)
        if has_facility else np.asarray([np.nan])
    )
    if has_facility and annual_facility_gas_values.size != 1:
        raise ValueError("Compact trajectory must contain one annual facility-gas total")
    complete_year = bool(mask.all())
    facility_gas = (
        float(annual_facility_gas_values[0])
        if has_facility and complete_year else float("nan")
    )
    gas = float(np.asarray(data["gas_hhv_kwh"])[mask].sum())
    heating_electric = float(np.asarray(data["heating_electric_kwh"])[mask].sum())
    cooling = float(np.asarray(data["cooling_electric_kwh"])[mask].sum())
    fans = float(np.asarray(data["fans_electric_kwh"])[mask].sum())
    conditioning_total = hvac_electricity + gas
    facility_plus_heating_gas = facility_electricity + gas
    facility_total = facility_electricity + facility_gas
    heating = heating_electric + gas
    if conditioning_total <= 0 or (
        has_facility and facility_plus_heating_gas <= 0
    ):
        raise ValueError("Purchased annual energy must be positive")
    return {
        "hvac_electric_kwh": hvac_electricity,
        "facility_electric_kwh": facility_electricity,
        "gas_hhv_kwh": gas,
        "facility_gas_hhv_kwh": facility_gas,
        "conditioning_purchased_kwh": conditioning_total,
        "facility_electric_plus_space_heating_gas_kwh": facility_plus_heating_gas,
        "facility_purchased_kwh": facility_total,
        # Backward-compatible main-total alias.  New outputs should use the
        # explicit facility-purchased-energy field above.
        "total_purchased_kwh": facility_total,
        "heating_electric_kwh": heating_electric,
        "cooling_electric_kwh": cooling,
        "fans_electric_kwh": fans,
        "conditioning_heating_share": heating / conditioning_total,
        "conditioning_cooling_share": cooling / conditioning_total,
        "heating_share": heating / conditioning_total,
        "cooling_share": cooling / conditioning_total,
        "load_dominance": "heating" if heating > cooling else "cooling",
    }


def meteorological_period(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    if month in (9, 10, 11):
        return "SON"
    raise ValueError(f"Invalid calendar month: {month}")


def local_season(period: str, latitude: float) -> str:
    north = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
    south = {"DJF": "summer", "MAM": "autumn", "JJA": "winter", "SON": "spring"}
    if period not in north or not np.isfinite(latitude):
        raise ValueError("A valid meteorological period and latitude are required")
    return (north if latitude >= 0 else south)[period]


def choose_minimum_energy(candidates: pd.DataFrame) -> dict | None:
    if candidates.empty:
        return None
    metric = "facility_purchased_kwh"
    minimum = float(candidates[metric].min())
    tolerance = max(1e-6, 1e-9 * abs(minimum))
    return candidates[candidates[metric].le(minimum + tolerance)].sort_values(
        ["run_id", "offset_c"], kind="mergesort"
    ).iloc[0].to_dict()


def select_common_service(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    fractions: tuple[float, ...] = SERVICE_FRACTIONS,
) -> pd.DataFrame:
    keys = ["climate", "composition", "layout"]
    joined = metrics.merge(
        anchors[keys + ["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]],
        on=keys, how="left", validate="many_to_one",
    )
    if joined[["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]].isna().any().any():
        raise ValueError("An annual candidate lacks its 25 C service anchor")
    rows = []
    group_keys = [*keys, "permission", "control"]
    for key, group in joined.groupby(group_keys, sort=True, dropna=False):
        for fraction in fractions:
            mean_limit = fraction * float(group.anchor_mean_mismatch_c.iloc[0])
            tail_limit = fraction * float(group.anchor_tail_mismatch_c.iloc[0])
            qualified = group[
                group.engineering_pass
                & group.mean_reference_mismatch_c.le(mean_limit + SERVICE_TOLERANCE_C)
                & group.fixed_tail_reference_mismatch_c.le(
                    tail_limit + SERVICE_TOLERANCE_C
                )
            ]
            chosen = choose_minimum_energy(qualified)
            item = dict(zip(group_keys, key, strict=True)) | {
                "service_fraction": fraction,
                "mean_limit_c": mean_limit,
                "tail_limit_c": tail_limit,
                "engineering_candidates": int(group.engineering_pass.sum()),
                "service_qualified_candidates": len(qualified),
                "selection_status": "selected" if chosen else "service_unavailable",
            }
            if chosen:
                item.update(chosen)
            rows.append(item)
    return pd.DataFrame(rows)


def _deduplicate_physical_plans(group: pd.DataFrame) -> pd.DataFrame:
    """Collapse aliases of the same input file after checking scored equality."""
    required = {
        "physical_plan_id", "run_id", "control", "offset_c",
        "facility_purchased_kwh", "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c",
        "zone_mean_mean_reference_mismatch_c",
        "zone_mean_fixed_tail_reference_mismatch_c",
    }
    missing = required - set(group.columns)
    if missing:
        raise ValueError(f"Fixed-plan diagnostic lacks {sorted(missing)}")
    checked = (
        "facility_purchased_kwh", "mean_reference_mismatch_c",
        "fixed_tail_reference_mismatch_c",
        "zone_mean_mean_reference_mismatch_c",
        "zone_mean_fixed_tail_reference_mismatch_c",
    )
    rows = []
    for plan_id, aliases in group.groupby("physical_plan_id", sort=True, dropna=False):
        for name in checked:
            values = pd.to_numeric(aliases[name], errors="raise").to_numpy(float)
            scale = max(1.0, float(np.max(np.abs(values))))
            if float(np.ptp(values)) > 1e-9 * scale:
                raise ValueError(
                    f"Physical plan {plan_id} has inconsistent {name} across aliases"
                )
        canonical = aliases.sort_values(
            ["run_id", "control", "offset_c"], kind="mergesort"
        ).iloc[0].to_dict()
        canonical.update({
            "physical_plan_id": str(plan_id),
            "physical_alias_count": len(aliases),
            "physical_source_run_ids": ";".join(
                sorted(aliases.run_id.astype(str).unique())
            ),
            "physical_source_controls": ";".join(
                sorted(aliases.control.astype(str).unique())
            ),
        })
        rows.append(canonical)
    return pd.DataFrame(rows)


def fixed_plan_information_diagnostic(
    metrics: pd.DataFrame,
    anchors: pd.DataFrame,
    fractions: tuple[float, ...] = SERVICE_FRACTIONS,
) -> pd.DataFrame:
    """Compare plan decisions from zone means with full-profile review.

    The candidate trajectories, energy ranking and service limits are held
    fixed.  Only the demand information used to qualify plans changes.
    """
    keys = ["climate", "composition", "layout"]
    joined = metrics.merge(
        anchors[keys + ["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]],
        on=keys, how="left", validate="many_to_one",
    )
    if joined[["anchor_mean_mismatch_c", "anchor_tail_mismatch_c"]].isna().any().any():
        raise ValueError("A fixed-plan candidate lacks its full-profile 25 C anchor")
    if "physical_plan_id" not in joined:
        raise ValueError("Fixed-plan candidates lack a stable physical plan identity")

    rows = []
    group_keys = [*keys, "permission"]
    for key, aliases in joined.groupby(group_keys, sort=True, dropna=False):
        plans = _deduplicate_physical_plans(aliases)
        for fraction in fractions:
            mean_limit = fraction * float(plans.anchor_mean_mismatch_c.iloc[0])
            tail_limit = fraction * float(plans.anchor_tail_mismatch_c.iloc[0])
            engineering = plans.engineering_pass.astype(bool)
            full_mask = (
                engineering
                & plans.mean_reference_mismatch_c.le(
                    mean_limit + SERVICE_TOLERANCE_C
                )
                & plans.fixed_tail_reference_mismatch_c.le(
                    tail_limit + SERVICE_TOLERANCE_C
                )
            )
            mean_mask = (
                engineering
                & plans.zone_mean_mean_reference_mismatch_c.le(
                    mean_limit + SERVICE_TOLERANCE_C
                )
                & plans.zone_mean_fixed_tail_reference_mismatch_c.le(
                    tail_limit + SERVICE_TOLERANCE_C
                )
            )
            full_qualified = plans[full_mask]
            mean_qualified = plans[mean_mask]
            full_choice = choose_minimum_energy(full_qualified)
            mean_choice = choose_minimum_energy(mean_qualified)
            full_ids = set(full_qualified.physical_plan_id.astype(str))
            mean_ids = set(mean_qualified.physical_plan_id.astype(str))
            both_ids = full_ids & mean_ids
            mean_only_ids = mean_ids - full_ids
            full_only_ids = full_ids - mean_ids
            full_id = str(full_choice["physical_plan_id"]) if full_choice else ""
            mean_id = str(mean_choice["physical_plan_id"]) if mean_choice else ""
            selection_disagreement = (
                (full_choice is None) != (mean_choice is None)
                or bool(full_choice and mean_choice and full_id != mean_id)
            )
            row = dict(zip(group_keys, key, strict=True)) | {
                "service_fraction": fraction,
                "mean_limit_c": mean_limit,
                "tail_limit_c": tail_limit,
                "physical_plan_count": len(plans),
                "physical_alias_count": int(len(aliases) - len(plans)),
                "full_qualified_plan_count": len(full_ids),
                "zone_mean_qualified_plan_count": len(mean_ids),
                "qualified_under_both_count": len(both_ids),
                "zone_mean_only_qualified_count": len(mean_only_ids),
                "full_only_qualified_count": len(full_only_ids),
                "eligibility_disagreement": full_ids != mean_ids,
                "full_unresolved": full_choice is None,
                "zone_mean_unresolved": mean_choice is None,
                "selection_disagreement": selection_disagreement,
                "zone_mean_selected_plan_fails_full_review": bool(
                    mean_choice and mean_id not in full_ids
                ),
                "zone_mean_false_reject_no_selection": bool(
                    full_choice and mean_choice is None
                ),
                "full_selected_plan_id": full_id,
                "zone_mean_selected_plan_id": mean_id,
                "full_selection_status": (
                    "selected" if full_choice else "no_qualifying_plan"
                ),
                "zone_mean_selection_status": (
                    "selected" if mean_choice else "no_qualifying_plan"
                ),
            }
            for prefix, choice in (
                ("full_selected", full_choice),
                ("zone_mean_selected", mean_choice),
            ):
                for name in (
                    "run_id", "control", "offset_c", "facility_purchased_kwh",
                    "mean_reference_mismatch_c",
                    "fixed_tail_reference_mismatch_c",
                    "zone_mean_mean_reference_mismatch_c",
                    "zone_mean_fixed_tail_reference_mismatch_c",
                ):
                    row[f"{prefix}_{name}"] = choice.get(name, np.nan) if choice else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarise_information_diagnostic(
    rows: pd.DataFrame, *, reference_target_c: float = 25.0
) -> dict:
    """Return auditable aggregate counts without hiding unresolved scenarios."""
    flags = (
        "eligibility_disagreement", "full_unresolved", "zone_mean_unresolved",
        "selection_disagreement", "zone_mean_selected_plan_fails_full_review",
        "zone_mean_false_reject_no_selection",
    )

    def counts(frame: pd.DataFrame) -> dict:
        distribution_required = (
            frame.eligibility_disagreement.astype(bool)
            | frame.selection_disagreement.astype(bool)
        )
        return {"scenarios": len(frame)} | {
            name: int(frame[name].astype(bool).sum()) for name in flags
        } | {
            "distribution_required": int(distribution_required.sum()),
            "zone_mean_only_qualified_plans": int(
                frame.zone_mean_only_qualified_count.sum()
            ),
            "full_only_qualified_plans": int(frame.full_only_qualified_count.sum()),
        }

    def scoped_counts(frame: pd.DataFrame) -> dict:
        full_resolvable = frame.loc[~frame.full_unresolved.astype(bool)].copy()
        return {
            "all_scenarios": counts(frame),
            "full_distribution_resolvable": counts(full_resolvable),
        }

    by_fraction = {
        f"{float(fraction):g}": scoped_counts(group)
        for fraction, group in rows.groupby("service_fraction", sort=True)
    }
    return {
        "status": "PASS",
        "definition": (
            "The same deduplicated physical plan library is selected by minimum "
            "annual facility-purchased energy under either zone-mean or "
            "full-profile centre-mismatch review. Both reviews use service limits "
            f"from the full-profile external {reference_target_c:g} C reference "
            "and the same fixed least-served profile group."
        ),
        "overall": scoped_counts(rows),
        "headline_scope": (
            "Use by_service_fraction['1'].full_distribution_resolvable for the "
            "main-text denominator. Other service fractions are sensitivity "
            "analyses and are not additional independent observations."
        ),
        "by_service_fraction": by_fraction,
    }


def pair_controls(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    keys = ["climate", "composition", "layout", "permission"]
    if mode == "common_service":
        keys.append("service_fraction")
    reference = rows[rows.control.eq("uniform_average")].set_index(keys)
    metrics = [
        "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
        "modeled_interval_person_hour_coverage", "modeled_interval_mean_excess_c",
        "hvac_electric_kwh", "facility_electric_kwh", "gas_hhv_kwh",
        "facility_gas_hhv_kwh",
        "conditioning_purchased_kwh",
        "facility_electric_plus_space_heating_gas_kwh", "facility_purchased_kwh",
        "total_purchased_kwh",
        "conditioning_operational_co2_country_kg",
        "conditioning_operational_co2_common_factor_kg",
        "operational_co2_country_kg", "operational_co2_common_factor_kg",
    ]
    output = []
    for item in rows[rows.control.ne("uniform_average")].to_dict("records"):
        key = tuple(item[name] for name in keys)
        ref = reference.loc[key] if key in reference.index else None
        selected = item.get("selection_status", "selected") == "selected"
        ref_selected = ref is not None and ref.get("selection_status", "selected") == "selected"
        matched = bool(selected and ref_selected)
        row = {name: item[name] for name in keys} | {
            "comparison_mode": mode,
            "alternative_control": item["control"],
            "status": "matched" if matched else "unmatched",
            "alternative_run_id": item.get("run_id", ""),
            "reference_run_id": ref.get("run_id", "") if ref is not None else "",
        }
        for metric in metrics:
            row[f"delta_{metric}"] = (
                float(item[metric]) - float(ref[metric]) if matched else np.nan
            )
        row["delta_total_purchased_pct"] = (
            100 * row["delta_total_purchased_kwh"] / float(ref.total_purchased_kwh)
            if matched else np.nan
        )
        row["delta_conditioning_purchased_pct"] = (
            100 * row["delta_conditioning_purchased_kwh"]
            / float(ref.conditioning_purchased_kwh)
            if matched else np.nan
        )
        row["delta_facility_electric_plus_space_heating_gas_pct"] = (
            100 * row["delta_facility_electric_plus_space_heating_gas_kwh"]
            / float(ref.facility_electric_plus_space_heating_gas_kwh)
            if matched else np.nan
        )
        row["delta_facility_purchased_pct"] = (
            100 * row["delta_facility_purchased_kwh"]
            / float(ref.facility_purchased_kwh)
            if matched else np.nan
        )
        output.append(row)
    return pd.DataFrame(output)


def _climate_table(path: Path) -> pd.DataFrame:
    content = json.loads(path.read_text(encoding="utf-8"))
    rows = content["climates"] if isinstance(content, dict) else content
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={
        _column(frame, ("climate", "context", "site"), "climate metadata key"):
            "climate"
    })
    if "latitude" not in frame:
        raise ValueError("Climate metadata must include latitude")
    return frame


def _carbon_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.rename(columns={
        _column(frame, ("site", "climate", "context"), "carbon site key"): "climate",
        _column(frame, ("electricity_co2_kg_kwh",), "electricity factor"):
            "electricity_co2_kg_kwh",
        _column(
            frame,
            ("common_world_2025_electricity_co2_kg_kwh", "common_electricity_co2_kg_kwh"),
            "common electricity factor",
        ): "common_electricity_co2_kg_kwh",
    })
    return frame


def analyse(
    run_manifest_path: Path,
    anchor_manifest_path: Path,
    profiles_path: Path,
    climates_path: Path,
    carbon_path: Path,
    output: Path,
    gas_factor: float = DEFAULT_GAS_CO2_KG_KWH_HHV,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    runs = normalise_manifest(pd.read_csv(run_manifest_path))
    if "hourly_sha256" not in runs:
        raise ValueError("Run manifest lacks hourly_sha256")
    if not runs.engineering_pass.all():
        failed = runs.loc[
            ~runs.engineering_pass,
            ["climate", "composition", "control", "offset_c", "run_id"],
        ]
        raise ValueError(
            "Global analysis requires every declared annual candidate to pass "
            f"engineering QA:\n{failed.to_string(index=False)}"
        )
    plan_identity_column = physical_plan_identity_column(runs)
    anchors_manifest = pd.read_csv(anchor_manifest_path)
    anchor_climate = _column(
        anchors_manifest, ("climate", "context", "site"), "anchor climate"
    )
    anchors_manifest = anchors_manifest.rename(columns={anchor_climate: "climate"})
    available_paths = [name for name in ("hourly_path", "compact_output", "path_hourly")
                       if name in anchors_manifest]
    if available_paths:
        anchors_manifest = anchors_manifest.rename(
            columns={available_paths[0]: "hourly_path"}
        )
    elif "run_id" in anchors_manifest:
        anchors_manifest["hourly_path"] = anchors_manifest.run_id.map(
            lambda value: f"hourly/{value}.npz"
        )
    else:
        raise ValueError("Anchor manifest lacks both a compact path and run_id")
    target_column = _column(
        anchors_manifest, ("target_c", "uniform_target_c"), "anchor target"
    )
    anchors_manifest = anchors_manifest[
        np.isclose(pd.to_numeric(anchors_manifest[target_column]), 25.0)
    ].copy()
    if anchors_manifest.duplicated("climate").any():
        raise ValueError("Expected one 25 C anchor trajectory per climate")
    if "hourly_sha256" not in anchors_manifest:
        raise ValueError("Anchor manifest lacks hourly_sha256")
    anchor_paths = dict(zip(anchors_manifest.climate, anchors_manifest.hourly_path, strict=True))
    anchor_hashes = dict(zip(
        anchors_manifest.climate, anchors_manifest.hourly_sha256, strict=True
    ))

    profiles = normalise_profiles(pd.read_csv(profiles_path))
    climates = _climate_table(climates_path)
    carbon = _carbon_table(carbon_path)
    climate = climates.merge(carbon, on="climate", how="left", validate="one_to_one")
    if climate[["latitude", "electricity_co2_kg_kwh",
                "common_electricity_co2_kg_kwh"]].isna().any().any():
        raise ValueError("Climate coordinates or carbon factors are incomplete")
    climate_info = climate.set_index("climate")

    profile_groups = {
        key: group.sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        for key, group in profiles.groupby(["climate", "composition", "layout"], sort=True)
    }
    mean_profile_groups = {
        key: zone_mean_profiles(group) for key, group in profile_groups.items()
    }
    declared = set(map(tuple, runs[["climate", "composition", "layout"]].drop_duplicates(
    ).to_numpy()))
    if declared != set(profile_groups):
        raise ValueError("Run manifest and profile groups do not match")

    def resolved(value: str, source: Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else source.parent / path

    anchor_data: dict[str, dict[str, np.ndarray]] = {}
    tails: dict[tuple, np.ndarray] = {}
    anchor_rows = []
    tail_rows = []
    for key, group in profile_groups.items():
        site = key[0]
        if site not in anchor_paths:
            raise ValueError(f"Missing 25 C anchor for {site}")
        if site not in anchor_data:
            anchor_path = resolved(str(anchor_paths[site]), anchor_manifest_path)
            verify_hourly_hash(
                anchor_path, anchor_hashes[site], f"Anchor trajectory for {site}"
            )
            anchor_data[site] = load_hourly(anchor_path)
        tail, anchor_score = fixed_tail_from_anchor(anchor_data[site], group)
        tails[key] = tail
        anchor_rows.append(dict(zip(
            ["climate", "composition", "layout"], key, strict=True
        )) | {
            "anchor_mean_mismatch_c":
                anchor_score["summary"]["mean_reference_mismatch_c"],
            "anchor_tail_mismatch_c":
                anchor_score["summary"]["fixed_tail_reference_mismatch_c"],
            "tail_profile_count": len(tail),
        })
        for position in tail:
            tail_rows.append(dict(zip(
                ["climate", "composition", "layout"], key, strict=True
            )) | {"profile_id": group.profile_id.iloc[position]})
    anchors = pd.DataFrame(anchor_rows)
    anchors.to_csv(output / "GLOBAL_25C_SERVICE_ANCHORS.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(output / "GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv", index=False)

    trajectory_cache: dict[str, dict[str, np.ndarray]] = {}
    annual_rows = []
    seasonal_rows = []
    for row in runs.itertuples(index=False):
        item = {
            name: getattr(row, name)
            for name in (
                "climate", "composition", "layout", "permission", "control",
                "offset_c", "run_id", "engineering_pass", "hourly_path",
            )
        }
        item["physical_plan_id"] = str(getattr(row, plan_identity_column))
        for name in ("input_sha256", "targets_sha256", "hourly_sha256"):
            if name in runs.columns:
                item[name] = str(getattr(row, name))
        if row.engineering_pass:
            path = resolved(str(row.hourly_path), run_manifest_path)
            verify_hourly_hash(
                path, row.hourly_sha256, f"Annual trajectory {row.run_id}"
            )
            cache_key = str(path.resolve())
            if cache_key not in trajectory_cache:
                trajectory_cache[cache_key] = load_hourly(path)
            data = trajectory_cache[cache_key]
            key = (row.climate, row.composition, row.layout)
            score = score_trajectory(data, profile_groups[key], tails[key])
            mean_score = score_trajectory(data, mean_profile_groups[key], tails[key])
            energy = energy_metrics(data)
            factor = climate_info.loc[row.climate]
            item.update(score["summary"])
            item.update({
                f"zone_mean_{name}": mean_score["summary"][name]
                for name in (
                    "mean_reference_mismatch_c",
                    "fixed_tail_reference_mismatch_c",
                )
            })
            item.update(energy)
            item["conditioning_operational_co2_country_kg"] = (
                energy["hvac_electric_kwh"] * factor.electricity_co2_kg_kwh
                + energy["gas_hhv_kwh"] * gas_factor
            )
            item["conditioning_operational_co2_common_factor_kg"] = (
                energy["hvac_electric_kwh"]
                * factor.common_electricity_co2_kg_kwh
                + energy["gas_hhv_kwh"] * gas_factor
            )
            item[
                "facility_electric_plus_space_heating_gas_operational_co2_country_kg"
            ] = (
                energy["facility_electric_kwh"] * factor.electricity_co2_kg_kwh
                + energy["gas_hhv_kwh"] * gas_factor
            )
            item[
                "facility_electric_plus_space_heating_gas_operational_co2_common_factor_kg"
            ] = (
                energy["facility_electric_kwh"]
                * factor.common_electricity_co2_kg_kwh
                + energy["gas_hhv_kwh"] * gas_factor
            )
            item["operational_co2_country_kg"] = (
                energy["facility_electric_kwh"] * factor.electricity_co2_kg_kwh
                + energy["facility_gas_hhv_kwh"] * gas_factor
            )
            item["operational_co2_common_factor_kg"] = (
                energy["facility_electric_kwh"]
                * factor.common_electricity_co2_kg_kwh
                + energy["facility_gas_hhv_kwh"] * gas_factor
            )
            for period in ("DJF", "MAM", "JJA", "SON"):
                mask = np.asarray(
                    [meteorological_period(int(month)) == period for month in data["month"]]
                )
                season_score = score_trajectory(
                    data, profile_groups[key], tails[key], mask=mask
                )["summary"]
                season_energy = energy_metrics(data, mask=mask)
                seasonal_rows.append({
                    **{name: item[name] for name in (
                        "climate", "composition", "layout", "permission", "control",
                        "offset_c", "run_id",
                    )},
                    "meteorological_period": period,
                    "local_season": local_season(period, float(factor.latitude)),
                    **season_score,
                    **season_energy,
                    "conditioning_operational_co2_country_kg": (
                        season_energy["hvac_electric_kwh"]
                        * factor.electricity_co2_kg_kwh
                        + season_energy["gas_hhv_kwh"] * gas_factor
                    ),
                    "conditioning_operational_co2_common_factor_kg": (
                        season_energy["hvac_electric_kwh"]
                        * factor.common_electricity_co2_kg_kwh
                        + season_energy["gas_hhv_kwh"] * gas_factor
                    ),
                    "facility_electric_plus_space_heating_gas_operational_co2_country_kg": (
                        season_energy["facility_electric_kwh"]
                        * factor.electricity_co2_kg_kwh
                        + season_energy["gas_hhv_kwh"] * gas_factor
                    ),
                    "facility_electric_plus_space_heating_gas_operational_co2_common_factor_kg": (
                        season_energy["facility_electric_kwh"]
                        * factor.common_electricity_co2_kg_kwh
                        + season_energy["gas_hhv_kwh"] * gas_factor
                    ),
                })
        annual_rows.append(item)
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(output / "GLOBAL_RUN_METRICS.csv", index=False)
    pd.DataFrame(seasonal_rows).to_csv(output / "GLOBAL_SEASONAL_METRICS.csv", index=False)

    direct = annual[np.isclose(pd.to_numeric(annual.offset_c), 0.0)].copy()
    direct["selection_status"] = np.where(
        direct.engineering_pass, "selected", "engineering_unavailable"
    )
    direct_pairs = pair_controls(direct, "direct")
    direct_pairs.to_csv(output / "GLOBAL_DIRECT_COMPARISONS.csv", index=False)

    selections = select_common_service(annual, anchors)
    selections.to_csv(output / "GLOBAL_COMMON_SERVICE_SELECTIONS.csv", index=False)
    common_pairs = pair_controls(selections, "common_service")
    common_pairs.to_csv(output / "GLOBAL_COMMON_SERVICE_COMPARISONS.csv", index=False)
    selections[selections.service_fraction.lt(1)].to_csv(
        output / "GLOBAL_STRICT_SERVICE_FEASIBILITY.csv", index=False
    )

    information = fixed_plan_information_diagnostic(annual, anchors)
    information_path = output / "GLOBAL_INFORMATION_DIAGNOSTIC.csv"
    information.to_csv(information_path, index=False)
    information_summary = summarise_information_diagnostic(information)
    information_summary.update({
        "physical_plan_identity_column": plan_identity_column,
        "input_sha256": {
            "run_manifest": sha256(run_manifest_path.resolve()),
            "anchor_manifest": sha256(anchor_manifest_path.resolve()),
            "profiles": sha256(profiles_path.resolve()),
        },
        "output_sha256": {
            information_path.name: sha256(information_path),
        },
    })
    (output / "GLOBAL_INFORMATION_DIAGNOSTIC_SUMMARY.json").write_text(
        json.dumps(information_summary, indent=2) + "\n", encoding="utf-8"
    )

    inputs = {
        "run_manifest": run_manifest_path,
        "anchor_manifest": anchor_manifest_path,
        "profiles": profiles_path,
        "climates": climates_path,
        "carbon_factors": carbon_path,
    }
    produced = sorted([
        *output.glob("GLOBAL_*.csv"),
        *output.glob("GLOBAL_*_SUMMARY.json"),
    ])
    audit = {
        "status": "PASS",
        "climates": int(runs.climate.nunique()),
        "profile_groups": len(profile_groups),
        "logical_candidates": len(runs),
        "engineering_qualified_candidates": int(runs.engineering_pass.sum()),
        "annual_scored_candidates": int(annual.mean_reference_mismatch_c.notna().sum()),
        "direct_matched_pairs": int(direct_pairs.status.eq("matched").sum()),
        "common_service_matched_pairs": int(common_pairs.status.eq("matched").sum()),
        "information_diagnostic_scenarios": len(information),
        "information_selection_disagreements": int(
            information.selection_disagreement.sum()
        ),
        "zone_mean_selected_plan_failures_under_full_review": int(
            information.zone_mean_selected_plan_fails_full_review.sum()
        ),
        "service_fractions": list(SERVICE_FRACTIONS),
        "fixed_tail_fraction": 0.10,
        "gas_co2_kg_kwh_hhv": gas_factor,
        "main_energy_boundary": (
            "annual Electricity:Facility plus annual facility Natural Gas from the "
            "AnnualBuildingUtilityPerformanceSummary (HHV)"
        ),
        "seasonal_facility_energy_boundary": (
            "hourly Electricity:Facility plus hourly Heating:NaturalGas (HHV); "
            "annual facility gas is not seasonally disaggregated"
        ),
        "conditioning_energy_boundary": (
            "heating, cooling and fan electricity plus Heating:NaturalGas (HHV)"
        ),
        "manifest_hourly_hashes_verified": True,
        "input_sha256": {name: sha256(path.resolve()) for name, path in inputs.items()},
        "output_sha256": {path.name: sha256(path) for path in produced},
    }
    (output / "GLOBAL_ANALYSIS_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--climates", type=Path, required=True)
    parser.add_argument("--carbon-factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gas-co2-kg-kwh-hhv", type=float, default=DEFAULT_GAS_CO2_KG_KWH_HHV
    )
    args = parser.parse_args()
    audit = analyse(
        args.run_manifest.resolve(),
        args.anchor_manifest.resolve(),
        args.profiles.resolve(),
        args.climates.resolve(),
        args.carbon_factors.resolve(),
        args.output.resolve(),
        args.gas_co2_kg_kwh_hhv,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
