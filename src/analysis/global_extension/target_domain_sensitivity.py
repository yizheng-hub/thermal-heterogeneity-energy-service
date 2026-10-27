"""Audit annual conclusions against the declared thermostat-target domain.

The annual controllers are optimized from cached 20--27 C probe trajectories.
This module does not infer behaviour outside that declared numerical domain.  It
reports boundary contact and repeats the service-selection and matched-service
comparisons after retaining only candidates whose every zone target is within
a declared subdomain. It can also recompute the offset-zero controller optima
on that subset of the same probes; no EnergyPlus run is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
TARGETED_EXTENSION = HERE.parents[1] / "targeted_extension"
sys.path.insert(0, str(TARGETED_EXTENSION))
sys.path.insert(0, str(HERE.parent))
from demand_aware_controls import generate_controls  # noqa: E402

from analyse_global_results import (  # noqa: E402
    fixed_plan_information_diagnostic,
    select_common_service,
    summarise_information_diagnostic,
)
from matched_service_sensitivity import (  # noqa: E402
    EXCESS_SERVICE,
    MEAN_SERVICE,
    TAIL_SERVICE,
    prepare_metrics,
    resolve_energy_column,
)


IDENTITY = ("climate", "composition", "layout", "permission")
ALTERNATIVE_CONTROLS = ("personal_center", "personal_range")
REFERENCE_CONTROL = "uniform_average"
DEFAULT_DECLARED_DOMAIN_C = (20.0, 27.0)
DEFAULT_CENTRAL_DOMAIN_C = (21.0, 26.0)
EPSILON_C = 1e-9


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _payload_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    result = values.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if result.isna().any():
        raise ValueError("Candidate qualification contains non-boolean values")
    return result.astype(bool)


def read_commands(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Control-command file must contain a non-empty list")
    rows: list[dict] = []
    for item in payload:
        targets = item.get("targets_c")
        if not isinstance(targets, dict) or not targets:
            raise ValueError("Every control command requires zone targets")
        values = np.asarray(list(targets.values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Control targets must be finite")
        rows.append(
            {
                "climate": str(item["context"]),
                "composition_id": int(item["composition_id"]),
                "control": str(item["control"]),
                "offset_c": float(item["offset_c"]),
                "targets_c": targets,
                "command_targets_sha256": _payload_hash(targets),
                "target_zone_count": int(values.size),
                "target_min_c": float(values.min()),
                "target_max_c": float(values.max()),
            }
        )
    commands = pd.DataFrame(rows)
    keys = ["climate", "composition_id", "control", "offset_c"]
    if commands.duplicated(keys).any():
        raise ValueError("Control-command file duplicates a candidate identity")
    return commands


def annotate_candidates(
    metrics: pd.DataFrame,
    run_manifest: pd.DataFrame,
    commands: pd.DataFrame,
    *,
    declared_domain_c: tuple[float, float] = DEFAULT_DECLARED_DOMAIN_C,
    central_domain_c: tuple[float, float] = DEFAULT_CENTRAL_DOMAIN_C,
) -> pd.DataFrame:
    """Attach exact zone-target provenance and domain flags to annual metrics."""
    prepared = prepare_metrics(metrics)
    required = {
        "run_id",
        "climate",
        "composition_id",
        "control",
        "offset_c",
        "targets_sha256",
    }
    missing = required - set(run_manifest)
    if missing:
        raise ValueError(f"Annual run manifest lacks {sorted(missing)}")
    manifest = run_manifest[list(required)].copy()
    if manifest.run_id.duplicated().any():
        raise ValueError("Annual run manifest duplicates run_id")
    manifest["offset_c"] = pd.to_numeric(manifest.offset_c, errors="raise")
    joined = prepared.merge(
        manifest,
        on="run_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    if joined[["climate_manifest", "control_manifest", "composition_id"]].isna().any().any():
        raise ValueError("Annual metrics do not have complete run-manifest provenance")
    if not joined.climate.astype(str).eq(joined.climate_manifest.astype(str)).all():
        raise ValueError("Metric and run-manifest climates differ")
    if not joined.control.astype(str).eq(joined.control_manifest.astype(str)).all():
        raise ValueError("Metric and run-manifest controls differ")
    if not np.allclose(joined.offset_c, joined.offset_c_manifest, rtol=0, atol=EPSILON_C):
        raise ValueError("Metric and run-manifest offsets differ")
    keys = ["climate_manifest", "composition_id", "control_manifest", "offset_c_manifest"]
    command_keys = ["climate", "composition_id", "control", "offset_c"]
    command_copy = commands.rename(columns=dict(zip(command_keys, keys, strict=True)))
    joined = joined.merge(command_copy, on=keys, how="left", validate="many_to_one")
    if joined.targets_c.isna().any():
        raise ValueError("An annual candidate lacks its declared zone targets")
    if not joined.targets_sha256.astype(str).eq(joined.command_targets_sha256).all():
        raise ValueError("Control-command targets differ from simulated annual inputs")

    declared_low, declared_high = map(float, declared_domain_c)
    central_low, central_high = map(float, central_domain_c)
    within_declared = declared_low <= central_low < central_high <= declared_high
    actually_restricted = central_low > declared_low or central_high < declared_high
    if not within_declared or not actually_restricted:
        raise ValueError(
            "Sensitivity target domain must be a proper subset of the declared domain"
        )
    zone_boundary_counts = []
    zone_central_counts = []
    for targets in joined.targets_c:
        values = np.asarray(list(targets.values()), dtype=float)
        boundary = np.isclose(values, declared_low, atol=EPSILON_C) | np.isclose(
            values, declared_high, atol=EPSILON_C
        )
        central = (values >= central_low - EPSILON_C) & (
            values <= central_high + EPSILON_C
        )
        zone_boundary_counts.append(int(boundary.sum()))
        zone_central_counts.append(int(central.sum()))
    joined["zones_at_declared_boundary"] = zone_boundary_counts
    joined["share_zones_at_declared_boundary"] = (
        joined.zones_at_declared_boundary / joined.target_zone_count
    )
    joined["candidate_touches_declared_boundary"] = joined.zones_at_declared_boundary.gt(0)
    joined["zones_within_central_domain"] = zone_central_counts
    joined["candidate_within_central_domain"] = joined.zones_within_central_domain.eq(
        joined.target_zone_count
    )
    joined["base_optimizer_touches_declared_boundary"] = np.isclose(
        joined.offset_c, 0.0, atol=EPSILON_C
    ) & joined.candidate_touches_declared_boundary
    joined["targets_c"] = joined.targets_c.map(
        lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
    )
    return joined.drop(
        columns=[
            "climate_manifest",
            "control_manifest",
            "offset_c_manifest",
            "command_targets_sha256",
        ]
    )


def summarise_candidate_domains(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["climate", "composition", "composition_id", "control"]
    for key, group in candidates.groupby(keys, sort=True, dropna=False):
        base = group[np.isclose(group.offset_c, 0.0, atol=EPSILON_C)]
        if len(base) != 1:
            raise ValueError("Every scenario-control requires one offset-zero candidate")
        base_row = base.iloc[0]
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "candidate_count": int(len(group)),
                "central_domain_candidate_count": int(
                    group.candidate_within_central_domain.sum()
                ),
                "candidate_boundary_contact_count": int(
                    group.candidate_touches_declared_boundary.sum()
                ),
                "base_target_min_c": float(base_row.target_min_c),
                "base_target_max_c": float(base_row.target_max_c),
                "base_zones_at_declared_boundary": int(
                    base_row.zones_at_declared_boundary
                ),
                "base_share_zones_at_declared_boundary": float(
                    base_row.share_zones_at_declared_boundary
                ),
                "base_optimizer_touches_declared_boundary": bool(
                    base_row.base_optimizer_touches_declared_boundary
                ),
            }
        )
    return pd.DataFrame(rows)


def direct_boundary_comparisons(candidates: pd.DataFrame) -> pd.DataFrame:
    direct = candidates[np.isclose(candidates.offset_c, 0.0, atol=EPSILON_C)].copy()
    rows = []
    energy_column, energy_scope = resolve_energy_column(direct)
    for key, group in direct.groupby(list(IDENTITY), sort=True, dropna=False):
        reference = group[group.control.eq(REFERENCE_CONTROL)]
        if len(reference) != 1:
            raise ValueError("Direct comparison requires one uniform candidate")
        uniform = reference.iloc[0]
        for control in ALTERNATIVE_CONTROLS:
            alternative = group[group.control.eq(control)]
            if len(alternative) != 1:
                raise ValueError("Direct comparison requires one personal candidate")
            personal = alternative.iloc[0]
            rows.append(
                dict(zip(IDENTITY, key, strict=True))
                | {
                    "alternative_control": control,
                    "alternative_base_optimizer_touches_declared_boundary": bool(
                        personal.base_optimizer_touches_declared_boundary
                    ),
                    "uniform_base_optimizer_touches_declared_boundary": bool(
                        uniform.base_optimizer_touches_declared_boundary
                    ),
                    "mean_mismatch_change_c": float(personal[MEAN_SERVICE])
                    - float(uniform[MEAN_SERVICE]),
                    "fixed_tail_mismatch_change_c": float(personal[TAIL_SERVICE])
                    - float(uniform[TAIL_SERVICE]),
                    "interval_excess_change_c": float(personal[EXCESS_SERVICE])
                    - float(uniform[EXCESS_SERVICE]),
                    "energy_scope": energy_scope,
                    "energy_change_kwh": float(personal[energy_column])
                    - float(uniform[energy_column]),
                    "energy_change_pct": 100.0
                    * (float(personal[energy_column]) - float(uniform[energy_column]))
                    / float(uniform[energy_column]),
                }
            )
    return pd.DataFrame(rows)


def matched_service_core(
    candidates: pd.DataFrame,
    *,
    domain_scope: str,
    expected_identities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Choose the closest mean-and-tail service pair at the available depth."""
    energy_column, energy_scope = resolve_energy_column(candidates)
    rows = []
    if expected_identities is None:
        expected_identities = candidates[list(IDENTITY)].drop_duplicates()
    expected_identities = expected_identities.sort_values(
        list(IDENTITY), kind="mergesort"
    )
    for key in expected_identities.itertuples(index=False, name=None):
        mask = pd.Series(True, index=candidates.index)
        for column, value in zip(IDENTITY, key, strict=True):
            mask &= candidates[column].eq(value)
        group = candidates[mask]
        reference = group[group.control.eq(REFERENCE_CONTROL)].copy()
        for control in ALTERNATIVE_CONTROLS:
            alternative = group[group.control.eq(control)].copy()
            record = dict(zip(IDENTITY, key, strict=True)) | {
                "domain_scope": domain_scope,
                "alternative_control": control,
                "uniform_candidate_count": int(len(reference)),
                "alternative_candidate_count": int(len(alternative)),
                "energy_scope": energy_scope,
                "energy_column": energy_column,
            }
            if reference.empty or alternative.empty:
                record["selection_status"] = "candidate_unavailable"
                rows.append(record)
                continue
            pairs = []
            for uniform in reference.itertuples(index=False):
                for personal in alternative.itertuples(index=False):
                    mean_gap = abs(
                        float(getattr(personal, MEAN_SERVICE))
                        - float(getattr(uniform, MEAN_SERVICE))
                    )
                    tail_gap = abs(
                        float(getattr(personal, TAIL_SERVICE))
                        - float(getattr(uniform, TAIL_SERVICE))
                    )
                    uniform_energy = float(getattr(uniform, energy_column))
                    personal_energy = float(getattr(personal, energy_column))
                    pairs.append(
                        {
                            "uniform_run_id": str(uniform.run_id),
                            "alternative_run_id": str(personal.run_id),
                            "uniform_offset_c": float(uniform.offset_c),
                            "alternative_offset_c": float(personal.offset_c),
                            "mean_gap_c": mean_gap,
                            "tail_gap_c": tail_gap,
                            "two_metric_linf_gap_c": max(mean_gap, tail_gap),
                            "two_metric_l1_gap_c": mean_gap + tail_gap,
                            "offset_gap_c": abs(
                                float(personal.offset_c) - float(uniform.offset_c)
                            ),
                            "uniform_energy_kwh": uniform_energy,
                            "alternative_energy_kwh": personal_energy,
                            "delta_energy_kwh": personal_energy - uniform_energy,
                            "delta_energy_pct": 100.0
                            * (personal_energy - uniform_energy)
                            / uniform_energy,
                        }
                    )
            chosen = sorted(
                pairs,
                key=lambda row: (
                    row["two_metric_linf_gap_c"],
                    row["two_metric_l1_gap_c"],
                    row["offset_gap_c"],
                    row["uniform_offset_c"],
                    row["alternative_offset_c"],
                    row["uniform_run_id"],
                    row["alternative_run_id"],
                ),
            )[0]
            record.update(chosen)
            record["selection_status"] = "selected"
            rows.append(record)
    return pd.DataFrame(rows)


def common_service_selections(
    candidates: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    declared_scope: str = "declared_20_27_c",
    sensitivity_scope: str = "central_21_26_c",
) -> pd.DataFrame:
    full = select_common_service(candidates, anchors)
    full["domain_scope"] = declared_scope
    central_candidates = candidates[candidates.candidate_within_central_domain].copy()
    central = select_common_service(central_candidates, anchors)
    central["domain_scope"] = sensitivity_scope
    expected_keys = [
        "climate",
        "composition",
        "layout",
        "permission",
        "control",
        "service_fraction",
    ]
    expected = full[expected_keys].drop_duplicates()
    missing = expected.merge(central[expected_keys], on=expected_keys, how="left", indicator=True)
    missing = missing[missing._merge.eq("left_only")].drop(columns="_merge")
    if not missing.empty:
        missing["selection_status"] = "candidate_unavailable"
        missing["engineering_candidates"] = 0
        missing["service_qualified_candidates"] = 0
        missing["domain_scope"] = sensitivity_scope
        central = pd.concat([central, missing], ignore_index=True, sort=False)
    result = pd.concat([full, central], ignore_index=True, sort=False)
    return result.sort_values([*expected_keys, "domain_scope"], kind="mergesort")


def _trim_probe_archive(
    source: Path,
    destination: Path,
    lower_c: float,
    upper_c: float,
) -> None:
    with np.load(source, allow_pickle=False) as saved:
        target = np.asarray(saved["uniform_target_c"], dtype=float)
        keep = (target >= lower_c - EPSILON_C) & (target <= upper_c + EPSILON_C)
        if keep.sum() < 2 or not np.isclose(target[keep].min(), lower_c):
            raise ValueError(f"Probe archive does not span the lower central bound: {source}")
        if not np.isclose(target[keep].max(), upper_c):
            raise ValueError(f"Probe archive does not span the upper central bound: {source}")
        np.savez_compressed(
            destination,
            uniform_target_c=target[keep],
            operative_c=np.asarray(saved["operative_c"])[keep],
            native_people=np.asarray(saved["native_people"]),
        )


def restricted_optimizers(
    controller_root: Path,
    commands_path: Path,
    *,
    central_domain_c: tuple[float, float] = DEFAULT_CENTRAL_DOMAIN_C,
) -> pd.DataFrame:
    """Recompute offset-zero optima on a subset of the same cached probes."""
    contexts_path = controller_root / "controller" / "CONTEXTS.json"
    profiles_path = controller_root / "GLOBAL_PROFILES.csv"
    diagnostics_path = controller_root / "TARGET_DIAGNOSTICS.json"
    for path in (contexts_path, profiles_path, diagnostics_path, commands_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    original_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    original_commands = [
        row
        for row in json.loads(commands_path.read_text(encoding="utf-8"))
        if np.isclose(float(row["offset_c"]), 0.0, atol=EPSILON_C)
    ]
    original_index = {
        (str(row["context"]), int(row["composition_id"]), str(row["control"])): row
        for row in original_commands
    }
    lower_c, upper_c = map(float, central_domain_c)
    with tempfile.TemporaryDirectory(prefix="target-domain-") as temporary:
        root = Path(temporary)
        (root / "controller").mkdir(parents=True)
        restricted_contexts = {}
        for climate, config in contexts.items():
            source = controller_root / str(config["probes"])
            destination = root / "controller" / f"{climate}_probes.npz"
            _trim_probe_archive(source, destination, lower_c, upper_c)
            restricted_contexts[climate] = dict(config) | {
                "probes": f"controller/{destination.name}"
            }
        (root / "controller" / "CONTEXTS.json").write_text(
            json.dumps(restricted_contexts, indent=2) + "\n", encoding="utf-8"
        )
        designs, restricted_diagnostics = generate_controls(
            root,
            profile_table=pd.read_csv(profiles_path),
            permissions=("native_zone",),
            offsets_c=(0.0,),
        )

    rows = []
    for restricted in designs:
        key = (
            str(restricted["context"]),
            int(restricted["composition_id"]),
            str(restricted["control"]),
        )
        original = original_index.get(key)
        if original is None:
            raise ValueError(f"Restricted optimizer lacks original command {key}")
        group_key = f"{key[1]}/{restricted['layout']}"
        objective_key = f"native_zone/{key[2]}"
        original_loss = original_diagnostics[key[0]]["groups"][group_key]["objectives"][
            objective_key
        ]["minimum_loss_c"]
        restricted_loss = restricted_diagnostics[key[0]]["groups"][group_key]["objectives"][
            objective_key
        ]["minimum_loss_c"]
        for zone, restricted_target in restricted["targets_c"].items():
            original_target = float(original["targets_c"][zone])
            rows.append(
                {
                    "climate": key[0],
                    "composition_id": key[1],
                    "control": key[2],
                    "zone_id": zone,
                    "declared_domain_target_c": original_target,
                    "central_domain_target_c": float(restricted_target),
                    "target_shift_c": float(restricted_target) - original_target,
                    "target_changed": not np.isclose(
                        float(restricted_target), original_target, atol=EPSILON_C
                    ),
                    "declared_domain_minimum_loss_c": float(original_loss),
                    "central_domain_minimum_loss_c": float(restricted_loss),
                }
            )
    return pd.DataFrame(rows)


def _scope_summary(frame: pd.DataFrame) -> dict:
    selected = frame[frame.selection_status.eq("selected")]
    energy_change = (
        pd.to_numeric(selected.delta_energy_kwh, errors="raise")
        if "delta_energy_kwh" in selected
        else pd.Series(dtype=float)
    )
    return {
        "comparisons": int(len(frame)),
        "selected": int(len(selected)),
        "unavailable": int(len(frame) - len(selected)),
        "median_two_metric_linf_gap_c": (
            float(selected.two_metric_linf_gap_c.median()) if not selected.empty else None
        ),
        "energy_increases": int((energy_change > 1e-9).sum()),
        "energy_decreases": int((energy_change < -1e-9).sum()),
    }


def _matched_scope_comparison(
    matched: pd.DataFrame,
    *,
    declared_scope: str = "declared_20_27_c",
    sensitivity_scope: str = "central_21_26_c",
) -> dict:
    keys = [*IDENTITY, "alternative_control"]
    declared = matched[matched.domain_scope.eq(declared_scope)].copy()
    central = matched[matched.domain_scope.eq(sensitivity_scope)].copy()
    joined = declared.merge(
        central,
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_declared", "_central"),
    )
    both = joined[
        joined.selection_status_declared.eq("selected")
        & joined.selection_status_central.eq("selected")
    ].copy()
    if both.empty:
        return {
            "comparisons": int(len(joined)),
            "selected_under_both": 0,
            "same_run_pair": 0,
            "same_energy_direction": 0,
            "energy_direction_reversals": 0,
        }
    declared_direction = np.sign(pd.to_numeric(both.delta_energy_kwh_declared))
    central_direction = np.sign(pd.to_numeric(both.delta_energy_kwh_central))
    return {
        "comparisons": int(len(joined)),
        "selected_under_both": int(len(both)),
        "same_run_pair": int(
            (
                both.uniform_run_id_declared.eq(both.uniform_run_id_central)
                & both.alternative_run_id_declared.eq(both.alternative_run_id_central)
            ).sum()
        ),
        "same_energy_direction": int((declared_direction == central_direction).sum()),
        "energy_direction_reversals": int(
            ((declared_direction * central_direction) < 0).sum()
        ),
    }


def _direct_boundary_summary(direct: pd.DataFrame) -> dict:
    result = {}
    grouped = direct.groupby(
        [
            "alternative_control",
            "alternative_base_optimizer_touches_declared_boundary",
        ],
        sort=True,
    )
    for (control, touches), group in grouped:
        result[f"{control}/boundary_contact_{str(bool(touches)).lower()}"] = {
            "comparisons": int(len(group)),
            "population_mean_mismatch_lower": int(
                (group.mean_mismatch_change_c < -EPSILON_C).sum()
            ),
            "fixed_tail_mismatch_lower": int(
                (group.fixed_tail_mismatch_change_c < -EPSILON_C).sum()
            ),
            "energy_increases": int((group.energy_change_kwh > 1e-9).sum()),
            "energy_decreases": int((group.energy_change_kwh < -1e-9).sum()),
        }
    return result


def _restricted_optimizer_summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"computed": False, "zone_targets": 0, "changed_zone_targets": 0}
    summary = {
        "computed": True,
        "zone_targets": int(len(rows)),
        "changed_zone_targets": int(rows.target_changed.astype(bool).sum()),
        "scenario_controls_changed": int(
            rows.groupby(["climate", "composition_id", "control"], sort=True)
            .target_changed.any()
            .sum()
        ),
        "maximum_absolute_target_shift_c": float(rows.target_shift_c.abs().max()),
        "by_control": {},
    }
    for control, group in rows.groupby("control", sort=True):
        summary["by_control"][str(control)] = {
            "zone_targets": int(len(group)),
            "changed_zone_targets": int(group.target_changed.astype(bool).sum()),
            "scenario_controls_changed": int(
                group.groupby(["climate", "composition_id"], sort=True)
                .target_changed.any()
                .sum()
            ),
        }
    return summary


def analyse_target_domain(
    metrics_path: Path,
    anchors_path: Path,
    run_manifest_path: Path,
    commands_path: Path,
    output_root: Path,
    *,
    controller_root: Path | None = None,
    declared_domain_c: tuple[float, float] = DEFAULT_DECLARED_DOMAIN_C,
    central_domain_c: tuple[float, float] = DEFAULT_CENTRAL_DOMAIN_C,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_path)
    anchors = pd.read_csv(anchors_path)
    run_manifest = pd.read_csv(run_manifest_path)
    commands = read_commands(commands_path)
    candidates = annotate_candidates(
        metrics,
        run_manifest,
        commands,
        declared_domain_c=declared_domain_c,
        central_domain_c=central_domain_c,
    )
    declared_scope = (
        f"declared_{declared_domain_c[0]:g}_{declared_domain_c[1]:g}_c"
    )
    sensitivity_scope = (
        f"central_{central_domain_c[0]:g}_{central_domain_c[1]:g}_c"
    )
    candidate_summary = summarise_candidate_domains(candidates)
    direct = direct_boundary_comparisons(candidates)
    information_full = fixed_plan_information_diagnostic(candidates, anchors)
    information_full["domain_scope"] = declared_scope
    information_sensitivity = fixed_plan_information_diagnostic(
        candidates[candidates.candidate_within_central_domain].copy(), anchors
    )
    information_sensitivity["domain_scope"] = sensitivity_scope
    information = pd.concat(
        [information_full, information_sensitivity], ignore_index=True, sort=False
    )
    selections = common_service_selections(
        candidates,
        anchors,
        declared_scope=declared_scope,
        sensitivity_scope=sensitivity_scope,
    )
    matched_full = matched_service_core(candidates, domain_scope=declared_scope)
    matched_central = matched_service_core(
        candidates[candidates.candidate_within_central_domain].copy(),
        domain_scope=sensitivity_scope,
        expected_identities=candidates[list(IDENTITY)].drop_duplicates(),
    )
    matched = pd.concat([matched_full, matched_central], ignore_index=True, sort=False)

    candidates.to_csv(output_root / "TARGET_DOMAIN_CANDIDATES.csv", index=False)
    candidate_summary.to_csv(output_root / "TARGET_DOMAIN_SUMMARY.csv", index=False)
    direct.to_csv(output_root / "TARGET_DOMAIN_DIRECT_COMPARISONS.csv", index=False)
    information.to_csv(
        output_root / "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv", index=False
    )
    selections.to_csv(
        output_root / "TARGET_DOMAIN_COMMON_SERVICE_SELECTIONS.csv", index=False
    )
    matched.to_csv(output_root / "TARGET_DOMAIN_MATCHED_SERVICE.csv", index=False)

    optimizer_rows = pd.DataFrame()
    if controller_root is not None:
        optimizer_rows = restricted_optimizers(
            controller_root,
            commands_path,
            central_domain_c=central_domain_c,
        )
        optimizer_rows.to_csv(
            output_root / "TARGET_DOMAIN_RESTRICTED_OPTIMIZERS.csv", index=False
        )

    base = candidate_summary.base_optimizer_touches_declared_boundary.astype(bool)
    summary = {
        "status": "COMPLETE",
        "interpretation_boundary": (
            "The probe experiment supports controller comparisons only within the "
            f"declared {declared_domain_c[0]:g}--{declared_domain_c[1]:g} C target domain; "
            "it does not identify optima beyond those bounds."
        ),
        "sensitivity_definition": (
            "The central-domain sensitivity retains simulated candidates only when "
            f"every zone target lies within {central_domain_c[0]:g}--"
            f"{central_domain_c[1]:g} C and recomputes offset-zero optima from the same "
            "cached probes restricted to that interval."
        ),
        "declared_target_domain_c": list(map(float, declared_domain_c)),
        "central_sensitivity_domain_c": list(map(float, central_domain_c)),
        "counts": {
            "annual_candidates": int(len(candidates)),
            "central_domain_candidates": int(
                candidates.candidate_within_central_domain.sum()
            ),
            "scenario_controls": int(len(candidate_summary)),
            "base_optimizers_touching_declared_boundary": int(base.sum()),
            "scenario_controls_with_central_candidate": int(
                candidate_summary.central_domain_candidate_count.gt(0).sum()
            ),
        },
        "base_boundary_contact_by_control": {
            str(control): {
                "scenario_controls": int(len(group)),
                "touching": int(
                    group.base_optimizer_touches_declared_boundary.astype(bool).sum()
                ),
            }
            for control, group in candidate_summary.groupby("control", sort=True)
        },
        "matched_service": {
            scope: _scope_summary(group)
            for scope, group in matched.groupby("domain_scope", sort=True)
        },
        "matched_service_scope_comparison": _matched_scope_comparison(
            matched,
            declared_scope=declared_scope,
            sensitivity_scope=sensitivity_scope,
        ),
        "information_diagnostic": {
            scope: summarise_information_diagnostic(group)
            for scope, group in information.groupby("domain_scope", sort=True)
        },
        "direct_results_stratified_by_boundary_contact": _direct_boundary_summary(
            direct
        ),
        "restricted_optimizer": _restricted_optimizer_summary(optimizer_rows),
        "provenance": {
            "metrics_sha256": sha256(metrics_path),
            "anchors_sha256": sha256(anchors_path),
            "run_manifest_sha256": sha256(run_manifest_path),
            "commands_sha256": sha256(commands_path),
        },
    }
    (output_root / "TARGET_DOMAIN_SENSITIVITY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--declared-domain-c", type=float, nargs=2, default=DEFAULT_DECLARED_DOMAIN_C
    )
    parser.add_argument(
        "--central-domain-c", type=float, nargs=2, default=DEFAULT_CENTRAL_DOMAIN_C
    )
    args = parser.parse_args()
    analyse_target_domain(
        args.metrics.resolve(),
        args.anchors.resolve(),
        args.run_manifest.resolve(),
        args.commands.resolve(),
        args.output_root.resolve(),
        controller_root=(args.controller_root.resolve() if args.controller_root else None),
        declared_domain_c=tuple(args.declared_domain_c),
        central_domain_c=tuple(args.central_domain_c),
    )


if __name__ == "__main__":
    main()
