#!/usr/bin/env python3
"""Build the deterministic, minimized Level-1 analysis data tree.

The builder reads only hash-locked scientific products. Human and grouping
identifiers are replaced with keyed release identifiers. No identifier
crosswalk is written.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True

WINDOWS_ABSOLUTE = re.compile(r"(?i)^[a-z]:[\\/]")
POSIX_USER_ABSOLUTE = re.compile(r"^/(?:Users|home)/")
PATH_KEY = re.compile(r"(?i)(?:^|_)(?:path|folder|root|directory|source_file)(?:$|_)")
FORBIDDEN_JSON_KEYS = {
    "subject_id", "user_id", "analysis_unit_id", "source_id", "building_id",
    "participant_timestamp", "timestamp", "latitude", "longitude", "room_name",
}

FIELD_DESCRIPTIONS = {
    "sampling_group_rank": (
        "Zero-based lexical rank of the original held-out group within split; "
        "carries no source or city identity."
    ),
    "sampling_unit_rank": (
        "Zero-based lexical rank of the original analysis unit within its held-out group; "
        "carries no participant identity."
    ),
    "sampling_group_row_rank": (
        "Zero-based position in the frozen filtered source order within held-out group; "
        "used only to reproduce record draws."
    ),
    "feedback_position": (
        "Zero-based position in the frozen first-40 unit permutation for one repetition."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_id(seed: str, domain: str, value: Any, prefix: str) -> str:
    """Return a deterministic random-looking ID; no reverse map is retained."""
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    message = f"{domain}\x00{value}".encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()[:15]
    token = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return f"{prefix}_{token}"


def stable_csv_bytes(frame: pd.DataFrame) -> bytes:
    text = frame.to_csv(index=False, lineterminator="\n", na_rep="", float_format="%.12g")
    return text.encode("utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stable_csv_bytes(frame)
    if path.suffix == ".gz":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                zipped.write(payload)
    else:
        path.write_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for index, key in enumerate(sorted(value)):
            lowered = str(key).lower()
            if lowered in FORBIDDEN_JSON_KEYS or PATH_KEY.search(lowered):
                continue
            clean_key = str(key)
            if WINDOWS_ABSOLUTE.match(clean_key) or POSIX_USER_ABSOLUTE.match(clean_key):
                clean_key = f"removed_machine_path_{index:03d}"
            clean[clean_key] = sanitize_json(value[key])
        return clean
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        if WINDOWS_ABSOLUTE.match(value) or POSIX_USER_ABSOLUTE.match(value):
            return "removed_machine_path"
        return value
    return value


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.map(lambda x: str(x).strip().lower() in {"1", "true", "yes"})


@dataclass
class Product:
    table_id: str
    path: str
    results: str
    release_tier: str
    primary_key: str
    sources: tuple[str, ...]
    transform: str
    row_count: int
    columns: tuple[str, ...]
    sha256: str
    size_bytes: int


class Builder:
    def __init__(self, release_root: Path, config_path: Path, output: Path) -> None:
        self.release_root = release_root
        self.config_path = config_path
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.seed = str(self.config["release_seed"])
        self.source_root = (release_root / self.config["source_project_root"]).resolve()
        self.output = output.resolve()
        self.products: list[Product] = []
        self.provenance: list[dict[str, Any]] = []
        self.codebook: list[dict[str, Any]] = []

    def source(self, source_id: str) -> Path:
        entry = self.config["sources"][source_id]
        path = (self.source_root / entry["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing locked source: {source_id}")
        actual = sha256_file(path)
        expected = str(entry["sha256"]).lower()
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {source_id}: {actual} != {expected}")
        return path

    def read_csv(self, source_id: str, **kwargs: Any) -> pd.DataFrame:
        return pd.read_csv(self.source(source_id), **kwargs)

    def add_frame(
        self,
        table_id: str,
        relative: str,
        frame: pd.DataFrame,
        *,
        results: str,
        tier: str,
        primary_key: list[str],
        sources: list[str],
        transform: str,
        sort_by: list[str] | None = None,
    ) -> None:
        if frame.empty:
            raise RuntimeError(f"refusing to write empty table: {table_id}")
        if sort_by:
            frame = frame.sort_values(sort_by, kind="mergesort", na_position="last").reset_index(drop=True)
        target = self.output / relative
        write_frame(target, frame)
        product = Product(
            table_id=table_id,
            path=relative.replace("\\", "/"),
            results=results,
            release_tier=tier,
            primary_key=";".join(primary_key),
            sources=tuple(sources),
            transform=transform,
            row_count=len(frame),
            columns=tuple(frame.columns),
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
        )
        self.products.append(product)
        for source_id in sources:
            entry = self.config["sources"][source_id]
            self.provenance.append({
                "table_id": table_id,
                "release_path": product.path,
                "source_asset_id": source_id,
                "source_locator": entry["path"],
                "source_sha256": entry["sha256"],
                "transform": transform,
                "release_sha256": product.sha256,
            })
        for column in frame.columns:
            dtype = str(frame[column].dtype)
            if column.startswith(("release_", "anonymous_")) and column.endswith("id"):
                privacy = "release_generated_identifier"
            elif column.startswith("sampling_") or column == "feedback_position":
                privacy = "privacy_safe_sampling_index"
            else:
                privacy = "scientific_value"
            self.codebook.append({
                "table_id": table_id,
                "release_path": product.path,
                "column": column,
                "storage_dtype": dtype,
                "privacy_class": privacy,
                "description": FIELD_DESCRIPTIONS.get(column, column.replace("_", " ")),
            })

    def add_json(
        self,
        table_id: str,
        relative: str,
        source_id: str,
        *,
        results: str,
        tier: str,
        transform: str = "remove paths and prohibited identifier keys; canonical JSON ordering",
    ) -> None:
        value = json.loads(self.source(source_id).read_text(encoding="utf-8"))
        value = sanitize_json(value)
        target = self.output / relative
        write_json(target, value)
        product = Product(
            table_id=table_id,
            path=relative,
            results=results,
            release_tier=tier,
            primary_key="",
            sources=(source_id,),
            transform=transform,
            row_count=1,
            columns=(),
            sha256=sha256_file(target),
            size_bytes=target.stat().st_size,
        )
        self.products.append(product)
        entry = self.config["sources"][source_id]
        self.provenance.append({
            "table_id": table_id,
            "release_path": relative,
            "source_asset_id": source_id,
            "source_locator": entry["path"],
            "source_sha256": entry["sha256"],
            "transform": transform,
            "release_sha256": product.sha256,
        })

    def build_human(self) -> None:
        human = self.read_csv("canonical_office", low_memory=False)
        centres = human[["reference_center_g033_c", "reference_center_g050_c", "reference_center_g067_c"]]
        out = pd.DataFrame({
            "release_profile_id": [release_id(self.seed, "archive_record", v, "pr") for v in human.record_id],
            "release_unit_id": [release_id(self.seed, "analysis_unit", v, "un") for v in human.analysis_unit_id],
            "release_building_group_id": [release_id(self.seed, "building", v, "bg") for v in human.building_id],
            "release_source_group_id": [release_id(self.seed, "source", v, "sg") for v in human.source_id],
            "geographic_stratum": human.country.astype(str) + "::" + human.city.astype(str),
            "country": human.country,
            "city": human.city,
            "climate_group": human.climate,
            "season": human.season,
            "observed_temperature_c": human.observed_temperature_c,
            "thermal_sensation": human.thermal_sensation,
            "thermal_preference": human.thermal_preference_normalized,
            "unacceptable": human.unacceptable,
            "reference_g033_c": human.reference_center_g033_c,
            "reference_g050_c": human.reference_center_g050_c,
            "reference_g067_c": human.reference_center_g067_c,
            "demand_lower_c": centres.min(axis=1),
            "demand_centre_c": human.reference_center_g050_c,
            "demand_upper_c": centres.max(axis=1),
            "record_equal_weight": human.weight_record_equal,
            "unit_equal_weight": human.weight_analysis_unit_equal,
            "building_equal_weight": human.weight_building_equal,
            "source_equal_weight": human.weight_source_equal,
            "model_version": "record_conditioned_griffiths_g033_g050_g067_v1",
        })
        self.add_frame("human_profiles_minimized", "human/human_profiles_minimized.csv.gz", out,
            results="R1;R3;R4;R5;R6", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["release_profile_id"], sources=["canonical_office"],
            transform="drop direct identifiers, times, demographics and coordinates; keyed release IDs; retain minimum hierarchy and demand fields",
            sort_by=["release_profile_id"])

        intervals = self.read_csv("ltpref", low_memory=False)
        out = pd.DataFrame({
            "release_profile_id": [release_id(self.seed, "archive_record", v, "pr") for v in intervals.record_id],
            "interval_lower_c": intervals.executable_low_c,
            "interval_centre_c": intervals.griffiths_reference_center_c,
            "interval_upper_c": intervals.executable_high_c,
            "interval_half_width_c": intervals.ltpref_symmetric_half_width_c,
            "virtual_model_derived": True,
            "main_control_eligible": bool_series(intervals.main_control_eligible),
            "reference_centre_inside_executable": bool_series(intervals.reference_center_in_executable_band),
            "algorithm_version": intervals.implementation,
        })
        self.add_frame("ltpref_intervals_minimized", "human/ltpref_intervals_minimized.csv.gz", out,
            results="R1;R3;R4;R5;R6", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["release_profile_id"], sources=["ltpref"],
            transform="retain executable virtual interval only; replace archive links with release profile ID; label model-derived",
            sort_by=["release_profile_id"])

        frames = []
        for source_id, dataset_code in (("nus_windows", "NUS_HAAS"), ("enth_windows", "ENTH")):
            frame = self.read_csv(source_id)
            frames.append(pd.DataFrame({
                "dataset_code": dataset_code,
                "anonymous_group_id": [release_id(self.seed, f"window_group:{dataset_code}", v, "wg") for v in frame.group_key],
                "anonymous_window_id": [release_id(self.seed, f"window:{dataset_code}", f"{g}|{w}", "ww") for g, w in zip(frame.group_key, frame.window_start)],
                "vote_count": frame.n_people.astype("int64"),
                "prefer_warmer_count": frame.n_prefer_warmer.astype("int64"),
                "no_change_count": frame.n_no_change.astype("int64"),
                "prefer_cooler_count": frame.n_prefer_cooler.astype("int64"),
                "median_temperature_c": frame.median_temperature_c,
                "temperature_coverage": frame.temperature_coverage,
                "compressed_burden": frame.compressed_burden,
                "full_burden": frame.full_burden,
                "jensen_gap": frame.jensen_gap,
                "disagreement_present": bool_series(frame.direct_disagreement),
                "simultaneous_warmer_cooler": bool_series(frame.opposite_direction_conflict),
                "eligible": bool_series(frame.eligible),
                "exclusion_reason": frame.exclusion_reason.fillna("eligible"),
            }))
        windows = pd.concat(frames, ignore_index=True)
        self.add_frame("direct_vote_windows_sanitized", "human/direct_vote_windows_sanitized.csv.gz", windows,
            results="R2", tier="GITHUB_DERIVED_AGGREGATE_ONLY",
            primary_key=["anonymous_window_id"], sources=["nus_windows", "enth_windows"],
            transform="replace room and time keys with no-crosswalk release IDs; remove exact dates, times, spaces and coordinates; retain eligible and excluded windows",
            sort_by=["dataset_code", "anonymous_window_id"])

        pred = self.read_csv("regional_predictions", low_memory=False)
        pred = pred.loc[pred.endpoint.eq("preference") & pred.model.eq("nested_hgb_climate_analogue") & pred.split.isin(["source", "city"])].copy()
        pred["sampling_group_row_rank"] = pred.groupby(["split", "held_out"], sort=False).cumcount().astype("int64")
        group_rank_maps = {
            split: {value: rank for rank, value in enumerate(sorted(group.held_out.astype(str).unique()))}
            for split, group in pred.groupby("split", sort=True)
        }
        pred["sampling_group_rank"] = [
            group_rank_maps[str(split)][str(held_out)]
            for split, held_out in pred[["split", "held_out"]].itertuples(index=False, name=None)
        ]
        pred["sampling_group_rank"] = pred.sampling_group_rank.astype("int64")
        pred["sampling_unit_rank"] = -1
        sampling_rows: list[dict[str, int | str]] = []
        sampling_seed = 20261021
        sampling_repetitions = 200
        sampling_max_feedback = 40
        sampling_min_units = 60
        for (split, held_out), group in pred.groupby(["split", "held_out"], sort=True):
            group = group.copy()
            unit_tokens = group.analysis_unit_id.astype(str)
            unique_units = np.array(sorted(unit_tokens.unique()), dtype=object)
            unit_ranks = {unit: rank for rank, unit in enumerate(unique_units)}
            pred.loc[group.index, "sampling_unit_rank"] = unit_tokens.map(unit_ranks).to_numpy(dtype="int64")
            if len(unique_units) < sampling_min_units:
                continue
            group = group.reset_index(drop=True)
            unit_tokens = group.analysis_unit_id.astype(str)
            by_unit = {
                unit: group.index[unit_tokens.eq(unit)].to_numpy()
                for unit in unique_units
            }
            group_rank = group_rank_maps[str(split)][str(held_out)]
            for repetition in range(sampling_repetitions):
                token = f"{sampling_seed}|{split}|{held_out}|{repetition}".encode("utf-8")
                rng_seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)
                rng = np.random.default_rng(rng_seed)
                permutation = rng.permutation(unique_units)
                for feedback_position, unit in enumerate(permutation[:sampling_max_feedback]):
                    selected = group.iloc[int(rng.choice(by_unit[unit]))]
                    sampling_rows.append({
                        "split": str(split),
                        "sampling_group_rank": int(group_rank),
                        "repetition": repetition,
                        "feedback_position": feedback_position,
                        "sampling_unit_rank": int(unit_ranks[str(unit)]),
                        "sampling_group_row_rank": int(selected.sampling_group_row_rank),
                    })
        if (pred.sampling_unit_rank < 0).any():
            raise RuntimeError("failed to assign a privacy-safe sampling unit rank")
        held = []
        for row in pred.itertuples(index=False):
            domain = "source" if row.split == "source" else "city"
            held.append(release_id(self.seed, domain, row.held_out, "hg"))
        out = pd.DataFrame({
            "split": pred.split,
            "held_out_group_id": held,
            "sampling_group_rank": pred.sampling_group_rank,
            "sampling_unit_rank": pred.sampling_unit_rank.astype("int64"),
            "sampling_group_row_rank": pred.sampling_group_row_rank,
            "release_record_id": [release_id(self.seed, "archive_record", v, "pr") for v in pred.record_id],
            "release_unit_id": [release_id(self.seed, "analysis_unit", v, "un") for v in pred.analysis_unit_id],
            "release_source_group_id": [release_id(self.seed, "source", v, "sg") for v in pred.source_id],
            "release_city_group_id": [release_id(self.seed, "city", v, "cg") for v in pred.city_key],
            "country": pred.country,
            "season": pred.season,
            "climate_group": pred.climate,
            "evaluation_weight": pred.evaluation_weight,
            "observed_preference": pred.observed,
            "p_warmer": pred.p_warmer,
            "p_no_change": pred.p_no_change,
            "p_cooler": pred.p_cooler,
            "model_version": "nested_hgb_climate_analogue_v2",
        })
        self.add_frame("regional_holdout_predictions_minimized", "human/regional_holdout_predictions_minimized.csv.gz", out,
            results="R1;R6", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["split", "release_record_id"], sources=["regional_predictions"],
            transform="filter frozen preference model and whole-source/city splits; replace identifiers with release IDs; retain only non-semantic within-group sampling ranks",
            sort_by=["split", "sampling_group_rank", "sampling_group_row_rank"])

        sampling_plan = pd.DataFrame(sampling_rows)
        self.add_frame("local_update_sampling_plan", "implementation/local_update_sampling_plan.csv.gz", sampling_plan,
            results="R6", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["split", "sampling_group_rank", "repetition", "feedback_position"], sources=["regional_predictions", "local_protocol"],
            transform="freeze the exact first-40 unit permutation and one-record draw using only non-semantic within-group ranks; omit raw identifiers, derived RNG seeds and crosswalks",
            sort_by=["split", "sampling_group_rank", "repetition", "feedback_position"])

        service = self.read_csv("service_curve")
        self.add_frame("service_curve", "human/service_curve.csv", service,
            results="R1;R3;R4;R5", tier="GITHUB_DIRECT",
            primary_key=["distance_from_reference_c"], sources=["service_curve"],
            transform="direct author-derived service curve; canonical row order",
            sort_by=["distance_from_reference_c"])

    def build_policy(self) -> None:
        source = self.read_csv("policy_lock", low_memory=False)
        columns = [
            "policy_id", "plot_label", "plot_order", "layer_order", "layer", "jurisdiction_key",
            "jurisdiction", "document", "version_date", "authority", "document_type", "mandatory_status",
            "numeric_semantics_short", "reference_metric", "document_metric", "metric_compatibility",
            "season_scope", "building_scope", "archive_match_rule", "lower_c", "upper_c", "target_c",
            "main_status", "source_url", "local_snapshot_sha256", "accessed_date", "snapshot_status",
        ]
        out = source[columns].copy().rename(columns={"local_snapshot_sha256": "upstream_snapshot_sha256"})
        self.add_frame("policy_semantics", "policy/policy_semantics.csv", out,
            results="R1", tier="GITHUB_DERIVED_AGGREGATE_ONLY", primary_key=["policy_id"], sources=["policy_lock"],
            transform="drop local snapshot paths and long source excerpts; retain short paraphrase, official locator, version and checksum",
            sort_by=["plot_order", "layer_order", "policy_id"])
        estimates = self.read_csv("policy_estimates")
        self.add_frame("policy_range_estimates", "policy/policy_range_estimates.csv", estimates,
            results="R1", tier="GITHUB_DIRECT", primary_key=["policy_id", "g_label", "class"], sources=["policy_estimates"],
            transform="direct aggregate estimates; canonical row order", sort_by=["plot_order", "g_label", "class"])
        support = self.read_csv("policy_support")
        self.add_frame("policy_support", "policy/policy_support.csv", support,
            results="R1", tier="GITHUB_DIRECT", primary_key=["policy_id"], sources=["policy_support"],
            transform="direct aggregate support and scope table; canonical row order", sort_by=["policy_id"])

    def build_population(self) -> None:
        frame = self.read_csv("assignments", low_memory=False)
        out = pd.DataFrame({
            "scenario_id": frame.population_scenario,
            "composition_seed": frame.composition_seed.astype("int64"),
            "context": frame.city,
            "allocation": frame.allocation,
            "zone_id": frame.zone,
            "assignment_order": frame.assignment_order_in_zone.astype("int64"),
            "release_profile_id": [release_id(self.seed, "synthetic_person", v, "sp") for v in frame.synthetic_person_id],
            "release_archive_profile_id": [release_id(self.seed, "archive_record", v, "pr") for v in frame.record_id],
            "demand_centre_c": frame.reference_center_g050_c,
            "interval_half_width_c": frame.ltpref_symmetric_half_width_c,
            "interval_lower_c": frame.executable_low_c,
            "interval_upper_c": frame.executable_high_c,
            "demand_centre_inside_executable": (
                frame.reference_center_g050_c.ge(frame.executable_low_c)
                & frame.reference_center_g050_c.le(frame.executable_high_c)
            ),
        })
        self.add_frame("population_assignments_minimized", "population/population_assignments_minimized.csv.gz", out,
            results="R3;R4;R5;R6", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["scenario_id", "composition_seed", "context", "allocation", "zone_id", "assignment_order"], sources=["assignments"],
            transform="replace synthetic and archive identities with separate no-crosswalk release IDs; remove source, building, unit and record links",
            sort_by=["scenario_id", "composition_seed", "context", "allocation", "zone_id", "assignment_order"])
        tails = self.read_csv("fixed_tails")
        out = pd.DataFrame({
            "scenario_id": tails.population_scenario,
            "composition_seed": tails.composition_seed.astype("int64"),
            "context": tails.city,
            "tail_rank": tails.tail_rank.astype("int64"),
            "release_profile_id": [release_id(self.seed, "synthetic_person", v, "sp") for v in tails.synthetic_person_id],
            "zone_id": tails.zone,
            "annual_profile_mean_deviation_c": tails.annual_profile_mean_deviation_c,
            "profile_person_hour_weight": tails.profile_person_hour_weight,
        })
        self.add_frame("fixed_tail_memberships_minimized", "population/fixed_tail_memberships_minimized.csv.gz", out,
            results="R3;R4;R5", tier="ZENODO_DERIVED_MINIMIZED",
            primary_key=["scenario_id", "composition_seed", "context", "tail_rank"], sources=["fixed_tails"],
            transform="replace frozen virtual-profile identity with matching release synthetic-profile ID",
            sort_by=["scenario_id", "composition_seed", "context", "tail_rank"])

    def build_building(self) -> None:
        annual = self.read_csv("annual_scores")
        annual_clean = annual.drop(columns=["folder"]).copy()
        self.add_frame("annual_service_scores", "building/annual_service_scores.csv.gz", annual_clean,
            results="R3;R4;R6", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "city", "composition_seed", "delta_c", "lambda", "representation", "physical_action_id"],
            sources=["annual_scores"], transform="remove run folder; retain every representation score including adverse actions",
            sort_by=["population_scenario", "city", "composition_seed", "delta_c", "lambda", "representation", "physical_action_id"])
        energy = ["heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh", "gas_kwh_fuel", "electric_kwh", "site_kwh", "operational_co2_kg"]
        grouped = annual_clean.groupby(["city", "physical_action_id"], sort=True, dropna=False)[energy].nunique(dropna=False)
        if int(grouped.max().max()) != 1:
            raise RuntimeError("annual physical action has inconsistent energy outcomes")
        actions = annual_clean[["city", "physical_action_id", *energy]].drop_duplicates().copy()
        actions.insert(0, "context_id", actions.pop("city"))
        actions["carrier"] = "electricity_and_gas"
        actions["purchased_site_energy_kwh"] = actions.site_kwh
        actions["run_hash"] = actions.physical_action_id
        actions = actions[["context_id", "physical_action_id", "carrier", "purchased_site_energy_kwh", *energy, "run_hash"]]
        self.add_frame("annual_action_library", "building/annual_action_library.csv.gz", actions,
            results="R3;R4;R6", tier="GITHUB_DIRECT", primary_key=["context_id", "physical_action_id"], sources=["annual_scores"],
            transform="deduplicate representation-invariant physical energy outcomes; logical action hash only",
            sort_by=["context_id", "physical_action_id"])

        dynamic = self.read_csv("dynamic_scores")
        dynamic_clean = dynamic.drop(columns=["baseline_folder", "folder"]).copy()
        self.add_frame("dynamic_service_scores", "building/dynamic_service_scores.csv.gz", dynamic_clean,
            results="R5;R6", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "city", "composition_seed", "event_month", "delta_c", "lambda", "representation", "event_action_id"],
            sources=["dynamic_scores"], transform="remove run folders; retain event, recovery, non-positive and adverse outcomes",
            sort_by=["population_scenario", "city", "composition_seed", "event_month", "delta_c", "lambda", "representation", "event_action_id"])
        metrics = [c for c in dynamic_clean.columns if c.endswith(("_kwh", "_kw", "_kwh_fuel"))]
        keys = ["city", "event_action_id", "event_month", "event_day"]
        grouped = dynamic_clean.groupby(keys, sort=True, dropna=False)[metrics].nunique(dropna=False)
        if int(grouped.max().max()) != 1:
            raise RuntimeError("dynamic physical action has inconsistent physical outcomes")
        dyn_actions = dynamic_clean[[*keys, *metrics]].drop_duplicates().copy()
        dyn_actions.insert(0, "context_id", dyn_actions.pop("city"))
        dyn_actions["physical_action_id"] = dyn_actions.event_action_id
        dyn_actions["event_id"] = dyn_actions.apply(lambda r: f"{r.context_id}:m{int(r.event_month):02d}:d{int(r.event_day):02d}", axis=1)
        dyn_actions["event_power_kw"] = dyn_actions.event_average_reduction_kw
        dyn_actions["recovery_window_h"] = 48.0
        dyn_actions["run_hash"] = dyn_actions.physical_action_id
        dyn_actions = dyn_actions[["context_id", "physical_action_id", "event_id", "event_power_kw", "recovery_window_h", *metrics, "run_hash"]]
        self.add_frame("dynamic_action_library", "building/dynamic_action_library.csv.gz", dyn_actions,
            results="R5;R6", tier="GITHUB_DIRECT", primary_key=["context_id", "physical_action_id", "event_id"], sources=["dynamic_scores"],
            transform="deduplicate representation-invariant event and recovery outcomes; retain non-positive response",
            sort_by=["context_id", "event_id", "physical_action_id"])

        aa = self.read_csv("annual_anchors")
        aa = aa.drop(columns=["service_row_indices_0based", "common25_folder"])
        self.add_frame("annual_service_anchors", "building/annual_service_anchors.csv", aa,
            results="R3;R4", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "composition_seed", "city", "allocation"], sources=["annual_anchors"],
            transform="remove machine folder and row-index payload; retain anchor values, counts and hashes",
            sort_by=["population_scenario", "composition_seed", "city", "allocation"])
        da = self.read_csv("dynamic_anchors")
        da = da.drop(columns=["physical_window_row_indices_0based", "service_row_indices_0based", "common25_folder"])
        self.add_frame("dynamic_service_anchors", "building/dynamic_service_anchors.csv", da,
            results="R5", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "composition_seed", "city", "event_month", "window"], sources=["dynamic_anchors"],
            transform="remove machine folder and row-index payload; retain event/recovery anchors, counts and hashes",
            sort_by=["population_scenario", "composition_seed", "city", "event_month", "window"])

        ledger = self.read_csv("failure_ledger")
        selected = ledger.compressed_selected_action_id.fillna(ledger.full_selected_action_id).fillna("NO_SELECTION")
        out = pd.DataFrame({
            "decision_id": [release_id(self.seed, "failure_decision", "|".join(map(str, row)), "fd") for row in ledger.astype(str).itertuples(index=False, name=None)],
            "representation": "compressed_vs_full",
            "selected_action_id": selected,
            "qualification_status": ledger.status,
            "failure_reason": ledger.reasons,
            "layer": ledger.layer,
            "population_scenario": ledger.population_scenario,
            "context_relation": ledger.context_relation,
            "city": ledger.city,
            "composition_seed": ledger.seed,
            "event_month": ledger.event_month,
            "service_fraction": ledger.service_fraction,
            "compressed_selected_action_id": ledger.compressed_selected_action_id,
            "full_selected_action_id": ledger.full_selected_action_id,
        })
        self.add_frame("decision_failure_ledger", "building/decision_failure_ledger.csv", out,
            results="R3;R4;R5", tier="GITHUB_DIRECT", primary_key=["decision_id"], sources=["failure_ledger"],
            transform="retain all failed, unmatched and unresolved outcomes; replace row identity with release decision ID",
            sort_by=["layer", "population_scenario", "city", "composition_seed", "event_month", "service_fraction", "decision_id"])
        nonpositive = self.read_csv("nonpositive_actions").drop(columns=["folder", "baseline_folder"])
        self.add_frame("dynamic_nonpositive_actions", "building/dynamic_nonpositive_actions.csv", nonpositive,
            results="R5", tier="GITHUB_DIRECT", primary_key=["city", "seed", "event_month", "event_action_id"], sources=["nonpositive_actions"],
            transform="remove run folders; preserve every non-positive action", sort_by=["city", "seed", "event_month", "event_action_id"])

        for table_id, relative, source_id, results in (
            ("annual_decisions_reference", "expected/annual_decisions_reference.csv.gz", "annual_decisions", "R4"),
            ("dynamic_decisions_reference", "expected/dynamic_decisions_reference.csv.gz", "dynamic_decisions", "R5"),
        ):
            frame = self.read_csv(source_id)
            self.add_frame(table_id, relative, frame, results=results, tier="GITHUB_DIRECT",
                primary_key=list(frame.columns[:7]), sources=[source_id],
                transform="canonical compressed expected decision table; failures and no-selection states retained",
                sort_by=list(frame.columns[:7]))

        matches = self.read_csv("three_definition_matches").drop(columns=["folder"])
        self.add_frame("three_definition_matches", "building/three_definition_matches.csv.gz", matches,
            results="R3", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"],
            sources=["three_definition_matches"], transform="remove run folder; retain matched, failed and direction-changing same-service comparisons",
            sort_by=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"])
        anchors = self.read_csv("demand_anchors")
        self.add_frame("demand_definition_anchors", "building/demand_definition_anchors.csv", anchors,
            results="R3", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier"], sources=["demand_anchors"],
            transform="direct frozen demand-definition anchors; canonical order",
            sort_by=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier"])
        realloc = self.read_csv("carrier_reallocation")
        realloc = realloc.drop(columns=[c for c in realloc.columns if "folder" in c])
        self.add_frame("carrier_reallocation", "building/carrier_reallocation.csv.gz", realloc,
            results="R3", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"], sources=["carrier_reallocation"],
            transform="remove all run folders; retain carrier reallocations and direction changes",
            sort_by=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"])
        carbon = self.read_csv("regional_carbon_cost").drop(columns=["folder"])
        self.add_frame("regional_carbon_cost", "building/regional_carbon_cost.csv.gz", carbon,
            results="R3", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"], sources=["regional_carbon_cost"],
            transform="remove run folder; retain carbon, carrier and economic-boundary outcomes including sign changes",
            sort_by=["city", "seed", "allocation", "griffiths_coefficient", "interval_width_multiplier", "service_fraction", "strategy"])

        factors = self.read_csv("regional_factors")
        rows = []
        specs = [
            ("electricity_co2_kg_kwh", "electricity_carbon", "kgCO2e/kWh", "grid_basis"),
            ("gas_co2_kg_kwh_hhv", "gas_carbon_hhv", "kgCO2e/kWh_HHV", "grid_basis"),
            ("electricity_price_currency_kwh", "electricity_price", "currency/kWh", "price_basis"),
            ("gas_price_currency_kwh_hhv", "gas_price_hhv", "currency/kWh_HHV", "price_basis"),
        ]
        upstream_hash = self.config["sources"]["regional_factors"]["sha256"]
        for row in factors.itertuples(index=False):
            for column, kind, unit, locator_column in specs:
                rows.append({
                    "factor_id": release_id(self.seed, "factor", f"{row.city}|{kind}", "fc"),
                    "geography": row.city,
                    "factor_type": kind,
                    "value": getattr(row, column),
                    "unit": unit.replace("currency", row.currency),
                    "source_year": "locked_source_year",
                    "source_locator": getattr(row, locator_column),
                    "extraction_locator": "regional_factor_contract_v1",
                    "upstream_hash": upstream_hash,
                })
        factor_frame = pd.DataFrame(rows)
        self.add_frame("regional_factor_contract", "building/regional_factor_contract.csv", factor_frame,
            results="R3", tier="GITHUB_DIRECT", primary_key=["factor_id"], sources=["regional_factors"],
            transform="normalize locked wide factors to one row per geography and factor; preserve basis labels including historic tariff proxy",
            sort_by=["geography", "factor_type"])
        self.add_json("carbon_boundary_contract", "contracts/carbon_boundary_contract.json", "carbon_contract", results="R3", tier="GITHUB_DIRECT")

    def build_implementation(self) -> None:
        rep = self.read_csv("representation_scores").drop(columns=["folder"])
        self.add_frame("representation_scores", "implementation/representation_scores.csv.gz", rep,
            results="R6", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "city", "seed", "allocation", "lambda", "representation", "physical_action_id"], sources=["representation_scores"],
            transform="remove run folder; retain full, mean, q3 and q5 candidate scores and adverse outcomes",
            sort_by=["population_scenario", "city", "seed", "allocation", "lambda", "representation", "physical_action_id"])
        pred = self.output / "human/regional_holdout_predictions_minimized.csv.gz"
        if not pred.is_file():
            raise RuntimeError("regional minimized predictions must be built before implementation inputs")

        phase = self.read_csv("phase_reference")
        self.add_frame("phase_map_reference", "expected/phase_map_reference.csv.gz", phase,
            results="R6", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "city", "seed", "allocation", "lambda", "service_fraction"], sources=["phase_reference"],
            transform="canonical expected phase-map decisions for independent regression; unresolved states retained",
            sort_by=["population_scenario", "city", "seed", "allocation", "lambda", "service_fraction"])
        margins = self.read_csv("quantile_margins")
        self.add_frame("quantile_fold_margins_reference", "expected/quantile_fold_margins_reference.csv", margins,
            results="R6", tier="GITHUB_DIRECT", primary_key=["split_scheme", "fold_id", "representation", "strategy"], sources=["quantile_margins"],
            transform="canonical expected training-fold margins", sort_by=["split_scheme", "fold_id", "representation", "strategy"])
        gate = self.read_csv("quantile_gate")
        self.add_frame("quantile_certification_gate_reference", "expected/quantile_certification_gate_reference.csv", gate,
            results="R6", tier="GITHUB_DIRECT", primary_key=["split_scheme", "representation", "strategy"], sources=["quantile_gate"],
            transform="canonical expected certification gate including failed strategies",
            sort_by=["split_scheme", "representation", "strategy"])
        decisions = self.read_csv("quantile_decisions")
        self.add_frame("quantile_decisions_reference", "expected/quantile_decisions_reference.csv.gz", decisions,
            results="R6", tier="GITHUB_DIRECT",
            primary_key=["population_scenario", "city", "seed", "lambda", "service_fraction", "split_scheme", "representation", "strategy"], sources=["quantile_decisions"],
            transform="canonical expected certification decisions including unresolved and adverse energy penalties",
            sort_by=["population_scenario", "city", "seed", "lambda", "service_fraction", "split_scheme", "representation", "strategy"])
        local = self.read_csv("local_summary")
        self.add_frame("local_update_learning_curve_reference", "expected/local_update_learning_curve_reference.csv", local,
            results="R6", tier="GITHUB_DIRECT", primary_key=["split", "feedback_n"], sources=["local_summary"],
            transform="canonical expected learning-curve summary including unstable small-feedback results",
            sort_by=["split", "feedback_n"])

        for table_id, relative, source_id in (
            ("phase_map_protocol", "contracts/phase_map_protocol.json", "phase_protocol"),
            ("protected_quantile_protocol", "contracts/protected_quantile_protocol.json", "quantile_protocol"),
            ("protected_quantile_summary", "expected/protected_quantile_summary.json", "quantile_summary"),
            ("local_update_protocol", "contracts/local_update_protocol.json", "local_protocol"),
            ("local_update_claim", "expected/local_update_claim.json", "local_claim"),
            ("gate2_protocol", "contracts/gate2_protocol.json", "gate2_protocol"),
            ("gate2_scientific_summary", "expected/gate2_scientific_summary.json", "gate2_summary"),
        ):
            self.add_json(table_id, relative, source_id, results="R3;R4;R5;R6", tier="GITHUB_DIRECT")

    def build_curves(self) -> None:
        support = self.read_csv("city_support")
        support["map_latitude_0p5deg"] = (support.pop("latitude") * 2).round() / 2
        support["map_longitude_0p5deg"] = (support.pop("longitude") * 2).round() / 2
        self.add_frame("city_season_demand_support", "curves/city_season_demand_support.csv", support,
            results="R1", tier="GITHUB_DIRECT", primary_key=["country", "city", "season_group"], sources=["city_support"],
            transform="coarsen map coordinates to 0.5-degree bin centres; retain supported and unsupported strata",
            sort_by=["country", "city", "season_group"])
        hierarchy = self.read_csv("hierarchy_quantiles")
        hierarchy["release_source_group_id"] = hierarchy.source_id.map(
            lambda v: release_id(self.seed, "source", "aggregate_not_applicable" if pd.isna(v) else v, "sg")
        )
        hierarchy = hierarchy.drop(columns=["source_id"])
        hierarchy["hierarchy_group_id"] = [
            release_id(
                self.seed,
                "hierarchy_group",
                "|".join("" if pd.isna(value) else str(value) for value in values),
                "hg",
            )
            for values in hierarchy[["aggregation_level", "country", "city", "release_source_group_id"]].itertuples(index=False, name=None)
        ]
        self.add_frame("hierarchy_quantile_curves", "curves/hierarchy_quantile_curves.csv.gz", hierarchy,
            results="R1", tier="GITHUB_DIRECT", primary_key=["aggregation_level", "season_group", "weighting", "hierarchy_group_id"], sources=["hierarchy_quantiles"],
            transform="replace source label with release group ID; retain city/source/season quantile curves",
            sort_by=["aggregation_level", "season_group", "weighting", "country", "city", "release_source_group_id"])
        power = self.read_csv("event_power_curves")
        self.add_frame("event_power_curves", "curves/event_power_curves.csv.gz", power,
            results="R5", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "strategy", "amplitude_c", "event_month", "hour_from_event_start"], sources=["event_power_curves"],
            transform="direct compact event and 48-hour recovery power/fuel curves; non-positive response retained",
            sort_by=["city", "seed", "allocation", "strategy", "amplitude_c", "event_month", "hour_from_event_start"])
        temp = self.read_csv("event_temperature_service")
        temp = temp.drop(columns=["source_row_index_0based", "source_datetime"])
        self.add_frame("event_temperature_service_curves", "curves/event_temperature_service_curves.csv.gz", temp,
            results="R5", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "event_month", "event_day", "event_action_id", "trajectory_state", "window", "hour_from_event_start"], sources=["event_temperature_service"],
            transform="remove source timestamp and row locator; retain compact event/recovery temperature and service curves",
            sort_by=["event_action_id", "trajectory_state", "window", "hour_from_event_start"])
        zone = self.read_csv("event_zone_temperature")
        zone = zone.drop(columns=["source_row_index_0based", "source_datetime"])
        self.add_frame("event_zone_temperature_curves", "curves/event_zone_temperature_curves.csv.gz", zone,
            results="R5", tier="GITHUB_DIRECT", primary_key=["city", "seed", "allocation", "event_month", "event_day", "event_action_id", "trajectory_state", "window", "hour_from_event_start", "zone"], sources=["event_zone_temperature"],
            transform="remove source timestamp and row locator; retain logical simulated zones and event/recovery temperatures",
            sort_by=["event_action_id", "trajectory_state", "window", "hour_from_event_start", "zone"])
        self.add_json(
            "panel_materialization_support",
            "panel_support/panel_materialization_support.json",
            "panel_materialization_support",
            results="R1;R2",
            tier="GITHUB_DIRECT",
            transform=(
                "publish privacy-safe frozen aggregate panel support; retain crossed-cluster "
                "intervals and input hashes; no record, room, date or participant identifiers"
            ),
        )

    def write_metadata(self) -> None:
        manifest_rows = []
        for product in sorted(self.products, key=lambda p: p.path):
            manifest_rows.append({
                "release_id": self.config["release_id"],
                "logical_path": product.path,
                "table_id": product.table_id,
                "results_used": product.results,
                "release_tier": product.release_tier,
                "row_count": product.row_count,
                "column_count": len(product.columns),
                "primary_key": product.primary_key,
                "source_asset_ids": ";".join(product.sources),
                "source_sha256s": ";".join(self.config["sources"][s]["sha256"] for s in product.sources),
                "transform": product.transform,
                "sha256": product.sha256,
                "size_bytes": product.size_bytes,
            })
        manifest_path = self.release_root / self.config["manifest"]
        write_frame(manifest_path, pd.DataFrame(manifest_rows))
        codebook_path = self.release_root / self.config["codebook"]
        write_frame(codebook_path, pd.DataFrame(self.codebook).sort_values(["release_path", "column"], kind="mergesort"))
        provenance_path = self.release_root / self.config["provenance"]
        write_frame(provenance_path, pd.DataFrame(self.provenance).sort_values(["release_path", "source_asset_id"], kind="mergesort"))

    def run(self) -> None:
        self.build_human()
        self.build_policy()
        self.build_population()
        self.build_building()
        self.build_implementation()
        self.build_curves()
        self.write_metadata()


def safe_prepare_output(release_root: Path, output: Path) -> Path:
    output = output.resolve()
    staging_root = (release_root / "staging").resolve()
    if output == staging_root or staging_root not in output.parents:
        raise RuntimeError("output must be a child of the release staging directory")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("src/settings/release_build.yaml"))
    parser.add_argument("--output", type=Path, default=Path("staging/data"))
    args = parser.parse_args()
    release_root = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else release_root / args.config
    output = args.output if args.output.is_absolute() else release_root / args.output
    output = safe_prepare_output(release_root, output)
    builder = Builder(release_root, config_path.resolve(), output)
    builder.run()
    result = {
        "status": "PASS",
        "release_id": builder.config["release_id"],
        "product_count": len(builder.products),
        "total_rows": sum(p.row_count for p in builder.products),
        "output": output.relative_to(release_root).as_posix(),
        "manifest": builder.config["manifest"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
