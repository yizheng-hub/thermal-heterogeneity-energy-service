#!/usr/bin/env python3
"""Fail-closed validation for the minimized Level-1 release data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.dont_write_bytecode = True

WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![a-z])[a-z]:[\\/]")
POSIX_USER_ABSOLUTE = re.compile(r"^/(?:Users|home)/[^/\s]+/")
RELEASE_ID = re.compile(r"^[a-z]{2}_[a-z2-7]{24}$")
FORBIDDEN_HEADERS = {
    "subject_id", "user_id", "analysis_unit_id", "source_id", "building_id",
    "participant_timestamp", "timestamp", "latitude", "longitude", "room_name",
    "record_id", "synthetic_person_id", "virtual_profile_id", "contributor", "age",
    "gender", "lat", "lon",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header_hash(columns: list[str]) -> str:
    return hashlib.sha256(("\n".join(columns) + "\n").encode("utf-8")).hexdigest()


def json_strings(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)
    elif isinstance(value, str):
        yield value


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


class Validator:
    def __init__(self, release_root: Path, data_root: Path, schema_path: Path, config_path: Path) -> None:
        self.release_root = release_root
        self.data_root = data_root
        self.schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.errors: list[str] = []
        self.checks: dict[str, Any] = {}
        self.frames: dict[str, pd.DataFrame] = {}

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def validate_sources(self) -> None:
        source_root = (self.release_root / self.config["source_project_root"]).resolve()
        verified = 0
        for source_id, entry in sorted(self.config["sources"].items()):
            path = (source_root / entry["path"]).resolve()
            if not path.is_file():
                self.fail(f"missing locked source: {source_id}")
                continue
            actual = sha256_file(path)
            if actual != str(entry["sha256"]).lower():
                self.fail(f"locked source hash mismatch: {source_id}")
            else:
                verified += 1
        self.checks["locked_source_hashes"] = {"verified": verified, "expected": len(self.config["sources"])}

    def expected_local_update_sampling_plan(self) -> pd.DataFrame:
        source_root = (self.release_root / self.config["source_project_root"]).resolve()
        entry = self.config["sources"]["regional_predictions"]
        predictions = pd.read_csv(source_root / entry["path"], low_memory=False)
        predictions = predictions.loc[
            predictions.endpoint.eq("preference")
            & predictions.model.eq("nested_hgb_climate_analogue")
            & predictions.split.isin(["source", "city"])
        ].copy()
        group_rank_maps = {
            split: {value: rank for rank, value in enumerate(sorted(group.held_out.astype(str).unique()))}
            for split, group in predictions.groupby("split", sort=True)
        }
        rows: list[dict[str, int | str]] = []
        for (split, held_out), group in predictions.groupby(["split", "held_out"], sort=True):
            group = group.reset_index(drop=True)
            unit_tokens = group.analysis_unit_id.astype(str)
            unique_units = np.array(sorted(unit_tokens.unique()), dtype=object)
            if len(unique_units) < 60:
                continue
            unit_ranks = {unit: rank for rank, unit in enumerate(unique_units)}
            by_unit = {
                unit: group.index[unit_tokens.eq(unit)].to_numpy()
                for unit in unique_units
            }
            for repetition in range(200):
                token = f"20261021|{split}|{held_out}|{repetition}".encode("utf-8")
                seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**32 - 1)
                rng = np.random.default_rng(seed)
                permutation = rng.permutation(unique_units)
                for position, unit in enumerate(permutation[:40]):
                    row_rank = int(rng.choice(by_unit[unit]))
                    rows.append({
                        "split": str(split),
                        "sampling_group_rank": int(group_rank_maps[str(split)][str(held_out)]),
                        "repetition": repetition,
                        "feedback_position": position,
                        "sampling_unit_rank": int(unit_ranks[str(unit)]),
                        "sampling_group_row_rank": row_rank,
                    })
        return pd.DataFrame(rows).sort_values(
            ["split", "sampling_group_rank", "repetition", "feedback_position"],
            kind="mergesort",
        ).reset_index(drop=True)

    def validate_contracts_and_hashes(self) -> None:
        manifest_path = self.release_root / self.config["manifest"]
        contract_path = self.release_root / self.schema["materialized_contract_index"]
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        contracts = pd.read_csv(contract_path, keep_default_na=False)
        if manifest.table_id.duplicated().any() or manifest.logical_path.duplicated().any():
            self.fail("manifest table_id or logical_path is duplicated")
        if contracts.table_id.duplicated().any() or contracts.logical_path.duplicated().any():
            self.fail("schema contract table_id or logical_path is duplicated")
        if set(manifest.table_id) != set(contracts.table_id):
            self.fail("manifest and schema table IDs differ")
        if set(manifest.logical_path) != set(contracts.logical_path):
            self.fail("manifest and schema logical paths differ")
        actual_files = {
            p.relative_to(self.data_root).as_posix()
            for p in self.data_root.rglob("*") if p.is_file()
        }
        if actual_files != set(manifest.logical_path):
            self.fail(f"materialized file set differs from manifest: extra={sorted(actual_files - set(manifest.logical_path))}, missing={sorted(set(manifest.logical_path) - actual_files)}")

        contract_by_id = contracts.set_index("table_id").to_dict("index")
        byte_hashes = 0
        schema_headers = 0
        primary_keys = 0
        for row in manifest.itertuples(index=False):
            path = self.data_root / row.logical_path
            if not path.is_file():
                continue
            if sha256_file(path) != row.sha256:
                self.fail(f"release byte hash mismatch: {row.logical_path}")
            else:
                byte_hashes += 1
            if path.stat().st_size != int(row.size_bytes):
                self.fail(f"release size mismatch: {row.logical_path}")
            contract = contract_by_id[row.table_id]
            if int(contract["expected_rows"]) != int(row.row_count):
                self.fail(f"manifest row count violates frozen contract: {row.table_id}")
            if path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    self.fail(f"invalid JSON {row.logical_path}: {exc}")
                continue
            frame = load_csv(path)
            self.frames[row.table_id] = frame
            if len(frame) != int(row.row_count):
                self.fail(f"row count mismatch: {row.table_id}")
            if len(frame.columns) != int(row.column_count):
                self.fail(f"column count mismatch: {row.table_id}")
            if header_hash(frame.columns.tolist()) != contract["header_sha256"]:
                self.fail(f"header schema mismatch: {row.table_id}")
            else:
                schema_headers += 1
            key = [c for c in str(contract["primary_key"]).split(";") if c]
            if key:
                missing = [c for c in key if c not in frame.columns]
                if missing:
                    self.fail(f"missing primary-key fields in {row.table_id}: {missing}")
                elif frame[key].isna().any().any() or frame[key].astype(str).eq("").any().any():
                    self.fail(f"null primary key in {row.table_id}")
                elif frame.duplicated(key).any():
                    self.fail(f"duplicate primary key in {row.table_id}")
                else:
                    primary_keys += 1
        self.checks["release_hashes"] = {"verified": byte_hashes, "expected": len(manifest)}
        self.checks["csv_header_schemas"] = {"verified": schema_headers, "expected": int((contracts.header_sha256 != "").sum())}
        self.checks["primary_keys"] = {"verified": primary_keys, "expected": int((contracts.primary_key != "").sum())}
        self.checks["products"] = {"count": len(manifest), "rows": int(manifest.row_count.sum()), "bytes": int(manifest.size_bytes.sum())}

    def validate_privacy_and_paths(self) -> None:
        bad_headers: dict[str, list[str]] = {}
        bad_paths: list[str] = []
        bad_ids: list[str] = []
        for table_id, frame in self.frames.items():
            prohibited = sorted(set(frame.columns) & FORBIDDEN_HEADERS)
            if prohibited:
                bad_headers[table_id] = prohibited
            for column in frame.columns:
                if column.startswith(("release_", "anonymous_")) and column.endswith("id"):
                    nonblank = frame[column].dropna().astype(str)
                    nonblank = nonblank.loc[nonblank.ne("")]
                    invalid = nonblank.loc[~nonblank.str.match(RELEASE_ID)]
                    if not invalid.empty:
                        bad_ids.append(f"{table_id}.{column}")
                if frame[column].dtype == object:
                    values = frame[column].dropna().astype(str)
                    if values.str.contains(WINDOWS_ABSOLUTE).any() or values.str.contains(POSIX_USER_ABSOLUTE).any():
                        bad_paths.append(f"{table_id}.{column}")
        for path in self.data_root.rglob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if any(WINDOWS_ABSOLUTE.search(s) or POSIX_USER_ABSOLUTE.search(s) for s in json_strings(value)):
                bad_paths.append(path.relative_to(self.data_root).as_posix())
        if bad_headers:
            self.fail(f"prohibited payload headers: {bad_headers}")
        if bad_paths:
            self.fail(f"machine paths present: {sorted(set(bad_paths))}")
        if bad_ids:
            self.fail(f"invalid release identifier syntax: {sorted(set(bad_ids))}")
        forbidden_names = [p.relative_to(self.data_root).as_posix() for p in self.data_root.rglob("*") if p.is_file() and ("crosswalk" in p.name.lower() or "mapping" in p.name.lower())]
        if forbidden_names:
            self.fail(f"private crosswalk-like payload is forbidden: {forbidden_names}")
        self.checks["privacy_path_scan"] = {
            "prohibited_header_tables": len(bad_headers),
            "machine_path_locations": len(bad_paths),
            "invalid_release_id_columns": len(bad_ids),
            "crosswalk_files": len(forbidden_names),
        }

    def validate_semantics(self) -> None:
        h = self.frames["human_profiles_minimized"]
        if not ((h.demand_lower_c <= h.demand_centre_c) & (h.demand_centre_c <= h.demand_upper_c)).all():
            self.fail("invalid human demand lower-centre-upper order")
        l = self.frames["ltpref_intervals_minimized"]
        if not (l.interval_lower_c <= l.interval_upper_c).all():
            self.fail("invalid LTPref executable interval order")
        l_inside = (l.interval_centre_c >= l.interval_lower_c) & (l.interval_centre_c <= l.interval_upper_c)
        if not l_inside.eq(as_bool(l.reference_centre_inside_executable)).all():
            self.fail("LTPref centre-inside-executable flag mismatch")
        if not as_bool(l.virtual_model_derived).all():
            self.fail("LTPref virtual_model_derived must be true")

        w = self.frames["direct_vote_windows_sanitized"]
        if not (w.prefer_warmer_count + w.no_change_count + w.prefer_cooler_count).eq(w.vote_count).all():
            self.fail("direct-vote counts do not sum to vote_count")
        eligible = w.loc[as_bool(w.eligible)].groupby("dataset_code").size().to_dict()
        if eligible != {"ENTH": 34, "NUS_HAAS": 63}:
            self.fail(f"eligible direct-vote windows changed: {eligible}")

        p = self.frames["regional_holdout_predictions_minimized"]
        if not np.allclose(p[["p_warmer", "p_no_change", "p_cooler"]].sum(axis=1), 1.0, atol=1e-6):
            self.fail("regional preference probabilities do not sum to one")
        rank_failures: list[str] = []
        for split, split_frame in p.groupby("split", sort=True):
            group_pairs = split_frame[["held_out_group_id", "sampling_group_rank"]].drop_duplicates()
            if group_pairs.held_out_group_id.duplicated().any() or group_pairs.sampling_group_rank.duplicated().any():
                rank_failures.append(f"{split}: group rank is not one-to-one")
            if sorted(group_pairs.sampling_group_rank.astype(int)) != list(range(len(group_pairs))):
                rank_failures.append(f"{split}: group ranks are not consecutive from zero")
        for (split, group_rank), group in p.groupby(["split", "sampling_group_rank"], sort=True):
            row_ranks = sorted(group.sampling_group_row_rank.astype(int))
            if row_ranks != list(range(len(group))):
                rank_failures.append(f"{split}/{group_rank}: row ranks are not consecutive and unique")
            unit_pairs = group[["release_unit_id", "sampling_unit_rank"]].drop_duplicates()
            if unit_pairs.release_unit_id.duplicated().any() or unit_pairs.sampling_unit_rank.duplicated().any():
                rank_failures.append(f"{split}/{group_rank}: unit rank is not one-to-one")
            if sorted(unit_pairs.sampling_unit_rank.astype(int)) != list(range(len(unit_pairs))):
                rank_failures.append(f"{split}/{group_rank}: unit ranks are not consecutive from zero")
        if rank_failures:
            self.fail(f"regional stable sampling ranks failed: {rank_failures[:8]}")

        plan = self.frames["local_update_sampling_plan"]
        plan_groups = plan[["split", "sampling_group_rank"]].drop_duplicates().groupby("split").size().to_dict()
        if plan_groups != {"city": 24, "source": 16}:
            self.fail(f"local-update eligible group cohort changed: {plan_groups}")
        per_group = plan.groupby(["split", "sampling_group_rank"]).size()
        if not per_group.eq(200 * 40).all():
            self.fail("local-update plan must contain 200 repetitions x 40 positions per eligible group")
        per_repetition = plan.groupby(["split", "sampling_group_rank", "repetition"])
        if not per_repetition.size().eq(40).all():
            self.fail("local-update sampling repetition does not contain 40 rows")
        if not per_repetition.feedback_position.apply(lambda values: sorted(values.astype(int)) == list(range(40))).all():
            self.fail("local-update feedback positions must be exactly 0 through 39")
        if not per_repetition.sampling_unit_rank.nunique().eq(40).all():
            self.fail("local-update first-40 draw repeats an analysis unit")
        joined = plan.merge(
            p[["split", "sampling_group_rank", "sampling_group_row_rank", "sampling_unit_rank"]],
            on=["split", "sampling_group_rank", "sampling_group_row_rank"],
            how="left",
            suffixes=("_plan", "_prediction"),
            validate="many_to_one",
            indicator=True,
        )
        if not joined._merge.eq("both").all() or not joined.sampling_unit_rank_plan.eq(joined.sampling_unit_rank_prediction).all():
            self.fail("local-update sampling plan does not resolve exactly to its selected prediction rows and units")
        expected_plan = self.expected_local_update_sampling_plan()
        actual_plan = plan[expected_plan.columns].sort_values(
            ["split", "sampling_group_rank", "repetition", "feedback_position"],
            kind="mergesort",
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(actual_plan, expected_plan, check_dtype=False)
        except AssertionError as exc:
            self.fail(f"local-update sampling plan differs from independently reconstructed locked draws: {exc}")
        a = self.frames["population_assignments_minimized"]
        if not (a.interval_lower_c <= a.interval_upper_c).all():
            self.fail("population assignment executable interval order failed")
        a_inside = (a.demand_centre_c >= a.interval_lower_c) & (a.demand_centre_c <= a.interval_upper_c)
        if not a_inside.eq(as_bool(a.demand_centre_inside_executable)).all():
            self.fail("population centre-inside-executable flag mismatch")

        annual = self.frames["annual_action_library"]
        if len(annual) != 346 or not annual.run_hash.eq(annual.physical_action_id).all():
            self.fail("annual action library physical/run contract failed")
        dynamic = self.frames["dynamic_action_library"]
        if len(dynamic) != 132 or set(dynamic.recovery_window_h.unique()) != {48.0}:
            self.fail("dynamic action library recovery contract failed")
        if not (dynamic.event_power_kw <= 0).any():
            self.fail("dynamic action library lost non-positive response")
        nonpositive = self.frames["dynamic_nonpositive_actions"]
        if not (nonpositive.event_average_reduction_kw <= 0).all():
            self.fail("non-positive action ledger contains a positive response")
        failures = self.frames["decision_failure_ledger"]
        if len(failures) != 377 or failures.failure_reason.fillna("").eq("").any():
            self.fail("failure ledger is incomplete")
        dyn_ref = self.frames["dynamic_decisions_reference"]
        states = set(dyn_ref.mean_selection_state) | set(dyn_ref.full_selection_state)
        if "eligible_but_no_positive_response" not in states:
            self.fail("dynamic no-positive-response decision state was lost")

        reps = self.frames["representation_scores"]
        if len(reps) != 47292 or set(reps.representation) != {"full", "mean_only", "quantile3", "quantile5"}:
            self.fail("representation score grid changed")
        phase = self.frames["phase_map_reference"]
        if set(phase.classification) != {"distribution-required", "finite-library-unresolved", "mean-safe"}:
            self.fail("phase map lost a classification, including unresolved decisions")
        gate = self.frames["quantile_certification_gate_reference"]
        if not as_bool(gate.passes_predefined_certification_gate).any() or as_bool(gate.passes_predefined_certification_gate).all():
            self.fail("protected-quantile gate must retain passing and failing strategies")
        target = gate.loc[
            gate.split_scheme.eq("leave_one_context_seed_out")
            & gate.representation.eq("quantile5")
            & gate.strategy.eq("pooled_q95")
        ]
        if len(target) != 1 or int(target.iloc[0].selected_false_accept_decisions) != 0:
            self.fail("locked q5 pooled_q95 certification result changed")
        learning = self.frames["local_update_learning_curve_reference"]
        if set(learning.feedback_n.astype(int)) != {0, 5, 10, 20, 40} or set(learning.split) != {"city", "source"}:
            self.fail("local-update learning curve grid changed")

        support = self.frames["city_season_demand_support"]
        for column in ("map_latitude_0p5deg", "map_longitude_0p5deg"):
            values = support[column].dropna().to_numpy(float)
            if not np.allclose(values * 2, np.round(values * 2)):
                self.fail(f"map coordinates are not coarsened to 0.5 degree: {column}")
        power = self.frames["event_power_curves"]
        if power.hour_from_event_start.min() != 0 or power.hour_from_event_start.max() != 49:
            self.fail("event/recovery curve must retain full 50-hour window")
        self.checks["semantic_invariants"] = {
            "human_profiles": len(h),
            "ltpref_intervals": len(l),
            "eligible_direct_vote_windows": eligible,
            "annual_actions": len(annual),
            "dynamic_actions": len(dynamic),
            "failure_ledger_rows": len(failures),
            "nonpositive_action_rows": len(nonpositive),
            "representation_score_rows": len(reps),
            "phase_map_rows": len(phase),
            "event_curve_hours": [int(power.hour_from_event_start.min()), int(power.hour_from_event_start.max())],
            "local_update_sampling_plan_rows": len(plan),
            "local_update_eligible_groups": plan_groups,
            "local_update_repetitions_per_group": 200,
            "local_update_feedback_positions": [0, 39],
        }

    def run(self) -> dict[str, Any]:
        if self.schema.get("status") != "R2_MATERIALIZED_LEVEL1_CONTRACT":
            self.fail("schema status is not the locked R2 materialized contract")
        self.validate_sources()
        self.validate_contracts_and_hashes()
        self.validate_privacy_and_paths()
        if not self.errors:
            self.validate_semantics()
        return {
            "status": "PASS" if not self.errors else "FAIL",
            "check": "release_data_materialization",
            "release_id": self.config.get("release_id"),
            "schema_version": self.schema.get("schema_version"),
            "checks": self.checks,
            "errors": self.errors,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("staging/data"))
    parser.add_argument("--schema", type=Path, default=Path("src/settings/data_schemas.yaml"))
    parser.add_argument("--config", type=Path, default=Path("src/settings/release_build.yaml"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    release_root = Path(__file__).resolve().parents[2]
    data_root = args.root if args.root.is_absolute() else release_root / args.root
    schema_path = args.schema if args.schema.is_absolute() else release_root / args.schema
    config_path = args.config if args.config.is_absolute() else release_root / args.config
    result = Validator(release_root, data_root.resolve(), schema_path.resolve(), config_path.resolve()).run()
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        report = args.report if args.report.is_absolute() else release_root / args.report
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
