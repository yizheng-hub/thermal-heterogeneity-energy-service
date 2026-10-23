"""Deterministically materialize 48 panel-specific figure-data v2 products."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, output_root, resolve_asset
from .io import sha256, write_json, write_table


PANEL_IDS = [f"Fig{figure}{letter}" for figure in range(1, 7) for letter in "abcdefgh"]
SCHEMATICS = {"Fig3a", "Fig3c", "Fig4a", "Fig5a", "Fig6a", "Fig6h"}


def validate_panel_registry(panels: list[dict]) -> None:
    observed = [panel.get("panel_id") for panel in panels]
    expected = set(PANEL_IDS)
    if len(observed) != 48 or set(observed) != expected or len(set(observed)) != 48:
        raise ValueError(
            "panel registry must contain Fig1a-Fig6h exactly; "
            f"missing={sorted(expected-set(observed))}, extra={sorted(set(observed)-expected)}"
        )
    for panel in panels:
        for field in ("result", "source_table", "citation_key"):
            if field not in panel:
                raise ValueError(f"{panel['panel_id']} missing {field}")


def _legacy_reproduce(config: dict[str, Any], root: Path) -> Path:
    """Retain the small synthetic smoke fixture used by package tests."""
    files = []
    for panel in sorted(config["figures"]["panels"], key=lambda item: item["panel_id"]):
        source = root / "tables" / panel["source_table"]
        frame = pd.read_csv(source)
        if frame.empty:
            raise ValueError(f"empty source data for {panel['panel_id']}")
        frame.insert(0, "panel_id", panel["panel_id"])
        target = write_table(
            frame,
            root / "figure_data" / f"{panel['panel_id']}.csv",
            sort_by=list(frame.columns),
        )
        files.append(
            {
                "panel_id": panel["panel_id"],
                "path": target.relative_to(root).as_posix(),
                "sha256": sha256(target),
                "rows": len(frame),
            }
        )
    return write_json(
        root / "qa" / "figure_data_manifest.json",
        {"panel_count": 48, "panels": files},
    )


def _stable_id(*values: Any) -> str:
    text = "\x1f".join("" if pd.isna(value) else str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _direction(values: list[float]) -> str | None:
    array = np.asarray(values, dtype=float)
    if np.isnan(array).all():
        return None
    if np.isnan(array).any():
        return "match_lost"
    if np.all(np.isclose(array, 0, atol=1e-9)):
        return "all_tied"
    if np.all(array > 1e-9):
        return "same_direction_increase"
    if np.all(array < -1e-9):
        return "same_direction_decrease"
    return "direction_not_stable"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class Materializer:
    def __init__(self, config: dict[str, Any], analysis_root: Path, panel_root: Path) -> None:
        self.config = config
        self.analysis_root = analysis_root
        self.panel_root = panel_root
        self.panel_root.mkdir(parents=True, exist_ok=True)
        self.frames: dict[str, pd.DataFrame] = {}
        self.input_records: dict[str, dict[str, Any]] = {}
        self.input_record_keys: dict[str, str] = {}
        self.panels: dict[str, dict[str, Any]] = {}
        self.panel_inputs: dict[str, list[str]] = {}
        self.filters: dict[str, str] = {}
        self.units: dict[str, dict[str, str]] = {}
        self.denominators: dict[str, list[str]] = {}
        self.transform_hash = sha256(Path(__file__))
        path = resolve_asset(config, "panel_materialization_support")
        self._record_input(
            path,
            f"clean_data/{config['assets']['panel_materialization_support']}",
            "panel_materialization_support",
        )
        self.support = json.loads(path.read_text(encoding="utf-8"))

    def _record_input(
        self, path: Path, logical_path: str, source_key: str | None = None
    ) -> None:
        key = str(path.resolve())
        if source_key is not None:
            self.input_record_keys[source_key] = key
        if key in self.input_records:
            return
        record: dict[str, Any] = {
            "path": logical_path,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".json":
            record.update({"row_count": 1, "ordered_header": []})
        else:
            frame = pd.read_csv(path, low_memory=False)
            record.update({"row_count": len(frame), "ordered_header": list(frame.columns)})
        self.input_records[key] = record

    def table(self, name: str) -> pd.DataFrame:
        key = f"table::{name}"
        if key not in self.frames:
            path = self.analysis_root / "tables" / f"{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"missing Level-1 table: {path}")
            self._record_input(path, f"analysis/tables/{name}.csv", name)
            self.frames[key] = pd.read_csv(path, low_memory=False)
        return self.frames[key].copy()

    def asset(self, asset_id: str) -> pd.DataFrame:
        key = f"asset::{asset_id}"
        if key not in self.frames:
            path = resolve_asset(self.config, asset_id)
            self._record_input(
                path, f"clean_data/{self.config['assets'][asset_id]}", asset_id
            )
            self.frames[key] = pd.read_csv(path, low_memory=False)
        return self.frames[key].copy()

    def add_csv(
        self,
        panel_id: str,
        frame: pd.DataFrame,
        *,
        inputs: list[str],
        filter_used: str,
        units: dict[str, str],
        denominator_fields: list[str],
        filename: str | None = None,
    ) -> None:
        if frame.empty or "panel_id" in frame:
            raise ValueError(f"invalid panel frame: {panel_id}")
        frame = frame.copy()
        frame.insert(0, "panel_id", panel_id)
        path = self.panel_root / (filename or f"{panel_id}.csv")
        write_table(frame, path, sort_by=list(frame.columns))
        self.panels[panel_id] = {
            "panel_id": panel_id,
            "path": path,
            "rows": len(frame),
            "ordered_fields": list(frame.columns),
        }
        self.panel_inputs[panel_id] = inputs
        self.filters[panel_id] = filter_used
        self.units[panel_id] = units
        self.denominators[panel_id] = denominator_fields

    def add_meta(
        self,
        panel_id: str,
        payload: dict[str, Any],
        *,
        inputs: list[str],
        filter_used: str,
        filename: str | None = None,
    ) -> None:
        value = {"panel_id": panel_id, **payload}
        path = self.panel_root / (filename or f"{panel_id}_meta.json")
        write_json(path, _json_safe(value))
        self.panels[panel_id] = {
            "panel_id": panel_id,
            "path": path,
            "rows": len(payload.get("items", payload.get("steps", []))) or 1,
            "ordered_fields": list(value),
        }
        self.panel_inputs[panel_id] = inputs
        self.filters[panel_id] = filter_used
        self.units[panel_id] = {}
        self.denominators[panel_id] = []

    def figure1(self) -> None:
        profiles = self.asset("human_profiles")
        intervals = self.asset("ltpref_intervals")
        executable_ids = set(
            intervals.loc[
                intervals["main_control_eligible"].astype(bool), "release_profile_id"
            ]
        )
        executable = profiles.loc[profiles["release_profile_id"].isin(executable_ids)]
        stages = []
        for label, frame in (("complete", profiles), ("executable", executable)):
            stages.append(
                {
                    "population": label,
                    "records": len(frame),
                    "analysis_units": frame["release_unit_id"].nunique(),
                    "buildings": frame["release_building_group_id"].nunique(),
                    "cities": frame[["country", "city"]].drop_duplicates().shape[0],
                    "countries": frame["country"].nunique(),
                }
            )
        self.add_csv(
            "Fig1a",
            pd.DataFrame(stages),
            inputs=["human_profiles", "ltpref_intervals"],
            filter_used="complete archive; executable=main_control_eligible release_profile_id",
            units={column: "count" for column in stages[0] if column != "population"},
            denominator_fields=["population"],
        )

        city_support = self.asset("city_season_support")
        geo_audit = []
        for panel_id, season in (("Fig1b", "warm"), ("Fig1c", "cold")):
            frame = city_support.loc[city_support["season_group"].eq(season)].copy()
            frame["coordinate_status"] = "source_coarsened_0p5_degree"
            bangalore = frame["city"].astype(str).str.casefold().eq("bangalore")
            frame.loc[bangalore, "map_latitude_0p5deg"] = 13.0
            frame.loc[bangalore, "map_longitude_0p5deg"] = 77.5
            frame.loc[bangalore, "coordinate_status"] = "display_only_not_source_geolocation"
            for row in frame.loc[bangalore].itertuples(index=False):
                geo_audit.append(
                    {
                        "panel_id": panel_id,
                        "city": row.city,
                        "country": row.country,
                        "latitude": row.map_latitude_0p5deg,
                        "longitude": row.map_longitude_0p5deg,
                        "coordinate_status": row.coordinate_status,
                        "scientific_use": "display_only",
                    }
                )
            frame = frame.rename(
                columns={
                    "season_group": "season",
                    "map_latitude_0p5deg": "latitude",
                    "map_longitude_0p5deg": "longitude",
                    "analysis_unit_count": "analysis_units",
                    "g050_p50_c": "demand_center_c",
                }
            )
            frame.loc[~frame["supported"].astype(bool), "demand_center_c"] = np.nan
            self.add_csv(
                panel_id,
                frame[
                    [
                        "city",
                        "country",
                        "season",
                        "latitude",
                        "longitude",
                        "coordinate_status",
                        "supported",
                        "analysis_units",
                        "demand_center_c",
                    ]
                ],
                inputs=["city_season_support"],
                filter_used=f"season_group={season}; g=0.50; unsupported centres blank",
                units={"latitude": "degree", "longitude": "degree", "analysis_units": "count", "demand_center_c": "degC"},
                denominator_fields=["analysis_units", "supported"],
            )
        write_table(pd.DataFrame(geo_audit), self.panel_root / "Fig1_geo_display_audit.csv")

        hierarchy = self.asset("hierarchy_quantiles")
        frame = hierarchy.loc[hierarchy["weighting"].eq("analysis_unit_equal")].copy()
        frame["name"] = np.where(
            frame["aggregation_level"].eq("city"),
            frame["country"].astype(str) + "::" + frame["city"].astype(str),
            frame["release_source_group_id"],
        )
        frame = frame.rename(
            columns={
                "season_group": "season",
                "analysis_unit_count": "n_analysis_units",
                "g050_p10_c": "p10_c",
                "g050_p50_c": "p50_c",
                "g050_p90_c": "p90_c",
            }
        )
        self.add_csv(
            "Fig1d",
            frame[["aggregation_level", "weighting", "season", "name", "n_analysis_units", "p10_c", "p50_c", "p90_c"]],
            inputs=["hierarchy_quantiles"],
            filter_used="weighting=analysis_unit_equal; g=0.50; city and source levels",
            units={"n_analysis_units": "count", "p10_c": "degC", "p50_c": "degC", "p90_c": "degC"},
            denominator_fields=["n_analysis_units"],
        )
        variance = self.table("result1_variance").rename(columns={"variance_c2": "variance"})
        self.add_csv(
            "Fig1e",
            variance[["component", "variance", "share"]],
            inputs=["result1_variance"],
            filter_used="all five frozen variance components",
            units={"variance": "degC2", "share": "fraction"},
            denominator_fields=["share"],
        )
        coverage = self.table("result1_policy_coverage")
        coverage = coverage.loc[coverage["g_label"].eq("g0.50")].merge(
            self.table("result1_policy_support")[["policy_id", "records", "analysis_units"]],
            on="policy_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_support"),
        )
        coverage = coverage.rename(
            columns={"plot_label": "policy", "layer": "policy_type", "g_label": "griffiths", "class": "position", "records_support": "records_denominator"}
        )
        self.add_csv(
            "Fig1f",
            coverage[["policy", "policy_type", "griffiths", "position", "proportion", "ci_low", "ci_high", "records_denominator", "analysis_units"]].rename(columns={"records_denominator": "records"}),
            inputs=["result1_policy_coverage", "result1_policy_support"],
            filter_used="g_label=g0.50; all nine bilateral formal/design documents",
            units={"proportion": "fraction", "ci_low": "fraction", "ci_high": "fraction", "records": "count", "analysis_units": "count"},
            denominator_fields=["records", "analysis_units"],
        )
        target = pd.DataFrame(self.support["target_delta_distributions"])
        target = target.loc[
            target["g_label"].eq("g0.50")
            & target["policy_id"].isin(["CHN_GB2007_SUMMER", "JPN_COOLBIZ28", "THA_EGAT2023"])
        ].rename(
            columns={"plot_label": "policy", "g_label": "griffiths", "q10": "q10_delta_c", "q25": "q25_delta_c", "median": "median_delta_c", "q75": "q75_delta_c", "q90": "q90_delta_c"}
        )
        self.add_csv(
            "Fig1g",
            target[["policy", "griffiths", "q10_delta_c", "q25_delta_c", "median_delta_c", "q75_delta_c", "q90_delta_c", "records"]],
            inputs=["panel_materialization_support"],
            filter_used="g=0.50; China, Japan and Thailand pre-specified campaigns",
            units={"q10_delta_c": "degC", "q25_delta_c": "degC", "median_delta_c": "degC", "q75_delta_c": "degC", "q90_delta_c": "degC", "records": "count"},
            denominator_fields=["records"],
        )
        robust = pd.DataFrame(self.support["hierarchy_variance_robustness"]).rename(
            columns={"leave_one_source_min_share": "loo_min_share", "leave_one_source_median_share": "loo_median_share", "leave_one_source_max_share": "loo_max_share"}
        )
        robust["griffiths"] = robust["reference"].map({"g033": 0.33, "g050": 0.50, "g067": 0.67})
        self.add_csv(
            "Fig1h",
            robust[["reference", "weighting", "component", "loo_min_share", "loo_median_share", "loo_max_share", "griffiths", "n_leave_one_source_runs"]],
            inputs=["panel_materialization_support"],
            filter_used="all frozen g, weighting and leave-one-source summaries",
            units={"loo_min_share": "fraction", "loo_median_share": "fraction", "loo_max_share": "fraction", "griffiths": "dimensionless", "n_leave_one_source_runs": "count"},
            denominator_fields=["reference", "weighting", "n_leave_one_source_runs"],
        )

    def figure2(self) -> None:
        flow = pd.DataFrame(self.support["direct_vote_row_flow"])
        base = flow.loc[flow["dataset"].eq("Humans-as-a-Sensor")]
        lookup = dict(zip(base["stage"], base["n_rows"]))
        window = base.loc[base["stage"].eq("window_construction_30min")].iloc[0]
        rows = [
            ("raw rows", lookup["raw_download"], "downloaded observations"),
            ("declared cohort", lookup["declared_cresh01_to_cresh30"], "declared cohort"),
            ("valid votes", lookup["valid_time_user_room_vote_and_room_temperature"], "valid time, user, room, vote and temperature"),
            ("candidate windows", window["candidate_group_windows"], "30-minute windows"),
            ("minimum-user windows", window["windows_after_minimum_user_rule"], "at least three unique users"),
            ("temperature-homogeneous windows", window["eligible_windows_after_all_rules"], "temperature spread at most 0.5 degC"),
            ("person-window votes", window["eligible_person_window_votes"], "eligible votes"),
        ]
        self.add_csv(
            "Fig2a",
            pd.DataFrame([{"dataset": "NUS_HAAS", "window_minutes": 30, "stage": stage, "rows_or_windows": int(value), "exclusion_rule": rule} for stage, value, rule in rows]),
            inputs=["panel_materialization_support"],
            filter_used="Humans-as-a-Sensor 30-minute primary flow",
            units={"window_minutes": "minute", "rows_or_windows": "count"},
            denominator_fields=["stage"],
        )
        windows = self.table("result2_windows")
        nus = windows.loc[windows["dataset_code"].eq("NUS_HAAS") & windows["eligible"]].copy()
        nus = nus.rename(columns={"dataset_code": "dataset", "anonymous_window_id": "window_id", "vote_count": "n_votes", "prefer_cooler_count": "n_cooler", "no_change_count": "n_no_change", "prefer_warmer_count": "n_warmer"})
        for label in ("cooler", "no_change", "warmer"):
            nus[f"share_{label}"] = nus[f"n_{label}"] / nus["n_votes"]
        self.add_csv(
            "Fig2b",
            nus[["dataset", "window_id", "n_votes", "n_cooler", "n_no_change", "n_warmer", "share_cooler", "share_no_change", "share_warmer"]],
            inputs=["result2_windows"],
            filter_used="dataset=NUS_HAAS; eligible=true; one row per room-window",
            units={"n_votes": "count", "share_cooler": "fraction", "share_no_change": "fraction", "share_warmer": "fraction"},
            denominator_fields=["n_votes"],
        )
        opposed = nus.loc[nus["simultaneous_warmer_cooler"]].merge(
            pd.DataFrame(self.support["opposed_window_temperature_spread"]).rename(columns={"anonymous_window_id": "window_id"}),
            on="window_id",
            how="left",
            validate="one_to_one",
        ).rename(columns={"median_temperature_c": "temperature_mean_c"})
        if len(opposed) != 6 or opposed["temperature_spread_c"].isna().any():
            raise ValueError("Fig2c requires six opposed windows")
        self.add_csv(
            "Fig2c",
            opposed[["window_id", "n_votes", "n_cooler", "n_no_change", "n_warmer", "temperature_mean_c", "temperature_spread_c"]],
            inputs=["result2_windows", "panel_materialization_support"],
            filter_used="NUS eligible windows with cooler>0 and warmer>0",
            units={"n_votes": "count", "n_cooler": "count", "n_no_change": "count", "n_warmer": "count", "temperature_mean_c": "degC", "temperature_spread_c": "degC"},
            denominator_fields=["n_votes"],
        )
        self.add_csv(
            "Fig2d",
            nus[["window_id", "n_votes", "full_burden", "compressed_burden", "jensen_gap"]],
            inputs=["result2_windows"],
            filter_used="all 63 eligible NUS windows",
            units={"n_votes": "count", "full_burden": "unitless", "compressed_burden": "unitless", "jensen_gap": "unitless"},
            denominator_fields=["n_votes"],
        )
        cancellation = nus.loc[np.isclose(nus["compressed_burden"], 0) & nus["full_burden"].gt(0)]
        self.add_csv(
            "Fig2e",
            cancellation[["window_id", "full_burden", "compressed_burden", "n_votes", "n_cooler", "n_warmer"]],
            inputs=["result2_windows"],
            filter_used="compressed_burden=0 and full_burden>0",
            units={"full_burden": "unitless", "compressed_burden": "unitless", "n_votes": "count", "n_cooler": "count", "n_warmer": "count"},
            denominator_fields=["n_votes"],
        )
        thresholds = self.table("result2_thresholds")
        thresholds = thresholds.loc[thresholds["dataset_code"].eq("NUS_HAAS")].rename(columns={"dataset_code": "dataset", "n_windows": "eligible_windows", "false_accept": "false_accept_count", "false_reject": "false_reject_count"})
        self.add_csv(
            "Fig2f",
            thresholds[["dataset", "threshold", "false_accept_count", "false_reject_count", "eligible_windows"]],
            inputs=["result2_thresholds"],
            filter_used="NUS thresholds 0,0.25,0.50,0.75",
            units={"threshold": "unitless", "false_accept_count": "count", "false_reject_count": "count", "eligible_windows": "count"},
            denominator_fields=["eligible_windows"],
        )
        sensitivity = pd.DataFrame(self.support["direct_vote_window_sensitivity"])
        self.add_csv(
            "Fig2g",
            sensitivity,
            inputs=["panel_materialization_support"],
            filter_used="NUS 15/30/60-minute windows; crossed room-date Bayesian bootstrap",
            units={"window_minutes": "minute", "estimate": "fraction_or_unitless", "ci_low": "fraction_or_unitless", "ci_high": "fraction_or_unitless", "eligible_windows": "count", "votes": "count"},
            denominator_fields=["eligible_windows", "votes"],
        )
        summary = self.table("result2_summary").rename(columns={"disagreement_windows": "disagreement_count", "opposing_direction_windows": "opposing_count", "exact_cancellation_windows": "cancellation_count"})
        ws = pd.DataFrame(self.support["direct_vote_window_summary"])
        primary_bootstrap = pd.DataFrame(self.support["direct_vote_primary_bootstrap"])
        rate_map = {"disagreement_rate": "rate_direct_disagreement", "opposing_request_rate": "rate_opposite_direction_conflict", "mean_jensen_gap": "mean_jensen_gap"}
        bootstrap_metric = {"disagreement_rate": "direct_disagreement_rate", "opposing_request_rate": "opposite_direction_rate", "mean_jensen_gap": "mean_jensen_gap"}
        metric_rows = []
        for row in summary.itertuples(index=False):
            label = "Humans-as-a-Sensor" if row.dataset_code == "NUS_HAAS" else "ENTH"
            base_row = ws.loc[ws["dataset"].eq(label) & ws["window_minutes"].eq(30)].iloc[0]
            for metric, source_column in rate_map.items():
                ci = primary_bootstrap.loc[
                    primary_bootstrap["dataset"].eq(label)
                    & primary_bootstrap["metric"].eq(bootstrap_metric[metric])
                ]
                if len(ci) != 1:
                    raise ValueError(f"missing primary crossed-bootstrap CI: {label}/{metric}")
                metric_rows.append({"dataset": row.dataset_code, "eligible_windows": row.eligible_windows, "votes": row.votes, "disagreement_count": row.disagreement_count, "opposing_count": row.opposing_count, "cancellation_count": row.cancellation_count, "metric": metric, "estimate": base_row[source_column], "ci_low": ci["ci_2_5"].iloc[0], "ci_high": ci["ci_97_5"].iloc[0]})
        self.add_csv(
            "Fig2h",
            pd.DataFrame(metric_rows),
            inputs=["result2_summary", "panel_materialization_support"],
            filter_used="NUS and ENTH 30-minute windows; frozen crossed-cluster bootstrap CIs",
            units={"eligible_windows": "count", "votes": "count", "disagreement_count": "count", "opposing_count": "count", "cancellation_count": "count", "estimate": "fraction_or_unitless", "ci_low": "fraction_or_unitless", "ci_high": "fraction_or_unitless"},
            denominator_fields=["eligible_windows", "votes"],
        )

    def _definition_directions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        matches = self.asset("three_definition_matches")
        carbon = self.asset("regional_carbon_cost")
        keys = ["city", "seed", "allocation", "service_fraction", "strategy"]
        base = matches.loc[
            matches["interval_width_multiplier"].eq(1)
            & matches["strategy"].isin(["personal_center", "ltpref_interval"])
        ]
        rows = []
        for key, group in base.groupby(keys, sort=True):
            energy_lookup = dict(
                zip(group["griffiths_coefficient"], group["site_change_vs_uniform_pct"])
            )
            # The locked 130-pair cohort is defined by a finite default-g (0.50)
            # comparison; sensitivity settings may subsequently lose that match.
            if pd.isna(energy_lookup.get(0.5, np.nan)):
                continue
            energy = _direction([energy_lookup.get(v, np.nan) for v in (0.33, 0.5, 0.67)])
            cgroup = carbon.loc[
                carbon["interval_width_multiplier"].eq(1)
                & carbon["city"].eq(key[0])
                & carbon["seed"].eq(key[1])
                & carbon["allocation"].eq(key[2])
                & carbon["service_fraction"].eq(key[3])
                & carbon["strategy"].eq(key[4])
            ]
            co2_lookup = dict(
                zip(cgroup["griffiths_coefficient"], cgroup["operational_co2_change_pct"])
            )
            co2 = _direction([co2_lookup.get(v, np.nan) for v in (0.33, 0.5, 0.67)])
            rows.append(
                {
                    "pair_id": _stable_id(*key),
                    "context": key[0],
                    "seed": key[1],
                    "allocation": key[2],
                    "service_fraction": key[3],
                    "alternative_control": key[4],
                    "energy_direction": energy,
                    "co2_direction": co2,
                    "matched": energy != "match_lost",
                    "griffiths": "0.33|0.50|0.67",
                }
            )
        directions = pd.DataFrame(rows)
        width_base = matches.loc[
            matches["griffiths_coefficient"].eq(0.5)
            & matches["strategy"].isin(["personal_center", "ltpref_interval"])
        ]
        width_rows = []
        for key, group in width_base.groupby(keys, sort=True):
            lookup = dict(
                zip(group["interval_width_multiplier"], group["site_change_vs_uniform_pct"])
            )
            classification = _direction([lookup.get(v, np.nan) for v in (0.5, 1, 1.5)])
            width_rows.append(
                {**dict(zip(keys, key)), "classification": classification or "all_unmatched"}
            )
        return directions, pd.DataFrame(width_rows)

    def figure3(self) -> None:
        self.add_meta(
            "Fig3a",
            {
                "title": "Three control definitions",
                "items": [
                    {"control_definition": "uniform_average", "shared_constraints": "one shared demand representation"},
                    {"control_definition": "personal_center", "shared_constraints": "individual demand centres; identical building and permissions"},
                    {"control_definition": "ltpref_interval", "shared_constraints": "individual virtual intervals; identical building and permissions"},
                ],
            },
            inputs=[],
            filter_used="locked conceptual definitions",
            filename="Fig3a_meta.json",
        )
        assignments = self.asset("population_assignments")
        group_keys = ["scenario_id", "context", "composition_seed"]
        grouped = assignments.groupby(group_keys, sort=True)["demand_centre_c"]
        summary = grouped.agg(
            n_assignments="size", centre_c="mean", sd_c=lambda values: values.std(ddof=0)
        ).reset_index()
        quantiles = (
            grouped.quantile([0.1, 0.9])
            .unstack()
            .reset_index()
            .rename(columns={0.1: "p10_c", 0.9: "p90_c"})
        )
        summary = summary.merge(quantiles, on=group_keys, validate="one_to_one").rename(
            columns={"scenario_id": "population", "composition_seed": "seed"}
        )
        self.add_csv(
            "Fig3b",
            summary[["population", "context", "seed", "n_assignments", "centre_c", "p10_c", "p90_c", "sd_c"]],
            inputs=["population_assignments"],
            filter_used="all 36 frozen population-context-seed assignment groups",
            units={"n_assignments": "count", "centre_c": "degC", "p10_c": "degC", "p90_c": "degC", "sd_c": "degC"},
            denominator_fields=["n_assignments"],
        )
        self.add_meta(
            "Fig3c",
            {
                "title": "Orthogonal demand transformation",
                "formula": "T_prime_i,z(delta,lambda)=mu_z+delta+lambda*(T_i,z-mu_z)",
                "items": [{"delta_c": [-2, -1, 0, 1, 2], "lambda": [0, 0.5, 1, 1.5], "transformation_formula": "location and spread varied independently"}],
            },
            inputs=[],
            filter_used="pre-specified Gate-2 orthogonal grid",
            filename="Fig3c_meta.json",
        )
        centre = self.table("result3_centre_effects")
        common = pd.DataFrame(
            {
                "context": centre["city"],
                "population": centre["population_scenario"],
                "seed": centre["composition_seed"],
                "service_budget": centre["service_fraction"],
                "delta_c": centre["target_delta_c"],
                "reference_action_id": centre["full_selected_action_id_reference"],
                "target_action_id": centre["full_selected_action_id_target"],
                "delta_energy_mwh": centre["change_full_selected_site_kwh"] / 1000,
                "matched": centre["change_full_selected_site_kwh"].notna(),
            }
        )
        self.add_csv(
            "Fig3d",
            common,
            inputs=["result3_centre_effects"],
            filter_used="all non-zero delta effects relative to delta=0, lambda=1; unmatched retained",
            units={"service_budget": "fraction", "delta_c": "degC", "delta_energy_mwh": "MWh_year"},
            denominator_fields=["matched"],
        )
        co2 = common.drop(columns=["delta_energy_mwh"]).copy()
        co2["delta_co2_t"] = centre["change_full_selected_operational_co2_kg"] / 1000
        co2["carbon_factor_version"] = "frozen_regional_factor_contract_v1"
        self.add_csv(
            "Fig3e",
            co2,
            inputs=["result3_centre_effects", "regional_factor_contract"],
            filter_used="same bilateral cells as Fig3d; frozen regional carbon factors",
            units={"service_budget": "fraction", "delta_c": "degC", "delta_co2_t": "tCO2_year"},
            denominator_fields=["matched"],
        )
        selected = centre.loc[centre["target_delta_c"].eq(2)].copy()
        selected["matched"] = selected["change_full_selected_site_kwh"].notna()
        eligibility = self.table("result3_annual_eligibility")
        score_keys = ["population_scenario", "city", "composition_seed", "service_fraction"]
        carrier_columns = ["heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh", "gas_kwh_fuel"]
        target_scores = eligibility.loc[
            eligibility["delta_c"].eq(2)
            & eligibility["lambda"].eq(1)
            & eligibility["representation"].eq("full"),
            score_keys + ["physical_action_id"] + carrier_columns,
        ].rename(columns={"physical_action_id": "full_selected_action_id_target", **{column: f"{column}_target" for column in carrier_columns}})
        reference_scores = eligibility.loc[
            eligibility["delta_c"].eq(0)
            & eligibility["lambda"].eq(1)
            & eligibility["representation"].eq("full"),
            score_keys + ["physical_action_id"] + carrier_columns,
        ].rename(columns={"physical_action_id": "full_selected_action_id_reference", **{column: f"{column}_reference" for column in carrier_columns}})
        selected = selected.merge(
            target_scores,
            on=score_keys + ["full_selected_action_id_target"],
            how="left",
            validate="many_to_one",
        ).merge(
            reference_scores,
            on=score_keys + ["full_selected_action_id_reference"],
            how="left",
            validate="many_to_one",
        )
        selected["matched"] = (
            selected["change_full_selected_site_kwh"].notna()
            & selected["full_selected_action_id_target"].notna()
            & selected["full_selected_action_id_reference"].notna()
            & selected[[f"{column}_{side}" for column in carrier_columns for side in ("reference", "target")]].notna().all(axis=1)
        )
        carriers = pd.DataFrame(
            {
                "action_id": selected["full_selected_action_id_target"],
                "target_action_id": selected["full_selected_action_id_target"],
                "reference_action_id": selected["full_selected_action_id_reference"],
                "context": selected["city"],
                "population": selected["population_scenario"],
                "seed": selected["composition_seed"],
                "service_budget": selected["service_fraction"],
                "delta_c": 2.0,
                "heating_electricity_reference_mwh": selected["heating_electric_kwh_reference"] / 1000,
                "heating_electricity_target_mwh": selected["heating_electric_kwh_target"] / 1000,
                "heating_electricity_change_mwh_signed": (selected["heating_electric_kwh_target"] - selected["heating_electric_kwh_reference"]) / 1000,
                "cooling_electricity_reference_mwh": selected["cooling_electric_kwh_reference"] / 1000,
                "cooling_electricity_target_mwh": selected["cooling_electric_kwh_target"] / 1000,
                "cooling_electricity_change_mwh_signed": (selected["cooling_electric_kwh_target"] - selected["cooling_electric_kwh_reference"]) / 1000,
                "fan_electricity_reference_mwh": selected["fans_electric_kwh_reference"] / 1000,
                "fan_electricity_target_mwh": selected["fans_electric_kwh_target"] / 1000,
                "fan_electricity_change_mwh_signed": (selected["fans_electric_kwh_target"] - selected["fans_electric_kwh_reference"]) / 1000,
                "gas_hhv_reference_mwh": selected["gas_kwh_fuel_reference"] / 1000,
                "gas_hhv_target_mwh": selected["gas_kwh_fuel_target"] / 1000,
                "gas_hhv_change_mwh_signed": (selected["gas_kwh_fuel_target"] - selected["gas_kwh_fuel_reference"]) / 1000,
                "matched": selected["matched"],
            }
        )
        carriers["heating_electricity_mwh"] = carriers["heating_electricity_target_mwh"]
        carriers["cooling_electricity_mwh"] = carriers["cooling_electricity_target_mwh"]
        carriers["fan_electricity_mwh"] = carriers["fan_electricity_target_mwh"]
        carriers["gas_hhv_mwh"] = carriers["gas_hhv_target_mwh"]
        self.add_csv(
            "Fig3f",
            carriers,
            inputs=["result3_centre_effects", "result3_annual_eligibility"],
            filter_used="signed target-minus-reference carrier change; delta=+2 versus delta=0, lambda=1, full-selected actions; all 144 rows, unmatched retained",
            units={"service_budget": "fraction", "delta_c": "degC", **{column: ("MWh_HHV_year" if column.startswith("gas_hhv") else "MWh_year") for column in carriers.columns if "_mwh" in column}},
            denominator_fields=["matched"],
        )
        directions, width = self._definition_directions()
        if len(directions) != 130:
            raise ValueError(f"locked three-definition cohort changed: {len(directions)}")
        self.add_csv(
            "Fig3g",
            directions[["pair_id", "context", "alternative_control", "energy_direction", "co2_direction", "matched", "griffiths"]],
            inputs=["three_definition_matches", "regional_carbon_cost"],
            filter_used="width=1; personal/LTPref alternatives; at least one g match",
            units={"griffiths": "dimensionless"},
            denominator_fields=["matched"],
        )
        rows = []
        for sensitivity_type, frame, column, value in (
            ("griffiths_energy", directions, "energy_direction", "0.33|0.50|0.67"),
            ("griffiths_carbon", directions, "co2_direction", "0.33|0.50|0.67"),
            ("interval_width_energy", width, "classification", "0.5|1.0|1.5"),
        ):
            for classification, count in frame[column].fillna("all_unmatched").value_counts().items():
                rows.append({"metric": "direction_classification", "sensitivity_type": sensitivity_type, "sensitivity_value": value, "classification": classification, "count": int(count), "denominator": len(frame)})
        self.add_csv(
            "Fig3h",
            pd.DataFrame(rows),
            inputs=["three_definition_matches", "regional_carbon_cost"],
            filter_used="pre-specified g, interval-width and carbon direction classifications",
            units={"count": "count", "denominator": "count"},
            denominator_fields=["denominator"],
        )

    def figure4(self) -> None:
        self.add_meta(
            "Fig4a",
            {"title": "Mean versus full service", "items": [{"mean_service_definition": "score one compressed mean demand", "full_service_definition": "score complete assigned demand distribution", "common_action_set": "same frozen physical actions"}]},
            inputs=[],
            filter_used="conceptual service contract",
            filename="Fig4a_meta.json",
        )
        assignments = self.asset("population_assignments")
        illustration = assignments.loc[
            assignments["scenario_id"].eq("multinational_source_balanced")
            & assignments["context"].eq("Miami")
            & assignments["composition_seed"].eq(2026102002)
        ]
        centre = illustration["demand_centre_c"].mean()
        rows = []
        for lam in (0, 0.5, 1, 1.5):
            for row in illustration.itertuples(index=False):
                rows.append({"reference_demand_c": row.demand_centre_c, "delta_c": 0.0, "lambda": lam, "perturbed_demand_c": centre + lam * (row.demand_centre_c - centre), "population": row.scenario_id, "context": row.context, "seed": row.composition_seed})
        self.add_csv(
            "Fig4b",
            pd.DataFrame(rows),
            inputs=["population_assignments"],
            filter_used="multinational_source_balanced, Miami, seed 2026102002, delta=0",
            units={"reference_demand_c": "degC", "delta_c": "degC", "lambda": "dimensionless", "perturbed_demand_c": "degC"},
            denominator_fields=["population", "context", "seed"],
        )
        decisions = self.table("result3_annual_decisions")
        subset = decisions.loc[
            decisions["delta_c"].eq(0) & decisions["lambda"].isin([0, 0.5, 1, 1.5])
        ]
        counts, false = [], []
        for lam, group in subset.groupby("lambda", sort=True):
            full_selected = group["full_selected_action_id"].notna()
            mean_selected = group["mean_selected_action_id"].notna()
            counts += [
                {"lambda": lam, "representation": "full", "selected_count": int(full_selected.sum()), "total_cells": len(group)},
                {"lambda": lam, "representation": "mean", "selected_count": int(mean_selected.sum()), "total_cells": len(group)},
            ]
            fa = mean_selected & ~group["mean_selected_full_eligible"].eq(True)
            false.append({"lambda": lam, "mean_selected_count": int(mean_selected.sum()), "false_accept_count": int(fa.sum()), "false_accept_rate": float(fa.sum() / mean_selected.sum())})
        self.add_csv(
            "Fig4c", pd.DataFrame(counts), inputs=["result3_annual_decisions"],
            filter_used="delta=0; lambda=0,0.5,1,1.5; exact selection counts",
            units={"lambda": "dimensionless", "selected_count": "count", "total_cells": "count"}, denominator_fields=["total_cells"])
        self.add_csv(
            "Fig4d", pd.DataFrame(false), inputs=["result3_annual_decisions"],
            filter_used="delta=0; mean-selected actions re-audited under full demand",
            units={"lambda": "dimensionless", "mean_selected_count": "count", "false_accept_count": "count", "false_accept_rate": "fraction"}, denominator_fields=["mean_selected_count"])
        action_counts = pd.DataFrame({
            "decision_id": [_stable_id(r.population_scenario, r.city, r.composition_seed, r.service_fraction, r.lambda_) for r in subset.rename(columns={"lambda": "lambda_"}).itertuples(index=False)],
            "lambda": subset["lambda"].to_numpy(), "full_eligible_action_count": subset["full_eligible_action_count"].to_numpy(), "mean_eligible_action_count": subset["mean_eligible_action_count"].to_numpy(),
            "full_selected": subset["full_selected_action_id"].notna().to_numpy(), "context": subset["city"].to_numpy(), "population": subset["population_scenario"].to_numpy(), "seed": subset["composition_seed"].to_numpy(), "service_budget": subset["service_fraction"].to_numpy()})
        self.add_csv(
            "Fig4e", action_counts, inputs=["result3_annual_decisions"],
            filter_used="all 576 delta=0 decision cells; zero action counts retained",
            units={"lambda": "dimensionless", "full_eligible_action_count": "count", "mean_eligible_action_count": "count", "service_budget": "fraction"}, denominator_fields=["decision_id"])
        spread = self.table("result4_spread_effects")
        energy = pd.DataFrame({
            "context": spread["city"], "population": spread["population_scenario"], "seed": spread["composition_seed"], "service_budget": spread["service_fraction"], "lambda": spread["target_lambda"], "reference_lambda": spread["lambda_reference"],
            "delta_energy_mwh": spread["change_full_selected_site_kwh"] / 1000, "bilateral_matched": spread["change_full_selected_site_kwh"].notna()})
        self.add_csv(
            "Fig4f", energy, inputs=["result4_spread_effects"],
            filter_used="delta=0 spread effects relative to lambda=1; unmatched retained",
            units={"service_budget": "fraction", "lambda": "dimensionless", "reference_lambda": "dimensionless", "delta_energy_mwh": "MWh_year"}, denominator_fields=["bilateral_matched"])
        eligibility = self.table("result3_annual_eligibility")
        score_keys = ["population_scenario", "city", "composition_seed", "service_fraction", "delta_c", "lambda", "physical_action_id"]
        carrier_columns = ["heating_electric_kwh", "cooling_electric_kwh", "fans_electric_kwh", "gas_kwh_fuel"]
        score_cols = score_keys + carrier_columns
        scores = eligibility.loc[eligibility["representation"].eq("full"), score_cols].drop_duplicates(score_keys)
        target_scores = scores.rename(columns={
            "delta_c": "delta_c_target", "lambda": "lambda_target",
            "physical_action_id": "full_selected_action_id_target",
            **{column: f"{column}_target" for column in carrier_columns},
        })
        reference_scores = scores.loc[
            scores["delta_c"].eq(0) & scores["lambda"].eq(1)
        ].rename(columns={
            "delta_c": "delta_c_reference", "lambda": "lambda_reference",
            "physical_action_id": "full_selected_action_id_reference",
            **{column: f"{column}_reference" for column in carrier_columns},
        })
        reference_scores["lambda_reference"] = reference_scores[
            "lambda_reference"
        ].astype("int64")
        target_keys = ["population_scenario", "city", "composition_seed", "service_fraction", "delta_c_target", "lambda_target", "full_selected_action_id_target"]
        reference_keys = ["population_scenario", "city", "composition_seed", "service_fraction", "delta_c_reference", "lambda_reference", "full_selected_action_id_reference"]
        joined = spread.merge(target_scores, on=target_keys, how="left", validate="many_to_one")
        joined = joined.merge(reference_scores, on=reference_keys, how="left", validate="many_to_one")
        carrier = pd.DataFrame({
            "context": spread["city"], "lambda": spread["target_lambda"], "delta_co2_t": spread["change_full_selected_operational_co2_kg"] / 1000,
            "heating_electricity_delta_mwh": (joined["heating_electric_kwh_target"] - joined["heating_electric_kwh_reference"]) / 1000,
            "cooling_electricity_delta_mwh": (joined["cooling_electric_kwh_target"] - joined["cooling_electric_kwh_reference"]) / 1000,
            "fan_delta_mwh": (joined["fans_electric_kwh_target"] - joined["fans_electric_kwh_reference"]) / 1000,
            "gas_hhv_delta_mwh": (joined["gas_kwh_fuel_target"] - joined["gas_kwh_fuel_reference"]) / 1000,
            "bilateral_matched": spread["change_full_selected_site_kwh"].notna().to_numpy()})
        self.add_csv(
            "Fig4g", carrier, inputs=["result4_spread_effects", "result3_annual_eligibility"],
            filter_used="same bilateral cells as Fig4f; selected target-minus-reference carrier outcomes",
            units={"lambda": "dimensionless", "delta_co2_t": "tCO2_year", "heating_electricity_delta_mwh": "MWh_year", "cooling_electricity_delta_mwh": "MWh_year", "fan_delta_mwh": "MWh_year", "gas_hhv_delta_mwh": "MWh_HHV_year"}, denominator_fields=["bilateral_matched"])
        interaction = self.table("result4_interactions").rename(columns={"population_scenario": "population", "city": "context", "composition_seed": "seed", "service_fraction": "service_budget", "interaction_full_eligible_action_count": "eligible_count_interaction"})
        interaction["energy_interaction_mwh"] = interaction["interaction_full_selected_site_kwh"] / 1000
        interaction["energy_interaction_resolved"] = interaction["interaction_full_selected_site_kwh"].notna()
        self.add_csv(
            "Fig4h", interaction[["context", "population", "seed", "service_budget", "delta_c", "lambda", "eligible_count_interaction", "energy_interaction_mwh", "energy_interaction_resolved"]],
            inputs=["result4_interactions"], filter_used="all 1,728 centre-by-spread interactions; unresolved retained",
            units={"service_budget": "fraction", "delta_c": "degC", "lambda": "dimensionless", "eligible_count_interaction": "count", "energy_interaction_mwh": "MWh_year"}, denominator_fields=["energy_interaction_resolved"])

    def figure5(self) -> None:
        self.add_meta(
            "Fig5a",
            {"title": "Event-response accounting protocol", "items": [{"event_hours": 2, "recovery_hours": 48, "service_checks": "event and recovery mean plus fixed-tail service", "energy_accounting_window": "50 hours"}]},
            inputs=[], filter_used="locked event protocol", filename="Fig5a_meta.json")
        decisions = self.table("result5_dynamic_decisions")
        rule = {
            "population_scenario": "multinational_source_balanced", "city": "Miami",
            "composition_seed": 2026102002, "event_month": 7, "delta_c": 0.0,
            "lambda": 1.0, "service_fraction": 0.9, "representation": "full",
            "selection_rule": "same mean/full-selected action; central pooled composition, median seed, warm-climate July case, locked 0.9 service fraction; never ranked by effect magnitude",
        }
        chosen = decisions.loc[
            decisions["population_scenario"].eq(rule["population_scenario"])
            & decisions["city"].eq(rule["city"])
            & decisions["composition_seed"].eq(rule["composition_seed"])
            & decisions["event_month"].eq(rule["event_month"])
            & decisions["delta_c"].eq(rule["delta_c"])
            & decisions["lambda"].eq(rule["lambda"])
            & decisions["service_fraction"].eq(rule["service_fraction"])
        ]
        if len(chosen) != 1 or pd.isna(chosen.iloc[0]["full_selected_action_id"]):
            raise ValueError("pre-specified Fig5b/c case does not resolve")
        decision = chosen.iloc[0]
        action_id = decision["full_selected_action_id"]
        if pd.isna(decision["mean_selected_action_id"]) or decision["mean_selected_action_id"] != action_id:
            raise ValueError("Fig5b/c lock requires the same mean/full-selected action")
        temperature = self.asset("event_temperature_service_curves")
        case_temperature = temperature.loc[
            temperature["event_action_id"].eq(action_id)
            & temperature["event_month"].eq(rule["event_month"])
        ].copy()
        power = self.asset("event_power_curves")
        case_power = power.loc[
            power["city"].eq(rule["city"])
            & power["seed"].eq(rule["composition_seed"])
            & power["allocation"].eq("random_mixing")
            & power["strategy"].eq("ltpref_interval")
            & power["amplitude_c"].eq(1)
            & power["event_month"].eq(rule["event_month"])
        ].copy()
        if len(case_temperature) != 100 or len(case_power) != 50:
            raise ValueError("Fig5b/c requires 50 hours x 2 states")
        hash_columns = ["trajectory_input_sha256", "trajectory_output_sha256", "profile_input_sha256"]
        state_hashes: dict[str, dict[str, str]] = {}
        for state, group in case_temperature.groupby("trajectory_state", sort=True):
            state_hashes[state] = {}
            for column in hash_columns:
                values = group[column].dropna().unique()
                if len(values) != 1:
                    raise ValueError(f"Fig5b/c requires one {column} per trajectory state")
                state_hashes[state][column] = values[0]
        if set(state_hashes) != {"baseline", "perturbed"}:
            raise ValueError("Fig5b/c requires baseline and perturbed trajectory hashes")
        rows = []
        for row in case_power.itertuples(index=False):
            for state, value in (("baseline", row.baseline_hvac_kw), ("perturbed", row.event_hvac_kw)):
                rows.append({
                    "event_action_id": action_id, "action_id": action_id, "selection_basis": rule["selection_rule"],
                    "state": state, "trajectory_state": state,
                    "hour": row.hour_from_event_start, "hvac_electric_power_kw": value,
                    "event_flag": row.hour_from_event_start < 2, "recovery_flag": row.hour_from_event_start >= 2,
                    "context": rule["city"], "population": rule["population_scenario"], "seed": rule["composition_seed"],
                    "month": rule["event_month"], "delta_c": rule["delta_c"], "lambda": rule["lambda"],
                    "service_budget": rule["service_fraction"], "service_fraction": rule["service_fraction"],
                    **state_hashes[state],
                })
        self.add_csv(
            "Fig5b", pd.DataFrame(rows),
            inputs=["result5_dynamic_decisions", "event_power_curves", "event_temperature_service_curves"],
            filter_used=json.dumps(rule, sort_keys=True),
            units={"hour": "hour_from_event_start", "hvac_electric_power_kw": "kW", "month": "month", "delta_c": "degC", "lambda": "dimensionless", "service_budget": "fraction"},
            denominator_fields=["action_id", "trajectory_state", "service_fraction"], filename="Fig5b_power_trace.csv")
        service = case_temperature.copy()
        service = service.rename(columns={
            "hour_from_event_start": "hour", "occupied_service_hour": "occupied",
            "zone_min_operative_temperature_c": "zone_min_c", "zone_arithmetic_mean_operative_temperature_c": "zone_mean_c", "zone_max_operative_temperature_c": "zone_max_c",
            "occupancy_weighted_mean_operative_temperature_c": "occupancy_weighted_temp_c", "full_distribution_mean_service_c": "full_mean_service",
            "fixed_27_tail_service_c": "fixed27_tail_service"})
        service["state"] = service["trajectory_state"]
        service["action_id"] = action_id
        service["context"] = rule["city"]
        service["population"] = rule["population_scenario"]
        service["seed"] = rule["composition_seed"]
        service["month"] = rule["event_month"]
        service["delta_c"] = rule["delta_c"]
        service["lambda"] = rule["lambda"]
        service["service_budget"] = rule["service_fraction"]
        service["service_fraction"] = rule["service_fraction"]
        service["mean_service_budget_c"] = decision["event_mean_budget_c"]
        service["tail_service_budget_c"] = decision["event_fixed_tail_budget_c"]
        self.add_csv(
            "Fig5c", service[["event_action_id", "action_id", "context", "population", "seed", "month", "delta_c", "lambda", "service_budget", "service_fraction", "state", "trajectory_state", "hour", "occupied", "zone_min_c", "zone_mean_c", "zone_max_c", "occupancy_weighted_temp_c", "full_mean_service", "fixed27_tail_service", "mean_service_budget_c", "tail_service_budget_c", "trajectory_input_sha256", "trajectory_output_sha256", "profile_input_sha256"]],
            inputs=["result5_dynamic_decisions", "event_temperature_service_curves", "event_zone_temperature_curves"],
            filter_used=f"same frozen action baseline-versus-perturbed physical trajectory; event_action_id={action_id}; month=7; hours 0-49",
            units={"hour": "hour_from_event_start", "zone_min_c": "degC", "zone_mean_c": "degC", "zone_max_c": "degC", "occupancy_weighted_temp_c": "degC", "full_mean_service": "degC_service_deviation", "fixed27_tail_service": "degC_service_deviation", "service_budget": "fraction", "mean_service_budget_c": "degC_service_deviation", "tail_service_budget_c": "degC_service_deviation"},
            denominator_fields=["action_id", "trajectory_state", "service_fraction"], filename="Fig5c_temperature_service_trace.csv")
        write_json(self.panel_root / "Fig5bc_case_lock.json", {
            **rule, "event_action_id": action_id, "mean_selected_action_id": decision["mean_selected_action_id"],
            "full_selected_action_id": decision["full_selected_action_id"], "mean_full_same_action": True,
            "panel_claim_scope": "same frozen selected action: baseline-versus-perturbed physical trajectory for event and recovery service",
            "mean_full_collection_difference_panels": ["Fig5d", "Fig5e", "Fig5f"],
            "trajectory_input_hashes": sorted(case_temperature["trajectory_input_sha256"].dropna().unique()),
            "trajectory_output_hashes": sorted(case_temperature["trajectory_output_sha256"].dropna().unique()),
            "profile_input_hashes": sorted(case_temperature["profile_input_sha256"].dropna().unique()),
            "event_power_rows": len(case_power), "temperature_service_rows": len(case_temperature), "new_energyplus_runs": 0})

        central = decisions.loc[
            decisions["delta_c"].eq(0) & decisions["event_month"].eq(7)
            & decisions["lambda"].isin([0, 0.5, 1, 1.5])]
        counts, false = [], []
        for lam, group in central.groupby("lambda", sort=True):
            full = group["full_selection_state"].eq("positive_action_selected")
            mean = group["mean_selection_state"].eq("positive_action_selected")
            counts += [
                {"month": 7, "lambda": lam, "representation": "full", "positive_selected_count": int(full.sum()), "total_cells": len(group)},
                {"month": 7, "lambda": lam, "representation": "mean", "positive_selected_count": int(mean.sum()), "total_cells": len(group)}]
            fa = mean & ~group["mean_selected_full_eligible"].eq(True)
            false.append({"month": 7, "lambda": lam, "mean_positive_count": int(mean.sum()), "false_accept_count": int(fa.sum()), "false_accept_rate": float(fa.sum() / mean.sum())})
        self.add_csv(
            "Fig5d", pd.DataFrame(counts), inputs=["result5_dynamic_decisions"],
            filter_used="July, delta=0, lambda 0/0.5/1/1.5; positive selected response",
            units={"month": "month", "lambda": "dimensionless", "positive_selected_count": "count", "total_cells": "count"}, denominator_fields=["total_cells"])
        self.add_csv(
            "Fig5e", pd.DataFrame(false), inputs=["result5_dynamic_decisions"],
            filter_used="July mean-positive selections re-audited under full demand",
            units={"month": "month", "lambda": "dimensionless", "mean_positive_count": "count", "false_accept_count": "count", "false_accept_rate": "fraction"}, denominator_fields=["mean_positive_count"])
        spread = self.table("result5_spread_effects")
        common = spread.loc[
            spread["event_month"].eq(7) & spread["delta_c_target"].eq(0)
            & spread["target_lambda"].eq(1.5)
            & spread["full_selection_state_target"].eq("positive_action_selected")
            & spread["full_selection_state_reference"].eq("positive_action_selected")].copy()
        if len(common) != 12:
            raise ValueError(f"Fig5f/g require 12 common selections, got {len(common)}")
        common["decision_id"] = [_stable_id(r.population_scenario, r.city, r.composition_seed, r.service_fraction, "July_common") for r in common.itertuples(index=False)]
        long_rows = []
        for row in common.itertuples(index=False):
            for lam, response in ((row.lambda_reference, row.full_selected_event_average_reduction_kw_reference), (row.lambda_target, row.full_selected_event_average_reduction_kw_target)):
                long_rows.append({"decision_id": row.decision_id, "context": row.city, "population": row.population_scenario, "seed": row.composition_seed, "service_budget": row.service_fraction, "lambda": lam, "selected_response_kw": response, "bilateral_matched": True})
        self.add_csv(
            "Fig5f", pd.DataFrame(long_rows), inputs=["result5_spread_effects"],
            filter_used="July delta=0; lambda 1 and 1.5 both retain positive full-service selection",
            units={"service_budget": "fraction", "lambda": "dimensionless", "selected_response_kw": "kW"}, denominator_fields=["decision_id", "bilateral_matched"])
        consequences = pd.DataFrame({
            "decision_id": common["decision_id"], "delta_response_kw": common["change_full_selected_event_average_reduction_kw"],
            "delta_recovery_peak_kw": common["change_full_selected_recovery_peak_hour_extra_kw"],
            "delta_recovery_extra_kwh": common["change_full_selected_recovery_net_extra_electric_kwh"],
            "delta_50h_net_saving_kwh": common["change_full_selected_event_plus48h_net_electric_saving_kwh"]})
        self.add_csv(
            "Fig5g", consequences, inputs=["result5_spread_effects"],
            filter_used="same 12 selections; lambda1.5 minus lambda1",
            units={"delta_response_kw": "kW", "delta_recovery_peak_kw": "kW", "delta_recovery_extra_kwh": "kWh", "delta_50h_net_saving_kwh": "kWh"}, denominator_fields=["decision_id"])
        grid_rows = []
        for (month, delta, lam), group in decisions.groupby(["event_month", "delta_c", "lambda"], sort=True):
            month_group = decisions.loc[decisions["event_month"].eq(month)]
            reference = month_group.loc[month_group["delta_c"].eq(0) & month_group["lambda"].eq(1)]
            grid_rows.append({
                "month": month, "delta_c": delta, "lambda": lam,
                "full_positive_selected_count": int(group["full_selection_state"].eq("positive_action_selected").sum()), "total_cells": len(group),
                "reference_flag": bool(delta == 0 and lam == 1),
                "reference_positive_selected_count": int(reference["full_selection_state"].eq("positive_action_selected").sum()), "reference_total_cells": len(reference),
                "month_grid_positive_selected_count": int(month_group["full_selection_state"].eq("positive_action_selected").sum()), "month_grid_total_cells": len(month_group)})
        grid = pd.DataFrame(grid_rows)
        feb = grid.loc[grid["month"].eq(2)]
        if not (set(feb["reference_positive_selected_count"]) == {0} and set(feb["reference_total_cells"]) == {144} and set(feb["month_grid_positive_selected_count"]) == {210} and set(feb["month_grid_total_cells"]) == {2880}):
            raise ValueError("Fig5h February denominators changed")
        self.add_csv(
            "Fig5h", grid, inputs=["result5_dynamic_decisions"],
            filter_used="complete month-delta-lambda grid; reference=delta0/lambda1",
            units={"month": "month", "delta_c": "degC", "lambda": "dimensionless", "full_positive_selected_count": "count", "total_cells": "count", "reference_positive_selected_count": "count", "reference_total_cells": "count", "month_grid_positive_selected_count": "count", "month_grid_total_cells": "count"},
            denominator_fields=["total_cells", "reference_total_cells", "month_grid_total_cells"])

    def figure6(self) -> None:
        self.add_meta(
            "Fig6a", {"title": "When is an average sufficient?", "items": [{"heterogeneity_sd_c": "increasing", "service_fraction": "increasing", "representation_question": "mean-safe, distribution-required, or unresolved"}]},
            inputs=[], filter_used="conceptual axes only; no numerical boundary", filename="Fig6a_meta.json")
        phase = self.table("result6_phase_map").rename(columns={"population_scenario": "population", "city": "context"})
        panel = phase[["context", "population", "seed", "lambda", "service_fraction", "heterogeneity_sd_c", "classification"]].copy()
        panel["service_budget"] = panel["service_fraction"]
        self.add_csv(
            "Fig6b", panel[["context", "population", "seed", "lambda", "service_budget", "heterogeneity_sd_c", "service_fraction", "classification"]],
            inputs=["result6_phase_map"], filter_used="all 1,008 frozen phase-map units",
            units={"lambda": "dimensionless", "service_budget": "fraction", "heterogeneity_sd_c": "degC", "service_fraction": "fraction"}, denominator_fields=["context", "population", "seed"])
        counts = phase["classification"].value_counts().rename_axis("classification").reset_index(name="count")
        counts["total_design_units"] = len(phase)
        counts["resolved_design_units"] = int((~phase["classification"].eq("finite-library-unresolved")).sum())
        self.add_csv(
            "Fig6c", counts, inputs=["result6_phase_map"], filter_used="exact phase-map class counts",
            units={"count": "count", "total_design_units": "count", "resolved_design_units": "count"}, denominator_fields=["total_design_units", "resolved_design_units"])
        threshold = 1.05868854309524
        resolved = phase.loc[~phase["classification"].eq("finite-library-unresolved")].copy()
        resolved["predicted_mean_safe"] = resolved["heterogeneity_sd_c"].le(threshold)
        resolved["actual_mean_safe"] = resolved["classification"].eq("mean-safe")
        threshold_rows = []
        for context, group in [("overall", resolved), *list(resolved.groupby("context", sort=True))]:
            false_safe = group["predicted_mean_safe"] & ~group["actual_mean_safe"]
            false_required = ~group["predicted_mean_safe"] & group["actual_mean_safe"]
            errors = int((false_safe | false_required).sum())
            threshold_rows.append({"threshold_sd_c": threshold, "context": context, "errors": errors, "denominator": len(group), "error_rate": errors / len(group), "false_safe": int(false_safe.sum()), "false_required": int(false_required.sum())})
        threshold_frame = pd.DataFrame(threshold_rows)
        if not np.isclose(threshold_frame.loc[threshold_frame.context.eq("overall"), "error_rate"].iloc[0], 0.15181518151815182) or not np.isclose(threshold_frame.loc[threshold_frame.context.ne("overall"), "error_rate"].max(), 0.20441988950276244):
            raise ValueError("Fig6d locked threshold errors changed")
        self.add_csv(
            "Fig6d", threshold_frame, inputs=["result6_phase_map"],
            filter_used="resolved only; predict mean-safe when SD<=1.05868854309524 degC",
            units={"threshold_sd_c": "degC", "errors": "count", "denominator": "count", "error_rate": "fraction", "false_safe": "count", "false_required": "count"}, denominator_fields=["denominator"])
        write_json(self.panel_root / "Fig6d_rule_lock.json", {
            "threshold_sd_c": threshold, "prediction_rule": "mean-safe when heterogeneity_sd_c <= threshold",
            "evaluation_scope": "606 resolved design decisions only", "overall_error_rate": 0.15181518151815182,
            "worst_context_error_rate": 0.20441988950276244, "accepted_as_universal_safe_boundary": False})
        approximation = self.table("result6_approximation")
        approximation["decision_id"] = [_stable_id(r.population_scenario, r.city, r.seed, r.lambda_, r.service_fraction) for r in approximation.rename(columns={"lambda": "lambda_"}).itertuples(index=False)]
        approximation["split"] = "all_frozen_design_cells"
        approximation = approximation.rename(columns={"city": "context", "eligibility_disagreement_rate": "eligibility_disagreement"})
        self.add_csv(
            "Fig6e", approximation[["decision_id", "representation", "candidate_tail_mae_c", "eligibility_disagreement", "context", "split"]],
            inputs=["result6_approximation"], filter_used="all 4,032 representation evaluations; full retained as benchmark",
            units={"candidate_tail_mae_c": "degC", "eligibility_disagreement": "fraction"}, denominator_fields=["decision_id"])
        gate = self.table("result6_quantile_gate")
        gate_panel = pd.DataFrame({
            "representation": gate["representation"], "margin_rule": gate["strategy"], "split_scheme": gate["split_scheme"],
            "candidate_rows": gate["candidate_rows"], "certified_candidates": gate["certified_candidates"],
            "full_eligible_candidates": gate["full_eligible_candidates"], "true_positive_candidates": gate["true_positive_candidates"],
            "false_accept_candidates": gate["false_accept_candidates"], "false_accept_selections": gate["selected_false_accept_decisions"],
            "decision_rows": gate["decision_rows"], "full_available_decisions": gate["full_available_decisions"],
            "certified_available_decisions": gate["certified_available_decisions"],
            "full_available_and_certified_decisions": gate["full_available_decisions"] - gate["unresolved_despite_full_available"],
            "unresolved_despite_full_available": gate["unresolved_despite_full_available"],
            "true_action_retention": gate["retained_full_eligible_fraction"], "decision_coverage": gate["certified_coverage_of_full_available"], "train_only": ~gate["strategy"].eq("none")})
        self.add_csv(
            "Fig6f", gate_panel, inputs=["result6_quantile_gate"], filter_used="all 36 frozen representation-margin-split evaluations",
            units={"candidate_rows": "count", "certified_candidates": "count", "full_eligible_candidates": "count", "true_positive_candidates": "count", "false_accept_candidates": "count", "false_accept_selections": "count", "decision_rows": "count", "full_available_decisions": "count", "certified_available_decisions": "count", "full_available_and_certified_decisions": "count", "unresolved_despite_full_available": "count", "true_action_retention": "fraction", "decision_coverage": "fraction"},
            denominator_fields=["candidate_rows", "certified_candidates", "full_eligible_candidates", "decision_rows", "full_available_decisions", "certified_available_decisions"])
        local = self.table("result6_local_update")
        local_rows = []
        for row in local.itertuples(index=False):
            for metric, estimate, low, high in (("CRPS", row.updated_crps, row.updated_crps_ci_low, row.updated_crps_ci_high), ("tail_MAE", row.updated_tail_mae, row.updated_tail_mae_ci_low, row.updated_tail_mae_ci_high)):
                local_rows.append({"transfer_split": row.split, "feedback_n": row.feedback_n, "metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high, "replicates": row.outer_groups * row.repeated_draws_per_group})
        self.add_csv(
            "Fig6g", pd.DataFrame(local_rows), inputs=["result6_local_update", "result6_local_update_replicates"],
            filter_used="city/source; feedback 0,5,10,20,40; frozen replicate intervals",
            units={"feedback_n": "count", "estimate": "score_or_degC", "ci_low": "score_or_degC", "ci_high": "score_or_degC", "replicates": "count"}, denominator_fields=["replicates"])
        self.add_meta(
            "Fig6h", {"title": "Service-qualified reporting workflow", "steps": [
                {"workflow_step": 1, "information_retained": "archive prior", "privacy_unit": "anonymous aggregate", "reported_metrics": "demand location and spread"},
                {"workflow_step": 2, "information_retained": "anonymous local update", "privacy_unit": "room/building aggregate", "reported_metrics": "update uncertainty"},
                {"workflow_step": 3, "information_retained": "five demand quantiles", "privacy_unit": "anonymous quantiles", "reported_metrics": "mean and tail service"},
                {"workflow_step": 4, "information_retained": "training-only protection margin", "privacy_unit": "no named profile", "reported_metrics": "service-qualified action set"},
                {"workflow_step": 5, "information_retained": "action and recovery contract", "privacy_unit": "building decision", "reported_metrics": "energy, carbon, flexibility, recovery"}]},
            inputs=[], filter_used="constructive workflow from frozen results; no new efficacy claim", filename="Fig6h_meta.json")

    def finalize(self) -> Path:
        if set(self.panels) != set(PANEL_IDS):
            raise ValueError(f"panel outputs incomplete: {sorted(set(PANEL_IDS)-set(self.panels))}")
        for source_key in sorted(
            {key for keys in self.panel_inputs.values() for key in keys}
        ):
            if source_key in self.input_record_keys:
                continue
            if source_key in self.config.get("assets", {}):
                self.asset(source_key)
            else:
                self.table(source_key)
        records, semantic_hashes = [], {}
        for panel_id in PANEL_IDS:
            item = self.panels[panel_id]
            path = item["path"]
            if path.suffix == ".csv":
                frame = pd.read_csv(path, low_memory=False)
                payload = frame.drop(columns=["panel_id"]).to_csv(
                    index=False, lineterminator="\n", float_format="%.12g"
                ).encode()
            else:
                value = json.loads(path.read_text(encoding="utf-8"))
                value.pop("panel_id", None)
                payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            semantic_hash = hashlib.sha256(payload).hexdigest()
            semantic_hashes[panel_id] = semantic_hash
            records.append({
                "panel_id": panel_id, "relative_path": path.relative_to(self.panel_root).as_posix(),
                "file_type": path.suffix.lstrip("."), "rows": item["rows"], "columns": len(item["ordered_fields"]),
                "ordered_fields": "|".join(item["ordered_fields"]), "units_json": json.dumps(self.units[panel_id], sort_keys=True),
                "denominator_fields": "|".join(self.denominators[panel_id]), "filter_used": self.filters[panel_id],
                "input_keys": "|".join(self.panel_inputs[panel_id]), "sha256": sha256(path), "size_bytes": path.stat().st_size,
                "semantic_sha256_without_panel_id": semantic_hash, "transform_sha256": self.transform_hash})
        if len(set(semantic_hashes.values())) != 48:
            collisions: dict[str, list[str]] = {}
            for panel_id, value in semantic_hashes.items():
                collisions.setdefault(value, []).append(panel_id)
            raise ValueError(f"semantic collisions: {[v for v in collisions.values() if len(v)>1]}")
        manifest = write_table(
            pd.DataFrame(records), self.panel_root / "PANEL_DATA_V2_MANIFEST.csv", sort_by=["panel_id"])
        try:
            git_commit = subprocess.check_output(
                ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        runtime = {"python": sys.version.split()[0], "platform": platform.platform(), "pandas": pd.__version__, "numpy": np.__version__}
        transform = {
            "materializer_script_sha256": self.transform_hash, "git_commit": git_commit, "runtime": runtime,
            "inputs": sorted(self.input_records.values(), key=lambda row: (row["path"], row["sha256"])), "filters": self.filters}
        write_json(self.panel_root / "PANEL_DATA_V2_TRANSFORM_MANIFEST.json", transform)
        groups = {
            "M1": [f"Fig1{x}" for x in "abcdefgh"], "M2": [f"Fig2{x}" for x in "abcdefgh"],
            "M3": [f"Fig3{x}" for x in "abcdefgh"] + [f"Fig4{x}" for x in "abcdefgh"],
            "M4": ["Fig5b", "Fig5c"], "M5": ["Fig6d"]}
        by_id = {row["panel_id"]: row for row in records}
        for group, panel_ids in groups.items():
            input_keys = sorted(
                {key for panel in panel_ids for key in self.panel_inputs[panel]}
            )
            write_json(self.panel_root / f"{group}_INPUT_OUTPUT_MANIFEST.json", {
                "materialization_group": group, "panels": panel_ids,
                "input_keys": input_keys,
                "inputs": sorted(
                    [
                        self.input_records[self.input_record_keys[key]]
                        for key in input_keys
                    ],
                    key=lambda row: (row["path"], row["sha256"]),
                ),
                "outputs": [by_id[panel] for panel in panel_ids],
                "transform_sha256": self.transform_hash, "filters": {panel: self.filters[panel] for panel in panel_ids}, "runtime": runtime})
        write_json(self.panel_root / "PANEL_DATA_V2_QA.json", {
            "status": "PASS", "panel_count": 48, "csv_panels": 42, "schematic_meta_panels": 6,
            "semantic_hashes_unique": len(set(semantic_hashes.values())), "schematic_panels": sorted(SCHEMATICS),
            "february_reference": "0/144", "february_complete_grid": "210/2880",
            "threshold_sd_c": 1.05868854309524, "threshold_overall_error": 0.15181518151815182,
            "threshold_worst_context_error": 0.20441988950276244, "new_energyplus_runs": 0,
            "manifest_sha256": sha256(manifest)})
        return manifest

    def run(self) -> Path:
        self.figure1()
        self.figure2()
        self.figure3()
        self.figure4()
        self.figure5()
        self.figure6()
        return self.finalize()


def reproduce_figure_data(
    config_path: str | Path,
    output: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
    analysis_root: str | Path | None = None,
    legacy: bool = False,
) -> Path:
    config = load_config(config_path, data_root=data_root)
    validate_panel_registry(config["figures"]["panels"])
    analysis = (
        Path(analysis_root).expanduser().resolve()
        if analysis_root
        else output_root(config, output)
    )
    if legacy or "panel_materialization_support" not in config.get("assets", {}):
        return _legacy_reproduce(config, analysis)
    panel_root = (
        Path(output).expanduser().resolve()
        if output is not None and analysis_root
        else analysis / "figure_data_v2"
    )
    return Materializer(config, analysis, panel_root).run()

