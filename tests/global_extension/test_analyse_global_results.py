from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

from analyse_global_results import (  # noqa: E402
    analyse,
    energy_metrics,
    fixed_plan_information_diagnostic,
    fixed_tail_from_anchor,
    local_season,
    meteorological_period,
    normalise_manifest,
    pair_controls,
    select_common_service,
    summarise_information_diagnostic,
    zone_mean_profiles,
)
from global_matrix import sha256  # noqa: E402


def _profiles() -> pd.DataFrame:
    return pd.DataFrame({
        "profile_id": ["a", "b", "c", "d"],
        "zone_id": ["z1", "z1", "z2", "z2"],
        "centre_c": [20.0, 24.0, 25.0, 26.0],
        "lower_c": [19.0, 23.0, 24.0, 25.0],
        "upper_c": [21.0, 25.0, 26.0, 27.0],
    })


def _trajectory() -> dict[str, np.ndarray]:
    hours = 4
    return {
        "zone_ids": np.asarray(["z1", "z2"]),
        "temperatures": np.full((hours, 2), 25.0),
        "people": np.ones((hours, 2)),
        "occupied": np.ones(hours, dtype=bool),
        "month": np.asarray([1, 4, 7, 10]),
        "facility_gas_hhv_kwh_annual": np.asarray(5.0),
        "facility_electric_kwh": np.asarray([5.0, 6.0, 7.0, 8.0]),
        "hvac_electric_kwh": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "heating_electric_kwh": np.asarray([0.5, 0.5, 0.0, 0.0]),
        "cooling_electric_kwh": np.asarray([0.0, 0.5, 2.0, 2.0]),
        "fans_electric_kwh": np.asarray([0.5, 1.0, 1.0, 2.0]),
        "gas_hhv_kwh": np.asarray([2.0, 1.0, 0.0, 0.0]),
    }


def test_anchor_fixes_the_worst_served_profile():
    tail, score = fixed_tail_from_anchor(_trajectory(), _profiles())
    assert tail.tolist() == [0]
    assert score["summary"]["mean_reference_mismatch_c"] == 1.75
    assert score["summary"]["fixed_tail_reference_mismatch_c"] == 5.0
    assert score["summary"]["modeled_interval_person_hour_coverage"] == 0.75


def test_zone_mean_representation_changes_centres_only_within_each_zone():
    profiles = _profiles()
    represented = zone_mean_profiles(profiles)
    assert represented.centre_c.tolist() == [22.0, 22.0, 25.5, 25.5]
    assert represented.lower_c.equals(profiles.lower_c)
    assert represented.upper_c.equals(profiles.upper_c)


def test_fixed_plan_diagnostic_detects_mean_selected_plan_failure():
    metrics = pd.DataFrame([
        {
            "climate": "Beijing", "composition": "China A",
            "layout": "random_mixing", "permission": "native_zone",
            "control": "uniform_average", "offset_c": 0.0,
            "run_id": "low", "physical_plan_id": "plan-low",
            "engineering_pass": True, "facility_purchased_kwh": 80.0,
            "mean_reference_mismatch_c": 1.2,
            "fixed_tail_reference_mismatch_c": 1.2,
            "zone_mean_mean_reference_mismatch_c": 0.8,
            "zone_mean_fixed_tail_reference_mismatch_c": 0.8,
        },
        {
            "climate": "Beijing", "composition": "China A",
            "layout": "random_mixing", "permission": "native_zone",
            "control": "personal_center", "offset_c": 0.0,
            "run_id": "safe", "physical_plan_id": "plan-safe",
            "engineering_pass": True, "facility_purchased_kwh": 100.0,
            "mean_reference_mismatch_c": 0.9,
            "fixed_tail_reference_mismatch_c": 0.9,
            "zone_mean_mean_reference_mismatch_c": 0.9,
            "zone_mean_fixed_tail_reference_mismatch_c": 0.9,
        },
    ])
    anchors = pd.DataFrame([{
        "climate": "Beijing", "composition": "China A",
        "layout": "random_mixing", "anchor_mean_mismatch_c": 1.0,
        "anchor_tail_mismatch_c": 1.0,
    }])
    result = fixed_plan_information_diagnostic(metrics, anchors, fractions=(1.0,))
    assert len(result) == 1
    row = result.iloc[0]
    assert row.full_selected_plan_id == "plan-safe"
    assert row.zone_mean_selected_plan_id == "plan-low"
    assert bool(row.eligibility_disagreement)
    assert bool(row.selection_disagreement)
    assert bool(row.zone_mean_selected_plan_fails_full_review)
    summary = summarise_information_diagnostic(result)
    assert (
        summary["overall"]["full_distribution_resolvable"]
        ["zone_mean_selected_plan_fails_full_review"]
        == 1
    )
    assert (
        summary["by_service_fraction"]["1"]["full_distribution_resolvable"]
        ["distribution_required"]
        == 1
    )


def test_energy_and_hemisphere_seasons():
    result = energy_metrics(_trajectory())
    assert result["hvac_electric_kwh"] == 10.0
    assert result["facility_electric_kwh"] == 26.0
    assert result["gas_hhv_kwh"] == 3.0
    assert result["conditioning_purchased_kwh"] == 13.0
    assert result["facility_electric_plus_space_heating_gas_kwh"] == 29.0
    assert result["facility_gas_hhv_kwh"] == 5.0
    assert result["facility_purchased_kwh"] == 31.0
    assert result["total_purchased_kwh"] == 31.0
    seasonal = energy_metrics(_trajectory(), mask=np.asarray([True, False, False, False]))
    assert np.isnan(seasonal["facility_gas_hhv_kwh"])
    assert np.isnan(seasonal["facility_purchased_kwh"])
    assert np.isnan(seasonal["total_purchased_kwh"])
    assert meteorological_period(1) == "DJF"
    assert meteorological_period(8) == "JJA"
    assert local_season("DJF", 40.0) == "winter"
    assert local_season("DJF", -34.0) == "summer"


def test_manifest_requires_explicit_engineering_qualification():
    raw = pd.DataFrame({
        "context": ["Beijing"],
        "composition_id": ["China A"],
        "control": ["ltpref_interval"],
        "offset_c": [0.0],
        "run_id": ["r1"],
        "compact_output": ["r1.npz"],
        "engineering_pass": ["true"],
    })
    result = normalise_manifest(raw)
    assert result.loc[0, "climate"] == "Beijing"
    assert result.loc[0, "control"] == "personal_range"
    assert bool(result.loc[0, "engineering_pass"])


def test_analysis_refuses_engineering_failed_candidates(tmp_path: Path):
    run_manifest = tmp_path / "runs.csv"
    pd.DataFrame([{
        "climate": "TestCity", "composition_id": 1,
        "control": "uniform_average", "offset_c": 0.0, "run_id": "failed",
        "relative_engineering_pass": False, "hourly_path": "missing.npz",
        "hourly_sha256": "unused-because-engineering-gate-fails-first",
    }]).to_csv(run_manifest, index=False)

    with pytest.raises(ValueError, match="requires every declared annual candidate"):
        analyse(
            run_manifest,
            tmp_path / "unused_anchor.csv",
            tmp_path / "unused_profiles.csv",
            tmp_path / "unused_climates.json",
            tmp_path / "unused_carbon.csv",
            tmp_path / "results",
        )


def test_common_service_selects_lowest_energy_qualified_candidate():
    rows = []
    for control in ("uniform_average", "personal_center", "personal_range"):
        for offset, energy, mismatch in ((-0.5, 80.0, 1.5), (0.0, 100.0, 0.8)):
            rows.append({
                "climate": "Beijing",
                "composition": "China A",
                "layout": "random_mixing",
                "permission": "native_zone",
                "control": control,
                "offset_c": offset,
                "run_id": f"{control}_{offset}",
                "engineering_pass": True,
                "mean_reference_mismatch_c": mismatch,
                "fixed_tail_reference_mismatch_c": mismatch,
                "facility_electric_plus_space_heating_gas_kwh": energy,
                "facility_purchased_kwh": energy,
                "total_purchased_kwh": energy,
            })
    anchors = pd.DataFrame([{
        "climate": "Beijing",
        "composition": "China A",
        "layout": "random_mixing",
        "anchor_mean_mismatch_c": 1.0,
        "anchor_tail_mismatch_c": 1.0,
    }])
    selected = select_common_service(pd.DataFrame(rows), anchors, fractions=(1.0,))
    assert selected.selection_status.eq("selected").all()
    assert selected.offset_c.eq(0.0).all()
    assert selected.total_purchased_kwh.eq(100.0).all()


def test_control_pair_reports_service_energy_and_carbon_deltas():
    base = {
        "climate": "Beijing",
        "composition": "China A",
        "layout": "random_mixing",
        "permission": "native_zone",
        "selection_status": "selected",
        "mean_reference_mismatch_c": 2.0,
        "fixed_tail_reference_mismatch_c": 3.0,
        "modeled_interval_person_hour_coverage": 0.5,
        "modeled_interval_mean_excess_c": 1.0,
        "hvac_electric_kwh": 100.0,
        "facility_electric_kwh": 150.0,
        "gas_hhv_kwh": 20.0,
        "facility_gas_hhv_kwh": 30.0,
        "conditioning_purchased_kwh": 120.0,
        "facility_electric_plus_space_heating_gas_kwh": 170.0,
        "facility_purchased_kwh": 180.0,
        "total_purchased_kwh": 180.0,
        "conditioning_operational_co2_country_kg": 50.0,
        "conditioning_operational_co2_common_factor_kg": 60.0,
        "operational_co2_country_kg": 50.0,
        "operational_co2_common_factor_kg": 60.0,
    }
    rows = []
    for control, scale in (("uniform_average", 1.0), ("personal_center", 0.8)):
        row = base | {"control": control, "run_id": control}
        for key in (
            "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c",
            "modeled_interval_mean_excess_c", "hvac_electric_kwh",
            "facility_electric_kwh", "gas_hhv_kwh",
            "facility_gas_hhv_kwh",
            "conditioning_purchased_kwh",
            "facility_electric_plus_space_heating_gas_kwh",
            "facility_purchased_kwh",
            "total_purchased_kwh", "conditioning_operational_co2_country_kg",
            "conditioning_operational_co2_common_factor_kg",
            "operational_co2_country_kg",
            "operational_co2_common_factor_kg",
        ):
            row[key] *= scale
        row["modeled_interval_person_hour_coverage"] = 0.5 if scale == 1 else 0.7
        rows.append(row)
    result = pair_controls(pd.DataFrame(rows), "direct")
    assert result.loc[0, "status"] == "matched"
    assert np.isclose(result.loc[0, "delta_mean_reference_mismatch_c"], -0.4)
    assert np.isclose(result.loc[0, "delta_total_purchased_pct"], -20.0)
    assert np.isclose(
        result.loc[0, "delta_conditioning_purchased_kwh"], -24.0
    )
    assert np.isclose(
        result.loc[0, "delta_facility_electric_plus_space_heating_gas_kwh"], -34.0
    )
    assert np.isclose(result.loc[0, "delta_facility_purchased_kwh"], -36.0)
    assert np.isclose(
        result.loc[0, "delta_modeled_interval_person_hour_coverage"], 0.2
    )


def test_analysis_writes_audited_annual_seasonal_and_pair_tables(tmp_path: Path):
    hours = 8760
    month = np.repeat(
        np.arange(1, 13),
        [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744],
    )
    hourly = tmp_path / "hourly"
    hourly.mkdir()

    def write_run(name: str, temperature: float) -> Path:
        path = hourly / f"{name}.npz"
        np.savez_compressed(
            path,
            zone_ids=np.asarray(["z1"]),
            temperatures=np.full((hours, 1), temperature),
            people=np.ones((hours, 1)),
            occupied=np.ones(hours, dtype=bool),
            month=month,
                facility_gas_hhv_kwh_annual=np.asarray(1_000.0),
            facility_electric_kwh=np.full(hours, 2.0),
            hvac_electric_kwh=np.ones(hours),
            heating_electric_kwh=np.full(hours, 0.2),
            cooling_electric_kwh=np.full(hours, 0.5),
            fans_electric_kwh=np.full(hours, 0.3),
            gas_hhv_kwh=np.full(hours, 0.1),
        )
        return path

    anchor = write_run("anchor", 25.0)
    controls = {
        "uniform_average": write_run("uniform", 25.0),
        "personal_center": write_run("center", 24.5),
        "personal_range": write_run("range", 25.0),
    }
    run_manifest = tmp_path / "runs.csv"
    pd.DataFrame([
        {
            "climate": "TestCity", "composition_id": 1,
            "layout": "random_mixing", "permission": "native_zone",
            "control": control, "offset_c": 0.0, "run_id": control,
            "relative_engineering_pass": True, "hourly_path": path,
            "hourly_sha256": sha256(path),
            "input_sha256": sha256(path), "targets_sha256": sha256(path),
        }
        for control, path in controls.items()
    ]).to_csv(run_manifest, index=False)
    anchor_manifest = tmp_path / "anchors.csv"
    pd.DataFrame([{
        "climate": "TestCity", "target_c": 25.0, "run_id": "anchor",
        "hourly_path": anchor, "hourly_sha256": sha256(anchor),
    }]).to_csv(anchor_manifest, index=False)
    profiles = tmp_path / "profiles.csv"
    pd.DataFrame({
        "context": ["TestCity", "TestCity"], "composition_id": [1, 1],
        "layout": ["random_mixing", "random_mixing"],
        "profile_id": ["a", "b"], "zone_id": ["z1", "z1"],
        "centre_c": [24.0, 26.0], "controller_lower_c": [23.0, 25.0],
        "controller_upper_c": [25.0, 27.0],
    }).to_csv(profiles, index=False)
    climates = tmp_path / "climates.json"
    climates.write_text(
        json.dumps({"climates": [{"climate": "TestCity", "latitude": -30.0}]}),
        encoding="utf-8",
    )
    carbon = tmp_path / "carbon.csv"
    pd.DataFrame([{
        "site": "TestCity", "electricity_co2_kg_kwh": 0.4,
        "common_world_2025_electricity_co2_kg_kwh": 0.5,
    }]).to_csv(carbon, index=False)
    output = tmp_path / "results"
    audit = analyse(
        run_manifest, anchor_manifest, profiles, climates, carbon, output
    )
    assert audit["status"] == "PASS"
    assert audit["annual_scored_candidates"] == 3
    assert len(pd.read_csv(output / "GLOBAL_SEASONAL_METRICS.csv")) == 12
    pairs = pd.read_csv(output / "GLOBAL_DIRECT_COMPARISONS.csv")
    assert len(pairs) == 2
    assert pairs.status.eq("matched").all()
    annual = pd.read_csv(output / "GLOBAL_RUN_METRICS.csv")
    assert annual.total_purchased_kwh.eq(annual.facility_purchased_kwh).all()
    assert annual.operational_co2_country_kg.gt(
        annual.conditioning_operational_co2_country_kg
    ).all()

    controls["uniform_average"].write_bytes(
        controls["uniform_average"].read_bytes() + b"tampered"
    )
    with pytest.raises(ValueError, match="SHA-256|(?i:hash)"):
        analyse(
            run_manifest, anchor_manifest, profiles, climates, carbon,
            tmp_path / "tampered_results",
        )
