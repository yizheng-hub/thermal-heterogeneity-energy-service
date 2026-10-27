from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

from global_matrix import ClimateSpec  # noqa: E402
from run_global_matrix import (  # noqa: E402
    COHORT_LABELS,
    ZONES,
    _hash_payload,
    _cached_run,
    apply_annual_solver_stability_exception,
    apply_solver_stability_exception,
    build_profiles,
    read_climates,
    read_annual_solver_stability_evidence,
    read_solver_stability_evidence,
    relative_engineering_qa,
    merge_control_manifests,
    retain_control_designs,
    run_controls,
    select_missing_control_designs,
    set_maximum_hvac_iterations,
    sha256,
    solver_configuration,
    warning_profile,
)


def _profiles(compositions: tuple[int, ...], prefix: str) -> pd.DataFrame:
    rows = []
    capacities = [53, 11, 7, 11, 7] * 3
    for composition in compositions:
        profile = 0
        for zone, count in zip(ZONES, capacities, strict=True):
            for _ in range(count):
                rows.append({
                    "composition_id": composition,
                    "layout": "random_mixing",
                    "profile_id": f"{prefix}_{composition}_{profile}",
                    "zone_id": zone,
                    "centre_c": 24.0 + profile / 1000,
                    "controller_lower_c": 22.0,
                    "controller_upper_c": 26.0,
                    "context": "Miami",
                })
                profile += 1
    return pd.DataFrame(rows)


def test_build_profiles_creates_six_named_compositions_per_climate(tmp_path: Path):
    multinational = _profiles((2026102001, 2026102002, 2026102003), "m")
    np.savez_compressed(
        tmp_path / "multinational.npz",
        **{
            name: (
                multinational[name].to_numpy(dtype=str)
                if name in {"layout", "profile_id", "zone_id"}
                else multinational[name].to_numpy()
            )
            for name in (
                "composition_id", "layout", "profile_id", "zone_id", "centre_c",
                "controller_lower_c", "controller_upper_c",
            )
        },
    )
    china = _profiles((2026102501, 2026102502, 2026102503), "c")
    china.to_csv(tmp_path / "china.csv", index=False)
    climates = [
        ClimateSpec("Beijing", "China", "Asia", "NewYork", "Beijing.epw", True),
        ClimateSpec("Miami", "USA", "North America", "Miami", "Miami.epw", False),
    ]
    result = build_profiles(
        climates, tmp_path / "multinational.npz", tmp_path / "china.csv"
    )
    assert len(result) == 2 * 6 * 267
    assert set(result["cohort_label"]) == set(COHORT_LABELS.values())
    assert result.groupby(["context", "composition_id"]).size().eq(267).all()


def test_build_profiles_rejects_unassigned_china_archive(tmp_path: Path):
    multinational = _profiles((2026102001, 2026102002, 2026102003), "m")
    np.savez_compressed(
        tmp_path / "multinational.npz",
        **{
            name: (
                multinational[name].to_numpy(dtype=str)
                if name in {"layout", "profile_id", "zone_id"}
                else multinational[name].to_numpy()
            )
            for name in (
                "composition_id", "layout", "profile_id", "zone_id", "centre_c",
                "controller_lower_c", "controller_upper_c",
            )
        },
    )
    raw_china = _profiles((2026102501, 2026102502, 2026102503), "c").drop(
        columns="context"
    )
    raw_china.to_csv(tmp_path / "raw_china.csv", index=False)
    climates = [
        ClimateSpec("Beijing", "China", "Asia", "NewYork", "Beijing.epw", True)
    ]

    with pytest.raises(ValueError, match="PROFILE_ASSIGNMENTS"):
        build_profiles(
            climates, tmp_path / "multinational.npz", tmp_path / "raw_china.csv"
        )


def test_relative_warning_gate_rejects_new_category_and_count_increase():
    run_qa = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    baseline = {"hvac_iterations": 14, "wet_bulb": 2,
                "psychrometric_temperature": 7, "coil_operating_range": 0}
    same = relative_engineering_qa(run_qa, dict(baseline), baseline)
    assert same["engineering_pass_relative_to_prototype_baseline"]
    increased = dict(baseline, hvac_iterations=15)
    assert not relative_engineering_qa(
        run_qa, increased, baseline
    )["engineering_pass_relative_to_prototype_baseline"]
    new = dict(baseline, coil_operating_range=1)
    checked = relative_engineering_qa(run_qa, new, baseline)
    assert not checked["engineering_pass_relative_to_prototype_baseline"]
    assert checked["new_warning_categories"] == ["coil_operating_range"]

    inherited_weather_drift = dict(
        baseline, wet_bulb=12, psychrometric_temperature=4
    )
    checked = relative_engineering_qa(
        run_qa, inherited_weather_drift, baseline
    )
    assert checked["engineering_pass_relative_to_prototype_baseline"]
    assert set(checked["inherited_environmental_warning_count_drift"]) == {
        "wet_bulb", "psychrometric_temperature",
    }

    no_weather_baseline = dict(baseline, wet_bulb=0)
    new_weather_category = dict(no_weather_baseline, wet_bulb=1)
    checked = relative_engineering_qa(
        run_qa, new_weather_category, no_weather_baseline
    )
    assert checked["engineering_pass_relative_to_prototype_baseline"]
    assert checked["new_warning_categories"] == []
    assert checked["inherited_environmental_warning_count_drift"] == {
        "wet_bulb": {"baseline": 0, "treatment": 1}
    }


def test_solver_stability_exception_is_probe_specific_and_audited(tmp_path: Path):
    evidence_path = tmp_path / "stability.csv"
    pd.DataFrame([{
        "climate": "Beijing", "target_c": 20.0,
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.0,
        "hvac_electricity_relative_difference": 0.0,
        "gas_hhv_relative_difference": 0.0,
        "total_purchased_energy_relative_difference": 0.0,
        "trajectory_stable_within_declared_tolerance": True,
        "declared_temperature_tolerance_c": 0.01,
        "declared_energy_relative_tolerance": 0.001,
        "first_treatment_warning_profile": '{"hvac_iterations": 3}',
        "second_treatment_warning_profile": '{"hvac_iterations": 3}',
    }]).to_csv(evidence_path, index=False)
    evidence, evidence_hash = read_solver_stability_evidence(evidence_path)
    assert evidence_hash
    relative = {
        "engineering_pass_relative_to_prototype_baseline": False,
        "new_warning_categories": ["hvac_iterations"],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 0, "treatment": 3}
        },
        "treatment_warning_profile": {"hvac_iterations": 3},
    }
    run_qa = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    passed, audit = apply_solver_stability_exception(
        relative,
        run_qa,
        evidence[("Beijing", 20.0)],
        current_maximum_hvac_iterations=40,
    )
    assert passed
    assert audit["solver_stability_exception_applied"]
    assert audit["raw_relative_engineering_pass"] is False
    assert audit["solver_stability_higher_iteration_limit"] == 100


def test_solver_stability_exception_never_overrides_other_failures():
    evidence = {
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.0,
        "hvac_electricity_relative_difference": 0.0,
        "gas_hhv_relative_difference": 0.0,
        "total_purchased_energy_relative_difference": 0.0,
        "first_treatment_warning_profile": '{"hvac_iterations": 3}',
    }
    relative = {
        "engineering_pass_relative_to_prototype_baseline": False,
        "new_warning_categories": ["coil_operating_range"],
        "critical_warning_count_increases": {
            "coil_operating_range": {"baseline": 0, "treatment": 1}
        },
        "treatment_warning_profile": {"coil_operating_range": 1},
    }
    run_qa = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    passed, audit = apply_solver_stability_exception(
        relative,
        run_qa,
        evidence,
        current_maximum_hvac_iterations=40,
    )
    assert not passed
    assert not audit["solver_stability_exception_applied"]


def test_solver_stability_evidence_rejects_changed_warning_profile(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame([{
        "climate": "Beijing", "target_c": 20.0,
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.0,
        "hvac_electricity_relative_difference": 0.0,
        "gas_hhv_relative_difference": 0.0,
        "total_purchased_energy_relative_difference": 0.0,
        "trajectory_stable_within_declared_tolerance": True,
        "declared_temperature_tolerance_c": 0.01,
        "declared_energy_relative_tolerance": 0.001,
        "first_treatment_warning_profile": '{"hvac_iterations": 3}',
        "second_treatment_warning_profile": '{"hvac_iterations": 2}',
    }]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="warning profile"):
        read_solver_stability_evidence(path)


def test_solver_stability_exception_requires_the_active_warning_profile():
    relative = {
        "engineering_pass_relative_to_prototype_baseline": False,
        "new_warning_categories": ["hvac_iterations"],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 0, "treatment": 3}
        },
        "treatment_warning_profile": {"hvac_iterations": 3},
    }
    run_qa = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    evidence = {
        "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.0,
        "total_purchased_energy_relative_difference": 0.0,
        "first_treatment_warning_profile": '{"hvac_iterations": 2}',
    }
    with pytest.raises(ValueError, match="active warning profile"):
        apply_solver_stability_exception(
            relative, run_qa, evidence, current_maximum_hvac_iterations=40
        )


def _annual_stability_row() -> dict:
    profile = json.dumps({
        "hvac_iterations": 3, "wet_bulb": 0,
        "psychrometric_temperature": 0, "coil_operating_range": 0,
    }, sort_keys=True)
    return {
        "run_id": "annual_test", "climate": "Beijing",
        "composition_id": 2026102001, "control": "personal_center",
        "offset_c": 0.0, "first_maximum_hvac_iterations": 40,
        "second_maximum_hvac_iterations": 100,
        "operative_temperature_max_abs_difference_c": 0.001,
        "hvac_electricity_relative_difference": 0.0001,
        "gas_hhv_relative_difference": 0.0001,
        "total_purchased_energy_relative_difference": 0.0001,
        "trajectory_stable_within_declared_tolerance": True,
        "declared_temperature_tolerance_c": 0.01,
        "declared_energy_relative_tolerance": 0.001,
        "higher_iteration_completed": True,
        "higher_iteration_severe_count": 0,
        "higher_iteration_fatal_occurrences": 0,
        "higher_iteration_relative_engineering_pass": False,
        "warning_profile_nonworsening": True,
        "first_treatment_warning_profile": profile,
        "second_treatment_warning_profile": profile,
        "first_input_sha256": "a" * 64,
        "first_hourly_sha256": "b" * 64,
    }


def test_annual_solver_stability_exception_requires_exact_hashes(tmp_path: Path):
    path = tmp_path / "annual_stability.csv"
    pd.DataFrame([_annual_stability_row()]).to_csv(path, index=False)
    evidence, evidence_hash = read_annual_solver_stability_evidence(path)
    assert evidence_hash
    relative = {
        "engineering_pass_relative_to_prototype_baseline": False,
        "new_warning_categories": ["hvac_iterations"],
        "critical_warning_count_increases": {
            "hvac_iterations": {"baseline": 0, "treatment": 3}
        },
        "treatment_warning_profile": json.loads(
            _annual_stability_row()["first_treatment_warning_profile"]
        ),
    }
    run_qa = {
        "completed": True, "reported_severe_count": 0, "fatal_occurrences": 0,
        "unexplained_recurring_groups": [],
    }
    passed, audit = apply_annual_solver_stability_exception(
        relative, run_qa, evidence["annual_test"],
        current_maximum_hvac_iterations=40,
        active_input_sha256="a" * 64, active_hourly_sha256="b" * 64,
    )
    assert passed
    assert audit["annual_solver_stability_exception_applied"]
    assert audit["annual_solver_stability_pass"]
    with pytest.raises(ValueError, match="active trajectory"):
        apply_annual_solver_stability_exception(
            relative, run_qa, evidence["annual_test"],
            current_maximum_hvac_iterations=40,
            active_input_sha256="a" * 64, active_hourly_sha256="c" * 64,
        )


def test_annual_solver_stability_reader_rejects_unstable_evidence(tmp_path: Path):
    row = _annual_stability_row()
    row["trajectory_stable_within_declared_tolerance"] = False
    path = tmp_path / "unstable.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="fails its declared limits"):
        read_annual_solver_stability_evidence(path)


def test_annual_solver_stability_accepts_count_drift_without_new_category(
    tmp_path: Path,
):
    row = _annual_stability_row()
    second = json.loads(row["second_treatment_warning_profile"])
    second["hvac_iterations"] += 2
    row["second_treatment_warning_profile"] = json.dumps(second, sort_keys=True)
    row["warning_profile_nonworsening"] = False
    row["higher_iteration_relative_engineering_pass"] = False
    path = tmp_path / "count_drift.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    evidence, _ = read_annual_solver_stability_evidence(path)
    selected = evidence["annual_test"]

    assert selected["warning_categories_stable"]
    assert selected["solver_stability_pass"]
    assert json.loads(selected["higher_iteration_new_warning_categories"]) == []
    assert json.loads(selected["warning_count_drift"])["hvac_iterations"]["delta"] == 2


def test_annual_solver_stability_rejects_new_warning_category(tmp_path: Path):
    row = _annual_stability_row()
    second = json.loads(row["second_treatment_warning_profile"])
    second["wet_bulb"] = 1
    row["second_treatment_warning_profile"] = json.dumps(second, sort_keys=True)
    path = tmp_path / "new_category.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="category-and-trajectory"):
        read_annual_solver_stability_evidence(path)


def test_cached_run_requires_matching_input_weather_runtime_and_hourly_hashes(
    tmp_path: Path,
):
    run_folder = tmp_path / "run"
    run_folder.mkdir()
    compact = tmp_path / "hourly.npz"
    compact.write_bytes(b"hourly")
    job = {
        "run_id": "probe_one",
        "run_folder": run_folder,
        "compact": compact,
        "input_sha256": "input",
        "weather_sha256": "weather",
    }
    qa = {
        "completed": True,
        "input_idf_sha256": "input",
        "weather_sha256": "weather",
        "runtime_sha256": "runtime",
        "hourly_sha256": sha256(compact),
    }
    (run_folder / "RUN_QA.json").write_text(json.dumps(qa), encoding="utf-8")
    assert _cached_run(job, "runtime") == qa
    with pytest.raises(ValueError, match="provenance mismatch"):
        _cached_run(job | {"weather_sha256": "different"}, "runtime")


def test_solver_limit_patch_preserves_other_convergence_fields():
    source = "ConvergenceLimits,1,20,2,8;\nBuilding,Office;"
    updated = set_maximum_hvac_iterations(source, 40)
    assert updated == "ConvergenceLimits,1,40,2,8;\nBuilding,Office;"
    assert solver_configuration(updated) == {
        "minimum_system_timestep_minutes": 1,
        "maximum_hvac_iterations": 40,
        "minimum_plant_iterations": 2,
        "maximum_plant_iterations": 8,
    }


def test_read_climates_retains_coordinates_and_zone(tmp_path: Path):
    path = tmp_path / "climates.json"
    path.write_text(json.dumps({"climates": [{
        "climate": "Sydney", "country": "Australia", "continent": "Oceania",
        "prototype_variant": "SanDiego", "weather": "Sydney.epw",
        "sentinel": True, "latitude": -33.95, "longitude": 151.18,
        "climate_zone": "3C",
    }]}), encoding="utf-8")
    result = read_climates(path)
    assert result[0].latitude == -33.95
    assert result[0].longitude == 151.18
    assert result[0].climate_zone == "3C"


def test_warning_profile_uses_explicit_recurring_total(tmp_path: Path):
    error = tmp_path / "eplusout.err"
    error.write_text(
        "   ** Warning ** SimHVAC: Maximum iterations (20) exceeded\n"
        "===== Recurring Error Summary =====\n"
        "   ** Warning ** SimHVAC: Exceeding Maximum iterations continues\n"
        "   **   ~~~   **   This error occurred 14 total times;\n"
        "   ** Warning ** WetBulb not converged after max iterations\n"
        "   **   ~~~   **   This error occurred 2 total times;\n",
        encoding="utf-8",
    )
    profile = warning_profile(error)
    assert profile["hvac_iterations"] == 14
    assert profile["wet_bulb"] == 2
    assert profile["psychrometric_temperature"] == 0


def test_manifest_filter_keeps_global_direct_and_sentinel_candidates():
    climates = [
        ClimateSpec("a", "A", "X", "v", "a.epw", True),
        ClimateSpec("b", "B", "Y", "v", "b.epw", False),
    ]
    designs = []
    for climate in ("a", "b"):
        for composition in COHORT_LABELS:
            for control in ("uniform_average", "personal_center", "personal_range"):
                for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    designs.append({
                        "context": climate, "composition_id": composition,
                        "control": control, "offset_c": offset,
                    })
    retained = retain_control_designs(climates, designs)
    assert len(retained) == 6 * 3 * 5 + 6 * 3
    counts = pd.DataFrame(retained).groupby("context").size().to_dict()
    assert counts == {"a": 90, "b": 18}
    assert json.loads(json.dumps(retained)) == retained

    full = retain_control_designs(climates, designs, "all")
    assert len(full) == 2 * 6 * 3 * 5
    assert pd.DataFrame(full).groupby("context").size().to_dict() == {"a": 90, "b": 90}


def test_manifest_filter_rejects_unknown_candidate_scope():
    climates = [ClimateSpec("a", "A", "X", "v", "a.epw", True)]
    with pytest.raises(ValueError, match="Unknown candidate scope"):
        retain_control_designs(climates, [], "unknown")


def test_candidate_extension_selects_only_missing_and_merges_exact_evidence(
    tmp_path: Path,
):
    designs = [
        {
            "context": "A", "composition_id": 1, "control": "uniform_average",
            "offset_c": offset, "targets_c": {"Zone": 25.0 + offset},
        }
        for offset in (0.0, 0.5)
    ]
    input_idf = tmp_path / "base.idf"
    hourly = tmp_path / "base.npz"
    input_idf.write_text("input", encoding="utf-8")
    hourly.write_bytes(b"hourly")
    existing = pd.DataFrame([{
        "climate": "A", "composition_id": 1, "control": "uniform_average",
        "offset_c": 0.0, "completed": True, "relative_engineering_pass": True,
        "input_sha256": sha256(input_idf),
        "targets_sha256": _hash_payload(designs[0]["targets_c"]),
        "input_idf_path": str(input_idf), "hourly_sha256": sha256(hourly),
        "hourly_path": str(hourly),
    }])
    existing_path = tmp_path / "existing.csv"
    existing.to_csv(existing_path, index=False)

    reused, missing = select_missing_control_designs(designs, existing_path)
    assert len(reused) == 1
    assert [row["offset_c"] for row in missing] == [0.5]

    extension = existing.copy()
    extension["offset_c"] = 0.5
    extension["targets_sha256"] = _hash_payload(designs[1]["targets_c"])
    extension_path = tmp_path / "extension.csv"
    extension.to_csv(extension_path, index=False)
    combined_path = tmp_path / "combined.csv"
    combined = merge_control_manifests(
        reused, extension_path, designs, combined_path
    )
    assert len(combined) == 2
    assert combined_path.is_file()


def test_manifest_merge_normalizes_legacy_solver_stability_fields(
    tmp_path: Path,
):
    profile = _annual_stability_row()["first_treatment_warning_profile"]
    designs = [{
        "context": "Beijing",
        "composition_id": 2026102001,
        "control": "personal_center",
        "offset_c": 0.0,
        "targets_c": {"Zone": 25.0},
    }]
    existing = pd.DataFrame([{
        "run_id": "annual_test",
        "climate": "Beijing",
        "composition_id": 2026102001,
        "control": "personal_center",
        "offset_c": 0.0,
        "completed": True,
        "relative_engineering_pass": True,
        "annual_solver_stability_exception_applied": True,
        "annual_solver_stability_evidence_sha256": pd.NA,
        "input_sha256": "a" * 64,
        "hourly_sha256": "b" * 64,
        "treatment_warning_profile": profile,
        "targets_sha256": _hash_payload(designs[0]["targets_c"]),
    }])
    extension_path = tmp_path / "extension.csv"
    pd.DataFrame(columns=existing.columns).to_csv(extension_path, index=False)
    evidence_path = tmp_path / "legacy_stability.csv"
    pd.DataFrame([_annual_stability_row()]).to_csv(evidence_path, index=False)

    combined = merge_control_manifests(
        existing,
        extension_path,
        designs,
        tmp_path / "combined.csv",
        annual_solver_stability_manifests=(evidence_path,),
    )

    assert bool(
        combined.annual_solver_stability_warning_categories_stable.iloc[0]
    )
    assert bool(combined.annual_solver_stability_pass.iloc[0])
    assert combined.annual_solver_stability_warning_count_drift.iloc[0] == "{}"
    assert (
        combined.annual_solver_stability_evidence_sha256.iloc[0]
        == sha256(evidence_path)
    )


def test_annual_controls_write_audit_then_fail_on_engineering_rejection(
    tmp_path: Path, monkeypatch,
):
    prepared = tmp_path / "prepared.idf"
    prepared.write_text("Building,Office;", encoding="utf-8")
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    runtime = tmp_path / "energyplus.exe"
    runtime.write_bytes(b"runtime")
    probes = tmp_path / "probes"
    (probes / "runs" / "anchor").mkdir(parents=True)
    pd.DataFrame([{
        "climate": "A", "target_c": 25.0, "run_id": "anchor",
    }]).to_csv(probes / "PROBE_RUN_MANIFEST.csv", index=False)
    output = tmp_path / "annual"

    monkeypatch.setattr(
        "run_global_matrix._write_run_idf",
        lambda model, targets, destination: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_text("input", encoding="utf-8"),
        ),
    )

    def fake_execute(jobs, executable, workers):
        return [job | {"run_qa": {
            "completed": True, "reported_severe_count": 0,
            "fatal_occurrences": 0, "unexplained_recurring_groups": [],
            "hourly_sha256": "hourly",
        }} for job in jobs]

    monkeypatch.setattr("run_global_matrix._execute_jobs", fake_execute)
    monkeypatch.setattr(
        "run_global_matrix.warning_profile",
        lambda path: {
            "hvac_iterations": 0 if "probe" in str(path) else 1,
            "wet_bulb": 0, "psychrometric_temperature": 0,
            "coil_operating_range": 0,
        },
    )
    designs = [{
        "context": "A", "composition_id": 2026102001,
        "control": "uniform_average", "offset_c": 0.0,
        "targets_c": dict.fromkeys(ZONES, 25.0),
    }]
    model_manifest = {"A": {
        "prepared_idf": str(prepared), "prepared_idf_sha256": "model",
        "weather_path": str(weather), "weather_sha256": "weather",
        "solver_configuration": {"maximum_hvac_iterations": 40},
    }}

    with pytest.raises(RuntimeError, match="Annual-control engineering gate failed"):
        run_controls(
            designs, model_manifest, probes, output, runtime, workers=1
        )
    manifest = pd.read_csv(output / "ANNUAL_CONTROL_RUN_MANIFEST.csv")
    audit = json.loads((output / "EXECUTION_QA.json").read_text(encoding="utf-8"))
    assert len(manifest) == 1
    assert not bool(manifest.raw_relative_engineering_pass.iloc[0])
    assert not bool(manifest.relative_engineering_pass.iloc[0])
    assert not bool(manifest.annual_solver_stability_exception_applied.iloc[0])
    assert audit["raw_relative_engineering_passes"] == 0
    assert audit["annual_solver_stability_exceptions"] == 0
    assert not audit["all_passed_relative_engineering_gate"]

    evidence_row = _annual_stability_row() | {
        "run_id": manifest.run_id.iloc[0],
        "climate": "A",
        "composition_id": 2026102001,
        "control": "uniform_average",
        "offset_c": 0.0,
        "first_input_sha256": manifest.input_sha256.iloc[0],
        "first_hourly_sha256": "b" * 64,
        "first_treatment_warning_profile": manifest.treatment_warning_profile.iloc[0],
        "second_treatment_warning_profile": manifest.treatment_warning_profile.iloc[0],
    }
    evidence_path = tmp_path / "annual_stability.csv"
    pd.DataFrame([evidence_row]).to_csv(evidence_path, index=False)

    def fake_execute_with_hash(jobs, executable, workers):
        return [job | {"run_qa": {
            "completed": True, "reported_severe_count": 0,
            "fatal_occurrences": 0, "unexplained_recurring_groups": [],
            "hourly_sha256": "b" * 64,
        }} for job in jobs]

    monkeypatch.setattr(
        "run_global_matrix._execute_jobs", fake_execute_with_hash
    )
    qualified = tmp_path / "qualified"
    run_controls(
        designs, model_manifest, probes, qualified, runtime, workers=1,
        annual_solver_stability_manifest=evidence_path,
    )
    qualified_manifest = pd.read_csv(
        qualified / "ANNUAL_CONTROL_RUN_MANIFEST.csv"
    )
    qualified_audit = json.loads(
        (qualified / "EXECUTION_QA.json").read_text(encoding="utf-8")
    )
    assert not bool(qualified_manifest.raw_relative_engineering_pass.iloc[0])
    assert bool(qualified_manifest.relative_engineering_pass.iloc[0])
    assert bool(
        qualified_manifest.annual_solver_stability_exception_applied.iloc[0]
    )
    assert qualified_audit["annual_solver_stability_exceptions"] == 1
    assert qualified_audit["all_passed_relative_engineering_gate"]
