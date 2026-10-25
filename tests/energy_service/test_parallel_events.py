"""Parallel cohort events preserve own baselines and balanced physical replicas."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/analysis/targeted_extension"))
from ctcd_parallel_events import joint_portfolio, portfolio  # noqa: E402
from baseline_decision_contract import KEYS, SERVICE, decision_sha256, rows, verify_baseline  # noqa: E402
from run_portfolio import read_csv, write_csv  # noqa: E402


def test_explicit_cohort_count_does_not_weaken_baseline_identity(tmp_path):
    records = []
    for control in ("uniform_average", "personal_center", "ltpref_interval"):
        record = dict(zip(KEYS, ["context", 1, "random_mixing", "native_zone", control]))
        record.update(selection_status="selected", run_id=control, variant=3)
        record.update({name: 1.25 for name in SERVICE})
        records.append(record)
    path = tmp_path / "DR_BASELINES.csv"
    write_csv(path, records)
    parsed = rows(path, expected_rows=3)
    generation = {"baseline_selection_sha256": "original"}
    contract = {"source_baseline_csv_sha256": "original", "decision_sha256": decision_sha256(parsed),
                "service_references": [{name: float(row[name]) for name in SERVICE} for row in parsed]}
    assert verify_baseline(path, generation, contract, expected_rows=3)["baseline_decisions_verified"] == 3
    with pytest.raises(ValueError, match="324"):
        rows(path)
    records[0]["run_id"] = "another-control-baseline"
    write_csv(path, records)
    with pytest.raises(ValueError, match="identifiers, states or variants"):
        verify_baseline(path, generation, contract, expected_rows=3)


def portfolio_fixture(tmp_path):
    types, hourly, design = [], [], []
    for seed in range(3):
        for layout in ("random_mixing", "demand_ordered"):
            key = {"climate_context": "test", "scope": "native_zone", "control": "personal_center",
                   "event_id": "month_07", "composition_id": str(seed), "layout": layout}
            design.append({"composition_id": str(seed), "layout": layout})
            types.append(key | {"baseline_case_id": "b", "event_case_id": "e", "floor_area_m2": 100,
                "baseline_engineering_status": "PASS", "event_engineering_status": "PASS",
                "baseline_service_qualified": True, "event_service_qualified": True})
            for hour in range(50):
                hourly.append(key | {"hour_index": hour, "phase": "event" if hour < 2 else "recovery",
                    "timestamp_local": (datetime(2001, 7, 1, 14) + timedelta(hours=hour)).isoformat(),
                    "timestep_hours": 1, "baseline_electric_kwh": 100,
                    "event_electric_kwh": 90 if hour < 2 else 100,
                    "baseline_gas_hhv_kwh": 0, "event_gas_hhv_kwh": 0})
    for name, records in (("PORTFOLIO_TYPES.csv", types), ("PORTFOLIO_HOURLY.csv", hourly),
                          ("PORTFOLIO_DESIGN_TYPES.csv", design)):
        write_csv(tmp_path / name, records)
    contexts = tmp_path / "contexts.json"
    contexts.write_text(json.dumps({"test": {"floor_binding_equals_building": False}}))
    return contexts


def test_matched_portfolio_uses_72_without_fractional_buildings(tmp_path):
    contexts = portfolio_fixture(tmp_path)
    report = portfolio(tmp_path, contexts, "CTCD", tmp_path / "portfolio")
    results = read_csv(tmp_path / "portfolio/portfolio_summary.csv")
    assert report["type_count"] == 6
    assert [int(row["nominal_participating_buildings"]) for row in results] == [18, 36, 72]
    assert [float(row["event_average_reduction_kw"]) for row in results] == [180, 360, 720]
    assert all(int(row["portfolio_buildings"]) == 72 for row in results)
    assert all(row["data_source"] == "CTCD" for row in results)


def test_missing_type_is_not_silently_reweighted(tmp_path):
    contexts = portfolio_fixture(tmp_path)
    types = read_csv(tmp_path / "PORTFOLIO_TYPES.csv")
    write_csv(tmp_path / "PORTFOLIO_TYPES.csv", types[:-1])
    with pytest.raises(ValueError, match="Missing or duplicated"):
        portfolio(tmp_path, contexts, "ASHRAE", tmp_path / "portfolio")


def test_joint_portfolio_closes_as_two_equal_source_inventories(tmp_path):
    source_roots = [tmp_path / "ashrae", tmp_path / "ctcd"]
    for index, source in enumerate(source_roots):
        source.mkdir()
        contexts = portfolio_fixture(source)
        if index:
            for name in ("PORTFOLIO_TYPES.csv", "PORTFOLIO_HOURLY.csv", "PORTFOLIO_DESIGN_TYPES.csv"):
                records = read_csv(source / name)
                for row in records:
                    row["composition_id"] = str(int(row["composition_id"]) + 1000)
                write_csv(source / name, records)
        portfolio(source, contexts, ("ASHRAE", "CTCD")[index], source / "portfolio")
    report = joint_portfolio(*source_roots, *(source / "portfolio" for source in source_roots),
                             contexts, tmp_path / "joint")
    values = read_csv(tmp_path / "joint/portfolio_summary.csv")
    assert report["portfolio_buildings"] == 144
    assert report["joint_equals_two_source_portfolios_hourly_max_residual_kwh"] == 0
    assert [int(row["nominal_participating_buildings"]) for row in values] == [36, 72, 144]
    assert [float(row["event_average_reduction_kw"]) for row in values] == [360, 720, 1440]
