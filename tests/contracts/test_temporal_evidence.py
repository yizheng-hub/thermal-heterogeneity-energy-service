from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_monthly_service_rollup_uses_occupied_weights_and_fixed_tail():
    seasonal = module("src/analysis/seasonal/derive_seasonal_comparison.py", "seasonal_numeric")
    months = np.repeat(np.arange(1, 13), 2)
    people = np.column_stack([np.arange(1, 25), np.arange(24, 0, -1)]).astype(float)
    data = {"zone_ids": ["A", "B"], "temperatures": np.column_stack([months, 2 * months]).astype(float),
            "people": people, "occupied": np.ones(24, dtype=bool), "month": months}
    for key in seasonal.CARRIERS:
        data[key] = np.arange(24).astype(float)
    profiles = pd.DataFrame({"zone_id": ["A", "B"], "centre_c": [0., 0.], "lower_c": [0., 0.], "upper_c": [30., 30.]})
    rows, rollups = seasonal.calculate(data, profiles, np.array([0, 1]),
                                      SimpleNamespace(electricity_co2_kg_kwh=.2, gas_co2_kg_kwh_hhv=.1))
    annual = rows[-1]
    expected = (data["temperatures"] * people).sum() / people.sum()
    expected_tail = ((data["temperatures"] * people).sum(axis=0) / people.sum(axis=0)).mean()
    assert np.isclose(annual["mean_reference_mismatch_c"], expected)
    assert np.isclose(annual["fixed_tail_reference_mismatch_c"], expected_tail)
    assert np.isclose(rollups["month"], expected_tail)
    assert np.isclose(sum(row["hvac_electric_kwh"] for row in rows[:12]), annual["hvac_electric_kwh"])
    assert not np.isclose(np.mean([row["fixed_tail_reference_mismatch_c"] for row in rows[:12]]), expected_tail)


def test_seasons_cover_each_month_once_and_keep_winter_heating():
    seasonal = module("src/analysis/seasonal/derive_seasonal_comparison.py", "seasonal_periods")
    assert sorted(sum(seasonal.SEASONS.values(), [])) == list(range(1, 13))
    assert seasonal.SEASONS["Winter"] == [12, 1, 2]
    assert {"heating_electric_kwh", "gas_hhv_kwh", "cooling_electric_kwh"} <= set(seasonal.CARRIERS)


def test_empty_result_panels_require_explicit_visible_status(tmp_path):
    renderer = module("src/analysis/energy_service_paper/scripts/render_control_service_figures.py", "temporal_renderer")
    figure, axis = renderer.plt.subplots()
    source = tmp_path / "source.csv"
    source.write_text("selection_status\nengineering_rejected\n")
    with pytest.raises(ValueError, match="Empty panels require explicit"):
        renderer.export_result(figure, [axis], tmp_path, "empty", caption="Status.", sources={"source": source})
    with pytest.raises(ValueError, match="visibly identify"):
        renderer.export_result(figure, [axis], tmp_path, "empty", caption="Status.", sources={"source": source},
                               documented_empty_axes=[axis])
    renderer.plt.close(figure)


def test_direct_common_gap_comparison_matches_same_conditions_and_removes_floor_aliases():
    seasonal = module("src/analysis/seasonal/derive_seasonal_comparison.py", "regime_pairing")
    rows = []
    for context, permission in [("Miami", "floor"), ("Seattle_small2019", "floor"), ("Seattle_small2019", "native_zone")]:
        for mode, fraction, gap in [("direct", -1, 5), ("equal_service", 1, 1), ("equal_service", .75, 3)]:
            rows.append({"context": context, "permission": permission, "composition_id": 1,
                         "layout": "random", "alternative": "personal_center", "comparison_mode": mode,
                         "service_fraction": fraction, "status": "matched", "delta_site_pct": gap})
    # A common-service-only condition cannot be substituted for an unavailable direct comparison.
    rows.append(rows[1] | {"composition_id": 2})
    paired = seasonal.match_comparison_regimes(pd.DataFrame(rows))
    assert len(paired) == 2
    assert set(paired.permission) == {"floor", "native_zone"}
    assert paired.composition_id.eq(1).all()
    assert paired.delta_site_pct_direct.eq(5).all() and paired.delta_site_pct_common.eq(1).all()


def test_temporal_compensation_closes_seasons_and_distinguishes_same_runs():
    seasonal = module("src/analysis/seasonal/derive_seasonal_comparison.py", "temporal_compensation")
    rows = []

    def add_pair(comparison_mode, fraction, alternative, reference_run, alternative_run, site, carbon):
        periods = list(seasonal.SEASONS) + ["Annual"]
        for index, period in enumerate(periods):
            rows.append({"context": "Miami", "composition_id": 1, "layout": "random_mixing",
                "permission": "native_zone", "comparison_mode": comparison_mode,
                "service_fraction": fraction, "alternative": alternative, "reference": "uniform_average",
                "period_type": "season" if period != "Annual" else "annual", "period": period,
                "pair_status": "matched", "alternative_status": "selected", "reference_status": "selected",
                "alternative_run_id": alternative_run, "reference_run_id": reference_run,
                "same_physical_operation": alternative_run == reference_run,
                "delta_purchased_site_kwh": site[index],
                "delta_operational_co2_kg": carbon[index]})

    add_pair("direct", -1., "personal_center", "uniform", "personal",
             [10., -4., 2., -1., 7.], [5., -2., 1., -.5, 3.5])
    add_pair("equal_service", 1., "personal_center", "shared", "shared",
             [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.])
    add_pair("equal_service", 1., "ltpref_interval", "uniform", "range",
             [2., -3., 1., -1., -1.], [1., -1.5, .5, -.5, -.5])
    pairs, summary = seasonal.temporal_compensation(pd.DataFrame(rows))
    direct = pairs[pairs.comparison_mode.eq("direct")].iloc[0]
    assert direct.alternative_run_id == "personal" and direct.reference_run_id == "uniform"
    assert not direct.same_physical_operation
    assert direct.seasonal_added_site_kwh == 12.
    assert direct.seasonal_avoided_site_kwh == 5.
    assert np.isclose(direct.seasonal_site_kwh_offset_fraction, 1 - 7 / 17)
    assert direct.seasonal_annual_closure_error_site_kwh == 0.
    equal = summary[summary.comparison_mode.eq("equal_service")].iloc[0]
    assert equal.matched_pairs == 2 and equal.same_physical_operation_pairs == 1
    assert equal.distinct_physical_operation_pairs == 1
    assert equal.opposing_seasonal_site_kwh_direction_pairs == 1
    assert equal.annual_site_kwh_decrease_pairs == 1 and equal.annual_site_kwh_zero_pairs == 1


def test_temporal_compensation_rejects_a_nonclosing_annual_total():
    seasonal = module("src/analysis/seasonal/derive_seasonal_comparison.py", "temporal_nonclosure")
    rows = []
    for period, value in [("Spring", 1.), ("Summer", 2.), ("Autumn", 3.), ("Winter", 4.),
                          ("Annual", 11.)]:
        rows.append({"context": "Miami", "composition_id": 1, "layout": "random_mixing",
            "permission": "native_zone", "comparison_mode": "direct", "service_fraction": -1.,
            "alternative": "personal_center", "reference": "uniform_average",
            "period_type": "annual" if period == "Annual" else "season", "period": period,
            "pair_status": "matched", "alternative_status": "selected", "reference_status": "selected",
            "alternative_run_id": "personal", "reference_run_id": "uniform",
            "same_physical_operation": False, "delta_purchased_site_kwh": value,
            "delta_operational_co2_kg": value})
    with pytest.raises(ValueError, match="Seasons do not close"):
        seasonal.temporal_compensation(pd.DataFrame(rows))
