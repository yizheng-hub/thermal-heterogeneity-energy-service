from pathlib import Path
import importlib.util
import json
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


def test_temporal_assets_preserve_existing_evidence_and_only_renumber_figures(tmp_path):
    finalizer = module("publication/energy_service_paper/scripts/finalize_temporal_assets.py", "temporal_assets")
    paper, seasonal, typical = [tmp_path / name for name in ["paper", "seasonal", "typical"]]

    def bundle(stem, marker):
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix in finalizer.SUFFIXES:
            Path(str(stem) + suffix).write_bytes(marker.encode())
        Path(str(stem) + "_caption.txt").write_text("Figure | Data; Supplementary Fig. S10; Supplementary Section S10.3.")

    order = []
    for number in range(1, 22):
        bundle(paper / f"supplementary_figures/Figure_S{number:02d}", str(number))
        order.append({"display_number": number, "source_figure_number": number})
    (paper / "SUPPLEMENTARY_FIGURE_ORDER.json").write_text(json.dumps(order))
    for number in range(1, 7):
        directory = paper / f"captions_alt/figure_{number:02d}"
        directory.mkdir(parents=True)
        for name in ["caption.txt", "alt_text.txt"]:
            (directory / name).write_text("Supplementary Fig. S10; Supplementary Table S10; Supplementary Section S10.")
    (paper / "figures/figure_03").mkdir(parents=True)
    for stem in ["Supplementary_monthly_direct", "Supplementary_monthly_common_service", "Figure_03_extended"]:
        bundle(seasonal / stem, stem)
    bundle(typical / "TYPICAL_DAY_DR", "typical")
    finalizer.finalize(paper, seasonal, typical)
    result = json.loads((paper / "SUPPLEMENTARY_FIGURE_ORDER.json").read_text())
    assert [item["display_number"] for item in result] == list(range(1, 25))
    for old, new in finalizer.OLD_TO_NEW.items():
        assert (paper / f"supplementary_figures/Figure_S{new:02d}.pdf").read_text() == str(old)
    caption = (paper / "captions_alt/figure_01/caption.txt").read_text()
    assert "Fig. S12" in caption and "Table S10" in caption and "Section S10" in caption
    assert (paper / "figures/figure_03/figure_03.pdf").read_text() == "Figure_03_extended"
    assert len(list((paper / "supplementary_figures").glob("*_word.png"))) == 24


def test_monthly_service_rollup_uses_occupied_weights_and_fixed_tail():
    seasonal = module("publication/seasonal/derive_seasonal_comparison.py", "seasonal_numeric")
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
    seasonal = module("publication/seasonal/derive_seasonal_comparison.py", "seasonal_periods")
    assert sorted(sum(seasonal.SEASONS.values(), [])) == list(range(1, 13))
    assert seasonal.SEASONS["Winter"] == [12, 1, 2]
    assert {"heating_electric_kwh", "gas_hhv_kwh", "cooling_electric_kwh"} <= set(seasonal.CARRIERS)


def test_empty_result_panels_require_explicit_visible_status(tmp_path):
    renderer = module("publication/energy_service_paper/scripts/render_control_service_figures.py", "temporal_renderer")
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
    seasonal = module("publication/seasonal/derive_seasonal_comparison.py", "regime_pairing")
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
