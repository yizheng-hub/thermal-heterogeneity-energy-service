"""Independent identity and accounting checks for the six-composition evidence."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/targeted_extension"))
from audit_joint_evidence import verify_compositions, verify_signed_energy  # noqa: E402
from build_joint_evidence import cooling_offsets, signed_enduses  # noqa: E402


def fixed_compositions():
    sources, exported = {}, []
    for origin, offset in (("ASHRAE", 0), ("CTCD", 3)):
        blocks = []
        for index in range(3):
            composition = offset + index + 1
            profiles = pd.DataFrame({"profile_id": [f"record_{composition}_{i}" for i in range(267)],
                "centre_c": 20 + np.arange(267) / 100, "lower_c": 19 + np.arange(267) / 100,
                "upper_c": 21 + np.arange(267) / 100}).assign(composition_id=composition)
            exported.append(profiles.assign(data_source=origin, range_width_c=2.))
            for context in range(4):
                for layout in ("random_mixing", "demand_ordered"):
                    blocks.append(profiles.assign(context=f"weather{context}", layout=layout))
        sources[origin] = pd.concat(blocks, ignore_index=True)
    return sources, pd.concat(exported, ignore_index=True)


def test_all_draw_members_and_values_survive_each_weather_and_layout():
    sources, exported = fixed_compositions()
    report = verify_compositions(sources, exported)
    assert report["context_layout_blocks_verified"] == 48
    assert report["distinct_profile_rows_in_display"] == 1602
    sources["CTCD"].loc[0, "centre_c"] += .1
    with pytest.raises(ValueError, match="changed between"):
        verify_compositions(sources, exported)


def test_composition_export_cannot_swap_a_sampled_record_or_its_range():
    sources, exported = fixed_compositions()
    wrong = exported.copy()
    wrong.loc[0, "profile_id"] = "different_record"
    with pytest.raises(ValueError, match="sampled record"):
        verify_compositions(sources, wrong)
    exported.loc[0, "range_width_c"] = 1.8
    with pytest.raises(ValueError, match="range widths"):
        verify_compositions(sources, exported)


def energy_example(tmp_path):
    values = {
        "heating_electric_kwh": ([5, 1, 4, 0], [2, 6, 4, 2]),
        "cooling_electric_kwh": ([8, 8, 8, 8], [3, 3, 3, 3]),
        "fans_electric_kwh": ([2, 2, 2, 2], [3, 2, 1, 3]),
        "pumps_electric_kwh": ([1, 1, 1, 1], [1, 1, 1, 1]),
        "gas_hhv_kwh": ([0, 4, 0, 5], [3, 0, 0, 5]),
    }
    for index, label in enumerate(("reference", "alternative")):
        arrays = {name: np.pad(np.array(pair[index], dtype=float), (0, 8756)) for name, pair in values.items()}
        arrays["hvac_electric_kwh"] = sum(arrays[name] for name in values if name not in {"gas_hhv_kwh", "pumps_electric_kwh"})
        np.savez(tmp_path / (label + ".npz"), **arrays)
    manifest = tmp_path / "ANNUAL_RUN_MANIFEST.csv"
    pd.DataFrame([{"run_id": label, "path_hourly": label + ".npz"} for label in ("reference", "alternative")]).to_csv(manifest, index=False)
    pairs = pd.DataFrame([{"context": "Miami", "composition_id": 1, "layout": "random_mixing",
        "control": "personal_center", "data_source": "ASHRAE", "comparison_mode": "direct",
        "permission": "native_zone", "status": "matched", "reference_run_id": "reference",
        "alternative_run_id": "alternative", "delta_annual_site_kwh": -16.,
        "delta_annual_operational_co2_kg": -7.7}])
    factors = pd.DataFrame([{"city": "Miami", "electricity_co2_kg_kwh": .5,
                             "gas_co2_kg_kwh_hhv": .2}]).set_index("city")
    return manifest, pairs, factors


def test_production_signed_ledger_preserves_simultaneous_additions_and_savings(tmp_path):
    manifest, pairs, factors = energy_example(tmp_path)
    ledger, error = signed_enduses(pairs, (manifest,))
    # These totals are computed by hand from the four nonzero hours above.
    expected = {"heating_electric": (7., 3., 4.), "cooling_electric": (0., 20., -20.),
                "fans_electric": (2., 1., 1.), "gas_hhv": (3., 4., -1.)}
    for row in ledger.itertuples():
        assert (row.added_kwh, row.avoided_kwh, row.net_kwh) == expected[row.component]
    assert error == 0
    report = verify_signed_energy(pairs, ledger, (manifest,), factors)
    assert report["actual_reference_alternative_pairs"] == 1
    assert report["maximum_absolute_errors"]["component_net_sum_kwh"] == 0


def test_independent_audit_catches_wrong_gross_totals_even_if_annual_net_closes(tmp_path):
    manifest, pairs, factors = energy_example(tmp_path)
    ledger, _ = signed_enduses(pairs, (manifest,))
    ledger.loc[0, ["added_kwh", "avoided_kwh"]] += 1.
    assert ledger.net_kwh.sum() == -16.
    with pytest.raises(ValueError, match="accounting failed"):
        verify_signed_energy(pairs, ledger, (manifest,), factors)


def test_independent_audit_rejects_wrong_carbon_despite_correct_energy(tmp_path):
    manifest, pairs, factors = energy_example(tmp_path)
    ledger, _ = signed_enduses(pairs, (manifest,))
    pairs.loc[0, "delta_annual_operational_co2_kg"] = -6.
    with pytest.raises(ValueError, match="accounting failed"):
        verify_signed_energy(pairs, ledger, (manifest,), factors)


def test_independent_audit_rejects_a_change_outside_the_space_energy_boundary(tmp_path):
    manifest, pairs, factors = energy_example(tmp_path)
    ledger, _ = signed_enduses(pairs, (manifest,))
    with np.load(tmp_path / "alternative.npz", allow_pickle=False) as stored:
        values = dict(stored)
    values["pumps_electric_kwh"][0] += 1.
    np.savez(tmp_path / "alternative.npz", **values)
    with pytest.raises(ValueError, match="accounting failed"):
        verify_signed_energy(pairs, ledger, (manifest,), factors)


def test_cooling_savings_do_not_imply_lower_total_energy_or_carbon():
    # Hand-set balances: -10 cooling + 8 heating + 5 fans = +3 total.
    # The other two outcomes retain a genuine saving and a carbon-only saving.
    cases = pd.DataFrame({"comparison_mode": ["direct"] * 3 + ["equal_service"],
        "permission": ["native_zone"] * 4, "status": ["matched"] * 4,
        "delta_annual_cooling_electric_kwh": [-10., -10., -10., -10.],
        "delta_annual_site_kwh": [3., -4., 3., 3.],
        "delta_annual_operational_co2_kg": [1.5, -2., -1., 1.5],
        "delta_interval_coverage_percentage_points": [5., 2., -1., 5.],
        "delta_mean_reference_mismatch_c": [-.2, -.1, .1, -.2],
        "delta_fixed_tail_reference_mismatch_c": [-.1, .1, .2, -.1]})
    result = cooling_offsets(cases)
    assert len(result) == 3
    assert result.cooling_decreased.tolist() == [True, True, True]
    assert result.cooling_saving_offset.tolist() == [True, False, False]
    assert result.mean_and_tail_mismatch_improved.tolist() == [True, False, False]
