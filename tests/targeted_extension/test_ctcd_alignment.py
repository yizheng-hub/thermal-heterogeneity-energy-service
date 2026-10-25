"""External records must retain source scales and not invent independence."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = spec_from_file_location("ctcd_alignment", Path(__file__).resolve().parents[2]
                              / "src/analysis/targeted_extension/ctcd_alignment.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def external_records():
    row = {name: "x" for name in MODULE.TEXT.values()}
    row.update({name: 1. for name in MODULE.NUMERIC.values()})
    row.update({"profile_id": "record_1", "date": "2019-01-01", "year": 2019,
                "city": "Beijing", "city_normalized": "beijing", "contributor": "University",
                "building_type": "Office", "operation": "Convection heating", "sex": "Female",
                "data_class": "I", "top": 24., "ta": 23., "rh": 50., "vel": .1,
                "met": 1.2, "clo": .7, "thermal_sensation": .5,
                "thermal_acceptability": -.1, "thermal_comfort": 1.3})
    return pd.DataFrame([row])


def ashrae_records():
    measurements = pd.DataFrame([{"building_id": 1, "top": 25., "thermal_sensation": 1.,
                                  "clo": .8, "met": 1.1, "rh": 55., "gender": "female"}])
    metadata = pd.DataFrame([{"building_id": 1, "city": "Harbin", "country": "china", "year": 2011}])
    return measurements, metadata


def test_city_harmonisation_does_not_create_building_identity():
    assert MODULE.city_name("Heilongjiang Province, Harbin City") == "harbin"
    assert MODULE.city_name("Shaanxi Province,Xi’an City") == "xian"
    assert MODULE.city_name("Beijing City") == MODULE.city_name("Beijing")


def test_official_two_header_rows_and_checksum_are_read(tmp_path, monkeypatch):
    names = ["ID", *MODULE.TEXT, *MODULE.NUMERIC,
             "E1.Indoor Air Temperature (℃).1", "E2.Indoor Relative Humidity (%).1",
             "E3.Indoor Air Velocity (m/s).1"]
    original = external_records().iloc[0]
    values = [1, *[original[n] for n in MODULE.TEXT.values()],
              *[original[n] for n in MODULE.NUMERIC.values()], 23., 50., .1]
    path = tmp_path / "Chinese Thermal Comfort Dataset Class I.csv"
    with path.open("w", encoding="gb18030", newline="") as stream:
        import csv
        writer = csv.writer(stream)
        writer.writerow(["section"] * len(names))
        writer.writerow(names)
        writer.writerow(values)
    monkeypatch.setattr(MODULE, "FILES", {"I": (1, MODULE.sha256(path))})
    result, manifest = MODULE.read_official(tmp_path)
    assert len(result) == 1
    assert result.ta.iloc[0] == 23.
    assert result.thermal_acceptability.iloc[0] == -.1
    assert not manifest["I"]["thermal_preference_column"]
    assert "ID" not in result and "A1.Code" not in result


def test_overlap_excludes_coarse_groups_and_complete_fingerprints():
    external = pd.concat([external_records()] * 3, ignore_index=True)
    external.loc[0, ["city_normalized", "year"]] = ["harbin", 2011]
    external.loc[1, MODULE.FINGERPRINT[:-1]] = [25., 1., .8, 1.1, 55.]
    external.loc[2, MODULE.FINGERPRINT[:-1]] = [25., 1., .8, 1.1, np.nan]
    city_year, fingerprint, _ = MODULE.overlap_flags(external, *ashrae_records())
    assert city_year.tolist() == [True, False, False]
    assert fingerprint.tolist() == [False, True, False]


def test_alignment_keeps_fractional_votes_and_broader_operation_separate():
    records = pd.concat([external_records()] * 3, ignore_index=True)
    records.loc[1, "profile_id"] = "record_2"
    records.loc[1, "operation"] = "Naturally Ventilated"
    records.loc[2, "profile_id"] = "same_observation_different_record_label"
    selected, stages, _ = MODULE.align_office(records, *ashrae_records())
    assert len(selected) == 2
    assert selected.active_conditioning.tolist() == [True, False]
    assert selected.thermal_sensation.eq(.5).all()
    assert selected.thermal_acceptability.eq(-.1).all()
    assert selected.thermal_comfort.eq(1.3).all()
    assert selected.reference_g050_c.eq(23.).all()
    assert selected.temperature_source.eq("operative").all()
    assert stages.remaining.iloc[-1] == 2


def test_interval_reuse_preserves_record_count_and_clipped_bounds(monkeypatch):
    records = pd.concat([external_records()] * 2, ignore_index=True)
    selected, _, _ = MODULE.align_office(records, *ashrae_records())
    selected = pd.concat([selected] * 2, ignore_index=True)
    calls = []
    def fake(reference, clo, met, scenes):
        calls.append((reference, clo, met))
        return {"interval_lower_c": 16., "interval_upper_c": 30.,
                "interval_centre_c": reference, "main_control_eligible": True}
    monkeypatch.setattr(MODULE, "virtual_interval", fake)
    result = MODULE.add_intervals(selected, [], 1)
    assert len(calls) == 1
    assert len(result) == 2
    assert result.lower_c.eq(16.).all() and result.upper_c.eq(30.).all()
    assert result.controller_lower_c.equals(result.lower_c)


def test_plot_bins_keep_reference_and_executable_denominators_distinct(tmp_path):
    frame = external_records()
    frame = pd.concat([frame] * 3, ignore_index=True)
    for suffix in ("033", "050", "067"):
        frame[f"reference_g{suffix}_c"] = [22., 26., 28.]
    frame["active_conditioning"] = [True, True, False]
    frame["main_control_eligible"] = [True, False, True]
    frame["lower_c"], frame["upper_c"] = [20., 30., 20.], [24., 30., 26.]
    MODULE.plot_distributions(frame, tmp_path)
    references = pd.read_csv(tmp_path / "CTCD_REFERENCE_HISTOGRAM.csv")
    widths = pd.read_csv(tmp_path / "CTCD_INTERVAL_HISTOGRAM.csv")
    assert references.scope_records.eq(2).all()
    assert widths.scope_records.eq(1).all()
    assert references.groupby("griffiths_coefficient").records.sum().eq(2).all()
    assert widths.records.sum() == 1
    assert ((widths.bin_upper_c - widths.bin_lower_c) * widths.density_per_c).sum() == 1.
    assert "profile_id" not in references and "profile_id" not in widths
