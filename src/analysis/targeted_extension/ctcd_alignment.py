"""Align official Chinese Thermal Comfort Dataset records with study inputs.

The released files contain sensation, comfort and acceptance votes, not the
thermal-preference variable described in the article. Records are not people,
and university/city/date fields do not establish a shared room or building.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from utilities.generate_demand_inputs import numeric_scenes, virtual_interval  # noqa: E402

FILES = {
    "I": (7246, "7a59ce782d9a55df8b0045c257e6fc66939feef8de7d1e66e0155efc55965370"),
    "II": (26184, "d4e6c64ac341e10e974cabfc6cf114256b6025852ee87a7580eda58706aa79f8"),
    "III": (8547, "ec94121c8ce92a7bb7429c0a8a17b059cf70928e9294acc3b006d7630a6454d8"),
}
TEXT = {"A2.Date": "date", "A3.Data Contributor": "contributor", "A5.City": "city",
        "A6.Climate Zone": "climate", "A4.Season": "season",
        "B1.Building Type": "building_type", "B2.Building Function": "building_function",
        "B4.Building Operation Mode": "operation", "C1.Sex": "sex", "C2.Age": "age_group"}
NUMERIC = {"D1.TSV": "thermal_sensation", "D2.TCV": "thermal_comfort",
           "D3.TAV": "thermal_acceptability", "D5.Clothing Insulation (clo)": "clo",
           "D6.Metabolic Rate (met)": "met", "F1.Operative Temperature (℃)": "top"}
ACTIVE = {"Convection cooling", "Convection heating", "Radiator heating",
          "Radiant floor heating", "Radiant floor cooling", "Floor radiation + fan coil",
          "Cold radiation ceiling cooling", "Air conditioner + fan", "Split air conditioner",
          "Air conditioning heating", "Ceiling capillary heating"}
FINGERPRINT = ["top", "thermal_sensation", "clo", "met", "rh", "sex"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def city_name(value: str) -> str:
    text = str(value).lower().replace("’", "'").strip()
    text = re.split(r"province\s*,?\s*", text)[-1]
    return re.sub(r"[^a-z]", "", text.replace("city", ""))


def read_official(root: Path) -> tuple[pd.DataFrame, dict]:
    frames, manifests = [], {}
    for label, (expected_rows, expected_sha) in FILES.items():
        path = root / f"Chinese Thermal Comfort Dataset Class {label}.csv"
        if sha256(path) != expected_sha:
            raise ValueError(f"Official Class {label} checksum differs")
        raw = pd.read_csv(path, header=1, encoding="gb18030", low_memory=False)
        raw = raw.loc[pd.to_numeric(raw.ID, errors="coerce").notna()].copy()
        if len(raw) != expected_rows:
            raise ValueError(f"Official Class {label} record count differs")
        frame = pd.DataFrame(index=raw.index)
        for original, name in TEXT.items():
            frame[name] = raw[original].astype("string").str.strip()
        for original, name in NUMERIC.items():
            frame[name] = pd.to_numeric(raw[original], errors="coerce")
        # All three files list physical heights 0.1/0.6/1.1 m. Retain 0.6 m
        # values where present, rather than silently using the first 0.1 m.
        for original, name in (("E1.Indoor Air Temperature (℃)", "ta"),
                               ("E2.Indoor Relative Humidity (%)", "rh"),
                               ("E3.Indoor Air Velocity (m/s)", "vel")):
            middle = f"{original}.1"
            frame[name] = pd.to_numeric(raw[middle], errors="coerce")
        frame["data_class"] = label
        frame["year"] = pd.to_datetime(frame.date, errors="coerce").dt.year
        frame["city_normalized"] = frame.city.map(city_name)
        frames.append(frame)
        manifests[label] = {"file": path.name, "sha256": expected_sha, "records": len(raw),
                            "thermal_preference_column": any("D4" in c for c in raw.columns)}
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "profile_id", [f"CTCD_record_{i:05d}" for i in range(len(result))])
    return result, manifests


def fingerprints(frame: pd.DataFrame) -> set[tuple]:
    numeric = frame[FINGERPRINT[:-1]].round(3)
    numeric["sex"] = frame.sex.astype("string").str.lower().str.strip()
    return set(numeric.dropna().itertuples(index=False, name=None))


def overlap_flags(frame: pd.DataFrame, measurements: pd.DataFrame,
                  metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict]:
    chinese = metadata.loc[metadata.country.astype("string").str.lower().eq("china")].copy()
    chinese["city_normalized"] = chinese.city.map(city_name)
    coarse = set(zip(chinese.city_normalized, chinese.year.astype(int), strict=True))
    city_year = pd.Series([(c, y) in coarse for c, y in
                          zip(frame.city_normalized, frame.year, strict=True)], index=frame.index)
    observed = measurements.merge(chinese[["building_id"]], on="building_id", validate="many_to_one")
    observed = observed.rename(columns={"gender": "sex"})
    keys = fingerprints(observed)
    probe = frame[FINGERPRINT[:-1]].round(3)
    probe["sex"] = frame.sex.astype("string").str.lower().str.strip()
    exact = pd.Series([all(pd.notna(x) for x in row) and row in keys
                      for row in probe.itertuples(index=False, name=None)], index=frame.index)
    report = {"ashrae_chinese_metadata_groups": len(chinese), "ashrae_chinese_rows": len(observed),
              "ashrae_chinese_years": sorted(chinese.year.astype(int).unique().tolist()),
              "city_year_rule": "Any Chinese ASHRAE metadata city and survey year, regardless of building type or contributor",
              "fingerprint_columns": FINGERPRINT, "numeric_rounding_decimals": 3,
              "fingerprint_rule": "Complete six-variable tuple across all Chinese ASHRAE records, irrespective of city/year",
              "interpretation": "Flags remove plausible overlap conservatively; lack of a matching tuple alone cannot establish source independence"}
    return city_year, exact, report


def align_office(frame: pd.DataFrame, measurements: pd.DataFrame,
                 metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    city_year, exact, overlap = overlap_flags(frame, measurements, metadata)
    audit = frame.copy()
    audit["overlap_city_year"] = city_year
    audit["overlap_fingerprint"] = exact
    audit["active_conditioning"] = frame.operation.isin(ACTIVE)
    audit["observed_temperature_c"] = frame.top.fillna(frame.ta)
    audit["temperature_source"] = np.where(frame.top.notna(), "operative", "air_fallback")
    stages = []
    keep = pd.Series(True, index=frame.index)
    for label, criterion in (
        ("all_official_records", keep),
        ("office_building", frame.building_type.str.lower().eq("office")),
        ("met_1_to_1_3", frame.met.between(1., 1.3)),
        ("valid_TSV_and_temperature", frame.thermal_sensation.between(-3, 3)
         & audit.observed_temperature_c.notna()),
        ("exclude_possible_ASHRAE_overlap", ~(city_year | exact)),
    ):
        before = int(keep.sum())
        keep &= criterion.fillna(False)
        stages.append({"stage": label, "remaining": int(keep.sum()), "excluded_here": before - int(keep.sum())})
    # Source row codes are not unique. Remove only identical released observed
    # content after harmonisation; do not infer people from those codes.
    dedup_columns = list(TEXT.values()) + list(NUMERIC.values()) + ["ta", "rh", "vel"]
    duplicates = audit.loc[keep].duplicated(dedup_columns, keep="first")
    duplicate_index = duplicates.index[duplicates]
    keep.loc[duplicate_index] = False
    stages.append({"stage": "exclude_identical_observed_rows", "remaining": int(keep.sum()),
                   "excluded_here": len(duplicate_index)})
    selected = audit.loc[keep].copy()
    selected["clo_analysis"] = selected.clo.where(selected.clo.ge(0))
    selected["clothing_imputation"] = np.where(selected.clo_analysis.notna(), "observed", "missing")
    for columns, label in ((["contributor", "season"], "source_season_median"),
                           (["contributor"], "source_median"),
                           (["climate", "season"], "climate_season_median"),
                           (["season"], "season_median")):
        median = selected.groupby(columns, dropna=False).clo_analysis.transform("median")
        replace = selected.clo_analysis.isna() & median.notna()
        selected.loc[replace, "clo_analysis"] = median[replace]
        selected.loc[replace, "clothing_imputation"] = label
    missing = selected.clo_analysis.isna()
    selected.loc[missing, "clo_analysis"] = selected.clo_analysis.median()
    selected.loc[missing, "clothing_imputation"] = "global_median"
    for suffix, coefficient in (("033", .33), ("050", .50), ("067", .67)):
        selected[f"reference_g{suffix}_c"] = (
            selected.observed_temperature_c - selected.thermal_sensation / coefficient)
    selected["centre_c"] = selected.reference_g050_c
    return selected, pd.DataFrame(stages), overlap


def interval_task(item: tuple) -> dict:
    reference, clo, met, scenes = item
    return virtual_interval(reference, clo, met, scenes)


def add_intervals(selected: pd.DataFrame, scenes: list[dict], workers: int) -> pd.DataFrame:
    keys = ["reference_g050_c", "clo_analysis", "met"]
    unique = selected[keys].drop_duplicates()
    items = [(float(r), float(c), float(m), scenes)
             for r, c, m in unique.itertuples(index=False, name=None)]
    if workers == 1:
        values = list(map(interval_task, items))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(interval_task, items, chunksize=32))
    interval = pd.concat([unique.reset_index(drop=True), pd.DataFrame(values)], axis=1)
    result = selected.merge(interval, on=keys, how="left", validate="many_to_one")
    result["lower_c"] = result.interval_lower_c
    result["upper_c"] = result.interval_upper_c
    result["controller_lower_c"] = result.lower_c
    result["controller_upper_c"] = result.upper_c
    return result


def describe(frame: pd.DataFrame, label: str) -> dict:
    result = {"scope": label, "records": len(frame), "cities": int(frame.city_normalized.nunique()),
              "contributors": int(frame.contributor.nunique()),
              "classes": frame.data_class.value_counts().to_dict(),
              "operations": frame.operation.fillna("missing").value_counts().to_dict(),
              "years": frame.year.value_counts(dropna=False).sort_index().to_dict()}
    for column in ("top", "ta", "thermal_sensation", "thermal_comfort", "thermal_acceptability", "clo", "met"):
        values = frame[column].dropna()
        result[column] = {"observed": len(values), "missing": len(frame) - len(values),
                          "min": float(values.min()), "median": float(values.median()), "max": float(values.max())}
    values = frame.thermal_acceptability.dropna()
    result["acceptability_sign"] = {"positive": int(values.gt(0).sum()), "negative": int(values.lt(0).sum()),
                                   "zero_unspecified": int(values.eq(0).sum())}
    result["observed_sensation_sign"] = {"cool": int(frame.thermal_sensation.lt(0).sum()),
                                         "neutral": int(frame.thermal_sensation.eq(0).sum()),
                                         "warm": int(frame.thermal_sensation.gt(0).sum())}
    result["scale_checks"] = {"TSV_outside_minus3_plus3": int((frame.thermal_sensation.notna()
                              & ~frame.thermal_sensation.between(-3., 3.)).sum()),
                              "TCV_outside_0_5": int((frame.thermal_comfort.notna()
                              & ~frame.thermal_comfort.between(0., 5.)).sum()),
                              "TAV_outside_minus1_plus1": int((frame.thermal_acceptability.notna()
                              & ~frame.thermal_acceptability.between(-1., 1.)).sum())}
    return result


def plot_distributions(records: pd.DataFrame, output: Path) -> None:
    """Emit only aggregate bins for figure inputs; no source record labels."""
    active = records.loc[records.active_conditioning]
    references = []
    for suffix, coefficient in (("033", .33), ("050", .50), ("067", .67)):
        edges = np.arange(10., 38., .5)
        values = active[f"reference_g{suffix}_c"].dropna().to_numpy()
        counts, _ = np.histogram(values, bins=edges)
        for lo, hi, count in zip(edges[:-1], edges[1:], counts, strict=True):
            references.append({"griffiths_coefficient": coefficient, "bin_lower_c": lo,
                               "bin_upper_c": hi, "bin_midpoint_c": (lo + hi) / 2,
                               "records": int(count), "scope_records": len(values),
                               "density_per_c": count / (counts.sum() * (hi - lo))})
    pd.DataFrame(references).to_csv(output / "CTCD_REFERENCE_HISTOGRAM.csv", index=False)
    executable = active.loc[active.main_control_eligible]
    widths = executable.upper_c - executable.lower_c
    edges = np.arange(0., 14.5, .5)
    counts, _ = np.histogram(widths, bins=edges)
    pd.DataFrame([{"bin_lower_c": lo, "bin_upper_c": hi, "bin_midpoint_c": (lo + hi) / 2,
                   "records": int(count), "scope_records": len(widths),
                   "density_per_c": count / (counts.sum() * (hi - lo))}
                  for lo, hi, count in zip(edges[:-1], edges[1:], counts, strict=True)]).to_csv(
                      output / "CTCD_INTERVAL_HISTOGRAM.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/upstream/chinese_thermal_comfort"))
    parser.add_argument("--ashrae-root", type=Path, default=Path("data/upstream/ashrae_db_ii"))
    parser.add_argument("--numeric-scenes", type=Path, default=Path("data/metadata/numeric_scene_spec_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("output/targeted_extension/ctcd"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    frame, raw = read_official(args.input_root)
    measurement = pd.read_csv(args.ashrae_root / "db_measurements_v2.1.0.csv.gz", low_memory=False)
    metadata = pd.read_csv(args.ashrae_root / "db_metadata.csv")
    selected, stages, overlap = align_office(frame, measurement, metadata)
    args.output_root.mkdir(parents=True, exist_ok=True)
    stages.to_csv(args.output_root / "CTCD_FILTER_STAGES.csv", index=False)
    city_year, exact, _ = overlap_flags(frame, measurement, metadata)
    overlap_rows = frame.loc[city_year | exact].copy()
    overlap_rows["city_year_flag"] = city_year[city_year | exact]
    overlap_rows["fingerprint_flag"] = exact[city_year | exact]
    overlap_rows.groupby(["data_class", "contributor", "city_normalized", "year",
                          "building_type", "city_year_flag", "fingerprint_flag"], dropna=False).size().rename(
                              "records").reset_index().to_csv(args.output_root / "CTCD_OVERLAP_GROUPS.csv", index=False)
    print(f"Aligned office records {len(selected)}; restoring numerical intervals", flush=True)
    scenes = numeric_scenes(args.numeric_scenes, numeric_only=True)
    result = add_intervals(selected, scenes, args.workers)
    result.to_csv(args.output_root / "CTCD_ALIGNED_OFFICE_RECORDS.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0}, float_format="%.17g")
    # Sampling uses the emitted numerical table, so pool records and the saved
    # normalization have the same CSV parsing boundary as the control runner.
    result = pd.read_csv(args.output_root / "CTCD_ALIGNED_OFFICE_RECORDS.csv.gz")
    broader_executable = result.loc[result.main_control_eligible].copy()
    broader_executable.to_csv(args.output_root / "CTCD_BROADER_EXECUTABLE_PROFILES.csv.gz", index=False,
                              compression={"method": "gzip", "mtime": 0}, float_format="%.17g")
    executable = result.loc[result.main_control_eligible & result.active_conditioning].copy()
    executable.to_csv(args.output_root / "CTCD_EXECUTABLE_PROFILES.csv.gz", index=False,
                      compression={"method": "gzip", "mtime": 0}, float_format="%.17g")
    plot_distributions(result, args.output_root)
    scopes = [describe(frame, "all_records"), describe(frame.loc[frame.building_type.str.lower().eq("office")], "all_office"),
              describe(result, "overlap_screened_office"),
              describe(result.loc[result.active_conditioning], "active_conditioning_office"),
              describe(result.loc[result.year.gt(2016)], "post_2016_office"),
              describe(result.loc[result.data_class.isin(["I", "II"]) & result.active_conditioning], "class_I_II_active_conditioning")]
    report = {"status": "PASS", "dataset_doi": "10.1038/s41597-023-02568-3",
              "official_osf": "https://osf.io/d465n/", "raw_inputs": raw,
              "license": "OSF node has no declared dataset license; article download/research-use statement does not establish a separate redistribution license",
              "scope_definitions": scopes, "filter_stages": stages.to_dict(orient="records"),
              "overlap": overlap, "executable_records": len(executable),
              "broader_executable_records": len(broader_executable),
              "collapsed_or_ineligible_intervals": len(result) - len(broader_executable),
              "active_conditioning_enumeration": sorted(ACTIVE),
              "numeric_scenes_sha256": sha256(args.numeric_scenes), "numeric_rounds": len(scenes),
              "observed_scales": {"TSV": "-3 cold to +3 hot, including released fractional votes",
                                   "TCV": "0 very comfortable to 5 very uncomfortable; preserve released fractional values",
                                   "TAV": "negative unacceptable, positive acceptable; do not reinterpret released +/-0.1 as missing",
                                   "preference": "not present in the three released CSVs"},
              "interpretation": {"population": "anonymous records, no verified subject/room/building identifiers",
                                 "intervals": "Griffiths centres then the same locked 19-scene numerical Li BAE adaptation; not observed personal ranges",
                                 "operation": "Physical main contrast uses explicit active-conditioning operation; broader-operation office records retained separately. CTCD survey operation is not ASHRAE annual air-conditioned metadata",
                                 "independence": "Conservative overlap-screened external dataset; post-2016 subset additionally separated from all recorded Chinese ASHRAE years"},
              "outputs": {p.name: sha256(p) for p in sorted(args.output_root.glob("CTCD_*.csv*"))}}
    (args.output_root / "CTCD_ALIGNMENT_QA.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"aligned_records": len(result), "executable_records": len(executable),
                      "output_root": str(args.output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
