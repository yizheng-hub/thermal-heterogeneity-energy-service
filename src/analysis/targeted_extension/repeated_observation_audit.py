"""Audit repeated observations without treating observed temperature span as a range.

The audit reports how many subject-within-building series can support a
time-ordered holdout.  It also performs an internal label-consistency check:
model-derived interval width and centre are estimated from the earlier
records only, then evaluated on later records with explicit acceptability or
preference labels.  The check is diagnostic; it does not identify a person's
complete acceptable-temperature boundary.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
UTILITIES = HERE.parents[2] / "utilities"
sys.path.insert(0, str(UTILITIES))
from generate_demand_inputs import build_cohort  # noqa: E402


RELEASE_SEED = "thermal-heterogeneity-level1-20261022-5d7c9e31"


def release_profile_id(record_id: object, seed: str = RELEASE_SEED) -> str:
    key = hashlib.sha256(seed.encode("utf-8")).digest()
    message = f"archive_record\x00{record_id}".encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()[:15]
    token = base64.b32encode(digest).decode("ascii").rstrip("=").lower()
    return f"pr_{token}"


def binary_metrics(predicted: pd.Series, observed: pd.Series) -> dict[str, float | int]:
    valid = observed.notna()
    prediction = predicted.loc[valid].astype(bool)
    truth = observed.loc[valid].astype(bool)
    tp = int((prediction & truth).sum())
    tn = int((~prediction & ~truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "records": int(len(truth)),
        "positive_prevalence": float(truth.mean()) if len(truth) else np.nan,
        "accuracy": float((prediction == truth).mean()) if len(truth) else np.nan,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
    }


def time_ordered_holdout(frame: pd.DataFrame, minimum_records: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    for subject_key, group in frame.groupby("subject_key", sort=True):
        group = group.sort_values(["timestamp_parsed", "record_id"], kind="mergesort")
        if len(group) < minimum_records:
            continue
        cut = max(3, int(np.floor(0.70 * len(group))))
        cut = min(cut, len(group) - 1)
        training = group.iloc[:cut]
        centre = float(training.interval_centre_c.median())
        half_width = float(training.interval_half_width_c.median())
        for row in group.iloc[cut:].itertuples(index=False):
            rows.append({
                "subject_key": subject_key,
                "observed_temperature_c": float(row.observed_temperature_c),
                "training_centre_c": centre,
                "training_model_half_width_c": half_width,
                "thermal_acceptability": str(row.thermal_acceptability).strip().lower(),
                "thermal_preference": str(row.thermal_preference).strip().lower(),
            })
    return pd.DataFrame(rows)


def support_table(frame: pd.DataFrame) -> pd.DataFrame:
    series = frame.groupby("subject_key", sort=True).agg(
        records=("record_id", "size"),
        temperature_min_c=("observed_temperature_c", "min"),
        temperature_max_c=("observed_temperature_c", "max"),
        acceptability_records=("thermal_acceptability", lambda x: int(x.notna().sum())),
        preference_records=("thermal_preference", lambda x: int(x.notna().sum())),
        sensation_records=("thermal_sensation", lambda x: int(x.notna().sum())),
    )
    series["observed_temperature_span_c"] = (
        series.temperature_max_c - series.temperature_min_c
    )
    rows = []
    for threshold in (1, 2, 3, 5, 8, 10):
        eligible = series.records.ge(threshold)
        rows.append({
            "minimum_records": threshold,
            "subject_within_building_series": int(eligible.sum()),
            "records_in_series": int(series.loc[eligible, "records"].sum()),
            "series_with_temperature_span_ge_2_c": int(
                (eligible & series.observed_temperature_span_c.ge(2.0)).sum()
            ),
            "series_with_acceptability_records_ge_threshold": int(
                series.acceptability_records.ge(threshold).sum()
            ),
            "series_with_preference_records_ge_threshold": int(
                series.preference_records.ge(threshold).sum()
            ),
        })
    return pd.DataFrame(rows)


def consistency_table(holdout: pd.DataFrame) -> pd.DataFrame:
    acceptability = holdout.thermal_acceptability.map(
        {"acceptable": True, "unacceptable": False}
    )
    preference = holdout.thermal_preference.map(
        {"no change": True, "warmer": False, "cooler": False}
    )
    rows = []
    widths: list[tuple[str, pd.Series | float]] = [
        ("model_derived_training_median", holdout.training_model_half_width_c),
        ("fixed_plus_minus_1_0_c", 1.0),
        ("fixed_plus_minus_1_5_c", 1.5),
        ("fixed_plus_minus_2_0_c", 2.0),
        ("fixed_plus_minus_2_5_c", 2.5),
        ("fixed_plus_minus_3_0_c", 3.0),
    ]
    for method, width in widths:
        inside = (holdout.observed_temperature_c - holdout.training_centre_c).abs().le(width)
        for endpoint, observed in (
            ("explicit_acceptability", acceptability),
            ("preference_no_change", preference),
        ):
            metrics = binary_metrics(inside, observed)
            valid_subjects = holdout.loc[observed.notna(), "subject_key"].nunique()
            rows.append({
                "interval_rule": method,
                "heldout_endpoint": endpoint,
                "subject_series": int(valid_subjects),
                **metrics,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    measurements = pd.read_csv(args.measurements.resolve(), low_memory=False)
    metadata = pd.read_csv(args.metadata.resolve(), low_memory=False)
    cohort, scope = build_cohort(measurements, metadata)
    cohort = cohort.loc[cohort.subject_id.notna()].copy()
    cohort["release_profile_id"] = cohort.record_id.map(release_profile_id)
    intervals = pd.read_csv(args.intervals.resolve())
    cohort = cohort.merge(intervals, on="release_profile_id", how="left", validate="one_to_one")
    records_without_executable_interval = int(cohort.interval_half_width_c.isna().sum())
    cohort = cohort.loc[cohort.interval_half_width_c.notna()].copy()
    cohort["subject_key"] = (
        cohort.building_id.astype(str) + "::" + cohort.subject_id.astype(str)
    )
    cohort["timestamp_parsed"] = pd.to_datetime(cohort.timestamp, errors="coerce", utc=True)

    support = support_table(cohort)
    holdout = time_ordered_holdout(cohort)
    consistency = consistency_table(holdout)
    args.output.mkdir(parents=True, exist_ok=True)
    support.to_csv(args.output / "REPEATED_OBSERVATION_SUPPORT.csv", index=False)
    consistency.to_csv(args.output / "HELDOUT_LABEL_CONSISTENCY.csv", index=False)
    report = {
        "status": "PASS",
        "office_cohort_records": int(scope["records"]),
        "subject_identified_records": int(len(cohort)),
        "subject_identified_records_without_executable_interval": records_without_executable_interval,
        "subject_within_building_series": int(cohort.subject_key.nunique()),
        "holdout_records": int(len(holdout)),
        "holdout_subject_series": int(holdout.subject_key.nunique()),
        "interpretation": (
            "Internal time-ordered label consistency only; observed temperatures do not "
            "identify complete personal acceptable-temperature boundaries."
        ),
        "raw_identifiers_exported": False,
    }
    (args.output / "REPEATED_OBSERVATION_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
