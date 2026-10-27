"""Validate personal temperature ranges against observations held out in time.

The publication simulation uses a temperature centre and a temperature range as
two different demand representations.  This audit asks two separate questions:

1. Does the variable half-width produced by the retained virtual elicitation
   procedure predict unseen acceptability or preference better than fixed bands?
2. Can a simple range learned from past observed preference votes predict future
   votes in an independent longitudinal study?

No held-out record contributes to its person's centre, width or class thresholds.
The independent ENTH study is opened only after the fixed half-width is selected
from the ASHRAE development cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.utilities.generate_demand_inputs import (  # noqa: E402
    build_cohort,
    numeric_scenes,
    virtual_interval,
)


CENTRE_GRID = np.round(np.arange(16.0, 30.0 + 1e-9, 0.1), 1)
WIDTH_GRID = np.arange(0.5, 4.0 + 1e-9, 0.25)
FIXED_WIDTH_AUDIT = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def preference_prediction(temperature: np.ndarray, centre: float, half_width: float) -> np.ndarray:
    """Return -1=warmer, 0=no change and 1=cooler."""
    return np.where(
        temperature < centre - half_width,
        -1,
        np.where(temperature > centre + half_width, 1, 0),
    )


def fit_centre(temperature: np.ndarray, preference: np.ndarray, half_width: float) -> float:
    """Minimize class-balanced training error; take the median tied centre."""
    classes = np.unique(preference)
    weights = np.zeros(len(preference), dtype=float)
    for label in classes:
        mask = preference == label
        weights[mask] = 1.0 / (len(classes) * mask.sum())
    losses = np.array([
        np.sum(weights * (preference_prediction(temperature, centre, half_width) != preference))
        for centre in CENTRE_GRID
    ])
    tied = CENTRE_GRID[np.isclose(losses, losses.min(), atol=1e-12, rtol=0)]
    return float(np.median(tied))


def metrics(frame: pd.DataFrame, prediction: str, outcome: str = "preference") -> dict:
    local = frame[frame[outcome].notna()].copy()
    truth = local[outcome].astype(int).to_numpy()
    predicted = local[prediction].astype(int).to_numpy()
    balanced_accuracy = float(np.mean([
        np.mean(predicted[truth == label] == label) for label in np.unique(truth)
    ]))
    return {
        "records": len(local),
        "people": local.person.nunique(),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "accuracy": float(np.mean(truth == predicted)),
        **{
            f"recall_{label}": float(np.mean(predicted[truth == value] == value))
            if np.any(truth == value) else np.nan
            for label, value in (("warmer", -1), ("no_change", 0), ("cooler", 1))
        },
    }


def load_ashrae(measurements: Path, metadata: Path) -> pd.DataFrame:
    raw = pd.read_csv(measurements, low_memory=False)
    meta = pd.read_csv(metadata, low_memory=False)
    cohort, _ = build_cohort(raw, meta)
    cohort = cohort[cohort.subject_id.notna()].copy()
    publication = cohort.publication.astype("string").str.strip()
    contributor = cohort.contributor.astype("string").str.strip()
    source = publication.where(publication.notna() & publication.ne(""), contributor)
    cohort["person"] = (
        source.fillna("unknown") + "::" + cohort.building_id.astype(str)
        + "::" + cohort.subject_id.astype(str)
    )
    cohort["time"] = pd.to_datetime(cohort.timestamp, errors="coerce", utc=True)
    cohort["temperature"] = cohort.observed_temperature_c.astype(float)
    cohort["reference_centre"] = cohort.reference_g050_c.astype(float)
    cohort["acceptability"] = clean_text(cohort.thermal_acceptability).map(
        {"acceptable": 1, "unacceptable": 0}
    )
    cohort["preference"] = clean_text(cohort.thermal_preference).map(
        {"warmer": -1, "no change": 0, "cooler": 1}
    )
    return cohort


def compute_model_widths(cohort: pd.DataFrame, scene_path: Path) -> pd.DataFrame:
    scenes = numeric_scenes(scene_path, numeric_only=True)
    rows = []
    for row in cohort.itertuples(index=False):
        interval = virtual_interval(
            float(row.reference_centre), float(row.clo_analysis), float(row.met), scenes
        )
        rows.append({
            "record_id": row.record_id,
            "model_half_width_c": interval["interval_half_width_c"],
        })
    return pd.DataFrame(rows)


def chronological_person_split(frame: pd.DataFrame, minimum_records: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    splits = []
    for _, group in frame.groupby("person", sort=True):
        group = group.sort_values(["time", "record_id"], na_position="last")
        if len(group) < minimum_records:
            continue
        cut = max(3, int(np.floor(0.7 * len(group))))
        if cut < len(group):
            splits.append((group.iloc[:cut].copy(), group.iloc[cut:].copy()))
    return splits


def audit_virtual_widths(cohort: pd.DataFrame, widths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = cohort.merge(widths, on="record_id", validate="one_to_one")
    rows = []
    for training, test in chronological_person_split(frame):
        centre = float(training.reference_centre.median())
        model_width = float(training.model_half_width_c.median())
        for row in test.itertuples(index=False):
            item = {
                "person": row.person,
                "record_id": row.record_id,
                "acceptability": row.acceptability,
                "preference_binary": np.nan if pd.isna(row.preference) else int(row.preference == 0),
                "personal_centre_c": centre,
                "model_half_width_c": model_width,
                "model_margin_c": model_width - abs(float(row.temperature) - centre),
            }
            for width in FIXED_WIDTH_AUDIT:
                item[f"fixed_{width:g}_margin_c"] = width - abs(float(row.temperature) - centre)
            rows.append(item)
    predictions = pd.DataFrame(rows)
    summary = []
    for outcome in ("acceptability", "preference_binary"):
        local = predictions[predictions[outcome].notna()].copy()
        truth = local[outcome].astype(int)
        for model, margin in [("virtual_width", "model_margin_c")] + [
            (f"fixed_{width:g}C", f"fixed_{width:g}_margin_c") for width in FIXED_WIDTH_AUDIT
        ]:
            predicted = local[margin].ge(0).astype(int)
            sensitivity = float(predicted[truth.eq(1)].mean())
            specificity = float((1 - predicted[truth.eq(0)]).mean())
            summary.append({
                "outcome": outcome,
                "range_model": model,
                "records": len(local),
                "people": local.person.nunique(),
                "inside_fraction": float(predicted.mean()),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "roc_auc": float(roc_auc_score(truth, local[margin])),
            })
    return predictions, pd.DataFrame(summary)


def compare_variable_with_fixed_widths(summary: pd.DataFrame) -> tuple[list[dict], bool]:
    """Compare the variable-width rule with the best declared fixed band.

    This audit is descriptive.  It prevents a completed analysis from being
    labelled a validation pass when the variable-width rule does not exceed
    the strongest fixed-width comparator on the held-out outcome.
    """
    required = {"outcome", "range_model", "balanced_accuracy", "roc_auc"}
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Width summary is missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for outcome, group in summary.groupby("outcome", sort=True):
        variable = group[group.range_model.eq("virtual_width")]
        fixed = group[group.range_model.str.startswith("fixed_")]
        if len(variable) != 1 or fixed.empty:
            raise ValueError(f"{outcome}: expected one variable and fixed-width comparators")
        best = fixed.sort_values(
            ["balanced_accuracy", "roc_auc", "range_model"],
            ascending=[False, False, True],
        ).iloc[0]
        variable_row = variable.iloc[0]
        rows.append({
            "outcome": str(outcome),
            "variable_balanced_accuracy": float(variable_row.balanced_accuracy),
            "best_fixed_model": str(best.range_model),
            "best_fixed_balanced_accuracy": float(best.balanced_accuracy),
            "balanced_accuracy_difference": float(
                variable_row.balanced_accuracy - best.balanced_accuracy
            ),
            "variable_roc_auc": float(variable_row.roc_auc),
            "best_fixed_roc_auc": float(best.roc_auc),
            "roc_auc_difference": float(variable_row.roc_auc - best.roc_auc),
            "variable_outperformed_best_fixed": bool(
                variable_row.balanced_accuracy > best.balanced_accuracy
            ),
        })
    return rows, bool(rows and all(row["variable_outperformed_best_fixed"] for row in rows))


def develop_observed_range(cohort: pd.DataFrame) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    cohort = cohort[cohort.preference.notna()].copy()
    splits = chronological_person_split(cohort)
    predictions = []
    summaries = []
    for width in WIDTH_GRID:
        local_rows = []
        for training, test in splits:
            centre = fit_centre(
                training.temperature.to_numpy(float), training.preference.to_numpy(int), width
            )
            predicted = preference_prediction(test.temperature.to_numpy(float), centre, width)
            for row, value in zip(test.itertuples(index=False), predicted):
                local_rows.append({
                    "person": row.person,
                    "record_id": row.record_id,
                    "preference": int(row.preference),
                    "prediction": int(value),
                    "half_width_c": float(width),
                    "personal_centre_c": centre,
                })
        local = pd.DataFrame(local_rows)
        score = metrics(local, "prediction")
        summaries.append({"half_width_c": float(width), **score})
        predictions.append(local)
    summary = pd.DataFrame(summaries)
    best_score = summary.balanced_accuracy.max()
    selected = float(summary.loc[
        np.isclose(summary.balanced_accuracy, best_score), "half_width_c"
    ].min())
    all_predictions = pd.concat(predictions, ignore_index=True)
    return selected, summary, all_predictions[all_predictions.half_width_c.eq(selected)].copy()


def load_enth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["time"] = pd.to_datetime(frame.time, errors="coerce")
    frame["day"] = frame.time.dt.date
    frame["temperature"] = pd.to_numeric(frame.temp_zone, errors="coerce")
    frame["preference"] = pd.to_numeric(frame.thermal, errors="coerce").map(
        {9: -1, 10: 0, 11: 1}
    )
    frame = frame[
        pd.to_numeric(frame["indoor/outdoor"], errors="coerce").eq(11)
        & frame.temperature.notna() & frame.preference.notna() & frame.user_id.notna()
    ].copy()
    frame["person"] = frame.user_id.astype(str)
    return frame.sort_values(["person", "time"]).drop_duplicates(["person", "time"])


def external_enth_validation(frame: pd.DataFrame, half_width: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for day in sorted(frame.day.unique()):
        previous = frame[frame.day < day]
        current = frame[frame.day == day]
        if len(previous) < 5:
            continue
        population_centre = fit_centre(
            previous.temperature.to_numpy(float), previous.preference.to_numpy(int), half_width
        )
        for person, test in current.groupby("person", sort=True):
            training = previous[previous.person.eq(person)]
            if training.day.nunique() < 2 or len(training) < 5:
                continue
            personal_centre = fit_centre(
                training.temperature.to_numpy(float), training.preference.to_numpy(int), half_width
            )
            personal = preference_prediction(test.temperature.to_numpy(float), personal_centre, half_width)
            population = preference_prediction(
                test.temperature.to_numpy(float), population_centre, half_width
            )
            fixed_25 = preference_prediction(test.temperature.to_numpy(float), 25.0, half_width)
            for record, p_personal, p_population, p_fixed in zip(
                test.itertuples(index=False), personal, population, fixed_25
            ):
                rows.append({
                    "person": person,
                    "time": record.time,
                    "day": day,
                    "temperature_c": float(record.temperature),
                    "preference": int(record.preference),
                    "personal_prediction": int(p_personal),
                    "population_prediction": int(p_population),
                    "fixed_25_prediction": int(p_fixed),
                    "personal_centre_c": personal_centre,
                    "population_centre_c": population_centre,
                    "half_width_c": half_width,
                })
    predictions = pd.DataFrame(rows)
    summaries = []
    for model, column in (
        ("personal_history", "personal_prediction"),
        ("population_history", "population_prediction"),
        ("fixed_25C", "fixed_25_prediction"),
    ):
        pooled = metrics(predictions, column)
        person_accuracy = []
        person_balanced = []
        for _, group in predictions.groupby("person"):
            person_accuracy.append(float(np.mean(group.preference == group[column])))
            if group.preference.nunique() >= 2:
                truth = group.preference.to_numpy(int)
                predicted = group[column].to_numpy(int)
                person_balanced.append(float(np.mean([
                    np.mean(predicted[truth == label] == label)
                    for label in np.unique(truth)
                ])))
        summaries.append({
            "model": model,
            **pooled,
            "person_equal_accuracy": float(np.mean(person_accuracy)),
            "person_equal_balanced_accuracy": float(np.mean(person_balanced)),
            "people_with_multiple_observed_classes": len(person_balanced),
        })
    return predictions, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--enth", type=Path, required=True)
    parser.add_argument("--width-cache", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_paths = {
        "measurements": args.measurements.resolve(),
        "metadata": args.metadata.resolve(),
        "scenes": args.scenes.resolve(),
        "enth": args.enth.resolve(),
    }
    if args.width_cache and args.width_cache.is_file():
        input_paths["width_cache"] = args.width_cache.resolve()
    input_hashes = {name: sha256(path) for name, path in input_paths.items()}

    ashrae = load_ashrae(args.measurements.resolve(), args.metadata.resolve())
    if args.width_cache and args.width_cache.is_file():
        model_widths = pd.read_csv(args.width_cache)
    else:
        model_widths = compute_model_widths(ashrae, args.scenes.resolve())
    width_predictions, width_summary = audit_virtual_widths(ashrae, model_widths)
    width_comparison, variable_width_supported = compare_variable_with_fixed_widths(
        width_summary
    )
    selected_width, development, development_predictions = develop_observed_range(ashrae)
    enth = load_enth(args.enth.resolve())
    external_predictions, external_summary = external_enth_validation(enth, selected_width)

    deterministic_gzip = {"method": "gzip", "mtime": 0}
    model_widths.to_csv(
        output / "VIRTUAL_WIDTHS.csv.gz", index=False, compression=deterministic_gzip
    )
    width_predictions.to_csv(output / "VIRTUAL_WIDTH_HELDOUT_PREDICTIONS.csv.gz", index=False,
                             compression=deterministic_gzip)
    width_summary.to_csv(output / "VIRTUAL_WIDTH_HELDOUT_SUMMARY.csv", index=False)
    development.to_csv(output / "OBSERVED_RANGE_DEVELOPMENT.csv", index=False)
    development_predictions.to_csv(output / "OBSERVED_RANGE_DEVELOPMENT_PREDICTIONS.csv.gz",
                                   index=False, compression=deterministic_gzip)
    external_predictions.to_csv(output / "ENTH_EXTERNAL_PREDICTIONS.csv", index=False)
    external_summary.to_csv(output / "ENTH_EXTERNAL_SUMMARY.csv", index=False)
    produced = sorted(output.glob("*.csv")) + sorted(output.glob("*.csv.gz"))
    report = {
        "status": "AUDIT_COMPLETE",
        "selected_fixed_half_width_c": selected_width,
        "selection_rule": "highest ASHRAE chronological held-out balanced accuracy; narrower tie",
        "enth_opened_after_width_selection": True,
        "variable_width_outperformed_best_fixed_all_outcomes": variable_width_supported,
        "heldout_width_comparison": width_comparison,
        "virtual_width_interpretation": (
            "The held-out comparison tests predictive discrimination against declared fixed "
            "bands. It does not convert a published elicitation range into measured satisfaction."
        ),
        "records": {
            "ashrae_named_office_records": len(ashrae),
            "ashrae_people": ashrae.person.nunique(),
            "enth_eligible_records": len(enth),
            "enth_external_predictions": len(external_predictions),
            "enth_people": external_predictions.person.nunique(),
        },
        "input_sha256": input_hashes,
        "output_sha256": {path.name: sha256(path) for path in produced},
        "outputs": [path.name for path in produced],
    }
    (output / "PERSONAL_RANGE_VALIDATION.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
