"""Evaluate one-record demand estimates on the frozen participant holdouts."""

from demand_validation import (
    CODE, DEMAND_REBUILD, ROOT, bootstrap_ba, evaluation, load_ashrae,
    training_only_widths, write_json, record_status,
)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss

OUT = ROOT / "results/demand_validation/single_record"


def frozen_splits(frame, assignments, timed):
    lookup = frame.set_index("record_id", drop=False)
    result = []
    for person, group in assignments.groupby("person", sort=True):
        calibration = group[group.stage.str.contains("calibration")]
        later = group[~group.stage.str.contains("calibration")]
        early = lookup.loc[calibration.record_id].reset_index(drop=True)
        late = lookup.loc[later.record_id].reset_index(drop=True)
        if timed:
            early = early.sort_values(["time", "record_id"])
            assert early.time.max() < late.time.min()
        assert set(early.record_id).isdisjoint(late.record_id)
        result.append((person, str(group.partition.iloc[0]), early, late))
    return result


def one_record_features(splits, widths, prior, k):
    rows = []
    for person, partition, early, late in splits:
        first = early.iloc[0]
        raw = float(first.reference_centre)
        local_prior = prior.get(str(first.source_group), prior["__all__"])
        for observation in late.itertuples():
            rows.append({
                "person": person, "partition": partition,
                "record_id": observation.record_id,
                "source": str(first.source_group), "building_id": first.building_id,
                "calibration_record_id": first.record_id,
                "raw_center": raw, "shrink_center": (raw + k * local_prior) / (1 + k),
                "published_half_width": float(widths.loc[first.record_id]),
                "temperature": observation.temperature,
                "later_reference_centre": observation.reference_centre,
                "acceptability": observation.acceptability,
                "no_change": float(observation.preference == 0)
                if pd.notna(observation.preference) else np.nan,
            })
    return pd.DataFrame(rows)


def signed_margin(temperature, center, half_width):
    lower = np.clip(np.asarray(center) - half_width, 16.0, 30.0)
    upper = np.clip(np.asarray(center) + half_width, 16.0, 30.0)
    return np.minimum(np.asarray(temperature) - lower, upper - np.asarray(temperature))


def probability_model(margin, truth):
    return LogisticRegression(C=1, random_state=2026103101).fit(
        np.asarray(margin).reshape(-1, 1), truth.astype(int)
    )


def evaluate_design(frame, assignment_path, design, endpoint):
    out = OUT / design
    out.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(assignment_path)
    splits = frozen_splits(frame, assignments, design == "timed")
    development_people = {p for p, partition, _, _ in splits if partition == "development"}
    dev = frame[frame.person.isin(development_people)]
    prior = dev.groupby("source_group").reference_centre.median().to_dict()
    prior["__all__"] = float(dev.reference_centre.median())
    widths = pd.read_csv(DEMAND_REBUILD / "ASHRAE_RECORDS.csv.gz", low_memory=False)
    widths = widths.set_index("record_id").interval_half_width_c
    widths = training_only_widths(frame, splits, widths, out)
    tuning = []
    for k in [2, 5, 10, 20]:
        features = one_record_features(splits, widths, prior, k)
        d = features[features.partition.eq("development")].dropna(subset=[endpoint])
        distance = abs(d.temperature - d.shrink_center).to_numpy()
        model = probability_model(-distance, d[endpoint])
        tuning.append({"k": k, "development_brier": brier_score_loss(
            d[endpoint], model.predict_proba((-distance).reshape(-1, 1))[:, 1])})
    k = min(tuning, key=lambda item: item["development_brier"])["k"]
    features = one_record_features(splits, widths, prior, k)
    d = features[features.partition.eq("development")].dropna(subset=[endpoint])
    fitted = []
    selections = []
    for center in ["raw_center", "shrink_center"]:
        grid = []
        for width in [.5, 1, 1.5, 2, 2.5, 3]:
            margin = signed_margin(d.temperature, d[center], width)
            grid.append((balanced_accuracy_score(d[endpoint], margin >= 0), width))
        width = sorted(grid, key=lambda pair: (-pair[0], pair[1]))[0][1]
        for kind in ["fixed_width", "published_range"]:
            half_width = width if kind == "fixed_width" else d.published_half_width.to_numpy()
            margin = signed_margin(d.temperature, d[center], half_width)
            model = probability_model(margin, d[endpoint])
            fitted.append((center, kind, width, model))
            selections.append({"center_model": center, "range_model": kind,
                "selected_fixed_half_width_c": width,
                "logit_intercept": float(model.intercept_[0]),
                "logit_slope": float(model.coef_[0, 0])})
    # Persist every fitted choice before final-test predictions are evaluated.
    write_json(out / "FROZEN_SELECTIONS.json", {
        "development_only": True, "shrinkage_k": k, "source_priors": prior,
        "shrinkage_search": tuning, "models": selections,
        "frozen_assignment_sha256": __import__("hashlib").sha256(assignment_path.read_bytes()).hexdigest(),
    })
    features.to_csv(out / "FROZEN_PREDICTOR_FEATURES.csv.gz", index=False)
    t = features[features.partition.eq("final_test")].dropna(subset=[endpoint])
    metrics, predictions, calibration = [], [], []
    for center, kind, width, model in fitted:
        half_width = width if kind == "fixed_width" else t.published_half_width.to_numpy()
        margin = signed_margin(t.temperature, t[center], half_width)
        pred = (margin >= 0).astype(int)
        probability = model.predict_proba(margin.reshape(-1, 1))[:, 1]
        low, high = bootstrap_ba(t, endpoint, pred)
        metrics.append({"design": design, "endpoint": endpoint, "center_model": center,
            "range_model": kind, "selected_fixed_half_width_c": width,
            "shrinkage_k": k, **evaluation(t, endpoint, pred, probability),
            "ba_lower95": low, "ba_upper95": high})
        p = t[["person", "source", "building_id", "record_id", "calibration_record_id"]].copy()
        p["truth"], p["prediction"], p["probability"] = t[endpoint], pred, probability
        p["center_model"], p["range_model"] = center, kind
        predictions.append(p)
        for lower in np.arange(0, 1, .1):
            mask = (probability >= lower) & (probability < lower + .1 if lower < .9 else probability <= 1)
            calibration.append({"center_model": center, "range_model": kind,
                "bin_low": lower, "records": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else None,
                "observed_fraction": float(t[endpoint].to_numpy()[mask].mean()) if mask.any() else None})
    pd.DataFrame(metrics).to_csv(out / "FINAL_METRICS.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "FINAL_PREDICTIONS.csv.gz", index=False)
    pd.DataFrame(calibration).to_csv(out / "CALIBRATION.csv", index=False)
    if design == "timed":
        errors = []
        for center in ["raw_center", "shrink_center"]:
            for (person, partition), block in features.groupby(["person", "partition"]):
                errors.append({"person": person, "partition": partition, "center_model": center,
                    "source": block.source.iloc[0], "building_id": block.building_id.iloc[0],
                    "later_records": len(block),
                    "signed_later_proxy_shift_c": float((block.later_reference_centre - block[center]).mean()),
                    "mean_absolute_record_proxy_error_c": float(abs(block.later_reference_centre - block[center]).mean())})
        error_frame = pd.DataFrame(errors)
        error_frame.to_csv(out / "CONTINUOUS_PROXY_ERRORS.csv", index=False)
        summaries = []
        for center, group in error_frame.groupby("center_model"):
            development = group[group.partition.eq("development")]
            final = group[group.partition.eq("final_test")]
            half_width = float(development.signed_later_proxy_shift_c.abs().quantile(.9))
            rng = np.random.default_rng(2026103102)
            samples = []
            for _ in range(500):
                values = []
                for _, block in final.groupby(["source", "building_id"], dropna=False):
                    shifts = block.signed_later_proxy_shift_c.to_numpy()
                    values.extend(shifts[rng.integers(0, len(shifts), len(shifts))])
                samples.append(float(np.mean(abs(np.asarray(values)))))
            lower, upper = np.quantile(samples, [.025, .975])
            summaries.append({"center_model": center, "final_people": len(final),
                "mean_absolute_unit_proxy_error_c": float(final.signed_later_proxy_shift_c.abs().mean()),
                "mean_absolute_record_proxy_error_c": float(final.mean_absolute_record_proxy_error_c.mean()),
                "mae_lower95_c": lower, "mae_upper95_c": upper,
                "development_90pct_half_width_c": half_width,
                "final_proxy_coverage": float(final.signed_later_proxy_shift_c.abs().le(half_width).mean())})
        pd.DataFrame(summaries).to_csv(out / "CONTINUOUS_PROXY_METRICS.csv", index=False)
    write_json(out / "AUDIT.json", {"status": "completed", "calibration_records_per_person": 1,
        "frozen_partition_unchanged": True, "final_endpoint_records": len(t),
        "final_endpoint_people": int(t.person.nunique()),
        "evaluation": "Strict later records" if design == "timed" else "Undated held-out records"})
    return metrics


def main():
    raw = CODE / "data/upstream/ashrae_db_ii"
    frame = load_ashrae(raw / "db_measurements_v2.1.0.csv.gz", raw / "db_metadata.csv")
    results = evaluate_design(frame, ROOT / "results/demand_validation/FROZEN_HOLDOUT_ASSIGNMENTS.csv.gz", "timed", "no_change")
    results += evaluate_design(frame, ROOT / "results/demand_validation/undated_acceptability/ASSIGNMENTS.csv.gz", "undated", "acceptability")
    pd.DataFrame(results).to_csv(OUT / "ALL_FINAL_METRICS.csv", index=False)
    record_status("single_record_validation", "completed", {"models": len(results), "EnergyPlus_runs": 0})
    print(pd.DataFrame(results).to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
