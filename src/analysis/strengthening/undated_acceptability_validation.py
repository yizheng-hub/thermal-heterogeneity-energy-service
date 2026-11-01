from common import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, balanced_accuracy_score
from demand_validation import (
    person_partition,
    features,
    evaluation,
    bootstrap_ba,
    training_only_widths,
)


def main():
    out = ROOT / "results/demand_validation/undated_acceptability"
    out.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(
        ROOT / "results/data_support/IDENTIFIED_AS_HRAE_RECORDS.csv.gz", low_memory=False
    )
    f["time"] = pd.to_datetime(f.time, errors="coerce", utc=True)
    f = f[f.time.isna() & f.acceptability.notna()].copy()
    keys = [
        x
        for x in [
            "person",
            "time",
            "temperature",
            "thermal_sensation",
            "clo_analysis",
            "met",
            "acceptability",
            "preference",
        ]
        if x in f
    ]
    f = f.drop_duplicates(keys)
    splits = []
    assignments = []
    for person, g in f.groupby("person", sort=True):
        if len(g) < 5:
            continue
        seed = int(hashlib.sha256(("2026103101|" + person).encode()).hexdigest()[:16], 16)
        order = np.random.default_rng(seed).permutation(len(g))
        g = g.iloc[order]
        k = max(3, int(0.7 * len(g)))
        early = g.iloc[:k]
        late = g.iloc[k:]
        if late.empty:
            continue
        part = person_partition(person)
        splits.append((person, part, early, late))
        for stage, rows in [("calibration_records", early), ("evaluation_records", late)]:
            assignments.extend(
                [
                    {"person": person, "record_id": i, "partition": part, "stage": stage}
                    for i in rows.record_id
                ]
            )
    pd.DataFrame(assignments).to_csv(out / "ASSIGNMENTS.csv.gz", index=False)
    people = {person for person, part, _, _ in splits if part == "development"}
    dev = f[f.person.isin(people)]
    prior = dev.groupby("source_group").reference_centre.median().to_dict()
    prior["__all__"] = float(dev.reference_centre.median())
    widths = (
        pd.read_csv(DEMAND_REBUILD / "ASHRAE_RECORDS.csv.gz", low_memory=False)
        .set_index("record_id")
        .interval_half_width_c
    )
    widths = training_only_widths(f, splits, widths, out)
    selection = []
    for k in [2, 5, 10, 20]:
        rows = features(splits, widths, prior, k)
        d = rows[rows.partition.eq("development")]
        x = abs(d.temperature - d.shrink_center).to_numpy().reshape(-1, 1)
        model = LogisticRegression(C=1, random_state=2026103101).fit(x, d.acceptability.astype(int))
        selection.append(
            {
                "k": k,
                "development_brier": brier_score_loss(
                    d.acceptability, model.predict_proba(x)[:, 1]
                ),
            }
        )
    k = min(selection, key=lambda x: x["development_brier"])["k"]
    rows = features(splits, widths, prior, k)
    d = rows[rows.partition.eq("development")]
    t = rows[rows.partition.eq("final_test")]
    summary = []
    predictions = []
    calibration = []
    models = []
    for center in ["raw_center", "shrink_center"]:
        dd = abs(d.temperature - d[center]).to_numpy()
        td = abs(t.temperature - t[center]).to_numpy()
        chosen = sorted(
            [
                (balanced_accuracy_score(d.acceptability, dd <= w), w)
                for w in [0.5, 1, 1.5, 2, 2.5, 3]
            ],
            key=lambda a: (-a[0], a[1]),
        )[0][1]
        for kind in ["fixed_width", "published_range"]:
            dw = (
                np.full(len(d), chosen)
                if kind == "fixed_width"
                else d.published_half_width.to_numpy()
            )
            tw = (
                np.full(len(t), chosen)
                if kind == "fixed_width"
                else t.published_half_width.to_numpy()
            )
            dx = (dw - dd).reshape(-1, 1)
            tx = (tw - td).reshape(-1, 1)
            model = LogisticRegression(C=1, random_state=2026103101).fit(
                dx, d.acceptability.astype(int)
            )
            prob = model.predict_proba(tx)[:, 1]
            prediction = (tx[:, 0] >= 0).astype(int)
            metric = evaluation(t, "acceptability", prediction, prob)
            ci = bootstrap_ba(t, "acceptability", prediction)
            summary.append(
                {
                    "endpoint": "acceptability",
                    "center_model": center,
                    "range_model": kind,
                    "selected_fixed_half_width_c": chosen,
                    "shrinkage_k": k,
                    **metric,
                    "ba_lower95": ci[0],
                    "ba_upper95": ci[1],
                }
            )
            p = t[["person", "source", "building_id", "record_id", "acceptability"]].copy()
            p["center_model"] = center
            p["range_model"] = kind
            p["prediction"] = prediction
            p["probability"] = prob
            predictions.append(p)
            models.append(
                {
                    "center_model": center,
                    "range_model": kind,
                    "half_width_c": chosen,
                    "logit_intercept": float(model.intercept_[0]),
                    "logit_slope": float(model.coef_[0, 0]),
                }
            )
            for lo in np.arange(0, 1, 0.1):
                mask = (prob >= lo) & (prob <= lo + 0.1 if lo > 0.8 else prob < lo + 0.1)
                calibration.append(
                    {
                        "center_model": center,
                        "range_model": kind,
                        "bin_low": lo,
                        "records": int(mask.sum()),
                        "predicted": float(prob[mask].mean()) if mask.any() else None,
                        "observed": float(t.acceptability.to_numpy()[mask].mean())
                        if mask.any()
                        else None,
                    }
                )
    pd.DataFrame(summary).to_csv(out / "FINAL_METRICS.csv", index=False)
    pd.concat(predictions).to_csv(out / "FINAL_PREDICTIONS.csv.gz", index=False)
    pd.DataFrame(calibration).to_csv(out / "FINAL_CALIBRATION.csv", index=False)
    write_json(
        out / "FROZEN_MODELS.json",
        {"development_selection": selection, "chosen_k": k, "models": models},
    )
    audit = {
        "status": "completed",
        "design": "participant-separated undated-record holdout; no temporal extrapolation claim",
        "eligible_participants": len(splits),
        "development_participants": sum(x[1] == "development" for x in splits),
        "final_test_participants": t.person.nunique(),
        "final_test_records": len(t),
        "calibration_and_evaluation_records_disjoint": True,
        "final_labels_used_for_model_or_width_selection": False,
        "protocol": "analysis_plan_amendment_01.yaml",
        "metrics": summary,
    }
    write_json(out / "AUDIT.json", audit)
    record_status("undated_acceptability_validation", "completed", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
