from common import *
import pandas as pd
import numpy as np

sys.path.insert(0, str(SNAPSHOT))
from src.analysis.validation.personal_range_validation import (  # noqa: E402
    load_enth,
    fit_centre,
    preference_prediction,
)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402


def main():
    source = (
        CODE
        / "data/upstream/shared_room_observations/longitudinal_personal_thermal_comfort_zenodo_5502441/enth_tabular_merged.csv"
    )
    f = load_enth(source)
    out = ROOT / "results/demand_validation"
    records = []
    support = []
    selected = json.loads((out / "FROZEN_MODEL_SELECTION.json").read_text())
    widths = {
        x["width_c"]
        for key, x in selected["models"].items()
        if key.startswith("no_change|") and key.endswith("fixed_width")
    }
    for person, g in f.groupby("person", sort=True):
        g = g[g.time.notna()].sort_values("time")
        times = np.sort(g.time.unique())
        if len(times) < 3 or len(g) < 5:
            continue
        cut = times[min(max(1, int(0.7 * len(times))), len(times) - 1)]
        early = g[g.time < cut]
        late = g[g.time >= cut]
        support.append(
            {
                "person": person,
                "records": len(g),
                "distinct_timestamps": len(times),
                "distinct_days": g.day.nunique(),
                "temperature_span_c": g.temperature.max() - g.temperature.min(),
                "early_records": len(early),
                "later_records": len(late),
                "early_preference_classes": early.preference.nunique(),
                "curve_support": bool(
                    len(early) >= 3
                    and early.temperature.max() - early.temperature.min() >= 1
                    and early.preference.nunique() >= 2
                ),
            }
        )
        if len(early) < 3 or not support[-1]["curve_support"]:
            continue
        for width in sorted(widths):
            center = fit_centre(
                early.temperature.to_numpy(float), early.preference.to_numpy(int), width
            )
            for row in late.itertuples():
                pred = preference_prediction(np.array([row.temperature]), center, width)[0]
                records.append(
                    {
                        "person": person,
                        "time": row.time,
                        "temperature": row.temperature,
                        "truth": row.preference,
                        "half_width_c": width,
                        "early_fitted_center_c": center,
                        "prediction": pred,
                        "no_change_truth": int(row.preference == 0),
                        "no_change_prediction": int(pred == 0),
                        "no_change_score": width - abs(row.temperature - center),
                    }
                )
    pd.DataFrame(support).to_csv(out / "ENTH_INDEPENDENT_SUPPORT.csv", index=False)
    predictions = pd.DataFrame(records)
    summary = []
    if len(predictions):
        predictions.to_csv(out / "ENTH_INDEPENDENT_LATER_PREDICTIONS.csv.gz", index=False)
        for width, g in predictions.groupby("half_width_c"):
            summary.append(
                {
                    "half_width_c": width,
                    "later_records": len(g),
                    "participants": g.person.nunique(),
                    "no_change_balanced_accuracy": balanced_accuracy_score(
                        g.no_change_truth, g.no_change_prediction
                    )
                    if g.no_change_truth.nunique() == 2
                    else None,
                    "no_change_auroc": roc_auc_score(g.no_change_truth, g.no_change_score)
                    if g.no_change_truth.nunique() == 2
                    else None,
                }
            )
    pd.DataFrame(summary).to_csv(out / "ENTH_INDEPENDENT_LATER_METRICS.csv", index=False)
    audit = {
        "status": "completed" if summary else "partial",
        "raw_merged_records": len(pd.read_csv(source)),
        "eligible_indoor_preference_records": len(f),
        "eligible_participants": f.person.nunique(),
        "curve_supported_participants": sum(x["curve_support"] for x in support),
        "thresholds": "widths selected from ASHRAE development participants; ENTH early data fit center only; later responses used only for evaluation",
        "scope": "fixed-width observed-preference response rule, not validation of published Personal-range width without its required inputs",
        "metrics": summary,
    }
    write_json(out / "ENTH_INDEPENDENT_VALIDATION.json", audit)
    record_status("external_preference_validation", audit["status"], audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
