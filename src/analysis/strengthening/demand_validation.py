from common import *
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss

sys.path.insert(0, str(SNAPSHOT))
from src.analysis.validation.personal_range_validation import load_ashrae  # noqa: E402
from src.utilities.generate_demand_inputs import numeric_scenes, virtual_interval  # noqa: E402
from src.analysis.targeted_extension.ctcd_alignment import read_official  # noqa: E402

OUT = ROOT / "results/demand_validation"


def support(frame):
    f = frame.copy()
    f["date_block"] = f.time.dt.floor("D")
    f["month"] = f.time.dt.month
    f["season"] = ((f["month"] % 12) // 3).astype("Int64")
    rows = []
    for person, g in f.groupby("person", sort=True):
        rows.append(
            {
                "person": person,
                "source": str(g.source_group.iloc[0]),
                "building_id": g.building_id.iloc[0],
                "records": len(g),
                "valid_timestamps": int(g.time.notna().sum()),
                "distinct_times": g.time.nunique(),
                "independent_days": g.date_block.nunique(),
                "seasons": g.season.nunique(),
                "temperature_min_c": g.temperature.min(),
                "temperature_max_c": g.temperature.max(),
                "temperature_span_c": g.temperature.max() - g.temperature.min(),
                "acceptability_votes": int(g.acceptability.notna().sum()),
                "acceptability_classes": g.acceptability.nunique(),
                "preference_votes": int(g.preference.notna().sum()),
                "no_change_votes": int(g.preference.eq(0).sum()),
                "no_change_binary_classes": g.loc[g.preference.notna(), "preference"]
                .eq(0)
                .nunique(),
            }
        )
    return pd.DataFrame(rows)


def person_partition(person):
    h = hashlib.sha256(("2026103101|" + person).encode()).hexdigest()
    return "development" if int(h[:8], 16) / 2**32 < 0.7 else "final_test"


def training_only_widths(frame, splits, widths, output):
    early = pd.concat([a for _, _, a, _ in splits], ignore_index=True)
    missing = early[pd.to_numeric(early.clo, errors="coerce").isna()]
    dev_ids = {p for p, part, _, _ in splits if part == "development"}
    dev = frame[frame.person.isin(dev_ids)].copy()
    dev["clo"] = pd.to_numeric(dev.clo, errors="coerce")
    prior = dev.groupby("source_group").clo.median().to_dict()
    fallback = float(dev.clo.median())
    assert np.isfinite(fallback)
    result = widths.copy()
    rows = []
    if len(missing):
        scenes = numeric_scenes(
            CODE / "data/metadata/numeric_scene_spec_v1.json", numeric_only=True
        )
        for row in missing.itertuples():
            clothing = prior.get(row.source_group, np.nan)
            if not np.isfinite(clothing):
                clothing = fallback
            value = virtual_interval(
                float(row.reference_centre), float(clothing), float(row.met), scenes
            )["interval_half_width_c"]
            rows.append(
                {
                    "record_id": row.record_id,
                    "person": row.person,
                    "development_imputed_clo": clothing,
                    "old_half_width_c": float(result.loc[row.record_id]),
                    "new_half_width_c": float(value),
                }
            )
            result.loc[row.record_id] = value
    pd.DataFrame(
        rows,
        columns=[
            "record_id",
            "person",
            "development_imputed_clo",
            "old_half_width_c",
            "new_half_width_c",
        ],
    ).to_csv(output / "DEVELOPMENT_ONLY_CLOTHING_WIDTH_REPLAY.csv", index=False)
    return result


def chronological_splits(frame):
    rows = []
    for person, g in frame.groupby("person", sort=True):
        g = g[g.time.notna()].sort_values(["time", "record_id"])
        times = np.sort(g.time.unique())
        if len(g) < 5 or len(times) < 3:
            continue
        k = max(1, int(np.floor(0.7 * len(times))))
        cut = times[min(k, len(times) - 1)]
        early = g[g.time < cut]
        late = g[g.time >= cut]
        if len(early) < 3 or late.empty:
            continue
        assert early.time.max() < late.time.min()
        rows.append((person, person_partition(person), early, late))
    return rows


def features(splits, widths, prior, k):
    rows = []
    for person, partition, early, late in splits:
        raw = float(early.reference_centre.median())
        source = str(early.source_group.iloc[0])
        p = prior.get(source, prior["__all__"])
        shrink = (len(early) * raw + k * p) / (len(early) + k)
        model_width = float(early.record_id.map(widths).median())
        for row in late.itertuples():
            rows.append(
                {
                    "person": person,
                    "partition": partition,
                    "record_id": row.record_id,
                    "source": source,
                    "building_id": row.building_id,
                    "time": row.time,
                    "temperature": row.temperature,
                    "raw_center": raw,
                    "shrink_center": shrink,
                    "early_records": len(early),
                    "published_half_width": model_width,
                    "acceptability": row.acceptability,
                    "no_change": float(row.preference == 0) if pd.notna(row.preference) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def evaluation(f, y, pred, prob):
    truth = f[y].to_numpy(int)
    v = pred.astype(int)
    if len(np.unique(truth)) < 2:
        return {"status": "single_endpoint_class", "records": len(f)}
    return {
        "status": "completed",
        "records": len(f),
        "participants": f.person.nunique(),
        "balanced_accuracy": balanced_accuracy_score(truth, v),
        "auroc": roc_auc_score(truth, prob),
        "brier": brier_score_loss(truth, prob),
        "observed_positive_fraction": truth.mean(),
        "predicted_positive_fraction": v.mean(),
    }


def bootstrap_ba(f, y, pred):
    table = f[["person", "source", "building_id", y]].copy()
    table["pred"] = pred
    table["tp"] = (table[y].eq(1) & table.pred.eq(1)).astype(int)
    table["fn"] = (table[y].eq(1) & table.pred.eq(0)).astype(int)
    table["tn"] = (table[y].eq(0) & table.pred.eq(0)).astype(int)
    table["fp"] = (table[y].eq(0) & table.pred.eq(1)).astype(int)
    g = table.groupby(["source", "building_id", "person"], dropna=False)[
        ["tp", "fn", "tn", "fp"]
    ].sum()
    rng = np.random.default_rng(2026103102)
    values = []
    for _ in range(500):
        total = np.zeros(4)
        for _, block in g.groupby(level=[0, 1], sort=True):
            a = block.to_numpy()
            total += a[rng.integers(0, len(a), len(a))].sum(axis=0)
        tp, fn, tn, fp = total
        if tp + fn > 0 and tn + fp > 0:
            values.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else [None, None]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = CODE / "data/upstream"
    f = load_ashrae(
        upstream / "ashrae_db_ii/db_measurements_v2.1.0.csv.gz",
        upstream / "ashrae_db_ii/db_metadata.csv",
    )
    keys = [
        c
        for c in [
            "person",
            "time",
            "temperature",
            "thermal_sensation",
            "clo_analysis",
            "met",
            "acceptability",
            "preference",
        ]
        if c in f
    ]
    duplicate = f.duplicated(keys, keep="first")
    f.loc[duplicate, ["record_id", "person", "time"]].to_csv(
        OUT / "DUPLICATE_RECORDS.csv", index=False
    )
    raw_support = support(f)
    raw_support.to_csv(OUT / "PERSON_SUPPORT_BEFORE_DEDUP.csv", index=False)
    f = f.loc[~duplicate].copy()
    support(f).to_csv(OUT / "PERSON_SUPPORT.csv", index=False)
    reused = f.groupby("subject_id", dropna=False).agg(
        qualified_people=("person", "nunique"),
        sources=("source_group", "nunique"),
        buildings=("building_id", "nunique"),
    )
    reused[reused.qualified_people.gt(1)].to_csv(OUT / "RAW_ID_REUSE.csv")
    ctcd, metadata = read_official(upstream / "chinese_thermal_comfort")
    write_json(
        OUT / "CTCD_SUPPORT.json",
        {
            "records": len(ctcd),
            "classes": metadata,
            "participant_identity_supported": False,
            "building_identity_supported": False,
            "date_nonmissing": int(pd.to_datetime(ctcd.date, errors="coerce").notna().sum()),
            "thermal_acceptability_values": ctcd.thermal_acceptability.value_counts(
                dropna=False
            ).to_dict(),
            "no_change_preference_present": False,
            "use": "source-level patterns; not personal longitudinal validation",
        },
    )
    splits = chronological_splits(f)
    assignments = []
    for person, partition, early, late in splits:
        assignments.extend(
            [
                {"person": person, "record_id": rid, "partition": partition, "stage": stage}
                for stage, g in [("early_calibration", early), ("later_evaluation", late)]
                for rid in g.record_id
            ]
        )
    pd.DataFrame(assignments).to_csv(OUT / "FROZEN_HOLDOUT_ASSIGNMENTS.csv.gz", index=False)
    dev_people = {p for p, s, _, _ in splits if s == "development"}
    dev = f[f.person.isin(dev_people)]
    prior = dev.groupby("source_group").reference_centre.median().to_dict()
    prior["__all__"] = float(dev.reference_centre.median())
    widths = (
        pd.read_csv(DEMAND_REBUILD / "ASHRAE_RECORDS.csv.gz", low_memory=False)
        .set_index("record_id")
        .interval_half_width_c
    )
    widths = training_only_widths(f, splits, widths, OUT)
    configs = []
    for k in [2, 5, 10, 20]:
        feat = features(splits, widths, prior, k)
        d = feat[feat.partition.eq("development")]
        scores = []
        for endpoint in ["acceptability", "no_change"]:
            g = d.dropna(subset=[endpoint])
            x = abs(g.temperature - g.shrink_center).to_numpy().reshape(-1, 1)
            if g[endpoint].nunique() < 2:
                continue
            model = LogisticRegression(C=1, random_state=2026103101).fit(x, g[endpoint].astype(int))
            scores.append(brier_score_loss(g[endpoint], model.predict_proba(x)[:, 1]))
        configs.append(
            {"k": k, "development_mean_brier": float(np.mean(scores)) if scores else np.nan}
        )
    pd.DataFrame(configs).to_csv(OUT / "DEVELOPMENT_SHRINKAGE_SELECTION.csv", index=False)
    if not configs or not splits:
        record_status(
            "independent_demand_validation",
            "partial",
            {"participants": len(f.person.unique())},
            ["No supported strict chronological split."],
        )
        return
    k = min(configs, key=lambda r: r["development_mean_brier"])["k"]
    feat = features(splits, widths, prior, k)
    feat.to_csv(OUT / "FROZEN_LATER_RECORD_FEATURES.csv.gz", index=False)
    summaries = []
    predictions = []
    calibration = []
    selected = {}
    for endpoint in ["acceptability", "no_change"]:
        d = feat[feat.partition.eq("development")].dropna(subset=[endpoint])
        t = feat[feat.partition.eq("final_test")].dropna(subset=[endpoint])
        if d[endpoint].nunique() < 2 or t[endpoint].nunique() < 2:
            continue
        for center in ["raw_center", "shrink_center"]:
            devdist = abs(d.temperature - d[center]).to_numpy()
            testdist = abs(t.temperature - t[center]).to_numpy()
            widths_grid = [0.5, 1, 1.5, 2, 2.5, 3]
            grid = [(balanced_accuracy_score(d[endpoint], devdist <= w), w) for w in widths_grid]
            width = sorted(grid, key=lambda a: (-a[0], a[1]))[0][1]
            for kind in ["fixed_width", "published_range"]:
                dw = (
                    np.full(len(d), width)
                    if kind == "fixed_width"
                    else d.published_half_width.to_numpy()
                )
                tw = (
                    np.full(len(t), width)
                    if kind == "fixed_width"
                    else t.published_half_width.to_numpy()
                )
                dx = (dw - devdist).reshape(-1, 1)
                tx = (tw - testdist).reshape(-1, 1)
                valid = np.isfinite(dx[:, 0])
                model = LogisticRegression(C=1, random_state=2026103101).fit(
                    dx[valid], d[endpoint].to_numpy(int)[valid]
                )
                prob = model.predict_proba(tx)[:, 1]
                pred = (tx[:, 0] >= 0).astype(int)
                metric = evaluation(t, endpoint, pred, prob)
                ci = bootstrap_ba(t, endpoint, pred)
                item = {
                    "endpoint": endpoint,
                    "center_model": center,
                    "range_model": kind,
                    "selected_fixed_half_width_c": width,
                    "shrinkage_k": k,
                    **metric,
                    "ba_lower95": ci[0],
                    "ba_upper95": ci[1],
                }
                summaries.append(item)
                selected[endpoint + "|" + center + "|" + kind] = {
                    "width_c": width,
                    "logit_intercept": model.intercept_[0],
                    "logit_slope": model.coef_[0, 0],
                }
                p = t[["person", "source", "building_id", "record_id", endpoint]].copy()
                p["endpoint"] = endpoint
                p["truth"] = p[endpoint]
                p["center_model"] = center
                p["range_model"] = kind
                p["prediction"] = pred
                p["probability"] = prob
                predictions.append(p.drop(columns=[endpoint]))
                for lo in np.arange(0, 1, 0.1):
                    mask = (prob >= lo) & (prob < lo + 0.1 if lo < 0.9 else prob <= 1)
                    calibration.append(
                        {
                            "endpoint": endpoint,
                            "center_model": center,
                            "range_model": kind,
                            "bin_low": lo,
                            "records": int(mask.sum()),
                            "mean_probability": float(prob[mask].mean()) if mask.any() else None,
                            "observed_fraction": float(t[endpoint].to_numpy()[mask].mean())
                            if mask.any()
                            else None,
                        }
                    )
    pd.DataFrame(summaries).to_csv(OUT / "FINAL_TEST_METRICS.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            OUT / "FINAL_TEST_PREDICTIONS.csv.gz", index=False
        )
    pd.DataFrame(calibration).to_csv(OUT / "FINAL_TEST_CALIBRATION.csv", index=False)
    write_json(
        OUT / "FROZEN_MODEL_SELECTION.json",
        {
            "selected_using": "development participants only",
            "shrinkage_k": k,
            "models": selected,
            "source_priors": prior,
        },
    )
    residual = []
    for person, partition, early, late in splits:
        residual.append(
            {
                "person": person,
                "partition": partition,
                "source": str(early.source_group.iloc[0]),
                "building_id": early.building_id.iloc[0],
                "later_minus_early_reconstructed_center_c": float(
                    late.reference_centre.median() - early.reference_centre.median()
                ),
                "early_temperature_span_c": float(
                    early.temperature.max() - early.temperature.min()
                ),
            }
        )
    pd.DataFrame(residual).to_csv(OUT / "TEMPORAL_CENTER_DIFFERENCES.csv", index=False)
    summary = {
        "known_id_records_before_dedup": len(f) + int(duplicate.sum()),
        "exact_duplicates_removed": int(duplicate.sum()),
        "qualified_person_units": f.person.nunique(),
        "chronological_participants": len(splits),
        "development_participants": sum(s == "development" for _, s, _, _ in splits),
        "final_test_participants": sum(s == "final_test" for _, s, _, _ in splits),
        "final_test_records": int(feat.partition.eq("final_test").sum()),
        "timestamp_rule": "strict earlier/later distinct timestamp blocks; same timestamp never crosses cut",
        "participant_definition": "source-qualified building and subject ID; qualified unit, not identity externally verified",
        "center_difference_interpretation": "temporal reconstructed-reference variation, not ground-truth center-estimation error",
        "status": "completed" if summaries else "partial",
    }
    write_json(OUT / "DEMAND_VALIDATION_AUDIT.json", summary)
    record_status("independent_demand_validation", summary["status"], summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
