"""Compare personal early information with development-only group centers."""
from common import *
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from demand_validation import evaluation
from single_record_validation import frozen_splits, signed_margin, probability_model

GROUP_MODELS = [scope + "_" + statistic for scope in ("pooled", "study", "building")
                for statistic in ("mean", "median")] + ["source_prior_only"]
MODELS = ["raw_center", "shrink_center", *GROUP_MODELS]
WIDTHS = [.5, 1, 1.5, 2, 2.5, 3]


def group_centers(frame):
    columns = ["person", "partition", "source", "building_id", "raw_center"]
    units = frame[columns].drop_duplicates().copy()
    if units.person.duplicated().any():
        raise ValueError("A calibration participant has inconsistent center or group metadata")
    development = units[units.partition.eq("development")]
    result, fits = frame.copy(), []
    for unit in units.itertuples(index=False):
        pool = development[development.person.ne(unit.person)] if unit.partition == "development" else development
        if pool.empty:
            raise ValueError("No independent development participants support a group center")
        for scope in ("pooled", "study", "building"):
            local, level = pool, "pooled"
            study = pool[pool.source.eq(unit.source)]
            if scope != "pooled" and len(study) >= 3:
                local, level = study, "study"
            if scope == "building" and pd.notna(unit.building_id):
                building = study[study.building_id.eq(unit.building_id)]
                if len(building) >= 3:
                    local, level = building, "building"
            for statistic in ("mean", "median"):
                model = scope + "_" + statistic
                center = float(getattr(local.raw_center, statistic)())
                result.loc[result.person.eq(unit.person), model] = center
                fits.append({"person": unit.person, "partition": unit.partition, "model": model,
                    "source": unit.source, "building_id": unit.building_id, "center_c": center,
                    "support_level": level, "development_participants": len(local)})
        if "source_prior_only" in frame:
            values = frame.loc[frame.person.eq(unit.person), "source_prior_only"].unique()
            if len(values) != 1:
                raise ValueError("A participant has inconsistent frozen source priors")
            fits.append({"person": unit.person, "partition": unit.partition, "model": "source_prior_only",
                "source": unit.source, "building_id": unit.building_id, "center_c": values[0],
                "support_level": "original_development_source_or_global_prior", "development_participants": np.nan})
    return result, pd.DataFrame(fits)


def paired_bootstrap(frame, predictions):
    """Keep every response of a sampled participant and all model pairs together."""
    units = frame[["person", "source", "building_id"]].drop_duplicates().reset_index(drop=True)
    if units.person.duplicated().any():
        raise ValueError("Endpoint participant crosses source-building strata")
    truth = frame.truth.to_numpy(int)
    counts = np.zeros((len(units), predictions.shape[1], 4))
    for index, unit in enumerate(units.itertuples()):
        local = frame.person.eq(unit.person).to_numpy()
        y, p = truth[local, None], predictions[local]
        counts[index] = np.stack([((y == 1) & (p == 1)).sum(axis=0),
            ((y == 1) & (p == 0)).sum(axis=0), ((y == 0) & (p == 0)).sum(axis=0),
            ((y == 0) & (p == 1)).sum(axis=0)], axis=-1)
    strata = [group.index.to_numpy() for _, group in units.groupby(["source", "building_id"], dropna=False, sort=True)]
    rng = np.random.default_rng(2026110101)
    draws = []
    for _ in range(500):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        tp, fn, tn, fp = counts[indices].sum(axis=0).T
        if ((tp + fn) > 0).all() and ((tn + fp) > 0).all():
            draws.append(.5 * (tp / (tp + fn) + tn / (tn + fp)))
    return np.asarray(draws)


def compare(frame, endpoint, design, out):
    augmented, fits = group_centers(frame)
    fits.to_csv(out / "GROUP_CENTER_FITS.csv", index=False)
    development = augmented[augmented.partition.eq("development")].dropna(subset=[endpoint])
    test = augmented[augmented.partition.eq("final_test")].dropna(subset=[endpoint]).copy()
    if set(development.person) & set(test.person):
        raise ValueError("Development and final-test participants overlap")
    if development[endpoint].nunique() != 2 or test[endpoint].nunique() != 2:
        raise ValueError("Both response classes are required for this matched comparison")
    test["truth"] = test[endpoint].astype(int)

    def width_for(model):
        return sorted([(balanced_accuracy_score(development[endpoint],
            signed_margin(development.temperature, development[model], width) >= 0), width)
            for width in WIDTHS], key=lambda pair: (-pair[0], pair[1]))[0][1]

    shared_width = width_for("raw_center")
    fitted, frozen = [], []
    for policy in ("shared_width", "tuned_width"):
        for model in MODELS:
            width = shared_width if policy == "shared_width" else width_for(model)
            margin = signed_margin(development.temperature, development[model], width)
            logit = probability_model(margin, development[endpoint])
            fitted.append((policy, model, width, logit))
            frozen.append({"width_policy": policy, "center_model": model, "half_width_c": width,
                "intercept": float(logit.intercept_[0]), "slope": float(logit.coef_[0, 0])})
    # Persist all model/width choices before evaluating final response outcomes.
    write_json(out / "FROZEN_MODEL_SELECTIONS.json", {"models": frozen,
        "development_participants": int(development.person.nunique()),
        "final_responses_used_for_selection": False, "group_centers_exclude_final_personal_responses": True})
    metrics, predictions, calibration, binary = [], [], [], []
    for policy, model, width, logit in fitted:
        margin = signed_margin(test.temperature, test[model], width)
        prediction = (margin >= 0).astype(int)
        probability = logit.predict_proba(margin.reshape(-1, 1))[:, 1]
        binary.append(prediction)
        metrics.append({"design": design, "endpoint": endpoint, "width_policy": policy,
            "center_model": model, "half_width_c": width, **evaluation(test, endpoint, prediction, probability)})
        local = test[["person", "source", "building_id", "record_id", "truth"]].copy()
        local["center_model"], local["width_policy"] = model, policy
        local["center_c"], local["prediction"], local["probability"] = test[model], prediction, probability
        predictions.append(local)
        for index in range(10):
            selected = (probability >= index / 10) & (probability < (index + 1) / 10 if index < 9 else probability <= 1)
            calibration.append({"center_model": model, "width_policy": policy, "bin_low": index / 10,
                "records": int(selected.sum()), "mean_probability": float(probability[selected].mean()) if selected.any() else None,
                "observed_fraction": float(test.truth.to_numpy()[selected].mean()) if selected.any() else None})
    draws = paired_bootstrap(test, np.column_stack(binary))
    if not len(draws):
        raise ValueError("No paired bootstrap draw contains both response classes")
    for index, metric in enumerate(metrics):
        metric["ba_lower95"], metric["ba_upper95"] = np.quantile(draws[:, index], [.025, .975])
        metric["valid_paired_bootstrap_draws"] = len(draws)
    contrasts = []
    for reference, item in enumerate(metrics):
        if item["center_model"] not in {"raw_center", "shrink_center"}:
            continue
        for comparator, group in enumerate(metrics):
            if group["width_policy"] != item["width_policy"] or group["center_model"] not in GROUP_MODELS:
                continue
            difference = draws[:, reference] - draws[:, comparator]
            low, high = np.quantile(difference, [.025, .975])
            contrasts.append({"design": design, "endpoint": endpoint, "width_policy": item["width_policy"],
                "individual_model": item["center_model"], "group_model": group["center_model"],
                "records": len(test), "participants": test.person.nunique(),
                "balanced_accuracy_difference": item["balanced_accuracy"] - group["balanced_accuracy"],
                "paired_ba_difference_lower95": low, "paired_ba_difference_upper95": high,
                "brier_difference": item["brier"] - group["brier"], "valid_bootstrap_draws": len(draws)})
    pd.DataFrame(metrics).to_csv(out / "FINAL_METRICS.csv", index=False)
    pd.DataFrame(contrasts).to_csv(out / "PAIRED_INFORMATION_CONTRASTS.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "FINAL_PREDICTIONS.csv.gz", index=False)
    pd.DataFrame(calibration).to_csv(out / "CALIBRATION.csv", index=False)
    return metrics, contrasts


def main():
    out = ROOT / "results/group_information_validation"
    base = ROOT / "results/demand_validation"
    paths = {"timed_multi": base / "FROZEN_LATER_RECORD_FEATURES.csv.gz",
        "timed_single": base / "single_record/timed/FROZEN_PREDICTOR_FEATURES.csv.gz",
        "undated_single": base / "single_record/undated/FROZEN_PREDICTOR_FEATURES.csv.gz",
        "undated_assignments": base / "undated_acceptability/ASSIGNMENTS.csv.gz",
        "undated_model": base / "undated_acceptability/FROZEN_MODELS.json",
        "timed_multi_model": base / "FROZEN_MODEL_SELECTION.json",
        "timed_single_model": base / "single_record/timed/FROZEN_SELECTIONS.json",
        "undated_single_model": base / "single_record/undated/FROZEN_SELECTIONS.json",
        "identified_records": ROOT / "results/data_support/IDENTIFIED_AS_HRAE_RECORDS.csv.gz",
        "protocol": CODE / "data/metadata/strengthening_protocol/analysis_plan_amendment_30_group_prediction.yaml"}
    inputs = {p.relative_to(CODE).as_posix(): digest(p) for p in paths.values()}
    functions = [main, group_centers, evaluation, frozen_splits, signed_margin, probability_model, digest]
    sources = {Path(inspect.getfile(f)).resolve().relative_to(CODE).as_posix(): digest(inspect.getfile(f)) for f in functions}
    receipt = out / "AUDIT.json"
    if receipt.is_file():
        audit = json.loads(receipt.read_text())
        if audit.get("status") != "completed" or audit["input_sha256"] != inputs or audit["source_sha256"] != sources:
            raise ValueError("Group prediction cache changed; preserve the prior attempt")
        if any(digest(CODE / name) != sha for name, sha in audit["output_sha256"].items()):
            raise ValueError("Group prediction output changed; preserve the prior attempt")
        print("Group prediction: verified completed cache", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Preserve the incomplete group-prediction attempt before retrying")
    frames = {name: pd.read_csv(paths[name]) for name in ("timed_multi", "timed_single", "undated_single")}
    for design, frame in frames.items():
        fitted = json.loads(paths[design + "_model"].read_text())
        priors = fitted["source_priors"]
        frame["source_prior_only"] = frame.source.map(priors).fillna(priors["__all__"])
        n = frame.early_records.to_numpy() if design == "timed_multi" else 1
        expected = (n * frame.raw_center + fitted["shrinkage_k"] * frame.source_prior_only) / (n + fitted["shrinkage_k"])
        if not np.allclose(frame.shrink_center, expected, rtol=0, atol=1e-10):
            raise ValueError("Pure source prior does not reproduce the frozen shrinkage model")
    observed = pd.read_csv(paths["identified_records"], low_memory=False)
    observed["time"] = pd.to_datetime(observed.time, errors="coerce", utc=True)
    observed = observed[observed.time.isna() & observed.acceptability.notna()]
    dedup = [name for name in ["person", "time", "temperature", "thermal_sensation", "clo_analysis",
             "met", "acceptability", "preference"] if name in observed]
    observed = observed.drop_duplicates(dedup)
    splits = frozen_splits(observed, pd.read_csv(paths["undated_assignments"]), False)
    dev_people = {person for person, partition, _, _ in splits if partition == "development"}
    development = observed[observed.person.isin(dev_people)]
    prior = development.groupby("source_group").reference_centre.median().to_dict()
    fallback = float(development.reference_centre.median())
    k = json.loads(paths["undated_model"].read_text())["chosen_k"]
    rows = []
    for person, partition, early, late in splits:
        center = float(early.reference_centre.median())
        source = str(early.source_group.iloc[0])
        shrink = (len(early) * center + k * prior.get(source, fallback)) / (len(early) + k)
        for row in late.itertuples():
            rows.append({"person": person, "partition": partition, "source": source,
                "building_id": row.building_id, "record_id": row.record_id, "temperature": row.temperature,
                "raw_center": center, "shrink_center": shrink, "acceptability": row.acceptability})
    frames["undated_multi"] = pd.DataFrame(rows)
    frames["undated_multi"]["source_prior_only"] = frames["undated_multi"].source.map(prior).fillna(fallback)
    all_metrics, all_contrasts = [], []
    for design in ("timed_multi", "timed_single", "undated_multi", "undated_single"):
        folder = out / design
        folder.mkdir(parents=True)
        endpoint = "no_change" if design.startswith("timed") else "acceptability"
        metrics, contrasts = compare(frames[design], endpoint, design, folder)
        all_metrics.extend(metrics)
        all_contrasts.extend(contrasts)
        print("GROUP_PREDICTION_COMPLETED", design, metrics[0]["records"], metrics[0]["participants"], flush=True)
    pd.DataFrame(all_metrics).to_csv(out / "ALL_FINAL_METRICS.csv", index=False)
    pd.DataFrame(all_contrasts).to_csv(out / "ALL_PAIRED_INFORMATION_CONTRASTS.csv", index=False)
    for name, sha in {**inputs, **sources}.items():
        if digest(CODE / name) != sha:
            raise ValueError("Group prediction input or source changed during execution")
    write_json(receipt, {"status": "completed", "designs": 4, "model_metric_rows": len(all_metrics),
        "paired_contrasts": len(all_contrasts), "EnergyPlus_runs": 0, "input_sha256": inputs,
        "source_sha256": sources, "output_sha256": {p.relative_to(CODE).as_posix(): digest(p)
            for p in out.rglob("*") if p.is_file() and p != receipt},
        "scope": "Participant-held-out request/acceptability prediction with fixed strata; not latent comfort truth or global population confidence intervals",
        "development_group_fits": "The six early-center group models leave the predicted development participant out; the additional exact original source-prior comparator retains its already-fitted development prior",
        "final_group_fits": "Six models use development early centers only; the additional original source/global prior uses its original development-record fit. Final participant early and late responses fit neither type of group center"})


if __name__ == "__main__":
    main()
