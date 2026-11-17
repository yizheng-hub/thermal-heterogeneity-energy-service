"""Evaluate frozen centers against later observational demand-center proxies."""

from demand_validation import *


def main():
    out = ROOT / "results/demand_validation/temporal_centers"
    out.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(OUT / "FROZEN_LATER_RECORD_FEATURES.csv.gz")
    upstream = CODE / "data/upstream"
    observed = load_ashrae(
        upstream / "ashrae_db_ii/db_measurements_v2.1.0.csv.gz",
        upstream / "ashrae_db_ii/db_metadata.csv",
    )
    actual = observed[["record_id", "reference_centre"]].rename(
        columns={"reference_centre": "later_observational_center_c"}
    )
    frame = features.merge(actual, on="record_id", validate="one_to_one")
    rows = []
    units = []
    for center in ["raw_center", "shrink_center"]:
        local = frame[
            [
                "person",
                "source",
                "building_id",
                "partition",
                "record_id",
                "later_observational_center_c",
                center,
            ]
        ].copy()
        local["model"] = center
        local["signed_error_c"] = local.later_observational_center_c - local[center]
        units.append(
            local.groupby(["person", "source", "building_id", "partition", "model"], dropna=False)
            .agg(
                later_records=("record_id", "count"),
                predicted_center_c=(center, "first"),
                later_mean_center_c=("later_observational_center_c", "mean"),
                signed_unit_center_shift_c=("signed_error_c", "mean"),
                mean_absolute_record_error_c=("signed_error_c", lambda a: abs(a).mean()),
            )
            .reset_index()
        )
        frame[center + "_signed_error_c"] = local.signed_error_c
    unit = pd.concat(units, ignore_index=True)
    for model, g in unit.groupby("model"):
        development = g[g.partition.eq("development")]
        test = g[g.partition.eq("final_test")]
        halfwidth = float(development.signed_unit_center_shift_c.abs().quantile(0.9))
        rng = np.random.default_rng(2026103102)
        bootstrap = []
        for _ in range(500):
            selected = []
            for _, block in test.groupby(["source", "building_id"], dropna=False):
                values = block.signed_unit_center_shift_c.to_numpy()
                selected.extend(values[rng.integers(0, len(values), len(values))])
            bootstrap.append(float(np.mean(abs(np.asarray(selected)))))
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        rows.append(
            {
                "model": model,
                "development_units": len(development),
                "final_units": len(test),
                "final_mean_absolute_unit_center_error_c": float(
                    test.signed_unit_center_shift_c.abs().mean()
                ),
                "final_unit_center_mae_lower95_c": float(low),
                "final_unit_center_mae_upper95_c": float(high),
                "final_median_absolute_unit_center_error_c": float(
                    test.signed_unit_center_shift_c.abs().median()
                ),
                "final_mean_absolute_later_record_error_c": float(
                    test.mean_absolute_record_error_c.mean()
                ),
                "final_signed_bias_c": float(test.signed_unit_center_shift_c.mean()),
                "development_selected_90pct_half_width_c": halfwidth,
                "final_unit_center_coverage": float(
                    test.signed_unit_center_shift_c.abs().le(halfwidth).mean()
                ),
                "endpoint": "Reconstructed later demand-center proxy; not a directly observed latent comfort target.",
            }
        )
    frame.to_csv(out / "LATER_RECORD_PREDICTIONS.csv.gz", index=False)
    unit.to_csv(out / "PAIRED_PERSON_TIME_CENTER_SHIFTS.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "FINAL_CONTINUOUS_CENTER_METRICS.csv", index=False)
    development = unit[unit.partition.eq("development") & unit.model.eq("raw_center")]
    development.to_csv(out / "DEVELOPMENT_EMPIRICAL_TEMPORAL_SHIFT_POOL.csv", index=False)
    report = {
        "status": "completed",
        "final_units": int(unit[unit.partition.eq("final_test")].person.nunique()),
        "endpoint": "Early estimates predicting strictly later reconstructed observational centers; temporal drift and observation noise are not separately identifiable.",
        "final_records": int(frame[frame.partition.eq("final_test")].record_id.nunique()),
        "profile_linkage": "Raw-source identities are reconstructed in audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz. Validation support and chronological overlap must be assessed per linked person; these held-out participants do not validate every simulated profile.",
    }
    write_json(out / "AUDIT.json", report)
    record_status("continuous_center_temporal_validation", "completed", report)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
