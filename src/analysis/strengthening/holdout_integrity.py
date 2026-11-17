"""Check participant separation, calibration/evaluation timing and covariate support."""

from demand_validation import *


def main():
    assignments = pd.read_csv(OUT / "FROZEN_HOLDOUT_ASSIGNMENTS.csv.gz")
    assert assignments.record_id.is_unique
    assert assignments.groupby("person").partition.nunique().eq(1).all()
    upstream = CODE / "data/upstream"
    f = load_ashrae(
        upstream / "ashrae_db_ii/db_measurements_v2.1.0.csv.gz",
        upstream / "ashrae_db_ii/db_metadata.csv",
    )
    joined = assignments.merge(
        f[["record_id", "time", "clothing_imputation", "temperature"]],
        on="record_id",
        validate="one_to_one",
    )
    rows = []
    for person, g in joined.groupby("person", sort=True):
        early = g[g.stage.eq("early_calibration")]
        late = g[g.stage.eq("later_evaluation")]
        assert early.time.max() < late.time.min()
        early_days = set(early.time.dt.date)
        late_days = set(late.time.dt.date)
        rows.append(
            {
                "person": person,
                "partition": g.partition.iloc[0],
                "early_records": len(early),
                "later_records": len(late),
                "early_days": len(early_days),
                "later_days": len(late_days),
                "days_shared_across_split": len(early_days & late_days),
                "minimum_forecast_gap_hours": (late.time.min() - early.time.max()).total_seconds()
                / 3600,
                "calibration_temperature_span_c": early.temperature.max() - early.temperature.min(),
                "early_clothing_imputed_records": int(
                    early.clothing_imputation.ne("observed").sum()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "HOLDOUT_TIME_BLOCK_AND_COVARIATE_AUDIT.csv", index=False)
    report = {
        "status": "completed",
        "participant_partitions_disjoint": True,
        "record_partitions_disjoint": True,
        "all_evaluations_strictly_after_calibration": True,
        "units": len(rows),
        "units_with_calendar_days_shared_across_split": sum(
            r["days_shared_across_split"] > 0 for r in rows
        ),
        "units_with_imputed_early_clothing": sum(
            r["early_clothing_imputed_records"] > 0 for r in rows
        ),
        "final_median_minimum_forecast_gap_hours": float(
            pd.DataFrame(rows).query("partition=='final_test'").minimum_forecast_gap_hours.median()
        ),
    }
    write_json(OUT / "HOLDOUT_INTEGRITY_AUDIT.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
