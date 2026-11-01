"""Reproduce the locked shared-room preference analysis from public raw files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (0.0, 0.25, 0.5, 0.75)


def load_votes(path, dataset):
    raw = pd.read_csv(path, low_memory=False)
    data = pd.DataFrame(index=raw.index)
    data["input_row"] = raw.index
    data["time"] = pd.to_datetime(raw.time, errors="coerce", utc=True).dt.tz_convert(
        "Asia/Singapore"
    )
    data["user"] = raw.user_id.astype("string").str.strip()
    data["vote"] = pd.to_numeric(
        raw["thermal_cozie" if dataset == "NUS" else "thermal"], errors="coerce"
    ).map({9: 1, 10: 0, 11: -1})
    if dataset == "NUS":
        data["room"] = raw.room.astype("string")
        data["temperature_c"] = pd.to_numeric(raw.temperature_sensing, errors="coerce")
        eligible = data.user.str.fullmatch(r"cresh(?:0[1-9]|[12][0-9]|30)", na=False)
        eligible &= np.isfinite(data.temperature_c)
    else:
        data["room"] = raw.building_name.astype("string") + "|" + raw.space_id.astype("string")
        data["temperature_c"] = pd.to_numeric(raw.temp_zone, errors="coerce")
        eligible = pd.to_numeric(raw["indoor/outdoor"], errors="coerce").eq(11)
    eligible &= data[["time", "user", "room", "vote"]].notna().all(axis=1)
    return data.loc[eligible].copy(), {
        "raw_records": len(raw),
        "valid_records": int(eligible.sum()),
    }


def windows(votes, minutes, minimum_users, temperature_limit=None):
    data = votes.copy()
    data["window_start"] = data.time.dt.floor(f"{minutes}min")
    data["midpoint_distance"] = abs(
        data.time - data.window_start - pd.Timedelta(minutes=minutes / 2)
    )
    data = data.sort_values(["midpoint_distance", "time", "input_row"], kind="mergesort")
    data = data.drop_duplicates(["room", "window_start", "user"])
    rows = []
    retained = []
    for (room, start), group in data.groupby(["room", "window_start"], sort=True):
        spread = group.temperature_c.max() - group.temperature_c.min()
        enough = len(group) >= minimum_users
        homogeneous = temperature_limit is None or spread <= temperature_limit + 1e-12
        vote = group.vote.to_numpy(float)
        full, mean = np.mean(abs(vote)), abs(np.mean(vote))
        opposed = bool((vote > 0).any() and (vote < 0).any())
        index = len(rows)
        row = {
            "window_id": index,
            "room": room,
            "window_start": start,
            "day": start.date(),
            "votes": len(group),
            "temperature_span_c": spread,
            "minimum_users_met": enough,
            "temperature_limit_met": homogeneous,
            "included": enough and homogeneous,
            "full_burden": full,
            "averaged_burden": mean,
            "gap": full - mean,
            "category": "Opposed" if opposed else "Same" if len(np.unique(vote)) == 1 else "Mixed",
            "cooler_votes": int((vote < 0).sum()),
            "no_change_votes": int((vote == 0).sum()),
            "warmer_votes": int((vote > 0).sum()),
        }
        rows.append(row)
        if row["included"]:
            retained.append(group.assign(window_id=index))
    all_windows = pd.DataFrame(rows)
    return all_windows, pd.concat(retained, ignore_index=True) if retained else data.iloc[:0].copy()


def bootstrap(windows_frame, votes, replicates=2000):
    users = sorted(votes.user.unique())
    rooms = sorted(windows_frame.room.unique())
    days = sorted(windows_frame.day.unique())
    ids = windows_frame.window_id.to_list()
    membership = np.zeros((len(ids), len(users)))
    positive, negative = membership.copy(), membership.copy()
    for row in votes.itertuples():
        i, j = ids.index(row.window_id), users.index(row.user)
        membership[i, j] = 1
        positive[i, j] = row.vote > 0
        negative[i, j] = row.vote < 0
    rng = np.random.default_rng(20261021)
    results = []
    for scheme in ["crossed", "date", "room", "user"]:
        sizes = [
            len(days) if scheme in ["crossed", "date"] else 0,
            len(rooms) if scheme in ["crossed", "room"] else 0,
            len(users) if scheme in ["crossed", "user"] else 0,
        ]
        # Draw date, room and user weights in that order within each replicate.
        draws = rng.exponential(size=(replicates, sum(sizes)))
        boundaries = np.cumsum([0, *sizes])
        dw, rw, uw = [
            draws[:, boundaries[i] : boundaries[i + 1]]
            if sizes[i]
            else np.ones((replicates, width))
            for i, width in enumerate([len(days), len(rooms), len(users)])
        ]
        total = uw @ membership.T
        pos, neg = uw @ positive.T, uw @ negative.T
        full, mean = (pos + neg) / total, abs(pos - neg) / total
        weights = (
            dw[:, [days.index(x) for x in windows_frame.day]]
            * rw[:, [rooms.index(x) for x in windows_frame.room]]
        )
        statistics = {"mean_gap": ((full - mean) * weights).sum(axis=1) / weights.sum(axis=1)}
        fixed_weights = weights * total / membership.sum(axis=1)
        for threshold in THRESHOLDS:
            label = (windows_frame.averaged_burden <= threshold) & (
                windows_frame.full_burden > threshold
            )
            statistics[f"false_acceptance_fraction_at_{threshold:g}"] = (
                fixed_weights * label.to_numpy()
            ).sum(axis=1) / fixed_weights.sum(axis=1)
        for endpoint, values in statistics.items():
            results.append(
                {
                    "cluster_scheme": scheme,
                    "endpoint": endpoint,
                    "p025": np.quantile(values, 0.025),
                    "p975": np.quantile(values, 0.975),
                    "replicates": replicates,
                    "seed": 20261021,
                    "method": "vote reweighting"
                    if endpoint == "mean_gap"
                    else "fixed window labels with crossed cluster weights",
                }
            )
    return pd.DataFrame(results)


def run(upstream, output):
    sources = {
        "NUS": (
            upstream / "humans_as_sensor_2019/2019-11-15_cozie_full_masked.csv",
            "d4adec0aa4937479d5bf946450f491b295cdf99a9070fd5b5d57fe30b86d9c11",
        ),
        "ENTH": (
            upstream
            / "longitudinal_personal_thermal_comfort_zenodo_5502441/enth_tabular_merged.csv",
            "fe6d5b6978fd273007964fa865e2af8dd79e3bafdecefcb00286f9650e83c50a",
        ),
    }
    summary, intervals, thresholds, flow = [], [], [], []
    for dataset, (path, expected) in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Shared-room raw input version differs: {dataset}")
        votes, support = load_votes(path, dataset)
        for minutes in [15, 30, 60]:
            folder = output / dataset / str(minutes)
            folder.mkdir(parents=True, exist_ok=True)
            all_windows, retained = windows(
                votes, minutes, 3 if dataset == "NUS" else 2, 0.5 if dataset == "NUS" else None
            )
            primary = all_windows.loc[all_windows.included]
            assert primary.gap.ge(-1e-12).all()
            all_windows.to_csv(folder / "WINDOW_INCLUSION_AND_SCORES.csv", index=False)
            retained.to_csv(
                folder / "RETAINED_VOTES.csv.gz",
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
            summary.append(
                {
                    "dataset": dataset,
                    "minutes": minutes,
                    "windows": len(primary),
                    "responses": len(retained),
                    "mean_gap": primary.gap.mean(),
                    **primary.category.value_counts().to_dict(),
                }
            )
            flow.append(
                {
                    "dataset": dataset,
                    "minutes": minutes,
                    **support,
                    "deduplicated_windows": len(all_windows),
                    "shared_windows": len(primary),
                }
            )
            if minutes == 30:
                intervals.append(bootstrap(primary, retained).assign(dataset=dataset))
            for threshold in THRESHOLDS:
                accepted = primary.averaged_burden <= threshold
                full = primary.full_burden <= threshold
                assert not (~accepted & full).any()
                thresholds.append(
                    {
                        "dataset": dataset,
                        "minutes": minutes,
                        "threshold": threshold,
                        "false_acceptance_windows": int((accepted & ~full).sum()),
                        "averaged_qualifying_windows": int(accepted.sum()),
                        "windows": len(primary),
                    }
                )
    pd.DataFrame(summary).to_csv(output / "SHARED_ROOM_SUMMARY.csv", index=False)
    pd.DataFrame(flow).to_csv(output / "ROW_AND_WINDOW_FLOW.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "BURDEN_THRESHOLD_RESULTS.csv", index=False)
    pd.concat(intervals, ignore_index=True).to_csv(
        output / "GROUPED_BOOTSTRAP_INTERVALS.csv", index=False
    )
    (output / "REPRODUCTION_AUDIT.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "raw_inputs": {k: v[1] for k, v in sources.items()},
                "primary_window_minutes": 30,
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream_root.resolve(), args.output_root.resolve())
