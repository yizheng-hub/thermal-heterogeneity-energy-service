"""Audit selected own-baseline responses through 168 hours from event start."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from analyse_common_baseline_dr import load_hourly, phase_indices  # noqa: E402


LAYOUTS = ("random_mixing", "demand_ordered")


def find_hourly(root: Path, run_id: str) -> Path:
    matches = list(root.rglob(f"{run_id}.npz"))
    if not matches:
        raise ValueError(f"Expected one hourly trajectory for {run_id}, found {matches}")
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}
    if len(digests) != 1:
        raise ValueError(f"Duplicate hourly trajectories disagree for {run_id}: {matches}")
    return sorted(matches)[0]


def anchors_by_context(manifest_path: Path) -> dict[str, dict[str, np.ndarray]]:
    manifest = pd.read_csv(manifest_path)
    result = {}
    for context, group in manifest[manifest.variant.eq(3)].groupby("context"):
        qualified = group[group.engineering_status.isin({"cached_qualified", "new_qualified"})]
        if qualified.run_id.nunique() != 1:
            raise ValueError(f"Expected one common reference trajectory for {context}")
        path = Path(qualified.iloc[0].path_hourly)
        if not path.is_absolute():
            path = manifest_path.parent / path
        result[str(context)] = load_hourly(path.resolve())
    return result


def build(selections_path: Path, trajectories: Path, anchor_manifest: Path,
          factors_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(selections_path)
    selected = selections[
        selections.quality_scope.eq("primary")
        & selections.layout.isin(LAYOUTS)
        & selections.qualified_positive_response.eq(True)
    ].copy()
    anchors = anchors_by_context(anchor_manifest)
    factors = pd.read_csv(factors_path).set_index("city")
    cache: dict[str, dict[str, np.ndarray]] = {}

    def trajectory(path: Path) -> dict[str, np.ndarray]:
        key = str(path.resolve())
        if key not in cache:
            cache[key] = load_hourly(path.resolve())
        return cache[key]

    rows = []
    for row in selected.itertuples(index=False):
        baseline_path = find_hourly(trajectories, str(row.baseline_run_id))
        event_path = find_hourly(trajectories, str(row.event_run_id))
        baseline = trajectory(baseline_path)
        event = trajectory(event_path)
        phases = phase_indices(
            anchors[str(row.context)], {"month": int(row.event_month), "day": int(row.event_day)}
        )
        event_start = int(phases["event"][0])
        horizon_50 = np.arange(event_start, event_start + 50)
        horizon_168 = np.arange(event_start, min(event_start + 168, len(event["hvac_electric_kwh"])))
        delta_e = event["hvac_electric_kwh"] - baseline["hvac_electric_kwh"]
        delta_g = event["gas_hhv_kwh"] - baseline["gas_hhv_kwh"]
        factor = factors.loc[str(row.context)]
        terminal_50 = horizon_50[-1]
        terminal_168 = horizon_168[-1]
        rows.append({
            "data_source": row.data_source,
            "context": row.context,
            "composition_id": row.composition_id,
            "layout": row.layout,
            "permission": row.permission,
            "control": row.control,
            "event_month": row.event_month,
            "event_day": row.event_day,
            "event_run_id": row.event_run_id,
            "baseline_run_id": row.baseline_run_id,
            "net_electric_saving_50h_kwh": float(-delta_e[horizon_50].sum()),
            "net_electric_saving_168h_kwh": float(-delta_e[horizon_168].sum()),
            "net_gas_change_168h_kwh_hhv": float(delta_g[horizon_168].sum()),
            "operational_co2_change_168h_kg": float(
                delta_e[horizon_168].sum() * factor.electricity_co2_kg_kwh
                + delta_g[horizon_168].sum() * factor.gas_co2_kg_kwh_hhv
            ),
            "terminal_50h_max_abs_zone_temperature_delta_c": float(
                np.max(np.abs(event["temperatures"][terminal_50] - baseline["temperatures"][terminal_50]))
            ),
            "terminal_168h_max_abs_zone_temperature_delta_c": float(
                np.max(np.abs(event["temperatures"][terminal_168] - baseline["temperatures"][terminal_168]))
            ),
        })
    table = pd.DataFrame(rows)
    table.to_csv(output / "LONG_RECOVERY_TERMINAL_AUDIT.csv", index=False)
    report = {
        "status": "PASS",
        "responses": int(len(table)),
        "net_positive_at_50h": int(table.net_electric_saving_50h_kwh.gt(0).sum()),
        "net_positive_at_168h": int(table.net_electric_saving_168h_kwh.gt(0).sum()),
        "maximum_terminal_168h_zone_temperature_difference_c": float(
            table.terminal_168h_max_abs_zone_temperature_delta_c.max()
        ),
        "time_origin": "start of the two-hour response event",
    }
    (output / "LONG_RECOVERY_TERMINAL_QA.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        args.selections.resolve(), args.trajectories.resolve(),
        args.anchor_manifest.resolve(), args.factors.resolve(), args.output.resolve()
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
