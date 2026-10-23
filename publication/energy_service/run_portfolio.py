"""Aggregate declared office replicas at synchronous event and recovery hours.

Input tables are minimized Stage8 outputs, not a real city inventory. Electricity
and gas columns contain interval kWh; timestep_hours converts electric power.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


GROUP = ("climate_context", "scope", "control", "event_id")
TYPE = ("composition_id", "layout")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def is_true(value):
    if str(value).lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"Not an explicit boolean: {value}")
    return str(value).lower() in {"true", "1"}


def aggregate(types, hourly, design_types, protocol):
    inventory = {tuple(row[key] for key in TYPE) for row in design_types}
    count = int(protocol["portfolio_buildings"])
    if not inventory or count % len(inventory):
        raise ValueError("Locked type inventory must divide portfolio count")
    replicas = count // len(inventory)
    for fraction in protocol["dr_participation_fractions"]:
        if abs(replicas * fraction - round(replicas * fraction)) > 1e-9:
            raise ValueError("Participation requires integer replicas of every type")
    groups = defaultdict(list)
    for row in types:
        groups[tuple(row[key] for key in GROUP)].append(row)
    hours = defaultdict(list)
    for row in hourly:
        hours[tuple(row[key] for key in GROUP + TYPE)].append(row)
    output, curves, components = [], [], []
    for group, members in sorted(groups.items()):
        if len(members) != len(inventory) or {
            tuple(row[key] for key in TYPE) for row in members
        } != inventory:
            raise ValueError(f"Missing or duplicated locked type in {group}")
        group_fields = dict(zip(GROUP, group))
        times = None
        parsed = []
        for row in sorted(members, key=lambda r: tuple(r[key] for key in TYPE)):
            type_id = tuple(row[key] for key in TYPE)
            series = sorted(hours[group + type_id], key=lambda h: int(h["hour_index"]))
            if [int(h["hour_index"]) for h in series] != list(range(50)):
                raise ValueError(f"Need exact 2+48 consecutive hourly rows: {group + type_id}")
            stamp = [(h["timestamp_local"], h["phase"]) for h in series]
            if times is not None and stamp != times:
                raise ValueError(f"Non-synchronous time or phase in {group}")
            times = stamp
            calendar = [datetime.fromisoformat(h["timestamp_local"]) for h in series]
            if any(b - a != timedelta(hours=1) for a, b in zip(calendar, calendar[1:])):
                raise ValueError("Expected 50 consecutive local model hours")
            if [h["phase"] for h in series] != ["event"] * 2 + ["recovery"] * 48:
                raise ValueError("Event/recovery phase must be 2 and 48 consecutive hours")
            if any(float(h["timestep_hours"]) != 1 for h in series):
                raise ValueError("This locked protocol uses hourly intervals")
            if row["baseline_engineering_status"] != "PASS":
                raise ValueError(f"No engineering-qualified baseline for {group + type_id}")
            event_engineering = row["event_engineering_status"] == "PASS"
            baseline_ok = is_true(row["baseline_service_qualified"])
            event_ok = is_true(row["event_service_qualified"])
            p0 = [float(h["baseline_electric_kwh"]) for h in series]
            p1 = [float(h["event_electric_kwh"]) for h in series] if event_engineering else p0
            g0 = [float(h["baseline_gas_hhv_kwh"]) for h in series]
            g1 = [float(h["event_gas_hhv_kwh"]) for h in series] if event_engineering else g0
            positive = sum(a - b for a, b in zip(p0[:2], p1[:2])) / 2 > 1e-9
            admitted = event_engineering and baseline_ok and event_ok and positive
            parsed.append((row, p0, p1, g0, g1, baseline_ok, admitted))
        total_area = sum(float(row[0]["floor_area_m2"]) * replicas for row in parsed)
        if total_area <= 0:
            raise ValueError("Positive complete-portfolio floor area required")
        for fraction in protocol["dr_participation_fractions"]:
            participants = int(round(replicas * fraction))
            keys = group_fields | {"dr_participation_fraction": fraction}
            delta_e = [0.0] * 50
            delta_g = [0.0] * 50
            baseline = [0.0] * 50
            gas_baseline = [0.0] * 50
            nonsynchronous_peak = 0.0
            for row, p0, p1, g0, g1, baseline_ok, admitted in parsed:
                active = participants if admitted else 0
                for hour in range(50):
                    baseline[hour] += replicas * p0[hour]
                    gas_baseline[hour] += replicas * g0[hour]
                    delta_e[hour] += active * (p1[hour] - p0[hour])
                    delta_g[hour] += active * (g1[hour] - g0[hour])
                nonsynchronous_peak += active * max(0.0, max(b - a for a, b in zip(p0[2:], p1[2:])))
                components.append(keys | {key: row[key] for key in TYPE} | {
                    "baseline_case_id": row["baseline_case_id"],
                    "event_case_id": row["event_case_id"],
                    "replicas": replicas,
                    "nominal_participants": participants,
                    "active_qualified_positive_response": active,
                    "baseline_service_qualified": baseline_ok,
                    "event_service_qualified": is_true(row["event_service_qualified"]),
                    "admitted": admitted,
                })
            metrics = {
                "event_average_reduction_kw": -sum(delta_e[:2]) / 2,
                "event_electric_saving_kwh": -sum(delta_e[:2]),
                "recovery_extra_electric_kwh": sum(delta_e[2:]),
                "recovery_synchronous_peak_extra_kw": max(0.0, max(delta_e[2:])),
                "event_plus_48h_net_electric_saving_kwh": -sum(delta_e),
                "event_plus_48h_gas_change_kwh_hhv": sum(delta_g),
                "sum_individual_recovery_peak_extra_kw_upper_bound": nonsynchronous_peak,
            }
            normalized = {key + "_per_1000m2": value * 1000 / total_area for key, value in metrics.items()}
            baseline_qualified = sum(replicas for row in parsed if row[5])
            output.append(keys | {
                "scope_label": "standardized office portfolio, not observed city stock",
                "portfolio_buildings": count,
                "portfolio_type_count": len(inventory),
                "portfolio_area_m2": total_area,
                "nominal_participating_buildings": participants * len(inventory),
                "active_qualified_positive_response_buildings": sum(participants for row in parsed if row[6]),
                "baseline_service_qualified_buildings": baseline_qualified,
                "baseline_service_unqualified_buildings": count - baseline_qualified,
                "nominal_participants_with_ineligible_baseline": sum(participants for row in parsed if not row[5]),
                "service_qualified_baseline_zero_tested_response_buildings": sum(participants for row in parsed if row[5] and not row[6]),
                "all_baselines_service_qualified": baseline_qualified == count,
                "capacity_interpretation": "Sum of individually qualified positive responses; no whole-portfolio service claim when baseline coverage is incomplete",
                "complete_portfolio_service_qualified_response_kw": metrics["event_average_reduction_kw"] if baseline_qualified == count else None,
            } | metrics | normalized)
            for hour in range(50):
                curves.append(keys | {
                    "hour_index": hour,
                    "timestamp_local": times[hour][0],
                    "phase": times[hour][1],
                    "baseline_electric_kwh": baseline[hour],
                    "perturbed_electric_kwh": baseline[hour] + delta_e[hour],
                    "electric_change_kwh": delta_e[hour],
                    "baseline_gas_hhv_kwh": gas_baseline[hour],
                    "perturbed_gas_hhv_kwh": gas_baseline[hour] + delta_g[hour],
                    "gas_change_hhv_kwh": delta_g[hour],
                })
    return output, curves, components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--types", required=True, type=Path)
    parser.add_argument("--hourly", required=True, type=Path)
    parser.add_argument("--design-types", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary, hourly, members = aggregate(read_csv(args.types), read_csv(args.hourly), read_csv(args.design_types), protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "portfolio_summary.csv", summary)
    write_csv(args.output / "portfolio_hourly.csv", hourly)
    write_csv(args.output / "portfolio_components.csv", members)
    metadata = {
        "status": "COMPUTED_FROM_SUPPLIED_MINIMIZED_RUN_OUTPUTS",
        "actual_city_prediction": False,
        "inputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.types, args.hourly, args.design_types, args.protocol)},
        "summary_rows": len(summary), "hourly_rows": len(hourly), "component_rows": len(members),
    }
    (args.output / "portfolio_QA.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
