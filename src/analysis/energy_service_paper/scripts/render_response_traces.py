"""Trace two deterministically selected qualified own-baseline July responses."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from render_control_service_figures import (
    CONTROL_COLORS, export_result, six_panel_canvas,
)


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def render(stage_root, output, results_root=None, engineering_root=None, service_module=None):
    results_root = results_root if results_root is not None else stage_root / "analysis/results"
    engineering_root = engineering_root if engineering_root is not None else stage_root / "engineering"
    module_path = service_module if service_module is not None else stage_root / "analysis/thermal_service.py"
    spec = importlib.util.spec_from_file_location("trace_thermal_service", module_path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    event_path = results_root / "events/EVENT_SELECTIONS.csv"
    profile_path = engineering_root / "PROFILE_ASSIGNMENTS.csv.gz"
    tail_path = results_root / "annual/FIXED_TAIL_MEMBERSHIPS.csv"
    manifest_path = engineering_root / "EVENT_RUN_MANIFEST.csv"
    events, profiles, tails, manifest = map(read, (event_path, profile_path, tail_path, manifest_path))
    possible = events.query("event_month == 7 and service_fraction == 1 and permission == 'native_zone' "
                            "and control == 'personal_center' and selection_status == 'selected'")
    fig, axes = six_panel_canvas([
        "Port Angeles electricity", "Seattle electricity", "Port Angeles temperature",
        "Seattle temperature", "Port Angeles tail deviation", "Seattle tail deviation",
    ], control_key=False)
    fig.legend(handles=[Line2D([], [], color=CONTROL_COLORS["uniform_average"], label="Own baseline"),
                        Line2D([], [], color=CONTROL_COLORS["personal_center"], linestyle="--", label="Response")],
               loc="upper center", bbox_to_anchor=(.5, .995), ncol=2)
    sources = {"events": event_path, "profile_assignment": profile_path, "fixed_tail": tail_path,
               "event_manifest": manifest_path, "service_code": module_path}
    records, selections = [], []
    for column, context in enumerate(("PortAngeles_small2019", "Seattle_small2019")):
        row = possible[possible.context.eq(context)].sort_values(["composition_id", "layout"]).iloc[0]
        physical = manifest[manifest.run_id.eq(row.selected_run_id)].iloc[0]
        p = profiles[profiles.context.eq(context) & profiles.composition_id.eq(row.composition_id)
                     & profiles.layout.eq(row.layout)].sort_values("profile_id")
        t = tails[tails.context.eq(context) & tails.composition_id.eq(row.composition_id)
                  & tails.layout.eq(row.layout)]
        tail_indices = np.flatnonzero(p.profile_id.isin(t.profile_id))
        if len(p) != 267 or len(tail_indices) != 27:
            raise ValueError("The plotted traces require the frozen 267/27 profile identities")
        selections.append(dict(context=context, composition_id=int(row.composition_id), layout=row.layout,
                               baseline_run_id=row.baseline_run_id, response_run_id=row.selected_run_id,
                               selection_rule="lexicographically first qualified PC native-zone July condition"))
        for operation, path, color, linestyle in (
            ("Own baseline", Path(physical.baseline_hourly_path), CONTROL_COLORS["uniform_average"], "-"),
            ("Response", Path(physical.path_hourly), CONTROL_COLORS["personal_center"], "--"),
        ):
            if not path.is_absolute():
                path = engineering_root / path
            sources[f"{context}_{operation}"] = path
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            civil_hour = data["hour"] - 1 + data["dst"]
            event_rows = np.flatnonzero((data["month"] == int(row.event_month))
                                        & (data["day"] == int(row.event_day))
                                        & (civil_hour >= 14) & (civil_hour < 16))
            if len(event_rows) != 2 or event_rows[1] != event_rows[0] + 1:
                raise ValueError("The source clock must identify two consecutive event hours")
            window = np.arange(event_rows[0], event_rows[0] + 50)
            people = data["people"][window] * data["occupied"][window, None]
            zone_map = {zone: index for index, zone in enumerate(data["zone_ids"].tolist())}
            zones = np.array([zone_map[zone] for zone in p.zone_id])
            hourly = service.score_service(data["temperatures"][window], people, zones,
                                           p.centre_c, p.lower_c, p.upper_c, tail_indices)["hourly"]
            denominator = people.sum(axis=1)
            mean_operative = np.divide((data["temperatures"][window] * people).sum(axis=1), denominator,
                                       out=np.full(50, np.nan), where=denominator > 0)
            series = (data["hvac_electric_kwh"][window], mean_operative,
                      hourly["fixed_tail_reference_mismatch_c"])
            for index, (values, ylabel) in enumerate(zip(series, (
                "HVAC electricity (kW)", "Operative temperature (°C)", "Tail mismatch (°C)"))):
                ax = axes[index * 2 + column]
                ax.plot(np.arange(50) + .5, values, color=color, linestyle=linestyle, linewidth=1.45)
                ax.set_ylabel(ylabel)
            records.extend(dict(context=context, composition_id=int(row.composition_id), layout=row.layout,
                                operation=operation, hour_after_event_start=int(hour),
                                source_row_index=int(window[hour]), source_month=int(data["month"][window[hour]]),
                                source_day=int(data["day"][window[hour]]), civil_hour=int(civil_hour[window[hour]]),
                                phase="event" if hour < 2 else "recovery",
                                hvac_electric_kw=float(series[0][hour]),
                                occupied_mean_operative_temperature_c=float(mean_operative[hour]),
                                mean_reference_mismatch_c=float(hourly["mean_reference_mismatch_c"][hour]),
                                fixed_tail_reference_mismatch_c=float(series[2][hour]))
                           for hour in range(50))
    for ax in axes:
        ax.axvspan(0, 2, color="#C8A36A", alpha=.2, zorder=0)
        ax.set_xlim(0, 50)
        ax.set_xticks([0, 24, 48])
        ax.set_xlabel("Hours from event start")
    for ax in (axes[2], axes[4]):
        ax.set_xlim(0, 6)
        ax.set_xticks([0, 2, 4, 6])
    caption = (
        "Supplementary Figure S21 | Hourly own-baseline response and recovery. a,b, Purchased HVAC "
        "electricity under own-baseline and perturbed July operation in Port Angeles and Seattle. c,d, "
        "Native-occupancy-weighted zone operative temperature. e,f, Hourly mean reference mismatch of the fixed "
        "27-profile tail. Solid grey and dashed gold lines show each Personal centre controller's own "
        "annual baseline and its selected response. Shading marks the two-hour event, followed by 48 "
        "calendar recovery hours. Temperature and service curves appear only during native occupied "
        "hours. Panels c,e enlarge the first six hours in Port Angeles, whose remaining recovery window "
        "is unoccupied; panels a,b,d,f retain the full 50-hour window. The first qualified native-zone "
        "Personal centre condition in composition–placement order "
        "was selected separately in each context. Both conditions use composition 2026102001 and random mixing."
    )
    result = export_result(fig, axes, output, "Supplementary_Figure_S21", caption=caption, sources=sources)
    pd.DataFrame(records).to_csv(output / "Supplementary_Figure_S21_source.csv", index=False)
    (output / "Supplementary_Figure_S21_selection.json").write_text(json.dumps(selections, indent=2)+"\n", encoding="utf-8")
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--engineering-root", type=Path)
    parser.add_argument("--service-module", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.stage_root is None and not all((args.results_root, args.engineering_root, args.service_module)):
        parser.error("Use --stage-root or provide --results-root, --engineering-root and --service-module")
    args.output.mkdir(parents=True, exist_ok=True)
    render(args.stage_root, args.output, args.results_root, args.engineering_root, args.service_module)
    print(json.dumps({"figure": "Supplementary Figure S21", "actual_visual_review": "PENDING"}))


if __name__ == "__main__":
    main()
