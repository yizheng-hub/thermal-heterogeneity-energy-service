"""Render frozen main Figures 1--3 from approved panel-data v2 products.

This module performs only display summaries specified by the Gate 4 design lock.
It never runs a simulation, changes a scientific denominator or imputes an
unmatched comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from .figures import (
    CONTEXT_COLORS as SHARED_CONTEXT_COLORS,
    DEMAND_CMAP,
    VARIANCE_COLORS,
    PALETTE,
    create_panel_figure,
    data_safe_legend,
    draw_world_land,
    style_panel,
)


CONTEXT_ORDER = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
CONTEXT_LABELS = {
    "Miami": "Miami",
    "NewDelhi": "New Delhi",
    "PortAngeles_small2019": "Port Angeles",
    "Seattle_small2019": "Seattle",
}
CONTEXT_LEGEND_LABELS = {
    "Miami": "Miami",
    "NewDelhi": "New Delhi",
    "PortAngeles_small2019": "Port Angeles",
    "Seattle_small2019": "Seattle",
}
CONTEXT_COLORS = SHARED_CONTEXT_COLORS
POPULATION_LABELS = {
    "us_archive_stratum": "United States",
    "multinational_source_balanced": "Multinational",
    "india_archive_stratum": "India",
}
OUTCOME_ORDER = [
    "same_direction_decrease",
    "same_direction_increase",
    "all_tied",
    "direction_not_stable",
    "match_lost",
    "all_unmatched",
]
OUTCOME_LABELS = {
    "same_direction_decrease": "Decrease",
    "same_direction_increase": "Increase",
    "all_tied": "Tie",
    "direction_not_stable": "Unstable",
    "match_lost": "Unmatched",
    "all_unmatched": "Unmatched",
}
OUTCOME_COLORS = {
    "same_direction_decrease": PALETTE["blue"],
    "same_direction_increase": PALETTE["vermillion"],
    "all_tied": PALETTE["grey"],
    "direction_not_stable": PALETTE["purple"],
    "match_lost": PALETTE["light_grey"],
    "all_unmatched": PALETTE["light_grey"],
}


def _read_csv(root: Path, panel_id: str, required: set[str]) -> pd.DataFrame:
    path = root / f"{panel_id}.csv"
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{panel_id} missing columns: {sorted(missing)}")
    if frame.empty or set(frame["panel_id"]) != {panel_id}:
        raise ValueError(f"{panel_id} is empty or carries the wrong panel_id")
    return frame


def _read_json(root: Path, filename: str, panel_id: str) -> dict:
    payload = json.loads((root / filename).read_text(encoding="utf-8"))
    if payload.get("panel_id") != panel_id:
        raise ValueError(f"{filename} carries the wrong panel_id")
    return payload


def _finish_axis(axis: Axes, letter: str, title: str, grid_axis: str | None = "y") -> None:
    style_panel(axis, letter, title, grid_axis=grid_axis)


def _context_line_summary(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    matched = frame.loc[frame["matched"].astype(str).str.lower().eq("true")].copy()
    matched[value] = pd.to_numeric(matched[value], errors="coerce")
    matched = matched.dropna(subset=[value])
    return (
        matched.groupby(["context", "delta_c"], sort=False)[value]
        .agg(
            median="median",
            q1=lambda series: series.quantile(0.25),
            q3=lambda series: series.quantile(0.75),
            n="size",
        )
        .reset_index()
    )


def build_figure_1(panel_root: Path | str) -> Figure:
    """Build Figure 1: observed geography, policy coordinates and variance scale."""

    root = Path(panel_root)
    a = _read_csv(root, "Fig1a", {"population", "records", "analysis_units"})
    b = _read_csv(
        root,
        "Fig1b",
        {"city", "latitude", "longitude", "supported", "analysis_units", "demand_center_c"},
    )
    c = _read_csv(
        root,
        "Fig1c",
        {"city", "latitude", "longitude", "supported", "analysis_units", "demand_center_c"},
    )
    d = _read_csv(
        root,
        "Fig1d",
        {"aggregation_level", "season", "name", "p10_c", "p50_c", "p90_c"},
    )
    e = _read_csv(root, "Fig1e", {"component", "share"})
    f = _read_csv(
        root,
        "Fig1f",
        {"policy", "position", "proportion", "ci_low", "ci_high", "records"},
    )
    g = _read_csv(
        root,
        "Fig1g",
        {"policy", "q10_delta_c", "q25_delta_c", "median_delta_c", "q75_delta_c", "q90_delta_c", "records"},
    )
    h = _read_csv(
        root,
        "Fig1h",
        {"reference", "weighting", "component", "loo_min_share", "loo_median_share", "loo_max_share", "n_leave_one_source_runs"},
    )

    figure, axes = create_panel_figure()
    figure.subplots_adjust(wspace=0.24, hspace=0.34)
    ax = axes.ravel()

    # a: separate record and analysis-unit flows prevent a people/record conflation.
    complete = a.set_index("population").loc["complete"]
    executable = a.set_index("population").loc["executable"]
    ax[0].set_xlim(-0.12, 1.04)
    ax[0].set_ylim(-0.4, 1.7)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].axis("off")
    for y, key, label, loss in [
        (1.05, "records", "Records", 19),
        (0.25, "analysis_units", "Analysis units", 17),
    ]:
        values = [int(complete[key]), int(executable[key])]
        for x, value, stage in zip([0.02, 0.86], values, ["Complete", "Executable"]):
            width = 0.34 if stage == "Complete" else 0.32
            box = FancyBboxPatch(
                (x - width / 2, y - 0.19),
                width,
                0.38,
                boxstyle="round,pad=0.02",
                facecolor="#EAF2F8" if x == 0 else "#D7EFE7",
                edgecolor="#666666",
                linewidth=0.65,
            )
            ax[0].add_patch(box)
            ax[0].text(x, y + 0.04, f"{value:,}", ha="center", va="center")
            ax[0].text(x, y - 0.10, stage, ha="center", va="center")
        ax[0].annotate("", xy=(0.68, y), xytext=(0.21, y), arrowprops={"arrowstyle": "->", "lw": 0.9})
        ax[0].text(0.45, y + 0.15, f"-{loss}", ha="center", va="center")
        ax[0].text(0.02, y + 0.30, label, ha="left", va="center")
    ax[0].text(0.46, 1.52, "187 buildings · 40 cities · 15 countries", ha="center", va="center")
    _finish_axis(ax[0], "a", "Archive and population", None)

    supported_centres = pd.concat([b, c], ignore_index=True)["demand_center_c"].dropna()
    vmin, vmax = float(supported_centres.min()), float(supported_centres.max())
    anchor_names = {"honolulu", "midland", "bangalore", "sydney"}
    anchor_offsets = {
        "honolulu": (4, -10),
        "midland": (3, 3),
        "bangalore": (3, -7),
        "sydney": (-4, 7),
    }
    anchor_labels = {"honolulu": "Honolulu", "midland": "UK", "bangalore": "Bengaluru", "sydney": "Sydney"}
    supported_frames = pd.concat([b, c], ignore_index=True)
    supported_frames = supported_frames.loc[
        supported_frames["supported"].astype(str).str.lower().eq("true")
    ]
    marker_unit_max = float(supported_frames["analysis_units"].max())
    for panel_axis, frame, letter, title in [
        (ax[1], b, "b", "Warm-season observations"),
        (ax[2], c, "c", "Cold-season observations"),
    ]:
        draw_world_land(panel_axis)
        is_supported = frame["supported"].astype(str).str.lower().eq("true")
        unsupported = frame.loc[~is_supported]
        panel_axis.scatter(
            unsupported["longitude"],
            unsupported["latitude"],
            s=22,
            facecolors="none",
            edgecolors=PALETTE["grey"],
            linewidths=0.8,
            zorder=2,
        )
        shown = frame.loc[is_supported].copy()
        sizes = 18 + 110 * np.sqrt(shown["analysis_units"].astype(float) / marker_unit_max)
        panel_axis.scatter(
            shown["longitude"],
            shown["latitude"],
            c=shown["demand_center_c"],
            cmap=DEMAND_CMAP,
            vmin=vmin,
            vmax=vmax,
            s=sizes,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        for row in shown.loc[shown["city"].isin(anchor_names)].itertuples():
            city = str(row.city)
            annotation = panel_axis.annotate(
                anchor_labels[str(row.city)],
                (row.longitude, row.latitude),
                xytext=anchor_offsets[city],
                textcoords="offset points",
                ha="right" if city == "sydney" else "left",
                va="bottom",
                annotation_clip=True,
            )
            annotation.set_clip_on(True)
        panel_axis.text(
            0.99,
            0.02 if letter == "b" else 0.08,
            f"{len(shown)}/40 supported",
            ha="right",
            va="bottom",
            transform=panel_axis.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6},
        )
        if letter == "b":
            colour_map = plt.get_cmap(DEMAND_CMAP)
            for colour_index in range(8):
                panel_axis.add_patch(
                    Rectangle(
                        (-34 + 12 * colour_index, -57),
                        12,
                        5,
                        facecolor=colour_map(colour_index / 7),
                        edgecolor="none",
                        zorder=4,
                    )
                )
            panel_axis.text(14, -48, f"{vmin:.1f}–{vmax:.1f} °C", ha="center", va="bottom")
        else:
            panel_axis.scatter([-170, -135], [-54, -54], s=[28, 82], color=PALETTE["grey"], edgecolors="white", linewidths=0.4, zorder=4)
            panel_axis.scatter([-70], [-54], s=35, facecolors="none", edgecolors=PALETTE["grey"], linewidths=0.8, zorder=4)
            panel_axis.text(-120, -54, "area=n", ha="left", va="center")
            panel_axis.text(-59, -54, "no support", ha="left", va="center")
        panel_axis.set_xticks([-120, 0, 120])
        panel_axis.set_yticks([-30, 30, 60])
        if letter == "b":
            panel_axis.set_ylabel("")
            panel_axis.tick_params(axis="y", direction="in", pad=-24)
            for tick_label in panel_axis.get_yticklabels():
                tick_label.set_horizontalalignment("left")
        _finish_axis(panel_axis, letter, title, None)

    # d: four compact ranked bands, preserving empirical P10--P90 intervals.
    group_order = [("city", "warm"), ("city", "cold"), ("source", "warm"), ("source", "cold")]
    y_base = 0
    yticks, ylabels = [], []
    for group_index, (level, season) in enumerate(group_order):
        group = d.loc[(d["aggregation_level"] == level) & (d["season"] == season)].sort_values("p50_c")
        count = len(group)
        ys = np.linspace(y_base, y_base + 0.68, count) if count > 1 else np.array([y_base + 0.34])
        ax[3].hlines(ys, group["p10_c"], group["p90_c"], color=PALETTE["sky"], linewidth=0.7)
        ax[3].scatter(group["p50_c"], ys, s=5, color=PALETTE["blue"], zorder=3)
        widths = group["p90_c"] - group["p10_c"]
        yticks.append(y_base + 0.34)
        ylabels.append(f"{'C' if level == 'city' else 'S'}/{season[0]}: {count}, {widths.median():.1f}°")
        y_base += 1.0
    ax[3].set_yticks([])
    ax[3].set_xlim(14, 32)
    ax[3].set_xticks([16, 20, 24, 28, 32])
    for yy, label in zip(yticks, ylabels):
        ax[3].text(14.15, yy, label, ha="left", va="center")
    ax[3].set_xlabel("Demand temperature (°C)", labelpad=-4.5)
    _finish_axis(ax[3], "d", "Within-place demand ranges", "x")

    # e: exact mutually exclusive variance allocation.
    component_labels = {
        "country": "Country",
        "city": "City",
        "release_building_group_id": "Building",
        "release_unit_id": "Within building",
        "within_unit": "Repeat",
    }
    order = ["country", "city", "release_building_group_id", "release_unit_id", "within_unit"]
    colors = VARIANCE_COLORS
    lookup = e.set_index("component")["share"]
    left = 0.0
    for component, color in zip(order, colors):
        value = float(lookup.loc[component]) * 100
        ax[4].barh([0], [value], left=left, color=color, height=0.45)
        offset_y = {"city": 0.13, "release_building_group_id": -0.13}.get(component, 0)
        ax[4].text(left + value / 2, offset_y, f"{value:.1f}%", ha="center", va="center")
        left += value
    ax[4].set_xlim(0, 100)
    ax[4].set_ylim(-0.65, 0.85)
    ax[4].set_yticks([])
    ax[4].set_xlabel("Share of variance (%)", labelpad=-4.5)
    handles = [Patch(facecolor=color, label=component_labels[name]) for name, color in zip(order, colors)]
    ax[4].legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=0.8, columnspacing=0.7, handletextpad=0.3)
    _finish_axis(ax[4], "e", "Where demand variance resides", "x")

    # f: each policy keeps its own denominator and within-range CI.
    positions = ["below", "within", "above"]
    position_colors = {"below": PALETTE["blue"], "within": PALETTE["light_grey"], "above": PALETTE["vermillion"]}
    policy_order = list(dict.fromkeys(f["policy"]))
    compact_policy = {
        "Australia model COP": "AUS COP",
        "Brazil NR-17": "BRA NR17",
        "Canada CCOHS": "CAN CCOHS",
        "India ECSBC 2024": "IND ECSBC",
        "Indonesia Permenkes 48": "IDN P48",
        "Malaysia ICOP IAQ": "MYS ICOP",
        "Singapore SS 553": "SGP SS553",
        "Sweden AFS 2023:12": "SWE AFS",
        "USA OSHA guidance": "USA OSHA",
    }
    for y, policy in enumerate(policy_order):
        group = f.loc[f["policy"] == policy].set_index("position")
        left = 0.0
        for position in positions:
            value = float(group.loc[position, "proportion"])
            ax[5].barh(y, value, left=left, color=position_colors[position], height=0.68)
            left += value
        within = group.loc["within"]
        within_left = float(group.loc["below", "proportion"])
        centre = within_left + float(within["proportion"]) / 2
        half = (float(within["ci_high"]) - float(within["ci_low"])) / 2
        ax[5].errorbar(centre, y, xerr=half, color="#222222", capsize=1.5, linewidth=0.8)
        ax[5].text(1.04, y, f"{float(within['proportion'])*100:.0f}%", ha="left", va="center")
        ax[5].text(1.50, y, f"{int(within['records']):,}", ha="right", va="center")
    ax[5].set_yticks([])
    for y, policy in enumerate(policy_order):
        ax[5].text(-0.02, y, compact_policy[policy], ha="right", va="center")
    ax[5].set_ylim(len(policy_order) - 0.5, -2.25)
    ax[5].set_xlim(-0.46, 1.55)
    ax[5].set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax[5].set_xlabel("Demand records by range (%)", labelpad=-4.5)
    handles = [Patch(facecolor=position_colors[p], label=p.title()) for p in positions]
    ax[5].legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99), handlelength=0.8, columnspacing=0.7, handletextpad=0.3)
    _finish_axis(ax[5], "f", "Policy-range coverage", "x")

    # g: empirical demand-minus-target quantiles.
    y = np.arange(len(g))
    ax[6].axvline(0, color="#333333", linewidth=0.7)
    for index, row in enumerate(g.itertuples()):
        ax[6].hlines(index, row.q10_delta_c, row.q90_delta_c, color=PALETTE["sky"], linewidth=1.2)
        ax[6].hlines(index, row.q25_delta_c, row.q75_delta_c, color=PALETTE["blue"], linewidth=4)
        ax[6].scatter(row.median_delta_c, index, s=26, color=PALETTE["vermillion"], zorder=3)
        label_y = index + 0.14 if index < len(g) - 1 else index - 0.14
        value_label = ax[6].text(
            2.55,
            label_y,
            f"{row.median_delta_c:+.1f}° (n={int(row.records)})",
            ha="right",
            va="top" if index < len(g) - 1 else "bottom",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.15},
        )
        value_label.set_gid("fig1g-value-label")
    ax[6].set_yticks(y, [])
    for index, value in enumerate(g["policy"]):
        label = value.replace("China summer rule", "China").replace("Japan Cool Biz", "Japan").replace("Thailand EGAT campaign", "Thailand")
        ax[6].text(-5.05, index, label, ha="left", va="center")
    ax[6].invert_yaxis()
    ax[6].set_xlim(-5.2, 4.8)
    ax[6].set_xticks([-4, -2, 0, 2, 4])
    ax[6].set_xlabel("Demand centre − campaign target (°C)")
    _finish_axis(ax[6], "g", "Guidance relative to demand", "x")

    # h: all 12 coefficient/weighting specifications remain visible.
    component_names = {
        "between_analysis_unit_within_building": "Within bldg.",
        "between_country": "Country",
        "between_city_within_country": "City",
        "between_building_within_city": "Building",
        "within_analysis_unit": "Repeat",
    }
    comp_order = list(component_names)
    for y, component in enumerate(comp_order):
        group = h.loc[h["component"] == component]
        for _, row in group.iterrows():
            ax[7].hlines(y, row["loo_min_share"] * 100, row["loo_max_share"] * 100, color="#C9C9C9", linewidth=0.6)
            ax[7].scatter(row["loo_median_share"] * 100, y, s=10, color=PALETTE["blue"], alpha=0.8)
        ax[7].hlines(y, group["loo_min_share"].min() * 100, group["loo_max_share"].max() * 100, color=PALETTE["vermillion"], linewidth=2.2)
        ax[7].scatter(group["loo_median_share"].median() * 100, y, s=28, color=PALETTE["vermillion"], zorder=3)
    ax[7].set_yticks([])
    for y, component in enumerate(comp_order):
        ax[7].text(-1.5, y, component_names[component], ha="right", va="center")
    ax[7].invert_yaxis()
    ax[7].set_xlim(-20, 75)
    ax[7].set_xticks([0, 25, 50, 75])
    ax[7].set_xlabel("Variance share (%)")
    _finish_axis(ax[7], "h", "Robustness: 12 specs × 27 removals", "x")

    return figure


def build_figure_2(panel_root: Path | str) -> Figure:
    """Build Figure 2: independent direct-vote evidence for compression loss."""

    root = Path(panel_root)
    a = _read_csv(root, "Fig2a", {"stage", "rows_or_windows", "exclusion_rule"})
    b = _read_csv(root, "Fig2b", {"window_id", "n_votes", "n_cooler", "n_no_change", "n_warmer", "share_cooler", "share_no_change", "share_warmer"})
    c = _read_csv(root, "Fig2c", {"window_id", "n_votes", "n_cooler", "n_no_change", "n_warmer", "temperature_mean_c", "temperature_spread_c"})
    d = _read_csv(root, "Fig2d", {"window_id", "n_votes", "full_burden", "compressed_burden", "jensen_gap"})
    e = _read_csv(root, "Fig2e", {"window_id", "full_burden", "compressed_burden", "n_votes", "n_cooler", "n_warmer"})
    f = _read_csv(root, "Fig2f", {"threshold", "false_accept_count", "false_reject_count", "eligible_windows"})
    g = _read_csv(root, "Fig2g", {"window_minutes", "metric", "estimate", "ci_low", "ci_high", "eligible_windows", "votes"})
    h = _read_csv(root, "Fig2h", {"dataset", "eligible_windows", "votes", "disagreement_count", "opposing_count", "cancellation_count", "metric", "estimate", "ci_low", "ci_high"})

    figure, axes = create_panel_figure()
    figure.subplots_adjust(left=0.13, wspace=0.38, hspace=0.40)
    ax = axes.ravel()

    # a: fixed construction flow.
    stage_order = ["raw rows", "declared cohort", "valid votes", "candidate windows", "minimum-user windows", "temperature-homogeneous windows", "person-window votes"]
    frame = a.set_index("stage").loc[stage_order].reset_index()
    y = np.arange(len(frame))[::-1]
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-0.55, len(frame) - 0.45)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].axis("off")
    for index, (yy, row) in enumerate(zip(y, frame.itertuples())):
        ax[0].text(0.04, yy, row.stage.replace("temperature-homogeneous", "temperature-homog."), ha="left", va="center")
        ax[0].text(0.80, yy, f"{int(row.rows_or_windows):,}", ha="right", va="center")
        if index < len(frame) - 1:
            ax[0].annotate("", xy=(0.88, yy - 0.72), xytext=(0.88, yy - 0.18), arrowprops={"arrowstyle": "->", "lw": 0.75, "color": "#666666"})
    ax[0].text(0.98, y[4], "≥3 users", ha="right", va="center")
    ax[0].text(0.98, y[5], "≤0.5 °C", ha="right", va="center")
    _finish_axis(ax[0], "a", "Independent cohort flow", None)

    # b: all 63 windows as thin composition bars.
    ordered = b.assign(net=b["share_warmer"] - b["share_cooler"], disagreement=(b[["n_cooler", "n_no_change", "n_warmer"]] > 0).sum(axis=1) > 1).sort_values(["net", "disagreement"])
    yy = np.arange(len(ordered))
    left = np.zeros(len(ordered))
    for field, label, color in [
        ("share_cooler", "Cooler", PALETTE["blue"]),
        ("share_no_change", "No change", PALETTE["light_grey"]),
        ("share_warmer", "Warmer", PALETTE["vermillion"]),
    ]:
        values = ordered[field].to_numpy(float)
        bars = ax[1].barh(yy, values, left=left, height=0.88, color=color, label=label)
        for patch in bars.patches:
            patch.set_gid("fig2b-window-bar")
        left += values
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(-1, len(ordered) + 30)
    ax[1].set_yticks([])
    ax[1].set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax[1].set_xlabel("Votes within room-window (%)")
    summary_text = ax[1].text(
        0.02,
        0.98,
        "49/63 disagreement · 6/63 opposite",
        ha="left",
        va="top",
        transform=ax[1].transAxes,
    )
    summary_text.set_gid("fig2b-summary")
    legend = ax[1].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.89),
        handlelength=0.8,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    legend.set_gid("fig2b-legend")
    _finish_axis(ax[1], "b", "Votes within shared rooms", "x")

    # c: six opposite-request windows.
    labels = [f"Window {i+1}" for i in range(len(c))]
    yy = np.arange(len(c))
    ax[2].barh(yy, -c["n_cooler"], color=PALETTE["blue"], height=0.58)
    ax[2].barh(yy, c["n_warmer"], color=PALETTE["vermillion"], height=0.58)
    ax[2].scatter(np.zeros(len(c)), yy, s=18 + c["n_no_change"] * 18, color=PALETTE["light_grey"], edgecolors="#555555", zorder=3)
    for index, row in enumerate(c.itertuples()):
        ax[2].text(
            1.18,
            index,
            f"n={row.n_votes} | {row.temperature_mean_c:.1f}°C | {row.temperature_spread_c:.1f}°",
            ha="left",
            va="center",
            clip_on=True,
        )
    ax[2].axvline(0, color="#444444", linewidth=0.7)
    ax[2].set_xlim(-3.5, 5.3)
    ax[2].set_xticks([-2, 0, 2, 4])
    ax[2].set_yticks(yy, labels)
    ax[2].set_xlabel("Cooler ← vote count → warmer", labelpad=-2)
    ax[2].xaxis.label.set_gid("fig2c-xlabel")
    _finish_axis(ax[2], "c", "Within-window opposing requests", "x")

    # d: Jensen compression relation and exact cancellation.
    exact = (d["compressed_burden"] == 0) & (d["full_burden"] > 0)
    ax[3].plot([0, 1], [0, 1], color="#777777", linewidth=0.8, linestyle="--")
    for row in d.itertuples():
        ax[3].plot([row.full_burden, row.full_burden], [row.compressed_burden, row.full_burden], color="#D0D0D0", linewidth=0.5)
    ax[3].scatter(d.loc[~exact, "full_burden"], d.loc[~exact, "compressed_burden"], s=22, color=PALETTE["blue"], alpha=0.8)
    ax[3].scatter(d.loc[exact, "full_burden"], d.loc[exact, "compressed_burden"], s=34, facecolors="white", edgecolors=PALETTE["vermillion"], linewidths=1.2)
    ax[3].set_xlim(-0.03, 1.03)
    ax[3].set_ylim(-0.03, 1.03)
    ax[3].set_xticks([0, 0.5, 1])
    ax[3].set_yticks([0, 0.5, 1])
    ax[3].set_xlabel("Full burden")
    ax[3].set_ylabel("Averaged burden")
    ax[3].text(0.98, 0.07, f"median gap={d['jensen_gap'].median():.3f}\nn=63 windows", ha="right", va="bottom")
    ax[3].xaxis.label.set_gid("fig2d-xlabel")
    _finish_axis(ax[3], "d", "Averaging compresses burden", "both")

    # e: four exact cancellations.
    yy = np.arange(len(e))
    for index, row in enumerate(e.itertuples()):
        ax[4].plot([row.compressed_burden, row.full_burden], [index, index], color="#888888", linewidth=1.3)
        ax[4].scatter(
            row.compressed_burden,
            index,
            marker="s",
            s=32,
            color=PALETTE["dark_grey"],
            label="Averaged" if index == 0 else None,
        )
        ax[4].scatter(
            row.full_burden,
            index,
            marker="o",
            s=36,
            color=PALETTE["blue"],
            label="Full" if index == 0 else None,
        )
        ax[4].text(row.full_burden + 0.04, index, f"{row.n_cooler}C/{row.n_warmer}W · n={row.n_votes}", va="center")
    ax[4].set_yticks(yy, [f"Case {value+1}" for value in yy])
    ax[4].set_xlim(-0.03, 1.02)
    ax[4].set_xticks([0, 0.5, 1])
    ax[4].set_ylim(-0.35, len(e) + 0.45)
    ax[4].set_xlabel("Request burden")
    ax[4].xaxis.label.set_gid("fig2e-xlabel")
    ax[4].legend(
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=0.8,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    _finish_axis(ax[4], "e", "Exact cancellation cases", "x")

    # f: false rejects remain visible at zero.
    x = np.arange(len(f))
    width = 0.34
    ax[5].bar(x - width / 2, f["false_accept_count"], width, color=PALETTE["vermillion"], label="False accept")
    ax[5].bar(x + width / 2, f["false_reject_count"], width, facecolor="white", edgecolor=PALETTE["blue"], hatch="//", label="False reject")
    for index, row in enumerate(f.itertuples()):
        ax[5].text(index - width / 2, row.false_accept_count + 0.18, str(row.false_accept_count), ha="center", va="bottom")
        ax[5].text(index + width / 2, 0.18, str(row.false_reject_count), ha="center", va="bottom")
    ax[5].set_xticks(x, [f"{value:g}" for value in f["threshold"]])
    ax[5].set_ylim(0, max(f["false_accept_count"]) + 2.2)
    ax[5].set_yticks([0, 2, 4, 6])
    ax[5].set_xlabel("Service threshold · 63 windows")
    ax[5].set_ylabel("Windows (count)")
    ax[5].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=0.9, columnspacing=0.8)
    _finish_axis(ax[5], "f", "One-sided qualification error", "y")

    # g: all metrics on a percentage scale; Jensen gap is multiplied by 100.
    metric_styles = {
        "disagreement_rate": ("Disagreement", PALETTE["blue"], "o"),
        "opposing_request_rate": ("Opposite", PALETTE["vermillion"], "^"),
        "mean_jensen_gap": ("Jensen gap", PALETTE["dark_grey"], "s"),
    }
    label_offsets = {
        "disagreement_rate": (8, 0),
        "opposing_request_rate": (8, 10),
        "mean_jensen_gap": (8, -1),
    }
    for metric, (label, color, marker) in metric_styles.items():
        group = g.loc[g["metric"] == metric].sort_values("window_minutes")
        xvals = group["window_minutes"].to_numpy(float)
        vals = group["estimate"].to_numpy(float) * 100
        low = group["ci_low"].to_numpy(float) * 100
        high = group["ci_high"].to_numpy(float) * 100
        ax[6].errorbar(xvals, vals, yerr=[vals-low, high-vals], color=color, marker=marker, linewidth=1.2, capsize=2, label=label)
        end_label = ax[6].annotate(
            label,
            (xvals[-1], vals[-1]),
            xytext=label_offsets[metric],
            textcoords="offset points",
            color=color,
            ha="left",
            va="center",
            annotation_clip=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
        )
        end_label.set_gid("fig2g-end-label")
        end_label.set_clip_on(True)
    ax[6].set_xlim(12, 92)
    ax[6].set_xticks([15, 30, 60], ["15 min\nn=65", "30 min\nn=63", "60 min\nn=56"])
    ax[6].set_ylim(0, 105)
    ax[6].set_yticks([0, 25, 50, 75, 100])
    ax[6].set_xlabel("")
    ax[6].set_ylabel("Rate or gap (%, 95% CI)")
    _finish_axis(ax[6], "g", "Window-definition sensitivity", "y")

    # h: side-by-side external replication on a common percentage scale.
    metric_order = ["disagreement_rate", "opposing_request_rate", "mean_jensen_gap"]
    metric_labels = ["Disagree", "Opposite", "Jensen gap"]
    x = np.arange(3)
    for dataset, offset, color, marker in [("NUS_HAAS", -0.11, PALETTE["blue"], "o"), ("ENTH", 0.11, PALETTE["orange"], "s")]:
        group = h.loc[h["dataset"] == dataset].set_index("metric").loc[metric_order]
        values = group["estimate"].to_numpy(float) * 100
        low = group["ci_low"].to_numpy(float) * 100
        high = group["ci_high"].to_numpy(float) * 100
        ax[7].errorbar(x + offset, values, yerr=[values-low, high-values], color=color, marker=marker, linestyle="", capsize=2, label="NUS" if dataset == "NUS_HAAS" else "ENTH")
    ax[7].set_xticks(x, metric_labels)
    ax[7].set_ylim(0, 105)
    ax[7].set_yticks([0, 25, 50, 75, 100])
    ax[7].set_ylabel("Estimate (%, 95% CI)")
    ax[7].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=0.8, columnspacing=0.8)
    nus = h.loc[h["dataset"] == "NUS_HAAS"].iloc[0]
    enth = h.loc[h["dataset"] == "ENTH"].iloc[0]
    ax[7].text(0.98, 0.56, f"NUS {int(nus.disagreement_count)}/{int(nus.eligible_windows)} disagree\n{int(nus.opposing_count)} opposite · {int(nus.cancellation_count)} cancel\nENTH {int(enth.disagreement_count)}/{int(enth.eligible_windows)} disagree\n{int(enth.opposing_count)} opposite · {int(enth.cancellation_count)} cancel", ha="right", va="center", transform=ax[7].transAxes)
    _finish_axis(ax[7], "h", "Independent ENTH replication", "y")

    return figure


def build_figure_3(panel_root: Path | str) -> Figure:
    """Build Figure 3: annual energy/carbon redistribution under three controls."""

    root = Path(panel_root)
    a = _read_json(root, "Fig3a_meta.json", "Fig3a")
    b = _read_csv(root, "Fig3b", {"population", "context", "seed", "n_assignments", "centre_c", "p10_c", "p90_c", "sd_c"})
    c = _read_json(root, "Fig3c_meta.json", "Fig3c")
    d = _read_csv(root, "Fig3d", {"context", "population", "seed", "service_budget", "delta_c", "delta_energy_mwh", "matched"})
    e = _read_csv(root, "Fig3e", {"context", "population", "seed", "service_budget", "delta_c", "delta_co2_t", "matched", "carbon_factor_version"})
    f = _read_csv(root, "Fig3f", {"context", "population", "seed", "service_budget", "delta_c", "matched", "heating_electricity_change_mwh_signed", "cooling_electricity_change_mwh_signed", "fan_electricity_change_mwh_signed", "gas_hhv_change_mwh_signed", "heating_electricity_reference_mwh", "heating_electricity_target_mwh"})
    g = _read_csv(root, "Fig3g", {"pair_id", "context", "alternative_control", "energy_direction", "co2_direction", "matched", "griffiths"})
    h = _read_csv(root, "Fig3h", {"metric", "sensitivity_type", "sensitivity_value", "classification", "count", "denominator"})

    figure, axes = create_panel_figure()
    figure.subplots_adjust(left=0.14, bottom=0.15, wspace=0.52, hspace=0.50)
    ax = axes.ravel()

    # a: common building boundary and three control families.
    ax[0].set_xlim(0, 3)
    ax[0].set_ylim(0, 2.1)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].axis("off")
    labels = [("Uniform average", "one shared\ndemand"), ("Personal centre", "individual\ncentres"), ("Personal range", "individual\nranges")]
    colors = [PALETTE["uniform"], PALETTE["orange"], PALETTE["green"]]
    for index, ((label, sublabel), color) in enumerate(zip(labels, colors)):
        box = FancyBboxPatch((index + 0.08, 0.84), 0.84, 0.72, boxstyle="round,pad=0.03", facecolor=color, alpha=0.20, edgecolor=color, linewidth=1.1)
        ax[0].add_patch(box)
        ax[0].text(index + 0.5, 1.33, label.replace(" ", "\n", 1), ha="center", va="center")
        ax[0].text(index + 0.5, 1.02, sublabel, ha="center", va="center")
        ax[0].annotate("", xy=(index + 0.5, 0.61), xytext=(index + 0.5, 0.83), arrowprops={"arrowstyle": "->", "lw": 0.8, "color": color})
    footer = "Same building and operating permissions"
    common = FancyBboxPatch((0.12, 0.16), 2.76, 0.42, boxstyle="round,pad=0.03", facecolor="#F1F1F1", edgecolor="#777777", linewidth=0.7)
    ax[0].add_patch(common)
    ax[0].text(1.5, 0.37, footer, ha="center", va="center")
    if len(a["items"]) != 3:
        raise ValueError("Fig3a must define exactly three controls")
    _finish_axis(ax[0], "a", "Three control families", None)

    # b: 12 population-context rows; all three seeds remain as thin intervals/points.
    ylabels, ypositions = [], []
    cursor = 0
    population_order = ["us_archive_stratum", "multinational_source_balanced", "india_archive_stratum"]
    for context in CONTEXT_ORDER:
        for population in population_order:
            group = b.loc[(b["context"] == context) & (b["population"] == population)].sort_values("seed")
            if len(group) != 3:
                raise ValueError("Fig3b requires three frozen seeds per population-context")
            yy = cursor
            for offset, row in zip([-0.15, 0, 0.15], group.itertuples()):
                ax[1].hlines(yy + offset, row.p10_c, row.p90_c, color=CONTEXT_COLORS[context], linewidth=0.55, alpha=0.55)
                ax[1].scatter(row.centre_c, yy + offset, s=9, color=CONTEXT_COLORS[context], zorder=3)
            ax[1].hlines(yy, group["p10_c"].median(), group["p90_c"].median(), color=CONTEXT_COLORS[context], linewidth=2.4)
            ax[1].scatter(group["centre_c"].median(), yy, s=25, color=CONTEXT_COLORS[context], edgecolors="white", linewidths=0.4, zorder=4)
            ypositions.append(yy)
            ylabels.append(f"{CONTEXT_LABELS[context]} · {POPULATION_LABELS[population]}")
            cursor += 1
    ax[1].set_yticks([1, 4, 7, 10], ["Miami", "New Delhi", "Port Angeles", "Seattle"])
    ax[1].invert_yaxis()
    ax[1].set_xlim(18, 31)
    ax[1].set_xticks([20, 24, 28])
    ax[1].set_xlabel("Assigned demand temperature (°C)")
    _finish_axis(ax[1], "b", "Assigned population demand", "x")

    # c: the pre-specified 5 x 4 orthogonal grid, shown without simulated outcomes.
    deltas = c["items"][0]["delta_c"]
    lambdas = c["items"][0]["lambda"]
    for yy, lam in enumerate(lambdas):
        for xx, delta in enumerate(deltas):
            length = 0.07 + 0.12 * float(lam)
            ax[2].hlines(yy, xx - length, xx + length, color=PALETTE["blue"], linewidth=1.3 + float(lam))
            face = "white" if (delta == 0 and lam == 1) else PALETTE["orange"]
            ax[2].scatter(xx, yy, s=38, facecolor=face, edgecolor=PALETTE["orange"], linewidth=1.0, zorder=3)
    ax[2].set_xticks(range(len(deltas)), [f"{value:+g}" for value in deltas])
    ax[2].set_yticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    ax[2].set_xlabel("Demand-location shift Δ (°C)")
    ax[2].set_ylabel("Spread scale λ")
    ax[2].text(0.98, 0.04, "open = reference (0, 1)", ha="right", va="bottom", transform=ax[2].transAxes)
    _finish_axis(ax[2], "c", "Centre × spread perturbation", "both")

    # d/e: matched medians and IQR only; matched n values use an aligned legend table.
    def plot_context_lines(panel_axis: Axes, frame: pd.DataFrame, value: str, ylabel: str, letter: str, title: str) -> None:
        summary = _context_line_summary(frame, value)
        n_vectors: dict[str, str] = {}
        for context in CONTEXT_ORDER:
            group = summary.loc[summary["context"] == context].sort_values("delta_c")
            xvals = group["delta_c"].to_numpy(float)
            med = group["median"].to_numpy(float)
            q1 = group["q1"].to_numpy(float)
            q3 = group["q3"].to_numpy(float)
            n_vector = "·".join(str(int(value)) for value in group["n"])
            n_vectors[context] = n_vector
            panel_axis.plot(
                xvals,
                med,
                marker="o",
                linewidth=1.2,
                color=CONTEXT_COLORS[context],
                label=CONTEXT_LEGEND_LABELS[context],
            )
            panel_axis.fill_between(xvals, q1, q3, color=CONTEXT_COLORS[context], alpha=0.24)
        panel_axis.axhline(0, color="#333333", linewidth=0.7)
        panel_axis.set_xticks([-2, -1, 1, 2])
        panel_axis.set_xlabel("Demand-centre shift Δ (°C)")
        panel_axis.set_ylabel(ylabel)
        _finish_axis(panel_axis, letter, title, "y")
        lower, original_upper = panel_axis.get_ylim()
        original_range = original_upper - lower
        panel_axis.set_ylim(lower, original_upper + original_range * 1.70)
        header = panel_axis.text(
            0.5,
            0.93,
            "matched n by Δ",
            ha="center",
            va="center",
            transform=panel_axis.transAxes,
        )
        header.set_gid(f"fig3{letter}-n-table")
        table_positions = [(0.02, 0.81), (0.02, 0.69), (0.02, 0.57), (0.02, 0.45)]
        table_labels = ["Miami", "New Delhi", "Port Angeles", "Seattle"]
        for context, label, (x_position, y_position) in zip(
            CONTEXT_ORDER,
            table_labels,
            table_positions,
        ):
            table_text = panel_axis.text(
                x_position,
                y_position,
                f"{label} {n_vectors[context]}",
                ha="left",
                va="center",
                color=CONTEXT_COLORS[context],
                transform=panel_axis.transAxes,
            )
            table_text.set_gid(f"fig3{letter}-n-table")
        if value == "delta_energy_mwh":
            panel_axis.set_yticks([-50, 0, 50, 100])
        else:
            panel_axis.set_yticks([-25, 0, 25, 50, 75])

    plot_context_lines(ax[3], d, "delta_energy_mwh", "Δ energy (MWh yr⁻¹)", "d", "Annual energy redistribution")
    plot_context_lines(ax[4], e, "delta_co2_t", "Δ CO₂ (t yr⁻¹)", "e", "Annual carbon redistribution")

    # f: median signed changes come directly from amended fields; unmatched rows excluded and counted.
    matched = f.loc[f["matched"].astype(str).str.lower().eq("true")].copy()
    carrier_fields = [
        ("heating_electricity_change_mwh_signed", "Heating", PALETTE["orange"]),
        ("cooling_electricity_change_mwh_signed", "Cooling", PALETTE["blue"]),
        ("fan_electricity_change_mwh_signed", "Fans", PALETTE["green"]),
        ("gas_hhv_change_mwh_signed", "Gas", PALETTE["purple"]),
    ]
    x = np.arange(len(CONTEXT_ORDER))
    pos = np.zeros(len(CONTEXT_ORDER))
    neg = np.zeros(len(CONTEXT_ORDER))
    for field, label, color in carrier_fields:
        values = np.array([matched.loc[matched["context"] == context, field].median() for context in CONTEXT_ORDER])
        bottoms = np.where(values >= 0, pos, neg)
        ax[5].bar(x, values, bottom=bottoms, color=color, width=0.68, label=label)
        pos += np.where(values >= 0, values, 0)
        neg += np.where(values < 0, values, 0)
    energy_summary = _context_line_summary(d.loc[d["delta_c"] == 2], "delta_energy_mwh").set_index("context")
    context_ns: list[int] = []
    for index, context in enumerate(CONTEXT_ORDER):
        total = float(energy_summary.loc[context, "median"])
        n = int((matched["context"] == context).sum())
        context_ns.append(n)
        ax[5].scatter(index, total, marker="D", s=30, facecolor="white", edgecolor="#222222", zorder=4)
    ax[5].axhline(0, color="#333333", linewidth=0.7)
    context_tick_labels = [
        f"{label}\nn={n}"
        for label, n in zip(["Miami", "New\nDelhi", "Port\nAngeles", "Seattle"], context_ns)
    ]
    ax[5].set_xticks(x, context_tick_labels)
    ax[5].set_yticks([-20, -10, 0, 10])
    ax[5].set_ylabel("Median signed change (MWh yr⁻¹)")
    _finish_axis(ax[5], "f", "Carrier redistribution at Δ=+2°C", "y")
    data_safe_legend(ax[5], ncol=2)
    ax[5].set_ylim(-22, 40)
    match_note = ax[5].text(
        0.98,
        0.54,
        "59 matched · 85 unmatched",
        ha="right",
        va="center",
        transform=ax[5].transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.2},
    )
    match_note.set_gid("fig3f-match-note")

    # g: retain every one of the 130 pair definitions, including unmatched cases.
    bars = []
    for alternative in ["personal_center", "ltpref_interval"]:
        for context in CONTEXT_ORDER:
            bars.append((alternative, context))
    counts = g.groupby(["alternative_control", "context", "energy_direction"]).size()
    x = np.arange(len(bars))
    bottoms = np.zeros(len(bars))
    for outcome in OUTCOME_ORDER[:5]:
        values = np.array([counts.get((alternative, context, outcome), 0) for alternative, context in bars])
        short_outcome = {"same_direction_decrease": "Decrease", "same_direction_increase": "Increase", "all_tied": "Tie", "direction_not_stable": "Unstable", "match_lost": "Unmatched"}[outcome]
        ax[6].bar(x, values, bottom=bottoms, color=OUTCOME_COLORS[outcome], width=0.76, label=short_outcome, hatch="//" if outcome == "match_lost" else None)
        for index, (value, bottom) in enumerate(zip(values, bottoms)):
            if value > 0:
                ax[6].text(index, bottom + value / 2, str(int(value)), ha="center", va="center")
        bottoms += values
    labels = [f"{'Personal centre' if alt == 'personal_center' else 'Personal range'}\n{CONTEXT_LABELS[ctx]}" for alt, ctx in bars]
    ax[6].set_xticks(x, labels, rotation=90, ha="center")
    for label in ax[6].get_xticklabels():
        label.set_linespacing(0.8)
    ax[6].set_ylabel("Paired definitions (count)")
    ax[6].set_ylim(0, max(bottoms) + 20)
    ax[6].legend(
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        handlelength=0.8,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    _finish_axis(ax[6], "g", "Three-control paired outcomes", "y")

    # h: three frozen sensitivity families, with their explicit denominators.
    sensitivity_order = ["griffiths_energy", "griffiths_carbon", "interval_width_energy"]
    sensitivity_labels = ["g · energy", "g · CO₂", "width · energy"]
    grouped = h.groupby(["sensitivity_type", "classification"], as_index=False).agg(count=("count", "sum"), denominator=("denominator", "first"))
    yy = np.arange(len(sensitivity_order))
    left = np.zeros(len(sensitivity_order))
    for outcome in OUTCOME_ORDER:
        values = []
        for sensitivity in sensitivity_order:
            row = grouped.loc[(grouped["sensitivity_type"] == sensitivity) & (grouped["classification"] == outcome)]
            values.append(0.0 if row.empty else float(row.iloc[0]["count"]) / float(row.iloc[0]["denominator"]) * 100)
        values_array = np.asarray(values)
        ax[7].barh(yy, values_array, left=left, color=OUTCOME_COLORS[outcome], height=0.62, hatch="//" if outcome in {"match_lost", "all_unmatched"} else None)
        left += values_array
    denominators = [int(h.loc[h["sensitivity_type"] == value, "denominator"].iloc[0]) for value in sensitivity_order]
    ax[7].set_yticks([])
    ax[7].invert_yaxis()
    ax[7].set_xlim(-25, 100)
    for y_value, label, denominator in zip(yy, sensitivity_labels, denominators):
        ax[7].text(-1.5, y_value, f"{label}\nn={denominator}", ha="right", va="center")
    ax[7].set_xticks([0, 50, 100])
    ax[7].set_xlabel("Outcome share (%)")
    _finish_axis(ax[7], "h", "Human-demand sensitivity", "x")

    return figure


FIGURE_BUILDERS: dict[int, Callable[[Path | str], Figure]] = {
    1: build_figure_1,
    2: build_figure_2,
    3: build_figure_3,
}


CAPTIONS = {
    1: (
        "**Figure 1 | Observed human demand is geographically structured but dominated by "
        "within-building heterogeneity.** **a,** The complete office archive contains 21,418 records "
        "from 12,130 analysis units; the executable set retains 21,399 records from 12,113 "
        "analysis units in 187 buildings, 40 cities and 15 countries. **b,c,** Observed-city warm- "
        "and cold-season demand centres. Filled circles denote season-supported cities, hollow "
        "circles observed but unsupported cities, circle area the number of analysis units and "
        "colour the demand centre on the same 19.5–30.6 °C scale; unsampled geography is left "
        "blank and no spatial interpolation is applied. **d,** Empirical P10–P90 intervals and "
        "medians for city- and source-level warm/cold aggregates under analysis-unit-equal "
        "weighting; labels give the number of aggregates and median interval width. **e,** "
        "Hierarchical variance allocation across country (22.3%), city within country (7.7%), "
        "building within city (5.8%), analysis unit within building (56.1%) and repeated "
        "measurements (8.2%). **f,** Weighted demand records below, within and above nine bilateral "
        "policy or design ranges at Griffiths coefficient 0.50; labels give the within-range "
        "share and eligible-record denominator, and whiskers are nested-bootstrap 95% confidence "
        "intervals for the within-range share. **g,** Empirical P10–P90 (thin), P25–P75 (thick) and "
        "median (point) of demand centre minus campaign target for three prespecified campaigns; "
        "positive values indicate a demand centre warmer than the target, and negative values "
        "indicate a demand centre cooler than the target. **h,** Leave-one-source sensitivity ranges "
        "and medians across 12 coefficient/weighting specifications, each based on 27 source "
        "removals. Policy documents in **f,g** are external temperature coordinates rather than "
        "individual comfort ground truth."
    ),
    2: (
        "**Figure 2 | Independent shared-room votes reveal demand disagreement hidden by "
        "averages.** **a,** Prespecified construction of the NUS 30-min validation cohort from "
        "4,378 raw rows to 63 temperature-homogeneous room-windows containing 225 valid "
        "person-window votes. **b,** Cooler, no-change and warmer vote shares in all 63 "
        "room-windows, ordered by net vote and disagreement; 49 windows contain more than one "
        "response category and six contain warmer and cooler requests within the same 30-min "
        "room-window. **c,** Vote counts in the six room-windows with within-window opposing "
        "requests; grey circles denote no-change votes and labels give vote count, mean room "
        "temperature and within-window temperature span. **d,** Full request burden, mean "
        "absolute vote, versus burden after averaging, absolute mean vote, for each window. "
        "Hollow orange points are exact cancellations; the dashed line is equality and the "
        "median compression gap is zero despite a positive mean gap. **e,** The four exact-"
        "cancellation windows, in which non-zero full burden is represented as zero after "
        "averaging. **f,** False acceptance and false rejection at burden thresholds 0, 0.25, "
        "0.50 and 0.75; all four thresholds retain zero false rejections and 2–5 false "
        "acceptances among 63 windows. **g,** Disagreement, within-window opposing requests and "
        "mean Jensen gap under 15-, 30- and 60-min windows; whiskers are crossed room/date "
        "cluster-bootstrap 95% confidence intervals and n is the eligible-window denominator. "
        "**h,** Independent ENTH replication (34 windows, 81 votes) beside NUS (63 windows, 225 "
        "votes). Windows, rather than votes, are the inferential unit throughout."
    ),
    3: (
        "**Figure 3 | Demand location redistributes annual energy and carbon under three "
        "controls.** **a,** Uniform average, personal centre and personal range control "
        "families generate physical actions in the same building under identical operating "
        "permissions. These controls are distinct from the mean and full-distribution "
        "qualification representations used to score the same actions. **b,** Assigned P10–P90 "
        "demand intervals, centres and seed variation for three population compositions in four "
        "building/climate contexts. **c,** Prespecified orthogonal transformation "
        "T′=μ+Δ+λ(T−μ), with demand-location shift Δ from −2 to +2 °C and spread scale λ from 0 "
        "to 1.5; the open point is the Δ=0, λ=1 reference. **d,e,** Median and interquartile "
        "range of matched service-qualified changes in purchased space-HVAC site energy and "
        "operational CO₂ relative to the reference. The coloured denominator tables give "
        "matched n for Miami, New Delhi, Port Angeles and Seattle in left-to-right Δ order (−2, "
        "−1, +1 and +2 °C). **f,** Median signed target-minus-reference changes in heating "
        "electricity, cooling electricity, fan electricity and gas HHV for Δ=+2 °C, using the "
        "amended reference/target carrier fields directly; diamonds show median total-energy "
        "change and the two-line context labels give 22, 20, 8 and 9 matched cells. The artwork "
        "states the full accounting boundary of 59 matched and 85 unmatched design rows; "
        "unmatched rows contribute no fabricated delta. **g,** Energy direction for all 130 "
        "default uniform-versus-alternative paired definitions, separated by control and "
        "context: 54 increases, 3 decreases, 36 ties, 7 direction-unstable comparisons and 30 "
        "unmatched comparisons. **h,** Outcome shares across Griffiths energy, Griffiths carbon "
        "and personal range-width sensitivity families with explicit denominators and the same outcome "
        "colour/hatch key as panel g. Population assignments are controlled archive scenarios "
        "rather than claims of local population representativeness; carbon factors are fixed "
        "regional scenarios."
    ),
}


ALT_TEXT = {
    1: (
        "Eight-panel portrait figure. An archive flow retains 21,399 of 21,418 records. Two "
        "observed-city maps show 31 warm-season and 26 cold-season supported cities without "
        "interpolation. Ranked empirical intervals show broad within-city and within-source "
        "demand ranges. A variance bar assigns the largest share, 56.1%, to analysis units "
        "within buildings. Policy-range bars show wide variation in observed-demand coverage. "
        "Panel g plots demand centre minus campaign target: positive values mean that the demand "
        "centre is warmer than the target, and negative values mean that it is cooler. "
        "Leave-one-source sensitivity preserves the importance of within-building heterogeneity, "
        "while showing that its magnitude depends on weighting and the Griffiths coefficient."
    ),
    2: (
        "Eight-panel portrait figure. A filtering flow yields 63 shared-room windows and 225 "
        "direct votes. A 63-row composition chart shows disagreement in 49 windows, including "
        "six with warmer and cooler requests within the same 30-min room-window. Six diverging "
        "bars detail those within-window opposing requests. A scatter shows that averaging can "
        "only preserve or reduce represented request burden and exactly cancels four non-neutral "
        "windows. Threshold bars show false acceptance but no false rejection. Results persist "
        "across 15-, 30- and 60-min windows and replicate in a second ENTH dataset, although "
        "confidence intervals are wider there."
    ),
    3: (
        "Eight-panel portrait figure. Panel a compares three control families—Uniform average, "
        "personal centre and personal range controls—under the same building and operating "
        "permissions. Mean and full-distribution are separate qualification representations "
        "used to score generated actions, not additional controls. Population scenarios differ "
        "in both demand centre and width. A centre-by-spread grid defines independent "
        "perturbations. Energy and carbon response curves change sign across contexts: warmer "
        "demand reduces median annual energy and carbon in Miami and New Delhi but increases "
        "them in Port Angeles and Seattle. Coloured tables above both curves report the matched "
        "denominator at each demand shift without covering the data. Carrier bars show "
        "concurrent shifts among heating electricity, cooling electricity, fans and gas and "
        "state that 59 rows match while 85 remain unmatched. Stacked outcome bars retain "
        "increases, decreases, ties, unstable directions and unmatched cases. Sensitivity bars "
        "reuse that outcome key and show that direction and match availability depend on demand "
        "assumptions, so personalization does not imply universal savings."
    ),
}


def close_figure(figure: Figure) -> None:
    """Close a rendered figure without relying on caller pyplot state."""

    plt.close(figure)
