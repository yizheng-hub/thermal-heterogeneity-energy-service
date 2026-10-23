#!/usr/bin/env python3
"""Render the result-first Stage 7 main figures from frozen panel-data v2.

This script performs presentation-only changes. It imports the frozen panel
renderers, removes the specified conceptual panels, reflows the retained axes,
and exports a complete publication bundle. It never edits the panel inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from PIL import Image


STAGE_REL = Path("results/stage7_service_first_revision_20261022")
PANEL_REL = Path("data/figure_panel_data")
RELEASE_REL = Path("release/thermal-heterogeneity")
WIDTH_MM = 180
HEIGHT_MM = 230
SIZE_IN = (WIDTH_MM / 25.4, HEIGHT_MM / 25.4)
PNG_600 = (4252, 5433)
PNG_WORD = (2126, 2717)
FONT_FAMILY = "Times New Roman"
BASE_FONT_PT = 11
TITLE_FONT_PT = 12


MAPPING: dict[int, list[str]] = {
    1: ["Fig1b", "Fig1c", "Fig1d", "Fig1e", "Fig1f", "Fig1g", "Fig1h", "Fig2h"],
    2: ["Fig2b", "Fig2c", "Fig2d", "Fig2e", "Fig2f", "Fig2g", "Fig2h"],
    3: ["Fig3b", "Fig3d", "Fig3e", "Fig3f", "Fig3g", "Fig3h"],
    4: ["Fig4b", "Fig4c", "Fig4d", "Fig4e", "Fig4f", "Fig4g", "Fig4h"],
    5: ["Fig5b", "Fig5c", "Fig5d", "Fig5e", "Fig5f", "Fig5g", "Fig5h"],
    6: ["Fig6b", "Fig6c", "Fig6d", "Fig6e", "Fig6f", "Fig6g", "MAIN_FIG6G_Q5_REGRET"],
}

EXCLUDED = {"Fig1a", "Fig2a", "Fig3a", "Fig3c", "Fig4a", "Fig5a", "Fig6a", "Fig6h"}


CAPTIONS = {
    1: "Figure 1 | Observed thermal demand and shared-room requests. a,b, Reconstructed city "
    "centres in warm/cold seasons. Marker area denotes analysis-unit count, colour denotes "
    "temperature, and hollow markers denote unsupported summaries. Support is 31/40 warm and "
    "26/40 cold cities. Maps contain no interpolation. c, Medians and P10–P90 ranges by city "
    "and source (Src), for warm (W) and cold (C) seasons. d, Unit-equal hierarchical variance "
    "shares. e, Record-equal shares below, within and above nine policy/design ranges at "
    "Griffiths coefficient 0.50. Labels give within-range percentages, whiskers give nested-bootstrap "
    "95% intervals, and country codes identify jurisdictions. f, Demand minus campaign target. "
    "Thin/thick segments give P10–P90/P25–P75 and points give medians. g, Leave-one-source "
    "ranges and medians across 12 coefficient/weighting specifications and 27 source removals. "
    "h, Direct requests in 63 NUS and 34 ENTH room-windows. Categories are mutually exclusive: "
    "identical (Same), different without opposing requests (Mixed), and simultaneous warmer/cooler "
    "requests (Opposed). Labels give window counts. Complete denominators accompany Supplementary Data.",
    2: "Figure 2 | Shared-room vote distributions and averaging. a, Cooler, no-change and warmer vote "
    "shares across sixty-three NUS room-windows, ordered by net vote and disagreement. "
    "Blue/grey/terracotta encode the three vote categories. b, Counts in the six windows "
    "containing both cooler and warmer requests. Labels give vote count, mean temperature and "
    "within-window temperature span. c, Mean absolute vote (full request burden) versus absolute "
    "mean vote (burden after averaging). Hollow terracotta points mark exact cancellation, and the "
    "dashed line is equality. d, Four exact-cancellation windows. Labels 1C/1W mean one cooler and "
    "one warmer vote. Total n includes all votes. e, False acceptance and rejection at four burden "
    "thresholds. f, Disagreement, opposing requests and mean Jensen gap (full minus net request "
    "burden) across window lengths, with crossed room/date cluster-bootstrap 95% intervals. g, "
    "ENTH (34 windows, 81 votes) and NUS (63 windows, 225 votes). Windows are the inferential "
    "unit. Definitions are given in Supplementary Section S7.",
    3: "Figure 3 | Annual energy and carbon under demand-centre perturbations. Panels a–d use exchanged archive pools and a common plan library. Panels e,f use separately retargeted controls. a, Assigned P10–P90 intervals and centres in four building/weather contexts. Thin/thick marks show seeds/medians. Terracotta/mauve/blue/green denote Miami/New Delhi/Port Angeles/Seattle. b,c, Median and interquartile matched changes in purchased HVAC site energy and operational CO₂ relative to Δ=0, λ=1. Count rows follow Δ=−2, −1, +1, +2 °C. d, Median signed carrier changes at Δ=+2 °C, with diamonds showing totals (59 matched, 85 unmatched). e, Three-coefficient directions among 130 comparisons matched at the default demand definition: 54 increases, 3 decreases, 36 ties, 7 unstable, 30 unmatched. Circle area and labels encode counts. Small grey dots denote zero. PC/LT denote personal-centre/LTPref controls. f, Griffiths energy/carbon and interval-width outcome shares, using e’s key. Mia/Del/PA/Sea identify contexts, Multi the multinational archive pool, and E energy. Carbon uses fixed regional factors.",
    4: "Figure 4 | Annual service qualification across demand-spread scenarios. a, Fixed-centre "
    "density ridges for the Miami multinational composition, scaled separately to equal height. "
    "Thick segments show P10–P90. b, Full-demand and mean-based selections among 144 design cells "
    "at each spread multiplier λ. c, Mean-based selections failing full-demand review (117/128 at "
    "λ=1). d, Eligible-action distributions, including zero-action cells. Boxes show medians and "
    "interquartile ranges, whiskers P10–P90, and points individual cells. Annotations give medians "
    "and zero counts. e, Median and interquartile energy changes for paired full-service "
    "selections. f, Carbon medians/interquartile ranges and signed carrier changes for those "
    "comparisons. g, Median eligibility interactions relative to the additive centre-plus-spread "
    "prediction. The f,g heatmaps use separate shared signed scales, with blue negative and "
    "terracotta positive. Selected-energy interactions are resolved in 585/1,728 cells. Full "
    "review uses population-mean and fixed-27 tail absolute temperature deviations from "
    "reconstructed demand centres. Definitions and matched denominators are in Supplementary "
    "Section S9 and Supplementary Data.",
    5: "Figure 5 | Event response and recovery under demand-spread scenarios. a,b, HVAC power, "
    "temperature and service for the prespecified Miami July action at Δ=0, λ=1 and service "
    "fraction 0.9. Grey dashed/blue solid traces denote baseline/perturbed operation. Gold and "
    "pale-blue backgrounds mark the two-hour event and forty-eight-hour recovery. In b, shading "
    "spans zone minimum/maximum temperatures, temperature lines give occupied means, and service "
    "lines give population-mean absolute deviations from reconstructed demand centres. Dotted "
    "lines give fixed-27 tail deviations. Horizontal grey/terracotta lines mark mean/tail budgets. "
    "c,d, July positive full-demand and mean-based selections, and mean-selected false accepts "
    "(78/122 at λ=1). e, Response in twelve common full-service cells at λ=1 and 1.5. f, Changes "
    "from λ=1 to 1.5 in response, recovery peak, recovery electricity and fifty-hour net savings. "
    "Rows have separate symmetric scales labelled by ± limits. Blue/grey/terracotta denote "
    "negative/zero/positive changes. g, July/February positive-selection counts on a common scale. "
    "There are 144 design cells per setting. February has 0/144 at the central reference and "
    "210/2,880 over its complete grid.",
    6: "Figure 6 | Demand summaries, service qualification and annual decision cost. a,b, Classes across demand-centre spread and service budgets. Safe denotes mean/full agreement on every candidate's eligibility and the selected plan. Distribution denotes disagreement despite an available full-qualified plan. Unresolved denotes no full-qualified plan. c, False-safe/false-required errors at the 1.059 °C threshold. d, Median and interquartile absolute tail-score errors and eligibility disagreement against full review. e, Mean-only/q3/q5 results (grey/gold/green) under context-seed/population holdout (circles/squares). q3/q5 retain three/five near-equal-count bin means per zone, with population and fixed-tail counts. In context-seed holdout, outlined q5+q95 retains 14,709/18,766 full-qualified candidate evaluations and covers 542/606 solvable cases, without false acceptance observed in tested folds. f, Ordinal ranked probability score (RPS) and tail-share mean absolute error (MAE) under city/source holdouts, with 95% bootstrap intervals. g, Extra annual HVAC energy versus each case's full-information minimum-energy qualified plan. Grey diamonds/lines show medians/P10–P90. Labels give matched/available counts. The 64 protected-unmatched and 402 full-unavailable cases are excluded.",
}


LONG_CAPTIONS = dict(CAPTIONS)
CAPTIONS = dict(LONG_CAPTIONS)


ALT_TEXT = dict(CAPTIONS)


LAYOUTS: dict[int, list[tuple[float, float, float, float]]] = {
    1: [
        (0.070, 0.790, 0.900, 0.180),
        (0.070, 0.585, 0.900, 0.180),
        (0.070, 0.430, 0.410, 0.120),
        (0.565, 0.430, 0.405, 0.120),
        (0.070, 0.220, 0.430, 0.145),
        (0.580, 0.220, 0.390, 0.145),
        (0.070, 0.045, 0.900, 0.130),
    ],
    2: [
        (0.070, 0.820, 0.900, 0.150),
        (0.070, 0.580, 0.410, 0.165),
        (0.565, 0.580, 0.405, 0.165),
        (0.070, 0.350, 0.410, 0.165),
        (0.565, 0.350, 0.405, 0.165),
        (0.070, 0.050, 0.410, 0.230),
        (0.565, 0.050, 0.405, 0.230),
    ],
    3: [
        (0.075, 0.720, 0.405, 0.250),
        (0.565, 0.720, 0.405, 0.250),
        (0.075, 0.395, 0.405, 0.250),
        (0.565, 0.395, 0.405, 0.250),
        (0.075, 0.055, 0.405, 0.260),
        (0.565, 0.055, 0.405, 0.260),
    ],
    4: [
        (0.090, 0.755, 0.880, 0.215),
        (0.070, 0.520, 0.410, 0.180),
        (0.565, 0.520, 0.405, 0.180),
        (0.070, 0.285, 0.410, 0.180),
        (0.565, 0.285, 0.405, 0.180),
        (0.070, 0.045, 0.410, 0.185),
        (0.565, 0.045, 0.405, 0.185),
    ],
    5: [
        (0.070, 0.850, 0.900, 0.120),
        (0.070, 0.650, 0.900, 0.150),
        (0.070, 0.450, 0.410, 0.140),
        (0.565, 0.450, 0.405, 0.140),
        (0.070, 0.270, 0.410, 0.140),
        (0.565, 0.270, 0.405, 0.140),
        (0.070, 0.015, 0.900, 0.200),
    ],
    6: [
        (0.070, 0.750, 0.900, 0.220),
        (0.070, 0.515, 0.410, 0.180),
        (0.565, 0.515, 0.405, 0.180),
        (0.070, 0.280, 0.410, 0.180),
        (0.565, 0.280, 0.405, 0.180),
        (0.070, 0.045, 0.410, 0.180),
        (0.565, 0.045, 0.405, 0.180),
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(panel_root: Path) -> dict[str, str]:
    return {
        path.relative_to(panel_root).as_posix(): sha256(path)
        for path in sorted(panel_root.rglob("*"))
        if path.is_file()
    }


def retitle(axis, letter: str) -> None:
    old = axis.get_title(loc="left")
    title = re.sub(r"^\([a-h]\)\s*", "", old)
    artist = axis.set_title(
        f"({letter}) {title}",
        loc="left",
        fontsize=TITLE_FONT_PT,
        fontweight="bold",
        pad=9,
    )
    artist.set_gid("stage7-result-panel-title")


def restructure(figure, figure_number: int):
    top_axes = list(figure.axes)
    keep_indices = {
        1: [1, 2, 3, 4, 5, 6, 7],
        2: [1, 2, 3, 4, 5, 6, 7],
        3: [1, 3, 4, 5, 6, 7],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [1, 2, 3, 4, 5, 6, 7],
        6: [1, 2, 3, 4, 5, 6, 7],
    }[figure_number]
    retained = [top_axes[index] for index in keep_indices]
    for index, axis in enumerate(top_axes):
        if index not in keep_indices:
            figure.delaxes(axis)
    figure.set_size_inches(*SIZE_IN, forward=True)
    for index, (axis, bounds) in enumerate(zip(retained, LAYOUTS[figure_number])):
        axis.set_position(bounds)
        retitle(axis, chr(ord("a") + index))
    if figure_number == 1:
        # The two full-width maps share the lower map's coordinate ticks. Removing
        # the upper x labels avoids duplicating longitude and protects the second
        # map title; the lower xlabel is also omitted because the degree ticks are
        # self-explanatory at journal scale.
        retained[0].set_xlabel("")
        retained[0].tick_params(axis="x", bottom=False, labelbottom=False)
        retained[1].set_xlabel("")
        for artist in retained[0].texts:
            if artist.get_text() == "19.5–30.6 °C":
                artist.set_x(-32)
        for artist in retained[1].texts:
            if artist.get_text() == "area=n":
                artist.set_text("marker area scales with n")
                artist.set_position((-126, -54))
            elif artist.get_text() == "no support":
                artist.set_position((110, -54))
        for collection in retained[1].collections:
            offsets = collection.get_offsets()
            if len(offsets) == 1 and abs(float(offsets[0, 0]) + 70) < 1e-8:
                collection.set_offsets([[95, -54]])

        policy_axis = retained[4]
        if policy_axis.get_legend() is not None:
            policy_axis.get_legend().remove()
        policy_axis.set_ylim(8.5, -3.0)
        direct_key = [
            (0.16, "Below", "#6F9EBC"),
            (0.49, "Within", "#D2D2D2"),
            (0.82, "Above", "#C78470"),
        ]
        for x, label, colour in direct_key:
            policy_axis.scatter(
                [x - 0.055],
                [0.94],
                marker="s",
                s=45,
                color=colour,
                transform=policy_axis.transAxes,
                clip_on=False,
                zorder=5,
            )
            policy_axis.text(
                x,
                0.94,
                label,
                transform=policy_axis.transAxes,
                ha="left",
                va="center",
            )
    elif figure_number == 2:
        retained[0].set_xlabel("Vote share (%)")
        retained[4].set_xlabel("Threshold")
    elif figure_number == 3:
        retained[0].set_xlim(14, 31)
        for label_artist in retained[0].texts:
            if label_artist.get_text().startswith(("Mia", "Del", "PA", "Sea")):
                label_artist.set_x(14.15)
        retitle_axis = retained[5]
        artist = retitle_axis.set_title(
            "(f) Demand-model sensitivity",
            loc="left",
            fontsize=TITLE_FONT_PT,
            fontweight="bold",
            pad=2,
        )
        artist.set_gid("stage7-result-panel-title")
        sensitivity_labels = {
            "g · energy\nn=130": "Griffiths E\nn=130",
            "g · CO₂\nn=130": "Griffiths CO₂\nn=130",
            "width · energy\nn=288": "Interval width\nn=288",
        }
        for label_artist in retitle_axis.texts:
            if label_artist.get_text() in sensitivity_labels:
                label_artist.set_text(sensitivity_labels[label_artist.get_text()])
    elif figure_number == 4:
        for artist in list(retained[2].texts):
            if artist.get_text() == "λ":
                artist.remove()
        retained[2].set_xlabel("Demand spread λ")
        for child in retained[6].child_axes:
            for artist in child.texts:
                if artist.get_text() in {"Miami", "New Delhi", "Port Angeles", "Seattle"}:
                    artist.set_position((0.05, 0.90))
                    artist.set_bbox(
                        {"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.4}
                    )
    elif figure_number == 5:
        retained[0].set_xlabel("")
        trajectory_legend = retained[0].get_legend()
        if trajectory_legend is not None:
            trajectory_legend.set_frame_on(True)
            trajectory_legend.get_frame().set_facecolor("white")
            trajectory_legend.get_frame().set_edgecolor("none")
            trajectory_legend.get_frame().set_alpha(0.92)
        retained[2].set_xlabel("")
        for artist in list(retained[3].texts):
            if artist.get_text() == "λ":
                artist.remove()
        retained[3].set_xlabel("")

        service_parent = retained[1]
        for child in service_parent.child_axes:
            child.set_xlabel("")
            ylabel = child.get_ylabel()
            if ylabel.startswith("Temp."):
                child.set_ylabel("Temp.", labelpad=9)
            elif ylabel.startswith("Service"):
                child.set_ylabel("Service", labelpad=9)

        recovery_parent = retained[5]
        recovery_labels = [
            "Response (kW)",
            "Rebound (kW)",
            "Extra (kWh)",
            "Net (kWh)",
        ]
        recovery_rows = [0.78, 0.58, 0.35, 0.15]
        recovery_values = []
        for child in list(recovery_parent.child_axes):
            offsets = child.collections[0].get_offsets()
            recovery_values.append([float(value) for value in offsets[:, 0]])
            child.remove()
        for text_artist in list(recovery_parent.texts):
            text_artist.remove()
        recovery_parent.set_xlim(0, 1)
        recovery_parent.set_ylim(0, 1)
        recovery_parent.set_xticks([])
        recovery_parent.set_yticks([])
        recovery_parent.plot([0.66, 0.66], [0.07, 0.88], color="#777777", linewidth=0.65)
        for label, row_y, values in zip(recovery_labels, recovery_rows, recovery_values):
            maximum = max(1.0, max(abs(value) for value in values))
            limit = math.ceil(maximum / 5) * 5
            positions = [0.66 + 0.14 * value / limit for value in values]
            colours = [
                "#6F9EBC" if value < -1e-9 else "#C78470" if value > 1e-9 else "#A5A5A5"
                for value in values
            ]
            jitter = [
                row_y + (index - (len(values) - 1) / 2) * 0.006 for index in range(len(values))
            ]
            recovery_parent.plot([0.48, 0.82], [row_y, row_y], color="#D9D9D9", linewidth=0.65)
            recovery_parent.scatter(positions, jitter, s=14, c=colours, zorder=3)
            lower = sum(value < -1e-9 for value in values)
            tied = sum(abs(value) <= 1e-9 for value in values)
            higher = sum(value > 1e-9 for value in values)
            recovery_parent.text(0.02, row_y, label, ha="left", va="center")
            recovery_parent.text(0.44, row_y, f"±{limit:g}", ha="right", va="center")
            recovery_parent.text(
                0.98, row_y, f"{lower}− | {tied}= | {higher}+", ha="right", va="center"
            )

        boundary_parent = retained[6]
        parent_bounds = boundary_parent.get_position()
        child_bounds = [(0.05, 0.58, 0.41, 0.30), (0.56, 0.58, 0.41, 0.30)]
        for child, (x, y, width, height) in zip(boundary_parent.child_axes, child_bounds):
            child.set_axes_locator(None)
            child.set_position(
                (
                    parent_bounds.x0 + x * parent_bounds.width,
                    parent_bounds.y0 + y * parent_bounds.height,
                    width * parent_bounds.width,
                    height * parent_bounds.height,
                )
            )
        for artist in boundary_parent.texts:
            text = artist.get_text()
            if text == "Demand shift (°C); rows are λ":
                artist.set_position((0.50, 0.43))
            elif text.startswith("July: ref"):
                artist.set_position((0.50, 0.27))
            elif text.startswith("February: ref"):
                artist.set_position((0.50, 0.08))
                artist.set_bbox(None)
    elif figure_number == 6:
        # Protect the long "Resolved" tick label at journal width.
        retained[1].set_position((0.095, 0.515, 0.385, 0.180))
        for artist in retained[4].texts:
            if artist.get_text().startswith("x retention"):
                artist.set_text("Retention (%) | false accepts (log) | size = coverage")
        for child in retained[4].child_axes:
            for artist in child.texts:
                if artist.get_text().startswith("q5+q95"):
                    artist.set_position((10, 7))
        # The local-update protocol scores ordinal preferences with a unitless
        # ranked probability score and tail shares as fractions. The frozen
        # renderer attached temperature units to these two metrics in error.
        feedback_parent = retained[5]
        feedback_bounds = feedback_parent.get_position()
        feedback_children = [(0.10, 0.28, 0.36, 0.31), (0.61, 0.28, 0.36, 0.31)]
        for child, (x, y, width, height) in zip(feedback_parent.child_axes, feedback_children):
            child.set_axes_locator(None)
            child.set_position(
                (
                    feedback_bounds.x0 + x * feedback_bounds.width,
                    feedback_bounds.y0 + y * feedback_bounds.height,
                    width * feedback_bounds.width,
                    height * feedback_bounds.height,
                )
            )
            child.set_xticks([0, 10, 20, 40])
        legend = feedback_parent.get_legend()
        if legend is not None:
            legend.set_bbox_to_anchor((0.50, 0.94))
        for artist in retained[5].texts:
            text = artist.get_text()
            if text == "CRPS (°C)":
                artist.set_text("Ordinal RPS\n(unitless)")
                artist.set_linespacing(0.9)
                artist.set_position((0.28, 0.69))
            elif text == "Tail MAE (°C)":
                artist.set_text("Tail-share MAE\n(fraction)")
                artist.set_linespacing(0.9)
                artist.set_position((0.79, 0.69))
            elif text.startswith("City Δ40: CRPS"):
                artist.set_text("Plotted n40 − n0: RPS −0.004; tail-share −0.017")
    presentation_refinement(figure, retained, figure_number)
    return retained


def presentation_refinement(figure, axes, number):
    """Apply common readable geometry without changing frozen results."""
    from matplotlib.text import Text

    geometry = {
        2: [
            (0.13, 0.85, 0.85, 0.12),
            (0.13, 0.60, 0.34, 0.16),
            (0.64, 0.60, 0.34, 0.16),
            (0.13, 0.375, 0.34, 0.135),
            (0.64, 0.375, 0.34, 0.135),
            (0.13, 0.07, 0.34, 0.21),
            (0.64, 0.07, 0.34, 0.21),
        ],
        3: [
            (0.13, 0.73, 0.34, 0.23),
            (0.64, 0.73, 0.34, 0.23),
            (0.13, 0.40, 0.34, 0.23),
            (0.64, 0.40, 0.34, 0.23),
            (0.13, 0.07, 0.34, 0.23),
            (0.64, 0.07, 0.34, 0.23),
        ],
        4: [
            (0.13, 0.84, 0.85, 0.12),
            (0.13, 0.62, 0.34, 0.13),
            (0.64, 0.62, 0.34, 0.13),
            (0.13, 0.40, 0.34, 0.13),
            (0.64, 0.40, 0.34, 0.13),
            (0.13, 0.08, 0.34, 0.23),
            (0.64, 0.08, 0.34, 0.23),
        ],
        5: [
            (0.13, 0.855, 0.85, 0.105),
            (0.13, 0.635, 0.85, 0.135),
            (0.13, 0.425, 0.34, 0.12),
            (0.64, 0.425, 0.34, 0.12),
            (0.13, 0.235, 0.34, 0.105),
            (0.64, 0.235, 0.34, 0.105),
            (0.13, 0.060, 0.85, 0.105),
        ],
        6: [
            (0.13, 0.76, 0.85, 0.20),
            (0.13, 0.535, 0.34, 0.135),
            (0.64, 0.535, 0.34, 0.135),
            (0.13, 0.31, 0.34, 0.135),
            (0.64, 0.31, 0.34, 0.135),
            (0.13, 0.06, 0.34, 0.16),
            (0.64, 0.06, 0.34, 0.16),
        ],
    }
    figure.canvas.draw()
    if number in geometry:
        for axis, bounds in zip(axes, geometry[number]):
            old = axis.get_position()
            children = []
            for child in axis.child_axes:
                cp = child.get_position()
                children.append(
                    (
                        child,
                        (
                            (cp.x0 - old.x0) / old.width,
                            (cp.y0 - old.y0) / old.height,
                            cp.width / old.width,
                            cp.height / old.height,
                        ),
                    )
                )
            axis.set_position(bounds)
            for child, (x, y, w, h) in children:
                child.set_axes_locator(None)
                child.set_position(
                    (
                        bounds[0] + x * bounds[2],
                        bounds[1] + y * bounds[3],
                        w * bounds[2],
                        h * bounds[3],
                    )
                )
    short_titles = {
        1: [
            "Warm-season demand",
            "Cold-season demand",
            "Demand within places",
            "Variance allocation",
            "Policy-range coverage",
            "Campaign offsets",
            "Source and coefficient sensitivity",
        ],
        2: [
            "Shared-room vote composition",
            "Opposing requests",
            "Loss after averaging",
            "Exact cancellations",
            "Qualification errors",
            "Window-length sensitivity",
            "Independent replication",
        ],
        3: [
            "Assigned demand ranges",
            "Annual energy change",
            "Operational carbon change",
            "Carrier contributions",
            "Three-control outcomes",
            "Demand-model sensitivity",
        ],
        4: [
            "Fixed-centre spread",
            "Qualified selections",
            "Mean-only false accepts",
            "Qualified action counts",
            "Energy consequence",
            "Carbon and carriers",
            "Centre–spread interaction",
        ],
        5: [
            "Power during event and recovery",
            "Temperature and service trajectories",
            "Qualified response choices",
            "Mean-only false accepts",
            "Common-cell response",
            "Recovery consequences",
            "Seasonal response availability",
        ],
        6: [
            "Decision classes across sampled demand settings",
            "Resolvable decisions",
            "Single-threshold errors",
            "Service reconstruction",
            "Held-out qualification",
            "Learning from local votes",
            "Additional annual energy",
        ],
    }
    for i, axis in enumerate(axes):
        axis.set_title(
            f"({chr(97+i)}) {short_titles[number][i]}",
            loc="left",
            fontsize=12,
            fontweight="bold",
            pad=9,
        ).set_gid("stage7-result-panel-title")
    for axis in figure.axes:
        axis.grid(False)
        axis.xaxis.labelpad = 6
        axis.yaxis.labelpad = 7
        axis.tick_params(axis="both", pad=3)
        for child in axis.child_axes:
            child.grid(False)
            child.xaxis.labelpad = 5
            child.yaxis.labelpad = 7
            child.tick_params(axis="both", pad=3)
    # Bboxes cannot serve as an eraser for dense observations.
    for artist in figure.findobj(Text):
        if artist.get_bbox_patch() is not None:
            artist.set_bbox(None)
    if number == 1:
        refine_figure_one(figure, axes)
    elif number == 2:
        axes[2].set_ylabel("Burden after averaging", labelpad=10)
        axes[2].set_xlabel("Full request burden", labelpad=7)
        axes[1].set_xlabel("Cooler ← votes → warmer")
        axes[5].set_ylabel("Rate or gap (%)", labelpad=7)
        axes[6].set_ylabel("Rate or gap (%)", labelpad=7)
        axes[4].set_ylim(0, 10)
        axes[0].set_ylim(-1, 115)
        axes[0].get_legend().set_bbox_to_anchor((0.5, 0.84))
    elif number == 3:
        axes[2].set_ylabel("CO₂ change (t/year)")
        axes[4].set_ylabel("Paired definitions")
        legend = axes[3].get_legend()
        handles, labels = axes[3].get_legend_handles_labels()
        if legend is not None:
            legend.remove()
        axes[3].legend(
            handles,
            labels,
            ncol=2,
            loc="upper left",
            frameon=False,
            handlelength=1,
            columnspacing=0.7,
        )
        axes[3].set_ylim(-22, 25)
        for artist in list(axes[3].texts):
            artist.remove()
        for artist in list(axes[4].texts):
            if re.fullmatch(r"\d+", artist.get_text()):
                artist.remove()
        refine_control_outcomes(figure, axes[4])
        labels = []
        positions = []
        for artist in list(axes[5].texts):
            if artist.get_text().startswith(("Griffiths", "Interval width")):
                positions.append(artist.get_position()[1])
                labels.append(artist.get_text())
                artist.remove()
        axes[5].set_xlim(0, 100)
        axes[5].set_yticks(positions, labels)
        axes[5].spines["left"].set_visible(False)
        axes[5].tick_params(axis="y", length=0, pad=8)
    elif number == 4:
        refine_figure_four(figure, axes)
    elif number == 5:
        # Child axes originally used inset margins inconsistent with the power
        # trace. Expand them to the exact same physical x endpoints.
        parent = axes[1]
        bounds = parent.get_position()
        for child, (y, h) in zip(parent.child_axes, [(0.58, 0.42), (0.02, 0.40)]):
            child.set_axes_locator(None)
            child.set_position(
                (bounds.x0, bounds.y0 + y * bounds.height, bounds.width, h * bounds.height)
            )
            child.set_ylabel("T (°C)" if y > 0.5 else "Gap (°C)", labelpad=9)
            for artist in list(child.texts):
                artist.remove()
        parent.child_axes[-1].set_xlabel("Hours after event start")
        axes[0].set_ylabel("HVAC power (kW)")
        axes[0].set_ylim(0, 160)
        legend = axes[0].get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            legend.set_bbox_to_anchor((0.5, 1.02))
            legend._loc = 9
        # Remove duplicated case statistics from the trajectory. The exact
        # response and recovery endpoints remain in the caption and data.
        for artist in list(axes[0].texts):
            artist.remove()
        axes[2].set_ylabel("Positive choices")
        axes[3].set_ylabel("False accepts")
        axes[2].set_xlabel("Demand spread λ")
        axes[3].set_xlabel("Demand spread λ")
        axes[4].set_ylabel("Response (kW)")
        axes[2].set_ylim(0, 220)
        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].get_legend().remove()
        axes[2].legend(
            handles,
            ["Full", "Mean"],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            handlelength=1,
            columnspacing=0.8,
        )
        for artist in list(axes[2].texts):
            if "5d-mean-" in str(artist.get_gid()):
                artist.remove()
        axes[3].set_ylim(0, 180)
        for artist in list(axes[4].texts):
            artist.remove()
        for artist in list(axes[5].texts):
            label = artist.get_text()
            if "− |" in label:
                artist.remove()
            elif label == "Response (kW)":
                artist.set_text("Resp. (kW)")
            elif label == "Rebound (kW)":
                artist.set_text("Reb. (kW)")
            elif label == "Extra (kWh)":
                artist.set_text("Extra kWh")
            elif label == "Net (kWh)":
                artist.set_text("Net kWh")
            elif label.startswith("±"):
                artist.set_x(0.44)
        refine_event_heatmaps(axes[6])
        color_axis = figure.add_axes([0.65, 0.026, 0.26, 0.008])
        image = axes[6].child_axes[0].images[0]
        bar = figure.colorbar(image, cax=color_axis, orientation="horizontal", ticks=[0, 60, 120])
        figure.text(0.13, 0.030, "Qualified selections", fontsize=11, ha="left", va="center")
        axes[6].set_title(
            "(g) Seasonal choices: July (left), February (right)",
            loc="left",
            fontsize=12,
            weight="bold",
            pad=9,
        ).set_gid("stage7-result-panel-title")
    elif number == 6:
        refine_figure_six(figure, axes)
    # Clear/rebuilt axes restore rc defaults, so apply spacing after all changes.
    for axis in figure.findobj(__import__("matplotlib").axes.Axes):
        axis.xaxis.labelpad = 6
        axis.yaxis.labelpad = 8
        axis.grid(False)
        axis.tick_params(pad=3)
    if number == 3:
        axes[5].tick_params(axis="y", length=0, pad=8)
    submission_layout(figure, axes, number)


def submission_layout(figure, axes, number):
    """Allocate page area to real data frames, including compound-panel axes."""
    from thermal_heterogeneity.figures import layout_snapshot

    before = layout_snapshot(figure)

    def place(axis, bounds):
        axis.set_axes_locator(None)
        axis.set_position(bounds)

    def panel_title(axis, y):
        title = axis.get_title(loc="left")
        relative_y = (y - axis.get_position().y0) / axis.get_position().height
        axis.set_title(
            title, loc="left", y=relative_y, pad=0, fontsize=12, fontweight="bold"
        ).set_gid("stage7-result-panel-title")

    if number == 1:
        for axis in axes[2:4]:
            place(axis, (axis.get_position().x0, 0.570, 0.34, 0.150))
    elif number == 2:
        handles, labels = axes[4].get_legend_handles_labels()
        axes[4].get_legend().remove()
        axes[4].legend(
            handles,
            labels,
            frameon=False,
            loc="upper right",
            ncol=1,
            handlelength=1,
            handletextpad=0.4,
            borderpad=0.2,
        )
    elif number == 3:
        for axis, bottom, heading, rows in [
            (axes[1], 0.730, 0.952, [0.930, 0.908]),
            (axes[2], 0.400, 0.622, [0.600, 0.578]),
        ]:
            lower, _ = axis.get_ylim()
            # Explicit ticks had also changed the lower bound after the old
            # annotation padding, so recover the data margin from dataLim.
            data_lower, data_upper = axis.dataLim.ymin, axis.dataLim.ymax
            axis.set_ylim(lower, data_upper + 0.05 * (data_upper - data_lower))
            if axis.dataLim.ymin < axis.get_ylim()[0] or axis.dataLim.ymax > axis.get_ylim()[1]:
                raise RuntimeError("Restored Fig. 3 display range clips data")
            place(axis, (axis.get_position().x0, bottom, 0.34, 0.155))
            panel_title(axis, bottom + 0.245)
            table = [artist for artist in axis.texts if "-n-table" in str(artist.get_gid())]
            for artist in table:
                text = artist.get_text()
                artist.set_transform(figure.transFigure)
                if text.startswith("matched n"):
                    artist.set_position((axis.get_position().x0 + 0.17, heading))
                else:
                    label = text.split()[0]
                    column = 0 if label in {"Mia", "PA"} else 1
                    row = 0 if label in {"Mia", "Del"} else 1
                    artist.set_position((axis.get_position().x0 + column * 0.17, rows[row]))
    elif number == 4:
        for index, x in [(1, 0.13), (2, 0.62), (3, 0.13), (4, 0.62)]:
            axis = axes[index]
            place(axis, (x, axis.get_position().y0, 0.36, axis.get_position().height))
        for parent, x in [(axes[5], 0.13), (axes[6], 0.62)]:
            place(parent, (x, 0.070, 0.36, 0.240))
            panel_title(parent, 0.335)
        carbon, carrier = axes[5].child_axes
        place(carbon, (0.13, 0.095, 0.17, 0.195))
        place(carrier, (0.36, 0.095, 0.13, 0.195))
        for artist in list(axes[5].texts):
            artist.remove()
        axes[5].text(0.215, 0.307, "Carbon", transform=figure.transFigure, ha="center", va="center")
        axes[5].text(
            0.425, 0.307, "Carriers", transform=figure.transFigure, ha="center", va="center"
        )
        for child, bounds in zip(
            axes[6].child_axes,
            [
                (0.62, 0.220, 0.15, 0.075),
                (0.83, 0.220, 0.15, 0.075),
                (0.62, 0.105, 0.15, 0.075),
                (0.83, 0.105, 0.15, 0.075),
            ],
        ):
            place(child, bounds)
        for axis in figure.axes:
            colorbar = getattr(axis, "_colorbar", None)
            if colorbar is not None and colorbar.mappable is carrier.images[0]:
                place(axis, (0.36, 0.035, 0.13, 0.007))
            elif colorbar is not None and colorbar.mappable is axes[6].child_axes[0].images[0]:
                place(axis, (0.64, 0.035, 0.32, 0.007))
    elif number == 5:
        parent = axes[5]
        for line in parent.lines:
            values = list(line.get_xdata())
            if values == [0.66, 0.66]:
                line.set_xdata([0.45, 0.45])
            elif values == [0.48, 0.82]:
                line.set_xdata([0.05, 0.85])
        for collection in parent.collections:
            points = collection.get_offsets().copy()
            points[:, 0] = 0.45 + 0.40 * (points[:, 0] - 0.66) / 0.14
            collection.set_offsets(points)
        for artist in parent.texts:
            if artist.get_text().startswith("±"):
                artist.set_x(0.98)
            else:
                artist.set_x(-0.07)
                artist.set_ha("right")
        for child, x in zip(axes[6].child_axes, [0.13, 0.64]):
            place(child, (x, 0.085, 0.34, 0.080))
        for artist in axes[6].texts:
            artist.set_transform(figure.transFigure)
            artist.set_position((0.555, 0.039))
        for artist in figure.texts:
            if artist.get_text() == "Qualified selections":
                artist.set_position((0.13, 0.016))
    elif number == 6:
        phase = axes[0]
        place(phase, (0.13, 0.730, 0.85, 0.230))
        for child, bounds in zip(
            phase.child_axes,
            [
                (0.13, 0.835, 0.36, 0.075),
                (0.62, 0.835, 0.36, 0.075),
                (0.13, 0.730, 0.36, 0.075),
                (0.62, 0.730, 0.36, 0.075),
            ],
        ):
            place(child, bounds)
            child.set_title(
                child.get_title(loc="left"),
                loc="left",
                y=1.08,
                pad=0,
                fontsize=11,
                fontweight="normal",
            )
        for artist in list(phase.texts):
            artist.remove()
        phase.text(
            0.555,
            0.695,
            "Assigned demand-centre SD (°C)",
            transform=figure.transFigure,
            ha="center",
            va="center",
        )
        phase.text(
            0.070,
            0.825,
            "Service-budget fraction",
            transform=figure.transFigure,
            ha="center",
            va="center",
            rotation=90,
        )
        handles, labels = phase.get_legend_handles_labels()
        old_legend = phase.get_legend()
        if old_legend is not None:
            handles = old_legend.legend_handles
            labels = [text.get_text() for text in old_legend.get_texts()]
            old_legend.remove()
        phase.legend(
            handles,
            labels,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.555, 0.965),
            bbox_transform=figure.transFigure,
            handlelength=0.8,
            handletextpad=0.2,
            columnspacing=0.6,
        )
        panel_title(phase, 0.980)

        for left, right, y, title_y in [
            (1, 2, 0.515, 0.665),
            (3, 4, 0.285, 0.435),
            (5, 6, 0.055, 0.205),
        ]:
            place(axes[left], (0.13, y, 0.36, 0.120))
            place(axes[right], (0.62, y, 0.36, 0.120))
            panel_title(axes[left], title_y)
            panel_title(axes[right], title_y)

        for parent, y, labels in [
            (axes[3], 0.285, ["Tail-score error\n(°C)", "Eligibility error\n(%)"]),
            (axes[5], 0.055, ["Ordinal RPS\n(unitless)", "Tail-share MAE\n(fraction)"]),
        ]:
            for artist in list(parent.texts):
                artist.remove()
            for child, x, label in zip(parent.child_axes, [0.13, 0.36], labels):
                place(child, (x, y, 0.13, 0.120))
                child.set_ylabel(label, labelpad=4)
                child.set_xlabel("")
                child.tick_params(pad=2)
            if parent is axes[5]:
                parent.text(
                    0.31,
                    0.015,
                    "Feedback observations",
                    transform=figure.transFigure,
                    ha="center",
                    va="center",
                )

        for axis, y, ncol in [(axes[4], 0.420, 3), (axes[5], 0.190, 2)]:
            legend = axis.get_legend()
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.get_texts()]
            legend.remove()
            axis.legend(
                handles,
                labels,
                frameon=False,
                loc="center",
                ncol=ncol,
                bbox_to_anchor=(axis.get_position().x0 + 0.18, y),
                bbox_transform=figure.transFigure,
                handlelength=1,
                handletextpad=0.3,
                columnspacing=0.7,
                borderpad=0,
            )

    after = layout_snapshot(figure)
    changed_domains = [
        {"axis": key, "field": field, "before": before[key][field], "after": after[key][field]}
        for key in before
        for field in ["xlim", "ylim", "xscale", "yscale"]
        if before[key][field] != after[key][field]
    ]
    permitted = number == 3 and all(
        item["axis"] in {"axis1", "axis2"} and item["field"] == "ylim" for item in changed_domains
    )
    if changed_domains and not permitted:
        raise RuntimeError("Submission layout changed a data-axis domain")
    figure._submission_layout_audit = {
        "before": before,
        "after": after,
        "data_axis_domains_unchanged": not changed_domains,
        "permitted_display_domain_changes": changed_domains,
        "display_change_reason": "Fig3 annual energy/carbon y padding for the former count table removed; all observations and intervals remain within limits"
        if changed_domains
        else None,
    }


def refine_figure_four(figure, axes):
    import numpy as np
    import pandas as pd
    from thermal_heterogeneity.figures_4_6 import _simple_density

    ax = axes[0]
    ax.clear()
    d = pd.read_csv(figure._stage7_panel_root / "Fig4b.csv")
    grid = np.linspace(d.perturbed_demand_c.min() - 0.5, d.perturbed_demand_c.max() + 0.5, 300)
    colors = ["#6F9EBC", "#ACCAD8", "#C8A36A", "#C78470"]
    for y, (lam, color) in enumerate(zip([0, 0.5, 1, 1.5], colors)):
        values = d.loc[d["lambda"] == lam, "perturbed_demand_c"].to_numpy()
        density = _simple_density(values, grid)
        ridge = density / density.max() * 0.68
        ax.fill_between(grid, y, y + ridge, color=color, alpha=0.22)
        ax.plot(grid, y + ridge, color=color, lw=1.2)
        q10, med, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        ax.plot([q10, q90], [y + 0.02, y + 0.02], color=color, lw=2.4)
    ax.set_yticks(np.arange(4) + 0.2, ["λ = 0", "λ = 0.5", "λ = 1", "λ = 1.5"])
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(-0.05, 3.85)
    ax.set_xlabel("Demand temperature (°C)")
    ax.set_ylabel("")
    ax.set_xticks([15, 20, 25, 30, 35])
    ax.set_title(
        "(a) Fixed-centre demand spread", loc="left", fontsize=12, weight="bold", pad=9
    ).set_gid("stage7-result-panel-title")
    axes[1].set_ylabel("Qualified selections")
    axes[1].set_xlabel("Demand spread λ")
    axes[2].set_xlabel("Demand spread λ")
    axes[2].set_ylabel("False accepts")
    axes[3].set_ylabel("Qualified actions")
    axes[3].set_xlabel("Demand spread λ")
    axes[4].set_xlabel("Energy change (MWh/year)")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].get_legend().remove()
    axes[1].legend(
        handles,
        ["Full", "Mean"],
        frameon=False,
        loc="lower left",
        ncol=2,
        handlelength=1,
        columnspacing=0.7,
    )
    axes[2].set_ylim(0, 185)
    axes[3].set_ylim(-2, 90)
    for text in list(axes[4].texts):
        text.remove()
    axes[4].set_xlim(-70, 70)
    for child in list(axes[6].child_axes)[:2]:
        child.tick_params(axis="x", labelbottom=False)
    for i, child in enumerate(axes[5].child_axes):
        if i == 0:
            child.set_xticks([-10, 10, 30])
        else:
            for label in child.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")
    for text in axes[3].texts:
        if text.get_text().startswith("Median"):
            text.set_position((0.03, 0.95))
            text.set_transform(axes[3].transAxes)
        elif text.get_text().startswith("Zero"):
            text.set_position((0.03, 0.78))
            text.set_transform(axes[3].transAxes)
    # Context names sit above each small heatmap, outside its data rectangle.
    for child in axes[6].child_axes:
        if len(child.get_yticks()) == 3:
            child.set_yticks([0, 2], ["0", "1.5"])
        for text in list(child.texts):
            if text.get_text() in {"Miami", "New Delhi", "Port Angeles", "Seattle"}:
                text.set_transform(child.transAxes)
                text.set_position((0.02, 1.07))
                text.set_va("bottom")
                text.set_bbox(None)
    parent = axes[6]
    pb = parent.get_position()
    # Four heatmaps have an independent title row and a common standard colourbar.
    for child, (x, y) in zip(
        parent.child_axes, [(0.08, 0.66), (0.58, 0.66), (0.08, 0.25), (0.58, 0.25)]
    ):
        child.set_position(
            (pb.x0 + x * pb.width, pb.y0 + y * pb.height, 0.38 * pb.width, 0.25 * pb.height)
        )
    for artist in list(parent.texts):
        artist.remove()
    for artist in list(parent.images):
        artist.remove()
    for artist in list(parent.patches):
        artist.remove()
    cax = figure.add_axes(
        (pb.x0 + 0.10 * pb.width, pb.y0 + 0.09 * pb.height, 0.80 * pb.width, 0.008)
    )
    image = parent.child_axes[0].images[0]
    limit = float(image.norm.vmax)
    figure.colorbar(
        image, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit], format="%.0f"
    )
    # The carrier scale is also explicit and signed, with units in the caption.
    parent = axes[5]
    carrier = parent.child_axes[1]
    cp = carrier.get_position()
    carrier.set_xlabel("")
    for artist in list(carrier.texts):
        artist.remove()
    cax = figure.add_axes((cp.x0, 0.043, cp.width, 0.007))
    image = carrier.images[0]
    limit = float(image.norm.vmax)
    figure.colorbar(
        image, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit], format="%.0f"
    )


def refine_control_outcomes(figure, ax):
    import pandas as pd
    from thermal_heterogeneity.main_figures_1_3 import OUTCOME_COLORS

    data = pd.read_csv(figure._stage7_panel_root / "Fig3g.csv")
    data.energy_direction = data.energy_direction.replace({"all_unmatched": "match_lost"})
    kinds = [
        "same_direction_decrease",
        "same_direction_increase",
        "all_tied",
        "direction_not_stable",
        "match_lost",
    ]
    labels = ["Decrease", "Increase", "Tie", "Unstable", "Unmatched"]
    cities = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
    ax.clear()
    for x, (control, city) in enumerate(
        (method, city) for method in ["personal_center", "ltpref_interval"] for city in cities
    ):
        q = data[
            (data.alternative_control == control) & (data.context == city)
        ].energy_direction.value_counts()
        for y, kind in enumerate(kinds):
            count = int(q.get(kind, 0))
            if count:
                ax.scatter(
                    x, y, s=18 + count * 23, color=OUTCOME_COLORS[kind], alpha=0.85, zorder=2
                )
                ax.text(x, y, str(count), ha="center", va="center")
            else:
                ax.scatter(x, y, s=4, color="#DDDDDD")
    ax.set_xlim(-0.6, 7.6)
    ax.set_ylim(4.6, -0.6)
    ax.set_yticks(range(5), labels)
    ax.set_xticks(
        range(8),
        ["PC\nMia", "PC\nDel", "PC\nPA", "PC\nSea", "LT\nMia", "LT\nDel", "LT\nPA", "LT\nSea"],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(
        "(e) Three-control outcomes", loc="left", fontsize=12, weight="bold", pad=9
    ).set_gid("stage7-result-panel-title")


def refine_event_heatmaps(parent):
    bounds = parent.get_position()
    for child, (x, width) in zip(parent.child_axes, [(0, 0.42), (0.58, 0.42)]):
        child.set_axes_locator(None)
        child.set_position(
            (
                bounds.x0 + x * bounds.width,
                bounds.y0 + 0.40 * bounds.height,
                width * bounds.width,
                0.52 * bounds.height,
            )
        )
        for text in list(child.texts):
            if text.get_text() in {"July", "February"}:
                text.remove()
        child.set_yticks([0, 3], ["0", "1.5"])
    for text in list(parent.texts):
        if text.get_text().startswith(("July: ref", "February: ref")):
            text.remove()
        elif text.get_text().startswith("Demand shift"):
            text.set_position((0.5, -0.05))
            text.set_text("Demand shift (°C), spread λ by row")


def refine_figure_six(figure, axes):
    import pandas as pd
    import numpy as np
    from thermal_heterogeneity.figures import CONTEXT_COLORS, REPRESENTATION_COLORS

    ax = axes[4]
    for child in list(ax.child_axes):
        child.remove()
    ax.clear()
    d = pd.read_csv(figure._stage7_panel_root / "Fig6f.csv")
    color = REPRESENTATION_COLORS
    for split, marker in [("leave_one_context_seed_out", "o"), ("leave_one_population_out", "s")]:
        q = d[d.split_scheme == split]
        for rep, col in color.items():
            z = q[q.representation == rep]
            ax.scatter(
                z.true_action_retention * 100,
                z.false_accept_candidates + 1,
                color=col,
                marker=marker,
                s=28,
                alpha=0.75,
            )
    highlight = d[
        (d.representation == "quantile5")
        & (d.margin_rule == "pooled_q95")
        & (d.split_scheme == "leave_one_context_seed_out")
    ].iloc[0]
    ax.scatter(
        [highlight.true_action_retention * 100],
        [highlight.false_accept_candidates + 1],
        s=100,
        facecolors="none",
        edgecolors="#7F7F7F",
        lw=1.1,
        zorder=4,
    )
    ax.set_yscale("log")
    ax.set_xlim(-5, 105)
    ax.set_ylim(0.65, 20000)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([1, 100, 10000], ["0", "99", "9,999"])
    ax.set_xlabel("Qualified actions retained (%)")
    ax.set_ylabel("False accepts + 1 (log)")
    ax.set_title(
        "(e) Held-out qualification", loc="left", fontsize=12, weight="bold", pad=9
    ).set_gid("stage7-result-panel-title")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=color[key], marker="o", linestyle="", label=label)
        for key, label in [("mean_only", "Mean"), ("quantile3", "q3"), ("quantile5", "q5")]
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.10, 1.0),
        handlelength=0.8,
        handletextpad=0.35,
        borderpad=0.1,
    )
    # Small learning axes use fewer tick labels, with the full sequence in data.
    for child in axes[5].child_axes:
        child.set_xticks([0, 20, 40])
        child.tick_params(pad=2)
    # Give the phase-map axes their physical definitions in the available gutter.
    axes[0].text(
        0.5,
        -0.08,
        "Assigned demand-centre SD (°C)",
        ha="center",
        va="top",
        transform=axes[0].transAxes,
    )
    axes[0].text(
        -0.005,
        0.42,
        "Service-budget fraction",
        ha="right",
        va="center",
        rotation=90,
        transform=axes[0].transAxes,
    )
    for artist in list(axes[2].texts):
        if artist.get_text().startswith("in-sample"):
            artist.remove()
    axes[2].legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=0.9,
        columnspacing=0.65,
        handletextpad=0.25,
    )
    axes[2].set_ylim(-0.6, 7.0)
    for line in list(axes[2].lines):
        if line.get_linestyle() == "--":
            line.remove()
    for artist in axes[3].texts:
        if artist.get_text() == "Tail MAE (°C)":
            artist.set_text("Tail error (°C)")
        elif artist.get_text() == "Eligibility error (%)":
            artist.set_text("Eligibility (%)")
    axes[3].child_axes[0].set_ylim(0, 6.5)
    axes[3].child_axes[1].set_ylim(0, 90)
    for child in axes[3].child_axes:
        pos = child.get_position()
        child.set_position((pos.x0, pos.y0, pos.width, pos.height * 0.88))
    legend = axes[5].get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.5, 1.08))
    for artist in list(axes[5].texts):
        if artist.get_text().startswith(("Plotted n40", "City Δ40")):
            artist.remove()
        elif artist.get_text() == "Normalized RPS":
            artist.set_text("Ordinal RPS\n(unitless)")
            artist.set_position((0.25, 0.69))
            artist.set_linespacing(0.9)
        elif artist.get_text() == "Tail-share MAE":
            artist.set_text("Tail-share MAE\n(fraction)")
            artist.set_position((0.79, 0.69))
            artist.set_linespacing(0.9)
    axes[5].text(
        0.5, 0.02, "Feedback observations", ha="center", va="center", transform=axes[5].transAxes
    )
    # Show the cost of protection among actual matched qualified decisions.
    ax = axes[6]
    for child in list(ax.child_axes):
        child.remove()
    ax.clear()
    q = pd.read_csv(figure._stage7_decision_value)
    matched = q[q.status.eq("matched_full_qualified")].copy()
    available = q.full_available.astype(str).str.lower().eq("true")
    if len(matched) != 542 or int(available.sum()) != 606:
        raise ValueError("Decision-value denominator changed from 542/606")
    cities = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
    names = ["Miami", "Delhi", "Port A.", "Seattle"]
    colors = [CONTEXT_COLORS[city] for city in cities]
    ax.axvline(0, color="#aaaaaa", lw=0.65, zorder=0)
    for y, (city, color) in enumerate(zip(cities, colors)):
        vals = np.sort(matched.loc[matched.city.eq(city), "energy_regret_percent"].to_numpy())
        jitter = np.tile(np.linspace(-0.16, 0.16, 11), int(np.ceil(len(vals) / 11)))[: len(vals)]
        ax.scatter(vals, y + jitter, s=9, color=color, alpha=0.55, edgecolors="none", zorder=2)
        lo, med, hi = np.quantile(vals, [0.1, 0.5, 0.9])
        ax.plot([lo, hi], [y, y], color="#7F7F7F", lw=1.1, zorder=3)
        ax.scatter(med, y, s=23, marker="D", color="#7F7F7F", zorder=4)
        total = int((available & q.city.eq(city)).sum())
        ax.text(46, y, f"{len(vals)}/{total}", ha="right", va="center")
    ax.text(46, -0.80, "n / available", ha="right", va="center")
    ax.set_xlim(-1, 48)
    ax.set_ylim(3.55, -1.15)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks(range(4), names)
    ax.set_xlabel("Extra HVAC energy (%)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=5)
    ax.set_title(
        "(g) Additional annual energy", loc="left", fontsize=12, weight="bold", pad=9
    ).set_gid("stage7-result-panel-title")


def refine_figure_one(figure, axes):
    """Rebuild congested annotations from the unchanged panel data."""
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from thermal_heterogeneity.figures import DEMAND_CMAP, VARIANCE_COLORS, draw_world_land

    panel = figure._stage7_panel_root
    # Common column endpoints. Maps span one column each without wide margins.
    bounds = [
        (0.13, 0.795, 0.34, 0.16),
        (0.64, 0.795, 0.34, 0.16),
        (0.13, 0.555, 0.34, 0.145),
        (0.64, 0.555, 0.34, 0.145),
        (0.13, 0.255, 0.34, 0.215),
        (0.64, 0.255, 0.34, 0.215),
        (0.13, 0.060, 0.34, 0.110),
    ]
    for ax, pos in zip(axes, bounds):
        ax.set_position(pos)
    for idx, name in enumerate(["Fig1b.csv", "Fig1c.csv"]):
        ax = axes[idx]
        ax.clear()
        d = pd.read_csv(panel / name)
        draw_world_land(ax)
        supported = d.supported.astype(str).str.lower().eq("true")
        q = d[supported]
        no = d[~supported]
        sizes = 12 + 65 * np.sqrt(q.analysis_units / q.analysis_units.max())
        ax.scatter(
            q.longitude,
            q.latitude,
            c=q.demand_center_c,
            cmap=DEMAND_CMAP,
            vmin=19.5,
            vmax=30.6,
            s=sizes,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        ax.scatter(
            no.longitude,
            no.latitude,
            s=15,
            facecolors="none",
            edgecolors="#777777",
            linewidths=0.7,
            zorder=2,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-58, 85)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_xticks([-120, 0, 120])
        ax.set_yticks([-30, 30, 60])
        ax.set_xlabel("Longitude (°)", labelpad=5)
        ax.tick_params(axis="x", bottom=True, labelbottom=True)
        ax.set_ylabel("Latitude (°)", labelpad=6)
        figure.text(
            0.13 if idx == 0 else 0.78,
            0.778,
            f"{len(q)}/40 supported",
            ha="left",
            va="center",
            fontsize=11,
        )
        title = ["Warm-season demand", "Cold-season demand"][idx]
        ax.set_title(
            f"({chr(97+idx)}) {title}", loc="left", fontsize=12, weight="bold", pad=9
        ).set_gid("stage7-result-panel-title")
    cax = figure.add_axes([0.385, 0.775, 0.28, 0.008])
    cb = figure.colorbar(
        ScalarMappable(norm=Normalize(19.5, 30.6), cmap=DEMAND_CMAP),
        cax=cax,
        orientation="horizontal",
        ticks=[20, 25, 30],
    )
    cb.set_label("")
    cax.xaxis.set_ticks_position("top")
    figure.text(0.69, 0.779, "°C", ha="left", va="center", fontsize=11)
    # Compact interval rows put group names outside data rather than over them.
    ax = axes[2]
    ax.clear()
    d = pd.read_csv(panel / "Fig1d.csv")
    for j, (level, season, label) in enumerate(
        [
            ("city", "warm", "City/W"),
            ("city", "cold", "City/C"),
            ("source", "warm", "Src/W"),
            ("source", "cold", "Src/C"),
        ]
    ):
        q = d[(d.aggregation_level == level) & (d.season == season)].sort_values("p50_c")
        y = j + np.linspace(-0.30, 0.30, len(q))
        ax.hlines(y, q.p10_c, q.p90_c, color="#ACCAD8", lw=0.65, alpha=0.75)
        ax.scatter(q.p50_c, y, s=7, color="#6F9EBC", zorder=3)
    ax.set_yticks(range(4), ["City/W", "City/C", "Src/W", "Src/C"])
    ax.invert_yaxis()
    ax.set_xlim(14, 33)
    ax.set_xticks([16, 24, 32])
    ax.set_xlabel("Demand temperature (°C)", labelpad=6)
    ax.set_title("(c) Demand within places", loc="left", fontsize=12, weight="bold", pad=9).set_gid(
        "stage7-result-panel-title"
    )
    # A lollipop allocation replaces overprinted numbers inside narrow stacks.
    ax = axes[3]
    ax.clear()
    d = pd.read_csv(panel / "Fig1e.csv").set_index("component")
    keys = ["country", "city", "release_building_group_id", "release_unit_id", "within_unit"]
    labels = ["Country", "City", "Building", "Within bldg", "Repeat"]
    v = d.loc[keys, "share"].to_numpy() * 100
    cols = VARIANCE_COLORS
    for y, val, col in zip(range(5), v, cols):
        ax.plot([0, val], [y, y], color=col, lw=2.4)
        ax.scatter(val, y, s=30, color=col, zorder=3)
        ax.text(72, y, f"{val:.1f}%", ha="right", va="center")
    ax.set_yticks(range(5), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Variance share (%)", labelpad=6)
    ax.set_title("(d) Variance allocation", loc="left", fontsize=12, weight="bold", pad=9).set_gid(
        "stage7-result-panel-title"
    )
    # Policy labels and numerical columns have reserved space on either side.
    ax = axes[4]
    ax.clear()
    d = pd.read_csv(panel / "Fig1f.csv")
    policies = list(dict.fromkeys(d.policy))
    compact = ["AUS", "BRA", "CAN", "IND", "IDN", "MYS", "SGP", "SWE", "USA"]
    colors = {"below": "#6F9EBC", "within": "#D2D2D2", "above": "#C78470"}
    for y, policy in enumerate(policies):
        q = d[d.policy == policy].set_index("position")
        left = 0
        for k in ["below", "within", "above"]:
            val = float(q.loc[k, "proportion"])
            ax.barh(y, val, left=left, height=0.58, color=colors[k])
            left += val
        row = q.loc["within"]
        center = float(q.loc["below", "proportion"]) + float(row.proportion) / 2
        ax.errorbar(
            center,
            y,
            xerr=(float(row.ci_high) - float(row.ci_low)) / 2,
            color="#333333",
            capsize=2,
            lw=0.8,
        )
        ax.text(1.20, y, f"{float(row.proportion)*100:.0f}%", ha="right", va="center")
    ax.set_yticks(range(9), compact)
    ax.set_ylim(8.6, -1.9)
    ax.set_xlim(0, 1.24)
    ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax.set_xlabel("Demand records (%)", labelpad=6)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=col, label=label)
            for label, col in [("Below", "#6F9EBC"), ("Within", "#D2D2D2"), ("Above", "#C78470")]
        ],
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.0),
        frameon=False,
        ncol=3,
        handlelength=0.7,
        handletextpad=0.3,
        columnspacing=0.65,
        borderpad=0,
    )
    ax.spines["left"].set_visible(False)
    ax.set_title(
        "(e) Policy-range coverage", loc="left", fontsize=12, weight="bold", pad=9
    ).set_gid("stage7-result-panel-title")
    ax = axes[5]
    ax.clear()
    d = pd.read_csv(panel / "Fig1g.csv")
    ax.axvline(0, color="#aaaaaa", lw=0.8)
    for y, row in enumerate(d.itertuples()):
        ax.hlines(y, row.q10_delta_c, row.q90_delta_c, color="#ACCAD8", lw=1.2)
        ax.hlines(y, row.q25_delta_c, row.q75_delta_c, color="#6F9EBC", lw=4)
        ax.scatter(row.median_delta_c, y, s=26, color="#C78470", zorder=3)
        ax.text(
            0.98,
            0.87 - y * 0.32,
            f"{row.median_delta_c:+.1f}°",
            ha="right",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_yticks(range(3), ["China", "Japan", "Thailand"])
    ax.invert_yaxis()
    ax.set_xlim(-5, 4.8)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_xlabel("Demand − target (°C)", labelpad=6)
    ax.set_title("(f) Campaign offsets", loc="left", fontsize=12, weight="bold", pad=9).set_gid(
        "stage7-result-panel-title"
    )
    # Sensitivity labels are outside the actual 0–75% plotting region.
    ax = axes[6]
    for text in list(ax.texts):
        text.remove()
    ax.set_xlim(0, 75)
    ax.set_yticks(range(5), ["Within bldg", "Country", "City", "Building", "Repeat"])
    ax.set_ylim(4.45, -0.45)
    ax.set_xlabel("Variance share (%)", labelpad=6)
    ax.set_title("(g) Sensitivity", loc="left", fontsize=12, weight="bold", pad=9).set_gid(
        "stage7-result-panel-title"
    )
    shared = pd.read_csv(panel / "Fig2h.csv").drop_duplicates("dataset").set_index("dataset")
    ax = figure.add_axes((0.64, 0.060, 0.34, 0.110))
    axes.append(ax)
    source_rows = []
    request_colors = ["#ACCAD8", "#C8A36A", "#C78470"]
    for y, (dataset, label) in enumerate([("NUS_HAAS", "NUS"), ("ENTH", "ENTH")]):
        row = shared.loc[dataset]
        count = int(row.eligible_windows)
        counts = [count - int(row.disagreement_count),
                  int(row.disagreement_count) - int(row.opposing_count),
                  int(row.opposing_count)]
        expected = {"NUS_HAAS": (63, [14, 43, 6]), "ENTH": (34, [12, 20, 2])}
        if (count, counts) != expected[dataset]:
            raise ValueError("Shared-room categories differ from the fixed evidence")
        left = 0
        for category, value, color in zip(["Same", "Mixed", "Opposed"], counts, request_colors):
            width = 100 * value / count
            ax.barh(y, width, left=left, height=.55, color=color, edgecolor="white", lw=.5)
            ax.text(left + width / 2, y, str(value), ha="center", va="center", fontsize=11)
            source_rows.append({"dataset": dataset, "eligible_windows": count,
                                "category": category, "window_count": value,
                                "share_percent": width})
            left += width
    ax.set_yticks([0, 1], ["NUS", "ENTH"])
    ax.set_ylim(2.1, -1.2)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Shared-room windows (%)", labelpad=6)
    ax.set_title("(h) Shared-room votes", loc="left", fontsize=12, weight="bold", pad=9).set_gid(
        "stage7-result-panel-title"
    )
    ax.legend(handles=[Patch(facecolor=color, label=label) for color, label in
                       zip(request_colors, ["Same", "Mixed", "Opposed"])],
              frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.48, 1.02),
              handlelength=.55, columnspacing=.55, handletextpad=.3, borderpad=0)
    figure._shared_room_source = source_rows
    figure._shared_room_input_sha256 = sha256(panel / "Fig2h.csv")
    for ax in figure.axes:
        ax.tick_params(pad=3)
        ax.xaxis.labelpad = max(ax.xaxis.labelpad, 5)
        ax.yaxis.labelpad = max(ax.yaxis.labelpad, 6)


def save_rgb_png(figure, path: Path, dpi: int, expected: tuple[int, int]) -> None:
    temporary = path.with_name(path.stem + ".tmp.png")
    original_size = tuple(figure.get_size_inches())
    try:
        figure.set_size_inches(expected[0] / dpi, expected[1] / dpi, forward=True)
        figure.savefig(temporary, format="png", dpi=dpi, facecolor="white", edgecolor="white")
        with Image.open(temporary) as image:
            rgb = image.convert("RGB")
            if rgb.size != expected:
                raise RuntimeError(f"{path.name}: pixel size {rgb.size}, expected {expected}")
            rgb.save(path, format="PNG", dpi=(dpi, dpi), optimize=False)
    finally:
        figure.set_size_inches(original_size, forward=True)
        temporary.unlink(missing_ok=True)


def stage7_text_audit(figure) -> dict:
    """Verify the revised 11-pt/12-pt title typography without an 8-panel assumption."""

    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.text import Text

    figure.canvas.draw()
    checked = 0
    title_count = 0
    errors: list[str] = []
    for artist in figure.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        checked += 1
        is_title = artist.get_gid() == "stage7-result-panel-title"
        expected_size = TITLE_FONT_PT if is_title else BASE_FONT_PT
        if abs(float(artist.get_fontsize()) - expected_size) > 1e-8:
            errors.append(f"size:{artist.get_text()!r}:{artist.get_fontsize()}!={expected_size}")
        weight = str(artist.get_fontweight()).lower()
        if is_title:
            title_count += 1
            if weight not in {"bold", "700"}:
                errors.append(f"title_not_bold:{artist.get_text()!r}:{weight}")
        elif weight not in {"normal", "400"}:
            errors.append(f"non_title_bold:{artist.get_text()!r}:{weight}")
        path = Path(findfont(artist.get_fontproperties(), fallback_to_default=False))
        actual = FontProperties(fname=str(path)).get_name()
        if actual != FONT_FAMILY:
            errors.append(f"font:{artist.get_text()!r}:{actual}")
    if title_count == 0:
        errors.append("no panel titles found")
    if errors:
        raise RuntimeError("Stage 7 text audit failed: " + "; ".join(errors[:12]))
    return {"status": "PASS", "visible_text_artists": checked, "panel_titles": title_count}


def export_figure(figure, number: int, directory: Path) -> dict:
    from matplotlib.text import Text
    from thermal_heterogeneity.figures import apply_data_axis_style

    map_axes = figure.axes[:2] if number == 1 else []
    apply_data_axis_style(figure, map_axes=map_axes)
    for artist in figure.findobj(Text):
        artist.set_color("black")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "submission_layout.json").write_text(
        json.dumps(figure._submission_layout_audit, indent=2), encoding="utf-8"
    )
    stem = f"figure_{number:02d}"
    if number == 1:
        with (directory / "figure_01_shared_room_source.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(figure._shared_room_source[0]))
            writer.writeheader()
            writer.writerows(figure._shared_room_source)
        (directory / "figure_01_shared_room_source_qa.json").write_text(
            json.dumps({"source": "Fig2h.csv", "sha256": figure._shared_room_input_sha256,
                        "NUS_windows": 63, "ENTH_windows": 34,
                        "categories_are_mutually_exclusive": True}, indent=2), encoding="utf-8")
    pdf = directory / f"{stem}.pdf"
    svg = directory / f"{stem}.svg"
    png_600 = directory / f"{stem}_600dpi.png"
    png_word = directory / f"{stem}_word.png"
    text_audit = stage7_text_audit(figure)
    from geometry_audit import audit_figure

    geometry = audit_figure(figure)
    (directory / f"{stem}_geometry.json").write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )
    figure.savefig(
        pdf,
        format="pdf",
        metadata={
            "Title": "Human thermal heterogeneity figure",
            "Author": "Study authors",
            "Subject": "Result-first Stage 7 figure",
            "Creator": "render_revised_main_figures.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        svg,
        format="svg",
        metadata={"Title": "Human thermal heterogeneity figure", "Date": None},
    )
    save_rgb_png(figure, png_600, 600, PNG_600)
    save_rgb_png(figure, png_word, 300, PNG_WORD)
    files = {}
    for path in (pdf, svg, png_600, png_word):
        files[path.name] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    svg_text = svg.read_text(encoding="utf-8")
    if FONT_FAMILY not in svg_text:
        raise RuntimeError(f"{stem}: SVG lacks live {FONT_FAMILY} text")
    return {
        "status": "PASS",
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "base_font_pt": BASE_FONT_PT,
        "panel_title_pt": TITLE_FONT_PT,
        "font_family": FONT_FAMILY,
        "panel_count": len(MAPPING[number]),
        "panel_ids": MAPPING[number],
        "text_contract": text_audit,
        "geometry_review": geometry,
        "files": files,
    }


def check_numbers(panel_root: Path, captions: dict[int, str]) -> dict:
    import pandas as pd

    warm = pd.read_csv(panel_root / "Fig1b.csv")
    cold = pd.read_csv(panel_root / "Fig1c.csv")
    warm_n = int(warm["supported"].astype(str).str.lower().eq("true").sum())
    cold_n = int(cold["supported"].astype(str).str.lower().eq("true").sum())
    annual = pd.read_csv(panel_root / "Fig4d.csv")
    annual_row = annual[annual["lambda"] == 1].iloc[0]
    event = pd.read_csv(panel_root / "Fig5e.csv")
    event_row = event[event["lambda"] == 1].iloc[0]
    gate = pd.read_csv(panel_root / "Fig6f.csv")
    gate_row = gate[
        gate["representation"].eq("quantile5")
        & gate["margin_rule"].eq("pooled_q95")
        & gate["split_scheme"].eq("leave_one_context_seed_out")
    ].iloc[0]
    observed = {
        "warm_supported": warm_n,
        "cold_supported": cold_n,
        "annual_false_accept": int(annual_row["false_accept_count"]),
        "annual_mean_positive": int(annual_row["mean_selected_count"]),
        "event_false_accept": int(event_row["false_accept_count"]),
        "event_mean_positive": int(event_row["mean_positive_count"]),
        "q5_true_positive": int(gate_row["true_positive_candidates"]),
        "q5_full_eligible": int(gate_row["full_eligible_candidates"]),
        "q5_covered_decisions": int(gate_row["full_available_and_certified_decisions"]),
        "q5_full_available_decisions": int(gate_row["full_available_decisions"]),
        "q5_false_accept": int(gate_row["false_accept_candidates"]),
    }
    expected = {
        "warm_supported": 31,
        "cold_supported": 26,
        "annual_false_accept": 117,
        "annual_mean_positive": 128,
        "event_false_accept": 78,
        "event_mean_positive": 122,
        "q5_true_positive": 14709,
        "q5_full_eligible": 18766,
        "q5_covered_decisions": 542,
        "q5_full_available_decisions": 606,
        "q5_false_accept": 0,
    }
    if observed != expected:
        raise RuntimeError(f"Frozen-number regression failed: {observed}")
    required_caption_strings = {
        1: ["31/40", "26/40"],
        4: ["117/128"],
        5: ["78/122"],
        6: ["14,709/18,766", "542/606"],
    }
    missing = {
        number: [token for token in tokens if token not in captions[number]]
        for number, tokens in required_caption_strings.items()
    }
    missing = {number: tokens for number, tokens in missing.items() if tokens}
    if missing:
        raise RuntimeError(f"Caption-number regression failed: {missing}")
    return {"status": "PASS", "observed": observed, "captions_checked": required_caption_strings}


def write_mapping(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["new_figure", "new_panel", "old_frozen_panel", "status"],
        )
        writer.writeheader()
        for number, old_panels in MAPPING.items():
            for index, old_panel in enumerate(old_panels):
                writer.writerow(
                    {
                        "new_figure": f"Fig{number}",
                        "new_panel": chr(ord("a") + index),
                        "old_frozen_panel": old_panel,
                        "status": "retained_and_relabelled",
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    project_root_default = Path(__file__).resolve().parents[3]
    parser.add_argument("--project-root", type=Path, default=project_root_default)
    parser.add_argument(
        "--release-root", type=Path, help="Standalone release root for public reproduction."
    )
    parser.add_argument(
        "--decision-value", type=Path, help="Frozen q5 annual decision-value plot table."
    )
    parser.add_argument(
        "--panel-data",
        type=Path,
        help="Panel-data directory; defaults to data/figure_panel_data under the project root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Stage-style output directory; defaults to results/stage7_service_first_revision_20261022.",
    )
    parser.add_argument(
        "--figures", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7))
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    stage_root = args.output_root.resolve() if args.output_root else project_root / STAGE_REL
    panel_root = args.panel_data.resolve() if args.panel_data else project_root / PANEL_REL
    release_root = args.release_root.resolve() if args.release_root else project_root / RELEASE_REL
    decision_value = (
        args.decision_value.resolve()
        if args.decision_value
        else project_root / STAGE_REL / "experiments/decision_value/MAIN_FIG6G_Q5_REGRET.csv"
    )
    sys.path.insert(0, str(release_root / "src"))

    from thermal_heterogeneity.figures import apply_figure_style
    from thermal_heterogeneity.figures_4_6 import build_figure_4, build_figure_5, build_figure_6
    from thermal_heterogeneity.main_figures_1_3 import (
        build_figure_1,
        build_figure_2,
        build_figure_3,
        close_figure,
    )

    apply_figure_style()
    figures_root = stage_root / "figures"
    captions_root = stage_root / "captions_alt"
    qa_root = stage_root / "qa"
    figures_root.mkdir(parents=True, exist_ok=True)
    captions_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)
    (captions_root / "CAPTION_DEFINITIONS_LONG.md").write_text(
        "\n\n".join(LONG_CAPTIONS.values()) + "\n", encoding="utf-8"
    )

    before = input_hashes(panel_root)
    decision_before = sha256(decision_value)
    write_mapping(stage_root / "panel_mapping.csv")
    builders = {
        1: build_figure_1,
        2: build_figure_2,
        3: build_figure_3,
        4: build_figure_4,
        5: build_figure_5,
        6: build_figure_6,
    }
    figure_audits = {}
    for number in args.figures:
        figure = builders[number](panel_root)
        try:
            figure._stage7_panel_root = panel_root
            figure._stage7_decision_value = decision_value
            retained = restructure(figure, number)
            if len(retained) != len(MAPPING[number]):
                raise RuntimeError(f"Figure {number}: panel mapping incomplete")
            output = figures_root / f"figure_{number:02d}"
            audit = export_figure(figure, number, output)
            (qa_root / f"figure_{number:02d}_qa.json").write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            text_dir = captions_root / f"figure_{number:02d}"
            text_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / "caption.txt").write_text(CAPTIONS[number] + "\n", encoding="utf-8")
            (text_dir / "alt_text.txt").write_text(ALT_TEXT[number] + "\n", encoding="utf-8")
            figure_audits[f"Figure {number}"] = audit
        finally:
            close_figure(figure)

    after = input_hashes(panel_root)
    if sha256(decision_value) != decision_before:
        raise RuntimeError("Decision-value input changed during rendering")
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [name for name in changed if before.get(name) != after.get(name)]
        raise RuntimeError(f"Frozen panel data changed: {changed}")
    with (qa_root / "input_sha256.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "sha256_before", "sha256_after", "unchanged"]
        )
        writer.writeheader()
        for relative, digest in before.items():
            writer.writerow(
                {
                    "relative_path": relative,
                    "sha256_before": digest,
                    "sha256_after": after[relative],
                    "unchanged": True,
                }
            )

    mapping_rows = sum(len(panels) for panels in MAPPING.values())
    if mapping_rows != 42 or EXCLUDED & {panel for panels in MAPPING.values() for panel in panels}:
        raise RuntimeError("Panel mapping or exclusion regression failed")
    number_audit = check_numbers(panel_root, CAPTIONS)
    overall = {
        "status": "PASS",
        "presentation_only": True,
        "panel_input_files": len(before),
        "panel_input_hashes_unchanged": True,
        "decision_value_input_sha256": decision_before,
        "decision_value_input_unchanged": True,
        "mapped_result_panels": mapping_rows,
        "excluded_concept_panels": sorted(EXCLUDED),
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "font_family": FONT_FAMILY,
        "base_font_pt": BASE_FONT_PT,
        "panel_title_pt": TITLE_FONT_PT,
        "frozen_number_regression": number_audit,
        "figures": figure_audits,
        "fig6f_metric_unit_correction": {
            "status": "PASS",
            "left_metric": "Ordinal RPS (unitless)",
            "right_metric": "Tail-share MAE (fraction)",
            "city_delta_40": "ordinal RPS −0.004; tail-share −0.017",
            "shared_legend_and_metric_headings_separated": True,
            "metric_headings_clear_of_data": True,
            "scientific_values_changed": False,
        },
    }
    (qa_root / "FIGURE_REVISION_QA.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md = [
        "# Stage 7 revised main-figure QA",
        "",
        "**Status: PASS**",
        "",
        f"- Frozen panel-data files hashed before and after: {len(before)}; all unchanged.",
        f"- Retained empirical/result panels mapped: {mapping_rows}.",
        f"- Removed conceptual panels: {', '.join(sorted(EXCLUDED))}.",
        f"- Canvas: {WIDTH_MM} × {HEIGHT_MM} mm; font: {FONT_FAMILY} {BASE_FONT_PT} pt; panel titles: {TITLE_FONT_PT} pt bold.",
        "- Every figure exported as PDF, live-text SVG, 600-dpi RGB PNG and 300-dpi Word RGB PNG.",
        "- Frozen-number checks passed: map supports 31/26; annual false acceptance 117/128; event false acceptance 78/122; q5+q95 retention 14,709/18,766 and coverage 542/606 with zero observed false acceptance.",
        "- Figure 1 maps occupy one aligned row. Figure 2 separates burden labels and ticks. Figure 3 uses an outcome-count matrix and preserves all signed results.",
        "- Figure 4 uses normalized density ridges and explicit signed colourbars. Figure 5 aligns power, temperature and service data axes and supplies the shared eligibility-count colourbar.",
        "- Figure 6g shows actual q5 protection energy overhead in 542 matched decisions among 606 full-information-resolvable decisions, using an independently hashed derived decision table. The mapping contains 42 result panels, including the added empirical shared-room panel in Figure 1h.",
        "- Figure 6f metric units were corrected to ordinal RPS (unitless) and tail-share MAE (fraction); its shared legend and metric headings occupy separate rows above the lowered child axes. The plotted values and intervals are unchanged.",
        "",
        "This QA confirms presentation integrity and number preservation; it does not redefine scientific denominators or validate new experiments.",
    ]
    (qa_root / "FIGURE_REVISION_QA.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    hash_roots = [figures_root, captions_root, stage_root / "scripts", qa_root]
    hash_candidates = sorted(
        {
            path
            for root in hash_roots
            for path in root.rglob("*")
            if path.is_file() and path.name != "output_sha256.csv"
        }
        | {stage_root / "panel_mapping.csv"}
    )
    with (qa_root / "output_sha256.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        for path in hash_candidates:
            writer.writerow(
                {
                    "relative_path": path.relative_to(stage_root).as_posix(),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    print(
        json.dumps(
            {"status": "PASS", "stage_root": str(stage_root), "figures": args.figures}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
