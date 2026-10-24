"""Render the four-context July source-day sensitivity comparison."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

from replay import digest, load, save

COLORS = {"uniform_average": "#7F7F7F", "personal_center": "#C8A36A", "ltpref_interval": "#7FA995"}
LABELS = {"uniform_average": "Uniform average", "personal_center": "Personal centre", "ltpref_interval": "LTPref interval"}
MARKERS = {"uniform_average": "o", "personal_center": "s", "ltpref_interval": "D"}
CONTEXTS = [("Miami", "Miami"), ("NewDelhi", "New Delhi"), ("PortAngeles_small2019", "Port Angeles"), ("Seattle_small2019", "Seattle")]
CLASSES = ["original_maximum", "median", "high90_nonmaximum"]
CLASS_LABELS = ["Peak", "Median", "High 90th"]

CAPTION = (
    "Supplementary Fig. 13 | July demand response across three source load classes. "
    "Panels show the Miami (a), New Delhi (b), Port Angeles (c) and Seattle (d) representative building contexts "
    "for composition 2026102001, random mixing and native-zone control. The peak date maximizes common-25 "
    "baseline weekday electrical HVAC use from 14:00 to 16:00; added dates are nearest the median and 90th "
    "percentile, excluding the peak. Dates use the source simulation calendar. Faint dashed curves show the "
    "physical response to a +1 °C shift. Solid curves show the largest positive service-qualified response "
    "among +0.5/+1 °C actions, with zero for a qualified baseline lacking a qualified action. Mean and fixed "
    "annual-tail limits apply separately to the event and next 48 hours. Each context has its own vertical scale."
)


def figure_source(replay):
    report = load(replay / "REPLAY_REPORT.json")
    names = ["EVENT_ALL_CANDIDATES.csv", "headline_replay/EVENT_ALL_CANDIDATES.csv", "THREE_DATE_COMPARISON.csv"]
    for name in names:
        if digest(replay / name) != report["outputs"][name]:
            raise ValueError(f"Replayed source changed: {name}")
    raw = pd.concat([pd.read_csv(replay / name) for name in names[:2]], ignore_index=True)
    raw = raw[raw.service_fraction.eq(1) & raw.amplitude_c.eq(1)]
    qualified = pd.read_csv(replay / names[2])
    qualified = qualified[qualified.service_fraction.eq(1)]
    if len(raw) != 36 or len(qualified) != 36 or not qualified.baseline_service_qualified.all():
        raise ValueError("Expected four contexts by three dates by three controls with qualified baselines")
    keys = ["context", "date_class", "control"]
    source = qualified.merge(raw[keys + ["event_average_reduction_kw", "run_id", "source_day_type"]],
                             on=keys, validate="one_to_one", suffixes=("_selected", "_physical_1c"))
    source = source.rename(columns={"event_average_reduction_kw_physical_1c": "physical_1c_response_kw", "run_id": "physical_1c_run_id"})
    source["composition_id"], source["layout"], source["permission"] = 2026102001, "random_mixing", "native_zone"
    source["physical_action_amplitude_c"] = 1.
    source["qualified_event_amplitudes_tested_c"] = "0.5|1.0"
    source["source_calendar_display_year_assumed"] = False
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    source = figure_source(args.replay_root.resolve())
    normal_path = font_manager.findfont(FontProperties(family="Times New Roman", style="normal", weight="normal"), fallback_to_default=False)
    bold_path = font_manager.findfont(FontProperties(family="Times New Roman", style="normal", weight="bold"), fallback_to_default=False)
    normal, bold = FontProperties(fname=normal_path, size=11), FontProperties(fname=bold_path, size=12)
    if normal.get_name() != "Times New Roman" or bold.get_name() != "Times New Roman":
        raise ValueError("Exact Times New Roman normal and bold fonts are required")
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "text.color": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False})
    fig = plt.figure(figsize=(180 / 25.4, 174 / 25.4), facecolor="white")
    axes, titles = [], []
    for index, (context, label) in enumerate(CONTEXTS):
        left, bottom = [.145, .635][index % 2], [.565, .18][index // 2]
        ax = fig.add_axes([left, bottom, .31, .25])
        axes.append(ax)
        rows = source[source.context.eq(context)]
        max_value = float(rows[["physical_1c_response_kw", "service_qualified_available_response_kw"]].to_numpy().max())
        ticks = MaxNLocator(nbins=4, min_n_ticks=4).tick_values(0, max_value * 1.12)
        upper = float(ticks[-1])
        ax.set_ylim(-upper * .065, upper)
        ax.set_yticks(ticks[ticks >= 0])
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        for offset, control in zip([-.14, 0., .14], COLORS):
            subset = rows[rows.control.eq(control)].set_index("date_class").loc[CLASSES]
            x = np.arange(3) + offset
            ax.plot(x, subset.physical_1c_response_kw, linestyle=(0, (3, 2)), linewidth=1.1,
                    color=COLORS[control], alpha=.46, marker=MARKERS[control], markersize=5.2,
                    markerfacecolor="white", markeredgewidth=.9, zorder=2)
            ax.plot(x, subset.service_qualified_available_response_kw, linewidth=1.15, color=COLORS[control],
                    marker=MARKERS[control], markersize=4.5, markeredgewidth=.5, markeredgecolor="white", zorder=3)
        days = rows.drop_duplicates("date_class").set_index("date_class").loc[CLASSES].event_day.astype(int)
        ax.set_xticks(np.arange(3), [f"{label}\n{day} Jul" for label, day in zip(CLASS_LABELS, days)])
        ax.set_xlim(-.37, 2.37)
        ax.set_ylabel("2-h electrical reduction (kW)", fontproperties=normal, labelpad=5)
        ax.axhline(0, color="#B0B0B0", linewidth=.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(.75)
        ax.tick_params(axis="both", direction="out", length=3, width=.7, pad=3)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontproperties(normal)
        title = fig.text(left, bottom + .276, f"{chr(97 + index)}  {label}", ha="left", va="bottom", fontproperties=bold)
        titles.append(title)
    control_handles = [Line2D([], [], color=COLORS[control], marker=MARKERS[control], linewidth=1.1, markersize=4.6,
                              label=LABELS[control]) for control in COLORS]
    control_legend = fig.legend(handles=control_handles, loc="upper center", bbox_to_anchor=(.545, .985),
                               ncol=3, frameon=False, prop=normal, handlelength=1.4, columnspacing=1.25, handletextpad=.45)
    meaning_handles = [Line2D([], [], color="#777777", linestyle=(0, (3, 2)), alpha=.46, marker="o", markerfacecolor="white", label="+1 °C physical"),
                       Line2D([], [], color="#777777", marker="o", label="Service-qualified availability")]
    meaning_legend = fig.legend(handles=meaning_handles, loc="upper center", bbox_to_anchor=(.545, .935), ncol=2,
                               frameon=False, prop=normal, handlelength=1.9, columnspacing=1.4, handletextpad=.5)
    footer = fig.text(.545, .071, "Source-day class and date", ha="center", va="center", fontproperties=normal)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = [text for text in fig.findobj(Text) if text.get_visible() and text.get_text()]
    outside, intersections = [], []
    supported_glyphs = set(font_manager.get_font(normal_path).get_charmap())
    for text in text_artists:
        if any(not char.isspace() and ord(char) not in supported_glyphs for char in text.get_text()):
            raise ValueError("Exact Times New Roman lacks a required text glyph")
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
            outside.append(text.get_text())
        if text.get_color() != "black" or text.get_fontproperties().get_name() != "Times New Roman":
            raise ValueError("Figure text style differs from the locked font/colour contract")
        if abs(text.get_fontsize() - (12 if text in titles else 11)) > 1e-9:
            raise ValueError("Figure text size differs from the locked 11/12-point contract")
    for first, second in combinations(text_artists, 2):
        a, b = first.get_window_extent(renderer), second.get_window_extent(renderer)
        if a.overlaps(b):
            intersection_width, intersection_height = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if intersection_width > 1.2 and intersection_height > 1.2:
                intersections.append({"first": first.get_text(), "second": second.get_text()})
    if outside or intersections:
        raise ValueError(f"Figure text geometry failed: outside={outside}, intersections={intersections}")
    out.mkdir(parents=True, exist_ok=True)
    stem = "TYPICAL_DAY_DR"
    source.to_csv(out / f"{stem}_source.csv", index=False)
    (out / f"{stem}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    for extension in ["pdf", "svg"]:
        fig.savefig(out / f"{stem}.{extension}", facecolor="white")
    fig.savefig(out / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(out / f"{stem}_word.png", dpi=300, facecolor="white")
    geometry = {"status": "PASS", "canvas_mm": [180, 174], "panels": 4, "source_rows": len(source),
                "panel_axes_bounds": [list(ax.get_position().bounds) for ax in axes], "text_outside_canvas": outside,
                "text_intersections": intersections, "font_family": "Times New Roman", "normal_pt": 11, "title_pt": 12,
                "all_text_black": True, "normal_font_sha256": digest(normal_path), "bold_font_sha256": digest(bold_path),
                "visual_inspection_required": True, "physical_and_qualified_outcomes_distinguished": True,
                "contexts_have_separate_y_scales": True, "regional_totals_or_stock_aggregation": False,
                "source_replay_report_sha256": digest(args.replay_root.resolve() / "REPLAY_REPORT.json"),
                "renderer_sha256": digest(__file__), "outputs": {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    save(out / f"{stem}_geometry_QA.json", geometry)
    plt.close(fig)
    print(json.dumps({k: v for k, v in geometry.items() if k not in ["outputs", "panel_axes_bounds"]}, indent=2))


if __name__ == "__main__":
    main()
