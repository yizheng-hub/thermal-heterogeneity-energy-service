"""Shared Stage 8 result-figure layout; numerical renderers follow the locked tables.

This module does not create sample data or export an empty conceptual figure.
The command-line style check resolves fonts and reports layout geometry only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


FONT = "Times New Roman"
WIDTH_MM, HEIGHT_MM = 180, 183
CONTROL_ORDER = ("uniform_average", "personal_center", "ltpref_interval")
CONTROL_LABELS = {
    "uniform_average": "Uniform average",
    "personal_center": "Personal centre",
    "ltpref_interval": "LTPref interval",
}
CONTROL_COLORS = {
    "uniform_average": "#7F7F7F",
    "personal_center": "#C8A36A",
    "ltpref_interval": "#7FA995",
}
CONTEXT_ORDER = ("Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019")
CONTEXT_COLORS = dict(zip(CONTEXT_ORDER, ("#C78470", "#AF94AD", "#6F9EBC", "#7FA995")))
CONTEXT_LABELS = dict(zip(CONTEXT_ORDER, ("Miami", "New Delhi", "Port Angeles", "Seattle")))
CONTEXT_SHORT = ("MIA", "DEL", "PTA", "SEA")
CONTROL_MARKERS = dict(zip(CONTROL_ORDER, ("o", "s", "^")))
ALTERNATIVES = CONTROL_ORDER[1:]

# Millimetres on the actual page, rather than parent boxes with unequal children.
DATA_BOUNDS_MM = tuple(
    (x, y, 67.0, 35.0) for y in (125.0, 72.0, 19.0) for x in (19.0, 109.0)
)


def configure_style() -> dict[str, str]:
    fonts = {}
    for weight in ("normal", "bold"):
        prop = FontProperties(family=FONT, weight=weight)
        path = Path(findfont(prop, fallback_to_default=False))
        if FontProperties(fname=str(path)).get_name() != FONT:
            raise ValueError(f"Exact {FONT} {weight} is required")
        fonts[weight] = str(path)
    plt.rcParams.update({
        "font.family": FONT, "font.size": 11, "text.color": "black",
        "axes.labelsize": 11, "axes.labelcolor": "black", "axes.labelpad": 6,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#777777", "axes.linewidth": 0.75,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.major.pad": 4, "ytick.major.pad": 4,
        "xtick.color": "black", "ytick.color": "black",
        "legend.fontsize": 11, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "thermal-heterogeneity-stage8",
    })
    return fonts


def six_panel_canvas(titles: list[str], *, control_key: bool = True, bounds=None, height=None):
    """Create equal data frames with separate title, axis and shared-key bands."""
    if len(titles) != 6:
        raise ValueError("The main result template has exactly six data panels")
    configure_style()
    page_height = height or HEIGHT_MM
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, page_height / 25.4))
    axes = []
    for index, (title, panel_bounds) in enumerate(zip(titles, bounds or DATA_BOUNDS_MM)):
        x, y, width, height_mm = panel_bounds
        ax = fig.add_axes((x / WIDTH_MM, y / page_height, width / WIDTH_MM, height_mm / page_height))
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", pad=8)
        ax.grid(False)
        ax.set_axisbelow(True)
        axes.append(ax)
    if control_key:
        handles = [Line2D([], [], marker=marker, linestyle="none", color=CONTROL_COLORS[key],
                          markersize=5, label=CONTROL_LABELS[key])
                   for key, marker in zip(CONTROL_ORDER, ("o", "s", "^"))]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                   ncol=3, handletextpad=0.3, columnspacing=0.9)
    return fig, axes


def layout_audit(fig, axes) -> dict:
    """Record geometric candidates. Actual image inspection is a separate gate."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, texts = [], []
    hidden_ticks = set()
    for ax in axes:
        for ticker, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in ticker.get_major_ticks() + ticker.get_minor_ticks():
                if tick.get_loc() < low - 1e-9 or tick.get_loc() > high + 1e-9:
                    hidden_ticks.update((id(tick.label1), id(tick.label2)))
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip() or id(text) in hidden_ticks:
            continue
        box = text.get_window_extent(renderer)
        item = {"text": text.get_text(), "bounds_px": [round(v, 2) for v in box.bounds]}
        texts.append((text, box, item))
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            outside.append(item)
    overlaps = []
    for index, (left, a, a_item) in enumerate(texts):
        for right, b, b_item in texts[index + 1:]:
            if left is right:
                continue
            dx, dy = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx > 1 and dy > 1:
                overlaps.append({"left": a_item, "right": b_item})
    return {"status": "REQUIRES_ACTUAL_IMAGE_REVIEW",
            "data_frames": [{"panel": chr(97 + i), "bounds": list(ax.get_position().bounds),
                             "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim())}
                            for i, ax in enumerate(axes)],
            "text_outside_canvas": outside, "text_overlap_candidates": overlaps}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_result(fig, axes, directory: Path, stem: str, *, caption: str,
                  sources: dict[str, Path]) -> dict:
    """Export real results only, with exact input hashes and no inferred visual pass."""
    if not sources or not caption.strip() or not axes or not all(ax.has_data() for ax in axes):
        raise ValueError("Every panel needs real data, a caption and traced source files before export")
    before = {name: file_sha256(path) for name, path in sources.items()}
    audit = layout_audit(fig, axes)
    typography = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        is_panel_title = any(text is ax.title or text is ax._left_title for ax in axes)
        expected_size = 12 if is_panel_title else 11
        expected_weight = "bold" if is_panel_title else "normal"
        prop = text.get_fontproperties()
        font_path = Path(findfont(prop, fallback_to_default=False))
        if (FontProperties(fname=str(font_path)).get_name() != FONT
                or abs(text.get_fontsize() - expected_size) > 1e-9
                or prop.get_weight() != expected_weight or text.get_color() != "black"):
            raise ValueError(f"Unexpected typography: {text.get_text()!r}")
        typography.append({"text": text.get_text(), "font_file": str(font_path),
                           "size_pt": expected_size, "weight": expected_weight, "colour": "black"})
    if audit["text_outside_canvas"]:
        raise ValueError("Figure has text outside its canvas; adjust layout before exporting")
    directory.mkdir(parents=True, exist_ok=True)
    pdf, svg = directory / f"{stem}.pdf", directory / f"{stem}.svg"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(svg, metadata={"Date": None})
    outputs = [pdf, svg]
    for suffix, dpi in (("", 600), ("_word", 300)):
        size = tuple(round(value * dpi) for value in fig.get_size_inches())
        png = directory / f"{stem}{suffix}.png"
        fig.savefig(png, dpi=dpi)
        with Image.open(png) as image:
            rgb = image.convert("RGB")
            if rgb.size != size:
                rgb = rgb.resize(size, Image.Resampling.LANCZOS)
            rgb.save(png)
        outputs.append(png)
    (directory / f"{stem}_caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")
    if before != {name: file_sha256(path) for name, path in sources.items()}:
        raise ValueError("Scientific input changed during figure rendering")
    result = {"figure": stem, "sources": before, "outputs": {p.name: file_sha256(p) for p in outputs},
              "layout_audit": audit, "typography_audit": {"status": "PASS", "text": typography},
              "caption_word_count": len(caption.split()), "actual_visual_review": "PENDING"}
    (directory / f"{stem}_qa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def point_summary(ax, frame, metric, ylabel, *, scale=1.0, controls=CONTROL_ORDER,
                  signed=False, percentage=False):
    """Show every finite scenario plus medians/IQRs, never a selected favourable subset."""
    records = []
    offsets = np.linspace(-0.23, 0.23, len(controls))
    for context_index, context in enumerate(CONTEXT_ORDER):
        cases = frame[frame.context.eq(context)]
        if all(column in cases for column in ("composition_id", "layout")):
            for _, case in cases.groupby(["composition_id", "layout"], sort=True):
                x_line, y_line = [], []
                for control, offset in zip(controls, offsets):
                    part = case[case.control.eq(control)]
                    if len(part) == 1 and pd.notna(part.iloc[0][metric]):
                        x_line.append(context_index + offset)
                        y_line.append(float(part.iloc[0][metric]) * scale)
                if len(y_line) > 1:
                    ax.plot(x_line, y_line, color="#BDB9B2", linewidth=.6, alpha=.65, zorder=1)
        for control, offset in zip(controls, offsets):
            selected = frame[frame.context.eq(context) & frame.control.eq(control)].copy()
            values = pd.to_numeric(selected[metric], errors="raise").to_numpy(dtype=float) * scale
            finite = values[np.isfinite(values)]
            if finite.size:
                x = context_index + offset
                jitter = np.linspace(-0.045, 0.045, finite.size) if finite.size > 1 else np.zeros(1)
                ax.scatter(x + jitter, finite, s=12, color=CONTROL_COLORS[control], alpha=0.78,
                           marker=CONTROL_MARKERS[control], linewidths=.2, edgecolors="white", zorder=2)
                low, median, high = np.quantile(finite, [0.25, 0.5, 0.75])
                ax.errorbar(x, median, yerr=[[median - low], [high - median]],
                            fmt=CONTROL_MARKERS[control], color=CONTROL_COLORS[control],
                            markersize=4.5, linewidth=1.5, capsize=2.3, zorder=3)
            else:
                low = median = high = np.nan
            records.append({"context": context, "control": control, "metric": metric,
                            "declared_cases": len(selected), "finite_cases": len(finite),
                            "q25_display": low, "median_display": median, "q75_display": high})
    if signed:
        ax.axhline(0, color="#A5A5A5", linewidth=0.65, zorder=0)
    if percentage:
        finite_values = pd.to_numeric(frame[metric], errors="raise").to_numpy(float) * scale
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values):
            span = max(np.ptp(finite_values), 8)
            ax.set_ylim(max(-2, float(finite_values.min()) - span * .12),
                        min(102, float(finite_values.max()) + span * .12))
    ax.set_xticks(range(4), CONTEXT_SHORT)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    return records


def annotated_heatmap(ax, values, xlabels, ylabel, *, signed, decimals=1, counts=None):
    """An exact-value matrix uses no extra inset that would shrink the data frame."""
    values = np.asarray(values, dtype=float)
    maximum = max(float(np.nanmax(np.abs(values))), 1e-9) if np.isfinite(values).any() else 1.0
    if signed:
        cmap = LinearSegmentedColormap.from_list("stage8_signed", ["#6F9EBC", "#F7F6F3", "#C78470"])
        limits = (-maximum, maximum)
    else:
        cmap = LinearSegmentedColormap.from_list("stage8_positive", ["#F7F6F3", "#7FA995"])
        limits = (0.0, max(100.0, maximum))
    cmap.set_bad("#EEEEEE")
    ax.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest",
              cmap=cmap, vmin=limits[0], vmax=limits[1])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            rounded = round(value, decimals) if np.isfinite(value) else value
            label = f"{0.0 if rounded == 0 else rounded:.{decimals}f}" if np.isfinite(value) else "NA"
            if counts is not None:
                label += f"\n({int(counts[row][col])}/3)"
            ax.text(col, row, label, ha="center", va="center", color="black", fontsize=11)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(4), CONTEXT_SHORT)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    return {"values": values.tolist(), "colour_limits": list(limits)}


def read_annual_tables(root: Path) -> dict[str, pd.DataFrame]:
    names = ("ANNUAL_DIRECT", "ANNUAL_SERVICE_SELECTIONS", "CONTROL_COMPARISONS",
             "SPATIAL_PAIRS", "LAYOUT_PAIRS", "STATIC_ZONE_INTERVAL_COMPATIBILITY")
    tables = {name: pd.read_csv(root / f"{name}.csv", keep_default_na=False,
                                na_values=[""]) for name in names}
    direct = tables["ANNUAL_DIRECT"]
    if len(direct) != 324 or set(direct.control) != set(CONTROL_ORDER):
        raise ValueError("Expected the complete 324-row three-control direct design")
    if len(tables["ANNUAL_SERVICE_SELECTIONS"]) != 1296:
        raise ValueError("Expected every controller, space and common-service fraction")
    return tables


def write_panel_summaries(directory, number, records):
    pd.DataFrame(records).to_csv(directory / f"Figure_{number:02d}_numerical_summary.csv", index=False)


def empirical_steps(ax, frame, metric, xlabel, scale=1.0):
    """Empirical case distributions, with no fitted density or synthetic observations."""
    for control in CONTROL_ORDER:
        values = np.sort(pd.to_numeric(frame.loc[frame.control.eq(control), metric]).dropna().to_numpy() * scale)
        if len(values):
            cumulative = np.arange(1, len(values) + 1) / len(values) * 100
            ax.step(values, cumulative, where="post", color=CONTROL_COLORS[control], linewidth=1.6)
            ax.scatter(values, cumulative, color=CONTROL_COLORS[control],
                       marker=CONTROL_MARKERS[control], s=12, linewidths=0, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Admitted cases at or below (%)")
    ax.set_ylim(-3, 103)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_locator(MaxNLocator(4))


def placement_forest(ax, frame, metric, xlabel, *, scale=1.0):
    """All paired placements, separated by context and placement, without a heat table."""
    for context_index, context in enumerate(CONTEXT_ORDER):
        for layout_index, layout in enumerate(("demand_ordered", "source_structured")):
            center = context_index * 2.5 + layout_index
            for offset, control in zip((-.23, 0, .23), CONTROL_ORDER):
                selected = frame[frame.context.eq(context) & frame.layout.eq(layout) & frame.control.eq(control)]
                values = pd.to_numeric(selected[metric]).dropna().to_numpy() * scale
                if len(values):
                    ax.scatter(values, center + offset + np.linspace(-.045, .045, len(values)),
                               s=15, marker=CONTROL_MARKERS[control], color=CONTROL_COLORS[control],
                               alpha=.85, linewidths=.2, edgecolors="white", zorder=3)
                    if len(values) > 1:
                        ax.plot([min(values), max(values)], [center + offset] * 2,
                                color=CONTROL_COLORS[control], linewidth=1.1, zorder=2)
    ax.axvline(0, color="#A5A5A5", linewidth=.75, zorder=0)
    ax.set_yticks([i * 2.5 + .5 for i in range(4)], CONTEXT_SHORT)
    ax.set_ylim(9.1, -.7)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_ylabel("Context (ordered / source)")


def render_figure_two(tables, annual_root, output):
    direct = tables["ANNUAL_DIRECT"].query("permission == 'native_zone'").copy()
    if len(direct) != 108:
        raise ValueError("The native-zone direct comparison must contain 108 logical rows")
    direct["strict_profile_coverage_pct"] = (
        direct.profiles_with_no_occupied_hour_interval_violation / direct.assigned_profile_count * 100
    )
    direct.to_csv(output / "Figure_02_source_data.csv", index=False)
    specs = [
        ("Mean reference mismatch", "mean_reference_mismatch_c", "Mean deviation (°C)", 1, False),
        ("Fixed-tail mismatch", "fixed_tail_reference_mismatch_c", "Tail deviation (°C)", 1, False),
        ("Interval coverage", "modeled_interval_person_hour_coverage", "Covered weighted hours (%)", 100, True),
        ("Least-served profiles", "profile_coverage_p10", "Profile P10 coverage (%)", 100, True),
        ("Interval excess", "modeled_interval_mean_excess_c", "Mean excess (°C)", 1, False),
        ("No interval violations", "strict_profile_coverage_pct", "Profiles without violations (%)", 1, True),
    ]
    fig, axes = six_panel_canvas([s[0] for s in specs], bounds=[
        (19, 125, 67, 35), (109, 125, 67, 35),
        (19, 72, 86, 35), (128, 72, 48, 35),
        (19, 19, 67, 35), (109, 19, 67, 35)])
    keys = ["context", "composition_id", "layout"]
    admitted = direct[direct.selection_status.eq("selected")]
    complete_keys = admitted.groupby(keys).control.nunique()
    complete_keys = complete_keys[complete_keys.eq(3)].reset_index()[keys]
    complete = admitted.merge(complete_keys, on=keys, how="inner", validate="many_to_one")
    if len(complete) != 48 or complete.groupby("control").size().nunique() != 1:
        raise ValueError("Expected the frozen 16 complete three-control direct comparisons")
    complete.to_csv(output / "Figure_02_matched_three_source_data.csv", index=False)
    summaries = []
    for letter, ax, (_, metric, label, scale, percentage) in zip("abcdef", axes, specs):
        if letter in "cd":
            empirical_steps(ax, complete, metric, label, scale)
            for control, group in complete.groupby("control"):
                summaries.append(dict(panel=letter, control=control, metric=metric,
                                      finite_cases=len(group), comparison="complete_three_control_cases"))
        else:
            summaries += [dict(panel=letter, **r) for r in point_summary(
                ax, direct, metric, label, scale=scale, percentage=percentage)]
    axes[2].set_ylabel("Cases at or below (%)")
    axes[3].set_ylabel("Cases at or below (%)")
    axes[2].set_xlabel("Covered weighted hours (%)", labelpad=3)
    axes[3].set_xlabel("Profile P10 coverage (%)", labelpad=3)
    counts = []
    for context in CONTEXT_ORDER:
        counts.append("/".join(str(int(direct[direct.context.eq(context) & direct.control.eq(c)]
                                      .selection_status.eq("selected").sum())) for c in CONTROL_ORDER))
    caption = ("Figure 2 | Thermal service under three direct control rules. All panels use native-zone "
               "permissions and predeclared zero-slack commands. Each context has nine composition–placement "
               "cases per controller. Admitted trajectories passed equipment, ventilation, numerical and "
               "energy-accounting checks. In a,b,e,f, small points denote admitted cases, connected grey lines "
               "link the same composition–placement case, and larger symbols and whiskers show medians "
               "and interquartile ranges. a, Native-person-hour-weighted mean reference mismatch. "
               "b, Equal mean of occupied-time mismatches for the same 27 reference-tail profiles. "
               "c, Empirical distribution of occupied weighted hours inside model-derived intervals. d, Empirical "
               "distribution of the tenth percentile of individual profile coverage. The steps in c,d use the same "
               "16 complete three-control conditions (16 cases per controller), without fitting a density. "
               "Other panels retain all admitted cases, including conditions missing another controller. "
               "e, Occupied-hour-normalized interval excess. "
               "f, Profiles with no occupied interval violation. Admitted counts in uniform/personal/interval "
               f"order are {', '.join(f'{name} {n}' for name,n in zip(CONTEXT_SHORT,counts))}. "
               "MIA, DEL, PTA and SEA identify Miami, New Delhi, Port Angeles and Seattle building–weather "
               "contexts. Service references are reconstructed centres and model-derived intervals.")
    result = export_result(fig, axes, output, "Figure_02", caption=caption,
                           sources={"ANNUAL_DIRECT.csv": annual_root / "ANNUAL_DIRECT.csv"})
    write_panel_summaries(output, 2, summaries)
    plt.close(fig)
    return result


def render_figure_three(tables, annual_root, output):
    comparisons = tables["CONTROL_COMPARISONS"]
    native = comparisons.query("permission == 'native_zone'")
    direct = native.query("comparison_mode == 'direct'")
    equal = native.query("comparison_mode == 'equal_service' and service_fraction == 1.0")
    pd.concat([direct, equal]).to_csv(output / "Figure_03_paired_source_data.csv", index=False)
    direct.to_csv(output / "Figure_03_panel_c_resource_reallocation_source.csv", index=False)
    fig, axes = six_panel_canvas(["Direct energy change", "Direct carbon change", "Added and released conditioning",
                                 "Common-service\nenergy", "Common-service\ncarbon", "Qualified service\nchoices"],
                                height=190, bounds=[(19, 133, 67, 34), (109, 133, 67, 34),
                                                    (19, 76, 157, 33), (19, 19, 43, 30),
                                                    (77, 19, 43, 30), (135, 19, 41, 30)])
    summaries = []
    for letter, ax, frame, metric, label, scale in (
        ("a", axes[0], direct, "delta_site_pct", "Site energy change (%)", 1),
        ("b", axes[1], direct, "delta_annual_operational_co2_kg", "CO₂ change (t/year)", 0.001),
        ("d", axes[3], equal, "delta_site_pct", "Energy change (%)", 1),
        ("e", axes[4], equal, "delta_annual_operational_co2_kg", "CO₂ change (t/year)", 0.001),
    ):
        summaries += [dict(panel=letter, **r) for r in point_summary(
            ax, frame, metric, label, scale=scale, controls=ALTERNATIVES, signed=True)]
    for context_index, context in enumerate(CONTEXT_ORDER):
        for offset, control in zip((-.19, .19), ALTERNATIVES):
            group = direct[direct.context.eq(context) & direct.control.eq(control)]
            totals = [0.0, 0.0]
            for metric in ("hourly_added_electric_kwh", "hourly_added_gas_hhv_kwh",
                           "hourly_released_electric_kwh", "hourly_released_gas_hhv_kwh"):
                values = pd.to_numeric(group[metric]).dropna().to_numpy() / 1000
                value = float(np.median(values)) if len(values) else np.nan
                summaries.append({"panel": "c", "context": context, "control": control,
                                  "metric": metric, "finite_cases": len(values), "median_display": value})
                if np.isfinite(value):
                    side = 0 if "added" in metric else 1
                    sign = 1 if side == 0 else -1
                    axes[2].bar(context_index + offset, sign * value, width=.31, bottom=totals[side],
                                color=CONTROL_COLORS[control], alpha=1 if side == 0 else .85,
                                hatch=None if "electric" in metric else "///",
                                linewidth=.65, edgecolor="white")
                    totals[side] += sign * value
            if len(group):
                net = pd.to_numeric(group.delta_annual_site_kwh).dropna().to_numpy() / 1000
                axes[2].scatter(context_index + offset + np.linspace(-.065, .065, len(net)), net, s=17,
                                color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                                edgecolors="#666666", linewidths=.45, zorder=4)
    axes[2].set_xticks(range(4), CONTEXT_SHORT)
    axes[2].set_ylabel("Added / released (MWh/year)")
    axes[2].axhline(0, color="#A5A5A5", linewidth=.75)
    axes[2].yaxis.set_major_locator(MaxNLocator(4))
    axes[2].text(.98, .97, "Above: added / below: released\nSolid: electricity / hatched: gas HHV", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=11, color="black")
    selections = tables["ANNUAL_SERVICE_SELECTIONS"].query("permission == 'native_zone'")
    for control in CONTROL_ORDER:
        counts, fractions = [], sorted(selections.service_fraction.unique())
        for fraction in fractions:
            group = selections[selections.control.eq(control) & selections.service_fraction.eq(fraction)]
            count = int(group.selection_status.eq("selected").sum())
            counts.append(count)
            summaries.append({"panel": "f", "control": control, "service_fraction": fraction,
                              "declared_cases": len(group), "selected_cases": count})
        axes[5].plot(fractions, counts, marker=CONTROL_MARKERS[control], markersize=4,
                     color=CONTROL_COLORS[control], linewidth=1.1)
    axes[5].set_xticks([.5, .75, 1], ["0.5", "0.75", "1"])
    axes[5].set_xlabel("Service budget")
    axes[5].set_ylabel("Qualified cases")
    axes[5].set_ylim(-1, 37)
    axes[5].yaxis.set_major_locator(MaxNLocator(4, integer=True))
    caption = ("Figure 3 | Energy and operational carbon consequences of three controls. All panels use "
               "native-zone permissions. a,b, Paired direct-operation changes relative to uniform average "
               "control. Small points are admitted composition–placement comparisons; symbols and whiskers "
               "show medians and interquartile ranges. c, Building-hour increases (above zero) and decreases "
               "(below zero) in electricity (solid) and heating gas (hatched), summed over the year and then "
               "summarized by the composition–placement median. Overlaid symbols give individual paired net "
               "site-energy changes. These quantities describe temporal building-load changes, not metered "
               "transfers between people. Gas uses the higher-heating-value basis. PC and LT denote personal-centre and "
               "LTPref controls. d,e, Paired minimum-site-energy selections at the same population and fixed-tail "
               "reference limits, using budget fraction 1 and equal four-variant candidate support. f, Available "
               "selections among all 36 declared context–composition–placement cases at each service budget. "
               "Carbon uses fixed regional electricity and gas accounting factors. MIA, DEL, PTA and SEA "
               "denote Miami, New Delhi, Port Angeles and Seattle.")
    result = export_result(fig, axes, output, "Figure_03", caption=caption, sources={
        "CONTROL_COMPARISONS.csv": annual_root / "CONTROL_COMPARISONS.csv",
        "ANNUAL_SERVICE_SELECTIONS.csv": annual_root / "ANNUAL_SERVICE_SELECTIONS.csv"})
    write_panel_summaries(output, 3, summaries)
    plt.close(fig)
    return result


def render_figure_four(tables, annual_root, output):
    spatial = tables["SPATIAL_PAIRS"].query("alternative == 'native_zone'")
    direct = spatial.query("comparison_mode == 'direct'")
    equal = spatial.query("comparison_mode == 'equal_service' and service_fraction == 1.0")
    fig, axes = six_panel_canvas(["Permissions and service", "Permissions and energy", "Common-service energy",
                                 "Placement and service", "Placement and energy", "Static interval overlap"],
                                height=192, bounds=[(19, 135, 67, 34), (19, 78, 67, 34),
                                                    (19, 19, 67, 34), (109, 121, 67, 48),
                                                    (109, 57, 67, 40), (109, 14, 67, 22)])
    summaries = []
    for letter, ax, frame, metric, label in (
        ("a", axes[0], direct, "delta_interval_coverage_percentage_points", "Coverage change (p.p.)"),
        ("b", axes[1], direct, "delta_site_pct", "Site energy change (%)"),
        ("c", axes[2], equal, "delta_site_pct", "Site energy change (%)"),
    ):
        summaries += [dict(panel=letter, **r) for r in point_summary(ax, frame, metric, label, signed=True)]
    common_layout_path = annual_root.parent / "common_layout/COMMON_LAYOUT_PAIRS.csv"
    common_layout = pd.read_csv(common_layout_path)
    layouts = common_layout.query("permission == 'native_zone' and service_fraction == 1 "
                                  "and weight_view == 'native_occupancy'").copy()
    tables["LAYOUT_PAIRS"].to_csv(output / "Figure_04_direct_and_relative_anchor_layout_source.csv", index=False)
    layouts.to_csv(output / "Figure_04_common_absolute_anchor_layout_source.csv", index=False)
    xlabels = [f"{control}\n{layout}" for layout in ("Ord", "Src") for control in ("UA", "PC", "LT")]
    for letter, ax, metric, ylabel in (
        ("d", axes[3], "delta_modeled_interval_coverage", "Coverage change (p.p.)"),
        ("e", axes[4], "delta_site_pct", "Energy change (%)"),
    ):
        matrix, cell_counts = [], []
        for context in CONTEXT_ORDER:
            row, count_row = [], []
            for layout in ("demand_ordered", "source_structured"):
                for control in CONTROL_ORDER:
                    group = layouts[layouts.context.eq(context) & layouts.layout.eq(layout)
                                    & layouts.control.eq(control)]
                    values = pd.to_numeric(group[metric]).dropna().to_numpy()
                    if letter == "d":
                        values = values * 100
                    value = float(np.median(values)) if len(values) else np.nan
                    row.append(value)
                    count_row.append(len(values))
                    summaries.append({"panel": letter, "context": context, "control": control,
                                      "layout": layout, "metric": metric, "finite_cases": len(values),
                                      "median_display": value})
            matrix.append(row)
            cell_counts.append(count_row)
        placement_forest(ax, layouts, metric, ylabel, scale=100 if letter == "d" else 1)
    static = tables["STATIC_ZONE_INTERVAL_COMPATIBILITY"]
    keys = ["context", "composition_id", "layout"]
    aggregate = static.groupby(keys, sort=True)[["assigned_profile_count", "max_simultaneously_covered_profiles"]].sum()
    aggregate["maximum_covered_pct"] = (100 * aggregate.max_simultaneously_covered_profiles / aggregate.assigned_profile_count)
    aggregate = aggregate.reset_index()
    matrix = []
    for context_index, context in enumerate(CONTEXT_ORDER):
        row = []
        medians = []
        for layout_index, layout in enumerate(("random_mixing", "demand_ordered", "source_structured")):
            values = aggregate[aggregate.context.eq(context) & aggregate.layout.eq(layout)].maximum_covered_pct
            value = float(values.median())
            medians.append(value)
            row.append(value)
            axes[5].scatter(values, np.repeat(context_index + (layout_index - 1) * .22, len(values)),
                            s=18, color=CONTEXT_COLORS[context], marker=("o", "s", "D")[layout_index],
                            facecolors="none" if layout_index == 0 else CONTEXT_COLORS[context],
                            linewidths=.9, zorder=4)
            summaries.append({"panel": "f", "context": context, "layout": layout,
                              "metric": "static_maximum_covered_pct", "finite_cases": len(values),
                              "median_display": value})
        axes[5].plot(medians, [context_index + (j - 1) * .22 for j in range(3)],
                     color=CONTEXT_COLORS[context], linewidth=1.2, zorder=2)
        matrix.append(row)
    axes[5].set_yticks(range(4), CONTEXT_SHORT)
    axes[5].set_ylim(3.5, -.6)
    axes[5].set_xlim(67, 101)
    axes[5].set_xlabel("Maximum static coverage (%)")
    axes[5].set_xticks([70, 80, 90, 100])
    caption = ("Figure 4 | Placement and existing control permissions. a,b, Native-zone minus building-bound "
               "direct-operation changes in virtual-interval coverage and purchased site energy. c, The same "
               "permission contrast for minimum-energy selections under common service limits at budget "
               "fraction 1. Points show admitted paired cases, with median symbols and interquartile whiskers. "
               "d,e, All matched ordered (upper) and source-structured (lower) placement changes relative to random "
               "mixing at native-zone permissions and budget 1, using the same random-placement absolute "
               "limits and tail identities. The two rows in each context show ordered and source placements; "
               "each symbol denotes a composition, and horizontal segments show the observed ranges. "
               "UA/PC/LT denote uniform-average/personal-centre/LTPref control. f, Static interval coverage "
               "with an independent "
               "temperature per existing zone. Symbols show all three compositions, connected through their "
               "placement medians. Open circles, squares and diamonds identify random, ordered and source "
               "placement. Missing marks in d,e denote no matched selection. Static coverage is a demand-overlap "
               "bound, not simulated annual coverage. MIA/DEL/PTA/SEA denote Miami/New Delhi/Port Angeles/Seattle.")
    sources = {name + ".csv": annual_root / (name + ".csv") for name in
               ("SPATIAL_PAIRS", "LAYOUT_PAIRS", "STATIC_ZONE_INTERVAL_COMPATIBILITY")}
    sources["COMMON_LAYOUT_PAIRS.csv"] = common_layout_path
    result = export_result(fig, axes, output, "Figure_04", caption=caption, sources=sources)
    write_panel_summaries(output, 4, summaries)
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-style", action="store_true")
    parser.add_argument("--annual-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.check_style:
        print(json.dumps({"status": "LAYOUT_INTERFACE_READY_NO_RESULTS_RENDERED",
                          "fonts": configure_style(), "page_mm": [WIDTH_MM, HEIGHT_MM],
                          "data_bounds_mm": DATA_BOUNDS_MM, "control_colors": CONTROL_COLORS}, indent=2))
        return
    if args.annual_root is None or args.output is None:
        parser.error("Use --check-style or provide both --annual-root and --output")
    tables = read_annual_tables(args.annual_root)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [renderer(tables, args.annual_root, args.output)
                for renderer in (render_figure_two, render_figure_three, render_figure_four)]
    (args.output / "ANNUAL_FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(manifest), "output": str(args.output),
                      "actual_visual_review": "PENDING"}, indent=2))


if __name__ == "__main__":
    main()
