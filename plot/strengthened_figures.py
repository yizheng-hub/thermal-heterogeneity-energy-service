"""Draw strengthened publication figures from completed contained analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl  # noqa: E402 - initialize local imports before loading project modules

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.collections import PolyCollection  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.collections import PathCollection  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.lines import Line2D  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.patches import Patch  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.text import Text  # noqa: E402 - initialize local imports before loading project modules
import numpy as np  # noqa: E402 - initialize local imports before loading project modules
import pandas as pd  # noqa: E402 - initialize local imports before loading project modules

from thermal_heterogeneity.figures import (  # noqa: E402 - initialize local imports before loading project modules
    audit_data_safe_legends,
    load_world_land,
    require_times_new_roman,
)

FONT = 9
TITLE = 10
WIDTH = 8306 / 1440
BLUE, PURPLE, GREEN, CORAL, GOLD = "#6B9CBF", "#AA8BAE", "#70A598", "#CC8176", "#C9AE70"


def style():
    mpl.rcdefaults()
    fonts = require_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": FONT,
            "axes.titlesize": TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT,
            "axes.labelpad": 4,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "xtick.major.pad": 3,
            "ytick.major.pad": 3,
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "legend.fontsize": FONT,
            "legend.title_fontsize": FONT,
            "legend.frameon": False,
            "legend.labelspacing": 0.45,
            "legend.facecolor": "none",
            "legend.edgecolor": "none",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "HEM-strengthened-publication",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return fonts


def title(ax, letter, text):
    label = ax.set_title(
        f"({letter}) {text}", loc="left", pad=6, fontsize=TITLE, fontweight="bold", color="black"
    )
    label.set_gid("thermal-heterogeneity-panel-title")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.6)


def equal_earth(longitude, latitude):
    """Spherical Equal Earth equations; doi:10.1080/13658816.2018.1504949.

    Coefficients and forward equations checked against PROJ's eqearth implementation:
    https://proj.org/en/stable/operations/projections/eqearth.html
    """
    longitude, latitude = np.broadcast_arrays(np.asarray(longitude), np.asarray(latitude))
    theta = np.arcsin(np.sqrt(3) / 2 * np.sin(np.deg2rad(latitude)))
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    x = (
        np.deg2rad(longitude)
        * np.cos(theta)
        / (np.sqrt(3) / 2 * (a1 + 3 * a2 * theta**2 + 7 * a3 * theta**6 + 9 * a4 * theta**8))
    )
    y = theta * (a1 + a2 * theta**2 + a3 * theta**6 + a4 * theta**8)
    return x, y


def world_base(ax):
    ax.add_collection(
        PolyCollection(
            [
                np.column_stack(equal_earth(np.asarray(ring)[:, 0], np.asarray(ring)[:, 1]))
                for ring in load_world_land()
            ],
            facecolors="#F0F1F0",
            edgecolors="#707070",
            linewidths=0.35,
            zorder=0,
        )
    )
    for longitude in (-120, -60, 0, 60, 120):
        ax.plot(
            *equal_earth(np.full(181, longitude), np.linspace(-90, 90, 181)),
            color="#D7D7D7",
            lw=0.35,
            ls=":",
            zorder=-1,
        )
    for latitude in (-60, -30, 0, 30, 60):
        ax.plot(
            *equal_earth(np.linspace(-180, 180, 361), np.full(361, latitude)),
            color="#D7D7D7",
            lw=0.35,
            ls=":",
            zorder=-1,
        )
    latitudes = np.linspace(-90, 90, 181)
    west = np.column_stack(equal_earth(np.full(181, -180), latitudes))
    east = np.column_stack(equal_earth(np.full(181, 180), latitudes[::-1]))
    boundary = np.vstack([west, east, west[:1]])
    ax.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=0.7, zorder=2)
    ax.set_xlim(boundary[:, 0].min(), boundary[:, 0].max())
    ax.set_ylim(boundary[:, 1].min(), boundary[:, 1].max())
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.set_anchor("N")


def figure_one(raw, analysis):
    sources = raw / "results/observation_figures"
    paths = [
        sources / name
        for name in (
            "CITY_MAP_POINTS.csv",
            "COUNTRY_DEMAND_SUMMARY.csv",
            "LEAVE_ONE_SOURCE_VARIANCE.csv",
        )
    ] + [
        raw / "results/observations/ASHRAE_NESTED_VARIANCE.csv",
        raw / "results/shared_room/SHARED_ROOM_SUMMARY.csv",
        ROOT / "src/analysis/global_extension/GLOBAL_CLIMATES.json",
        ROOT / "src/thermal_heterogeneity/assets/ne_110m_observed_countries.json",
        ROOT / "src/thermal_heterogeneity/assets/ne_110m_land.geojson",
    ]
    points, country_demand, loo, variance, rooms = [pd.read_csv(p) for p in paths[:5]]
    climates = pd.DataFrame(json.loads(paths[5].read_text())["climates"])
    countries = json.loads(paths[6].read_text())
    assert points.record_count.sum() == 33845 and len(points) == 49
    assert len(country_demand) == 15 and country_demand.record_count.sum() == 33845
    assert len(climates) == 9
    fig = plt.figure(figsize=(WIDTH, 7.65))
    # These bounds reserve physical space for labels rather than squeezing data.
    a = fig.add_axes([0.14, 0.67, 0.74, 0.28])
    b = fig.add_axes([0.14, 0.355, 0.74, 0.28])
    c = fig.add_axes([0.14, 0.075, 0.31, 0.24])
    d = fig.add_axes([0.61, 0.075, 0.27, 0.24])

    def projected(ring):
        ring = np.asarray(ring)
        return np.column_stack(equal_earth(ring[:, 0], ring[:, 1]))

    world_base(a)
    world_base(b)
    a.add_collection(
        PolyCollection(
            [projected(ring) for feature in countries["features"] for ring in feature["rings"]],
            facecolors="#E2DCE8",
            edgecolors="#A39AA6",
            linewidths=0.3,
            zorder=1,
        )
    )
    cmap = LinearSegmentedColormap.from_list(
        "reference_temperature", ["#497BB8", "#82B4C0", "#F2E7C9", "#E5B185", "#C76E7A"]
    )
    norm = Normalize(points.demand_center_c.min(), points.demand_center_c.max())
    scale = 140 / points.record_count.max()
    cloud = a.scatter(
        *equal_earth(points.longitude, points.latitude),
        s=scale * points.record_count,
        c=points.demand_center_c,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.95,
        zorder=3,
    )
    a.scatter(
        *equal_earth(climates.longitude, climates.latitude),
        marker="*",
        s=48,
        facecolor=GOLD,
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    labels = {
        "Miami": (-60, 28),
        "Montreal": (-126, 56),
        "Malmo": (5, 72),
        "Tunis": (-25, 19),
        "Delhi": (50, 10),
        "Beijing": (140, 56),
        "Singapore": (145, 0),
        "Sydney": (151, -53),
        "Brasilia": (-53, -43),
    }
    for row in climates.itertuples():
        a.annotate(
            "Malmö" if row.climate == "Malmo" else row.climate,
            equal_earth(row.longitude, row.latitude),
            xytext=equal_earth(*labels[row.climate]),
            ha="center",
            va="center",
            fontsize=FONT,
            color="black",
            arrowprops={"arrowstyle": "-", "color": "#707070", "lw": 0.5},
            zorder=6,
        )
    title(a, "a", "Global observations and simulation climates")
    handles = [
        a.scatter([], [], s=scale * value, facecolor="#AEAEAE", edgecolor="white", linewidth=0.35)
        for value in (100, 500, 2000)
    ]
    handles += [
        Patch(facecolor="#E2DCE8", edgecolor="#A39AA6", linewidth=0.3),
        Line2D(
            [],
            [],
            ls="",
            marker="*",
            markersize=np.sqrt(48),
            markerfacecolor=GOLD,
            markeredgecolor="black",
            markeredgewidth=0.5,
        ),
    ]
    a.legend(
        handles,
        ["100 records", "500 records", "2,000 records", "Countries", "Climates"],
        loc="lower left",
        bbox_to_anchor=(0.09, 0.17),
        borderaxespad=0,
        handletextpad=0.65,
        labelspacing=0.55,
        frameon=False,
        facecolor="none",
        edgecolor="none",
    )
    fig.canvas.draw()
    box = a.get_position()
    cb = fig.add_axes([box.x1 + 0.022, box.y0, 0.018, box.height])
    ticks = np.arange(2 * np.ceil(norm.vmin / 2), 2 * np.floor(norm.vmax / 2) + 0.01, 2)
    colorbar = fig.colorbar(cloud, cax=cb, ticks=ticks)
    colorbar.set_label("Mean demand center (°C)", labelpad=5)
    cb.yaxis.set_ticks_position("right")
    cb.yaxis.set_label_position("right")
    colorbar.outline.set_linewidth(0.5)
    country_demand["span_c"] = (
        country_demand.demand_center_p90_c - country_demand.demand_center_p10_c
    )
    spread = country_demand.set_index("country").span_c
    spread_cmap = LinearSegmentedColormap.from_list(
        "within_country_variation", ["#F4EFF4", "#CEB9D1", "#8C6BA6"]
    )
    spread_norm = Normalize(0, 8)
    rings, colors = [], []
    for feature in countries["features"]:
        for ring in feature["rings"]:
            rings.append(projected(ring))
            colors.append(spread_cmap(spread_norm(spread.loc[feature["country_key"]])))
    b.add_collection(
        PolyCollection(rings, facecolors=colors, edgecolors="#707070", linewidths=0.3, zorder=1)
    )
    singapore = points[points.country.eq("singapore")]
    b.scatter(
        *equal_earth(singapore.longitude, singapore.latitude),
        s=15,
        c=[spread.loc["singapore"]],
        cmap=spread_cmap,
        norm=spread_norm,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    title(b, "b", "Demand varies within countries")
    fig.canvas.draw()
    box_b = b.get_position()
    cb_b = fig.add_axes([box_b.x1 + 0.022, box_b.y0, 0.018, box_b.height])
    spread_scale = mpl.cm.ScalarMappable(norm=spread_norm, cmap=spread_cmap)
    spread_bar = fig.colorbar(spread_scale, cax=cb_b, ticks=[0, 4, 8])
    spread_bar.set_label("10th–90th percentile span (°C)", labelpad=5)
    spread_bar.outline.set_linewidth(0.5)
    variance = variance[
        variance.weighting.eq("unit_weight") & variance.griffiths_coefficient.eq(0.5)
    ]
    order = ["country", "city", "building_id", "unit_group", "within_unit"]
    names = ["Country", "City", "Building", "Within\nbuilding", "Repeated\nrecords"]
    shares = variance.set_index("component").loc[order, "share"].to_numpy() * 100
    colors = [BLUE, "#99B9CA", GREEN, PURPLE, "#CBCBCB"]
    c.barh(range(5), shares, color=colors, height=0.65)
    sensitivity = (
        loo[loo.weighting.eq("unit_weight") & loo.griffiths_coefficient.eq(0.5)]
        .groupby("component")
        .share.agg(["min", "max"])
    )
    limits = sensitivity.loc[order].to_numpy() * 100
    c.errorbar(
        shares,
        range(5),
        xerr=np.vstack([shares - limits[:, 0], limits[:, 1] - shares]),
        fmt="none",
        ecolor="black",
        elinewidth=0.65,
        capsize=2,
        capthick=0.65,
    )
    c.set_yticks(range(5), names)
    c.invert_yaxis()
    c.set_xlim(0, 71)
    c.set_xticks([0, 30, 60])
    c.set_xlabel("Variance share (%)")
    for y, value in enumerate(shares):
        c.text(limits[y, 1] + 1.2, y, f"{value:.1f}", va="center", fontsize=FONT)
    title(c, "c", "Sources of variation")
    rooms = rooms[rooms.minutes.eq(30)].set_index("dataset").loc[["NUS", "ENTH"]]
    start = np.zeros(2)
    for key, color in (("Same", GREEN), ("Mixed", PURPLE), ("Opposed", CORAL)):
        width = rooms[key].to_numpy() / rooms.windows.to_numpy() * 100
        d.barh(range(2), width, left=start, color=color, height=0.55)
        start += width
    d.set_yticks(range(2), [f"{key}\nn = {int(row.windows)}" for key, row in rooms.iterrows()])
    d.set_ylim(1.6, -1.6)
    d.set_xlim(0, 100)
    d.set_xticks([0, 50, 100])
    d.set_xlabel("Room–time\nwindows (%)")
    title(d, "d", "Shared-room requests")
    d.legend(
        [Patch(facecolor=color) for color in (GREEN, PURPLE, CORAL)],
        ["Aligned requests", "Mixed requests", "Opposite requests"],
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.99),
        borderaxespad=0,
        handlelength=1,
        labelspacing=0.35,
        frameon=False,
        facecolor="none",
        edgecolor="none",
    )
    return fig, paths, "abcd"


def figure_two(raw, analysis):
    paths = [
        analysis / "results/nominal_information/NOMINAL_QUALIFICATION_COUNTS.csv",
        analysis
        / "results/information_uncertainty_summary/NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv",
        ROOT / "src/analysis/global_extension/GLOBAL_CLIMATES.json",
    ]
    counts, choices = [pd.read_csv(p) for p in paths[:2]]
    climates = [row["climate"] for row in json.loads(paths[2].read_text())["climates"]]
    compositions = [
        f"{archive} {letter}" for archive in ("Multinational", "China") for letter in "ABC"
    ]
    joined = counts.merge(
        choices, on=["climate", "composition"], validate="one_to_one", suffixes=("", "_paired")
    )
    if len(joined) != 54 or not joined.full_choice.eq(joined.full_choice_paired).all():
        raise ValueError("Nominal qualification and selected-operation tables differ")
    matrix = joined.pivot(
        index="climate", columns="composition", values="zone_mean_selected_full_service_excess_c"
    ).loc[climates, compositions]
    if matrix.isna().any().any():
        raise ValueError("Incomplete climate-composition service matrix")
    fig = plt.figure(figsize=(WIDTH, 5.7))
    a = fig.add_axes([0.14, 0.59, 0.74, 0.32])
    b = fig.add_axes([0.13, 0.17, 0.205, 0.25])
    c = fig.add_axes([0.495, 0.17, 0.205, 0.25])
    d = fig.add_axes([0.835, 0.17, 0.135, 0.25])
    cmap = LinearSegmentedColormap.from_list("service_excess", ["#F7F5F1", "#D5B4C8", "#856EAB"])
    im = a.imshow(
        matrix.to_numpy(),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=max(1.0, float(matrix.to_numpy().max())),
    )
    a.set_xticks(range(6), range(1, 7))
    a.set_yticks(range(9), ["Malmö" if climate == "Malmo" else climate for climate in climates])
    a.set_xlabel("Demand composition")
    a.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    a.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    a.grid(which="minor", color="white", linewidth=0.7)
    a.tick_params(which="minor", length=0)
    title(a, "a", "Individual service under mean-based choices")
    cb = fig.add_axes([0.904, 0.59, 0.018, 0.32])
    colorbar = fig.colorbar(im, cax=cb, ticks=[0, 0.5, 1.0])
    colorbar.set_label("Service excess (°C)", labelpad=5)
    colorbar.outline.set_linewidth(0.5)
    changed = joined.choices_differ.eq(True)
    for mask, color in ((~changed, BLUE), (changed, PURPLE)):
        group = joined[mask]
        b.scatter(
            100 * group.zone_mean_qualified_operations / group.candidate_operations,
            100 * group.full_qualified_operations / group.candidate_operations,
            s=16,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
        c.scatter(
            group.zone_mean_selected_full_service_excess_c,
            group.energy_difference_kwh / 1000,
            s=18,
            color=color,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    b.plot([0, 100], [0, 100], ls="--", lw=0.7, color="#909090")
    b.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        xticks=[0, 50, 100],
        yticks=[0, 50, 100],
        xlabel="Mean review (%)",
        ylabel="Individual review (%)",
    )
    title(b, "b", "Qualifying\noperations")
    c.axhline(0, color="#909090", lw=0.6, ls="--")
    c.set(
        xlim=(-0.04, 1.08),
        ylim=(-2, 60),
        xticks=[0, 0.5, 1],
        yticks=[0, 30, 60],
        xlabel="Mean-choice\nservice excess (°C)",
        ylabel="Selected energy\ndifference (MWh)",
    )
    title(c, "c", "Selected\nenergy difference")
    scopes = choices.full_permission.value_counts()
    if set(scopes.index) - {"building", "floor", "native_zone"}:
        raise ValueError("Unknown selected command scope")
    values = [int(scopes.get(scope, 0)) for scope in ("building", "floor", "native_zone")]
    d.barh(range(3), values, color=[BLUE, PURPLE, GREEN], height=0.55)
    d.set_yticks(range(3), ["Building", "Floor", "Zone"])
    d.set_ylim(2.7, -0.7)
    d.set_xlim(0, 75)
    d.set_xticks([0, 54])
    d.set_xlabel("Settings")
    for y, value in enumerate(values):
        d.text(value + 2, y, str(value), va="center", fontsize=FONT)
    title(d, "d", "Command\nscope")
    fig.legend(
        [
            Line2D([], [], ls="", marker="o", color=PURPLE, ms=4),
            Line2D([], [], ls="", marker="o", color=BLUE, ms=4),
        ],
        [f"Changed choice ({int(changed.sum())})", f"Same choice ({int((~changed).sum())})"],
        loc="lower center",
        bbox_to_anchor=(0.51, 0.015),
        ncol=2,
        frameon=False,
        facecolor="none",
        edgecolor="none",
        handletextpad=0.45,
        columnspacing=1.4,
    )
    return fig, paths, "abcd"


def export(fig, output, number, paths, letters, fonts):
    fig.canvas.draw()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        expected = TITLE if text.get_gid() == "thermal-heterogeneity-panel-title" else FONT
        actual = FontProperties(
            fname=findfont(text.get_fontproperties(), fallback_to_default=False)
        ).get_name()
        if actual != "Times New Roman" or abs(text.get_fontsize() - expected) > 1e-8:
            raise ValueError(f"Font contract differs: {text.get_text()!r}")
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [Text.get_window_extent(t, renderer) for t in texts]
    collision = []
    for i, first in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            second = boxes[j]
            width = min(first.x1, second.x1) - max(first.x0, second.x0)
            height = min(first.y1, second.y1) - max(first.y0, second.y0)
            if width > 0.5 and height > 0.5:
                collision.append([texts[i].get_text(), texts[j].get_text()])
    clipped = [
        t.get_text()
        for t, box in zip(texts, boxes)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height
    ]
    marker_overlap = []
    if number == 1:
        map_axis = fig.axes[0]
        for collection in map_axis.collections:
            if not isinstance(collection, PathCollection) or not len(collection.get_offsets()):
                continue
            positions = collection.get_offset_transform().transform(collection.get_offsets())
            sizes = np.broadcast_to(collection.get_sizes(), (len(positions),))
            for text, box in zip(texts, boxes):
                if text.axes is not map_axis:
                    continue
                radius = np.sqrt(sizes) * fig.dpi / 144
                hit = (
                    (positions[:, 0] + radius > box.x0)
                    & (positions[:, 0] - radius < box.x1)
                    & (positions[:, 1] + radius > box.y0)
                    & (positions[:, 1] - radius < box.y1)
                )
                if np.any(hit):
                    marker_overlap.append(text.get_text())
    legends = audit_data_safe_legends(fig)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"Figure_{number:02}"
    for extension in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=300, facecolor="white")
    audit = {
        "status": "requires_layout_correction"
        if collision or clipped or marker_overlap
        else "rendered_pending_visual_review",
        "font_pt": FONT,
        "title_pt": TITLE,
        "figure_width_in": WIDTH,
        "fonts": fonts,
        "panels": letters,
        "text_collisions": collision,
        "clipped_text": clipped,
        "map_text_marker_overlaps": marker_overlap,
        "legends": legends,
        "input_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    prefix.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    plt.close(fig)
    print(
        json.dumps(
            {
                "figure": number,
                "status": audit["status"],
                "collision": collision,
                "clipped": clipped,
                "marker_overlap": marker_overlap,
            }
        )
    )
    if collision or clipped or marker_overlap:
        raise ValueError("Figure layout requires correction; preview and diagnostics retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "output/from_raw")
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "output/strengthening")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/publication_development")
    parser.add_argument("--figure", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    args = parser.parse_args()
    raw, analysis, output = [
        p.resolve() for p in (args.raw_root, args.analysis_root, args.output_root)
    ]
    if not all(p.is_relative_to(ROOT) for p in (raw, analysis, output)):
        raise ValueError("Publication inputs and outputs must remain inside the project")
    fonts = style()
    for number in args.figure:
        fig, paths, letters = {1: figure_one, 2: figure_two}[number](raw, analysis)
        export(fig, output, number, paths, letters, fonts)


if __name__ == "__main__":
    main()
