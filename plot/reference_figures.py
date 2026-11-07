"""Six publication figures from project-local result tables, geometry and trajectories.

The publication notebook reads the Data_for_Figures tree in strict numerical input mode.
Use --data-root PATH to select another complete Data_for_Figures tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("hem_figures")
DISPLAY = {"Malmo": "Malmö", "Brasilia": "Brasília"}
COMPOSITIONS = [f"{source} {letter}" for source in ["Multinational", "China"] for letter in "ABC"]
P = {
    "blue": "#4778A8",
    "plum": "#AF668A",
    "teal": "#26898B",
    "coral": "#DF7D73",
    "gold": "#E6B34B",
    "gray": "#9CA3AA",
    "grid": "#E2E5E8",
    "ink": "#000000",
    "cool": "#A3BDD2",
    "green": "#8BA88B",
    "sand": "#CCB16D",
}
C = {
    "width_in": 7.6,
    "height_in": 9.4,
    "font_size": 9,
    "title_size": 10,
    "panel_size": 10,
    "font": "Times New Roman",
    "figure6_font": "Times New Roman",
    "dpi": 300,
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class Study:
    def __init__(self, data_root: Path, output: Path, strict_raw: bool = False):
        self.root = data_root.resolve()
        self.output = output.resolve()
        self.strict_raw = strict_raw
        self.used = {}
        self.checks = {}
        audit = json.loads((ROOT / "plot/reference_figure_inputs.json").read_text(encoding="utf-8"))
        self.expected_paths = {
            Path(x["path"]).name: x["original_project_path"].removeprefix("Data_for_Figures/") for x in audit
        }
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "plot_data").mkdir(exist_ok=True)

    def path(self, name: str) -> Path:
        preferred = self.root / self.expected_paths.get(name, name)
        paths = [preferred] if preferred.is_file() else list(self.root.rglob(name))
        if len(paths) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {name} under {self.root}; found {len(paths)}"
            )
        p = paths[0]
        self.used["Data_for_Figures/" + p.relative_to(self.root).as_posix()] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        return p

    def has(self, name: str) -> bool:
        preferred = self.root / self.expected_paths.get(name, name)
        return preferred.is_file() or len(list(self.root.rglob(name))) == 1

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path(name), low_memory=False)

    def json(self, name: str) -> dict:
        return json.loads(self.path(name).read_text(encoding="utf-8-sig"))

    def table(self, name: str, df: pd.DataFrame):
        df.to_csv(self.output / "plot_data" / name, index=False, float_format="%.12g")

    def check(self, name: str, condition: bool, value: Any = None):
        self.checks[name] = {"passed": bool(condition), "value": _jsonable(value)}
        if not condition:
            raise ValueError(f"Data integrity check failed: {name}; value={value}")

    def climates(self) -> list[str]:
        return [x["climate"] for x in self.json("GLOBAL_CLIMATES.json")["climates"]]


def choose_font(request: str) -> str:
    options = {
        "sans": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    }.get(request, [request])
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in options:
        if name in available:
            return name
    raise RuntimeError(f"Font {request!r} is not installed. Use --font sans or --font serif.")


def style(number: int):
    mpl.rcdefaults()
    family = choose_font(C["figure6_font"] if number == 6 else C["font"])
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": C["font_size"],
            "axes.titlesize": C["title_size"],
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
            "axes.labelsize": C["font_size"],
            "axes.labelpad": 5,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#454B51",
            "axes.labelcolor": P["ink"],
            "xtick.labelsize": C["font_size"],
            "ytick.labelsize": C["font_size"],
            "xtick.color": P["ink"],
            "ytick.color": P["ink"],
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "text.color": P["ink"],
            "legend.frameon": False,
            "legend.fontsize": C["font_size"],
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.45,
            "grid.color": P["grid"],
            "grid.linestyle": ":",
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "hem-data-faithful-20261101",
            "path.simplify": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )
    return family


def page(number: int):
    family = style(number)
    f = plt.figure(figsize=(C["width_in"], C["height_in"]), facecolor="white")
    f._hem_meta = {"figure": number, "font": family, "source_mode": "CSV numerical results"}
    return f


def axes(fig, rect, letter: str, heading: str, grid: str | None = "both", letter_x=-0.16):
    a = fig.add_axes(rect)
    text = f"({letter.strip('()')}) {heading}" if heading else ""
    title = a.set_title(text, loc="left", fontsize=C["title_size"], fontweight="bold", pad=8)
    title.set_gid("panel-title")
    if heading:
        fig._hem_panel_headings = getattr(fig, "_hem_panel_headings", []) + [title]
    if grid:
        a.grid(axis=grid)
    return a


def panel_heading(fig, x, y, letter, title):
    heading = fig.text(
        x,
        y,
        f"({letter.strip('()')}) {title}",
        fontweight="bold",
        fontsize=C["title_size"],
        va="center",
        gid="panel-title",
    )
    fig._hem_panel_headings = getattr(fig, "_hem_panel_headings", []) + [heading]


def bubble_legend_check(study, name, legend, values, area_scale):
    """Validate legend areas using the same points-squared scale as the data."""
    observed = [float(handle.get_sizes()[0]) for handle in legend.legend_handles[:len(values)]]
    study.check(name + "_legend_areas", np.allclose(observed, np.asarray(values) * area_scale),
                {"values": values, "areas_pt2": observed, "area_scale": area_scale})
    study.check(name + "_legend_marker_scale", legend.markerscale == 1.0, legend.markerscale)


def marker_handle(color, marker="o", label=None, size=4.6):
    return Line2D(
        [],
        [],
        color=color,
        marker=marker,
        ls="",
        ms=size,
        label=label,
        markeredgecolor="white",
        markeredgewidth=0.45,
    )


def equal_earth(longitude, latitude):
    # Same forward projection and coefficients as the supplied plotting source.
    lon, lat = np.broadcast_arrays(np.asarray(longitude), np.asarray(latitude))
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(np.deg2rad(lat)))
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    x = (
        np.deg2rad(lon)
        * np.cos(th)
        / (np.sqrt(3) / 2 * (a1 + 3 * a2 * th**2 + 7 * a3 * th**6 + 9 * a4 * th**8))
    )
    y = th * (a1 + a2 * th**2 + a3 * th**6 + a4 * th**8)
    return x, y


def project(ring):
    a = np.asarray(ring)
    return np.column_stack(equal_earth(a[:, 0], a[:, 1]))


def map_base(ax, land, countries):
    rings = []
    for f in land["features"]:
        g = f["geometry"]
        gs = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        rings.extend(project(v[0]) for v in gs)
    land_layer = PolyCollection(rings, facecolors="#F3F1ED", edgecolors="#929A9E", lw=0.4, zorder=1)
    land_layer.set_path_effects(
        [
            pe.SimplePatchShadow(
                shadow_rgbFace=(0.20, 0.24, 0.28), offset=(0.65, -0.65), alpha=0.16
            ),
            pe.Normal(),
        ]
    )
    ax.add_collection(land_layer)
    country_layer = PolyCollection(
        [project(r) for f in countries["features"] for r in f["rings"]],
        facecolors="#D9CFE5",
        edgecolors="#A79BB4",
        lw=0.4,
        zorder=2,
    )
    country_layer.set_path_effects(
        [
            pe.SimplePatchShadow(
                shadow_rgbFace=(0.20, 0.24, 0.28), offset=(0.35, -0.35), alpha=0.12
            ),
            pe.Normal(),
        ]
    )
    ax.add_collection(country_layer)
    for lon in range(-180, 181, 60):
        ax.plot(
            *equal_earth(np.full(146, lon), np.linspace(-60, 85, 146)),
            c="#D8DBDF",
            lw=0.4,
            ls=":",
            zorder=0,
        )
    for lat in [-60, -30, 0, 30, 60]:
        ax.plot(
            *equal_earth(np.linspace(-180, 180, 361), np.full(361, lat)),
            c="#D8DBDF",
            lw=0.4,
            ls=":",
            zorder=0,
        )
    boundary = np.vstack(
        [
            project(np.column_stack([np.linspace(-180, 180, 361), np.full(361, -60)])),
            project(np.column_stack([np.full(146, 180), np.linspace(-60, 85, 146)])),
            project(np.column_stack([np.linspace(180, -180, 361), np.full(361, 85)])),
            project(np.column_stack([np.full(146, -180), np.linspace(85, -60, 146)])),
        ]
    )
    ax.add_patch(Polygon(boundary, closed=True, fc="#F5F8FA", ec="none", zorder=-3))
    ax.plot(*np.vstack([boundary, boundary[:1]]).T, c="#555B60", lw=0.65, zorder=3)
    ax.set(
        xlim=(boundary[:, 0].min(), boundary[:, 0].max()),
        ylim=(boundary[:, 1].min(), boundary[:, 1].max()),
    )
    ax.set_aspect("equal")
    ax.set_axis_off()


def figure_one(s: Study):
    points = s.csv("CITY_MAP_POINTS.csv")
    countries = s.csv("COUNTRY_DEMAND_SUMMARY.csv")
    variance = s.csv("ASHRAE_NESTED_VARIANCE.csv")
    sens = s.csv("LEAVE_ONE_SOURCE_VARIANCE.csv")
    rooms = s.csv("SHARED_ROOM_SUMMARY.csv")
    climates = pd.DataFrame(s.json("GLOBAL_CLIMATES.json")["climates"])
    s.check(
        "fig1_observation_count",
        points.record_count.sum() == countries.record_count.sum() == 33845,
        33845,
    )
    s.check(
        "fig1_locations",
        len(points) == 49 and len(countries) == 15 and len(climates) == 9,
        [len(points), len(countries), len(climates)],
    )
    f = page(1)
    f.set_size_inches(7.6, 8.0)
    a = f.add_axes([0.065, 0.58, 0.82, 0.37])
    panel_heading(f, 0.1516394530, 0.962, "a", "Global observations and simulation climates")
    map_base(a, s.json("ne_110m_land.geojson"), s.json("ne_110m_observed_countries.json"))
    cmap = LinearSegmentedColormap.from_list(
        "demand", ["#365889", "#56A9BE", "#F4DE91", "#E77C5E", "#A92C46"]
    )
    norm = Normalize(np.floor(points.demand_center_c.min()), np.ceil(points.demand_center_c.max()))
    area_scale = 360 / points.record_count.max()
    cloud = a.scatter(
        *equal_earth(points.longitude, points.latitude),
        s=area_scale * points.record_count,
        c=points.demand_center_c,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        lw=0.8,
        zorder=4,
    )
    a.scatter(
        *equal_earth(climates.longitude, climates.latitude),
        marker="*",
        s=118,
        c=P["gold"],
        edgecolors="#5E4A13",
        lw=0.7,
        zorder=6,
    )
    labels = {
        "Miami": (-71, 19),
        "Montreal": (-66, 50),
        "Malmo": (27, 65),
        "Tunis": (14, 29),
        "Delhi": (86, 22),
        "Beijing": (133, 44),
        "Singapore": (127, -10),
        "Sydney": (145, -46),
        "Brasilia": (-39, -26),
    }
    for row in climates.itertuples():
        a.annotate(
            DISPLAY.get(row.climate, row.climate),
            equal_earth(row.longitude, row.latitude),
            xytext=equal_earth(*labels[row.climate]),
            ha="center",
            va="center",
            fontsize=7.6,
            arrowprops={"arrowstyle": "-", "lw": 0.55, "color": "#6B7075"},
            zorder=7,
            bbox={"facecolor": "white", "alpha": 0.62, "edgecolor": "none", "pad": 0.3},
        )
    for latitude in [-60, -30, 0, 30, 60]:
        x, y = equal_earth(-180, latitude)
        label = "0°" if latitude == 0 else f"{abs(latitude)}°" + ("N" if latitude > 0 else "S")
        a.text(x - 0.065, y, label, ha="right", va="center", clip_on=False)
    for longitude in [-180, -120, -60, 0, 60, 120, 180]:
        x, y = equal_earth(longitude, -60)
        label = "0°" if longitude == 0 else f"{abs(longitude)}°" + ("E" if longitude > 0 else "W")
        a.text(x, y - 0.075, label, ha="center", va="top", clip_on=False)
    f.canvas.draw()
    box = a.get_position()
    cax = f.add_axes([0.906, box.y0, 0.018, box.height])
    cb = f.colorbar(cloud, cax=cax)
    cb.outline.set_linewidth(0.5)
    cax.set_title("Mean demand\ncenter (°C)", fontsize=8.7, fontweight="normal", pad=8)
    cb.ax.yaxis.set_major_locator(MultipleLocator(2))
    handles = [
        a.scatter([], [], s=area_scale * n, c="#9B9D9F", ec="white", lw=0.8)
        for n in [100, 500, 2000, 5000]
    ]
    handles += [
        Patch(fc="#E4DFEC", ec="#AAA4B4", lw=0.5),
        Line2D([], [], ls="", marker="*", ms=8, c=P["gold"], mec="#5E4A13", mew=0.6),
    ]
    size_legend = f.legend(
        handles,
        ["100", "500", "2,000", "5,000 records", "Countries with observations", "9 weather contexts"],
        loc="center",
        bbox_to_anchor=(0.51, 0.540),
        ncol=6,
        columnspacing=1.0,
        handletextpad=0.9,
        fontsize=7.2,
        markerscale=1,
    )
    bubble_legend_check(s, "fig1_records", size_legend, [100, 500, 2000, 5000], area_scale)
    b = axes(f, [0.1516394530, 0.075, 0.30, 0.390], "b", "Demand within countries", "x", letter_x=-0.49)
    cc = axes(f, [0.6316394530, 0.295, 0.29, 0.170], "c", "Variation by scale", "x", letter_x=-0.24)
    d = axes(f, [0.6316394530, 0.075, 0.29, 0.105], "d", "", "x", letter_x=-0.24)
    panel_heading(f, 0.6316394530, 0.225, "d", "Shared-room requests")
    countries = countries.sort_values("demand_center_median_c", ascending=False).reset_index(
        drop=True
    )
    names = (
        countries.country.str.replace("_", " ")
        .str.title()
        .replace({"Uk": "United Kingdom", "Usa": "United States"})
    )
    y = np.arange(15)
    for i, row in countries.iterrows():
        paint = cmap(norm(row.demand_center_median_c))
        b.plot([row.demand_center_p10_c, row.demand_center_p90_c], [i, i], c=paint, lw=1.8)
        b.plot(
            [row.demand_center_p10_c, row.demand_center_p90_c],
            [i, i],
            ls="",
            marker="|",
            c=paint,
            ms=5,
            mew=1,
        )
        b.scatter(row.demand_center_median_c, i, s=48, c=[paint], ec="white", lw=0.55, zorder=3)
        b.text(
            1.05,
            i,
            f"{int(row.record_count):,}",
            transform=b.get_yaxis_transform(),
            va="center",
            fontsize=7.8,
            clip_on=False,
        )
    b.set(
        yticks=y, yticklabels=names, ylim=(14.7, -0.8), xlabel="Demand center (°C)", xlim=(18, 32.7)
    )
    b.set_xticks([20, 24, 28, 32])
    b.tick_params(axis="y", length=0, pad=4)
    b.text(1.05, 1.025, "$n$ (records)", transform=b.transAxes, fontsize=7.2, ha="left")
    # Country intervals and medians are defined in the caption.
    order = ["country", "city", "building_id", "unit_group", "within_unit"]
    v = (
        variance[variance.weighting.eq("unit_weight") & variance.griffiths_coefficient.eq(0.5)]
        .set_index("component")
        .loc[order]
    )
    lim = (
        sens[sens.weighting.eq("unit_weight") & sens.griffiths_coefficient.eq(0.5)]
        .groupby("component")
        .share.agg(["min", "max"])
        .loc[order]
        * 100
    )
    shares = v.share.to_numpy() * 100
    cc.barh(
        range(5),
        shares,
        color=["#68A7B0", "#9DB6D2", "#B09FC5", "#8A365E", "#DBB562"],
        height=0.65,
        zorder=2,
    )
    # Plot range endpoints directly, even if the nominal value is outside a leave-one-source range.
    for i, val in enumerate(shares):
        cc.plot([lim.iloc[i]["min"], lim.iloc[i]["max"]], [i, i], c="#232323", lw=0.8, zorder=4)
        cc.plot(
            [lim.iloc[i]["min"], lim.iloc[i]["max"]],
            [i, i],
            ls="",
            marker="|",
            ms=5,
            mew=0.8,
            c="#232323",
            zorder=4,
        )
        cc.text(max(val, lim.iloc[i]["max"]) + 2, i, f"{val:.1f}", va="center", fontsize=7.8)
    cc.set(
        yticks=range(5),
        yticklabels=["Country", "City", "Building", "Within\nbuilding", "Repeated\nrecords"],
        ylim=(4.65, -0.65),
        xlim=(0, 78),
        xlabel="Variance share (%)",
    )
    cc.set_xticks([0, 20, 40, 60])
    cc.tick_params(axis="y", length=0)
    # Whisker definition is provided in the caption.
    rr = rooms[rooms.minutes.eq(30)].set_index("dataset").loc[["NUS", "ENTH"]]
    left = np.zeros(2)
    for field, paint, label in [
        ("Same", P["teal"], "Aligned"),
        ("Mixed", "#B1A1CA", "Mixed"),
        ("Opposed", P["coral"], "Opposite"),
    ]:
        width = rr[field].to_numpy() / rr.windows.to_numpy() * 100
        d.barh([0, 1], width, left=left, height=0.55, color=paint, ec="white", lw=0.4, zorder=2)
        for i, w in enumerate(width):
            if w < 12:
                d.annotate(
                    f"{w:.1f}%",
                    (left[i] + w / 2, i - 0.275),
                    xytext=(98.5, i - 0.46),
                    ha="right",
                    va="center",
                    fontsize=6.7,
                    arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#6F6261"},
                    zorder=4,
                )
            else:
                d.text(
                    left[i] + w / 2,
                    i,
                    f"{w:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="white" if field == "Same" else "#28262A",
                    zorder=3,
                )
        left += width
    d.set(
        yticks=[0, 1],
        yticklabels=[f"{dict(NUS='Singapore', ENTH='Longitudinal')[name]}\n($n = {int(row.windows)}$)" for name, row in rr.iterrows()],
        ylim=(1.6, -0.85),
        xlim=(0, 100),
        xlabel="Room-time windows (%)",
    )
    d.set_xticks([0, 25, 50, 75, 100])
    d.tick_params(axis="y", length=0)
    f.legend(
        [Patch(fc=c) for c in [P["teal"], "#B1A1CA", P["coral"]]],
        ["Aligned", "Mixed", "Opposite"],
        bbox_to_anchor=(0.7766394530, 0.196),
        loc="center",
        ncol=3,
        columnspacing=0.7,
        fontsize=7,
    )
    s.table("Figure_01_countries.csv", countries)
    s.table("Figure_01_map_points.csv", points)
    s.table(
        "Figure_01_variance.csv",
        pd.DataFrame(
            {
                "component": order,
                "share_pct": shares,
                "min_pct": lim["min"].values,
                "max_pct": lim["max"].values,
            }
        ),
    )
    s.table("Figure_01_rooms.csv", rr.reset_index())
    f._hem_meta.update(
        observations=33845,
        map_locations=49,
        countries=15,
        weather_contexts=9,
        uncertainty="leave-one-source ranges; not confidence intervals",
    )
    return f


def heatmap(ax, matrix, cmap, vmin, vmax, fmt, climates):
    im = ax.pcolormesh(
        np.arange(-0.5, 6),
        np.arange(-0.5, 9),
        np.asarray(matrix),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        edgecolors="white",
        linewidth=0.5,
        rasterized=False,
    )
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_yticks(range(9), [DISPLAY.get(x, x) for x in climates])
    ax.set_xticks(range(6), range(1, 7))
    ax.set_xticks(np.arange(-0.5, 6), minor=True)
    ax.set_yticks(np.arange(-0.5, 9), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    ax.set_xlabel("Demand composition")
    for (i, j), val in np.ndenumerate(np.asarray(matrix)):
        label = "0" if abs(val) < 0.005 else format(val, fmt)
        ax.text(
            j,
            i,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            color="white" if val > (vmin + 0.58 * (vmax - vmin)) else "#3C3448",
        )
    return im


def figure_two(s: Study):
    counts = s.csv("NOMINAL_QUALIFICATION_COUNTS.csv")
    choices = s.csv("NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv")
    uncertainty = s.csv("SETTING_SELECTION_STABILITY_ALL_BRANCHES.csv")
    climates = s.climates()
    data = counts.merge(
        choices, on=["climate", "composition"], validate="one_to_one", suffixes=("", "_choice")
    )
    s.check("fig2_pairs", len(data) == 54, len(data))
    s.check("fig2_choice_difference_count", int(data.choices_differ.sum()) == 24, 24)
    s.check(
        "fig2_energy_agreement",
        np.allclose(data.energy_difference_kwh, data.extra_energy_kwh, atol=1e-8, rtol=0),
    )
    s.check("fig2_qualifying_set_count", int(data.qualified_operation_sets_differ.sum()) == 54, 54)
    data["row"] = data.climate.map({c: i for i, c in enumerate(climates)})
    data["column"] = data.composition.map({c: i for i, c in enumerate(COMPOSITIONS)})
    f = page(2)
    a = axes(f, [0.12, 0.565, 0.315, 0.36], "a", "Full-demand limit exceedance", None, letter_x=-0.34)
    b = axes(f, [0.635, 0.565, 0.325, 0.36], "b", "Energy of the choices", "x", letter_x=-0.26)
    c = axes(
        f, [0.12, 0.10, 0.34, 0.36], "c", "Service shifts with selection", "both", letter_x=-0.30
    )
    d = axes(f, [0.635, 0.10, 0.27, 0.36], "d", "Choices under demand error", None, letter_x=-0.37)
    cm = LinearSegmentedColormap.from_list("excess", ["#F7F4F8", "#DEC5DF", "#B277B8", "#713775"])
    matrix = data.pivot(
        index="climate", columns="composition", values="zone_mean_selected_full_service_excess_c"
    ).loc[climates, COMPOSITIONS]
    s.check("fig2_excess_bounds", matrix.to_numpy().max() < 1.0, float(matrix.to_numpy().max()))
    im = heatmap(a, matrix, cm, 0, 1, ".2f", climates)
    cax = f.add_axes([0.452, 0.565, 0.013, 0.36])
    cb = f.colorbar(im, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("Excess mismatch (°C)", labelpad=4)
    cb.outline.set_linewidth(0.5)
    for i in range(9):
        if i % 2 == 1:
            b.axhspan(i - 0.5, i + 0.5, fc="#F3F4F6", zorder=-2)
    for row in data.itertuples():
        b.scatter(
            row.energy_difference_kwh / 1000,
            row.row + (row.column - 2.5) * 0.055,
            s=43,
            c=P["plum"] if row.choices_differ else "#C5C7C9",
            ec="#5D4663" if row.choices_differ else "#82878C",
            lw=0.45,
            alpha=0.9,
            zorder=3,
        )
    b.axvline(0, c="#676C72", ls="--", lw=0.7)
    b.set(
        yticks=range(9),
        yticklabels=[DISPLAY.get(x, x) for x in climates],
        ylim=(8.5, -0.5),
        xlim=(-3, 60),
        xlabel="Full minus mean choice\n(MWh yr$^{-1}$)",
    )
    b.set_xticks([0, 15, 30, 45, 60])
    b.tick_params(axis="y", length=0)
    b.legend(
        [marker_handle("#C5C7C9"), marker_handle(P["plum"])],
        ["Same choice (30)", "Different choice (24)"],
        loc="upper right",
        fontsize=7,
        handlelength=0.8,
        borderaxespad=0.4,
    )
    changed = data[data.choices_differ]
    energies = changed.extra_energy_kwh / 1000
    c.scatter(
        changed.zone_mean_selected_population_loss_vs_full_choice_c,
        changed.zone_mean_selected_fixed_tail_loss_vs_full_choice_c,
        s=3.4 * energies,
        c=P["plum"],
        alpha=0.62,
        ec="#8A508C",
        lw=0.65,
        zorder=3,
    )
    c.scatter([0], [0], s=25, c="#BFC1C4", ec="white", lw=0.4, zorder=4)
    c.axvline(0, c="#72767C", ls="--", lw=0.65)
    c.axhline(0, c="#72767C", ls="--", lw=0.65)
    c.set(
        xlim=(-0.08, 1.05),
        ylim=(-1.55, 1.65),
        xlabel="Population mismatch,\nmean minus full choice (°C)",
        ylabel="Least-served mismatch,\nmean minus full choice (°C)",
    )
    c.set_xticks([0, 0.25, 0.5, 0.75, 1])
    c.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1])
    # The coincident unchanged choices are described in the legend and caption.
    handles = [
        c.scatter([], [], s=3.4 * e, c=P["plum"], alpha=0.62, ec="#8A508C", lw=0.5)
        for e in [10, 30, 50]
    ]
    size_legend = c.legend(
        handles,
        ["10", "30", "50"],
        title="Extra energy (MWh yr$^{-1}$)",
        title_fontsize=7,
        ncol=3,
        loc="upper right",
        fontsize=9,
        handlelength=2,
        handletextpad=0.6,
        columnspacing=0.8,
        borderaxespad=0.3,
        markerscale=1,
    )
    bubble_legend_check(s, "fig2_extra_energy", size_legend, [10, 30, 50], 3.4)
    selected = uncertainty[
        uncertainty.library_and_scenario_design.eq("pooled_study_shared")
        & uncertainty.scenario.eq("studyGroupSD0.5_personSD0.5")
    ]
    matrix2 = (
        selected.pivot(
            index="climate",
            columns="composition",
            values="conditional_choice_disagreement_fraction",
        ).loc[climates, COMPOSITIONS]
        * 100
    )
    s.check("fig2_uncertainty_complete", not matrix2.isna().any().any(), matrix2.shape)
    cm2 = LinearSegmentedColormap.from_list(
        "choice_frequency", ["#F6F8F6", "#B6D3CC", "#52958B", "#17665F"]
    )
    im2 = heatmap(d, matrix2, cm2, 0, 100, ".0f", climates)
    cax2 = f.add_axes([0.923, 0.10, 0.013, 0.36])
    cb2 = f.colorbar(im2, cax=cax2, ticks=[0, 25, 50, 75, 100])
    cb2.set_label("Changed choices (%)", fontsize=8, labelpad=3)
    cb2.outline.set_linewidth(0.5)
    # Demand-error design is documented in the caption.
    s.table("Figure_02_choices.csv", data)
    s.table("Figure_02_uncertainty.csv", selected)
    f._hem_meta.update(
        settings=54,
        changed_choices=24,
        changed_qualifying_sets=54,
        heatmap_excess_range=[0, 1],
        uncertainty_scenario="studyGroupSD0.5_personSD0.5",
    )
    return f


def native_floor_panels(s: Study):
    profiles = s.csv("CONSTRAINED_PLACEMENT_PROFILES.csv")
    rep = profiles[profiles.context.eq("Miami") & profiles.composition_id.eq(2026102001)]
    model = s.json("Miami.geometry.json")
    polygons = {
        z: np.asarray(v, dtype=float) for z, v in model["floors"].items() if z in set(rep.zone_id)
    }
    s.check("fig3_native_zones", len(polygons) == 15, len(polygons))
    levels = sorted({float(v[:, 2].mean()) for v in polygons.values()})
    slots = {}
    initial = rep[rep.layout.eq("within_floor_relocation_0pct")]
    # Display-only positions: spaced schematic markers, one per actual profile.
    for zone, v in sorted(polygons.items()):
        starts = v[:, :2]
        edges = np.roll(starts, -1, axis=0) - starts
        boundary = MplPath(np.vstack([starts, starts[:1]]), closed=True)
        n = int(initial.zone_id.eq(zone).sum())
        gx, gy = np.meshgrid(
            np.arange(starts[:, 0].min(), starts[:, 0].max(), 0.55),
            np.arange(starts[:, 1].min(), starts[:, 1].max(), 0.55),
        )
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        t = np.clip(
            ((xy[:, None, :] - starts) * edges).sum(axis=2) / (edges * edges).sum(axis=1), 0, 1
        )
        dist = np.linalg.norm(xy[:, None, :] - starts - t[:, :, None] * edges, axis=2)
        candidates = xy[boundary.contains_points(xy) & (dist.min(axis=1) > 1.1)]
        chosen = []
        nearest = np.full(len(candidates), np.inf)
        idx = int(np.argmin(np.linalg.norm(candidates - candidates.mean(axis=0), axis=1)))
        for _ in range(n):
            chosen.append(candidates[idx])
            nearest = np.minimum(nearest, np.linalg.norm(candidates - candidates[idx], axis=1))
            idx = int(np.argmax(nearest))
        slots[zone] = np.asarray(chosen)
    norm = Normalize(np.floor(rep.centre_c.min()), np.ceil(rep.centre_c.max()))
    out = []
    for ri, layout in enumerate(["within_floor_relocation_0pct", "within_floor_relocation_25pct"]):
        group = rep[rep.layout.eq(layout)]
        s.check(f"fig3_native_layout_{ri}", len(group) == 267, len(group))
        for fi, lev in enumerate(levels):
            panel = {"row": ri, "floor": fi, "polygons": [], "points": [], "values": []}
            for zone, v in polygons.items():
                if float(v[:, 2].mean()) != lev:
                    continue
                local = group[group.zone_id.eq(zone)].sort_values("profile_id")
                if len(local) != len(slots[zone]):
                    raise ValueError("Rearrangement changed zone capacity")
                panel["polygons"].append(v[:, :2])
                panel["points"].append(slots[zone])
                panel["values"].extend(local.centre_c.tolist())
            panel["points"] = np.vstack(panel["points"])
            panel["values"] = np.asarray(panel["values"])
            out.append(panel)
    return (
        out,
        (norm.vmin, norm.vmax),
        {
            "mode": "native_CSV_and_geometry_JSON",
            "coordinates": "schematic slots within real zone polygons",
        },
    )


def figure_three(s: Study):
    effects = s.csv("PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv")
    frontier = s.csv("DESCRIPTIVE_BUDGET_SUMMARY.csv")
    if s.has("CONSTRAINED_PLACEMENT_PROFILES.csv") and s.has("Miami.geometry.json"):
        panels, bounds, provenance = native_floor_panels(s)
    elif s.strict_raw:
        raise FileNotFoundError(
            "Figure 3a needs CONSTRAINED_PLACEMENT_PROFILES.csv and Miami.geometry.json"
        )
    else:
        raise FileNotFoundError("Native profile and geometry inputs are required")
        LOG.warning(
            "Figure 3a: replaying source SVG geometry/markers; raw profiles and geometry are absent."
        )
    f = page(3)
    f._hem_meta["panel_a"] = provenance
    panel_heading(f, 0.095, 0.963, "a", "Office-floor demand placement")
    cmap = LinearSegmentedColormap.from_list(
        "placement", ["#2468A3", "#52A9CB", "#F3DFA2", "#E58E6E", "#B73555"]
    )
    norm = Normalize(*bounds)
    cbax = f.add_axes([0.92, 0.405, 0.014, 0.51])
    cb = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbax)
    cb.set_ticks([18, 22, 26, 30])
    cb.ax.tick_params(pad=2, length=2)
    cb.set_label("Demand center (°C)", labelpad=5)
    cb.outline.set_linewidth(0.5)
    columns = [0.115, 0.47]
    rows = [0.755, 0.58, 0.405]
    for ri, x in enumerate(columns):
        f.text(
            x + 0.20, 0.935, ["Random mixing", "25% move allowance"][ri], ha="center", va="center"
        )
    for fi, y in enumerate(rows):
        f.add_artist(
            Rectangle(
                (0.045, y), 0.026, 0.16, transform=f.transFigure, fc="#ECF0F3", ec="none", zorder=-2
            )
        )
        f.text(
            0.058,
            y + 0.08,
            ["Lower floor", "Middle floor", "Upper floor"][fi],
            rotation=90,
            ha="center",
            va="center",
        )
    for panel in panels:
        ri, fi = panel["row"], panel["floor"]
        ax = f.add_axes([columns[ri], rows[fi], 0.40, 0.17])
        polygons = panel["polygons"]
        dots = panel["points"]
        allxy = np.vstack(polygons)
        lo = allxy.min(axis=0)
        hi = allxy.max(axis=0)
        base = Rectangle(lo, *(hi - lo), fc="#FBFAF7", ec="#7F8A92", lw=0.8, zorder=0)
        base.set_path_effects(
            [
                pe.SimplePatchShadow(
                    shadow_rgbFace=(0.20, 0.24, 0.28), offset=(1.5, -1.5), alpha=0.16
                ),
                pe.Normal(),
            ]
        )
        ax.add_patch(base)
        for poly in polygons:
            # These are thermal-zone boundaries, not invented interior walls.
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    fc="#EDF1F2",
                    alpha=0.48,
                    ec="#9CA8AD",
                    lw=0.55,
                    ls=(0, (3, 2)),
                    zorder=1,
                )
            )
        points = ax.scatter(
            *dots.T, c=cmap(norm(panel["values"])), s=24, ec="white", lw=0.6, zorder=3
        )
        points.set_path_effects(
            [
                pe.SimplePatchShadow(
                    shadow_rgbFace=(0.20, 0.24, 0.28), offset=(0.4, -0.4), alpha=0.20
                ),
                pe.Normal(),
            ]
        )
        ax.set_xlim(lo[0] - 1, hi[0] + 1)
        ax.set_ylim(lo[1] - 1, hi[1] + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        if ri == 1 and fi == 2:
            ax.plot(
                [hi[0] - 13, hi[0] - 3],
                [lo[1] - 2.4, lo[1] - 2.4],
                lw=1.5,
                color="black",
                clip_on=False,
            )
            ax.text(hi[0] - 8, lo[1] - 3.8, "10 m", ha="center", va="top", clip_on=False)
    b = axes(f, [0.095, 0.07, 0.235, 0.255], "b", "Population service gain", letter_x=-0.30)
    c = axes(f, [0.425, 0.07, 0.235, 0.255], "c", "Least-served service gain", letter_x=-0.25)
    d = axes(f, [0.755, 0.07, 0.205, 0.255], "d", "Service requirements", letter_x=-0.28)
    all_stats = []
    for ax, field in [(b, "population_gain_c"), (c, "fixed_tail_gain_c")]:
        for control, paint, marker, label in [
            ("personal_center", P["blue"], "o", "Personal center"),
            ("personal_range", P["plum"], "s", "Personal range"),
        ]:
            stats = (
                effects[effects.control.eq(control)]
                .groupby("maximum_relocation_fraction")[field]
                .agg(
                    median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75)
                )
            )
            s.check(f"fig3_{field}_{control}_groups", len(stats) == 3, len(stats))
            x = np.r_[0, stats.index.to_numpy() * 100]
            med, low, high = (np.r_[0, stats[k].to_numpy()] for k in ["median", "q25", "q75"])
            ax.fill_between(x, low, high, color=paint, alpha=0.15, lw=0)
            ax.plot(x, med, c=paint, marker=marker, ms=4.8, lw=1.5, label=label, zorder=3)
            for xx, m, lower, h in zip(x, med, low, high):
                all_stats.append(
                    {
                        "metric": field,
                        "control": control,
                        "move_allowance_pct": xx,
                        "median": m,
                        "q25": lower,
                        "q75": h,
                    }
                )
        ax.set(
            xlim=(-4, 104),
            xticks=[0, 25, 50, 100],
            xlabel="Move allowance (%)",
            ylabel="Mismatch reduction (°C)",
        )
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylim(bottom=-0.02)
    b.set_ylim(-0.02, 0.56)
    c.set_ylim(-0.04, 1.43)
    f.legend(
        *b.get_legend_handles_labels(),
        loc="center",
        bbox_to_anchor=(0.38, 0.375),
        ncol=2,
        handlelength=1.4,
    )
    for fraction, paint, linestyle, label in [
        (1.0, "#454B51", "--", "Reference"),
        (0.9, "#38876F", "-", "90% reference"),
    ]:
        g = frontier[
            frontier.arm.eq("vav_strong_mixed_library")
            & frontier.budget_type.eq("reference_relative")
            & frontier.reference_fraction.eq(fraction)
        ].sort_values("maximum_relocation_fraction")
        s.check(f"fig3_frontier_{fraction}", len(g) == 4, g.feasible_settings.tolist())
        d.plot(
            g.maximum_relocation_fraction * 100,
            g.feasible_settings,
            c=paint,
            ls=linestyle,
            marker="o",
            ms=3.7,
            lw=1.3,
            label=label,
        )
        if fraction == 0.9:
            for xx, yy in zip(g.maximum_relocation_fraction * 100, g.feasible_settings):
                d.annotate(
                    str(int(yy)),
                    (xx, yy),
                    xytext=(0, -12 if yy > 50 else 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.9,
                    color=paint,
                )
    d.set(
        xlim=(-4, 108),
        ylim=(20, 60),
        xticks=[0, 25, 50, 100],
        yticks=[20, 30, 40, 50, 60],
        xlabel="Move allowance (%)",
        ylabel="Feasible settings",
    )
    d.legend(loc="lower right", handlelength=1.2)
    # Median and IQR definitions are in the caption.
    s.table("Figure_03_service_curves.csv", pd.DataFrame(all_stats))
    s.table("Figure_03_frontiers.csv", frontier)
    f._hem_meta.update(
        profile_count_per_layout=267,
        move_allowance=0.25,
        maximum_moves=66,
        uncertainty="interquartile ranges; not confidence intervals",
        furniture_added=False,
    )
    return f


def figure_four(s: Study):
    hp = s.csv("independent_pthp_MATCHED_CROSS_SYSTEM_EFFECTS.csv")
    reset = s.csv("vav_coordinated_outdoor_reset_MATCHED_CROSS_SYSTEM_EFFECTS.csv")
    decomp = s.csv("DECOMPOSITION.csv")
    hp = hp[hp.layout.eq("within_floor_relocation_25pct")]
    reset = reset[reset.layout.eq("within_floor_relocation_25pct")]
    keys = ["climate", "composition_id", "control", "layout"]
    common = hp[keys].merge(reset, on=keys, validate="one_to_one")
    s.check(
        "fig4_pair_support",
        len(hp) == 105 and len(reset) == 108 and len(common) == 105,
        [len(hp), len(reset), len(common)],
    )
    f = page(4)
    a = axes(f, [0.12, 0.565, 0.34, 0.36], "a", "HVAC changes energy effects", letter_x=-0.27)
    b = axes(f, [0.63, 0.565, 0.34, 0.36], "b", "Effect of supply-air reset", letter_x=-0.28)
    c = axes(f, [0.12, 0.115, 0.34, 0.35], "c", "End uses behind the change", "x", letter_x=-0.27)
    d = axes(f, [0.63, 0.115, 0.34, 0.35], "d", "Sources of service gain", "y", letter_x=-0.28)
    for ax, frame in [(a, hp), (b, reset)]:
        ax.axhline(0, c="#676D74", lw=0.7, ls="--")
        ax.axvline(0, c="#676D74", lw=0.7, ls="--")
        for control, paint, marker, label in [
            ("personal_center", P["blue"], "o", "Personal center"),
            ("personal_range", P["plum"], "s", "Personal range"),
        ]:
            g = frame[frame.control.eq(control)]
            ax.scatter(
                g.energy_change_pct_vav,
                g.energy_change_pct_alternative,
                c=paint,
                marker=marker,
                s=39,
                alpha=0.80,
                ec="white",
                lw=0.4,
                label=label,
                zorder=3,
            )
        ax.set_xlim(-2.5, 13)
        ax.set_xticks([0, 5, 10])
        ax.set_xlabel("Original VAV energy change (%)")
    f.legend(
        *a.get_legend_handles_labels(), loc="center", bbox_to_anchor=(0.545, 0.505), ncol=2
    )
    a.add_patch(Rectangle((0, -3), 13, 3, fc="#DDE9F0", alpha=0.45, zorder=-2))
    a.set(
        ylim=(-3, 3),
        yticks=[-3, -2, -1, 0, 1, 2, 3],
        ylabel="Zone heat-pump\nenergy change (%)",
    )
    a.text(
        0.97,
        0.035,
        "VAV increases;\nheat pumps decrease",
        transform=a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#626B74",
        fontstyle="italic",
    )
    b.plot([-2.5, 13], [-2.5, 13], c="#4C5157", ls="--", lw=0.8, zorder=1)
    b.set(ylim=(-2.5, 13), yticks=[0, 5, 10], xlabel="Original air supply:\nenergy change (%)", ylabel="After supply-air reset:\nenergy change (%)")
    b.text(0.85, 0.96, "Equal\nenergy effects", ha="right", va="top", transform=b.transAxes, fontsize=7)
    carriers = [
        ("heating_electric_change_mwh", "Electric heating", "#B65F69"),
        ("cooling_electric_change_mwh", "Cooling", "#648DB4"),
        ("fans_electric_change_mwh", "Fans", "#90AA90"),
        ("facility_gas_hhv_change_mwh", "Gas", "#CEB27B"),
    ]
    systems = [
        (hp, "vav", "Original VAV"),
        (common, "alternative", "VAV reset"),
        (hp, "alternative", "Zone heat pumps"),
    ]
    part_rows = []
    for y, (frame, suffix, label) in enumerate(systems):
        total = (
            frame["facility_electric_change_mwh_" + suffix]
            + frame["facility_gas_hhv_change_mwh_" + suffix]
        )
        parts = frame[[field + "_" + suffix for field, _, _ in carriers]].sum(axis=1)
        s.check(f"fig4_energy_closure_{label}", np.allclose(parts, total, atol=1e-6, rtol=1e-8))
        positive = negative = 0.0
        for field, component, paint in carriers:
            val = float(frame[field + "_" + suffix].mean())
            start = positive if val >= 0 else negative
            c.barh(y, val, left=start, color=paint, height=0.5, ec="white", lw=0.4, zorder=2)
            if abs(val) >= 1.25:
                c.text(
                    start + val / 2,
                    y,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7.2,
                    zorder=4,
                )
            if val >= 0:
                positive += val
            else:
                negative += val
            part_rows.append({"system": label, "component": component, "mean_change_mwh": val})
        value = float(total.mean())
        c.scatter(value, y, c="black", s=34, ec="white", lw=0.6, zorder=5)
        tx = max(value, positive) + 0.7
        c.text(
            tx, y, f"{value:+.2f}", va="center", ha="left", fontsize=8, fontweight="bold", zorder=5
        )
    c.axvline(0, c="#60676F", lw=0.7, ls="--")
    c.set(
        yticks=range(3),
        yticklabels=["Original\nVAV", "VAV\nreset", "Zone heat\npumps"],
        ylim=(3.05, -0.45),
        xlim=(-3.5, 24),
        xlabel="Mean annual energy change (MWh)",
    )
    c.set_xticks([0, 5, 10, 15, 20])
    c.tick_params(axis="y", length=0)
    c.legend(
        [Patch(fc=paint) for _, _, paint in carriers] + [marker_handle("black")],
        [label for _, label, _ in carriers] + ["Total"],
        loc="lower right",
        ncol=2,
        fontsize=7,
        columnspacing=0.9,
        handlelength=0.9,
    )
    dec_rows = []
    for x, system in enumerate(["vav", "pthp"]):
        local = decomp[decomp.system.eq(system)]
        s.check(f"fig4_service_support_{system}", len(local) == 105, len(local))
        bottom = 0.0
        for field, paint, label in [
            ("population_assignment_step_c", P["blue"], "Assignment"),
            ("population_trajectory_step_c", P["plum"], "Temperature"),
        ]:
            val = float(local[field].mean())
            d.bar(x, val, bottom=bottom, width=0.55, color=paint, ec="white", lw=0.4, zorder=2)
            d.text(
                x,
                bottom + val / 2,
                f"{val:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
            bottom += val
            dec_rows.append({"system": system, "step": label, "mean_gain_c": val})
        s.check(
            f"fig4_service_closure_{system}",
            abs(bottom - local.population_total_step_c.mean()) < 1e-10,
        )
        d.text(x, bottom + 0.012, f"{bottom:.3f}", ha="center", fontsize=9, fontweight="bold")
    d.set(
        xticks=[0, 1],
        xticklabels=["VAV", "Heat pumps"],
        ylim=(0, 0.4),
        yticks=[0, 0.1, 0.2, 0.3, 0.4],
        ylabel="Population mismatch\nreduction (°C)",
    )
    d.legend(
        [Patch(fc=P["blue"]), Patch(fc=P["plum"])],
        ["Assignment step", "Temperature step"],
        loc="upper left",
        fontsize=7,
        borderaxespad=0.3,
        handlelength=0.9,
    )
    # Fixed-order accounting definition is in the caption.
    s.table("Figure_04_heatpump_pairs.csv", hp)
    s.table("Figure_04_reset_pairs.csv", reset)
    s.table("Figure_04_energy_components.csv", pd.DataFrame(part_rows))
    s.table("Figure_04_service_decomposition.csv", pd.DataFrame(dec_rows))
    f._hem_meta.update(
        common_pairs=105,
        reset_pairs=108,
        comparison="arranged minus mixed, within each system",
        decomposition="order-dependent accounting steps; not causal effects",
    )
    return f


def figure_five(s: Study):
    pairs = s.csv("PAIRED_EVENT_INFORMATION_REVIEW.csv")
    scores = s.csv("FULL_REPLAY_ACTION_SCORES.csv")
    draws = s.csv("PAIRED_FROZEN_POWER_DRAWS.csv")
    pairs = pairs[
        pairs.control.eq("building_search")
        & pairs.normal_energy_reference_factor.eq(1.0)
        & pairs.policy.eq("maximum_power")
    ].copy()
    s.check(
        "fig5_all_events",
        len(pairs) == 432 and pairs.normal_eligible_engineering_resolved_support.all(),
        len(pairs),
    )
    pairs["checked_mean_kw"] = np.where(
        pairs.mean_selected_full_service_qualified.eq(True), pairs.mean_available_power_kw, 0.0
    )
    climates = s.climates()
    fields = ["mean_available_power_kw", "checked_mean_kw", "full_available_power_kw"]
    summary = pairs.groupby("climate")[fields].mean().loc[climates]
    f = page(5)
    a = axes(
        f,
        [0.235, 0.535, 0.725, 0.39],
        "a",
        "Individual demands change response estimates",
        "x",
        letter_x=-0.29,
    )
    a._left_title.set_x((0.09 - 0.235) / 0.725)
    b = axes(f, [0.09, 0.105, 0.24, 0.33], "b", "Service behind\npower estimates", letter_x=-0.30)
    c = axes(
        f, [0.435, 0.105, 0.20, 0.33], "c", "Service across\nresponse windows", "x", letter_x=-0.39
    )
    d = axes(f, [0.755, 0.105, 0.205, 0.33], "d", "Demand error and\nusable power", letter_x=-0.40)
    for i, row in enumerate(summary.itertuples()):
        if i % 2 == 1:
            a.axhspan(i - 0.5, i + 0.5, fc="#F5F6F7", zorder=-2)
        a.plot(
            [row.checked_mean_kw, row.mean_available_power_kw],
            [i, i],
            c="#969DA4",
            lw=0.8,
            zorder=1,
        )
    a.scatter(
        summary.mean_available_power_kw,
        range(9),
        marker="o",
        s=90,
        c=P["gold"],
        ec="#4F4840",
        lw=0.6,
        zorder=4,
    )
    # Small categorical y offsets distinguish nearly identical values without changing x.
    a.scatter(
        summary.checked_mean_kw,
        np.arange(9) - 0.16,
        marker="s",
        s=60,
        c=P["coral"],
        ec="#595052",
        lw=0.6,
        zorder=5,
    )
    a.scatter(
        summary.full_available_power_kw,
        np.arange(9) + 0.16,
        marker="D",
        s=64,
        c=P["teal"],
        ec=P["teal"],
        lw=1.2,
        zorder=6,
    )
    a.set(
        yticks=range(9),
        yticklabels=[DISPLAY.get(x, x) for x in climates],
        ylim=(8.55, -0.6),
        xlim=(-0.7, 29),
        xticks=[0, 6, 12, 18, 24],
        xlabel="Mean event-power reduction (kW)",
    )
    a.tick_params(axis="y", length=0, labelleft=False)
    geography = s.json("ne_110m_observed_countries.json")["features"]
    land = s.json("ne_110m_land.geojson")["features"]
    context = {r["climate"]: r for r in s.json("GLOBAL_CLIMATES.json")["climates"]}
    country_keys = {
        "United States": "usa",
        "Canada": "canada",
        "Sweden": "sweden",
        "India": "india",
        "China": "china",
        "Tunisia": "tunisia",
        "Australia": "australia",
        "Brazil": "brazil",
    }
    for i, climate in enumerate(climates):
        row = context[climate]
        yp = 0.535 + 0.39 * (8.55 - i) / 9.15
        f.text(0.137, yp, DISPLAY.get(climate, climate), ha="right", va="center")
        mini = f.add_axes([0.155, yp - 0.018, 0.055, 0.036])
        rings = []
        for feature in geography:
            if feature["country_key"] == country_keys.get(row["country"]):
                rings = feature["rings"]
                break
        if rings:
            # Use the mainland component containing the named city context.
            rings = sorted(
                rings,
                key=lambda r: np.ptp(np.asarray(r)[:, 0]) * np.ptp(np.asarray(r)[:, 1]),
                reverse=True,
            )[:1]
            for ring in rings:
                mini.add_patch(Polygon(ring, fc="#B9BDC2", ec="none"))
            xy = np.vstack(rings)
            lo = xy.min(axis=0)
            hi = xy.max(axis=0)
            mini.set_xlim(lo[0] - 1, hi[0] + 1)
            mini.set_ylim(lo[1] - 1, hi[1] + 1)
        else:
            # Singapore lies below the resolution of this country layer: show
            # the actual surrounding coastline with a city locator instead.
            for feature in land:
                geometry = feature["geometry"]
                polys = (
                    [geometry["coordinates"]]
                    if geometry["type"] == "Polygon"
                    else geometry["coordinates"]
                )
                for poly in polys:
                    mini.add_patch(Polygon(poly[0], fc="#B9BDC2", ec="none"))
            mini.set_xlim(98, 109)
            mini.set_ylim(-3, 8)
        mini.scatter(row["longitude"], row["latitude"], s=5, c=P["teal"], ec="white", lw=0.25)
        mini.set_aspect("equal")
        mini.axis("off")
    a.legend(
        [marker_handle(P["gold"]), marker_handle(P["coral"], "s"), marker_handle(P["teal"], "D")],
        ["Zone-mean selection and scoring", "Zone-mean selection, full-demand check", "Full-demand selection and scoring"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7.1,
        borderaxespad=0.3,
    )
    selected = pairs[pairs.mean_selected_action.notna()].merge(
        scores,
        left_on=["climate", "composition_id", "control", "event_id", "mean_selected_action"],
        right_on=["climate", "composition_id", "control", "event_id", "action_id"],
        validate="one_to_one",
    )
    s.check("fig5_selected_actions", len(selected) == 423, len(selected))
    excess_fields = [
        "mean_selected_full_baseline_window_excess_c",
        "mean_selected_full_action_precondition_excess_c",
        "mean_selected_full_action_event_excess_c",
        "mean_selected_full_action_recovery_excess_c",
    ]
    selected["display_maximum_service_excess_c"] = selected[excess_fields].max(axis=1)
    b.scatter(
        selected.mean_available_power_kw,
        selected.display_maximum_service_excess_c,
        c=P["plum"],
        s=20,
        alpha=0.38,
        lw=0,
        zorder=3,
    )
    b.axhline(0, c="#777D83", ls="--", lw=0.65)
    b.set(
        ylim=(-0.035, 1.02),
        xlim=(-1, 56),
        xticks=[0, 20, 40],
        yticks=[0, 0.3, 0.6, 0.9],
        xlabel="Event-power\nreduction (kW)",
        ylabel="Service excess (°C)",
    )
    b.text(
        0.95,
        0.95,
        "$n = 423$\nselected actions",
        transform=b.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
    )
    common = selected[selected.common_baseline_window_support]
    s.check("fig5_common_baseline", len(common) == 376, len(common))
    stage_rows = []
    for j, (metric, paint, label) in enumerate(
        [
            ("population_mean_mismatch_c", P["blue"], "Population"),
            ("fixed_tail_mean_mismatch_c", P["coral"], "Least-served"),
        ]
    ):
        vals = []
        for window in ["precondition", "event", "recovery"]:
            obs = common["action_" + window + "_" + metric]
            ref = common["reference_" + window + "_" + metric]
            evaluated = np.isfinite(obs) & np.isfinite(ref)
            val = int((evaluated & obs.gt(ref + 1e-9)).sum())
            vals.append(val)
            stage_rows.append(
                {"window": window, "metric": label, "count": val, "evaluated": int(evaluated.sum())}
            )
        y = np.arange(3) + (j - 0.5) * 0.36
        c.barh(y, vals, height=0.29, color=paint, zorder=2)
        for yy, val in zip(y, vals):
            c.text(val + 7, yy, str(val), va="center", fontsize=7.6)
    c.set(
        yticks=range(3),
        yticklabels=["Before", "Event", "Recovery"],
        ylim=(2.55, -1.35),
        xlim=(0, 315),
        xticks=[0, 150, 300],
        xlabel="Events outside\nservice limits",
    )
    c.tick_params(axis="y", length=0)
    c.legend(
        [Patch(fc=P["blue"]), Patch(fc=P["coral"])],
        ["Population", "Least-served"],
        loc="upper left",
        fontsize=6.9,
        borderaxespad=0.25,
        handlelength=0.8,
    )
    local = draws[
        draws.control.eq("building_search")
        & draws.budget.eq(1.0)
        & draws.support.eq("all_normal_available")
    ]
    dr = []
    for prefix, ls, source in [
        ("sourceSD", "-", "Source-shared"),
        ("studyGroupSD", "--", "Study-shared"),
    ]:
        for field, paint, name in [
            ("mean_selection_full_checked_power_kw", P["coral"], "Zone-mean selection, full-demand check"),
            ("full_selection_full_checked_power_kw", P["teal"], "Full-demand selection and scoring"),
        ]:
            means = []
            lo = []
            hi = []
            for val in [0, 0.25, 0.5]:
                vv = local[local.scenario.eq(f"{prefix}{val:g}_personSD0.5")][field]
                s.check(f"fig5_draws_{prefix}_{field}_{val}", len(vv) == 256, len(vv))
                means.append(vv.mean())
                lo.append(vv.quantile(0.05))
                hi.append(vv.quantile(0.95))
                dr.append(
                    {
                        "shared_error": source,
                        "selection": name,
                        "sd_c": val,
                        "mean_kw": vv.mean(),
                        "q05_kw": vv.quantile(0.05),
                        "q95_kw": vv.quantile(0.95),
                        "draws": len(vv),
                    }
                )
            d.fill_between([0, 0.25, 0.5], lo, hi, color=paint, alpha=0.10, lw=0)
            d.plot([0, 0.25, 0.5], means, c=paint, ls=ls, lw=1.2, zorder=3)
    d.set(
        xlim=(-0.015, 0.515),
        xticks=[0, 0.25, 0.5],
        xlabel="Shared error SD (°C)",
        ylabel="Service-preserving\npower (kW)",
    )
    d.yaxis.set_major_locator(MaxNLocator(4))
    d.set_ylim(0.8, max(d.get_ylim()[1], 3.1))
    d.legend(
        [
            Line2D([], [], c=P["coral"], lw=1.5),
            Line2D([], [], c=P["teal"], lw=1.5),
            Line2D([], [], c="black", ls="-", lw=1),
            Line2D([], [], c="black", ls="--", lw=1),
        ],
        ["Zone-mean selection", "Full-demand selection", "Source-shared", "Study-shared"],
        loc="lower left",
        ncol=1,
        labelspacing=0.3,
        columnspacing=1.2,
    )
    # The common action menu and event denominator are stated in the caption.
    # Bands are defined in the caption.
    s.table("Figure_05_climate_means.csv", summary.reset_index())
    s.table("Figure_05_selected_actions.csv", selected)
    s.table("Figure_05_stage_counts.csv", pd.DataFrame(stage_rows))
    s.table("Figure_05_uncertainty.csv", pd.DataFrame(dr))
    means = pairs[fields].mean().to_dict()
    s.check(
        "fig5_published_means",
        np.allclose([means[x] for x in fields], [11.532, 2.494, 2.514], atol=0.001),
        means,
    )
    f._hem_meta.update(
        all_events=432,
        mean_selected=423,
        common_baseline=376,
        all_event_means=means,
        error_bands="5th–95th draw percentiles; not confidence intervals",
        selection="finite-menu perfect-foresight capability",
    )
    return f


def native_response_curves(s: Study, example: pd.Series):
    plan = s.csv("SINGLE_EVENT_ACTION_PLAN.csv")
    normals = s.csv("NORMAL_OPERATION_REPLAY_COMPLETED.csv").set_index("normal_id")
    actions = s.csv("ACTION_FULL_COMPLETED.csv").set_index("run_id")
    context = plan[
        plan.climate.eq(example.climate)
        & plan.composition_id.eq(example.composition_id)
        & plan.control.eq(example.control)
        & plan.event_id.eq(example.event_id)
    ]
    curves = []
    for column, net in [
        ("selected_action", "net_full_window_electricity_reduction_kwh"),
        ("positive_net_selected_action", "positive_net_full_window_kwh"),
    ]:
        operation = context[context.action_id.eq(example[column])].iloc[0]
        normal_name = Path(
            str(normals.loc[operation.normal_id, "hourly_path"]).replace("\\", "/")
        ).name
        action_name = Path(
            str(actions.loc[operation.run_id, "hourly_path"]).replace("\\", "/")
        ).name
        with np.load(s.path(normal_name), allow_pickle=False) as file:
            baseline = dict(file)
        with np.load(s.path(action_name), allow_pickle=False) as file:
            action = dict(file)

        def stamps(a):
            return pd.DatetimeIndex(
                [
                    pd.Timestamp(2001, int(m), int(d)) + pd.Timedelta(hours=int(h) - 1)
                    for m, d, h in zip(a["month"], a["day"], a["hour"])
                ]
            )

        clock = stamps(baseline)
        if not clock.equals(stamps(action)):
            raise ValueError("Normal and action clocks differ")
        event = json.loads(operation.event_json)
        start = pd.Timestamp(event["start_timestamp"])
        begin = start - pd.Timedelta(hours=int(event["precondition_hours"]))
        end = start + pd.Timedelta(
            hours=int(event["duration_hours"]) + int(event["recovery_hours"])
        )
        mask = (clock >= begin) & (clock < end)
        delta = baseline["facility_electric_kwh"] - action["facility_electric_kwh"]
        if abs(delta[mask].sum() - example[net]) > 1e-8:
            raise ValueError("Native time-series endpoint disagrees with selected action")
        hours = (clock[mask] - start).total_seconds().to_numpy() / 3600
        curves.append((np.r_[hours[0], hours + 1], np.r_[0, np.cumsum(delta[mask])]))
    return curves, {"mode": "native_hourly_NPZ", "no_smoothing": True, "no_uncertainty_bands": True}


def figure_six(s: Study):
    choices = s.csv("ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")
    scores = s.csv("ISOLATED_EVENT_ACTION_SCORES.csv")
    choices = choices[
        choices.control.eq("building_search")
        & choices.normal_energy_reference_factor.eq(1.0)
        & choices.status.eq("selected")
    ].copy()
    s.check(
        "fig6_selected_events",
        len(choices) == 112 and choices.positive_net_selected_action.notna().all(),
        len(choices),
    )
    keys = ["climate", "composition_id", "control", "event_id"]
    lookup = scores.set_index(keys + ["action_id"], verify_integrity=True)
    negative = choices[choices.net_full_window_electricity_reduction_kwh.lt(0)].copy()
    negative["distance"] = (
        negative.net_full_window_electricity_reduction_kwh
        - negative.net_full_window_electricity_reduction_kwh.median()
    ).abs()
    example = negative.sort_values(["distance", *keys]).iloc[0]
    s.check("fig6_negative_net_events", len(negative) == 44, len(negative))
    npzs = [
        "normal_f620f82ff7dd4584423772e3.npz",
        "isolated_4caeea1e4b24d6220d74d1dc.npz",
        "isolated_0e545d46d7ce446398d31866.npz",
    ]
    if all(s.has(name) for name in npzs):
        curves, provenance = native_response_curves(s, example)
    elif s.strict_raw:
        raise FileNotFoundError(
            "Figure 6a/b needs the three original hourly NPZ files listed in INPUT_AUDIT.json"
        )
    else:
        raise FileNotFoundError("Native hourly response inputs are required")
        LOG.warning(
            "Figure 6a/b: replaying original SVG curve coordinates; hourly NPZ files are absent."
        )
        for (_, y), column in zip(
            curves, ["net_full_window_electricity_reduction_kwh", "positive_net_full_window_kwh"]
        ):
            s.check(
                f"fig6_vector_endpoint_{column}",
                abs(y[-1] - example[column]) < 2e-5,
                {"svg": y[-1], "csv": example[column]},
            )
    f = page(6)
    f._hem_meta["panel_ab"] = provenance
    a = axes(
        f, [0.12, 0.565, 0.34, 0.36], "(a)", "Event savings and recovery", None, letter_x=-0.27
    )
    b = axes(f, [0.63, 0.565, 0.34, 0.36], "(b)", "First six hours", None, letter_x=-0.28)
    c = axes(
        f, [0.12, 0.115, 0.34, 0.35], "(c)", "Power and electricity balance", None, letter_x=-0.27
    )
    d = axes(f, [0.63, 0.115, 0.34, 0.35], "(d)", "Where electricity shifts", None, letter_x=-0.28)
    for ax in [a, b]:
        ax.fill_between(
            curves[0][0], curves[0][1], curves[1][1], color=P["teal"], alpha=0.12, lw=0, zorder=1
        )
        ax.axvspan(-2, 0, fc="#DCE6EF", alpha=0.6, lw=0, zorder=-2)
        ax.axvspan(0, 2, fc="#F4E8D5", alpha=0.55, lw=0, zorder=-2)
        ax.axhline(0, c="#767B80", ls="--", lw=0.65, zorder=0)
        for (t, y), paint, label in zip(
            curves, [P["coral"], P["teal"]], ["Maximum event power", "Positive net electricity"]
        ):
            ax.plot(
                t,
                y,
                c=paint,
                lw=1.65,
                marker="o",
                markevery=4,
                ms=3.8,
                mec="white",
                mew=0.35,
                label=label,
                zorder=3,
            )
        ax.set(ylim=(-25, 55), xlabel="Hours from event start (h)")
    a.set(
        xlim=(-3, 52),
        xticks=[0, 12, 24, 36, 48],
        yticks=[-20, 0, 20, 40],
        ylabel="Cumulative electricity\nreduction (kWh)",
    )
    a.add_patch(Rectangle((-2, -23), 6, 64, fill=False, ec="#99A4AD", ls="--", lw=0.6, zorder=2))
    a.text(0.19, 0.80, "Zoom (b)", ha="left", va="center", transform=a.transAxes, fontsize=8)
    a.legend(loc="upper right", fontsize=7.1, borderaxespad=0.3, handlelength=1.2)
    for (t, y), paint in zip(curves, [P["coral"], P["teal"]]):
        mask = (t >= -2) & (t <= 4)
        b.plot(
            t[mask], y[mask], ls="", marker="o", ms=3.7, c=paint, mec="white", mew=0.35, zorder=4
        )
    b.set(
        xlim=(-2, 4),
        xticks=[-2, -1, 0, 1, 2, 3, 4],
        yticks=[-20, 0, 20, 40],
        ylabel="Cumulative electricity\nreduction (kWh)",
    )
    b.text(-1, 52, "Before\n−2 to 0 h", ha="center", va="top", fontsize=7.0)
    b.text(1, 52, "Event\n0 to 2 h", ha="center", va="top", fontsize=7.0)
    b.text(3, 52, "Recovery", ha="center", va="top", fontsize=7.0)
    power = choices.event_power_reduction_kw - choices.positive_net_event_power_kw
    net = choices.positive_net_full_window_kwh - choices.net_full_window_electricity_reduction_kwh
    s.check(
        "fig6_nested_policy_order",
        power.min() > -1e-9 and net.min() > -1e-9,
        [power.min(), net.min()],
    )
    coincident = (
        pd.DataFrame({"power_cost_kw": power.round(6), "net_gain_kwh": net.round(6)})
        .groupby(["power_cost_kw", "net_gain_kwh"])
        .size()
        .reset_index(name="cases")
    )
    s.check("fig6_bubble_count", coincident.cases.sum() == 112, int(coincident.cases.sum()))
    c.scatter(
        coincident.power_cost_kw,
        coincident.net_gain_kwh,
        s=5.0 * coincident.cases,
        c="#AD8CB7",
        alpha=0.52,
        ec="#92729C",
        lw=0.55,
        zorder=3,
    )
    c.axvline(0, c="#737A80", ls="--", lw=0.65)
    c.axhline(0, c="#737A80", ls="--", lw=0.65)
    c.set(
        xlim=(-0.23, max(power) * 1.12),
        ylim=(-1, net.max() * 1.18),
        xticks=[0, 0.5, 1, 1.5, 2],
        yticks=[0, 3, 6, 9, 12],
        xlabel="Event-power cost (kW)",
        ylabel="Net electricity gained (kWh)",
    )
    c.text(
        0.96,
        0.96,
        "$n = 112$ events",
        ha="right",
        va="top",
        transform=c.transAxes,
        fontsize=8,
    )
    hs = [
        c.scatter([], [], s=5.0 * n, c="#AD8CB7", alpha=0.52, ec="#92729C", lw=0.55)
        for n in [1, 10, 50]
    ]
    size_legend = c.legend(
        hs,
        ["1", "10", "50"],
        title="Coincident cases",
        title_fontsize=7.5,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.15),
        ncol=1,
        fontsize=9,
        columnspacing=1.1,
        handlelength=2.2,
        handletextpad=0.65,
        markerscale=1,
    )
    bubble_legend_check(s, "fig6_coincident_cases", size_legend, [1, 10, 50], 5.0)
    values = []
    part_rows = []
    for col, policy in [
        ("selected_action", "Maximum event power"),
        ("positive_net_selected_action", "Positive net electricity"),
    ]:
        rows = [
            lookup.loc[tuple(row[k] for k in keys) + (row[col],)]
            for row in choices.to_dict("records")
        ]
        frame = pd.DataFrame(rows)
        parts = np.column_stack(
            [
                -frame.preconditioning_facility_electricity_change_kwh,
                frame.event_facility_electricity_reduction_kwh,
                -frame.recovery_facility_electricity_change_kwh,
            ]
        )
        s.check(
            f"fig6_energy_closure_{policy}",
            np.allclose(
                parts.sum(axis=1),
                frame.precondition_event_recovery_facility_net_reduction_kwh,
                atol=1e-8,
                rtol=0,
            ),
        )
        value = np.r_[parts.mean(axis=0), parts.sum(axis=1).mean()]
        values.append(value)
        for stage, v in zip(["Before", "Event", "Recovery", "Net"], value):
            part_rows.append(
                {"policy": policy, "stage": stage, "mean_reduction_kwh": v, "paired_events": 112}
            )
    for j, (value, paint) in enumerate(zip(values, [P["coral"], P["teal"]])):
        yy = np.arange(4) + (j - 0.5) * 0.36
        d.barh(yy, value, height=0.29, color=paint, alpha=0.9, zorder=2)
        for y, v in zip(yy, value):
            d.text(
                v + (0.6 if v >= 0 else -0.6),
                y,
                f"{v:.2f}",
                ha="left" if v >= 0 else "right",
                va="center",
                fontsize=6.6,
            )
    for v in [0.5, 1.5, 2.5]:
        d.axhline(v, c="#E0E3E5", lw=0.45, zorder=0)
    d.plot([0, 0], [-0.6, 3.4], c="#737A80", ls="--", lw=0.65)
    d.set(
        yticks=range(4),
        yticklabels=[
            "Before\n(−2 to 0 h)",
            "Event\n(0 to 2 h)",
            "Recovery\n(2 to 50 h)",
            "Net\n(−2 to 50 h)",
        ],
        ylim=(4.8, -0.6),
        xlim=(-23, 29),
        xticks=[-20, -10, 0, 10, 20],
        xlabel="Mean electricity\nreduction (kWh)",
    )
    d.tick_params(axis="y", length=0, labelsize=7)
    d.legend(
        [Patch(fc=P["coral"]), Patch(fc=P["teal"])],
        ["Maximum event power", "Positive net electricity"],
        loc="lower right",
        ncol=1,
        fontsize=7.5,
        labelspacing=0.3,
        handlelength=1,
    )
    # Example and aggregate scopes are specified in the caption.
    s.table("Figure_06_coincident_events.csv", coincident)
    s.table("Figure_06_phase_means.csv", pd.DataFrame(part_rows))
    curve_rows = []
    for name, (t, y) in zip(["maximum_power", "positive_net"], curves):
        for tt, yy in zip(t, y):
            curve_rows.append(
                {
                    "policy": name,
                    "hour_from_event_start": tt,
                    "cumulative_electricity_reduction_kwh": yy,
                    "source_mode": provenance["mode"],
                }
            )
    s.table("Figure_06_example_curves.csv", pd.DataFrame(curve_rows))
    f._hem_meta.update(
        paired_events=112,
        negative_net_maxpower_events=44,
        example={k: str(example[k]) for k in keys},
        curve_origin=provenance,
        electricity_phase_means=part_rows,
        uncertainty_bands="not present for a single event",
    )
    return f


FIGURES = {
    1: figure_one,
    2: figure_two,
    3: figure_three,
    4: figure_four,
    5: figure_five,
    6: figure_six,
}


def export(fig, number: int, s: Study, formats: list[str], dpi: int):
    base = s.output / f"Figure_{number:02}"
    for artist in fig.findobj(mpl.collections.Collection):
        artist.set_rasterized(False)  # Keep heatmap cells and colorbars vector-editable.
    for ax in fig.axes:
        cb = getattr(ax, "_colorbar", None)
        if cb is not None and cb.solids is not None:
            cb.solids.set_edgecolor("face")  # Avoid PDF-viewer seams in vector colorbars.
            cb.solids.set_linewidth(0.05)
    for ax in fig.axes:
        for collection in ax.collections:
            if (
                isinstance(collection, mpl.collections.PathCollection)
                and len(collection.get_offsets()) > 0
            ):
                collection.set_path_effects(
                    [
                        pe.SimplePatchShadow(
                            shadow_rgbFace=(0.20, 0.24, 0.28), offset=(0.35, -0.35), alpha=0.14
                        ),
                        pe.Normal(),
                    ]
                )
        for patch in ax.patches:
            if isinstance(patch, Rectangle) and patch.get_zorder() >= 1:
                patch.set_path_effects(
                    [
                        pe.SimplePatchShadow(
                            shadow_rgbFace=(0.20, 0.24, 0.28), offset=(0.45, -0.45), alpha=0.10
                        ),
                        pe.Normal(),
                    ]
                )
        for line in ax.lines:
            if (
                line.get_linestyle() == "-"
                and line.get_linewidth() >= 1.2
                and line.get_color() not in ("black", "#333333")
            ):
                line.set_path_effects(
                    [
                        pe.Stroke(
                            linewidth=line.get_linewidth() + 0.7, foreground="white", alpha=0.8
                        ),
                        pe.Normal(),
                    ]
                )
    for text in fig.findobj(mpl.text.Text):
        text.set_fontfamily("Times New Roman")
        is_title = text.get_gid() == "panel-title" or any(
            text is ax._left_title or text is ax.title for ax in fig.axes
        )
        text.set_fontsize(10 if is_title and text.get_fontweight() == "bold" else 9)
        if text.get_color() not in ("white", "#ffffff"):
            text.set_color("black")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A single text artist keeps the panel label and heading one space apart.
    fig._hem_meta["heading_alignment"] = [
        {
            "text": heading.get_text(),
            "single_text_artist": True,
            "left_fraction": float(heading.get_transform().transform(heading.get_position())[0] / fig.bbox.width),
        }
        for heading in fig._hem_panel_headings
    ]
    same_column = {
        1: [(0, 1), (2, 3)], 2: [(0, 2), (1, 3)], 3: [(0, 1)],
        4: [(0, 2), (1, 3)], 5: [(0, 1)], 6: [(0, 2), (1, 3)],
    }
    anchors = fig._hem_meta["heading_alignment"]
    for first, second in same_column[number]:
        assert abs(anchors[first]["left_fraction"] - anchors[second]["left_fraction"]) < 1e-6
    if number in (4, 6):
        boxes = [ax.get_position().bounds for ax in fig.axes[:4]]
        assert max(b[2] for b in boxes) - min(b[2] for b in boxes) < 1e-12
        assert abs(boxes[0][0] - boxes[2][0]) < 1e-12 and abs(boxes[1][0] - boxes[3][0]) < 1e-12
        fig._hem_meta["paired_panel_axes"] = [list(b) for b in boxes]

    # Detect items crossing the page boundary. This complements manual visual checks.
    boundary = fig.bbox
    outside = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bbox = text.get_window_extent(renderer)
        if (
            bbox.x0 < boundary.x0 - 2
            or bbox.y0 < boundary.y0 - 2
            or bbox.x1 > boundary.x1 + 2
            or bbox.y1 > boundary.y1 + 2
        ):
            if text.axes is not None and text.get_clip_on():
                continue
            outside.append(text.get_text())
    fig._hem_meta["canvas_overflow_text"] = outside
    visible_text = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        if text.axes is not None and text.get_clip_on():
            continue
        # Matplotlib keeps tick artists outside the displayed limits.
        off_axis_tick = False
        for axis in fig.axes:
            for coordinate, limits in [(axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())]:
                for tick in coordinate.get_major_ticks():
                    if text in (tick.label1, tick.label2) and not min(limits) <= tick.get_loc() <= max(limits):
                        off_axis_tick = True
        if not off_axis_tick:
            visible_text.append(text)
    collisions = []
    boxes = [text.get_window_extent(renderer) for text in visible_text]
    for i, first in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            second = boxes[j]
            if min(first.x1, second.x1) - max(first.x0, second.x0) > 0.5 and min(first.y1, second.y1) - max(first.y0, second.y0) > 0.5:
                collisions.append([visible_text[i].get_text(), visible_text[j].get_text()])
    fig._hem_meta["text_collisions"] = collisions
    fig._hem_meta["typography"] = {
        "family": "Times New Roman",
        "normal_pt": 9,
        "panel_title_pt": 10,
        "panel_number_format": "(a)",
    }
    fig._hem_meta["figure_size_in"] = list(fig.get_size_inches())
    fig._hem_meta["input_sha256"] = dict(s.used)
    fig._hem_meta["notes"] = (
        "Input checksums and plotting definitions accompany each figure."
    )
    for fmt in formats:
        opts = {"dpi": dpi}
        if fmt == "pdf":
            opts["metadata"] = {
                "Title": f"HEM Figure {number}: data-faithful style recreation",
                "Creator": "HEM Matplotlib renderer",
            }
        fig.savefig(base.with_suffix("." + fmt), **opts)
    base.with_suffix(".json").write_text(
        json.dumps(_jsonable(fig._hem_meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOG.info("Figure %s saved: %s | overflow=%s", number, ", ".join(formats), outside)
    return base.with_suffix(".png")


def draw(
    number: int,
    data_root: Path | None = None,
    output: Path | None = None,
    strict_raw=True,
    formats=None,
    dpi=None,
):
    if number not in FIGURES:
        raise ValueError("Figure number must be 1–6")
    s = Study(data_root or ROOT / "Data_for_Figures", output or ROOT / "output/publication/main", strict_raw)
    f = FIGURES[number](s)
    path = export(f, number, s, formats or ["png", "svg", "pdf"], dpi or C["dpi"])
    plt.close(f)
    (s.output / "checks").mkdir(exist_ok=True)
    (s.output / "checks" / f"Figure_{number:02}.json").write_text(
        json.dumps(_jsonable(s.checks), indent=2), encoding="utf-8"
    )
    return path


def make_contact_sheet(paths: list[Path], output: Path):
    from PIL import Image, ImageOps, ImageDraw, ImageFont

    thumbs = []
    for path in paths:
        if not path.is_file():
            continue
        im = Image.open(path).convert("RGB")
        im.thumbnail((480, 600))
        tile = Image.new("RGB", (500, 640), "white")
        tile.paste(im, ((500 - im.width) // 2, 30))
        ImageDraw.Draw(tile).text((15, 9), path.stem, fill="#202327")
        thumbs.append(tile)
    if not thumbs:
        return
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 500, rows * 640), "#ECEFF1")
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % columns) * 500, (i // columns) * 640))
    sheet.save(output / "Six_Figures_Overview.png")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--figures", "--figure", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7))
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "Data_for_Figures",
        help="Data_for_Figures tree containing output/ and src/",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output/publication/main")
    parser.add_argument(
        "--strict-raw",
        action="store_true",
        default=True,
        help="Require the complete original numerical inputs",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "svg", "pdf"], default=["png", "svg", "pdf"]
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font",
        default=None,
        help="sans, serif, or an installed font name; overrides all six figures",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "plot/reference_style_config.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if args.config.is_file():
        config = json.loads(args.config.read_text(encoding="utf-8"))
        C.update(config.get("layout", {}))
        P.update(config.get("palette", {}))
    if args.font:
        C["font"] = C["figure6_font"] = args.font
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    out = []
    for i in args.figures:
        out.append(draw(i, args.data_root, args.output, args.strict_raw, args.formats, args.dpi))
    if "png" in args.formats:
        make_contact_sheet(out, args.output)
    print(f"Finished {len(out)} figures. Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
