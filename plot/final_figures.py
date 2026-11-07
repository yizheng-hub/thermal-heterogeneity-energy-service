"""Render the verified study evidence at the author's final manuscript size."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib as mpl  # noqa: E402 - initialize local imports before loading project modules

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.collections import PolyCollection  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.lines import Line2D  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.patches import Patch  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.patches import Rectangle, ConnectionPatch  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.path import Path as GeometryPath  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.ticker import MaxNLocator  # noqa: E402 - initialize local imports before loading project modules
import numpy as np  # noqa: E402 - initialize local imports before loading project modules
import pandas as pd  # noqa: E402 - initialize local imports before loading project modules

from strengthened_figures import style, title, equal_earth, export, WIDTH, FONT, TITLE  # noqa: E402 - initialize local imports before loading project modules
from strengthened_figures import BLUE, PURPLE, GREEN, CORAL, GOLD  # noqa: E402 - initialize local imports before loading project modules


CLIMATE_NAMES = {"Malmo": "Malmö", "Brasilia": "Brasília", "Delhi": "Delhi"}


def project(ring):
    vertices = np.asarray(ring)
    return np.column_stack(equal_earth(vertices[:, 0], vertices[:, 1]))


def data_asset(raw, name):
    base = ROOT / "Data_for_Figures" if raw.resolve().is_relative_to(ROOT / "Data_for_Figures") else ROOT
    return base / name


def retained_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def map_base(ax, land_path):
    payload = json.loads(land_path.read_text(encoding="utf-8"))
    rings = []
    for feature in payload["features"]:
        geometry = feature["geometry"]
        if geometry["type"] not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Unexpected land geometry")
        groups = (
            [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        )
        rings.extend(group[0] for group in groups)
    ax.add_collection(
        PolyCollection(
            [project(ring) for ring in rings],
            facecolors="#F1F1EF",
            edgecolors="#777777",
            linewidths=0.32,
            zorder=0,
        )
    )
    for longitude in [-120, -60, 0, 60, 120]:
        ax.plot(
            *equal_earth(np.full(146, longitude), np.linspace(-60, 85, 146)),
            color="#D4D4D4",
            lw=0.35,
            ls=":",
            zorder=-1,
        )
    for latitude in [-30, 0, 30, 60]:
        ax.plot(
            *equal_earth(np.linspace(-180, 180, 361), np.full(361, latitude)),
            color="#D4D4D4",
            lw=0.35,
            ls=":",
            zorder=-1,
        )
    boundary = np.vstack(
        [
            project(np.column_stack([np.linspace(-180, 180, 361), np.full(361, -60)])),
            project(np.column_stack([np.full(146, 180), np.linspace(-60, 85, 146)])),
            project(np.column_stack([np.linspace(180, -180, 361), np.full(361, 85)])),
            project(np.column_stack([np.full(146, -180), np.linspace(85, -60, 146)])),
        ]
    )
    ax.plot(*np.vstack([boundary, boundary[:1]]).T, color="black", lw=0.6, zorder=2)
    ax.set_xlim(boundary[:, 0].min(), boundary[:, 0].max())
    ax.set_ylim(boundary[:, 1].min(), boundary[:, 1].max())
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_axis_off()


def figure_one(raw, analysis):
    source = raw / "results/observation_figures"
    paths = [
        source / n
        for n in [
            "CITY_MAP_POINTS.csv",
            "COUNTRY_DEMAND_SUMMARY.csv",
            "LEAVE_ONE_SOURCE_VARIANCE.csv",
        ]
    ]
    paths += [
        raw / "results/observations/ASHRAE_NESTED_VARIANCE.csv",
        raw / "results/shared_room/SHARED_ROOM_SUMMARY.csv",
        data_asset(raw, "src/analysis/global_extension/GLOBAL_CLIMATES.json"),
        data_asset(raw, "src/thermal_heterogeneity/assets/ne_110m_observed_countries.json"),
        data_asset(raw, "src/thermal_heterogeneity/assets/ne_110m_land.geojson"),
    ]
    points, countries, sensitivity, variance, rooms = [pd.read_csv(p) for p in paths[:5]]
    climates = pd.DataFrame(json.loads(paths[5].read_text())["climates"])
    borders = json.loads(paths[6].read_text())
    assert points.record_count.sum() == countries.record_count.sum() == 33845
    fig = plt.figure(figsize=(WIDTH, 6.05))
    a = fig.add_axes([0.18, 0.625, 0.70, 0.33])
    b = fig.add_axes([0.18, 0.085, 0.34, 0.52])
    c = fig.add_axes([0.685, 0.410, 0.195, 0.195])
    d = fig.add_axes([0.685, 0.093, 0.195, 0.207])
    map_base(a, paths[7])
    a.add_collection(
        PolyCollection(
            [project(ring) for feature in borders["features"] for ring in feature["rings"]],
            facecolors="#E4DDEB",
            edgecolors="#A297AA",
            linewidths=0.32,
            zorder=1,
        )
    )
    cmap = LinearSegmentedColormap.from_list(
        "demand_center", ["#497BB8", "#8BBBC3", "#F1E7C7", "#E5B084", "#C57485"]
    )
    norm = Normalize(np.floor(points.demand_center_c.min()), np.ceil(points.demand_center_c.max()))
    scale = 125 / points.record_count.max()
    cloud = a.scatter(
        *equal_earth(points.longitude, points.latitude),
        s=scale * points.record_count,
        c=points.demand_center_c,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )
    a.scatter(
        *equal_earth(climates.longitude, climates.latitude),
        marker="*",
        s=39,
        facecolor=GOLD,
        edgecolor="black",
        linewidth=0.45,
        zorder=4,
    )
    positions = {
        "Miami": (-56, 31),
        "Montreal": (-125, 57),
        "Malmo": (45, 68),
        "Tunis": (-20, 19),
        "Delhi": (50, 7),
        "Beijing": (144, 58),
        "Singapore": (146, -2),
        "Sydney": (149, -51),
        "Brasilia": (-50, -47),
    }
    for row in climates.itertuples():
        a.annotate(
            CLIMATE_NAMES.get(row.climate, row.climate),
            equal_earth(row.longitude, row.latitude),
            xytext=equal_earth(*positions[row.climate]),
            ha="center",
            va="center",
            fontsize=FONT,
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.45},
            zorder=5,
        )
    title(a, "a", "Global thermal-demand evidence")
    a.set_title("(a) Global thermal-demand evidence", loc="left", pad=3)
    handles = [
        a.scatter([], [], s=scale * n, facecolor="#A8A8A8", edgecolor="white", linewidth=0.35)
        for n in [100, 500, 2000]
    ]
    handles += [
        Patch(facecolor="#E4DDEB", edgecolor="#A297AA", linewidth=0.32),
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            ms=np.sqrt(39),
            markerfacecolor=GOLD,
            markeredgecolor="black",
            markeredgewidth=0.45,
        ),
    ]
    a.legend(
        handles,
        ["100 records", "500 records", "2,000 records", "Data coverage", "Weather sites"],
        loc="lower left",
        bbox_to_anchor=(0.075, 0.10),
        handletextpad=0.45,
        labelspacing=0.33,
        handlelength=1.0,
        borderaxespad=0,
    )
    fig.canvas.draw()
    box = a.get_position()
    cax = fig.add_axes([box.x1 + 0.022, box.y0, 0.016, box.height])
    colorbar = fig.colorbar(cloud, cax=cax)
    colorbar.set_label("Mean demand center (°C)", labelpad=4)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cax.yaxis.set_major_locator(MaxNLocator(5))
    colorbar.outline.set_linewidth(0.5)

    countries = countries.sort_values("demand_center_median_c", ascending=False).reset_index(
        drop=True
    )
    labels = (
        countries.country.str.replace("_", " ")
        .str.title()
        .replace({"Uk": "United Kingdom", "Usa": "United States"})
    )
    y = np.arange(len(countries))
    b.hlines(
        y,
        countries.demand_center_p10_c,
        countries.demand_center_p90_c,
        color=BLUE,
        lw=3.0,
        alpha=0.65,
    )
    b.scatter(
        countries.demand_center_median_c,
        y,
        color=PURPLE,
        edgecolor="black",
        linewidth=0.3,
        s=19,
        zorder=3,
    )
    b.set_yticks(y, labels)
    b.set_ylim(len(countries) - 0.4, -1.1)
    lo, hi = (
        np.floor(countries.demand_center_p10_c.min()) - 1,
        np.ceil(countries.demand_center_p90_c.max()) + 1,
    )
    b.set_xlim(lo, hi + 3.3)
    b.set_xticks(np.arange(np.ceil(lo / 5) * 5, hi + 1, 5))
    for row in countries.itertuples():
        b.text(hi + 0.35, row.Index, f"{row.record_count:,}", va="center")
    b.text(hi + 0.35, -0.9, "n", va="center")
    b.set_xlabel("Demand center (°C)")
    title(b, "b", "Demand within countries")

    variance = variance[
        variance.weighting.eq("unit_weight") & variance.griffiths_coefficient.eq(0.5)
    ]
    order = ["country", "city", "building_id", "unit_group", "within_unit"]
    shares = variance.set_index("component").loc[order, "share"].to_numpy() * 100
    limits = sensitivity[
        sensitivity.weighting.eq("unit_weight") & sensitivity.griffiths_coefficient.eq(0.5)
    ]
    limits = limits.groupby("component").share.agg(["min", "max"]).loc[order].to_numpy() * 100
    cy = [0, 1, 2, 3.2, 4.7]
    c.barh(cy, shares, height=0.65, color=[BLUE, "#9EBECE", GREEN, PURPLE, "#CBCBCB"])
    c.errorbar(
        shares,
        cy,
        xerr=np.vstack([shares - limits[:, 0], limits[:, 1] - shares]),
        fmt="none",
        ecolor="black",
        elinewidth=0.6,
        capsize=1.8,
    )
    c.set_yticks(cy, ["Country", "City", "Building", "Within\nbuilding", "Repeated\nrecords"])
    c.set_ylim(5.3, -0.5)
    c.set_xlim(0, 78)
    c.set_xticks([0, 30, 60])
    c.set_xlabel("Variance share (%)")
    for i, value in enumerate(shares):
        c.text(limits[i, 1] + 1.2, cy[i], f"{value:.1f}", va="center")
    title(c, "c", "Variation by scale")

    rooms = rooms[rooms.minutes.eq(30)].set_index("dataset").loc[["NUS", "ENTH"]]
    left = np.zeros(2)
    for name, color in [("Same", GREEN), ("Mixed", PURPLE), ("Opposed", CORAL)]:
        width = rooms[name].to_numpy() / rooms.windows.to_numpy() * 100
        d.barh([0, 1], width, left=left, height=0.48, color=color)
        left += width
    d.set_yticks([0, 1], [f"{key}\nn = {int(row.windows)}" for key, row in rooms.iterrows()])
    d.set_ylim(1.6, -2.2)
    d.set_xlim(0, 100)
    d.set_xticks([0, 50, 100])
    d.set_xlabel("Room-time\nwindows (%)")
    d.legend(
        [Patch(facecolor=c) for c in [GREEN, PURPLE, CORAL]],
        ["Aligned", "Mixed", "Opposite"],
        loc="upper left",
        bbox_to_anchor=(0, 0.98),
        borderaxespad=0,
        handlelength=0.8,
        labelspacing=0.3,
    )
    title(d, "d", "Shared-room requests")
    return fig, paths, "abcd"


def figure_two(raw, analysis):
    paths = [
        analysis / "results/nominal_information/NOMINAL_QUALIFICATION_COUNTS.csv",
        analysis
        / "results/information_uncertainty_summary/NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv",
        analysis
        / "results/information_uncertainty_summary/SETTING_SELECTION_STABILITY_ALL_BRANCHES.csv",
        data_asset(raw, "src/analysis/global_extension/GLOBAL_CLIMATES.json"),
    ]
    counts, choices, uncertainty = [pd.read_csv(p) for p in paths[:3]]
    climates = [row["climate"] for row in json.loads(paths[3].read_text())["climates"]]
    compositions = [
        f"{source} {letter}" for source in ["Multinational", "China"] for letter in "ABC"
    ]
    data = counts.merge(
        choices, on=["climate", "composition"], validate="one_to_one", suffixes=("", "_choice")
    )
    if len(data) != 54 or not np.allclose(
        data.energy_difference_kwh, data.extra_energy_kwh, atol=1e-8, rtol=0
    ):
        raise ValueError("Annual information-effect sources disagree")
    data["row"] = data.climate.map({climate: i for i, climate in enumerate(climates)})
    data["column"] = data.composition.map(
        {composition: i for i, composition in enumerate(compositions)}
    )
    fig = plt.figure(figsize=(WIDTH, 6.0))
    a = fig.add_axes([0.16, 0.61, 0.34, 0.32])
    b = fig.add_axes([0.71, 0.61, 0.24, 0.32])
    c = fig.add_axes([0.16, 0.16, 0.32, 0.27])
    d = fig.add_axes([0.665, 0.16, 0.217, 0.27])
    cmap = LinearSegmentedColormap.from_list("service_excess", ["#F5F3EE", "#D2B7D1", "#8568A0"])
    matrix = data.pivot(
        index="climate", columns="composition", values="zone_mean_selected_full_service_excess_c"
    ).loc[climates, compositions]
    image = a.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="none")
    a.set_yticks(range(9), [CLIMATE_NAMES.get(name, name) for name in climates])
    a.set_xticks(range(6), range(1, 7))
    a.set_xlabel("Demand\ncomposition")
    a.set_xticks(np.arange(-0.5, 6), minor=True)
    a.set_yticks(np.arange(-0.5, 9), minor=True)
    a.grid(which="minor", color="white", linewidth=0.5)
    a.tick_params(which="minor", length=0)
    title(a, "a", "Service omitted by means")
    cax = fig.add_axes([0.525, 0.61, 0.015, 0.32])
    colorbar = fig.colorbar(image, cax=cax, ticks=[0, 0.5, 1])
    colorbar.set_label("Excess mismatch (°C)", labelpad=4)
    colorbar.outline.set_linewidth(0.5)
    for row in data.itertuples():
        b.scatter(
            row.energy_difference_kwh / 1000,
            row.row + (row.column - 2.5) * 0.055,
            s=18,
            color=PURPLE if row.choices_differ else "#BFBFBF",
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )
    b.set_ylim(8.5, -0.5)
    b.set_xlim(-3, 59)
    b.set_yticks(range(9), [""] * 9)
    b.set_xticks([0, 25, 50])
    b.axvline(0, color="#8D8D8D", linewidth=0.5, zorder=0)
    b.set_xlabel("Full minus mean\nchoice (MWh yr⁻¹)")
    title(b, "b", "Energy of the choices")
    fig.legend(
        [
            Line2D([], [], marker="o", ls="", color=PURPLE, ms=4),
            Line2D([], [], marker="o", ls="", color="#BFBFBF", ms=4),
        ],
        [
            f"Different choice ({int(data.choices_differ.sum())})",
            f"Same choice ({int((~data.choices_differ).sum())})",
        ],
        loc="center",
        bbox_to_anchor=(0.54, 0.495),
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.4,
    )

    x = data.zone_mean_selected_population_loss_vs_full_choice_c
    y = data.zone_mean_selected_fixed_tail_loss_vs_full_choice_c
    sizes = 12 + 1.2 * data.extra_energy_kwh / 1000
    c.axhline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
    c.axvline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
    c.scatter(
        x,
        y,
        s=sizes,
        c=np.where(data.choices_differ, PURPLE, "#BFBFBF"),
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        zorder=3,
    )
    c.set_xlim(-0.07, 1.03)
    c.set_ylim(-1.55, 1.25)
    c.set_xticks([0, 0.5, 1])
    c.set_yticks([-1, 0, 1])
    c.set_xlabel("Population mismatch\nmean minus full choice (°C)")
    c.set_ylabel("Least-served mismatch\nmean minus full choice (°C)", labelpad=5)
    title(c, "c", "Service shifts with selection")

    selected = uncertainty[
        uncertainty.library_and_scenario_design.eq("pooled_study_shared")
        & uncertainty.scenario.eq("studyGroupSD0.5_personSD0.5")
    ]
    matrix = selected.pivot(
        index="climate", columns="composition", values="conditional_choice_disagreement_fraction"
    ).loc[climates, compositions]
    if matrix.isna().any().any():
        raise ValueError("Annual paired-error setting coverage is incomplete")
    im = d.imshow(
        matrix * 100,
        cmap=LinearSegmentedColormap.from_list(
            "choice_frequency", ["#F5F3EE", "#C1D6D1", "#507F78"]
        ),
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="none",
    )
    d.set_xticks(range(6), range(1, 7))
    d.set_yticks(range(9), [CLIMATE_NAMES.get(name, name) for name in climates])
    d.set_xlabel("Demand\ncomposition")
    d.set_xticks(np.arange(-0.5, 6), minor=True)
    d.set_yticks(np.arange(-0.5, 9), minor=True)
    d.grid(which="minor", color="white", linewidth=0.5)
    d.tick_params(which="minor", length=0)
    title(d, "d", "Choices under demand error")
    cbax = fig.add_axes([0.90, 0.16, 0.015, 0.27])
    cb = fig.colorbar(im, cax=cbax, ticks=[0, 50, 100])
    cb.set_label("Changed choices (%)", labelpad=3)
    cb.outline.set_linewidth(0.5)
    handles = [
        c.scatter([], [], s=12 + 1.2 * value, color=PURPLE, edgecolor="white", linewidth=0.4)
        for value in [10, 30, 50]
    ]
    fig.legend(
        handles,
        ["10 MWh", "30 MWh", "50 MWh"],
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.54, 0.025),
        columnspacing=1.5,
        handletextpad=0.4,
    )
    return fig, paths, "abcd"


def figure_three(raw, analysis):
    paths = [
        analysis / "config/CONSTRAINED_PLACEMENT_PROFILES.csv",
        analysis / "results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv",
        analysis / "results/placement_service_frontiers/DESCRIPTIVE_BUDGET_SUMMARY.csv",
        raw / "matrix/models/MODEL_MANIFEST.json",
    ]
    profiles, effects, frontier = [pd.read_csv(path) for path in paths[:3]]
    model_info = json.loads(paths[3].read_text())["Miami"]
    model = retained_path(model_info.get("floor_geometry", model_info.get("prepared_idf")))
    if not model.resolve().is_relative_to(ROOT):
        raise ValueError("Floor geometry must remain inside the project")
    paths.append(model)
    representative = profiles[profiles.context.eq("Miami") & profiles.composition_id.eq(2026102001)]
    zones = set(representative.zone_id)
    polygons = {}
    if "floor_geometry" in model_info:
        polygons = {
            zone: np.array(vertices, dtype=float)
            for zone, vertices in json.loads(model.read_text())["floors"].items()
            if zone in zones
        }
    else:
        sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
        from global_matrix import parse_idf

        for obj in parse_idf(model.read_text()):
            if (
                obj[0].lower() == "buildingsurface:detailed"
                and obj[2].lower() == "floor"
                and obj[4] in zones
            ):
                vertices = np.array(obj[12:], dtype=float).reshape(-1, 3)
                if len(vertices) != int(obj[11]) or obj[4] in polygons:
                    raise ValueError("Unexpected native occupied-zone floor geometry")
                polygons[obj[4]] = vertices
    if set(polygons) != zones or len(zones) != 15:
        raise ValueError("The placement figure does not contain all15occupied zones")
    levels = sorted({float(v[:, 2].mean()) for v in polygons.values()})
    if len(levels) != 3:
        raise ValueError("The retained office must have three occupied floors")
    fig = plt.figure(figsize=(WIDTH, 6.1))
    a = fig.add_axes([0.14, 0.54, 0.74, 0.39])
    b = fig.add_axes([0.14, 0.16, 0.19, 0.275])
    c = fig.add_axes([0.45, 0.16, 0.19, 0.275])
    d = fig.add_axes([0.76, 0.16, 0.19, 0.275])
    color = LinearSegmentedColormap.from_list(
        "profile_center", ["#547FB0", "#9FC3C7", "#F1E8CE", "#D39AAC", "#8C6EA3"]
    )
    norm = Normalize(
        np.floor(representative.centre_c.min()), np.ceil(representative.centre_c.max())
    )
    rng = np.random.default_rng(2026103104)
    slots = {}
    first = representative[representative.layout.eq("within_floor_relocation_0pct")]
    for zone, vertices in sorted(polygons.items()):
        boundary = GeometryPath(np.vstack([vertices[:, :2], vertices[:1, :2]]), closed=True)
        count = int(first.zone_id.eq(zone).sum())
        accepted = []
        while len(accepted) < count:
            candidates = rng.uniform(
                vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0), (100, 2)
            )
            starts, ends = vertices[:, :2], np.roll(vertices[:, :2], -1, axis=0)
            edges = ends - starts
            projection = np.clip(
                ((candidates[:, None, :] - starts) * edges).sum(axis=2)
                / (edges * edges).sum(axis=1),
                0,
                1,
            )
            distances = np.linalg.norm(
                candidates[:, None, :] - starts - projection[:, :, None] * edges, axis=2
            )
            valid = boundary.contains_points(candidates) & (distances.min(axis=1) > 0.85)
            accepted.extend(candidates[valid])
        slots[zone] = np.array(accepted[:count])
    for row_index, layout in enumerate(
        ["within_floor_relocation_0pct", "within_floor_relocation_25pct"]
    ):
        group = representative[representative.layout.eq(layout)]
        if len(group) != 267:
            raise ValueError("Incomplete illustrative placement")
        for zone, vertices in polygons.items():
            floor = levels.index(float(vertices[:, 2].mean()))
            offset = np.array([floor * 59.0, (1 - row_index) * 40.0])
            polygon = vertices[:, :2] + offset
            a.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor="#F5F4F0",
                edgecolor="#777777",
                linewidth=0.5,
                zorder=0,
            )
            local = group[group.zone_id.eq(zone)].sort_values("profile_id")
            if len(local) != len(slots[zone]):
                raise ValueError("Placement changed native zone capacity")
            xy = slots[zone] + offset
            cloud = a.scatter(
                xy[:, 0],
                xy[:, 1],
                c=local.centre_c,
                cmap=color,
                norm=norm,
                s=4.0,
                edgecolors="none",
                zorder=2,
            )
    for i, label in enumerate(["Lower floor", "Middle floor", "Upper floor"]):
        a.text(i * 59 + 25, 77.0, label, ha="center", va="bottom")
    a.text(-7, 56.0, "Random\nmixing", ha="right", va="center")
    a.text(-7, 16.0, "25% move\nbudget", ha="right", va="center")
    a.plot([5, 15], [-4, -4], color="black", linewidth=1.0)
    a.text(10, -6, "10 m", ha="center", va="top")
    a.set_xlim(-2, 169)
    a.set_ylim(-10, 85)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_axis_off()
    title(a, "a", "Demand placement within the same office")
    fig.canvas.draw()
    box = a.get_position()
    cbax = fig.add_axes([box.x1 + 0.025, box.y0, 0.015, box.height])
    cb = fig.colorbar(cloud, cax=cbax, ticks=[18, 24, 30])
    cb.set_label("Demand center (°C)", labelpad=3)
    cb.outline.set_linewidth(0.5)
    for ax, field, letter, heading in [
        (b, "population_gain_c", "b", "Population\nservice gain"),
        (c, "fixed_tail_gain_c", "c", "Least-served\nservice gain"),
    ]:
        for control, paint, marker in [
            ("personal_center", BLUE, "o"),
            ("personal_range", PURPLE, "s"),
        ]:
            g = effects[effects.control.eq(control)]
            stats = g.groupby("maximum_relocation_fraction")[field].agg(
                median="median", q25=lambda v: v.quantile(0.25), q75=lambda v: v.quantile(0.75)
            )
            x = np.r_[0, stats.index.to_numpy() * 100]
            median, lower, upper = (
                np.r_[0, stats[key].to_numpy()] for key in ["median", "q25", "q75"]
            )
            ax.fill_between(x, lower, upper, color=paint, alpha=0.18, linewidth=0)
            ax.plot(x, median, color=paint, marker=marker, markersize=3, linewidth=1.0)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel("Move budget (%)")
        ax.set_ylabel("Mismatch reduction (°C)", labelpad=4)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        title(ax, letter, heading)
    for fraction, paint, line in [(0.9, GREEN, "-"), (1.0, "#999999", "--")]:
        g = frontier[
            frontier.arm.eq("vav_strong_mixed_library")
            & frontier.budget_type.eq("reference_relative")
            & frontier.reference_fraction.eq(fraction)
        ].sort_values("maximum_relocation_fraction")
        if len(g) != 4:
            raise ValueError("Common-service placement frontier is incomplete")
        d.plot(
            g.maximum_relocation_fraction * 100,
            g.feasible_settings,
            color=paint,
            marker="o",
            markersize=3,
            linewidth=1.0,
            linestyle=line,
        )
    d.set_xlim(-4, 104)
    d.set_ylim(31, 57)
    d.set_xticks([0, 50, 100])
    d.set_yticks([35, 45, 54])
    d.set_xlabel("Move budget (%)")
    d.set_ylabel("Feasible settings", labelpad=4)
    title(d, "d", "Common service\nrequirements")
    fig.legend(
        [
            Line2D([], [], color=BLUE, marker="o", ms=3),
            Line2D([], [], color=PURPLE, marker="s", ms=3),
        ],
        ["Personal center", "Personal range"],
        loc="center",
        bbox_to_anchor=(0.35, 0.035),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
        handlelength=1.0,
    )
    fig.legend(
        [
            Line2D([], [], color=GREEN, marker="o", ms=3),
            Line2D([], [], color="#999999", ls="--", marker="o", ms=3),
        ],
        ["90% reference", "Reference"],
        loc="center",
        bbox_to_anchor=(0.83, 0.035),
        ncol=1,
        labelspacing=0.25,
        handletextpad=0.4,
        handlelength=1.0,
    )
    return fig, paths, "abcd"


def figure_four(raw, analysis):
    folder = analysis / "results/placement_hvac_interaction"
    completed = []
    for receipt in sorted((analysis / "results/spatial_hvac_mechanism").glob("run_*/STATUS.json")):
        status = json.loads(receipt.read_text())
        source = receipt.parent / "DECOMPOSITION.csv"
        if status.get("decomposition", {}).get("status") == "completed" and source.is_file():
            completed.append(source)
    if not completed:
        raise ValueError("No completed spatial service decomposition in the supplied analysis root")
    if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in completed}) != 1:
        raise ValueError(
            "Completed spatial decompositions differ; resolve their analysis version before publication"
        )
    paths = [
        folder / "independent_pthp_MATCHED_CROSS_SYSTEM_EFFECTS.csv",
        folder / "vav_coordinated_outdoor_reset_MATCHED_CROSS_SYSTEM_EFFECTS.csv",
        completed[-1],
    ]
    heatpump, reset, decomposition = [pd.read_csv(p) for p in paths]
    heatpump = heatpump[heatpump.layout.eq("within_floor_relocation_25pct")]
    reset = reset[reset.layout.eq("within_floor_relocation_25pct")]
    keys = ["climate", "composition_id", "control", "layout"]
    common = heatpump[keys].merge(reset, on=keys, validate="one_to_one")
    if len(heatpump) != 105 or len(common) != 105 or len(reset) != 108:
        raise ValueError("Quarter-budget matched system denominators differ")
    fig = plt.figure(figsize=(WIDTH, 6.4))
    a = fig.add_axes([0.15, 0.60, 0.32, 0.32])
    b = fig.add_axes([0.65, 0.60, 0.30, 0.32])
    c = fig.add_axes([0.15, 0.175, 0.36, 0.27])
    d = fig.add_axes([0.65, 0.175, 0.30, 0.27])
    for ax, frame, letter, text in [
        (a, heatpump, "a", "HVAC changes energy effects"),
        (b, reset, "b", "VAV supply-air reset"),
    ]:
        ax.axhline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
        ax.axvline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
        if ax is a:
            ax.axhspan(-3.1, 0, facecolor=GREEN, alpha=0.07, zorder=-1)
        else:
            ax.plot([-3, 13], [-3, 13], color="#B5B5B5", ls="--", linewidth=0.6, zorder=0)
        for control, paint, marker in [
            ("personal_center", BLUE, "o"),
            ("personal_range", PURPLE, "s"),
        ]:
            local = frame[frame.control.eq(control)]
            ax.scatter(
                local.energy_change_pct_vav,
                local.energy_change_pct_alternative,
                c=paint,
                marker=marker,
                s=18,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.35,
                zorder=3,
            )
        ax.set_xlim(-2.5, 13)
        ax.set_xticks([0, 5, 10])
        ax.set_xlabel("Original VAV\nenergy change (%)")
        title(ax, letter, text)
    a.set_ylim(-3.1, 3.1)
    a.set_yticks([-2, 0, 2])
    a.set_ylabel("Zone heat pumps\nenergy change (%)", labelpad=4)
    b.set_ylim(-2.5, 13)
    b.set_yticks([0, 5, 10])
    b.set_ylabel("VAV with supply-air reset\nenergy change (%)", labelpad=4)
    fig.legend(
        [
            Line2D([], [], marker="o", ls="", color=BLUE, ms=4),
            Line2D([], [], marker="s", ls="", color=PURPLE, ms=4),
        ],
        ["Personal center", "Personal range"],
        loc="center",
        bbox_to_anchor=(0.54, 0.51),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.4,
    )
    carriers = [
        ("heating_electric_change_mwh", "Electric heating", "#D89987"),
        ("cooling_electric_change_mwh", "Cooling", "#8EB8C9"),
        ("fans_electric_change_mwh", "Fans", "#BFBFC7"),
        ("facility_gas_hhv_change_mwh", "Gas", "#9EBBAA"),
    ]
    systems = [
        (heatpump, "vav", "Original VAV"),
        (common, "alternative", "VAV reset"),
        (heatpump, "alternative", "Zone heat\npumps"),
    ]
    for y, (frame, suffix, label) in enumerate(systems):
        contributions = frame[[field + "_" + suffix for field, _, _ in carriers]].sum(axis=1)
        total = (
            frame["facility_electric_change_mwh_" + suffix]
            + frame["facility_gas_hhv_change_mwh_" + suffix]
        )
        if not np.allclose(contributions, total, rtol=1e-8, atol=1e-6):
            raise ValueError("End-use contributions do not close to the system energy change")
        positive, negative = 0.0, 0.0
        for field, _, paint in carriers:
            value = frame[field + "_" + suffix].mean()
            c.barh(
                y,
                value,
                left=positive if value >= 0 else negative,
                height=0.50,
                color=paint,
                edgecolor="white",
                linewidth=0.4,
            )
            if value >= 0:
                positive += value
            else:
                negative += value
        c.plot(total.mean(), y, marker="|", color="black", markersize=10, markeredgewidth=1.0)
    c.axvline(0, color="#777777", linewidth=0.5)
    c.set_yticks(range(3), [s[2] for s in systems])
    c.set_ylim(2.65, -0.65)
    c.xaxis.set_major_locator(MaxNLocator(4))
    c.set_xlabel("Mean annual change (MWh)")
    title(c, "c", "End uses behind the change")

    labels = ["Assignment", "Temperature"]
    colors = [GOLD, "#9D8DB6"]
    for x, system in enumerate(["vav", "pthp"]):
        local = decomposition[decomposition.system.eq(system)]
        if len(local) != 105:
            raise ValueError("Service-accounting hardware scope differs")
        bottom = 0.0
        for field, paint in zip(
            ["population_assignment_step_c", "population_trajectory_step_c"], colors
        ):
            value = local[field].mean()
            d.bar(
                x, value, bottom=bottom, color=paint, width=0.55, edgecolor="white", linewidth=0.4
            )
            bottom += value
        if abs(bottom - local.population_total_step_c.mean()) > 1e-10:
            raise ValueError("Fixed-order service accounting does not close")
    d.set_xticks([0, 1], ["VAV", "Heat\npumps"])
    d.set_ylim(0, 0.4)
    d.set_yticks([0, 0.2, 0.4])
    d.set_ylabel("Population mismatch\nreduction (°C)", labelpad=4)
    title(d, "d", "Sources of service gain")
    fig.legend(
        [Patch(facecolor=paint) for _, _, paint in carriers],
        [label for _, label, _ in carriers],
        loc="center",
        bbox_to_anchor=(0.34, 0.055),
        ncol=2,
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.3,
    )
    fig.legend(
        [Patch(facecolor=paint) for paint in colors],
        labels,
        loc="center",
        bbox_to_anchor=(0.83, 0.055),
        ncol=1,
        handlelength=0.9,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    return fig, paths, "abcd"


def figure_five(raw, analysis):
    paths = [
        analysis / "results/dr_information_review/PAIRED_EVENT_INFORMATION_REVIEW.csv",
        analysis / "results/dr_information_review/FULL_REPLAY_ACTION_SCORES.csv",
        analysis / "results/dr_information_uncertainty/PAIRED_FROZEN_POWER_DRAWS.csv",
        data_asset(raw, "src/analysis/global_extension/GLOBAL_CLIMATES.json"),
    ]
    pairs, scores, draws = [pd.read_csv(p) for p in paths[:3]]
    pairs = pairs[
        pairs.control.eq("building_search")
        & pairs.normal_energy_reference_factor.eq(1.0)
        & pairs.policy.eq("maximum_power")
    ].copy()
    if len(pairs) != 432 or not pairs.normal_eligible_engineering_resolved_support.all():
        raise ValueError("The primary response information inventory is incomplete")
    pairs["checked_mean_kw"] = np.where(
        pairs.mean_selected_full_service_qualified.eq(True), pairs.mean_available_power_kw, 0.0
    )
    climates = [row["climate"] for row in json.loads(paths[3].read_text())["climates"]]
    fig = plt.figure(figsize=(WIDTH, 6.4))
    a = fig.add_axes([0.16, 0.63, 0.73, 0.30])
    b = fig.add_axes([0.15, 0.16, 0.18, 0.25])
    c = fig.add_axes([0.46, 0.16, 0.18, 0.25])
    d = fig.add_axes([0.77, 0.16, 0.18, 0.25])
    summary = (
        pairs.groupby("climate")[
            ["mean_available_power_kw", "checked_mean_kw", "full_available_power_kw"]
        ]
        .mean()
        .loc[climates]
    )
    for i, row in enumerate(summary.itertuples()):
        a.plot(
            [row.mean_available_power_kw, row.checked_mean_kw],
            [i, i],
            color="#C6C6C6",
            lw=1.0,
            zorder=0,
        )
    for field, paint, marker in [
        ("mean_available_power_kw", GOLD, "o"),
        ("checked_mean_kw", CORAL, "s"),
        ("full_available_power_kw", GREEN, "D"),
    ]:
        a.scatter(
            summary[field],
            range(9),
            color=paint,
            marker=marker,
            s=23,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    a.set_yticks(range(9), [CLIMATE_NAMES.get(name, name) for name in climates])
    a.set_ylim(8.5, -0.5)
    a.set_xlim(-0.6, summary.to_numpy().max() * 1.06)
    a.xaxis.set_major_locator(MaxNLocator(5))
    a.set_xlabel("Mean event-power reduction (kW)")
    title(a, "a", "Individual demands change response estimates")
    fig.legend(
        [
            Line2D([], [], marker=marker, ls="", color=paint, ms=4)
            for paint, marker in [(GOLD, "o"), (CORAL, "s"), (GREEN, "D")]
        ],
        ["Mean-demand estimate", "Mean choice retaining service", "Individual-demand choice"],
        loc="center",
        bbox_to_anchor=(0.53, 0.515),
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    selected = pairs[pairs.mean_selected_action.notna()].merge(
        scores,
        left_on=["climate", "composition_id", "control", "event_id", "mean_selected_action"],
        right_on=["climate", "composition_id", "control", "event_id", "action_id"],
        validate="one_to_one",
    )
    fields = [
        "mean_selected_full_baseline_window_excess_c",
        "mean_selected_full_action_precondition_excess_c",
        "mean_selected_full_action_event_excess_c",
        "mean_selected_full_action_recovery_excess_c",
    ]
    maximum = selected[fields].max(axis=1)
    b.scatter(
        selected.mean_available_power_kw, maximum, s=8, color=PURPLE, alpha=0.38, linewidths=0
    )
    b.axhline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
    b.set_ylim(-0.06, maximum.max() * 1.08)
    b.xaxis.set_major_locator(MaxNLocator(3))
    b.yaxis.set_major_locator(MaxNLocator(4))
    b.set_xlabel("Event-power\nreduction (kW)")
    b.set_ylabel("Service excess (°C)", labelpad=4)
    title(b, "b", "Service behind\npower estimates")
    common = selected[selected.common_baseline_window_support]
    if len(common) != 376:
        raise ValueError("The common-baseline selected-action support changed")
    for j, (metric, paint) in enumerate(
        [("population_mean_mismatch_c", "#A8ADB4"), ("fixed_tail_mean_mismatch_c", PURPLE)]
    ):
        values = []
        for window in ["precondition", "event", "recovery"]:
            observed, reference = (
                common["action_" + window + "_" + metric],
                common["reference_" + window + "_" + metric],
            )
            evaluated = np.isfinite(observed) & np.isfinite(reference)
            values.append(int((evaluated & observed.gt(reference + 1e-9)).sum()))
        y = np.arange(3) + (j - 0.5) * 0.42
        c.barh(y, values, height=0.34, color=paint)
        for position, value in zip(y, values):
            c.text(value + 6, position, str(value), va="center")
    c.set_yticks(range(3), ["Before", "Event", "Recovery"])
    c.set_ylim(2.6, -1.6)
    c.set_xlim(0, 320)
    c.set_xticks([0, 150, 300])
    c.set_xlabel("Events outside\nservice limits")
    c.legend(
        [Patch(facecolor=paint) for paint in ["#A8ADB4", PURPLE]],
        ["Population", "Least-served"],
        loc="upper left",
        bbox_to_anchor=(0, 0.99),
        borderaxespad=0,
        handlelength=0.7,
        labelspacing=0.25,
    )
    title(c, "c", "Service across\nresponse windows")
    local = draws[
        draws.control.eq("building_search")
        & draws.budget.eq(1.0)
        & draws.support.eq("all_normal_available")
    ]
    for prefix, line in [("sourceSD", "-"), ("studyGroupSD", "--")]:
        for field, paint in [
            ("mean_selection_full_checked_power_kw", CORAL),
            ("full_selection_full_checked_power_kw", GREEN),
        ]:
            medians, low, high = [], [], []
            for value in [0, 0.25, 0.5]:
                v = local[local.scenario.eq(f"{prefix}{value:g}_personSD0.5")][field]
                if len(v) != 256:
                    raise ValueError("The paired response-error draws are incomplete")
                medians.append(v.mean())
                low.append(v.quantile(0.05))
                high.append(v.quantile(0.95))
            d.fill_between([0, 0.25, 0.5], low, high, color=paint, alpha=0.07, linewidth=0)
            d.plot([0, 0.25, 0.5], medians, color=paint, ls=line, lw=1.0)
    d.set_xticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
    d.yaxis.set_major_locator(MaxNLocator(4))
    d.set_xlabel("Shared error\nSD (°C)")
    d.set_ylabel("Service-preserving\npower (kW)", labelpad=4)
    title(d, "d", "Demand error and\nusable power")
    fig.legend(
        [Line2D([], [], color="black", ls=line, lw=0.8) for line in ["-", "--"]],
        ["Source-shared", "Study-shared"],
        loc="center",
        bbox_to_anchor=(0.83, 0.035),
        ncol=1,
        labelspacing=0.2,
        handlelength=1.1,
        handletextpad=0.4,
    )
    return fig, paths, "abcd"


def figure_six(raw, analysis):
    folder = analysis / "results/dr_capability"
    paths = [
        folder / name
        for name in [
            "ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv",
            "ISOLATED_EVENT_ACTION_SCORES.csv",
            "SINGLE_EVENT_ACTION_PLAN.csv",
            "NORMAL_OPERATION_REPLAY_COMPLETED.csv",
            "ACTION_FULL_COMPLETED.csv",
        ]
    ]
    choices, scores, plan, normals, actions = [pd.read_csv(path) for path in paths]
    choices = choices[
        choices.control.eq("building_search")
        & choices.normal_energy_reference_factor.eq(1.0)
        & choices.status.eq("selected")
    ].copy()
    if len(choices) != 112 or choices.positive_net_selected_action.isna().any():
        raise ValueError("Response-policy paired support is incomplete")
    keys = ["climate", "composition_id", "control", "event_id"]
    lookup = scores.set_index(keys + ["action_id"], verify_integrity=True)
    negative = choices[choices.net_full_window_electricity_reduction_kwh.lt(0)].copy()
    negative["distance"] = (
        negative.net_full_window_electricity_reduction_kwh
        - negative.net_full_window_electricity_reduction_kwh.median()
    ).abs()
    example = negative.sort_values(["distance", *keys]).iloc[0]
    context = plan[
        (plan.climate.eq(example.climate))
        & plan.composition_id.eq(example.composition_id)
        & plan.control.eq(example.control)
        & plan.event_id.eq(example.event_id)
    ]
    normals, actions = normals.set_index("normal_id"), actions.set_index("run_id")
    sys.path.insert(0, str(ROOT / "src/analysis/global_extension"))
    from multi_event_dr_extension import compact_timestamps, _window_masks

    curves = []
    for column, net_column in [
        ("selected_action", "net_full_window_electricity_reduction_kwh"),
        ("positive_net_selected_action", "positive_net_full_window_kwh"),
    ]:
        operation = context[context.action_id.eq(example[column])].iloc[0]
        normal_path = retained_path(normals.loc[operation.normal_id, "hourly_path"])
        action_path = retained_path(actions.loc[operation.run_id, "hourly_path"])
        for path in [normal_path, action_path]:
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError("Illustrative response trajectory is outside the project")
            paths.append(path)
        with np.load(normal_path, allow_pickle=False) as saved:
            baseline = dict(saved)
        with np.load(action_path, allow_pickle=False) as saved:
            action = dict(saved)
        event = json.loads(operation.event_json)
        start = pd.Timestamp(event["start_timestamp"])
        masks = _window_masks(
            baseline,
            start,
            int(event["duration_hours"]),
            int(event["recovery_hours"]),
            int(event["precondition_hours"]),
        )
        window = np.logical_or.reduce(masks)
        delta = baseline["facility_electric_kwh"] - action["facility_electric_kwh"]
        if abs(delta[window].sum() - example[net_column]) > 1e-8:
            raise ValueError(
                "Illustrative cumulative electricity does not reproduce the selected operation"
            )
        hours = (compact_timestamps(baseline)[window] - start).total_seconds().to_numpy() / 3600
        curves.append((np.r_[hours[0], hours + 1], np.r_[0, np.cumsum(delta[window])]))
    fig = plt.figure(figsize=(WIDTH, 5.9))
    a = fig.add_axes([0.16, 0.61, 0.43, 0.31])
    b = fig.add_axes([0.74, 0.61, 0.21, 0.31])
    c = fig.add_axes([0.16, 0.145, 0.32, 0.28])
    d = fig.add_axes([0.65, 0.145, 0.30, 0.28])
    ymin = min(v.min() for _, v in curves) - 5
    ymax = max(v.max() for _, v in curves) + 5
    for ax, letter, heading in [
        (a, "a", "Event savings and recovery"),
        (b, "b", "First six hours"),
    ]:
        ax.axvspan(-2, 0, color=GOLD, alpha=0.12, linewidth=0, zorder=-1)
        ax.axvspan(0, 2, color=BLUE, alpha=0.10, linewidth=0, zorder=-1)
        ax.axhline(0, color="#AAAAAA", linewidth=0.5, zorder=0)
        for (hours, cumulative), paint in zip(curves, [CORAL, GREEN]):
            ax.plot(hours, cumulative, color=paint, linewidth=1.2)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Hours from event start")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        title(ax, letter, heading)
    a.set_xlim(-3, 52)
    a.set_xticks([0, 12, 24, 36, 48])
    a.set_ylabel("Cumulative electricity\nreduction (kWh)", labelpad=4)
    b.set_xlim(-2, 4)
    b.set_xticks([-2, 0, 2, 4])
    b.set_yticks([-20, 0, 20])
    rectangle_low = min(v[np.asarray(t) <= 4].min() for t, v in curves) - 3
    rectangle_high = max(v[np.asarray(t) <= 4].max() for t, v in curves) + 1
    a.add_patch(
        Rectangle(
            (-2, rectangle_low),
            6,
            rectangle_high - rectangle_low,
            fill=False,
            edgecolor="#A4A4A4",
            linewidth=0.55,
        )
    )
    for level in [rectangle_low, rectangle_high]:
        fig.add_artist(
            ConnectionPatch(
                (4, level),
                (-2, level),
                coordsA="data",
                coordsB="data",
                axesA=a,
                axesB=b,
                color="#A4A4A4",
                linewidth=0.45,
                zorder=2.5,
            )
        )
    fig.legend(
        [Line2D([], [], color=paint, lw=1.2) for paint in [CORAL, GREEN]],
        ["Maximum event power", "Positive net electricity"],
        loc="center",
        bbox_to_anchor=(0.54, 0.51),
        ncol=2,
        columnspacing=1.3,
        handletextpad=0.5,
        handlelength=1.5,
    )
    power_cost = choices.event_power_reduction_kw - choices.positive_net_event_power_kw
    net_gain = (
        choices.positive_net_full_window_kwh - choices.net_full_window_electricity_reduction_kwh
    )
    if power_cost.min() < -1e-9 or net_gain.min() < -1e-9:
        raise ValueError("Nested response-policy comparison is inconsistent")
    coincident = (
        pd.DataFrame({"power": power_cost.round(6), "net": net_gain.round(6)})
        .groupby(["power", "net"])
        .size()
        .reset_index(name="cases")
    )
    if coincident.cases.sum() != len(choices):
        raise ValueError("Coincident response-policy cases were lost")
    c.scatter(
        coincident.power,
        coincident.net,
        color=PURPLE,
        s=3 * coincident.cases,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
    )
    c.axhline(0, color="#AAAAAA", lw=0.5, zorder=0)
    c.axvline(0, color="#AAAAAA", lw=0.5, zorder=0)
    c.set_xlim(-0.3, power_cost.max() * 1.1)
    c.set_ylim(-1.0, net_gain.max() * 1.12)
    c.xaxis.set_major_locator(MaxNLocator(4))
    c.yaxis.set_major_locator(MaxNLocator(4))
    c.set_xlabel("Event-power cost (kW)")
    c.set_ylabel("Net electricity gained (kWh)", labelpad=4)
    title(c, "c", "Power and electricity balance")
    handles = [
        c.scatter([], [], s=3 * n, color=PURPLE, alpha=0.8, edgecolor="white", linewidth=0.3)
        for n in [1, 10, 50]
    ]
    fig.legend(
        handles,
        ["1 case", "10 cases", "50 cases"],
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.34, 0.025),
        columnspacing=0.6,
        handletextpad=0.3,
        handlelength=1.0,
    )
    values = []
    for column in ["selected_action", "positive_net_selected_action"]:
        rows = [
            lookup.loc[tuple(row[k] for k in keys) + (row[column],)]
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
        if not np.allclose(
            parts.sum(axis=1),
            frame.precondition_event_recovery_facility_net_reduction_kwh,
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Policy electricity components do not close")
        values.append(np.r_[parts.mean(axis=0), parts.sum(axis=1).mean()])
    for j, (value, paint) in enumerate(zip(values, [CORAL, GREEN])):
        d.barh(np.arange(4) + (j - 0.5) * 0.38, value, height=0.30, color=paint)
    d.axvline(0, color="#777777", linewidth=0.5)
    d.set_yticks(range(4), ["Before", "Event", "Recovery", "Net"])
    d.set_ylim(3.7, -0.7)
    d.xaxis.set_major_locator(MaxNLocator(4))
    d.set_xlabel("Mean electricity\nreduction (kWh)")
    title(d, "d", "Where electricity shifts")
    fig._hem_example = {k: str(example[k]) for k in keys}
    fig._hem_example.update(
        event_start=str(start),
        selection_rule="Median negative-net maximum-power event, tie-broken by declared case keys",
        paired_events=len(choices),
        changed_actions=int(choices.selected_action.ne(choices.positive_net_selected_action).sum()),
    )
    return fig, paths, "abcd"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "output/from_raw")
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "output/strengthening")
    parser.add_argument("--output-root", type=Path, default=ROOT / "output/publication/main")
    parser.add_argument(
        "--figure", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4, 5, 6]
    )
    args = parser.parse_args(argv)
    raw, analysis, output = [
        p.resolve() for p in [args.raw_root, args.analysis_root, args.output_root]
    ]
    if not all(p.is_relative_to(ROOT) for p in [raw, analysis, output]):
        raise ValueError("Publication input and output must remain inside the project")
    fonts = style()
    for number in args.figure:
        fig, paths, letters = {
            1: figure_one,
            2: figure_two,
            3: figure_three,
            4: figure_four,
            5: figure_five,
            6: figure_six,
        }[number](raw, analysis)
        example = getattr(fig, "_hem_example", None)
        for axis in fig.axes:
            if not axis.axison:
                continue
            for coordinate, limits in [
                (axis.xaxis, axis.get_xlim()),
                (axis.yaxis, axis.get_ylim()),
            ]:
                positions = coordinate.get_majorticklocs()
                inside = (positions >= min(limits) - 1e-10) & (positions <= max(limits) + 1e-10)
                if not inside.all():
                    coordinate.set_ticks(positions[inside])
        export(fig, output, number, paths, letters, fonts)
        audit_path = output / f"Figure_{number:02}.json"
        audit = json.loads(audit_path.read_text())
        audit["plot_sources_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "plot/final_figures.py",
                "plot/strengthened_figures.py",
                "src/thermal_heterogeneity/figures.py",
                "src/analysis/global_extension/global_matrix.py",
                "src/analysis/global_extension/multi_event_dr_extension.py",
            ]
        }
        if example:
            audit["illustrative_trajectory"] = example
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
