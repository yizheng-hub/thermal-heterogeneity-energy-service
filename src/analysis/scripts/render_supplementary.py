"""Rebuild supplementary evidence figures from anonymous, frozen analysis products."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from thermal_heterogeneity.figures import (
    CONTEXT_COLORS,
    CONTROL_COLORS,
    COUNT_CMAP,
    DEMAND_CMAP,
    GRIFFITHS_COLORS,
    PALETTE,
    REPRESENTATION_COLORS,
    VARIANCE_COLORS,
    apply_data_axis_style,
    layout_snapshot,
)

COLORS = ["#6F9EBC", "#C78470", "#7FA995", "#AF94AD"]
CONTEXTS = ["Miami", "New Delhi", "Port Angeles", "Seattle"]
CONTEXT_COLORS = dict(zip(CONTEXTS, CONTEXT_COLORS.values()))
G_COLORS = list(GRIFFITHS_COLORS.values())
POPULATION_COLORS = {
    "india_archive_stratum": PALETTE["purple"],
    "multinational_source_balanced": PALETTE["green"],
    "us_archive_stratum": PALETTE["blue"],
}
SEASON_COLORS = {"cold": PALETTE["blue"], "warm": PALETTE["vermillion"]}


def configure():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "thermal-heterogeneity-stage6-supplement",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.labelpad": 6,
            "boxplot.whiskerprops.color": PALETTE["dark_grey"],
            "boxplot.capprops.color": PALETTE["dark_grey"],
            "xtick.major.pad": 4,
            "ytick.major.pad": 4,
            "savefig.facecolor": "white",
        }
    )


def canvas(titles):
    if len(titles) == 6:
        fig, axes = plt.subplots(3, 2, figsize=(180 / 25.4, 225 / 25.4))
        bounds = [(x, y, 0.34, 0.22) for y in (0.735, 0.425, 0.115) for x in (0.13, 0.64)]
    elif len(titles) == 2:
        fig, axes = plt.subplots(2, 1, figsize=(180 / 25.4, 177 / 25.4))
        bounds = [(0.14, 0.59, 0.82, 0.32), (0.14, 0.12, 0.82, 0.32)]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 177 / 25.4))
        bounds = [
            (0.13, 0.59, 0.34, 0.32),
            (0.64, 0.59, 0.34, 0.32),
            (0.13, 0.12, 0.34, 0.32),
            (0.64, 0.12, 0.34, 0.32),
        ]
    for i, (ax, title, rectangle) in enumerate(zip(axes.flat, titles, bounds)):
        ax.set_position(rectangle)
        ax.set_title(f"({chr(97+i)}) {title}", loc="left", pad=10)
        ax.grid(False)
        ax.set_axisbelow(True)
    return fig, list(axes.flat)


def save(fig, out, number, caption, sources):
    stem = f"Figure_S{number:02d}"
    from geometry_audit import audit_figure
    from matplotlib.text import Text

    for artist in fig.findobj(Text):
        title = bool(re.match(r"^\([a-z]\) ", artist.get_text()))
        artist.set_fontfamily("Times New Roman")
        artist.set_fontsize(12 if title else 11)
        artist.set_fontweight("bold" if title else "normal")
        artist.set_color("black")
    for ax in fig.axes:
        ax.grid(False)
        ax.xaxis.labelpad = 6
        ax.yaxis.labelpad = 7
    if number == 10:
        fig.axes[1].set_xlabel("Assumed gas price\n(INR/kWh HHV)")
    elif number == 9:
        for ax, title in zip(
            fig.axes,
            [
                "City: ordinal RPS",
                "Source: ordinal RPS",
                "City: tail-share error",
                "Source: tail-share error",
            ],
        ):
            label = ax.get_title(loc="left")[:3]
            ax.set_title(label + " " + title, loc="left", pad=18, fontsize=12, weight="bold")
    elif number == 11:
        fig.axes[2].set_xticklabels(
            ["<20", "20–29", "30–39", "40–49", "50–59", "60+"], rotation=45, ha="right"
        )
    elif number == 14:
        maps = fig.axes[:2]
        for ax, bounds in zip(maps, [(0.13, 0.65, 0.84, 0.26), (0.13, 0.34, 0.84, 0.26)]):
            ax.set_position(bounds)
        maps[0].tick_params(axis="x", bottom=False, labelbottom=False)
        for ax in maps:
            for artist in list(ax.texts):
                artist.remove()
        # Common map columns exactly match the data rectangle below.
        fig.axes[3].set_position((0.13, 0.065, 0.33, 0.19))
        fig.axes[4].set_position((0.64, 0.065, 0.33, 0.19))
    elif number == 15:
        ax = fig.axes[2]
        for artist in list(ax.texts):
            artist.remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()
        ax.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=0.6,
            handletextpad=0.3,
        )
        ax.set_title("(c) GPR interval calibration", loc="left", pad=28, fontsize=12, weight="bold")
        ax = fig.axes[3]
        for artist in list(ax.texts):
            artist.remove()
        for line in list(ax.lines):
            if len(line.get_xdata()) == 1:
                line.remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            frameon=False,
            ncol=2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            columnspacing=0.6,
            handletextpad=0.3,
        )
        ax.set_title("(d) Coefficient sensitivity", loc="left", pad=28, fontsize=12, weight="bold")
    submission_supplement_layout(fig, number, out)
    apply_data_axis_style(fig, map_axes=fig.axes[:2] if number == 14 else [])
    for artist in fig.findobj(Text):
        title = bool(re.match(r"^\([a-z]\) ", artist.get_text()))
        artist.set_fontfamily("Times New Roman")
        artist.set_fontsize(12 if title else 11)
        artist.set_fontweight("bold" if title else "normal")
        artist.set_color("black")
    audit = audit_figure(fig)
    (out / f"{stem}_geometry.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(out / f"{stem}.png", dpi=300)
    fig.savefig(out / f"{stem}_word.png", dpi=300)
    plt.close(fig)
    (out / f"{stem}_caption.txt").write_text(caption + "\n", encoding="utf-8")
    return {
        "figure": f"S{number}",
        "stem": stem,
        "caption": caption,
        "sources": sources,
        "panels": 6 if number == 2 else 2 if number == 12 else 4,
    }


def submission_supplement_layout(fig, number, out):
    """Keep ordinary SI panels on one grid and reserve map/diagram layouts."""
    before = layout_snapshot(fig)
    if number != 14:
        for axis in fig.axes[:6 if number == 2 else 4]:
            if number != 15:
                axis.set_title(
                    axis.get_title(loc="left"), loc="left", pad=10, fontsize=12, fontweight="bold"
                )
        if number == 3:
            fig.axes[4].set_position((0.49, 0.59, 0.012, 0.32))
    after = layout_snapshot(fig)
    unchanged = all(
        before[key][field] == after[key][field]
        for key in before
        for field in ["xlim", "ylim", "xscale", "yscale"]
    )
    if not unchanged:
        raise RuntimeError("Supplementary layout changed a data-axis domain")
    (out / f"Figure_S{number:02d}_submission_layout.json").write_text(
        json.dumps(
            {"before": before, "after": after, "data_axis_domains_unchanged": unchanged}, indent=2
        ),
        encoding="utf-8",
    )


def box(ax, values, labels, ylabel, colors=None):
    bp = ax.boxplot(
        values,
        tick_labels=labels,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "#7F7F7F", "linewidth": 1},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor((colors or COLORS)[i % len(colors or COLORS)])
        patch.set_alpha(0.82)
    ax.set_ylabel(ylabel)


def plot_control_inputs(input_root, out):
    """Plot the anonymous inputs and commands already executed in the main comparison."""
    engineering = input_root / "engineering"
    profile_path = engineering / "PROFILE_ASSIGNMENTS.csv.gz"
    command_path = engineering / "CONTROL_REFERENCE.json"
    group_path = engineering / "ZONE_EXECUTION_GROUPS.csv"
    profiles = pd.read_csv(profile_path)
    commands = json.loads(command_path.read_text(encoding="utf-8"))
    groups = pd.read_csv(group_path)
    example = profiles[(profiles.context == "Miami") &
                       (profiles.layout == "random_mixing")].copy()
    composition_ids = sorted(example.composition_id.unique())
    if len(composition_ids) != 3 or any(
        len(example[example.composition_id == key]) != 267 for key in composition_ids
    ):
        raise ValueError("Expected the three fixed compositions of 267 anonymous profiles")
    source = []
    for row in commands:
        if row["variant"] != 0:
            continue
        for zone, target in row["targets_c"].items():
            source.append({**{k: row[k] for k in ("context", "composition_id", "layout",
                                                "permission", "control", "variant")},
                           "zone_id": zone, "target_c": target})
    source = pd.DataFrame(source)
    source.to_csv(out / "actual_control_target_source.csv", index=False)
    example.to_csv(out / "actual_assigned_profile_source.csv", index=False)
    colors = [PALETTE["blue"], PALETTE["purple"], PALETTE["green"]]
    fig, axes = canvas(["Assigned interval spectra", "Demand centres and widths",
                        "Executed zone commands", "Range across controlled zones"])
    for offset, (composition, color) in enumerate(zip(composition_ids, colors)):
        local = example[example.composition_id == composition].sort_values("centre_c")
        y = np.arange(267) + 267 * offset
        axes[0].hlines(y, local.lower_c, local.upper_c, color=color, lw=.42, alpha=.45)
        axes[0].scatter(local.centre_c, y, s=2.5, color=color, linewidths=0, zorder=3)
        axes[1].scatter(local.centre_c, local.upper_c - local.lower_c, s=10,
                        color=color, alpha=.65, linewidths=0,
                        label=f"Composition {offset + 1}")
    axes[0].set_yticks([133, 400, 667], ["1", "2", "3"])
    axes[0].set_ylim(-10, 811)
    axes[0].set_xlim(16, max(32, np.ceil(example.centre_c.max())))
    axes[0].set_xlabel("Assigned demand interval (°C)")
    axes[0].set_ylabel("Composition (267 profiles each)")
    axes[1].set_xlabel("Demand centre (°C)")
    axes[1].set_ylabel("Model-derived width (°C)")
    axes[1].legend(frameon=False, loc="upper right", handletextpad=.3)
    native = source[source.permission == "native_zone"]
    context_names = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
    controls = ["uniform_average", "personal_center", "ltpref_interval"]
    control_colors = [CONTROL_COLORS["uniform"], CONTROL_COLORS["personal_center"],
                      CONTROL_COLORS["ltpref_interval"]]
    for control_index, (control, color) in enumerate(zip(controls, control_colors)):
        for context_index, context in enumerate(context_names):
            local = native[(native.context == context) & (native.control == control)]
            position = context_index + (control_index - 1) * .24
            bp = axes[2].boxplot([local.target_c.to_numpy()], positions=[position], widths=.20,
                                patch_artist=True, manage_ticks=False, showfliers=False,
                                medianprops={"color": "black", "linewidth": .8},
                                boxprops={"linewidth": .65}, whiskerprops={"linewidth": .65},
                                capprops={"linewidth": .65})
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(.85)
            points = np.sort(local.target_c.to_numpy())
            jitter = (np.arange(len(points)) % 7 - 3) * .018
            axes[2].scatter(position + jitter, points, s=3, color=color, alpha=.55,
                            linewidths=0, zorder=3)
    axes[2].set_xticks(range(4), ["Miami", "New\nDelhi", "Port\nAngeles", "Seattle"])
    axes[2].set_ylabel("Occupied target (°C)")
    axes[2].set_xlabel("Building–weather context")
    ranges = (source.groupby(["context", "composition_id", "layout", "permission", "control"])
              .target_c.agg(lambda value: value.max() - value.min()).reset_index(name="span_c"))
    group_lookup = {}
    for context, local in groups.groupby("context"):
        group_lookup[(context, "building")] = 1
        group_lookup[(context, "floor")] = local.floor.nunique()
        group_lookup[(context, "native_zone")] = local.zone.nunique()
    ranges["independent_command_groups"] = [group_lookup[(row.context, row.permission)]
                                           for row in ranges.itertuples()]
    ranges.to_csv(out / "actual_control_command_range_source.csv", index=False)
    for i, permission in enumerate(["building", "floor", "native_zone"]):
        for j, (control, color) in enumerate(zip(controls, control_colors)):
            values = ranges[(ranges.permission == permission) &
                            (ranges.control == control)].span_c.to_numpy()
            position = i + (j - 1) * .22
            lo, med, hi = np.quantile(values, [.1, .5, .9])
            axes[3].plot([lo, hi], [position, position], color=color, lw=2)
            axes[3].scatter(values, position + (np.arange(len(values)) % 5 - 2) * .025,
                            color=color, s=8, alpha=.55, linewidths=0)
            axes[3].scatter([med], [position], color=color, s=23, edgecolors="white", lw=.4,
                            zorder=4)
    axes[3].set_yticks(range(3), ["Building", "Floor", "Zone"])
    axes[3].invert_yaxis()
    axes[3].set_xlabel("Within-plan target range (°C)")
    axes[3].set_ylabel("Command permission")
    # One shared legend sits between the two rows, outside all data rectangles.
    fig.legend(handles=[Patch(facecolor=color, label=label) for color, label in
                         zip(control_colors, ["Uniform average", "Personal centre", "Personal range"])],
               frameon=False, ncol=3, loc="center", bbox_to_anchor=(.53, .505),
               columnspacing=1.5, handlelength=1.2)
    caption = (
        "Figure S13 | Assigned demand inputs and control commands in the main comparison. "
        "a, The three fixed compositions, each containing 267 archive-derived virtual profiles. "
        "Profiles are ordered by reconstructed centre within each composition. Horizontal marks "
        "give model-derived intervals and points give centres. These are the Miami random-mixing "
        "assignment, whose three profile sets are also used in the other layouts and contexts. "
        "b, Centres and model-derived interval widths for the same 801 profile assignments. "
        "c, Distributions of the actually generated occupied zone-level thermostat targets for "
        "the minimum-predicted-mismatch candidate of each control, including all three compositions and layouts. "
        "Boxes give medians and interquartile ranges, whiskers extend to 1.5 interquartile ranges, "
        "and points retain every zone command. d, The maximum minus minimum zone target of each "
        "minimum-predicted-mismatch plan, separated by command permission and control. Points give all declared "
        "plans, thick segments span P10–P90 and larger white-edged points mark medians. "
        "Uniform, Personal centre and Personal range controls are grey, gold and green. Small-building "
        "floor and building permissions coincide. Commands are model inputs, not measured room temperatures."
    )
    metadata = {path.relative_to(input_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [profile_path, command_path, group_path]}
    (out / "actual_control_input_sha256.json").write_text(json.dumps(metadata, indent=2),
                                                       encoding="utf-8")
    return save(fig, out, 13, caption, list(metadata))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--panel-data", type=Path)
    ap.add_argument("--output", "--output-root", dest="output", type=Path)
    ap.add_argument("--science-audit", type=Path)
    ap.add_argument("--land", type=Path)
    ap.add_argument("--energy-service-input-root", type=Path,
                    help="Combined-release energy_service directory for actual control-input plots")
    ap.add_argument(
        "--spatial-audit",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "spatial_method_audit",
    )
    args = ap.parse_args()
    project = args.project_root.resolve()
    s6 = project / "results/stage6_revision_20261022"
    args.data_root = args.data_root or project / "release/thermal-heterogeneity/staging/data_v3"
    args.panel_data = args.panel_data or project / "data/figure_panel_data"
    args.output = (
        args.output
        or project / "results/stage7_service_first_revision_20261022/supplementary_figures"
    )
    args.science_audit = args.science_audit or s6 / "science_audit"
    args.land = (
        args.land
        or project
        / "release/thermal-heterogeneity/src/thermal_heterogeneity/assets/ne_110m_land.geojson"
    )
    if not args.spatial_audit.exists():
        args.spatial_audit = s6 / "spatial_method_audit"
    b = args.data_root
    p = args.panel_data
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    configure()
    manifest = []

    def normalized(frame):
        names = {
            "NewDelhi": "New Delhi",
            "PortAngeles_small2019": "Port Angeles",
            "Seattle_small2019": "Seattle",
        }
        for key in ["city", "context"]:
            if key in frame:
                frame[key] = frame[key].replace(names)
        return frame

    def read(path):
        return normalized(pd.read_csv(b / path))

    def panel(name):
        return normalized(pd.read_csv(p / name))

    human = read("human/human_profiles_minimized.csv.gz")

    fig, a = canvas(["Uniform 24 °C", "Uniform 25 °C", "Uniform 26 °C", "Climate-group contrast"])
    summary = []
    for ax, target in zip(a[:3], [24, 25, 26]):
        for j, (g, col) in enumerate(
            zip(["reference_g033_c", "reference_g050_c", "reference_g067_c"], G_COLORS)
        ):
            values = human[g].to_numpy() - target
            ax.hist(
                values,
                bins=np.arange(-12, 12.5, 0.5),
                weights=human.unit_equal_weight,
                histtype="step",
                density=True,
                color=col,
                label=["g = 0.33", "g = 0.50", "g = 0.67"][j],
            )
            summary.append(
                {
                    "target_c": target,
                    "g_column": g,
                    "record_count": len(values),
                    "mean_absolute_gap_c": np.average(
                        np.abs(values), weights=human.unit_equal_weight
                    ),
                }
            )
        ax.axvline(0, color="#555555", lw=0.7)
        ax.set_xlim(-10, 10)
        ax.set_xlabel("Demand centre − target (°C)")
        ax.set_ylabel("Weighted density")
    a[0].legend(frameon=False, fontsize=11)
    groups = human.groupby("climate_group").size().sort_values(ascending=False).head(6).index
    vals = [human.loc[human.climate_group == g, "reference_g050_c"].values for g in groups]
    box(
        a[3],
        vals,
        list(range(1, len(groups) + 1)),
        "Demand centre (°C)",
        colors=[PALETTE[k] for k in ["blue", "vermillion", "green", "purple", "sky", "orange"]],
    )
    a[3].set_xlabel("Climate label (see caption)")
    a[3].tick_params(axis="x", labelsize=11)
    pd.DataFrame(summary).to_csv(out / "baseline_coordinate_summary.csv", index=False)
    manifest.append(
        save(
            fig,
            out,
            1,
            "Figure S1 | Reconstructed demand relative to uniform thermal coordinates. a–c, Analysis-unit-weighted demand-centre-minus-target distributions for 24, 25 and 26 °C coordinates at Griffiths coefficients 0.33, 0.50 and 0.67 °C−1, shown in blue, gold and green. d, Record-level demand-centre distributions for the six most represented climate labels, numbered in order: "
            + "; ".join(f"{i+1}, {g}" for i, g in enumerate(groups))
            + ". Boxes show medians and interquartile ranges, whiskers extend to 1.5 interquartile ranges, and the underlying data retain outliers.",
            ["human/human_profiles_minimized.csv.gz"],
        )
    )

    variance = panel("Fig1h.csv")
    components = list(variance.component.drop_duplicates())
    fig, a = canvas(
        [
            "Record-equal weighting",
            "Unit-equal weighting",
            "Building-equal weighting",
            "Source-equal weighting",
            "Identified-unit check",
            "Variance components",
        ]
    )
    labels = {
        "between_country": "Country",
        "between_city_within_country": "City",
        "between_building_within_city": "Building",
        "between_analysis_unit_within_building": "Within\nbuilding",
        "within_analysis_unit": "Within-unit\nvariation",
    }
    for ax, (weight, d) in zip(a, variance.groupby("weighting", sort=True)):
        # The title follows the actual group to avoid relying on sort order.
        ax.set_title(
            f"({chr(97+a.index(ax))}) "
            + weight.replace("_equal", " equal").replace("_", " ").capitalize(),
            loc="left",
            pad=18,
        )
        for j, g in enumerate(sorted(d.griffiths.unique())):
            q = d[d.griffiths == g].set_index("component").reindex(components)
            y = np.arange(len(q)) + (j - 1) * 0.22
            center = q.loo_median_share.to_numpy() * 100
            ax.errorbar(
                center,
                y,
                xerr=[
                    center - q.loo_min_share.to_numpy() * 100,
                    q.loo_max_share.to_numpy() * 100 - center,
                ],
                fmt="o",
                ms=3,
                capsize=2,
                color=GRIFFITHS_COLORS[g],
                label=f"g = {g:g}",
            )
        ax.set_yticks(
            range(len(components)),
            [labels.get(x, x.replace("_", " ")) for x in components],
            fontsize=11,
        )
        ax.set_xlabel("Archive variance share (%)")
        ax.set_xlim(-2, 85)
    a[0].legend(frameon=False, fontsize=11, loc="upper right")
    identity_variance = pd.read_csv(args.science_audit / "variance_recheck.csv")
    comps = list(identity_variance.component.drop_duplicates())
    groups = list(identity_variance.cohort.drop_duplicates())
    y = np.arange(len(groups))
    left = np.zeros(len(groups))
    for j, comp in enumerate(comps):
        q = identity_variance[identity_variance.component == comp].set_index("cohort").reindex(groups)
        a[4].barh(
            y, q.share * 100, left=left, color=VARIANCE_COLORS[j], label=comp.replace("_", " ")
        )
        left += q.share.to_numpy() * 100
    a[4].set_yticks(y, ["All records", "Identified\nsubjects"])
    a[4].set_xlabel("Archive variance share (%)")
    a[4].set_ylim(-0.5, 2.45)
    component_labels = ["Country", "City", "Building", "Within\nbuilding", "Within-unit\nvariation"]
    handles = [Patch(facecolor=VARIANCE_COLORS[j], edgecolor="none") for j in range(len(comps))]
    a[4].legend(
        handles, [label.replace("\n", " ") for label in component_labels], frameon=False, loc="upper left", ncol=2,
        handlelength=1, columnspacing=0.7, labelspacing=0.2,
    )
    arrays = [
        identity_variance[identity_variance.cohort == group].set_index("component").reindex(comps).share.to_numpy() * 100
        for group in groups
    ]
    for y, (v, w) in enumerate(zip(*arrays)):
        a[5].plot([v, w], [y, y], color="#aaaaaa", lw=1)
    for values, color, label in zip(
        arrays, [PALETTE["blue"], PALETTE["vermillion"]], ["All records", "Identified subjects"]
    ):
        a[5].scatter(values, range(5), s=35, color=color, label=label, zorder=3)
    a[5].set_yticks(range(5), component_labels)
    a[5].invert_yaxis()
    a[5].set_xlim(0, 65)
    a[5].set_xlabel("Variance share (%)")
    a[5].legend(frameon=False, loc="upper right")
    manifest.append(
        save(
            fig,
            out,
            2,
            "Figure S2 | Hierarchical variance and identity sensitivity. a–d, Leave-one-source variance shares under the four weighting schemes identified in the panel titles. Points are medians across twenty-seven source removals and whiskers give the minimum and maximum. Blue, gold and green denote Griffiths coefficients 0.33, 0.50 and 0.67 °C−1. The five components are country, city within country, building within city, analysis unit within building and repeated measurement within analysis unit. Building labels denote database groups; 150 identifiers are original and 37 inferred. e,f, Hierarchical variance shares for all canonical records and the subset with subject-within-building identifiers. Component colours in e match Figure 1d. Blue and terracotta in f identify all records and identified subjects, respectively. The identified-subject subset uses only original-source building identifiers.",
            ["Fig1h.csv", "science_audit/variance_recheck.csv"],
        )
    )

    fig, a = canvas(
        [
            "Temperature and votes",
            "Reference sensitivity",
            "Records per unit",
            "Virtual interval widths",
        ]
    )
    h = a[0].hist2d(
        human.observed_temperature_c, human.thermal_sensation, bins=[35, 7], cmap=COUNT_CMAP
    )
    a[0].set_xlabel("Observed temperature (°C)")
    a[0].set_ylabel("Thermal-sensation vote")
    cax = fig.add_axes([0.48, 0.625, 0.012, 0.25])
    bar = fig.colorbar(h[3], cax=cax, ticks=[0, 750, 1500])
    bar.ax.set_title("n", fontsize=11, weight="normal", pad=5)
    for g, c, label in zip(
        ["reference_g033_c", "reference_g050_c", "reference_g067_c"],
        G_COLORS,
        ["g = 0.33", "g = 0.50", "g = 0.67"],
    ):
        a[1].hist(
            human[g],
            bins=np.arange(10, 38, 0.5),
            histtype="step",
            density=True,
            color=c,
            label=label,
        )
    a[1].set_xlabel("Reference centre (°C)")
    a[1].set_ylabel("Record density")
    a[1].legend(frameon=False)
    counts = human.groupby("release_unit_id").size()
    a[2].hist(
        counts, bins=np.arange(0, min(counts.max(), 40) + 2) - 0.5, color=COLORS[0], alpha=0.9
    )
    a[2].set_yscale("log")
    a[2].set_xlabel("Records per analysis unit")
    a[2].set_ylabel("Analysis units (log)")
    intervals = read("human/ltpref_intervals_minimized.csv.gz")
    width_cols = [c for c in intervals if "width" in c and "multiplier" not in c]
    lo = next((c for c in intervals if "lower" in c and c.endswith("_c")), None)
    hi = next((c for c in intervals if "upper" in c and c.endswith("_c")), None)
    if lo and hi:
        widths = intervals[hi] - intervals[lo]
    elif width_cols:
        widths = intervals[width_cols[0]]
    else:
        raise ValueError("No interval-width fields")
    a[3].hist(widths.dropna(), bins=30, color=COLORS[2], alpha=0.9)
    a[3].set_xlabel("Model-derived width (°C)")
    a[3].set_ylabel("Executable records")
    manifest.append(
        save(
            fig,
            out,
            3,
            "Figure S3 | Observation and derived-input diagnostics. a, Observed temperature and thermal-sensation votes in the retained office cohort. Colour and n give records per bin. b, Reconstructed reference centres at three Griffiths coefficients. c, Records per analysis unit, with unit counts on a logarithmic axis. Analysis units include identified subject-within-building groups and single-record units without a usable subject identifier. d, Model-derived executable Personal range widths.",
            ["human/human_profiles_minimized.csv.gz", "human/ltpref_intervals_minimized.csv.gz"],
        )
    )

    votes = read("human/direct_vote_windows_sanitized.csv.gz")
    eligible = votes[votes.eligible.astype(str).str.lower() == "true"]
    fig, a = canvas(
        ["NUS room windows", "ENTH room windows", "Requests within windows", "Compression gaps"]
    )
    for ax, code in zip(a[:2], ["NUS_HAAS", "ENTH"]):
        q = eligible[eligible.dataset_code == code]
        ax.scatter(
            q.full_burden,
            q.compressed_burden,
            s=22,
            color=PALETTE["blue"] if code == "NUS_HAAS" else PALETTE["orange"],
            alpha=0.85,
        )
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=0.8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Full vote burden")
        ax.set_ylabel("Burden after averaging")
        ax.text(0.04, 0.94, f"n = {len(q)} windows", transform=ax.transAxes, va="top")
    d = panel("Fig2h.csv").drop_duplicates("dataset")
    xs = np.arange(len(d))
    w = 0.24
    for j, (col, label) in enumerate(
        [
            ("disagreement_count", "Differing"),
            ("opposing_count", "Opposing"),
            ("cancellation_count", "Cancelled"),
        ]
    ):
        a[2].bar(
            xs + (j - 1) * w,
            d[col],
            w,
            label=label,
            color=[PALETTE["blue"], PALETTE["vermillion"], PALETTE["dark_grey"]][j],
        )
    a[2].set_xticks(xs, d.dataset.replace({"NUS_HAAS": "NUS"}))
    a[2].set_ylabel("Windows")
    a[2].legend(frameon=False, fontsize=11)
    gap_values = [
        eligible.loc[eligible.dataset_code == x, "jensen_gap"].values for x in ["NUS_HAAS", "ENTH"]
    ]
    box(
        a[3],
        gap_values,
        ["NUS", "ENTH"],
        "Full minus averaged burden",
        colors=[PALETTE["blue"], PALETTE["orange"]],
    )
    for j, values in enumerate(gap_values):
        a[3].scatter(
            j + 1 + np.linspace(-0.15, 0.15, len(values)),
            values,
            s=10,
            color=[PALETTE["blue"], PALETTE["orange"]][j],
            alpha=0.8,
        )
    manifest.append(
        save(
            fig,
            out,
            4,
            "Figure S4 | Shared-room vote replication. a,b, Mean absolute vote (full burden) versus absolute mean vote (burden after averaging) for eligible NUS and ENTH thirty-minute room-windows. Dashed lines mark equality. c, Window counts containing differing categories, opposing warmer/cooler requests, and exact cancellation. d, Full-minus-averaged burden distributions. The National University of Singapore (NUS) and ENTH datasets contain 63/34 windows and 225/81 votes. Boxes show medians and interquartile ranges, whiskers extend to 1.5 interquartile ranges, and points show individual windows.",
            ["human/direct_vote_windows_sanitized.csv.gz", "Fig2h.csv"],
        )
    )

    energies = panel("Fig3d.csv")
    carbon = panel("Fig3e.csv")
    fig, a = canvas(CONTEXTS)
    for ax, city in zip(a, CONTEXTS):
        d = energies[
            (energies.context == city) & (energies.matched.astype(str).str.lower() == "true")
        ]
        population_labels = {
            "india_archive_stratum": "India",
            "us_archive_stratum": "United States",
            "multinational_source_balanced": "Pooled",
        }
        for j, (pop, q) in enumerate(d.groupby("population")):
            q = q.groupby("delta_c").delta_energy_mwh.agg(["median", "count"])
            ax.plot(
                q.index,
                q["median"],
                marker=["o", "s", "^"][j],
                lw=1.2,
                color=POPULATION_COLORS[pop],
                label=population_labels.get(pop, pop),
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel("Centre shift (°C)")
        ax.set_ylabel("Annual energy change (MWh)")
    a[0].legend(frameon=False, fontsize=11)
    manifest.append(
        save(
            fig,
            out,
            5,
            "Figure S5 | Demand-centre energy effects by archive composition. a–d, Median annual site-energy changes relative to the Δ=0 reference among cases with qualifying selections in both conditions at archive-derived spread in Miami, New Delhi, Port Angeles and Seattle, respectively. The horizontal coordinate is the applied demand-centre shift. Blue, mauve and green denote the United States, India and multinational archive compositions. Curves have no value where a comparator is unavailable. These compositions are assigned archive scenarios.",
            ["Fig3d.csv"],
        )
    )

    annual = read("expected/annual_decisions_reference.csv.gz")
    fig, a = canvas(
        [
            "Annual false acceptance",
            "Qualified annual choices",
            "Eligible annual actions",
            "Selected annual energy",
        ]
    )
    base = annual[annual.delta_c == 0]
    for j, (budget, d) in enumerate(base.groupby("service_fraction")):
        rows = []
        for lam, q in d.groupby("lambda"):
            mean = q.mean_selected_action_id.notna()
            false = q.mean_selected_full_reaudit_status.eq("mean_selected_false_accept")
            rows.append(
                (lam, int(false.sum()), int(mean.sum()), q.full_selected_action_id.notna().sum())
            )
        z = pd.DataFrame(rows, columns=["lambda", "false", "mean", "full"])
        # Status spelling is checked below against actual non-empty selections.
        if z["false"].sum() == 0:
            for k, (lam, q) in enumerate(d.groupby("lambda")):
                selected = q.mean_selected_action_id.notna()
                z.loc[k, "false"] = (
                    selected & ~q.mean_selected_full_eligible.fillna(False).astype(bool)
                ).sum()
        a[0].plot(
            z["lambda"],
            100 * z["false"] / z["mean"].replace(0, np.nan),
            marker="o",
            color=COLORS[j],
            label=f"Budget {budget:g}",
        )
        a[1].plot(z["lambda"], z["full"], marker="o", color=COLORS[j], label=f"{budget:g}")
    a[0].set_ylabel("False acceptance (%)")
    a[0].set_xlabel("Demand spread λ")
    a[0].legend(frameon=False, fontsize=11)
    a[1].set_ylabel("Selections among 36 cells")
    a[1].set_xlabel("Demand spread λ")
    for city, col in CONTEXT_COLORS.items():
        d = base[base.city == city].groupby("lambda").full_eligible_action_count.median()
        a[2].plot(d.index, d, marker="o", color=col, label=city)
    a[2].set_ylabel("Median eligible actions")
    a[2].set_xlabel("Demand spread λ")
    a[2].legend(frameon=False, fontsize=11)
    q = base[
        (base["lambda"] == 1)
        & base.full_selected_site_kwh.notna()
        & base.mean_selected_site_kwh.notna()
    ]
    box(
        a[3],
        [
            (
                q.loc[q.city == c, "full_selected_site_kwh"]
                - q.loc[q.city == c, "mean_selected_site_kwh"]
            )
            / 1000
            for c in CONTEXTS
        ],
        ["Miami", "New\nDelhi", "Port\nAngeles", "Seattle"],
        "Full minus mean energy (MWh)",
        colors=list(CONTEXT_COLORS.values()),
    )
    a[3].axhline(0, color="gray", lw=0.7)
    manifest.append(
        save(
            fig,
            out,
            6,
            "Figure S6 | Annual service-budget and decision diagnostics. a, Fraction of mean-selected cases failing full-distribution review under four nested service budgets. b, Full-distribution selections among the thirty-six composition–seed–context cells per budget. c, Median full-qualified action count in each context. Terracotta, mauve, blue and green denote Miami, New Delhi, Port Angeles and Seattle, respectively. d, Full-minus-mean selected annual energy among bilateral selections at archive spread. Boxes show medians and interquartile ranges, whiskers extend to 1.5 interquartile ranges, and outliers remain in the underlying data. Budgets share the same underlying scenarios.",
            ["expected/annual_decisions_reference.csv.gz"],
        )
    )

    dynamic = read("expected/dynamic_decisions_reference.csv.gz")
    fig, a = canvas(CONTEXTS)
    for ax, city in zip(a, CONTEXTS):
        for j, month in enumerate([2, 7]):
            d = dynamic[
                (dynamic.city == city) & (dynamic.event_month == month) & (dynamic["lambda"] == 1)
            ]
            z = d.groupby("delta_c").full_selected_event_average_reduction_kw.count()
            ax.plot(
                z.index,
                z,
                marker="o",
                color=[SEASON_COLORS["cold"], SEASON_COLORS["warm"]][j],
                label=["February", "July"][j],
            )
        ax.set_xlabel("Centre shift (°C)")
        ax.set_ylabel("Positive full selections / 36")
        ax.set_ylim(-1, 37)
    a[0].legend(frameon=False)
    manifest.append(
        save(
            fig,
            out,
            7,
            "Figure S7 | Seasonal response availability by demand centre. a–d, Positive event selections satisfying population-mean, fixed-tail and recovery service limits at archive-derived spread in Miami, New Delhi, Port Angeles and Seattle. Blue/terracotta denote February/July. Each centre–context–month point contains nine composition–seed scenarios and four nested budgets, for thirty-six design cells. Events last two hours with forty-eight consecutive hours of recovery.",
            ["expected/dynamic_decisions_reference.csv.gz"],
        )
    )

    qt = read("expected/quantile_certification_gate_reference.csv")
    fig, a = canvas(
        [
            "Context–seed holdout",
            "Population holdout",
            "Qualification margins",
            "Protection energy penalty",
        ]
    )
    for ax, (scheme, d) in zip(a[:2], qt.groupby("split_scheme", sort=True)):
        for j, (rep, q) in enumerate(d.groupby("representation")):
            ax.scatter(
                q.retained_full_eligible_fraction * 100,
                q.false_accept_candidates + 1,
                color=REPRESENTATION_COLORS[rep],
                marker=["o", "s", "^"][j],
                s=35,
                label={"mean_only": "Mean only", "quantile3": "Three bins", "quantile5": "Five bins"}[rep],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Qualified actions retained (%)")
        ax.set_ylabel("False accepts + 1 (log)")
        ax.set_xlim(-3, 103)
        ax.set_title(
            f"({chr(97+a.index(ax))}) "
            + scheme.replace("leave_one_", "").replace("_out", "").replace("_", " ").capitalize(),
            loc="left",
            pad=18,
        )
    a[0].legend(frameon=False)
    margins = read("expected/quantile_fold_margins_reference.csv")
    # The protected q5 rule is identified semantically rather than by position.
    q = margins[
        margins.representation.astype(str).str.contains("5")
        & margins.strategy.astype(str).str.contains("95")
    ]
    if q.empty:
        q = margins[margins.representation.astype(str).str.contains("5")]
    box(
        a[2],
        [q.mean_margin_c.values, q.tail_margin_c.values],
        ["Mean", "Tail"],
        "Training-derived margin (°C)",
        colors=[PALETTE["dark_grey"], PALETTE["vermillion"]],
    )
    q = qt[qt.representation.astype(str).str.contains("5")]
    a[3].scatter(
        q.retained_full_eligible_fraction * 100,
        q.energy_penalty_median_kwh / 1000,
        s=35,
        c=REPRESENTATION_COLORS["quantile5"],
    )
    a[3].set_xlabel("Qualified actions retained (%)")
    a[3].set_ylabel("Median energy penalty (MWh)")
    manifest.append(
        save(
            fig,
            out,
            8,
            "Figure S8 | Anonymous-summary rule comparisons. a,b, Mean-only, three-bin and five-bin summaries under all tested protection strategies for context–seed and population holdout, respectively. Grey/gold/green identify the summaries. Three-bin and five-bin summaries use at most three and five near-equal-count bins per zone with bin means, bin counts and fixed-tail membership counts. One is added to false-accept counts for logarithmic display. c, Training-fold mean and tail margins for five-bin summaries with 95th-percentile protection. Boxes show medians and interquartile ranges, with whiskers at 1.5 interquartile ranges. d, Median selection-energy penalty versus qualified-action retention for the tested five-bin protection strategies.",
            [
                "expected/quantile_certification_gate_reference.csv",
                "expected/quantile_fold_margins_reference.csv",
            ],
        )
    )

    learn = read("expected/local_update_learning_curve_reference.csv")
    fig, a = canvas(
        [
            "City transfer: ordinal RPS",
            "Source transfer: ordinal RPS",
            "City transfer: tail-share error",
            "Source transfer: tail-share error",
        ]
    )
    for ax, (split, metric) in zip(
        a, [("city", "crps"), ("source", "crps"), ("city", "tail_mae"), ("source", "tail_mae")]
    ):
        d = learn[learn.split.astype(str).str.contains(split)].sort_values("feedback_n")
        key = "updated_" + metric
        ax.errorbar(
            d.feedback_n,
            d[key],
            yerr=[d[key] - d[key + "_ci_low"], d[key + "_ci_high"] - d[key]],
            fmt="o-",
            capsize=3,
            color=PALETTE["blue"] if split == "city" else PALETTE["orange"],
        )
        ax.plot(d.feedback_n, d["base_" + metric], ls="--", color="gray", label="Prior")
        ax.set_xlabel("Local feedback observations")
        ax.set_ylabel("Ordinal RPS (unitless)" if metric == "crps" else "Tail-share MAE (fraction)")
        ax.set_xticks([0, 10, 20, 40])
    a[0].legend(frameon=False)
    manifest.append(
        save(
            fig,
            out,
            9,
            "Figure S9 | Local-update learning curves. a,b, Analysis-unit-balanced ordinal ranked probability score (RPS, unitless) under city-held-out and source-held-out transfer. c,d, Mean absolute error of predicted versus observed warmer/cooler vote shares (fraction) under the same splits. Dashed lines show the prior. Points and whiskers show updated estimates and hierarchical-bootstrap 95% intervals. The horizontal coordinate is the number of local feedback observations.",
            ["expected/local_update_learning_curve_reference.csv"],
        )
    )

    econ = read("building/regional_carbon_cost.csv.gz")
    matched = econ[(econ.strategy != "uniform") & econ.annual_energy_cost_delta_currency.notna()]
    fig, a = canvas(CONTEXTS)
    for ax, city in zip(a, CONTEXTS):
        col = CONTEXT_COLORS[city]
        if city == "New Delhi":
            gas = np.linspace(0, 15, 100)
            delhi = econ[
                (econ.city == city)
                & (econ.strategy != "uniform")
                & (econ.comparison_status == "matched")
                & (econ.service_fraction == 0.75)
            ]
            for method, line_color, label in [
                ("personal_center", CONTROL_COLORS["personal_center"], "Personal centre"),
                ("ltpref_interval", CONTROL_COLORS["ltpref_interval"], "Personal range"),
            ]:
                q = delhi[delhi.strategy == method].dropna(
                    subset=[
                        "delhi_affine_headroom_intercept_at_pf095_INR_year",
                        "delhi_affine_headroom_slope_per_INR_kwh_gas",
                    ]
                )
                values = (
                    q.delhi_affine_headroom_intercept_at_pf095_INR_year.to_numpy()[:, None]
                    + q.delhi_affine_headroom_slope_per_INR_kwh_gas.to_numpy()[:, None]
                    * gas[None, :]
                )
                ax.fill_between(
                    gas, values.min(axis=0), values.max(axis=0), color=line_color, alpha=0.25
                )
                ax.plot(gas, values.mean(axis=0), color=line_color, label=f"{label} (n={len(q)})")
            ax.axhline(0, color="gray", lw=0.7)
            ax.set_xlabel("Assumed gas price (INR/kWh HHV)")
            ax.set_ylabel("Energy-bill headroom (INR/year)")
            ax.legend(frameon=False, fontsize=11)
            continue
        d = matched[matched.city == city]
        if len(d):
            ax.scatter(
                d.delta_operational_co2_kg / 1000,
                d.signed_management_headroom_currency_year,
                s=12,
                alpha=0.75,
                color=col,
            )
        ax.axhline(0, color="gray", lw=0.7)
        ax.axvline(0, color="gray", lw=0.7)
        ax.set_xlabel("Operational CO₂ change (t/year)")
        currency = d.currency.iloc[0] if len(d) else ("INR" if city == "New Delhi" else "USD")
        ax.set_ylabel(f"Energy-bill headroom ({currency}/year)")
    manifest.append(
        save(
            fig,
            out,
            10,
            "Figure S10 | Energy-bill headroom and carbon accounting. a,c,d, Annual operational-carbon change versus energy-bill headroom for matched non-uniform controls in Miami, Port Angeles and Seattle, using historical state-average commercial energy prices. b, New Delhi headroom at service-budget fraction 0.75 and assumed power factor 0.95 as the assumed commercial gas price varies from 0 to 15 INR/kWh HHV. Gold/green denote Personal centre/Personal range control. Lines are scenario means and bands span matched constructed scenarios. Positive/negative headroom denotes energy-bill saving/increase before implementation costs. The gas-price axis is an assumed stress range. USD and INR denote United States dollars and Indian rupees. HHV denotes higher heating value.",
            ["building/regional_carbon_cost.csv.gz", "contracts/carbon_boundary_contract.json"],
        )
    )

    gender = pd.read_csv(args.science_audit / "demographic_city_season_gender_summary.csv")
    age = pd.read_csv(args.science_audit / "demographic_city_season_age_summary.csv")
    fig, a = canvas(
        [
            "Gender centre contrast",
            "Gender width contrast",
            "Age-band descriptions",
            "Demographic coverage",
        ]
    )
    for ax, metric in zip(a[:2], ["median", "p10_p90_width"]):
        for j, g in enumerate(["033", "050", "067"]):
            field = f"reference_center_g{g}_{metric}_c"
            q = gender.pivot(index=["country", "city", "season"], columns="gender", values=field)
            values = (q["female"] - q["male"]).dropna()
            ax.hist(
                values,
                bins=np.arange(-4, 4.5, 0.5),
                histtype="step",
                color=G_COLORS[j],
                label=["g = 0.33", "g = 0.50", "g = 0.67"][j],
            )
        ax.axvline(0, color="gray", lw=0.7)
        ax.set_ylabel("City–season settings")
    a[0].set_xlabel("Female − male median (°C)")
    a[1].set_xlabel("Female − male width (°C)")
    a[0].legend(frameon=False, fontsize=11)
    bands = [
        x
        for x in ["under_20", "20_29", "30_39", "40_49", "50_59", "60_plus"]
        if x in set(age.age_band)
    ]
    if not bands:
        bands = list(age.age_band.drop_duplicates())
    box(
        a[2],
        [age.loc[age.age_band == x, "reference_center_g050_median_c"] for x in bands],
        [x.replace("under_", "<").replace("_plus", "+").replace("_", "–") for x in bands],
        "City–season median (°C)",
        colors=[PALETTE[k] for k in ["blue", "vermillion", "green", "purple", "sky", "orange"]],
    )
    a[2].set_xlabel("Age band (years)")
    a[2].tick_params(axis="x", rotation=30)
    a[3].bar(
        ["Age", "Gender"], [10283 / 21418 * 100, 12941 / 21418 * 100], color=COLORS[:2], alpha=0.8
    )
    a[3].set_ylim(0, 100)
    a[3].set_ylabel("Office records with data (%)")
    a[3].text(0, 52, "10,283 / 21,418", ha="center", fontsize=11)
    a[3].text(1, 64, "12,941 / 21,418", ha="center", fontsize=11)
    manifest.append(
        save(
            fig,
            out,
            11,
            "Figure S11 | Demographic demand descriptions and coverage. a,b, Female-minus-male contrasts in reconstructed median demand and P10–P90 width within matched country–city–season settings, under Griffiths coefficients 0.33, 0.50 and 0.67 °C−1. c, City–season median demand-centre distributions by recorded age band. Boxes show medians and interquartile ranges, with whiskers at 1.5 interquartile ranges. d, Original age/gender field coverage among 21,418 office records. Displayed setting–group summaries contain at least twenty records and ten analysis units.",
            [
                "science_audit/demographic_city_season_gender_summary.csv",
                "science_audit/demographic_city_season_age_summary.csv",
                "science_audit/demographic_coverage.csv",
            ],
        )
    )

    severity = pd.read_csv(args.science_audit / "false_accept_exceedance_cells.csv")
    fig, a = canvas(
        [
            "Service exceedance",
            "Substitute energy penalty",
        ]
    )
    exceed = next(c for c in severity if c in ["maximum_positive_excess_c"])
    a[0].hist(severity[exceed], bins=20, color=COLORS[1], alpha=0.75)
    a[0].set_xlabel("Maximum budget exceedance (°C)")
    a[0].set_ylabel("False-accepted selections")
    cost_col = next(c for c in severity if c in ["full_alternative_energy_penalty_percent"])
    z = np.sort(severity[cost_col].dropna().to_numpy())
    a[1].plot(z, np.arange(1, len(z) + 1) / len(z), color=COLORS[0])
    a[1].set_xlabel("Substitute energy increase (%)")
    a[1].set_ylabel("Cumulative fraction")
    manifest.append(
        save(
            fig,
            out,
            12,
            "Figure S12 | Annual service exceedance and substitute energy cost. a, Maximum excess over the mean or tail service budget in the 117 false-accepted annual selections at archive spread. b, Cumulative distribution of the annual energy increase for the full-qualified substitute in sixty-three cases with an available substitute. Fifty-four cases without such a substitute are excluded from b.",
            ["science_audit/false_accept_exceedance_cells.csv"],
        )
    )

    control_input = args.energy_service_input_root or b.parent / "energy_service"
    manifest.append(plot_control_inputs(control_input, out))

    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Polygon

    land = json.loads(args.land.read_text(encoding="utf-8"))
    polygons = []
    for feature in land["features"]:
        geom = feature["geometry"]
        shapes = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        polygons.extend(np.asarray(shape[0], dtype=float) for shape in shapes)
    lon, lat = np.meshgrid(np.arange(-179.5, 180, 1), np.arange(-54.5, 80, 1))
    xy = np.column_stack([lon.ravel(), lat.ravel()])
    land_mask = np.zeros(len(xy), dtype=bool)
    for poly in polygons:
        land_mask |= MplPath(poly).contains_points(xy)

    def distance(points, observed):
        lp = np.radians(points)
        lo = np.radians(observed)
        aa = (
            np.sin((lp[:, None, 1] - lo[None, :, 1]) / 2) ** 2
            + np.cos(lp[:, None, 1])
            * np.cos(lo[None, :, 1])
            * np.sin((lp[:, None, 0] - lo[None, :, 0]) / 2) ** 2
        )
        return 6371 * 2 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))

    def interpolate(dist, values):
        idx = np.argsort(dist, axis=1)[:, :4]
        near = np.take_along_axis(dist, idx, axis=1)
        weights = 1 / np.maximum(near, 1) ** 2
        return (weights * values[idx]).sum(axis=1) / weights.sum(axis=1)

    fig = plt.figure(figsize=(180 / 25.4, 220 / 25.4))
    gs = fig.add_gridspec(
        3,
        2,
        left=0.13,
        right=0.97,
        top=0.90,
        bottom=0.10,
        hspace=0.55,
        wspace=0.5,
        height_ratios=[1, 1, 1.1],
    )
    mapaxes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    cv = []
    grid_summaries = []
    for ax, season, label in zip(
        mapaxes,
        ["warm", "cold"],
        ["(a) Warm-season interpolation", "(b) Cold-season interpolation"],
    ):
        d = read("curves/city_season_demand_support.csv")
        d = d[(d.season_group == season) & d.supported]
        supported_count = len(d)
        d = d.dropna(subset=["map_longitude_0p5deg", "map_latitude_0p5deg", "g050_p50_c"]).copy()
        coords = d[["map_longitude_0p5deg", "map_latitude_0p5deg"]].to_numpy()
        values = d.g050_p50_c.to_numpy()
        dist = distance(xy, coords)
        nearest = dist.min(axis=1)
        z = interpolate(dist, values)
        keep = land_mask & (nearest <= 1000)
        shown = np.where(keep, z, np.nan).reshape(lon.shape)
        for poly in polygons:
            ax.add_patch(Polygon(poly, facecolor="#efefef", edgecolor="#bdbdbd", lw=0.35))
        im = ax.pcolormesh(
            lon,
            lat,
            shown,
            cmap=DEMAND_CMAP,
            vmin=19.5,
            vmax=30.6,
            shading="nearest",
            rasterized=True,
        )
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=values,
            cmap=DEMAND_CMAP,
            vmin=19.5,
            vmax=30.6,
            s=15,
            edgecolor="#7F7F7F",
            linewidth=0.4,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-55, 80)
        ax.set_aspect("equal")
        ax.set_title(label, loc="left", fontsize=12, weight="bold", pad=8)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_yticks([-30, 0, 30, 60])
        ax.set_ylabel("Latitude (°)")
        ax.text(
            0.99,
            0.02,
            f"{len(d)} geolocated cities; 1,000-km mask",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
        )
        cvdist = distance(coords, coords)
        np.fill_diagonal(cvdist, np.inf)
        pred = interpolate(cvdist, values)
        near = cvdist.min(axis=1)
        for city, obs, pr, km in zip(d.city, values, pred, near):
            cv.append(
                {
                    "city": city,
                    "season": season,
                    "observed_c": obs,
                    "heldout_prediction_c": pr,
                    "absolute_error_c": abs(pr - obs),
                    "nearest_training_city_km": km,
                }
            )
        for km in [500, 1000, 1500]:
            grid_summaries.append(
                {
                    "season": season,
                    "supported_city_summaries": supported_count,
                    "geolocated_city_summaries": len(d),
                    "excluded_without_scientific_coordinates": supported_count - len(d),
                    "mask_radius_km": km,
                    "supported_land_cells": int((land_mask & (nearest <= km)).sum()),
                    "land_cells": int(land_mask.sum()),
                }
            )
    cax = fig.add_axes([0.60, 0.945, 0.34, 0.012])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.set_label("Reconstructed demand centre (°C)", labelpad=1)
    cax.xaxis.set_label_position("top")
    ax = fig.add_subplot(gs[2, 0])
    ay = fig.add_subplot(gs[2, 1])
    cv = pd.DataFrame(cv)
    for j, (season, d) in enumerate(cv.groupby("season")):
        ax.scatter(
            d.observed_c,
            d.heldout_prediction_c,
            s=20,
            color=SEASON_COLORS[season],
            alpha=0.8,
            label=season.title(),
        )
        ay.scatter(
            d.nearest_training_city_km,
            d.absolute_error_c,
            s=20,
            color=SEASON_COLORS[season],
            alpha=0.8,
        )
    ax.plot([18, 32], [18, 32], ls="--", color="gray", lw=0.8)
    ax.set_xlabel("Observed city median (°C)")
    ax.set_ylabel("Held-out prediction (°C)")
    ax.legend(frameon=False, loc="upper left")
    ay.set_xlabel("Nearest training city (km)")
    ay.set_ylabel("Absolute held-out error (°C)")
    ay.axvline(1000, ls="--", color="gray", lw=0.8)
    ax.set_title("(c) Leave-one-city-out", loc="left", fontsize=12, weight="bold", pad=14)
    ay.set_title("(d) Transfer distance and error", loc="left", fontsize=12, weight="bold", pad=14)
    cv.to_csv(out / "spatial_interpolation_city_holdout.csv", index=False)
    pd.DataFrame(grid_summaries).to_csv(
        out / "spatial_interpolation_support_masks.csv", index=False
    )
    manifest.append(
        save(
            fig,
            out,
            14,
            "Figure S14 | Support-masked interpolation and out-of-city diagnostics. a,b, Four-neighbour inverse-distance-squared interpolation of observed warm- and cold-season city medians at Griffiths coefficient 0.50 °C−1. The common colour scale matches Figure 1a,b. Coloured land lies within 1,000 km of a geolocated observation. Grey land lies outside this mask. Points mark thirty warm-season and twenty-five cold-season geolocated city summaries. One supported Bangalore summary per season has a display-only coordinate and is excluded. c, Leave-one-city-out predictions without the held-out city in training. d, Absolute prediction error versus nearest training-city distance. Blue/terracotta denote cold/warm seasons. Data include errors and support masks at 500, 1,000 and 1,500 km.",
            ["curves/city_season_demand_support.csv", "natural_earth_110m_land"],
        )
    )

    audit = args.spatial_audit
    metrics = pd.read_csv(audit / "metrics_summary.csv")
    oof = pd.read_csv(audit / "oof_predictions.csv")
    paired = pd.read_csv(audit / "paired_cluster_bootstrap_vs_mean.csv")
    fig, a = canvas(
        [
            "Out-of-city RMSE",
            "Out-of-city MAE",
            "GPR interval calibration",
            "Coefficient sensitivity",
        ]
    )
    primary = metrics[metrics.target.eq("g050_p50_c")].copy()
    methods = ["global_training_mean", "idw2_4nn", "gpr_spherical_xyz"]
    method_labels = ["Training mean", "4-neighbour IDW²", "Spherical GPR"]
    scopes = ["all", "cold", "warm"]
    scope_labels = ["Pooled", "Cold", "Warm"]
    x = np.arange(len(scopes))
    width = 0.23
    for ax, metric, ylabel in [(a[0], "rmse_c", "RMSE (°C)"), (a[1], "mae_c", "MAE (°C)")]:
        for j, (method, label) in enumerate(zip(methods, method_labels)):
            q = primary[primary.model.eq(method)].set_index("season_scope").reindex(scopes)
            ax.bar(
                x + (j - 1) * width, q[metric], width, color=(COLORS + ["#777777"])[j], label=label
            )
        ax.set_xticks(x, scope_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 2.8)
    a[1].legend(frameon=False, fontsize=11, loc="upper left")

    gp = oof[(oof.target == "g050_p50_c") & (oof.model == "gpr_spherical_xyz")].copy()
    covered = gp.covered_95.astype(str).str.lower().eq("true")
    for mask, color, label in [(covered, COLORS[0], "Covered"), (~covered, COLORS[1], "Missed")]:
        q = gp[mask]
        a[2].vlines(q.observed_c, q.lower_95_c, q.upper_95_c, color=color, alpha=0.55, lw=0.7)
        a[2].scatter(
            q.observed_c, q.prediction_c, s=19, color=color, alpha=0.8, label=label, zorder=3
        )
    lo = min(gp.observed_c.min(), gp.lower_95_c.min())
    hi = max(gp.observed_c.max(), gp.upper_95_c.max())
    a[2].plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="gray")
    a[2].set_xlim(lo - 0.3, hi + 0.3)
    a[2].set_ylim(lo - 0.3, hi + 0.3)
    a[2].set_xlabel("Observed city median (°C)")
    a[2].set_ylabel("GPR prediction and 95% interval (°C)")
    coverage = covered.mean() * 100
    wilson_low = (
        primary[
            (primary.model == "gpr_spherical_xyz") & (primary.season_scope == "all")
        ].coverage_95_wilson_low.iloc[0]
        * 100
    )
    wilson_high = (
        primary[
            (primary.model == "gpr_spherical_xyz") & (primary.season_scope == "all")
        ].coverage_95_wilson_high.iloc[0]
        * 100
    )
    a[2].text(
        0.03,
        0.96,
        f"{covered.sum()}/{len(gp)} covered ({coverage:.1f}%)\nWilson 95%: {wilson_low:.1f}–{wilson_high:.1f}%",
        transform=a[2].transAxes,
        va="top",
    )
    a[2].legend(frameon=False, loc="lower right")

    pooled = paired[
        (paired.season_scope == "all") & paired.model.isin(["idw2_4nn", "gpr_spherical_xyz"])
    ].copy()
    target_order = ["g033_p50_c", "g050_p50_c", "g067_p50_c"]
    rows = []
    for target in target_order:
        for method in ["idw2_4nn", "gpr_spherical_xyz"]:
            rows.append(pooled[(pooled.target == target) & (pooled.model == method)].iloc[0])
    y = np.arange(len(rows))
    labels = [
        f'g = {r.target[1:4].replace("0", ".", 1)} ' + ("IDW²" if r.model == "idw2_4nn" else "GPR")
        for r in rows
    ]
    for offset, metric, marker, color, label in [
        (-0.10, "mae", "o", COLORS[0], "ΔMAE"),
        (0.10, "rmse", "^", COLORS[1], "ΔRMSE"),
    ]:
        center = np.array([r[f"delta_{metric}_c_model_minus_mean"] for r in rows])
        low = np.array([r[f"delta_{metric}_c_ci_low"] for r in rows])
        high = np.array([r[f"delta_{metric}_c_ci_high"] for r in rows])
        a[3].errorbar(
            center,
            y + offset,
            xerr=[center - low, high - center],
            fmt=marker,
            color=color,
            capsize=2,
            ms=4,
            label=label,
        )
    a[3].axvline(0, color="gray", lw=0.8, ls="--")
    a[3].set_yticks(y, labels)
    a[3].invert_yaxis()
    a[3].set_xlabel("Spatial model − training mean (°C)")
    a[3].set_xlim(-0.7, 1.25)
    a[3].plot(0.84, 2.15, marker="o", color=COLORS[0], ms=5)
    a[3].text(0.94, 2.15, "ΔMAE", va="center")
    a[3].plot(0.84, 2.55, marker="^", color=COLORS[1], ms=5)
    a[3].text(0.94, 2.55, "ΔRMSE", va="center")
    manifest.append(
        save(
            fig,
            out,
            15,
            "Figure S15 | Whole-city holdout audit of coordinate-based spatial prediction. a,b, Held-out root-mean-square error (RMSE) and mean absolute error (MAE) for the training mean, four-neighbour inverse-distance-squared interpolation (IDW²) and spherical Gaussian-process regression (GPR), at Griffiths coefficient 0.50 °C−1. c, GPR predictions and nominal 95% intervals for fifty-five held-out city–season summaries. Blue/terracotta mark covered/missed observations. Forty-five are covered (81.8%, Wilson interval 69.7–89.8%), with mean interval width 6.82 °C. d, Paired city-cluster bootstrap differences from the training-mean baseline in pooled MAE and RMSE under three Griffiths coefficients. Points and horizontal lines show estimates and intervals from fixed out-of-fold predictions. Negative values favour the spatial method.",
            [
                "spatial_method_audit/metrics_summary.csv",
                "spatial_method_audit/oof_predictions.csv",
                "spatial_method_audit/paired_cluster_bootstrap_vs_mean.csv",
            ],
        )
    )

    pd.DataFrame(manifest).to_json(
        out / "SUPPLEMENTARY_FIGURE_MANIFEST.json", orient="records", indent=2, force_ascii=False
    )
    print(
        json.dumps(
            {
                "figures": len(manifest),
                "panels": sum(x["panels"] for x in manifest),
                "output": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
