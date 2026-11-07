"""Supplementary evidence figures from the contained, completed analysis tables."""

from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402 - initialize local imports before loading project modules

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.lines import Line2D  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.patches import Patch  # noqa: E402 - initialize local imports before loading project modules
from matplotlib.ticker import MaxNLocator  # noqa: E402 - initialize local imports before loading project modules
import numpy as np  # noqa: E402 - initialize local imports before loading project modules
import pandas as pd  # noqa: E402 - initialize local imports before loading project modules
from strengthened_figures import style, title, export, WIDTH, BLUE, PURPLE, GREEN, CORAL, GOLD  # noqa: E402 - initialize local imports before loading project modules

CONTROL = {
    "uniform_average": "Average",
    "personal_center": "Personal center",
    "personal_range": "Personal range",
}
COLORS = {"uniform_average": GOLD, "personal_center": BLUE, "personal_range": PURPLE}


class Sources:
    def __init__(self, raw, analysis):
        self.raw, self.analysis, self.paths = raw, analysis, []

    def read(self, name, raw=False):
        path = (self.raw if raw else self.analysis) / name
        if not path.is_file() and path.suffix == ".csv" and path.with_suffix(".csv.gz").is_file():
            path = path.with_suffix(".csv.gz")
        self.paths.append(path)
        return pd.read_csv(path, low_memory=False)

    def latest(self, folder, filename):
        paths = sorted((self.analysis / folder).glob("run_*/" + filename))
        if not paths:
            raise FileNotFoundError(folder + "/" + filename)
        if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}) != 1:
            raise ValueError(f"Conflicting diagnostic outputs: {folder}/{filename}")
        self.paths.append(paths[-1])
        return pd.read_csv(paths[-1])


def canvas(count=4, legend_gap=False):
    rows = count // 2
    height = 5.5 if rows == 2 else 7.0
    fig = plt.figure(figsize=(WIDTH, height))
    positions = ([0.625, 0.135] if legend_gap else [0.61, 0.15]) if rows == 2 else [0.745, 0.44, 0.135]
    axes = [
        fig.add_axes([x, y, 0.30, 0.29 if rows == 2 else 0.19])
        for y in positions
        for x in [0.17, 0.67]
    ]
    return fig, axes


def ecdf(ax, values, color, label=None):
    values = np.sort(pd.Series(values).dropna().to_numpy())
    ax.step(
        values,
        np.arange(1, len(values) + 1) / len(values),
        where="post",
        color=color,
        lw=1.2,
        label=label,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Cumulative fraction")


def common_legend(fig, labels, colors, y=0.035, columns=None):
    fig.legend(
        [Patch(facecolor=c) for c in colors],
        labels,
        loc="center",
        bbox_to_anchor=(0.56, y),
        ncol=columns or len(labels),
        columnspacing=1,
        handlelength=1,
        handletextpad=0.4,
    )


def s1(s):
    ash = s.read("demand/ASHRAE_RECORDS.csv", True)
    ct = s.read("demand/CTCD_RECORDS.csv", True)
    people = s.read("results/data_support/IDENTIFIED_AS_HRAE_RECORDS.csv")
    variance = s.read("results/observations/ASHRAE_NESTED_VARIANCE.csv", True)
    fig, (a, b, c, d) = canvas()
    for frame, color in [(ash, BLUE), (ct, PURPLE)]:
        a.hist(
            frame.reference_g050_c,
            bins=np.arange(10, 40.1, 0.5),
            density=True,
            histtype="step",
            color=color,
            linewidth=1.1,
        )
    a.set(xlabel="Demand center (°C)", ylabel="Density (°C⁻¹)", xlim=(10, 45))
    title(a, "a", "Record-level demand")
    coverage = (
        people
        if {"records", "lo", "hi"}.issubset(people.columns)
        else people.groupby("person").agg(
            records=("temperature", "size"), lo=("temperature", "min"), hi=("temperature", "max")
        )
    )
    counts = coverage.records
    ecdf(b, counts, GREEN)
    b.set_xlabel("Records per analysis unit")
    b.set_xscale("log")
    title(b, "b", "Repeated-record support")
    c.scatter(
        coverage.records, coverage.hi - coverage.lo, s=5, color=BLUE, alpha=0.35, linewidths=0
    )
    c.set(xlabel="Records per analysis unit", ylabel="Observed temperature\nspan (°C)", xscale="log")
    title(c, "c", "Temperature coverage")
    x = variance[variance.component.eq("unit_group")]
    names = {
        "record_weight": "Record",
        "unit_weight": "Unit",
        "building_weight": "Building",
        "source_weight": "Source",
    }
    for (weighting, group), paint in zip(x.groupby("weighting"), [BLUE, GOLD, GREEN, PURPLE]):
        d.plot(
            group.griffiths_coefficient,
            group.share * 100,
            marker="o",
            ms=3,
            lw=0.9,
            color=paint,
            label=names.get(weighting, weighting),
        )
    d.set(xlabel="Griffiths coefficient (°C⁻¹)", ylabel="Within-building\nvariance share (%)")
    title(d, "d", "Reconstruction sensitivity")
    d.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, columnspacing=0.5, handlelength=0.8
    )
    d.set_ylim(0, 100)
    a.legend([Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)], ["ASHRAE", "CTCD"], loc="upper right")
    return fig, "abcd"


def s2(s):
    cities = s.read("results/observations/CITY_OBSERVATION_SUMMARY.csv", True)
    policies = s.read("results/observation_figures/POLICY_RANGE_POSITIONS.csv", True)
    seasons = s.read("results/seasonal_command/SEASONAL_PLACE_EFFECTS.csv", True)
    rooms = s.read("results/shared_room/BURDEN_THRESHOLD_RESULTS.csv", True)
    fig, (a, b, c, d) = canvas()
    for ax in (b, d):
        bounds = ax.get_position()
        ax.set_position([0.64, bounds.y0, 0.33, bounds.height])
    for archive, g in cities.groupby("archive"):
        a.scatter(
            g.observation_records,
            g.demand_center_p90_c - g.demand_center_p10_c,
            s=10,
            alpha=0.65,
            color=BLUE if "ASH" in archive.upper() else PURPLE,
            label="ASHRAE" if "ASH" in archive.upper() else "CTCD",
        )
    a.set(xscale="log", xlabel="City records", ylabel="Demand 10th–90th\npercentile span (°C)")
    title(a, "a", "Variation within cities")
    a.set_ylim(0, 14)
    a.legend(loc="upper right", handlelength=0.7)
    p = policies[policies.griffiths.eq("g0.50")].pivot(
        index="policy", columns="position", values="proportion"
    )
    short = {name: name.split()[0].replace("USA", "United States") for name in p.index}
    left = np.zeros(len(p))
    for key, color in [("below", BLUE), ("within", GREEN), ("above", CORAL)]:
        b.barh(np.arange(len(p)), 100 * p[key], left=left, color=color, height=0.65)
        left += 100 * p[key]
    b.set_yticks(range(len(p)), [short[n] for n in p.index])
    b.invert_yaxis()
    b.set(xlabel="Demand centers (%)", xlim=(0, 100))
    b.set_xticks([0, 50, 100])
    title(b, "b", "Published temperature ranges")
    for source, g in seasons.groupby("demand_source"):
        c.scatter(
            g.cool_mean_c,
            g.warm_mean_c,
            s=10,
            alpha=0.7,
            color=BLUE if "ASH" in source.upper() else PURPLE,
        )
    lim = [
        min(seasons.cool_mean_c.min(), seasons.warm_mean_c.min()) - 1,
        max(seasons.cool_mean_c.max(), seasons.warm_mean_c.max()) + 1,
    ]
    c.plot(lim, lim, color="#777777", lw=0.6, ls="--")
    c.set(xlabel="Cool-season center (°C)", ylabel="Warm-season center (°C)", xlim=lim, ylim=lim)
    title(c, "c", "Seasonal place contrasts")
    for dataset, g in rooms[rooms.minutes.eq(30)].groupby("dataset"):
        d.plot(
            g.threshold,
            100 * g.false_acceptance_windows / g.windows,
            marker="o",
            ms=3,
            lw=1,
            label="Singapore" if dataset == "NUS" else "Independent",
        )
    d.set(xlabel="Mean-request threshold", ylabel="Windows with hidden\nindividual requests (%)")
    title(d, "d", "Averaging shared requests")
    d.legend(loc="upper right", handlelength=0.8)
    d.set_ylim(0, 19)
    b.set_ylim(len(p) - 0.5, -2.3)
    b.legend([Patch(fc=x) for x in [BLUE, GREEN, CORAL]], ["Below", "Within", "Above"], loc="upper center", ncol=3, columnspacing=0.6, handlelength=0.8)
    c.legend([Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)], ["ASHRAE", "CTCD"], loc="lower right")
    return fig, "abcd"


def s3(s):
    fig, axes = canvas(6)
    files = [
        "results/demand_validation/FINAL_TEST_METRICS.csv",
        "results/demand_validation/undated_acceptability/FINAL_METRICS.csv",
        "results/demand_validation/single_record/timed/FINAL_METRICS.csv",
        "results/demand_validation/single_record/undated/FINAL_METRICS.csv",
    ]
    headings = [
        "Later requests\nRepeated calibration",
        "Acceptance\nRepeated calibration",
        "Later requests\nSingle-record calibration",
        "Acceptance\nSingle-record calibration",
    ]
    for i, (path, heading) in enumerate(zip(files, headings)):
        x = s.read(path)
        x = x[x.status.eq("completed")].copy()
        labels = [
            ("Center" if r.center_model == "raw_center" else "Shrink")
            + " + "
            + ("fixed" if r.range_model == "fixed_width" else "range")
            for r in x.itertuples()
        ]
        ax = axes[i]
        ax.hlines(np.arange(len(x)), x.ba_lower95, x.ba_upper95, color="#777777", lw=0.8)
        ax.scatter(
            x.balanced_accuracy,
            np.arange(len(x)),
            c=[BLUE if k == "fixed_width" else PURPLE for k in x.range_model],
            s=19,
            zorder=3,
        )
        ax.set_yticks(range(len(x)), labels)
        ax.set_ylim(len(x) - 0.5, -0.5)
        ax.set(xlabel="Balanced accuracy", xlim=(0.25, 0.85))
        ax.set_xticks([0.3, 0.5, 0.7])
        ax.axvline(0.5, color="#999999", lw=0.5, ls=":")
        title(ax, "abcdef"[i], heading)
    contrasts = s.read("results/group_information_validation/ALL_PAIRED_INFORMATION_CONTRASTS.csv")
    x = contrasts[
        contrasts.width_policy.eq("shared_width")
        & contrasts.individual_model.eq("raw_center")
        & contrasts.group_model.eq("building_median")
    ].sort_values("design")
    names = {
        "timed_multi": "Later, repeated",
        "timed_single": "Later, single",
        "undated_multi": "Acceptance, repeated",
        "undated_single": "Acceptance, single",
    }
    axes[4].hlines(
        range(len(x)),
        x.paired_ba_difference_lower95,
        x.paired_ba_difference_upper95,
        color=BLUE,
        lw=1,
    )
    axes[4].scatter(x.balanced_accuracy_difference, range(len(x)), s=18, color=PURPLE)
    axes[4].set_yticks(range(len(x)), [names[z].replace(", ", "\n") for z in x.design])
    axes[4].invert_yaxis()
    axes[4].axvline(0, color="#777777", lw=0.6)
    axes[4].set_xlabel("Accuracy difference")
    title(axes[4], "e", "Individual minus building")
    proxy = s.read("results/demand_validation/single_record/timed/CONTINUOUS_PROXY_METRICS.csv")
    y = np.arange(len(proxy))
    axes[5].hlines(y, proxy.mae_lower95_c, proxy.mae_upper95_c, color=BLUE, lw=1)
    axes[5].scatter(proxy.mean_absolute_unit_proxy_error_c, y, s=20, color=PURPLE)
    axes[5].set_yticks(
        y, ["Individual" if n == "raw_center" else "Shrinkage" for n in proxy.center_model]
    )
    axes[5].invert_yaxis()
    axes[5].set_xlabel("Later-center proxy MAE (°C)")
    title(axes[5], "f", "Center prediction error")
    return fig, "abcdef"


def s4(s):
    profiles = s.read("results/observation_figures/DEMAND_COMPOSITION_PROFILES.csv", True)
    profiles = profiles[profiles.context.eq("Miami")]
    domain = s.read("results/target_domain_final/TARGET_DOMAIN_SUMMARY.csv", True)
    anchors = s.read("results/global_final/GLOBAL_25C_SERVICE_ANCHORS.csv", True)
    fig, (a, b, c, d) = canvas()
    fig.set_size_inches(WIDTH, 5.1)
    for ax, x, y in [(a, 0.13, 0.60), (b, 0.62, 0.60), (c, 0.13, 0.15), (d, 0.62, 0.15)]:
        ax.set_position([x, y, 0.35, 0.31])
    for ax, column, paint, label in [
        (a, "centre_c", BLUE, "Demand center (°C)"),
        (b, "range_width_c", PURPLE, "Personal range width (°C)"),
    ]:
        values = [g[column].to_numpy() for _, g in profiles.groupby("composition_id")]
        violin = ax.violinplot(values, positions=np.arange(1, 7), showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(paint)
            body.set_edgecolor("none")
            body.set_alpha(0.65)
        ax.scatter(range(1, 7), [np.median(v) for v in values], s=12, color="black", zorder=3)
        ax.set(xlabel="Demand composition", ylabel=label)
        ax.set_xticks(range(1, 7))
    title(a, "a", "Six demand compositions")
    title(b, "b", "Range-width distributions")
    for control, g in domain.groupby("control"):
        c.scatter(
            g.base_target_min_c,
            g.base_target_max_c,
            s=14,
            alpha=0.6,
            color=COLORS[control],
            label=CONTROL[control],
        )
    c.set(
        xlabel="Lowest zone target (°C)",
        ylabel="Highest zone target (°C)",
        xlim=(19.5, 27.5),
        ylim=(19.5, 34.5),
    )
    c.plot([20, 27], [20, 27], color="#999999", lw=0.6, ls=":")
    title(c, "c", "Target extent by control")
    for climate, g in anchors.groupby("climate"):
        d.scatter(g.anchor_mean_mismatch_c, g.anchor_tail_mismatch_c, s=14, color=GREEN, alpha=0.7)
    d.set(
        xlabel="Population reference\nmismatch (°C)", ylabel="Least-served reference\nmismatch (°C)"
    )
    title(d, "d", "Reference service limits")
    c.legend(loc="upper left", handlelength=0.8)
    return fig, "abcd"


def s5(s):
    levels = s.read("results/aligned_grid_sensitivity/ALIGNED_GRID_SUMMARY.csv")
    changes = s.read("results/aligned_grid_sensitivity/ALIGNED_GRID_LEVEL_CHANGES.csv")
    budgets = s.read("results/budget_frontiers/COMMON_BUDGET_MINIMUM_ENERGY.csv")
    tol = s.read("results/decision_tolerance_sensitivity/DECISION_TOLERANCE_SUMMARY.csv", True)
    fig, (a, b, c, d) = canvas()
    for col, label, paint in [
        ("qualified_set_changes", "Qualifying set", PURPLE),
        ("choice_changes", "Energy choice", BLUE),
    ]:
        a.plot(range(len(levels)), levels[col], marker="o", color=paint, lw=1, label=label)
    a.set_xticks(range(len(levels)), ["0.5", "0.25", "0.125"])
    a.set(xlabel="Aligned offset grid (°C)", ylabel="Settings with a change", ylim=(0, 85))
    a.legend(loc="upper center", ncol=1, handlelength=0.8)
    title(a, "a", "Choices across grids")
    for level, g in changes.groupby("to_level"):
        ecdf(
            b,
            -g.full_selected_energy_change_pct,
            BLUE if len(b.lines) == 0 else PURPLE,
            {"eighth": "0.125 °C", "quarter": "0.25 °C"}[str(level)],
        )
    b.set_xlabel("Energy improvement (%)")
    title(b, "b", "Refining energy minima")
    b.legend(loc="lower right", handlelength=0.8)
    # Separate absolute population and least-served budgets form a two-dimensional grid.
    x = budgets[budgets.budget_type.eq("absolute_mismatch") & budgets.permission.eq("native_zone")]
    if "full" in x.representation.unique():
        x = x[x.representation.eq("full")]
    table = (
        x.assign(feasible=x.status.eq("feasible"))
        .groupby(["fixed_tail_budget_c", "population_budget_c"])
        .feasible.sum()
        .unstack()
    )
    if table.empty:
        raise ValueError("Absolute service grid missing")
    image = c.imshow(
        table.to_numpy(), origin="lower", aspect="auto", cmap="Purples", vmin=0, vmax=54
    )
    c.set_xticks(range(len(table.columns)), [f"{z:g}" for z in table.columns])
    c.set_yticks(range(len(table)), [f"{z:g}" for z in table.index])
    c.set(xlabel="Population budget (°C)", ylabel="Least-served budget (°C)")
    title(c, "c", "Feasible settings out of 54")
    fig.colorbar(image, cax=fig.add_axes([0.49, 0.15, 0.014, 0.29]), ticks=[0, 27, 54])
    d.plot(tol.delta_c, tol.eligibility_set_changed_settings, marker="o", ms=3, color=PURPLE)
    d.plot(tol.delta_c, tol.selected_physical_plan_changed_settings, marker="o", ms=3, color=BLUE)
    d.set(xlabel="Added service margin (°C)", ylabel="Settings with a change", ylim=(0, 30))
    title(d, "d", "Original-library margins")
    return fig, "abcd"


def s6(s):
    x = s.read(
        "results/information_uncertainty_summary/SETTING_SELECTION_STABILITY_ALL_BRANCHES.csv"
    )
    fig, (a, b, c, d) = canvas(legend_gap=True)
    branch = x.library_and_scenario_design.unique()
    # Compare identical declared error scales where the scenario is available.
    for label, paint in [("pooled_archive_shared", BLUE), ("pooled_study_shared", PURPLE)]:
        names = [
            z
            for z in branch
            if ("study" in z) == ("study" in label)
            and "pooled" in z
            and "temporal" not in z
            and "single" not in z
        ]
        if not names:
            continue
        g = x[
            x.library_and_scenario_design.eq(names[0])
            & x.scenario.str.contains("SD0.5_personSD0.5", regex=False)
        ]
        if g.empty:
            continue
        ecdf(a, g.conditional_choice_disagreement_fraction, paint)
        ecdf(b, g.zone_mean_selected_excess_p95_c, paint)
        c.scatter(
            g.conditional_choice_disagreement_fraction,
            g.extra_selected_energy_median_kwh / 1000,
            color=paint,
            s=12,
            alpha=0.6,
        )
    for name in [z for z in branch if "temporal" in z or "single" in z]:
        g = x[
            x.library_and_scenario_design.eq(name)
            & x.scenario.eq("observed_temporal_proxy_shift_fraction1")
        ]
        ecdf(d, g.conditional_choice_disagreement_fraction, BLUE if "single" in name else PURPLE)
    a.set_xlabel("Choice-disagreement frequency")
    title(a, "a", "Shared-error structures")
    b.set_xlabel("95th-percentile service\nexceedance (°C)")
    title(b, "b", "Magnitude of exceedance")
    c.set(xlabel="Choice-disagreement frequency", ylabel="Median extra energy (MWh)")
    title(c, "c", "Resource consequence")
    d.set_xlabel("Choice-disagreement\nfrequency")
    title(d, "d", "Temporal error scenarios")
    if not a.lines or not d.lines:
        raise ValueError("Expected uncertainty branches not found")
    common_legend(fig, ["Archive shared", "Contributor shared"], [BLUE, PURPLE], y=0.500)
    d.legend(
        [Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)],
        ["Single record", "Repeated records"],
        loc="lower right",
        handlelength=0.8,
    )
    return fig, "abcd"


def s7(s):
    x = s.read("results/constrained_placement/PAIRED_WITHIN_FLOOR_PLACEMENT_EFFECTS.csv")
    fig, (a, b, c, d) = canvas(legend_gap=True)
    budgets = sorted(x.maximum_relocation_fraction.unique())
    for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
        g = x[x.control.eq(control)]
        for ax, column in [
            (a, "actual_relocation_fraction"),
            (b, "population_gain_c"),
            (c, "fixed_tail_gain_c"),
            (d, "energy_change_pct"),
        ]:
            for _, setting in g.groupby(["climate", "composition_id"]):
                setting = setting.sort_values("maximum_relocation_fraction")
                multiplier = 100 if column == "actual_relocation_fraction" else 1
                ax.plot(
                    setting.maximum_relocation_fraction * 100,
                    setting[column] * multiplier,
                    color=paint,
                    lw=0.45,
                    alpha=0.16,
                )
            median = g.groupby("maximum_relocation_fraction")[column].median()
            ax.plot(
                median.index * 100,
                median * (100 if column == "actual_relocation_fraction" else 1),
                color=paint,
                lw=1.5,
                marker="o",
                ms=3,
            )
    for ax in [a, b, c, d]:
        ax.set_xticks([v * 100 for v in budgets])
        ax.set_xlabel("Reassignment allowance (%)")
    a.set_ylabel("People reassigned (%)")
    b.set_ylabel("Population mismatch\nreduction (°C)")
    c.set_ylabel("Least-served mismatch\nreduction (°C)")
    d.set_ylabel("Purchased-energy change (%)")
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        ["Actual movement", "Population service", "Least-served service", "Energy consequence"],
    ):
        title(ax, letter, name)
    common_legend(fig, ["Personal center", "Personal range"], [BLUE, PURPLE], y=0.500)
    return fig, "abcd"


def s8(s):
    x = s.read("results/hvac_retuned/WITHIN_SYSTEM_STRATEGY_COMPARISONS.csv")
    operations = s.read("results/hvac_retuned/SYSTEM_OPERATION_SCORES.csv")
    unavailable = s.read("results/hvac_retuned/UNAVAILABLE_RETUNED_SETTINGS.csv")
    x = x[x.engineering_pass_demand.eq(True) & x.engineering_pass_average.eq(True)]
    fig, (a, b, c, d) = canvas(legend_gap=True)
    for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
        g = x[x.compared_control.eq(control)]
        a.scatter(
            g.population_mismatch_reduction_c,
            g.delta_purchased_energy_pct,
            s=15,
            color=paint,
            alpha=0.65,
        )
        b.scatter(
            g.fixed_tail_mismatch_reduction_c,
            g.delta_purchased_energy_pct,
            s=15,
            color=paint,
            alpha=0.65,
        )
        ecdf(c, g.heating_unmet_zone_hours_demand + g.cooling_unmet_zone_hours_demand, paint)
    a.set(xlabel="Population mismatch\nreduction (°C)", ylabel="Purchased-energy change (%)")
    b.set(xlabel="Least-served mismatch\nreduction (°C)", ylabel="Purchased-energy change (%)")
    c.set_xlabel("Unmet heating and cooling\n(zone-hours)")
    counts = operations.groupby("climate").engineering_pass.agg(["sum", "size"]).sort_index()
    for climate, g in unavailable.groupby("climate"):
        if climate not in counts.index:
            counts.loc[climate] = [0, len(g)]
        else:
            counts.loc[climate, "size"] += len(g)
    counts = counts.sort_index()
    positions = np.arange(len(counts))
    d.barh(positions, counts["size"], color="#e8e8e8", height=0.6)
    d.barh(positions, counts["sum"], color=GREEN, height=0.6)
    d.set_yticks(positions, [str(v).replace("_", " ").title() for v in counts.index])
    d.invert_yaxis()
    d.set_xlabel("Annual operations")
    d.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        ["Population service", "Least-served service", "Load tracking", "Annual operation support"],
    ):
        title(ax, letter, name)
    common_legend(fig, ["Personal center", "Personal range"], [BLUE, PURPLE], y=0.500)
    d.set_ylim(len(counts) - 0.5, -3.2)
    d.legend([Patch(fc="#e8e8e8"), Patch(fc=GREEN)], ["Planned", "Engineering accepted"], loc="upper center", bbox_to_anchor=(0.5, 1.03), handlelength=0.8)
    return fig, "abcd"


def s9(s):
    energy = s.latest("results/spatial_hvac_mechanism", "ENERGY_WINDOW_LEVELS.csv")
    tracking = s.latest("results/spatial_hvac_mechanism", "TRACKING_WINDOW_LEVELS.csv")
    vrp = s.latest("results/vrp_request_diagnostic", "AIRLOOP_REQUEST_FULFILLMENT.csv")
    fig, (a, b, c, d) = canvas(legend_gap=True)
    keys = ["climate", "composition_id", "control", "system", "placement"]
    annual = energy.groupby(keys, as_index=False).sum(numeric_only=True)
    tracking = tracking.groupby(keys, as_index=False).sum(numeric_only=True)
    systems = sorted(annual.system.unique())
    for system, paint in zip(systems, [BLUE, PURPLE]):
        g = annual[annual.system.eq(system)]
        a.scatter(
            g.cooling_electric_kwh / 1000,
            g.heating_electric_kwh / 1000,
            s=10,
            color=paint,
            alpha=0.4,
        )
        ecdf(b, g.joint_heating_cooling_hour_bins, paint)
        t = tracking[tracking.system.eq(system)]
        ecdf(c, t.heating_unmet_zone_hours + t.cooling_unmet_zone_hours, paint)
    a.set(xlabel="Cooling electricity (MWh)", ylabel="Heating electricity (MWh)")
    b.set_xlabel("Hours with heating\nand cooling")
    c.set_xlabel("Unmet heating and cooling\n(zone-hours)")
    valid = vrp.dropna(subset=["actual_to_vrp_request_ratio_min"])
    for placement, paint in [("mixed", BLUE), ("arranged", PURPLE)]:
        g = valid[valid.placement.eq(placement)]
        if len(g):
            ecdf(d, g.actual_to_vrp_request_ratio_min, paint)
    d.set_xlabel("Minimum outdoor-air\ndelivery / native request")
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "End-use relationship",
            "Coincident operation",
            "Annual load tracking",
            "Outdoor-air request",
        ],
    ):
        title(ax, letter, name)
    common_legend(
        fig, [{"pthp": "Zone heat pump", "vav": "VAV"}[v] for v in systems], [BLUE, PURPLE], y=0.500
    )
    d.legend(
        [Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)],
        ["Mixed", "Arranged"],
        loc="lower right",
        handlelength=0.7,
    )
    return fig, "abcd"


def s10(s):
    x = s.read("results/dr_capability_summary/RESOURCE_SUMMARY_BY_CLIMATE.csv")
    choices = s.read("results/dr_capability/ONE_ACTION_PER_EVENT_BUDGET_CHOICES.csv")
    fig, (a, b, c, d) = canvas(legend_gap=True)
    x = x[x.normal_energy_reference_factor.eq(1)]
    menus = [
        ("building_search", "Building", GOLD),
        ("zone_summary", "Zone summary", BLUE),
        ("zone_full", "Full zone", PURPLE),
    ]
    climates = sorted(x.climate.unique())
    for menu, label, paint in menus:
        g = x[x.control.eq(menu)].set_index("climate").reindex(climates)
        if g.available_event_power_mean_kw.isna().any():
            raise ValueError(f"Missing normal menu {menu}")
        a.plot(
            g.available_event_power_mean_kw,
            np.arange(len(climates)),
            marker="o",
            ms=3,
            color=paint,
            ls="none",
        )
        b.plot(
            g.selected_positive_response_events,
            np.arange(len(climates)),
            marker="o",
            ms=3,
            color=paint,
            ls="none",
        )
    for ax in [a, b]:
        ax.set_yticks(np.arange(len(climates)), [v.replace("_", " ").title() for v in climates])
        ax.invert_yaxis()
    a.set_xlabel("All-event mean power (kW)")
    b.set_xlabel("Positive-response events")
    g = choices[
        choices.control.eq("building_search")
        & choices.normal_energy_reference_factor.eq(1)
        & choices.selected_action.notna()
    ].copy()
    c.scatter(
        g.baseline_facility_mean_kw, g.absolute_event_facility_mean_kw, color=BLUE, s=10, alpha=0.5
    )
    limits = [
        min(g.baseline_facility_mean_kw.min(), g.absolute_event_facility_mean_kw.min()),
        max(g.baseline_facility_mean_kw.max(), g.absolute_event_facility_mean_kw.max()),
    ]
    c.plot(limits, limits, color="#888888", ls=":", lw=0.7)
    c.set(xlabel="Normal event power (kW)", ylabel="Action event power (kW)")
    ecdf(d, g.event_power_reduction_kw, BLUE)
    d.set_xlabel("Selected event reduction (kW)")
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        ["Available power", "Event availability", "Absolute event power", "Response size"],
    ):
        title(ax, letter, name)
    common_legend(fig, [v[1] for v in menus], [v[2] for v in menus], y=0.500)
    return fig, "abcd"


def s11(s):
    x = s.read("results/dr_information_review/PAIRED_EVENT_INFORMATION_REVIEW.csv")
    summary = s.read("results/dr_information_uncertainty/PAIRED_FROZEN_POWER_SUMMARY.csv")
    fig, (a, b, c, d) = canvas()
    primary = x[
        x.control.eq("building_search")
        & x.normal_energy_reference_factor.eq(1)
        & x.policy.eq("maximum_power")
    ]
    if primary.empty:
        raise ValueError("Primary information comparison missing")
    primary = primary.copy()
    for representation in ["full", "mean"]:
        primary[representation + "_count"] = primary[representation + "_positive_actions"].map(
            lambda value: len(json.loads(value))
        )
    counts = (
        primary.groupby(["full_count", "mean_count"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(6), columns=range(6), fill_value=0)
    )
    if counts.to_numpy().sum() != len(primary):
        raise ValueError("Action-count heatmap lost events")
    image = a.imshow(counts.to_numpy(), origin="lower", cmap="Purples", vmin=0)
    a.set(
        xticks=range(6),
        yticks=range(6),
        xlabel="Mean-screened actions",
        ylabel="Individual-screened actions",
    )
    fig.colorbar(image, cax=fig.add_axes([0.49, 0.61, 0.014, 0.29]), ticks=[0, 100, 200])
    failed = primary.groupby("climate").mean_selected_fails_full_review.sum().sort_index()
    b.barh(np.arange(len(failed)), failed, color=CORAL, height=0.65)
    b.set_yticks(np.arange(len(failed)), failed.index)
    b.invert_yaxis()
    b.set_xlabel("Mean choices exceeding\nindividual service limits")
    q = summary[
        summary.control.eq("building_search")
        & summary.budget.eq(1)
        & summary.support.eq("all_normal_available")
        & summary.stratum.eq("all")
    ]
    for prefix, paint in [("sourceSD", BLUE), ("studyGroupSD", PURPLE)]:
        g = q[q.scenario.str.startswith(prefix)]
        ecdf(c, g.paired_mean_minus_full_power_mean_kw, paint)
    c.set_xlabel("Frozen mean minus full\navailable power (kW)")
    temporal = (
        q[q.scenario.str.contains("temporal_proxy", regex=False)].copy().sort_values("scenario")
    )
    names = {
        "observed_temporal_proxy_shift_fraction0.5": "Repeated, half",
        "observed_temporal_proxy_shift_fraction1": "Repeated, full",
        "single_record_temporal_proxy_shift_fraction0.5": "Single, half",
        "single_record_temporal_proxy_shift_fraction1": "Single, full",
    }
    for i, row in enumerate(temporal.itertuples()):
        d.plot(
            [row.paired_mean_minus_full_power_p05_kw, row.paired_mean_minus_full_power_p95_kw],
            [i, i],
            color=PURPLE,
            lw=1.3,
        )
        d.scatter(row.paired_mean_minus_full_power_mean_kw, i, s=16, color=PURPLE)
    d.set_yticks(range(len(temporal)), [names[n] for n in temporal.scenario])
    d.invert_yaxis()
    d.set_xlabel("Frozen mean minus full\navailable power (kW)")
    for ax in [c, d]:
        ax.axvline(0, color="#888888", ls=":", lw=0.7)
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "Available action counts",
            "Where averaging matters",
            "Shared-error structures",
            "Temporal error scenarios",
        ],
    ):
        title(ax, letter, name)
    c.legend([Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)], ["Archive shared", "Contributor shared"], loc="lower right", handlelength=0.8)
    return fig, "abcd"


def s12(s):
    hourly = s.read("results/dr_terminal_recovery/PHYSICAL_PAIR_HOURLY_RESIDUALS.csv")
    totals = s.read("results/dr_terminal_recovery/PHYSICAL_PAIR_POST_RECOVERY_SUMMARY.csv")
    fig, (a, b, c, d) = canvas()
    terminal = (
        hourly.sort_values("audit_hour_index")
        .groupby(["normal_id", "run_id", "event_id"], sort=False)
        .head(1)
    )
    ecdf(a, terminal.operative_temperature_max_abs_difference_c, BLUE)
    a.set_xlabel("Temperature residual in\nlast recovery hour (°C)")
    maximum = hourly.groupby(
        ["normal_id", "run_id", "event_id"]
    ).operative_temperature_max_abs_difference_c.max()
    ecdf(b, maximum, PURPLE)
    b.set_xlabel("Largest follow-on\ntemperature residual (°C)")
    ecdf(c, totals.facility_electricity_post_recovery_change_kwh, GREEN)
    c.set_xlabel("Follow-on electricity\nchange (kWh)")
    for column, paint in [
        ("facility_electricity_interval_mean_kw_difference", BLUE),
        ("hvac_electricity_interval_mean_kw_difference", PURPLE),
    ]:
        quantiles = hourly.groupby("audit_hour_index")[column].quantile([0.05, 0.5, 0.95]).unstack()
        d.plot(np.arange(len(quantiles)), quantiles[0.5], color=paint, lw=1)
        d.fill_between(
            np.arange(len(quantiles)),
            quantiles[0.05],
            quantiles[0.95],
            color=paint,
            alpha=0.15,
            linewidth=0,
        )
    d.set(xlabel="Hours after accounting end", ylabel="Action minus baseline (kW)")
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "Last recovery hour",
            "Follow-on temperature",
            "Six-hour electricity",
            "Hourly electricity residual",
        ],
    ):
        title(ax, letter, name)
    lo, hi = d.get_ylim()
    d.set_ylim(lo, hi + 0.55 * (hi - lo))
    d.legend([Line2D([], [], color=BLUE), Line2D([], [], color=PURPLE)], ["Whole building", "HVAC"], loc="upper right", handlelength=0.8)
    return fig, "abcd"


def s13(s):
    direct = s.read("results/global_final/GLOBAL_DIRECT_COMPARISONS.csv", True)
    matched = s.read(
        "results/matched_service_final/MATCHED_REALIZED_SERVICE_TWO_METRIC_TOLERANCE_SUMMARY.csv",
        True,
    )
    complete = s.read(
        "results/energy_boundary_final/GLOBAL_ENERGY_BOUNDARY_DIRECT_COMPARISONS.csv", True
    )
    fig, (a, b, c, d) = canvas(legend_gap=True)
    for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
        g = direct[direct.alternative_control.eq(control)]
        a.scatter(
            -g.delta_mean_reference_mismatch_c,
            g.delta_facility_purchased_pct,
            s=13,
            color=paint,
            alpha=0.6,
        )
        b.scatter(
            -g.delta_fixed_tail_reference_mismatch_c,
            g.delta_facility_purchased_pct,
            s=13,
            color=paint,
            alpha=0.6,
        )
        t = matched[matched.alternative_control.eq(control)].sort_values("tolerance_c")
        c.plot(t.tolerance_c, t.matched_context_control_groups, marker="o", ms=3, color=paint, lw=1)
        z = complete[complete.alternative_control.eq(control)]
        d.scatter(
            z.delta_complete_facility_kwh / 1000,
            z.delta_complete_facility_carbon_kg / 1000,
            s=13,
            color=paint,
            alpha=0.6,
        )
    for ax in [a, b, d]:
        ax.axhline(0, color="#aaaaaa", ls=":", lw=0.6)
        ax.axvline(0, color="#aaaaaa", ls=":", lw=0.6)
    a.set(xlabel="Population mismatch\nreduction (°C)", ylabel="Purchased-energy change (%)")
    b.set(xlabel="Least-served mismatch\nreduction (°C)", ylabel="Purchased-energy change (%)")
    c.set(xlabel="Service-matching tolerance (°C)", ylabel="Matched comparisons", ylim=(0, 56))
    d.set(xlabel="Annual energy change (MWh)", ylabel="Annual operational\ncarbon change (t CO₂)")
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "Direct population service",
            "Direct tail service",
            "Matching both services",
            "Annual carbon inventory",
        ],
    ):
        title(ax, letter, name)
    common_legend(fig, ["Personal center", "Personal range"], [BLUE, PURPLE], y=0.500)
    return fig, "abcd"


def s14(s):
    annual = s.read(
        "results/seasonal_facility_energy_final/SEASONAL_COMMAND_ANNUAL_ENERGY_COMPLETE_FACILITY.csv",
        True,
    )
    seasons = s.read(
        "results/seasonal_facility_energy_final/SEASONAL_COMMAND_SEASON_METRICS_COMPLETE_FACILITY.csv",
        True,
    )
    fig, (a, b, c, d) = canvas(legend_gap=True)
    for fraction, ax in [(0.5, a), (1, b)]:
        g = annual[annual.seasonal_contrast_fraction.eq(fraction)]
        baseline = (
            g[g.control.eq("uniform_average")]
            .set_index(["climate", "composition_id"])
            .facility_purchased_kwh
        )
        for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
            h = g[g.control.eq(control)].set_index(["climate", "composition_id"])
            change = (h.facility_purchased_kwh / baseline.reindex(h.index) - 1) * 100
            ecdf(ax, change, paint)
        ax.set_xlabel("Annual energy change (%)")
    # Paired seasonal comparisons retain the source-specific command and demand shifts.
    g = seasons[seasons.seasonal_contrast_fraction.eq(1)]
    keys = ["climate", "composition_id", "local_season"]
    baseline = g[g.control.eq("uniform_average")].set_index(keys)
    for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
        h = g[g.control.eq(control)].set_index(keys)
        ref = baseline.reindex(h.index)
        c.scatter(
            ref.mean_reference_mismatch_c - h.mean_reference_mismatch_c,
            (h.facility_purchased_kwh - ref.facility_purchased_kwh) / 1000,
            s=8,
            color=paint,
            alpha=0.4,
        )
        d.scatter(
            ref.fixed_tail_reference_mismatch_c - h.fixed_tail_reference_mismatch_c,
            (h.facility_purchased_kwh - ref.facility_purchased_kwh) / 1000,
            s=8,
            color=paint,
            alpha=0.4,
        )
    c.set(
        xlabel="Seasonal population\nmismatch reduction (°C)", ylabel="Seasonal energy change (MWh)"
    )
    d.set(
        xlabel="Seasonal least-served\nmismatch reduction (°C)",
        ylabel="Seasonal energy change (MWh)",
    )
    for ax in [c, d]:
        ax.axhline(0, color="#aaaaaa", ls=":", lw=0.6)
        ax.axvline(0, color="#aaaaaa", ls=":", lw=0.6)
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "Half seasonal contrast",
            "Full seasonal contrast",
            "Seasonal population service",
            "Seasonal tail service",
        ],
    ):
        title(ax, letter, name)
    common_legend(fig, ["Personal center", "Personal range"], [BLUE, PURPLE], y=0.500)
    return fig, "abcd"


def s15(s):
    permission = s.read("results/spatial_final/GLOBAL_PERMISSION_COMPARISONS.csv", True)
    placement = s.read("results/spatial_final/GLOBAL_PLACEMENT_COMPARISONS.csv", True)
    fig, (a, b, c, d) = canvas(legend_gap=True)
    for frame, axes in [(permission, (a, b)), (placement, (c, d))]:
        for control, paint in [("personal_center", BLUE), ("personal_range", PURPLE)]:
            g = frame[frame.control.eq(control)]
            axes[0].scatter(
                -g.delta_mean_reference_mismatch_c,
                -g.delta_fixed_tail_reference_mismatch_c,
                color=paint,
                s=8,
                alpha=0.45,
            )
            axes[1].scatter(
                -g.delta_mean_reference_mismatch_c,
                g.delta_facility_purchased_pct,
                color=paint,
                s=8,
                alpha=0.45,
            )
        axes[0].set(
            xlabel="Population mismatch\nreduction (°C)",
            ylabel="Least-served mismatch\nreduction (°C)",
        )
        axes[1].set(
            xlabel="Population mismatch\nreduction (°C)", ylabel="Purchased-energy change (%)"
        )
    for ax, letter, name in zip(
        [a, b, c, d],
        "abcd",
        [
            "Command-scope service",
            "Command-scope energy",
            "Ordered-placement service",
            "Ordered-placement energy",
        ],
    ):
        title(ax, letter, name)
    common_legend(fig, ["Personal center", "Personal range"], [BLUE, PURPLE], y=0.500)
    return fig, "abcd"


def s16(s):
    x = s.read("results/response_mean_decomposition/RESPONSE_MEAN_DECOMPOSITION.csv", True)
    fig, (a, b, c, d) = canvas(legend_gap=True)
    total = x[x.group_dimension.eq("overall")].set_index("control").loc[list(CONTROL)]
    positions = np.arange(3)
    for ax, col, name in [
        (a, "raw_positive_zero_filled_mean_kw", "Physical response"),
        (b, "service_preserving_zero_filled_mean_kw", "Service-preserving power"),
    ]:
        ax.bar(positions, total[col], color=[GOLD, BLUE, PURPLE], width=0.6)
        ax.set_xticks(positions, ["Average", "Personal\ncenter", "Personal\nrange"])
        ax.set_ylabel("Fraction of cases" if ax is c else "Mean event power (kW)")
        title(ax, "abcd"[[a, b, c, d].index(ax)], name)
    climates = x[x.group_dimension.eq("climate")]
    for control, paint in [
        ("uniform_average", GOLD),
        ("personal_center", BLUE),
        ("personal_range", PURPLE),
    ]:
        g = climates[climates.control.eq(control)]
        c.scatter(
            g.qualified_positive_fraction,
            g.qualified_positive_conditional_mean_kw,
            color=paint,
            s=18,
        )
    c.set(xlabel="Positive-response fraction", ylabel="Conditional power (kW)", xlim=(-0.04, 1.04))
    title(c, "c", "Frequency and intensity")
    actions = (
        x[x.group_dimension.eq("action_id")]
        .pivot(
            index="group_value", columns="control", values="service_preserving_zero_filled_mean_kw"
        )
        .reindex(columns=list(CONTROL))
    )
    action_names = {
        "uniform_0.5": "Uniform 0.5 °C",
        "uniform_1.0": "Uniform 1 °C",
        "protected_1.0": "Protected 1 °C",
        "precondition_0.5_uniform_1.0": "Precondition +\nuniform",
        "precondition_0.5_protected_1.0": "Precondition +\nprotected",
    }
    image = d.imshow(actions.to_numpy(), aspect="auto", origin="upper", cmap="Purples")
    d.set_xticks(range(3), ["Average", "Personal\ncenter", "Personal\nrange"])
    d.set_yticks(range(len(actions)), [action_names[n] for n in actions.index])
    title(d, "d", "Action-specific power")
    d.set_position([0.67, 0.165, 0.30, 0.26])
    colorbar = fig.colorbar(
        image, cax=fig.add_axes([0.67, 0.065, 0.30, 0.014]), orientation="horizontal"
    )
    colorbar.set_label("All-case mean power (kW)", labelpad=0)
    common_legend(fig, list(CONTROL.values()), [GOLD, BLUE, PURPLE], y=0.500)
    return fig, "abcd"


FUNCTIONS = {
    1: s1,
    2: s2,
    3: s3,
    4: s4,
    5: s5,
    6: s6,
    7: s7,
    8: s8,
    9: s9,
    10: s10,
    11: s11,
    12: s12,
    13: s13,
    14: s14,
    15: s15,
    16: s16,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "output/from_raw")
    parser.add_argument("--analysis-root", type=Path, default=ROOT / "output/strengthening")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "output/publication/supplementary"
    )
    parser.add_argument("--figure", type=int, nargs="+", default=list(FUNCTIONS))
    parser.add_argument("--published-number", type=int)
    args = parser.parse_args(argv)
    fonts = style()
    if not all(
        p.resolve().is_relative_to(ROOT)
        for p in [args.raw_root, args.analysis_root, args.output_root]
    ):
        raise ValueError("All publication inputs and outputs must remain inside the project")
    for number in args.figure:
        published = args.published_number or number
        if args.published_number and len(args.figure) != 1:
            raise ValueError("A published number requires exactly one source figure")
        source = Sources(args.raw_root, args.analysis_root)
        fig, letters = FUNCTIONS[number](source)
        for axis in fig.axes:
            for coordinate, limits in [
                (axis.xaxis, axis.get_xlim()),
                (axis.yaxis, axis.get_ylim()),
            ]:
                ticks = coordinate.get_majorticklocs()
                inside = (ticks >= min(limits) - 1e-10) & (ticks <= max(limits) + 1e-10)
                if not inside.all():
                    coordinate.set_ticks(ticks[inside])
        export(fig, args.output_root, 100 + published, source.paths, letters, fonts)
        for extension in ["png", "pdf", "svg", "json"]:
            old = args.output_root / f"Figure_{100+published}.{extension}"
            old.replace(args.output_root / f"Figure_S{published:02}.{extension}")
        path = args.output_root / f"Figure_S{published:02}.json"
        receipt = json.loads(path.read_text())
        receipt["supplementary_plot_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
