"""Render the actual own-baseline response and implementation results.

Scientific inputs are the completed annual, event, cost and portfolio tables.
No observations, service states, financial assumptions or parameters are changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from render_control_service_figures import (
    ALTERNATIVES,
    CONTROL_COLORS, CONTROL_MARKERS, CONTROL_ORDER, export_result,
    point_summary, six_panel_canvas, write_panel_summaries,
)


def read_table(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def dump_source(output, number, panel, frame):
    frame.to_csv(output / f"Figure_{number:02d}_panel_{panel}_source.csv", index=False)


def zero_axes(ax, *, x=False, y=True):
    if x:
        ax.axvline(0, color="#B0B0B0", linewidth=0.65, zorder=0)
    if y:
        ax.axhline(0, color="#B0B0B0", linewidth=0.65, zorder=0)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)


def portfolio_points(ax, frame, metric, label):
    values = frame.rename(columns={"climate_context": "context"})
    if len(values) != 12 or values.duplicated(["context", "control"]).any():
        raise ValueError("Expected four contexts by three controls, without aliases")
    records = point_summary(ax, values, metric, label, signed=True)
    return records


def render_five(results, output):
    event_path = results / "events/EVENT_SELECTIONS.csv"
    summary_path = results / "portfolio/portfolio_summary.csv"
    hourly_path = results / "portfolio/portfolio_hourly.csv"
    events = read_table(event_path)
    summary = read_table(summary_path)
    july = events.query("event_month == 7 and service_fraction == 1 and permission == 'native_zone'").copy()
    if len(july) != 108 or not july.baseline_service_qualified.eq(True).all():
        raise ValueError("Figure 5 July baseline qualification changed; do not infer unavailable as zero")
    no_response = (july.baseline_service_qualified.eq(True)
                   & july.service_qualified_available_response_kw.eq(0)
                   & july.selected_run_id.isna())
    # These are explicitly eligible, unperturbed fallback operations. Ineligible
    # baselines are never imputed or admitted by this display transformation.
    for column in ("recovery_extra_electric_kwh", "event_plus48h_net_electric_saving_kwh"):
        july.loc[no_response, column] = 0.0
    july["available_response_kw"] = july.service_qualified_available_response_kw
    full = summary.query("event_id == 'month_07' and dr_participation_fraction == 1").copy()
    if not full.all_baselines_service_qualified.eq(True).all():
        raise ValueError("Complete July portfolio baseline qualification is required")
    fig, axes = six_panel_canvas([
        "Response and recovery power", "Event-plus-recovery energy", "Zone-bound portfolio",
        "Building-bound portfolio", "Synchronized rebound", "Service during selected responses",
    ], height=207, bounds=[(19, 155, 157, 28), (19, 105, 67, 27), (109, 105, 67, 27),
                           (19, 56, 67, 27), (109, 56, 67, 27), (19, 14, 157, 22)])
    records = []
    hourly = read_table(hourly_path)
    trace = hourly.query("event_id == 'month_07' and scope == 'native_zone' and dr_participation_fraction == 1").copy()
    for control in CONTROL_ORDER:
        for context, style in (("PortAngeles_small2019", "-"), ("Seattle_small2019", "--")):
            d = trace[trace.control.eq(control) & trace.climate_context.eq(context)].sort_values("hour_index")
            axes[0].plot(d.hour_index + .5, d.electric_change_kwh,
                         color=CONTROL_COLORS[control], linestyle=style, linewidth=1.4)
    axes[0].axvspan(0, 2, color="#C8A36A", alpha=.2, zorder=0)
    axes[0].set_xlim(0, 50)
    axes[0].set_xticks([0, 2, 12, 24, 36, 48])
    axes[0].set_xlabel("Hours after event start", labelpad=3)
    axes[0].set_ylabel("Power change (kW)")
    zero_axes(axes[0])
    axes[0].set_xticks([0, 2, 12, 24, 36, 48])
    axes[0].text(.99, .08, "PTA solid / SEA dashed", ha="right", va="bottom",
                 transform=axes[0].transAxes, color="black", fontsize=11)
    records.extend(dict(panel="a", context=context, control=control, metric="available_response_kw",
                        finite_cases=len(g), median_display=float(g.available_response_kw.median()))
                   for (context, control), g in july.groupby(["context", "control"]))
    records += [dict(panel="b", **r) for r in point_summary(
        axes[1], july, "event_plus48h_net_electric_saving_kwh", "Net electricity saved (kWh)", signed=True)]
    for panel, ax, scope in (("c", axes[2], "native_zone"), ("d", axes[3], "building")):
        d = full[full.scope.eq(scope)].copy()
        records += [dict(panel=panel, **r) for r in portfolio_points(
            ax, d, "complete_portfolio_service_qualified_response_kw", "Available reduction (kW)")]
        dump_source(output, 5, panel, d)
    native = full[full.scope.eq("native_zone")].copy()
    records += [dict(panel="e", **r) for r in portfolio_points(
        axes[4], native, "recovery_synchronous_peak_extra_kw", "Recovery peak excess (kW)")]
    participating = summary.query("event_id == 'month_07' and scope == 'native_zone'")
    selected = july[july.selected_run_id.notna() & july.available_response_kw.gt(0)].copy()
    service_rows = []
    criteria = (("event", "mean", "Event mean"), ("event", "fixed_tail", "Event tail"),
                ("recovery", "mean", "Recovery mean"), ("recovery", "fixed_tail", "Recovery tail"))
    for criterion_index, (phase, metric, _) in enumerate(criteria):
        for offset, control in zip((-.22, 0, .22), CONTROL_ORDER):
            group = selected[selected.control.eq(control)]
            budget = pd.to_numeric(group[f"{phase}_{metric}_budget_c"]).to_numpy(float)
            reference_column = "mean_reference_mismatch_c" if metric == "mean" else "fixed_tail_reference_mismatch_c"
            baseline = pd.to_numeric(group[f"{phase}_baseline_{reference_column}"]).to_numpy(float) / budget
            response = pd.to_numeric(group[f"{phase}_perturbed_{reference_column}"]).to_numpy(float) / budget
            if not (np.isfinite(baseline).all() and np.isfinite(response).all()
                    and (baseline <= 1 + 1e-9).all() and (response <= 1 + 1e-9).all()):
                raise ValueError("Every displayed selected response must retain both service limits")
            x = criterion_index + offset + np.linspace(-.06, .06, len(group))
            for xx, before, after in zip(x, baseline, response):
                axes[5].plot([xx, xx], [before, after], color=CONTROL_COLORS[control],
                             linewidth=.65, alpha=.6, zorder=1)
            axes[5].scatter(x, baseline, s=17, facecolors="white", edgecolors=CONTROL_COLORS[control],
                            marker=CONTROL_MARKERS[control], linewidths=.75, zorder=2)
            axes[5].scatter(x, response, s=11, color=CONTROL_COLORS[control],
                            marker=CONTROL_MARKERS[control], linewidths=0, zorder=3)
            for row, before, after, limit in zip(group.itertuples(), baseline, response, budget):
                service_rows.append(dict(context=row.context, composition_id=row.composition_id, layout=row.layout,
                                         control=control, criterion=f"{phase}_{metric}",
                                         baseline_run_id=row.baseline_run_id, response_run_id=row.selected_run_id,
                                         budget_c=limit,
                                         baseline_mismatch_c=getattr(row, f"{phase}_baseline_{reference_column}"),
                                         response_mismatch_c=getattr(row, f"{phase}_perturbed_{reference_column}"),
                                         baseline_ratio=before, response_ratio=after))
            records.append(dict(panel="f", control=control, criterion=f"{phase}_{metric}",
                                selected_positive_cases=len(group), maximum_baseline_ratio=max(baseline),
                                maximum_response_ratio=max(response)))
    axes[5].axhline(1, color="#7F7F7F", linestyle="--", linewidth=.85)
    axes[5].set_xticks(range(4), [entry[2] for entry in criteria])
    axes[5].set_ylabel("Mismatch / limit")
    axes[5].set_ylim(0, 1.08)
    axes[5].set_xlim(-.5, 3.5)
    axes[5].set_yticks([0, .5, 1])
    dump_source(output, 5, "a", july)
    dump_source(output, 5, "a_hourly", trace)
    dump_source(output, 5, "b", july)
    dump_source(output, 5, "e", native)
    dump_source(output, 5, "f", pd.DataFrame(service_rows))
    dump_source(output, 5, "participation_supplement", participating)
    caption = (
        (
               "Figure 5 | Own-baseline July response and recovery on selected high-load weekdays. Response is the largest qualified positive "
               "power reduction among two amplitudes, with a two-hour event and 48-hour recovery. Qualification "
               "requires both baseline and perturbed operation to meet population and fixed-tail "
               "temperature-mismatch limits during the event and recovery. All shown baselines meet the window "
               "limits. a, Hourly perturbed-minus-own-baseline electricity for complete 108-office portfolios "
               "with native-zone commands. Shading marks the two-hour event. Solid and dashed lines denote Port "
               "Angeles and Seattle, respectively. The Miami and Delhi portfolios have zero change because no "
               "tested positive action qualifies. b, Single-office net electricity saving over 50 hours; points "
               "and whiskers denote cases and median interquartile ranges. c,d, Service-qualified response from "
               "standardized 108-office portfolios with native-zone or building-bound commands. e, Synchronized "
               "recovery excess peak with native-zone commands. f, Population and fixed-tail mismatch divided by "
               "their own event and recovery limits, for all selected positive native-zone July responses (16 "
               "uniform-average, 12 personal-centre and 12 LTPref conditions). Hollow and solid symbols identify "
               "baseline and perturbed operation. The dashed line is the qualification limit. Panel b uses nine "
               "composition–placement cases per control and context. Zero denotes an eligible baseline with no "
               "positive qualified tested action. Portfolios share declared composition, not observed city "
               "stock. MIA/DEL/PTA/SEA denote Miami/New Delhi/Port Angeles/Seattle."
    )
    )
    result = export_result(fig, axes, output, "Figure_05", caption=caption, sources={
        "EVENT_SELECTIONS.csv": event_path, "portfolio_summary.csv": summary_path,
        "portfolio_hourly.csv": hourly_path})
    write_panel_summaries(output, 5, records)
    plt.close(fig)
    return result


def nonalias(frame):
    return frame[~(frame.permission.eq("floor") & frame.context.str.contains("small2019"))].copy()


def render_six(results, output):
    cost_root = results / "costs"
    paths = {name: cost_root / (name + ".csv") for name in (
        "all_signed_paired_costs", "bill_implementation_boundaries",
        "command_scope_management_boundaries", "implementation_cost_scenarios")}
    costs = read_table(paths["all_signed_paired_costs"])
    bills = read_table(paths["bill_implementation_boundaries"])
    groups = read_table(paths["command_scope_management_boundaries"])
    packages = read_table(paths["implementation_cost_scenarios"])
    base = nonalias(costs[costs.scenario_id.eq("information_reuse__median")])
    direct = base.query("comparison_mode == 'direct' and permission == 'native_zone'").copy()
    common = base.query("comparison_mode == 'equal_service' and service_fraction == 1").copy()
    if len(common) != 126:
        raise ValueError("The common-service US nonalias comparison must have 126 matched rows")
    fig, axes = six_panel_canvas([
        "Direct service and bill", "Common-service bill", "Implementation scenarios",
        "Delhi charges", "Extra-group boundary", "Value of improved service",
    ], control_key=False, height=187,
                                bounds=[(19, 130, 67, 34), (109, 130, 67, 34),
                                        (19, 73, 87, 34), (130, 73, 46, 34),
                                        (19, 19, 67, 31), (109, 19, 67, 31)])
    fig.legend(handles=[Line2D([], [], marker=CONTROL_MARKERS[c], linestyle="none",
                               color=CONTROL_COLORS[c], markersize=5,
                               label={"personal_center": "Personal centre", "ltpref_interval": "LTPref interval"}[c])
                        for c in ALTERNATIVES], loc="upper center", bbox_to_anchor=(.5, .995), ncol=2)
    records = []
    for control in ALTERNATIVES:
        d = direct[direct.control.eq(control)]
        axes[0].scatter(d.delta_interval_coverage_percentage_points, d.delta_energy_charge_usd_year,
                        color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                        s=21, alpha=.85, linewidths=.35, edgecolors="white")
    axes[0].set_xlabel("Coverage change (p.p.)")
    axes[0].set_ylabel("Bill change (US$/year)")
    zero_axes(axes[0], x=True)
    # Each US context retains three compositions and placements at each distinct
    # permission. Small-office floor aliases do not add observations.
    for index, context in enumerate(("Miami", "PortAngeles_small2019", "Seattle_small2019")):
        for offset, control in zip((-.13, .13), ALTERNATIVES):
            d = common[common.context.eq(context) & common.control.eq(control)]
            values = d.delta_energy_charge_usd_year.to_numpy(float)
            axes[1].scatter(index + offset + np.linspace(-.035, .035, len(values)), values,
                            color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                            s=13, alpha=.75, linewidths=.25, edgecolors="white")
            q25, median, q75 = np.quantile(values, [.25, .5, .75])
            axes[1].errorbar(index + offset, median, yerr=[[median-q25], [q75-median]],
                             fmt=CONTROL_MARKERS[control], color=CONTROL_COLORS[control],
                             markersize=4, capsize=2, linewidth=1.2)
            records.append(dict(panel="b", context=context, control=control, n=len(values),
                                q25=q25, median=median, q75=q75))
    axes[1].set_xticks([0, 1, 2], ["MIA", "PTA", "SEA"])
    axes[1].set_ylabel("Bill change (US$/year)")
    axes[1].axhline(0, color="#B0B0B0", linewidth=.65)
    axes[1].yaxis.set_major_locator(MaxNLocator(4))
    amounts = np.sort(packages.annual_resource_cost_usd.to_numpy(float))
    cumulative = np.arange(1, len(amounts)+1)/len(amounts)*100
    axes[2].step(amounts, cumulative, where="post", color="#7F7F7F", linewidth=1.6)
    axes[2].fill_between(amounts, 0, cumulative, step="post", color="#7F7F7F", alpha=.13)
    axes[2].scatter(amounts, cumulative, s=5, color="#7F7F7F", alpha=.9, linewidths=0)
    light = packages.loc[packages.scenario_id.eq("existing_bms_light__median"), "annual_resource_cost_usd"].item()
    axes[2].axvline(light, color="#C8A36A", linewidth=1.1, linestyle="--")
    axes[2].set_xlabel("Resource cost (US$/year)")
    axes[2].set_ylabel("Scenarios at or below (%)")
    axes[2].set_xlim(-30, max(amounts)*1.04)
    axes[2].set_ylim(0, 103)
    axes[2].xaxis.set_major_locator(MaxNLocator(3))
    axes[2].text(.98, .08, "118 declared scenarios", transform=axes[2].transAxes,
                 ha="right", va="bottom", fontsize=11, color="black")
    delhi = bills.query("context == 'NewDelhi' and comparison_mode == 'direct' and permission == 'native_zone'")
    for control in ALTERNATIVES:
        d = delhi[delhi.control.eq(control)]
        axes[3].scatter(d.delta_bill_gas_slope_kwh_hhv,
                        d.delta_bill_intercept_INR_year_pf095 / 1000,
                        color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                        s=22, alpha=.75, linewidths=.3, edgecolors="white")
    axes[3].set_xlabel("Gas change\n(kWh HHV/year)", labelpad=3)
    axes[3].set_ylabel("Electric charge (10³ INR/year)")
    zero_axes(axes[3], x=True)
    d = groups.query("scenario_id == 'information_reuse__median' and comparison_mode == 'equal_service' "
                     "and service_fraction == 1 and extra_groups_over_uniform > 0 and climate_context != 'NewDelhi'").copy()
    d = d[~(d.scope.eq("floor") & d.climate_context.str.contains("small2019"))]
    if len(d) != 72:
        raise ValueError("Expected 72 additional-command-group US comparisons")
    d["minutes_per_group_year"] = d.signed_resource_manager_hour_allowance_per_extra_group_year * 60
    for control in ALTERNATIVES:
        selected = d[d.control.eq(control)]
        for count, chunk in selected.groupby("extra_groups_over_uniform"):
            axes[4].scatter(count + np.linspace(-.16, .16, len(chunk)), chunk.minutes_per_group_year,
                            color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                            s=17, alpha=.8, linewidths=.3, edgecolors="white")
    axes[4].set_xticks([2, 4, 14])
    axes[4].set_xlim(.8, 15)
    axes[4].set_xlabel("Extra independent groups")
    axes[4].set_ylabel("Funded time (min/group/year)")
    axes[4].axhline(0, color="#B0B0B0", linewidth=.65)
    axes[4].yaxis.set_major_locator(MaxNLocator(4))
    miami = costs.query("scenario_id == 'existing_bms_light__median' and comparison_mode == 'direct' "
                        "and permission == 'native_zone' and context == 'Miami'").copy()
    for control in ALTERNATIVES:
        selected = miami[miami.control.eq(control)]
        axes[5].scatter(selected.delta_interval_coverage_percentage_points,
                        selected.minimum_additional_annual_external_value_for_resource_breakeven_usd,
                        color=CONTROL_COLORS[control], marker=CONTROL_MARKERS[control],
                        s=21, alpha=.85, linewidths=.35, edgecolors="white")
    axes[5].set_xlabel("Coverage change (p.p.)")
    axes[5].set_ylabel("Break-even value (US$/year)")
    zero_axes(axes[5], x=True)
    for letter, data in zip("abcdef", (direct, common, packages, delhi, d, miami)):
        dump_source(output, 6, letter, data)
    for letter, data in (("a", direct), ("f", miami)):
        for control, group in data.groupby("control"):
            records.append(dict(panel=letter, control=control, n=len(group),
                                median_coverage_change_pp=group.delta_interval_coverage_percentage_points.median(),
                                median_bill_change_usd_year=group.delta_energy_charge_usd_year.median()))
    records.append(dict(panel="c", package_count=len(packages), light_package_usd_year=light))
    records.append(dict(panel="e", n=len(d), positive_allowances=int(d.minutes_per_group_year.gt(0).sum()),
                        maximum_manager_minutes_per_group_year=d.minutes_per_group_year.max()))
    caption = (
        (
               "Figure 6 | Service, energy bills and implementation resources. Control changes are relative to "
               "uniform-average control under matched conditions. a, Native-zone direct US controls: interval "
               "coverage versus annual energy-bill change. b, US bill changes under common temperature-mismatch "
               "requirements across distinct permissions (126 matched comparisons). Points and whiskers show "
               "cases and median interquartile ranges. c, Cumulative distribution of 118 declared resource-cost "
               "scenarios, with the US$294.70 existing-system package marked. The package includes annualized "
               "setup and recurring management. d, Delhi direct native-zone bill components: electricity charge "
               "and gas use, one matched pair per control, before assigning a gas tariff. e, Annual manager-time "
               "allowance per additional command group before fixed costs (72 US comparisons). Positive and "
               "negative values convert bill savings and increases into management time, respectively. f, Miami "
               "direct service changes and additional annual benefit needed to cover energy and the light "
               "package. Currency values use declared historical prices. Costs are implementation scenarios, not "
               "project quotes. MIA/PTA/SEA denote Miami/Port Angeles/Seattle."
    )
    )
    result = export_result(fig, axes, output, "Figure_06", caption=caption, sources=paths)
    write_panel_summaries(output, 6, records)
    plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = [render_five(args.results_root, args.output), render_six(args.results_root, args.output)]
    (args.output / "RESPONSE_COST_FIGURE_MANIFEST.json").write_text(
        json.dumps(figures, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": 2, "output": str(args.output), "actual_visual_review": "PENDING"}))


if __name__ == "__main__":
    main()
