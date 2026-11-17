"""Append two seasonal evidence panels to the current six-panel Figure 3 builder.

All inputs are explicit CLI paths. The current builder is imported unchanged;
its export is captured in memory, then the same six axes are repositioned.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

WIDTH_MM, HEIGHT_MM = 180.0, 215.0
BOUNDS_MM = [(19, 167, 67, 30), (109, 167, 67, 30),
             (19, 118, 157, 30), (19, 69, 43, 27),
             (76, 69, 43, 27), (133, 69, 43, 27),
             (19, 15, 67, 24), (109, 15, 67, 24)]


def artist_signature(axes):
    """Hash a-f scientific artists independently of their page positions."""
    def values(array):
        return np.ma.asarray(array).filled(np.nan).tolist()
    panels = []
    for ax in axes:
        panels.append({'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim()),
            'lines': [{'x': values(line.get_xdata()), 'y': values(line.get_ydata()),
                       'color': str(line.get_color()), 'marker': str(line.get_marker()),
                       'linestyle': line.get_linestyle()} for line in ax.lines],
            'collections': [{'offsets': values(c.get_offsets()),
                             'segments': [values(s) for s in c.get_segments()] if hasattr(c, 'get_segments') else [],
                             'facecolors': values(c.get_facecolors()), 'edgecolors': values(c.get_edgecolors())}
                            for c in ax.collections],
            'patches': [{'vertices': values(p.get_path().vertices),
                         'transform': values(p.get_patch_transform().get_matrix()),
                         'facecolor': p.get_facecolor(), 'hatch': p.get_hatch()} for p in ax.patches],
            'texts': [{'text': t.get_text(), 'position': t.get_position()} for t in ax.texts],
            'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(), 'title': ax.get_title(loc='left')})
    return hashlib.sha256(json.dumps(panels, sort_keys=True).encode()).hexdigest(), panels


def capture_original(builder, tables, annual_root, output):
    captured = {}
    export, close = builder.export_result, builder.plt.close
    def capture(fig, axes, directory, stem, *, caption, sources):
        captured.update(figure=fig, axes=axes, caption=caption, sources=sources)
        return {'status': 'CAPTURED_UNCHANGED_SIX_PANEL_FIGURE'}
    builder.export_result = capture
    builder.plt.close = lambda *args, **kwargs: None
    try:
        builder.render_figure_three(tables, annual_root, output)
    finally:
        builder.export_result, builder.plt.close = export, close
    if len(captured.get('axes', [])) != 6:
        raise ValueError('The current Figure 3 builder must produce its six original panels')
    return captured


def add_seasonal_panels(builder, figure, axes, args):
    diagnostic = pd.read_csv(args.seasonal_root / 'SEASONAL_SERVICE_BUDGET_DIAGNOSTIC.csv')
    periods = pd.read_csv(args.seasonal_root / 'PERIOD_TOTALS.csv')
    g = figure.add_axes(np.array(BOUNDS_MM[6]) / [WIDTH_MM, HEIGHT_MM, WIDTH_MM, HEIGHT_MM])
    h = figure.add_axes(np.array(BOUNDS_MM[7]) / [WIDTH_MM, HEIGHT_MM, WIDTH_MM, HEIGHT_MM])
    g.set_title('(g) Seasonal service checks', loc='left', pad=8)
    h.set_title('(h) Winter heating', loc='left', pad=8)
    selected = diagnostic[diagnostic.context.eq('Seattle_small2019')
        & diagnostic.comparison_mode.eq('equal_service') & diagnostic.service_fraction.eq(1)
        & diagnostic.data_status.eq('qualified')].copy()
    selected['both_pass'] = selected.both_budgets_pass.map({True: True, False: False, 'True': True, 'False': False})
    if selected.both_pass.isna().any():
        raise ValueError('Seasonal pass flags are incomplete')
    order = ['Annual', 'Spring', 'Summer', 'Autumn', 'Winter']
    g_summary = []
    for control, offset in zip(builder.CONTROL_ORDER, [-.19, 0, .19]):
        counts = []
        for period in order:
            group = selected[selected.control.eq(control) & selected.period.eq(period)]
            if len(group) != 9 or group.duplicated(['composition_id', 'layout']).any():
                raise ValueError('Panel g must retain all nine Seattle composition/placement strata')
            count = int(group.both_pass.sum())
            counts.append(count)
            g_summary.append({'control': control, 'period': period, 'declared_cases': 9,
                              'qualified_cases': 9, 'passing_both_reference_budgets': count,
                              'fraction_passing': count / 9})
        g.scatter(np.arange(5) + offset, counts, color=builder.CONTROL_COLORS[control],
                  marker=builder.CONTROL_MARKERS[control], s=22, linewidths=.4,
                  edgecolors='white', zorder=3)
    g.set_xlim(-.55, 4.55)
    g.set_ylim(-.5, 9.9)
    g.set_yticks([0, 3, 6, 9])
    g.set_xticks(range(5), order, rotation=30, ha='right')
    g.set_ylabel('Cases / 9')
    g.grid(axis='y', color='#D8D8D8', linewidth=.45)
    g.set_axisbelow(True)
    selected.to_csv(args.output / 'Figure_03_panel_g_all_seattle_cases.csv', index=False)
    pd.DataFrame(g_summary).to_csv(args.output / 'Figure_03_panel_g_counts.csv', index=False)
    winter = periods[periods.period_type.eq('season') & periods.period.eq('Winter')
        & periods.comparison_mode.eq('equal_service') & periods.service_fraction.eq(1)
        & periods.data_status.eq('qualified')].copy()
    if len(winter) != 108 or not winter.groupby(['context', 'control']).size().eq(9).all():
        raise ValueError('Panel h must retain all nine strata for each context and control')
    winter['heating_input_kwh'] = winter.heating_electric_kwh + winter.gas_hhv_kwh
    winter['winter_heating_share_pct'] = 100 * winter.heating_input_kwh / winter.purchased_site_kwh
    h_summary = builder.point_summary(h, winter, 'winter_heating_share_pct', 'Heating share (%)', percentage=True)
    h.set_xticks(range(4), ['Miami', 'New\nDelhi', 'Port\nAngeles', 'Seattle'])
    h.set_ylim(0, 100)
    h.set_yticks([0, 50, 100])
    winter.to_csv(args.output / 'Figure_03_panel_h_winter_heating_source.csv', index=False)
    pd.DataFrame(h_summary).to_csv(args.output / 'Figure_03_panel_h_summary.csv', index=False)
    axes.extend([g, h])
    return g_summary, h_summary


def render_monthly_support(builder, args, mode):
    """Reflow monthly service and carrier evidence to the final 180-mm width."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    table = pd.read_csv(args.seasonal_root / 'PERIOD_TOTALS.csv')
    selected = table[table.composition_id.eq(2026102001) & table.layout.eq('random_mixing')
        & table.comparison_mode.eq(mode) & table.period_type.eq('month')
        & table.data_status.eq('qualified')].copy()
    selected['month_number'] = selected.period.astype(int)
    height = 230.0
    figure = builder.plt.figure(figsize=(180 / 25.4, height / 25.4))
    axes = []
    stack_keys = ['heating_electric_kwh', 'cooling_electric_kwh', 'fans_electric_kwh', 'gas_hhv_kwh']
    stack_colors = ['#C78470', '#6F9EBC', '#C8A36A', '#AF94AD']
    for i, context in enumerate(builder.CONTEXT_ORDER):
        label = builder.CONTEXT_LABELS[context]
        left = figure.add_axes(np.array([19, 184-i*54, 67, 30]) / [180, height, 180, height])
        right = figure.add_axes(np.array([109, 184-i*54, 67, 30]) / [180, height, 180, height])
        axes.extend([left, right])
        left.set_title(f'({chr(97+2*i)}) {label}: service', loc='left', pad=8)
        right.set_title(f'({chr(98+2*i)}) {label}: energy', loc='left', pad=8)
        block = selected[selected.context.eq(context)]
        baseline = block[block.control.eq('uniform_average')].sort_values('month_number')
        if len(baseline) == 12:
            right.stackplot(baseline.month_number.to_numpy(), *[baseline[k].to_numpy()/1000 for k in stack_keys],
                            colors=stack_colors, alpha=.85, linewidth=0, zorder=1)
        elif len(baseline) != 0:
            raise ValueError('A qualified annual trajectory must retain twelve months')
        if block.empty:
            for ax in [left, right]:
                ax.text(.5, .5, 'No qualified trajectory', transform=ax.transAxes, ha='center', va='center')
        for control in builder.CONTROL_ORDER:
            values = block[block.control.eq(control)].sort_values('month_number')
            if len(values) == 0:
                continue
            if len(values) != 12:
                raise ValueError('A qualified annual trajectory must retain all twelve months')
            size = {'uniform_average': 4.5, 'personal_center': 3.7, 'ltpref_interval': 3.0}[control]
            for ax, metric, scale in [(left, 'mean_reference_mismatch_c', 1), (right, 'purchased_site_kwh', .001)]:
                ax.plot(values.month_number, values[metric]*scale, color=builder.CONTROL_COLORS[control],
                        marker=builder.CONTROL_MARKERS[control], linewidth=1.3, markersize=size,
                        markerfacecolor='white', markeredgewidth=.7, zorder=4)
        if block.groupby('control').run_id.first().nunique() == 1:
            left.text(.02, .96, 'All three identical', transform=left.transAxes, va='top', fontsize=11)
            low, high = left.get_ylim()
            left.set_ylim(low, high + .2*(high-low))
        left.set_ylabel('Mismatch (°C)')
        right.set_ylabel('Energy (MWh)')
        for ax in [left, right]:
            ax.set_xlim(.8, 12.2)
            ax.set_xticks([1, 4, 7, 10], ['January', 'April', 'July', 'October'], rotation=25, ha='right')
            ax.grid(axis='y', color='#D8D8D8', linewidth=.45)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(builder.MaxNLocator(3))
    figure.legend([Line2D([], [], color=builder.CONTROL_COLORS[c], marker=builder.CONTROL_MARKERS[c],
                         linestyle='-', lw=1.3, markersize=4, markerfacecolor='white') for c in builder.CONTROL_ORDER],
                  [builder.CONTROL_LABELS[c] for c in builder.CONTROL_ORDER], loc='upper center',
                  bbox_to_anchor=(.5, .995), ncol=3, frameon=False, columnspacing=.9, handletextpad=.3)
    figure.legend([Patch(facecolor=color, alpha=.85) for color in stack_colors],
                  ['Heating electricity', 'Cooling electricity', 'Fans', 'Gas HHV'], loc='lower center',
                  bbox_to_anchor=(.5, .005), ncol=4, frameon=False, columnspacing=.8, handletextpad=.3,
                  handlelength=1.1)
    label = 'direct control operation' if mode == 'direct' else 'common-service annual selections'
    stem = 'Supplementary_monthly_direct' if mode == 'direct' else 'Supplementary_monthly_common_service'
    caption = ('Monthly service and purchased HVAC energy: ' + label + '. '
        'The first demand composition, random-mixing placement and zone-level commands are shown in four building–weather contexts. '
        'a,c,e,g, Population-weighted absolute temperature-reference mismatch during occupied hours. '
        'b,d,f,h, Monthly energy: Uniform average operation is stacked by heating electricity, cooling electricity, fan electricity and gas HHV; '
        'lines show HVAC electricity plus gas HHV for each control. Every qualified trajectory contains all twelve months and 8,760 hours. '
        'Identical selections overlap; trajectories failing engineering checks are left blank. HVAC denotes heating, ventilation and air conditioning; HHV denotes higher heating value. '
        'Each context retains its prototype’s raw energy totals. Prototype occupancy weights are shared among assigned demand profiles.')
    result = builder.export_result(figure, axes, args.output, stem,
                                   caption=caption, sources={'PERIOD_TOTALS.csv': args.seasonal_root / 'PERIOD_TOTALS.csv',
                                                             'current_main_builder': args.main_builder},
                                   documented_empty_axes=[ax for ax in axes if not ax.has_data()])
    selected.to_csv(args.output / (stem + '_source.csv'), index=False)
    builder.plt.close(figure)
    return result


CAPTION = 'Figure 3 | Energy, carbon and seasonal service. All panels allow independent commands in existing thermal zones. a,b, Direct site-energy and operational-carbon changes from Uniform average control. Small points show composition–placement cases; larger symbols and whiskers show medians and interquartile ranges. c, Positive and negative building-hour electricity (solid) and gas HHV (hatched) changes were summed over each matched direct-control year. Bars show each component\'s composition–placement median separately; symbols show each case\'s net site-energy change. d,e, Minimum-energy choices from four variants at common-service fraction 1, constraining population-mean and fixed-tail reference mismatch. f, Qualifying choices among 36 context–composition–placement cases as limits tighten. Panels g,h retain the annual minimum-energy selections at common-service fraction 1. g, Counts meeting annual or corresponding seasonal common-25 °C reference values for both mismatch measures: nine Seattle cases per control. Selected plans and annual tail membership remain fixed. h, Winter heating share: (heating electricity + gas HHV)/(HVAC electricity + gas HHV); nine cases per context/control. Winter is December–February. Carbon uses fixed regional factors. HVAC denotes heating, ventilation and air conditioning; HHV denotes higher heating value.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-builder', type=Path, required=True)
    parser.add_argument('--annual-root', type=Path, required=True)
    parser.add_argument('--seasonal-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('current_main_figure_builder', args.main_builder)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    tables = builder.read_annual_tables(args.annual_root)
    captured = capture_original(builder, tables, args.annual_root, args.output)
    figure, axes = captured['figure'], captured['axes']
    before, original_artists = artist_signature(axes)
    figure.set_size_inches(WIDTH_MM / 25.4, HEIGHT_MM / 25.4, forward=True)
    for ax, bounds in zip(axes, BOUNDS_MM[:6]):
        ax.set_position(np.array(bounds) / [WIDTH_MM, HEIGHT_MM, WIDTH_MM, HEIGHT_MM])
    original_caption = captured['caption']
    (args.output / 'Figure_03_original_af_caption.txt').write_text(original_caption + '\n', encoding='utf-8')
    g_summary, h_summary = add_seasonal_panels(builder, figure, axes, args)
    after, _ = artist_signature(axes[:6])
    if before != after:
        raise ValueError('An original a-f scientific artist or axis value changed')
    sources = captured['sources'] | {
        'current_main_builder': args.main_builder,
        'PERIOD_TOTALS.csv': args.seasonal_root / 'PERIOD_TOTALS.csv',
        'SEASONAL_SERVICE_BUDGET_DIAGNOSTIC.csv': args.seasonal_root / 'SEASONAL_SERVICE_BUDGET_DIAGNOSTIC.csv',
        'COMMON25_PERIOD_BUDGETS.csv': args.seasonal_root / 'COMMON25_PERIOD_BUDGETS.csv'}
    result = builder.export_result(figure, axes, args.output, 'Figure_03_extended', caption=CAPTION, sources=sources)
    qa = {'status': 'EXPORTED_PENDING_IMAGE_REVIEW', 'page_mm': [WIDTH_MM, HEIGHT_MM],
          'original_af_artists_identical': before == after, 'original_af_sha256': before,
          'data_bounds_mm': BOUNDS_MM, 'original_af_artist_records': original_artists,
          'g_summary': g_summary, 'h_summary': h_summary,
          'seasonal_definition': 'Diagnostic of retained annual plans against same-period common25 references; unchanged annual tail; no reselection',
          'winter_heating_definition': '(heating electricity + gas HHV) / (HVAC electricity + gas HHV)',
          'caption_word_count': len(CAPTION.split()),
          'layout': result['layout_audit']}
    (args.output / 'EXTENSION_GEOMETRY_QA.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: qa[k] for k in ['status', 'page_mm', 'original_af_artists_identical', 'caption_word_count']}, indent=2))
    print(json.dumps({'text_outside_canvas': len(result['layout_audit']['text_outside_canvas']),
                      'text_overlap_candidates': len(result['layout_audit']['text_overlap_candidates'])}, indent=2))
    builder.plt.close(figure)
    for mode in ['direct', 'equal_service']:
        si = render_monthly_support(builder, args, mode)
        print(json.dumps({'mode': mode, 'text_overlap_candidates': len(si['layout_audit']['text_overlap_candidates']),
                          'text_outside_canvas': len(si['layout_audit']['text_outside_canvas'])}, indent=2))


if __name__ == '__main__':
    main()
