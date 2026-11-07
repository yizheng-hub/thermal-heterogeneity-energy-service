"""Extract complete monthly/meteorological-season evidence from frozen annual runs.

No simulation, control selection, profile reassignment, or physics is changed.
Run with --help; all inputs are supplied as paths and all outputs stay in --output.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

KEYS = ["context", "composition_id", "layout", "permission", "control"]
PROFILE_KEYS = KEYS[:3]
CONTROLS = ["uniform_average", "personal_center", "ltpref_interval"]
LABELS = ["Uniform average", "Personal centre", "Personal range"]
COLORS = ["#7F7F7F", "#C8A36A", "#7FA995"]
CONTEXTS = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
CONTEXT_LABELS = ["Miami", "New Delhi", "Port Angeles", "Seattle"]
SEASONS = {"Spring": [3, 4, 5], "Summer": [6, 7, 8],
           "Autumn": [9, 10, 11], "Winter": [12, 1, 2]}
CARRIERS = ["hvac_electric_kwh", "gas_hhv_kwh", "heating_electric_kwh",
            "cooling_electric_kwh", "fans_electric_kwh", "pumps_electric_kwh",
            "electricity_hvac_meter_kwh"]
SERVICE = ["occupied_native_person_hours", "mean_reference_mismatch_c",
           "fixed_tail_reference_mismatch_c", "modeled_interval_person_hour_coverage"]
RETAINED = {"hvac_electric_kwh": "annual_electric_kwh",
            "gas_hhv_kwh": "annual_gas_hhv_kwh",
            "heating_electric_kwh": "annual_heating_electric_kwh",
            "cooling_electric_kwh": "annual_cooling_electric_kwh",
            "fans_electric_kwh": "annual_fans_electric_kwh",
            "purchased_site_kwh": "annual_site_kwh",
            "operational_co2_kg": "annual_operational_co2_kg"}
PAIR_KEYS = PROFILE_KEYS + ['permission', 'comparison_mode', 'service_fraction',
                            'alternative', 'reference']


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def period_definitions():
    return ([('month', f'{month:02d}', [month]) for month in range(1, 13)]
            + [('season', name, months) for name, months in SEASONS.items()]
            + [('annual', 'Annual', list(range(1, 13)))])


def match_comparison_regimes(compared):
    """Compare direct and common-service gaps on identical spatial conditions."""
    compared = compared[~(compared.context.isin(['PortAngeles_small2019', 'Seattle_small2019'])
                          & compared.permission.eq('floor'))]
    keys = ['context', 'composition_id', 'layout', 'permission', 'alternative']
    direct = compared[compared.comparison_mode.eq('direct') & compared.status.eq('matched')]
    common = compared[compared.comparison_mode.eq('equal_service')
                      & compared.service_fraction.eq(1) & compared.status.eq('matched')]
    return direct.merge(common, on=keys, suffixes=('_direct', '_common'), validate='one_to_one')


def export_comparison_regimes(args):
    paired = match_comparison_regimes(pd.read_csv(args.annual_root / 'CONTROL_COMPARISONS.csv'))
    if getattr(args, 'expected_layouts', 3) == 3 and len(paired) != 114:
        raise ValueError('Expected 114 like-for-like direct/common-service comparisons')
    paired.to_csv(args.output / 'PAIRED_COMPARISON_REGIMES.csv', index=False)
    direct, common = paired.delta_site_pct_direct.abs(), paired.delta_site_pct_common.abs()
    report = {'status': 'PASS', 'matched_conditions': len(paired),
              'matching_keys': ['context', 'composition_id', 'layout', 'permission', 'alternative'],
              'small_office_floor_aliases_removed': True,
              'median_absolute_direct_energy_difference_pct': float(direct.median()),
              'median_absolute_common_service_energy_difference_pct': float(common.median()),
              'absolute_energy_gap_smaller': int((common < direct - 1e-8).sum()),
              'absolute_energy_gap_equal': int(((common - direct).abs() <= 1e-8).sum()),
              'absolute_energy_gap_larger': int((common > direct + 1e-8).sum())}
    (args.output / 'COMPARISON_REGIMES_QA.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def calculate(data, group, tail, factor):
    """Independently accumulate numerators with the locked native people weights."""
    positions = {str(z).strip().upper(): i for i, z in enumerate(data["zone_ids"])}
    zone = np.array([positions[str(z).strip().upper()] for z in group.zone_id])
    counts = np.bincount(zone, minlength=data["temperatures"].shape[1])
    native = data["people"] * data["occupied"].astype(bool)[:, None]
    weight = native[:, zone] / counts[zone]
    error = np.abs(data["temperatures"][:, zone] - group.centre_c.to_numpy())
    excess = (np.maximum(group.lower_c.to_numpy() - data["temperatures"][:, zone], 0)
              + np.maximum(data["temperatures"][:, zone] - group.upper_c.to_numpy(), 0))
    error_weighted = error * weight
    covered_weighted = (excess <= 1e-9) * weight
    rows = []
    tail_parts = {kind: ([], []) for kind in ['month', 'season']}
    for kind, period, months in period_definitions():
        mask = np.isin(data["month"], months)
        denominators = weight[mask].sum(axis=0)
        numerators = error_weighted[mask].sum(axis=0)
        if (denominators <= 0).any():
            raise ValueError("Every profile must retain occupied time in every reported period")
        people_hours = float(denominators.sum())
        item = {"period_type": kind, "period": period,
                "months": "|".join(map(str, months)), "hours": int(mask.sum()),
                "occupied_native_person_hours": people_hours,
                "reference_mismatch_native_person_c_hours": float(numerators.sum()),
                "covered_native_person_hours": float(covered_weighted[mask].sum()),
                "mean_reference_mismatch_c": float(numerators.sum() / people_hours),
                "fixed_tail_reference_mismatch_c": float((numerators[tail] / denominators[tail]).mean()),
                "modeled_interval_person_hour_coverage": float(covered_weighted[mask].sum() / people_hours)}
        item.update({key: float(data[key][mask].sum()) for key in CARRIERS})
        item["purchased_site_kwh"] = item["hvac_electric_kwh"] + item["gas_hhv_kwh"]
        item["operational_co2_kg"] = (item["hvac_electric_kwh"] * factor.electricity_co2_kg_kwh
                                      + item["gas_hhv_kwh"] * factor.gas_co2_kg_kwh_hhv)
        rows.append(item)
        if kind in tail_parts:
            tail_parts[kind][0].append(numerators[tail])
            tail_parts[kind][1].append(denominators[tail])
    # Tail membership is fixed annually. Its endpoint is the equal mean of 27
    # profile-level occupied-time averages, never a second native-weighted mean.
    annual_tail_rollups = {kind: float((np.sum(parts[0], axis=0) / np.sum(parts[1], axis=0)).mean())
                          for kind, parts in tail_parts.items()}
    return rows, annual_tail_rollups


def check_value(checks, identity, metric, method, actual, expected):
    passed = bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-7))
    checks.append(identity | {"metric": metric, "method": method,
                              "actual": float(actual), "expected": float(expected),
                              "absolute_error": float(abs(actual - expected)), "passed": passed})
    if not passed:
        raise ValueError(f"Annual closure failed: {identity}, {metric}, {method}")


def validate_selection_grid(selections, direct, expected_layouts=3):
    layouts = {'random_mixing', 'demand_ordered'}
    if expected_layouts == 3:
        layouts.add('source_structured')
    elif expected_layouts != 2:
        raise ValueError('Expected two comparable layouts or the original three layouts')
    expected = 4 * 3 * expected_layouts * 3
    native_direct = direct[direct.permission.eq('native_zone')]
    if (len(selections) != expected * 2 or len(native_direct) != expected
            or selections.duplicated(KEYS + ['comparison_mode']).any()
            or set(selections.context) != set(CONTEXTS)
            or set(selections.layout) != layouts
            or set(selections.control) != set(CONTROLS)
            or set(selections.comparison_mode) != {'direct', 'equal_service'}):
        raise ValueError(f'Expected all 4 contexts x 3 compositions x {expected_layouts} layouts '
                         'x 3 controls x 2 modes, with unique native-zone cells')
    compositions = set(selections.composition_id)
    if len(compositions) != 3:
        raise ValueError('Expected the same three declared compositions throughout')
    for _, group in selections.groupby(['context', 'layout', 'control', 'comparison_mode']):
        if set(group.composition_id) != compositions:
            raise ValueError('A seasonal comparison changed the declared compositions')


def temporal_compensation(pairs):
    """Summarize seasonal additions and savings for each fixed annual pair."""
    expected_periods = set(SEASONS)
    rows = []
    for key, group in pairs.groupby(PAIR_KEYS, sort=True, dropna=False):
        record = dict(zip(PAIR_KEYS, key))
        annual = group[group.period_type.eq('annual')]
        seasons = group[group.period_type.eq('season')]
        if len(annual) != 1 or set(seasons.period) != expected_periods or len(seasons) != 4:
            raise ValueError(f'Each temporal pair needs one annual and four seasonal rows: {record}')
        for column in ['pair_status', 'alternative_status', 'reference_status',
                       'alternative_run_id', 'reference_run_id', 'same_physical_operation']:
            values = group[column].drop_duplicates()
            if len(values) != 1:
                raise ValueError(f'Temporal pair metadata changed within the year: {record}, {column}')
            record[column] = values.iloc[0]
        matched = record['pair_status'] == 'matched'
        for metric, unit in [('purchased_site_kwh', 'site_kwh'),
                             ('operational_co2_kg', 'operational_co2_kg')]:
            annual_value = annual['delta_' + metric].iloc[0] if matched else np.nan
            values = (seasons.set_index('period')['delta_' + metric]
                      .reindex(SEASONS).astype(float)) if matched else pd.Series(np.nan, index=SEASONS)
            for season, value in values.items():
                record[season.lower() + '_delta_' + unit] = value
            added = float(values.clip(lower=0).sum()) if matched else np.nan
            avoided = float((-values.clip(upper=0)).sum()) if matched else np.nan
            net = float(values.sum()) if matched else np.nan
            gross = added + avoided if matched else np.nan
            error = abs(net - annual_value) if matched else np.nan
            if matched and not np.isclose(net, annual_value, rtol=1e-10, atol=1e-7):
                raise ValueError(f'Seasons do not close to the annual pair: {record}, {metric}')
            record.update({
                'seasonal_added_' + unit: added,
                'seasonal_avoided_' + unit: avoided,
                'seasonal_net_' + unit: net,
                'annual_delta_' + unit: annual_value,
                'seasonal_annual_closure_error_' + unit: error,
                'opposing_seasonal_' + unit + '_directions':
                    bool(added > 1e-7 and avoided > 1e-7) if matched else pd.NA,
                'seasonal_' + unit + '_offset_fraction':
                    float(1 - abs(net) / gross) if matched and gross > 1e-7 else (0. if matched else np.nan),
            })
        rows.append(record)
    table = pd.DataFrame(rows)
    summary = []
    for key, group in table.groupby(['comparison_mode', 'service_fraction'], sort=True):
        matched = group[group.pair_status.eq('matched')]
        distinct = matched[~matched.same_physical_operation.astype(bool)]
        row = dict(zip(['comparison_mode', 'service_fraction'], key))
        row.update(layouts_included='|'.join(sorted(group.layout.unique())),
                   designed_pairs=len(group), matched_pairs=len(matched),
                   same_physical_operation_pairs=int(matched.same_physical_operation.astype(bool).sum()),
                   distinct_physical_operation_pairs=len(distinct))
        for unit in ['site_kwh', 'operational_co2_kg']:
            annual = matched['annual_delta_' + unit]
            opposing = matched['opposing_seasonal_' + unit + '_directions'].astype(bool)
            distinct_offsets = distinct['seasonal_' + unit + '_offset_fraction']
            row.update({
                'opposing_seasonal_' + unit + '_direction_pairs': int(opposing.sum()),
                'annual_' + unit + '_increase_pairs': int(annual.gt(1e-7).sum()),
                'annual_' + unit + '_decrease_pairs': int(annual.lt(-1e-7).sum()),
                'annual_' + unit + '_zero_pairs': int(annual.abs().le(1e-7).sum()),
                'opposing_and_annual_' + unit + '_increase_pairs': int((opposing & annual.gt(1e-7)).sum()),
                'opposing_and_annual_' + unit + '_decrease_pairs': int((opposing & annual.lt(-1e-7)).sum()),
                'distinct_seasonal_' + unit + '_offset_fraction_median':
                    float(distinct_offsets.median()) if len(distinct_offsets) else np.nan,
            })
        summary.append(row)
    return table, pd.DataFrame(summary)


def derive(args):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.analysis_module.resolve().parent))
    annual_module = importlib.import_module(args.analysis_module.stem)
    engineering = args.engineering_root.resolve()
    manifest_path = engineering / "ANNUAL_RUN_MANIFEST.csv"
    profiles_path = engineering / "PROFILE_ASSIGNMENTS.csv.gz"
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    profiles = pd.read_csv(profiles_path)
    factors = pd.read_csv(args.factors).set_index("city")
    tails = pd.read_csv(args.annual_root / "FIXED_TAIL_MEMBERSHIPS.csv")
    direct = pd.read_csv(args.annual_root / "ANNUAL_DIRECT.csv")
    service = pd.read_csv(args.annual_root / "ANNUAL_SERVICE_SELECTIONS.csv")
    selections = pd.concat([direct, service[service.service_fraction.eq(1.0)]], ignore_index=True)
    selections = selections[selections.permission.eq("native_zone")].copy()
    selections = selections.sort_values(KEYS + ["comparison_mode"]).reset_index(drop=True)
    expected_layouts = getattr(args, 'expected_layouts', 3)
    validate_selection_grid(selections, direct, expected_layouts)
    profile_groups = {key: g.sort_values('profile_id', kind='mergesort').reset_index(drop=True)
                      for key, g in profiles.groupby(PROFILE_KEYS, sort=True)}
    tail_groups = {key: set(g.profile_id) for key, g in tails.groupby(PROFILE_KEYS, sort=True)}
    locations = {r.run_id: annual_module.resolve_hourly(engineering, r)
                 for r in manifest.itertuples(index=False)}
    rows, checks, cache, energy_curves, source_hashes = [], [], {}, {}, {}
    for selected in selections.to_dict('records'):
        identity = {k: selected[k] for k in KEYS + ['comparison_mode', 'service_fraction']}
        metadata = identity | {k: selected.get(k, '') for k in ['variant', 'run_id', 'engineering_status', 'selection_status']}
        key = tuple(selected[k] for k in PROFILE_KEYS)
        if selected['selection_status'] != 'selected':
            for kind, period, months in period_definitions():
                rows.append(metadata | {'period_type': kind, 'period': period,
                                        'months': '|'.join(map(str, months)),
                                        'data_status': selected['selection_status']})
            continue
        score_key = key + (selected['run_id'],)
        if score_key not in cache:
            path = locations[selected['run_id']]
            data = annual_module.load_hourly(path)
            for name in CARRIERS:
                if data[name].shape != (8760,) or not np.isfinite(data[name]).all():
                    raise ValueError(f"Invalid frozen energy component: {name}")
            if set(np.unique(data['month'])) != set(range(1, 13)):
                raise ValueError('All twelve months are required')
            if not np.array_equal(np.bincount(data['month'].astype(int))[1:],
                                  [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]):
                raise ValueError('The hourly calendar does not close to the complete nonleap year')
            group = profile_groups[key]
            tail = np.flatnonzero(group.profile_id.isin(tail_groups[key]))
            if len(tail) != 27:
                raise ValueError('Exactly 27 retained annual-tail profiles are required')
            original = annual_module.score(data, group, tail)['summary']
            periods, tail_rollup = calculate(data, group, tail, factors.loc[selected['context']])
            cache[score_key] = periods, tail_rollup, original
            energy_curves[selected['run_id']] = (data['hvac_electric_kwh'], data['gas_hhv_kwh'], data['month'])
            source_hashes[selected['run_id']] = digest(path)
        periods, tail_rollup, original = cache[score_key]
        whole = periods[-1]
        for metric in SERVICE:
            check_value(checks, identity, metric, 'independent_vs_original_score', whole[metric], original[metric])
            check_value(checks, identity, metric, 'independent_vs_retained_annual', whole[metric], selected[metric])
        for metric, retained in RETAINED.items():
            check_value(checks, identity, metric, 'hourly_vs_retained_annual', whole[metric], selected[retained])
        check_value(checks, identity, 'hvac_electric_kwh', 'heating_plus_cooling_plus_fans',
                    whole['heating_electric_kwh'] + whole['cooling_electric_kwh'] + whole['fans_electric_kwh'],
                    whole['hvac_electric_kwh'])
        for kind in ('month', 'season'):
            pieces = [p for p in periods if p['period_type'] == kind]
            for metric in CARRIERS + ['purchased_site_kwh', 'operational_co2_kg', 'occupied_native_person_hours',
                                      'reference_mismatch_native_person_c_hours', 'covered_native_person_hours']:
                check_value(checks, identity, metric, kind + '_sum_vs_annual', sum(p[metric] for p in pieces), whole[metric])
            check_value(checks, identity, 'mean_reference_mismatch_c', kind + '_weighted_rollup',
                        sum(p['reference_mismatch_native_person_c_hours'] for p in pieces)
                        / sum(p['occupied_native_person_hours'] for p in pieces), whole['mean_reference_mismatch_c'])
            check_value(checks, identity, 'modeled_interval_person_hour_coverage', kind + '_weighted_rollup',
                        sum(p['covered_native_person_hours'] for p in pieces)
                        / sum(p['occupied_native_person_hours'] for p in pieces), whole['modeled_interval_person_hour_coverage'])
            check_value(checks, identity, 'fixed_tail_reference_mismatch_c', kind + '_profile_numerators_rollup',
                        tail_rollup[kind], whole['fixed_tail_reference_mismatch_c'])
        for period in periods:
            rows.append(metadata | period | {'data_status': 'qualified'})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'PERIOD_TOTALS.csv', index=False)
    selections[[*KEYS, 'comparison_mode', 'service_fraction', 'variant', 'run_id',
                'engineering_status', 'selection_status']].to_csv(args.output / 'SELECTION_STATUS.csv', index=False)
    pairs = []
    pairing = PROFILE_KEYS + ['permission', 'comparison_mode', 'service_fraction', 'period_type', 'period']
    index = {tuple(r[k] for k in pairing): r for r in table[table.control.eq('uniform_average')].to_dict('records')}
    for alternative in table[~table.control.eq('uniform_average')].to_dict('records'):
        reference = index.get(tuple(alternative[k] for k in pairing))
        matched = reference is not None and reference['data_status'] == alternative['data_status'] == 'qualified'
        pair = {k: alternative[k] for k in pairing} | {'alternative': alternative['control'],
                'reference': 'uniform_average', 'pair_status': 'matched' if matched else 'unmatched',
                'alternative_status': alternative['selection_status'],
                'reference_status': reference['selection_status'] if reference else 'missing_design',
                'alternative_run_id': alternative.get('run_id', ''),
                'reference_run_id': reference.get('run_id', '') if reference else '',
                'same_physical_operation': (alternative.get('run_id', '') == reference.get('run_id', ''))
                    if matched else pd.NA}
        for metric in SERVICE + CARRIERS + ['purchased_site_kwh', 'operational_co2_kg']:
            pair['delta_' + metric] = alternative.get(metric, np.nan) - reference.get(metric, np.nan) if matched else np.nan
        if matched:
            ae, ag, months = energy_curves[alternative['run_id']]
            re, rg, rm = energy_curves[reference['run_id']]
            if not np.array_equal(months, rm):
                raise ValueError('Paired hourly calendars differ')
            mask = np.isin(months, list(map(int, alternative['months'].split('|'))))
            for name, delta in [('electric', ae-re), ('gas_hhv', ag-rg), ('purchased_site', ae+ag-re-rg)]:
                pair['hourly_added_' + name + '_kwh'] = float(np.maximum(delta[mask], 0).sum())
                pair['hourly_released_' + name + '_kwh'] = float(np.maximum(-delta[mask], 0).sum())
                target = {'electric': 'hvac_electric_kwh', 'gas_hhv': 'gas_hhv_kwh',
                          'purchased_site': 'purchased_site_kwh'}[name]
                if not np.isclose(pair['hourly_added_' + name + '_kwh'] - pair['hourly_released_' + name + '_kwh'],
                                  pair['delta_' + target], rtol=1e-10, atol=1e-7):
                    raise ValueError('Added minus released hourly energy does not close to the period difference')
        pairs.append(pair)
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(args.output / 'PAIRED_PERIOD_DIFFERENCES.csv', index=False)
    compensation, compensation_summary = temporal_compensation(pair_table)
    compensation.to_csv(args.output / 'TEMPORAL_COMPENSATION_PAIRS.csv', index=False)
    compensation_summary.to_csv(args.output / 'TEMPORAL_COMPENSATION_SUMMARY.csv', index=False)
    # Describe every declared composition/placement stratum, including unmatched
    # designs. These strata are not independent experiments.
    seasonal_pairs = pair_table
    seasonal_pairs = seasonal_pairs[seasonal_pairs.period_type.eq('season')]
    delta_summary = []
    for key, group in seasonal_pairs.groupby(['context', 'comparison_mode', 'period', 'alternative'], sort=True):
        matched = group[group.pair_status.eq('matched')]
        summary_row = dict(zip(['context', 'comparison_mode', 'season', 'alternative'], key))
        summary_row.update(reference='uniform_average', designed_strata=len(group), matched_strata=len(matched),
                           unmatched_strata=len(group)-len(matched),
                           mean_mismatch_decreased=int((matched.delta_mean_reference_mismatch_c < -1e-9).sum()),
                           mean_mismatch_increased=int((matched.delta_mean_reference_mismatch_c > 1e-9).sum()),
                           tail_mismatch_decreased=int((matched.delta_fixed_tail_reference_mismatch_c < -1e-9).sum()),
                           interval_coverage_increased=int((matched.delta_modeled_interval_person_hour_coverage > 1e-9).sum()),
                           interval_coverage_decreased=int((matched.delta_modeled_interval_person_hour_coverage < -1e-9).sum()),
                           mean_and_coverage_both_improved=int(((matched.delta_mean_reference_mismatch_c < -1e-9)
                                                               & (matched.delta_modeled_interval_person_hour_coverage > 1e-9)).sum()),
                           purchased_site_energy_decreased=int((matched.delta_purchased_site_kwh < -1e-7).sum()))
        for metric in ['mean_reference_mismatch_c', 'fixed_tail_reference_mismatch_c',
                       'modeled_interval_person_hour_coverage', 'purchased_site_kwh',
                       'hvac_electric_kwh', 'gas_hhv_kwh', 'heating_electric_kwh',
                       'cooling_electric_kwh', 'fans_electric_kwh', 'pumps_electric_kwh', 'operational_co2_kg']:
            values = matched['delta_' + metric]
            name = 'interval_coverage_percentage_points' if metric == 'modeled_interval_person_hour_coverage' else metric
            if metric == 'modeled_interval_person_hour_coverage':
                values = values * 100
            for statistic in ['median', 'min', 'max']:
                summary_row['delta_' + name + '_' + statistic] = float(getattr(values, statistic)()) if len(values) else np.nan
        delta_summary.append(summary_row)
    pd.DataFrame(delta_summary).to_csv(args.output / 'SEASONAL_PAIRED_DELTA_SUMMARY.csv', index=False)
    # Seasonal common25 reference budgets are diagnostic only. Keep the annual
    # selected plan and the annual fixed-tail membership; never reselect either.
    anchors_path = args.annual_root / 'ANNUAL_ANCHORS.csv'
    anchors = pd.read_csv(anchors_path)
    budget_rows = []
    for anchor in anchors.to_dict('records'):
        key = tuple(anchor[k] for k in PROFILE_KEYS)
        score_key = key + (anchor['anchor_run_id'],)
        if score_key not in cache:
            path = locations[anchor['anchor_run_id']]
            data = annual_module.load_hourly(path)
            group = profile_groups[key]
            tail = np.flatnonzero(group.profile_id.isin(tail_groups[key]))
            original = annual_module.score(data, group, tail)['summary']
            periods, tail_rollups = calculate(data, group, tail, factors.loc[anchor['context']])
            cache[score_key] = periods, tail_rollups, original
            source_hashes[anchor['anchor_run_id']] = digest(path)
        periods, tail_rollups, original = cache[score_key]
        anchor_identity = dict(zip(PROFILE_KEYS, key)) | {'permission': 'native_zone', 'control': 'common25_anchor',
                                                        'comparison_mode': 'diagnostic_reference', 'service_fraction': 1.0}
        check_value(checks, anchor_identity, 'mean_reference_mismatch_c', 'common25_vs_retained_anchor',
                    periods[-1]['mean_reference_mismatch_c'], anchor['anchor_mean_c'])
        check_value(checks, anchor_identity, 'fixed_tail_reference_mismatch_c', 'common25_vs_retained_anchor',
                    periods[-1]['fixed_tail_reference_mismatch_c'], anchor['anchor_fixed_tail_c'])
        for period in periods:
            budget_rows.append(dict(zip(PROFILE_KEYS, key)) | {'period_type': period['period_type'], 'period': period['period'],
                    'reference_run_id': anchor['anchor_run_id'], 'diagnostic_service_fraction': 1.0,
                    'mean_budget_c': period['mean_reference_mismatch_c'],
                    'fixed_annual_tail_budget_c': period['fixed_tail_reference_mismatch_c'],
                    'reference_occupied_native_person_hours': period['occupied_native_person_hours']})
    budgets = pd.DataFrame(budget_rows)
    budgets.to_csv(args.output / 'COMMON25_PERIOD_BUDGETS.csv', index=False)
    diagnostic = table[table.period_type.isin(['season', 'annual'])].merge(
        budgets, on=PROFILE_KEYS + ['period_type', 'period'], validate='many_to_one')
    diagnostic['mean_budget_margin_c'] = diagnostic.mean_budget_c - diagnostic.mean_reference_mismatch_c
    diagnostic['fixed_tail_budget_margin_c'] = diagnostic.fixed_annual_tail_budget_c - diagnostic.fixed_tail_reference_mismatch_c
    diagnostic['mean_budget_pass'] = diagnostic.mean_budget_margin_c.ge(-1e-9)
    diagnostic['fixed_tail_budget_pass'] = diagnostic.fixed_tail_budget_margin_c.ge(-1e-9)
    diagnostic['both_budgets_pass'] = diagnostic.mean_budget_pass & diagnostic.fixed_tail_budget_pass
    diagnostic['budget_evaluation_status'] = np.where(diagnostic.data_status.eq('qualified'),
            np.where(diagnostic.both_budgets_pass, 'pass', 'reference_budget_exceeded'), 'not_evaluable')
    diagnostic['is_retained_selection_requirement'] = diagnostic.period_type.eq('annual') & diagnostic.comparison_mode.eq('equal_service')
    annual_required = diagnostic[diagnostic.is_retained_selection_requirement & diagnostic.data_status.eq('qualified')]
    if not annual_required.both_budgets_pass.all():
        raise ValueError('A retained common-service annual selection fails its unchanged annual budgets')
    for column in ['mean_budget_pass', 'fixed_tail_budget_pass', 'both_budgets_pass']:
        diagnostic[column] = diagnostic[column].astype('boolean').mask(~diagnostic.data_status.eq('qualified'), pd.NA)
    diagnostic.to_csv(args.output / 'SEASONAL_SERVICE_BUDGET_DIAGNOSTIC.csv', index=False)
    budget_summary = []
    for key, group in diagnostic.groupby(['context', 'comparison_mode', 'control', 'period_type', 'period'], sort=True):
        valid = group[group.data_status.eq('qualified')]
        budget_summary.append(dict(zip(['context', 'comparison_mode', 'control', 'period_type', 'period'], key))
              | {'designed_strata': len(group), 'qualified_strata': len(valid),
                 'not_evaluable_strata': len(group)-len(valid),
                 'mean_budget_pass_strata': int(valid.mean_budget_pass.sum()),
                 'fixed_tail_budget_pass_strata': int(valid.fixed_tail_budget_pass.sum()),
                 'both_budgets_pass_strata': int(valid.both_budgets_pass.sum())})
    pd.DataFrame(budget_summary).to_csv(args.output / 'SEASONAL_BUDGET_SUMMARY.csv', index=False)
    check_table = pd.DataFrame(checks)
    check_table.to_csv(args.output / 'ANNUAL_ROLLUP_CHECKS.csv', index=False)
    primary = selections[selections.composition_id.eq(args.figure_composition) & selections.layout.eq('random_mixing')]
    warnings = []
    for (context, mode), group in primary.groupby(['context', 'comparison_mode']):
        if group.selection_status.ne('selected').any():
            warnings.append(f'{context} {mode}: engineering-rejected controls retained as gaps; no qualified comparison')
        elif group.run_id.nunique() == 1:
            warnings.append(f'{context} {mode}: all three controls select the identical physical trajectory; curves overlap')
    summary = {'status': 'PASS', 'read_only_frozen_inputs': True,
               'simulation_reruns': 0, 'physics_or_profiles_changed': False,
               'contexts': CONTEXTS, 'compositions': sorted(selections.composition_id.unique().tolist()),
               'layouts': sorted(selections.layout.unique().tolist()), 'permission': 'native_zone',
               'modes': {'direct': 'retained variant 0', 'equal_service': 'retained service fraction 1.0 selection'},
               'selection_rows': len(selections), 'period_rows': len(table),
               'qualified_selection_rows': int(selections.selection_status.eq('selected').sum()),
               'rejected_selection_rows': int(selections.selection_status.ne('selected').sum()),
               'profile_trajectory_pairs_scored': len(cache), 'unique_hourly_inputs': len(source_hashes),
               'annual_checks': len(checks), 'all_annual_checks_passed': bool(check_table.passed.all()),
               'maximum_absolute_error_by_metric': check_table.groupby('metric').absolute_error.max().to_dict(),
               'month_hours': {'01': 744, '02': 672, '03': 744, '04': 720, '05': 744, '06': 720,
                               '07': 744, '08': 744, '09': 720, '10': 744, '11': 720, '12': 744},
               'seasons': SEASONS, 'winter_definition': 'December + January + February from the same retained annual simulation',
               'weighting': 'Native people within the locked occupied source window, divided equally among assigned profiles in each zone',
               'tail': 'Same retained annual 27-profile tail in every month and season; equal mean of profile-level occupied-time averages',
               'seasonal_budget_diagnostic': 'Each selected annual plan is compared with the same-period common25 reference mean/tail using unchanged annual-tail membership. Seasonal checks are descriptive and are not new selection requirements; no plan is reselected.',
               'temporal_compensation': {
                   'pair_table': 'TEMPORAL_COMPENSATION_PAIRS.csv',
                   'summary_table': 'TEMPORAL_COMPENSATION_SUMMARY.csv',
                   'definition': 'Seasonal additions sum positive spring/summer/autumn/winter paired changes; seasonal savings sum their absolute negative changes. Offset fraction is 1 - abs(annual net)/(additions + savings).',
                   'matched_pairs': int(compensation.pair_status.eq('matched').sum()),
                   'maximum_site_energy_closure_error_kwh': float(compensation.seasonal_annual_closure_error_site_kwh.max()),
                   'maximum_operational_co2_closure_error_kg': float(compensation.seasonal_annual_closure_error_operational_co2_kg.max()),
               },
               'paired_summary_interpretation': f'All {3 * expected_layouts} composition/placement strata per context/mode/alternative/season; retained zero-match rows. Strata share fixed profiles and physical runs and are not independent experimental replicates.',
               'service_interpretation': 'Reference mismatch and model-derived interval coverage; not measured satisfaction',
               'accounting': 'Purchased site = frozen HVAC electricity + gas HHV. Pumps and the separate HVAC meter remain separate channels; neither is silently added to the locked HVAC denominator.',
               'figure_primary': {'composition_id': args.figure_composition, 'layout': 'random_mixing', 'permission': 'native_zone'},
               'warnings': warnings,
               'source_sha256': {'annual_module': digest(args.analysis_module), 'manifest': digest(manifest_path),
                                 'profiles': digest(profiles_path), 'factors': digest(args.factors),
                                 'direct': digest(args.annual_root / 'ANNUAL_DIRECT.csv'),
                                 'selections': digest(args.annual_root / 'ANNUAL_SERVICE_SELECTIONS.csv'),
                                 'anchors': digest(anchors_path),
                                 'tail_memberships': digest(args.annual_root / 'FIXED_TAIL_MEMBERSHIPS.csv')},
               'hourly_inputs_sha256': source_hashes}
    provenance = {key: summary.pop(key) for key in ['source_sha256', 'hourly_inputs_sha256']}
    (args.output / 'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    (args.output / 'ROLLUP_SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if expected_layouts == 2:
        scope = {'layouts': sorted(selections.layout.unique().tolist()),
                 'compositions': sorted(selections.composition_id.unique().tolist()),
                 'contexts': CONTEXTS, 'permission': 'native_zone',
                 'retained_selection_rows': len(selections),
                 'operation_modes': ['direct', 'equal_service_fraction_1'],
                 'coverage': 'Complete retained annual histories summarized by all twelve months and four meteorological seasons; annual selections and annual tail identities remain fixed.',
                 'new_physical_runs': 0, 'reselection': False,
                 'cohort_comparison': 'Each cohort retains its own demand inputs and service references; records from different cohorts are not pooled.'}
        (args.output / 'COHORT_SCOPE.json').write_text(json.dumps(scope, indent=2) + '\n', encoding='utf-8')
    return table, summary


def render(table, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for weight in ['normal', 'bold']:
        findfont(FontProperties(family='Times New Roman', weight=weight), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'text.color': 'black',
                         'axes.labelcolor': 'black', 'axes.labelsize': 11, 'axes.titlesize': 12,
                         'axes.titleweight': 'bold', 'xtick.labelsize': 11, 'ytick.labelsize': 11,
                         'xtick.color': 'black', 'ytick.color': 'black', 'legend.fontsize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'hem-seasonal-evidence'})
    primary = table[table.composition_id.eq(args.figure_composition) & table.layout.eq('random_mixing')
                    & table.period_type.eq('month')].copy()
    primary['month_number'] = primary.period.astype(int)
    stack_keys = ['heating_electric_kwh', 'cooling_electric_kwh', 'fans_electric_kwh', 'gas_hhv_kwh']
    stack_colors = ['#C78470', '#6F9EBC', '#C8A36A', '#AF94AD']
    stack_labels = ['Heating electricity', 'Cooling electricity', 'Fan electricity', 'Gas (HHV)']
    for mode, name in [('direct', 'direct'), ('equal_service', 'common_service')]:
        figure = plt.figure(figsize=(10.2, 11.6))
        axes = []
        for i, (context, label) in enumerate(zip(CONTEXTS, CONTEXT_LABELS)):
            y = .755 - i * .216
            left = figure.add_axes([.083, y, .36, .155])
            right = figure.add_axes([.586, y, .36, .155])
            axes.extend([left, right])
            left.set_title(f'({chr(97+2*i)}) {label}: reference mismatch', loc='left', pad=7)
            right.set_title(f'({chr(98+2*i)}) {label}: purchased energy', loc='left', pad=7)
            block = primary[primary.context.eq(context) & primary.comparison_mode.eq(mode)]
            valid = block[block.data_status.eq('qualified')]
            uniform = valid[valid.control.eq('uniform_average')].sort_values('month_number')
            if len(uniform):
                right.stackplot(uniform.month_number.to_numpy(), *[uniform[k].to_numpy()/1000 for k in stack_keys],
                                colors=stack_colors, alpha=.85, linewidth=0, zorder=1)
            for control, color, marker in zip(CONTROLS, COLORS, ['o', 's', '^']):
                values = valid[valid.control.eq(control)].sort_values('month_number')
                if len(values):
                    # Nested hollow markers preserve visibility when the same
                    # fallback trajectory is selected by all three controls.
                    size = {'uniform_average': 5.8, 'personal_center': 4.8, 'ltpref_interval': 3.7}[control]
                    left.plot(values.month_number, values.mean_reference_mismatch_c, color=color,
                              linewidth=1.8, marker=marker, markersize=size, markerfacecolor='white',
                              markeredgewidth=.85, zorder=4)
                    right.plot(values.month_number, values.purchased_site_kwh/1000, color=color,
                               linewidth=1.8, marker=marker, markersize=size, markerfacecolor='white',
                               markeredgewidth=.85, zorder=4)
            if valid.empty:
                for ax in [left, right]:
                    ax.text(.5, .52, 'All direct controls\nengineering-rejected', ha='center', va='center',
                            transform=ax.transAxes, fontsize=11, color='black')
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
            elif valid.groupby('control').run_id.first().nunique() == 1:
                left.text(.02, .96, 'All three controls identical', transform=left.transAxes,
                          ha='left', va='top', fontsize=11)
                low, high = left.get_ylim()
                left.set_ylim(low, high + .15*(high-low))
            left.set_ylabel('Mismatch (°C)')
            right.set_ylabel('Monthly energy (MWh)')
            for ax in [left, right]:
                ax.set_xlim(.8, 12.2)
                ax.set_xticks([1, 4, 7, 10])
                ax.set_xticklabels(['January', 'April', 'July', 'October'], rotation=25, ha='right')
                ax.tick_params(length=3, width=.7)
                ax.grid(axis='y', color='#D8D8D8', linewidth=.45, zorder=0)
                ax.set_axisbelow(True)
        title = 'Direct controls (variant 0)' if mode == 'direct' else 'Controls selected at the common annual service budget (fraction 1.0)'
        figure.text(.5, .972, title, ha='center', fontsize=11)
        figure.legend([Line2D([], [], color=c, lw=1.8, marker=m, markerfacecolor='white', markersize=5)
                       for c, m in zip(COLORS, ['o', 's', '^'])], LABELS,
                      loc='upper center', bbox_to_anchor=(.5, .957), ncol=3, frameon=False,
                      columnspacing=2.0, handletextpad=.5)
        figure.legend([Patch(facecolor=c, alpha=.85) for c in stack_colors], stack_labels,
                      loc='lower center', bbox_to_anchor=(.5, .024), ncol=4, frameon=False,
                      columnspacing=1.4, handlelength=1.4, handletextpad=.45)
        figure.text(.5, .014, 'Energy stack: Uniform average baseline; lines: total purchased energy for each control',
                    ha='center', fontsize=11)
        figure.canvas.draw()
        # Verify that all panel frames use identical dimensions and align in rows.
        frames = [ax.get_position().bounds for ax in axes]
        if len({tuple(np.round(b[2:], 8)) for b in frames}) != 1:
            raise ValueError('Panel dimensions differ')
        stem = args.output / ('SEASONAL_MONTHLY_' + name)
        for ext in ['png', 'pdf', 'svg']:
            figure.savefig(stem.with_suffix('.' + ext), dpi=300, facecolor='white')
        plt.close(figure)
        caption = (f'Monthly service and energy from retained complete 8760-hour annual trajectories: {title.lower()}. '
                   'Panels show all twelve months for the prespecified first composition, random mixing, and zone-level control permission. '
                   'Left panels use absolute temperature-reference mismatch weighted by actual native people within the locked occupied window. '
                   'Right panels show Uniform average electricity components (heating, cooling, fans) and gas HHV as a stack; '
                   'the three control lines show total purchased HVAC electricity plus gas HHV. '
                   'The frozen HVAC electricity channel closes to heating, cooling, and fans; the separately retained pumps and HVAC-meter channels '
                   'are available in PERIOD_TOTALS.csv and are not added to that locked electricity denominator. '
                   'Identical selections overlap and are explicitly labelled. Engineering-rejected direct trajectories remain gaps. '
                   'Monthly and meteorological-season fixed annual-tail mismatch, model-derived interval coverage, native person-hour weights, '
                   'all energy components, and carbon computed with the frozen regional factors are supplied in the accompanying tables. '
                   'Selections use annual service budgets; they do not establish that every monthly or seasonal budget is met. '
                   'No simulation was rerun; no control, physics, profile, annual-tail membership, or annual selection was changed. '
                   'Service endpoints are model-reference measures, not observed satisfaction. HVAC denotes heating, ventilation and air conditioning; HHV denotes higher heating value.\n')
        stem.with_name(stem.name + '_caption.txt').write_text(caption, encoding='utf-8')


def render_service_details(table, args):
    """Show the three model-reference endpoints without crowding energy panels."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    primary = table[table.composition_id.eq(args.figure_composition) & table.layout.eq('random_mixing')
                    & table.period_type.eq('month')].copy()
    primary['month_number'] = primary.period.astype(int)
    for mode, name in [('direct', 'direct'), ('equal_service', 'common_service')]:
        figure = plt.figure(figsize=(10.2, 11.6))
        for i, (context, label) in enumerate(zip(CONTEXTS, CONTEXT_LABELS)):
            y = .755 - i * .216
            left = figure.add_axes([.083, y, .36, .155])
            right = figure.add_axes([.586, y, .36, .155])
            left.set_title(f'({chr(97+2*i)}) {label}: reference mismatch', loc='left', pad=7)
            right.set_title(f'({chr(98+2*i)}) {label}: model interval coverage', loc='left', pad=7)
            block = primary[primary.context.eq(context) & primary.comparison_mode.eq(mode)
                            & primary.data_status.eq('qualified')]
            identical = not block.empty and block.groupby('control').run_id.first().nunique() == 1
            for control, color, marker in zip(CONTROLS, COLORS, ['o', 's', '^']):
                values = block[block.control.eq(control)].sort_values('month_number')
                if len(values):
                    size = {'uniform_average': 5.8, 'personal_center': 4.8, 'ltpref_interval': 3.7}[control]
                    for metric, linestyle in [('mean_reference_mismatch_c', '-'), ('fixed_tail_reference_mismatch_c', '--')]:
                        left.plot(values.month_number, values[metric], color=color, linestyle=linestyle,
                                  linewidth=1.6, marker=marker, markersize=size, markerfacecolor='white',
                                  markeredgewidth=.8)
                    right.plot(values.month_number, values.modeled_interval_person_hour_coverage * 100,
                               color=color, linewidth=1.8, marker=marker, markersize=size,
                               markerfacecolor='white', markeredgewidth=.8)
            if block.empty:
                for ax in [left, right]:
                    ax.text(.5, .52, 'All direct controls\nengineering-rejected', ha='center', va='center',
                            transform=ax.transAxes, fontsize=11)
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])
            elif identical:
                left.text(.02, .96, 'All three controls identical', transform=left.transAxes,
                          ha='left', va='top', fontsize=11)
                low, high = left.get_ylim()
                left.set_ylim(low, high + .15*(high-low))
            left.set_ylabel('Reference mismatch (°C)')
            right.set_ylabel('Native person-hours\ncovered (%)')
            for ax in [left, right]:
                ax.set_xlim(.8, 12.2)
                ax.set_xticks([1, 4, 7, 10])
                ax.set_xticklabels(['January', 'April', 'July', 'October'], rotation=25, ha='right')
                ax.tick_params(length=3, width=.7)
                ax.grid(axis='y', color='#D8D8D8', linewidth=.45)
                ax.set_axisbelow(True)
        title = 'Direct controls (variant 0)' if mode == 'direct' else 'Controls selected at the common annual service budget (fraction 1.0)'
        figure.text(.5, .972, title, ha='center', fontsize=11)
        figure.legend([Line2D([], [], color=c, lw=1.8, marker=m, markerfacecolor='white', markersize=5)
                       for c, m in zip(COLORS, ['o', 's', '^'])], LABELS,
                      loc='upper center', bbox_to_anchor=(.5, .957), ncol=3, frameon=False,
                      columnspacing=2.0, handletextpad=.5)
        figure.legend([Line2D([], [], color='black', lw=1.6, linestyle=s) for s in ['-', '--']],
                      ['Population mean mismatch', 'Fixed annual-tail mismatch'],
                      loc='lower center', bbox_to_anchor=(.5, .024), ncol=2, frameon=False,
                      columnspacing=2.0, handlelength=2.0)
        figure.text(.5, .014, 'Model-derived temperature references and interval coverage; not measured comfort or satisfaction',
                    ha='center', fontsize=11)
        figure.canvas.draw()
        stem = args.output / ('SEASONAL_SERVICE_' + name)
        for ext in ['png', 'pdf', 'svg']:
            figure.savefig(stem.with_suffix('.' + ext), dpi=300, facecolor='white')
        plt.close(figure)
        caption = (f'Monthly model-reference service from retained complete annual simulations: {title.lower()}. '
                   'The prespecified first composition, random-mixing placement, and zone-level permission are shown for all twelve months. '
                   'Solid lines in left panels are native-population-weighted absolute temperature-reference mismatch; '
                   'dashed lines are the equal mean of the same retained annual 27-tail profiles after each profile’s own native-occupied-time average. '
                   'The annual tail is never selected again within a month or season. Right panels show native-person-hour-weighted coverage '
                   'of model-derived profile intervals. Actual native zone people counts within the locked occupied source window '
                   'are shared equally among assigned profiles. Identical physical selections overlap; engineering rejection is retained as a gap. '
                   'Selections were made using annual service budgets; these traces do not establish that every monthly or seasonal budget is met. '
                   'These modeled service endpoints are not observed comfort, satisfaction, or experimental measurements. '
                   'Every month and all four meteorological seasons are supplied in PERIOD_TOTALS.csv, including qualified and rejected design rows.\n')
        stem.with_name(stem.name + '_caption.txt').write_text(caption, encoding='utf-8')


def verify_frozen_inputs(args):
    retained = json.loads((args.annual_root / 'ANALYSIS_MANIFEST.json').read_text(encoding='utf-8'))
    provenance = json.loads((args.output / 'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))
    source = provenance['source_sha256']
    checks = {key + '_matches_retained_input': source[key] == retained['source_sha256'][key]
              for key in ['manifest', 'profiles', 'factors']}
    for key, filename in [('direct', 'ANNUAL_DIRECT.csv'), ('selections', 'ANNUAL_SERVICE_SELECTIONS.csv'),
                          ('tail_memberships', 'FIXED_TAIL_MEMBERSHIPS.csv'), ('anchors', 'ANNUAL_ANCHORS.csv')]:
        checks[key + '_matches_retained_output'] = source[key] == retained['output_sha256'][filename]
    known = set(retained['hourly_input_sha256'].values())
    checks['all_hourly_sha256_match_retained_inputs'] = all(value in known for value in provenance['hourly_inputs_sha256'].values())
    if not all(checks.values()):
        raise ValueError('Frozen input identity verification failed')
    result = {'status': 'PASS', 'checks': checks,
              'verified_hourly_input_count': len(provenance['hourly_inputs_sha256'])}
    (args.output / 'FROZEN_INPUT_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engineering-root', type=Path, required=True)
    parser.add_argument('--annual-root', type=Path, required=True)
    parser.add_argument('--factors', type=Path, required=True)
    parser.add_argument('--analysis-module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure-composition', type=int, default=2026102001)
    parser.add_argument('--expected-layouts', type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table, summary = derive(args)
    export_comparison_regimes(args)
    verify_frozen_inputs(args)
    render(table, args)
    render_service_details(table, args)
    print(json.dumps({k: summary[k] for k in ['status', 'selection_rows', 'period_rows', 'qualified_selection_rows',
                                           'rejected_selection_rows', 'annual_checks', 'warnings']}, indent=2))


if __name__ == '__main__':
    main()
