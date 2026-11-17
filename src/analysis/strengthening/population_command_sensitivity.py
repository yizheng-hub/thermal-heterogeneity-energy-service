"""Evaluate additional demand compositions using 31 shared temperature commands."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "src/analysis/global_extension"))
from service_threshold_sensitivity import (Trajectories, SCORE_COLUMNS, digest,
    evaluate_thresholds, verify_reference)
from service_distribution import ServiceDistribution
from analyse_global_results import normalise_profiles
from paired_uncertainty import choose

def sample(pool, zones, seed, ashrae):
    p = pool.source_equal_probability.to_numpy() if ashrae else None
    selected = np.random.default_rng(seed).choice(len(pool), 267, replace=False, p=p)
    x = pool.iloc[selected].copy().reset_index(drop=True)
    x['pool_row'] = selected
    if ashrae:
        x['profile_id'] = ['sp_' + hashlib.sha256(f's{seed}_v{i:03d}'.encode()).hexdigest()[:24]
                           for i in range(267)]
    x = x.iloc[np.random.default_rng(seed + 1000).permutation(267)].copy()
    x['zone_id'] = zones
    return x.sort_values(['zone_id', 'profile_id']).reset_index(drop=True)

def scores(curve, centers, means, tails):
    a, b = curve.individual(centers), curve.individual(means)
    return np.column_stack([a @ curve.weights, np.take_along_axis(a, tails, axis=1).mean(axis=1),
                            b @ curve.weights, np.take_along_axis(b, tails, axis=1).mean(axis=1)])

def select(frame, values, limits):
    eligible = np.column_stack([np.all(values[:, :2] <= limits + 1e-9, axis=1),
                                np.all(values[:, 2:] <= limits + 1e-9, axis=1)])
    a, b = choose(eligible.T, frame.facility_purchased_kwh.to_numpy(), frame.run_id.to_numpy())
    if a < 0 or b < 0:
        raise AssertionError('The retained 25 C reference must provide a feasible individual operation')
    energy = frame.facility_purchased_kwh.to_numpy()
    return dict(feasibility='both_feasible', qualifying_sets_differ=bool(np.any(eligible[:, 0] != eligible[:, 1])),
                choices_differ=bool(frame.iloc[a].physical_plan_id != frame.iloc[b].physical_plan_id),
                individual_qualifying_count=int(eligible[:, 0].sum()), zone_average_qualifying_count=int(eligible[:, 1].sum()),
                individual_physical_plan_id=frame.iloc[a].physical_plan_id,
                zone_average_physical_plan_id=frame.iloc[b].physical_plan_id,
                individual_energy_kwh=float(energy[a]), zone_average_energy_kwh=float(energy[b]),
                additional_purchased_energy_kwh=float(energy[a] - energy[b]),
                additional_purchased_energy_percent=float(100 * (energy[a] - energy[b]) / energy[b]),
                zone_selected_individual_exceedance_c=float(max(0, np.max(values[b, :2] - limits))),
                population_limit_c=float(limits[0]), fixed_group_limit_c=float(limits[1]))

def summarize(frame, label):
    changed = frame.loc[frame.choices_differ]
    return dict(sample=label, settings=len(frame), different_sets=int(frame.qualifying_sets_differ.sum()),
                different_choices=int(frame.choices_differ.sum()), choice_difference_percent=100 * frame.choices_differ.mean(),
                positive_energy_penalty_count=int((frame.additional_purchased_energy_kwh > 1e-6).sum()),
                changed_energy_percent_min=float(changed.additional_purchased_energy_percent.min()),
                changed_energy_percent_median=float(changed.additional_purchased_energy_percent.median()),
                changed_energy_percent_max=float(changed.additional_purchased_energy_percent.max()),
                changed_exceedance_c_median=float(changed.zone_selected_individual_exceedance_c.median()))

def run(args):
    global INPUTS, OUT
    OUT = args.output_dir
    INPUTS = {
        "scores": args.results_root / "service_threshold_sensitivity/PAIRED_OPERATION_SERVICE_SCORES.csv",
        "profiles": args.matrix_root / "GLOBAL_PROFILES.csv",
        "references": args.matrix_root / "probe_runs/PROBE_RUN_MANIFEST.csv",
        "memberships": args.memberships or args.results_root / "base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv",
        "nominal": args.results_root / "information_uncertainty_summary/NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv",
        "ashrae_pool": args.demand_root / "ASHRAE_ANALYSIS_UNIT_POOL.csv.gz",
        "ctcd_pool": args.demand_root / "CTCD_EXECUTABLE_POOL.csv.gz",
        "inventory": args.demand_root / "ZONE_INVENTORY.csv",
    }
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(INPUTS['scores'])
    limits_path = args.results_root / 'service_threshold_sensitivity/REFERENCE_SERVICE_LIMITS.csv'
    primary = evaluate_thresholds(saved, pd.read_csv(limits_path), [1.0])
    verify_reference(primary, pd.read_csv(INPUTS['nominal']))
    primary.to_csv(OUT / 'PRIMARY_FINAL_LIBRARY_REPRODUCTION.csv', index=False)
    original = normalise_profiles(pd.read_csv(INPUTS['profiles']))
    inventory = pd.read_csv(INPUTS['inventory'])
    zones = np.repeat(inventory.zone_id.to_numpy(), inventory.profiles.to_numpy())
    pools = {s: pd.read_csv(INPUTS[k], low_memory=False) for s, k in
             [('ASHRAE', 'ashrae_pool'), ('CTCD', 'ctcd_pool')]}
    samples, metadata = [], []
    for composition, g in original[original.climate.eq('Beijing')].groupby('composition', sort=True):
        g = g.sort_values(['zone_id', 'profile_id']).reset_index(drop=True)
        source = 'ASHRAE' if str(composition).startswith('Multinational') else 'CTCD'
        samples.append(g)
        seed = int(g.composition_id.iloc[0])
        metadata.append(dict(composition=composition, source=source, sample_kind='original', draw=0, seed=seed))
        replay = sample(pools[source], zones, seed, source == 'ASHRAE')
        assert replay.profile_id.tolist() == g.profile_id.tolist()
        np.testing.assert_allclose(replay.centre_c, g.centre_c, rtol=0, atol=1e-9)
    for source, start in [('ASHRAE', 2026111701), ('CTCD', 2026111801)]:
        for draw in range(1, 31):
            samples.append(sample(pools[source], zones, start+draw-1, source == 'ASHRAE'))
            metadata.append(dict(composition=f'{source} draw {draw:02d}', source=source, sample_kind='new', draw=draw, seed=start+draw-1))
    template = samples[0]
    assert all(x.zone_id.tolist() == template.zone_id.tolist() for x in samples)
    centers = np.stack([x.centre_c.to_numpy() for x in samples])
    means = centers.copy()
    for z in template.zone_id.unique():
        pos = np.flatnonzero(template.zone_id.eq(z))
        means[:, pos] = centers[:, pos].mean(axis=1, keepdims=True)
    rows = []
    for m, x in zip(metadata, samples):
        keep = [c for c in ['profile_id','zone_id','centre_c','lower_c','upper_c','pool_row'] if c in x]
        rows.append(x[keep].assign(**m))
    pd.concat(rows).to_csv(OUT/'SAMPLED_COMPOSITIONS.csv.gz', index=False)
    probe = pd.read_csv(INPUTS['references'])
    memberships = pd.read_csv(INPUTS['memberships'])
    trajectories = Trajectories(args.matrix_root)
    results, times, candidate_rows, validations = [], [], [], []
    for climate, raw in probe.groupby('climate', sort=True):
        tick = time.perf_counter()
        raw = raw.sort_values('run_id').reset_index(drop=True)
        assert len(raw) == 31 and raw.target_c.eq(25).sum() == 1
        reference = raw[raw.target_c.eq(25)].iloc[0]
        ref = trajectories.load(reference.hourly_path, reference.hourly_sha256)
        curve = ServiceDistribution(ref, template)
        individual = curve.individual(centers)
        tails = []
        for i, x in enumerate(samples):
            if metadata[i]['sample_kind'] == 'original':
                members = memberships[memberships.climate.eq(climate) & memberships.composition.eq(metadata[i]['composition'])]
                tail = np.flatnonzero(x.profile_id.isin(members.profile_id))
            else:
                tail = np.sort(np.lexsort((x.profile_id.to_numpy(str), -individual[i]))[:27])
            assert len(tail) == 27
            tails.append(tail)
        tails = np.stack(tails)
        limits = np.column_stack([individual @ curve.weights, np.take_along_axis(individual,tails,axis=1).mean(axis=1)])
        values = np.empty((66,31,4))
        candidates = raw.copy()
        candidates['physical_plan_id'] = candidates.hourly_sha256
        candidates['facility_purchased_kwh'] = np.nan
        for j, row in enumerate(raw.itertuples()):
            data = trajectories.load(row.hourly_path, row.hourly_sha256)
            assert np.array_equal(data['occupied'], ref['occupied']) and np.array_equal(data['zone_ids'],ref['zone_ids'])
            np.testing.assert_allclose(data['people'], ref['people'], rtol=0, atol=1e-9)
            values[:,j] = scores(ServiceDistribution(data, template), centers, means, tails)
            candidates.loc[j,'facility_purchased_kwh'] = float(np.sum(data['facility_electric_kwh'])+np.sum(data['facility_gas_hhv_kwh_annual']))
        candidate_rows.append(candidates[['climate','run_id','target_c','physical_plan_id','facility_purchased_kwh','hourly_sha256']])
        by_hash = dict(zip(candidates.hourly_sha256, candidates.index))
        for i, m in enumerate(metadata[:6]):
            prior = saved[saved.climate.eq(climate) & saved.composition.eq(m['composition']) & saved.hourly_sha256.isin(by_hash)]
            ix = np.asarray([by_hash[h] for h in prior.hourly_sha256])
            error = float(np.max(np.abs(values[i,ix]-prior[SCORE_COLUMNS].to_numpy())))
            np.testing.assert_allclose(values[i,ix],prior[SCORE_COLUMNS],rtol=0,atol=1e-8)
            np.testing.assert_allclose(candidates.iloc[ix].facility_purchased_kwh,prior.facility_purchased_kwh,rtol=0,atol=1e-6)
            validations.append(dict(climate=climate,composition=m['composition'],matched_operations=len(prior),max_score_error_c=error))
        for i, m in enumerate(metadata):
            r = select(candidates, values[i], limits[i])
            targets = candidates.set_index('physical_plan_id').target_c
            r.update(individual_target_c=float(targets[r['individual_physical_plan_id']]),zone_average_target_c=float(targets[r['zone_average_physical_plan_id']]))
            results.append(dict(climate=climate,scope='31_uniform_shared_commands',**m,**r))
        np.savez_compressed(OUT/f'SCORES_{climate}.npz',scores=values,limits=limits,tails=tails,commands_c=candidates.target_c.to_numpy(),composition=np.asarray([m['composition'] for m in metadata]))
        times.append(dict(climate=climate,seconds=time.perf_counter()-tick))
        print(f'{climate}: {time.perf_counter()-tick:.1f} s',flush=True)
    frame = pd.DataFrame(results)
    frame.to_csv(OUT/'ALL_SELECTIONS.csv',index=False)
    pd.concat(candidate_rows).to_csv(OUT/'SHARED_COMMANDS.csv',index=False)
    primary_summary = primary.rename(columns={'additional_purchased_energy_pct':'additional_purchased_energy_percent'})
    summary = [summarize(primary_summary,'original_6_final_libraries'),summarize(frame[frame.sample_kind.eq('original')],'original_6_shared_command_family')]
    groups = []
    for n in [3,10,30]:
        sub = frame[frame.sample_kind.eq('new') & frame.draw.le(n)]
        summary.append(summarize(sub,f'new_{n}_per_source'))
        for source,g in sub.groupby('source'):
            summary.append(summarize(g,f'new_{n}_{source}'))
        for climate,g in sub.groupby('climate'):
            groups.append(dict(climate=climate,n_per_source=n,**summarize(g,f'new_{n}_per_source')))
    pd.DataFrame(summary).to_csv(OUT/'NESTED_SAMPLE_SUMMARY.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'CLIMATE_SUMMARY.csv',index=False)
    pd.DataFrame(validations).to_csv(OUT/'NUMERICAL_REPRODUCTION.csv',index=False)
    manifest = dict(design='Fixed complete 31-command family, same whole-building command in every zone; 30 new demand draws per source crossed with nine weather contexts',
                    primary='Primary final 8586-entry library is preserved and its stored scores reproduce all54 qualifying-set differences and24 choice differences separately',
                    sampling='267 profiles per draw without replacement; ASHRAE source-equal weighting, CTCD uniform sampling; fixed zone capacities; seed+1000 permutation',
                    nested_draw_counts=[3,10,30],source_pool_sizes={k:len(v) for k,v in pools.items()},
                    reference='The 25C-command temperature trajectory defines both limits and a fixed worst27 group separately for each demand draw and weather',
                    scoring='Exact occupied native person-hour weighted absolute mismatch for population, equal profile mean for fixed group; zone-average comparison substitutes within-zone arithmetic demand means',
                    inference='Sensitivity of demand-representation decisions to new sampled populations within shared-command family; does not establish convergence of three draws or regenerate the final demand-dependent candidate library',
                    inputs={k:dict(path=str(v),sha256=digest(v)) for k,v in INPUTS.items()},reference_limits_hash=digest(limits_path),
                    source_sha256=digest(Path(__file__)),
                    validation=dict(matched_retained_score_rows=sum(x['matched_operations'] for x in validations),max_score_error_c=max(x['max_score_error_c'] for x in validations)),
                    trajectory_files=trajectories.used,timings=times,elapsed_seconds=time.perf_counter()-started)
    (OUT/'METHODS_AND_PROVENANCE.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demand-root", type=Path, default=CODE / "output/from_raw/demand")
    parser.add_argument("--matrix-root", type=Path, default=CODE / "output/from_raw/matrix")
    parser.add_argument("--results-root", type=Path, default=CODE / "output/from_raw/strengthening/results")
    parser.add_argument("--memberships", type=Path)
    parser.add_argument("--output-dir", type=Path, default=CODE / "output/from_raw/strengthening/results/population_command_sensitivity")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
