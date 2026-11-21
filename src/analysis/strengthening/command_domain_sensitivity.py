"""Compare final-library choices with all command midpoints restricted to 21–26 °C."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / 'src/analysis/global_extension'))
from service_threshold_sensitivity import evaluate_thresholds, summarise, Trajectories, digest
from analyse_global_results import energy_metrics


def run(args):
    results, matrix, out = args.results_root, args.matrix_root, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    library = pd.read_csv(results / 'budget_frontiers/ALL_OPERATION_LIBRARY.csv')
    scores = pd.read_csv(results / 'service_threshold_sensitivity/PAIRED_OPERATION_SERVICE_SCORES.csv')
    refs = pd.read_csv(results / 'service_threshold_sensitivity/REFERENCE_SERVICE_LIMITS.csv')
    nominal = evaluate_thresholds(scores, refs, [1.0])
    if nominal.qualifying_sets_differ.sum() != 54 or nominal.choices_differ.sum() != 24:
        raise AssertionError('Primary reference choices differ')
    mapping, sources = {}, []

    def add(run_id, targets):
        values = np.asarray(sorted(targets.values()) if isinstance(targets, dict) else targets, float)
        if run_id in mapping:
            np.testing.assert_allclose(mapping[run_id], values, rtol=0, atol=1e-8)
        mapping[run_id] = values

    for path in results.glob('*MANIFEST.csv'):
        if {'run_id', 'targets_c'} <= set(pd.read_csv(path, nrows=0).columns):
            for row in pd.read_csv(path, usecols=['run_id','targets_c']).dropna().itertuples(index=False):
                add(row.run_id, json.loads(row.targets_c))
            sources.append(path)
    probe_path = matrix / 'probe_runs/PROBE_RUN_MANIFEST.csv'
    probes = pd.read_csv(probe_path)
    for row in probes.itertuples():
        add(row.run_id, [row.target_c] * 15)
    command_path = matrix / 'control_generation/controller_input/CONTROL_COMMANDS.json'
    commands = json.loads(command_path.read_text())
    comp = {f'Multinational {x}':2026102000+i for i,x in enumerate('ABC',1)} | {f'China {x}':2026102500+i for i,x in enumerate('ABC',1)}
    lookup = {(x['context'],x['composition_id'],x['control'],x['offset_c']):x['targets_c'] for x in commands}
    for row in library.itertuples():
        if row.run_id not in mapping:
            key = (row.climate,comp[row.composition],row.control,row.offset_c)
            if key in lookup:
                add(row.run_id,lookup[key])
    if not scores.run_id.isin(mapping).all():
        raise ValueError('Command vectors are missing for final-library operations')
    for frame in [library,scores]:
        frame['target_min_c'] = frame.run_id.map(lambda k:mapping[k].min())
        frame['target_max_c'] = frame.run_id.map(lambda k:mapping[k].max())
        frame['in_21_26'] = frame.target_min_c.ge(21-1e-9) & frame.target_max_c.le(26+1e-9)
        assert frame.target_min_c.min() >= 20-1e-9 and frame.target_max_c.max() <= 27+1e-9
    restricted = evaluate_thresholds(scores[scores.in_21_26], refs, [1.0])
    keys = ['climate','composition']
    paired = nominal.merge(restricted,on=keys,suffixes=('_20_27','_21_26'))
    nominal.to_csv(out/'NOMINAL_SELECTIONS.csv',index=False)
    restricted.to_csv(out/'RESTRICTED_21_26_SELECTIONS.csv',index=False)
    paired.to_csv(out/'PAIRED_DOMAIN_SELECTIONS.csv',index=False)
    scores.to_csv(out/'CANDIDATE_DOMAINS.csv',index=False)
    summaries = pd.concat([summarise(nominal).assign(command_domain='20–27 °C'),
                           summarise(restricted).assign(command_domain='21–26 °C')],ignore_index=True)
    summaries.to_csv(out/'DOMAIN_SUMMARY.csv',index=False)
    trajectories = Trajectories(matrix)
    curves = []
    for row in probes.itertuples():
        data = trajectories.load(row.hourly_path,row.hourly_sha256)
        curves.append(dict(climate=row.climate,target_c=row.target_c,**energy_metrics(data)))
    pd.DataFrame(curves).to_csv(out/'UNIFORM_COMMAND_ENERGY_CURVES.csv',index=False)
    source_paths = [results/'budget_frontiers/ALL_OPERATION_LIBRARY.csv',
                    results/'service_threshold_sensitivity/PAIRED_OPERATION_SERVICE_SCORES.csv',
                    results/'service_threshold_sensitivity/REFERENCE_SERVICE_LIMITS.csv',probe_path,command_path,*sources]
    metadata = dict(domain_rule='Retain a candidate when all 15 occupied command midpoints lie in the closed 21–26 °C interval',
                    reference='Same 25 °C reference temperatures, fixed group of 27 profiles and service limits',
                    library_entries=len(library),restricted_library_entries=int(library.in_21_26.sum()),
                    physical_operations_within_settings=len(scores),restricted_physical_operations_within_settings=int(scores.in_21_26.sum()),
                    source_files={str(p):digest(p) for p in source_paths},trajectory_files=trajectories.used)
    (out/'COMMAND_DOMAIN_METHODS.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(summaries[['command_domain','settings','qualifying_set_differences','different_choices']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=CODE/'results/new_run/strengthening/results')
    parser.add_argument('--matrix-root',type=Path,default=CODE/'results/new_run/matrix')
    parser.add_argument('--output-dir',type=Path,default=CODE/'results/new_run/strengthening/results/command_domain_sensitivity')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
