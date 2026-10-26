"""Compare iteration-warning trajectories with a targeted 40-iteration rerun."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parent))
from analyse_repaired_controls import load_hourly, score  # noqa: E402

STAMP=re.compile(r'(\d{1,2})/(\d{1,2})\s+\d{1,2}:\d{2}\s+-\s+(\d{1,2}):\d{2}')


def one(root:Path, pattern:str, require_identical:bool=True)->Path:
    matches=list(root.rglob(pattern))
    if not matches: raise ValueError(f'Expected one {pattern}, found {matches}')
    if require_identical:
        digests={hashlib.sha256(path.read_bytes()).hexdigest() for path in matches}
        if len(digests)!=1: raise ValueError(f'Duplicate files disagree for {pattern}: {matches}')
    preferred=[path for path in matches if 'pilot' not in str(path).lower()]
    return sorted(preferred or matches)[0]


def warning_rows(run_id:str, err:Path, calendar:dict[str,np.ndarray], label:str)->list[dict]:
    text=err.read_text(encoding='utf-8',errors='replace')
    rows=[]
    for line in text.splitlines():
        if 'SimHVAC: Maximum iterations' not in line: continue
        match=STAMP.search(line)
        occupied=None; stamp=''
        if match:
            month,day,hour=map(int,match.groups());stamp=f'{month:02d}/{day:02d} {hour:02d}:00'
            idx=np.flatnonzero((calendar['month']==month)&(calendar['day']==day)&(calendar['hour']==hour))
            occupied=bool(calendar['occupied'][idx[0]]) if len(idx)==1 else None
        rows.append({'source_run_id':run_id,'run':label,'timestamp':stamp,'occupied_hour':occupied,'warning_line':line.strip()})
    return rows


def build(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    new_manifest=pd.read_csv(args.new_root/'RUN_MANIFEST.csv')
    pairs=json.loads(args.targets.read_text(encoding='utf-8'))['pairs']
    tails=pd.read_csv(args.tails)
    profiles={'ASHRAE':pd.read_csv(args.ashrae_profiles),'CTCD':pd.read_csv(args.ctcd_profiles)}
    anchor_manifest=pd.read_csv(args.anchor_manifest)
    anchor=anchor_manifest[(anchor_manifest.context=='NewDelhi')&anchor_manifest.variant.eq(3)&anchor_manifest.engineering_status.isin(['cached_qualified','new_qualified'])].iloc[0]
    anchor_path=Path(anchor.path_hourly)
    if not anchor_path.is_absolute(): anchor_path=args.anchor_manifest.parent/anchor_path
    calendar=load_hourly(anchor_path.resolve())
    mapping=new_manifest.drop_duplicates(['source_run_id','run_id']).set_index('source_run_id').run_id.to_dict()
    run_ids=sorted(set(mapping))
    warning=[];run_rows=[]
    for old_id in run_ids:
        new_id=mapping[old_id]
        old_hourly=one(args.old_root,f'{old_id}.npz');new_hourly=args.new_root/'hourly'/f'{new_id}.npz'
        old=load_hourly(old_hourly);new=load_hourly(new_hourly)
        old_err=one(args.old_root,f'{old_id}/eplusout.err',require_identical=False);new_err=args.new_root/'runs'/new_id/'eplusout.err'
        old_w=warning_rows(old_id,old_err,calendar,'original');new_w=warning_rows(old_id,new_err,calendar,'40_iterations')
        warning.extend(old_w+new_w)
        delta_temp=new['temperatures']-old['temperatures'];occupied=old['occupied'].astype(bool)
        old_site=float(old['hvac_electric_kwh'].sum()+old['gas_hhv_kwh'].sum());new_site=float(new['hvac_electric_kwh'].sum()+new['gas_hhv_kwh'].sum())
        run_rows.append({'source_run_id':old_id,'rerun_id':new_id,'original_warning_occurrences':len(old_w),'rerun_warning_occurrences':len(new_w),'original_warnings_in_occupied_hours':sum(r['occupied_hour'] is True for r in old_w),'rerun_warnings_in_occupied_hours':sum(r['occupied_hour'] is True for r in new_w),'maximum_absolute_temperature_difference_c':float(np.abs(delta_temp).max()),'maximum_occupied_temperature_difference_c':float(np.abs(delta_temp[occupied]).max()),'temperature_rmse_c':float(np.sqrt(np.mean(delta_temp**2))),'annual_site_energy_difference_kwh':new_site-old_site,'annual_site_energy_difference_percent':100*(new_site-old_site)/old_site,'rerun_engineering_pass':bool(new_manifest[new_manifest.run_id.eq(new_id)].engineering_pass.iloc[0])})
    logical=[]
    new_by_source={r.source_run_id:r for r in new_manifest.itertuples(index=False)}
    cache={}
    def data(path):
        key=str(path)
        if key not in cache: cache[key]=load_hourly(path)
        return cache[key]
    for p in pairs:
        source=p['data_source'];prof=profiles[source]
        prof=prof[(prof.context==p['context'])&(prof.composition_id==p['composition_id'])&(prof.layout==p['layout'])].sort_values('profile_id').reset_index(drop=True)
        tail_ids=set(tails[(tails.data_source==source)&(tails.context==p['context'])&(tails.composition_id==p['composition_id'])&(tails.layout==p['layout'])].profile_id)
        tail=np.flatnonzero(prof.profile_id.isin(tail_ids))
        metrics={}
        for role,old_id in [('alternative',p['alternative_run_id']),('reference',p['reference_run_id'])]:
            new_id=mapping[old_id]
            old=data(one(args.old_root,f'{old_id}.npz'));new=data(args.new_root/'hourly'/f'{new_id}.npz')
            for version,d in [('original',old),('rerun',new)]:
                service=score(d,prof,tail)['summary']
                metrics[(role,version)]={'site':float(d['hvac_electric_kwh'].sum()+d['gas_hhv_kwh'].sum()),'mean':float(service['mean_reference_mismatch_c']),'tail':float(service['fixed_tail_reference_mismatch_c']),'coverage':float(service['modeled_interval_person_hour_coverage'])}
        row={k:p[k] for k in ['data_source','context','composition_id','layout','permission','alternative']}
        for name in ['site','mean','tail','coverage']:
            old_delta=metrics[('alternative','original')][name]-metrics[('reference','original')][name]
            new_delta=metrics[('alternative','rerun')][name]-metrics[('reference','rerun')][name]
            row[f'original_delta_{name}']=old_delta;row[f'rerun_delta_{name}']=new_delta;row[f'change_in_pair_delta_{name}']=new_delta-old_delta
        logical.append(row)
    runs=pd.DataFrame(run_rows);comp=pd.DataFrame(logical);warnings=pd.DataFrame(warning)
    runs.to_csv(out/'ENGINEERING_ITERATION_STABILITY.csv',index=False)
    comp.to_csv(out/'ENGINEERING_ITERATION_PAIR_COMPARISON.csv',index=False)
    warnings.to_csv(out/'ENGINEERING_WARNING_TIMES.csv',index=False)
    criteria={'maximum_occupied_temperature_difference_c':0.05,'absolute_annual_site_energy_difference_percent':0.10,'absolute_pair_mean_or_tail_difference_c':0.01,'absolute_pair_coverage_difference':0.005}
    passed=bool(runs.maximum_occupied_temperature_difference_c.le(criteria['maximum_occupied_temperature_difference_c']).all() and runs.annual_site_energy_difference_percent.abs().le(criteria['absolute_annual_site_energy_difference_percent']).all() and comp[['change_in_pair_delta_mean','change_in_pair_delta_tail']].abs().max().max()<=criteria['absolute_pair_mean_or_tail_difference_c'] and comp.change_in_pair_delta_coverage.abs().max()<=criteria['absolute_pair_coverage_difference'])
    report={'status':'PASS' if passed else 'FAIL','scope':'29 unique physical runs in the 23 primary-only New Delhi direct comparisons','declared_stability_criteria':criteria,'runs':len(runs),'pairs':len(comp),'reruns_passing_strict_engineering_gate':int(runs.rerun_engineering_pass.sum()),'original_warning_occurrences':int(runs.original_warning_occurrences.sum()),'rerun_warning_occurrences':int(runs.rerun_warning_occurrences.sum()),'maximum_occupied_temperature_difference_c':float(runs.maximum_occupied_temperature_difference_c.max()),'maximum_absolute_annual_site_energy_difference_percent':float(runs.annual_site_energy_difference_percent.abs().max()),'maximum_absolute_change_in_pair_mean_mismatch_c':float(comp.change_in_pair_delta_mean.abs().max()),'maximum_absolute_change_in_pair_tail_mismatch_c':float(comp.change_in_pair_delta_tail.abs().max()),'maximum_absolute_change_in_pair_interval_coverage':float(comp.change_in_pair_delta_coverage.abs().max()),'energy_direction_changes':int((np.sign(comp.original_delta_site)!=np.sign(comp.rerun_delta_site)).sum()),'building_physics_changed':False,'solver_iteration_limit':40}
    (out/'ENGINEERING_STABILITY_QA.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--new-root',type=Path,required=True);p.add_argument('--old-root',type=Path,required=True);p.add_argument('--targets',type=Path,required=True);p.add_argument('--ashrae-profiles',type=Path,required=True);p.add_argument('--ctcd-profiles',type=Path,required=True);p.add_argument('--tails',type=Path,required=True);p.add_argument('--anchor-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);build(p.parse_args())
if __name__=='__main__':main()
