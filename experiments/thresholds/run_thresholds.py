#!/usr/bin/env python3
"""Reproduce the linear threshold-ratio calculation for Figure 18."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import threshold_core as core
from threshold_helpers import law_grid, production_numerics

K_KEYS = ['1','2','4','8','16','32','64','128','inf']
RATIO_BETAS = [0.1,0.3111111111111111,0.5222222222222223,
               0.7333333333333333,0.9444444444444444,1.0]
RATIO_RHOS = [0.05,0.25,0.5,0.75,0.95]
RATIO_LAWS = ['homogeneous','bounded_matched_lognormal_moments']
ALL_LAWS = RATIO_LAWS + ['bounded_heterogenous','lognormal_legacy_grid']


def write_json(path, value):
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if isinstance(v,np.generic): return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v): return None
        return v
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def write_csv(path, frame):
    tmp=path.with_suffix('.csv.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def source_hashes():
    base=Path(__file__).resolve().parent
    return {name:hashlib.sha256((base/name).read_bytes()).hexdigest() for name in
            ('run_thresholds.py','threshold_core.py','threshold_helpers.py','gamma_distributions.py')}


def load_config(root):
    config=json.loads((root/'config.json').read_text())
    if config['source_sha256']!=source_hashes():
        raise ValueError('Source changed after preparation; prepare a new run directory')
    return config


def prepare(args):
    laws=args.laws or RATIO_LAWS
    betas=args.betas or RATIO_BETAS
    rhos=args.rhos or RATIO_RHOS
    ks=args.k_values or K_KEYS
    for values in (laws,betas,rhos,ks):
        if len(set(values))!=len(values): raise ValueError('Grid entries must be unique')
    if any(not np.isfinite(b) or b<=0 for b in betas): raise ValueError('Need beta>0')
    if any(not np.isfinite(r) or not 0<r<1 for r in rhos): raise ValueError('Need 0<rho<1')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use an empty output directory')
    tasks=[dict(task_id=i,gamma_dist=law,beta=float(beta),rho=float(rho),K_key=k)
           for i,(law,beta,rho,k) in enumerate((law,beta,rho,k) for law in laws for beta in betas for rho in rhos for k in ks)]
    config=dict(preset=args.preset,activation='linear',laws=laws,betas=betas,rhos=rhos,k_values=ks,
                tasks=tasks,task_count=len(tasks),quadrature_order=200,
                matched_target_second_moment=1.0927746165423835,
                legacy=dict(s=1.,eta_n=.75,eta_h=.25,n_points=200,tail_eps=1e-5,
                            renormalize_lognormal_mean=True),
                solver=dict(n_scan=36,bisect_steps=24,damping=.22,tol=2e-9,max_iter=4000,ridge=0.),
                numerics=production_numerics(),alpha_continuation_warm_start=True,
                scan_descending=False,source_sha256=source_hashes())
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'tasks').mkdir()
    write_json(args.output/'config.json',config)
    print(json.dumps(dict(preset=args.preset,tasks=len(tasks),task_id_range=[0,len(tasks)-1])))


def variance_grid(config, law):
    if law!='lognormal_legacy_grid':
        return law_grid(law,config['quadrature_order'],config['matched_target_second_moment'])
    params=config['legacy']
    gamma,weights=core.generalized_gamma_grid(**params)
    summary=core.noise_summary(gamma,weights,s=params['s'],eta_n=params['eta_n'],eta_h=params['eta_h'])
    meta=dict(gamma_dist=law,gamma_low=float(gamma.min()),gamma_high=float(gamma.max()),
              quadrature='legacy-quantile',quadrature_order=len(gamma),truncated=True,
              gamma_mean_grid=float(summary['E[gamma]']),
              gamma_second_moment_grid=float(summary['E[gamma^2] grid']),
              gamma_second_moment_exact=float(summary['E[gamma^2] exact']),
              gamma_second_moment_relative_error=float(summary['E[gamma^2] relative error']),
              s=params['s'],eta_n=params['eta_n'],eta_h=params['eta_h'],
              legacy_n_gamma=params['n_points'],legacy_tail_eps=params['tail_eps'],
              physical_gamma_support_unbounded=True,numerical_gamma_support_unbounded=False)
    return gamma,weights,meta


def calculate(config, task):
    start=time.perf_counter()
    gamma,weights,meta=variance_grid(config,task['gamma_dist'])
    k=float('inf') if task['K_key']=='inf' else int(task['K_key'])
    result=core.critical_alpha(beta=task['beta'],rho=task['rho'],K=k,activation=config['activation'],
                              gamma=gamma,weights=weights,numerics=config['numerics'],
                              gamma_support_unbounded=False,gamma_lower_bound=meta['gamma_low'],
                              scan_descending=config['scan_descending'],**config['solver'])
    row=core.threshold_row(result)
    tau=float(result.get('tau_c',np.nan))
    support=min(1+tau*meta['gamma_low'],1+tau*meta['gamma_high'])
    if task['gamma_dist']!='lognormal_legacy_grid' and row.get('status')=='converged' and not support>0:
        row['status']='failed: true bounded support curvature'
    saddle=result.get('zero_saddle',{})
    row.update(task,activation=config['activation'],K=k,K_numeric=k,**{key:val for key,val in meta.items() if key!='gamma_dist'})
    row.update(noise_discretization=meta['quadrature'],ridge=config['solver']['ridge'],
               alpha_continuation_warm_start=True,runtime_seconds=time.perf_counter()-start,
               regularity_true_support_min=support,critical_gain=result.get('gain',np.nan),
               critical_gain_residual=abs(result.get('gain',np.nan)-1),
               zero_saddle_status=saddle.get('status','unavailable'),
               zero_saddle_residual=saddle.get('residual',np.nan),zero_saddle_iterations=saddle.get('n_iter',0))
    return row,result.get('curve',pd.DataFrame())


def task(args):
    config=load_config(args.output)
    index=args.task_id if args.task_id is not None else os.environ.get('SLURM_ARRAY_TASK_ID')
    if index is None: raise ValueError('Provide --task-id or SLURM_ARRAY_TASK_ID')
    index=int(index)
    if not 0<=index<len(config['tasks']): raise IndexError('Task ID outside prepared grid')
    t=config['tasks'][index];folder=args.output/'tasks'/f'task_{index:04d}'
    if (folder/'COMPLETED.json').exists():
        print(f'Task {index} already complete; preserved');return 0
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'progress.json',dict(**t,phase='threshold_search'))
    row,curve=calculate(config,t)
    write_csv(folder/'result.csv',pd.DataFrame([row]))
    write_csv(folder/'zero_branch.csv',curve)
    write_json(folder/'COMPLETED.json',row)
    print(json.dumps({key:row[key] for key in ('task_id','gamma_dist','beta','rho','K_key','status')})+
          f" alpha_c={row.get('alpha_c')} seconds={row['runtime_seconds']:.3f}",flush=True)
    return 0 if row['status']=='converged' else 2


def ratio_tables(raw):
    rows=[]
    for _,group in raw.groupby(['gamma_dist','activation','beta','rho'],sort=False):
        dynamic=group[group.K_key.eq('inf')]
        dynamic_value=np.nan
        if len(dynamic)==1 and dynamic.iloc[0].status=='converged' and np.isfinite(dynamic.iloc[0].alpha_c) and dynamic.iloc[0].alpha_c>0:
            dynamic_value=float(dynamic.iloc[0].alpha_c)
        for row in group.to_dict('records'):
            row['alpha_dynamic']=dynamic_value
            valid=row['status']=='converged' and np.isfinite(row['alpha_c']) and row['alpha_c']>0 and np.isfinite(dynamic_value)
            row['threshold_ratio']=row['alpha_c']/dynamic_value if valid else np.nan
            row['ratio_status']='available' if valid else 'unavailable'
            row['inverse_K']=0. if row['K_key']=='inf' else 1/float(row['K_key'])
            row['scaled_threshold']=row['beta']**2*row['alpha_c']
            rows.append(row)
    all_ratios=pd.DataFrame(rows)
    finite=all_ratios[np.isfinite(all_ratios.K_numeric)].copy()
    available=finite[finite.ratio_status.eq('available')].copy()
    return all_ratios,available


def aggregate(args):
    config=load_config(args.output);frames=[];missing=[]
    for t in config['tasks']:
        folder=args.output/'tasks'/f"task_{t['task_id']:04d}"
        if not (folder/'COMPLETED.json').exists():missing.append(t['task_id']);continue
        f=pd.read_csv(folder/'result.csv',dtype={'K_key':str},float_precision='round_trip')
        if len(f)!=1:raise ValueError('Expected one result per task')
        for key,value in t.items():
            if key in ('beta','rho'):
                if not np.isclose(f.iloc[0][key],value,rtol=1e-12,atol=0):raise ValueError(f'{key} mismatch')
            elif f.iloc[0][key]!=value:raise ValueError(f'{key} mismatch')
        frames.append(f)
    if missing and not args.allow_missing:raise RuntimeError(f'{len(missing)} tasks missing; --allow-missing explicitly permits partial aggregation')
    if not frames:raise RuntimeError('No completed results')
    raw=pd.concat(frames,ignore_index=True)
    if raw.duplicated(['gamma_dist','activation','beta','rho','K_key']).any():raise ValueError('Duplicate threshold')
    ratios,available=ratio_tables(raw)
    write_csv(args.output/'all_thresholds.csv',ratios)
    write_csv(args.output/'validated_ratio_data_matched_three_laws_grid.csv',available)
    coverage=dict(expected_tasks=len(config['tasks']),completed_tasks=len(raw),missing_task_ids=missing,
                  partial=bool(missing),status_counts={str(k):int(v) for k,v in raw.status.value_counts().items()},
                  available_finite_ratios=len(available))
    write_json(args.output/'coverage.json',coverage);print(json.dumps(coverage))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preset',choices=['ratios'],default='ratios')
    p.add_argument('--laws',choices=ALL_LAWS,nargs='+');p.add_argument('--betas',type=float,nargs='+')
    p.add_argument('--rhos',type=float,nargs='+');p.add_argument('--k-values',choices=K_KEYS,nargs='+')
    t=sub.add_parser('task');t.add_argument('--output',type=Path,required=True);t.add_argument('--task-id',type=int)
    a=sub.add_parser('aggregate');a.add_argument('--output',type=Path,required=True);a.add_argument('--allow-missing',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args);return 0
    if args.command=='task':return task(args)
    aggregate(args);return 0


if __name__=='__main__':raise SystemExit(main())
