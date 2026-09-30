#!/usr/bin/env python3
"""Run the recorded AMP presets; see README.md for initialization and readouts."""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch

from amp_spike import fit_amp, squared_cosine
from amp_downstream import fit_amp_downstream
import synthetic_data as data

PAPER_PRESETS = {'bounded': {'d': 4000, 'beta': 1.0, 'gamma_dist': 'bounded_heterogenous', 'gamma_sigma': 1.0, 'gamma_floor': 0.75, 'gamma_eta': 0.25, 'n_ds': 100, 'probe_n_test': 10000, 'alphas': [0.1, 0.1333521432163324, 0.1778279410038923, 0.2371373705661655, 0.3162277660168379, 0.4216965034285823, 0.5623413251903491, 0.7498942093324559, 1.0, 1.333521432163324, 1.7782794100389228, 2.3713737056616555, 3.1622776601683795, 4.216965034285822, 5.62341325190349, 7.498942093324557, 10.0], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'amp': {'max_iter': 20000, 'min_iter': 30, 'tol': 1e-06, 'damping': 0.75, 'init_std': 0.31622776601683794, 'patience': 5, 'variance_floor': 1e-12, 'relative_variance_floor': 1e-08, 'zero_u_init': False, 'adaptive_cycle_damping': True}, 'oracle_initialization_plant': 0.05, 'gamma_bounded_a': 0.25}, 'homogeneous': {'d': 4000, 'beta': 1.0, 'gamma_dist': 'lognormal', 'gamma_sigma': 0.0, 'gamma_floor': 0.75, 'gamma_eta': 0.25, 'n_ds': 100, 'probe_n_test': 10000, 'alphas': [0.5, 0.6029542755153482, 0.7271077167244768, 0.8768254131184519, 1.057371263440564, 1.2750930481971074, 1.537645610180688, 1.8542599897717045, 2.23606797749979, 2.696493494752912, 3.251724563121181, 3.9212824562643886, 4.728708045015879, 5.702389466812296, 6.876560219336321, 8.29250277017519, 10.0], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'amp': {'max_iter': 20000, 'min_iter': 30, 'tol': 1e-06, 'damping': 0.75, 'init_std': 0.31622776601683794, 'patience': 5, 'variance_floor': 1e-12, 'relative_variance_floor': 1e-08, 'zero_u_init': False, 'adaptive_cycle_damping': True}, 'oracle_initialization_plant': 0.05}, 'unbounded': {'d': 4000, 'beta': 1.0, 'gamma_dist': 'lognormal', 'gamma_sigma': 1.0, 'gamma_floor': 0.75, 'gamma_eta': 0.25, 'n_ds': 100, 'probe_n_test': 10000, 'alphas': [0.5, 0.6029542755153482, 0.7271077167244768, 0.8768254131184519, 1.057371263440564, 1.2750930481971074, 1.537645610180688, 1.8542599897717045, 2.23606797749979, 2.696493494752912, 3.251724563121181, 3.9212824562643886, 4.728708045015879, 5.702389466812296, 6.876560219336321, 8.29250277017519, 10.0], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'amp': {'max_iter': 20000, 'min_iter': 30, 'tol': 1e-06, 'damping': 0.75, 'init_std': 0.31622776601683794, 'patience': 5, 'variance_floor': 1e-12, 'relative_variance_floor': 1e-08, 'zero_u_init': False, 'adaptive_cycle_damping': True}, 'oracle_initialization_plant': 0.05}}




def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare(args):
    root = args.output
    if (root/'config.json').exists():
        raise FileExistsError('Choose a new output directory; config.json already exists.')
    c = json.loads(json.dumps(PAPER_PRESETS[args.law]))
    c.update(law=args.law, initialization=args.initialization, amp_device=args.device)
    if args.initialization == 'random':
        c.pop('oracle_initialization_plant', None)
    for name in ('d', 'beta', 'n_ds', 'probe_n_test', 'alphas', 'seeds'):
        if getattr(args, name) is not None:
            c[name] = getattr(args, name)
    for name in ('max_iter', 'min_iter', 'tol', 'damping', 'init_std', 'patience'):
        if getattr(args, name) is not None:
            c['amp'][name] = getattr(args, name)
    if (c['d'] < 1 or c['n_ds'] < 1 or c['probe_n_test'] < 1 or c['beta'] <= 0
            or not np.isfinite(c['beta']) or any(not np.isfinite(a) or a <= 0 or round(a*c['d']) < 1 for a in c['alphas'])):
        raise ValueError('Dimensions, alpha, beta and sample counts must be positive.')
    if len(set(c['alphas'])) != len(c['alphas']) or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Duplicate alpha values or seeds.')
    if any(s < 0 for s in c['seeds']):
        raise ValueError('Seeds must be nonnegative.')
    o = c['amp']
    if not (0 < o['damping'] <= 1 and np.isfinite(o['tol']) and o['tol'] > 0
            and o['max_iter'] >= 1 and 1 <= o['min_iter'] <= o['max_iter'] and o['patience'] >= 1
            and np.isfinite(o['init_std']) and o['init_std'] > 0):
        raise ValueError('Invalid AMP iteration settings.')
    c['tasks'] = [dict(task_id=i, seed=s, alpha=a) for i,(s,a) in enumerate(
        (s,a) for s in c['seeds'] for a in c['alphas'])]
    c['task_count'] = len(c['tasks'])
    c['priors'] = dict(u='gaussian', sample_amplitude='gaussian')
    c['data_dtype'] = 'float32; AMP and probe computation in float64'
    c['variance_estimation'] = 'mean of squared observations, with recorded floors'
    c['probe'] = dict(feature='raw_encoder_preactivation', lr=1., max_iter=100, history_size=25)
    write_json(root/'config.json', c)
    print(f"Prepared {len(c['tasks'])} independent tasks; initialization={c['initialization']}; output={root}")


def task(args):
    c = json.loads(args.config.read_text())
    idx = args.task_id if args.task_id is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= idx < len(c['tasks']):
        raise ValueError('Task index outside grid.')
    item = c['tasks'][idx]
    output = args.config.parent/'tasks'/f'task_{idx:04d}.csv'
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a new run directory for reruns.')
    torch.set_num_threads(1)
    start = time.monotonic()
    cfg = data.experiment_config(c,item['alpha'])
    seed = item['seed']
    spike = data.make_quenched_spike(cfg,seed=10000*seed+1)
    train = data.make_spiked_heteroskedastic_data(cfg.n_train,spike,cfg,seed=10000*seed+2)
    x = train.x.to(device=c['amp_device'],dtype=torch.float64)
    del train
    options = dict(c['amp'])
    plant = c.get('oracle_initialization_plant')
    if plant is not None:
        z = torch.randn(cfg.d,generator=torch.Generator().manual_seed(10000*seed+71),dtype=x.dtype)
        options['initial_u'] = np.sqrt(plant)*spike.u_star.to(x.dtype)+np.sqrt(1-plant)*z
    inference_cfg = SimpleNamespace(beta=cfg.beta,u_prior=cfg.u_prior,lambda_prior=cfg.lambda_prior)
    result = fit_amp(x,inference_cfg,seed=10000*seed+70,**options)
    del x
    result = dataclasses.replace(result,w=result.w.cpu(),v=result.v.cpu(),gamma_hat=result.gamma_hat.cpu())
    cs = squared_cosine(result.w,spike.u_star)
    resolved = float(result.w.square().mean().sqrt()) > 1e-12
    row = dict(**item,law=c['law'],d=cfg.d,n_train=cfg.n_train,beta=cfg.beta,
               gamma_dist=cfg.gamma_dist,n_ds=c['n_ds'],probe_n_test=c['probe_n_test'],
               initialization=c['initialization'],oracle_initialization=plant is not None,
               oracle_plant=plant,cosine_sq=cs,cosine_abs=np.sqrt(cs),
               direction_resolved=resolved,cosine_resolved=np.sqrt(cs) if resolved else np.nan,
               converged=result.converged,n_iter=result.n_iter,residual=result.residual,
               q=float(result.w.square().mean()),w_norm=float(result.w.norm()),
               variance_floor=result.variance_floor,n_floored=result.n_floored,
               gamma_relative_rmse=float(((result.gamma_hat/spike.gamma.double()-1).square().mean()).sqrt()),
               amp_seconds=time.monotonic()-start)
    if plant is not None:
        row['initial_cosine_abs'] = np.sqrt(squared_cosine(options['initial_u'],spike.u_star))
    labeled = data.make_labeled_spiked_data(c['n_ds'],spike,cfg,seed=10000*seed+20)
    test = data.make_labeled_spiked_data(c['probe_n_test'],spike,cfg,seed=10000*seed+21)
    row.update(data.fit_scalar_logistic_probe(
        data.encoder_fields(result.w,labeled.x.double()),labeled.y.double(),
        data.encoder_fields(result.w,test.x.double()),test.y.double(),
        lr=1.,max_iter=100,history_size=25))
    row['bayesian_readout_accuracy'] = np.nan
    row['bayesian_readout_sign_log_odds'] = np.nan
    if result.converged:
        head = fit_amp_downstream(result,labeled.x,labeled.y)
        row['bayesian_readout_accuracy'] = head.accuracy(test.x,test.y)
        row['bayesian_readout_sign_log_odds'] = head.sign_log_odds
    row['elapsed_seconds'] = time.monotonic()-start
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.csv.tmp')
    pd.DataFrame([row]).to_csv(temporary,index=False)
    temporary.replace(output)
    print(f"Task {idx}: alpha={item['alpha']:g}, seed={seed}, cosine={row['cosine_abs']:.6g}, converged={result.converged}")
    return 0 if result.converged else 2


def aggregate(args):
    c = json.loads(args.config.read_text()); root = args.config.parent
    rows=[]; missing=[]
    for item in c['tasks']:
        path=root/'tasks'/f"task_{item['task_id']:04d}.csv"
        if not path.exists():
            missing.append(item); continue
        frame=pd.read_csv(path)
        if len(frame)!=1:
            raise ValueError(f'Invalid result length: {path}')
        r=frame.iloc[0]
        if r.task_id!=item['task_id'] or r.seed!=item['seed'] or not np.isclose(r.alpha,item['alpha'],rtol=1e-13,atol=0) or r.d!=c['d'] or r.initialization!=c['initialization']:
            raise ValueError(f'Result/config mismatch: {path}')
        rows.append(frame)
    if missing and not args.allow_missing:
        raise RuntimeError(f'{len(missing)} missing tasks; use --allow-missing for an explicitly partial summary.')
    if not rows:
        raise RuntimeError('No saved results.')
    df=pd.concat(rows,ignore_index=True)
    if df.duplicated(['alpha','seed']).any():
        raise ValueError('Duplicate alpha/seed results.')
    df.to_csv(root/'amp_per_seed.csv',index=False)
    metrics=['cosine_abs','cosine_sq','cosine_resolved','probe_test_accuracy','probe_test_loss',
             'bayesian_readout_accuracy','n_iter','residual','q','gamma_relative_rmse']
    named=dict(num_seeds=('seed','nunique'),num_converged=('converged','sum'),num_resolved=('direction_resolved','sum'))
    for metric in metrics:
        named[metric+'_count']=(metric,'count')
        for stat in ('mean','std','sem'):
            named[metric+'_'+stat]=(metric,stat)
    # Include terminal iterates and preserve convergence counts, as in the archive.
    df.groupby('alpha',as_index=False).agg(**named).to_csv(root/'amp_summary.csv',index=False)
    good=df[df.converged]
    good.groupby('alpha',as_index=False).agg(**named).to_csv(root/'amp_converged_summary.csv',index=False)
    write_json(root/'coverage.json',dict(expected=len(c['tasks']),saved=len(df),
               converged=int(df.converged.sum()),missing=missing))
    print(f'Saved {len(df)}/{len(c["tasks"])} results; {int(df.converged.sum())} converged.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare')
    q.add_argument('--law',choices=tuple(PAPER_PRESETS),default='bounded')
    q.add_argument('--output',type=Path,required=True)
    q.add_argument('--initialization',choices=('teacher-informed','random'),default='teacher-informed')
    q.add_argument('--device',default='cpu',help='cpu reproduces the recorded execution; cuda is optional.')
    for name in ('d','n-ds','probe-n-test','max-iter','min-iter','patience'):
        q.add_argument('--'+name,type=int)
    for name in ('beta','tol','damping','init-std'):
        q.add_argument('--'+name,type=float)
    q.add_argument('--alphas',nargs='+',type=float)
    q.add_argument('--seeds',nargs='+',type=int)
    q=sub.add_parser('task'); q.add_argument('--config',type=Path,required=True); q.add_argument('--task-id',type=int)
    q=sub.add_parser('aggregate'); q.add_argument('--config',type=Path,required=True); q.add_argument('--allow-missing',action='store_true')
    args=p.parse_args(); return globals()[args.command](args)

if __name__=='__main__':
    raise SystemExit(main())
