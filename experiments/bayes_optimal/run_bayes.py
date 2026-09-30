#!/usr/bin/env python3
"""Gaussian-prior Bayes reference and raw-feature downstream prediction."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from bayes_core import variance_quadrature, alpha_c_BO_from_grid, run_bayes_gaussian_prior_se, flatten_result_for_dataframe

PAPER_ALPHAS = [0.08999999999999997, 0.09443234181700585, 0.09908296867826477, 0.10396263073855093, 0.10908260757886667, 0.1144547342797539, 0.12009142877866998, 0.12600572057466644, 0.1322112808467225, 0.13872245405535402, 0.1455542911005453, 0.15272258411265188, 0.160243902956693, 0.1681356335344172, 0.17641601797267686, 0.18510419679100965, 0.19422025314589936, 0.2037852592539886, 0.2138213251015534, 0.2243516495528336, 0.2354005739753588, 0.24699363850622727, 0.2591576410894004, 0.27192069942047975, 0.28531231594215617, 0.2993634460405695, 0.31410656960022054, 0.3295757660828352, 0.33, 0.33235, 0.3347, 0.33705, 0.3394, 0.34175, 0.3441, 0.3458067933037317, 0.34645, 0.3488, 0.35115, 0.3535, 0.35585, 0.3582, 0.36055, 0.3628371700877854, 0.3629, 0.36525, 0.3676, 0.36995, 0.3723, 0.37465, 0.377, 0.37935, 0.38070626299605376, 0.3817, 0.38405, 0.3864, 0.38875, 0.3911, 0.39345, 0.3958, 0.39815, 0.3994553773235364, 0.4005, 0.40285, 0.4052, 0.40755, 0.4099, 0.41225, 0.4146, 0.41695, 0.4191278525784138, 0.4193, 0.42165, 0.424, 0.42635, 0.4287, 0.43105, 0.4334, 0.43575, 0.4381, 0.43976916266347116, 0.44045, 0.4428, 0.44515, 0.4475, 0.44985, 0.4522, 0.45455, 0.4569, 0.45925, 0.46142702099128174, 0.4616, 0.46395, 0.4663, 0.46865, 0.471, 0.47335, 0.4757, 0.47805, 0.4804, 0.48275, 0.4841514907761273, 0.4851, 0.48745, 0.4898, 0.49215, 0.4945, 0.49685, 0.4992, 0.50155, 0.5039, 0.5062500000000001, 0.5079951007576022, 0.5086, 0.51095, 0.5133000000000001, 0.51565, 0.518, 0.5203500000000001, 0.5227, 0.52505, 0.5274000000000001, 0.52975, 0.5321, 0.5330129666234025, 0.5344500000000001, 0.5368, 0.53915, 0.5415000000000001, 0.5438500000000001, 0.5462, 0.5485500000000001, 0.5509000000000001, 0.55325, 0.5556000000000001, 0.5579500000000001, 0.559262918411972, 0.5603, 0.56265, 0.5650000000000001, 0.56735, 0.5697, 0.5720500000000001, 0.5744, 0.57675, 0.5791000000000001, 0.58145, 0.5838000000000001, 0.58615, 0.5868056341895066, 0.5885, 0.5908500000000001, 0.5932, 0.59555, 0.5979000000000001, 0.60025, 0.6026, 0.6049500000000001, 0.6073, 0.60965, 0.6120000000000001, 0.61435, 0.6157047803103154, 0.6167, 0.6190500000000001, 0.6214, 0.62375, 0.6261000000000001, 0.62845, 0.6308, 0.6331500000000001, 0.6355, 0.63785, 0.6402000000000001, 0.64255, 0.6449, 0.6460271585847579, 0.6472500000000001, 0.6496, 0.65195, 0.6543000000000001, 0.65665, 0.659, 0.6613500000000001, 0.6637, 0.66605, 0.6684000000000001, 0.67075, 0.6731, 0.6754500000000001, 0.6778, 0.6778428606949434, 0.68015, 0.6825000000000001, 0.68485, 0.6872, 0.6895500000000001, 0.6919, 0.69425, 0.6966000000000001, 0.69895, 0.7013, 0.7036500000000001, 0.706, 0.70835, 0.7107000000000001, 0.7112254302151331, 0.71305, 0.7154, 0.7177500000000001, 0.7201, 0.72245, 0.7248000000000001, 0.72715, 0.7295, 0.7318500000000001, 0.7342, 0.73655, 0.7389000000000001, 0.74125, 0.7436, 0.7459500000000001, 0.746252032611361, 0.7483, 0.75065, 0.7530000000000001, 0.75535, 0.7577, 0.7600500000000001, 0.7624, 0.76475, 0.7671000000000001, 0.76945, 0.7718, 0.7741500000000001, 0.7765, 0.77885, 0.7812000000000001, 0.7830036336132382, 0.78355, 0.7859, 0.78825, 0.7906, 0.79295, 0.7953, 0.79765, 0.8, 0.8215651863702547, 0.8620258278252001, 0.9044790847586306, 0.9490230899806621, 0.9957608091698334, 1.044800278883378, 1.0962548562890913, 1.1502434811960451, 1.2068909509898635, 1.2663282091080668, 1.3286926477223318, 1.3941284253273083, 1.4627867999701392, 1.534826478890937, 1.6104139853824502, 1.6897240437169128, 1.7729399830298767, 1.860254161094598, 1.9518684089665843, 2.047994497526088, 2.148854626997007, 2.2546819405737337, 2.365721063343209, 2.4822286677479615, 2.6044740668971835, 2.7327398370973586, 2.8673224710414065, 3.008533063166267, 3.156698028763117, 3.3121598585025254, 3.4752779101186237, 3.646429239082352, 3.8260094701838794, 4.014433712038952, 4.212137516633042, 4.419577886121397, 4.637234329212179, 4.865609969574646, 5.105232708834495, 5.3566564468446405, 5.620462362052236, 5.897260254921474, 6.187689957517644, 6.492422812510671, 6.812163225017048, 7.147650290867185, 7.4996595050621275, 7.869004554368707, 8.256539198196908, 8.663159242107085, 9.089804608509002, 9.537461509339078, 10.007164725738267, 10.5]


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def prepare(args):
    if (args.output/'config.json').exists():
        raise FileExistsError('Choose a new output directory; config.json already exists.')
    alphas=args.alphas if args.alphas is not None else PAPER_ALPHAS
    if args.beta<=0 or not np.isfinite(args.beta) or any(a<=0 or not np.isfinite(a) for a in alphas):
        raise ValueError('alpha and beta must be finite and positive.')
    if len(set(alphas))!=len(alphas):
        raise ValueError('Duplicate alpha values.')
    if not (0<args.damping<=1 and args.max_iter>=1 and args.tol>0 and np.isfinite(args.tol)
            and 0<args.q_lambda_init<=1 and args.order>=2 and args.n_ds>=1 and args.mc_samples>=1 and args.mc_seed>=0):
        raise ValueError('Invalid numerical settings.')
    gamma,weights=variance_quadrature(args.law,args.order)
    c=dict(law=args.law,beta=args.beta,alphas=alphas,quadrature_order=len(gamma),
           gamma_nodes=gamma.tolist(),gamma_weights=weights.tolist(),
           noise_discretization={'bounded':'gauss_legendre','homogeneous':'point_mass','unbounded':'gauss_hermite'}[args.law],
           alpha_c=alpha_c_BO_from_grid(args.beta,gamma,weights),
           q_lambda_init=args.q_lambda_init,max_iter=args.max_iter,tol=args.tol,damping=args.damping,
           warm_start=False,n_ds=args.n_ds,mc_samples=args.mc_samples,mc_seed=args.mc_seed,
           downstream_feature='raw_encoder_preactivation',
           tasks=[dict(task_id=i,alpha=a) for i,a in enumerate(alphas)])
    c['task_count']=len(c['tasks'])
    write_json(args.output/'config.json',c)
    print(f"Prepared {c['task_count']} independent tasks; alpha_c={c['alpha_c']:.12g}")


def raw_probe_prediction(result,gamma,weights,n_ds,mc_samples,mc_seed):
    """Original exported scalar-logistic-probe prediction for Gaussian labels.

    This is the raw feature readout used in the comparison plot, not the
    inverse-variance Bayesian head implemented in amp_downstream.py.
    """
    t=result['alpha']*result['beta']*result['q_lambda']
    qu=t/(gamma+t)
    M=float(weights@qu); C=float(weights@(gamma*qu))
    snr=result['beta']*M*M/C if C>0 else 0.
    score=np.abs(np.random.default_rng(mc_seed).normal(size=(mc_samples,n_ds))).sum(axis=1)/np.sqrt(n_ds)
    orientation=float(ndtr(np.sqrt(snr)*score).mean())
    accuracy=.5+(2*orientation-1)*np.arctan(np.sqrt(snr))/np.pi
    return dict(bayes_M=M,bayes_C=C,bayes_snr_rep=snr,
                bayes_downstream_test_accuracy=float(accuracy),raw_probe_orientation_probability=orientation)


def task(args):
    c=json.loads(args.config.read_text())
    index=args.task_id if args.task_id is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0<=index<len(c['tasks']):
        raise ValueError('Task index outside grid.')
    item=c['tasks'][index]
    path=args.config.parent/'tasks'/f'task_{index:04d}.csv'
    if path.exists():
        raise FileExistsError(f'{path} already exists; choose a new run directory for reruns.')
    gamma=np.asarray(c['gamma_nodes']); weights=np.asarray(c['gamma_weights'])
    result=run_bayes_gaussian_prior_se(item['alpha'],c['beta'],gamma,weights,
            **{k:c[k] for k in ('q_lambda_init','max_iter','tol','damping')})
    row=flatten_result_for_dataframe(result)
    row.update(task_id=index,law=c['law'],alpha_c=c['alpha_c'],n_gamma=len(gamma),
               n_ds=c['n_ds'],downstream_mc_samples=c['mc_samples'],downstream_mc_seed=c['mc_seed'])
    row.update(raw_probe_prediction(result,gamma,weights,c['n_ds'],c['mc_samples'],c['mc_seed']))
    row.update(bayes_optimal_cosine=result['cosine'],bayes_optimal_cosine_sq=result['cosine_sq'],
               bayes_optimal_status=result['status'],bayes_q_lambda=result['q_lambda'],
               bayes_saddle_status=result['status'],bayes_saddle_n_iter=result['n_iter'],bayes_saddle_n_gamma=len(gamma))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.csv.tmp')
    pd.DataFrame([row]).to_csv(temporary,index=False); temporary.replace(path)
    print(f"Task {index}: alpha={item['alpha']:g}, cosine={result['cosine']:.8g}, status={result['status']}")
    return 0 if result['status']=='converged' else 2


def aggregate(args):
    c=json.loads(args.config.read_text()); root=args.config.parent
    rows=[]; missing=[]
    for item in c['tasks']:
        path=root/'tasks'/f"task_{item['task_id']:04d}.csv"
        if not path.exists():
            missing.append(item); continue
        frame=pd.read_csv(path)
        if len(frame)!=1 or frame.iloc[0].task_id!=item['task_id'] or not np.isclose(frame.iloc[0].alpha,item['alpha'],rtol=1e-13,atol=0) or frame.iloc[0].law!=c['law']:
            raise ValueError(f'Invalid result: {path}')
        rows.append(frame)
    if missing and not args.allow_missing:
        raise RuntimeError(f'{len(missing)} missing tasks; use --allow-missing for a partial export.')
    if not rows:
        raise RuntimeError('No saved results.')
    frame=pd.concat(rows,ignore_index=True).sort_values('alpha')
    frame.to_csv(root/'bayes_optimal.csv',index=False)
    counts=frame.status.value_counts().to_dict()
    write_json(root/'coverage.json',dict(expected=len(c['tasks']),saved=len(frame),statuses=counts,missing=missing))
    print(f'Saved {len(frame)}/{len(c["tasks"])} results; statuses={counts}')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--output',type=Path,required=True)
    q.add_argument('--law',choices=('bounded','homogeneous','unbounded'),default='bounded')
    q.add_argument('--alphas',nargs='+',type=float)
    q.add_argument('--beta',type=float,default=1.)
    q.add_argument('--order',type=int,default=40)
    q.add_argument('--q-lambda-init',type=float,default=1e-3)
    q.add_argument('--max-iter',type=int,default=20000)
    q.add_argument('--tol',type=float,default=1e-12)
    q.add_argument('--damping',type=float,default=.2)
    q.add_argument('--n-ds',type=int,default=100)
    q.add_argument('--mc-samples',type=int,default=100000)
    q.add_argument('--mc-seed',type=int,default=12345)
    q=sub.add_parser('task'); q.add_argument('--config',type=Path,required=True); q.add_argument('--task-id',type=int)
    q=sub.add_parser('aggregate'); q.add_argument('--config',type=Path,required=True); q.add_argument('--allow-missing',action='store_true')
    a=p.parse_args(); return globals()[a.command](a)

if __name__=='__main__':
    raise SystemExit(main())
