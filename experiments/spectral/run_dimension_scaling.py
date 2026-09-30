#!/usr/bin/env python3
"""Independent ordinary-PCA fits for the bounded-noise dimension sweep."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
import synthetic_data as base
from run_io import prepare_run, load_config, task_index, task_folder, write_json, write_rows, aggregate_run

DIMENSIONS = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
ALPHAS = [1.0, 1.5444521049463789, 2.385332304473301, 3.6840314986403864, 5.689810202763908, 8.7876393444041, 13.572088082974531, 20.96144000826768, 32.37394014347626, 50.0]
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

def solve(x, seed):
    n,d=x.shape
    calls=0
    def mv(v):
        nonlocal calls
        calls+=1
        return x.T@(x@v)/n
    op=LinearOperator((d,d),matvec=mv,dtype=np.float64)
    vals,vecs=eigsh(op,k=2,which='LA',tol=1e-9,maxiter=5000,ncv=min(40,d),
                    v0=np.random.default_rng(10000*seed+80).normal(size=d))
    order=np.argsort(vals)[::-1]
    vals,vecs=vals[order],vecs[:,order]
    residual=float(np.linalg.norm(mv(vecs[:,0])-vals[0]*vecs[:,0])/abs(vals[0]))
    if residual>1e-7: raise RuntimeError(f'Unresolved leading PC: residual {residual}')
    return vecs[:,0],vals,residual,calls


def generate_data(config, t):
    config = dict(config, d=t['d'])
    cfg = base.experiment_config(config, t['alpha'], 'linear')
    sp = base.make_quenched_spike(cfg, seed=10000*t['seed']+1)
    u = sp.u_star.double().numpy(); gamma = sp.gamma.double().numpy()
    n, d = cfg.n_train, cfg.d
    x = np.empty((n,d), dtype=np.float64)
    z_rng = np.random.default_rng(10000*t['seed']+102)
    a_rng = np.random.default_rng(10000*t['seed']+103)
    sqrt_gamma = np.sqrt(gamma)
    for lo in range(0,n,1024):
        block = x[lo:min(n,lo+1024)]
        z_rng.standard_normal(size=block.shape, out=block)
        block *= sqrt_gamma
        block += np.sqrt(cfg.beta/d)*a_rng.standard_normal((len(block),1))*u[None,:]
    return x, u, gamma


def prepare(a):
    if min(a.dimensions)<4 or min(a.seeds)<0 or any(not np.isfinite(x) or x<=0 or round(x*min(a.dimensions))<2 for x in a.alphas):
        raise ValueError('Need d>=4, alpha>0, n>=2 and nonnegative seeds')
    for values in (a.dimensions, a.alphas, a.seeds):
        if len(set(values)) != len(values): raise ValueError('Grid entries must be unique')
    tasks = [dict(d=d, alpha=alpha, seed=seed) for d in a.dimensions for seed in a.seeds for alpha in a.alphas]
    prepare_run(a.root, dict(preset='bounded_pca_dimension_scaling', dimensions=a.dimensions,
                alphas=a.alphas, seeds=a.seeds, tasks=tasks, beta=1., gamma_dist='bounded_heterogenous',
                gamma_floor=.75, gamma_eta=.25, gamma_sigma=1., normalize_gamma=False,
                gamma_formula='0.25 + 2.25 * Beta(1,2)', pca_center=False, pca_standardize=False,
                teacher_dtype='float32', observation_dtype='float64',
                teacher_gamma_seed='10000*seed+1', noise_seed='10000*seed+102',
                amplitude_seed='10000*seed+103', data_block_size=1024,
                eig_tol=1e-9, eig_maxiter=5000, eigenpair_residual_tolerance=1e-7))


def task(a):
    config = load_config(a.root); index = task_index(a.task_id, config)
    t = config['tasks'][index]; folder = task_folder(a.root, index)
    if folder is None: return
    torch.set_num_threads(1)
    start = time.monotonic()
    write_json(folder/'progress.json', dict(**t, stage='generating', data_bytes=8*round(t['alpha']*t['d'])*t['d']))
    x, u, gamma = generate_data(config, t)
    generated = time.monotonic()
    write_json(folder/'progress.json', dict(**t, stage='eigensolver'))
    v, vals, residual, calls = solve(x, t['seed'])
    n, d = x.shape
    cosine_sq = float((v@u)**2/(v@v)/(u@u))
    row = dict(**t, method='pca', n_train=n, alpha_effective=n/d, beta=config['beta'],
               gamma_dist=config['gamma_dist'], cosine_sq=cosine_sq, cosine_abs=np.sqrt(cosine_sq),
               eigenvalue=float(vals[0]), eigenvalue_second=float(vals[1]), spectral_gap=float(vals[0]-vals[1]),
               eigenpair_relative_residual=residual, matvec_count=calls,
               inverse_participation_ratio=float(np.sum(v**4)), gamma_max=float(gamma.max()),
               data_bytes=x.nbytes, generation_seconds=generated-start,
               pca_seconds=time.monotonic()-generated, elapsed_seconds=time.monotonic()-start,
               converged=True, pca_center=False, pca_standardize=False)
    write_rows(folder/'final.csv', [row]); write_json(folder/'COMPLETED.json', row)
    print(row, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--dimensions', type=int, nargs='+', default=DIMENSIONS)
    prep.add_argument('--alphas', type=float, nargs='+', default=ALPHAS)
    prep.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    run = sub.add_parser('task'); run.add_argument('--root', type=Path, required=True)
    run.add_argument('--task-id', type=int)
    agg = sub.add_parser('aggregate'); agg.add_argument('--root', type=Path, required=True)
    agg.add_argument('--allow-missing', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a)
    elif a.command == 'task': task(a)
    else: aggregate_run(a.root, methods=('pca',), group_columns=['d','alpha'],
                        raw_name='pca_per_seed.csv', summary_name='pca_summary.csv', allow_missing=a.allow_missing)


if __name__ == '__main__': main()
