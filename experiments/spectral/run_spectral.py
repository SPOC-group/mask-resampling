#!/usr/bin/env python3
"""Prepare, run and aggregate the heterogeneous-noise spectral comparison."""
from __future__ import annotations
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import synthetic_data as base
from heteroskedastic_pca_estimators import first_pc, diagonal_deletion_pca, heteropca, estimated_whitening_pca
from heteropca_continuation import resume_heteropca
from run_io import prepare_run, load_config, task_index, task_folder, write_json, write_rows, aggregate_run

PANELS = {'bounded': {'d': 2000, 'beta': 1.0, 'gamma_dist': 'bounded_heterogenous', 'gamma_floor': 0.75, 'gamma_eta': 0.25, 'gamma_sigma': 1.0, 'pca_center': False, 'alphas': [0.1, 0.1333521432163324, 0.1778279410038923, 0.2371373705661655, 0.3162277660168379, 0.4216965034285823, 0.5623413251903491, 0.7498942093324559, 1.0, 1.333521432163324, 1.7782794100389228, 2.3713737056616555, 3.1622776601683795, 4.216965034285822, 5.62341325190349, 7.498942093324557, 10.0], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}, 'unbounded': {'d': 2000, 'beta': 1.0, 'gamma_dist': 'lognormal', 'gamma_floor': 0.75, 'gamma_eta': 0.25, 'gamma_sigma': 1.0, 'pca_center': True, 'alphas': [0.5, 0.6029542755153482, 0.7271077167244768, 0.8768254131184519, 1.057371263440564, 1.2750930481971074, 1.537645610180688, 1.8542599897717045, 2.23606797749979, 2.696493494752912, 3.251724563121181, 3.9212824562643886, 4.728708045015879, 5.702389466812296, 6.876560219336321, 8.29250277017519, 10.0], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}}
METHODS = ('pca', 'diagonal_deletion_pca', 'heteropca', 'estimated_whitening_pca')

def generate_covariance(panel_config, alpha, seed):
    """Use the original ERM generator: no new RNG, no Gamma clipping/rescaling."""
    started = time.monotonic()
    cfg = base.experiment_config(panel_config, alpha, 'linear')
    spike = base.make_quenched_spike(cfg, seed=10_000*seed+1)
    batch = base.make_spiked_heteroskedastic_data(cfg.n_train, spike, cfg, seed=10_000*seed+2)
    observation_hash = hashlib.sha256(memoryview(batch.x.numpy()).cast('B')).hexdigest()
    x = batch.x.numpy().astype(np.float64)
    del batch  # Masks are not used by any of the PCA estimators.
    if panel_config['pca_center']:
        x -= x.mean(axis=0, keepdims=True)
    data_seconds = time.monotonic() - started
    started = time.monotonic()
    s = x.T @ x / len(x)
    del x
    return s, spike, dict(n_train=cfg.n_train, alpha_effective=cfg.n_train/cfg.d,
                          data_seconds=data_seconds, covariance_seconds=time.monotonic()-started,
                          observation_sha256=observation_hash)


def fit_heteropca(s, seed, stages, tol, progress=None):
    result = heteropca(s, seed=seed, max_iter=stages[0], tol=tol,
                       progress=None if progress is None else lambda row: progress(row, None))
    calls = result.diagnostics['matvec_count']
    residual = result.diagnostics['max_eigenpair_relative_residual']
    used = [stages[0]]
    for cap in stages[1:]:
        if result.diagnostics['converged']: break
        result = resume_heteropca(s, initial_diagonal=result.auxiliary['imputed_diagonal'],
                                 initial_iteration=result.diagnostics['n_iter'], seed=seed,
                                 max_iter=cap, tol=tol, progress=progress)
        calls += result.diagnostics['matvec_count']
        residual = max(residual, result.diagnostics['max_eigenpair_relative_residual'])
        used.append(cap)
    result.diagnostics.update(matvec_count=calls, max_eigenpair_relative_residual=residual,
                              heteropca_stages_used=len(used), heteropca_retried=len(used)>1)
    return result


def prepare(a):
    import copy
    panels = copy.deepcopy(PANELS)
    for panel in panels.values():
        if a.d is not None: panel['d'] = a.d
        if a.alphas is not None: panel['alphas'] = a.alphas
        if a.seeds is not None: panel['seeds'] = a.seeds
        if panel['d'] < 4 or any(not np.isfinite(x) or x <= 0 or round(x*panel['d']) < 2 for x in panel['alphas']):
            raise ValueError('Need d>=4, finite alpha>0, n>=2')
        if len(set(panel['alphas'])) != len(panel['alphas']) or len(set(panel['seeds'])) != len(panel['seeds']) or min(panel['seeds']) < 0:
            raise ValueError('Unique alpha values and nonnegative seeds required')
    if sorted(set(a.heteropca_stages)) != a.heteropca_stages or min(a.heteropca_stages)<1 or not np.isfinite(a.heteropca_tol) or a.heteropca_tol<=0:
        raise ValueError('Positive increasing iteration caps and positive tolerance required')
    tasks = [dict(panel=name, alpha=alpha, seed=seed)
             for name, panel in panels.items() for alpha in panel['alphas'] for seed in panel['seeds']]
    prepare_run(a.root, dict(preset='heterogeneous_spectral', panels=panels, tasks=tasks,
                methods=list(METHODS), heteropca_stages=a.heteropca_stages,
                heteropca_tol=a.heteropca_tol, eig_tol=1e-10, eig_maxiter=5000,
                variance_floor=1e-12, normalize_gamma=False, gamma_truncation=False,
                data_dtype='float32', covariance_dtype='float64', covariance_normalization='1/n',
                teacher_gamma_seed='10000*seed+1', observation_seed='10000*seed+2',
                use_teacher_for_fitting=False, use_true_gamma_for_fitting=False))


def task(a):
    config = load_config(a.root); index = task_index(a.task_id, config)
    t = config['tasks'][index]; folder = task_folder(a.root, index)
    if folder is None: return
    torch.set_num_threads(1)
    p = config['panels'][t['panel']]
    write_json(folder/'progress.json', dict(**t, phase='generating_covariance'))
    s, spike, shared = generate_covariance(p, t['alpha'], t['seed'])
    teacher = spike.u_star.numpy().astype(np.float64)
    rows = []
    for method in METHODS:
        start = time.monotonic()
        write_json(folder/'progress.json', dict(**t, phase=method))
        if method == 'heteropca':
            def progress(row, _diagonal):
                write_json(folder/'progress.json', dict(**t, phase=method, **row))
            result = fit_heteropca(s, t['seed'], config['heteropca_stages'], config['heteropca_tol'], progress)
        else:
            estimator = dict(pca=first_pc, diagonal_deletion_pca=diagonal_deletion_pca,
                             estimated_whitening_pca=estimated_whitening_pca)[method]
            result = estimator(s, seed=t['seed'])
        cosine = float(abs(result.direction @ teacher)/(np.linalg.norm(result.direction)*np.linalg.norm(teacher)))
        row = dict(**t, method=method, d=p['d'], beta=p['beta'], gamma_dist=p['gamma_dist'],
                   pca_center=p['pca_center'], cosine_abs=cosine, cosine_sq=cosine**2,
                   fit_seconds=time.monotonic()-start, **shared, **result.diagnostics)
        rows.append(row)
        write_rows(folder/'final.csv', rows)
        print({k:row[k] for k in ('task_id','method','cosine_abs','converged','n_iter')}, flush=True)
    write_json(folder/'COMPLETED.json', dict(**t, num_converged=sum(r['converged'] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--d', type=int); prep.add_argument('--alphas', type=float, nargs='+')
    prep.add_argument('--seeds', type=int, nargs='+')
    prep.add_argument('--heteropca-stages', type=int, nargs='+', default=[500,50000,100000])
    prep.add_argument('--heteropca-tol', type=float, default=1e-8)
    run = sub.add_parser('task'); run.add_argument('--root', type=Path, required=True)
    run.add_argument('--task-id', type=int)
    agg = sub.add_parser('aggregate'); agg.add_argument('--root', type=Path, required=True)
    agg.add_argument('--allow-missing', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare': prepare(a)
    elif a.command == 'task': task(a)
    else: aggregate_run(a.root, methods=METHODS, group_columns=['panel','method','alpha'],
                        raw_name='pca_estimators_per_seed.csv', summary_name='pca_estimators_summary.csv',
                        allow_missing=a.allow_missing)


if __name__ == '__main__': main()
