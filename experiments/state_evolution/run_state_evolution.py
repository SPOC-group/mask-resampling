#!/usr/bin/env python3
"""Bounded-noise masked-autoencoder state evolution: independent alpha tasks.

The paper preset keeps the original activation-specific solvers, initializations,
quadratures and stopping tolerances. See README.md for run commands.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

PAPER_ALPHAS = [0.09,
 0.2,
 0.35,
 0.45,
 0.6,
 0.8,
 0.9,
 0.9348314606741573,
 0.9696629213483147,
 1.0044943820224719,
 1.0393258426966292,
 1.0741573033707865,
 1.1089887640449438,
 1.1438202247191012,
 1.1786516853932585,
 1.2134831460674158,
 1.2483146067415731,
 1.2831460674157302,
 1.3179775280898878,
 1.3528089887640449,
 1.3876404494382022,
 1.4224719101123595,
 1.4573033707865168,
 1.4921348314606742,
 1.5269662921348315,
 1.5617977528089888,
 1.596629213483146,
 1.6314606741573034,
 1.6662921348314605,
 1.701123595505618,
 1.7359550561797752,
 1.7707865168539325,
 1.8056179775280898,
 1.8404494382022472,
 1.8752808988764045,
 1.9101123595505616,
 1.9449438202247191,
 1.9797752808988762,
 2.0146067415730338,
 2.049438202247191,
 2.0842696629213484,
 2.1191011235955055,
 2.153932584269663,
 2.18876404494382,
 2.2235955056179773,
 2.258426966292135,
 2.293258426966292,
 2.3280898876404494,
 2.3629213483146065,
 2.397752808988764,
 2.432584269662921,
 2.4674157303370787,
 2.502247191011236,
 2.5370786516853934,
 2.5719101123595505,
 2.6067415730337076,
 2.641573033707865,
 2.676404494382022,
 2.7112359550561798,
 2.746067415730337,
 2.7808988764044944,
 2.8157303370786515,
 2.850561797752809,
 2.885393258426966,
 2.9202247191011232,
 2.955056179775281,
 2.989887640449438,
 3.0247191011235954,
 3.0595505617977525,
 3.09438202247191,
 3.129213483146067,
 3.1640449438202243,
 3.198876404494382,
 3.233707865168539,
 3.2685393258426965,
 3.3033707865168536,
 3.338202247191011,
 3.373033707865168,
 3.4078651685393258,
 3.442696629213483,
 3.4775280898876404,
 3.5123595505617975,
 3.5471910112359546,
 3.582022471910112,
 3.6168539325842692,
 3.651685393258427,
 3.686516853932584,
 3.7213483146067414,
 3.7561797752808985,
 3.791011235955056,
 3.825842696629213,
 3.8606741573033707,
 3.895505617977528,
 3.930337078651685,
 3.9651685393258425,
 4.0,
 4.2,
 4.3795494865122535,
 4.566774691621367,
 4.7620037513589315,
 4.96557882953177,
 5.177856717407615,
 5.399209459037271,
 5.6300250033092345,
 5.870707883879548,
 6.121679928168599,
 6.383380996667419,
 6.656269753849255,
 6.9408244720375105,
 7.2375438696389445,
 7.546947985211224,
 7.869579088896787,
 8.206002632820356,
 8.556808242115855,
 8.922610748319599,
 9.304051266940906,
 9.701798321098744,
 10.116549013193676,
 10.54903024666867,
 11.0]

PAPER_SETTINGS = {'linear': {'1': {'beta': 1.0,
                  'rho': 0.75,
                  'lambda_r': 0.0001,
                  'init_m_scale': 0.05,
                  'max_iter': 160000,
                  'tol': 1e-06,
                  'damping': 0.3,
                  'quad_order': 21,
                  'mask_quad_order': 21,
                  'prox_grid_size': 201,
                  'prox_newton_steps': 25,
                  'sobol_power': 11,
                  'qmc_seed': 1234,
                  'finite_k_prox_steps': 35,
                  'finite_k_prox_tol': 1e-09,
                  'k1_backend': 'generic',
                  'antithetic_qmc': False,
                  'finite_k_backend': 'numpy',
                  'endpoint_backend': 'numpy',
                  'torch_device': 'cpu',
                  'n_gamma': 200,
                  'solver': 'linear_core'},
            '2': {'beta': 1.0,
                  'rho': 0.75,
                  'lambda_r': 0.0001,
                  'init_m_scale': 0.05,
                  'max_iter': 160000,
                  'tol': 1e-06,
                  'damping': 0.3,
                  'quad_order': 21,
                  'mask_quad_order': 21,
                  'prox_grid_size': 201,
                  'prox_newton_steps': 25,
                  'sobol_power': 11,
                  'qmc_seed': 1234,
                  'finite_k_prox_steps': 35,
                  'finite_k_prox_tol': 1e-09,
                  'k1_backend': 'generic',
                  'antithetic_qmc': False,
                  'finite_k_backend': 'numpy',
                  'endpoint_backend': 'numpy',
                  'torch_device': 'cpu',
                  'n_gamma': 200,
                  'solver': 'linear_core'},
            '4': {'beta': 1.0,
                  'rho': 0.75,
                  'lambda_r': 0.0001,
                  'init_m_scale': 0.05,
                  'max_iter': 160000,
                  'tol': 1e-06,
                  'damping': 0.3,
                  'quad_order': 21,
                  'mask_quad_order': 21,
                  'prox_grid_size': 201,
                  'prox_newton_steps': 25,
                  'sobol_power': 11,
                  'qmc_seed': 1234,
                  'finite_k_prox_steps': 35,
                  'finite_k_prox_tol': 1e-09,
                  'k1_backend': 'generic',
                  'antithetic_qmc': False,
                  'finite_k_backend': 'numpy',
                  'endpoint_backend': 'numpy',
                  'torch_device': 'cpu',
                  'n_gamma': 200,
                  'solver': 'linear_core'},
            'dynamic': {'beta': 1.0,
                        'rho': 0.75,
                        'lambda_r': 0.0001,
                        'init_m_scale': 0.05,
                        'max_iter': 160000,
                        'tol': 1e-06,
                        'damping': 0.3,
                        'quad_order': 21,
                        'mask_quad_order': 21,
                        'prox_grid_size': 201,
                        'prox_newton_steps': 25,
                        'sobol_power': 11,
                        'qmc_seed': 1234,
                        'finite_k_prox_steps': 35,
                        'finite_k_prox_tol': 1e-09,
                        'k1_backend': 'generic',
                        'antithetic_qmc': False,
                        'finite_k_backend': 'numpy',
                        'endpoint_backend': 'numpy',
                        'torch_device': 'cpu',
                        'n_gamma': 200,
                        'solver': 'linear_core'}},
 'relu': {'1': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-06,
                'damping': 0.3,
                'quad_order': 120,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 14,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'specialized',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'numpy',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'relu_core'},
          '2': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-06,
                'damping': 0.3,
                'quad_order': 21,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 18,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'specialized',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'numpy',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'relu_core'},
          '4': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-06,
                'damping': 0.3,
                'quad_order': 21,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 18,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'specialized',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'numpy',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'relu_core'},
          'dynamic': {'beta': 1.0,
                      'rho': 0.75,
                      'lambda_r': 0.0001,
                      'init_m_scale': 0.05,
                      'max_iter': 160000,
                      'tol': 1e-06,
                      'damping': 0.3,
                      'quad_order': 120,
                      'mask_quad_order': 21,
                      'prox_grid_size': 201,
                      'prox_newton_steps': 25,
                      'sobol_power': 14,
                      'qmc_seed': 1234,
                      'finite_k_prox_steps': 35,
                      'finite_k_prox_tol': 1e-09,
                      'k1_backend': 'specialized',
                      'antithetic_qmc': True,
                      'finite_k_backend': 'torch',
                      'endpoint_backend': 'numpy',
                      'torch_device': 'cuda',
                      'n_gamma': 200,
                      'solver': 'relu_core'}},
 'elu': {'1': {'beta': 1.0,
               'rho': 0.75,
               'lambda_r': 0.0001,
               'init_m_scale': 0.05,
               'max_iter': 160000,
               'tol': 1e-06,
               'damping': 0.3,
               'quad_order': 21,
               'mask_quad_order': 21,
               'prox_grid_size': 201,
               'prox_newton_steps': 25,
               'sobol_power': 14,
               'qmc_seed': 1234,
               'finite_k_prox_steps': 35,
               'finite_k_prox_tol': 1e-09,
               'k1_backend': 'generic',
               'antithetic_qmc': True,
               'finite_k_backend': 'torch',
               'endpoint_backend': 'torch',
               'torch_device': 'cuda',
               'n_gamma': 200,
               'solver': 'elu_core'},
         '2': {'beta': 1.0,
               'rho': 0.75,
               'lambda_r': 0.0001,
               'init_m_scale': 0.05,
               'max_iter': 160000,
               'tol': 1e-06,
               'damping': 0.3,
               'quad_order': 21,
               'mask_quad_order': 21,
               'prox_grid_size': 201,
               'prox_newton_steps': 25,
               'sobol_power': 14,
               'qmc_seed': 1234,
               'finite_k_prox_steps': 35,
               'finite_k_prox_tol': 1e-09,
               'k1_backend': 'generic',
               'antithetic_qmc': True,
               'finite_k_backend': 'torch',
               'endpoint_backend': 'torch',
               'torch_device': 'cuda',
               'n_gamma': 200,
               'solver': 'elu_core'},
         '4': {'beta': 1.0,
               'rho': 0.75,
               'lambda_r': 0.0001,
               'init_m_scale': 0.05,
               'max_iter': 160000,
               'tol': 1e-06,
               'damping': 0.3,
               'quad_order': 21,
               'mask_quad_order': 21,
               'prox_grid_size': 201,
               'prox_newton_steps': 25,
               'sobol_power': 14,
               'qmc_seed': 1234,
               'finite_k_prox_steps': 35,
               'finite_k_prox_tol': 1e-09,
               'k1_backend': 'generic',
               'antithetic_qmc': True,
               'finite_k_backend': 'torch',
               'endpoint_backend': 'torch',
               'torch_device': 'cuda',
               'n_gamma': 200,
               'solver': 'elu_core'},
         'dynamic': {'beta': 1.0,
                     'rho': 0.75,
                     'lambda_r': 0.0001,
                     'init_m_scale': 0.05,
                     'max_iter': 160000,
                     'tol': 1e-06,
                     'damping': 0.3,
                     'quad_order': 64,
                     'mask_quad_order': 64,
                     'prox_grid_size': 201,
                     'prox_newton_steps': 25,
                     'sobol_power': 14,
                     'qmc_seed': 1234,
                     'finite_k_prox_steps': 35,
                     'finite_k_prox_tol': 1e-09,
                     'k1_backend': 'generic',
                     'antithetic_qmc': True,
                     'finite_k_backend': 'torch',
                     'endpoint_backend': 'torch',
                     'torch_device': 'cuda',
                     'n_gamma': 200,
                     'solver': 'elu_core'}},
 'tanh': {'1': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-07,
                'damping': 0.3,
                'quad_order': 21,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 18,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'generic',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'torch',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'tanh_core'},
          '2': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-06,
                'damping': 0.3,
                'quad_order': 21,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 14,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'specialized',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'torch',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'tanh_core'},
          '4': {'beta': 1.0,
                'rho': 0.75,
                'lambda_r': 0.0001,
                'init_m_scale': 0.05,
                'max_iter': 160000,
                'tol': 1e-06,
                'damping': 0.3,
                'quad_order': 21,
                'mask_quad_order': 21,
                'prox_grid_size': 201,
                'prox_newton_steps': 25,
                'sobol_power': 14,
                'qmc_seed': 1234,
                'finite_k_prox_steps': 35,
                'finite_k_prox_tol': 1e-09,
                'k1_backend': 'specialized',
                'antithetic_qmc': True,
                'finite_k_backend': 'torch',
                'endpoint_backend': 'torch',
                'torch_device': 'cuda',
                'n_gamma': 200,
                'solver': 'tanh_core'},
          'dynamic': {'beta': 1,
                      'rho': 0.75,
                      'lambda_r': 0.0001,
                      'damping': 0.3,
                      'tol': 1e-08,
                      'max_iter': 160000,
                      'init_m': 0.01,
                      'init_v': 1,
                      'quad_order': 64,
                      'mask_quad_order': 64,
                      'prox_grid_size': 241,
                      'prox_newton_steps': 35,
                      'gamma_law': 'gamma=0.25+2.25*Beta(1,2)',
                      'n_gamma': 200,
                      'solver': 'dynamic_tanh_jax'}}}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prepare(args):
    alphas = PAPER_ALPHAS if args.alphas is None else sorted(set(args.alphas))
    if not alphas or any(not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("Alphas must be finite and positive.")
    counts = list(dict.fromkeys(str(k).lower() for k in args.k_values))
    for k in counts:
        if k != "dynamic" and (not k.isdigit() or int(k) < 1):
            raise ValueError("K must be a positive integer or 'dynamic'.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory to avoid mixing experiments.")
    overrides = {name:getattr(args, name) for name in (
        "max_iter", "tol", "damping", "n_gamma", "sobol_power", "quad_order",
        "mask_quad_order", "prox_grid_size", "prox_newton_steps", "finite_k_prox_steps"
    ) if getattr(args, name) is not None}
    for name, value in overrides.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if "damping" in overrides and overrides["damping"] > 1:
        raise ValueError("Damping must be at most one.")
    if "sobol_power" in overrides and not 4 <= overrides["sobol_power"] <= 24:
        raise ValueError("Sobol power must be between 4 and 24.")
    for name in ("quad_order", "mask_quad_order", "n_gamma"):
        if name in overrides and overrides[name] < 2:
            raise ValueError(f"{name} must be at least two.")
    profiles = {}
    tasks = []
    activations = list(dict.fromkeys(args.activations))
    for activation in activations:
        for k in counts:
            # Additional finite K values reuse this activation's finite-K settings.
            base = k if k in PAPER_SETTINGS[activation] else "4"
            settings = dict(PAPER_SETTINGS[activation][base])
            settings.update(overrides)
            key = f"{activation}/{k}"
            profiles[key] = settings
            for i, alpha in enumerate(alphas):
                tasks.append(dict(task_id=len(tasks), activation=activation,
                                  k=k, alpha=float(alpha), alpha_index=i, profile=key))
    config = dict(
        preset="bounded-beta", gamma_law="gamma=0.25+2.25*Beta(1,2)",
        gamma_support=[0.25,2.5], beta=1.0, rho=0.75, lambda_r=0.0001,
        alpha_values=alphas, activations=activations, k_values=counts,
        profiles=profiles, tasks=tasks, task_count=len(tasks),
        numerical_overrides=overrides, continuation=False,
        downstream=dict(n_ds=100,mc_samples=100000,mc_seed=12345),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"points").mkdir()
    write_json(args.output/"config.json", config)
    print(f"Prepared {len(tasks)} independent tasks (IDs 0–{len(tasks)-1}) in {args.output}")


def needs_gpu(task, settings):
    return not (task["activation"] == "linear" or
                (task["activation"] == "relu" and task["k"] in ("1", "dynamic")))


def solve_point(task, settings, device):
    settings = dict(settings)
    module = importlib.import_module("solvers." + settings.pop("solver"))
    if task["activation"] == "tanh" and task["k"] == "dynamic":
        import jax
        settings["noise_nodes"] = settings.pop("n_gamma")
        settings["alpha"] = task["alpha"]
        with jax.default_device(jax.devices(device)[0]):
            return module.solve(settings, task_id=task["task_id"])
    from solvers.gamma_distributions import bounded_heterogenous_quadrature
    gamma, weights = bounded_heterogenous_quadrature(settings.pop("n_gamma"))
    settings["torch_device"] = "cuda" if device == "gpu" else "cpu"
    return module.run_smooth_mae_se(
        alpha=task["alpha"], activation=task["activation"],
        mask_count=np.inf if task["k"] == "dynamic" else int(task["k"]),
        gamma=gamma, weights=weights, init_state=None, verbose=False, **settings,
    )


def downstream_accuracy(row, settings):
    from scipy.special import ndtr
    m, c = float(row.get("M", math.nan)), float(row.get("C", math.nan))
    if not math.isfinite(m) or not math.isfinite(c) or c <= 0:
        return math.nan
    snr = row["beta"] * m*m / c
    n = settings["n_ds"]
    score = np.abs(np.random.default_rng(settings["mc_seed"]).normal(
        size=(settings["mc_samples"], n))).sum(axis=1) / np.sqrt(n)
    p = float(ndtr(np.sqrt(snr)*score).mean())
    return .5 + (2*p-1)*np.arctan(np.sqrt(snr))/np.pi


def normalized_row(result, task, settings, config):
    row = {key:value.item() if isinstance(value,np.generic) else value
           for key,value in result.items()
           if np.isscalar(value) and not isinstance(value,(bytes,bytearray))}
    dynamic = task["k"] == "dynamic"
    row.update(task_id=task["task_id"],activation=task["activation"],
        alpha=task["alpha"], beta=1.0, rho=.75, lambda_r=1e-4,
        k_views=0 if dynamic else int(task["k"]),
        masking_protocol="dynamic" if dynamic else "finite_k_fixed",
        k_label="dynamic" if dynamic else "K="+task["k"],
        mask_count="inf" if dynamic else int(task["k"]),
        mask_label="K=inf (dynamic)" if dynamic else "K="+task["k"],
        sweep_parameter="alpha",sweep_value=task["alpha"],
        gamma_law=config["gamma_law"],n_gamma=settings["n_gamma"],
        noise_discretization="full_support_gauss_legendre_beta12",warm_start=False)
    for key,value in settings.items():
        if key not in row:
            row[key]=value
    m,q=float(row.get("M",math.nan)),float(row.get("Q",math.nan))
    row["cosine"] = abs(m)/math.sqrt(q) if math.isfinite(m) and q>0 else math.nan
    row["cosine_sq"] = row["cosine"]**2
    row["downstream_test_accuracy"] = downstream_accuracy(row,config["downstream"])
    row.update(downstream_n_ds=config["downstream"]["n_ds"],
               downstream_mc_samples=config["downstream"]["mc_samples"],
               downstream_mc_seed=config["downstream"]["mc_seed"])
    return row


def task(args):
    config = json.loads(args.config.read_text())
    if not 0 <= args.task_id < config["task_count"]:
        raise IndexError("Task ID outside the prepared array.")
    point = config["tasks"][args.task_id]
    settings = config["profiles"][point["profile"]]
    directory = args.config.resolve().parent/"points"/f"{args.task_id:04d}"
    directory.mkdir(parents=True,exist_ok=True)
    done = directory/"completion.json"
    if done.exists() and not args.force:
        previous = json.loads(done.read_text())
        print(f"Task {args.task_id} already finished: {previous['status']}; use --force to rerun.")
        return 0 if previous["converged"] else 2
    lock = directory/"running.lock"
    with lock.open("x") as handle:
        handle.write("Task running; remove this lock only after its process has stopped.\n")
    device = args.device
    if device == "auto":
        device = "gpu" if needs_gpu(point,settings) else "cpu"
    try:
        import torch
        torch.set_num_threads(1)
        if device == "gpu" and settings["solver"] != "dynamic_tanh_jax" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for a CPU check.")
        started = time.perf_counter()
        result = solve_point(point,settings,device)
        row = normalized_row(result,point,settings,config)
        row["elapsed_seconds"] = time.perf_counter()-started
        row["compute_device"] = device
        write_csv(directory/"result.csv",[row])
        residual = float(row.get("se_raw_residual",math.inf))
        converged = (row.get("status")=="converged" and math.isfinite(residual)
                     and residual < settings["tol"]
                     and row.get("full_support_admissible",True) is not False)
        write_json(done,dict(task_id=args.task_id,status=row.get("status","unknown"),converged=converged))
        print(f"{point['activation']} K={point['k']} alpha={point['alpha']:g}: "
              f"{row.get('status')}, residual={residual:.3g}, cosine={row['cosine']:.6g}")
        return 0 if converged else 2
    except Exception as error:
        write_json(directory/"error.json",dict(task_id=args.task_id,error=str(error)))
        raise
    finally:
        lock.unlink()


def aggregate(args):
    config = json.loads(args.config.read_text())
    rows, missing = [], []
    for point in config["tasks"]:
        path = args.config.resolve().parent/"points"/f"{point['task_id']:04d}"/"result.csv"
        if path.exists():
            with path.open(newline="") as handle:
                saved = list(csv.DictReader(handle))
            if len(saved)!=1 or int(saved[0]["task_id"]) != point["task_id"]:
                raise ValueError(f"Unexpected result at task {point['task_id']}")
            rows.append(saved[0])
        else:
            missing.append(point["task_id"])
            # Preserve absent solutions as gaps, never as zero overlap.
            rows.append(dict(task_id=point["task_id"],activation=point["activation"],alpha=point["alpha"],
                k_views=0 if point["k"]=="dynamic" else int(point["k"]),
                masking_protocol="dynamic" if point["k"]=="dynamic" else "finite_k_fixed",
                status="no_saved_result",cosine="",downstream_test_accuracy=""))
    if missing and not args.allow_missing:
        raise RuntimeError(f"{len(missing)} results missing; use --allow-missing to preserve gaps.")
    output = args.config.resolve().parent/"state_evolution.csv"
    write_csv(output,rows)
    print(f"Wrote {len(rows)} rows to {output.name}; {len(missing)} missing. Solver statuses are retained.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    prep=commands.add_parser("prepare",help="Create a preset configuration without running solvers.")
    prep.add_argument("--preset",choices=["bounded-beta"],default="bounded-beta")
    prep.add_argument("--output",type=Path,required=True)
    prep.add_argument("--activations",choices=list(PAPER_SETTINGS),nargs="+",default=list(PAPER_SETTINGS))
    prep.add_argument("--k-values",nargs="+",default=["1","2","4","dynamic"])
    prep.add_argument("--alphas",type=float,nargs="+")
    for name in ("max-iter","n-gamma","sobol-power","quad-order","mask-quad-order",
                 "prox-grid-size","prox-newton-steps","finite-k-prox-steps"):
        prep.add_argument("--"+name,type=int)
    for name in ("tol","damping"):
        prep.add_argument("--"+name,type=float)
    run=commands.add_parser("task",help="Run one independent alpha/activation/K point.")
    run.add_argument("--config",type=Path,required=True)
    run.add_argument("--task-id",type=int,required=True)
    run.add_argument("--device",choices=["auto","cpu","gpu"],default="auto")
    run.add_argument("--force",action="store_true")
    collect=commands.add_parser("aggregate",help="Combine scalar CSVs without discarding unconverged rows.")
    collect.add_argument("--config",type=Path,required=True)
    collect.add_argument("--allow-missing",action="store_true")
    args=parser.parse_args()
    if args.command=="prepare":
        prepare(args)
    elif args.command=="task":
        raise SystemExit(task(args))
    else:
        aggregate(args)


if __name__=="__main__":
    main()
