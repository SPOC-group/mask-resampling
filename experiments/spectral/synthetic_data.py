"""Original float32 teacher and observation generators for the spectral presets."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
from typing import Any
import numpy as np
import torch
from torch import Tensor
from gamma_distributions import BOUNDED_HETEROGENOUS, canonical_gamma_dist, sample_bounded_heterogenous

@dataclass(frozen=True)
class ExperimentConfig:
    d: int
    alpha: float
    beta: float
    rho: float
    n_test: int | None
    epochs: int
    number_restarts: int
    eval_every: int
    lr: float
    init_std: float
    activation: str
    lambda_prior: str
    u_prior: str
    gamma_dist: str
    gamma_delta: float
    gamma_sigma: float
    gamma_shape: float
    normalize_gamma: bool
    dtype: torch.dtype
    device: torch.device
    gamma_floor: float = 1.0
    gamma_eta: float = 0.25
    normalize_lognormal_multiplier: bool = False
    gamma_low: float = 0.01
    gamma_high: float = 3.0
    gamma_high_probability: float = 99.0 / 299.0

    @property
    def n_train(self) -> int:
        return int(round(self.alpha * self.d))

    @property
    def resolved_n_test(self) -> int:
        return self.n_train if self.n_test is None else self.n_test


@dataclass
class DataBatch:
    x: Tensor
    hidden_mask: Tensor


@dataclass
class QuenchedSpike:
    u_star: Tensor
    gamma: Tensor


def prior_samples(
    shape: tuple[int, ...],
    prior: str,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Sample zero-mean, unit-variance priors used for lambda and u_star."""
    if prior == "gaussian":
        return torch.randn(shape, generator=generator, dtype=dtype, device=device)
    if prior == "rademacher":
        bits = torch.randint(0, 2, shape, generator=generator, device=device)
        return (2 * bits.to(dtype) - 1).to(device)
    if prior == "laplace":
        uniform = torch.rand(shape, generator=generator, dtype=dtype, device=device)
        centered = uniform - 0.5
        scale = 1.0 / math.sqrt(2.0)
        return -scale * torch.sign(centered) * torch.log1p(-2.0 * centered.abs())
    raise ValueError(f"unknown prior {prior!r}")


def gamma_samples(
    d: int,
    cfg: ExperimentConfig,
    *,
    generator: torch.Generator,
    seed: int,
) -> Tensor:
    """Sample positive quenched noise variances with population mean one."""
    dtype = cfg.dtype
    device = cfg.device

    gamma_dist = canonical_gamma_dist(cfg.gamma_dist)
    if gamma_dist == "constant":
        gamma = torch.ones(d, dtype=dtype, device=device)
    elif gamma_dist == "two_point":
        if not 0.0 <= cfg.gamma_delta < 1.0:
            raise ValueError("--gamma-delta must lie in [0, 1) for two_point")
        bits = torch.randint(0, 2, (d,), generator=generator, device=device)
        gamma = 1.0 + cfg.gamma_delta * (2 * bits.to(dtype) - 1)
    elif gamma_dist == "binary_mixture":
        if cfg.gamma_low < 0 or cfg.gamma_high < 0:
            raise ValueError("binary-mixture variances must be nonnegative")
        if not 0.0 <= cfg.gamma_high_probability <= 1.0:
            raise ValueError("gamma_high_probability must lie in [0,1]")
        high = torch.rand(d, generator=generator, dtype=dtype, device=device)
        high = high < cfg.gamma_high_probability
        gamma = torch.where(
            high,
            torch.as_tensor(cfg.gamma_high, dtype=dtype, device=device),
            torch.as_tensor(cfg.gamma_low, dtype=dtype, device=device),
        )
    elif gamma_dist == "lognormal":
        if cfg.gamma_sigma < 0:
            raise ValueError("gamma_sigma must be nonnegative for lognormal")
        if cfg.gamma_floor < 0 or cfg.gamma_eta < 0:
            raise ValueError("gamma_floor and gamma_eta must be nonnegative for lognormal")
        if cfg.gamma_floor == 0 and cfg.gamma_eta == 0:
            raise ValueError("at least one of gamma_floor or gamma_eta must be positive")
        z = torch.randn(d, generator=generator, dtype=dtype, device=device)
        ell = torch.exp(cfg.gamma_sigma * z - 0.5 * cfg.gamma_sigma**2)
        if cfg.normalize_lognormal_multiplier:
            ell = ell / ell.mean().clamp_min(torch.finfo(dtype).eps)
        gamma = cfg.gamma_floor + cfg.gamma_eta * ell
    elif gamma_dist == "gamma":
        if cfg.gamma_shape <= 0:
            raise ValueError("--gamma-shape must be positive")
        rng = np.random.default_rng(seed)
        gamma_np = rng.gamma(shape=cfg.gamma_shape, scale=1.0 / cfg.gamma_shape, size=d)
        gamma = torch.as_tensor(gamma_np, dtype=dtype, device=device)
    elif gamma_dist == BOUNDED_HETEROGENOUS:
        rng = np.random.default_rng(seed)
        gamma_np = sample_bounded_heterogenous(rng, size=d)
        gamma = torch.as_tensor(gamma_np, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown gamma distribution {gamma_dist!r}")

    if cfg.normalize_gamma:
        gamma = gamma / gamma.mean().clamp_min(torch.finfo(dtype).eps)
    return gamma


def make_quenched_spike(cfg: ExperimentConfig, seed: int) -> QuenchedSpike:
    """Draw the teacher vector and coordinate noise variances."""
    sample_device = torch.device("cpu")
    sample_cfg = replace(cfg, device=sample_device)
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    u_star = prior_samples(
        (cfg.d,),
        cfg.u_prior,
        generator=generator,
        dtype=cfg.dtype,
        device=sample_device,
    )
    gamma = gamma_samples(cfg.d, sample_cfg, generator=generator, seed=seed + 17)
    return QuenchedSpike(u_star=u_star.to(cfg.device), gamma=gamma.to(cfg.device))


def make_spiked_heteroskedastic_data(
    n: int,
    spike: QuenchedSpike,
    cfg: ExperimentConfig,
    seed: int,
) -> DataBatch:
    """Generate samples and iid Bernoulli(rho) hidden masks."""
    sample_device = torch.device("cpu")
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    u_star = spike.u_star.to(sample_device)
    gamma = spike.gamma.to(sample_device)
    lambdas = prior_samples(
        (n, 1),
        cfg.lambda_prior,
        generator=generator,
        dtype=cfg.dtype,
        device=sample_device,
    )
    z = torch.randn(n, cfg.d, generator=generator, dtype=cfg.dtype, device=sample_device)
    signal = math.sqrt(cfg.beta / cfg.d) * lambdas * u_star.view(1, cfg.d)
    noise = gamma.sqrt().view(1, cfg.d) * z
    x = signal + noise
    hidden_mask = torch.bernoulli(
        torch.full((n, cfg.d), cfg.rho, dtype=cfg.dtype, device=sample_device),
        generator=generator,
    )
    return DataBatch(x=x.to(cfg.device), hidden_mask=hidden_mask.to(cfg.device))


def experiment_config(config, alpha, activation='linear'):
    # Optimization-only fields do not enter the original data generator.
    return ExperimentConfig(
        d=int(config['d']), alpha=float(alpha), beta=float(config['beta']),
        rho=0.75, n_test=None, epochs=0, number_restarts=1, eval_every=1,
        lr=0., init_std=0., activation=activation, lambda_prior='gaussian',
        u_prior='gaussian', gamma_dist=config['gamma_dist'], gamma_delta=.5,
        gamma_sigma=float(config.get('gamma_sigma',1.)), gamma_shape=2.,
        normalize_gamma=False, dtype=torch.float32, device=torch.device('cpu'),
        gamma_floor=float(config.get('gamma_floor',.75)),
        gamma_eta=float(config.get('gamma_eta',.25)), normalize_lognormal_multiplier=False)
