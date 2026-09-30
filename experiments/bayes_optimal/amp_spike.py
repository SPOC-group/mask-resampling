"""Data-only, Fisher-averaged heteroskedastic AMP. Requires PyTorch >= 2.

Usage with your existing generator:
    result = fit_amp(train.x, cfg, seed=10)
    w = result.w

Only cfg.beta, cfg.u_prior and cfg.lambda_prior are used. Coordinate noise
variances are estimated from train.x; no teacher vector or true gamma is read.
The complete samples are used, independently of the MAE augmentation masks.
Gaussian and Rademacher priors are supported. Estimated-variance AMP is a
plug-in implementation, not a proven finite-sample Bayes-optimal estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass
class AMPResult:
    w: Tensor                  # Spike estimate, with its AMP scale and arbitrary sign.
    v: Tensor                  # Estimates of the pretraining sample amplitudes.
    gamma_hat: Tensor          # Estimated coordinate noise variances.
    beta: float
    lambda_prior: str
    converged: bool
    n_iter: int
    residual: float
    variance_floor: float
    n_floored: int

    @property
    def u_hat(self) -> Tensor:
        return self.w

    @property
    def direction(self) -> Tensor:
        norm = torch.linalg.vector_norm(self.w)
        return self.w / norm if norm.item() > 0 else torch.zeros_like(self.w)


def _denoise(a: Tensor, b: Tensor, prior: str) -> tuple[Tensor, Tensor]:
    if prior == "gaussian":
        variance = (1.0 + a).reciprocal()
        return b * variance, variance.expand_as(b)
    if prior == "rademacher":
        mean = torch.tanh(b)
        return mean, (1.0 - mean.square()).clamp_min(0.0)
    raise ValueError(f"Unsupported prior {prior!r}; use gaussian or rademacher.")


def _relative_change(new: Tensor, old: Tensor) -> Tensor:
    scale = torch.maximum(new.square().mean().sqrt(), old.square().mean().sqrt())
    return (new - old).square().mean().sqrt() / scale.clamp_min(1e-12)


def _sign_cycle_metrics(u_new, u, v_new, v):
    """Detect stable magnitudes/directions with at least one alternating sign."""
    direct_u = _relative_change(u_new, u).item()
    direct_v = _relative_change(v_new, v).item()
    flipped_u = _relative_change(u_new, -u).item()
    flipped_v = _relative_change(v_new, -v).item()
    detected = (max(direct_u, direct_v) > 1.5 and
                max(min(direct_u, flipped_u), min(direct_v, flipped_v)) < 1e-3)
    return detected, direct_u, direct_v, flipped_u, flipped_v


@torch.no_grad()
def fit_amp(
    x: Tensor,
    cfg: Any,
    *,
    seed: int = 0,
    max_iter: int = 2000,
    min_iter: int = 30,
    tol: float = 1e-6,
    damping: float = 0.5,
    init_std: float = 0.01,
    patience: int = 5,
    variance_floor: float = 1e-12,
    relative_variance_floor: float = 1e-8,
    progress_callback=None,
    zero_u_init: bool = False,
    adaptive_cycle_damping: bool = False,
    initial_u: Tensor | None = None,
) -> AMPResult:
    """Fit from x of shape (n_samples, d), without accessing teacher quantities.

    damping is the fraction of NEW fields retained (1 = undamped AMP).
    Variances use mean(x**2), including its finite-dimension signal bias.
    Both feature and sample-amplitude estimates start from independent Gaussian
    draws with standard deviation init_std (default variance 0.01**2).
    The random seed is independent of the data and teacher. Convergence means
    numerical stationarity, not certified spike recovery. No restarts are
    selected using true overlap. Inputs are not modified.
    """
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("x must have nonempty shape (n_samples, d).")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 samples.")
    if not torch.isfinite(x).all().item():
        raise ValueError("x must contain complete, finite samples.")
    for name, value in (("max_iter", max_iter), ("min_iter", min_iter),
                        ("patience", patience)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if min_iter > max_iter:
        raise ValueError("min_iter must not exceed max_iter.")
    for name, value in (("tol", tol), ("init_std", init_std),
                        ("variance_floor", variance_floor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must lie in (0, 1].")
    if not math.isfinite(relative_variance_floor) or relative_variance_floor < 0:
        raise ValueError("relative_variance_floor must be finite and nonnegative.")

    beta = float(cfg.beta)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("cfg.beta must be finite and nonnegative.")
    u_prior, v_prior = cfg.u_prior, cfg.lambda_prior
    if u_prior not in ("gaussian", "rademacher") or v_prior not in ("gaussian", "rademacher"):
        raise ValueError("AMP supports gaussian and rademacher priors; no silent prior substitution.")

    x = x.detach()
    n, d = x.shape
    # var + mean**2 equals the uncentered second moment, without allocating x**2.
    sample_var, sample_mean = torch.var_mean(x, dim=0, correction=0)
    raw_gamma = sample_var + sample_mean.square()
    if not torch.isfinite(raw_gamma).all().item():
        raise FloatingPointError("Variance estimation overflowed; use float64 or rescale the data.")
    floor = max(variance_floor, relative_variance_floor * raw_gamma.mean().item())
    gamma = raw_gamma.clamp_min(floor)
    inv_gamma = gamma.reciprocal()
    if not torch.isfinite(inv_gamma).all().item():
        raise FloatingPointError("Inverse variances overflowed; increase variance_floor.")
    n_floored = int((raw_gamma < floor).sum().item())
    alpha, coupling = n / d, math.sqrt(beta / d)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    v = init_std * torch.randn(n, generator=generator, dtype=x.dtype, device="cpu").to(x.device)
    cv = torch.ones_like(v)
    u = init_std * torch.randn(d, generator=generator, dtype=x.dtype, device="cpu").to(x.device)
    if zero_u_init:  # Reproduce archived runs only; Gaussian remains the default.
        u.zero_()
    if initial_u is not None:
        if zero_u_init:
            raise ValueError('initial_u conflicts with zero_u_init')
        if initial_u.shape != (d,) or not torch.isfinite(initial_u).all().item():
            raise ValueError('initial_u must be a finite vector with shape (d,)')
        u = initial_u.detach().to(dtype=x.dtype, device=x.device).clone()
    au, bu = torch.zeros_like(u), torch.zeros_like(u)
    av = x.new_zeros(())
    bv = torch.zeros_like(v)
    converged, stable = False, 0
    residual = float("inf")
    cycle_streak = 0

    for iteration in range(1, max_iter + 1):
        # Coordinate update: the Onsager term uses the previous u and current cv.
        au_raw = alpha * beta * v.square().mean() * inv_gamma
        bu_raw = coupling * (x.T @ v) * inv_gamma
        bu_raw = bu_raw - alpha * beta * cv.mean() * inv_gamma * u
        au_new = (1 - damping) * au + damping * au_raw
        bu_new = (1 - damping) * bu + damping * bu_raw
        u_new, cu_new = _denoise(au_new, bu_new, u_prior)

        # Sample update: use the NEW coordinate means and variances.
        av_raw = beta * (u_new.square() * inv_gamma).mean()
        bv_raw = coupling * (x @ (u_new * inv_gamma))
        bv_raw = bv_raw - beta * (cu_new * inv_gamma).mean() * v
        av_new = (1 - damping) * av + damping * av_raw
        bv_new = (1 - damping) * bv + damping * bv_raw
        v_new, cv_new = _denoise(av_new, bv_new, v_prior)

        values = (au_raw, bu_raw, av_raw, bv_raw, u_new, v_new, cu_new, cv_new)
        if not all(torch.isfinite(value).all().item() for value in values):
            raise FloatingPointError(
                f"AMP became nonfinite at iteration {iteration}; try smaller damping or float64."
            )
        # Check undamped field gaps too, so small damping cannot fake convergence.
        residual = torch.stack([
            _relative_change(u_new, u), _relative_change(v_new, v),
            _relative_change(au_raw, au), _relative_change(bu_raw, bu),
            _relative_change(av_raw, av), _relative_change(bv_raw, bv),
        ]).max().item()
        # A sign-alternating orbit is NOT a fixed point. Reduce the fraction of
        # new fields, reject this step, and retry from the last accepted state.
        # The original undamped-field convergence test remains unchanged.
        if adaptive_cycle_damping and residual > 1.5:
            metrics = _sign_cycle_metrics(u_new, u, v_new, v)
            cycle_streak = cycle_streak + 1 if metrics[0] else 0
            if cycle_streak >= 5 and damping > 0.1:
                previous_damping = damping
                damping = max(0.1, damping * 0.5)
                print(f'AMP sign-cycle safeguard: iteration={iteration}, '
                      f'direct_u={metrics[1]:.6g}, direct_v={metrics[2]:.6g}, '
                      f'flipped_u={metrics[3]:.6g}, flipped_v={metrics[4]:.6g}, '
                      f'damping={previous_damping}->{damping}', flush=True)
                cycle_streak, stable = 0, 0
                continue
        else:
            cycle_streak = 0
        u, v, cv = u_new, v_new, cv_new
        au, bu, av, bv = au_new, bu_new, av_new, bv_new
        stable = stable + 1 if iteration >= min_iter and residual <= tol else 0
        if progress_callback is not None:
            progress_callback(iteration, residual, float(u.square().mean()), stable)
        if stable >= patience:
            converged = True
            break

    return AMPResult(
        w=u.clone(), v=v.clone(), gamma_hat=gamma.clone(), beta=beta,
        lambda_prior=v_prior, converged=converged, n_iter=iteration,
        residual=residual, variance_floor=floor, n_floored=n_floored,
    )


@torch.no_grad()
def squared_cosine(w: Tensor, u_star: Tensor) -> float:
    """Evaluation only: call AFTER fitting; never used by fit_amp."""
    if w.ndim != 1 or u_star.shape != w.shape:
        raise ValueError("w and u_star must be vectors of the same shape.")
    target = u_star.to(device=w.device, dtype=w.dtype)
    nw, nt = torch.linalg.vector_norm(w), torch.linalg.vector_norm(target)
    if nw.item() == 0 or nt.item() == 0:
        return 0.0
    return torch.dot(w / nw, target / nt).square().clamp(0, 1).item()
