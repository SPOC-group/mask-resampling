#!/usr/bin/env python3
"""Batched PyTorch proximal for smooth finite-K MAE output channels."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

try:
    from .state_evolution_gtrain_gtest_core import (
        EPS,
        centered_loss_to_gain,
        gh_standard_normal,
    )
except ImportError:
    from state_evolution_gtrain_gtest_core import (
        EPS,
        centered_loss_to_gain,
        gh_standard_normal,
    )


def _activation_terms(x: torch.Tensor, activation: str):
    name = str(activation).lower()
    if name == "linear":
        value = x
        first = torch.ones_like(x)
        second = torch.zeros_like(x)
    elif name == "relu":
        value = torch.clamp_min(x, 0.0)
        first = (x > 0.0).to(dtype=x.dtype)
        second = torch.zeros_like(x)
    elif name == "tanh":
        value = torch.tanh(x)
        first = 1.0 - value.square()
        second = -2.0 * value * first
    elif name == "elu":
        positive = x > 0.0
        exp_x = torch.exp(x)
        value = torch.where(positive, x, exp_x - 1.0)
        first = torch.where(positive, torch.ones_like(x), exp_x)
        second = torch.where(positive, torch.zeros_like(x), exp_x)
    else:
        raise ValueError(f"unsupported torch activation: {activation!r}")
    return value, first, second


def finite_k_loss_grad_hess_torch(
    g: torch.Tensor,
    *,
    Q: float,
    rho: float,
    activation: str,
    need_hess: bool = True,
):
    """Torch equivalent of ``_finite_k_loss_grad_hess_batch``."""
    if g.ndim != 2 or g.shape[1] < 2:
        raise ValueError("g must have shape (n_samples, K+1), K>=1")
    h = g[:, 0]
    x = g[:, 1:]
    K = int(x.shape[1])
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = mask_variance**0.5

    y = visible * h[:, None] + root_mask_variance * x
    t = hidden * h[:, None] - root_mask_variance * x
    sigma, sigma_prime, sigma_second = _activation_terms(y, activation)
    loss = torch.mean(hidden * float(Q) * sigma.square() - 2.0 * sigma * t, dim=1)
    U = 2.0 * (
        visible * t * sigma_prime
        + hidden * sigma
        - visible * hidden * float(Q) * sigma * sigma_prime
    )
    W = 2.0 * root_mask_variance * (
        t * sigma_prime - sigma - hidden * float(Q) * sigma * sigma_prime
    )
    gradient = torch.empty_like(g)
    gradient[:, 0] = -torch.mean(U, dim=1)
    gradient[:, 1:] = -W / K
    if not need_hess:
        return loss, gradient, None

    h_hh = 2.0 * (
        visible**2
        * hidden
        * float(Q)
        * (sigma_prime.square() + sigma * sigma_second)
        - 2.0 * visible * hidden * sigma_prime
        - visible**2 * t * sigma_second
    )
    h_xx = 2.0 * mask_variance * (
        hidden * float(Q) * (sigma_prime.square() + sigma * sigma_second)
        + 2.0 * sigma_prime
        - t * sigma_second
    )
    h_hx = 2.0 * root_mask_variance * (
        visible
        * hidden
        * float(Q)
        * (sigma_prime.square() + sigma * sigma_second)
        + (visible - hidden) * sigma_prime
        - visible * t * sigma_second
    )
    sample_count, dimension = g.shape
    hessian = torch.zeros(
        (sample_count, dimension, dimension), dtype=g.dtype, device=g.device
    )
    hessian[:, 0, 0] = torch.mean(h_hh, dim=1)
    hessian[:, 0, 1:] = h_hx / K
    hessian[:, 1:, 0] = h_hx / K
    indices = torch.arange(K, device=g.device)
    hessian[:, 1 + indices, 1 + indices] = h_xx / K
    return loss, gradient, hessian


@torch.no_grad()
def finite_k_smooth_prox_batch_torch(
    g0: torch.Tensor,
    *,
    Q: float,
    Delta: float,
    rho: float,
    activation: str,
    max_steps: int = 35,
    tol: float = 1e-9,
    multistart: bool = True,
):
    """Vectorized multistart proximal matching the NumPy implementation."""
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    sample_count, dimension = g0.shape
    starts = [g0.clone()]
    if multistart:
        starts.extend([0.5 * g0, torch.zeros_like(g0)])
    best_g = None
    best_objective = None
    diagonal = torch.arange(dimension, device=g0.device)

    for start in starts:
        g = start.clone()
        for _ in range(int(max_steps)):
            loss, gradient_loss, hessian = finite_k_loss_grad_hess_torch(
                g, Q=Q, rho=rho, activation=activation, need_hess=True
            )
            gradient = (g - g0) / float(Delta) + gradient_loss
            hessian[:, diagonal, diagonal] += 1.0 / float(Delta)
            step, info = torch.linalg.solve_ex(
                hessian, gradient.unsqueeze(-1), check_errors=False
            )
            step = step[..., 0]
            bad_solve = info.ne(0) | (~torch.isfinite(step).all(dim=1))
            if bool(torch.any(bad_solve).item()):
                step[bad_solve] = float(Delta) * gradient[bad_solve]
            not_descent = torch.sum(gradient * step, dim=1) <= 1e-12
            step[not_descent] = float(Delta) * gradient[not_descent]
            step_norm = torch.linalg.vector_norm(step, dim=1)
            cap = 4.0 * (float(Delta) + 1.0) ** 0.5
            step *= torch.minimum(
                torch.ones_like(step_norm), cap / torch.clamp_min(step_norm, 1e-12)
            )[:, None]

            objective = (
                0.5 * torch.sum((g - g0).square(), dim=1) / float(Delta) + loss
            )
            accepted = torch.zeros(sample_count, dtype=torch.bool, device=g.device)
            line_scale = torch.ones(sample_count, dtype=g.dtype, device=g.device)
            g_new = g.clone()
            for _ in range(14):
                candidate = g - line_scale[:, None] * step
                candidate_loss, _, _ = finite_k_loss_grad_hess_torch(
                    candidate,
                    Q=Q,
                    rho=rho,
                    activation=activation,
                    need_hess=False,
                )
                candidate_objective = (
                    0.5
                    * torch.sum((candidate - g0).square(), dim=1)
                    / float(Delta)
                    + candidate_loss
                )
                okay = (
                    (~accepted)
                    & torch.isfinite(candidate_objective)
                    & (candidate_objective <= objective + 1e-12)
                )
                g_new[okay] = candidate[okay]
                accepted |= okay
                line_scale[~accepted] *= 0.5
                if bool(torch.all(accepted).item()):
                    break
            if bool(torch.any(~accepted).item()):
                bad = torch.nonzero(~accepted, as_tuple=False)[:, 0]
                candidate = g[bad] - 0.05 * float(Delta) * gradient[bad]
                candidate_loss, _, _ = finite_k_loss_grad_hess_torch(
                    candidate,
                    Q=Q,
                    rho=rho,
                    activation=activation,
                    need_hess=False,
                )
                candidate_objective = (
                    0.5
                    * torch.sum((candidate - g0[bad]).square(), dim=1)
                    / float(Delta)
                    + candidate_loss
                )
                improved = candidate_objective < objective[bad]
                g_new[bad[improved]] = candidate[improved]
            change = torch.max(torch.abs(g_new - g))
            g = g_new
            if float(change.detach().cpu()) < float(tol):
                break

        loss, _, _ = finite_k_loss_grad_hess_torch(
            g, Q=Q, rho=rho, activation=activation, need_hess=False
        )
        objective = (
            0.5 * torch.sum((g - g0).square(), dim=1) / float(Delta) + loss
        )
        if best_g is None:
            best_g = g.clone()
            best_objective = objective.clone()
        else:
            improved = objective < best_objective
            best_g[improved] = g[improved]
            best_objective[improved] = objective[improved]
    return best_g, best_objective


@lru_cache(maxsize=32)
def _gh_tensors(order: int, device: str):
    nodes, weights = gh_standard_normal(int(order))
    resolved = torch.device(device)
    return (
        torch.as_tensor(nodes, dtype=torch.float64, device=resolved),
        torch.as_tensor(weights, dtype=torch.float64, device=resolved),
    )


@torch.no_grad()
def fixed_smooth_channel_torch(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
    activation: str,
    *,
    quad_order: int = 21,
    grid_size: int = 201,
    newton_steps: int = 25,
    device: str = "cuda",
):
    """GPU equivalent of the specialized static K=1 smooth channel."""
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    z, weights = _gh_tensors(int(quad_order), str(torch.device(device)))
    latent = z[:, None, None]
    noise_y = z[None, :, None]
    noise_t = z[None, None, :]
    quad_weights = (
        weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )
    visible = 1.0 - float(rho)
    hidden = float(rho)
    C = max(float(C), 0.0)
    sqrt_C = max(C, EPS) ** 0.5
    y0 = (
        visible * np.sqrt(beta) * float(M) * latent
        + (visible * C) ** 0.5 * noise_y
    )
    t0 = hidden * np.sqrt(beta) * float(M) * latent + (hidden * C) ** 0.5 * noise_t

    scale = (
        max(visible * float(Delta), EPS)
        + float(torch.mean(y0.square()).cpu())
        + float(torch.mean(t0.square()).cpu())
        + 1.0
    ) ** 0.5
    offsets = torch.linspace(
        -8.0 * scale,
        8.0 * scale,
        int(grid_size),
        dtype=torch.float64,
        device=z.device,
    )
    candidates = y0[..., None] + offsets
    sigma, _, _ = _activation_terms(candidates, activation)
    objective = (
        (candidates - y0[..., None]).square() / (2.0 * visible * float(Delta))
        + hidden * (float(Q) - 2.0 * float(Delta)) * sigma.square()
        - 2.0 * t0[..., None] * sigma
    )
    indices = torch.argmin(objective, dim=-1, keepdim=True)
    y = torch.gather(candidates.expand_as(objective), -1, indices)[..., 0]
    for _ in range(int(newton_steps)):
        sigma, sigma_prime, sigma_second = _activation_terms(y, activation)
        gradient = (
            (y - y0) / (visible * float(Delta))
            + 2.0
            * hidden
            * (float(Q) - 2.0 * float(Delta))
            * sigma
            * sigma_prime
            - 2.0 * t0 * sigma_prime
        )
        hessian = (
            1.0 / (visible * float(Delta))
            + 2.0
            * hidden
            * (float(Q) - 2.0 * float(Delta))
            * (sigma_prime.square() + sigma * sigma_second)
            - 2.0 * t0 * sigma_second
        )
        step = torch.where(hessian.abs() > 1e-11, gradient / hessian, 0.0)
        step = torch.clamp(step, -2.0 * scale, 2.0 * scale)
        candidate = y - step
        sigma_candidate, _, _ = _activation_terms(candidate, activation)
        candidate_objective = (
            (candidate - y0).square() / (2.0 * visible * float(Delta))
            + hidden
            * (float(Q) - 2.0 * float(Delta))
            * sigma_candidate.square()
            - 2.0 * t0 * sigma_candidate
        )
        current_objective = (
            (y - y0).square() / (2.0 * visible * float(Delta))
            + hidden * (float(Q) - 2.0 * float(Delta)) * sigma.square()
            - 2.0 * t0 * sigma
        )
        y = torch.where(
            torch.isfinite(candidate_objective)
            & (candidate_objective <= current_objective),
            candidate,
            y,
        )

    sigma, _, _ = _activation_terms(y, activation)
    force_y = (y - y0) / (visible * float(Delta))
    force_t = 2.0 * sigma
    force_signal = visible * force_y + hidden * force_t
    force_square = visible * force_y.square() + hidden * force_t.square()
    derivative_C = (
        -force_y * visible**0.5 * noise_y / (2.0 * sqrt_C)
        - force_t * hidden**0.5 * noise_t / (2.0 * sqrt_C)
    )
    t_star = t0 + 2.0 * hidden * float(Delta) * sigma
    train_loss = hidden * float(Q) * sigma.square() - 2.0 * sigma * t_star
    sigma0, _, _ = _activation_terms(y0, activation)
    test_loss = hidden * float(Q) * sigma0.square() - 2.0 * sigma0 * t0
    moments = torch.stack(
        [
            torch.sum(quad_weights * latent * force_signal),
            torch.sum(quad_weights * sigma.square()),
            torch.sum(quad_weights * force_square),
            torch.sum(quad_weights * derivative_C),
            torch.sum(quad_weights * train_loss),
            torch.sum(quad_weights * test_loss),
        ]
    ).cpu().numpy()
    expected_latent_force, J, G, H, train_centered_loss, test_centered_loss = map(
        float, moments
    )
    U = hidden * J
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "J": J,
        "G": G,
        "force_square": G,
        "U": U,
        "H": H,
        "E_lam_A": expected_latent_force,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "quad_order": int(quad_order),
        "endpoint_backend": "torch",
        "torch_device": str(z.device),
    }


def _dynamic_loss_terms_torch(
    h: torch.Tensor,
    *,
    Q: float,
    C: float,
    rho: float,
    activation: str,
    xi: torch.Tensor,
    weights: torch.Tensor,
):
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    C_effective = max(float(C), EPS)
    root = (mask_variance * C_effective) ** 0.5
    field = h[..., None]
    y = visible * field + root * xi
    t = hidden * field - root * xi
    sigma, sigma_prime, sigma_second = _activation_terms(y, activation)
    loss = torch.sum(
        weights * (hidden * float(Q) * sigma.square() - 2.0 * sigma * t), dim=-1
    )
    derivative_Q = torch.sum(weights * hidden * sigma.square(), dim=-1)
    derivative_C = (mask_variance / C_effective) ** 0.5 * torch.sum(
        weights
        * xi
        * (hidden * float(Q) * sigma * sigma_prime - t * sigma_prime + sigma),
        dim=-1,
    )
    derivative_h = torch.sum(
        weights
        * 2.0
        * (
            visible * hidden * float(Q) * sigma * sigma_prime
            - visible * t * sigma_prime
            - hidden * sigma
        ),
        dim=-1,
    )
    derivative_hh = torch.sum(
        weights
        * 2.0
        * (
            visible**2
            * hidden
            * float(Q)
            * (sigma_prime.square() + sigma * sigma_second)
            - 2.0 * visible * hidden * sigma_prime
            - visible**2 * t * sigma_second
        ),
        dim=-1,
    )
    return loss, derivative_h, derivative_hh, derivative_Q, derivative_C


@torch.no_grad()
def dynamic_smooth_channel_torch(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
    activation: str,
    *,
    quad_order: int = 21,
    mask_quad_order: int | None = None,
    grid_size: int = 201,
    newton_steps: int = 25,
    device: str = "cuda",
):
    """GPU equivalent of the smooth dynamic/annealed output channel."""
    if mask_quad_order is None:
        mask_quad_order = quad_order
    z, weights = _gh_tensors(int(quad_order), str(torch.device(device)))
    xi, xi_weights = _gh_tensors(int(mask_quad_order), str(torch.device(device)))
    latent = z[:, None]
    noise = z[None, :]
    quad_weights = weights[:, None] * weights[None, :]
    C = max(float(C), 0.0)
    sqrt_C = max(C, EPS) ** 0.5
    h0 = np.sqrt(beta) * float(M) * latent + C**0.5 * noise
    scale = (
        max(float(Delta), EPS)
        + C
        + float(torch.mean(h0.square()).cpu())
        + 1.0
    ) ** 0.5
    offsets = torch.linspace(
        -8.0 * scale,
        8.0 * scale,
        int(grid_size),
        dtype=torch.float64,
        device=z.device,
    )
    candidates = h0[..., None] + offsets
    loss, _, _, _, _ = _dynamic_loss_terms_torch(
        candidates,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        xi=xi,
        weights=xi_weights,
    )
    objective = 0.5 * (candidates - h0[..., None]).square() / float(Delta) + loss
    indices = torch.argmin(objective, dim=-1, keepdim=True)
    h = torch.gather(candidates, -1, indices)[..., 0]
    for _ in range(int(newton_steps)):
        loss, derivative_h, derivative_hh, _, _ = _dynamic_loss_terms_torch(
            h,
            Q=Q,
            C=C,
            rho=rho,
            activation=activation,
            xi=xi,
            weights=xi_weights,
        )
        gradient = (h - h0) / float(Delta) + derivative_h
        hessian = 1.0 / float(Delta) + derivative_hh
        step = torch.where(hessian.abs() > 1e-11, gradient / hessian, 0.0)
        step = torch.clamp(step, -2.0 * scale, 2.0 * scale)
        candidate = h - step
        candidate_loss, _, _, _, _ = _dynamic_loss_terms_torch(
            candidate,
            Q=Q,
            C=C,
            rho=rho,
            activation=activation,
            xi=xi,
            weights=xi_weights,
        )
        candidate_objective = (
            0.5 * (candidate - h0).square() / float(Delta) + candidate_loss
        )
        current_objective = 0.5 * (h - h0).square() / float(Delta) + loss
        h = torch.where(
            torch.isfinite(candidate_objective)
            & (candidate_objective <= current_objective),
            candidate,
            h,
        )

    force = (h - h0) / float(Delta)
    train_loss, _, _, derivative_Q, derivative_C_loss = _dynamic_loss_terms_torch(
        h,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        xi=xi,
        weights=xi_weights,
    )
    test_loss, _, _, _, _ = _dynamic_loss_terms_torch(
        h0,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        xi=xi,
        weights=xi_weights,
    )
    derivative_C = derivative_C_loss - force * noise / (2.0 * sqrt_C)
    moments = torch.stack(
        [
            torch.sum(quad_weights * latent * force),
            torch.sum(quad_weights * force.square()),
            torch.sum(quad_weights * derivative_Q),
            torch.sum(quad_weights * derivative_C),
            torch.sum(quad_weights * train_loss),
            torch.sum(quad_weights * test_loss),
        ]
    ).cpu().numpy()
    expected_latent_force, force_square, U, H, train_centered_loss, test_centered_loss = map(
        float, moments
    )
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "J": U / rho,
        "A2": force_square,
        "force_square": force_square,
        "U": U,
        "H": H,
        "E_lam_A": expected_latent_force,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "quad_order": int(quad_order),
        "mask_quad_order": int(mask_quad_order),
        "endpoint_backend": "torch",
        "torch_device": str(z.device),
    }


__all__ = [
    "_activation_terms",
    "dynamic_smooth_channel_torch",
    "finite_k_loss_grad_hess_torch",
    "finite_k_smooth_prox_batch_torch",
    "fixed_smooth_channel_torch",
]
