#!/usr/bin/env python3
"""Smooth-activation MAE state evolution extracted from MAE_stateevolution.ipynb.

This module adds the notebook's generic K=1 and K=infinity output channels to
the already-extracted generic finite-K channel.  It intentionally lives in a
separate module so the long-running ReLU sweep can continue using its frozen
runner while ELU and tanh are added.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .state_evolution_generic_activation_core import (
        _final_result,
        generic_finite_k_channel,
        get_activation_spec,
        torch_generic_finite_k_channel,
    )
    from .state_evolution_gtrain_gtest_core import (
        EPS,
        centered_loss_to_gain,
        compute_macros,
        coordinate_update,
        gh_standard_normal,
        initial_state,
    )
    from .state_evolution_smooth_torch_backend import (
        dynamic_smooth_channel_torch,
        fixed_smooth_channel_torch,
    )
except ImportError:
    from state_evolution_generic_activation_core import (
        _final_result,
        generic_finite_k_channel,
        get_activation_spec,
        torch_generic_finite_k_channel,
    )
    from state_evolution_gtrain_gtest_core import (
        EPS,
        centered_loss_to_gain,
        compute_macros,
        coordinate_update,
        gh_standard_normal,
        initial_state,
    )
    from state_evolution_smooth_torch_backend import (
        dynamic_smooth_channel_torch,
        fixed_smooth_channel_torch,
    )


SOURCE_NOTEBOOK = "scripts/MAE_stateevolution.ipynb"


def generic_fixed_y_star(
    y0,
    t0,
    Q,
    Delta,
    rho,
    activation,
    grid_size=161,
    newton_steps=25,
):
    """Globally seeded, damped-Newton scalar proximal for static masking."""
    spec = get_activation_spec(activation)
    y0 = np.asarray(y0, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    visible = 1.0 - float(rho)
    hidden = float(rho)
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")

    scale = np.sqrt(
        max(visible * float(Delta), EPS)
        + float(np.mean(y0 * y0))
        + float(np.mean(t0 * t0))
        + 1.0
    )
    radius = 8.0 * scale
    offsets = np.linspace(-radius, radius, int(grid_size))
    candidates = y0[..., None] + offsets
    sigma = spec.fn(candidates)
    objective = (
        (candidates - y0[..., None]) ** 2 / (2.0 * visible * Delta)
        + hidden * (Q - 2.0 * Delta) * sigma**2
        - 2.0 * t0[..., None] * sigma
    )
    indices = np.argmin(objective, axis=-1)
    y = np.take_along_axis(candidates, indices[..., None], axis=-1)[..., 0]

    for _ in range(int(newton_steps)):
        sigma = spec.fn(y)
        sigma_prime = spec.d1(y)
        sigma_second = spec.d2(y)
        gradient = (
            (y - y0) / (visible * Delta)
            + 2.0
            * hidden
            * (Q - 2.0 * Delta)
            * sigma
            * sigma_prime
            - 2.0 * t0 * sigma_prime
        )
        hessian = (
            1.0 / (visible * Delta)
            + 2.0
            * hidden
            * (Q - 2.0 * Delta)
            * (sigma_prime**2 + sigma * sigma_second)
            - 2.0 * t0 * sigma_second
        )
        step = np.where(np.abs(hessian) > 1e-11, gradient / hessian, 0.0)
        step = np.clip(step, -2.0 * scale, 2.0 * scale)
        candidate = y - step
        sigma_candidate = spec.fn(candidate)
        objective_candidate = (
            (candidate - y0) ** 2 / (2.0 * visible * Delta)
            + hidden * (Q - 2.0 * Delta) * sigma_candidate**2
            - 2.0 * t0 * sigma_candidate
        )
        sigma_y = spec.fn(y)
        objective_y = (
            (y - y0) ** 2 / (2.0 * visible * Delta)
            + hidden * (Q - 2.0 * Delta) * sigma_y**2
            - 2.0 * t0 * sigma_y
        )
        accept = np.isfinite(objective_candidate) & (
            objective_candidate <= objective_y
        )
        y = np.where(accept, candidate, y)
    return y


def generic_fixed_channel(
    M,
    Q,
    C,
    Delta,
    beta,
    rho,
    activation,
    quad_order=21,
    prox_grid_size=161,
    prox_newton_steps=25,
):
    """Notebook static (K=1) output channel for a registered activation."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    spec = get_activation_spec(activation)
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    visible = 1.0 - rho
    hidden = rho

    z, weights = gh_standard_normal(quad_order)
    latent = z[:, None, None]
    noise_y = z[None, :, None]
    noise_t = z[None, None, :]
    quadrature_weights = (
        weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )
    y0 = (
        visible * np.sqrt(beta) * M * latent
        + np.sqrt(visible * C) * noise_y
    )
    t0 = hidden * np.sqrt(beta) * M * latent + np.sqrt(hidden * C) * noise_t
    y_star = generic_fixed_y_star(
        y0,
        t0,
        Q=Q,
        Delta=Delta,
        rho=rho,
        activation=spec,
        grid_size=prox_grid_size,
        newton_steps=prox_newton_steps,
    )
    sigma = spec.fn(y_star)
    force_y = (y_star - y0) / (visible * Delta)
    force_t = 2.0 * sigma
    force_signal = visible * force_y + hidden * force_t
    force_square = visible * force_y**2 + hidden * force_t**2
    derivative_C = (
        -force_y * np.sqrt(visible) * noise_y / (2.0 * sqrt_C)
        - force_t * np.sqrt(hidden) * noise_t / (2.0 * sqrt_C)
    )
    expectation = lambda value: float(np.sum(quadrature_weights * value))
    expected_latent_force = expectation(latent * force_signal)
    J = expectation(sigma**2)
    U = hidden * J
    G = expectation(force_square)
    H = expectation(derivative_C)
    t_star = t0 + 2.0 * hidden * Delta * sigma
    train_centered_loss = expectation(hidden * Q * sigma**2 - 2.0 * sigma * t_star)
    sigma0 = spec.fn(y0)
    test_centered_loss = expectation(hidden * Q * sigma0**2 - 2.0 * sigma0 * t0)
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
    }


def generic_dynamic_loss_terms(h, Q, C, rho, activation, mask_quad_order=21):
    """Dynamic loss and its h, Q, and C derivatives by Gaussian quadrature."""
    spec = get_activation_spec(activation)
    h = np.asarray(h, dtype=float)
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    C_effective = max(float(C), EPS)
    xi, weights = gh_standard_normal(mask_quad_order)
    root = np.sqrt(mask_variance * C_effective)
    field = h[..., None]
    y = visible * field + root * xi
    t = hidden * field - root * xi
    sigma = spec.fn(y)
    sigma_prime = spec.d1(y)
    sigma_second = spec.d2(y)
    loss = np.sum(weights * (hidden * Q * sigma**2 - 2.0 * sigma * t), axis=-1)
    derivative_Q = np.sum(weights * hidden * sigma**2, axis=-1)
    derivative_C = np.sqrt(mask_variance / C_effective) * np.sum(
        weights
        * xi
        * (hidden * Q * sigma * sigma_prime - t * sigma_prime + sigma),
        axis=-1,
    )
    derivative_h = np.sum(
        weights
        * 2.0
        * (
            visible * hidden * Q * sigma * sigma_prime
            - visible * t * sigma_prime
            - hidden * sigma
        ),
        axis=-1,
    )
    derivative_hh = np.sum(
        weights
        * 2.0
        * (
            visible**2
            * hidden
            * Q
            * (sigma_prime**2 + sigma * sigma_second)
            - 2.0 * visible * hidden * sigma_prime
            - visible**2 * t * sigma_second
        ),
        axis=-1,
    )
    return loss, derivative_h, derivative_hh, derivative_Q, derivative_C


def generic_dynamic_prox(
    h0,
    Q,
    C,
    Delta,
    rho,
    activation,
    mask_quad_order=21,
    grid_size=201,
    newton_steps=25,
):
    """Globally seeded, damped-Newton proximal for the dynamic channel."""
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    h0 = np.asarray(h0, dtype=float)
    scale = np.sqrt(
        max(float(Delta), EPS)
        + max(float(C), 0.0)
        + float(np.mean(h0**2))
        + 1.0
    )
    radius = 8.0 * scale
    offsets = np.linspace(-radius, radius, int(grid_size))
    candidates = h0[..., None] + offsets
    loss, _, _, _, _ = generic_dynamic_loss_terms(
        candidates,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        mask_quad_order=mask_quad_order,
    )
    objective = 0.5 * (candidates - h0[..., None]) ** 2 / Delta + loss
    indices = np.argmin(objective, axis=-1)
    h = np.take_along_axis(candidates, indices[..., None], axis=-1)[..., 0]

    for _ in range(int(newton_steps)):
        loss, derivative_h, derivative_hh, _, _ = generic_dynamic_loss_terms(
            h,
            Q=Q,
            C=C,
            rho=rho,
            activation=activation,
            mask_quad_order=mask_quad_order,
        )
        gradient = (h - h0) / Delta + derivative_h
        hessian = 1.0 / Delta + derivative_hh
        step = np.where(np.abs(hessian) > 1e-11, gradient / hessian, 0.0)
        step = np.clip(step, -2.0 * scale, 2.0 * scale)
        candidate = h - step
        candidate_loss, _, _, _, _ = generic_dynamic_loss_terms(
            candidate,
            Q=Q,
            C=C,
            rho=rho,
            activation=activation,
            mask_quad_order=mask_quad_order,
        )
        candidate_objective = 0.5 * (candidate - h0) ** 2 / Delta + candidate_loss
        current_objective = 0.5 * (h - h0) ** 2 / Delta + loss
        accept = np.isfinite(candidate_objective) & (
            candidate_objective <= current_objective
        )
        h = np.where(accept, candidate, h)
    return h


def generic_dynamic_channel(
    M,
    Q,
    C,
    Delta,
    beta,
    rho,
    activation,
    quad_order=21,
    mask_quad_order=None,
    prox_grid_size=201,
    prox_newton_steps=25,
):
    """Notebook K=infinity (dynamic/annealed) registered-activation channel."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    if mask_quad_order is None:
        mask_quad_order = quad_order
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    z, weights = gh_standard_normal(quad_order)
    latent = z[:, None]
    noise = z[None, :]
    quadrature_weights = weights[:, None] * weights[None, :]
    h0 = np.sqrt(beta) * M * latent + np.sqrt(C) * noise
    h_star = generic_dynamic_prox(
        h0,
        Q=Q,
        C=C,
        Delta=Delta,
        rho=rho,
        activation=activation,
        mask_quad_order=mask_quad_order,
        grid_size=prox_grid_size,
        newton_steps=prox_newton_steps,
    )
    force = (h_star - h0) / Delta
    train_loss, _, _, derivative_Q, derivative_C_loss = generic_dynamic_loss_terms(
        h_star,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        mask_quad_order=mask_quad_order,
    )
    test_loss, _, _, _, _ = generic_dynamic_loss_terms(
        h0,
        Q=Q,
        C=C,
        rho=rho,
        activation=activation,
        mask_quad_order=mask_quad_order,
    )
    derivative_C = derivative_C_loss - force * noise / (2.0 * sqrt_C)
    expectation = lambda value: float(np.sum(quadrature_weights * value))
    expected_latent_force = expectation(latent * force)
    force_square = expectation(force**2)
    U = expectation(derivative_Q)
    H = expectation(derivative_C)
    train_centered_loss = expectation(train_loss)
    test_centered_loss = expectation(test_loss)
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
    }


def run_smooth_mae_se(
    *,
    alpha,
    beta,
    rho,
    lambda_r,
    gamma,
    weights,
    mask_count,
    activation,
    init_state=None,
    init_m_scale=5e-2,
    max_iter=20_000,
    tol=1e-10,
    damping=0.05,
    verbose=False,
    quad_order=21,
    mask_quad_order=None,
    prox_grid_size=201,
    prox_newton_steps=25,
    sobol_power=9,
    qmc_seed=1234,
    finite_k_prox_steps=35,
    finite_k_prox_tol=1e-9,
    k1_backend="specialized",
    antithetic_qmc=False,
    finite_k_backend="numpy",
    endpoint_backend="numpy",
    torch_device="cuda",
):
    """Run ELU/tanh state evolution for finite K or dynamic masking."""
    spec = get_activation_spec(activation)
    if not spec.smooth_at_zero:
        raise ValueError(
            f"{spec.name} is not a smooth activation; use its dedicated runner"
        )
    dynamic = bool(np.isinf(mask_count))
    if dynamic:
        K = None
    else:
        K = int(mask_count)
        if K < 1 or float(mask_count) != float(K):
            raise ValueError("mask_count must be a positive integer or infinity")
    if k1_backend not in {"specialized", "generic"}:
        raise ValueError("k1_backend must be 'specialized' or 'generic'")
    if finite_k_backend not in {"numpy", "torch"}:
        raise ValueError("finite_k_backend must be 'numpy' or 'torch'")
    if endpoint_backend not in {"numpy", "torch"}:
        raise ValueError("endpoint_backend must be 'numpy' or 'torch'")
    if init_state is None:
        m, q, v = initial_state(
            alpha, rho, gamma, weights, init_m_scale=init_m_scale
        )
    else:
        m, q, v = [np.asarray(value, dtype=float).copy() for value in init_state]
    status = "max_iter"
    diff = np.nan
    raw_residual = np.nan
    info: dict[str, Any] = {}
    iteration = -1

    def step(current_m, current_q, current_v):
        M, Q, C, Delta, Ev = compute_macros(
            current_m, current_q, current_v, gamma, weights
        )
        if dynamic and endpoint_backend == "torch":
            channel = dynamic_smooth_channel_torch(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                activation=spec.name,
                quad_order=quad_order,
                mask_quad_order=mask_quad_order,
                grid_size=prox_grid_size,
                newton_steps=prox_newton_steps,
                device=torch_device,
            )
        elif dynamic:
            channel = generic_dynamic_channel(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                activation=spec,
                quad_order=quad_order,
                mask_quad_order=mask_quad_order,
                prox_grid_size=prox_grid_size,
                prox_newton_steps=prox_newton_steps,
            )
        elif (
            K == 1
            and k1_backend == "specialized"
            and endpoint_backend == "torch"
        ):
            channel = fixed_smooth_channel_torch(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                activation=spec.name,
                quad_order=quad_order,
                grid_size=prox_grid_size,
                newton_steps=prox_newton_steps,
                device=torch_device,
            )
        elif K == 1 and k1_backend == "specialized":
            channel = generic_fixed_channel(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                activation=spec,
                quad_order=quad_order,
                prox_grid_size=prox_grid_size,
                prox_newton_steps=prox_newton_steps,
            )
        elif finite_k_backend == "torch":
            channel = torch_generic_finite_k_channel(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                mask_count=K,
                activation=spec,
                sobol_power=sobol_power,
                qmc_seed=qmc_seed,
                antithetic_qmc=antithetic_qmc,
                prox_steps=finite_k_prox_steps,
                prox_tol=finite_k_prox_tol,
                test_quad_order=quad_order,
                device=torch_device,
            )
        else:
            channel = generic_finite_k_channel(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                mask_count=K,
                activation=spec,
                sobol_power=sobol_power,
                qmc_seed=qmc_seed,
                antithetic_qmc=antithetic_qmc,
                prox_steps=finite_k_prox_steps,
                prox_tol=finite_k_prox_tol,
                test_quad_order=quad_order,
            )
        hat_m = alpha * np.sqrt(beta) * channel["E_lam_A"]
        hat_q = alpha * gamma * channel["force_square"]
        hat_v = 2.0 * alpha * (channel["U"] + gamma * channel["H"])
        return coordinate_update(
            hat_m, hat_q, hat_v, lambda_r, M, Q, C, Delta, Ev, channel
        )

    for iteration in range(int(max_iter)):
        try:
            m_raw, q_raw, v_raw, info = step(m, q, v)
        except Exception as error:
            status = f"failed: {error}"
            break
        raw_residual = max(
            float(np.max(np.abs(m_raw - m))),
            float(np.max(np.abs(q_raw - q))),
            float(np.max(np.abs(v_raw - v))),
        )
        # The residual and channel moments above belong to (m, q, v).
        # Return that checked state, rather than applying one unchecked update.
        if raw_residual < tol:
            diff = 0.0  # No update is applied on the accepting iteration.
            status = "converged"
            break
        m_next = (1.0 - damping) * m + damping * m_raw
        q_next = (1.0 - damping) * q + damping * q_raw
        v_next = (1.0 - damping) * v + damping * v_raw
        diff = max(
            float(np.max(np.abs(m_next - m))),
            float(np.max(np.abs(q_next - q))),
            float(np.max(np.abs(v_next - v))),
        )
        m, q, v = m_next, q_next, v_next
        if verbose and iteration % 100 == 0:
            label = "dynamic" if dynamic else f"K={K}"
            print(
                f"{spec.name} {label} iter={iteration} diff={diff:.6e} "
                f"raw_residual={raw_residual:.6e}",
                flush=True,
            )
        if not np.isfinite(diff):
            status = "failed: non-finite diff"
            break

    if status != "converged":
        try:
            m_raw, q_raw, v_raw, info = step(m, q, v)
            raw_residual = max(
                float(np.max(np.abs(m_raw - m))),
                float(np.max(np.abs(q_raw - q))),
                float(np.max(np.abs(v_raw - v))),
            )
        except Exception:
            pass
    result = _final_result(
        m=m,
        q=q,
        v=v,
        gamma=gamma,
        weights=weights,
        info=info,
        model=("annealed_mask_MAE" if dynamic else f"finite_{K}_mask_MAE"),
        mask_count=(np.inf if dynamic else K),
        activation=spec.name,
        alpha=alpha,
        beta=beta,
        rho=rho,
        lambda_r=lambda_r,
        n_iter=iteration + 1,
        diff=diff,
        raw_residual=raw_residual,
        status=status,
    )
    result["k1_backend"] = (
        k1_backend if (not dynamic and K == 1) else "not_applicable"
    )
    return result
