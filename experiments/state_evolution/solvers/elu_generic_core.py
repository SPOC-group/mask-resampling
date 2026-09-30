#!/usr/bin/env python3
"""Generic finite-K state evolution extracted from the MAE notebooks.

The nonlinear K>1 output channel is evaluated by a fixed scrambled Sobol
Gaussian rule. ReLU K>1 uses the exact active-set/profile proximal from
three_state_evolutions_relu_finiteK_fixed.ipynb; smooth activations retain the
vectorized multistart Moreau proximal. ReLU K=1 and K=infinity deliberately use
their dedicated endpoint channels. All solvers test convergence using the
*undamped* fixed-point residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

try:
    from .state_evolution_gtrain_gtest_core import (
        EPS,
        annealed_relu_channel,
        centered_loss_to_gain,
        compute_macros,
        coordinate_update,
        finite_k_linear_channel,
        gh_standard_normal,
        initial_state,
        relu,
    )
except ImportError:
    from state_evolution_gtrain_gtest_core import (
        EPS,
        annealed_relu_channel,
        centered_loss_to_gain,
        compute_macros,
        coordinate_update,
        finite_k_linear_channel,
        gh_standard_normal,
        initial_state,
        relu,
    )


SOURCE_NOTEBOOK = "scripts/MAE_stateevolution.ipynb"
RELU_FINITE_K_SOURCE_NOTEBOOK = (
    "scripts/three_state_evolutions_relu_finiteK_fixed.ipynb"
)


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    fn: object
    d1: object
    d2: object
    sigma0: float = np.nan
    slope0: float = np.nan
    smooth_at_zero: bool = True


ACTIVATIONS: dict[str, ActivationSpec] = {}


def register_activation(
    name,
    fn,
    d1,
    d2,
    sigma0=np.nan,
    slope0=np.nan,
    smooth_at_zero=True,
):
    """Register an elementwise activation and its first two derivatives."""
    key = str(name).lower()
    ACTIVATIONS[key] = ActivationSpec(
        name=key,
        fn=fn,
        d1=d1,
        d2=d2,
        sigma0=float(sigma0) if np.isfinite(sigma0) else np.nan,
        slope0=float(slope0) if np.isfinite(slope0) else np.nan,
        smooth_at_zero=bool(smooth_at_zero),
    )
    return ACTIVATIONS[key]


def _linear_fn(x):
    return np.asarray(x, dtype=float)


def _linear_d1(x):
    return np.ones_like(np.asarray(x, dtype=float))


def _linear_d2(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _relu_fn(x):
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def _relu_d1(x):
    return (np.asarray(x, dtype=float) > 0.0).astype(float)


def _relu_d2(x):
    return np.zeros_like(np.asarray(x, dtype=float))


def _tanh_fn(x):
    return np.tanh(np.asarray(x, dtype=float))


def _tanh_d1(x):
    value = np.tanh(np.asarray(x, dtype=float))
    return 1.0 - value * value


def _tanh_d2(x):
    value = np.tanh(np.asarray(x, dtype=float))
    derivative = 1.0 - value * value
    return -2.0 * value * derivative


def _elu_fn(x):
    value = np.asarray(x, dtype=float)
    return np.where(value > 0.0, value, np.expm1(value))


def _elu_d1(x):
    value = np.asarray(x, dtype=float)
    return np.where(value > 0.0, 1.0, np.exp(value))


def _elu_d2(x):
    value = np.asarray(x, dtype=float)
    return np.where(value > 0.0, 0.0, np.exp(value))


register_activation("linear", _linear_fn, _linear_d1, _linear_d2, 0.0, 1.0)
register_activation(
    "relu", _relu_fn, _relu_d1, _relu_d2,
    sigma0=0.0, slope0=np.nan, smooth_at_zero=False,
)
register_activation("tanh", _tanh_fn, _tanh_d1, _tanh_d2, 0.0, 1.0)
register_activation("elu", _elu_fn, _elu_d1, _elu_d2, 0.0, 1.0)


def get_activation_spec(activation) -> ActivationSpec:
    if isinstance(activation, ActivationSpec):
        return activation
    key = str(activation).lower()
    if key not in ACTIVATIONS:
        raise ValueError(
            f"unknown activation {activation!r}; registered: {sorted(ACTIVATIONS)}"
        )
    return ACTIVATIONS[key]


def fixed_relu_y_star(y0, t0, Q, Delta, rho):
    """Exact scalar ReLU proximal for the K=1 static endpoint."""
    visible = 1.0 - rho
    precision = 1.0 / (visible * Delta) + 2.0 * rho * (Q - 2.0 * Delta)
    if precision <= 0.0:
        raise FloatingPointError(
            f"unstable fixed-mask ReLU positive branch: {precision=:.6e}"
        )
    constant = y0**2 / (2.0 * visible * Delta)
    y_negative = np.minimum(y0, 0.0)
    objective_negative = np.where(y0 <= 0.0, 0.0, constant)
    linear = y0 / (visible * Delta) + 2.0 * t0
    y_positive = np.maximum(linear / precision, 0.0)
    objective_positive = (
        constant + 0.5 * precision * y_positive**2 - linear * y_positive
    )
    return np.where(
        objective_positive <= objective_negative, y_positive, y_negative
    )


def fixed_relu_channel(M, Q, C, Delta, beta, rho, quad_order=21):
    """Dedicated K=1 ReLU channel from MAE_stateevolution.ipynb."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    z, weights = gh_standard_normal(quad_order)
    latent = z[:, None, None]
    noise_y = z[None, :, None]
    noise_t = z[None, None, :]
    quadrature_weights = (
        weights[:, None, None]
        * weights[None, :, None]
        * weights[None, None, :]
    )
    visible = 1.0 - rho
    y0 = (
        visible * np.sqrt(beta) * M * latent
        + np.sqrt(visible * C) * noise_y
    )
    t0 = rho * np.sqrt(beta) * M * latent + np.sqrt(rho * C) * noise_t
    y_star = fixed_relu_y_star(y0, t0, Q=Q, Delta=Delta, rho=rho)
    phi = relu(y_star)
    force_y = (y_star - y0) / (visible * Delta)
    force_t = 2.0 * phi
    force_signal = visible * force_y + rho * force_t
    force_square = visible * force_y**2 + rho * force_t**2
    derivative_C = (
        -force_y * np.sqrt(visible) * noise_y / (2.0 * sqrt_C)
        - force_t * np.sqrt(rho) * noise_t / (2.0 * sqrt_C)
    )
    expectation = lambda value: float(np.sum(quadrature_weights * value))
    expected_latent_force = expectation(latent * force_signal)
    J = expectation(phi**2)
    G = expectation(force_square)
    H = expectation(derivative_C)
    t_star = t0 + 2.0 * rho * Delta * phi
    train_centered_loss = expectation(rho * Q * phi**2 - 2.0 * phi * t_star)
    phi0 = relu(y0)
    test_centered_loss = expectation(rho * Q * phi0**2 - 2.0 * phi0 * t0)
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "J": J,
        "G": G,
        "force_square": G,
        "H": H,
        "U": rho * J,
        "E_lam_A": expected_latent_force,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "quad_order": int(quad_order),
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "k1_backend": "tensor3d",
    }


@lru_cache(maxsize=16)
def _gh_standard_normal_pairs(order):
    """Cached tensor-product standard-normal nodes and weights in two dimensions."""
    nodes, weights = gh_standard_normal(int(order))
    normal_1, normal_2 = np.meshgrid(nodes, nodes, indexing="ij")
    normal_pair = np.column_stack([normal_1.ravel(), normal_2.ravel()])
    quadrature_weights = np.multiply.outer(weights, weights).ravel()
    return normal_pair, quadrature_weights


def fixed_relu_channel_gaussian_2d(M, Q, C, Delta, beta, rho, quad_order=81):
    """K=1 ReLU channel reduced to the joint Gaussian pair ``(y0, t0)``.

    The dedicated notebook expression is a three-dimensional expectation over
    the latent variable and two independent noises.  The scalar proximal only
    depends on the two correlated Gaussian fields ``(y0, t0)``.  Conditional
    Gaussian regression supplies the latent/noise factors needed by the
    derivatives, reducing the quadrature from ``order**3`` to ``order**2``.
    This is mathematically equivalent but permits a much finer deterministic
    rule for the nonsmooth ReLU active-set boundary.  An even ``quad_order``
    is recommended so that Gauss--Hermite does not place a node exactly at
    the ReLU kink.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    visible = 1.0 - rho

    signal_y = visible * np.sqrt(beta) * M
    signal_t = rho * np.sqrt(beta) * M
    noise_y_scale = np.sqrt(visible * C)
    noise_t_scale = np.sqrt(rho * C)
    covariance = np.array(
        [
            [signal_y**2 + noise_y_scale**2, signal_y * signal_t],
            [signal_y * signal_t, signal_t**2 + noise_t_scale**2],
        ],
        dtype=float,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    covariance_root = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    normal_pair, quadrature_weights = _gh_standard_normal_pairs(quad_order)
    fields = normal_pair @ covariance_root.T
    y0 = fields[:, 0]
    t0 = fields[:, 1]

    covariance_inverse = np.linalg.pinv(covariance, rcond=1e-13)
    field_pair = np.column_stack([y0, t0])
    latent_regression = (
        np.array([signal_y, signal_t]) @ covariance_inverse
    )
    noise_y_regression = (
        np.array([noise_y_scale, 0.0]) @ covariance_inverse
    )
    noise_t_regression = (
        np.array([0.0, noise_t_scale]) @ covariance_inverse
    )
    latent_conditional_mean = field_pair @ latent_regression
    noise_y_conditional_mean = field_pair @ noise_y_regression
    noise_t_conditional_mean = field_pair @ noise_t_regression

    y_star = fixed_relu_y_star(y0, t0, Q=Q, Delta=Delta, rho=rho)
    phi = relu(y_star)
    force_y = (y_star - y0) / (visible * Delta)
    force_t = 2.0 * phi
    force_signal = visible * force_y + rho * force_t
    force_square = visible * force_y**2 + rho * force_t**2
    derivative_C = (
        -force_y
        * np.sqrt(visible)
        * noise_y_conditional_mean
        / (2.0 * sqrt_C)
        - force_t
        * np.sqrt(rho)
        * noise_t_conditional_mean
        / (2.0 * sqrt_C)
    )
    expectation = lambda value: float(np.sum(quadrature_weights * value))
    expected_latent_force = expectation(latent_conditional_mean * force_signal)
    J = expectation(phi**2)
    G = expectation(force_square)
    H = expectation(derivative_C)
    t_star = t0 + 2.0 * rho * Delta * phi
    train_centered_loss = expectation(rho * Q * phi**2 - 2.0 * phi * t_star)
    phi0 = relu(y0)
    test_centered_loss = expectation(rho * Q * phi0**2 - 2.0 * phi0 * t0)
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "J": J,
        "G": G,
        "force_square": G,
        "H": H,
        "U": rho * J,
        "E_lam_A": expected_latent_force,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "quad_order": int(quad_order),
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "k1_backend": "gaussian2d",
    }


@lru_cache(maxsize=128)
def _sobol_standard_normals(dim, sobol_power, seed, antithetic=False):
    sampler = qmc.Sobol(d=int(dim), scramble=True, seed=int(seed))
    sample_power = int(sobol_power) - (1 if antithetic else 0)
    if sample_power < 1:
        raise ValueError("sobol_power must be at least 2 for antithetic sampling")
    uniforms = sampler.random_base2(m=sample_power)
    uniforms = np.clip(uniforms, 1e-12, 1.0 - 1e-12)
    normals = ndtri(uniforms)
    if antithetic:
        normals = np.concatenate([normals, -normals], axis=0)
    return normals


@lru_cache(maxsize=128)
def _torch_sobol_standard_normals(
    dim, sobol_power, seed, antithetic=False, device="cuda"
):
    """Cache the deterministic Sobol rule directly on a PyTorch device."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the torch finite-K backend requires PyTorch") from error
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved} requested but torch.cuda.is_available() is false"
        )
    normals = np.ascontiguousarray(
        _sobol_standard_normals(dim, sobol_power, seed, antithetic),
        dtype=np.float64,
    )
    return torch.as_tensor(normals, dtype=torch.float64, device=resolved)


def _finite_k_loss_grad_hess_batch(g, Q, rho, activation, need_hess=True):
    """Finite-K loss and derivatives in whitened fields g=(h,x_1,...,x_K)."""
    spec = get_activation_spec(activation)
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.shape[1] < 2:
        raise ValueError("g must have shape (n_samples, K+1), K>=1")
    h = g[:, 0]
    x = g[:, 1:]
    K = x.shape[1]
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = np.sqrt(mask_variance)

    y = visible * h[:, None] + root_mask_variance * x
    t = hidden * h[:, None] - root_mask_variance * x
    sigma = spec.fn(y)
    sigma_prime = spec.d1(y)
    loss = np.mean(hidden * Q * sigma**2 - 2.0 * sigma * t, axis=1)
    U = 2.0 * (
        visible * t * sigma_prime
        + hidden * sigma
        - visible * hidden * Q * sigma * sigma_prime
    )
    W = 2.0 * root_mask_variance * (
        t * sigma_prime - sigma - hidden * Q * sigma * sigma_prime
    )
    gradient = np.empty_like(g)
    gradient[:, 0] = -np.mean(U, axis=1)
    gradient[:, 1:] = -W / K
    if not need_hess:
        return loss, gradient, None

    sigma_second = spec.d2(y)
    h_hh = 2.0 * (
        visible**2 * hidden * Q * (sigma_prime**2 + sigma * sigma_second)
        - 2.0 * visible * hidden * sigma_prime
        - visible**2 * t * sigma_second
    )
    h_xx = 2.0 * mask_variance * (
        hidden * Q * (sigma_prime**2 + sigma * sigma_second)
        + 2.0 * sigma_prime
        - t * sigma_second
    )
    h_hx = 2.0 * root_mask_variance * (
        visible * hidden * Q * (sigma_prime**2 + sigma * sigma_second)
        + (visible - hidden) * sigma_prime
        - visible * t * sigma_second
    )
    sample_count, dimension = g.shape
    hessian = np.zeros((sample_count, dimension, dimension), dtype=float)
    hessian[:, 0, 0] = np.mean(h_hh, axis=1)
    hessian[:, 0, 1:] = h_hx / K
    hessian[:, 1:, 0] = h_hx / K
    indices = np.arange(K)
    hessian[:, 1 + indices, 1 + indices] = h_xx / K
    return loss, gradient, hessian


def _finite_k_relu_active_quadratic_coeff(x0, Q, Delta, rho, K):
    """Profiled active-branch coefficients for one finite-K ReLU view."""
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = np.sqrt(mask_variance)
    shifted_noise = root_mask_variance * float(x0)
    output_curvature = hidden * float(Q) + 2.0
    linear_h = visible / (Delta * mask_variance) + 2.0 / K
    curvature_y = (
        1.0 / (Delta * mask_variance) + 2.0 * output_curvature / K
    )
    A = (
        visible**2 / (Delta * mask_variance)
        - linear_h**2 / curvature_y
    )
    B = (
        visible * shifted_noise / (Delta * mask_variance)
        - linear_h
        * (shifted_noise / (Delta * mask_variance))
        / curvature_y
    )
    C0 = (
        shifted_noise**2 / (2.0 * Delta * mask_variance)
        - (shifted_noise / (Delta * mask_variance)) ** 2
        / (2.0 * curvature_y)
    )
    return A, B, C0, linear_h, curvature_y


def _finite_k_relu_conditional_y(h, x0, Q, Delta, rho, K):
    """Exact per-view minimizers conditional on the shared field h."""
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = np.sqrt(mask_variance)
    x0 = np.asarray(x0, dtype=float)
    cavity_y = visible * float(h) + root_mask_variance * x0

    y_negative = np.minimum(cavity_y, 0.0)
    value_negative = (
        (y_negative - cavity_y) ** 2 / (2.0 * Delta * mask_variance)
    )

    output_curvature = hidden * float(Q) + 2.0
    curvature_y = (
        1.0 / (Delta * mask_variance) + 2.0 * output_curvature / K
    )
    linear_y = cavity_y / (Delta * mask_variance) + 2.0 * float(h) / K
    y_positive = np.maximum(linear_y / curvature_y, 0.0)
    value_positive = (
        (y_positive - cavity_y) ** 2 / (2.0 * Delta * mask_variance)
        + (
            output_curvature * y_positive**2
            - 2.0 * float(h) * y_positive
        )
        / K
    )

    use_positive = value_positive < value_negative
    y = np.where(use_positive, y_positive, y_negative)
    value = np.where(use_positive, value_positive, value_negative)
    return y, value, use_positive


def _finite_k_relu_prox_single(g0, Q, Delta, rho):
    """Global finite-K ReLU proximal via a scalar active-set profile."""
    g0 = np.asarray(g0, dtype=float)
    K = g0.size - 1
    if K < 1:
        raise ValueError("finite-K ReLU proximal requires K>=1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0<rho<1")

    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = np.sqrt(mask_variance)
    h0 = float(g0[0])
    x0 = g0[1:]

    # On each interval the profiled objective is 0.5*A*h^2 + B*h + C.
    A_total = 1.0 / Delta
    B_total = -h0 / Delta
    C_total = h0**2 / (2.0 * Delta)
    events = []

    for x_value in x0:
        A_active, B_active, C_active, linear_h, curvature_y = (
            _finite_k_relu_active_quadratic_coeff(
                x_value, Q=Q, Delta=Delta, rho=rho, K=K
            )
        )
        shifted_noise = root_mask_variance * float(x_value)
        cavity_zero = -shifted_noise / visible
        active_zero = -(
            shifted_noise / (Delta * mask_variance)
        ) / linear_h

        if x_value >= 0.0:
            A_zero = visible**2 / (Delta * mask_variance)
            B_zero = visible * shifted_noise / (Delta * mask_variance)
            C_zero = shifted_noise**2 / (2.0 * Delta * mask_variance)
            events.append((cavity_zero, A_zero, B_zero, C_zero))
            events.append(
                (
                    active_zero,
                    A_active - A_zero,
                    B_active - B_zero,
                    C_active - C_zero,
                )
            )
        else:
            ratio = np.sqrt(curvature_y / (Delta * mask_variance))
            crossing = -shifted_noise * (
                1.0 / (Delta * mask_variance) + ratio
            ) / (linear_h + visible * ratio)
            events.append((crossing, A_active, B_active, C_active))

    events.sort(key=lambda event: event[0])
    best_value = np.inf
    best_h = None
    left = -np.inf

    def consider(h, A, B, C0):
        nonlocal best_value, best_h
        value = 0.5 * A * h**2 + B * h + C0
        if np.isfinite(value) and value < best_value:
            best_value = float(value)
            best_h = float(h)

    event_index = 0
    while event_index < len(events):
        right = float(events[event_index][0])
        if A_total > 1e-14:
            stationary = -B_total / A_total
            if left < stationary < right:
                consider(stationary, A_total, B_total, C_total)
        consider(right, A_total, B_total, C_total)

        next_index = event_index
        while (
            next_index < len(events)
            and abs(events[next_index][0] - right)
            <= 1e-12 * (1.0 + abs(right))
        ):
            _, delta_A, delta_B, delta_C = events[next_index]
            A_total += delta_A
            B_total += delta_B
            C_total += delta_C
            next_index += 1
        left = right
        event_index = next_index

    if A_total < -1e-12 or (
        abs(A_total) <= 1e-12 and B_total < -1e-12
    ):
        raise FloatingPointError(
            "Unbounded finite-K ReLU Moreau profile on the positive-h tail: "
            f"final curvature={A_total:.6e}, final slope={B_total:.6e}"
        )
    if A_total > 1e-14:
        stationary = -B_total / A_total
        if stationary > left:
            consider(stationary, A_total, B_total, C_total)

    if best_h is None:
        raise FloatingPointError("Could not locate a finite finite-K ReLU proximal")

    y_star, _, _ = _finite_k_relu_conditional_y(
        best_h, x0, Q=Q, Delta=Delta, rho=rho, K=K
    )
    x_star = (y_star - visible * best_h) / root_mask_variance
    return np.concatenate([[best_h], x_star]), best_value


def _finite_k_relu_prox_batch(g0, Q, Delta, rho):
    """Batch wrapper for the exact finite-K ReLU profile solver."""
    g0 = np.asarray(g0, dtype=float)
    stars = np.empty_like(g0)
    values = np.empty(g0.shape[0], dtype=float)
    for sample_index in range(g0.shape[0]):
        stars[sample_index], values[sample_index] = _finite_k_relu_prox_single(
            g0[sample_index], Q=Q, Delta=Delta, rho=rho
        )
    return stars, values


def _finite_k_prox_batch(
    g0,
    Q,
    Delta,
    rho,
    activation,
    max_steps=35,
    tol=1e-9,
    multistart=True,
):
    """Vectorized multistart Moreau proximal from the generalized notebook."""
    g0 = np.asarray(g0, dtype=float)
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    sample_count, dimension = g0.shape
    starts = [g0.copy()]
    if multistart:
        starts += [0.5 * g0, np.zeros_like(g0)]
    best_g = None
    best_objective = None
    diagonal = np.arange(dimension)

    for start in starts:
        g = start.copy()
        for _ in range(int(max_steps)):
            loss, loss_gradient, loss_hessian = _finite_k_loss_grad_hess_batch(
                g, Q=Q, rho=rho, activation=activation, need_hess=True
            )
            gradient = (g - g0) / Delta + loss_gradient
            hessian = loss_hessian.copy()
            hessian[:, diagonal, diagonal] += 1.0 / Delta
            try:
                step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
            except np.linalg.LinAlgError:
                step = Delta * gradient
            not_descent = np.sum(gradient * step, axis=1) <= 1e-12
            step[not_descent] = Delta * gradient[not_descent]
            step_norm = np.linalg.norm(step, axis=1)
            cap = 4.0 * np.sqrt(Delta + 1.0)
            step *= np.minimum(1.0, cap / np.maximum(step_norm, 1e-12))[:, None]

            objective = 0.5 * np.sum((g - g0) ** 2, axis=1) / Delta + loss
            accepted = np.zeros(sample_count, dtype=bool)
            line_scale = np.ones(sample_count, dtype=float)
            g_new = g.copy()
            for _ in range(14):
                candidate = g - line_scale[:, None] * step
                candidate_loss, _, _ = _finite_k_loss_grad_hess_batch(
                    candidate, Q=Q, rho=rho, activation=activation,
                    need_hess=False,
                )
                candidate_objective = (
                    0.5 * np.sum((candidate - g0) ** 2, axis=1) / Delta
                    + candidate_loss
                )
                okay = (
                    (~accepted)
                    & np.isfinite(candidate_objective)
                    & (candidate_objective <= objective + 1e-12)
                )
                g_new[okay] = candidate[okay]
                accepted |= okay
                line_scale[~accepted] *= 0.5
                if accepted.all():
                    break
            if (~accepted).any():
                bad = np.where(~accepted)[0]
                candidate = g[bad] - 0.05 * Delta * gradient[bad]
                candidate_loss, _, _ = _finite_k_loss_grad_hess_batch(
                    candidate, Q=Q, rho=rho, activation=activation,
                    need_hess=False,
                )
                candidate_objective = (
                    0.5 * np.sum((candidate - g0[bad]) ** 2, axis=1) / Delta
                    + candidate_loss
                )
                improved = candidate_objective < objective[bad]
                g_new[bad[improved]] = candidate[improved]
            change = float(np.max(np.abs(g_new - g)))
            g = g_new
            if change < tol:
                break

        loss, _, _ = _finite_k_loss_grad_hess_batch(
            g, Q=Q, rho=rho, activation=activation, need_hess=False
        )
        objective = 0.5 * np.sum((g - g0) ** 2, axis=1) / Delta + loss
        if best_g is None:
            best_g = g.copy()
            best_objective = objective.copy()
        else:
            improved = objective < best_objective
            best_g[improved] = g[improved]
            best_objective[improved] = objective[improved]
    return best_g, best_objective


def generic_fresh_mask_centered_loss(M, Q, C, beta, rho, activation, quad_order=21):
    """Fresh-sample/fresh-mask centered loss, independent of finite K."""
    spec = get_activation_spec(activation)
    visible = 1.0 - float(rho)
    hidden = float(rho)
    C = max(float(C), 0.0)
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
    sigma = spec.fn(y0)
    return float(
        np.sum(quadrature_weights * (hidden * Q * sigma**2 - 2.0 * sigma * t0))
    )


def generic_finite_k_channel(
    M,
    Q,
    C,
    Delta,
    beta,
    rho,
    mask_count,
    activation,
    sobol_power=9,
    qmc_seed=1234,
    antithetic_qmc=False,
    prox_steps=35,
    prox_tol=1e-9,
    test_quad_order=21,
    compute_test_loss=True,
):
    """General nonlinear finite-K output channel from MAE_stateevolution.ipynb."""
    K = int(mask_count)
    if K < 1 or float(mask_count) != float(K):
        raise ValueError("mask_count must be a positive integer")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    spec = get_activation_spec(activation)
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    normals = _sobol_standard_normals(
        K + 2,
        int(sobol_power),
        int(qmc_seed),
        bool(antithetic_qmc),
    )
    latent = normals[:, 0]
    noise_h = normals[:, 1]
    noise_x = normals[:, 2:]
    h0 = np.sqrt(beta) * M * latent + sqrt_C * noise_h
    x0 = sqrt_C * noise_x
    g0 = np.column_stack([h0, x0])
    if spec.name == "relu":
        g_star, _ = _finite_k_relu_prox_batch(
            g0, Q=Q, Delta=Delta, rho=rho
        )
        prox_solver = "relu_exact_piecewise_profile"
    else:
        g_star, _ = _finite_k_prox_batch(
            g0,
            Q=Q,
            Delta=Delta,
            rho=rho,
            activation=spec,
            max_steps=prox_steps,
            tol=prox_tol,
            multistart=True,
        )
        prox_solver = "generic_newton_multistart"
    force = (g_star - g0) / Delta
    loss_star, loss_gradient, _ = _finite_k_loss_grad_hess_batch(
        g_star, Q=Q, rho=rho, activation=spec, need_hess=False
    )
    visible = 1.0 - rho
    mask_variance = rho * visible
    y_star = (
        visible * g_star[:, [0]]
        + np.sqrt(mask_variance) * g_star[:, 1:]
    )
    sigma = spec.fn(y_star)
    dQ = rho * np.mean(sigma**2, axis=1)
    derivative_C = -np.sum(force * normals[:, 1:], axis=1) / (2.0 * sqrt_C)
    expected_latent_force = float(np.mean(latent * force[:, 0]))
    force_square = float(np.mean(np.sum(force**2, axis=1)))
    U = float(np.mean(dQ))
    H = float(np.mean(derivative_C))
    train_centered_loss = float(np.mean(loss_star))
    test_centered_loss = (
        generic_fresh_mask_centered_loss(
            M, Q, C, beta, rho, activation=spec, quad_order=test_quad_order
        )
        if compute_test_loss
        else np.nan
    )
    prox_residual = (
        np.nan
        if spec.name == "relu"
        else float(np.max(np.abs(force + loss_gradient)))
    )
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "mask_count": K,
        "E_lam_A": expected_latent_force,
        "force_square": force_square,
        "G": force_square,
        "U": U,
        "J": U / rho,
        "H": H,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "prox_residual_max": prox_residual,
        "sobol_samples": int(2 ** int(sobol_power)),
        "qmc_seed": int(qmc_seed),
        "antithetic_qmc": bool(antithetic_qmc),
        "prox_solver": prox_solver,
        "finite_k_backend": "numpy",
    }


def torch_finite_k_relu_channel(
    M,
    Q,
    C,
    Delta,
    beta,
    rho,
    mask_count,
    sobol_power=9,
    qmc_seed=1234,
    antithetic_qmc=False,
    test_quad_order=21,
    compute_test_loss=True,
    device="cuda",
):
    """Finite-K ReLU channel evaluated by batched float64 PyTorch tensors."""
    try:
        import torch
        from .state_evolution_relu_torch_backend import (
            finite_k_relu_prox_batch_torch,
        )
    except ImportError:
        try:
            import torch
            from state_evolution_relu_torch_backend import (
                finite_k_relu_prox_batch_torch,
            )
        except ImportError as error:
            raise RuntimeError(
                "the torch finite-K backend requires PyTorch and its backend module"
            ) from error

    K = int(mask_count)
    if K < 2 or float(mask_count) != float(K):
        raise ValueError("the torch ReLU channel requires an integer K>=2")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved_device} requested but CUDA is unavailable"
        )

    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    normals = _torch_sobol_standard_normals(
        K + 2,
        int(sobol_power),
        int(qmc_seed),
        bool(antithetic_qmc),
        str(resolved_device),
    )
    latent = normals[:, 0]
    noise_h = normals[:, 1]
    noise_x = normals[:, 2:]
    h0 = np.sqrt(beta) * float(M) * latent + sqrt_C * noise_h
    x0 = sqrt_C * noise_x
    g0 = torch.cat([h0[:, None], x0], dim=1)
    g_star, _ = finite_k_relu_prox_batch_torch(
        g0, Q=float(Q), Delta=float(Delta), rho=float(rho)
    )
    force = (g_star - g0) / float(Delta)

    visible = 1.0 - float(rho)
    hidden = float(rho)
    root_mask_variance = np.sqrt(hidden * visible)
    h_star = g_star[:, [0]]
    x_star = g_star[:, 1:]
    y_star = visible * h_star + root_mask_variance * x_star
    t_star = hidden * h_star - root_mask_variance * x_star
    sigma = torch.relu(y_star)
    loss_star = torch.mean(
        hidden * float(Q) * sigma.square() - 2.0 * sigma * t_star,
        dim=1,
    )
    dQ = hidden * torch.mean(sigma.square(), dim=1)
    derivative_C = -torch.sum(force * normals[:, 1:], dim=1) / (2.0 * sqrt_C)

    # Copy all channel moments to the host together.  Calling ``item`` on each
    # moment separately would introduce five CUDA synchronization points per
    # state-evolution iteration.
    moments = torch.stack(
        [
            torch.mean(latent * force[:, 0]),
            torch.mean(torch.sum(force.square(), dim=1)),
            torch.mean(dQ),
            torch.mean(derivative_C),
            torch.mean(loss_star),
        ]
    ).detach().cpu().numpy()
    (
        expected_latent_force,
        force_square,
        U,
        H,
        train_centered_loss,
    ) = map(float, moments)
    test_centered_loss = (
        generic_fresh_mask_centered_loss(
            M, Q, C, beta, rho, activation="relu", quad_order=test_quad_order
        )
        if compute_test_loss
        else np.nan
    )
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "mask_count": K,
        "E_lam_A": expected_latent_force,
        "force_square": force_square,
        "G": force_square,
        "U": U,
        "J": U / rho,
        "H": H,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "prox_residual_max": np.nan,
        "sobol_samples": int(2 ** int(sobol_power)),
        "qmc_seed": int(qmc_seed),
        "antithetic_qmc": bool(antithetic_qmc),
        "prox_solver": "relu_exact_piecewise_profile_torch_batched",
        "finite_k_backend": "torch",
        "torch_device": str(resolved_device),
        "torch_dtype": "float64",
    }


def torch_generic_finite_k_channel(
    M,
    Q,
    C,
    Delta,
    beta,
    rho,
    mask_count,
    activation,
    sobol_power=9,
    qmc_seed=1234,
    antithetic_qmc=False,
    prox_steps=35,
    prox_tol=1e-9,
    test_quad_order=21,
    compute_test_loss=True,
    device="cuda",
):
    """Smooth finite-K channel using the batched float64 PyTorch proximal."""
    try:
        import torch
        from .elu_torch_backend import (
            _activation_terms,
            finite_k_loss_grad_hess_torch,
            finite_k_smooth_prox_batch_torch,
        )
    except ImportError:
        try:
            import torch
            from elu_torch_backend import (
                _activation_terms,
                finite_k_loss_grad_hess_torch,
                finite_k_smooth_prox_batch_torch,
            )
        except ImportError as error:
            raise RuntimeError(
                "the torch smooth finite-K backend requires PyTorch"
            ) from error

    spec = get_activation_spec(activation)
    if spec.name not in {"linear", "relu", "elu", "tanh"}:
        raise ValueError(f"unsupported smooth torch activation: {spec.name}")
    K = int(mask_count)
    if K < 1 or float(mask_count) != float(K):
        raise ValueError("mask_count must be a positive integer")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {resolved_device} requested but CUDA is unavailable"
        )

    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    normals = _torch_sobol_standard_normals(
        K + 2,
        int(sobol_power),
        int(qmc_seed),
        bool(antithetic_qmc),
        str(resolved_device),
    )
    latent = normals[:, 0]
    noise_h = normals[:, 1]
    noise_x = normals[:, 2:]
    h0 = np.sqrt(beta) * float(M) * latent + sqrt_C * noise_h
    x0 = sqrt_C * noise_x
    g0 = torch.cat([h0[:, None], x0], dim=1)
    g_star, _ = finite_k_smooth_prox_batch_torch(
        g0,
        Q=float(Q),
        Delta=float(Delta),
        rho=float(rho),
        activation=spec.name,
        max_steps=int(prox_steps),
        tol=float(prox_tol),
        multistart=True,
    )
    force = (g_star - g0) / float(Delta)
    loss_star, loss_gradient, _ = finite_k_loss_grad_hess_torch(
        g_star,
        Q=float(Q),
        rho=float(rho),
        activation=spec.name,
        need_hess=False,
    )
    visible = 1.0 - float(rho)
    hidden = float(rho)
    root_mask_variance = np.sqrt(hidden * visible)
    y_star = visible * g_star[:, [0]] + root_mask_variance * g_star[:, 1:]
    sigma, _, _ = _activation_terms(y_star, spec.name)
    dQ = hidden * torch.mean(sigma.square(), dim=1)
    derivative_C = -torch.sum(force * normals[:, 1:], dim=1) / (2.0 * sqrt_C)
    prox_residual = torch.max(torch.abs(force + loss_gradient))
    moments = torch.stack(
        [
            torch.mean(latent * force[:, 0]),
            torch.mean(torch.sum(force.square(), dim=1)),
            torch.mean(dQ),
            torch.mean(derivative_C),
            torch.mean(loss_star),
            prox_residual,
        ]
    ).detach().cpu().numpy()
    (
        expected_latent_force,
        force_square,
        U,
        H,
        train_centered_loss,
        prox_residual_max,
    ) = map(float, moments)
    test_centered_loss = (
        generic_fresh_mask_centered_loss(
            M, Q, C, beta, rho, activation=spec, quad_order=test_quad_order
        )
        if compute_test_loss
        else np.nan
    )
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "mask_count": K,
        "E_lam_A": expected_latent_force,
        "force_square": force_square,
        "G": force_square,
        "U": U,
        "J": U / rho,
        "H": H,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "train_centered_loss": train_centered_loss,
        "test_centered_loss": test_centered_loss,
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
        "prox_residual_max": prox_residual_max,
        "sobol_samples": int(2 ** int(sobol_power)),
        "qmc_seed": int(qmc_seed),
        "antithetic_qmc": bool(antithetic_qmc),
        "prox_solver": "generic_newton_multistart_torch_batched",
        "finite_k_backend": "torch",
        "torch_device": str(resolved_device),
        "torch_dtype": "float64",
    }


def _final_result(
    *,
    m,
    q,
    v,
    gamma,
    weights,
    info,
    model,
    mask_count,
    activation,
    alpha,
    beta,
    rho,
    lambda_r,
    n_iter,
    diff,
    raw_residual,
    status,
):
    M, Q, C, Delta, Ev = compute_macros(m, q, v, gamma, weights)
    cosine = M / np.sqrt(Q) if Q > 0.0 else np.nan
    scalar_info = {
        key: value for key, value in info.items()
        if np.isscalar(value) and key not in {"M", "Q", "C", "Delta", "Ev"}
    }
    return {
        **scalar_info,
        "model": model,
        "model_family": (
            "finite_K_mask_MAE" if np.isfinite(mask_count) else "dynamic_mask_MAE"
        ),
        "mask_count": mask_count,
        "mask_label": (
            "K=1 (static)" if mask_count == 1
            else (f"K={int(mask_count)}" if np.isfinite(mask_count) else "K=inf (dynamic)")
        ),
        "mask_protocol": (
            "finite quenched masks" if np.isfinite(mask_count) else "annealed/dynamic"
        ),
        "activation": str(activation),
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "lambda_r": float(lambda_r),
        "M": M,
        "Q": Q,
        "C": C,
        "Delta": Delta,
        "Ev": Ev,
        "cosine": float(cosine) if np.isfinite(cosine) else np.nan,
        "cosine_sq": float(cosine**2) if np.isfinite(cosine) else np.nan,
        "theta_u_star": float(abs(cosine)) if np.isfinite(cosine) else np.nan,
        "q_gamma_over_q_I": float(C / Q) if Q > 0.0 else np.nan,
        "q_ratio": float(C / Q) if Q > 0.0 else np.nan,
        "n_iter": int(n_iter),
        "diff": float(diff) if np.isfinite(diff) else np.nan,
        "se_raw_residual": (
            float(raw_residual) if np.isfinite(raw_residual) else np.nan
        ),
        "status": status,
        "m": m,
        "q": q,
        "v": v,
    }


def run_relu_mae_se(
    *,
    alpha,
    beta,
    rho,
    lambda_r,
    gamma,
    weights,
    mask_count,
    init_state=None,
    init_m_scale=5e-2,
    init_fallback_delta=1.0,
    init_delta_override=None,
    max_iter=20_000,
    tol=1e-10,
    damping=0.3,
    verbose=False,
    quad_order=31,
    prox_grid_size=301,
    prox_newton_steps=25,
    sobol_power=9,
    qmc_seed=1234,
    antithetic_qmc=False,
    finite_k_prox_steps=35,
    finite_k_prox_tol=1e-9,
    k1_backend="tensor3d",
    finite_k_backend="numpy",
    torch_device="cuda",
):
    """Run one ReLU fixed point for finite K or dynamic masking.

    ``mask_count=np.inf`` selects the dedicated dynamic ReLU endpoint.
    Finite K=1 selects the dedicated static ReLU endpoint.  K>1 selects the
    notebook's generalized finite-K QMC channel.
    """
    dynamic = bool(np.isinf(mask_count))
    if not dynamic:
        K = int(mask_count)
        if K < 1 or float(mask_count) != float(K):
            raise ValueError("mask_count must be a positive integer or infinity")
    else:
        K = None
    if k1_backend not in {"tensor3d", "gaussian2d"}:
        raise ValueError("k1_backend must be 'tensor3d' or 'gaussian2d'")
    if finite_k_backend not in {"numpy", "torch"}:
        raise ValueError("finite_k_backend must be 'numpy' or 'torch'")
    if init_state is None:
        if init_delta_override is None:
            m, q, v = initial_state(
                alpha,
                rho,
                gamma,
                weights,
                init_m_scale=init_m_scale,
                fallback_Delta=init_fallback_delta,
            )
        else:
            delta0 = float(init_delta_override)
            if not np.isfinite(delta0) or delta0 <= 0.0:
                raise ValueError("init_delta_override must be positive and finite")
            m = float(init_m_scale) * np.ones_like(gamma)
            v = delta0 * np.ones_like(gamma)
            q = 4.0 * gamma * delta0 + m**2
    else:
        m, q, v = [np.asarray(value, dtype=float).copy() for value in init_state]
    status = "max_iter"
    diff = np.nan
    raw_residual = np.nan
    info: dict[str, Any] = {}
    iteration = -1

    def step(current_m, current_q, current_v, *, compute_diagnostics=False):
        M, Q, C, Delta, Ev = compute_macros(
            current_m, current_q, current_v, gamma, weights
        )
        if dynamic:
            channel = annealed_relu_channel(
                M, Q, C, Delta, beta, rho,
                quad_order=quad_order,
                prox_grid_size=prox_grid_size,
                prox_newton_steps=prox_newton_steps,
            )
            force_square = channel["A2"]
        elif K == 1:
            if k1_backend == "gaussian2d":
                channel = fixed_relu_channel_gaussian_2d(
                    M, Q, C, Delta, beta, rho, quad_order=quad_order
                )
            else:
                channel = fixed_relu_channel(
                    M, Q, C, Delta, beta, rho, quad_order=quad_order
                )
            force_square = channel["G"]
        elif finite_k_backend == "torch":
            channel = torch_finite_k_relu_channel(
                M, Q, C, Delta, beta, rho,
                mask_count=K,
                sobol_power=sobol_power,
                qmc_seed=qmc_seed,
                antithetic_qmc=antithetic_qmc,
                test_quad_order=quad_order,
                compute_test_loss=compute_diagnostics,
                device=torch_device,
            )
            force_square = channel["force_square"]
        else:
            channel = generic_finite_k_channel(
                M, Q, C, Delta, beta, rho,
                mask_count=K,
                activation="relu",
                sobol_power=sobol_power,
                qmc_seed=qmc_seed,
                antithetic_qmc=antithetic_qmc,
                prox_steps=finite_k_prox_steps,
                prox_tol=finite_k_prox_tol,
                test_quad_order=quad_order,
                compute_test_loss=compute_diagnostics,
            )
            force_square = channel["force_square"]
        hat_m = alpha * np.sqrt(beta) * channel["E_lam_A"]
        hat_q = alpha * gamma * force_square
        if dynamic:
            hat_v = 2.0 * alpha * (channel["U"] + gamma * channel["H"])
        elif K == 1:
            hat_v = 2.0 * alpha * (rho * channel["J"] + gamma * channel["H"])
        else:
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
        # Preserve the iterate whose undamped residual passed tolerance.
        if raw_residual < tol:
            diff = 0.0
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
                f"relu {label} iter={iteration} diff={diff:.6e} "
                f"raw_residual={raw_residual:.6e}",
                flush=True,
            )
        if not np.isfinite(diff):
            status = "failed: non-finite diff"
            break

    if status != "converged":
        try:
            m_raw, q_raw, v_raw, info = step(m, q, v, compute_diagnostics=True)
            raw_residual = max(
                float(np.max(np.abs(m_raw - m))),
                float(np.max(np.abs(q_raw - q))),
                float(np.max(np.abs(v_raw - v))),
            )
        except Exception:
            pass
    return _final_result(
        m=m,
        q=q,
        v=v,
        gamma=gamma,
        weights=weights,
        info=info,
        model=("annealed_mask_MAE" if dynamic else f"finite_{K}_mask_MAE"),
        mask_count=(np.inf if dynamic else K),
        activation="relu",
        alpha=alpha,
        beta=beta,
        rho=rho,
        lambda_r=lambda_r,
        n_iter=iteration + 1,
        diff=diff,
        raw_residual=raw_residual,
        status=status,
    )


def validate_notebook_extraction() -> dict[str, float]:
    """Check deterministic QMC and the generic linear channel against exact theory."""
    first = _sobol_standard_normals(5, 7, 1234)
    second = _sobol_standard_normals(5, 7, 1234)
    if not np.array_equal(first, second):
        raise AssertionError("cached Sobol normal rule is not deterministic")
    state = dict(M=0.5, Q=2.1, C=1.2, Delta=0.2, beta=1.0, rho=0.75)
    exact = finite_k_linear_channel(**state, mask_count=2)
    generic = generic_finite_k_channel(
        **state,
        mask_count=2,
        activation="linear",
        sobol_power=14,
        qmc_seed=1234,
        prox_steps=20,
        prox_tol=1e-11,
        test_quad_order=31,
    )
    relative_force_error = abs(generic["force_square"] - exact["force_square"]) / max(
        abs(exact["force_square"]), EPS
    )
    relative_u_error = abs(generic["U"] - exact["U"]) / max(abs(exact["U"]), EPS)
    if relative_force_error > 0.025 or relative_u_error > 0.025:
        raise AssertionError(
            "generic finite-K linear validation failed: "
            f"force error={relative_force_error:.3g}, U error={relative_u_error:.3g}"
        )
    return {
        "linear_K2_relative_force_error": float(relative_force_error),
        "linear_K2_relative_U_error": float(relative_u_error),
        "linear_K2_prox_residual_max": float(generic["prox_residual_max"]),
    }
