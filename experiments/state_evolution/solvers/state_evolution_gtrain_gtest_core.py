#!/usr/bin/env python3
"""State-evolution core extracted from the G_train/G_test notebook.

The finite-K channel is exact for linear activation.  The dynamic channel is
the K -> infinity theory and supports linear and ReLU activations.
"""

from __future__ import annotations

from math import erf
from statistics import NormalDist
from typing import Any
import warnings

import numpy as np
from numpy.polynomial.hermite import hermgauss

try:
    from .gamma_distributions import (
        BOUNDED_HETEROGENOUS,
        BOUNDED_HETEROGENOUS_LOW,
        BOUNDED_HETEROGENOUS_MEAN,
        BOUNDED_HETEROGENOUS_SECOND_MOMENT,
        bounded_heterogenous_quadrature,
        canonical_gamma_dist,
    )
except ImportError:  # Direct execution with ``scripts`` on sys.path.
    from gamma_distributions import (
        BOUNDED_HETEROGENOUS,
        BOUNDED_HETEROGENOUS_LOW,
        BOUNDED_HETEROGENOUS_MEAN,
        BOUNDED_HETEROGENOUS_SECOND_MOMENT,
        bounded_heterogenous_quadrature,
        canonical_gamma_dist,
    )

try:
    from scipy.special import ndtr
except ImportError:
    def ndtr(value):
        value = np.asarray(value)
        return 0.5 * (1.0 + np.vectorize(erf)(value / np.sqrt(2.0)))


EPS = 1e-12
SQRT_2PI = np.sqrt(2.0 * np.pi)


def generalized_gamma_grid(
    s: float,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    n_points: int = 400,
    tail_eps: float = 1e-5,
    renormalize_lognormal_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy finite-quantile approximation of the shifted lognormal law.

    This routine truncates the Gaussian quantile range and renormalizes the
    retained lognormal nodes.  It is preserved for reproducing archived runs;
    new production calculations should use
    :func:`generalized_gamma_hermite_grid`.
    """
    if s < 0.0:
        raise ValueError("s must be non-negative")
    if eta_n < 0.0 or eta_h < 0.0:
        raise ValueError("eta_n and eta_h must be non-negative")
    if not 0.0 <= tail_eps < 0.5:
        raise ValueError("tail_eps must satisfy 0 <= tail_eps < 0.5")
    if eta_h == 0.0 or s == 0.0:
        gamma0 = eta_n + eta_h
        if gamma0 <= 0.0:
            raise ValueError("gamma must be strictly positive")
        return np.array([float(gamma0)]), np.array([1.0])
    warnings.warn(
        "generalized_gamma_grid is a truncated legacy approximation; "
        "use generalized_gamma_hermite_grid for the full unbounded law",
        RuntimeWarning,
        stacklevel=2,
    )
    normal = NormalDist()
    probabilities = tail_eps + (1.0 - 2.0 * tail_eps) * (
        np.arange(n_points) + 0.5
    ) / n_points
    standard_normal = np.array(
        [normal.inv_cdf(float(probability)) for probability in probabilities]
    )
    weights = np.ones(n_points, dtype=float) / n_points
    lognormal = np.exp(s * standard_normal - 0.5 * s**2)
    if renormalize_lognormal_mean:
        lognormal /= np.sum(weights * lognormal)
    gamma = eta_n + eta_h * lognormal
    if np.any(gamma <= 0.0):
        raise ValueError("all gamma grid points must be strictly positive")
    return gamma.astype(float), weights.astype(float)


def generalized_gamma_hermite_grid(
    s: float,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    order: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature for the full shifted-lognormal law.

    The quadrature is performed in the underlying standard-normal variable,
    so there is no explicit tail cutoff, clipping, or empirical mean
    renormalization.  The law is unbounded above whenever ``s>0`` and
    ``eta_h>0``; callers must enforce support-level saddle regularity rather
    than checking only the largest quadrature node.
    """
    if s < 0.0:
        raise ValueError("s must be non-negative")
    if eta_n < 0.0 or eta_h < 0.0 or eta_n + eta_h <= 0.0:
        raise ValueError("eta_n and eta_h must be non-negative with positive sum")
    if int(order) < 2:
        raise ValueError("Hermite order must be at least two")
    if eta_h == 0.0 or s == 0.0:
        return np.array([float(eta_n + eta_h)]), np.array([1.0])
    points, raw_weights = hermgauss(int(order))
    standard_normal = np.sqrt(2.0) * points
    weights = raw_weights / np.sqrt(np.pi)
    gamma = eta_n + eta_h * np.exp(
        float(s) * standard_normal - 0.5 * float(s) ** 2
    )
    if np.any(~np.isfinite(gamma)) or np.any(gamma <= 0.0):
        raise FloatingPointError("non-finite or non-positive full-law gamma node")
    return gamma.astype(float), weights.astype(float)


def bounded_heterogenous_gamma_grid(
    order: int = 40, *, gamma_bounded_a: float = BOUNDED_HETEROGENOUS_LOW
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the shared full-law quadrature."""
    return bounded_heterogenous_quadrature(n_points=order, a=gamma_bounded_a)


def gamma_kappa2_exact(
    s: float, eta_n: float = 0.75, eta_h: float = 0.25
) -> float:
    """Exact second moment of the shifted-lognormal variance law."""
    return float(
        eta_n**2 + 2.0 * eta_n * eta_h + eta_h**2 * np.exp(float(s) ** 2)
    )


def resolve_gamma_discretization(gamma_dist: str, requested: str = "auto") -> str:
    """Resolve a distribution-aware integration method.

    ``auto`` deliberately retains the archived bounded-quantile default for
    lognormal noise while selecting full-support Gauss-Legendre integration
    for the bounded Beta law.
    """
    gamma_dist = canonical_gamma_dist(gamma_dist)
    requested = str(requested).strip().lower().replace("-", "_")
    if gamma_dist == "lognormal":
        resolved = "legacy_quantile" if requested == "auto" else requested
        if resolved not in {"legacy_quantile", "gauss_hermite"}:
            raise ValueError(
                "lognormal noise requires legacy_quantile or gauss_hermite"
            )
        return resolved
    if gamma_dist == BOUNDED_HETEROGENOUS:
        resolved = "gauss_legendre" if requested == "auto" else requested
        if resolved != "gauss_legendre":
            raise ValueError(
                "bounded_heterogenous noise requires gauss_legendre integration"
            )
        return resolved
    raise ValueError(f"unknown gamma distribution {gamma_dist!r}")


def build_gamma_quadrature(
    *,
    gamma_dist: str = "lognormal",
    noise_discretization: str = "auto",
    s: float = 1.0,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    n_points: int = 200,
    tail_eps: float = 1e-5,
    hermite_order: int = 40,
    gamma_bounded_a: float = BOUNDED_HETEROGENOUS_LOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Build quadrature nodes and probabilities for a supported variance law."""
    gamma_dist = canonical_gamma_dist(gamma_dist)
    integration = resolve_gamma_discretization(gamma_dist, noise_discretization)
    if gamma_dist == BOUNDED_HETEROGENOUS:
        return bounded_heterogenous_quadrature(n_points=n_points, a=gamma_bounded_a)
    if integration == "gauss_hermite":
        return generalized_gamma_hermite_grid(
            s=s,
            eta_n=eta_n,
            eta_h=eta_h,
            order=hermite_order,
        )
    return generalized_gamma_grid(
        s=s,
        eta_n=eta_n,
        eta_h=eta_h,
        n_points=n_points,
        tail_eps=tail_eps,
    )


def gamma_mean_exact(
    gamma_dist: str,
    *,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
) -> float:
    gamma_dist = canonical_gamma_dist(gamma_dist)
    if gamma_dist == BOUNDED_HETEROGENOUS:
        return BOUNDED_HETEROGENOUS_MEAN
    if gamma_dist == "lognormal":
        return float(eta_n + eta_h)
    raise ValueError(f"unknown gamma distribution {gamma_dist!r}")


def gamma_second_moment_exact(
    gamma_dist: str,
    *,
    s: float = 1.0,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
    gamma_bounded_a: float = BOUNDED_HETEROGENOUS_LOW,
) -> float:
    gamma_dist = canonical_gamma_dist(gamma_dist)
    if gamma_dist == BOUNDED_HETEROGENOUS:
        a = float(gamma_bounded_a)
        if not np.isfinite(a) or not 0 <= a <= 1:
            raise ValueError('gamma_bounded_a must be finite and in [0,1]')
        return float(1.0 + 0.5 * (1.0 - a)**2)
    if gamma_dist == "lognormal":
        return gamma_kappa2_exact(s=s, eta_n=eta_n, eta_h=eta_h)
    raise ValueError(f"unknown gamma distribution {gamma_dist!r}")


def gamma_support_is_unbounded(
    gamma_dist: str,
    *,
    s: float = 1.0,
    eta_h: float = 0.25,
) -> bool:
    gamma_dist = canonical_gamma_dist(gamma_dist)
    if gamma_dist == BOUNDED_HETEROGENOUS:
        return False
    if gamma_dist == "lognormal":
        return bool(s > 0.0 and eta_h > 0.0)
    raise ValueError(f"unknown gamma distribution {gamma_dist!r}")


def gamma_support_regularity_infimum(
    tau: float,
    *,
    s: float,
    eta_n: float = 0.75,
    eta_h: float = 0.25,
) -> float:
    """Return ``inf_gamma (1 + tau*gamma)`` for the actual gamma support."""
    tau = float(tau)
    if not np.isfinite(tau):
        return np.nan
    unbounded_above = float(s) > 0.0 and float(eta_h) > 0.0
    tolerance = 1e-12
    if unbounded_above and tau < -tolerance:
        return -np.inf
    if abs(tau) <= tolerance:
        return 1.0
    gamma_lower = float(eta_n) if unbounded_above else float(eta_n + eta_h)
    return float(1.0 + tau * gamma_lower)


def compute_macros(
    m: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float, float]:
    M = float(np.sum(weights * m))
    Q = float(np.sum(weights * q))
    C = float(np.sum(weights * gamma * q))
    Delta = float(np.sum(weights * gamma * v))
    Ev = float(np.sum(weights * v))
    return M, Q, C, Delta, Ev


def initial_state(
    alpha: float,
    rho: float,
    gamma: np.ndarray,
    weights: np.ndarray,
    init_m_scale: float = 5e-2,
    fallback_Delta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kappa2 = float(np.sum(weights * gamma**2))
    branch_value = 2.0 * alpha * kappa2 - 1.0
    if branch_value > 0.0 and 0.0 < rho < 1.0:
        Delta0 = 1.0 / (
            2.0
            * np.sqrt(max(rho * (1.0 - rho) * branch_value, 1e-12))
        )
    else:
        Delta0 = fallback_Delta
    m0 = init_m_scale * np.ones_like(gamma)
    v0 = Delta0 * np.ones_like(gamma)
    q0 = 4.0 * gamma * Delta0 + m0**2
    return m0, q0, v0


def gh_standard_normal(order: int = 25) -> tuple[np.ndarray, np.ndarray]:
    points, weights = hermgauss(order)
    return (
        (np.sqrt(2.0) * points).astype(float),
        (weights / np.sqrt(np.pi)).astype(float),
    )


def normal_pdf(value):
    return np.exp(-0.5 * np.asarray(value) ** 2) / SQRT_2PI


def relu(value):
    return np.maximum(value, 0.0)


def relu_gaussian_moments(mu, var):
    var = max(float(var), EPS)
    sigma = np.sqrt(var)
    k = mu / sigma
    Phi = ndtr(k)
    pdf = normal_pdf(k)
    first = sigma * pdf + mu * Phi
    second = (mu**2 + var) * Phi + mu * sigma * pdf
    return first, second, Phi, pdf, sigma


def annealed_relu_loss_terms(h, Q: float, C: float, rho: float):
    visible = 1.0 - rho
    variance = max(rho * visible * float(C), EPS)
    mu = visible * h
    first, second, Phi, pdf, sigma = relu_gaussian_moments(mu, variance)
    loss = (rho * Q + 2.0) * second - 2.0 * h * first
    dQ = rho * second
    dC = rho * visible * (
        (rho * Q + 2.0) * Phi - h * pdf / sigma
    )
    d1 = 2.0 * (
        (visible * (rho * Q + 2.0) - 1.0) * first
        - visible * h * Phi
    )
    d2 = 2.0 * visible * (
        (visible * (rho * Q + 2.0) - 2.0) * Phi
        - (visible * h / sigma) * pdf
    )
    return loss, d1, d2, dQ, dC


def annealed_relu_prox(
    h0,
    Q: float,
    C: float,
    Delta: float,
    rho: float,
    grid_size: int = 201,
    newton_steps: int = 20,
):
    h0 = np.asarray(h0, dtype=float)
    scale = np.sqrt(
        max(float(Delta), EPS)
        + max(float(C), 0.0)
        + float(np.mean(h0**2))
        + 1.0
    )
    radius = 8.0 * scale
    grid = np.linspace(-radius, radius, int(grid_size))
    candidates = h0[..., None] + grid
    loss, _, _, _, _ = annealed_relu_loss_terms(
        candidates, Q=Q, C=C, rho=rho
    )
    objective = 0.5 * (candidates - h0[..., None]) ** 2 / Delta + loss
    indices = np.argmin(objective, axis=-1)
    h = np.take_along_axis(candidates, indices[..., None], axis=-1)[..., 0]
    for _ in range(int(newton_steps)):
        _, d1, d2, _, _ = annealed_relu_loss_terms(h, Q=Q, C=C, rho=rho)
        gradient = (h - h0) / Delta + d1
        hessian = 1.0 / Delta + d2
        step = np.where(np.abs(hessian) > 1e-10, gradient / hessian, 0.0)
        step = np.clip(step, -2.0 * scale, 2.0 * scale)
        candidate = h - step
        candidate_loss, _, _, _, _ = annealed_relu_loss_terms(
            candidate, Q=Q, C=C, rho=rho
        )
        candidate_objective = (
            0.5 * (candidate - h0) ** 2 / Delta + candidate_loss
        )
        current_loss, _, _, _, _ = annealed_relu_loss_terms(
            h, Q=Q, C=C, rho=rho
        )
        current_objective = 0.5 * (h - h0) ** 2 / Delta + current_loss
        accept = np.isfinite(candidate_objective) & (
            candidate_objective <= current_objective
        )
        h = np.where(accept, candidate, h)
    return h


def coordinate_update(
    hat_m,
    hat_q,
    hat_v,
    lambda_r: float,
    M: float,
    Q: float,
    C: float,
    Delta: float,
    Ev: float,
    channel: dict[str, Any],
):
    denominator = lambda_r + hat_v
    if np.any(denominator <= 0.0):
        raise FloatingPointError(
            "non-positive coordinate precision: "
            f"min(lambda_r + hat_v)={np.min(denominator):.6e}"
        )
    v_new = 1.0 / denominator
    m_new = v_new * hat_m
    q_new = v_new**2 * (hat_m**2 + hat_q)
    info = {
        "M": M,
        "Q": Q,
        "C": C,
        "Delta": Delta,
        "Ev": Ev,
        "hat_m": float(np.asarray(hat_m).mean()),
        "hat_v_min": float(np.min(hat_v)),
        "hat_v_max": float(np.max(hat_v)),
        **channel,
    }
    return m_new, q_new, v_new, info


def centered_loss_to_gain(centered_loss: float, rho: float) -> float:
    """Return the dimension-free reconstruction gain over a zero decoder."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    return float(-float(centered_loss) / float(rho))


def linear_fresh_mask_centered_loss(
    M: float,
    Q: float,
    C: float,
    beta: float,
    rho: float,
) -> float:
    """Population centered loss on a fresh sample and fresh mask."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    C = max(float(C), 0.0)
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    a_loss = mask_variance * (visible * float(Q) - 2.0)
    b_loss = mask_variance * (hidden * float(Q) + 2.0)
    second_h = float(beta) * float(M) ** 2 + C
    return float(a_loss * second_h + b_loss * C)


def fixed_linear_channel(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
) -> dict[str, Any]:
    """Static K=1 linear channel, retained for endpoint validation."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    C = max(float(C), 0.0)
    visible = 1.0 - rho
    precision = 1.0 / (visible * Delta) + 2.0 * rho * (Q - 2.0 * Delta)
    if precision <= 0.0:
        raise FloatingPointError(
            f"unstable fixed-mask quadratic branch: K={precision:.6e}"
        )
    a = (1.0 / Delta + 2.0 * rho) / precision
    by = (
        np.sqrt(C) / (np.sqrt(visible) * Delta * precision)
        if C > 0.0
        else 0.0
    )
    bt = 2.0 * np.sqrt(rho * C) / precision if C > 0.0 else 0.0
    J = a**2 * beta * M**2 + by**2 + bt**2
    dmu = a - visible
    dy = by - np.sqrt(visible * C)
    force_square = (
        dmu**2 * beta * M**2 + dy**2 + bt**2
    ) / (visible * Delta**2) + 4.0 * rho * J
    curvature = 1.0 / (visible * Delta**2) + 4.0 * rho
    H = 1.0 / (2.0 * Delta) - curvature / (2.0 * precision)
    chi = (a - visible) / Delta + 2.0 * rho * a

    expected_y_t0 = rho * a * beta * M**2
    if C > 0.0:
        expected_y_t0 += bt * np.sqrt(rho * C)
    expected_y_tstar = expected_y_t0 + 2.0 * rho * Delta * J
    train_centered_loss = rho * Q * J - 2.0 * expected_y_tstar
    test_centered_loss = linear_fresh_mask_centered_loss(
        M=M, Q=Q, C=C, beta=beta, rho=rho
    )
    return {
        "K": float(precision),
        "a": float(a),
        "by": float(by),
        "bt": float(bt),
        "J": float(J),
        "G": float(force_square),
        "H": float(H),
        "chi": float(chi),
        "train_centered_loss": float(train_centered_loss),
        "test_centered_loss": float(test_centered_loss),
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
    }


def annealed_linear_channel(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
) -> dict[str, Any]:
    """Dynamic K=infinity linear channel with train/test gains."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    C = max(float(C), 0.0)
    a_loss = rho * (1.0 - rho) * ((1.0 - rho) * Q - 2.0)
    b_loss = rho * (1.0 - rho) * (rho * Q + 2.0)
    denominator = 1.0 + 2.0 * a_loss * Delta
    if denominator <= 0.0:
        raise FloatingPointError(
            f"unstable annealed-mask quadratic branch: D={denominator:.6e}"
        )
    chi = -2.0 * a_loss / denominator
    J = (beta * M**2 + C) / denominator**2
    force_square = chi**2 * (beta * M**2 + C)
    U = rho * (1.0 - rho) * ((1.0 - rho) * J + rho * C)
    H = a_loss / denominator + b_loss
    train_centered_loss = a_loss * J + b_loss * C
    test_centered_loss = linear_fresh_mask_centered_loss(
        M=M, Q=Q, C=C, beta=beta, rho=rho
    )
    return {
        "D": float(denominator),
        "a_loss": float(a_loss),
        "b_loss": float(b_loss),
        "J": float(J),
        "A2": float(force_square),
        "U": float(U),
        "H": float(H),
        "chi": float(chi),
        "train_centered_loss": float(train_centered_loss),
        "test_centered_loss": float(test_centered_loss),
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
    }


def annealed_relu_channel(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
    quad_order: int = 31,
    prox_grid_size: int = 301,
    prox_newton_steps: int = 25,
) -> dict[str, Any]:
    """Dynamic K=infinity ReLU channel with train/test gains."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    C = max(float(C), 0.0)
    sqrt_C = np.sqrt(max(C, EPS))
    z, weights = gh_standard_normal(quad_order)
    latent = z[:, None]
    noise = z[None, :]
    quadrature_weights = weights[:, None] * weights[None, :]
    h0 = np.sqrt(beta) * M * latent + np.sqrt(C) * noise
    h_star = annealed_relu_prox(
        h0,
        Q=Q,
        C=C,
        Delta=Delta,
        rho=rho,
        grid_size=prox_grid_size,
        newton_steps=prox_newton_steps,
    )
    force = (h_star - h0) / Delta
    ell_star, _, _, dQ, dC_loss = annealed_relu_loss_terms(
        h_star, Q=Q, C=C, rho=rho
    )
    ell_test, _, _, _, _ = annealed_relu_loss_terms(
        h0, Q=Q, C=C, rho=rho
    )
    D_C = -force * noise / (2.0 * sqrt_C) + dC_loss
    expectation = lambda value: float(np.sum(quadrature_weights * value))
    expected_latent_force = expectation(latent * force)
    force_square = expectation(force**2)
    U = expectation(dQ)
    H = expectation(D_C)
    J = expectation(relu((1.0 - rho) * h_star) ** 2)
    train_centered_loss = expectation(ell_star)
    test_centered_loss = expectation(ell_test)
    chi = (
        expected_latent_force / (np.sqrt(beta) * M)
        if abs(M) > 1e-12 and beta > 0.0
        else np.nan
    )
    return {
        "J": J,
        "A2": force_square,
        "U": U,
        "H": H,
        "E_lam_A": expected_latent_force,
        "chi": float(chi) if np.isfinite(chi) else np.nan,
        "quad_order": int(quad_order),
        "prox_grid_size": int(prox_grid_size),
        "train_centered_loss": float(train_centered_loss),
        "test_centered_loss": float(test_centered_loss),
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
    }


def finite_k_linear_channel(
    M: float,
    Q: float,
    C: float,
    Delta: float,
    beta: float,
    rho: float,
    mask_count: int,
) -> dict[str, Any]:
    """Exact correlated-mask channel for K quenched views of each sample."""
    try:
        K = int(mask_count)
    except Exception as error:
        raise ValueError("mask_count must be a positive integer") from error
    if K < 1 or float(mask_count) != float(K):
        raise ValueError("mask_count must be a positive integer")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0 < rho < 1")
    if Delta <= 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")

    C = max(float(C), 0.0)
    visible = 1.0 - float(rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    A = mask_variance * (visible * Q - 2.0)
    B = visible - hidden + mask_variance * Q
    Gcoef = hidden * Q + 2.0

    sqrt_c_over_k = np.sqrt(mask_variance / K)
    S_long = 2.0 * np.array(
        [
            [A, B * sqrt_c_over_k],
            [B * sqrt_c_over_k, mask_variance * Gcoef / K],
        ],
        dtype=float,
    )
    P_long = np.eye(2) + Delta * S_long
    eigvals = np.linalg.eigvalsh(P_long)
    s_perp = 2.0 * mask_variance * Gcoef / K
    transverse_denom = 1.0 + Delta * s_perp
    if eigvals[0] <= 0.0 or transverse_denom <= 0.0:
        raise FloatingPointError(
            "unstable finite-K quadratic branch: "
            f"min eig(I+Delta S_long)={eigvals[0]:.6e}, "
            f"1+Delta s_perp={transverse_denom:.6e}"
        )

    R_long = np.linalg.solve(P_long, np.eye(2))
    L_long = -S_long @ R_long
    r_perp = 1.0 / transverse_denom
    l_perp = -s_perp * r_perp
    chi = float(L_long[0, 0])
    signal_second = float(beta) * float(M) ** 2
    force_square = (
        signal_second * float(np.dot(L_long[:, 0], L_long[:, 0]))
        + C * float(np.sum(L_long**2))
        + (K - 1) * C * l_perp**2
    )

    y_direction = np.array([visible, sqrt_c_over_k], dtype=float)
    y_signal_gain = float(y_direction @ R_long[:, 0])
    y_noise_gain = R_long.T @ y_direction
    mean_y2 = (
        signal_second * y_signal_gain**2
        + C * float(np.dot(y_noise_gain, y_noise_gain))
        + (K - 1) * (mask_variance / K) * C * r_perp**2
    )
    U = hidden * mean_y2
    H = -0.5 * (float(np.trace(L_long)) + (K - 1) * l_perp)

    cov_long_0 = np.diag([signal_second + C, C])
    cov_long_star = R_long @ cov_long_0 @ R_long.T
    train_centered_loss = 0.5 * float(np.trace(S_long @ cov_long_star))
    train_centered_loss += 0.5 * (K - 1) * s_perp * r_perp**2 * C
    test_centered_loss = linear_fresh_mask_centered_loss(
        M=M, Q=Q, C=C, beta=beta, rho=rho
    )
    return {
        "mask_count": K,
        "A": float(A),
        "B": float(B),
        "Gcoef": float(Gcoef),
        "S_long": S_long,
        "s_perp": float(s_perp),
        "R_long": R_long,
        "L_long": L_long,
        "r_perp": float(r_perp),
        "l_perp": float(l_perp),
        "J": float(mean_y2),
        "force_square": float(force_square),
        "G": float(force_square),
        "U": float(U),
        "H": float(H),
        "chi": float(chi),
        "longitudinal_min_eig": float(eigvals[0]),
        "transverse_denom": float(transverse_denom),
        "train_centered_loss": float(train_centered_loss),
        "test_centered_loss": float(test_centered_loss),
        "G_train": centered_loss_to_gain(train_centered_loss, rho),
        "G_test": centered_loss_to_gain(test_centered_loss, rho),
    }


def _step_finite_k_linear(
    m: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    lambda_r: float,
    mask_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    M, Q, C, Delta, Ev = compute_macros(m, q, v, gamma, weights)
    channel = finite_k_linear_channel(
        M=M,
        Q=Q,
        C=C,
        Delta=Delta,
        beta=beta,
        rho=rho,
        mask_count=mask_count,
    )
    hat_m = alpha * beta * channel["chi"] * M
    hat_q = alpha * gamma * channel["force_square"]
    hat_v = 2.0 * alpha * (channel["U"] + gamma * channel["H"])
    return coordinate_update(
        hat_m,
        hat_q,
        hat_v,
        lambda_r,
        M,
        Q,
        C,
        Delta,
        Ev,
        channel,
    )


def _step_dynamic(
    m: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    lambda_r: float,
    activation: str,
    quad_order: int,
    prox_grid_size: int,
    prox_newton_steps: int,
    gamma_support: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    M, Q, C, Delta, Ev = compute_macros(m, q, v, gamma, weights)
    if activation == "linear":
        channel = annealed_linear_channel(M, Q, C, Delta, beta, rho)
        hat_m = alpha * beta * channel["chi"] * M
    elif activation == "relu":
        channel = annealed_relu_channel(
            M,
            Q,
            C,
            Delta,
            beta,
            rho,
            quad_order=quad_order,
            prox_grid_size=prox_grid_size,
            prox_newton_steps=prox_newton_steps,
        )
        hat_m = alpha * np.sqrt(beta) * channel["E_lam_A"]
    else:
        raise ValueError("dynamic activation must be 'linear' or 'relu'")
    hat_q = alpha * gamma * channel["A2"]
    hat_v = 2.0 * alpha * (channel["U"] + gamma * channel["H"])
    if gamma_support is not None:
        low, high = gamma_support
        H = float(channel['H'])
        endpoint = low if H >= 0 else high
        precision = float(lambda_r + 2.0 * alpha * (channel['U'] + endpoint * H))
        if not np.isfinite(precision) or precision <= 0:
            raise FloatingPointError('non-positive coordinate precision somewhere on the actual gamma support: '
                                     f'inf={precision:.6e}, support={gamma_support}')
        channel = dict(channel, support_precision_infimum=precision,
                       gamma_support_low=float(low), gamma_support_high=float(high))
    return coordinate_update(
        hat_m,
        hat_q,
        hat_v,
        lambda_r,
        M,
        Q,
        C,
        Delta,
        Ev,
        channel,
    )


def _iterate(
    *,
    step,
    alpha: float,
    beta: float,
    rho: float,
    lambda_r: float,
    gamma: np.ndarray,
    weights: np.ndarray,
    init_state,
    init_m_scale: float,
    max_iter: int,
    tol: float,
    damping: float,
    verbose: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str, int, float]:
    if init_state is None:
        m, q, v = initial_state(
            alpha, rho, gamma, weights, init_m_scale=init_m_scale
        )
    else:
        m, q, v = [np.asarray(value, dtype=float).copy() for value in init_state]
    status = "max_iter"
    difference = np.nan
    info: dict[str, Any] = {}
    iteration = -1
    for iteration in range(int(max_iter)):
        try:
            m_raw, q_raw, v_raw, info = step(m, q, v)
        except Exception as error:
            status = f"failed: {error}"
            break
        m_next = (1.0 - damping) * m + damping * m_raw
        q_next = (1.0 - damping) * q + damping * q_raw
        v_next = (1.0 - damping) * v + damping * v_raw
        difference = max(
            float(np.max(np.abs(m_next - m))),
            float(np.max(np.abs(q_next - q))),
            float(np.max(np.abs(v_next - v))),
        )
        m, q, v = m_next, q_next, v_next
        if verbose and iteration % 500 == 0:
            print(
                label,
                "iter",
                iteration,
                "diff",
                difference,
                "macros",
                compute_macros(m, q, v, gamma, weights),
                flush=True,
            )
        if not np.isfinite(difference):
            status = "failed: non-finite diff"
            break
        if difference < tol:
            status = "converged"
            break
    return m, q, v, info, status, iteration + 1, float(difference)


def _final_scalars(
    *,
    m: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    gamma: np.ndarray,
    weights: np.ndarray,
    info: dict[str, Any],
    reconstruction: dict[str, float],
) -> dict[str, Any]:
    M, Q, C, Delta, Ev = compute_macros(m, q, v, gamma, weights)
    cosine = M / np.sqrt(Q) if Q > 0.0 else np.nan
    multiplier = np.nan
    if abs(M) > 1e-12 and "hat_m" in info:
        multiplier = info["hat_m"] * Ev / M
    scalar_info = {
        key: value
        for key, value in info.items()
        if np.isscalar(value) and key not in {"M", "Q", "C", "Delta", "Ev"}
    }
    return {
        **scalar_info,
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
        "multiplier": float(multiplier) if np.isfinite(multiplier) else np.nan,
        **reconstruction,
        "G_train_minus_G_test": float(
            reconstruction["G_train"] - reconstruction["G_test"]
        ),
        "reconstruction_generalization_gap": float(
            reconstruction["G_train"] - reconstruction["G_test"]
        ),
        "test_minus_train_centered_loss": float(
            reconstruction["test_centered_loss"]
            - reconstruction["train_centered_loss"]
        ),
    }


def run_finite_k_linear_mae_se(
    *,
    alpha: float,
    beta: float,
    rho: float,
    lambda_r: float,
    gamma: np.ndarray,
    weights: np.ndarray,
    mask_count: int,
    init_state=None,
    init_m_scale: float = 5e-2,
    max_iter: int = 20_000,
    tol: float = 1e-10,
    damping: float = 0.05,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one exact finite-K linear fixed point."""
    K = int(mask_count)
    if K < 1 or float(mask_count) != float(K):
        raise ValueError("mask_count must be a positive integer")
    step = lambda m, q, v: _step_finite_k_linear(
        m,
        q,
        v,
        gamma,
        weights,
        alpha,
        beta,
        rho,
        lambda_r,
        K,
    )
    m, q, v, info, status, n_iter, difference = _iterate(
        step=step,
        alpha=alpha,
        beta=beta,
        rho=rho,
        lambda_r=lambda_r,
        gamma=gamma,
        weights=weights,
        init_state=init_state,
        init_m_scale=init_m_scale,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        verbose=verbose,
        label=f"finite_K={K}",
    )
    M, Q, C, Delta, _ = compute_macros(m, q, v, gamma, weights)
    try:
        final_channel = finite_k_linear_channel(
            M=M,
            Q=Q,
            C=C,
            Delta=Delta,
            beta=beta,
            rho=rho,
            mask_count=K,
        )
        reconstruction = {
            key: final_channel[key]
            for key in (
                "train_centered_loss",
                "test_centered_loss",
                "G_train",
                "G_test",
            )
        }
    except Exception:
        reconstruction = {
            "train_centered_loss": np.nan,
            "test_centered_loss": np.nan,
            "G_train": np.nan,
            "G_test": np.nan,
        }
    final = _final_scalars(
        m=m,
        q=q,
        v=v,
        gamma=gamma,
        weights=weights,
        info=info,
        reconstruction=reconstruction,
    )
    return {
        **final,
        "model": f"finite_{K}_mask_MAE",
        "model_family": "finite_K_mask_MAE",
        "mask_count": K,
        "mask_label": "K=1 (static)" if K == 1 else f"K={K}",
        "mask_protocol": "finite quenched masks",
        "activation": "linear",
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "lambda_r": float(lambda_r),
        "n_iter": n_iter,
        "diff": difference if np.isfinite(difference) else np.nan,
        "status": status,
        "m": m,
        "q": q,
        "v": v,
    }


def run_dynamic_mae_se(
    *,
    alpha: float,
    beta: float,
    rho: float,
    lambda_r: float,
    gamma: np.ndarray,
    weights: np.ndarray,
    activation: str = "linear",
    init_state=None,
    init_m_scale: float = 5e-2,
    max_iter: int = 20_000,
    tol: float = 1e-10,
    damping: float = 0.05,
    verbose: bool = False,
    quad_order: int = 31,
    prox_grid_size: int = 301,
    prox_newton_steps: int = 25,
    gamma_support: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run one dynamic (K=infinity) linear or ReLU fixed point."""
    activation = str(activation).lower()
    if activation not in {"linear", "relu"}:
        raise ValueError("activation must be 'linear' or 'relu'")
    if gamma_support is not None:
        low, high = map(float, gamma_support)
        if not np.isfinite(low) or low < 0 or np.isnan(high) or high < low:
            raise ValueError('invalid gamma_support')
        gamma_support = (low, high)
    step = lambda m, q, v: _step_dynamic(
        m,
        q,
        v,
        gamma,
        weights,
        alpha,
        beta,
        rho,
        lambda_r,
        activation,
        quad_order,
        prox_grid_size,
        prox_newton_steps,
        gamma_support,
    )
    m, q, v, info, status, n_iter, difference = _iterate(
        step=step,
        alpha=alpha,
        beta=beta,
        rho=rho,
        lambda_r=lambda_r,
        gamma=gamma,
        weights=weights,
        init_state=init_state,
        init_m_scale=init_m_scale,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
        verbose=verbose,
        label=f"dynamic_{activation}",
    )
    M, Q, C, Delta, _ = compute_macros(m, q, v, gamma, weights)
    try:
        if activation == "linear":
            final_channel = annealed_linear_channel(
                M, Q, C, Delta, beta, rho
            )
        else:
            final_channel = annealed_relu_channel(
                M,
                Q,
                C,
                Delta,
                beta,
                rho,
                quad_order=quad_order,
                prox_grid_size=prox_grid_size,
                prox_newton_steps=prox_newton_steps,
            )
        reconstruction = {
            key: final_channel[key]
            for key in (
                "train_centered_loss",
                "test_centered_loss",
                "G_train",
                "G_test",
            )
        }
    except Exception:
        reconstruction = {
            "train_centered_loss": np.nan,
            "test_centered_loss": np.nan,
            "G_train": np.nan,
            "G_test": np.nan,
        }
    final = _final_scalars(
        m=m,
        q=q,
        v=v,
        gamma=gamma,
        weights=weights,
        info=info,
        reconstruction=reconstruction,
    )
    return {
        **final,
        "model": "annealed_mask_MAE",
        "model_family": "dynamic_mask_MAE",
        "mask_count": np.inf,
        "mask_label": "K->infinity (dynamic)",
        "mask_protocol": "annealed/dynamic",
        "activation": activation,
        "alpha": float(alpha),
        "beta": float(beta),
        "rho": float(rho),
        "lambda_r": float(lambda_r),
        "n_iter": n_iter,
        "diff": difference if np.isfinite(difference) else np.nan,
        "status": status,
        "m": m,
        "q": q,
        "v": v,
    }


def validate_finite_k_linear_channel(
    rtol: float = 1e-10,
    atol: float = 1e-11,
) -> bool:
    """Check K=1 and K->infinity, including G_train and G_test."""
    states = [
        dict(M=0.2, Q=1.3, C=0.8, Delta=0.4, beta=2.0, rho=0.3),
        dict(M=0.5, Q=2.1, C=1.2, Delta=0.2, beta=1.0, rho=0.75),
    ]
    for state in states:
        finite_one = finite_k_linear_channel(mask_count=1, **state)
        fixed = fixed_linear_channel(**state)
        for finite_key, fixed_key in (
            ("J", "J"),
            ("force_square", "G"),
            ("H", "H"),
            ("chi", "chi"),
            ("train_centered_loss", "train_centered_loss"),
            ("test_centered_loss", "test_centered_loss"),
            ("G_train", "G_train"),
            ("G_test", "G_test"),
        ):
            np.testing.assert_allclose(
                finite_one[finite_key],
                fixed[fixed_key],
                rtol=rtol,
                atol=atol,
            )
        finite_large = finite_k_linear_channel(mask_count=1_000_000, **state)
        dynamic = annealed_linear_channel(**state)
        for finite_key, dynamic_key in (
            ("force_square", "A2"),
            ("U", "U"),
            ("H", "H"),
            ("chi", "chi"),
            ("train_centered_loss", "train_centered_loss"),
            ("test_centered_loss", "test_centered_loss"),
            ("G_train", "G_train"),
            ("G_test", "G_test"),
        ):
            np.testing.assert_allclose(
                finite_large[finite_key],
                dynamic[dynamic_key],
                rtol=2e-5,
                atol=2e-7,
            )
    return True


def scalar_result_row(result: dict[str, Any]) -> dict[str, Any]:
    """Drop vector fixed-point profiles before writing a result to CSV."""
    return {
        key: value
        for key, value in result.items()
        if isinstance(value, str) or np.isscalar(value)
    }


__all__ = [
    "bounded_heterogenous_gamma_grid",
    "annealed_linear_channel",
    "annealed_relu_channel",
    "build_gamma_quadrature",
    "centered_loss_to_gain",
    "finite_k_linear_channel",
    "fixed_linear_channel",
    "gamma_mean_exact",
    "gamma_kappa2_exact",
    "gamma_second_moment_exact",
    "gamma_support_is_unbounded",
    "gamma_support_regularity_infimum",
    "generalized_gamma_grid",
    "generalized_gamma_hermite_grid",
    "linear_fresh_mask_centered_loss",
    "resolve_gamma_discretization",
    "run_dynamic_mae_se",
    "run_finite_k_linear_mae_se",
    "scalar_result_row",
    "validate_finite_k_linear_channel",
]
