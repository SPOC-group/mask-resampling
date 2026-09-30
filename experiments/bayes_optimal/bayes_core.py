"""Gaussian-prior Bayes state evolution and full-law quadrature routines."""
from __future__ import annotations
import numpy as np
from numpy.polynomial.hermite import hermgauss
from gamma_distributions import bounded_heterogenous_quadrature

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


def alpha_c_BO_from_grid(beta, gamma, weights):
    return float(1.0 / (beta**2 * np.sum(weights / gamma**2)))



def run_bayes_gaussian_prior_se(alpha, beta, gamma, weights, q_lambda_init=1e-3, max_iter=20_000, tol=1e-12, damping=0.2):
    q_lambda = float(q_lambda_init)
    q_u = np.zeros_like(gamma, dtype=float)
    status = 'max_iter'; diff = np.nan; it = -1
    for it in range(max_iter):
        hat_q_u = alpha * beta * q_lambda / gamma
        q_u_raw = hat_q_u / (1.0 + hat_q_u)
        hat_q_lambda = beta * float(np.sum(weights * q_u_raw / gamma))
        q_lambda_raw = hat_q_lambda / (1.0 + hat_q_lambda)
        q_lambda_next = (1.0 - damping) * q_lambda + damping * q_lambda_raw
        q_u_next = (1.0 - damping) * q_u + damping * q_u_raw
        diff = max(abs(q_lambda_next - q_lambda), np.max(np.abs(q_u_next - q_u)))
        q_lambda, q_u = q_lambda_next, q_u_next
        if not np.isfinite(diff):
            status = 'failed: non-finite diff'; break
        if diff < tol:
            status = 'converged'; break
    q_u_mean = float(np.sum(weights * q_u))
    fisher_q_u = float(np.sum(weights * q_u / gamma))
    hat_q_lambda = beta * fisher_q_u
    return {'model': 'bayes_optimal', 'activation': 'bayes', 'alpha': float(alpha), 'beta': float(beta), 'rho': np.nan, 'lambda_r': np.nan, 'q_lambda': float(q_lambda), 'q_u': q_u, 'q_u_mean': q_u_mean, 'fisher_q_u': fisher_q_u, 'hat_q_lambda': float(hat_q_lambda), 'cosine': float(np.sqrt(max(q_u_mean, 0.0))), 'cosine_sq': q_u_mean, 'n_iter': it + 1, 'diff': float(diff) if np.isfinite(diff) else np.nan, 'status': status}



def flatten_result_for_dataframe(res):
    row = {}
    for key, value in res.items():
        if isinstance(value, str): row[key] = value
        elif np.isscalar(value): row[key] = value
    return row



def variance_quadrature(law, order=40):
    if law == 'bounded':
        return bounded_heterogenous_quadrature(n_points=order, a=0.25)
    if law == 'homogeneous':
        return np.ones(1), np.ones(1)
    if law == 'unbounded':
        return generalized_gamma_hermite_grid(s=1.0, eta_n=0.75, eta_h=0.25, order=order)
    raise ValueError(f'Unknown law: {law}')
