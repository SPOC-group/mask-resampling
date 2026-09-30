"""Rank-one, data-only PCA baselines for coordinate-heteroskedastic noise.

Diagonal deletion and HeteroPCA follow Zhang, Cai and Wu (2022),
https://arxiv.org/abs/1810.08316, especially Algorithm 1 for HeteroPCA.
This is not an attribution of the invention of diagonal deletion.

Estimated-whitening PCA is our rank-one specialization of the
whitening/unwhitening construction of Leeb and Romanov (2021),
https://arxiv.org/abs/1811.02201, with the diagonal sample-variance estimate
discussed in Section 4.3.1. It is not their full optimal-shrinkage algorithm.
The estimators never receive the planted direction or true noise variances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


@dataclass
class PCAEstimate:
    direction: np.ndarray
    diagnostics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    auxiliary: dict[str, np.ndarray] = field(default_factory=dict)


def _covariance(value: np.ndarray) -> np.ndarray:
    s = np.asarray(value, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or len(s) < 2:
        raise ValueError("covariance must be square with dimension >= 2")
    if not np.isfinite(s).all() or not np.allclose(s, s.T, rtol=1e-12, atol=1e-14):
        raise ValueError("covariance must be finite and symmetric")
    return s


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("no nonzero spectral direction")
    v = v / norm
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def _leading(
    s: np.ndarray, *, seed: int, which: str, diagonal: np.ndarray | None = None,
    scale: np.ndarray | None = None, tol: float = 1e-10, maxiter: int = 5000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetric SVD via largest-magnitude *signed* eigenpairs when which=LM."""
    d = len(s)
    correction = None if diagonal is None else diagonal - np.diag(s)
    calls = 0

    def matvec(v):
        nonlocal calls
        calls += 1
        result = s @ v if scale is None else scale * (s @ (scale * v))
        return result if correction is None else result + correction * v

    if tol <= 0 or maxiter < 1:
        raise ValueError("eigensolver tol and maxiter must be positive")
    if d <= 3:
        matrix = s.copy() if scale is None else scale[:, None] * s * scale[None, :]
        if diagonal is not None:
            np.fill_diagonal(matrix, diagonal)
        values, vectors = np.linalg.eigh(matrix)
    else:
        op = LinearOperator((d, d), matvec=matvec, dtype=np.float64)
        initial = np.random.default_rng(10_000 * int(seed) + 80).normal(size=d)
        values, vectors = eigsh(op, k=2, which=which, tol=tol, maxiter=maxiter,
                               ncv=min(40, d), v0=initial)
    order = np.argsort(np.abs(values) if which == "LM" else values)[::-1]
    values, vectors = values[order], vectors[:, order]
    eigenvalue = float(values[0])
    if abs(eigenvalue) <= np.finfo(float).tiny:
        raise ValueError("zero matrix has no identifiable leading direction")
    v = _normalize(vectors[:, 0])
    residual = float(np.linalg.norm(matvec(v) - eigenvalue * v) / abs(eigenvalue))
    if not np.isfinite(residual) or residual > max(1e-8, 100 * tol):
        raise RuntimeError(f"eigenpair residual too large: {residual}")
    gap = (abs(values[0]) - abs(values[1]) if which == "LM" else values[0] - values[1])
    return v, dict(eigenvalue=eigenvalue, eigenvalue_second=float(values[1]),
                   leading_singular_value=abs(eigenvalue), spectral_gap=float(gap),
                   eigenpair_relative_residual=residual, matvec_count=calls,
                   spectral_selection=which)


def diagonal_deletion_pca(s: np.ndarray, *, seed: int = 0,
                          eig_tol: float = 1e-10, eig_maxiter: int = 5000) -> PCAEstimate:
    """Leading left singular vector after setting the covariance diagonal to zero."""
    s = _covariance(s)
    v, diagnostics = _leading(s, seed=seed, which="LM", diagonal=np.zeros(len(s)),
                              tol=eig_tol, maxiter=eig_maxiter)
    diagnostics.update(converged=True, n_iter=0, status="converged")
    return PCAEstimate(v, diagnostics)


def heteropca(s: np.ndarray, *, seed: int = 0, max_iter: int = 500,
             tol: float = 1e-8, eig_tol: float = 1e-10,
             eig_maxiter: int = 5000,
             progress: Callable[[dict[str, Any]], None] | None = None) -> PCAEstimate:
    """Zhang--Cai--Wu Algorithm 1, r=1, with undamped diagonal imputation.

    N_0=offdiag(S); N_{t+1}=offdiag(S)+diag(lambda_t*v_t*v_t.T).
    lambda_t is the SIGNED eigenvalue of largest absolute magnitude: using
    abs(lambda_t) for diagonal imputation would not be the SVD algorithm.
    Stop on relative diagonal change; return the leading vector of the final
    updated matrix. No previous-alpha direction is used for initialization.
    """
    s = _covariance(s)
    if max_iter < 1 or tol <= 0:
        raise ValueError("HeteroPCA max_iter and tol must be positive")
    diagonal = np.zeros(len(s))
    history = []
    total_calls = 0
    max_residual = 0.0
    previous_direction = None
    converged = False
    for iteration in range(1, max_iter + 1):
        v, diagnostics = _leading(s, seed=seed, which="LM", diagonal=diagonal,
                                  tol=eig_tol, maxiter=eig_maxiter)
        total_calls += diagnostics["matvec_count"]
        max_residual = max(max_residual, diagnostics["eigenpair_relative_residual"])
        updated = diagnostics["eigenvalue"] * v**2
        denominator = max(np.linalg.norm(updated), np.linalg.norm(diagonal),
                          np.finfo(float).tiny)
        change = float(np.linalg.norm(updated - diagonal) / denominator)
        direction_change = (None if previous_direction is None else
                            float(min(np.linalg.norm(v - previous_direction),
                                      np.linalg.norm(v + previous_direction))))
        history.append(dict(iteration=iteration, diagonal_relative_change=change,
                            direction_change=direction_change, **diagnostics))
        if progress is not None and (iteration == 1 or iteration % 25 == 0
                                     or change <= tol or iteration == max_iter):
            progress(history[-1])
        diagonal = updated
        previous_direction = v
        if change <= tol:
            converged = True
            break
    # Algorithm 1 outputs U^(T), not U^(T-1).
    v, diagnostics = _leading(s, seed=seed, which="LM", diagonal=diagonal,
                              tol=eig_tol, maxiter=eig_maxiter)
    total_calls += diagnostics["matvec_count"]
    max_residual = max(max_residual, diagnostics["eigenpair_relative_residual"])
    diagnostics.update(converged=converged, n_iter=len(history),
                       status="converged" if converged else "max_iterations",
                       diagonal_relative_change=history[-1]["diagonal_relative_change"],
                       max_eigenpair_relative_residual=max_residual,
                       matvec_count=total_calls, heteropca_tol=tol, heteropca_max_iter=max_iter)
    return PCAEstimate(v, diagnostics, history, {"imputed_diagonal": diagonal})


def estimated_whitening_pca(s: np.ndarray, *, seed: int = 0,
                            variance_floor: float = 1e-12,
                            eig_tol: float = 1e-10, eig_maxiter: int = 5000) -> PCAEstimate:
    """Estimate Gamma=diag(S), whiten, take PC1, unwhiten, normalize.

    C_w=Gamma_hat^(-1/2) S Gamma_hat^(-1/2),
    v_hat=normalize(Gamma_hat^(1/2) v_w).
    The output is the reconstructed *feature direction*, not the
    inverse-whitened score/readout vector. No spectral shrinkage or threshold
    is applied: a rank-one direction is evaluated even below detectability.
    """
    s = _covariance(s)
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive and finite")
    raw = np.diag(s).copy()
    if np.any(raw < 0):
        raise ValueError("sample variances cannot be negative")
    estimated = np.maximum(raw, variance_floor)
    v_w, diagnostics = _leading(s, seed=seed, which="LA", scale=estimated**-0.5,
                                tol=eig_tol, maxiter=eig_maxiter)
    v = _normalize(estimated**0.5 * v_w)
    diagnostics.update(converged=True, n_iter=0, status="converged",
                       noise_estimator="diag(S)", variance_floor=variance_floor,
                       n_variance_floored=int(np.count_nonzero(raw < variance_floor)),
                       estimated_gamma_min=float(estimated.min()),
                       estimated_gamma_max=float(estimated.max()),
                       eigenpair_coordinates="whitened")
    return PCAEstimate(v, diagnostics, auxiliary={"estimated_gamma": estimated,
                                                "whitened_direction": v_w})


def first_pc(s: np.ndarray, *, seed: int = 0, eig_tol: float = 1e-10,
             eig_maxiter: int = 5000) -> PCAEstimate:
    """Reference ordinary PC1, used to check matching to archived datasets."""
    v, diagnostics = _leading(_covariance(s), seed=seed, which="LA",
                              tol=eig_tol, maxiter=eig_maxiter)
    diagnostics.update(converged=True, n_iter=0, status="converged")
    return PCAEstimate(v, diagnostics)
