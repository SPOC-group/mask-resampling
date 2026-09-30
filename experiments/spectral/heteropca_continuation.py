"""Continue the original undamped HeteroPCA fit and validate its fixed point."""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from heteroskedastic_pca_estimators import PCAEstimate, _covariance, _leading, _normalize

def relative_change(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b),
                                          np.finfo(float).tiny))


def leading_rank_one(s, diagonal, *, seed, tol, maxiter):
    if len(s) <= 3:
        return _leading(s, seed=seed, which='LM', diagonal=diagonal, tol=tol, maxiter=maxiter)
    correction = diagonal - np.diag(s)
    calls = 0

    def matvec(v):
        nonlocal calls
        calls += 1
        return s @ v + correction*v

    op = LinearOperator(s.shape, matvec=matvec, dtype=np.float64)
    initial = np.random.default_rng(10_000*int(seed)+80).normal(size=len(s))
    values, vectors = eigsh(op, k=1, which='LM', tol=tol, maxiter=maxiter,
                           ncv=min(20, len(s)), v0=initial)
    value = float(values[0])
    if abs(value) <= np.finfo(float).tiny:
        raise ValueError('No identifiable leading direction')
    v = _normalize(vectors[:, 0])
    residual = float(np.linalg.norm(matvec(v)-value*v)/abs(value))
    if not np.isfinite(residual) or residual > max(1e-8, 100*tol):
        raise RuntimeError(f'Eigenpair residual too large: {residual}')
    return v, dict(eigenvalue=value, leading_singular_value=abs(value),
                   eigenpair_relative_residual=residual, matvec_count=calls,
                   spectral_selection='LM')


def resume_heteropca(s, *, initial_diagonal, initial_iteration, seed=0,
                    max_iter=50_000, tol=1e-8, eig_tol=1e-10,
                    eig_maxiter=5000, progress=None):
    """Continue the same fit, stopping only at a reference-validated fixed point."""
    s = _covariance(s)
    diagonal = np.asarray(initial_diagonal, dtype=np.float64).copy()
    if (diagonal.shape != (len(s),) or not np.isfinite(diagonal).all()
            or initial_iteration < 0 or max_iter <= initial_iteration
            or tol <= 0 or eig_tol <= 0 or eig_maxiter < 1):
        raise ValueError('Invalid continuation configuration')
    history, previous_direction = [], None
    calls, max_residual, converged = 0, 0., False
    reference = None
    fixed_residual = np.nan
    agreement = np.nan
    for iteration in range(initial_iteration+1, max_iter+1):
        v, diagnostics = leading_rank_one(s, diagonal, seed=seed, tol=eig_tol,
                                          maxiter=eig_maxiter)
        calls += diagnostics['matvec_count']
        max_residual = max(max_residual, diagnostics['eigenpair_relative_residual'])
        updated = diagnostics['eigenvalue']*v**2
        change = relative_change(updated, diagonal)
        direction_change = (np.nan if previous_direction is None else
                            min(np.linalg.norm(v-previous_direction),
                                np.linalg.norm(v+previous_direction)))
        diagonal = updated
        row = dict(iteration=iteration, diagonal_relative_change=change,
                   direction_change=float(direction_change), **diagnostics)
        if change <= tol:
            # Verify the FINAL updated matrix, with the original k=2 spectral solve.
            ref_v, ref_diag = _leading(s, seed=seed, which='LM', diagonal=diagonal,
                                       tol=eig_tol, maxiter=eig_maxiter)
            calls += ref_diag['matvec_count']
            max_residual = max(max_residual, ref_diag['eigenpair_relative_residual'])
            fixed_residual = relative_change(ref_diag['eigenvalue']*ref_v**2, diagonal)
            agreement = float(min(np.linalg.norm(ref_v-v), np.linalg.norm(ref_v+v)))
            converged = (np.isfinite(fixed_residual) and fixed_residual <= tol
                         and ref_diag['eigenpair_relative_residual'] <= 1e-8
                         and agreement <= max(1e-6, 100*tol))
            row.update(reference_fixed_point_residual=fixed_residual,
                       reference_direction_discrepancy=agreement,
                       reference_validated=converged)
            reference = (ref_v, ref_diag) if converged else None
        history.append(row)
        if progress is not None and (iteration == initial_iteration+1 or iteration % 100 == 0
                                     or converged or iteration == max_iter):
            progress(dict(row), diagonal.copy())
        previous_direction = v
        if converged:
            break
    if reference is None:
        ref_v, ref_diag = _leading(s, seed=seed, which='LM', diagonal=diagonal,
                                   tol=eig_tol, maxiter=eig_maxiter)
        calls += ref_diag['matvec_count']
        max_residual = max(max_residual, ref_diag['eigenpair_relative_residual'])
        fixed_residual = relative_change(ref_diag['eigenvalue']*ref_v**2, diagonal)
        agreement = float(min(np.linalg.norm(ref_v-v), np.linalg.norm(ref_v+v)))
    else:
        ref_v, ref_diag = reference
    ref_diag.update(converged=converged, n_iter=iteration,
                    status='converged' if converged else 'max_iterations',
                    diagonal_relative_change=history[-1]['diagonal_relative_change'],
                    reference_fixed_point_residual=fixed_residual,
                    reference_direction_discrepancy=agreement,
                    reference_validated=converged, matvec_count=calls,
                    max_eigenpair_relative_residual=max_residual,
                    heteropca_tol=tol, heteropca_max_iter=max_iter,
                    continuation_from_iteration=initial_iteration,
                    retry_iterations=len(history), inner_eigenpairs=1)
    return PCAEstimate(ref_v, ref_diag, history, {'imputed_diagonal': diagonal})

