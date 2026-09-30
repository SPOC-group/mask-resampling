#!/usr/bin/env python3
"""Shared definitions of coordinate-noise variance distributions.

The same constants and transformations are imported by finite-dimensional ERM
simulations and state-evolution quadratures.  Keeping the law here prevents a
silent theory/simulation mismatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np


BOUNDED_HETEROGENOUS = "bounded_heterogenous"
BOUNDED_HETEROGENOUS_LOW = 0.25
BOUNDED_HETEROGENOUS_SCALE = 2.25
BOUNDED_HETEROGENOUS_BETA_A = 1.0
BOUNDED_HETEROGENOUS_BETA_B = 2.0
BOUNDED_HETEROGENOUS_HIGH = (
    BOUNDED_HETEROGENOUS_LOW + BOUNDED_HETEROGENOUS_SCALE
)
BOUNDED_HETEROGENOUS_MEAN = 1.0
BOUNDED_HETEROGENOUS_SECOND_MOMENT = 1.28125


def canonical_gamma_dist(name: str) -> str:
    """Return the stored distribution name while accepting spelling aliases."""
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "bounded_heterogeneous": BOUNDED_HETEROGENOUS,
        "bounded_heterogenous": BOUNDED_HETEROGENOUS,
    }
    return aliases.get(normalized, normalized)


def sample_bounded_heterogenous(
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    *,
    a: float = BOUNDED_HETEROGENOUS_LOW,
) -> np.ndarray:
    """Draw ``gamma = a + 3*(1-a)*Beta(1,2)``; default preserves the old law."""
    low, scale = _bounded_parameters(a)
    return low + scale * rng.beta(
        BOUNDED_HETEROGENOUS_BETA_A,
        BOUNDED_HETEROGENOUS_BETA_B,
        size=size,
    )


def bounded_heterogenous_quadrature(
    n_points: int = 200,
    *,
    a: float = BOUNDED_HETEROGENOUS_LOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-support Gauss-Legendre rule for the transformed Beta(1, 2) law.

    If ``U ~ Beta(1, 2)``, then its density is ``2(1-U)`` on ``[0,1]``.
    The density is absorbed into the Legendre weights, so there is no tail
    cutoff or truncation.
    """
    low, scale = _bounded_parameters(a)
    n_points = int(n_points)
    if n_points < 2:
        raise ValueError("bounded_heterogenous quadrature needs at least two nodes")
    points, raw_weights = np.polynomial.legendre.leggauss(n_points)
    beta_values = 0.5 * (points + 1.0)
    weights = raw_weights * (1.0 - beta_values)
    weights /= np.sum(weights)
    gamma = low + scale * beta_values
    if np.any(gamma <= 0.0) or np.any(weights <= 0.0):
        raise FloatingPointError("invalid bounded_heterogenous quadrature")
    return gamma.astype(float), weights.astype(float)


def _bounded_parameters(a: float) -> tuple[float, float]:
    a = float(a)
    if not np.isfinite(a) or not 0.0 <= a <= 1.0:
        raise ValueError("bounded_heterogenous a must be finite and in [0,1]")
    return a, 3.0 * (1.0 - a)


def bounded_heterogenous_metadata(
    *, a: float = BOUNDED_HETEROGENOUS_LOW,
) -> dict[str, Any]:
    low, scale = _bounded_parameters(a)
    return {
        "gamma_dist": BOUNDED_HETEROGENOUS,
        "gamma_formula": f"{low:g} + {scale:g} * Beta(1, 2)",
        "gamma_bounded_a": low,
        "gamma_low": low,
        "gamma_high": low + scale,
        "gamma_beta_a": BOUNDED_HETEROGENOUS_BETA_A,
        "gamma_beta_b": BOUNDED_HETEROGENOUS_BETA_B,
        "gamma_mean_exact": BOUNDED_HETEROGENOUS_MEAN,
        "gamma_second_moment_exact": 1.0 + 0.5 * (1.0 - low)**2,
        "physical_gamma_support_unbounded": False,
    }


__all__ = [
    "BOUNDED_HETEROGENOUS",
    "BOUNDED_HETEROGENOUS_BETA_A",
    "BOUNDED_HETEROGENOUS_BETA_B",
    "BOUNDED_HETEROGENOUS_HIGH",
    "BOUNDED_HETEROGENOUS_LOW",
    "BOUNDED_HETEROGENOUS_MEAN",
    "BOUNDED_HETEROGENOUS_SCALE",
    "BOUNDED_HETEROGENOUS_SECOND_MOMENT",
    "bounded_heterogenous_metadata",
    "bounded_heterogenous_quadrature",
    "canonical_gamma_dist",
    "sample_bounded_heterogenous",
]
