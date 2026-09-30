"""Frozen AMP readout with a finite-label Bayesian orientation update.

    head = fit_amp_downstream(amp, labeled.x, labeled.y)
    probabilities = head.predict_proba(test.x)

No teacher vector or true variances are used by the readout. Synthetic
observations and labels can be generated with synthetic_data.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

try:
    from .amp_spike import AMPResult
except ImportError:
    from amp_spike import AMPResult


def _samples(x: Tensor, weights: Tensor) -> Tensor:
    if x.ndim != 2 or x.shape[1] != weights.numel():
        raise ValueError("x must have shape (n_samples, d), matching the AMP weights.")
    x = x.detach().to(device=weights.device, dtype=weights.dtype)
    if not torch.isfinite(x).all().item():
        raise ValueError("Samples must be finite and complete.")
    return x


def _labels(y: Tensor, n: int, like: Tensor) -> Tensor:
    if y.ndim != 1 or y.numel() != n:
        raise ValueError("y must have shape (n_samples,).")
    y = y.to(device=like.device, dtype=like.dtype)
    if not ((y == 1) | (y == -1)).all().item():
        raise ValueError("Labels must be -1 or +1, representing sign(lambda).")
    return y


@dataclass
class AMPDownstream:
    weights: Tensor            # Frozen u_hat / gamma_hat; do not unit-normalize.
    beta: float
    precision: float           # beta * mean(u_hat**2 / gamma_hat).
    lambda_prior: str
    sign_log_odds: float
    n_labels: int

    @torch.no_grad()
    def score(self, x: Tensor) -> Tensor:
        x = _samples(x, self.weights)
        scores = math.sqrt(self.beta / self.weights.numel()) * (x @ self.weights)
        if not torch.isfinite(scores).all().item():
            raise FloatingPointError("Downstream scores overflowed.")
        return scores

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Return P(y=+1), integrating uncertainty in the global sign."""
        b = self.score(x)
        if self.precision == 0:
            return torch.full_like(b, 0.5)
        if self.lambda_prior == "gaussian":
            p_positive_sign = torch.special.ndtr(b / math.sqrt(1 + self.precision))
        elif self.lambda_prior == "rademacher":
            p_positive_sign = torch.sigmoid(2 * b)
        else:
            raise ValueError(f"Unsupported amplitude prior {self.lambda_prior!r}.")
        eta = torch.sigmoid(b.new_tensor(self.sign_log_odds))
        return eta * p_positive_sign + (1 - eta) * (1 - p_positive_sign)

    @torch.no_grad()
    def predict(self, x: Tensor, *, seed: int = 0) -> Tensor:
        """Return +/-1; exact ties use a reproducible fair random choice."""
        p = self.predict_proba(x)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        ties = torch.randint(0, 2, p.shape, generator=generator, device="cpu").to(p.device)
        ties = 2 * ties.to(p.dtype) - 1
        return torch.where(p > 0.5, torch.ones_like(p),
                           torch.where(p < 0.5, -torch.ones_like(p), ties))

    @torch.no_grad()
    def accuracy(self, x: Tensor, y: Tensor) -> float:
        """Test accuracy, with exact probability ties contributing one half."""
        p = self.predict_proba(x)
        y = _labels(y, p.numel(), p)
        if p.numel() == 0:
            raise ValueError("Accuracy requires at least one test sample.")
        predicted = torch.where(p > 0.5, torch.ones_like(p), -torch.ones_like(p))
        correct = (predicted == y).to(p.dtype)
        return torch.where(p == 0.5, torch.full_like(p, 0.5), correct).mean().item()


@torch.no_grad()
def fit_amp_downstream(
    amp: AMPResult,
    x_labeled: Tensor,
    y_labeled: Tensor,
    *,
    allow_unconverged: bool = False,
) -> AMPDownstream:
    """Freeze AMP and learn only its orientation from independent labels.

    Gaussian amplitudes use log Phi(t) - log Phi(-t), with
    t = y*B/sqrt(1+A). Rademacher amplitudes use the exact log odds 2*y*B.
    Empty labeled data are allowed and yield probability 1/2 for every test
    sample. Convergence and plug-in variance calibration do not prove Bayes
    optimality; the likelihood is the matched scalar-channel prescription.
    """
    if not amp.converged and not allow_unconverged:
        raise ValueError("AMP did not converge. Inspect amp.residual or explicitly allow_unconverged.")
    if amp.lambda_prior not in ("gaussian", "rademacher"):
        raise ValueError("Supported amplitude priors are gaussian and rademacher.")
    weights = (amp.w / amp.gamma_hat).detach().clone()
    precision = (amp.beta * (amp.w.square() / amp.gamma_hat).mean()).item()
    if not torch.isfinite(weights).all().item() or not math.isfinite(precision):
        raise FloatingPointError("AMP readout calibration is nonfinite.")
    if precision < 0:
        raise ValueError("The calibrated precision must be nonnegative.")
    x_labeled = _samples(x_labeled, weights)
    y = _labels(y_labeled, x_labeled.shape[0], weights)
    b = math.sqrt(amp.beta / weights.numel()) * (x_labeled @ weights)
    if not torch.isfinite(b).all().item():
        raise FloatingPointError("Labeled scores overflowed.")
    if precision == 0 or y.numel() == 0:
        log_odds = 0.0
    elif amp.lambda_prior == "gaussian":
        t = y * b / math.sqrt(1 + precision)
        # Log-domain CDF avoids taking log(0) for confidently classified labels.
        log_odds = (torch.special.log_ndtr(t) - torch.special.log_ndtr(-t)).sum().item()
    else:
        log_odds = (2 * y * b).sum().item()
    if not math.isfinite(log_odds):
        raise FloatingPointError("Orientation log odds overflowed; use float64.")
    return AMPDownstream(weights, amp.beta, precision, amp.lambda_prior,
                         log_odds, y.numel())

