#!/usr/bin/env python3
"""Batched PyTorch backend for the exact finite-K ReLU Moreau proximal.

The reference implementation evaluates one Sobol sample at a time in Python.
This module expresses the same active-set/profile calculation as batched tensor
operations.  It therefore runs on either CPU or CUDA without changing the
state-evolution equations.
"""

from __future__ import annotations

import torch


def finite_k_relu_prox_batch_torch(
    g0: torch.Tensor,
    *,
    Q: float,
    Delta: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact finite-K ReLU proximal points for a batch of fields.

    Parameters
    ----------
    g0:
        Tensor with shape ``(n_samples, K + 1)``.  Column zero is the shared
        field and the remaining columns are the K view-specific fields.
    Q, Delta, rho:
        State-evolution scalars.  Double precision is strongly recommended.
    """
    if g0.ndim != 2 or g0.shape[1] < 2:
        raise ValueError("g0 must have shape (n_samples, K+1), K>=1")
    if not g0.is_floating_point():
        raise TypeError("g0 must be floating point")
    if not Delta > 0.0:
        raise FloatingPointError(f"Delta must be positive; got {Delta}")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must satisfy 0<rho<1")

    dtype, device = g0.dtype, g0.device
    K = int(g0.shape[1] - 1)
    x0 = g0[:, 1:]
    h0 = g0[:, 0]
    visible = float(1.0 - rho)
    hidden = float(rho)
    mask_variance = visible * hidden
    root_mask_variance = mask_variance**0.5
    shifted_noise = root_mask_variance * x0

    output_curvature = hidden * float(Q) + 2.0
    linear_h = visible / (Delta * mask_variance) + 2.0 / K
    curvature_y = 1.0 / (Delta * mask_variance) + 2.0 * output_curvature / K

    A_active = visible**2 / (Delta * mask_variance) - linear_h**2 / curvature_y
    B_active = (
        visible * shifted_noise / (Delta * mask_variance)
        - linear_h * (shifted_noise / (Delta * mask_variance)) / curvature_y
    )
    C_active = (
        shifted_noise.square() / (2.0 * Delta * mask_variance)
        - (shifted_noise / (Delta * mask_variance)).square()
        / (2.0 * curvature_y)
    )
    A_zero = visible**2 / (Delta * mask_variance)
    B_zero = visible * shifted_noise / (Delta * mask_variance)
    C_zero = shifted_noise.square() / (2.0 * Delta * mask_variance)

    cavity_zero = -shifted_noise / visible
    active_zero = -(shifted_noise / (Delta * mask_variance)) / linear_h
    ratio = (curvature_y / (Delta * mask_variance)) ** 0.5
    crossing = -shifted_noise * (
        1.0 / (Delta * mask_variance) + ratio
    ) / (linear_h + visible * ratio)
    positive = x0 >= 0.0

    infinity = torch.full_like(x0, torch.inf)
    zeros = torch.zeros_like(x0)
    thresholds = torch.stack(
        [torch.where(positive, cavity_zero, crossing),
         torch.where(positive, active_zero, infinity)],
        dim=2,
    ).reshape(g0.shape[0], 2 * K)
    delta_A = torch.stack(
        [torch.where(positive, torch.full_like(x0, A_zero),
                     torch.full_like(x0, A_active)),
         torch.where(positive, torch.full_like(x0, A_active - A_zero), zeros)],
        dim=2,
    ).reshape(g0.shape[0], 2 * K)
    delta_B = torch.stack(
        [torch.where(positive, B_zero, B_active),
         torch.where(positive, B_active - B_zero, zeros)],
        dim=2,
    ).reshape(g0.shape[0], 2 * K)
    delta_C = torch.stack(
        [torch.where(positive, C_zero, C_active),
         torch.where(positive, C_active - C_zero, zeros)],
        dim=2,
    ).reshape(g0.shape[0], 2 * K)

    thresholds, order = torch.sort(thresholds, dim=1)
    delta_A = torch.gather(delta_A, 1, order)
    delta_B = torch.gather(delta_B, 1, order)
    delta_C = torch.gather(delta_C, 1, order)

    A0 = 1.0 / Delta
    B0 = -h0 / Delta
    C0 = h0.square() / (2.0 * Delta)
    prefix_A = torch.cumsum(delta_A, dim=1) - delta_A
    prefix_B = torch.cumsum(delta_B, dim=1) - delta_B
    prefix_C = torch.cumsum(delta_C, dim=1) - delta_C
    A_before = A0 + prefix_A
    B_before = B0[:, None] + prefix_B
    C_before = C0[:, None] + prefix_C

    minus_infinity = torch.full(
        (g0.shape[0], 1), -torch.inf, dtype=dtype, device=device
    )
    left = torch.cat([minus_infinity, thresholds[:, :-1]], dim=1)
    finite_event = torch.isfinite(thresholds)
    safe_A = torch.where(A_before.abs() > 1e-30, A_before, torch.ones_like(A_before))
    stationary = -B_before / safe_A
    stationary_valid = (
        (A_before > 1e-14)
        & finite_event
        & (stationary > left)
        & (stationary < thresholds)
    )
    stationary_value = (
        0.5 * A_before * stationary.square()
        + B_before * stationary
        + C_before
    )
    stationary_value = torch.where(
        stationary_valid, stationary_value, torch.full_like(stationary_value, torch.inf)
    )

    safe_thresholds = torch.where(finite_event, thresholds, torch.zeros_like(thresholds))
    boundary_value = (
        0.5 * A_before * safe_thresholds.square()
        + B_before * safe_thresholds
        + C_before
    )
    boundary_value = torch.where(
        finite_event, boundary_value, torch.full_like(boundary_value, torch.inf)
    )

    A_tail = A0 + delta_A.sum(dim=1)
    B_tail = B0 + delta_B.sum(dim=1)
    C_tail = C0 + delta_C.sum(dim=1)
    unbounded = (A_tail < -1e-12) | (
        (A_tail.abs() <= 1e-12) & (B_tail < -1e-12)
    )
    if bool(torch.any(unbounded).item()):
        index = int(torch.nonzero(unbounded, as_tuple=False)[0, 0].item())
        raise FloatingPointError(
            "Unbounded finite-K ReLU Moreau profile on the positive-h tail: "
            f"sample={index}, final curvature={A_tail[index].item():.6e}, "
            f"final slope={B_tail[index].item():.6e}"
        )
    finite_thresholds = torch.where(
        finite_event, thresholds, torch.full_like(thresholds, -torch.inf)
    )
    tail_left = finite_thresholds.max(dim=1).values
    safe_A_tail = torch.where(
        A_tail.abs() > 1e-30, A_tail, torch.ones_like(A_tail)
    )
    tail_stationary = -B_tail / safe_A_tail
    tail_valid = (A_tail > 1e-14) & (tail_stationary > tail_left)
    tail_value = (
        0.5 * A_tail * tail_stationary.square()
        + B_tail * tail_stationary
        + C_tail
    )
    tail_value = torch.where(
        tail_valid, tail_value, torch.full_like(tail_value, torch.inf)
    )

    candidate_values = torch.cat(
        [boundary_value, stationary_value, tail_value[:, None]], dim=1
    )
    candidate_h = torch.cat(
        [safe_thresholds, stationary, tail_stationary[:, None]], dim=1
    )
    best_index = torch.argmin(candidate_values, dim=1, keepdim=True)
    best_value = torch.gather(candidate_values, 1, best_index)[:, 0]
    best_h = torch.gather(candidate_h, 1, best_index)[:, 0]
    if bool(torch.any(~torch.isfinite(best_value)).item()):
        raise FloatingPointError("Could not locate a finite finite-K ReLU proximal")

    cavity_y = visible * best_h[:, None] + root_mask_variance * x0
    y_negative = torch.minimum(cavity_y, torch.zeros_like(cavity_y))
    value_negative = (y_negative - cavity_y).square() / (
        2.0 * Delta * mask_variance
    )
    linear_y = cavity_y / (Delta * mask_variance) + 2.0 * best_h[:, None] / K
    y_positive = torch.clamp_min(linear_y / curvature_y, 0.0)
    value_positive = (
        (y_positive - cavity_y).square() / (2.0 * Delta * mask_variance)
        + (output_curvature * y_positive.square()
           - 2.0 * best_h[:, None] * y_positive) / K
    )
    y_star = torch.where(value_positive < value_negative, y_positive, y_negative)
    x_star = (y_star - visible * best_h[:, None]) / root_mask_variance
    return torch.cat([best_h[:, None], x_star], dim=1), best_value


__all__ = ["finite_k_relu_prox_batch_torch"]
