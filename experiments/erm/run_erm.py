#!/usr/bin/env python3
"""Run synthetic masked or unmasked rank-one autoencoder experiments.

Dataset generation, initialization, JAX optimization, reconstruction metrics
and the scalar downstream probe retain the numerical functions used in the
experiments. Only packaging, defaults and output handling are simplified.
See README.md for the paper presets and independent-task commands.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")



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


Tensor = torch.Tensor


@dataclass(frozen=True)
class ExperimentConfig:
    d: int
    alpha: float
    beta: float
    rho: float | None  # None is reserved for mask-free full reconstruction.
    n_test: int | None
    epochs: int
    number_restarts: int
    eval_every: int
    lr: float
    init_std: float
    activation: str
    lambda_prior: str
    u_prior: str
    gamma_dist: str
    gamma_delta: float
    gamma_sigma: float
    gamma_shape: float
    normalize_gamma: bool
    dtype: torch.dtype
    device: torch.device
    gamma_floor: float = 1.0
    gamma_eta: float = 0.25
    normalize_lognormal_multiplier: bool = False
    gamma_low: float = 0.01
    gamma_high: float = 3.0
    gamma_high_probability: float = 99.0 / 299.0
    gamma_bounded_a: float = BOUNDED_HETEROGENOUS_LOW

    @property
    def n_train(self) -> int:
        return int(round(self.alpha * self.d))

    @property
    def resolved_n_test(self) -> int:
        return self.n_train if self.n_test is None else self.n_test


@dataclass
class DataBatch:
    x: Tensor
    hidden_mask: Tensor


@dataclass
class QuenchedSpike:
    u_star: Tensor
    gamma: Tensor


def prior_samples(
    shape: tuple[int, ...],
    prior: str,
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Sample zero-mean, unit-variance priors used for lambda and u_star."""
    if prior == "gaussian":
        return torch.randn(shape, generator=generator, dtype=dtype, device=device)
    if prior == "rademacher":
        bits = torch.randint(0, 2, shape, generator=generator, device=device)
        return (2 * bits.to(dtype) - 1).to(device)
    if prior == "laplace":
        uniform = torch.rand(shape, generator=generator, dtype=dtype, device=device)
        centered = uniform - 0.5
        scale = 1.0 / math.sqrt(2.0)
        return -scale * torch.sign(centered) * torch.log1p(-2.0 * centered.abs())
    raise ValueError(f"unknown prior {prior!r}")


def gamma_samples(
    d: int,
    cfg: ExperimentConfig,
    *,
    generator: torch.Generator,
    seed: int,
) -> Tensor:
    """Sample positive quenched noise variances with population mean one."""
    dtype = cfg.dtype
    device = cfg.device

    gamma_dist = canonical_gamma_dist(cfg.gamma_dist)
    if gamma_dist == "constant":
        gamma = torch.ones(d, dtype=dtype, device=device)
    elif gamma_dist == "two_point":
        if not 0.0 <= cfg.gamma_delta < 1.0:
            raise ValueError("--gamma-delta must lie in [0, 1) for two_point")
        bits = torch.randint(0, 2, (d,), generator=generator, device=device)
        gamma = 1.0 + cfg.gamma_delta * (2 * bits.to(dtype) - 1)
    elif gamma_dist == "binary_mixture":
        if cfg.gamma_low < 0 or cfg.gamma_high < 0:
            raise ValueError("binary-mixture variances must be nonnegative")
        if not 0.0 <= cfg.gamma_high_probability <= 1.0:
            raise ValueError("gamma_high_probability must lie in [0,1]")
        high = torch.rand(d, generator=generator, dtype=dtype, device=device)
        high = high < cfg.gamma_high_probability
        gamma = torch.where(
            high,
            torch.as_tensor(cfg.gamma_high, dtype=dtype, device=device),
            torch.as_tensor(cfg.gamma_low, dtype=dtype, device=device),
        )
    elif gamma_dist == "lognormal":
        if cfg.gamma_sigma < 0:
            raise ValueError("gamma_sigma must be nonnegative for lognormal")
        if cfg.gamma_floor < 0 or cfg.gamma_eta < 0:
            raise ValueError("gamma_floor and gamma_eta must be nonnegative for lognormal")
        if cfg.gamma_floor == 0 and cfg.gamma_eta == 0:
            raise ValueError("at least one of gamma_floor or gamma_eta must be positive")
        z = torch.randn(d, generator=generator, dtype=dtype, device=device)
        ell = torch.exp(cfg.gamma_sigma * z - 0.5 * cfg.gamma_sigma**2)
        if cfg.normalize_lognormal_multiplier:
            ell = ell / ell.mean().clamp_min(torch.finfo(dtype).eps)
        gamma = cfg.gamma_floor + cfg.gamma_eta * ell
    elif gamma_dist == "gamma":
        if cfg.gamma_shape <= 0:
            raise ValueError("--gamma-shape must be positive")
        rng = np.random.default_rng(seed)
        gamma_np = rng.gamma(shape=cfg.gamma_shape, scale=1.0 / cfg.gamma_shape, size=d)
        gamma = torch.as_tensor(gamma_np, dtype=dtype, device=device)
    elif gamma_dist == BOUNDED_HETEROGENOUS:
        rng = np.random.default_rng(seed)
        gamma_np = sample_bounded_heterogenous(rng, size=d, a=cfg.gamma_bounded_a)
        gamma = torch.as_tensor(gamma_np, dtype=dtype, device=device)
    else:
        raise ValueError(f"unknown gamma distribution {gamma_dist!r}")

    if cfg.normalize_gamma:
        gamma = gamma / gamma.mean().clamp_min(torch.finfo(dtype).eps)
    return gamma


def make_quenched_spike(cfg: ExperimentConfig, seed: int) -> QuenchedSpike:
    """Draw the teacher vector and coordinate noise variances."""
    sample_device = torch.device("cpu")
    sample_cfg = replace(cfg, device=sample_device)
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    u_star = prior_samples(
        (cfg.d,),
        cfg.u_prior,
        generator=generator,
        dtype=cfg.dtype,
        device=sample_device,
    )
    gamma = gamma_samples(cfg.d, sample_cfg, generator=generator, seed=seed + 17)
    return QuenchedSpike(u_star=u_star.to(cfg.device), gamma=gamma.to(cfg.device))


def make_spiked_heteroskedastic_data(
    n: int,
    spike: QuenchedSpike,
    cfg: ExperimentConfig,
    seed: int,
) -> DataBatch:
    """Generate samples and iid Bernoulli(rho) hidden masks."""
    sample_device = torch.device("cpu")
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    u_star = spike.u_star.to(sample_device)
    gamma = spike.gamma.to(sample_device)
    lambdas = prior_samples(
        (n, 1),
        cfg.lambda_prior,
        generator=generator,
        dtype=cfg.dtype,
        device=sample_device,
    )
    z = torch.randn(n, cfg.d, generator=generator, dtype=cfg.dtype, device=sample_device)
    signal = math.sqrt(cfg.beta / cfg.d) * lambdas * u_star.view(1, cfg.d)
    noise = gamma.sqrt().view(1, cfg.d) * z
    x = signal + noise
    hidden_mask = torch.bernoulli(
        torch.full((n, cfg.d), cfg.rho, dtype=cfg.dtype, device=sample_device),
        generator=generator,
    )
    return DataBatch(x=x.to(cfg.device), hidden_mask=hidden_mask.to(cfg.device))


def activation_function(name: str) -> Callable[[Tensor], Tensor]:
    if name == "identity":
        return lambda x: x
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "elu":
        return torch.nn.functional.elu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "erf":
        return lambda x: torch.erf(x / math.sqrt(2.0))
    raise ValueError(f"unknown activation {name!r}")


class MaskedRankOneAutoencoder(nn.Module):
    """Tied rank-one masked autoencoder with reconstruction w_i sigma(y)/sqrt(d)."""

    def __init__(
        self,
        d: int,
        *,
        activation: str,
        init_std: float,
        generator: torch.Generator,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        generator_device = getattr(generator, "device", torch.device("cpu"))
        w0 = init_std * torch.randn(d, generator=generator, dtype=dtype, device=generator_device)
        w0 = w0.to(device)
        self.w = nn.Parameter(w0)
        self._activation = activation_function(activation)

    def visible_field(self, x: Tensor, hidden_mask: Tensor) -> Tensor:
        d = x.shape[1]
        visible_mask = 1.0 - hidden_mask
        return (visible_mask * x * self.w.view(1, d)).sum(dim=1) / math.sqrt(d)

    def forward(self, x: Tensor, hidden_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d = x.shape[1]
        y = self.visible_field(x, hidden_mask)
        sigma_y = self._activation(y)
        reconstruction = sigma_y.view(-1, 1) * self.w.view(1, d) / math.sqrt(d)
        return reconstruction, y, sigma_y


def restart_init_seed(seed: int, restart: int) -> int:
    """Initialization seed for a seed/restart pair.

    Restart zero intentionally matches the original single-restart seed.
    """
    return 10_000 * seed + 4 + 1_000_000 * restart


@dataclass
class LabeledBatch:
    x: torch.Tensor
    y: torch.Tensor
    lambdas: torch.Tensor


def make_labeled_spiked_data(
    n: int,
    spike: QuenchedSpike,
    cfg: ExperimentConfig,
    seed: int,
) -> LabeledBatch:
    """Draw fresh samples using the seed's existing teacher and gamma vector."""
    sample_device = torch.device("cpu")
    generator = torch.Generator(device=sample_device).manual_seed(seed)
    u_star = spike.u_star.to(sample_device)
    gamma = spike.gamma.to(sample_device)

    lambdas = prior_samples(
        (n, 1),
        cfg.lambda_prior,
        generator=generator,
        dtype=cfg.dtype,
        device=sample_device,
    )
    z = torch.randn(n, cfg.d, generator=generator, dtype=cfg.dtype, device=sample_device)
    signal = math.sqrt(cfg.beta / cfg.d) * lambdas * u_star.view(1, cfg.d)
    noise = gamma.sqrt().view(1, cfg.d) * z
    x = signal + noise
    y = torch.where(
        lambdas.view(-1) >= 0,
        torch.ones(n, dtype=cfg.dtype),
        -torch.ones(n, dtype=cfg.dtype),
    )
    return LabeledBatch(
        x=x.to(cfg.device),
        y=y.to(cfg.device),
        lambdas=lambdas.view(-1).to(cfg.device),
    )


@torch.no_grad()
def encoder_fields(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the raw scalar encoder preactivation w^T x / sqrt(d)."""
    return x.matmul(w) / math.sqrt(x.shape[1])


def fit_scalar_logistic_probe(
    g_train: torch.Tensor,
    y_train: torch.Tensor,
    g_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    lr: float,
    max_iter: int,
    history_size: int,
) -> dict[str, float | int]:
    """Fit the notebook's scalar coefficient a, with no intercept."""
    g_train = g_train.detach()
    y_train = y_train.detach()
    g_test = g_test.detach()
    y_test = y_test.detach()

    coefficient = torch.zeros((), dtype=g_train.dtype, device=g_train.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [coefficient],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )

    def logistic_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(-labels * coefficient * features).mean()

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = logistic_loss(g_train, y_train)
        loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state.get(coefficient, {})

    with torch.no_grad():
        train_logits = coefficient * g_train
        test_logits = coefficient * g_test
        train_pred = torch.where(train_logits >= 0, torch.ones_like(y_train), -torch.ones_like(y_train))
        test_pred = torch.where(test_logits >= 0, torch.ones_like(y_test), -torch.ones_like(y_test))
        return {
            "probe_a": float(coefficient.detach().cpu()),
            "probe_train_loss": float(logistic_loss(g_train, y_train).detach().cpu()),
            "probe_test_loss": float(logistic_loss(g_test, y_test).detach().cpu()),
            "probe_train_accuracy": float(
                (train_pred == y_train).to(torch.float32).mean().detach().cpu()
            ),
            "probe_test_accuracy": float(
                (test_pred == y_test).to(torch.float32).mean().detach().cpu()
            ),
            "probe_train_margin_mean": float((y_train * train_logits).mean().detach().cpu()),
            "probe_test_margin_mean": float((y_test * test_logits).mean().detach().cpu()),
            "probe_lbfgs_n_iter": int(state.get("n_iter", 0)),
            "probe_lbfgs_func_evals": int(state.get("func_evals", 0)),
        }


def resampled_mask_seed(seed: int) -> int:
    """Training-mask RNG seed for a data seed.

    All restarts for the same seed use the same fresh-mask sequence; restarts
    differ only through their weight initialization, matching the fixed-mask
    restart convention.
    """
    return 10_000 * seed + 5


def make_mask_generator(device: torch.device, seed: int) -> torch.Generator:
    """Create a torch Generator on the training device when supported."""
    device = torch.device(device)
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(seed)
    return torch.Generator(device=torch.device("cpu")).manual_seed(seed)


def appendix_observables(row):
    full = row.get("reconstruction_mode") == "full"
    gain_scale = 1.0 if full else row["rho"]
    result={"optimizer_updates":row["epoch"],"realized_alpha":row["n_train"]/row["d"],
            "gain_normalization": gain_scale}
    for split in ("train","test"):
        count=row.get(f"{split}_target_count", row.get(f"{split}_hidden_count"))
        baseline=row[f"{split}_zero_decoder_sse"]
        denominator=row[f"{split}_sample_view_count"]
        raw=row[f"pretrain_{split}_loss"]
        centered=(raw*count-baseline)/denominator
        result.update({
            f"{split}_loss_per_sample":raw*count/denominator,
            f"{split}_zero_decoder_per_sample":baseline/denominator,
            f"{split}_zero_decoder_mse":baseline/count,
            f"{split}_centered_loss_per_sample":centered,
            f"{split}_gain":-centered/gain_scale,
            f"{split}_relative_gain":row["d"]*(1-raw*count/baseline)})
    ridge=row["lambda_r"]*row["d"]*row["R"]/(2*row["n_train"])
    result.update(ridge_penalty_per_sample=ridge,
        regularized_train_objective_per_sample=result["train_loss_per_sample"]+ridge,
        regularized_centered_train_objective_per_sample=result["train_centered_loss_per_sample"]+ridge)
    return result


ACTIVATION_NAMES = {
    "linear": "identity",
    "relu": "relu",
    "elu": "elu",
    "tanh": "tanh",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if "control_root" in config:
        root = path.resolve().parent
        config["control_root"] = str(root / config["control_root"])
        config["output_dir"] = str(root / config["output_dir"])
    return config


def extra_view_mask_seed(seed: int) -> int:
    return 10_000 * int(seed) + 6


def task_dir(config: dict[str, Any], task_id: int) -> Path:
    return Path(config["control_root"]) / "tasks" / f"task_{task_id:04d}"


def condition_dir(output: Path, k_label: str | int) -> Path:
    return output / f"condition_{k_label}"


def finite_condition(k_views: int) -> dict[str, Any]:
    return {
        "k_views": int(k_views),
        "k_label": str(int(k_views)),
        "masking_protocol": "finite_k_fixed",
    }


def dynamic_condition() -> dict[str, Any]:
    return {
        "k_views": 0,
        "k_label": "dynamic",
        "masking_protocol": "dynamic",
    }


def full_condition() -> dict[str, Any]:
    return {"k_views": 0, "k_label": "full", "masking_protocol": "full_reconstruction"}


def full_base_and_count(x: torch.Tensor, row_chunk_size: int) -> tuple[float, int]:
    """All-coordinate target norm/count, without a mask or reconstruction tensor."""
    baseline = sum(float(x[start:start + row_chunk_size].double().square().sum())
                   for start in range(0, len(x), row_chunk_size))
    return baseline, x.numel()


def experiment_config(
    config: dict[str, Any], alpha: float, activation: str, rho: float | None = None
) -> ExperimentConfig:
    return ExperimentConfig(
        d=int(config["d"]),
        alpha=float(alpha),
        beta=float(config["beta"]),
        rho=(None if config.get("reconstruction_mode", "masked") == "full"
             else float(config["rho"] if rho is None else rho)),
        n_test=None,
        epochs=int(config["epochs"]),
        number_restarts=1,
        eval_every=int(config["eval_every"]),
        lr=float(config["learning_rate"]),
        init_std=float(config["init_std"]),
        activation=ACTIVATION_NAMES[activation],
        lambda_prior="gaussian",
        u_prior="gaussian",
        gamma_dist=str(config.get("gamma_dist", "lognormal")),
        gamma_delta=0.5,
        gamma_sigma=float(config["gamma_sigma"]),
        gamma_shape=2.0,
        normalize_gamma=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
        gamma_floor=float(config["gamma_floor"]),
        gamma_eta=float(config["gamma_eta"]),
        normalize_lognormal_multiplier=False,
        gamma_bounded_a=float(config.get("gamma_bounded_a", BOUNDED_HETEROGENOUS_LOW)),
    )


def prepare(args: argparse.Namespace) -> None:
    full = getattr(args, "reconstruction_mode", "masked") == "full"
    if not np.isfinite(args.lambda_r) or args.lambda_r < 0:
        raise ValueError("lambda-r must be finite and nonnegative")
    alphas = tuple(sorted(set(float(value) for value in args.alphas)))
    rhos = tuple() if full else tuple(
        sorted(
            set(
                float(value)
                for value in (args.rhos if args.rhos is not None else [args.rho])
            )
        )
    )
    if full:
        rhos = (None,)  # A single unmasked condition; rho is inapplicable.
    seeds = tuple(range(int(args.seed_start), int(args.seed_start) + int(args.n_seeds)))
    activations = tuple(dict.fromkeys(args.activations))
    k_values = (
        tuple()
        if args.dynamic_only or full
        else tuple(sorted(set(int(value) for value in args.k_values)))
    )
    include_dynamic = not full and bool(args.include_dynamic or args.dynamic_only)
    gamma_dist = canonical_gamma_dist(args.gamma_dist)
    if not alphas or any(not np.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("all alpha values must be finite and positive")
    if args.d <= 0 or not np.isfinite(args.beta) or args.beta < 0 or (not full and any(not 0 < value < 1 for value in rhos)):
        raise ValueError("d must be positive, beta nonnegative, and every rho in (0,1)")
    if len(alphas) > 1 and len(rhos) > 1:
        raise ValueError("sweep alpha or rho, not both in the same run")
    if args.n_seeds <= 0 or args.epochs < 0 or args.eval_every <= 0:
        raise ValueError("seed/evaluation counts must be positive and epochs nonnegative")
    if args.row_chunk_size <= 0:
        raise ValueError("row chunk size must be positive")
    if not activations or any(value not in ACTIVATION_NAMES for value in activations):
        raise ValueError(f"activations must be chosen from {sorted(ACTIVATION_NAMES)}")
    if any(value < 1 for value in k_values):
        raise ValueError("all K values must be positive")
    if not k_values and not include_dynamic and not full:
        raise ValueError("request at least one finite K or the dynamic condition")
    conditions = [full_condition()] if full else [finite_condition(value) for value in k_values]
    if include_dynamic:
        conditions.append(dynamic_condition())

    tasks: list[dict[str, Any]] = []
    for seed in seeds:
        for alpha in alphas:
            for rho in rhos:
                for activation in activations:
                    tasks.append(
                        {
                            "task_id": len(tasks),
                            "seed": int(seed),
                            "alpha": float(alpha),
                            "rho": None if full else float(rho),
                            "n_train": int(round(alpha * args.d)),
                            "activation": activation,
                        }
                    )

    config_path = args.config
    control_root = config_path.parent
    output_dir = control_root / "results"
    if config_path.exists() and not args.force:
        raise FileExistsError(config_path)
    sweep_axis = "rho" if len(rhos) > 1 else "alpha"
    if (output_dir / f"k_{sweep_axis}_summary.csv").exists() and not args.force:
        raise FileExistsError(output_dir)

    config = {
        "format_version": 1,
        "training_backend": "jax",
        "save_weights": False,
        "save_data": False,
        "control_name": "full_reconstruction_alpha" if full else f"boolean_mask_bank_finite_k_{sweep_axis}",
        "reconstruction_mode": "full" if full else "masked",
        "control_root": ".",
        "output_dir": "results",
        "d": int(args.d),
        "beta": float(args.beta),
        "rho": None if full else float(rhos[0]),
        "rhos": [] if full else list(rhos),
        "gamma_dist": gamma_dist,
        "gamma_sigma": float(args.gamma_sigma),
        "gamma_floor": float(args.gamma_floor),
        "gamma_eta": float(args.gamma_eta),
        "alphas": list(alphas),
        "sweep_axis": sweep_axis,
        "seeds": list(seeds),
        "activations": list(activations),
        "k_values": list(k_values),
        "include_dynamic": include_dynamic,
        "dynamic_only": not full and bool(args.dynamic_only),
        "conditions": conditions,
        "epochs": int(args.epochs),
        "lambda_r": float(args.lambda_r),
        "eval_every": int(args.eval_every),
        "learning_rate": float(args.learning_rate),
        "init_std": float(args.init_std),
        "row_chunk_size": int(args.row_chunk_size),
        "n_ds": int(args.n_ds),
        "probe_n_test": int(args.probe_n_test),
        "probe_lbfgs_lr": float(args.probe_lbfgs_lr),
        "probe_lbfgs_max_iter": int(args.probe_lbfgs_max_iter),
        "probe_lbfgs_history_size": int(args.probe_lbfgs_history_size),
        "downstream_feature": "raw_encoder_preactivation",
        "warm_start": False,
        "number_restarts": 1,
        "dataset_storage": "one shared torch.float32 matrix per train/test set",
        "mask_storage": "none" if full else "separate nested torch.bool bank; one byte per entry",
        "optimization": (
            "exact algebraic rank-one loss; view/row gradients accumulated; "
            "one Adam step per epoch"
        ),
        "tasks": tasks,
        "task_count": len(tasks),
        "fit_count": len(tasks) * len(conditions),
    }
    if gamma_dist == BOUNDED_HETEROGENOUS:
        config.update(bounded_heterogenous_metadata())
    (control_root / "tasks").mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    print(
        json.dumps(
            {
                "config": str(config_path),
                "task_count": len(tasks),
                "fit_count": config["fit_count"],
                "output_dir": "results",
            },
            sort_keys=True,
        )
    )


def bool_mask_bank(
    first_mask: torch.Tensor,
    max_k: int,
    rho: float,
    seed: int,
) -> list[torch.Tensor]:
    """Build a nested fixed-mask bank, retaining masks only as Boolean tensors."""
    masks = [first_mask.to(dtype=torch.bool)]
    generator = make_mask_generator(first_mask.device, extra_view_mask_seed(seed))
    for _ in range(1, max_k):
        uniform = torch.rand(
            first_mask.shape,
            generator=generator,
            dtype=torch.float32,
            device=first_mask.device,
        )
        masks.append(uniform.lt(rho))
        del uniform
    if any(mask.dtype != torch.bool for mask in masks):
        raise AssertionError("mask bank is not Boolean")
    return masks


def row_slices(n_rows: int, chunk_size: int):
    for start in range(0, n_rows, chunk_size):
        yield slice(start, min(start + chunk_size, n_rows))


def variable_loss_numerator(
    w: torch.Tensor,
    x: torch.Tensor,
    hidden_mask: torch.Tensor,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Parameter-dependent SSE for one row chunk and one hidden mask.

    If a=sigma(sum_i (1-B_i)x_i w_i/sqrt(d)), the omitted constant is
    sum_i B_i x_i^2 and the returned expression is
    sum_mu [-2 a_mu sum_i B_i x_i w_i/sqrt(d)
            +a_mu^2 sum_i B_i w_i^2/d].
    """
    if hidden_mask.dtype != torch.bool:
        raise TypeError("hidden_mask must use torch.bool storage")
    d = x.shape[1]
    sqrt_d = math.sqrt(d)
    hidden_x = x.masked_fill(~hidden_mask, 0.0)
    hidden_projection = hidden_x.matmul(w)
    visible_field = (x.matmul(w) - hidden_projection) / sqrt_d
    activated = activation(visible_field)
    hidden_w2 = hidden_mask.to(dtype=x.dtype).matmul(w.square())
    return (
        -2.0 * (activated * hidden_projection).sum() / sqrt_d
        + (activated.square() * hidden_w2).sum() / d
    )


@torch.no_grad()
def mask_base_and_count(
    x: torch.Tensor, hidden_mask: torch.Tensor, chunk_size: int
) -> tuple[float, int]:
    base = 0.0
    count = 0
    for rows in row_slices(x.shape[0], chunk_size):
        x_chunk = x[rows]
        mask_chunk = hidden_mask[rows]
        base += float(x_chunk.square().masked_select(mask_chunk).sum().item())
        count += int(mask_chunk.sum().item())
    return base, count


def bank_variable_numerator(
    w: torch.Tensor,
    x: torch.Tensor,
    masks: list[torch.Tensor],
    activation: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: int,
) -> torch.Tensor:
    total = w.new_zeros(())
    for mask in masks:
        for rows in row_slices(x.shape[0], chunk_size):
            total = total + variable_loss_numerator(
                w, x[rows], mask[rows], activation
            )
    return total


@torch.no_grad()
def raw_bank_loss(
    w: torch.Tensor,
    x: torch.Tensor,
    masks: list[torch.Tensor],
    bases: list[float],
    counts: list[int],
    activation: Callable[[torch.Tensor], torch.Tensor],
    chunk_size: int,
) -> float:
    variable = bank_variable_numerator(
        w, x, masks, activation, chunk_size
    )
    return (sum(bases) + float(variable.item())) / max(sum(counts), 1)


def representation_metrics(
    w: torch.Tensor, spike: Any, epoch: int, seed: int
) -> dict[str, float | int]:
    with torch.no_grad():
        detached = w.detach()
        u = spike.u_star
        gamma = spike.gamma
        cosine_sq = (
            torch.dot(detached, u).square()
            / (detached.square().sum() * u.square().sum()).clamp_min(1e-30)
        ).item()
        return {
            "seed": seed,
            "restart": 0,
            "epoch": epoch,
            "cosine_sq": cosine_sq,
            "cosine_abs": math.sqrt(max(0.0, cosine_sq)),
            "overlap": (torch.dot(detached, u) / detached.numel()).item(),
            "R": detached.square().mean().item(),
            "S_gamma": (gamma * detached.square()).mean().item(),
            "w_norm": detached.norm().item(),
            "u_norm": u.norm().item(),
            "gamma_mean": gamma.mean().item(),
            "gamma_second_moment": gamma.square().mean().item(),
        }


def storage_metadata(
    x: torch.Tensor,
    masks: list[torch.Tensor],
    test_x: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict[str, float | int | str]:
    data_bytes = x.numel() * x.element_size()
    mask_bytes = sum(mask.numel() * mask.element_size() for mask in masks)
    test_data_bytes = test_x.numel() * test_x.element_size()
    test_mask_bytes = test_mask.numel() * test_mask.element_size()
    float_mask_viewwise_bytes = data_bytes + sum(
        mask.numel() * x.element_size() for mask in masks
    )
    fully_repeated_bytes = len(masks) * data_bytes + sum(
        mask.numel() * x.element_size() for mask in masks
    )
    persistent_bytes = data_bytes + mask_bytes
    return {
        "dataset_dtype": str(x.dtype),
        "mask_dtype": str(masks[0].dtype),
        "train_dataset_storage_bytes": data_bytes,
        "train_mask_bank_storage_bytes": mask_bytes,
        "persistent_train_storage_bytes": persistent_bytes,
        "test_dataset_storage_bytes": test_data_bytes,
        "test_mask_storage_bytes": test_mask_bytes,
        "float_mask_viewwise_storage_bytes": float_mask_viewwise_bytes,
        "fully_repeated_nk_storage_bytes": fully_repeated_bytes,
        "saving_vs_float_mask_viewwise_fraction": (
            1.0 - persistent_bytes / float_mask_viewwise_bytes
        ),
        "saving_vs_fully_repeated_nk_fraction": (
            1.0 - persistent_bytes / fully_repeated_bytes
        ),
    }


def run_task(args: argparse.Namespace, train_fn=None) -> None:
    config = load_json(args.config.resolve())
    full = config.get("reconstruction_mode", "masked") == "full"
    if train_fn is None:
        raise ValueError("The task command must select a JAX device.")
    task_id = int(args.task_id)
    if not 0 <= task_id < int(config["task_count"]):
        raise IndexError(task_id)
    task = dict(config["tasks"][task_id])
    if int(task["task_id"]) != task_id:
        raise ValueError("task ordering is inconsistent")
    output = task_dir(config, task_id)
    completed_path = output / "COMPLETED.json"
    if completed_path.exists() and not args.force:
        print(f"task {task_id} already complete")
        return
    output.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    seed = int(task["seed"])
    cfg = experiment_config(
        config,
        float(task["alpha"]),
        str(task["activation"]),
        None if full else float(task.get("rho", config["rho"])),
    )
    spike = make_quenched_spike(cfg, seed=10_000 * seed + 1)
    k_values = [int(value) for value in config["k_values"]]
    conditions = list(config.get("conditions", [finite_condition(value) for value in k_values]))
    if full:
        if conditions != [full_condition()]:
            raise ValueError("Full reconstruction must have one full condition, without K or dynamic masks.")
        # Same latent/noise draws as the masked generator, but never draw masks.
        train_x = make_labeled_spiked_data(cfg.n_train, spike, cfg, seed=10_000 * seed + 2).x
        test_x = make_labeled_spiked_data(cfg.resolved_n_test, spike, cfg, seed=10_000 * seed + 3).x
        mask_bank = test_mask = None
        train_base, train_count = full_base_and_count(train_x, int(config["row_chunk_size"]))
        train_bases, train_counts = [train_base], [train_count]
        test_base, test_count = full_base_and_count(test_x, int(config["row_chunk_size"]))
    else:
        raw_train = make_spiked_heteroskedastic_data(
            cfg.n_train, spike, cfg, seed=10_000 * seed + 2)
        raw_test = make_spiked_heteroskedastic_data(
            cfg.resolved_n_test, spike, cfg, seed=10_000 * seed + 3)
        train_x, test_x = raw_train.x, raw_test.x
        mask_bank = bool_mask_bank(raw_train.hidden_mask, max(k_values, default=1), cfg.rho, seed)
        test_mask = raw_test.hidden_mask.to(dtype=torch.bool)
        del raw_train, raw_test
        train_stats = [mask_base_and_count(train_x, mask, int(config["row_chunk_size"]))
                       for mask in mask_bank]
        train_bases = [value[0] for value in train_stats]
        train_counts = [value[1] for value in train_stats]
        test_base, test_count = mask_base_and_count(test_x, test_mask, int(config["row_chunk_size"]))
    downstream_train = make_labeled_spiked_data(
        int(config["n_ds"]), spike, cfg, seed=10_000 * seed + 20
    )
    downstream_test = make_labeled_spiked_data(
        int(config["probe_n_test"]), spike, cfg, seed=10_000 * seed + 21
    )

    completed_conditions = 0
    for condition in conditions:
        k_views = int(condition["k_views"])
        k_label = str(condition["k_label"])
        condition_output = condition_dir(output, k_label)
        condition_completed = condition_output / "COMPLETED.json"
        if condition_completed.exists() and not args.force:
            completed_conditions += 1
            continue
        condition_output.mkdir(parents=True, exist_ok=True)
        trajectory, final, weight = train_fn(
            config,
            task,
            cfg,
            spike,
            train_x,
            test_x,
            mask_bank,
            train_bases,
            train_counts,
            test_mask,
            test_base,
            test_count,
            downstream_train,
            downstream_test,
            condition,
        )
        trajectory.to_csv(condition_output / "trajectory.csv", index=False)
        pd.DataFrame([final]).to_csv(condition_output / "final.csv", index=False)
        write_json(
            condition_completed,
            {
                "task_id": task_id,
                "seed": seed,
                "alpha": float(task["alpha"]),
                "activation": str(task["activation"]),
                "k_views": k_views,
                "k_label": k_label,
                "masking_protocol": condition["masking_protocol"],
                "mask_dtype": "none" if full else "torch.bool",
            },
        )
        completed_conditions += 1
        print(
            f"task={task_id} activation={task['activation']} "
            f"alpha={task['alpha']:g} seed={seed} K={k_label} "
            f"cos2={final['cosine_sq']:.6g} "
            f"accuracy={final['probe_test_accuracy']:.6g} "
            f"grad={final['final_gradient_norm']:.3g}",
            flush=True,
        )

    write_json(
        completed_path,
        {
            **task,
            "condition_count": completed_conditions,
        },
    )


def aggregate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = [
        "cosine_sq",
        "cosine_abs",
        "probe_test_accuracy",
        "probe_train_accuracy",
        "pretrain_train_loss",
        "pretrain_test_loss",
        "overlap",
        "R",
        "S_gamma",
        "final_gradient_norm",
        "last_step_norm",
        "last_step_relative_norm",
    ]
    # Preserve the existing schema and aggregate the recorded appendix observables too.
    values += [name for name in (
        "train_loss_per_sample", "test_loss_per_sample",
        "train_centered_loss_per_sample", "test_centered_loss_per_sample",
        "train_gain", "test_gain", "train_relative_gain", "test_relative_gain",
        "ridge_penalty_per_sample", "regularized_centered_train_objective_per_sample",
        "probe_test_loss", "probe_train_loss",
    ) if name in frame]
    named: dict[str, tuple[str, str]] = {"num_seeds": ("seed", "nunique")}
    for value in values:
        named[f"{value}_mean"] = (value, "mean")
        named[f"{value}_std"] = (value, "std")
        named[f"{value}_sem"] = (value, "sem")
    return (
        frame.groupby(
            [
                "activation",
                "activation_name",
                "masking_protocol",
                "k_views",
                "k_label",
                "alpha",
                "rho",
            ],
            as_index=False, dropna=False,
        )
        .agg(**named)
        .sort_values(["activation", "alpha", "rho", "k_views"])
    )


def aggregate(args: argparse.Namespace) -> None:
    config = load_json(args.config.resolve())
    missing: list[str] = []
    trajectories: list[pd.DataFrame] = []
    finals: list[pd.DataFrame] = []
    for task in config["tasks"]:
        output = task_dir(config, int(task["task_id"]))
        conditions = list(
            config.get(
                "conditions",
                [finite_condition(int(value)) for value in config["k_values"]],
            )
        )
        for condition in conditions:
            k_views = int(condition["k_views"])
            k_label = str(condition["k_label"])
            directory = condition_dir(output, k_label)
            if not (directory / "COMPLETED.json").exists():
                missing.append(f"{task['task_id']}:{k_label}")
                continue
            trajectories.append(pd.read_csv(directory / "trajectory.csv"))
            finals.append(pd.read_csv(directory / "final.csv"))
    if missing:
        raise RuntimeError(f"{len(missing)} conditions incomplete: {missing[:20]}")

    trajectory = pd.concat(trajectories, ignore_index=True)
    final = pd.concat(finals, ignore_index=True)
    expected = int(config["fit_count"])
    if len(final) != expected:
        raise ValueError(f"final table has {len(final)} rows; expected {expected}")
    keys = ["activation", "alpha", "rho", "k_views", "seed"]
    if final.duplicated(keys).any():
        raise ValueError("duplicate activation/alpha/K/seed rows")
    expected_mask_dtype = "none" if config.get("reconstruction_mode") == "full" else "torch.bool"
    if not final["mask_dtype"].eq(expected_mask_dtype).all():
        raise ValueError("a result used incompatible mask storage")
    summary = aggregate_frame(final)
    if not summary["num_seeds"].eq(len(config["seeds"])).all():
        raise ValueError("not every aggregate contains all seeds")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_axis = str(config.get("sweep_axis", "alpha"))
    output_stem = "full_reconstruction_alpha" if config.get("reconstruction_mode") == "full" else f"k_{sweep_axis}"
    trajectory.to_csv(output_dir / f"{output_stem}_trajectories.csv", index=False)
    final.to_csv(output_dir / f"{output_stem}_per_seed.csv", index=False)
    summary.to_csv(output_dir / f"{output_stem}_summary.csv", index=False)
    public_config = {key: value for key, value in config.items()
                     if key not in ("control_root", "output_dir")}
    public_config["validated_fits"] = len(final)
    write_json(output_dir / "run_config.json", public_config)
    print(f"Aggregated and validated {len(final)} fits")


@torch.no_grad()
def raw_full_loss(weight, x, baseline, activation, row_chunk_size):
    """Full SSE per coordinate, accumulated in float64 without n-by-d predictions."""
    d = weight.numel()
    w = weight.double()
    q = float(w.square().sum()) / d
    excess = 0.0
    for start in range(0, len(x), row_chunk_size):
        h = x[start:start + row_chunk_size].double().matmul(w) / np.sqrt(d)
        a = activation(h)
        excess += float((q*a.square() - 2*h*a).sum())
    return (baseline + excess) / x.numel()


def train_condition(config, task, cfg, spike, train_x, test_x, mask_bank,
                    train_bases, train_counts, test_mask, test_base, test_count,
                    downstream_train, downstream_test, condition, *, device):
    import jax
    from erm_core import make_trainer

    started = time.perf_counter()
    seed = int(task['seed'])
    full = config.get('reconstruction_mode', 'masked') == 'full'
    dynamic = condition['masking_protocol'] == 'dynamic'
    k = 1 if full or dynamic else int(condition['k_views'])
    masks = None if full else mask_bank[:k]
    if full and (mask_bank is not None or test_mask is not None):
        raise ValueError('Full reconstruction received masks; use the mask-free task loader.')
    generator = torch.Generator().manual_seed(restart_init_seed(seed, 0))
    model = MaskedRankOneAutoencoder(
        cfg.d, activation=cfg.activation, init_std=cfg.init_std,
        generator=generator, dtype=cfg.dtype, device=cfg.device)
    initialization = config.get('initialization', 'gaussian')
    if initialization == 'planted':
        with torch.no_grad():
            model.w.copy_(spike.u_star)
    elif initialization != 'gaussian':
        raise ValueError(f'Unknown initialization: {initialization}')
    state, step, final_gradient = make_trainer(
        train_x.numpy(), None if full else np.stack([b.numpy() for b in masks]),
        model.w.detach().numpy(), activation=cfg.activation, rho=cfg.rho,
        dynamic=dynamic, reconstruction_mode='full' if full else 'masked',
        chunk_size=config['row_chunk_size'],
        learning_rate=cfg.lr, lambda_r=float(config.get('lambda_r', 0.)),
        seed=resampled_mask_seed(seed), device=device)
    activation = activation_function(cfg.activation)
    if full:
        storage = dict(dataset_dtype=str(train_x.dtype), mask_dtype='none',
            train_dataset_storage_bytes=train_x.numel()*train_x.element_size(),
            train_mask_bank_storage_bytes=0,
            persistent_train_storage_bytes=train_x.numel()*train_x.element_size(),
            test_dataset_storage_bytes=test_x.numel()*test_x.element_size(),
            test_mask_storage_bytes=0)
    else:
        storage = storage_metadata(train_x, masks, test_x, test_mask)
    common = dict(initialization=initialization, init_std=cfg.init_std,
                  alpha=float(task['alpha']), n_train=cfg.n_train,
                  activation=task['activation'], activation_name=cfg.activation,
                  **condition, d=cfg.d, beta=cfg.beta, rho=None if full else cfg.rho,
                  reconstruction_mode="full" if full else "masked",
                  gamma_dist=cfg.gamma_dist, gamma_sigma=cfg.gamma_sigma,
                  gamma_floor=cfg.gamma_floor, gamma_eta=cfg.gamma_eta,
                  epochs=cfg.epochs, lambda_r=float(config.get('lambda_r', 0.)),
                  learning_rate=cfg.lr, row_chunk_size=config['row_chunk_size'],
                  method=('jax_full_reconstruction_adam' if full else
                          'jax_bool_dynamic_mae_adam' if dynamic else 'jax_bool_fixed_k_mae_adam'),
                  training_backend='jax', training_device=str(device),
                  jax_version=jax.__version__, device_mask_dtype='none' if full else 'bool',
                  dynamic_rng='none' if full else 'jax_threefry' if dynamic else 'matched_torch_cpu_bank',
                  **storage)
    for field in ('gamma_low', 'gamma_high', 'gamma_beta_a', 'gamma_beta_b'):
        common[field] = config.get(field, np.nan)
    common.update(
        n_ds=int(config["n_ds"]), probe_n_test=int(config["probe_n_test"]),
        train_target_count=int(sum(train_counts[:k])),
        train_zero_decoder_sse=float(sum(train_bases[:k])),
        train_sample_view_count=int(cfg.n_train*k),
        test_target_count=int(test_count), test_zero_decoder_sse=float(test_base),
        test_sample_view_count=int(test_x.shape[0]),
        n_test_reconstruction=int(test_x.shape[0]),
        test_views_per_sample=1, train_evaluation_views_per_sample=k,
        test_population_zero_decoder_per_sample=(1.0 if full else cfg.rho)*(
            spike.gamma.double().sum().item()
            +cfg.beta*spike.u_star.double().square().sum().item()/cfg.d),
        train_loss_estimator=(
            "Full reconstruction on all training coordinates" if full else
            "Monte Carlo mask-averaged training risk; one fixed evaluation mask per training sample, independent of dynamic training masks"
            if dynamic else "Empirical loss on all K fixed training masks"),
        test_loss_estimator=("Full reconstruction on independent test samples, all coordinates" if full else
            "Monte Carlo population risk; independent test samples and one independent mask per sample"))
    if not full:
        # Preserve the historical masked-output fields.
        common.update(train_hidden_count=common['train_target_count'],
                      test_hidden_count=common['test_target_count'])
    history = []
    last_step = (np.nan, np.nan)
    for epoch in range(cfg.epochs + 1):
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            weight = torch.from_numpy(np.array(jax.device_get(state[0]), copy=True))
            if not torch.isfinite(weight).all():
                raise FloatingPointError(f'Nonfinite weights at epoch {epoch}')
            row = representation_metrics(weight, spike, epoch, seed)
            row.update(common)
            if full:
                train_loss = raw_full_loss(weight, train_x, train_bases[0], activation, config['row_chunk_size'])
                test_loss = raw_full_loss(weight, test_x, test_base, activation, config['row_chunk_size'])
            else:
                train_loss = raw_bank_loss(
                    weight, train_x, masks, train_bases[:k], train_counts[:k], activation, config['row_chunk_size'])
                test_loss = raw_bank_loss(
                    weight, test_x, [test_mask], [test_base], [test_count], activation, config['row_chunk_size'])
            row.update(pretrain_train_loss=train_loss, pretrain_test_loss=test_loss,
                       last_step_norm=float(last_step[0]), last_step_relative_norm=float(last_step[1]))
            row.update(appendix_observables(row))
            history.append(row)
            print(f"JAX task={task['task_id']} K={condition['k_label']} "
                  f"epoch={epoch}/{cfg.epochs} cosine={row['cosine_abs']:.6g}", flush=True)
        if epoch < cfg.epochs:
            state, last_step = step(state)
    grad_norm = float(final_gradient(state))
    if not np.isfinite(grad_norm):
        raise FloatingPointError('Nonfinite final gradient')
    history[-1].update(
        final_gradient_norm=grad_norm,
        final_gradient_kind='all_training_coordinates' if full else 'fresh_dynamic_mask' if dynamic else 'fixed_training_mask_bank',
        train_loss_evaluation_protocol='full_reconstruction' if full else 'original_fixed_evaluation_mask' if dynamic
        else 'fixed_training_mask_bank')
    final = dict(history[-1])
    with torch.no_grad():
        train_features = encoder_fields(weight, downstream_train.x)
        test_features = encoder_fields(weight, downstream_test.x)
    for label, features in (('train', train_features), ('test', test_features)):
        final[f'probe_feature_{label}_mean'] = float(features.mean())
        final[f'probe_feature_{label}_std'] = float(features.std())
    final.update(fit_scalar_logistic_probe(
        train_features, downstream_train.y, test_features, downstream_test.y,
        lr=float(config['probe_lbfgs_lr']), max_iter=int(config['probe_lbfgs_max_iter']),
        history_size=int(config['probe_lbfgs_history_size'])))
    for label in ('train', 'test'):
        final[f'probe_{label}_error'] = 1. - final[f'probe_{label}_accuracy']
    final['elapsed_seconds'] = time.perf_counter() - started
    return pd.DataFrame(history), final, weight


# The 17 alpha values shared by the reported ERM and AMP comparisons.
PAPER_ALPHAS = [
    0.1, 0.1333521432163324, 0.1778279410038923, 0.2371373705661655,
    0.3162277660168379, 0.4216965034285823, 0.5623413251903491,
    0.7498942093324559, 1.0, 1.333521432163324, 1.7782794100389228,
    2.3713737056616555, 3.1622776601683795, 4.216965034285822,
    5.62341325190349, 7.498942093324557, 10.0,
]
PAPER_RHOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Write the configuration; do not train.")
    prep.add_argument("--experiment", choices=("masked-alpha", "full-alpha", "masked-rho"), required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--d", type=int, default=2000)
    prep.add_argument("--beta", type=float, default=1.0)
    prep.add_argument("--rho", type=float, default=0.75)
    prep.add_argument("--rhos", type=float, nargs="+")
    prep.add_argument("--alphas", type=float, nargs="+")
    prep.add_argument("--activations", choices=tuple(ACTIVATION_NAMES), nargs="+")
    prep.add_argument("--k-values", type=int, nargs="+")
    prep.add_argument("--dynamic-only", action="store_true")
    prep.add_argument("--include-dynamic", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--n-seeds", type=int, default=30)
    prep.add_argument("--seed-start", type=int, default=0)
    prep.add_argument("--updates", dest="epochs", type=int, default=4000,
                      help="Full-batch Adam updates; no early stopping.")
    prep.add_argument("--eval-every", type=int, default=200)
    prep.add_argument("--learning-rate", type=float, default=0.05)
    prep.add_argument("--lambda-r", type=float, default=1e-4)
    prep.add_argument("--init-std", type=float, default=1.0)
    prep.add_argument("--row-chunk-size", type=int, default=1024)
    prep.add_argument("--n-ds", type=int, default=100)
    prep.add_argument("--probe-n-test", type=int, default=10000)
    prep.add_argument("--probe-lbfgs-lr", type=float, default=1.0)
    prep.add_argument("--probe-lbfgs-max-iter", type=int, default=100)
    prep.add_argument("--probe-lbfgs-history-size", type=int, default=25)
    prep.add_argument("--gamma-dist", choices=(BOUNDED_HETEROGENOUS, "lognormal", "constant"),
                      default=BOUNDED_HETEROGENOUS)
    prep.add_argument("--gamma-sigma", type=float, default=1.0)
    prep.add_argument("--gamma-floor", type=float, default=0.75)
    prep.add_argument("--gamma-eta", type=float, default=0.25)
    task = commands.add_parser("task", help="Run one independent seed/alpha/rho/activation task.")
    task.add_argument("--config", type=Path, required=True)
    task.add_argument("--task-id", type=int, required=True)
    task.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    task.add_argument("--force", action="store_true", help="Rerun this task even if completed.")
    collect = commands.add_parser("aggregate", help="Collect completed tasks; write CSV only.")
    collect.add_argument("--config", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        args.reconstruction_mode = "full" if args.experiment == "full-alpha" else "masked"
        rho_sweep = args.experiment == "masked-rho"
        if args.alphas is None:
            args.alphas = [8.0] if rho_sweep else PAPER_ALPHAS
        if args.rhos is None and rho_sweep:
            args.rhos = PAPER_RHOS
        if args.activations is None:
            args.activations = ["linear"] if rho_sweep else ["linear", "tanh", "elu", "relu"]
        if args.k_values is None:
            args.k_values = [1, 2, 4, 8] if rho_sweep else [1, 2, 4]
        if any(round(a * args.d) < 1 for a in args.alphas):
            raise ValueError("Every alpha must give at least one training sample.")
        if args.seed_start < 0 or args.n_ds < 1 or args.probe_n_test < 1:
            raise ValueError("Seeds must be nonnegative and probe sample counts positive.")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("Learning rate must be positive and finite.")
        if not np.isfinite(args.init_std) or args.init_std <= 0:
            raise ValueError("Initialization standard deviation must be positive and finite.")
        args.config = args.output / "config.json"
        args.force = False
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Choose an empty output directory to avoid mixing experiments.")
        prepare(args)
    elif args.command == "task":
        import jax
        device = jax.devices(args.device)[0]
        run_task(args, train_fn=functools.partial(train_condition, device=device))
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
