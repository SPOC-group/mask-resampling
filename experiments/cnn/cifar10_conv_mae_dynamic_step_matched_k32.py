#!/usr/bin/env python3
"""MAE mask-schedule controls with a configurable comparison clock.

With ``--reference-k 1``, an epoch contains N image-mask presentations and is
ordinary dynamic masking: every original appears once and receives a freshly
sampled mask. Larger reference K values make one comparison epoch contain N*K
presentations so its validation, scheduling, and early-stopping boundaries
match a paired fixed-K control.

``--fixed-mask-k`` selects a frozen bank of K Boolean masks per original while
retaining ``reference_k`` presentations and optimizer steps per comparison
epoch. For K larger than the reference clock, successive epochs rotate through
the bank. This decouples mask-bank size from the exposure/update clock.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cifar10_conv_mae import (
    atomic_torch_save,
    create_fixed_patch_bank,
    evaluate,
    save_plots_and_examples,
    write_rows,
)
from cifar10_conv_mae_core import (
    IMAGE_SIZE,
    ConvMAE,
    IndexedTensorDataset,
    count_parameters,
    device_metadata,
    expand_patch_masks,
    load_cifar10_train,
    make_exact_patch_masks,
    normalize_batch,
    reconstruction_loss,
    resolve_device,
    rho_token,
    save_json,
    seed_everything,
    stable_seed,
    validate_splits,
)
from cifar10_conv_mae_kmask_control import (
    DynamicAugmentedFixedKMaskViewDataset,
    FixedKMaskViewDataset,
    UniqueImageFixedKBatchSampler,
    create_or_load_boolean_patch_bank,
    gather_boolean_patch_masks,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = Path("runs/cnn/protocols/seed0/protocol")


@dataclass(frozen=True)
class StepMatchedDynamicCondition:
    ae_seed: int
    rho: float
    reference_k: int
    architecture: str = "conv"
    fixed_mask_k1: bool = False
    fixed_mask_k: int | None = None

    @property
    def mask_bank_size(self) -> int | None:
        return 1 if self.fixed_mask_k1 else self.fixed_mask_k

    @property
    def schedule(self) -> str:
        if self.mask_bank_size is not None:
            return f"static_k{self.mask_bank_size}_step_matched_k{self.reference_k}"
        if self.reference_k == 1:
            return "dynamic"
        return f"dynamic_step_matched_k{self.reference_k}"

    @property
    def name(self) -> str:
        architecture = "" if self.architecture == "conv" else f"{self.architecture}_"
        if self.mask_bank_size is not None:
            clock = f"static_k{self.mask_bank_size}_stepmatched_k{self.reference_k}"
        else:
            clock = (
                "dynamic"
                if self.reference_k == 1
                else f"dynamic_stepmatched_k{self.reference_k}"
            )
        return (
            f"seed{self.ae_seed}_{architecture}{clock}_rho{rho_token(self.rho)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--ae-seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--reference-k", type=int, default=32)
    parser.add_argument(
        "--fixed-mask-k1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use one frozen mask per original while retaining reference-k image "
            "presentations and optimizer steps per comparison epoch."
        ),
    )
    parser.add_argument(
        "--fixed-mask-k",
        type=int,
        default=None,
        help="Frozen masks per image; independent of the reference-k training clock.",
    )
    parser.add_argument(
        "--architecture",
        choices=("conv",),
        default="conv",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=460)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-schedule", choices=("plateau", "constant"), default="plateau")
    parser.add_argument("--diagnostic-first-epoch", action="store_true",
                        help="Replay initialization through one epoch with diagnostic validation and batch losses.")
    parser.add_argument("--diagnostic-validation-updates", type=int, nargs="+",
                        default=[0, 50, 100, 250, 500, 1000, 1500, 2000, 2500])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume a completed condition only when it stopped at its epoch ceiling.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-validation-images", type=int, default=None)
    parser.add_argument(
        "--dynamic-train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resample U-MAE-style RandomResizedCrop(scale=0.2..1.0) and "
            "horizontal flip for every image presentation; mask schedule is separate."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.rho < 1:
        raise ValueError("rho must lie strictly between zero and one")
    if args.reference_k < 1:
        raise ValueError("reference-k must be positive")
    if IMAGE_SIZE % args.patch_size:
        raise ValueError("patch-size must divide 32")
    if min(args.epochs, args.min_epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epoch, patience, and batch settings must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if args.dynamic_train_augmentation and args.architecture != "conv":
        raise ValueError("The dynamic-augmentation control is restricted to the old CNN")
    if args.fixed_mask_k is not None and args.fixed_mask_k < 1:
        raise ValueError("fixed-mask-k must be positive")
    if args.fixed_mask_k1 and args.fixed_mask_k not in (None, 1):
        raise ValueError("fixed-mask-k1 conflicts with fixed-mask-k other than 1")


def condition_seeds(condition: StepMatchedDynamicCondition) -> dict[str, int]:
    return {
        "model": stable_seed(condition.ae_seed, 1),
        "shuffle": stable_seed(condition.ae_seed, 2),
        "dynamic_mask": stable_seed(condition.ae_seed, 3),
        "fixed_mask": stable_seed(condition.ae_seed, 4),
        "validation_mask": stable_seed(condition.ae_seed, 5),
        "augmentation": stable_seed(condition.ae_seed, 6),
    }


def diagnostic_evaluate(model, loader, mean, std, device, bank, patch_size, generators):
    """Extra evaluation must not change training mode or any training RNG stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    generator_states = [(g, g.get_state()) for g in generators if g is not None]
    modes = [(module, module.training) for module in model.modules()]
    try:
        return evaluate(model, loader, mean, std, device, bank, patch_size)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for generator, state in generator_states:
            generator.set_state(state)
        for module, mode in modes:
            module.training = mode


def make_autoencoder(architecture: str) -> ConvMAE:
    if architecture != "conv":
        raise ValueError("This package contains the convolutional autoencoder only")
    return ConvMAE()


def expected_parameter_count(architecture: str) -> int:
    if architecture != "conv":
        raise ValueError("This package contains the convolutional autoencoder only")
    return 854_563


def train_condition(
    condition: StepMatchedDynamicCondition,
    args: argparse.Namespace,
    output_dir: Path,
    train_dataset: FixedKMaskViewDataset,
    validation_dataset: IndexedTensorDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    condition_dir = output_dir / "conditions" / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    status_path = condition_dir / "status.json"
    if args.diagnostic_first_epoch and any((condition_dir / name).exists() for name in ("history.csv", "last.pt", "batch_history.csv")):
        raise ValueError("Diagnostic replay requires a fresh output directory; it must start from initialization")
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("state") == "complete" and (condition_dir / "best.pt").is_file():
            hit_ceiling = bool(status.get("summary", {}).get("hit_epoch_ceiling", False))
            can_continue = (
                args.continue_complete and hit_ceiling and (condition_dir / "last.pt").is_file()
            )
            if can_continue:
                print(f"[resume] continuing epoch-limited {condition.name}", flush=True)
            else:
                print(f"[resume] skipping complete {condition.name}", flush=True)
                return status["summary"]

    seeds = condition_seeds(condition)
    seed_everything(seeds["model"])
    loader_generator = torch.Generator(device="cpu").manual_seed(seeds["shuffle"])
    batch_sampler = UniqueImageFixedKBatchSampler(
        train_dataset.samples,
        condition.reference_k,
        args.batch_size,
        loader_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_bank, masked_count, realized_rho = create_fixed_patch_bank(
        len(validation_dataset),
        args.patch_size,
        condition.rho,
        seeds["validation_mask"],
    )
    dynamic_generator = None
    fixed_bank = None
    fixed_bank_path = None
    if condition.mask_bank_size is not None:
        fixed_bank, fixed_count, fixed_rho, fixed_bank_path = (
            create_or_load_boolean_patch_bank(
                condition_dir,
                train_dataset.samples,
                condition.mask_bank_size,
                args.patch_size,
                condition.rho,
                seeds["fixed_mask"],
            )
        )
        if fixed_count != masked_count or not math.isclose(fixed_rho, realized_rho):
            raise AssertionError("Fixed training and validation mask realizations differ")
    else:
        dynamic_generator = torch.Generator(device=device).manual_seed(
            seeds["dynamic_mask"]
        )

    model = make_autoencoder(condition.architecture).to(device)
    expected_parameters = expected_parameter_count(condition.architecture)
    if count_parameters(model) != expected_parameters:
        raise AssertionError(f"Unexpected model size: {count_parameters(model)}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None if args.lr_schedule == "constant" else ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    optimizer_steps = 0
    last_path = condition_dir / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        saved_schedule = checkpoint.get("lr_schedule", "plateau")
        if saved_schedule != args.lr_schedule:
            raise ValueError("Cannot change the learning-rate schedule when resuming a run")
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        elif any(group["lr"] != args.lr for group in optimizer.param_groups):
            raise ValueError("Resumed optimizer does not have the requested constant learning rate")
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        loader_generator.set_state(
            checkpoint["loader_generator_state"].detach().to(device="cpu", dtype=torch.uint8)
        )
        if condition.mask_bank_size is None:
            assert dynamic_generator is not None
            dynamic_generator.set_state(
                checkpoint["dynamic_generator_state"].detach().to(
                    device="cpu", dtype=torch.uint8
                )
            )
        history = checkpoint["history"]
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        optimizer_steps = int(checkpoint["optimizer_steps"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] {condition.name} from comparison epoch {start_epoch}", flush=True)

    steps_per_epoch = len(batch_sampler)
    diagnostic_updates = set(args.diagnostic_validation_updates)
    if args.diagnostic_first_epoch:
        if 0 not in diagnostic_updates or steps_per_epoch not in diagnostic_updates or any(u < 0 or u > steps_per_epoch for u in diagnostic_updates):
            raise ValueError("Diagnostic updates must include 0 and the epoch boundary and lie within the first epoch")
    expected_pairs = train_dataset.samples * condition.reference_k
    expected_steps = math.ceil(expected_pairs / args.batch_size)
    if steps_per_epoch != expected_steps:
        raise RuntimeError(
            f"Batch sampler reports {steps_per_epoch} steps, expected {expected_steps}"
        )
    ordinary_dynamic = condition.reference_k == 1 and condition.mask_bank_size is None
    condition_metadata = {
        **asdict(condition),
        "name": condition.name,
        "schedule": condition.schedule,
        "seeds": seeds,
        "control": (
            f"{condition.mask_bank_size} frozen masks per image with a "
            f"K={condition.reference_k}-matched presentation/update clock"
            if condition.mask_bank_size is not None
            else (
                "ordinary dynamic masks: one fresh mask per original per epoch"
                if ordinary_dynamic
                else f"dynamic masks with fixed-K{condition.reference_k}-matched optimizer-step clock"
            )
        ),
        "architecture": condition.architecture,
        "encoder_decoder_skip_connections": False,
        "residual_connections": "inside ResNet-18 encoder only",
        "mask_resampling_during_training": condition.mask_bank_size is None,
        "fixed_masks_per_original": condition.mask_bank_size,
        "fixed_bank_epoch_rotation": (
            "(view_index + (epoch-1)*reference_k) mod fixed_mask_k"
            if condition.mask_bank_size is not None else None
        ),
        "fixed_mask_bank": (
            str(fixed_bank_path.resolve()) if fixed_bank_path is not None else None
        ),
        "fixed_mask_bank_shape": (
            list(fixed_bank.shape) if fixed_bank is not None else None
        ),
        "ordinary_dynamic_epoch_clock": ordinary_dynamic,
        "original_training_images": train_dataset.samples,
        "presentations_per_original_per_comparison_epoch": condition.reference_k,
        "training_pairs_per_comparison_epoch": expected_pairs,
        "optimizer_steps_per_comparison_epoch": steps_per_epoch,
        "validation_interval_optimizer_steps": steps_per_epoch,
        "scheduler_interval_optimizer_steps": steps_per_epoch,
        "early_stopping_interval_optimizer_steps": steps_per_epoch,
        "unique_original_images_per_training_batch": True,
        "training_batch_sampling": (
            "identical unique-image batch-sampler construction to paired fixed K=32 control"
        ),
        "batch_size": args.batch_size,
        "training_image_augmentation": (
            "dynamic_random_resized_crop_32_scale_0p2_1p0_bicubic_plus_horizontal_flip_p0p5"
            if args.dynamic_train_augmentation
            else "none"
        ),
        "augmentation_resampling": (
            "fresh deterministic draw per (seed, comparison_epoch, image_presentation)"
            if args.dynamic_train_augmentation
            else "not_applicable"
        ),
        "mask_and_image_augmentation_jointly_resampled": (
            args.dynamic_train_augmentation and condition.mask_bank_size is None
        ),
        "fixed_mask_relative_to_dynamic_augmentation": (
            "bank-selected spatial mask is applied after augmentation; bank is frozen"
            if condition.mask_bank_size is not None
            else "not_applicable"
        ),
        "reconstruction_target": (
            "dynamically_augmented_standardized_image"
            if args.dynamic_train_augmentation
            else "clean_standardized_image"
        ),
        "reconstruction_loss_support": "masked RGB coordinates only",
        "unmasked_prediction_gradient_by_construction": 0.0,
        "patch_size": args.patch_size,
        "patches_per_image": (IMAGE_SIZE // args.patch_size) ** 2,
        "masked_patch_count": masked_count,
        "realized_rho": realized_rho,
        "optimizer": "AdamW",
        "initial_lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": "ReduceLROnPlateau" if scheduler is not None else "constant",
        "scheduler_factor": args.scheduler_factor if scheduler is not None else None,
        "scheduler_patience_comparison_epochs": args.scheduler_patience if scheduler is not None else None,
        "minimum_lr": args.min_lr if scheduler is not None else args.lr,
        "early_stopping_patience_comparison_epochs": args.patience,
        "early_stopping_enabled": args.early_stopping,
        "device": str(device),
        "device_metadata": device_metadata(device),
        "parameter_count": count_parameters(model),
        "official_test_batch_opened": False,
    }
    save_json(condition_dir / "config.json", condition_metadata)
    save_json(status_path, {"state": "running", "condition": condition_metadata})

    started = time.time()
    batch_history = []
    diagnostic_history = []
    if args.diagnostic_first_epoch:
        atomic_torch_save(validation_bank, condition_dir / "diagnostic_validation_masks.pt")
        save_json(condition_dir / "diagnostic_config.json", {
            "validation_updates": sorted(diagnostic_updates),
            "checkpoint_selection_interval_updates": steps_per_epoch,
            "early_stopping_interval_updates": steps_per_epoch,
            "extra_evaluations_affect_selection": False,
            "training_rng_preserved": True,
            "training_loss_timing": "forward pass before the numbered optimizer update",
            "validation_loss_timing": "after the numbered optimizer update; 0 is initialization",
            "stop_after_updates": steps_per_epoch,
        })
        model.train()
        initial_validation = diagnostic_evaluate(model, validation_loader, mean, std, device,
                                                 validation_bank, args.patch_size,
                                                 [loader_generator, dynamic_generator])
        diagnostic_history.append({"optimizer_updates": 0, "validation_mse": initial_validation,
                                   "validation_samples": len(validation_dataset),
                                   "selection_check": False, "seconds_elapsed": time.time() - started})
        write_rows(condition_dir / "validation_updates.csv", diagnostic_history)
    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        train_sum = 0.0
        train_pairs = 0
        steps_this_epoch = 0
        for raw, _labels, pair_positions in train_loader:
            target = normalize_batch(raw, mean, std, device)
            if condition.mask_bank_size is not None:
                assert fixed_bank is not None
                sample_positions = pair_positions.remainder(train_dataset.samples)
                view_positions = pair_positions.div(
                    train_dataset.samples, rounding_mode="floor"
                )
                bank_views = (
                    view_positions + (epoch - 1) * condition.reference_k
                ).remainder(condition.mask_bank_size)
                patch_mask = gather_boolean_patch_masks(
                    fixed_bank,
                    bank_views * train_dataset.samples + sample_positions,
                    device,
                )
            else:
                assert dynamic_generator is not None
                patch_mask, count, batch_rho = make_exact_patch_masks(
                    len(raw),
                    args.patch_size,
                    condition.rho,
                    dynamic_generator,
                    device,
                )
                if count != masked_count or not math.isclose(batch_rho, realized_rho):
                    raise AssertionError("Dynamic mask realization mismatch")
            pixel_mask = expand_patch_masks(patch_mask, args.patch_size)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = model(target, pixel_mask)
                loss = (
                    reconstruction_loss(prediction, target, pixel_mask)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(raw)
            train_pairs += len(raw)
            steps_this_epoch += 1
            optimizer_steps += 1
            if args.diagnostic_first_epoch:
                batch_mse = float(loss.detach())
                elements_per_sample = 3 * masked_count * args.patch_size ** 2
                batch_history.append({
                    "optimizer_updates": optimizer_steps,
                    "comparison_epoch": epoch,
                    "batch_samples": len(raw),
                    "train_loss": batch_mse,
                    "batch_loss_times_samples": batch_mse * len(raw),
                    "cumulative_samples": train_pairs,
                    "cumulative_loss_times_samples": train_sum,
                    "cumulative_mean_train_loss": train_sum / train_pairs,
                    "batch_masked_rgb_elements": len(raw) * elements_per_sample,
                    "cumulative_masked_rgb_elements": train_pairs * elements_per_sample,
                    "cumulative_squared_error": train_sum * elements_per_sample,
                    "lr": optimizer.param_groups[0]["lr"],
                    "seconds_elapsed": time.time() - started,
                })
                if optimizer_steps in diagnostic_updates and optimizer_steps != steps_per_epoch:
                    mse = diagnostic_evaluate(model, validation_loader, mean, std, device,
                                              validation_bank, args.patch_size,
                                              [loader_generator, dynamic_generator])
                    diagnostic_history.append({"optimizer_updates": optimizer_steps,
                                               "validation_mse": mse,
                                               "validation_samples": len(validation_dataset),
                                               "selection_check": False,
                                               "seconds_elapsed": time.time() - started})
                    write_rows(condition_dir / "validation_updates.csv", diagnostic_history)
                if optimizer_steps % 50 == 0 or optimizer_steps == steps_per_epoch:
                    write_rows(condition_dir / "batch_history.csv", batch_history)

        if train_pairs != expected_pairs or steps_this_epoch != steps_per_epoch:
            raise AssertionError(
                f"Comparison epoch saw {train_pairs} pairs/{steps_this_epoch} steps; "
                f"expected {expected_pairs}/{steps_per_epoch}"
            )
        train_loss = train_sum / train_pairs
        validation_loss = evaluate(
            model,
            validation_loader,
            mean,
            std,
            device,
            validation_bank,
            args.patch_size,
        )
        if scheduler is not None:
            scheduler.step(validation_loss)
        if args.diagnostic_first_epoch:
            diagnostic_history.append({"optimizer_updates": optimizer_steps,
                                       "validation_mse": validation_loss,
                                       "validation_samples": len(validation_dataset),
                                       "selection_check": True,
                                       "seconds_elapsed": time.time() - started})
            write_rows(condition_dir / "validation_updates.csv", diagnostic_history)
            model.train()
        improved = validation_loss < best_loss - 1e-10
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.encoder.state_dict(),
                    "epoch": epoch,
                    "optimizer_steps": optimizer_steps,
                    "validation_loss": validation_loss,
                    "condition": condition_metadata,
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                    "test_data_opened": False,
                },
                condition_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        history.append(
            {
                "comparison_epoch": epoch,
                "optimizer_steps_this_epoch": steps_this_epoch,
                "optimizer_steps_total": optimizer_steps,
                "training_pairs": train_pairs,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "best_comparison_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "seconds_elapsed": time.time() - started,
            }
        )
        write_rows(condition_dir / "history.csv", history)
        last_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "lr_schedule": args.lr_schedule,
                "scaler_state_dict": scaler.state_dict(),
                "loader_generator_state": loader_generator.get_state(),
                "history": history,
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "encoder_state_dict": model.encoder.state_dict(),
                "mean": mean.cpu(),
                "std": std.cpu(),
                "condition": condition_metadata,
                "validation_loss": validation_loss,
                "test_data_opened": False,
            }
        if dynamic_generator is not None:
            last_payload["dynamic_generator_state"] = dynamic_generator.get_state()
        atomic_torch_save(last_payload, last_path)
        print(
            f"[{condition.name}] comparison_epoch={epoch:03d} "
            f"steps={optimizer_steps} train={train_loss:.6f} val={validation_loss:.6f} "
            f"best={best_loss:.6f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if args.diagnostic_first_epoch:
            break
        if args.early_stopping and epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    best_checkpoint = torch.load(
        condition_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    fresh_validation_mse = evaluate(
        model,
        validation_loader,
        mean,
        std,
        device,
        validation_bank,
        args.patch_size,
    )
    save_plots_and_examples(
        condition_dir,
        [
            {
                "epoch": row["comparison_epoch"],
                "train_loss": row["train_loss"],
                "validation_loss": row["validation_loss"],
                "lr": row["lr"],
            }
            for row in history
        ],
        model,
        validation_loader,
        mean,
        std,
        device,
        validation_bank,
        args.patch_size,
    )
    summary = {
        "ae_seed": condition.ae_seed,
        "schedule": condition.schedule,
        "architecture": condition.architecture,
        "rho": condition.rho,
        "condition": condition.name,
        "reference_k": condition.reference_k,
        "original_training_images": train_dataset.samples,
        "training_pairs_per_comparison_epoch": expected_pairs,
        "optimizer_steps_per_comparison_epoch": steps_per_epoch,
        "unique_original_images_per_training_batch": True,
        "requested_rho": condition.rho,
        "realized_rho": realized_rho,
        "masked_patch_count": masked_count,
        "best_epoch": best_epoch,
        "best_optimizer_steps": best_epoch * steps_per_epoch,
        "epochs_ran": history[-1]["comparison_epoch"],
        "optimizer_steps_ran": optimizer_steps,
        "hit_epoch_ceiling": history[-1]["comparison_epoch"] == args.epochs,
        "diagnostic_first_epoch": args.diagnostic_first_epoch,
        "fresh_validation_mse": fresh_validation_mse,
        "dynamic_train_augmentation": args.dynamic_train_augmentation,
        "mask_resampling_during_training": condition.mask_bank_size is None,
        "fixed_masks_per_original": condition.mask_bank_size,
        "runtime_seconds": time.time() - started,
        "checkpoint": str((condition_dir / "best.pt").resolve()),
        "official_test_batch_opened": False,
    }
    save_json(status_path, {"state": "complete", "condition": condition_metadata, "summary": summary})
    write_rows(output_dir / "pretraining_summary.csv", [summary])
    return summary


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"resolved device: {device}", flush=True)

    train_images, train_labels = load_cifar10_train(args.data_root)
    reference_payload = np.load(reference_root / "split_indices.npz")
    splits = {
        name: reference_payload[name].astype(np.int64)
        for name in ("pretrain", "validation", "probe")
    }
    validate_splits(splits, train_labels, 4000, 500, 500)
    np.savez(output_dir / "split_indices.npz", **splits)
    normalization = json.loads((reference_root / "normalization.json").read_text())
    mean_np = np.asarray(normalization["mean"], dtype=np.float32)
    std_np = np.asarray(normalization["std"], dtype=np.float32)
    mean = torch.tensor(mean_np, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_np, device=device).view(1, 3, 1, 1)
    save_json(
        output_dir / "normalization.json",
        {"mean": mean_np.tolist(), "std": std_np.tolist(), "source": str(reference_root)},
    )
    save_json(
        output_dir / "config.json",
        {
            **vars(args),
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "output_dir": str(output_dir),
            "reference_root": str(reference_root),
        "experiment": (
            f"static K={1 if args.fixed_mask_k1 else args.fixed_mask_k} mask MAE "
            f"with K={args.reference_k}-matched exposure/update clock"
            if args.fixed_mask_k1 or args.fixed_mask_k is not None
            else (
                "ordinary dynamic MAE"
                if args.reference_k == 1
                else f"dynamic MAE with K={args.reference_k}-matched optimizer-step clock"
            )
        ),
            "test_data_opened": False,
            **device_metadata(device),
            "parameter_count": expected_parameter_count(args.architecture),
        },
    )
    train_indices = splits["pretrain"]
    validation_indices = splits["validation"]
    if args.max_train_images is not None:
        train_indices = train_indices[: args.max_train_images]
    if args.max_validation_images is not None:
        validation_indices = validation_indices[: args.max_validation_images]
    np.savez(
        output_dir / "effective_split_indices.npz",
        pretrain=train_indices,
        validation=validation_indices,
        probe_train=splits["pretrain"],
        probe_validation=np.concatenate((splits["validation"], splits["probe"])),
    )
    if args.dynamic_train_augmentation:
        train_dataset = DynamicAugmentedFixedKMaskViewDataset(
            train_images,
            train_labels,
            train_indices,
            args.reference_k,
            augmentation_seed=stable_seed(args.ae_seed, 6),
        )
    else:
        train_dataset = FixedKMaskViewDataset(
            train_images,
            train_labels,
            train_indices,
            args.reference_k,
        )
    validation_dataset = IndexedTensorDataset(
        train_images, train_labels, validation_indices
    )
    condition = StepMatchedDynamicCondition(
        args.ae_seed,
        args.rho,
        args.reference_k,
        args.architecture,
        args.fixed_mask_k1,
        args.fixed_mask_k,
    )
    train_condition(
        condition,
        args,
        output_dir,
        train_dataset,
        validation_dataset,
        mean,
        std,
        device,
    )


if __name__ == "__main__":
    main()
