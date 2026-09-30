#!/usr/bin/env python3
"""Static/dynamic convolutional MAE experiments on clean CIFAR-10 images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cifar10_conv_mae_core import (
    IMAGE_SIZE,
    ConvMAE,
    IndexedTensorDataset,
    channel_statistics,
    count_parameters,
    device_metadata,
    expand_patch_masks,
    load_cifar10_train,
    load_or_create_splits,
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


@dataclass(frozen=True)
class Condition:
    ae_seed: int
    schedule: str
    rho: float

    @property
    def name(self) -> str:
        if self.schedule == "full_reconstruction":
            return f"seed{self.ae_seed}_full_reconstruction"
        return f"seed{self.ae_seed}_{self.schedule}_rho{rho_token(self.rho)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help="Optional saved split_indices.npz and normalization.json to reuse exactly",
    )
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--schedules", nargs="+", choices=["static", "dynamic"], default=["static", "dynamic"])
    parser.add_argument(
        "--condition-spec",
        nargs="*",
        default=None,
        metavar="SCHEDULE:RHO",
        help="Exact conditions, e.g. dynamic:0.5 or full_reconstruction:0",
    )
    parser.add_argument("--ae-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--pretrain-per-class", type=int, default=4000)
    parser.add_argument("--validation-per-class", type=int, default=500)
    parser.add_argument("--probe-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="*",
        default=[],
        help="Optional exact epochs whose model weights are retained for later probing",
    )
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-full-reconstruction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--continue-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume a completed run only when it previously stopped at its epoch ceiling",
    )
    parser.add_argument(
        "--continue-all-complete",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume every complete condition from last.pt, including patience-stopped runs",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-images", type=int, default=None, help="Smoke-test limiter; balanced selection is not guaranteed")
    parser.add_argument("--max-validation-images", type=int, default=None, help="Smoke-test limiter")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if IMAGE_SIZE % args.patch_size:
        raise ValueError("patch-size must divide 32")
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        raise ValueError("epochs, min-epochs, and patience must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if len(set(args.checkpoint_epochs)) != len(args.checkpoint_epochs):
        raise ValueError("checkpoint-epochs must not contain duplicates")
    if any(epoch < 1 or epoch > args.epochs for epoch in args.checkpoint_epochs):
        raise ValueError("checkpoint-epochs must lie between 1 and epochs")
    if len(set(args.rhos)) != len(args.rhos):
        raise ValueError("rhos must not contain duplicates")
    if any(not 0 < rho < 1 for rho in args.rhos):
        raise ValueError("rhos must lie strictly between zero and one")
    if args.condition_spec:
        for spec in args.condition_spec:
            try:
                schedule, rho_text = spec.split(":")
                rho = float(rho_text)
            except ValueError as error:
                raise ValueError(f"Invalid condition spec {spec!r}; expected SCHEDULE:RHO") from error
            if schedule not in {"static", "dynamic", "full_reconstruction"}:
                raise ValueError(f"Invalid condition spec {spec!r}")
            if schedule == "full_reconstruction" and rho != 0:
                raise ValueError(f"Full-reconstruction condition must use rho=0: {spec!r}")
            if schedule != "full_reconstruction" and not 0 < rho < 1:
                raise ValueError(f"Masked condition must use 0<rho<1: {spec!r}")


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def create_fixed_patch_bank(length: int, patch_size: int, rho: float, seed: int) -> tuple[torch.Tensor, int, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return make_exact_patch_masks(length, patch_size, rho, generator, torch.device("cpu"))


def condition_seeds(condition: Condition) -> dict[str, int]:
    return {
        # An AE seed denotes one shared initialization and minibatch ordering.
        "model": stable_seed(condition.ae_seed, 1),
        "shuffle": stable_seed(condition.ae_seed, 2),
        # Identical random scores make masks nested across ratios. Validation
        # excludes schedule, so both schedules use the same fresh bank.
        "dynamic_mask": stable_seed(condition.ae_seed, 3),
        "static_mask": stable_seed(condition.ae_seed, 4),
        "validation_mask": stable_seed(condition.ae_seed, 5),
    }


def evaluate(
    model: ConvMAE,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    patch_bank: torch.Tensor | None,
    patch_size: int,
) -> float:
    model.eval()
    loss_numerator = 0.0
    examples = 0
    with torch.inference_mode():
        for raw, _labels, positions in loader:
            target = normalize_batch(raw, mean, std, device)
            if patch_bank is None:
                pixel_mask = None
            else:
                patch_mask = patch_bank[positions].to(device, non_blocking=True)
                pixel_mask = expand_patch_masks(patch_mask, patch_size)
            prediction = model(target, pixel_mask)
            batch_loss = reconstruction_loss(prediction, target, pixel_mask)
            loss_numerator += float(batch_loss) * len(raw)
            examples += len(raw)
    return loss_numerator / max(examples, 1)


def save_plots_and_examples(*args, **kwargs):
    """Plotting is centralized in the supplementary plotting notebook."""
    return None


def train_condition(
    condition: Condition,
    args: argparse.Namespace,
    output_dir: Path,
    train_dataset: IndexedTensorDataset,
    validation_dataset: IndexedTensorDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    condition_dir = output_dir / "conditions" / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    status_path = condition_dir / "status.json"
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("state") == "complete" and (condition_dir / "best.pt").is_file():
            hit_epoch_ceiling = status.get("summary", {}).get("hit_epoch_ceiling", False)
            can_continue = (
                (args.continue_all_complete or (args.continue_complete and hit_epoch_ceiling))
                and (condition_dir / "last.pt").is_file()
            )
            if can_continue:
                print(f"[resume] continuing epoch-limited {condition.name}", flush=True)
            else:
                print(f"[resume] skipping complete {condition.name}", flush=True)
                return status["summary"]

    seeds = condition_seeds(condition)
    seed_everything(seeds["model"])
    loader_generator = torch.Generator(device="cpu").manual_seed(seeds["shuffle"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
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

    masked = condition.schedule != "full_reconstruction"
    if masked:
        static_bank, masked_count, realized_rho = create_fixed_patch_bank(
            len(train_dataset), args.patch_size, condition.rho, seeds["static_mask"]
        )
        validation_bank, validation_count, validation_rho = create_fixed_patch_bank(
            len(validation_dataset), args.patch_size, condition.rho, seeds["validation_mask"]
        )
        if masked_count != validation_count or not math.isclose(realized_rho, validation_rho):
            raise AssertionError("Training and validation mask realization mismatch")
        actual_counts = static_bank.sum(dim=1)
        if not torch.all(actual_counts == masked_count):
            raise AssertionError("Static masks do not have exact patch counts")
    else:
        static_bank = None
        validation_bank = None
        masked_count = 0
        realized_rho = 0.0

    dynamic_generator = torch.Generator(device=device).manual_seed(seeds["dynamic_mask"])
    model = ConvMAE().to(device)
    if count_parameters(model) != 854_563:
        raise AssertionError(f"Unexpected model size: {count_parameters(model)}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    last_path = condition_dir / "last.pt"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        # map_location=device also moves generator-state ByteTensors to CUDA,
        # while Generator.set_state requires its serialized state on CPU.
        loader_generator.set_state(
            checkpoint["loader_generator_state"].detach().to(device="cpu", dtype=torch.uint8)
        )
        dynamic_generator.set_state(
            checkpoint["dynamic_generator_state"].detach().to(device="cpu", dtype=torch.uint8)
        )
        history = checkpoint["history"]
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint["best_epoch"]
        stale_epochs = checkpoint["stale_epochs"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"[resume] {condition.name} from epoch {start_epoch}", flush=True)

    condition_metadata = {
        **asdict(condition),
        "name": condition.name,
        "seeds": seeds,
        "reconstruction_target": "clean standardized image",
        "patch_size": args.patch_size,
        "patches_per_image": (IMAGE_SIZE // args.patch_size) ** 2,
        "masked_patch_count": masked_count,
        "realized_rho": realized_rho,
        "device": str(device),
        "device_metadata": device_metadata(device),
        "parameter_count": count_parameters(model),
    }
    save_json(condition_dir / "config.json", condition_metadata)
    save_json(status_path, {"state": "running", "condition": condition_metadata})

    train_started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_sum = 0.0
        train_examples = 0
        for raw, _labels, positions in train_loader:
            target = normalize_batch(raw, mean, std, device)
            if condition.schedule == "static":
                patch_mask = static_bank[positions].to(device, non_blocking=True)
                pixel_mask = expand_patch_masks(patch_mask, args.patch_size)
            elif condition.schedule == "dynamic":
                patch_mask, count, dynamic_rho = make_exact_patch_masks(
                    len(raw), args.patch_size, condition.rho, dynamic_generator, device
                )
                if count != masked_count or not math.isclose(dynamic_rho, realized_rho):
                    raise AssertionError("Dynamic mask realization mismatch")
                pixel_mask = expand_patch_masks(patch_mask, args.patch_size)
            else:
                pixel_mask = None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = model(target, pixel_mask)
                loss = reconstruction_loss(prediction, target, pixel_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            train_sum += float(loss.detach()) * len(raw)
            train_examples += len(raw)

        train_loss = train_sum / train_examples
        validation_loss = evaluate(
            model,
            validation_loader,
            mean,
            std,
            device,
            validation_bank,
            args.patch_size,
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss - 1e-10
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_payload = {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "condition": condition_metadata,
                "mean": mean.cpu(),
                "std": std.cpu(),
            }
            atomic_torch_save(best_payload, condition_dir / "best.pt")
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "seconds_elapsed": time.time() - train_started,
        }
        history.append(row)
        write_rows(condition_dir / "history.csv", history)
        last_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "dynamic_generator_state": dynamic_generator.get_state(),
            "history": history,
            "epoch": epoch,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
        }
        atomic_torch_save(last_payload, last_path)
        if epoch in args.checkpoint_epochs:
            milestone_payload = {
                "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "condition": condition_metadata,
                "mean": mean.cpu(),
                "std": std.cpu(),
            }
            atomic_torch_save(
                milestone_payload,
                condition_dir / f"epoch{epoch:04d}.pt",
            )
        print(
            f"[{condition.name}] epoch={epoch:03d} train={train_loss:.6f} "
            f"val={validation_loss:.6f} best={best_loss:.6f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    best_checkpoint = torch.load(condition_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    fresh_validation_loss = evaluate(
        model,
        validation_loader,
        mean,
        std,
        device,
        validation_bank,
        args.patch_size,
    )
    reused_train_loss = None
    if condition.schedule == "static":
        ordered_train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        reused_train_loss = evaluate(
            model,
            ordered_train_loader,
            mean,
            std,
            device,
            static_bank,
            args.patch_size,
        )
    save_plots_and_examples(
        condition_dir,
        history,
        model,
        validation_loader,
        mean,
        std,
        device,
        validation_bank,
        args.patch_size,
    )
    summary = {
        **asdict(condition),
        "condition": condition.name,
        "requested_rho": condition.rho,
        "realized_rho": realized_rho,
        "masked_patch_count": masked_count,
        "best_epoch": best_epoch,
        "epochs_ran": history[-1]["epoch"],
        "hit_epoch_ceiling": history[-1]["epoch"] == args.epochs,
        "fresh_validation_mse": fresh_validation_loss,
        "static_reused_train_mse": reused_train_loss,
        "runtime_seconds": time.time() - train_started,
        "checkpoint": str((condition_dir / "best.pt").resolve()),
    }
    save_json(status_path, {"state": "complete", "condition": condition_metadata, "summary": summary})
    return summary


def update_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    path = output_dir / "pretraining_summary.csv"
    if path.is_file():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                existing[row["condition"]] = dict(row)
    for row in summaries:
        existing[row["condition"]] = row
    write_rows(path, list(existing.values()))


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"resolved device: {device}", flush=True)
    train_images, train_labels = load_cifar10_train(args.data_root)
    if args.reference_root is None:
        splits = load_or_create_splits(
            output_dir / "split_indices.npz",
            train_labels,
            args.split_seed,
            args.pretrain_per_class,
            args.validation_per_class,
            args.probe_per_class,
        )
        mean_np, std_np = channel_statistics(train_images, splits["pretrain"])
        normalization_source = "pretraining split only"
    else:
        reference_root = args.reference_root.expanduser().resolve()
        payload = np.load(reference_root / "split_indices.npz")
        splits = {
            name: payload[name].astype(np.int64)
            for name in ("pretrain", "validation", "probe")
        }
        validate_splits(
            splits,
            train_labels,
            args.pretrain_per_class,
            args.validation_per_class,
            args.probe_per_class,
        )
        np.savez(output_dir / "split_indices.npz", **splits)
        normalization = json.loads((reference_root / "normalization.json").read_text())
        mean_np = np.asarray(normalization["mean"], dtype=np.float32)
        std_np = np.asarray(normalization["std"], dtype=np.float32)
        normalization_source = str(reference_root)
    mean = torch.tensor(mean_np, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std_np, device=device).view(1, 3, 1, 1)
    save_json(
        output_dir / "normalization.json",
        {"mean": mean_np.tolist(), "std": std_np.tolist(), "source": normalization_source},
    )
    config = vars(args).copy()
    config.update(
        {
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "output_dir": str(output_dir),
            **device_metadata(device),
            "input": "unaltered CIFAR-10 image with channel standardization only",
            "reconstruction_target": "clean standardized image",
            "parameter_count": 854_563,
        }
    )
    save_json(output_dir / "config.json", config)

    train_indices = splits["pretrain"]
    validation_indices = splits["validation"]
    if args.max_train_images is not None:
        train_indices = train_indices[: args.max_train_images]
    if args.max_validation_images is not None:
        validation_indices = validation_indices[: args.max_validation_images]

    if args.condition_spec:
        templates = []
        for spec in args.condition_spec:
            schedule, rho_text = spec.split(":")
            templates.append((schedule, float(rho_text)))
        if len(set(templates)) != len(templates):
            raise ValueError("condition-spec contains duplicates")
    else:
        templates = [(schedule, rho) for schedule in args.schedules for rho in args.rhos]
        if args.include_full_reconstruction:
            templates.append(("full_reconstruction", 0.0))

    summaries: list[dict[str, Any]] = []
    for ae_seed in args.ae_seeds:
        train_dataset = IndexedTensorDataset(train_images, train_labels, train_indices)
        validation_dataset = IndexedTensorDataset(train_images, train_labels, validation_indices)
        conditions = [
            Condition(ae_seed=ae_seed, schedule=schedule, rho=rho)
            for schedule, rho in templates
        ]
        for condition in conditions:
            summary = train_condition(
                condition,
                args,
                output_dir,
                train_dataset,
                validation_dataset,
                mean,
                std,
                device,
            )
            summaries.append(summary)
            update_summary(output_dir, summaries)
    print(f"completed {len(summaries)} condition(s); summary: {output_dir / 'pretraining_summary.csv'}")


if __name__ == "__main__":
    main()
