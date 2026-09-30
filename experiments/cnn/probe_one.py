#!/usr/bin/env python3
"""Train one 128-to-10 frozen probe on the pretraining 40k; validate on its 10k complement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


from cifar10_conv_mae_core import (  # noqa: E402
    ConvMAE, IndexedTensorDataset, load_cifar10_train, resolve_device,
    save_json, validate_splits,
)
from probe_core import extract_embeddings, train_one_classifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--pretraining-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--full-reconstruction", action="store_true")
    parser.add_argument("--validation-order", choices=("sorted", "protocol"), default="sorted")
    parser.add_argument("--checkpoint", type=Path,
                        help="Explicit encoder checkpoint instead of the best-validation encoder.")
    parser.add_argument("--expected-updates", type=int,
                        help="Require this exact optimizer update count in the selected encoder.")
    return parser.parse_args()


def load_encoder_checkpoint(condition_dir: Path, explicit_path: Path | None, expected_updates: int | None):
    checkpoint_path = explicit_path.resolve() if explicit_path is not None else condition_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_updates is not None and checkpoint.get("optimizer_steps") != expected_updates:
        raise ValueError(f"Encoder must have exactly {expected_updates} updates; got {checkpoint.get('optimizer_steps')}")
    return checkpoint_path, checkpoint


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    if status_path.is_file() and (args.output_dir / "best_head.pt").is_file():
        status = json.loads(status_path.read_text())
        if status.get("state") == "complete":
            if args.checkpoint is not None and Path(status["summary"]["pretraining_checkpoint"]).resolve() != args.checkpoint.resolve():
                raise ValueError("Existing probe used a different encoder checkpoint")
            if args.expected_updates is not None and status["summary"].get("pretraining_optimizer_updates") != args.expected_updates:
                raise ValueError("Existing probe used a different optimizer update count")
            print(f"[probe resume] {args.output_dir}", flush=True)
            return
    train_images, train_labels = load_cifar10_train(args.data_root)
    with np.load(args.reference_root / "split_indices.npz") as payload:
        splits = {key: payload[key].astype(np.int64) for key in ("pretrain", "validation", "probe")}
    validate_splits(splits, train_labels, 4000, 1000 if args.full_reconstruction else 500, 0 if args.full_reconstruction else 500)
    train_indices = splits["pretrain"]
    validation_indices = np.setdiff1d(
        np.arange(50_000, dtype=np.int64), train_indices, assume_unique=True
    )
    if args.validation_order == "protocol":
        validation_indices = np.concatenate((splits["validation"], splits["probe"]))
    if len(validation_indices) != 10_000 or np.intersect1d(train_indices, validation_indices).size:
        raise AssertionError("Probe train/validation partitions are invalid")
    with (args.reference_root / "normalization.json").open() as handle:
        normalization = json.load(handle)
    device = resolve_device(args.device)
    mean = torch.tensor(normalization["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(normalization["std"], device=device).view(1, 3, 1, 1)
    summary_path = args.pretraining_root / "pretraining_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Pretraining summary is missing: {summary_path}")
    import csv
    with summary_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Expected exactly one pretraining condition per run directory")
    condition = rows[0]["condition"]
    condition_dir = args.pretraining_root / "conditions" / condition
    status = json.loads((condition_dir / "status.json").read_text())
    if status.get("state") != "complete":
        raise RuntimeError("Pretraining is not complete")
    checkpoint_path, checkpoint = load_encoder_checkpoint(condition_dir, args.checkpoint, args.expected_updates)
    if checkpoint.get("test_data_opened", False if args.full_reconstruction else None) is not False:
        raise RuntimeError("Pretraining checkpoint has invalid test-data audit flag")
    if not torch.allclose(checkpoint["mean"].reshape(-1), mean.cpu().reshape(-1)):
        raise AssertionError("Checkpoint and protocol normalization disagree")
    if not torch.allclose(checkpoint["std"].reshape(-1), std.cpu().reshape(-1)):
        raise AssertionError("Checkpoint and protocol normalization disagree")
    model = ConvMAE().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    train_features, train_targets = extract_embeddings(
        model, IndexedTensorDataset(train_images, train_labels, train_indices),
        mean, std, device, args.embedding_batch_size, args.num_workers,
    )
    val_features, val_targets = extract_embeddings(
        model, IndexedTensorDataset(train_images, train_labels, validation_indices),
        mean, std, device, args.embedding_batch_size, args.num_workers,
    )
    if train_features.shape != (40_000, 128) or val_features.shape != (10_000, 128):
        raise AssertionError("Unexpected old-CNN encoder embedding shape")
    if not np.array_equal(train_targets, train_labels[train_indices]):
        raise AssertionError("Probe training labels changed")
    if not np.array_equal(val_targets, train_labels[validation_indices]):
        raise AssertionError("Probe validation labels changed")
    probe_args = SimpleNamespace(
        resolved_device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, lr=args.lr, weight_decay=0.0,
        epochs=args.epochs, min_epochs=20, patience=args.patience,
    )
    result = train_one_classifier(
        train_features, train_targets, val_features, val_targets,
        probe_args, args.output_dir,
    )
    converged = result["classifier_patience_gap"] >= args.patience
    summary = {
        "condition": condition,
        "pretraining_checkpoint": str(checkpoint_path.resolve()),
        "pretraining_optimizer_updates": checkpoint.get("optimizer_steps"),
        "encoder_checkpoint_selection": "explicit" if args.checkpoint is not None else "best_validation",
        "classifier_training_images": 40_000,
        "classifier_validation_images": 10_000,
        "encoder_frozen": True,
        "classifiers_trained_for_encoder": 1,
        "classifier_converged_by_patience": converged,
        "official_test_batch_opened": False,
        **result,
    }
    save_json(status_path, {"state": "complete" if converged else "needs_continuation", "summary": summary})
    if not converged:
        raise RuntimeError("Linear probe reached epoch limit without exhausting patience")
    print(f"[probe] {condition}: validation accuracy={result['validation_accuracy']:.5f}", flush=True)


if __name__ == "__main__":
    main()
