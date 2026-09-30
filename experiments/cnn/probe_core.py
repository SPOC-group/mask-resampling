"""Original frozen linear-probe routines; no joint encoder fine-tuning."""
from __future__ import annotations
import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from cifar10_conv_mae_core import ConvMAE, IndexedTensorDataset, normalize_batch, seed_everything
CLASSIFIER_SEED = 314159

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

def extract_embeddings(
    model: ConvMAE,
    dataset: IndexedTensorDataset,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for raw, targets, _positions in loader:
            images = normalize_batch(raw, mean, std, device)
            features.append(model.embedding(images).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features).astype(np.float32), np.concatenate(labels).astype(np.int64)

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean())

def train_one_classifier(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    args: argparse.Namespace,
    artifact_dir: Path,
) -> dict[str, Any]:
    # Identical classifier initialization and minibatch RNG for every encoder.
    seed_everything(CLASSIFIER_SEED)
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    feature_std[feature_std < 1e-6] = 1.0

    def standardized(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(((values - feature_mean) / feature_std).astype(np.float32))

    train_x = standardized(train_features)
    train_y = torch.from_numpy(train_labels).long()
    validation_x = standardized(validation_features).to(args.resolved_device)
    validation_y = torch.from_numpy(validation_labels).long().to(args.resolved_device)
    loader_generator = torch.Generator(device="cpu").manual_seed(CLASSIFIER_SEED + 1)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.num_workers,
        pin_memory=args.resolved_device.type == "cuda",
    )
    head = nn.Linear(train_x.shape[1], 10).to(args.resolved_device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total_count = 0
        for features, labels in loader:
            features = features.to(args.resolved_device, non_blocking=True)
            labels = labels.to(args.resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            total_count += len(features)
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
            validation_loss = float(nn.functional.cross_entropy(validation_logits, validation_y))
            validation_accuracy = accuracy(validation_logits, validation_y)
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in head.state_dict().items()
            }
        else:
            stale += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_count,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "best_epoch": best_epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds_elapsed": time.time() - started,
            }
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)
        write_rows(artifact_dir / "history.csv", history)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    if best_state is None:
        raise AssertionError("Classifier never produced a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    with torch.inference_mode():
        validation_logits = head(validation_x)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_rows(artifact_dir / "history.csv", history)
    atomic_torch_save(
        {
            "head_state_dict": best_state,
            "feature_mean": torch.from_numpy(feature_mean),
            "feature_std": torch.from_numpy(feature_std),
            "best_epoch": best_epoch,
            "classifier_seed": CLASSIFIER_SEED,
            "test_data_opened": False,
        },
        artifact_dir / "best_head.pt",
    )
    return {
        "classifier_seed": CLASSIFIER_SEED,
        "classifier_best_epoch": best_epoch,
        "classifier_epochs_ran": history[-1]["epoch"],
        "classifier_patience_gap": history[-1]["epoch"] - best_epoch,
        "validation_loss": float(nn.functional.cross_entropy(validation_logits, validation_y)),
        "validation_accuracy": accuracy(validation_logits, validation_y),
        "runtime_seconds": time.time() - started,
        "test_data_opened": False,
    }
