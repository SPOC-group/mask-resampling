"""Shared, dependency-light utilities for the CIFAR-10 convolutional MAE runs.

The module deliberately does not depend on torchvision: the experimental host
already has the official ``cifar-10-batches-py`` files but not torchvision.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


IMAGE_SIZE = 32
CHANNELS = 3
N_CLASSES = 10
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "resolved_device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    else:
        metadata["device_name"] = "CPU"
    return metadata


def _read_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    data = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(payload.get(b"labels", payload.get(b"fine_labels")), dtype=np.int64)
    return data, labels


def _resolve_cifar10_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if (root / "cifar-10-batches-py").is_dir():
        root = root / "cifar-10-batches-py"
    return root


def load_cifar10_train(data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the official CIFAR-10 training batches.

    This function deliberately never opens ``test_batch``.  It is used by
    strict train/validation pipelines whose test data must remain physically
    untouched until every model-selection decision has been finalized.
    """
    root = _resolve_cifar10_root(data_root)
    required = [root / f"data_batch_{i}" for i in range(1, 6)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CIFAR-10 batch files: " + ", ".join(missing))
    train_parts = [_read_cifar_batch(root / f"data_batch_{i}") for i in range(1, 6)]
    train_images = np.concatenate([part[0] for part in train_parts])
    train_labels = np.concatenate([part[1] for part in train_parts])
    if train_images.shape != (50_000, 3, 32, 32):
        raise ValueError(f"Unexpected CIFAR training shape: {train_images.shape}")
    return train_images, train_labels


def load_cifar10_test(data_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the official CIFAR-10 test batch for final evaluation."""
    root = _resolve_cifar10_root(data_root)
    path = root / "test_batch"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CIFAR-10 test batch: {path}")
    test_images, test_labels = _read_cifar_batch(root / "test_batch")
    if test_images.shape != (10_000, 3, 32, 32):
        raise ValueError(f"Unexpected CIFAR test shape: {test_images.shape}")
    return test_images, test_labels


def load_cifar10(data_root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load official CIFAR-10 Python batches as uint8 NCHW arrays."""
    train_images, train_labels = load_cifar10_train(data_root)
    test_images, test_labels = load_cifar10_test(data_root)
    return train_images, train_labels, test_images, test_labels


def balanced_splits(
    labels: np.ndarray,
    seed: int = 2026,
    pretrain_per_class: int = 4000,
    validation_per_class: int = 500,
    probe_per_class: int = 500,
) -> dict[str, np.ndarray]:
    requested = pretrain_per_class + validation_per_class + probe_per_class
    rng = np.random.default_rng(seed)
    result: dict[str, list[np.ndarray]] = {"pretrain": [], "validation": [], "probe": []}
    for class_id in range(N_CLASSES):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) < requested:
            raise ValueError(f"Class {class_id} has {len(candidates)} examples, need {requested}")
        shuffled = rng.permutation(candidates)
        a = pretrain_per_class
        b = a + validation_per_class
        result["pretrain"].append(shuffled[:a])
        result["validation"].append(shuffled[a:b])
        result["probe"].append(shuffled[b:requested])
    arrays: dict[str, np.ndarray] = {}
    for offset, (name, chunks) in enumerate(result.items()):
        values = np.concatenate(chunks).astype(np.int64)
        arrays[name] = np.random.default_rng(seed + 100 + offset).permutation(values)
    validate_splits(arrays, labels, pretrain_per_class, validation_per_class, probe_per_class)
    return arrays


def validate_splits(
    splits: dict[str, np.ndarray],
    labels: np.ndarray,
    pretrain_per_class: int,
    validation_per_class: int,
    probe_per_class: int,
) -> None:
    expected = {
        "pretrain": pretrain_per_class,
        "validation": validation_per_class,
        "probe": probe_per_class,
    }
    sets: dict[str, set[int]] = {}
    for name, per_class in expected.items():
        values = np.asarray(splits[name], dtype=np.int64)
        counts = np.bincount(labels[values], minlength=N_CLASSES)
        if not np.all(counts == per_class):
            raise ValueError(f"Unbalanced {name} split: {counts.tolist()}")
        sets[name] = set(values.tolist())
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if sets[left].intersection(sets[right]):
                raise ValueError(f"Splits {left} and {right} overlap")


def load_or_create_splits(
    path: Path,
    labels: np.ndarray,
    seed: int,
    pretrain_per_class: int,
    validation_per_class: int,
    probe_per_class: int,
) -> dict[str, np.ndarray]:
    if path.is_file():
        loaded = np.load(path)
        splits = {name: loaded[name].astype(np.int64) for name in ("pretrain", "validation", "probe")}
        validate_splits(splits, labels, pretrain_per_class, validation_per_class, probe_per_class)
        return splits
    splits = balanced_splits(
        labels,
        seed=seed,
        pretrain_per_class=pretrain_per_class,
        validation_per_class=validation_per_class,
        probe_per_class=probe_per_class,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **splits)
    return splits


def channel_statistics(images: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Float64 accumulation keeps this stable while avoiding a second 500 MB image copy.
    sums = np.zeros(CHANNELS, dtype=np.float64)
    square_sums = np.zeros(CHANNELS, dtype=np.float64)
    count = 0
    chunk_size = 1024
    for start in range(0, len(indices), chunk_size):
        chunk = images[indices[start : start + chunk_size]].astype(np.float64) / 255.0
        sums += chunk.sum(axis=(0, 2, 3))
        square_sums += np.square(chunk).sum(axis=(0, 2, 3))
        count += chunk.shape[0] * IMAGE_SIZE * IMAGE_SIZE
    mean = sums / count
    variance = np.maximum(square_sums / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    if np.any(std <= 0):
        raise ValueError(f"Invalid channel standard deviations: {std}")
    return mean.astype(np.float32), std.astype(np.float32)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(value):
        value = asdict(value)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def rho_token(rho: float) -> str:
    return f"{rho:g}".replace("-", "m").replace(".", "p")


def stable_seed(*values: int) -> int:
    """Mix small integer configuration fields into a reproducible 31-bit seed."""
    state = 0x345678
    for value in values:
        state = ((state ^ int(value)) * 1_000_003) & 0x7FFFFFFF
    return state


class IndexedTensorDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.images = torch.from_numpy(np.ascontiguousarray(images[indices]))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels[indices])).long()
        self.positions = torch.arange(len(indices), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index], self.positions[index]


def normalize_batch(
    raw_images: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    raw = raw_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    return (raw - mean) / std


def make_exact_patch_masks(
    batch_size: int,
    patch_size: int,
    rho: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, int, float]:
    grid = IMAGE_SIZE // patch_size
    patches = grid * grid
    masked_count = int(round(float(rho) * patches))
    if masked_count < 1 or masked_count >= patches:
        raise ValueError(f"rho={rho} realizes {masked_count}/{patches}; masked MAE needs 1..{patches - 1}")
    scores = torch.rand((batch_size, patches), generator=generator, device=device)
    selected = scores.topk(masked_count, dim=1, largest=False).indices
    patch_mask = torch.zeros((batch_size, patches), dtype=torch.bool, device=device)
    patch_mask.scatter_(1, selected, True)
    return patch_mask, masked_count, masked_count / patches


def expand_patch_masks(patch_masks: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch_size, patches = patch_masks.shape
    grid = int(round(patches**0.5))
    if grid * grid != patches or grid * patch_size != IMAGE_SIZE:
        raise ValueError("Patch-mask shape is inconsistent with image and patch sizes")
    return (
        patch_masks.view(batch_size, 1, grid, grid)
        .repeat_interleave(patch_size, dim=2)
        .repeat_interleave(patch_size, dim=3)
    )


class ConvMAE(nn.Module):
    """The exact 854,563-parameter architecture described in the handoff."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def encode(self, model_images: torch.Tensor, pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if pixel_mask is None:
            pixel_mask = torch.zeros(
                (len(model_images), 1, IMAGE_SIZE, IMAGE_SIZE),
                dtype=torch.bool,
                device=model_images.device,
            )
        masked = model_images.masked_fill(pixel_mask, 0.0)
        return self.encoder(torch.cat((masked, pixel_mask.to(model_images.dtype)), dim=1))

    def embedding(self, model_images: torch.Tensor) -> torch.Tensor:
        return self.encode(model_images).mean(dim=(2, 3))

    def forward(self, model_images: torch.Tensor, pixel_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(self.encode(model_images, pixel_mask))


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, pixel_mask: torch.Tensor | None) -> torch.Tensor:
    squared = torch.square(prediction - target)
    if pixel_mask is None:
        return squared.mean()
    denominator = target.shape[1] * pixel_mask.sum().clamp_min(1)
    return (squared * pixel_mask).sum() / denominator


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
