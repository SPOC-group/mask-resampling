#!/usr/bin/env python3
"""Fixed-K-mask control experiments for the Figure 1 CIFAR-10 protocol.

Each original pretraining image is paired with exactly K independently sampled,
frozen patch masks.  An epoch traverses all N*K fixed image-mask pairs.  K=1 is
therefore the original static objective, while increasing K gives a finite-view
approximation to dynamic masking without resampling masks during optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Sampler

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


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_ROOT = Path("runs/cnn/protocols/seed0/protocol")
FIGURE1_RHOS = (0.1, 0.171875, 0.25, 0.375, 0.5, 0.625, 0.75, 0.828125, 0.9)
MASK_BANK_CHUNK_PAIRS = 32_768
PIL_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
PIL_FLIP_LEFT_RIGHT = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


@dataclass(frozen=True)
class KMaskCondition:
    ae_seed: int
    masks_per_sample: int
    rho: float
    architecture: str = "conv"

    @property
    def schedule(self) -> str:
        return "fixed_k_mask"

    @property
    def name(self) -> str:
        architecture = "" if self.architecture == "conv" else f"{self.architecture}_"
        return (
            f"seed{self.ae_seed}_{architecture}fixed_k{self.masks_per_sample}_"
            f"rho{rho_token(self.rho)}"
        )


class FixedKMaskViewDataset(Dataset):
    """Store N originals once and expose a virtual view-major N*K index.

    Pair index ``view*N + sample`` makes the generated mask bank nested in K:
    the K=2 bank is an exact prefix of K=4, then K=8, K=16, K=32, and K=64. Its first N rows
    also exactly reproduce the original K=1 static mask bank.  No image tensor is
    expanded or repeated in memory; only a batch of selected originals is collated.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        masks_per_sample: int,
    ) -> None:
        self.images = torch.from_numpy(np.ascontiguousarray(images[indices]))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels[indices])).long()
        self.samples = len(indices)
        self.masks_per_sample = masks_per_sample

    def __len__(self) -> int:
        return self.samples * self.masks_per_sample

    def __getitem__(self, pair_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_index = pair_index % self.samples
        return self.images[sample_index], self.labels[sample_index], pair_index


class DynamicAugmentedFixedKMaskViewDataset(FixedKMaskViewDataset):
    """Fixed masks with a freshly augmented image view on every epoch.

    Augmentation is stateless in ``(seed, epoch, pair_index)``. This provides
    exact resume behavior while changing the crop and flip across epochs. The
    fixed Boolean mask selected by ``pair_index`` never changes.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        masks_per_sample: int,
        augmentation_seed: int,
        scale: tuple[float, float] = (0.2, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        horizontal_flip_probability: float = 0.5,
    ) -> None:
        super().__init__(images, labels, indices, masks_per_sample)
        self.augmentation_seed = int(augmentation_seed)
        self.scale = tuple(float(value) for value in scale)
        self.ratio = tuple(float(value) for value in ratio)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _random_resized_crop_parameters(
        self, rng: np.random.Generator
    ) -> tuple[int, int, int, int]:
        """Mirror torchvision RandomResizedCrop's area/aspect sampling."""

        height = width = IMAGE_SIZE
        area = height * width
        log_ratio = np.log(np.asarray(self.ratio, dtype=np.float64))
        for _ in range(10):
            target_area = area * rng.uniform(self.scale[0], self.scale[1])
            aspect_ratio = float(np.exp(rng.uniform(log_ratio[0], log_ratio[1])))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = int(rng.integers(0, height - crop_height + 1))
                left = int(rng.integers(0, width - crop_width + 1))
                return top, left, crop_height, crop_width

        input_ratio = width / height
        if input_ratio < self.ratio[0]:
            crop_width = width
            crop_height = int(round(crop_width / self.ratio[0]))
        elif input_ratio > self.ratio[1]:
            crop_height = height
            crop_width = int(round(crop_height * self.ratio[1]))
        else:
            crop_width = width
            crop_height = height
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        return top, left, crop_height, crop_width

    def __getitem__(self, pair_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_index = pair_index % self.samples
        rng = np.random.default_rng(
            stable_seed(self.augmentation_seed, self.epoch, int(pair_index))
        )
        top, left, crop_height, crop_width = self._random_resized_crop_parameters(rng)
        array = np.ascontiguousarray(
            self.images[sample_index].permute(1, 2, 0).numpy()
        )
        image = Image.fromarray(array, mode="RGB")
        image = image.crop(
            (left, top, left + crop_width, top + crop_height)
        ).resize((IMAGE_SIZE, IMAGE_SIZE), resample=PIL_BICUBIC)
        if rng.random() < self.horizontal_flip_probability:
            image = image.transpose(PIL_FLIP_LEFT_RIGHT)
        augmented = torch.from_numpy(
            np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
        )
        return augmented, self.labels[sample_index], pair_index


class UniqueImageFixedKBatchSampler(Sampler[list[int]]):
    """Visit every fixed pair once while forbidding repeated images per batch.

    Each mask view receives an independently shuffled permutation of the N
    original images. At boundaries between views, the minimum necessary swaps
    prevent the partial batch from receiving an image already present in it.
    """

    def __init__(
        self,
        samples: int,
        masks_per_sample: int,
        batch_size: int,
        generator: torch.Generator,
    ) -> None:
        if batch_size > samples:
            raise ValueError(
                "A unique-image batch cannot be larger than the original-image pool"
            )
        self.samples = samples
        self.masks_per_sample = masks_per_sample
        self.batch_size = batch_size
        self.generator = generator

    def __len__(self) -> int:
        return math.ceil(self.samples * self.masks_per_sample / self.batch_size)

    def __iter__(self):
        pending: list[int] = []
        pairs_yielded = 0
        view_order = torch.randperm(
            self.masks_per_sample, generator=self.generator
        ).tolist()
        for view_index in view_order:
            image_order = torch.randperm(self.samples, generator=self.generator)
            if pending:
                needed = self.batch_size - len(pending)
                forbidden = torch.zeros(self.samples, dtype=torch.bool)
                forbidden[
                    torch.tensor(
                        [pair_index % self.samples for pair_index in pending],
                        dtype=torch.long,
                    )
                ] = True
                collision_positions = torch.nonzero(
                    forbidden[image_order[:needed]], as_tuple=False
                ).flatten()
                collision_count = len(collision_positions)
                if collision_count:
                    safe_later = torch.nonzero(
                        ~forbidden[image_order[needed:]], as_tuple=False
                    ).flatten()[:collision_count] + needed
                    if len(safe_later) != collision_count:
                        raise AssertionError("Could not repair a cross-view batch boundary")
                    displaced = image_order[collision_positions].clone()
                    image_order[collision_positions] = image_order[safe_later]
                    image_order[safe_later] = displaced

            for sample_index in image_order.tolist():
                pending.append(view_index * self.samples + sample_index)
                if len(pending) == self.batch_size:
                    original_ids = [pair_index % self.samples for pair_index in pending]
                    if len(set(original_ids)) != len(original_ids):
                        raise AssertionError("A training batch contains a repeated image")
                    pairs_yielded += len(pending)
                    yield pending
                    pending = []

        if pending:
            original_ids = [pair_index % self.samples for pair_index in pending]
            if len(set(original_ids)) != len(original_ids):
                raise AssertionError("The final training batch contains a repeated image")
            pairs_yielded += len(pending)
            yield pending
        expected_pairs = self.samples * self.masks_per_sample
        if pairs_yielded != expected_pairs:
            raise AssertionError(
                f"Unique-image sampler yielded {pairs_yielded} pairs, expected {expected_pairs}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_boolean_patch_bank(
    bank: np.ndarray,
    masks_per_sample: int,
    samples: int,
    patches: int,
    masked_count: int,
) -> None:
    expected_shape = (masks_per_sample, samples, patches)
    if bank.shape != expected_shape:
        raise AssertionError(
            f"Fixed patch bank has shape {bank.shape}, expected {expected_shape}"
        )
    if bank.dtype != np.bool_:
        raise AssertionError(f"Fixed patch bank has dtype {bank.dtype}, expected bool")
    flat_bank = bank.reshape(-1, patches)
    for start in range(0, len(flat_bank), MASK_BANK_CHUNK_PAIRS):
        stop = min(start + MASK_BANK_CHUNK_PAIRS, len(flat_bank))
        if not np.all(np.count_nonzero(flat_bank[start:stop], axis=1) == masked_count):
            raise AssertionError("Fixed patch bank contains an incorrect masked-patch count")


def create_or_load_boolean_patch_bank(
    condition_dir: Path,
    samples: int,
    masks_per_sample: int,
    patch_size: int,
    rho: float,
    seed: int,
) -> tuple[np.ndarray, int, float, Path]:
    """Persist only K*N patch masks as bool and memory-map them on reuse.

    Masks are stored as ``[K, N, P]`` in view-major order. Generation is
    chunked, so temporary floating-point random-score memory is bounded rather
    than scaling as K*N*P. Each row still contains exactly round(rho*P) masked
    patches, preserving the existing fixed-rho protocol rather than changing it
    to an unconditioned Bernoulli draw with a fluctuating patch count.
    """

    grid = IMAGE_SIZE // patch_size
    patches = grid * grid
    masked_count = int(round(float(rho) * patches))
    if masked_count < 1 or masked_count >= patches:
        raise ValueError(
            f"rho={rho} realizes {masked_count}/{patches}; masked MAE needs 1..{patches - 1}"
        )
    realized_rho = masked_count / patches
    bank_path = condition_dir / "fixed_patch_masks.bool.npy"
    metadata_path = condition_dir / "fixed_patch_masks.json"
    expected = {
        "format": "numpy_bool_memmap",
        "layout": "view_major_[K,N,P]",
        "masks_per_sample": masks_per_sample,
        "samples": samples,
        "patches_per_image": patches,
        "masked_patch_count": masked_count,
        "requested_rho": float(rho),
        "realized_rho": realized_rho,
        "seed": seed,
        "dtype": "bool",
        "shape": [masks_per_sample, samples, patches],
    }

    if bank_path.is_file() or metadata_path.is_file():
        if not bank_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(
                f"Incomplete fixed-mask cache: expected both {bank_path} and {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"Fixed-mask cache metadata mismatch for {key}: "
                    f"found {metadata.get(key)!r}, expected {value!r}"
                )
        actual_sha256 = _sha256_file(bank_path)
        if metadata.get("sha256") != actual_sha256:
            raise RuntimeError(f"Fixed-mask cache checksum mismatch: {bank_path}")
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        _validate_boolean_patch_bank(
            bank, masks_per_sample, samples, patches, masked_count
        )
        return bank, masked_count, realized_rho, bank_path

    temporary_path = bank_path.with_name(bank_path.name + ".tmp")
    bank_writer = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.bool_,
        shape=(masks_per_sample, samples, patches),
    )
    flat_writer = bank_writer.reshape(-1, patches)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_pairs = samples * masks_per_sample
    for start in range(0, total_pairs, MASK_BANK_CHUNK_PAIRS):
        stop = min(start + MASK_BANK_CHUNK_PAIRS, total_pairs)
        block, block_count, block_rho = make_exact_patch_masks(
            stop - start,
            patch_size,
            rho,
            generator,
            torch.device("cpu"),
        )
        if block_count != masked_count or not math.isclose(block_rho, realized_rho):
            raise AssertionError("Chunked mask generation changed the mask realization")
        flat_writer[start:stop] = block.numpy()
    bank_writer.flush()
    del flat_writer
    del bank_writer
    os.replace(temporary_path, bank_path)

    metadata = {
        **expected,
        "pairs": total_pairs,
        "storage_bytes": bank_path.stat().st_size,
        "generation_chunk_pairs": MASK_BANK_CHUNK_PAIRS,
        "sha256": _sha256_file(bank_path),
    }
    save_json(metadata_path, metadata)
    bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
    _validate_boolean_patch_bank(bank, masks_per_sample, samples, patches, masked_count)
    return bank, masked_count, realized_rho, bank_path


def gather_boolean_patch_masks(
    bank: np.ndarray,
    pair_positions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Materialize only the Boolean patch masks needed by the current batch."""

    patches = bank.shape[-1]
    flat_bank = bank.reshape(-1, patches)
    selected = np.array(
        flat_bank[pair_positions.detach().cpu().numpy()],
        dtype=np.bool_,
        order="C",
        copy=True,
    )
    return torch.from_numpy(selected).to(device=device, non_blocking=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--masks-per-sample", type=int, required=True)
    parser.add_argument("--rhos", type=float, nargs="+", default=list(FIGURE1_RHOS))
    parser.add_argument("--ae-seed", type=int, default=0)
    parser.add_argument(
        "--architecture",
        choices=("conv",),
        default="conv",
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=460)
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
    parser.add_argument(
        "--unique-images-per-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Forbid two fixed-mask views of the same original image in one training batch.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-complete", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-validation-images", type=int, default=None)
    parser.add_argument(
        "--dynamic-train-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resample U-MAE-style RandomResizedCrop(scale=0.2..1.0) and "
            "horizontal flip for every training image every epoch; masks stay fixed."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.masks_per_sample < 1:
        raise ValueError("masks-per-sample must be positive")
    if args.unique_images_per_batch and args.batch_size > 40_000:
        raise ValueError("unique-images-per-batch requires batch-size <= 40000")
    if IMAGE_SIZE % args.patch_size:
        raise ValueError("patch-size must divide 32")
    if args.epochs < 1 or args.min_epochs < 1 or args.patience < 1:
        raise ValueError("epochs, min-epochs, and patience must be positive")
    if args.min_epochs > args.epochs:
        raise ValueError("min-epochs cannot exceed epochs")
    if len(set(args.rhos)) != len(args.rhos):
        raise ValueError("rhos must not contain duplicates")
    if any(not 0 < rho < 1 for rho in args.rhos):
        raise ValueError("rhos must lie strictly between zero and one")
    if args.dynamic_train_augmentation and args.architecture != "conv":
        raise ValueError("The dynamic-augmentation control is restricted to the old CNN")


def condition_seeds(condition: KMaskCondition) -> dict[str, int]:
    # These deliberately equal the Figure 1 K=1 static seeds.  K is omitted so
    # initialization is matched and the fixed mask banks are nested across K.
    return {
        "model": stable_seed(condition.ae_seed, 1),
        "shuffle": stable_seed(condition.ae_seed, 2),
        "fixed_mask": stable_seed(condition.ae_seed, 4),
        "validation_mask": stable_seed(condition.ae_seed, 5),
        "augmentation": stable_seed(condition.ae_seed, 6),
    }


def make_autoencoder(architecture: str) -> ConvMAE:
    if architecture != "conv":
        raise ValueError("This package contains the convolutional autoencoder only")
    return ConvMAE()


def expected_parameter_count(architecture: str) -> int:
    if architecture != "conv":
        raise ValueError("This package contains the convolutional autoencoder only")
    return 854_563


def evaluate_fixed_pairs(
    model: ConvMAE,
    loader: DataLoader,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    patch_bank: np.ndarray,
    patch_size: int,
) -> float:
    model.eval()
    loss_numerator = 0.0
    pairs = 0
    with torch.inference_mode():
        for raw, _labels, pair_positions in loader:
            target = normalize_batch(raw, mean, std, device)
            patch_mask = gather_boolean_patch_masks(patch_bank, pair_positions, device)
            pixel_mask = expand_patch_masks(patch_mask, patch_size)
            prediction = model(target, pixel_mask)
            batch_loss = (
                reconstruction_loss(prediction, target, pixel_mask)
            )
            loss_numerator += float(batch_loss) * len(raw)
            pairs += len(raw)
    return loss_numerator / max(pairs, 1)


def train_condition(
    condition: KMaskCondition,
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
    if args.unique_images_per_batch:
        unique_batch_sampler = UniqueImageFixedKBatchSampler(
            train_dataset.samples,
            condition.masks_per_sample,
            args.batch_size,
            loader_generator,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=unique_batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
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

    fixed_bank, masked_count, realized_rho, fixed_bank_path = (
        create_or_load_boolean_patch_bank(
            condition_dir,
            train_dataset.samples,
            condition.masks_per_sample,
            args.patch_size,
            condition.rho,
            seeds["fixed_mask"],
        )
    )
    validation_bank, validation_count, validation_rho = create_fixed_patch_bank(
        len(validation_dataset), args.patch_size, condition.rho, seeds["validation_mask"]
    )
    if masked_count != validation_count or not math.isclose(realized_rho, validation_rho):
        raise AssertionError("Training and validation mask realization mismatch")
    model = make_autoencoder(condition.architecture).to(device)
    expected_parameters = expected_parameter_count(condition.architecture)
    if count_parameters(model) != expected_parameters:
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
        loader_generator.set_state(
            checkpoint["loader_generator_state"].detach().to(device="cpu", dtype=torch.uint8)
        )
        history = checkpoint["history"]
        best_loss = float(checkpoint["best_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] {condition.name} from epoch {start_epoch}", flush=True)

    condition_metadata = {
        **asdict(condition),
        "name": condition.name,
        "schedule": condition.schedule,
        "seeds": seeds,
        "control": "K independently sampled masks per image, frozen for the entire run",
        "architecture": condition.architecture,
        "encoder_decoder_skip_connections": False,
        "residual_connections": "inside ResNet-18 encoder only",
        "mask_bank_order": "view-major; banks are nested across K=1,2,4,8,16,32,64",
        "fixed_mask_storage": "separate memory-mapped NumPy bool patch bank",
        "fixed_mask_bank": str(fixed_bank_path.resolve()),
        "fixed_mask_bank_shape": list(fixed_bank.shape),
        "fixed_mask_bank_storage_bytes": fixed_bank_path.stat().st_size,
        "mask_generation_chunk_pairs": MASK_BANK_CHUNK_PAIRS,
        "original_training_images": train_dataset.samples,
        "fixed_masks_per_sample": condition.masks_per_sample,
        "training_pairs_per_epoch": len(train_dataset),
        "unique_original_images_per_training_batch": args.unique_images_per_batch,
        "training_batch_sampling": (
            "all N*K pairs once per epoch; independently permuted image order per mask view; "
            "cross-view boundaries repaired to prohibit repeated original images"
            if args.unique_images_per_batch
            else "global random permutation of all N*K image-mask pairs"
        ),
        "validation_masks_per_sample": 1,
        "training_image_augmentation": (
            "dynamic_random_resized_crop_32_scale_0p2_1p0_bicubic_plus_horizontal_flip_p0p5"
            if args.dynamic_train_augmentation
            else "none"
        ),
        "augmentation_resampling": (
            "fresh deterministic draw per (seed, epoch, image-mask pair)"
            if args.dynamic_train_augmentation
            else "not_applicable"
        ),
        "fixed_mask_relative_to_dynamic_augmentation": (
            "same spatial mask coordinates every epoch, applied after augmentation"
            if args.dynamic_train_augmentation
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
        "device": str(device),
        "device_metadata": device_metadata(device),
        "parameter_count": count_parameters(model),
        "official_test_batch_opened": False,
    }
    save_json(condition_dir / "config.json", condition_metadata)
    save_json(status_path, {"state": "running", "condition": condition_metadata})

    train_started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        train_sum = 0.0
        train_pairs = 0
        for raw, _labels, pair_positions in train_loader:
            target = normalize_batch(raw, mean, std, device)
            patch_mask = gather_boolean_patch_masks(fixed_bank, pair_positions, device)
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

        if train_pairs != len(train_dataset):
            raise AssertionError(f"Epoch saw {train_pairs} pairs, expected {len(train_dataset)}")
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
        scheduler.step(validation_loss)
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
                "epoch": epoch,
                "optimizer_steps_this_epoch": math.ceil(train_pairs / args.batch_size),
                "training_pairs": train_pairs,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "seconds_elapsed": time.time() - train_started,
            }
        )
        write_rows(condition_dir / "history.csv", history)
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loader_generator_state": loader_generator.get_state(),
                "history": history,
                "epoch": epoch,
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            },
            last_path,
        )
        print(
            f"[{condition.name}] epoch={epoch:03d} pairs={train_pairs} "
            f"train={train_loss:.6f} val={validation_loss:.6f} "
            f"best={best_loss:.6f}@{best_epoch} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        if epoch >= args.min_epochs and stale_epochs >= args.patience:
            break

    best_checkpoint = torch.load(
        condition_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    validation_mse = evaluate(
        model,
        validation_loader,
        mean,
        std,
        device,
        validation_bank,
        args.patch_size,
    )
    ordered_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    reused_train_mse = evaluate_fixed_pairs(
        model,
        ordered_train_loader,
        mean,
        std,
        device,
        fixed_bank,
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
        "ae_seed": condition.ae_seed,
        "schedule": condition.schedule,
        "architecture": condition.architecture,
        "rho": condition.rho,
        "condition": condition.name,
        "masks_per_sample": condition.masks_per_sample,
        "original_training_images": train_dataset.samples,
        "training_pairs_per_epoch": len(train_dataset),
        "unique_original_images_per_training_batch": args.unique_images_per_batch,
        "requested_rho": condition.rho,
        "realized_rho": realized_rho,
        "masked_patch_count": masked_count,
        "best_epoch": best_epoch,
        "best_optimizer_steps": best_epoch * math.ceil(len(train_dataset) / args.batch_size),
        "epochs_ran": history[-1]["epoch"],
        "optimizer_steps_ran": history[-1]["epoch"]
        * math.ceil(len(train_dataset) / args.batch_size),
        "hit_epoch_ceiling": history[-1]["epoch"] == args.epochs,
        "fresh_validation_mse": validation_mse,
        "fixed_k_reused_train_mse": reused_train_mse,
        "dynamic_train_augmentation": args.dynamic_train_augmentation,
        "runtime_seconds": time.time() - train_started,
        "checkpoint": str((condition_dir / "best.pt").resolve()),
        "official_test_batch_opened": False,
    }
    save_json(
        status_path,
        {"state": "complete", "condition": condition_metadata, "summary": summary},
    )
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
            "experiment": "Figure 1 fixed-K-mask control",
            "training_objective_size": f"40000*{args.masks_per_sample}",
            "mask_resampling_during_training": False,
            "mask_banks_nested_across_k": True,
            "validation_masks_per_sample": 1,
            "official_test_batch_opened": False,
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
    if args.dynamic_train_augmentation:
        train_dataset = DynamicAugmentedFixedKMaskViewDataset(
            train_images,
            train_labels,
            train_indices,
            args.masks_per_sample,
            augmentation_seed=stable_seed(args.ae_seed, 6),
        )
    else:
        train_dataset = FixedKMaskViewDataset(
            train_images,
            train_labels,
            train_indices,
            args.masks_per_sample,
        )
    validation_dataset = IndexedTensorDataset(
        train_images, train_labels, validation_indices
    )

    summaries: list[dict[str, Any]] = []
    for rho in args.rhos:
        condition = KMaskCondition(
            args.ae_seed, args.masks_per_sample, rho, args.architecture
        )
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
    print(
        f"completed {len(summaries)} K={args.masks_per_sample} condition(s); "
        f"summary: {output_dir / 'pretraining_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
