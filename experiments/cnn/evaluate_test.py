#!/usr/bin/env python3
"""Explicit final-only evaluation of a frozen encoder/head on official CIFAR-10 test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


from cifar10_conv_mae_core import (  # noqa: E402
    ConvMAE, IndexedTensorDataset, load_cifar10_test, normalize_batch,
    resolve_device, save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--probe-dir", type=Path, help="Probe directory for a specified full-reconstruction epoch")
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint after relocating a completed run")
    parser.add_argument("--expected-updates", type=int, default=None)
    args = parser.parse_args()
    probe_dir = args.probe_dir or args.run_dir / "probe"
    probe_status = json.loads((probe_dir / "status.json").read_text())
    if probe_status.get("state") != "complete":
        raise RuntimeError("Probe must finish before official-test evaluation")
    checkpoint_path = args.checkpoint or Path(probe_status["summary"]["pretraining_checkpoint"])
    if not checkpoint_path.is_file():
        if args.expected_updates is not None:
            raise FileNotFoundError(f"Required fixed-update checkpoint is missing: {checkpoint_path}")
        # Supports moving a completed run directory between machines.
        matches = list((args.run_dir / "pretraining" / "conditions").glob("*/best.pt"))
        if len(matches) != 1:
            raise FileNotFoundError("Cannot locate the pretrained best.pt")
        checkpoint_path = matches[0]
    device = resolve_device(args.device)
    pretrained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.expected_updates is not None:
        if pretrained.get("optimizer_steps") != args.expected_updates or probe_status["summary"].get("pretraining_optimizer_updates") != args.expected_updates:
            raise ValueError("Encoder and probe must match the requested optimizer update count")
    head_checkpoint = torch.load(probe_dir / "best_head.pt", map_location="cpu", weights_only=False)
    model = ConvMAE().to(device).eval()
    model.load_state_dict(pretrained["model_state_dict"])
    model.requires_grad_(False)
    head = nn.Linear(128, 10).to(device).eval()
    head.load_state_dict(head_checkpoint["head_state_dict"])
    mean = pretrained["mean"].to(device)
    std = pretrained["std"].to(device)
    feature_mean = head_checkpoint["feature_mean"].to(device)
    feature_std = head_checkpoint["feature_std"].to(device)
    test_images, test_labels = load_cifar10_test(args.data_root)
    loader = DataLoader(
        IndexedTensorDataset(test_images, test_labels, list(range(10_000))),
        batch_size=args.batch_size, shuffle=False,
    )
    correct = 0
    total = 0
    with torch.inference_mode():
        for raw, targets, _ in loader:
            features = model.embedding(normalize_batch(raw, mean, std, device))
            logits = head((features - feature_mean) / feature_std)
            correct += int((logits.argmax(1) == targets.to(device)).sum())
            total += len(raw)
    if total != 10_000:
        raise AssertionError("Official CIFAR-10 test set has the wrong size")
    result = {
        "condition": probe_status["summary"]["condition"],
        "test_correct": correct,
        "test_total": total,
        "test_accuracy": correct / total,
        "test_error": (total - correct) / total,
        "pretraining_checkpoint": str(checkpoint_path.resolve()),
        "pretraining_optimizer_updates": pretrained.get("optimizer_steps"),
        "official_test_batch_opened": True,
        "model_selection_used_test_data": False,
    }
    save_json(probe_dir / "test_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
