"""Original CPU full-reconstruction validation functions."""
import sitecustomize
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import torch
import models_mae

def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)

def validation_fingerprint(dataset):
    root = Path(dataset.root)
    manifest = [
        {"path": str(Path(path).relative_to(root)), "label": int(label),
         "size_bytes": Path(path).stat().st_size}
        for path, label in dataset.samples
    ]
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()

def load_frozen_model(path, epoch, args):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if int(checkpoint.get("epoch", -1)) != epoch:
        raise ValueError("Checkpoint epoch mismatch: {}".format(path))
    model = models_mae.__dict__[args["model"]](norm_pix_loss=args["norm_pix_loss"])
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    gc.collect()
    model.eval().requires_grad_(False)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError("Non-finite checkpoint parameter: {}".format(name))
    return model

def weighted_mse(batches):
    total, count = 0.0, 0
    for value, samples in batches:
        if not math.isfinite(value) or value < 0 or samples <= 0:
            raise ValueError("Invalid reconstruction loss or batch sample count")
        total += value * samples
        count += samples
    if count == 0:
        raise ValueError("No validation samples evaluated")
    return total / count, count

@torch.inference_mode()
def evaluate(model, loader, epoch, status_path, benchmark_only=False):
    start = last_report = time.monotonic()
    batches = []
    seen = 0
    for index, (images, _) in enumerate(loader):
        loss, _, mask, _, _ = model(
            images, mask_ratio=0.0, mask_mode="none",
            reconstruction_loss_scope="all")
        if mask.any():
            raise ValueError("Unexpected masking during full-reconstruction validation")
        value = float(loss.item())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError("Non-finite or negative validation reconstruction MSE")
        samples = images.shape[0]
        batches.append((value, samples))
        seen += samples
        now = time.monotonic()
        if now - last_report >= 30 or seen == len(loader.dataset):
            elapsed = now - start
            progress = {
                "state": "running", "pretraining_epoch": epoch,
                "num_samples_evaluated": seen, "num_validation_samples": len(loader.dataset),
                "elapsed_seconds": elapsed,
                "eta_seconds": elapsed * (len(loader.dataset) - seen) / seen,
            }
            if status_path is not None:
                write_json(status_path, progress)
            print("Epoch {}: {}/{} images; {:.2f} images/s; ETA {:.1f} min".format(
                epoch, seen, len(loader.dataset), seen / elapsed,
                progress["eta_seconds"] / 60), flush=True)
            last_report = now
        if benchmark_only and index == 2:
            break
    mse, count = weighted_mse(batches)
    if not benchmark_only and count != len(loader.dataset):
        raise ValueError("Validation set was not evaluated in full")
    return mse, count, time.monotonic() - start

def load_cached_result(path, provenance):
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    for key, value in provenance.items():
        if result.get(key) != value:
            raise ValueError("Cached evaluation provenance mismatch: {} ({})".format(path, key))
    mse = float(result["validation_reconstruction_mse"])
    if int(result.get("num_samples_evaluated", -1)) != provenance["num_validation_samples"]:
        raise ValueError("Cached reconstruction MSE does not cover the full validation split")
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("Invalid cached reconstruction MSE")
    return result
