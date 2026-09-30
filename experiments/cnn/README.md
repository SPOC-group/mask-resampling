# CIFAR-10 convolutional autoencoder

Run all commands below from the repository root (`mask-resampling/`).

Original convolutional-autoencoder training, frozen linear probing, augmentation controls and optimization-budget controls. Use `run_cnn.py` to dispatch one independent seed/condition to a GPU. No training images or pretrained weights are distributed. Reproduction runs write checkpoints, Boolean mask banks, split indices and scalar histories to your chosen output directory; these are needed for resumption and probing. Training scripts do not generate figures. The plotting notebook remains the single plotting entry point.

## Setup

Use Python 3.11 in a separate environment. Install the pinned dependencies below, selecting a compatible PyTorch CUDA wheel for GPU training if needed. Place the official CIFAR-10 **Python version** in a local directory containing `cifar-10-batches-py/` (or pass that extracted directory directly). The scripts do not download data.

```bash
python -m pip install -r requirements/cnn.txt
python experiments/cnn/prepare_protocols.py --data-root /path/to/cifar10 --output-root runs/cnn/protocols
```

Preparation reads only the five CIFAR-10 training batches. The 40,000-image pretraining set and 5,000-image reconstruction-validation set follow the original balanced splits, with split seed `2026 + 1009*seed`. The remaining 5,000 training images join the reconstruction-validation set for frozen probing: **40,000 labeled training images and their 10,000-image complement for validation**. Channel normalization is fitted to the pretraining images only. The legacy split key `probe` denotes that remaining 5,000-image subset, not the probe's training set.

## Presets

Every preset uses seeds 0–4. `--list` prints every indexed condition without loading images or starting training.

| Preset | Tasks | Protocol |
|---|---:|---|
| `original` | 330 | K=1,2,4,8,16,32 over nine masking ratios; dynamic over eleven ratios; dynamic with crop/flip at rho=0.625. Original plateau scheduler and validation-selected encoders. |
| `augmentation` | 220 | K=1 and dynamic, each with and without crop/flip, over eleven masking ratios. All use 32 presentations per image per epoch and a plateau scheduler. |
| `fixed_lr` | 35 | K=1,2,4,8,16,32 and dynamic at rho=0.625; constant learning rate 0.001; validation-selected encoder. |
| `fixed_budget` | 35 | Same conditions and constant learning rate, but exactly 125,000 updates; pretraining early stopping disabled; probe the final encoder. |
| `early_diagnostics` | 35 | Separate first-epoch reruns of `fixed_lr`, recording batch losses and additional validation evaluations. No probe. |
| `full_reconstruction` | 5 | Unmasked reconstruction, 800 epochs per seed; frozen probes at 13 pretraining checkpoints. |

The nine masking ratios are `0.1, 0.171875, 0.25, 0.375, 0.5, 0.625, 0.75, 0.828125, 0.9`; the eleven-ratio grid adds `0.05, 0.95`.

```bash
python experiments/cnn/run_cnn.py --preset augmentation --list
python experiments/cnn/run_cnn.py --preset augmentation --task 0 \
  --data-root /path/to/cifar10 --output-root runs/cnn --device cuda

python experiments/cnn/run_cnn.py --preset fixed_lr --task 0 \
  --data-root /path/to/cifar10 --output-root runs/cnn --device cuda

python experiments/cnn/run_cnn.py --preset fixed_budget --task 0 \
  --data-root /path/to/cifar10 --output-root runs/cnn --device cuda
```

Replace the preset with `original`, `early_diagnostics` or `full_reconstruction` to run those experiments. By default a task runs pretraining followed by its frozen probe; `--stage train` and `--stage probe` separate them. `--dry-run` prints the training command. A full-reconstruction task trains one encoder trajectory and then probes its 13 checkpoints; it does not retrain the encoder 13 times.

Run different tasks concurrently on allocated GPUs. If `--task` is omitted, the runner reads `SLURM_ARRAY_TASK_ID`. For example, after preparing protocols, submit a cluster array of `0-34` for either learning-rate control and execute this command inside each GPU allocation:

```bash
python experiments/cnn/run_cnn.py --preset fixed_budget \
  --data-root /path/to/cifar10 --output-root runs/cnn --device cuda
```

Use one task per allocated GPU and site-appropriate resource requests. Existing completed pretraining and probes are reused with matching parameters. A diagnostic replay requires a fresh output directory. Use a separate output root when changing parameters. The low-level trainer filenames retain their original names for traceability; `reference-k=32` specifies presentations per epoch, not a finite mask bank for dynamic masking.

## Recorded settings

- Convolutional encoder/decoder: 854,563 parameters, 128-dimensional spatially averaged representation. Input size 32x32, patch size 4. A Boolean mask hides exactly `round(64*rho)` patches; this is fixed-count patch masking. Dynamic masks are freshly sampled at every presentation. Fixed-K masks are drawn once and reused, with nested banks across K.
- Pretraining: AdamW, batch size 512, initial learning rate 0.001, weight decay 0.0001, gradient clipping at 5, full precision. The original scheduler is `ReduceLROnPlateau`, factor 0.5, patience 5 epochs, minimum learning rate 0.00001. Validation selects the encoder by reconstruction MSE. Default minimum duration 20 epochs, stopping patience 15, maximum 460.
- In `original`, a finite-K epoch traverses 40,000K image-mask pairs; a dynamic epoch contains 32 presentations per image, or 2,500 updates. These epoch definitions and their scheduling intervals are retained. The augmentation and constant-learning-rate controls instead use 2,500 updates per epoch for **all** K.
- `fixed_lr`: learning rate stays at 0.001, stopping patience 50 epochs, maximum 460. `fixed_budget`: the same training setup for 50 epochs, exactly 125,000 updates and 64,000,000 image presentations; no pretraining early stopping. The runner checks these values before probing `update_125000.pt`.
- Augmentation: fresh random resized crop with scale [0.2,1], aspect ratio [3/4,4/3], bicubic resize, and horizontal flip probability 1/2. For static masking the patch positions stay fixed while the image transforms are resampled. Frozen probing uses unaugmented images.
- Frozen probe: standardize the 128 features using probe-training statistics, fit a 128-to-10 linear head with AdamW, learning rate 0.01, weight decay 0, batch size 512. Classifier seed 314159 is held fixed. Minimum 20 epochs, maximum 200, patience 25; select by validation cross-entropy. A probe that hits its ceiling without exhausting patience is flagged rather than silently accepted as complete. There is no joint encoder fine-tuning in this package.
- Full reconstruction preserves its separate original split seed `2026 + seed`: 40,000 training and 10,000 validation images. Train for 800 epochs, with the plateau scheduler and the same pretraining optimizer. Probe checkpoints at epochs `1,2,4,8,16,25,50,100,200,300,400,600,800`. Select the common pretraining epoch by maximum mean probe-validation accuracy across five seeds, breaking ties by smaller mean validation cross-entropy. The saved plotted experiment selected epoch 16; reproduction recomputes that choice from validation data.
- First-epoch diagnostics evaluate validation MSE at updates `0,50,100,250,500,1000,1500,2000,2500`. Extra evaluations preserve the training RNG streams and do not affect scheduling or checkpoint selection. These are separate reruns; GPU kernels can cause small differences from an earlier trajectory.

## Test evaluation and collected results

Official test images are opened only through explicit evaluation, after the preset's probes finish. The full-reconstruction validation choice is recorded before test evaluation.

```bash
python experiments/cnn/collect_cnn.py --preset fixed_budget --output-root runs/cnn \
  --evaluate --data-root /path/to/cifar10 --device cuda
```

Use the same command with `original`, `augmentation`, `fixed_lr` or `full_reconstruction`. Without `--evaluate`, collection only reads existing outputs. `--allow-partial` permits inspecting incomplete runs, reports their coverage and cannot be combined with official test evaluation. For diagnostics:

```bash
python experiments/cnn/collect_cnn.py --preset early_diagnostics --output-root runs/cnn
```

Collected files include `training_history.csv`, `early_diagnostics.csv`, `per_run_results.csv`, `aggregate_results.csv` and `coverage.json`, as applicable. Scalar results report the updates and presentations at the selected encoder and the total training updates. Aggregation reports both sample SD and SEM across independent encoder/split seeds. Incomplete conditions are identified explicitly.

The existing notebook reads the curated saved exports under `data/vision/cnn/`: `masking_ratio`, `mask_diversity` and `plateau_lr` correspond to `original`; `augmentation`, `fixed_lr`, `fixed_budget` and `full_reconstruction` correspond to their named presets. The early points in `fixed_lr` come from `early_diagnostics`. Newly collected results use a common tidy schema and do not overwrite those published inputs automatically.

The four `cifar10_conv_mae*.py` files retain the original numerical routines. Packaging removes unused ResNet routes and auxiliary image/plot exports. `probe_core.py` contains the original frozen-head functions; portable runners replace machine-specific orchestration and archived-invocation dependencies. CUDA hardware and software differences can change floating-point trajectories; the short packaging checks are not a new reproduction of the full training sweeps.
