# ViT-Base on ImageNet-100

Run all commands below from the repository root (`mask-resampling/`).

The original U-MAE-derived numerical implementation, with portable commands for the paper's masking comparison, augmentation comparison, longer-training checks, probe-augmentation check and unmasked baseline. See `ATTRIBUTION.md` and `LICENSE` for upstream credits. No images, pretrained weights, private run logs or account configuration are included. External experiment tracking is disabled. Training writes checkpoints and scalar logs into your chosen run directory; figures remain in the plotting notebook.

## Setup and data

Use a separate Python 3.11 environment. Install the recorded dependencies, selecting a PyTorch CUDA wheel suitable for your system if needed:

```bash
python -m pip install -r requirements/vit.txt
```

The original `timm==0.3.2` is intentional. `sitecustomize.py` supplies its compatibility shim for modern PyTorch; the portable runner sets the import path automatically. Do not replace timm with a newer version without adapting and checking the model code.

Provide ImageFolder directories `train/<synset>/...` and `val/<synset>/...`, containing 126,689 training and 5,000 validation images. `imagenet100_classes.txt` contains the exact 100 synsets, in the saved class-index order. From an existing ImageNet-1K installation with organized validation folders, prepare the subset with:

```bash
python experiments/vit/prepare_imagenet100.py --imagenet1k_root /path/to/imagenet1k \
  --output_root /path/to/imagenet100 --class_list experiments/vit/imagenet100_classes.txt
```

This creates symlinks; `--copy` copies instead. The runner verifies the class list and image counts. ImageFolder's sorted filenames also determine the image indices used for static masks; preserve the original ImageNet filenames. This subset is distinct from the subset used for the published U-MAE benchmark, so the longer-run table supplies context rather than an identical-data reproduction.

Create each shared initialization once, before launching independent tasks:

```bash
for seed in 0 1 2; do
  python experiments/vit/create_initialization.py --output "runs/vit/initializations/seed${seed}.pth" \
    --model mae_vit_base_patch16 --norm_pix_loss --seed "$seed"
done
```

The initialization generator refuses to overwrite a file. All conditions with the same initialization seed load the same weights. The main three-seed curves vary **initialization seeds 0,1,2**; the training seed, static-mask seed and probe seed remain zero.

## Presets and commands

| Preset | Tasks | Purpose |
|---|---:|---|
| `main` | 48 | Static/dynamic, eight masking ratios, three initializations; no pretraining crop/flip. |
| `augmentation` | 32 | Static/dynamic with/without pretraining crop/flip, eight ratios, initialization zero. |
| `longer` | 2 | Dynamic, rho=0.75, no pretraining crop/flip: 200-epoch U-MAE recipe and 641 epochs of the 1,600-epoch MAE recipe. Each encoder gets both probes. |
| `probe_no_aug` | 2 | Reuse static/dynamic main encoders at rho=0.75, initialization zero; remove crop/flip from the probe. |
| `unmasked` | 1 | Full reconstruction through 27 completed epochs; probe nine encoder checkpoints. |

The masking grid is `0.125,0.25,0.375,0.5,0.625,0.75,0.825,0.95`.

```bash
python experiments/vit/run_vit.py --preset main --list
python experiments/vit/run_vit.py --preset main --task 0 \
  --data-root /path/to/imagenet100 --output-root runs/vit

python experiments/vit/run_vit.py --preset augmentation --task 0 \
  --data-root /path/to/imagenet100 --output-root runs/vit

python experiments/vit/run_vit.py --preset longer --task 0 \
  --data-root /path/to/imagenet100 --output-root runs/vit
```

Select `--stage pretrain`, `--stage probe` or the default `both`. `--dry-run` prints commands. `probe_no_aug` requires its two pretrained main encoders; it does not repeat pretraining. Main and augmentation tasks share the identical unaugmented initialization-zero encoders through the same output paths. Do not launch overlapping tasks concurrently. Completed runs with matching commands are reused.

Different conditions can run concurrently on allocated GPUs. Assign one visible GPU per task; the runner launches **one distributed process** to retain the original epoch-dependent sampler behavior. It reads `SLURM_ARRAY_TASK_ID` when `--task` is omitted. Use array indices `0-47` for main, `0-31` for augmentation, and `0-1` for longer or probe-no-augmentation. Do not launch multiple distributed processes per task: that would change the effective batch size. The original physical batches are retained, so memory needs vary across presets.

`--resume` resumes an interrupted task only when its history ends at the newest saved checkpoint. If training continued beyond that checkpoint before interruption, the runner stops for inspection instead of mixing histories. Resume can change stochastic trajectories because the inherited checkpoints do not retain every RNG state. For a fresh reproduction use a new output root.

## Recorded protocol

- ViT-Base/16: 12 encoder blocks, width 768, 12 heads; eight decoder blocks, width 512, 16 heads. Input 224x224, 196 patches. Reconstruction uses within-patch standardized targets. Uniformity regularization is disabled (`reg=none`, `lamb=0`); weight decay is retained. The inherited online classifier receives detached features and does not train the encoder.
- Dynamic masking redraws a patch permutation every presentation; static masking fixes it per training-image index. Both retain `floor(196*(1-rho))` visible patches. No-augmentation preprocessing is bicubic resize to shorter side 256, center crop 224 and ImageNet normalization. Augmentation uses random resized crop with scale [0.2,1], aspect ratio [3/4,4/3] and horizontal flip probability 1/2.
- Main pretraining: 100 epochs of AdamW, betas (0.9,0.95), weight decay 0.05, effective batch 128, peak LR 0.000075, ten warmup epochs and cosine decay to zero. Physical batch/accumulation are 64/2 without crop/flip and 128/1 with crop/flip. Training retains the original mixed precision and exclusions of biases/normalization parameters from weight decay.
- Longer pretraining: 200 epochs with physical batch 128, accumulation 8, peak LR 0.0006 and 40 warmup epochs; or physical batch 64, accumulation 64, peak LR 0.0024 and 40 warmup epochs on a 1,600-epoch schedule. The table evaluates checkpoint 640 (641 completed epochs); the original run continued to 647 completed epochs before stopping. The portable preset stops at the evaluated checkpoint. The latter retains the **1,600-epoch LR horizon**, not a 641-epoch cosine schedule. Both use initialization zero, rho=0.75 and no crop/flip. They are trained on ImageNet-100, not initialized from external pretrained weights.
- Frozen probing: all patches, 768-dimensional class token, non-affine batch normalization and a 100-class linear head. LARS, momentum 0.9, no weight decay, ten warmup epochs, cosine decay. Standard probe: 50 epochs, batch 256, accumulation 1, peak LR 0.1. Longer probe: 90 epochs, physical batch 512, accumulation 32, peak LR 6.4. Batch-normalization statistics use the physical batch. Incomplete accumulation windows at an epoch's end do not produce an optimizer update.
- Probe training normally uses random resized crop with scale [0.08,1] and horizontal flipping. `probe_no_aug` replaces these by deterministic resize/center crop, holding the encoder and other probe settings fixed. Validation always uses deterministic preprocessing. All accuracies are ImageNet-100 **validation** top-1/top-5, from the final probe epoch. Inherited log keys named `test_acc1`, `test_acc5` and `test_loss` refer to this validation set.
- Unmasked reconstruction uses the same architecture, normalized targets and main no-augmentation optimizer, with no masking and loss over all patches. The schedule still spans 100 epochs, while the run stops after epoch index 26. Probe completed epochs `1,2,3,6,11,16,21,22,27`. The reproduction preset saves all these checkpoints in one trajectory; the archived experiment recovered some early checkpoints in a resumed run. Select the baseline by highest final-probe validation top-5 across the nine checkpoints (six completed epochs in the saved results). This is validation selection, not an independent test estimate.

## Collect results and reconstruction errors

```bash
python experiments/vit/collect_vit.py --preset main --output-root runs/vit
python experiments/vit/collect_vit.py --preset longer --output-root runs/vit
```

Use the corresponding preset for other experiments. Outputs go to `runs/vit/summaries/<preset>/`: per-run final validation metrics, aggregated means/SD/SEM, pretraining histories and coverage. Accuracies in the collected CSVs are fractions. `--allow-partial` explicitly reports missing conditions. For the unmasked validation-MSE curve, run the retained float32 CPU evaluator and collect again:

```bash
python experiments/vit/evaluate_reconstruction.py --data-root /path/to/imagenet100 --output-root runs/vit
python experiments/vit/collect_vit.py --preset unmasked --output-root runs/vit
```

The evaluator covers all 5,000 validation images and weights batch MSEs by sample count. `train_loss_mae` is reconstruction MSE; total pretraining loss also includes the detached classifier loss and must not be plotted as reconstruction error. The collector preserves that distinction and records the unmasked validation-based checkpoint choice.

The existing notebook uses saved exports under `data/vision/vit/{masking_ratio,augmentation,full_reconstruction}`. Newly generated tidy CSVs do not overwrite those curated plotting inputs automatically. The saved scalar records for both ViT control tables are in `data/vision/vit/tables/`. `probe_results.csv` records eight completed probes and their settings; `probe_history.csv` retains all 480 validation evaluations. Run from the repository root:

```bash
python experiments/vit/make_vit_tables.py --input-csv data/vision/vit/tables/probe_results.csv \
  --output-dir runs/vit/tables
```

This standard-library command verifies full probe histories and terminal scores, then writes `training_budget.csv`, `probe_augmentation.csv` and `tables.md`. The same three outputs are supplied with the data. Scores are validation percentages from the final probe epoch, with no seed uncertainty estimated for these single-run checks. Console-derived values retain their original three-decimal precision; JSON scalar values retain their saved precision. `pretraining_completed_epochs` identifies the evaluated encoder (641 for the longer MAE recipe), not the later interruption point of that original run. Repeated pretraining rows in the longer-training table reuse the same encoder.

## Scope of packaging checks

The numerical model, losses, training engines and shared utilities are retained from the experiment source. Portable wrappers replace private launch scripts; external tracking and Git revision recording are disabled. Small checks exercise masks, preprocessing, gradients, scheduler horizons, task parameters, class mapping and collection. These checks do not rerun the full GPU training sweeps or imply bitwise reproducibility across hardware. Some checkpoint/output directories linked from the experiment repository were unavailable during packaging. Preserved scalar logs and configurations nevertheless verify all eight probes behind the two ViT control tables, including their final scores and full histories. This verification does not require rereading the unavailable model checkpoints.
