# Experiment guide

All commands in this guide and the experiment READMEs run from the repository root (`mask-resampling/`). Run an experiment in its own environment, independently of the plotting environment. Full numerical settings and task counts are in each experiment's README. The source routines and recorded presets are preserved from the supplementary package.

## Synthetic experiments

Use Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/synthetic-cpu.txt
```

For supported Linux/CUDA 12 systems, use `requirements/synthetic-cuda12.txt` instead. PyTorch GPU routines require a compatible PyTorch installation too. Select CPU/GPU explicitly using the runner's documented flags.

| Calculation | Prepare configuration (does not run the sweep) | Next step |
|---|---|---|
| Masked ERM | `python experiments/erm/run_erm.py prepare --experiment masked-alpha --output runs/erm` | `python experiments/erm/run_erm.py task --config runs/erm/config.json --task-id 0 --device cpu` |
| Unmasked ERM | Same command with `--experiment full-alpha` and a new output directory | Same task/aggregate interface |
| Masked state evolution | `python experiments/state_evolution/run_state_evolution.py prepare --preset bounded-beta --output runs/se` | `python experiments/state_evolution/run_state_evolution.py task --config runs/se/config.json --task-id 0 --device cpu` |
| AMP | `python experiments/bayes_optimal/run_amp.py prepare --law bounded --output runs/amp` | `python experiments/bayes_optimal/run_amp.py task --config runs/amp/config.json --task-id 0` |
| Bayes reference | `python experiments/bayes_optimal/run_bayes.py prepare --output runs/bayes` | `python experiments/bayes_optimal/run_bayes.py task --config runs/bayes/config.json --task-id 0` |
| Spectral comparison | `python experiments/spectral/run_spectral.py prepare --root runs/spectral` | `python experiments/spectral/run_spectral.py task --root runs/spectral --task-id 0` |
| Thresholds | `python experiments/thresholds/run_thresholds.py prepare --preset ratios --output runs/thresholds` | `python experiments/thresholds/run_thresholds.py task --output runs/thresholds --task-id 0` |

These defaults describe full paper configurations, not quick demonstrations. For a small ERM execution check, the [ERM README](../experiments/erm/README.md) lists reduced dimension, updates and seed settings. Such a check is not a paper result. Use each runner's `--help` and `prepare --help` for overrides.

Run all task IDs independently and call `aggregate` only after the tasks finish. Use a separate output directory for each configuration. Missing points, solver failures and convergence flags remain explicit; the wrappers do not turn missing overlaps into zero. The state-evolution and threshold solvers have documented internal iteration/continuation behavior, but independent alpha tasks do not depend on one another.

## CNN, ViT and BERT

| Model | Environment | Data | List tasks before training |
|---|---|---|---|
| CNN | `python -m pip install -r requirements/cnn.txt` | External CIFAR-10 Python batches; prepare splits first | `python experiments/cnn/run_cnn.py --preset original --list` |
| ViT | `python -m pip install -r requirements/vit.txt` | External ImageNet with the supplied 100-class subset; prepare shared initializations | `python experiments/vit/run_vit.py --preset main --list` |
| BERT | Follow the ordered torch/source setup in the [BERT README](../experiments/bert/README.md) | External preprocessed Wikipedia–BookCorpus shard and GLUE | `python experiments/bert/run_bert.py --stage pretrain --list` |

The CNN and ViT runners also provide augmentation and training-budget controls. BERT contains the final nine pretraining runs and 144 fine-tuning runs used for the reported table. `--dry-run` on these runners prints commands without training. BERT's source preparation downloads a pinned public DinkyTrain archive and applies the supplied overlay; that large upstream source is not vendored here.

Follow [CNN](../experiments/cnn/README.md), [ViT](../experiments/vit/README.md) or [BERT](../experiments/bert/README.md) for external data preparation, device allocation, probing and collection. No workstation hostname, account or cluster partition is required by the release.
