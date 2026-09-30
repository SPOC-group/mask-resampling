# Synthetic autoencoder ERM

Run all commands below from the repository root (`mask-resampling/`).

Two Python files: `erm_core.py` implements JAX full-batch Adam; `run_erm.py` generates the data, runs masked or unmasked training, evaluates reconstruction and the frozen scalar logistic probe, and aggregates results. No private modules or input datasets are required.

## Install

Use Python 3.11 in a separate environment. The experiments used JAX/JAXlib 0.4.30, PyTorch 2.8.0, NumPy 2.0.2 and pandas 2.3.3. For a compatible NVIDIA CUDA 12 environment:

```bash
python -m pip install -r requirements/synthetic-cuda12.txt
```

For CPU checks, install `requirements/synthetic-cpu.txt` instead and use `--device cpu`. GPU training uses JAX; PyTorch generates the seeded data and fits the CPU probe.

## Prepare the reported experiments

From the repository root, each command creates a configuration without running simulations:

```bash
python experiments/erm/run_erm.py prepare --experiment masked-alpha --output runs/masked
python experiments/erm/run_erm.py prepare --experiment full-alpha --output runs/full
python experiments/erm/run_erm.py prepare --experiment masked-rho --output runs/rho
```

All presets use gamma = 0.25 + 2.25 Beta(1,2), d = 2000, beta = 1, ridge = 0.0001, Gaussian spike/latent/initial weights, 4,000 full-batch Adam updates at learning rate 0.05, row chunks of 1,024, and seeds 0–29. Diagnostics are recorded every 200 updates. The probe uses 100 independent labeled training samples and 10,000 test samples, the raw encoder preactivation, and a scalar logistic coefficient without intercept. Reconstruction test samples number n = round(alpha d).

| Preset | Conditions | Independent tasks |
|---|---|---:|
| `masked-alpha` | Linear, tanh, ELU, ReLU; rho = 0.75; K = 1, 2, 4, dynamic | 2,040 |
| `full-alpha` | Same activations and alpha grid; all input/target coordinates, no masking | 2,040 |
| `masked-rho` | Linear, alpha = 8; 11 rho values; K = 1, 2, 4, 8, dynamic | 330 |

The alpha grid has 17 logarithmically spaced values from 0.1 to 10; the rho grid is 0.05, 0.1, 0.2, ..., 0.9, 0.95. Exact values are saved in `config.json`. Each independent task fixes a seed, alpha, rho and activation, then runs its K conditions separately from the same initialization. Finite K reuses nested Boolean Bernoulli mask banks; dynamic training draws fresh Bernoulli masks at every update. Unmasked training allocates no masks and ignores rho. Chunks accumulate gradients before one Adam update; they do not change the update budget. There is one fit per condition/seed, no restarts or early stopping.

## Run and collect

```bash
python experiments/erm/run_erm.py task --config runs/masked/config.json --task-id 0 --device gpu
```

Run every task ID (0–2039 for either alpha preset; 0–329 for the rho preset) independently, in parallel on available GPUs. For example, with Slurm and an already activated environment:

```bash
sbatch --array=0-2039 --gpus=1 --cpus-per-task=2 --mem=8G --time=01:00:00 \
  --wrap='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/erm/run_erm.py task --config runs/masked/config.json --task-id "$SLURM_ARRAY_TASK_ID" --device gpu'
```

Use your cluster's partition/account options as needed. Substitute `runs/full/config.json` or `runs/rho/config.json` and the corresponding task range for the other presets. There is no continuation across alpha values. Completed conditions are skipped when rerunning a task; `--force` reruns that task. Wait for all tasks before collecting:

```bash
python experiments/erm/run_erm.py aggregate --config runs/masked/config.json
```

`runs/masked/results/` contains per-seed and mean/SD/SEM CSVs, trajectories and the numerical configuration. Loss columns include raw reconstruction MSE and the appendix's centered losses/gains; dynamic training loss is evaluated using a separate fixed mask, and test loss uses independent test data/masks. No weights, generated datasets, mask banks or plots are saved. Use the plotting notebook for figures.

For a small check, prepare a new folder with `--d 32 --alphas 1 --n-seeds 1 --updates 3 --eval-every 1 --row-chunk-size 13 --n-ds 8 --probe-n-test 32`; this reduces the configuration and is not a paper result. `--help` lists all overrides. Configurations use relative paths and can be moved together with their task outputs.
