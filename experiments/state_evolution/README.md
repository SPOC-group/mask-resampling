# Masked-autoencoder state evolution

Run all commands below from the repository root (`mask-resampling/`).

`run_state_evolution.py` provides the paper's bounded-noise preset. `solvers/` contains the numerical implementations used for the reported curves, with common helpers stored once. No input datasets, private notebooks or cluster paths are required.

## Install

Use Python 3.11 in a separate environment. For a compatible NVIDIA CUDA 12 environment:

```bash
python -m pip install -r requirements/synthetic-cuda12.txt
```

For CPU checks, install `requirements/synthetic-cpu.txt` instead and pass `--device cpu`. Numerical calculations use float64. JAX is used for dynamic tanh; the other channels use NumPy/SciPy or PyTorch.

## Paper preset

The model is gamma = 0.25 + 2.25 Beta(1,2), beta = 1, rho = 0.75 and ridge = 0.0001. Variance expectations use 200-node Gauss–Legendre quadrature over the full bounded support. All profiles use damping 0.3, a 160,000-iteration cap and independent initialization at each alpha.

The **120-point alpha array** is `[0.09, 0.2, 0.35, 0.45, 0.6, 0.8]`, followed by `linspace(0.9, 4, 90)` and `geomspace(4.2, 11, 24)`. Its exact saved values are built into the script. Default conditions are linear, ReLU, ELU and tanh, each with K = 1, 2, 4 and dynamic masking.

| Curve | Solver and field integration | Outer tolerance |
|---|---|---:|
| Linear, all K and dynamic | Analytical linear channels, same outer iteration | 1e-6 |
| ReLU, K=1 and dynamic | Dedicated ReLU channels; Gaussian quadrature order 120 | 1e-6 |
| ReLU, K=2,4 | Analytical active-set proximal; 262,144 antithetic Sobol samples | 1e-6 |
| ELU, K=1,2,4 | Generic joint proximal; 16,384 antithetic Sobol samples | 1e-6 |
| ELU, dynamic | Generic scalar proximal; field/mask quadrature order 64 | 1e-6 |
| Tanh, K=1 | Generic joint proximal; 262,144 antithetic Sobol samples | 1e-7 |
| Tanh, K=2,4 | Generic joint proximal; 16,384 antithetic Sobol samples | 1e-6 |
| Tanh, dynamic | Specialized bounded-law JAX solver; field/mask quadrature order 64 | 1e-8 |

Generic finite-K smooth proximals use 35 steps and tolerance 1e-9, with Sobol seed 1234. Generic initialization uses `init_m_scale=0.05`; dynamic tanh uses coordinate m=0.01, v=1 and q=4 gamma v+m². Dynamic ELU uses a 201-point proximal grid and 25 refinements; dynamic tanh uses 241 points and 35 refinements. Exact per-curve settings are saved in the generated configuration.

## Prepare, run and collect

From the repository root:

```bash
python experiments/state_evolution/run_state_evolution.py prepare --preset bounded-beta --output runs/bounded
python experiments/state_evolution/run_state_evolution.py task --config runs/bounded/config.json --task-id 0
```

Preparation writes the configuration only. The full preset has **1,920 independent tasks**, IDs 0–1919. Run alpha values in parallel; there is no continuation, automatic retry or multiple-initialization sweep. Device selection defaults to CPU for linear and ReLU endpoints, and GPU for the remaining curves.

For a single curve, prepare a separate directory, for example:

```bash
python experiments/state_evolution/run_state_evolution.py prepare --output runs/tanh_k2 --activations tanh --k-values 2
sbatch --array=0-119 --gpus=1 --cpus-per-task=2 --mem=16G --time=01:00:00 \
  --wrap='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiments/state_evolution/run_state_evolution.py task --config runs/tanh_k2/config.json --task-id "$SLURM_ARRAY_TASK_ID"'
python experiments/state_evolution/run_state_evolution.py aggregate --config runs/tanh_k2/config.json
```

Use your cluster's partition/account options and an activated environment. Wait for the array before aggregating. Replace the activation/K to select another curve; use `dynamic` for fresh masking. Other positive K values are supported using that activation's finite-K settings, but the reported preset contains K=1,2,4 only. `--alphas` overrides the array; `--help` lists numerical overrides.

Each task saves a scalar `result.csv` with M, Q, C, cosine similarity, residuals, solver status and available reconstruction losses. Downstream accuracy uses the same raw-preactivation prediction as the plotting notebook: 100 labels, 100,000 half-normal Monte Carlo samples and seed 12345. Dynamic tanh retains its original output schema; reconstruction losses that its solver does not return remain unavailable. No simulation datasets, coordinate-state arrays or plots are saved.

Aggregation writes `state_evolution.csv` and retains nonconverged results. A task returns exit code 2 when its recorded convergence/admissibility checks fail. `--allow-missing` preserves absent solutions as blank rows; it never fills missing overlaps with zero. Existing completed tasks are skipped unless `--force` is supplied. After an interrupted process, remove its `running.lock` only after confirming it has stopped. Numerical overrides are recorded explicitly; they do not describe the paper curves.
