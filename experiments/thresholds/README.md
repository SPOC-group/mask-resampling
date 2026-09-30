# Recovery-threshold calculations

Run all commands below from the repository root (`mask-resampling/`).

Four Python files reproduce the linear threshold-ratio figure without a solver notebook or private input files. `threshold_core.py` retains the original notebook's numerical definitions; `threshold_helpers.py` retains the distribution adapter, numerical settings; `gamma_distributions.py` defines the bounded law. `run_thresholds.py` supplies portable presets, independent tasks and CSV aggregation.

## Install and run

Use Python 3.11 in a separate environment with the pinned synthetic dependencies:

```bash
python -m pip install -r requirements/synthetic-cpu.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/thresholds/run_thresholds.py prepare --preset ratios --output runs/ratios
python experiments/thresholds/run_thresholds.py task --output runs/ratios --task-id 0
```

The ratio preset has **540 independent tasks, IDs 0–539**: two laws, six beta values, five masking ratios, and K=1,2,4,8,16,32,64,128,infinity. For parallel Slurm execution from the repository root, with the environment activated:

```bash
sbatch --array=0-539%32 --cpus-per-task=1 --mem=3G --time=02:00:00 \
  --wrap='python experiments/thresholds/run_thresholds.py task --output runs/ratios'
# After all tasks finish:
python experiments/thresholds/run_thresholds.py aggregate --output runs/ratios
```

Adapt resources to your cluster. `SLURM_ARRAY_TASK_ID` is used when `--task-id` is omitted. No cluster account or hostname is embedded.

## Exact presets

**`ratios` — Figure 18.** Linear activation, zero ridge; beta=0.1, 0.3111111111111111, 0.5222222222222223, 0.7333333333333333, 0.9444444444444444, 1; rho=0.05,0.25,0.5,0.75,0.95. Homogeneous noise has gamma=1. The bounded panel uses gamma=1+a(0.25+2.25B−1), B~Beta(1,2), with a=sqrt((1.0927746165423835−1)/0.28125). This is the matched-moment bounded law, distinct from the unscaled Beta law in the main recovery experiments. Integration uses a point mass or 200-node full-support Gauss–Legendre quadrature.

Each finite-K threshold is divided by the **numerically computed dynamic threshold at the same beta, rho and variance law**. The black weak-signal reference is 1+[1/(2rho(1−rho))−1]/K. The preset yields 480 finite-K ratios and 60 dynamic denominators.

The preset uses the original analytical linear channels, damping 0.22, saddle tolerance 2e−9, maximum 4000 iterations, 36 logarithmic scan points and up to 24 bisections. Tasks initialize independently; **within each task**, the alpha scan continues the exactly uninformative M=0 branch, as in the original calculation. The crossing condition is alpha*beta*chi*E[v]=1. These are replica-symmetric local-instability predictions. Original status flags, saddle residuals, gain residuals and factorization diagnostics are retained; packaging does not strengthen their convergence guarantees.

## Outputs

Each task saves `result.csv`, scalar `zero_branch.csv` diagnostics and JSON status. It saves no observations, weights or coordinate-state arrays. Completed tasks are preserved; interrupted tasks restart. Nonconverged results are saved and return exit code 2.

Aggregation writes `all_thresholds.csv` and `validated_ratio_data_matched_three_laws_grid.csv`; the latter contains only finite-K ratios with converged, finite numerator and denominator. Here “validated” retains the archived filename convention and does not mean a separate mathematical or quadrature validation. `--allow-missing` explicitly permits partial aggregation, with coverage recorded in `coverage.json`; missing or failed dynamic thresholds do not produce ratios.

`prepare --help` lists law/beta/rho/K overrides, which change the reported preset. Existing plotting data are not overwritten. Figure generation remains in the notebooks/figures.ipynb and its single `plots/` directory.
