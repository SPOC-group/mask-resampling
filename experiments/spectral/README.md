# Spectral baselines and PCA dimension scaling

Run all commands below from the repository root (`mask-resampling/`).

These scripts reproduce the spectral estimators and bounded-noise PCA dimension sweep. They retain the original numerical routines, random-number streams and recorded presets. The command-line wrappers remove cluster paths and archival dependencies. No synthetic observations, teacher arrays, fitted directions or checkpoints are saved; outputs contain configuration, scalar measurements and convergence diagnostics. Existing plotting inputs and PDFs are separate and remain unchanged.

## Install and run

From the repository root, use Python 3.11 in a separate environment with the pinned synthetic dependencies:

```bash
python -m pip install -r requirements/synthetic-cpu.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python experiments/spectral/run_spectral.py prepare --root results/spectral
python experiments/spectral/run_spectral.py task --root results/spectral --task-id 0
```

The preset has **680 independent tasks (IDs 0–679)**. Each task fits ordinary PCA, diagonal-deletion PCA, HeteroPCA and estimated-whitening PCA to the same observations. Run all task IDs before aggregation. For example, on Slurm, with the environment activated:

```bash
sbatch --array=0-679%32 --cpus-per-task=1 --mem=4G --time=24:00:00 \
  --wrap='python experiments/spectral/run_spectral.py task --root results/spectral'
# After all tasks finish:
python experiments/spectral/run_spectral.py aggregate --root results/spectral
```

The runner reads `SLURM_ARRAY_TASK_ID` when `--task-id` is omitted. Array concurrency and resource limits are examples to adapt to available hardware. Tasks do not initialize from other alpha values. Completed tasks are preserved on rerun; interrupted tasks restart without saved directions. `--allow-missing` produces an explicitly partial aggregate. Convergence flags and iteration counts remain in the results, including terminal fits that hit the cap; aggregation does not certify convergence or drop these fits.

## Spectral preset and algorithms

- Dimension 2000, beta=1, Gaussian teacher and latent amplitudes, seeds 0–19.
- Bounded panel: gamma=0.25+2.25 Beta(1,2); 17 logarithmically spaced alpha values from 0.1 to 10; **uncentered** observations.
- Unbounded panel: gamma=0.75+0.25 exp(Z−1/2), Z standard normal; 17 logarithmically spaced alpha values from 0.5 to 10; **sample-centered** observations.
- Exact saved alpha values are embedded in `run_spectral.py`. Variances are neither normalized nor truncated. Original float32 observations are converted to float64 before forming S=XᵀX/n. Teacher/gamma and observation seeds are 10000*seed+1 and 10000*seed+2. Bounded gamma uses the generator's additional seed offset 17. Unused masks drawn by the original generator never enter the estimators.

`heteroskedastic_pca_estimators.py` is the original numerical core. Diagonal deletion takes a leading singular direction of offdiag(S). HeteroPCA uses rank-one, undamped diagonal imputation with the **signed** eigenvalue of largest absolute magnitude. These implementations follow Algorithm 1 and the diagonal-deletion baseline discussed by [Zhang, Cai and Wu, “Heteroskedastic PCA: Algorithm, Optimality, and Applications”](https://arxiv.org/abs/1810.08316). The citation does not attribute the invention of diagonal deletion.

HeteroPCA initially stops at relative diagonal change 1e−8 or 500 iterations. Only unfinished fits continue, first to a total of 50,000 and then 100,000 iterations. `heteropca_continuation.py` retains the original continuation routine, using one inner eigenpair and checking candidate fixed points with the original two-eigenpair solver. Acceptance requires relative diagonal change and fixed-point residual ≤1e−8, eigenpair residual ≤1e−8, and sign-invariant direction discrepancy ≤1e−6. Initially converged fits retain their original stopping rule. The packaged runner carries the diagonal between stages in memory. All 680 archived HeteroPCA fits used for the figure eventually converged; this is not a guarantee for arbitrary new settings or numerical libraries.

Estimated whitening uses Gamma_hat=diag(S), floored at 1e−12, then the first PC of Gamma_hat^(-1/2) S Gamma_hat^(-1/2). It returns the normalized **unwhitened feature direction**, Gamma_hat^(1/2) times that PC. This specializes the whitening/unwhitening construction and estimated diagonal variances in [Leeb and Romanov, “Optimal spectral shrinkage and PCA with heteroscedastic noise”](https://arxiv.org/abs/1811.02201), rather than implementing their full optimal-shrinkage estimator. No estimator receives the teacher or true noise variances during fitting. The teacher is used only for cosine evaluation.

All spectral solves use tolerance 1e−10 and maximum 5000 iterations. Ordinary PCA is also saved as a reference; its float64 covariance can differ slightly from older float32 ordinary-PCA exports. Results are `pca_estimators_per_seed.csv` and `pca_estimators_summary.csv`, including mean absolute cosine, sample SD and SEM. This runner does not regenerate the masked or AMP comparison curves; those have separate simulation folders.

## PCA dimension sweep

```bash
python experiments/spectral/run_dimension_scaling.py prepare --root results/dimension
python experiments/spectral/run_dimension_scaling.py task --root results/dimension --task-id 0
# Run all 2700 independent task IDs (0–2699), then:
python experiments/spectral/run_dimension_scaling.py aggregate --root results/dimension
python experiments/spectral/fit_dimension_scaling.py --run-dir results/dimension
```

The preset uses dimensions 100, 200, 400, 800, 1600, 3200, 6400, 12800 and 25600; ten logarithmically spaced alpha values from 1 to 50; seeds 0–29; beta=1; and gamma=0.25+2.25 Beta(1,2). PCA is uncentered and unstandardized. The original float32 teacher/gamma draws are retained. Unlike the preceding comparison, observations are generated directly in float64 with separate NumPy streams (noise: 10000*seed+102; amplitudes: 10000*seed+103), in blocks of 1024 rows. Increasing alpha extends the same sample stream for a given dimension and seed. The covariance is applied as Xᵀ(Xv)/n; eigensolver tolerance is 1e−9, maximum 5000 iterations, with a required relative residual ≤1e−7.

**Size resources per dimension.** The largest task holds an approximately 244 GiB observation matrix in RAM, plus working memory. It requires a high-memory CPU node; the 4 GiB spectral-array example above is not suitable for this sweep. Use `--dimensions`, `--alphas` and `--seeds` when preparing a smaller or dimension-specific independent array. Those overrides change the default full reproduction grid.

The fitting script checks complete per-seed coverage and reconstructs the three largest-alpha profiles. It fits A_alpha d^(−1/2) in log space, both across all dimensions and over d≥1600, and also writes the original free-exponent diagnostic. Default uncertainty is 2000 joint bootstrap resamples of complete seed profiles, seed 1729; reported fit intervals are 95% bootstrap intervals. No simulation is rerun by this fitting step. All figure generation remains in the notebooks/figures.ipynb and its single `plots/` folder.
