# AMP and Bayes reference

Run all commands below from the repository root (`mask-resampling/`).

This folder contains the AMP spike estimator (`amp_spike.py`), its Bayesian downstream head (`amp_downstream.py`), the Gaussian-prior Bayes state evolution (`bayes_core.py`), and two command-line runners. Shared helpers generate synthetic observations and implement the raw-feature logistic probe. Nothing is downloaded; runs save configurations and scalar CSV results, without datasets, weights or figures.

## Install

Use Python 3.11 in a separate environment:

```bash
python -m pip install -r requirements/synthetic-cpu.txt
```

CPU execution reproduces the recorded AMP setup; `run_amp.py prepare --device cuda ...` optionally moves AMP inference to a GPU. Data generation and downstream probing remain on CPU.

## Reproduce the AMP settings

```bash
python experiments/bayes_optimal/run_amp.py prepare --law bounded --output runs/amp_bounded
python experiments/bayes_optimal/run_amp.py task --config runs/amp_bounded/config.json --task-id 0
# Run task IDs 0 through 169 independently, then:
python experiments/bayes_optimal/run_amp.py aggregate --config runs/amp_bounded/config.json
```

The bounded preset uses gamma = 0.25 + 2.25 Beta(1,2), beta=1, d=4000, seeds 0–9, and the 17 recorded alpha values from 0.1 to 10. `--law homogeneous` uses gamma=1; `--law unbounded` uses gamma = 0.75 + 0.25 exp(Z−1/2), Z standard normal. These two AMP presets use the 17 recorded alpha values from 0.5 to 10. Variances are sampled from the full laws, without empirical mean normalization. Gaussian spike and sample-amplitude priors are used throughout.

**The plotted AMP points use teacher-informed initialization**, explicitly reproduced by the default `--initialization teacher-informed`: u0 = sqrt(0.05) u_star + sqrt(0.95) z. The initial sample amplitudes have Gaussian variance 0.1. This reproduction preset is not data-only. The inference updates estimate coordinate variances from the observations and never receive the true variances. To run without teacher information, select `--initialization random`; this changes the experiment and retains Gaussian initialization standard deviation sqrt(0.1).

AMP uses at most 20,000 iterations, minimum 30, tolerance 1e-6, damping 0.75, five successive stable iterations, variance floors 1e-12 and 1e-8 relative to the mean estimated variance, and the original adaptive sign-cycle damping safeguard. The observations are generated in float32 and cast to float64 for inference. Each task starts independently. The same quenched spike is reused across alpha for a seed; the archived sampling seeds are retained.

Each fit evaluates two frozen downstream readouts using 100 independent labels and 10,000 test samples:

- `probe_test_accuracy`: scalar logistic regression on the raw preactivation wᵀx/sqrt(d), without an intercept. This is the readout in the plotted comparison with ERM. The original LBFGS settings are lr=1, max_iter=100, history_size=25.
- `bayesian_readout_accuracy`: the separate inverse-variance head w/gamma_hat with a finite-label Bayesian sign update. It is evaluated only when AMP converges. Exact probability ties receive half credit; the logistic probe predicts +1 at a zero score, as in the original experiments.

`amp_per_seed.csv` retains every saved terminal iterate, convergence flag, residual and near-zero-direction flag. `amp_summary.csv` includes mean, sample SD, SEM and per-metric counts; `amp_converged_summary.csv` additionally restricts to converged fits. A nonconverged task saves its result and exits with code 2. Aggregation requires all tasks unless `--allow-missing` is supplied; `coverage.json` always records coverage. No result is silently retried or selected using teacher alignment.

## Bayes state evolution

```bash
python experiments/bayes_optimal/run_bayes.py prepare --output runs/bayes_bounded
python experiments/bayes_optimal/run_bayes.py task --config runs/bayes_bounded/config.json --task-id 0
# Run task IDs 0 through 300 independently, then:
python experiments/bayes_optimal/run_bayes.py aggregate --config runs/bayes_bounded/config.json
```

The bounded preset retains the 301 alpha values used for the saved reference: 100 logarithmically spaced points on [0.09,10.5], plus 201 transition-refinement points. It uses 40 Gauss–Legendre variance nodes, beta=1, q_lambda(0)=1e-3, damping 0.2, tolerance 1e-12 and at most 20,000 iterations. The reported threshold is 1/(beta² E[gamma⁻²]), approximately 0.3779443315 for this law. Numerical residuals and finite-tolerance near-zero overlaps are preserved.

The output includes cosine similarity and the **raw-feature logistic-probe prediction used in the plot**, with n_ds=100 and the original orientation integration (100,000 Monte Carlo draws, seed 12345). This downstream curve is distinct from the Bayesian inverse-variance head above. `--n-ds` changes the label count. `--law homogeneous` and `--law unbounded` evaluate the same fixed-point equations with a point mass or full-law Gauss–Hermite quadrature; these options do not reproduce any earlier truncated-lognormal grid. `--alphas`, `--order`, `--tol` and other options allow explicit numerical checks; all settings are recorded in the run configuration.

## Parallel execution

Both runners accept `SLURM_ARRAY_TASK_ID` when `--task-id` is omitted. Submit one independent task per array index using your site's CPU or GPU resource settings. The AMP paper preset has 170 tasks and the Bayes bounded preset has 301. Each writes a separate scalar CSV; run aggregation after the array completes. Use a new output directory for a different configuration. `prepare --help` lists overrides, including custom alpha values and AMP seed lists.

## Use with existing data

```python
from types import SimpleNamespace
from amp_spike import fit_amp
from amp_downstream import fit_amp_downstream

cfg = SimpleNamespace(beta=1.0, u_prior="gaussian", lambda_prior="gaussian")
amp = fit_amp(train.x.double(), cfg, seed=0, max_iter=20000,
              damping=0.75, adaptive_cycle_damping=True)
w = amp.w  # No teacher or true variances passed; hidden masks are unused.
head = fit_amp_downstream(amp, labeled.x, labeled.y)  # Labels in {-1,+1}.
accuracy = head.accuracy(test.x, test.y)
```

The direct API defaults to random initialization with standard deviation 0.01; the figure runner passes its distinct recorded initialization explicitly. `fit_amp_downstream` requires a converged AMP result by default. Its outputs follow the matched scalar-channel prescription; convergence alone does not establish finite-sample Bayes optimality.
