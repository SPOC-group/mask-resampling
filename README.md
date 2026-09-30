# The Hidden Advantage of Mask Resampling

Code and results for **The Hidden Advantage of Mask Resampling: A Theory of Masked Autoencoders**, by **Jorge Medina Moreira, Lorenzo Bardone, and Lenka Zdeborová**.

Study how the number of masks per example changes feature recovery and downstream performance. The repository includes replica/state-evolution solvers, synthetic autoencoder training, AMP and spectral baselines, and CNN, ViT and BERT experiments.

## Reproduce the figures

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/plotting.txt
python scripts/verify.py
python scripts/plot.py --no-tex
```

This regenerates the 18 paper figures from the supplied results. All PDFs go to **`plots/`**. No GPU, training datasets or simulation runs are needed. On Windows, activate with `.venv\Scripts\activate`.

For one figure, use `python scripts/plot.py --figure 3 --no-tex`; use `--list` for available IDs. For interactive plotting, install `notebook` and open [notebooks/figures.ipynb](notebooks/figures.ipynb). See the [figure guide](docs/figures.md) for the paper-to-code mapping.

## Run experiments

Each experiment has a documented preset and standalone entry point. Use separate environments for synthetic, CNN, ViT and BERT runs: their recorded dependencies differ. The [experiment guide](docs/experiments.md) gives setup and run commands.

| Experiment | Entry point | Instructions |
|---|---|---|
| Masked and unmasked synthetic autoencoders | `experiments/erm/run_erm.py` | [ERM](experiments/erm/README.md) |
| Masked-autoencoder replica/state evolution | `experiments/state_evolution/run_state_evolution.py` | [State evolution](experiments/state_evolution/README.md) |
| AMP and Bayes state evolution | `experiments/bayes_optimal/run_amp.py`, `run_bayes.py` | [AMP / Bayes](experiments/bayes_optimal/README.md) |
| PCA, diagonal deletion, HeteroPCA and whitening | `experiments/spectral/run_spectral.py` | [Spectral estimators](experiments/spectral/README.md) |
| Recovery thresholds | `experiments/thresholds/run_thresholds.py` | [Thresholds](experiments/thresholds/README.md) |
| CNN on CIFAR-10 | `experiments/cnn/run_cnn.py` | [CNN](experiments/cnn/README.md) |
| ViT-Base on ImageNet-100 | `experiments/vit/run_vit.py` | [ViT](experiments/vit/README.md) |
| BERT-Medium and GLUE | `experiments/bert/run_bert.py` | [BERT](experiments/bert/README.md) |

Independent simulation tasks can run in parallel. Vision/language training requires external datasets and GPUs; each guide specifies the data, initialization, checkpoints and evaluation protocol.

## Repository layout

```text
experiments/    Simulation and training scripts, with a README per experiment
notebooks/      One plotting notebook
scripts/        Figure rendering and input verification
requirements/   Separate plotting and experiment environments
data/           Saved scalar results, histories and input checksums
plots/          All generated figure PDFs (ignored by Git)
docs/           Figure mapping, data descriptions and reproduction details
CITATION.cff    Paper title and author metadata
THIRD_PARTY.md  Upstream code credits and license scope
```

## Results and interpretation

The [data guide](docs/data.md) describes the saved inputs. The [reproduction guide](docs/reproduction.md) distinguishes regenerating figures from rerunning experiments and documents differences between protocols.

- Figure-reproduction presets retain **SD**, as used in the supplied plots; SEM columns are not silently substituted.
- The plotted AMP runs use **teacher-informed initialization with p = 0.05**. A separate random-initialization option is available; the two experiments are explicitly distinguished.
- Saved solver status and residuals are retained. The released presets preserve activation-specific solvers, quadratures and tolerances.
- No training images, text corpora, model weights or checkpoints are bundled. New training runs write their outputs locally.

## Citation and reuse

Original project code is released under [MIT](LICENSE). Use [CITATION.cff](CITATION.cff) for the paper title and author list. The arXiv identifier and public repository URL will be added when assigned. See [THIRD_PARTY.md](THIRD_PARTY.md) and the retained component licenses before reusing third-party code, particularly the MAE-derived ViT implementation.
