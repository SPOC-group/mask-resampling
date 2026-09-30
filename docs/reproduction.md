# Reproducing the paper

## Figures from saved results

`python scripts/plot.py --no-tex` executes the code cells in `notebooks/figures.ipynb` without changing that notebook. The setup verifies the supplied data hashes. All figures are written to the one `plots/` directory; repeat runs replace those PDFs. Use `--figure 1 3 18` to render a selection, or `--list` to inspect the mapping without importing plotting libraries.

The notebook can be opened from the repository root or from `notebooks/`. Run its setup cell before individual figure cells. TeX rendering is optional; without TeX, Matplotlib uses its built-in mathtext and a fallback serif font. This can change typography relative to the paper, but not the data or calculations.

Figure numbers, historical asset names, cell IDs and data paths are recorded in [figure_index.csv](figure_index.csv). All 18 figures appear in the manuscript. Do not infer figure numbers from the old asset filenames.

## Tables from saved records

These commands use the Python standard library and do not train a model:

```bash
python experiments/bert/collect_bert.py \
  --input-csv data/language/bert/glue_runs.csv --output-dir results/tables/bert
python experiments/vit/make_vit_tables.py \
  --input-csv data/vision/vit/tables/probe_results.csv --output-dir results/tables/vit
```

BERT aggregation first averages fine-tuning runs within each pretraining seed, then computes the mean and sample SD across pretraining seeds. ViT control tables retain their single-run interpretation. Outputs do not replace the distributed data.

## Rerunning experiments

The [experiment guide](experiments.md) points to each training or solver entry point. Scripts create run configurations and scalar outputs; vision and language training also writes local checkpoints needed for continuation and probing. Datasets, downloaded upstream sources and checkpoints are not bundled with the repository.

The plotting inputs are curated exports from the reported runs. A new experiment's output does not automatically replace them: schemas and aggregation conventions differ between the original exports and newer portable collectors. Inspect the new output and use the figure-to-data mapping to update a plotting input deliberately. Regenerate its checksum only after that update is verified. Editing data while leaving its checksum unchanged will correctly fail verification.

## Protocol distinctions preserved in the code

- Synthetic masks are Boolean Bernoulli masks; CNN patch masking fixes the number of hidden patches. These are different experimental models.
- Dynamic masking keeps resampling. The CNN convention of 32 presentations per image per epoch does not limit its total mask diversity to 32.
- The original CNN sweep, constant-LR control and 125,000-update control retain their distinct scheduler and checkpoint rules.
- AMP paper presets use p=0.05 teacher-informed initialization. Direct data-only inference uses a separately documented random initialization; do not conflate the two.
- Raw-feature logistic probing and the inverse-variance Bayesian downstream head are distinct readouts.
- The main bounded recovery law and the matched-moment bounded law in the threshold-ratio figure differ.
- Reported plot uncertainty uses sample SD. Saved SEM fields are also available where provided.

No packaging check establishes numerical convergence or bitwise reproducibility of full GPU training on different hardware. Residuals, solver statuses, seeds and checkpoint-selection settings remain part of the recorded results.
