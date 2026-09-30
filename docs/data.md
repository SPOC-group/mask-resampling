# Data guide

All CSV files use their first row as a header. Empty entries, NaNs and numerical status strings are preserved. The manifest lists row counts, columns and hashes for every distributed CSV. Every retained measurement is copied without changing its text representation or row order.

## Synthetic experiments

`data/synthetic/masking_ratio/` contains linear masked-autoencoder predictions and empirical results at alpha=8, beta=1, d=2000, with gamma=1/4+(9/4)B and B distributed as Beta(1,2). ERM uses 30 seeds, 4,000 full-batch Adam updates and ridge 1e-4. The theory uses 200-node Gauss-Legendre integration over the full bounded support. The reconstruction files contain losses centered by the zero-decoder baseline, without the ridge penalty. The dynamic empirical training objective is estimated with evaluation masks on the training samples.

`data/synthetic/bounded_alpha/` contains state-evolution curves for linear, ReLU, ELU and tanh, with finite K=1,2,4 and dynamic masking, at beta=1 and rho=0.75 for the same bounded law. The autoencoders use ridge 1e-4. The empirical summary has 17 sample-complexity values from 0.1 to 10, d=2000 and 30 seeds per activation/protocol/alpha group. The state-evolution file retains its status, outer residual and inner-proximal diagnostics, including incomplete entries. Missing values are not filled during packaging. The theory plotting behavior is preserved from the source notebook.

The AMP summary has d=4000 and ten seeds per alpha. It uses estimated coordinate variances and teacher-informed initialization u0=sqrt(p)u_star+sqrt(1-p)z with p=0.05. It is not a data-only initialization. The available configuration uses initial damping 0.75 with adaptive cycle damping, a 20,000-iteration cap and tolerance 1e-6. The Bayes CSV supplies the state-evolution reference. For the downstream comparison, all learned directions use the raw-feature logistic probe with 100 labels and 10,000 test examples.

`data/synthetic/full_reconstruction/` contains corresponding standard unmasked-autoencoder summaries and 2,040 per-seed records: four activations, 17 alpha values and 30 seeds. These runs use d=2000, 4,000 updates and ridge 1e-4. The absence of masking is explicit in the data. The notebook checks that means and sample SDs reproduce from these per-seed records.

`data/synthetic/three_noise_laws/` supplies the combined three-panel recovery figure. The variance laws are homogeneous gamma=1; bounded gamma=1/4+(9/4)Beta(1,2); and unbounded gamma=3/4+(1/4)exp(Z-1/2), Z standard normal. All have mean one. Method-specific dimensions, seed counts, status and uncertainty are in the CSV. The theoretical zero PCA reference in the unbounded panel is supplied analytically; failed numerical records are not thereby relabeled converged.

`data/synthetic/downstream_labels/` contains 960 saved predictions for linear dynamic masking: 120 alpha values for each label count 1,2,4,8,16,32,64,128. This is a deterministic prediction plot, not a new training experiment.

## Spectral comparisons

`data/spectral/dimension_scaling/` contains 90 PCA groups (nine dimensions and ten sample complexities), each with 30 seeds. Cosine is absolute, not squared. The large-dimension fits use d>=1600, fixed exponent -1/2 and a fitted amplitude; the exponent is not estimated.

`data/spectral/heterogeneous_estimators/` compares ordinary PCA, diagonal deletion, HeteroPCA, estimated whitening, dynamic masking and Bayes/AMP under bounded and shifted-lognormal noise. The non-AMP empirical methods use d=2000 and 20 seeds; AMP uses d=4000 and ten seeds. Eligibility and convergence-related flags are retained. Hollow AMP markers identify groups containing iteration-limited runs. The single combined CSV contains the plotted numerical summaries and theoretical references.

## Thresholds

`data/thresholds/finite_k_ratios/` contains the archived finite-K stability calculations. The figure selects only homogeneous and bounded matched-moment laws, beta from 0.1 to 1, five masking ratios and eight finite K values. The denominator is the corresponding finite-beta dynamic threshold. The black line is the weak-signal prediction 1+[1/(2 rho(1-rho))-1]/K. The matched-moment bounded law is different from the Beta law used in the main recovery figures. Other archived rows are retained in the CSV but not drawn.


## Vision experiments

`data/vision/cnn/masking_ratio/` and `mask_diversity/` contain consolidated frozen-probe summaries from the original CNN protocol. Results are CIFAR-10 test accuracies. The finite-K bank and dynamic resampling are distinct conditions; the original validation-selected sweep does not impose the same total update count across K.

`data/vision/cnn/augmentation/` contains per-run results at eleven masking ratios, five seeds, two masking modes and two augmentation conditions. The plotting code converts percentage accuracy to a fraction and computes sample SD across seeds.

`data/vision/cnn/full_reconstruction/` contains training/validation reconstruction losses and frozen-probe results for the unmasked model. The marked checkpoint is selected by validation accuracy; the displayed downstream outcome is test accuracy.

`data/vision/cnn/plateau_lr/` contains the original reconstruction histories, validation-selected checkpoints and downstream test evaluations. `data/vision/cnn/fixed_lr/` contains the fixed-learning-rate control, its selected checkpoints and separate short diagnostic reruns covering early updates. Their manifests preserve the relevant schedule settings. `data/vision/cnn/fixed_budget/` contains the separate five-seed control with constant LR=0.001, exactly 125,000 pretraining updates and pretraining early stopping disabled. These three collections share rho=0.625 but differ in training and selection policy; the notebook keeps them distinct.

`data/vision/vit/masking_ratio/` contains top-1 and top-5 validation summaries, including three-seed masking curves and the separately supplied reference points. The source pretraining-checkpoint epoch is retained as scientific metadata. `data/vision/vit/augmentation/` contains the paired single-seed static/dynamic comparison with and without cropping/flipping. `data/vision/vit/full_reconstruction/` contains the unmasked reconstruction and frozen-probe trajectories. All ViT accuracy plots in this collection use ImageNet-100 validation results, not an independent test set.

`data/vision/vit/tables/` contains the two ViT control tables. `probe_results.csv` has eight completed probes with scientific settings and final validation scores; `probe_history.csv` has all 480 epoch evaluations. Accuracies are percentages. The history field `epoch` is zero-based; `pretraining_completed_epochs` is the count at the evaluated encoder checkpoint. Both longer encoders are probed for 50 and 90 epochs. The MAE encoder is evaluated after 641 epochs of a 1,600-epoch LR schedule; the original pretraining run stopped after 647 completed epochs. The probe-augmentation pairs share the same 100-epoch encoder. All rows use rho=0.75, initialization/training/probe seed zero, no pretraining crop/flip and no uniformity penalty. No across-seed uncertainty is estimated. Console-summary values retain three decimals, while native scalar-JSON values retain their saved precision. `training_budget.csv`, `probe_augmentation.csv` and `tables.md` are derived by `experiments/vit/make_vit_tables.py`. Raw logs, private paths and model weights are excluded.

## Language experiments and tables

`data/language/bert/glue_runs.csv` contains exactly 162 metric entries from 144 final fine-tuning runs: three masking conditions, three pretraining seeds, two fixed fine-tuning seeds per pretraining seed, and eight tasks. MNLI has two development splits. Only the final MRPC/CoLA/RTE recipe at LR=5e-5 is included. Original metric text, scientific settings and row order are preserved; private source/checkpoint paths are replaced by checkpoint names and verified update counts. Values are fractions: accuracy for MNLI/QNLI/QQP/SST-2, Spearman for STS-B, and MCC for MRPC/CoLA/RTE. The evaluator's original three-decimal rounding of accuracy and MCC is retained.

`pretraining.csv` contains nine scalar summaries, checked against the final checkpoints. Every run completes 99,953 updates. The selected K=1 checkpoints are at update 29,985; K=10 and dynamic use the final checkpoint. MLM losses are in bits per predicted token. `per_pretraining_seed.csv`, `summary.csv` and `tables.md` are recomputed summaries on the x100 scale: first average the two fine-tuning runs within each pretraining seed, then calculate the mean and sample SD across three pretraining seeds. SEM is provided separately. AVG is an eight-task average using MNLI-m; it is not an official GLUE benchmark score. These tables are reproduced with `experiments/bert/collect_bert.py` and do not add another plotting notebook.

## Units and uncertainty

Sample complexity is alpha=n/d; rho is the masking ratio. Cosine means the absolute cosine similarity wherever the teacher sign is arbitrary. Accuracy is shown as a fraction unless a diagnostic table explicitly labels percent. Fields ending in `std`, `sd` or `sample_std` supply sample SD. Available `sem` fields are separate quantities, and assertions sometimes check SD/sqrt(seed count) consistency. No uncertainty bars are invented for a single seed or deterministic theoretical curve.
