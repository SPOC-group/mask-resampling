# Figure guide

Run `python scripts/plot.py --figure ID --no-tex` from the repository root. All figure code is in [notebooks/figures.ipynb](../notebooks/figures.ipynb); outputs share `plots/`. The machine-readable [figure index](figure_index.csv) lists every input path.

| ID | Figure | Output PDF | Notebook cell |
|---|---|---|---|
| 1 | Mask resampling across synthetic, CNN and ViT models | `plots/figure_01_mask_resampling.pdf` | `plot-01` |
| 2 | Recovery under homogeneous, bounded and unbounded noise | `plots/figure_02_noise_laws.pdf` | `plot-02` |
| 3 | Recovery and downstream accuracy across activations and mask diversity | `plots/figure_03_activation_and_mask_diversity.pdf` | `plot-03` |
| 4 | Finite mask diversity in the synthetic model and CNN | `plots/figure_04_finite_mask_diversity.pdf` | `plot-04` |
| 5 | ViT augmentation comparison | `plots/figure_05_vit_augmentation.pdf` | `plot-05` |
| 6 | PCA dimension scaling | `plots/figure_06_pca_dimension_scaling.pdf` | `plot-06` |
| 7 | ViT top-1 validation accuracy | `plots/figure_07_vit_top1.pdf` | `plot-07` |
| 8 | Spectral estimators under heterogeneous noise | `plots/figure_08_spectral_estimators.pdf` | `plot-08` |
| 9 | Dynamic masking versus full reconstruction | `plots/figure_09_dynamic_vs_full.pdf` | `plot-09` |
| 10 | Centered reconstruction errors | `plots/figure_10_reconstruction_errors.pdf` | `plot-10` |
| 11 | Downstream label-count dependence | `plots/figure_11_downstream_label_count.pdf` | `plot-11` |
| 12 | CNN full-reconstruction training and probing | `plots/figure_12_cnn_full_reconstruction.pdf` | `plot-12` |
| 13 | ViT full-reconstruction training and probing | `plots/figure_13_vit_full_reconstruction.pdf` | `plot-13` |
| 14 | CNN augmentation comparison | `plots/figure_14_cnn_augmentation.pdf` | `plot-14` |
| 15 | CNN training with plateau learning-rate scheduling | `plots/figure_15_cnn_plateau_training.pdf` | `plot-15` |
| 16 | CNN training with a fixed learning rate | `plots/figure_16_cnn_fixed_lr_training.pdf` | `plot-16` |
| 17 | CNN downstream comparison and fixed-budget control | `plots/figure_17_cnn_budget_control.pdf` | `plot-17` |
| 18 | Finite-K threshold ratios | `plots/figure_18_finite_k_threshold_ratios.pdf` | `plot-18` |

## Figure inputs

### Figure 1: Mask resampling across synthetic, CNN and ViT models

- [data/synthetic/masking_ratio/downstream_probe_per_seed.csv](../data/synthetic/masking_ratio/downstream_probe_per_seed.csv)
- [data/synthetic/masking_ratio/generic_k_downstream_theory.csv](../data/synthetic/masking_ratio/generic_k_downstream_theory.csv)
- [data/vision/cnn/masking_ratio/comparison_plot_data.csv](../data/vision/cnn/masking_ratio/comparison_plot_data.csv)
- [data/vision/vit/masking_ratio/linear_probe_top5_accuracy_mean_across_initialization_seeds.csv](../data/vision/vit/masking_ratio/linear_probe_top5_accuracy_mean_across_initialization_seeds.csv)

### Figure 2: Recovery under homogeneous, bounded and unbounded noise

- [data/synthetic/three_noise_laws/three_noise_oracle_amp_plant0p05.csv](../data/synthetic/three_noise_laws/three_noise_oracle_amp_plant0p05.csv)

### Figure 3: Recovery and downstream accuracy across activations and mask diversity

- [data/synthetic/bounded_alpha/amp_d4000_oracle_plant0p05_summary.csv](../data/synthetic/bounded_alpha/amp_d4000_oracle_plant0p05_summary.csv)
- [data/synthetic/bounded_alpha/bayes_optimal.csv](../data/synthetic/bounded_alpha/bayes_optimal.csv)
- [data/synthetic/bounded_alpha/erm_summary.csv](../data/synthetic/bounded_alpha/erm_summary.csv)
- [data/synthetic/bounded_alpha/state_evolution.csv](../data/synthetic/bounded_alpha/state_evolution.csv)

### Figure 4: Finite mask diversity in the synthetic model and CNN

- [data/synthetic/masking_ratio/downstream_probe_per_seed.csv](../data/synthetic/masking_ratio/downstream_probe_per_seed.csv)
- [data/synthetic/masking_ratio/generic_k_downstream_theory.csv](../data/synthetic/masking_ratio/generic_k_downstream_theory.csv)
- [data/synthetic/masking_ratio/k_view_downstream_summary.csv](../data/synthetic/masking_ratio/k_view_downstream_summary.csv)
- [data/vision/cnn/mask_diversity/aggregate_test_accuracy.csv](../data/vision/cnn/mask_diversity/aggregate_test_accuracy.csv)
- [data/vision/cnn/masking_ratio/comparison_plot_data.csv](../data/vision/cnn/masking_ratio/comparison_plot_data.csv)

### Figure 5: ViT augmentation comparison

- [data/vision/vit/augmentation/linear_probe_val_accuracy_geom_vs_no_geom.csv](../data/vision/vit/augmentation/linear_probe_val_accuracy_geom_vs_no_geom.csv)

### Figure 6: PCA dimension scaling

- [data/spectral/dimension_scaling/pca_cosine_dimension_fixed_slope_fits.csv](../data/spectral/dimension_scaling/pca_cosine_dimension_fixed_slope_fits.csv)
- [data/spectral/dimension_scaling/pca_summary.csv](../data/spectral/dimension_scaling/pca_summary.csv)

### Figure 7: ViT top-1 validation accuracy

- [data/vision/vit/augmentation/linear_probe_val_accuracy_geom_vs_no_geom.csv](../data/vision/vit/augmentation/linear_probe_val_accuracy_geom_vs_no_geom.csv)
- [data/vision/vit/masking_ratio/linear_probe_top1_accuracy_mean_across_initialization_seeds.csv](../data/vision/vit/masking_ratio/linear_probe_top1_accuracy_mean_across_initialization_seeds.csv)

### Figure 8: Spectral estimators under heterogeneous noise

- [data/spectral/heterogeneous_estimators/comparison.csv](../data/spectral/heterogeneous_estimators/comparison.csv)

### Figure 9: Dynamic masking versus full reconstruction

- [data/synthetic/bounded_alpha/amp_d4000_oracle_plant0p05_summary.csv](../data/synthetic/bounded_alpha/amp_d4000_oracle_plant0p05_summary.csv)
- [data/synthetic/bounded_alpha/bayes_optimal.csv](../data/synthetic/bounded_alpha/bayes_optimal.csv)
- [data/synthetic/bounded_alpha/erm_summary.csv](../data/synthetic/bounded_alpha/erm_summary.csv)
- [data/synthetic/bounded_alpha/state_evolution.csv](../data/synthetic/bounded_alpha/state_evolution.csv)
- [data/synthetic/full_reconstruction/full_reconstruction_alpha_per_seed.csv](../data/synthetic/full_reconstruction/full_reconstruction_alpha_per_seed.csv)
- [data/synthetic/full_reconstruction/full_reconstruction_alpha_summary.csv](../data/synthetic/full_reconstruction/full_reconstruction_alpha_summary.csv)

### Figure 10: Centered reconstruction errors

- [data/synthetic/masking_ratio/reconstruction_erm_summary.csv](../data/synthetic/masking_ratio/reconstruction_erm_summary.csv)
- [data/synthetic/masking_ratio/reconstruction_state_evolution.csv](../data/synthetic/masking_ratio/reconstruction_state_evolution.csv)

### Figure 11: Downstream label-count dependence

- [data/synthetic/downstream_labels/linear_dynamic_nds_state_evolution.csv](../data/synthetic/downstream_labels/linear_dynamic_nds_state_evolution.csv)

### Figure 12: CNN full-reconstruction training and probing

- [data/vision/cnn/full_reconstruction/linear_probe_only_plot_data.csv](../data/vision/cnn/full_reconstruction/linear_probe_only_plot_data.csv)
- [data/vision/cnn/full_reconstruction/pretraining_loss_aggregate.csv](../data/vision/cnn/full_reconstruction/pretraining_loss_aggregate.csv)

### Figure 13: ViT full-reconstruction training and probing

- [data/vision/vit/full_reconstruction/full_reconstruction_linear_probe_accuracy_vs_pretraining_epoch_early_recovery.csv](../data/vision/vit/full_reconstruction/full_reconstruction_linear_probe_accuracy_vs_pretraining_epoch_early_recovery.csv)
- [data/vision/vit/full_reconstruction/full_reconstruction_train_val_loss_and_linear_probe_top5_vs_pretraining_epochs.csv](../data/vision/vit/full_reconstruction/full_reconstruction_train_val_loss_and_linear_probe_top5_vs_pretraining_epochs.csv)

### Figure 14: CNN augmentation comparison

- [data/vision/cnn/augmentation/per_run_test_accuracy.csv](../data/vision/cnn/augmentation/per_run_test_accuracy.csv)

### Figure 15: CNN training with plateau learning-rate scheduling

- [data/vision/cnn/plateau_lr/selected_checkpoints.csv](../data/vision/cnn/plateau_lr/selected_checkpoints.csv)
- [data/vision/cnn/plateau_lr/test_accuracy.csv](../data/vision/cnn/plateau_lr/test_accuracy.csv)
- [data/vision/cnn/plateau_lr/training_history.csv](../data/vision/cnn/plateau_lr/training_history.csv)

### Figure 16: CNN training with a fixed learning rate

- [data/vision/cnn/fixed_lr/early_training.csv](../data/vision/cnn/fixed_lr/early_training.csv)
- [data/vision/cnn/fixed_lr/early_validation.csv](../data/vision/cnn/fixed_lr/early_validation.csv)
- [data/vision/cnn/fixed_lr/selected_checkpoints.csv](../data/vision/cnn/fixed_lr/selected_checkpoints.csv)
- [data/vision/cnn/fixed_lr/test_accuracy.csv](../data/vision/cnn/fixed_lr/test_accuracy.csv)
- [data/vision/cnn/fixed_lr/training_history.csv](../data/vision/cnn/fixed_lr/training_history.csv)

### Figure 17: CNN downstream comparison and fixed-budget control

- [data/vision/cnn/fixed_budget/per_run_test_metrics.csv](../data/vision/cnn/fixed_budget/per_run_test_metrics.csv)
- [data/vision/cnn/fixed_lr/test_accuracy.csv](../data/vision/cnn/fixed_lr/test_accuracy.csv)
- [data/vision/cnn/plateau_lr/test_accuracy.csv](../data/vision/cnn/plateau_lr/test_accuracy.csv)

### Figure 18: Finite-K threshold ratios

- [data/thresholds/finite_k_ratios/validated_ratio_data_matched_three_laws_grid.csv](../data/thresholds/finite_k_ratios/validated_ratio_data_matched_three_laws_grid.csv)
