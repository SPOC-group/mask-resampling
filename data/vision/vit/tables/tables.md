# ViT control tables

ImageNet-100 validation accuracy (%, higher is better), from the final probe epoch. Each condition uses initialization and training/probe seed zero. All encoders use rho=0.75, no pretraining crop/flip and no uniformity penalty. Batch sizes below are effective batches. These are single-run results; no seed uncertainty is estimated.

## Longer training

| schedule | pretraining_epochs | pretraining_effective_batch | probe_epochs | probe_effective_batch | top1_pct | top5_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Main comparison | 100 | 128 | 50 | 256 | 39.96 | 68.78 |
| U-MAE | 200 | 1024 | 50 | 256 | 47.56 | 76.14 |
| U-MAE | 200 | 1024 | 90 | 16384 | 61.50 | 86.04 |
| MAE | 641 | 4096 | 50 | 256 | 51.38 | 79.60 |
| MAE | 641 | 4096 | 90 | 16384 | 59.12 | 84.76 |

Schedule names denote adaptations of U-MAE and MAE optimization recipes. Repeated encoder rows evaluate the same pretrained checkpoint. The MAE encoder is checkpoint 640 (641 completed epochs) of a 1,600-epoch learning-rate schedule; the original run continued to 647 completed epochs before stopping.

## Probe augmentation

| masks | crop_flip_top1_pct | crop_flip_top5_pct | no_crop_flip_top1_pct | no_crop_flip_top5_pct |
| --- | --- | --- | --- | --- |
| Static | 25.34 | 51.20 | 26.22 | 53.28 |
| Dynamic | 39.96 | 68.78 | 41.98 | 71.06 |

Each pair reuses the same 100-epoch encoder and 50-epoch probe settings; only probe-training cropping/flipping changes. Validation preprocessing is always deterministic.

The adjacent probe_results.csv retains the measured terminal metrics and settings; probe_history.csv contains all 480 probe epochs. Console-derived metrics retain their original three-decimal precision; scalar-JSON metrics retain the saved precision.
