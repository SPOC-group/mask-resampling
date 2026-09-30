# BERT-Medium: final paper experiments

Run all commands below from the repository root (`mask-resampling/`).

This directory reproduces the **final** K=1, K=10 and dynamic experiments: nine pretraining runs and 144 GLUE fine-tuning runs. The presets use the successful learning rate **5e-5 for MRPC, CoLA and RTE**. Earlier collapsed runs, learning-rate sweeps, shorter pretraining pilots and unused fine-tuning seeds are not included.

## Install

Use Linux, Python 3.11 and a separate environment. The recorded runtime used PyTorch 2.1.2 with CUDA 12.1; choose a compatible NVIDIA driver. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pip==23.3.2
python -m pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/bert.txt
python experiments/bert/prepare_source.py
python -m pip install --no-build-isolation --no-deps -e experiments/bert/vendor/dinkytrain
PYTHONPATH=experiments/bert/vendor/dinkytrain python experiments/bert/check_masking.py
```

A C/C++ compiler and Python development headers are required for fairseq's extensions. `--no-deps` is intentional: this source includes the experiment's Python 3.11/OmegaConf compatibility changes, while upstream package metadata declares older Hydra/OmegaConf bounds. Use the supplied versions together. DeepSpeed and Apex are not used in these presets.

`prepare_source.py` downloads a pinned **public upstream** DinkyTrain archive, verifies its SHA-256 and applies the supplied numerical source overlay. `--archive FILE` supports an already downloaded copy. No private repository is needed. [ATTRIBUTION.md](ATTRIBUTION.md) and `LICENSE` retain the DinkyTrain/fairseq credits and MIT license.

## External data

Use DinkyTrain's public preprocessed [Wikipedia–BookCorpus](https://huggingface.co/datasets/princeton-nlp/wikibook_fairseq_format) and [GLUE](https://huggingface.co/datasets/princeton-nlp/glue_fairseq_format) resources. Download these separately; text and weights are not redistributed here.

Pretraining uses **only `bin-shard0-8`**, not rotation through the other shards. It contains `dict.txt`, `train.bin`, `train.idx`, `valid.bin`, `valid.idx`. The wrapper checks the original file hashes in `pretraining_data.json`. Preserve the directory basename `bin-shard0-8`: it enters the deterministic mask RNG namespace. Parent directories can change freely. Fine-tuning expects `<GLUE_ROOT>/<task>/bin`, with lowercase task names `mnli,qnli,qqp,sst2,stsb,mrpc,cola,rte`.

## Run

Create an output root separate from the supplied plotting inputs. Each pretraining task uses **four GPUs**; each fine-tuning task uses **one GPU**. Allocate GPUs externally and select them through `CUDA_VISIBLE_DEVICES`.

```bash
python experiments/bert/run_bert.py --stage pretrain --list
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/bert/run_bert.py --stage pretrain --task 0 \
  --data-root /path/to/wikibook_fairseq_format/bin-shard0-8 --output-root runs/bert

python experiments/bert/run_bert.py --stage finetune --list
CUDA_VISIBLE_DEVICES=0 python experiments/bert/run_bert.py --stage finetune --task 0 \
  --data-root /path/to/glue_fairseq_format --output-root runs/bert

python experiments/bert/collect_bert.py --runs-root runs/bert --output-dir runs/bert/summary
```

Pretraining task indices are `0-8`; fine-tuning indices are `0-143`. Independent tasks can run in parallel on allocated GPUs. If `--task` is omitted, the wrapper reads `SLURM_ARRAY_TASK_ID`. Each fine-tuning task requires its pretraining run to have finished. `--dry-run` prints the exact commands without loading data or using a GPU. Completed outputs with matching invocation records are verified and reused. Use `--resume` only for interrupted runs with their original output tree; inherited fairseq checkpoint/resumption behavior is retained. External experiment tracking is disabled.

## Exact protocol

- **Model/data:** `roberta_medium`, pre-LayerNorm, eight layers, hidden width 512, FFN width 2048, eight heads, maximum positions 512, approximately 51.5M parameters. Fixed contiguous token blocks have length at most 128; the deterministic training prefix contains 3,756,032 sequences, with 1,941 held-out sequences. Dropout and attention dropout are 0.1.
- **Masking:** select 15% of token positions and replace every selected position with the mask token. No random-token replacements or unchanged selected targets. Fixed masks depend on seed, sequence and view; K=1 and K=10 reuse one or ten views. Dynamic masking uses the same fixed text blocks and changes corruption by training epoch. Validation uses one fixed view within each seed, shared across conditions.
- **Pretraining:** seeds 0,1,2; training seed and mask seed coincide. All nine runs finish 99,953 optimizer updates. Four GPUs, 128 sequences per GPU, accumulation 8: nominal effective batch 4,096. Adam betas (0.9,0.98), epsilon 1e-6, weight decay 0.01, no clipping. Peak LR 0.002; cosine schedule, 1,999 warmup updates. FP16 initial loss scale 8, scale window 99,954 and tolerance 0.1. Validation/checkpoint interval 9,995 updates, plus the terminal evaluation. The budget was sized to 109 nominal corpus passes; this is not a guarantee that every sequence has exactly 109 presentations, because the inherited iterator, partial epochs and FP16 overflow handling determine actual presentations.
- **Encoder checkpoint used for GLUE:** K=1 loads `checkpoint_best.pt` (minimum held-out MLM loss; update 29,985 for all three archived seeds). K=10 and dynamic load `checkpoint_last.pt` at update 99,953. Dynamic seeds 1 and 2 have slightly lower recorded MLM loss at the preceding evaluation; their **final** checkpoints are nevertheless the ones used for the paper's downstream results. All K=1 runs also complete the full pretraining budget.
- **Fine-tuning seeds:** pretraining seed 0 uses fine-tuning seeds 0,2; seed 1 uses 0,1; seed 2 uses 0,2. The same pairs are used for every condition and task. These are fixed selections for reproducing the table, not a search for the best fine-tuning seeds.
- **Fine-tuning:** full-model training for ten epochs; final fine-tuned checkpoint evaluated on development sets. Adam betas (0.9,0.98), epsilon 1e-6, weight decay 0.1, no clipping, dropout/attention dropout 0.1, FP16. Warmup covers floor(6% of the task's update horizon), followed by polynomial decay with power one. The horizon is floor(10 × training examples / batch size); actual updates retain the original batching and overflow behavior.

| Tasks | Learning rate | Physical/effective batch | Reported metric |
|---|---:|---:|---|
| MNLI, QNLI, QQP, SST-2 | 1e-5 | 32 | Accuracy |
| STS-B | 2e-5 | 16 | Spearman correlation |
| MRPC, CoLA, RTE | **5e-5** | 16 | Matthews correlation |

The 5e-5 learning rate was selected in a preliminary sweep using K=10, pretraining seed 0, and applied to all masking conditions and seeds. It is the final successful recipe, not a condition-specific choice. Classification uses cross-entropy; STS-B uses squared error. MNLI-m and MNLI-mm are two development splits of one task. No test-set claim or checkpoint selection by GLUE score is implied.

## Reproduce the reported tables without training

The final scalar records are in `data/language/bert/`. They contain exactly the selected 144 runs (162 metric entries because MNLI has two splits), plus nine pretraining summaries. Personal paths and source filenames are removed; metric values, hyperparameters, seeds and actual checkpoint/update counts are retained.

```bash
python experiments/bert/collect_bert.py --input-csv data/language/bert/glue_runs.csv \
  --output-dir runs/bert/saved_tables
```

First average the two fine-tuning runs within each pretraining seed. Then calculate the mean and **sample SD across the three pretraining seeds**. The collector also supplies SEM as a separate column; the paper tables use SD. AVG is the unweighted mean over eight task scores, using MNLI-m and excluding MNLI-mm. It mixes task-specific metrics and is not an official GLUE score. The archived averages are K=1 **62.71**, K=10 **65.79**, dynamic **66.42**.

The original evaluator rounds accuracy and MCC to three decimal places before saving; the table reproduction retains that precision. Spearman uses average ranks for ties. Small and variable scores, including RTE, remain included. New runs save checkpoints locally for pretraining/fine-tuning and scalar JSON/CSV outputs; these runtime files are not part of the release. No plots are generated here.
