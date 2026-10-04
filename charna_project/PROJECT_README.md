# GBST + BERT (ChaRNABERT-style) on DNA-xLSTM

This branch (`charna-gbst`) adds a ChaRNABERT-style model to the DNA-xLSTM code base: a
**Gradient-Based Subword Tokenizer (GBST)** in front of a **bidirectional BERT-style Transformer
encoder**, built as described in ChaRNABERT §3 and trained with masked language modelling (MLM)
or, optionally, UL2 masking (ChaRNABERT §4.1). It is pretrained on hg38 with the short-context
DNA-xLSTM protocol and evaluated on the 18-task **Nucleotide Transformer (NT) benchmark**, so the
results are directly comparable with the Bio-xLSTM paper.

**Quick links**

| | |
|---|---|
| Everything in one folder (code, data, results) | [`charna_project/`](./) |
| Results of the runs | [`charna_project/results/`](results/) |
| Data (interval list, hg38 sample, download script) | [`charna_project/datasets/`](datasets/), [`scripts_charna/download_data.sh`](../scripts_charna/download_data.sh) |
| Model code (GBST + encoder) | [`models/charna/`](../models/charna/) |
| Original DNA-xLSTM README (installation, data, other models) | [`README_DNA_XLSTM.md`](../README_DNA_XLSTM.md) |

Papers used (all hyperparameters below cite one of them):

| Short name | Paper |
|---|---|
| Bio-xLSTM | Schmidinger et al., *Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences*, arXiv:2411.04165 |
| ChaRNABERT | Morales-Pastor et al., *Character-level Tokenizations as Powerful Inductive Biases for RNA Foundational Models*, arXiv:2411.11808 |
| Charformer | Tay et al., *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization*, arXiv:2106.12672 |
| UL2 | Tay et al., *UL2: Unifying Language Learning Paradigms*, arXiv:2205.05131 |

## Status (4 October 2026)

| Step | State |
|---|---|
| Model, UL2, configs, tests | done, 23 unit tests pass |
| Overfit one batch | done: loss < 0.05 after 164 steps (MLM) and 230 steps (UL2) |
| Smoke run (50M tokens, both models) | done, see [Smoke run](#smoke-run-50m-tokens) |
| Small experiment (≈0.5B tokens, 4 NT tasks, both models) | **running** since 4 Oct 2026 16:20 CEST, see [Small experiment](#small-experiment-05b-tokens-4-nt-tasks) |
| Full pretraining (10.5B tokens, both models) | on hold until the small experiment is reviewed. Two started runs were stopped: one with the earlier GBST and LR 1e-2 (~1 h), one with the final settings (~30 min) |
| NT benchmark (18 tasks) | pipeline ready and tested end to end; waits for the full pretraining |
| Ablations (d_s, MLM vs UL2, RC aug, GBST vs embedding) | not started |

No final results are reported yet. Numbers in this file are from smoke runs and the small
experiment and are labelled as such. All result files are in
[`charna_project/results/`](results/); the data that fits in git is in
[`charna_project/datasets/`](datasets/).

## What was added

All changes are additive: existing models and default behaviour are unchanged
(`mlm_strategy` defaults to `"mlm"`, new registry names only).

| File | Purpose |
|---|---|
| `models/charna/gbst.py` | GBST tokenizer (candidate blocks, scoring, calibration, optional downsampling) |
| `models/charna/modeling_charna.py` | Encoder (RoPE, SwiGLU, QKNorm, LayerNorm), MLM head, reverse-complement helpers |
| `models/charna/configuration_charna.py` | `CharnaConfig` (defaults = 2M model) |
| `configs/model/charna.yaml` | Hydra model config (`model=charna`) |
| `src/dataloaders/utils/ul2.py` | UL2 masking (ChaRNABERT §4.1) adapted to in-place `[MASK]` |
| `src/dataloaders/datasets/hg38_dataset.py`, `src/dataloaders/genomics.py` | `mlm_strategy: mlm \| ul2` switch (train split only) |
| `src/dataloaders/datasets/nucleotide_transformer_dataset.py` | Fix: load each NT task's own parquet file (see [Known issues](#known-issues)) |
| `src/models/sequence/dna_embedding.py` | Downstream wrappers `DNAEmbeddingModelCharna` and `DNAEmbeddingModelTransformer` |
| `src/utils/registry.py` | `charna_lm`, `dna_embedding_charna`, `dna_embedding_llama`, `metrics_csv` |
| `src/callbacks/metrics_csv.py` | Writes validation/test metrics to `metrics.csv` (wandb is off on our server) |
| `scripts_pretrain/run_pretrain_charna.sh` | Pretraining (GBST + BERT or Transformer++, same protocol) |
| `scripts_downstream/run_nucleotide_charna.sh` | NT benchmark fine-tuning |
| `scripts_charna/summarize_nt.py` | NT results table (best grid point per task, mean ± std and range) |
| `scripts_charna/train_charna.py` | Launcher for GPUs without mamba_ssm / flash_attn (see [Hardware](#hardware-and-environment)) |
| `scripts_charna/overfit_one_batch.py`, `scripts_charna/smoke_run.sh` | Sanity checks |
| `tests/test_charna.py`, `tests/test_downstream.py` | Unit tests |
| `scripts_charna/download_data.sh`, `scripts_charna/make_hg38_sample.py` | Download and check the full data; build the small hg38 sample |
| `scripts_charna/collect_results.sh` | Copy run results (metrics, configs, summaries) into `charna_project/results/` |
| `charna_project/` | The whole project in one folder: copies of all files above, a patch of the changes to existing files, the data that fits in git (`datasets/`) and the run results (`results/`) |

## Model

**GBST** (ChaRNABERT §3.2, Eq. 1–7; Charformer §2):

1. Character embedding, then a depthwise 1D convolution with kernel size M (Eq. 1) followed by
   a pointwise d × d projection. Eq. 1 names only the depthwise convolution, but Table 9 counts
   the GBST convolution as d_model × M + d_model² parameters, which is a depthwise-separable
   convolution; we follow the table (`gbst_pointwise_conv=true`).
2. For every position *i*, candidate blocks of size *b* = 1..M with all offsets (the blocks of
   size *b* that contain *i*), each represented by sum pooling: K = M(M+1)/2 candidates.
3. A linear scorer (no bias) gives one score per candidate; softmax over the K candidates.
4. Score calibration P̃ = softmax(PPᵀ)P (ChaRNABERT Eq. 6).
5. Output: score-weighted sum of the candidates, one latent per nucleotide.
6. Optional Charformer mean-pool downsampling by d_s. ChaRNABERT removes downsampling
   ("We also differ from the original implementation in the removal of the downsampling
   procedure", §3.2), so the default is **d_s = 1** (single-nucleotide resolution).

**Encoder** (ChaRNABERT §3.3): pre-norm Transformer encoder blocks with RoPE, SwiGLU, QKNorm and
LayerNorm, bidirectional attention (PyTorch SDPA), MLM head without bias.

**Size**: the shape of Bio-xLSTM's DNA-Transformer++-2M (Table A3): d_model 256, 4 blocks,
8 heads, up-projection 1.25:1 (SwiGLU hidden size 320), no norm/linear biases. Parameter count
2,107,392 (Transformer++-2M with the same shape: 2,036,992, 3.5% fewer); the difference is the
GBST (70,144 parameters). The ChaRNABERT sizes (8M to 650M, Table 1) are for RNA; the project
uses the 2M size so that results compare with DNA-xLSTM-2M.

| GBST setting | Value | Source |
|---|---|---|
| Max block size M | 4 | **Not stated in ChaRNABERT.** Charformer illustrates M = 4 (§2.1, Fig. 2); chosen as default |
| Offsets | on | ChaRNABERT §3.2 (Eq. 3 uses offsets o) |
| Block pooling | sum | ChaRNABERT §3.2 |
| Conv kernel size | M | ChaRNABERT §3.2 ("convolution of kernel size equal to the maximum block size") |
| Pointwise d × d after the depthwise conv | on | ChaRNABERT Table 9 (d_model × M + d_model²) |
| Score calibration | on | ChaRNABERT Eq. 6 |
| Downsampling d_s | 1 | ChaRNABERT §3.2 (downsampling removed) |

**Paper vs. this implementation.** Everything in ChaRNABERT §3 is implemented as written, with
these exceptions:

| ChaRNABERT | Here | Why |
|---|---|---|
| FlashAttention 2 (§3.3) | PyTorch scaled-dot-product attention | FlashAttention 2 needs Ampere GPUs; ours are GTX 1080 Ti. The attention computed is the same |
| bf16, DeepSpeed ZeRO (§4.3) | fp32, Lightning DDP | no bf16 on GTX 1080 Ti |
| Eq. 3 offsets written as X[i+o : i+o+b] | blocks that contain position i, X[i−o : i−o+b] | the literal index gives position i blocks that do not contain it; the text ("starting from position 0 and position 1", reading frames) describes shifted segmentations, which is what we implement (see `models/charna/gbst.py`) |
| Eq. 9 written as σ(xW₁) ⊙ swish(xW₂) | standard SwiGLU, swish(xW₁) ⊙ xW₂ | the text cites SwiGLU (Shazeer, 2020), which is the standard form; Eq. 9 as printed is a different function |
| Max block size M | 4 | not stated in ChaRNABERT (see table above) |
| Norm biases | none | not stated in ChaRNABERT; Bio-xLSTM Table A3 uses no biases |
| RNA data, context 2k–8k, bf16 (§4–5) | hg38, context 1k, fp32 | the project follows the DNA-xLSTM pretraining protocol so the NT results compare with Bio-xLSTM |
| BEACON downstream tasks, class-token heads (§6) | NT benchmark, mean pooling + linear head | same reason |

## Training protocol

Pretraining on hg38 follows the short-context DNA-xLSTM protocol (Bio-xLSTM App. C.5,
Tables A3 and A4), identical for GBST + BERT and the Transformer++-2M baseline:

| Setting | Value | Source |
|---|---|---|
| Context length | 1,024 | Table A3 |
| Tokens per step | 2²⁰ (1,024 sequences × 1,024) | App. C.5 |
| Update steps | 10,000 (≈10.5B tokens) | Table A4 |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 | Table A4 |
| Learning rate | GBST + BERT: 5e-4; Transformer++-2M: 1e-2 | GBST + BERT: ChaRNABERT Table 2 (best value for the 5M model, with GBST and with EM); Transformer++: Bio-xLSTM Table A3 (sweep 6e-3, 8e-3, **1e-2**) |
| Schedule | cosine decay, 1,000 warmup steps, minimum LR 1e-4 | Table A4 (repo `hg38.yaml`: `warmup_t = 0.1 × max_steps`, `lr_min = 1e-4`). ChaRNABERT gives no schedule |
| Dropout | 0 | Table A4 |
| Gradient clipping | 1.0 | repo `run_pretrain_*.sh` |
| MLM | 15% of tokens; 80% `[MASK]`, 10% random, 10% unchanged | App. C.5 |
| Reverse complement | RC augmentation during pretraining (PH setting) | App. C.4 |
| Validation | every 2,000 steps (≈2/4/6/8/10B tokens) on 12.5% of the validation set (repo default) | — |

Tables A3/A4 list the optimizer and schedule for DNA-xLSTM (A4); A3 gives only shape and
learning rate for Transformer++, so the A4 values are used for both, as the repo scripts do.
Note that the repo's `scripts_pretrain/run_pretrain_llama.sh` uses `HIDDEN_DIM=1024` (4:1) and
LR 8e-3, which differ from Table A3 (1.25:1, 1e-2 in bold). We follow Table A3.

**UL2** (`MLM_STRATEGY=ul2`; ChaRNABERT §4.1): ChaRNABERT trains with MLM and, as a second
masking strategy, UL2 (§4.1; "MLM/UL2 loss", §4.3), with a shared mask token and the same
prediction as MLM (Eq. 15). It gives no span lengths, rates or mode probabilities, so we use the
seven denoisers of UL2 Table 1, sampled uniformly per sequence: R (μ=3, r=0.15), R (μ=8, r=0.15),
S (r=0.25), X (μ=3, r=0.5), X (μ=8, r=0.5), X (μ=64, r=0.15), X (μ=64, r=0.5). Spans are drawn with
T5's `random_spans_noise_mask` (UL2 §9.2). ChaRNABERT labels the modes differently: its
"S-denoising" (short spans) is UL2's R, its "R-denoising" (predict the continuation) is UL2's S
(suffix masking); X is the same. Adaptations for an encoder-only model (our choices): corrupted
positions are replaced in place by `[MASK]` and predicted at their positions; the suffix length is
drawn uniformly from [1, L/2]; no mode tokens. Validation always uses standard 15% MLM so that
MLM and UL2 runs are comparable.

**NT benchmark** (Bio-xLSTM App. C.5, Table A5; repo `scripts_downstream/run_nucleotide.sh`):

| Setting | Value | Source |
|---|---|---|
| Head | mean pooling over the final hidden states + linear layer | App. C.5 ("classification head ... trained on average-pooled activations"), repo `SequenceDecoder` |
| Fine-tuning | all parameters trained | App. C.5 |
| RC | PH: RC augmentation on the training split, test prediction averaged over forward and RC | App. C.4, repo `conjoin_test` |
| Epochs, seeds | 20 epochs, 10 train/validation splits | App. C.5 |
| Grid | batch {64, 128, 256, 512} × LR {4e-4, 6e-4, 8e-4, 1e-3, 2e-3} | App. C.5 |
| Selection | best grid point per task by validation score averaged over the 10 splits | Table A5 caption |
| Reported | mean over 10 seeds ± std; the range (max − min), used for error bars in Bio-xLSTM, is reported too | — |

The project spec asked for an MLP head; the paper and the repo use mean pooling with a single
linear layer, which we keep for comparability. The full grid is 18 tasks × 20 grid points × 10
seeds = 3,600 fine-tuning runs; the default in our script is the repo script's grid
(batch {64, 128}), and the budget will be decided once pretraining is done.
Table A5 lists 3e-3 for two tasks (Promoter TATA and Splice Sites Donors, PH), which is outside
the LR grid stated in App. C.5.

No Transformer++ NT results are reported in Bio-xLSTM, so we fine-tune our own Transformer++-2M
baseline with exactly the same pipeline.

## Hardware and environment

Runs use the JKU server `student01` with 4× NVIDIA GTX 1080 Ti (11 GB, compute capability 6.1),
shared with other users. Consequences:

- **fp32** training: the GPUs have no bf16 and no fast fp16 (the paper used bf16 on A100s).
- `mamba_ssm`, `causal_conv1d` and `flash_attn` cannot be built for these GPUs. The repo imports
  them at module level even when they are unused, so `scripts_charna/train_charna.py` installs
  inert placeholders for the missing packages and then runs the unchanged `train.py`. Using a
  Mamba/Caduceus model would fail loudly; GBST + BERT and Transformer++ do not need them.
  DNA-xLSTM itself cannot be retrained here, so its numbers are taken from the paper.
- Global batch of 1,024 sequences = 4 GPUs × 32 × 32 gradient-accumulation steps.
  Lightning 1.8 counts `val_check_interval` in micro-batches, so the scripts scale it by the
  accumulation factor.

Python environment (Python 3.9): `torch==2.2.0+cu121`, `torchvision==0.17.0`,
`pytorch-lightning==1.8.6`, `hydra-core==1.3.2`, `transformers==4.38.1`,
`torchmetrics==1.3.0`, `datasets==2.15.0`. Newer torch wheels (e.g. 2.8 + cu128) have no kernels
for compute capability 6.1.

## How to run

All commands from the repository root unless stated otherwise.

**Data**

```bash
bash scripts_charna/download_data.sh   # full hg38 (3.1 GB) into data/hg38/ + md5 check, NT tasks into the HF cache
```

The full genome is too large for GitHub (files over 100 MB are refused). A 4.3 MB sample of 32 real
hg38 intervals is in `charna_project/datasets/hg38_sample/`; it works with every command below
in place of `data/hg38/...` (set `HG38_FASTA` / `HG38_BED` for the tests, or `dataset.fasta_file`
/ `dataset.bed_file` for training), but is only meant for trying the code.

**Tests**

```bash
python -m unittest tests.test_charna tests.test_downstream -v
```

**Overfit one batch** (should reach loss < 0.05)

```bash
python scripts_charna/overfit_one_batch.py --fasta data/hg38/hg38.ml.fa --bed data/hg38/human-sequences.bed
python scripts_charna/overfit_one_batch.py --fasta data/hg38/hg38.ml.fa --bed data/hg38/human-sequences.bed --strategy ul2
```

**Pretraining** (from `scripts_pretrain/`; every setting is an environment variable, see the script)

```bash
bash run_pretrain_charna.sh                                    # GBST + BERT, MLM, 10k steps
MODEL=llama bash run_pretrain_charna.sh                        # Transformer++-2M baseline
MLM_STRATEGY=ul2 RUN_NAME=charna_ul2 bash run_pretrain_charna.sh   # UL2 masking
DOWNSAMPLE=2 RUN_NAME=charna_ds2 bash run_pretrain_charna.sh       # ablation d_s = 2
TOKENIZER=embedding RUN_NAME=charna_emb bash run_pretrain_charna.sh  # ablation: no GBST
```

Outputs go to `outputs/pretrain/hg38/<RUN_NAME>/`: `metrics.csv` (step, tokens, val/test loss),
`model_config.json`, `checkpoints/test/loss.ckpt` (best) and `checkpoints/last.ckpt`.

**NT benchmark** (from `scripts_downstream/`)

```bash
PRETRAIN_DIR=../outputs/pretrain/hg38/full_charna_mlm MODEL=charna bash run_nucleotide_charna.sh
PRETRAIN_DIR=../outputs/pretrain/hg38/full_llama_mlm  MODEL=llama  bash run_nucleotide_charna.sh
# subsets: TASKS="H3 enhancers" LRS="1e-3" BATCH_SIZES="128" SEEDS="1 2 3"
python ../scripts_charna/summarize_nt.py ../outputs/downstream/nt_cv10_ep20 --markdown nt_results.md
```

Finished runs are skipped when the script is restarted.

## Small experiment (0.5B tokens, 4 NT tasks)

A small version of the whole pipeline, to see results before the 40 h full runs
(`~/charna/run_small_queue.sh` on the server):

1. **Pretraining**, both models, 500 steps × 2²⁰ tokens ≈ 0.5B tokens (5% of the full run), with
   the full-run settings; the learning-rate schedule scales with the step count (50 warmup steps,
   cosine to 1e-4). GBST + BERT at LR 5e-4, Transformer++-2M at LR 1e-2. Validation every 100 steps
   on 2% of the validation set (the full run uses 12.5%).
2. **NT fine-tuning** on 4 of the 18 tasks, one per group, with the DNA-xLSTM-PH settings of
   Bio-xLSTM Table A5, 3 seeds (the full benchmark uses 10), 20 epochs, PH:

   | Task | Metric | LR | Batch |
   |---|---|---|---|
   | H3 | MCC | 8e-4 | 128 |
   | enhancers | MCC | 2e-3 | 512 |
   | promoter_tata | F1 | 3e-3 | 128 |
   | splice_sites_all | accuracy | 8e-4 | 64 |

**Results:** running; this section will be filled in from `charna_project/results/` when the runs
finish. These are early results after 5% of the pretraining, for checking the pipeline and the
trend, not final numbers.

## Smoke run (50M tokens)

48 optimizer steps (≈50M tokens) per model, same data and logging as the full run, learning-rate
schedule compressed to 48 steps (≈5 warmup steps), validation on 64 batches. This only checks that
everything runs; it says nothing about final quality.
The GBST + BERT smoke run used the earlier GBST without the pointwise projection
(`gbst_pointwise_conv=false`, 2,041,856 parameters).

| Tokens | GBST + BERT val / test loss | Transformer++-2M val / test loss |
|---|---|---|
| 12.6M | 1.392 / 1.414 | 1.397 / 1.423 |
| 25.2M | 1.370 / 1.379 | 1.363 / 1.374 |
| 37.7M | 1.366 / 1.378 | 1.343 / 1.350 |
| 50.3M | 1.364 / 1.377 | 1.325 / 1.339 |

Loss in nats per masked nucleotide. Throughput on the shared 1080 Tis: ≈13.7 s per optimizer step
(≈75k tokens/s) for both models; peak memory 3.4 GB (GBST + BERT) and 1.2 GB (Transformer++,
which uses gradient checkpointing). Estimated full run: ≈40 h per model including validation.

## Known issues

- **NT data loading (repo bug, fixed in this branch).** The Hugging Face dataset
  `InstaDeepAI/nucleotide_transformer_downstream_tasks` is now stored as one parquet file per task,
  and the original `load_dataset(..., name=<task>)` silently loaded **all 18 tasks** (461,850
  training rows). `src/dataloaders/datasets/nucleotide_transformer_dataset.py` now loads
  `data_files={split: f"{task}/{split}.parquet"}`, which gives the training-set sizes in
  `configs/dataset/nucleotide_transformer.yaml` for 17 of 18 tasks; `promoter_tata` has 5,509 rows
  instead of 5,517 (the config value is only used to size the learning-rate schedule).
- The global batch is reached with gradient accumulation (4 GPUs × 32 × 32) instead of the
  paper's larger per-step GPU batch; mathematically the same update, but slower.
