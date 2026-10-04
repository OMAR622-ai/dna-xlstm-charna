# GBST + BERT (ChaRNABERT-style) on DNA-xLSTM

This branch (`charna-gbst`) adds a ChaRNABERT-style model to the DNA-xLSTM code base: a
**Gradient-Based Subword Tokenizer (GBST)** in front of a **bidirectional BERT-style Transformer
encoder**, trained with **MLM** or **UL2**. It is pretrained on hg38 with the short-context
DNA-xLSTM protocol and evaluated on the 18-task **Nucleotide Transformer (NT) benchmark**, so the
results are directly comparable with the Bio-xLSTM paper.

Papers used (all hyperparameters below cite one of them):

| Short name | Paper |
|---|---|
| Bio-xLSTM | Schmidinger et al., *Bio-xLSTM: Generative modeling, representation and in-context learning of biological and chemical sequences*, arXiv:2411.04165 |
| ChaRNABERT | *ChaRNABERT*, arXiv:2411.11808 |
| Charformer | Tay et al., *Charformer: Fast Character Transformers via Gradient-based Subword Tokenization*, arXiv:2106.12672 |
| UL2 | Tay et al., *UL2: Unifying Language Learning Paradigms*, arXiv:2205.05131 |

## Status (4 October 2026)

| Step | State |
|---|---|
| Model, UL2, configs, tests | done, 22 unit tests pass |
| Overfit one batch | done: MLM loss < 0.05 after 158 steps, UL2 after 229 steps |
| Smoke run (50M tokens, both models) | done, see [Smoke run](#smoke-run-50m-tokens) |
| Full pretraining, GBST + BERT (10.5B tokens) | **running** (started 4 Oct 2026, ~40 h) |
| Full pretraining, Transformer++-2M baseline | queued after GBST + BERT |
| NT benchmark | pipeline ready and tested end to end; waits for pretrained checkpoints |
| Ablations (d_s, MLM vs UL2, RC aug, GBST vs embedding) | not started |

No final results are reported yet. Numbers in this file are from smoke runs only and are
labelled as such.

## What was added

All changes are additive: existing models and default behaviour are unchanged
(`mlm_strategy` defaults to `"mlm"`, new registry names only).

| File | Purpose |
|---|---|
| `models/charna/gbst.py` | GBST tokenizer (candidate blocks, scoring, calibration, optional downsampling) |
| `models/charna/modeling_charna.py` | Encoder (RoPE, SwiGLU, QKNorm, LayerNorm), MLM head, reverse-complement helpers |
| `models/charna/configuration_charna.py` | `CharnaConfig` (defaults = 2M model) |
| `configs/model/charna.yaml` | Hydra model config (`model=charna`) |
| `src/dataloaders/utils/ul2.py` | UL2 mixture of denoisers adapted to in-place `[MASK]` |
| `src/dataloaders/datasets/hg38_dataset.py`, `src/dataloaders/genomics.py` | `mlm_strategy: mlm \| ul2` switch (train split only) |
| `src/models/sequence/dna_embedding.py` | Downstream wrappers `DNAEmbeddingModelCharna` and `DNAEmbeddingModelTransformer` |
| `src/utils/registry.py` | `charna_lm`, `dna_embedding_charna`, `dna_embedding_llama`, `metrics_csv` |
| `src/callbacks/metrics_csv.py` | Writes validation/test metrics to `metrics.csv` (wandb is off on our server) |
| `scripts_pretrain/run_pretrain_charna.sh` | Pretraining (GBST + BERT or Transformer++, same protocol) |
| `scripts_downstream/run_nucleotide_charna.sh` | NT benchmark fine-tuning |
| `scripts_charna/summarize_nt.py` | NT results table (best grid point per task, mean ± std and range) |
| `scripts_charna/train_charna.py` | Launcher for GPUs without mamba_ssm / flash_attn (see [Hardware](#hardware-and-environment)) |
| `scripts_charna/overfit_one_batch.py`, `scripts_charna/smoke_run.sh` | Sanity checks |
| `tests/test_charna.py`, `tests/test_downstream.py` | Unit tests |

## Model

**GBST** (ChaRNABERT §3.2, Eq. 1–7; Charformer §2):

1. Character embedding, then a depthwise 1D convolution with kernel size M.
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
2,041,856 (Transformer++-2M with the same shape: 2,036,992); the difference is the GBST.

| GBST setting | Value | Source |
|---|---|---|
| Max block size M | 4 | **Not stated in ChaRNABERT.** Charformer illustrates M = 4 (§2.1, Fig. 2); chosen as default |
| Offsets | on | ChaRNABERT §3.2 (Eq. 3 uses offsets o) |
| Block pooling | sum | ChaRNABERT §3.2 |
| Conv kernel size | M | ChaRNABERT §3.2 ("convolution of kernel size equal to the maximum block size") |
| Score calibration | on | ChaRNABERT Eq. 6 |
| Downsampling d_s | 1 | ChaRNABERT §3.2 (downsampling removed) |

## Training protocol

Pretraining on hg38 follows the short-context DNA-xLSTM protocol (Bio-xLSTM App. C.5,
Tables A3 and A4), identical for GBST + BERT and the Transformer++-2M baseline:

| Setting | Value | Source |
|---|---|---|
| Context length | 1,024 | Table A3 |
| Tokens per step | 2²⁰ (1,024 sequences × 1,024) | App. C.5 |
| Update steps | 10,000 (≈10.5B tokens) | Table A4 |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 | Table A4 |
| Learning rate | 1e-2 (Transformer++-2M sweep 6e-3, 8e-3, **1e-2**) | Table A3 |
| Schedule | cosine decay, 1,000 warmup steps | Table A4 (repo: `warmup_t = 0.1 × max_steps`, `lr_min = 1e-4`) |
| Dropout | 0 | Table A4 |
| Gradient clipping | 1.0 | repo `run_pretrain_*.sh` |
| MLM | 15% of tokens; 80% `[MASK]`, 10% random, 10% unchanged | App. C.5 |
| Reverse complement | RC augmentation during pretraining (PH setting) | App. C.4 |
| Validation | every 2,000 steps (≈2/4/6/8/10B tokens) on 12.5% of the validation set (repo default) | — |

Tables A3/A4 list the optimizer and schedule for DNA-xLSTM (A4); A3 gives only shape and
learning rate for Transformer++, so the A4 values are used for both, as the repo scripts do.
Note that the repo's `scripts_pretrain/run_pretrain_llama.sh` uses `HIDDEN_DIM=1024` (4:1) and
LR 8e-3, which differ from Table A3 (1.25:1, 1e-2 in bold). We follow Table A3.

**UL2** (`MLM_STRATEGY=ul2`): the seven denoisers of UL2 Table 1, sampled uniformly per sequence:
R (μ=3, r=0.15), R (μ=8, r=0.15), S (r=0.25), X (μ=3, r=0.5), X (μ=8, r=0.5), X (μ=64, r=0.15),
X (μ=64, r=0.5). Spans are drawn with T5's `random_spans_noise_mask` (UL2 §9.2). Adaptations for
an encoder-only model (our choices, not from the paper): corrupted positions are replaced in place
by `[MASK]` and predicted at their positions (no sentinel tokens, no decoder); the S denoiser
masks a suffix whose length is drawn uniformly from [1, L/2]; no mode tokens. Validation always
uses standard 15% MLM so that the losses of MLM and UL2 runs are comparable.

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

All commands from the repository root unless stated otherwise. Data: `data/hg38/hg38.ml.fa`,
`hg38.ml.fa.fai`, `human-sequences.bed` (see the main README, "Data Preparation").

**Tests**

```bash
python -m unittest tests.test_charna tests.test_downstream -v
```

**Overfit one batch** (should reach loss < 0.05)

```bash
python scripts_charna/overfit_one_batch.py --strategy mlm
python scripts_charna/overfit_one_batch.py --strategy ul2
```

**Pretraining** (from `scripts_pretrain/`; every setting is an environment variable, see the script)

```bash
bash run_pretrain_charna.sh                                    # GBST + BERT, MLM, 10k steps
MODEL=llama bash run_pretrain_charna.sh                        # Transformer++-2M baseline
MLM_STRATEGY=ul2 RUN_NAME=charna_ul2 bash run_pretrain_charna.sh   # UL2
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

## Smoke run (50M tokens)

48 optimizer steps (≈50M tokens) per model, same data and logging as the full run, learning-rate
schedule compressed to 48 steps (≈5 warmup steps), validation on 64 batches. This only checks that
everything runs; it says nothing about final quality.

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

- **NT data loading (repo bug, not yet fixed in this branch).** The Hugging Face dataset
  `InstaDeepAI/nucleotide_transformer_downstream_tasks` is now stored as one parquet file per task,
  and `load_dataset(..., name=<task>)` in `src/dataloaders/datasets/nucleotide_transformer_dataset.py`
  silently loads **all 18 tasks** (461,850 training rows). Loading
  `data_files={split: f"{task}/{split}.parquet"}` instead gives the training-set sizes in
  `configs/dataset/nucleotide_transformer.yaml` for 17 of 18 tasks; `promoter_tata` has 5,509 rows
  instead of 5,517. The fix changes shared code and is pending approval.
- The global batch is reached with gradient accumulation (4 GPUs × 32 × 32) instead of the
  paper's larger per-step GPU batch; mathematically the same update, but slower.
