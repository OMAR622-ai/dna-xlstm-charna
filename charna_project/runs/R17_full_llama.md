# R17_full_llama: Full pretraining, Transformer++ (10.5 B bases)

| | |
|---|---|
| Status | ⏳ running |
| Kind | pretraining |
| Started | 2026-10-06 08:31 (student07) |
| Ended | – |
| Model | Transformer++-2M, LR 1e-2, fixed data loader |

**Purpose.** The comparison model at full scale (from scratch, after R16; replaces R19).

**Command.**

```bash
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh
```

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `llama_lm` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| Steps | 10,000 (≈10.49 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.01, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 1000.0 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 2000 steps, on 12.5% of the batches of the validation and test sets |
| Seed | 2222 |

## Validation during training (`metrics.csv`)

Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.

| Step | Bases seen | Validation loss | Test loss |
|---|---|---|---|
| 2000 | 2.097 B | 1.0516 | 1.0369 |
| 4000 | 4.194 B | 1.0153 | 1.0051 |

## Files

- [`results/pretrain/hg38/full_llama_mlm/config.json`](../results/pretrain/hg38/full_llama_mlm/config.json)
- [`results/pretrain/hg38/full_llama_mlm/metrics.csv`](../results/pretrain/hg38/full_llama_mlm/metrics.csv)
- [`results/pretrain/hg38/full_llama_mlm/model_config.json`](../results/pretrain/hg38/full_llama_mlm/model_config.json)
- [`results/pretrain/hg38/full_llama_mlm/train.log`](../results/pretrain/hg38/full_llama_mlm/train.log)
- [`results/full/eval_charna_only/eval_test.csv`](../results/full/eval_charna_only/eval_test.csv)
- [`results/full/eval_charna_only/eval_test.md`](../results/full/eval_charna_only/eval_test.md)
- [`results/full/eval_charna_only/eval_valid.csv`](../results/full/eval_charna_only/eval_valid.csv)
- [`results/full/eval_charna_only/eval_valid.md`](../results/full/eval_charna_only/eval_valid.md)
- [`results/full/queue.txt`](../results/full/queue.txt)
- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
