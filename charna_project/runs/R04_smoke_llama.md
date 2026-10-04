# R04_smoke_llama: Smoke run, Transformer++ (48 steps)

| | |
|---|---|
| Status | ✅ done |
| Kind | pretraining |
| Started | 2026-10-04 13:42 |
| Ended | 2026-10-04 13:57 |
| Model | Transformer++-2M, LR 1e-2 |

**Purpose.** Same smoke run for the comparison model.

**Command.**

```bash
(second half of scripts_charna/smoke_run.sh, MODEL=llama)
```

**Outcome.** Done in 894 s, peak GPU memory 1,200 MiB (uses gradient checkpointing). Validation loss 1.397 → 1.325.

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `llama_lm` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| Steps | 48 (≈0.05 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.01, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 4.800000000000001 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 12 steps, on 64 batches of the validation and test sets |
| Seed | 2222 |

## Validation during training (`metrics.csv`)

Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.

| Step | Bases seen | Validation loss | Test loss |
|---|---|---|---|
| 12 | 0.013 B | 1.3969 | 1.4234 |
| 24 | 0.025 B | 1.3628 | 1.3738 |
| 36 | 0.038 B | 1.3431 | 1.3499 |
| 48 | 0.050 B | 1.3252 | 1.3393 |

## Summary file

```
smoke_charna exit=0 wall_s=930 peak_mem_mib=3388
smoke_llama exit=0 wall_s=894 peak_mem_mib=1200
SMOKE_DONE
```

## Files

- [`results/pretrain/hg38/smoke_llama/config.json`](../results/pretrain/hg38/smoke_llama/config.json)
- [`results/pretrain/hg38/smoke_llama/metrics.csv`](../results/pretrain/hg38/smoke_llama/metrics.csv)
- [`results/pretrain/hg38/smoke_llama/model_config.json`](../results/pretrain/hg38/smoke_llama/model_config.json)
- [`results/pretrain/hg38/smoke_llama/train.log`](../results/pretrain/hg38/smoke_llama/train.log)
- [`results/smoke/summary.txt`](../results/smoke/summary.txt)

[← All runs](README.md)
