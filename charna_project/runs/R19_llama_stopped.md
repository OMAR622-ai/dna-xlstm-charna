# R19_llama_stopped: Full pretraining, Transformer++, first attempt (stopped)

| | |
|---|---|
| Status | ⏹ stopped |
| Kind | pretraining |
| Started | 2026-10-05 18:51 (student07) |
| Ended | 2026-10-05 21:56 (stopped) |
| Model | Transformer++-2M, LR 1e-2 |

**Purpose.** The comparison model at full scale (started automatically after R16 crashed).

**Command.**

```bash
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 NUM_DEVICES=4 MICRO_BATCH=32 NUM_WORKERS=4 bash scripts_pretrain/run_pretrain_charna.sh
```

**Outcome.** Stopped at ≈ step 2,500 because it used the buggy data loader with 4 workers per GPU (many wrong sequences). Validation loss at step 2,000: 1.0512 — not used for comparisons. Restarted from scratch as R17 with the fixed loader.

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
| 2000 | 2.097 B | 1.0512 | 1.0369 |

## Files

- [`results/pretrain/hg38/stopped_llama_buggy_loader/config.json`](../results/pretrain/hg38/stopped_llama_buggy_loader/config.json)
- [`results/pretrain/hg38/stopped_llama_buggy_loader/metrics.csv`](../results/pretrain/hg38/stopped_llama_buggy_loader/metrics.csv)
- [`results/pretrain/hg38/stopped_llama_buggy_loader/model_config.json`](../results/pretrain/hg38/stopped_llama_buggy_loader/model_config.json)
- [`results/pretrain/hg38/stopped_llama_buggy_loader/train.log`](../results/pretrain/hg38/stopped_llama_buggy_loader/train.log)

[← All runs](README.md)
