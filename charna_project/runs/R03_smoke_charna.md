# R03_smoke_charna: Smoke run, GBST-BERT (48 steps)

| | |
|---|---|
| Status | ✅ done |
| Kind | pretraining |
| Started | 2026-10-04 13:26 |
| Ended | 2026-10-04 13:42 |
| Model | GBST-BERT, first version (no pointwise layer), LR 1e-2 |

**Purpose.** Short trial of the full pipeline (≈50 M bases) to measure speed and memory and to see that the loss falls.

**Command.**

```bash
cd scripts_pretrain && bash ../scripts_charna/smoke_run.sh   (MAX_STEPS=48 VAL_EVERY=12 LIMIT_VAL=64)
```

**Outcome.** Done in 930 s, peak GPU memory 3,388 MiB. Validation loss 1.392 → 1.364. Learning-rate schedule compressed to 48 steps.

**Notes.** Superseded model version and learning rate; a check, not a result.

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `charna_lm`, tokenizer `gbst` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| GBST | max block 4, offsets True, pooling sum, pointwise layer False, calibration True, downsampling 1 |
| Norms | layernorm, QKNorm True |
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
| 12 | 0.013 B | 1.3921 | 1.4141 |
| 24 | 0.025 B | 1.3703 | 1.3786 |
| 36 | 0.038 B | 1.3656 | 1.3777 |
| 48 | 0.050 B | 1.3641 | 1.3771 |

## Summary file

```
smoke_charna exit=0 wall_s=930 peak_mem_mib=3388
smoke_llama exit=0 wall_s=894 peak_mem_mib=1200
SMOKE_DONE
```

## Files

- [`results/pretrain/hg38/smoke_charna/config.json`](../results/pretrain/hg38/smoke_charna/config.json)
- [`results/pretrain/hg38/smoke_charna/metrics.csv`](../results/pretrain/hg38/smoke_charna/metrics.csv)
- [`results/pretrain/hg38/smoke_charna/model_config.json`](../results/pretrain/hg38/smoke_charna/model_config.json)
- [`results/pretrain/hg38/smoke_charna/train.log`](../results/pretrain/hg38/smoke_charna/train.log)
- [`results/smoke/summary.txt`](../results/smoke/summary.txt)

[← All runs](README.md)
