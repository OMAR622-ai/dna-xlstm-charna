# R08_full_v2_stopped: Full pretraining attempt 2 (stopped)

| | |
|---|---|
| Status | ⏹ stopped |
| Kind | pretraining |
| Started | 2026-10-04 15:10 |
| Ended | 2026-10-04 ≈16:18 (stopped) |
| Model | GBST-BERT, final version, LR 5e-4 |

**Purpose.** The planned 10,000-step run with the final settings.

**Command.**

```bash
MODEL=charna RUN_NAME=full_charna_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh
```

**Outcome.** Stopped after 1 h 8 min (≈ step 200, training loss ≈1.36) to run the small experiment first and see results sooner (decided together). No validation point had been reached.

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `charna_lm`, tokenizer `gbst` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| GBST | max block 4, offsets True, pooling sum, pointwise layer True, calibration True, downsampling 1 |
| Norms | layernorm, QKNorm True |
| Steps | 10,000 (≈10.49 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.0005, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 1000.0 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 2000 steps, on 12.5% of the batches of the validation and test sets |
| Seed | 2222 |

## Files

- [`results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/config.json`](../results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/config.json)
- [`results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/model_config.json`](../results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/model_config.json)
- [`results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/train.log`](../results/pretrain/hg38/stopped_charna_pointwise_lr5e-4/train.log)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
