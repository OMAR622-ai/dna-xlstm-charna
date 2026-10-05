# R16_full_charna: Full pretraining, GBST-BERT (10.5 B bases)

| | |
|---|---|
| Status | ⏳ running |
| Kind | pretraining |
| Started | 2026-10-05 05:00 |
| Ended | – |
| Model | GBST-BERT, final version, LR 2e-3 (best of R09/R13–R15 by validation loss at step 500) |

**Purpose.** The main run.

**Command.**

```bash
MODEL=charna LR=<best> RUN_NAME=full_charna_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh   (from ~/charna/run_full_after_lr.sh)
```

**Outcome.** First validation point (step 2,000, 2.1 B bases): validation loss 1.0559, test loss 1.0402. Checkpoints of steps 2,000 and 2,500 saved to charna_project/models/full_charna_mlm/ on 5 Oct 16:52 (moving to a new server); the run kept training on student01.

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `charna_lm`, tokenizer `gbst` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| GBST | max block 4, offsets True, pooling sum, pointwise layer True, calibration True, downsampling 1 |
| Norms | layernorm, QKNorm True |
| Steps | 10,000 (≈10.49 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.002, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 1000.0 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 2000 steps, on 12.5% of the batches of the validation and test sets |
| Seed | 2222 |

## Validation during training (`metrics.csv`)

Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.

| Step | Bases seen | Validation loss | Test loss |
|---|---|---|---|
| 2000 | 2.097 B | 1.0559 | 1.0402 |

## Files

- [`results/pretrain/hg38/full_charna_mlm/config.json`](../results/pretrain/hg38/full_charna_mlm/config.json)
- [`results/pretrain/hg38/full_charna_mlm/metrics.csv`](../results/pretrain/hg38/full_charna_mlm/metrics.csv)
- [`results/pretrain/hg38/full_charna_mlm/model_config.json`](../results/pretrain/hg38/full_charna_mlm/model_config.json)
- [`results/pretrain/hg38/full_charna_mlm/train.log`](../results/pretrain/hg38/full_charna_mlm/train.log)
- [`results/full/queue.txt`](../results/full/queue.txt)
- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
