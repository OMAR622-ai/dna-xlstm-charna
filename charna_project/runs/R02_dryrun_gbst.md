# R02_dryrun_gbst: 3-step trial on 4 GPUs

| | |
|---|---|
| Status | ✅ done |
| Kind | check |
| Started | 2026-10-04 13:11 |
| Ended | 2026-10-04 13:24 |
| Model | GBST-BERT, first version |

**Purpose.** Check that training, validation and checkpointing work end to end with 4 GPUs (DDP).

**Command.**

```bash
MAX_STEPS=3 ... bash scripts_pretrain/run_pretrain_charna.sh (3 steps)
```

**Outcome.** Passed: trained, validated (val loss 1.68, test loss 1.69), saved a checkpoint, exited 0. Measured speed: about 0.9 s per micro-batch.

**Notes.** Several launcher problems were found and fixed before this worked (placeholder packages, DDP processes, validation interval).

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `charna_lm`, tokenizer `gbst` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| GBST | max block 4, offsets True, pooling sum, pointwise layer False, calibration True, downsampling 1 |
| Norms | layernorm, QKNorm True |
| Steps | 3 (≈0.00 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.01, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 0.30000000000000004 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 3 steps, on 4 batches of the validation and test sets |
| Seed | 2222 |

## Files

- [`results/pretrain/hg38/dryrun_gbst/config.json`](../results/pretrain/hg38/dryrun_gbst/config.json)
- [`results/pretrain/hg38/dryrun_gbst/model_config.json`](../results/pretrain/hg38/dryrun_gbst/model_config.json)
- [`results/pretrain/hg38/dryrun_gbst/train.log`](../results/pretrain/hg38/dryrun_gbst/train.log)

[← All runs](README.md)
