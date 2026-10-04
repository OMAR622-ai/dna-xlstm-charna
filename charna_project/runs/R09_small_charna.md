# R09_small_charna: Small experiment, GBST-BERT (500 steps)

| | |
|---|---|
| Status | ✅ done |
| Kind | pretraining |
| Started | 2026-10-04 16:20 |
| Ended | 2026-10-04 19:55 |
| Model | GBST-BERT, final version, LR 5e-4 |

**Purpose.** Pretrain on ≈0.5 B bases (5% of the full run) to get early results.

**Command.**

```bash
MODEL=charna RUN_NAME=small_charna_500 MAX_STEPS=500 VAL_EVERY=100 LIMIT_VAL=0.02 CKPT_EVERY=100 bash scripts_pretrain/run_pretrain_charna.sh
```

**Outcome.** Validation loss 1.340 → 1.194 (test 1.196). Finished normally (the queue script was replaced while it ran, so its exit status was not logged; all 5 validation points and the final checkpoint exist).

## Settings (from the run's `config.json`)

| Setting | Value |
|---|---|
| Model | `charna_lm`, tokenizer `gbst` |
| Shape | d_model 256, 4 blocks, 8 heads, feed-forward 320 |
| GBST | max block 4, offsets True, pooling sum, pointwise layer True, calibration True, downsampling 1 |
| Norms | layernorm, QKNorm True |
| Steps | 500 (≈0.52 B bases) |
| Batch | 4 GPUs × 32 sequences × 8 accumulation = 1,024 sequences of 1024 bases |
| Optimizer | adamw, LR 0.0005, weight decay 0.1, betas [0.9, 0.95] |
| Schedule | cosine_warmup_timm, warmup 50.0 steps, minimum LR 0.0001 |
| Masking | mlm, probability 0.15; strand flipping (RC aug) True |
| Precision, clipping | fp32, gradient clip 1.0 |
| Validation | every 100 steps, on 2% of the batches of the validation and test sets |
| Seed | 2222 |

## Validation during training (`metrics.csv`)

Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.

| Step | Bases seen | Validation loss | Test loss |
|---|---|---|---|
| 100 | 0.105 B | 1.3399 | 1.3437 |
| 200 | 0.210 B | 1.3011 | 1.3013 |
| 300 | 0.315 B | 1.2146 | 1.2150 |
| 400 | 0.419 B | 1.2004 | 1.2018 |
| 500 | 0.524 B | 1.1944 | 1.1955 |

## Accuracy evaluation (test split)

Masked-base prediction on 2,048 test sequences of 1,024 bases (311,767 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 500 | 1.1944 | 3.302 | 43.94% | 53.9% | 32.0% | 36.1% | 47.9% |
| Transformer++ | 500 | 1.1958 | 3.306 | 43.77% | 51.0% | 33.0% | 31.5% | 52.8% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "A" (most common hidden base) | – | – | – | 29.42% | 100% | 0% | 0% | 0% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.4%, C 20.6%, G 20.8%, T 29.2%.

## Accuracy evaluation (valid split)

Masked-base prediction on 2,048 valid sequences of 1,024 bases (312,461 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 500 | 1.1945 | 3.302 | 44.12% | 54.6% | 30.0% | 33.5% | 50.5% |
| Transformer++ | 500 | 1.1955 | 3.305 | 43.95% | 51.5% | 30.9% | 29.3% | 55.2% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "T" (most common hidden base) | – | – | – | 29.97% | 0% | 0% | 0% | 100% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.5%, C 20.3%, G 20.2%, T 30.0%.

## Files

- [`results/pretrain/hg38/small_charna_500/config.json`](../results/pretrain/hg38/small_charna_500/config.json)
- [`results/pretrain/hg38/small_charna_500/metrics.csv`](../results/pretrain/hg38/small_charna_500/metrics.csv)
- [`results/pretrain/hg38/small_charna_500/model_config.json`](../results/pretrain/hg38/small_charna_500/model_config.json)
- [`results/pretrain/hg38/small_charna_500/train.log`](../results/pretrain/hg38/small_charna_500/train.log)
- [`results/small/eval_test.csv`](../results/small/eval_test.csv)
- [`results/small/eval_test.md`](../results/small/eval_test.md)
- [`results/small/eval_valid.csv`](../results/small/eval_valid.csv)
- [`results/small/eval_valid.md`](../results/small/eval_valid.md)
- [`results/small/queue.txt`](../results/small/queue.txt)

[← All runs](README.md)
