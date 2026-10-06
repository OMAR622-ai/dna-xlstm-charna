# R16_full_charna: Full pretraining, GBST-BERT (10.5 B bases)

| | |
|---|---|
| Status | ✅ done |
| Kind | pretraining |
| Started | 2026-10-05 05:00 (student01); resumed 17:21 and, after the fix, 21:58 (student07) |
| Ended | 2026-10-06 08:31 |
| Model | GBST-BERT, final version, LR 2e-3 (best of R09/R13–R15 by validation loss at step 500) |

**Purpose.** The main run.

**Command.**

```bash
MODEL=charna LR=<best> RUN_NAME=full_charna_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh   (from ~/charna/run_full_after_lr.sh)
```

**Outcome.** First validation point (step 2,000, 2.1 B bases): validation loss 1.0559, test loss 1.0402 (student01). Moved to student07 at step 3,000 (17:21). Crashed at 18:51 during the step-4,000 validation: the data loader returned an empty sequence. Cause: a bug in the repository's hg38 reader (DataLoader workers shared one open FASTA file and its read position; measured: 0.35% of sequences wrong with 2 workers per GPU as on student01, 13–33% with 12). Fixed (genome file reopened per worker, test added) and resumed from the step-3,000 checkpoint at 21:58 (5 Oct), so steps 3,000–4,000 are redone with correct data. Finished at 08:31 on 6 Oct (exit 0): validation loss 1.0014 (step 4,000), 0.9853 (6,000), 0.9765 (8,000); the step-10,000 validation was not run (after resuming, Lightning counts one step ahead, so the scheduled check fell on step 10,001); final model saved as charna_project/models/full_charna_mlm/final_step10000.ckpt and measured with evaluate_mlm.py (R18).

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
| 4001 | 4.195 B | 1.0014 | 0.9929 |
| 6001 | 6.293 B | 0.9853 | 0.9783 |
| 8001 | 8.390 B | 0.9765 | 0.9703 |

## Accuracy evaluation (test split)

Masked-base prediction on 2,048 test sequences of 1,024 bases (311,767 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 10000 | 0.9753 | 2.652 | 56.44% | 61.8% | 48.5% | 48.8% | 62.1% |
| Transformer++ | 10000 | 0.9774 | 2.657 | 56.33% | 61.3% | 48.9% | 48.9% | 61.9% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "A" (most common hidden base) | – | – | – | 29.42% | 100% | 0% | 0% | 0% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.4%, C 20.6%, G 20.8%, T 29.2%.

## Accuracy evaluation (valid split)

Masked-base prediction on 2,048 valid sequences of 1,024 bases (312,461 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT | 10000 | 0.9689 | 2.635 | 56.80% | 62.4% | 47.3% | 47.4% | 64.0% |
| Transformer++ | 10000 | 0.9728 | 2.645 | 56.65% | 62.1% | 47.6% | 47.2% | 63.7% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "T" (most common hidden base) | – | – | – | 29.97% | 0% | 0% | 0% | 100% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.5%, C 20.3%, G 20.2%, T 30.0%.

## Files

- [`results/pretrain/hg38/full_charna_mlm/config.json`](../results/pretrain/hg38/full_charna_mlm/config.json)
- [`results/pretrain/hg38/full_charna_mlm/metrics.csv`](../results/pretrain/hg38/full_charna_mlm/metrics.csv)
- [`results/pretrain/hg38/full_charna_mlm/model_config.json`](../results/pretrain/hg38/full_charna_mlm/model_config.json)
- [`results/pretrain/hg38/full_charna_mlm/train.log`](../results/pretrain/hg38/full_charna_mlm/train.log)
- [`results/full/eval_charna_only/eval_test.csv`](../results/full/eval_charna_only/eval_test.csv)
- [`results/full/eval_charna_only/eval_test.md`](../results/full/eval_charna_only/eval_test.md)
- [`results/full/eval_charna_only/eval_valid.csv`](../results/full/eval_charna_only/eval_valid.csv)
- [`results/full/eval_charna_only/eval_valid.md`](../results/full/eval_charna_only/eval_valid.md)
- [`results/full/eval_test.csv`](../results/full/eval_test.csv)
- [`results/full/eval_test.md`](../results/full/eval_test.md)
- [`results/full/eval_valid.csv`](../results/full/eval_valid.csv)
- [`results/full/eval_valid.md`](../results/full/eval_valid.md)
- [`results/full/queue.txt`](../results/full/queue.txt)
- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
