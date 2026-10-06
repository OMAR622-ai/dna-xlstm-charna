# R20_eval_charna_final: Accuracy of the final GBST-BERT model

| | |
|---|---|
| Status | ✅ done |
| Kind | evaluation |
| Started | 2026-10-06 ≈14:00 |
| Ended | 2026-10-06 ≈14:50 |
| Model | R16 final checkpoint (step 10,000), and R14 (step 500) for comparison |

**Purpose.** Measure the final GBST-BERT model right away (on the CPU, without waiting for Transformer++); same 2,048 sequences and masks as R12.

**Command.**

```bash
python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/full_charna_mlm:GBST-BERT-full outputs/pretrain/hg38/small_charna_500_lr2e-3:GBST-BERT-small-2e-3 --split {valid,test} --n_seqs 2048 --device cpu
```

**Outcome.** Final GBST-BERT on test DNA: loss 0.9753, accuracy 56.44% (A 61.8%, C 48.5%, G 48.8%, T 62.1%); validation: 0.9689, 56.80%. Random guess 25%, always the most common base 29.4%. The 500-step model reached 47.37%.

## Accuracy evaluation (test split)

Masked-base prediction on 2,048 test sequences of 1,024 bases (311,767 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT-full | 10000 | 0.9753 | 2.652 | 56.44% | 61.8% | 48.5% | 48.8% | 62.1% |
| GBST-BERT-small-2e-3 | 500 | 1.1375 | 3.119 | 47.37% | 55.9% | 37.5% | 35.7% | 54.1% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "A" (most common hidden base) | – | – | – | 29.42% | 100% | 0% | 0% | 0% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.4%, C 20.6%, G 20.8%, T 29.2%.

## Accuracy evaluation (valid split)

Masked-base prediction on 2,048 valid sequences of 1,024 bases (312,461 hidden bases, 15% masking, seed 0).

| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |
|---|---|---|---|---|---|---|---|---|
| GBST-BERT-full | 10000 | 0.9689 | 2.635 | 56.80% | 62.4% | 47.3% | 47.4% | 64.0% |
| GBST-BERT-small-2e-3 | 500 | 1.1468 | 3.148 | 47.07% | 56.2% | 34.7% | 32.7% | 56.1% |
| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |
| Always "T" (most common hidden base) | – | – | – | 29.97% | 0% | 0% | 0% | 100% |

Accuracy = the model's best guess among A, C, G, T is the true base. Base composition of the hidden positions: A 29.5%, C 20.3%, G 20.2%, T 30.0%.

## Files

- [`results/full/eval_charna_only/eval_test.csv`](../results/full/eval_charna_only/eval_test.csv)
- [`results/full/eval_charna_only/eval_test.md`](../results/full/eval_charna_only/eval_test.md)
- [`results/full/eval_charna_only/eval_valid.csv`](../results/full/eval_charna_only/eval_valid.csv)
- [`results/full/eval_charna_only/eval_valid.md`](../results/full/eval_charna_only/eval_valid.md)

[← All runs](README.md)
