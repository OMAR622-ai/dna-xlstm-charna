# R18_eval_full: Accuracy of the full models

| | |
|---|---|
| Status | ✅ done |
| Kind | evaluation |
| Started | 2026-10-06 23:26 |
| Ended | 2026-10-06 23:27 |
| Model | R16 and R17 final checkpoints |

**Purpose.** Same accuracy test as R12, after full training.

**Command.**

```bash
python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ --split {valid,test} --n_seqs 2048
```

**Outcome.** Test (2,048 sequences, 311,767 hidden bases): GBST-BERT loss 0.9753, accuracy 56.44%; Transformer++ 0.9774, 56.33%. Validation: 0.9689 / 56.80% vs 0.9728 / 56.65%. Random 25%, always-most-common 29.4%. GBST-BERT very slightly ahead (0.1 percentage points) on both splits; one training run per model.

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
