# R12_eval_small: Accuracy of the small-experiment models

| | |
|---|---|
| Status | ✅ done |
| Kind | evaluation |
| Started | 2026-10-04 22:02 |
| Ended | 2026-10-04 22:04 |
| Model | R09 and R10 final checkpoints (step 500) |

**Purpose.** Masked-base accuracy and loss on 2,048 held-out sequences (validation and test splits), same masks for both models.

**Command.**

```bash
python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/small_charna_500:GBST-BERT outputs/pretrain/hg38/small_llama_500:Transformer++ --split {valid,test} --n_seqs 2048
```

**Outcome.** Test accuracy GBST-BERT 43.94% (loss 1.1944), Transformer++ 43.77% (1.1958); random 25%, always-most-common 29.42%. Difference too small to call a winner.

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

- [`results/small/eval_test.csv`](../results/small/eval_test.csv)
- [`results/small/eval_test.md`](../results/small/eval_test.md)
- [`results/small/eval_valid.csv`](../results/small/eval_valid.csv)
- [`results/small/eval_valid.md`](../results/small/eval_valid.md)
- [`results/small/queue.txt`](../results/small/queue.txt)

[← All runs](README.md)
