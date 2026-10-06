# R18_eval_full: Accuracy of the full models

| | |
|---|---|
| Status | 🕒 queued |
| Kind | evaluation |
| Started | – |
| Ended | – |
| Model | R16 and R17 final checkpoints |

**Purpose.** Same accuracy test as R12, after full training.

**Command.**

```bash
python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ --split {valid,test} --n_seqs 2048
```

## Files

- [`results/full/eval_charna_only/eval_valid.csv`](../results/full/eval_charna_only/eval_valid.csv)
- [`results/full/eval_charna_only/eval_valid.md`](../results/full/eval_charna_only/eval_valid.md)
- [`results/full/queue.txt`](../results/full/queue.txt)
- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
