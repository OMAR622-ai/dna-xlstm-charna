# R16_full_charna: Full pretraining, GBST-BERT (10.5 B bases)

| | |
|---|---|
| Status | 🕒 queued |
| Kind | pretraining |
| Started | – |
| Ended | – |
| Model | GBST-BERT, final version, best LR of R09/R13–R15 (by validation loss at step 500) |

**Purpose.** The main run.

**Command.**

```bash
MODEL=charna LR=<best> RUN_NAME=full_charna_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh   (from ~/charna/run_full_after_lr.sh)
```

## Files

- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
