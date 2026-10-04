# R17_full_llama: Full pretraining, Transformer++ (10.5 B bases)

| | |
|---|---|
| Status | 🕒 queued |
| Kind | pretraining |
| Started | – |
| Ended | – |
| Model | Transformer++-2M, LR 1e-2 |

**Purpose.** The comparison model at full scale.

**Command.**

```bash
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 bash scripts_pretrain/run_pretrain_charna.sh
```

## Files

- [`results/full/stopped/queue.txt`](../results/full/stopped/queue.txt)
- [`results/full/stopped2/queue.txt`](../results/full/stopped2/queue.txt)

[← All runs](README.md)
