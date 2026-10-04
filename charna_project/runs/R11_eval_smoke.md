# R11_eval_smoke: Accuracy script test on the smoke models

| | |
|---|---|
| Status | ✅ done |
| Kind | evaluation |
| Started | 2026-10-04 ≈17:45 |
| Ended | – |
| Model | R03 and R04 checkpoints |

**Purpose.** Test scripts_charna/evaluate_mlm.py (64 test sequences, CPU).

**Command.**

```bash
python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/smoke_charna:GBST-BERT-smoke outputs/pretrain/hg38/smoke_llama:Transformer++-smoke --n_seqs 64 --batch 16 --device cpu
```

**Outcome.** Script works. Smoke GBST-BERT 29.1% accuracy = always answering A (the most common hidden base, 29.1%); smoke Transformer++ 32.8%. Showed that accuracy must be compared with the majority-base reference.

**Notes.** Printed to the terminal only; numbers are in the README and the report.

[← All runs](README.md)
