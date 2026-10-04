# R07_overfit_v2: Overfit one batch (paper-exact GBST)

| | |
|---|---|
| Status | ✅ done |
| Kind | check |
| Started | 2026-10-04 ≈15:05 |
| Ended | – |
| Model | GBST-BERT, final version (with pointwise layer) |

**Purpose.** Repeat the learning check for the changed model.

**Command.**

```bash
python scripts_charna/overfit_one_batch.py --fasta data/hg38/hg38.ml.fa --bed data/hg38/human-sequences.bed --strategy {mlm,ul2}
```

**Outcome.** Passed: loss < 0.05 after 164 steps (MLM) and 230 steps (UL2), on one GPU.

**Notes.** Printed to the terminal only; no result files.

[← All runs](README.md)
