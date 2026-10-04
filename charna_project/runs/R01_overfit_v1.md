# R01_overfit_v1: Overfit one batch (first GBST version)

| | |
|---|---|
| Status | ✅ done |
| Kind | check |
| Started | 2026-10-04, before 13:16 |
| Ended | – |
| Model | GBST-BERT, first version (no pointwise layer) |

**Purpose.** Check that the model can learn at all, by memorising one fixed masked batch of hg38.

**Command.**

```bash
python scripts_charna/overfit_one_batch.py --fasta data/hg38/hg38.ml.fa --bed data/hg38/human-sequences.bed --strategy {mlm,ul2}
```

**Outcome.** Passed: loss < 0.05 after 158 steps (MLM) and 229 steps (UL2), on one GPU. A first attempt on the CPU was too slow (about 9 s per step) and was stopped.

**Notes.** Printed to the terminal only; no result files.

[← All runs](README.md)
