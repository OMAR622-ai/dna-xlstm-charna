# Saved models

Checkpoints of the project's training runs, copied from the server student01 on 5 Oct 2026
(16:52 CEST). Each file is a PyTorch Lightning checkpoint (`.ckpt`, about 25 MB): the model
weights **plus** the optimizer and learning-rate-schedule state, so training can be continued
from it. `model_config.json` next to it describes the model (needed to rebuild it).

| Folder / file | Run | Model | Training step (bases seen) | Validation loss at that step | md5 (first 12) |
|---|---|---|---|---|---|
| **`full_charna_mlm/final_step10000.ckpt`** | [R16](../runs/R16_full_charna.md) full run, **final model** | GBST-BERT, LR 2e-3 | **10,000 (10.5 B)** | step 8,000: 0.9765 (final: see [R18](../runs/R18_eval_full.md)) | `8e91f91a3a86` |
| **`full_llama_mlm/final_step10000.ckpt`** | [R17](../runs/R17_full_llama.md) full run, **final model** | Transformer++-2M, LR 1e-2 | **10,000 (10.5 B)** | 0.9755 (test 0.9686) | `8020840dad0c` |
| `full_charna_mlm/last.ckpt` | R16, copy taken before moving servers | GBST-BERT, LR 2e-3 | 2,500 (2.6 B) | not measured at this step | `8dd49cd8e709` |
| `full_charna_mlm/best_test_loss.ckpt` | R16 | GBST-BERT, LR 2e-3 | 2,000 (2.1 B) | 1.0559 (test 1.0402) | `b2a163855865` |
| `small_charna_500_lr2e-3/last.ckpt` | [R14](../runs/R14_lr_2e-3.md) | GBST-BERT, LR 2e-3 | 500 (0.52 B) | 1.1541 | `869420a6f1ce` |
| `small_charna_500_lr5e-3/last.ckpt` | [R15](../runs/R15_lr_5e-3.md) | GBST-BERT, LR 5e-3 | 500 | 1.1605 | `bf0064ad2778` |
| `small_charna_500_lr1e-3/last.ckpt` | [R13](../runs/R13_lr_1e-3.md) | GBST-BERT, LR 1e-3 | 500 | 1.1869 | `91699090f744` |
| `small_charna_500/last.ckpt` | [R09](../runs/R09_small_charna.md) | GBST-BERT, LR 5e-4 | 500 | 1.1944 | `dd6922b6134a` |
| `small_llama_500/last.ckpt` | [R10](../runs/R10_small_llama.md) | Transformer++-2M, LR 1e-2 | 500 | 1.1960 | `c5287ef042cc` |

All GBST-BERT checkpoints are the final model version (with the pointwise layer, 2,107,392
parameters). Not copied (still on student01 only): the smoke-run and 3-step-trial checkpoints,
which use the superseded first GBST version.

The full run R16 was still training on student01 when these files were copied (step ≈2,900 of
10,000). To continue it on another server from `full_charna_mlm/last.ckpt`, see
[docs/MOVING_TO_NEW_SERVER.md](../../docs/MOVING_TO_NEW_SERVER.md).

**Use a model**, for example to measure its accuracy (from the repository root):

```bash
mkdir -p /tmp/m/checkpoints && cp charna_project/models/small_charna_500_lr2e-3/last.ckpt /tmp/m/checkpoints/ \
  && cp charna_project/models/small_charna_500_lr2e-3/model_config.json /tmp/m/
python scripts_charna/evaluate_mlm.py /tmp/m:GBST-BERT --split test --n_seqs 2048
```

Check a file after downloading: `md5sum charna_project/models/*/*.ckpt` and compare with the table.
