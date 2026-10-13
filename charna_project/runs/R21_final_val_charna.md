# R21_final_val_charna: Missing step-10,000 validation of the full GBST-BERT run

| | |
|---|---|
| Status | ✅ done |
| Kind | evaluation |
| Started | 2026-10-13 16:29 (student07) |
| Ended | 2026-10-13 ≈16:45 |
| Model | R16 final checkpoint (step 10,000) |

**Purpose.** R16 logged no validation at step 10,000 (after resuming, Lightning counted one step ahead, so the check fell on step 10,001). Measure it with exactly the same settings as during training (validation and test sets, 12.5% of the batches, 4 GPUs) so the curve ends at 10.5 B bases like Transformer++.

**Command.**

```bash
bash scripts_charna/server/final_val_charna.sh   (run_pretrain_charna.sh with train.pretrained_model_path=<final ckpt> train.validate_at_start=true, in a scratch run directory)
```

**Outcome.** Validation loss 0.9728, test loss 0.9670 (Transformer++ at step 10,000: 0.9755 / 0.9686). Added as the last point of the GBST-BERT curve (charna_project/results/full/full_charna_mlm_curve_with_final_val.csv).

## Validation during training (`metrics.csv`)

Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.

| Step | Bases seen | Validation loss | Test loss |
|---|---|---|---|
| 0 | 0.000 B | 0.9728 | 0.9670 |

## Files

- [`results/full/final_val_charna/metrics.csv`](../results/full/final_val_charna/metrics.csv)

[← All runs](README.md)
