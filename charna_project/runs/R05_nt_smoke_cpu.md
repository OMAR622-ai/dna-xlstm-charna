# R05_nt_smoke_cpu: NT benchmark pipeline test on the CPU

| | |
|---|---|
| Status | ✅ done |
| Kind | fine-tuning test |
| Started | 2026-10-04 ≈14:20 |
| Ended | 2026-10-04 14:37 |
| Model | R03 and R04 checkpoints |

**Purpose.** Check that fine-tuning on a Nucleotide Transformer task runs end to end (loading weights, training, best-checkpoint test, reverse-complement averaging, results table).

**Command.**

```bash
PRETRAIN_DIR=... MODEL={charna,llama} TASKS=promoter_tata LRS=1e-3 BATCH_SIZES=64 SEEDS="1 2" EPOCHS=2 bash scripts_downstream/run_nucleotide_charna.sh trainer.accelerator=cpu trainer.limit_train_batches=15 trainer.limit_val_batches=5
```

**Outcome.** Pipeline works for both models (exit 0). Test F1 0.679 for all runs = always predicting the positive class after 15 batches: the numbers carry no information.

**Notes.** Found and fixed two bugs on the way: GBST checkpoints could not be loaded for fine-tuning; the NT data loader mixed all 18 tasks (fix applied in a test copy first, committed 15:24).

## Files

- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/metrics.csv`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/metrics.csv)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/model_config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/model_config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/train.log`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-1/train.log)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/metrics.csv`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/metrics.csv)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/model_config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/model_config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/train.log`](../results/nt_smoke_cpu/promoter_tata/smoke_charna_ph/LR-1e-3_BATCH_SIZE-64/seed-2/train.log)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/metrics.csv`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/metrics.csv)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/model_config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/model_config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/train.log`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-1/train.log)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/metrics.csv`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/metrics.csv)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/model_config.json`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/model_config.json)
- [`results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/train.log`](../results/nt_smoke_cpu/promoter_tata/smoke_llama_ph/LR-1e-3_BATCH_SIZE-64/seed-2/train.log)

[← All runs](README.md)
