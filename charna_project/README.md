# charna_project: all project files in one place

This folder collects every file the GBST + BERT (ChaRNABERT-style) project added to or changed in
the DNA-xLSTM code base, so the whole project can be browsed or downloaded from one folder.
The full project documentation is [README_CHARNA.md](README_CHARNA.md).

**These are copies.** The code that actually runs lives at the same paths in the repository root
(e.g. `models/charna/gbst.py`), because the repo's training code (Hydra configs, model registry)
expects it there. The folder is regenerated from the repository with
`bash scripts_charna/export_project_folder.sh`, so it always matches the committed code.

## Contents

| Path | What it is |
|---|---|
| `README_CHARNA.md` | Project documentation: goal, status, model, protocol with paper sources, how to run, smoke results, known issues |
| `models/charna/gbst.py` | GBST tokenizer |
| `models/charna/modeling_charna.py` | Bidirectional encoder, MLM head, reverse-complement helpers |
| `models/charna/configuration_charna.py`, `models/charna/__init__.py` | Model config (2M defaults) |
| `configs/model/charna.yaml` | Hydra model config |
| `src/dataloaders/utils/ul2.py` | UL2 mixture-of-denoisers masking |
| `src/callbacks/metrics_csv.py` | Writes validation/test metrics to CSV |
| `src/dataloaders/datasets/hg38_dataset.py`, `src/dataloaders/genomics.py` | Changed upstream files: `mlm_strategy` switch (mlm / ul2) |
| `src/models/sequence/dna_embedding.py` | Changed upstream file: downstream wrappers for GBST + BERT and Transformer++ |
| `src/utils/registry.py` | Changed upstream file: new registry names |
| `scripts_pretrain/run_pretrain_charna.sh` | Pretraining on hg38 |
| `scripts_downstream/run_nucleotide_charna.sh` | Nucleotide Transformer benchmark fine-tuning |
| `scripts_charna/train_charna.py` | Launcher for GPUs without mamba_ssm / flash_attn |
| `scripts_charna/summarize_nt.py` | NT results table |
| `scripts_charna/overfit_one_batch.py`, `scripts_charna/smoke_run.sh` | Sanity checks |
| `scripts_charna/export_project_folder.sh` | Rebuilds this folder |
| `tests/` | Unit tests (`python -m unittest tests.test_charna tests.test_downstream -v` from the repo root) |
| `changes_to_existing_files.patch` | Exact line-by-line changes to the files that already existed in DNA-xLSTM |

For the four changed upstream files, the copies here are the full new versions; the patch shows
only what was changed.
