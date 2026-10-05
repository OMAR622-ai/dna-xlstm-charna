# charna_project: the whole project in one folder

This folder holds everything the GBST + BERT (ChaRNABERT-style) project added to the DNA-xLSTM
code base: the code, the documentation, the data that fits in git, and the results of the runs.
The full project documentation is the repository's main [README.md](../README.md); a copy is in
[PROJECT_README.md](PROJECT_README.md).

- **Code** (`models/`, `src/`, `configs/`, `scripts_*/`, `tests/`): copies. The code that runs
  lives at the same paths in the repository root (e.g. `models/charna/gbst.py`), because the
  repo's training code (Hydra configs, model registry) expects it there. The copies are rebuilt
  with `bash scripts_charna/export_project_folder.sh`, so they always match the committed code.
- **Data** (`datasets/`): kept only here.
- **Results** (`results/`): kept only here, copied from the server's `outputs/` with
  `bash scripts_charna/collect_results.sh`.
- **Models** (`models/`): checkpoints of the trained models (≈25 MB each, resumable); see
  [models/README.md](models/README.md).
- **Environment** (`environment/`): the exact Python packages used on the training server.
- **Runs** (`runs/`): one page per run (purpose, exact command, settings read from the run's
  config, results, files) and a log of every project step; see [runs/README.md](runs/README.md).
  Generated from `runs/runs.yaml` and `results/` by `python scripts_charna/make_run_cards.py`.

## Data

| Path | What it is |
|---|---|
| `datasets/hg38/human-sequences.bed` | The full list of pretraining regions (38,171 rows: chromosome, start, end, train/valid/test split), as used in all runs. The dataset reads 2²⁰ = 1,048,576 bases from each start and cuts them into 1,024 windows of 1,024 bases |
| `datasets/hg38/hg38.ml.fa.fai` | Index of the full genome FASTA |
| `datasets/hg38_sample/hg38_sample.fa`, `human-sequences_sample.bed` | **Sample** for trying the code: 8 real hg38 regions of up to 1,048,576 bases (6 train, 1 valid, 1 test; 7.9 MB), made with `scripts_charna/make_hg38_sample.py` (seed 0). The dataset reads it exactly like the full data |

**Not in git:** the full genome `hg38.ml.fa` is 3.1 GB (about 900 MB compressed); GitHub accepts
files up to 100 MB. `bash scripts_charna/download_data.sh` downloads it from the DNA-xLSTM
authors' server, checks its md5 sum (`c2cc5a203b71fff7fbf750f9276b3e8a`), and also caches the
Nucleotide Transformer benchmark tasks (public, Hugging Face `InstaDeepAI/nucleotide_transformer_downstream_tasks`).
Model checkpoints are not in git either (the repo ignores `*.ckpt`).

Use the sample, for example, with the tests:

```bash
HG38_FASTA=charna_project/datasets/hg38_sample/hg38_sample.fa \
HG38_BED=charna_project/datasets/hg38_sample/human-sequences_sample.bed \
python -m unittest tests.test_charna -v
```

## Results

| Path | What it is |
|---|---|
| `results/smoke/summary.txt`, `results/pretrain/hg38/smoke_*/metrics.csv` | Smoke run (48 steps, ≈50M tokens) of both models |
| `results/small/`, `results/pretrain/hg38/small_*_500/`, `results/downstream/nt_small/` | Small experiment (500 steps, ≈0.5B tokens, then 4 NT tasks); see the main README, "Small experiment" |
| `results/pretrain/hg38/stopped_*`, `results/full/stopped*/` | Configurations of the two full runs that were stopped (no validation point yet when stopped) |
| `results/nt_smoke_cpu/` | The CPU test of the NT benchmark pipeline (run R05; numbers carry no information) |
| `results/*/*/config.json`, `train.log` | Full resolved settings and main log of every training run |

Each `metrics.csv` has one row per validation: `global_step`, `tokens` seen, and the validation and
test losses (pretraining) or metrics (NT fine-tuning).

## Code

| Path | What it is |
|---|---|
| `PROJECT_README.md` | Copy of the project documentation (main README): goal, status, model, protocol with paper sources, how to run, results, known issues |
| `models/charna/gbst.py` | GBST tokenizer |
| `models/charna/modeling_charna.py` | Bidirectional encoder, MLM head, reverse-complement helpers |
| `models/charna/configuration_charna.py`, `models/charna/__init__.py` | Model config (2M defaults) |
| `configs/model/charna.yaml` | Hydra model config |
| `src/dataloaders/utils/ul2.py` | UL2 masking (ChaRNABERT §4.1) |
| `src/dataloaders/datasets/hg38_dataset.py`, `src/dataloaders/genomics.py` | Changed upstream files: `mlm_strategy` switch (mlm / ul2) |
| `src/dataloaders/datasets/nucleotide_transformer_dataset.py` | Changed upstream file: loads each NT task's own data file |
| `src/models/sequence/dna_embedding.py` | Changed upstream file: downstream wrappers for GBST + BERT and Transformer++ |
| `src/utils/registry.py` | Changed upstream file: new registry names |
| `src/callbacks/metrics_csv.py` | Writes validation/test metrics to CSV |
| `scripts_pretrain/run_pretrain_charna.sh` | Pretraining on hg38 |
| `scripts_downstream/run_nucleotide_charna.sh` | Nucleotide Transformer benchmark fine-tuning |
| `scripts_charna/train_charna.py` | Launcher for GPUs without mamba_ssm / flash_attn |
| `scripts_charna/summarize_nt.py` | NT results table |
| `scripts_charna/download_data.sh`, `make_hg38_sample.py` | Full data download and check; builds the sample |
| `scripts_charna/collect_results.sh` | Copies run results into `results/` |
| `scripts_charna/make_run_cards.py` | Writes the run pages in `runs/` |
| `scripts_charna/overfit_one_batch.py`, `scripts_charna/smoke_run.sh` | Sanity checks |
| `scripts_charna/export_project_folder.sh` | Rebuilds the code copies in this folder |
| `tests/` | Unit tests (`python -m unittest tests.test_charna tests.test_downstream -v` from the repo root) |
| `changes_to_existing_files.patch` | Exact line-by-line changes to the files that already existed in DNA-xLSTM |

For the changed upstream files, the copies here are the full new versions; the patch shows only
what was changed. The original DNA-xLSTM README is kept as `README_DNA_XLSTM.md` in the
repository root.
