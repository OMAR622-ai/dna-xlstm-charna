# Moving the project to a new server

Everything needed to continue the project on another machine is on GitHub (branch
`charna-gbst` of `OMAR622-ai/dna-xlstm-charna`), except the large public data, which is
downloaded by a script. Saved on 5 October 2026.

## 1. What is saved where

| What | Where in the repository |
|---|---|
| All code (model, training, evaluation, benchmark) | repository root (`models/charna/`, `src/`, `scripts_*/`, `configs/`), also copied in `charna_project/` |
| **Trained models** (checkpoints, resumable) | `charna_project/models/` — see its [README](../charna_project/models/README.md) |
| Results of every run (metrics, settings, logs, accuracy tables) | `charna_project/results/` |
| One page per run, project log | `charna_project/runs/` |
| Exact Python packages used | `charna_project/environment/requirements-student01.txt` |
| Queue scripts used on student01 | `scripts_charna/server/` |
| Interval list, genome index, small hg38 sample | `charna_project/datasets/` |
| Documentation, report | `README.md`, `docs/EXPLAINED.md`, `docs/report/REPORT.pdf` |

**Not on GitHub** (too large or not needed): the full genome `hg38.ml.fa` (3.1 GB) and the
Nucleotide Transformer data — both downloaded by `scripts_charna/download_data.sh`; the
checkpoints of the smoke runs and the 3-step trial (superseded model version, still on
student01); progress-bar logs.

**State of the training when this was saved:** the full GBST-BERT run (R16, LR 2e-3) was
running on student01 at step ≈2,900 of 10,000; its latest checkpoint at that time (step 2,500)
is `charna_project/models/full_charna_mlm/last.ckpt`. Its first validation point: step 2,000,
validation loss 1.0559, test loss 1.0402. Transformer++ (R17) and the final evaluation (R18)
had not started. If student01 stays reachable, later checkpoints are in
`~/charna/dna-xlstm/outputs/pretrain/hg38/full_charna_mlm/checkpoints/` there.

## 2. Set up the new server

```bash
# 1. Get the code (the repository is private: log in first with `gh auth login`,
#    or use an SSH key added to the GitHub account OMAR622-ai)
git clone -b charna-gbst https://github.com/OMAR622-ai/dna-xlstm-charna.git
cd dna-xlstm-charna

# 2. Python environment (Python 3.9 was used; conda or venv)
python3.9 -m venv ~/envs/charna && source ~/envs/charna/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r charna_project/environment/requirements-student01.txt

# 3. Data (3.1 GB genome with md5 check + NT benchmark tasks)
bash scripts_charna/download_data.sh

# 4. Check that everything works (expect: Ran 23 tests ... OK)
python -m unittest tests.test_charna tests.test_downstream -v
```

Notes:
- torch 2.2.0 + CUDA 12.1 was chosen because the GTX 1080 Ti needs it. On newer GPUs (e.g.
  A100, RTX 30xx/40xx) this version works too; keep it so the runs stay comparable.
- `scripts_charna/train_charna.py` lets the code run without `mamba_ssm` / `flash_attn`
  (needed for older GPUs). It is harmless on newer GPUs; the scripts always use it.
- The model files (`.ckpt`, ≈25 MB each) are normal git files; `git clone` downloads them.

## 3. Continue the full GBST-BERT training from step 2,500

The checkpoint holds the weights, the optimizer state and the learning-rate schedule, so
training continues where it stopped (tested on 5 Oct 2026 with the step-500 checkpoint of R09:
it restored all states and continued at step 501).

```bash
cd scripts_pretrain
MODEL=charna LR=2e-3 RUN_NAME=full_charna_mlm VAL_EVERY=2000 \
  NUM_DEVICES=4 MICRO_BATCH=32 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash run_pretrain_charna.sh \
  train.ckpt=$(realpath ../charna_project/models/full_charna_mlm/last.ckpt)
```

Keep the **global batch** at 1,024 sequences: `NUM_DEVICES × MICRO_BATCH` must divide 1,024
(the script adds gradient accumulation to fill the rest), e.g. 8 GPUs × 32, 2 GPUs × 64, or
1 GPU × 128 if memory allows. Keep `PRECISION=32` (the default) so the run stays comparable with
the first 2,500 steps, even if the new GPUs support bf16.

The validation results of the new part go to `outputs/pretrain/hg38/full_charna_mlm/metrics.csv`
on the new server; the step-2,000 point from student01 is in
`charna_project/results/pretrain/hg38/full_charna_mlm/metrics.csv` — combine the two files for
the full curve.

The training runs for hours or days; start it detached so it survives logging out, e.g.
`setsid -f bash -c '<the command above>' > full_charna.log 2>&1`.

## 4. The remaining steps after that

```bash
# Transformer++ full run (comparison model, LR 1e-2 is its script default)
cd scripts_pretrain
MODEL=llama RUN_NAME=full_llama_mlm VAL_EVERY=2000 bash run_pretrain_charna.sh

# Final accuracy test of both models (from the repository root)
cd ..
for SPLIT in valid test; do
  python scripts_charna/evaluate_mlm.py \
    outputs/pretrain/hg38/full_charna_mlm:GBST-BERT outputs/pretrain/hg38/full_llama_mlm:Transformer++ \
    --split $SPLIT --n_seqs 2048 --markdown eval_full_$SPLIT.md
done
```

`scripts_charna/server/run_full_after_lr.sh` is the queue script that did all of this
automatically on student01 (it uses `~/charna/dna-xlstm` as the repository path; adjust paths).

## 5. Keep GitHub up to date

```bash
bash scripts_charna/collect_results.sh outputs charna_project/results
python scripts_charna/make_run_cards.py          # needs pyyaml
bash scripts_charna/export_project_folder.sh
git add -A charna_project && git commit -m "Update results" && git push
```

Before running `make_run_cards.py`, update the run's status and outcome in
`charna_project/runs/runs.yaml`.
