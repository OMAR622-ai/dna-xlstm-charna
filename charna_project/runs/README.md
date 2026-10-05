# All runs and steps of the project

One page per run (training, checks, evaluations). Hand-written facts come from [`runs.yaml`](runs.yaml); settings and results are read from [`../results/`](../results/) by `scripts_charna/make_run_cards.py`. Times are CEST.

## Runs

| Run | What | Status | Started | Outcome |
|---|---|---|---|---|
| [R01_overfit_v1](R01_overfit_v1.md) | Overfit one batch (first GBST version) | ✅ done | 2026-10-04, before 13:16 | Passed: loss < 0.05 after 158 steps (MLM) and 229 steps (UL2), on one GPU. |
| [R02_dryrun_gbst](R02_dryrun_gbst.md) | 3-step trial on 4 GPUs | ✅ done | 2026-10-04 13:11 | Passed: trained, validated (val loss 1.68, test loss 1.69), saved a checkpoint, exited 0. |
| [R03_smoke_charna](R03_smoke_charna.md) | Smoke run, GBST-BERT (48 steps) | ✅ done | 2026-10-04 13:26 | Done in 930 s, peak GPU memory 3,388 MiB. |
| [R04_smoke_llama](R04_smoke_llama.md) | Smoke run, Transformer++ (48 steps) | ✅ done | 2026-10-04 13:42 | Done in 894 s, peak GPU memory 1,200 MiB (uses gradient checkpointing). |
| [R05_nt_smoke_cpu](R05_nt_smoke_cpu.md) | NT benchmark pipeline test on the CPU | ✅ done | 2026-10-04 ≈14:20 | Pipeline works for both models (exit 0). |
| [R06_full_v1_stopped](R06_full_v1_stopped.md) | Full pretraining attempt 1 (stopped) | ⏹ stopped | 2026-10-04 13:59 | Stopped after 1 h 10 min (≈ step 320 of 10,000, training loss ≈1.2) because the model was changed to match the ChaRNABERT paper exactly (pointwise layer) and the learning rate was changed to 5e-4. |
| [R07_overfit_v2](R07_overfit_v2.md) | Overfit one batch (paper-exact GBST) | ✅ done | 2026-10-04 ≈15:05 | Passed: loss < 0.05 after 164 steps (MLM) and 230 steps (UL2), on one GPU. |
| [R08_full_v2_stopped](R08_full_v2_stopped.md) | Full pretraining attempt 2 (stopped) | ⏹ stopped | 2026-10-04 15:10 | Stopped after 1 h 8 min (≈ step 200, training loss ≈1.36) to run the small experiment first and see results sooner (decided together). |
| [R09_small_charna](R09_small_charna.md) | Small experiment, GBST-BERT (500 steps) | ✅ done | 2026-10-04 16:20 | Validation loss 1.340 → 1.194 (test 1.196). |
| [R10_small_llama](R10_small_llama.md) | Small experiment, Transformer++ (500 steps) | ✅ done | 2026-10-04 19:55 | Validation loss 1.263 → 1.196 (test 1.196). |
| [R11_eval_smoke](R11_eval_smoke.md) | Accuracy script test on the smoke models | ✅ done | 2026-10-04 ≈17:45 | Script works. |
| [R12_eval_small](R12_eval_small.md) | Accuracy of the small-experiment models | ✅ done | 2026-10-04 22:02 | Test accuracy GBST-BERT 43.94% (loss 1.1944), Transformer++ 43.77% (1.1958); random 25%, always-most-common 29.42%. |
| [R13_lr_1e-3](R13_lr_1e-3.md) | Learning-rate check, GBST-BERT LR 1e-3 | ✅ done | 2026-10-04 22:10 | Validation loss at step 500: 1.1869 (5e-4 run: 1.1944). |
| [R14_lr_2e-3](R14_lr_2e-3.md) | Learning-rate check, GBST-BERT LR 2e-3 | ✅ done | 2026-10-05 00:23 | Best of the four: validation loss at step 500 1.1541 (test 1.1380). |
| [R15_lr_5e-3](R15_lr_5e-3.md) | Learning-rate check, GBST-BERT LR 5e-3 | ✅ done | 2026-10-05 02:41 | Validation loss at step 500: 1.1605, second best. |
| [R16_full_charna](R16_full_charna.md) | Full pretraining, GBST-BERT (10.5 B bases) | ⏳ running | 2026-10-05 05:00 | – |
| [R17_full_llama](R17_full_llama.md) | Full pretraining, Transformer++ (10.5 B bases) | 🕒 queued | – | – |
| [R18_eval_full](R18_eval_full.md) | Accuracy of the full models | 🕒 queued | – | – |

## Project log (steps that are not runs)

| When | What |
|---|---|
| 2026-10-03/04 | Read the code base; set up SSH key login to the JKU server student01 (4× GTX 1080 Ti, shared). |
| 2026-10-03 | Copied hg38 (3.1 GB) to the server and verified it (md5 c2cc5a203b71fff7fbf750f9276b3e8a); the direct download from ml.jku.at stalled. |
| 2026-10-03/04 | Built the Python environment (torch 2.2.0 + CUDA 12.1; newer torch has no kernels for the 1080 Ti). |
| 2026-10-03/04 | Read the four papers (Bio-xLSTM, ChaRNABERT, Charformer, UL2) and collected every hyperparameter with its source. |
| 2026-10-04 12:45 | Model, UL2 masking and 16 tests committed (GBST + BERT, 2.04 M parameters). |
| 2026-10-04 13:16–13:33 | Launcher for the older GPUs, pretraining script, metrics file, smoke-run script. |
| 2026-10-04 13:59 | Full pretraining started (R06), after the smoke runs; code pushed to a private GitHub repository. |
| 2026-10-04 14:40 | Benchmark wrappers and scripts; two bugs fixed (weight loading, results file). Project README written. |
| 2026-10-04 15:07 | Project scope set to GBST + bidirectional BERT; UL2 removed. |
| 2026-10-04 15:13 | Decision: follow ChaRNABERT exactly — pointwise layer added (Table 9), UL2 restored as an option (§4.1). |
| 2026-10-04 15:17 | Decision: learning rate 5e-4 (ChaRNABERT Table 2); R06 stopped, R08 started. |
| 2026-10-04 15:24 | Approved fix of the NT data loader committed (it mixed all 18 tasks). |
| 2026-10-04 16:18 | Decision: small experiment first; R08 stopped, R09 started. |
| 2026-10-04 16:37–17:28 | Data (interval list, hg38 sample, download script), project README as front page, beginner's explanation, loss plot script. |
| 2026-10-04 17:52 | Decision: NT benchmark postponed; accuracy evaluation script added; queue changed to pretraining + evaluation only. |
| 2026-10-04 18:28 | Progress report (PDF) added. |
| 2026-10-04 22:17 | Small-experiment results added to README and report. |
| 2026-10-04 22:30 | Learning-rate check (R13–R15) running; full runs (R16–R18) queued to start automatically after it. |
| 2026-10-04 22:40 | Every run documented: one page per run (charna_project/runs/), settings and logs of all runs collected. |
| 2026-10-05 05:00 | Learning-rate check done: 2e-3 best (val 1.1541 at step 500, vs 1.1605 / 1.1869 / 1.1944 for 5e-3 / 1e-3 / 5e-4). Full GBST-BERT run (R16) started automatically with 2e-3. |

## Code changes (git commits on this branch)

| When | Commit | Message |
|---|---|---|
| 2026-10-04 12:45 | `9e2402b` | Add ChaRNABERT-style GBST + BERT model, UL2 masking and tests |
| 2026-10-04 13:16 | `8142c67` | Add pretraining launcher, run script and overfit check for GBST + BERT |
| 2026-10-04 13:32 | `fdd56c2` | Fix train_charna launcher for DDP and optional-import fallbacks |
| 2026-10-04 13:33 | `418fdaa` | Add metrics CSV callback, Transformer++ switch and smoke-run driver |
| 2026-10-04 14:40 | `6eff3ea` | MetricsCSV: keep columns aligned when later rows add metrics |
| 2026-10-04 14:40 | `21444fa` | Downstream wrappers for GBST + BERT and Transformer++ with load tests |
| 2026-10-04 14:40 | `288854a` | NT benchmark script and results summary for GBST + BERT and Transformer++ |
| 2026-10-04 14:40 | `d1fa3df` | Document the GBST + BERT project in README_CHARNA.md |
| 2026-10-04 14:51 | `b80fe13` | Add script that collects all project files into charna_project/ |
| 2026-10-04 14:51 | `e472436` | Collect all project code and documentation in charna_project/ |
| 2026-10-04 15:07 | `8d0cbec` | Focus on GBST + BERT with MLM: remove the UL2 objective |
| 2026-10-04 15:07 | `8327afd` | Rebuild charna_project/ without UL2 |
| 2026-10-04 15:13 | `0b8f8c8` | Match ChaRNABERT exactly: depthwise-separable GBST conv, UL2 restored as option |
| 2026-10-04 15:13 | `fccbabc` | Rebuild charna_project/ |
| 2026-10-04 15:17 | `790e54d` | Pretrain GBST + BERT with ChaRNABERT's learning rate 5e-4 |
| 2026-10-04 15:17 | `4e6720b` | Rebuild charna_project/ |
| 2026-10-04 15:24 | `b449e6a` | Fix NT benchmark loader: load each task's own parquet file |
| 2026-10-04 15:24 | `6c85bf9` | Rebuild charna_project/ |
| 2026-10-04 16:37 | `77a0f21` | Add data scripts and the small-experiment section to the docs |
| 2026-10-04 16:37 | `b70166a` | charna_project: add data (bed, index, hg38 sample) and run results |
| 2026-10-04 16:56 | `6557167` | Make the project documentation the main README |
| 2026-10-04 16:57 | `8f852fd` | export: fix link rewriting in the folder copy of the README |
| 2026-10-04 16:57 | `51c55a5` | charna_project: add PROJECT_README.md, update index |
| 2026-10-04 17:28 | `b3fed34` | Add beginner's explanation, loss plot script; sample holds full 2^20-base regions |
| 2026-10-04 17:28 | `cfc4a4c` | charna_project: full-length hg38 sample, smoke loss plot, docs |
| 2026-10-04 17:52 | `699a548` | Add masked-base accuracy evaluation; small experiment without NT for now |
| 2026-10-04 17:52 | `d10a0bb` | Rebuild charna_project/ |
| 2026-10-04 18:28 | `62ddafb` | Add progress report (PDF): how ChaRNABERT works, how it is applied to DNA, what was done, results so far |
| 2026-10-04 18:28 | `76866a8` | Rebuild charna_project/ |
| 2026-10-04 22:17 | `c6b09b1` | Small experiment results: loss curves and masked-base accuracy of GBST-BERT and Transformer++ |
| 2026-10-04 22:17 | `81736f9` | charna_project: small experiment results |
| 2026-10-04 22:43 | `9389abb` | Record every run: one page per run plus project log |
| 2026-10-04 22:43 | `b114945` | charna_project: run pages (R01-R18), configs and logs of every run |
| 2026-10-04 22:43 | `84bd6b5` | export: tidy header comment |
| 2026-10-04 22:43 | `3de13f1` | Rebuild charna_project/ |
| 2026-10-04 22:50 | `dba5c1c` | Run status 4 Oct 22:43: R13 (LR 1e-3) step 100 |
| 2026-10-05 05:08 | `f0235da` | Learning-rate check done: 2e-3 best for GBST-BERT; full run R16 started with it |
| 2026-10-05 05:08 | `0742db0` | Rebuild charna_project/ |
