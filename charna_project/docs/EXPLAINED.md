# The GBST + BERT project, explained step by step

Written for a reader who comes from molecular biology and is new to coding and AI.
State of the project: 4 October 2026, evening. Technical words are defined the first time
they appear and collected in the [glossary](#glossary) at the end.

---

## 1. The big picture

We are teaching a small neural network to "read" human DNA by playing a fill-in-the-blank game:
we hide about 15% of the bases in a 1,024-base piece of the human genome (hg38) and the
network must guess them. The special part is **how the network reads the sequence**: instead
of looking at one base at a time, it first uses **GBST**, a component that learns by itself
whether it should look at single bases, pairs, triplets (like codons) or groups of four, at
every position. After this practice (called *pretraining*), we test the network on 18
real biology questions (the *Nucleotide Transformer benchmark*: is this a promoter? an
enhancer? a splice site? a histone-marked region?) and compare it with the DNA-xLSTM models
from the Bio-xLSTM paper. The question we want to answer: does learned, flexible "reading" of
DNA help a small model?

---

## 2. Files

"New" files were written for this project; "modified" files existed in the DNA-xLSTM code and
were changed. Everything is also copied into one folder, `charna_project/`.

| File | New / modified | What it does |
|---|---|---|
| `models/charna/gbst.py` | new | GBST: turns the bases into numbers and learns which block size (1–4 bases) to use at each position |
| `models/charna/modeling_charna.py` | new | The rest of the model: the BERT-style encoder and the final layer that guesses the hidden bases |
| `models/charna/configuration_charna.py` | new | The model's settings (size, number of layers, GBST options) with their default values |
| `models/charna/__init__.py` | new | Makes the folder importable as a Python package |
| `configs/model/charna.yaml` | new | The same model settings in the format the training program reads |
| `src/dataloaders/utils/ul2.py` | new | An optional second way of hiding bases (UL2: hides longer stretches) |
| `src/dataloaders/datasets/hg38_dataset.py` | modified | Reads the genome; now can use UL2 hiding instead of standard hiding |
| `src/dataloaders/genomics.py` | modified | Passes the "which hiding method" setting through to the genome reader |
| `src/dataloaders/datasets/nucleotide_transformer_dataset.py` | modified | Reads the 18 benchmark tasks; fixed a bug that mixed all 18 tasks together |
| `src/models/sequence/dna_embedding.py` | modified | Adapters that plug our pretrained model (and the comparison model) into the benchmark code |
| `src/utils/registry.py` | modified | The list of names the training program knows; our new parts were added |
| `src/callbacks/metrics_csv.py` | new | Writes the loss values into a simple table file (`metrics.csv`) |
| `scripts_pretrain/run_pretrain_charna.sh` | new | Starts pretraining with all the settings from the papers |
| `scripts_downstream/run_nucleotide_charna.sh` | new | Starts the benchmark fine-tuning |
| `scripts_charna/train_charna.py` | new | A starter that lets the training program run on our older GPUs |
| `scripts_charna/summarize_nt.py` | new | Turns the benchmark runs into a results table |
| `scripts_charna/plot_metrics.py` | new | Draws the loss curve as a picture |
| `scripts_charna/overfit_one_batch.py` | new | Sanity check: can the model memorize one small batch? |
| `scripts_charna/smoke_run.sh` | new | A short trial run of both models |
| `scripts_charna/download_data.sh` | new | Downloads the full genome and checks it is the right file |
| `scripts_charna/make_hg38_sample.py` | new | Makes the small genome sample that is stored on GitHub |
| `scripts_charna/collect_results.sh` | new | Copies result files from the server into `charna_project/results/` |
| `scripts_charna/export_project_folder.sh` | new | Rebuilds the `charna_project/` folder |
| `tests/test_charna.py`, `tests/test_downstream.py`, `tests/_shim.py`, `tests/__init__.py` | new | Automatic checks (23 tests) |
| `README.md` | modified | Now the project documentation; the original is kept as `README_DNA_XLSTM.md` |
| `docs/EXPLAINED.md` | new | This document |

---

## 3. The data flow: one DNA sequence from hg38 to a loss value

The real sizes are 1,024 bases per sequence and 256 numbers per base. To keep the examples
readable we follow **12 bases**. "Shape" means the size of a table of numbers, written as
(rows, columns, ...). A **tensor** is just such a table of numbers, possibly with more than two
dimensions, that the computer can do math on quickly.

**Step 1 — Find a region.** The file `human-sequences.bed` lists 38,171 start positions in the
genome, each labelled train, valid or test. From each start, the reader takes about 1 million
bases (2²⁰) and cuts them into 1,024 windows of 1,024 bases. So "sequence number 5,000" means
"region 4, window 904". (Code: `src/dataloaders/datasets/hg38_dataset.py`.)

**Step 2 — Read the bases** from the genome file `hg38.ml.fa` with a library (pyfaidx) that can
jump directly to any position.
Example: `"ACGTTGCAAGTC"` — a string of 12 letters.

**Step 3 — Maybe flip the strand.** With probability 50%, the sequence is replaced by its
reverse complement (`ACGTTGCAAGTC` → `GACTTGCAACGT`). Both strands of DNA carry the same
information, so this teaches the model that both readings are valid. This is called *RC
augmentation*.

**Step 4 — Turn letters into numbers (tokenizing).** Each letter gets a fixed number:
A=7, C=8, G=9, T=10, N=11; special symbols have small numbers ([SEP]=1 marks the end, [MASK]=3
means "hidden", [PAD]=4 means "nothing here"). Unknown bases (N) are replaced by [PAD] so the
model is never asked to guess them.
`ACGTTGCAAGTC` → `[7, 8, 9, 10, 10, 9, 8, 7, 7, 9, 10, 8]` → shape (12,).

**Step 5 — Hide some bases (masking).** For each position, a coin with 15% chance decides if it
is hidden. Say positions 2 and 9 are chosen (the two G's). Two lists are made:

- the **input** the model sees: 80% of chosen positions become [MASK]=3, 10% become a random
  token, 10% stay unchanged (a trick from the BERT paper so the model never fully trusts
  what it sees). Here: `[7, 8, 3, 10, 10, 9, 8, 7, 7, 3, 10, 8]`
- the **target** (the answer key): the true base at chosen positions, [PAD]=4 everywhere else.
  Here: `[4, 4, 9, 4, 4, 4, 4, 4, 4, 9, 4, 4]`

(Code: `src/dataloaders/utils/mlm.py`, from the original repo.)

**Step 6 — Make a batch.** A **dataloader** (the part of the program that reads examples and
stacks them) puts 32 sequences together. Shape (32, 12) in our example, (32, 1024) in reality.
Each of the 4 GPUs gets its own batch of 32.

**Step 7 — GBST: from numbers to "reading" vectors.** (Details in section 4.)
- Each token number is looked up in a table of learned vectors (an **embedding**: a learned list
  of numbers that describes a token, like a profile). Shape (32, 12, 256): every base is now
  described by 256 numbers.
- A short smoothing filter (a *convolution*) mixes each base's vector with its neighbours.
- GBST builds 10 candidate "readings" per position (blocks of 1 to 4 bases), scores them,
  and blends them. Output shape: still (32, 12, 256), one vector per base.

**Step 8 — The BERT encoder: bases look at each other.** Four identical *encoder blocks*. In each:
- **Attention**: every position asks every other position "how relevant are you to me?" and
  collects information from the relevant ones. It looks in both directions (left and right),
  which is what "bidirectional" means. 8 "heads" do this in parallel, each with its own idea
  of relevance. Position is encoded with *RoPE* (the vectors are rotated by an angle that
  depends on the position, so the model knows how far apart two bases are).
- **Feed-forward (SwiGLU)**: each position's vector is processed on its own by a small network
  (256 → 320 → 256 numbers).
- *Normalization* (LayerNorm) before each part keeps the numbers in a stable range.
Shape stays (32, 12, 256) throughout.

**Step 9 — Upsampling.** Only needed if GBST shortens the sequence (an optional setting called
downsampling, d_s). The paper does **not** shorten (d_s = 1), and neither do we, so this step does
nothing in our runs.

**Step 10 — Prediction.** A final layer turns each 256-number vector into 12 scores, one per
token type (A, C, G, T, and the special tokens). These scores are called **logits**. Shape
(32, 12, 12): for each sequence, each position, a score for each of the 12 possible tokens.

**Step 11 — Loss.** The scores are turned into probabilities (softmax), and at each **hidden**
position we take the probability the model gave to the true base. The loss is the average of
−log(probability) over the hidden positions (*cross-entropy*). Positions whose target is [PAD]
are ignored. If the model is sure and right, the loss is near 0; if it guesses uniformly among
the 4 bases, the loss is −log(1/4) = **1.386**. This is the number to beat: everything below
1.386 means the model has learned something about DNA.
Example: if the model gives the true G at position 2 a probability of 0.40 and at position 9
a probability of 0.30, the loss is (−ln 0.40 − ln 0.30) / 2 = (0.916 + 1.204) / 2 = **1.06**.

---

## 4. GBST: how it "chooses" block sizes

Biology analogy: a ribosome reads mRNA in triplets, but a transcription factor may recognize a
6-base motif, and a methylation enzyme looks at CG pairs. There is no single "right" way to cut
DNA into words. GBST does not fix one; it learns a soft mixture, separately at every position.

Take `ACGTTGCA` (positions 0–7) and look at **position 3, the first T**. With a maximum block
size of 4, GBST lists every block of 1–4 bases that **contains position 3**:

| Block size | Candidate blocks containing position 3 |
|---|---|
| 1 | `T` |
| 2 | `TT` (positions 3–4), `GT` (2–3) |
| 3 | `TTG` (3–5), `GTT` (2–4), `CGT` (1–3) |
| 4 | `TTGC` (3–6), `GTTG` (2–5), `CGTT` (1–4), `ACGT` (0–3) |

That is 1 + 2 + 3 + 4 = **10 candidates**. The different starting points of blocks of the same
size are called *offsets* — like the three reading frames of a gene.

What GBST does with them:

1. **Represent each block**: add up the (smoothed) vectors of the bases in it ("sum pooling").
   So the candidate `GT` is vector(G) + vector(T).
2. **Score each block**: a learned scorer turns each candidate's vector into one number,
   "how useful is this reading here?".
3. **Turn scores into weights** with softmax (they become positive and add up to 1).
   Made-up numbers for illustration: `T` 0.05, `TT` 0.10, `GT` 0.05, `TTG` 0.30, `GTT` 0.20, `CGT`
   0.05, the four 4-blocks 0.25 together. Here GBST would mostly "read" position 3 as part of a
   triplet starting at it.
4. **Calibrate**: each position's weights are adjusted towards those of positions with similar
   weight patterns elsewhere in the sequence, so the "reading style" is consistent along the
   sequence (P̃ = softmax(P·Pᵀ)·P, ChaRNABERT Eq. 6).
5. **Blend**: the output for position 3 is the weighted sum of the 10 candidate vectors.

Important: the choice is **soft** — every candidate contributes a little — so the whole thing
can be trained by gradient descent. Nobody tells GBST which block sizes are right; it learns the
weights from the fill-in-the-blank game. Every one of the 1,024 positions gets its own weights.

---

## 5. Training: one training step

1. Each of the 4 GPUs takes a batch of 32 masked sequences (steps 1–6 above).
2. **Forward pass**: the model computes the loss on its batch (steps 7–11).
3. **Backward pass**: the computer works out, for each of the 2.1 million adjustable numbers
   (*parameters* or *weights*), in which direction a small change would lower the loss (the
   *gradient*).
4. Steps 1–3 are repeated 8 times (*gradient accumulation*) and the gradients are added up. In
   total: 4 GPUs × 32 sequences × 8 = **1,024 sequences = 1,048,576 bases** per step. This is how
   we reach the paper's batch size on small GPUs.
5. The gradients are clipped (limited in size) so one strange batch cannot push the model too
   far.
6. The optimizer (**AdamW**) changes every parameter a little in the helpful direction. How big
   the change is depends on the **learning rate**.
7. The learning rate is updated by the schedule: it rises slowly over the first 10% of the steps
   (*warmup*), then decreases along a cosine curve.

Every 2,000 steps (full run; every 100 in the small experiment) the model is evaluated on
validation and test sequences it never trains on, and the losses are written to `metrics.csv`.
A **checkpoint** (a saved copy of all parameters, so the model can be reloaded later) is saved
regularly.

| Setting | Value | Where it is set | Source |
|---|---|---|---|
| Learning rate | 5e-4 (GBST + BERT), 1e-2 (Transformer++) | `scripts_pretrain/run_pretrain_charna.sh` (`LR`) | ChaRNABERT Table 2; Bio-xLSTM Table A3 |
| Warmup and decay | 10% of steps warmup, cosine down to 1e-4 | `configs/experiment/hg38/hg38.yaml` (`scheduler`) | Bio-xLSTM Table A4 |
| Batch size | 1,024 sequences of 1,024 bases per step | `run_pretrain_charna.sh` (`BATCH_SIZE`, `MICRO_BATCH`=32, `NUM_DEVICES`=4) | Bio-xLSTM App. C.5 |
| Masking ratio | 15% (80% [MASK], 10% random, 10% unchanged) | `run_pretrain_charna.sh` (`dataset.mlm_probability`); split in `src/dataloaders/utils/mlm.py` | Bio-xLSTM App. C.5 (BERT) |
| Token budget | 10,000 steps × 1,048,576 = 10.5 billion bases (small experiment: 500 steps = 0.5 billion) | `run_pretrain_charna.sh` (`MAX_STEPS`) | Bio-xLSTM Table A4 |
| Weight decay, optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 | `hg38.yaml`, `run_pretrain_charna.sh` | Bio-xLSTM Table A4 |
| Model size | 256 numbers per base, 4 blocks, 8 heads, 2.1 M parameters | `configs/model/charna.yaml` | Bio-xLSTM Table A3 |
| GBST | blocks up to 4, offsets, sum pooling, calibration, no downsampling | `configs/model/charna.yaml` | ChaRNABERT §3.2 |

**Hydra** is the tool the repo uses to read settings: it combines the `.yaml` files (a **config**
is such a file of settings) with overrides written on the command line, like
`optimizer.lr=5e-4`. That is why the scripts are mostly long lists of `name=value`.

---

## 6. Testing and results so far

**Automatic tests (23, all pass).** Run with one command (section 8). They check, among others:
- GBST gives exactly the same output as a slow, obviously-correct loop version of the paper's
  equations, with every option on and off.
- GBST has the number of weights the paper's Table 9 states.
- The model has about 2 million parameters, close to the comparison model.
- The model really looks in both directions (changing a base on the right changes the output
  on the left).
- The loss is computed only on hidden positions.
- About 15% of bases are hidden, 80% of those become [MASK].
- UL2 hiding never lets the true base leak into the input.
- Reverse-complement handling is consistent.
- The genome reader returns sequences of the right length, and N is never a target.
- Pretrained weights load correctly into the benchmark adapters (this test caught a real bug).
- The results-table script picks the right settings.

**Overfit check (passed).** Can the model memorize one fixed batch? Yes: the loss went below
0.05 after 164 steps (standard hiding) and 230 steps (UL2). This shows the model *can* learn.

**Smoke run (done, but with an older version).** 48 steps (50 million bases) for both models:

| Bases seen | GBST + BERT validation loss | Transformer++ validation loss |
|---|---|---|
| 13 M | 1.392 | 1.397 |
| 25 M | 1.370 | 1.363 |
| 38 M | 1.366 | 1.343 |
| 50 M | 1.364 | 1.325 |

Both are below 1.386, so both learned something. The comparison model was ahead after 50 M bases
and GBST seemed to flatten. But: this was a tiny run, GBST used the older version (before the
change in section 7) and a learning rate of 1e-2 that we later replaced with 5e-4. It only
showed the pipeline works.

**Small experiment (running now).** 500 steps (0.5 billion bases), then 4 benchmark tasks.
First result: after 100 steps (0.1 billion bases) the GBST + BERT validation loss is **1.340**
(test 1.344). The remaining points, the comparison model and the benchmark results are expected
by early Monday morning (5 October).

**Benchmark pipeline test (works, numbers meaningless).** A 2-epoch fine-tuning on one task on
the computer's main processor ran end to end for both models. The score (F1 0.679) is what you
get by always answering "promoter"; it only proves the plumbing works.

**Bugs found and fixed along the way:**
- Loading pretrained GBST weights into the benchmark adapter crashed (fixed, now tested).
- The benchmark data loader silently mixed all 18 tasks (461,850 examples instead of e.g.
  5,509). This is a bug in the original repository caused by a change on Hugging Face (fixed).
- The results file wrote values under the wrong column names in one case (fixed).
- The GitHub genome sample first held only 1/8 of each region (fixed).

**Not done yet:** the full 10.5-billion-base training (stopped twice: once to match the paper
exactly, once to run the small experiment first), the full 18-task benchmark, and the
comparisons asked for in the plan (*ablations*: with/without GBST, MLM vs UL2, with/without
strand flipping, downsampling).

**Things I am unsure about:**
- The maximum block size (4) is not stated in ChaRNABERT.
- Two of the paper's equations look misprinted (block offsets, Eq. 3; SwiGLU, Eq. 9); I followed
  the text (details in section 7).
- The extra "pointwise" layer in GBST is inferred from the paper's parameter table (Table 9),
  not from its text.
- The learning rate 5e-4 was best for ChaRNABERT's 5 M-parameter model on RNA with its own
  batch size; it may not be best for our 2 M model on DNA with 1,024-sequence batches.
- The small experiment validates on only 2% of the validation data, so its curve is noisier.
- The random-replacement part of masking can insert any of the 12 tokens, not only A/C/G/T. This
  is how the original repository works, for all models, so we kept it.

---

## 7. Differences from the papers

| Paper says | We do | Why |
|---|---|---|
| FlashAttention 2 (fast attention code) | PyTorch's standard attention | FlashAttention 2 needs newer GPUs than our GTX 1080 Ti; the result is the same, just slower |
| bf16 numbers (half-size numbers) | fp32 (normal-size numbers) | Our GPUs do not support bf16 |
| Eq. 3 writes block offsets as X[i+o : i+o+b] | blocks that contain position i | Read literally, position i would get blocks that do not include it; the text (reading frames) means our version |
| Eq. 9 writes SwiGLU as σ(xW₁)⊙swish(xW₂) | standard SwiGLU, swish(xW₁)⊙xW₂ | The text cites the standard SwiGLU paper; the printed formula is a different function |
| Eq. 1 says "depthwise convolution" only | depthwise convolution + a 256×256 "pointwise" layer | The paper's own parameter table (Table 9) counts d × M + d² weights, which is exactly that |
| Max block size not stated | 4 | Charformer uses 4 in its figure |
| RNA data, 2,000–8,000-base context, model sizes 8 M–650 M | human DNA, 1,024 bases, 2 M | So the results compare with DNA-xLSTM (Bio-xLSTM protocol, as asked in the project plan) |
| BEACON RNA benchmark | Nucleotide Transformer DNA benchmark | Same reason |
| Bio-xLSTM baseline learning rate in the repo script: 8e-3, hidden size 1,024 | 1e-2, hidden size 320 | We follow the paper's Table A3, which differs from the repo script |

---

## 8. How to run it yourself

On the JKU server, everything is installed. Start with:

```bash
ssh -p 5792 omohamed@student01.ai-lab.jku.at
source ~/envs/charna/bin/activate        # switch on the Python environment we installed
cd ~/charna/dna-xlstm                    # the project folder
```

**(a) Tests** (under a minute):

```bash
python -m unittest tests.test_charna tests.test_downstream -v
```

Expect a list of test names each followed by `ok`, ending in:

```
Ran 23 tests in ...s
OK
```

On a computer without the training libraries you will see `OK (skipped=6)`: the 6 benchmark
tests need them. To use the small genome sample from GitHub instead of the full genome, put
`HG38_FASTA=charna_project/datasets/hg38_sample/hg38_sample.fa HG38_BED=charna_project/datasets/hg38_sample/human-sequences_sample.bed`
in front of the command.

**(b) Smoke run** (both models, 48 steps each, ~30 min, all 4 GPUs):

```bash
cd scripts_pretrain
bash ../scripts_charna/smoke_run.sh
cat ../outputs/smoke/summary.txt
```

Expect two lines like `smoke_charna exit=0 wall_s=930 peak_mem_mib=3388`, then `SMOKE_DONE`.
`exit=0` means success. The losses are in `../outputs/pretrain/hg38/smoke_charna/metrics.csv`.

**(c) Full training** (~40 hours per model, so start it detached so it survives logging out):

```bash
cd ~/charna/dna-xlstm/scripts_pretrain
MODEL=charna RUN_NAME=full_charna_mlm VAL_EVERY=2000 setsid -f bash run_pretrain_charna.sh > ../outputs/full_charna.log 2>&1
tail -c 300 ../outputs/full_charna.log      # progress bar: Epoch 0: ... loss=1.3...
```

Every 2,000 steps a new line appears in `../outputs/pretrain/hg38/full_charna_mlm/metrics.csv`
(step, bases seen, test loss, validation loss). Use `MODEL=llama` for the comparison model.

**(d) Benchmark** (after training; one GPU per command):

```bash
cd ~/charna/dna-xlstm/scripts_downstream
PRETRAIN_DIR=../outputs/pretrain/hg38/full_charna_mlm MODEL=charna bash run_nucleotide_charna.sh
cd .. && python scripts_charna/summarize_nt.py outputs/downstream/nt_cv10_ep20
```

The first command prints a line per run (`NT: ..., TASK: H3, LR: ..., SEED: 1` then `exit=0`).
The full grid is very long (18 tasks × 10 grid points × 10 seeds); use for example
`TASKS="H3" LRS="8e-4" BATCH_SIZES="128" SEEDS="1 2 3"` in front of the command for a part. The
second command prints a table: one row per task, `mean ± std (range ...)`.

**(e) The loss plot** (on your own computer, after downloading the results from GitHub;
matplotlib is not installed on the server):

```bash
python scripts_charna/plot_metrics.py \
  charna_project/results/pretrain/hg38/smoke_charna/metrics.csv:GBST-BERT \
  charna_project/results/pretrain/hg38/smoke_llama/metrics.csv:Transformer++ \
  --out val_loss.png
```

Expect the numbers printed (`GBST-BERT: 0.013B -> 1.3921, ...`), then `saved val_loss.png`: a
picture with validation loss (up) against billions of bases seen (right), one line per model.

---

## 9. Where to look if something breaks

| Symptom | Likely cause | Where to look |
|---|---|---|
| `no kernel image is available` or CUDA errors at start | Wrong PyTorch version for the GTX 1080 Ti (needs torch 2.2.0 + cu121; newer versions dropped these GPUs) | `pip list | grep torch` in `~/envs/charna` |
| `ModuleNotFoundError: mamba_ssm` or `flash_attn` | `train.py` was started directly | Always start through `scripts_charna/train_charna.py` (the scripts do this) |
| `.bed file must exist` / FASTA not found | The genome is not in `data/hg38/` | `ls data/hg38/`; `bash scripts_charna/download_data.sh` |
| `CUDA out of memory` or very slow training | GPUs shared with other users, or batch too large | `nvidia-smi`; lower `MICRO_BATCH` in `run_pretrain_charna.sh` |
| Benchmark crash `Missing key in pretrained model` | Model settings of the checkpoint and of the fine-tuning do not match (e.g. `gbst_pointwise_conv`) | `model_config.json` in the pretraining run folder; `src/models/sequence/dna_embedding.py` |

Logs: pretraining prints into the `.log` file you chose and into
`outputs/pretrain/hg38/<run>/train.log`; each benchmark run writes `stdout.log` in its folder.

---

## 10. Three key pieces of code, line by line

### 10.1 GBST (`models/charna/gbst.py`, `forward`)

```python
def forward(self, input_ids):
    # input_ids: token numbers, shape (batch, length), e.g. (32, 1024)

    x = self.embedding(input_ids)
    # look up the learned vector of every token -> shape (32, 1024, 256)

    length = x.size(1)
    # number of positions (1024)

    x = self.conv(F.pad(x.transpose(1, 2), self.conv_pad)).transpose(1, 2)
    # smoothing filter over 4 neighbouring positions, separately for each of the 256 numbers
    # (depthwise convolution). The padding keeps the length at 1024. Shape unchanged.

    if self.pointwise is not None:
        x = self.pointwise(x)
    # mix the 256 numbers of each position with each other (the "pointwise" layer, Table 9)

    prefix = F.pad(x.cumsum(dim=1), (0, 0, 1, 0))
    # running totals along the sequence: prefix[j] = sum of vectors of positions 0..j-1.
    # Trick: the sum of any block [start, end) is prefix[end] - prefix[start],
    # so we never have to add up each block separately. Shape (32, 1025, 256)

    prefix_scores = self.scorer(prefix).squeeze(-1)
    # the scorer is a simple linear function, so the score of a block sum equals
    # score(prefix[end]) - score(prefix[start]); scores of running totals, shape (32, 1025)

    bounds, scores = [], []
    for b, o in self.candidates:
        # loop over the 10 (block size, offset) candidates: (1,0), (2,0), (2,1), (3,0), ...

        start, end = self._block_bounds(b, o, length, x.device)
        # for every position i: the block of size b that contains i, starting at i - o
        # (clipped at the sequence ends)

        scale = 1.0 / b if self.pooling == "mean" else 1.0
        # sum pooling (paper) -> scale 1; mean pooling (Charformer option) -> divide by b

        bounds.append((start, end, scale))
        # remember the block edges for the blending step

        scores.append((prefix_scores[:, end] - prefix_scores[:, start]) * scale)
        # the score of this candidate at every position, shape (32, 1024)

    probs = torch.stack(scores, dim=-1).softmax(dim=-1)
    # put the 10 scores side by side -> (32, 1024, 10), turn into weights that add up to 1

    if self.score_calibration:
        probs = torch.softmax(probs @ probs.transpose(1, 2), dim=-1) @ probs
    # calibration (Eq. 6): compare every position's weight pattern with every other
    # position's (probs @ probs^T is a 1024 x 1024 similarity table), and pull each
    # position's weights towards those of similar positions. Shape stays (32, 1024, 10)

    latents = 0
    for k, (start, end, scale) in enumerate(bounds):
        block = (prefix[:, end] - prefix[:, start]) * scale
        # the vector of candidate k at every position (block sum), shape (32, 1024, 256)
        latents = latents + probs[..., k:k + 1] * block
        # add it, weighted by its probability -> the blended reading of each position

    char_latents = latents
    # keep the per-base result (used only when downsampling is switched on)

    if self.downsample_rate > 1:
        latents = downsample(latents, self.downsample_rate)
    # optional shortening (off in our runs: downsample_rate = 1)

    return latents, char_latents
    # shape (32, 1024, 256): one blended vector per base, ready for the encoder
```

### 10.2 Hiding bases (`src/dataloaders/utils/mlm.py`, `mlm_getitem`, from the original repo)

```python
def mlm_getitem(seq, mlm_probability=0.15, contains_eos=False, tokenizer=None, eligible_replacements=None):
    data = seq[:-1].clone() if contains_eos else seq.clone()
    # copy the token numbers; drop the last token if it is the end marker [SEP]
    # (our data has it, so 1025 tokens become 1024)

    target = data.clone()
    # the answer key starts as a copy of the true sequence

    probability_matrix = torch.full(target.shape, mlm_probability)
    # a list with 0.15 at every position

    masked_indices = torch.bernoulli(probability_matrix).bool()
    # flip a 15% coin at every position: True = this base will be hidden

    target[~masked_indices] = tokenizer.pad_token_id
    # in the answer key, every position that is NOT hidden becomes [PAD],
    # which the loss ignores: the model is graded only on hidden bases

    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    # of the hidden positions, pick 80% ...
    data[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # ... and write [MASK] there in the input

    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    # of the remaining 20%, pick half (= 10% of the hidden ones)

    if eligible_replacements is not None:
        rand_choice = torch.randint(eligible_replacements.shape[0], size=target.shape)
        random_words = eligible_replacements[rand_choice]
    else:
        random_words = torch.randint(len(tokenizer), size=target.shape, dtype=torch.long)
    # random tokens; in our setup any of the 12 token types (eligible_replacements is None)

    data[indices_random] = random_words[indices_random]
    # put the random tokens into the input at those positions

    return data, target
    # the last 10% of hidden positions keep their true base in the input;
    # data = what the model sees, target = what it must predict
```

### 10.3 The whole model (`models/charna/modeling_charna.py`)

```python
class CharnaModel(nn.Module):
    def forward(self, input_ids, return_char_latents=False):
        if self.config.tokenizer == "gbst":
            h, char_latents = self.gbst(input_ids)
        else:
            h = char_latents = self.embedding(input_ids)
        # step 7: GBST (our model) or a plain lookup table (the "no GBST" comparison)

        h = self.dropout(h)
        # dropout randomly zeroes numbers during training; set to 0 in our runs (paper setting)

        seqlen = h.size(1)
        freqs_cos, freqs_sin = self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]
        # the RoPE rotation angles for each position, computed once in advance

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # step 8: pass through the 4 encoder blocks
        # (a memory-saving variant, "gradient checkpointing", can be switched on; it is off)

        h = self.norm(h)
        # final normalization, shape (32, 1024, 256)
        return (h, char_latents) if return_char_latents else h


class EncoderBlock(nn.Module):
    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        # normalize, let every position gather information from all others (attention),
        # and ADD the result to the input (a "residual connection": the block only learns
        # a correction, which makes deep networks easier to train)
        return x + self.feed_forward(self.ffn_norm(x))
        # normalize, process each position on its own (SwiGLU), add again


class CharnaForMaskedLM(nn.Module):
    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        h, char_latents = self.backbone(input_ids, return_char_latents=True)
        # run GBST + encoder (CharnaModel above)

        rate = self.config.downsample_rate
        if rate > 1:
            h = h.repeat_interleave(rate, dim=1)[:, :input_ids.size(1)]
            h = self.upsample_norm(h + char_latents)
        # step 9: only if GBST shortened the sequence (off in our runs)

        return LMOutput(logits=self.lm_head(h))
        # step 10: 12 scores per position, shape (32, 1024, 12).
        # The loss (step 11) is computed outside, by the repo's training code:
        # cross-entropy with ignore_index=4, i.e. [PAD] positions do not count.
```

---

## Glossary

| Term | Meaning |
|---|---|
| **AdamW** | The optimizer: the rule that changes the parameters after each step, with a gentle pull towards small values (weight decay) |
| **Attention** | The part of the encoder where every position weighs and gathers information from all other positions |
| **Batch** | A group of sequences processed together (32 per GPU here) |
| **BERT** | A type of model (Devlin et al., 2019) that reads in both directions and is trained by guessing hidden tokens |
| **Bidirectional** | Uses context on both sides of a position (left and right), unlike text generators that only look left |
| **Checkpoint** | A saved copy of all model parameters (and optimizer state) at some point in training |
| **Config** | A file of settings (`.yaml` here) that the program reads instead of having values written in the code |
| **Convolution** | A small sliding filter that combines each position with its neighbours |
| **Cross-entropy** | The loss used here: the average of −log(probability given to the correct answer) |
| **Dataloader** | The part of the program that reads examples, prepares them and stacks them into batches |
| **Downsampling (d_s)** | Shortening the sequence by averaging neighbours; Charformer does it, ChaRNABERT and we do not |
| **Embedding** | A learned list of numbers that represents a token, like a profile describing it |
| **Encoder block** | One repeated unit of the model: attention + feed-forward, each with normalization and a residual connection |
| **Epoch** | One pass through all training examples (our pretraining uses steps instead) |
| **Fine-tuning** | Further training of a pretrained model on a specific task (here: the benchmark tasks) |
| **fp32 / bf16** | Number formats: 32-bit (normal precision) and 16-bit "brain float" (faster, needs newer GPUs) |
| **GBST** | Gradient-Based Subword (Subsequence) Tokenization: a learned, soft choice of block size at every position |
| **GPU** | Graphics card used for the many parallel calculations of training; the server has four GTX 1080 Ti |
| **Gradient** | For each parameter, the direction and size of change that would lower the loss |
| **Gradient accumulation** | Adding up gradients of several batches before one update, to reach a large batch on small GPUs |
| **Hydra** | The tool that reads the `.yaml` configs and command-line overrides |
| **hg38** | The human reference genome, version 38 |
| **Learning rate** | How big each parameter change is; too big makes training unstable, too small makes it slow |
| **LayerNorm** | Rescales each position's numbers to a standard range, keeping training stable |
| **Logits** | The raw scores for each possible token before they are turned into probabilities |
| **Loss** | The number measuring how wrong the model is; training lowers it. 1.386 = random guessing among 4 bases |
| **MLM** | Masked language modelling: the fill-in-the-blank training game |
| **NT benchmark** | Nucleotide Transformer benchmark: 18 DNA classification tasks (promoters, enhancers, splice sites, histone marks) |
| **Offset** | Where a block starts relative to a position, like a reading frame |
| **Overfitting a batch** | Training on one fixed batch until memorized, to check the model can learn at all |
| **Parameters (weights)** | The adjustable numbers inside the model (2.1 million here) |
| **PH (post-hoc conjoining)** | At test time, run the model on both strands and average the two answers |
| **Pretraining** | The first, general training (fill-in-the-blank on the genome) before any specific task |
| **RC augmentation** | Randomly using the reverse-complement strand during training |
| **Residual connection** | Adding a block's output to its input, so the block learns a correction |
| **RoPE** | Rotary position encoding: tells the model where each base is by rotating its vectors by a position-dependent angle |
| **Smoke run** | A short trial run to check everything works before a long run |
| **Softmax** | Turns a list of scores into positive weights that add up to 1 |
| **SwiGLU** | The type of small per-position network inside each encoder block |
| **Tensor** | A table of numbers with any number of dimensions, e.g. (32, 1024, 256) |
| **Token / tokenizer** | A token is one unit the model reads (here one base or special symbol); the tokenizer turns letters into token numbers |
| **Transformer++** | The comparison model from the repo: the same encoder without GBST (plain lookup table) |
| **UL2** | An alternative hiding method that hides spans of different lengths, including long stretches |
| **Validation / test set** | Sequences never used for training, used to measure how well the model generalizes |
| **Warmup** | Starting training with a small learning rate that rises over the first steps |
