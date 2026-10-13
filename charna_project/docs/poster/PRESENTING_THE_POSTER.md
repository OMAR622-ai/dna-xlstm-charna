# How to present the poster

A guide to the A0 poster (`docs/poster/POSTER.pdf`): for each of the 9 blocks, what it means in
simple words, what to say (about 5–7 minutes in total), and questions you may be asked.

**Your one-sentence summary** (say it first, repeat it at the end):
> "We tested whether a model that learns by itself how to read DNA predicts DNA better than a
> model that reads one base at a time. Both learned DNA well, but the learned reader gave almost
> no advantage, because it ended up reading single bases anyway."

---

## Title
**"Does a learned tokenizer help a small DNA model?"**
- *Tokenizer* = the way a model cuts a sequence into units ("words") before reading it.
- *Learned* = the model decides the units itself, instead of a human fixing them.

**Say:** "My project asks one question: does it help if a DNA model learns by itself how to cut
the sequence into units?"

---

## Block 1 — The question
**In simple words:** Most DNA models decide beforehand how to read DNA: one base at a time, or in
groups (codons, k-mers). The ChaRNABERT paper (2024) built a model for RNA that learns the
best grouping at every position — a method called **GBST** — and it worked a bit better on RNA.
We wanted to know if this also works for **human DNA** with a **small** model.

**Say:** "ChaRNABERT showed this idea helps on RNA. I applied the same model to the human genome
and compared it with a normal model of the same size."

---

## Block 2 — The model: GBST-BERT
**In simple words:** Four steps, shown as boxes:
1. **DNA** goes in (1,024 bases).
2. **GBST** looks at every position and considers blocks of 1, 2, 3 or 4 bases around it (10
   possibilities, like reading frames). It gives each a score and mixes them.
3. The **BERT encoder** lets every base "look at" every other base, in both directions, to
   collect context.
4. The model **guesses the hidden bases**.

The small table shows the 10 blocks for the first T in `ACGTTGCA`.

**Say:** "GBST is like trying different reading frames at once and learning which one is useful.
The encoder then lets each base use information from the whole sequence. The model has
2.1 million parameters — the same size as the DNA-xLSTM models I compare with."

**Possible question:** *Why 2 million parameters?* — "To compare fairly with the published
DNA-xLSTM models of that size, and because it fits our GPUs."

---

## Block 3 — Data and training
**In simple words:**
- The data is the **human reference genome hg38** (3 billion bases), split into **training
  (77%)**, **validation (10%)** and **test (9%)**. The model never trains on test DNA.
- **The game:** hide 15% of the bases, the model guesses them (the example shows `?` and the
  answer). No labels needed — the DNA itself is the teacher. Half of the time the other DNA
  strand (reverse complement) is used.
- **Settings table:** 10,000 training steps = 10.5 billion bases read; each step reads 1,024
  pieces of 1,024 bases. Each setting comes from a paper (Bio-xLSTM or ChaRNABERT), except the
  learning rate of GBST-BERT, which I chose with a short test.

**Say:** "The model learns by filling in hidden bases — like a fill-in-the-blank exercise. It
read about 10 billion bases, so it saw the training part of the genome about four to five times."

**Possible question:** *What is the learning rate?* — "How big each learning step is. I tested
four values in short runs and chose 2e-3, the best on validation data."

---

## Block 4 — How we test
**In simple words:**
- **Loss** = how wrong the guesses are. Lower is better. Random guessing gives 1.386.
- **Accuracy** = how often the best guess is the correct base, on DNA the model has never seen.
  Both models get exactly the same hidden positions, so it is fair.
- **References:** random guessing = 25%; always guessing the most common base = 29.4%. A good
  model must clearly beat both.

**Say:** "I report two numbers: the loss, which the model is trained on, and the accuracy, which
is easier to understand. And I compare with what you would get without any model."

---

## Block 5 — Engineering notes
**In simple words:**
- 24 automatic tests check the code; every run is documented.
- **A bug was found and fixed** in the original repository: when several processes read the
  genome file at the same time, some got wrong DNA. I measured it (0.35% wrong in our main
  setup), fixed it, and proved the fix with a test (841 of 1,024 wrong before, 0 after).
  Training restarted from a clean checkpoint. Test DNA never got into training.
- Training moved from one server to another in the middle and continued from a saved checkpoint.

**Say:** "Along the way I found a data-loading bug in the original code, measured how big it
was, fixed it and tested the fix. The results I show use correct data."

**Possible question:** *Did the bug change the results?* — "Very little: 0.35% of sequences in
the early runs, always from the same data split, and the accuracy test never used the buggy path."

---

## Block 6 — Results (the big figure and table)
**In simple words:**
- **Left graph — learning-rate check (500 short steps):** four learning rates for GBST-BERT and
  Transformer++. The lowest line is the best: **2e-3**.
- **Right graph — full training (10,000 steps):** loss goes down from about 1.05 to about 0.97
  for both models. The blue line (GBST-BERT) is below the orange line (Transformer++) from
  4 billion bases on; at the end the gap is small (0.973 vs 0.975).
- **Table:** one row per run. The key numbers are the bold ones: **final test accuracy 56.44%
  (GBST-BERT) vs 56.33% (Transformer++)**, test loss 0.9753 vs 0.9774.

**Say:** "Both models learned DNA well: they guess 56% of hidden bases correctly — more than twice
random guessing. GBST-BERT was ahead during most of training, but at the end the difference is
only 0.1 percentage points. With one run per model, that is too small to call a winner."

**Possible questions:**
- *Is 56% good?* — "Yes for a small model: random is 25%, and our loss of 0.975 is in the range of
  the published models of the same size (about 0.96–1.03)."
- *Was the last blue point measured differently?* — "After restarting the training, the step
  counter was one ahead, so the last check was skipped during training. I ran exactly the same
  check on the final model afterwards: 0.973."

---

## Block 7 — What GBST learned
**In simple words:** I looked inside the trained model. GBST can choose blocks of 1, 2, 3 or 4
bases — but after training it puts **98% of its weight on single bases**. So the "learned
tokenizer" learned to read one base at a time, just like the normal model.

**Say:** "This is the most interesting finding: the learned reader decided that single bases are
best for human DNA. That explains why both models end so close."

**Possible question:** *Why would it choose single bases?* — "We can only guess: the convolution
before GBST already mixes each base with its neighbours, and the attention collects context, so
bigger blocks add little. Human DNA may also have less 'word-like' structure than RNA."

---

## Block 8 — Conclusions
**In simple words:**
1. Both small models learned human DNA well (56% accuracy, loss about 0.975).
2. GBST gave only a tiny advantage at the end (0.1 points).
3. GBST learned to read mostly single bases — the reason for (2).
4. Even on RNA, ChaRNABERT found only small gains.

**Caveats (say them yourself — it shows you understand the limits):** each model was trained
once; only GBST-BERT's learning rate was tuned; the two models also differ slightly in other
parts (normalization).

**Say:** "So my answer is: on human DNA, with a small model, learned tokenization gives no clear
benefit — and the model itself chose to read single bases."

---

## Block 9 — Next steps
1. **Clean test of GBST:** the exact same model with GBST replaced by a simple lookup table —
   this isolates the effect of GBST.
2. **Biology tasks:** test both models on 18 real tasks (promoters, enhancers, splice sites,
   histone marks) — the code is ready.
3. Repeat the runs with other random seeds to see how much results vary.

The QR code opens the GitHub repository (private: log in as OMAR622-ai).

**Say:** "Next, I want a cleaner comparison and to test whether these models help on real
biological questions."

---

## A 5-minute order
| Time | Block | Main sentence |
|---|---|---|
| 0:00 | Title + 1 | "Does learning how to read DNA help a small model?" |
| 0:45 | 2 | "GBST tries blocks of 1–4 bases; BERT collects context." |
| 1:30 | 3 + 4 | "Fill-in-the-blank on 10 billion bases; measured on unseen DNA." |
| 2:30 | 6 | "Both reach 56% — the difference is only 0.1 points." |
| 3:30 | 7 | "GBST learned to read single bases (98%)." |
| 4:15 | 8 + 9 | "No clear benefit; next: clean test and biology tasks." |
| (if asked) | 5 | "Found and fixed a data bug; results use correct data." |
