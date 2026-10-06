"""Demo: load the final GBST-BERT model, hide bases in a real DNA sequence, and look at the guesses.

Run from the repository root (needs torch; pyfaidx for reading the genome):
    python scripts_charna/demo_predict.py
    python scripts_charna/demo_predict.py --seq ACGT...   # your own sequence (up to 1,024 bases)

What it shows:
  1. how many hidden bases the model guesses correctly,
  2. a 60-base window: the true DNA, what the model saw, and its guesses,
  3. which block sizes (1-4 bases) GBST used, on average and at a few positions.
"""

import argparse
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import _shim  # noqa: E402,F401  (import the model without mamba_ssm / flash_attn)
from models.charna import CharnaConfig, CharnaForMaskedLM  # noqa: E402
MODEL_DIR = REPO / "charna_project/models/full_charna_mlm"

# Token numbers used in training (the repository's tokenizer)
TOKEN = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
BASE = {v: k for k, v in TOKEN.items()}
MASK, PAD = 3, 4
SAMPLE = REPO / "data/hg38/hg38.ml.fa"  # full genome (scripts_charna/download_data.sh)


def load_model():
    config = json.load(open(MODEL_DIR / "model_config.json"))["config"]
    config.pop("_target_")
    model = CharnaForMaskedLM(CharnaConfig(**config))
    ckpt = torch.load(MODEL_DIR / "final_step10000.ckpt", map_location="cpu", weights_only=False)
    weights = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(weights)  # strict: every weight must match
    print(f"Loaded GBST-BERT after {ckpt['global_step']:,} training steps "
          f"({sum(p.numel() for p in model.parameters()):,} parameters)")
    return model.eval()


def example_sequence():
    """1,024 bases from chromosome 21 of hg38 (a region without unknown bases)."""
    from pyfaidx import Fasta
    return str(Fasta(str(SAMPLE))["chr21"][33_000_000:33_001_024]).upper()


def gbst_block_weights(gbst, ids):
    """Repeat the first half of GBST.forward to get the weight of each candidate block."""
    x = gbst.embedding(ids)
    x = gbst.conv(F.pad(x.transpose(1, 2), gbst.conv_pad)).transpose(1, 2)
    if gbst.pointwise is not None:
        x = gbst.pointwise(x)
    prefix_scores = gbst.scorer(F.pad(x.cumsum(dim=1), (0, 0, 1, 0))).squeeze(-1)
    scores = []
    for b, o in gbst.candidates:
        start, end = gbst._block_bounds(b, o, ids.size(1), ids.device)
        scores.append(prefix_scores[:, end] - prefix_scores[:, start])
    probs = torch.stack(scores, dim=-1).softmax(dim=-1)
    if gbst.score_calibration:
        probs = torch.softmax(probs @ probs.transpose(1, 2), dim=-1) @ probs
    return probs[0]  # (length, 10 candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", help="DNA sequence (default: 1,024 bases of chr21)")
    ap.add_argument("--hide", type=float, default=0.15, help="fraction of bases to hide")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq = (args.seq or example_sequence()).upper()[:1024]
    ids = torch.tensor([[TOKEN.get(c, TOKEN["N"]) for c in seq]])
    torch.manual_seed(args.seed)
    hidden = (torch.rand(ids.shape) < args.hide) & (ids != TOKEN["N"])
    inputs = ids.masked_fill(hidden, MASK)  # hidden bases become [MASK]

    model = load_model()
    with torch.no_grad():
        logits = model(inputs).logits[0]                   # (length, 12 token scores)
        base_logits = logits[:, [7, 8, 9, 10]]             # keep only A, C, G, T
        probs = base_logits.softmax(-1)
        guess = torch.tensor([7, 8, 9, 10])[probs.argmax(-1)]

    h = hidden[0]
    correct = (guess[h] == ids[0][h]).float().mean().item()
    loss = F.cross_entropy(logits[h], ids[0][h]).item()
    print(f"\nHidden {int(h.sum())} of {len(seq)} bases: {100 * correct:.1f}% guessed correctly "
          f"(loss {loss:.3f}; random guessing: 25%, loss 1.386)")

    start = 300
    window = range(start, start + 60)
    print(f"\nPositions {start}-{start + 59}:")
    print("true DNA     " + "".join(seq[i] for i in window))
    print("model sees   " + "".join("?" if h[i] else seq[i] for i in window))
    print("model guess  " + "".join(BASE[int(guess[i])] if h[i] else " " for i in window))
    print("             " + "".join(("✓" if guess[i] == ids[0][i] else "✗") if h[i] else " " for i in window))

    with torch.no_grad():
        w = gbst_block_weights(model.backbone.gbst, inputs)
    sizes = torch.tensor([b for b, _ in model.backbone.gbst.candidates])
    per_size = torch.stack([w[:, sizes == b].sum(-1) for b in range(1, 5)], -1)  # (length, 4)
    print("\nGBST: average weight of each block size over the sequence")
    for b in range(4):
        print(f"  blocks of {b + 1} base{'s' if b else ' '}: {100 * per_size[:, b].mean():5.1f}%")
    print("\nGBST at a few positions (weight per block size 1 / 2 / 3 / 4):")
    for i in (start, start + 15, start + 30, start + 45):
        print(f"  position {i} ({seq[max(i - 3, 0):i]}[{seq[i]}]{seq[i + 1:i + 4]}): "
              + " / ".join(f"{100 * p:.0f}%" for p in per_size[i]))


if __name__ == "__main__":
    main()
