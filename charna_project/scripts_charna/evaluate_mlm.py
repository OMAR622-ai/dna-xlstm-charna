"""Evaluate pretrained MLM checkpoints: masked-base accuracy and loss on held-out hg38 sequences.

For each run directory (model_config.json + a checkpoint), the same fixed set of sequences from
the chosen split is masked in exactly the same way (15%: 80% [MASK], 10% random, 10% unchanged,
as in training; fixed random seed, no strand flipping), and the model predicts the hidden bases.
Reported over the hidden positions:
  loss      cross-entropy in nats (the training loss; 1.386 = uniform guess among 4 bases)
  acc       the highest-scoring token (out of all 12) is the true base
  acc_ACGT  the highest-scoring of A, C, G, T is the true base
  acc_A..T  acc_ACGT per true base
plus two reference values that need no model: 25% (random guess) and the frequency of the most
common base among the hidden positions (always guessing that base).

  python scripts_charna/evaluate_mlm.py outputs/pretrain/hg38/small_charna_500:GBST-BERT \
      outputs/pretrain/hg38/small_llama_500:Transformer++ --split test --n_seqs 2048 \
      --csv results.csv --markdown results.md
"""

import argparse
import csv
import importlib
import json
import math
import pathlib
import sys

import torch
import torch.nn.functional as F

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts_charna"))
import train_charna  # noqa: E402  (placeholders for mamba_ssm / flash_attn, as in training)

train_charna.install_placeholders()
train_charna.ensure_cuda_home()
sys.path.insert(0, str(REPO))

from models.caduceus.tokenization_caduceus import CaduceusTokenizer  # noqa: E402
from src.dataloaders.datasets.hg38_dataset import HG38Dataset  # noqa: E402

BASES = "ACGT"


def build_model(run_dir, ckpt_name):
    """Model from model_config.json, weights from the Lightning checkpoint (prefix 'model.')."""
    config = dict(json.load(open(run_dir / "model_config.json"))["config"])
    module, _, cls = config.pop("_target_").rpartition(".")
    config_cls = getattr(importlib.import_module(module), cls)
    config.pop("complement_map", None)
    if cls == "CharnaConfig":
        from models.charna import CharnaForMaskedLM
        model = CharnaForMaskedLM(config_cls(**config))
    else:
        from models.llama.modelling_llama import TransformerLMHeadModel
        model = TransformerLMHeadModel(config_cls(**config))
    ckpt = torch.load(run_dir / "checkpoints" / ckpt_name, map_location="cpu")
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model, ckpt.get("global_step")


def masked_batches(args, tokenizer):
    """Fixed masked sequences: evenly spaced over the split, same masks for every model."""
    ds = HG38Dataset(split=args.split, bed_file=args.bed, fasta_file=args.fasta, max_length=1024,
                     tokenizer=tokenizer, tokenizer_name="char", add_eos=True, rc_aug=False,
                     mlm=True, mlm_probability=0.15)
    step = max(len(ds) // args.n_seqs, 1)
    indices = list(range(0, len(ds), step))[:args.n_seqs]
    torch.manual_seed(args.seed)  # masking uses torch's global random generator
    items = [ds[i] for i in indices]
    for start in range(0, len(items), args.batch):
        data, target = map(torch.stack, zip(*items[start:start + args.batch]))
        yield data, target


@torch.no_grad()
def evaluate(model, batches, tokenizer, device):
    base_ids = torch.tensor([tokenizer._vocab_str_to_int[b] for b in BASES], device=device)
    pad = tokenizer.pad_token_id
    total_loss, n, correct, correct_acgt = 0.0, 0, 0, 0
    per_base = {b: [0, 0] for b in BASES}  # correct, total
    for data, target in batches:
        data, target = data.to(device), target.to(device)
        logits = model(data).logits.float()
        hidden = target != pad
        logits_h, target_h = logits[hidden], target[hidden]
        total_loss += F.cross_entropy(logits_h, target_h, reduction="sum").item()
        n += target_h.numel()
        correct += (logits_h.argmax(-1) == target_h).sum().item()
        pred_acgt = base_ids[logits_h[:, base_ids].argmax(-1)]
        correct_acgt += (pred_acgt == target_h).sum().item()
        for b, bid in zip(BASES, base_ids):
            is_b = target_h == bid
            per_base[b][0] += (pred_acgt[is_b] == bid).sum().item()
            per_base[b][1] += is_b.sum().item()
    loss = total_loss / n
    row = {"hidden_bases": n, "loss": loss, "perplexity": math.exp(loss),
           "acc": correct / n, "acc_ACGT": correct_acgt / n}
    row.update({f"acc_{b}": c / t if t else float("nan") for b, (c, t) in per_base.items()})
    row.update({f"share_{b}": t / n for b, (c, t) in per_base.items()})
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run_dir[:label]")
    ap.add_argument("--fasta", default=str(REPO / "data/hg38/hg38.ml.fa"))
    ap.add_argument("--bed", default=str(REPO / "data/hg38/human-sequences.bed"))
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--n_seqs", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="last.ckpt", help="checkpoint file in <run_dir>/checkpoints/")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv")
    ap.add_argument("--markdown")
    args = ap.parse_args()

    tokenizer = CaduceusTokenizer(model_max_length=1024)
    rows = []
    for run in args.runs:
        path, _, label = run.partition(":")
        model, step = build_model(pathlib.Path(path), args.ckpt)
        model.to(args.device).eval()
        row = {"model": label or path, "step": step, "split": args.split}
        row.update(evaluate(model, masked_batches(args, tokenizer), tokenizer, args.device))
        rows.append(row)
        print(f"{row['model']}: step {step}, loss {row['loss']:.4f}, acc {row['acc']:.4f}, "
              f"acc_ACGT {row['acc_ACGT']:.4f} on {row['hidden_bases']:,d} hidden bases")
        del model

    majority = max(BASES, key=lambda b: rows[0][f"share_{b}"])
    lines = [
        f"Masked-base prediction on {args.n_seqs:,d} {args.split} sequences of 1,024 bases "
        f"({rows[0]['hidden_bases']:,d} hidden bases, 15% masking, seed {args.seed}).",
        "",
        "| Model | Step | Loss (nats) | Perplexity | Accuracy | Acc. A | Acc. C | Acc. G | Acc. T |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['step']} | {r['loss']:.4f} | {r['perplexity']:.3f} | "
                     f"{100 * r['acc_ACGT']:.2f}% | " +
                     " | ".join(f"{100 * r[f'acc_{b}']:.1f}%" for b in BASES) + " |")
    lines.append(f"| Random guess | – | 1.3863 | 4.000 | 25.00% | 25% | 25% | 25% | 25% |")
    lines.append(f"| Always \"{majority}\" (most common hidden base) | – | – | – | "
                 f"{100 * rows[0][f'share_{majority}']:.2f}% | " +
                 " | ".join("100%" if b == majority else "0%" for b in BASES) + " |")
    lines += ["", "Accuracy = the model's best guess among A, C, G, T is the true base. "
              "Base composition of the hidden positions: " +
              ", ".join(f"{b} {100 * rows[0][f'share_{b}']:.1f}%" for b in BASES) + "."]
    table = "\n".join(lines)
    print("\n" + table)
    if args.markdown:
        pathlib.Path(args.markdown).write_text(table + "\n")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
