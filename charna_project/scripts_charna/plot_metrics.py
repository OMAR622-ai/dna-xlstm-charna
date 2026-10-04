"""Plot validation loss against tokens seen, from one or more pretraining metrics.csv files.

  python scripts_charna/plot_metrics.py \
      charna_project/results/pretrain/hg38/smoke_charna/metrics.csv:GBST-BERT \
      charna_project/results/pretrain/hg38/smoke_llama/metrics.csv:Transformer++ \
      --out charna_project/results/val_loss.png

Each argument is <path to metrics.csv>[:<label>]. The metrics.csv files are written by the
MetricsCSV callback (one row per validation, with `tokens` and `val/loss`).
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # write files, no window needed
import matplotlib.pyplot as plt  # noqa: E402


def read(path, column):
    tokens, values = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get(column):
                tokens.append(int(row["tokens"]) / 1e9)
                values.append(float(row[column]))
    return tokens, values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="metrics.csv[:label]")
    ap.add_argument("--metric", default="val/loss")
    ap.add_argument("--out", default="val_loss.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in args.runs:
        path, _, label = run.partition(":")
        tokens, values = read(path, args.metric)
        ax.plot(tokens, values, marker="o", label=label or path)
        print(f"{label or path}: " + ", ".join(f"{t:.3f}B -> {v:.4f}" for t, v in zip(tokens, values)))
    ax.set_xlabel("tokens seen (billions)")
    ax.set_ylabel(f"{args.metric} (nats per masked base)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
