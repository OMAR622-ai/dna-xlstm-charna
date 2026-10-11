"""Validation-loss curves of every learning rate that was trained, in two panels.

Left: the learning-rate check (500 steps, 0.52 B bases): GBST-BERT at 5e-4, 1e-3, 2e-3, 5e-3 and
Transformer++ at 1e-2. Right: the full training (10,000 steps, 10.5 B bases), which was run only for
GBST-BERT at 2e-3 (best in the check) and Transformer++ at 1e-2. Each series keeps its colour and
marker in both panels. Run from the repository root:

    python scripts_charna/plot_lr_curves.py [--out charna_project/results/lr_and_full_val_loss.png]
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
RUNS = REPO / "charna_project/results/pretrain/hg38"

# Categorical palette in fixed order (validated: lightness, chroma, CVD and normal-vision
# separation pass on the light surface); marker shapes as secondary encoding.
SERIES = {
    "GBST-BERT, LR 2e-3": dict(color="#2a78d6", marker="o"),
    "Transformer++, LR 1e-2": dict(color="#eb6834", marker="s"),
    "GBST-BERT, LR 1e-3": dict(color="#1baf7a", marker="^"),
    "GBST-BERT, LR 5e-3": dict(color="#eda100", marker="D"),
    "GBST-BERT, LR 5e-4": dict(color="#e87ba4", marker="v"),
}
SMALL = {
    "GBST-BERT, LR 5e-4": "small_charna_500",
    "GBST-BERT, LR 1e-3": "small_charna_500_lr1e-3",
    "GBST-BERT, LR 2e-3": "small_charna_500_lr2e-3",
    "GBST-BERT, LR 5e-3": "small_charna_500_lr5e-3",
    "Transformer++, LR 1e-2": "small_llama_500",
}
FULL = {
    "GBST-BERT, LR 2e-3": "full_charna_mlm",
    "Transformer++, LR 1e-2": "full_llama_mlm",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def read(run):
    xs, ys = [], []
    with open(RUNS / run / "metrics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("val/loss"):
                xs.append(int(row["tokens"]) / 1e9)
                ys.append(float(row["val/loss"]))
    return xs, ys


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, pad=10)
    ax.set_xlabel("bases seen (billions)", color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "charna_project/results/lr_and_full_val_loss.png"))
    args = ap.parse_args()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for panel, runs, title, min_gap, label_x in (
            (left, SMALL, "Learning-rate check (500 steps)", 0.0085, 0.555),
            (right, FULL, "Full training (10,000 steps)", 0.0042, 10.9)):
        ends = []
        for name in SERIES:  # fixed slot order
            if name not in runs:
                continue
            xs, ys = read(runs[name])
            s = SERIES[name]
            panel.plot(xs, ys, color=s["color"], linewidth=2, marker=s["marker"], markersize=6.5,
                       markeredgecolor="white", markeredgewidth=1.2, label=name, zorder=3)
            ends.append([ys[-1], ys[-1], xs[-1], name])
            print(f"{title:32s} {name:24s} " + ", ".join(f"{x:.2f}B {y:.4f}" for x, y in zip(xs, ys)))
        # direct labels in a column right of the lines, pushed apart so they never overlap,
        # each joined to its line end by a thin leader (text in ink; the marker carries the colour)
        ends.sort(key=lambda e: e[0])
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + min_gap)
        for y_end, y_label, x_end, name in ends:
            panel.annotate(f"{name.replace('GBST-BERT, LR ', 'GBST ').replace('Transformer++, LR ', 'T++ ')}  {y_end:.3f}", (x_end, y_end), xytext=(label_x, y_label),
                           textcoords="data", va="center", fontsize=8.5, color=INK,
                           arrowprops=dict(arrowstyle="-", color=GRID, linewidth=1, shrinkA=0, shrinkB=4))
        style(panel, title)
    left.set_ylabel("validation loss (nats per hidden base)", color=MUTED)
    left.set_xlim(0.05, 0.72)
    right.set_xlim(1.5, 14.2)
    right.text(14.0, 1.055, "GBST-BERT has no step-10,000 point (see README).\n"
               "Final test loss: GBST-BERT 0.9753, Transformer++ 0.9774", fontsize=8, color=MUTED,
               va="top", ha="right")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle("Validation loss for every trained learning rate (lower is better; random guessing = 1.386)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(args.out, dpi=160, facecolor="#fcfcfb")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
