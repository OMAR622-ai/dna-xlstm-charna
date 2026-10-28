"""Figures of the Genomic Benchmarks results (from gb_comparison.csv, written by compare_gb.py).

gb_table.{png,pdf}: test accuracy of all six models (four Bio-xLSTM Table A2 models and ours), best
mean per task highlighted. gb_simple.{png,pdf}: GBST-BERT minus Transformer++ per task, with the
number of the 5 shared train/validation splits on which GBST-BERT was better.

    python scripts_charna/plot_gb.py
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_gb import PAPER
from compare_nt import DOWN

MODELS = PAPER + ["GBST-BERT", "Transformer++"]
LABELS = ["Mamba-PH\n500k", "xLSTM-PH\n500k", "Mamba-PS\n500k", "xLSTM-PS\n500k", "GBST-BERT\n2M", "Transformer++\n2M"]
TEXT, MUTED, BEST, OURS_BG, BLUE, GREY = "#0b0b0b", "#52514e", "#cfe2f8", "#eef4fc", "#2a78d6", "#a3a9b4"


def table(rows):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(13, 0.55 * n + 2.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(n + 1.7, -1.6)
    ax.axis("off")
    x_task, x0, w = 0.5, 25, 12.5
    xs = [x0 + w * i + w / 2 for i in range(len(MODELS))]
    for i in (4, 5):
        ax.add_patch(plt.Rectangle((x0 + w * i, -1.5), w, n + 2.6, color=OURS_BG, zorder=0))
    ax.text(x0 + w * 5, -1.45, "ours", ha="center", fontsize=10, color="#1f5fae", style="italic", va="top")
    for x, m in zip(xs, LABELS):
        ax.text(x, -0.55, m, ha="center", va="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.text(x_task, -0.55, "Task (top-1 accuracy)", va="center", fontsize=11, fontweight="bold", color=TEXT)
    ax.plot([0, 100], [0.15, 0.15], color=TEXT, lw=1.2)
    wins = {m: 0 for m in MODELS}
    for y, r in enumerate(rows, start=1):
        means = {m: float(r[f"{m} mean"]) for m in MODELS}
        best = max(means.values())
        ax.text(x_task, y - 0.1, r["name"], va="center", fontsize=11, color=TEXT)
        for x, m in zip(xs, MODELS):
            is_best = abs(means[m] - best) < 5e-4
            if is_best:
                wins[m] += 1
                ax.add_patch(plt.Rectangle((x - w / 2 + 0.4, y - 0.45), w - 0.8, 0.9, color=BEST, zorder=1))
            ax.text(x, y - 0.12, f"{means[m]:.3f}", ha="center", va="center", fontsize=11,
                    fontweight="bold" if is_best else "normal", color=TEXT, zorder=2)
            ax.text(x, y + 0.25, f"± {float(r[f'{m} range']):.3f}", ha="center", va="center", fontsize=7.5,
                    color=MUTED, zorder=2)
    y = n + 1.0
    ax.plot([0, 100], [y - 0.45, y - 0.45], color=TEXT, lw=1.2)
    ax.text(x_task, y + 0.15, "Tasks with the best mean", va="center", fontsize=11.5, fontweight="bold", color=TEXT)
    for x, m in zip(xs, MODELS):
        ax.text(x, y + 0.15, str(wins[m]), ha="center", va="center", fontsize=13, fontweight="bold",
                color="#1f5fae" if m == "GBST-BERT" else TEXT)
    fig.text(0.01, 0.01, "Mean test accuracy over 5 train/validation splits (higher is better); small number = range "
             "(best minus worst split). Highlighted = best mean.\nFirst four columns: Bio-xLSTM paper, Table A2 "
             "(smaller 500k-parameter models).", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(DOWN / f"gb_table.{ext}", dpi=200)
    print(wins)


def simple(rows):
    rows = rows[::-1]
    diffs = [100 * (float(r["GBST-BERT mean"]) - float(r["Transformer++ mean"])) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ys = range(len(rows))
    ax.barh(ys, diffs, color=[BLUE if d > 0 else GREY for d in diffs], height=0.55)
    ax.axvline(0, color=TEXT, lw=1.2)
    for y, d, r in zip(ys, diffs, rows):
        side = 0.06 if d >= 0 else -0.06
        ax.text(d + side, y, f"{d:+.1f}", va="center", ha="left" if d >= 0 else "right", fontsize=11, color=TEXT)
        ax.text(2.35, y, f"{r['GBST-BERT better splits']} of {r['splits']}", va="center", ha="center", fontsize=11,
                color=TEXT)
    ax.set_ylim(-1.0, len(rows) + 0.2)
    ax.text(2.35, len(rows) - 0.45, "GBST better\non splits", ha="center", va="bottom", fontsize=10, color=MUTED)
    ax.set_yticks(list(ys), [r["name"] for r in rows], fontsize=11)
    ax.set_xlim(-2.0, 2.8)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xlabel("Difference in mean test accuracy (percentage points)", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e2e2e2", zorder=0)
    ax.set_axisbelow(True)
    n_up = sum(d > 0 for d in diffs)
    ax.set_title(f"GBST-BERT vs. the same model without GBST (Transformer++): higher mean on {n_up} of {len(rows)} tasks",
                 fontsize=12.5, loc="left", pad=28)
    ax.text(-1.95, -0.8, "◀ GBST-BERT worse", fontsize=10, color=MUTED, va="center")
    ax.text(1.95, -0.8, "GBST-BERT better ▶", fontsize=10, color=MUTED, ha="right", va="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(DOWN / f"gb_simple.{ext}", dpi=200)


def main():
    rows = list(csv.DictReader(open(DOWN / "gb_comparison.csv", newline="")))
    table(rows)
    simple(rows)


if __name__ == "__main__":
    main()
