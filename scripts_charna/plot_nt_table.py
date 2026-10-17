"""Table image of the NT-benchmark results of all seven models (five Bio-xLSTM Table 1 models and ours).

Cell: mean test score over 10 seeds, with the range (max - min) below it. Best mean per task (within
0.0005) highlighted. Last row: number of tasks where the model has the best mean (ties counted).

    python scripts_charna/plot_nt_table.py   # -> charna_project/results/downstream/nt_table.{png,pdf}
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_nt import DOWN, TASKS

MODELS = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH", "GBST-BERT", "Transformer++"]
GROUPS = [("Histone marks (MCC)", TASKS[:10]), ("Enhancers (MCC)", TASKS[10:12]),
          ("Promoters (F1)", TASKS[12:15]), ("Splice sites (accuracy / F1)", TASKS[15:])]
TEXT, MUTED, RULE, BAND, BEST, OURS_BG = "#0b0b0b", "#52514e", "#c9c7c0", "#f1f0ec", "#cfe2f8", "#eef4fc"


def main():
    rows = {r["task"]: r for r in csv.DictReader(open(DOWN / "nt_comparison.csv", newline=""))}
    wins = {m: 0 for m in MODELS}
    n_rows = len(TASKS) + len(GROUPS) + 2
    fig, ax = plt.subplots(figsize=(15, 0.42 * n_rows + 1.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(n_rows, -1.2)
    ax.axis("off")
    x_task, x0, w = 0.5, 20.5, 11.3
    xs = [x0 + w * i + w / 2 for i in range(len(MODELS))]
    # our two models: light column background
    for i in (5, 6):
        ax.add_patch(plt.Rectangle((x0 + w * i, -1.1), w, n_rows + 0.9, color=OURS_BG, zorder=0))
    ax.text(x0 + w * 6, -1.0, "ours", ha="center", fontsize=10, color="#1f5fae", style="italic")
    for x, m in zip(xs, MODELS):
        ax.text(x, -0.3, m, ha="center", va="center", fontsize=11.5, fontweight="bold", color=TEXT)
    ax.text(x_task, -0.3, "Task", va="center", fontsize=11.5, fontweight="bold", color=TEXT)
    ax.plot([0, 100], [0.25, 0.25], color=TEXT, lw=1.2)
    y = 0.9
    for title, tasks in GROUPS:
        ax.text(x_task, y, title, va="center", fontsize=11, fontweight="bold", color="#1f5fae")
        y += 1
        for t in tasks:
            r = rows[t]
            means = {m: float(r[f"{m} mean"]) for m in MODELS}
            best = max(means.values())
            ax.text(x_task + 1.2, y, t, va="center", fontsize=11, color=TEXT)
            for x, m in zip(xs, MODELS):
                is_best = abs(means[m] - best) < 5e-4
                if is_best:
                    wins[m] += 1
                    ax.add_patch(plt.Rectangle((x - w / 2 + 0.3, y - 0.45), w - 0.6, 0.9, color=BEST, zorder=1))
                ax.text(x, y - 0.1, f"{means[m]:.3f}", ha="center", va="center", fontsize=11,
                        fontweight="bold" if is_best else "normal", color=TEXT, zorder=2)
                ax.text(x, y + 0.27, f"± {float(r[f'{m} range']):.3f}", ha="center", va="center", fontsize=7.5,
                        color=MUTED, zorder=2)
            y += 1
    ax.plot([0, 100], [y - 0.4, y - 0.4], color=TEXT, lw=1.2)
    ax.text(x_task, y + 0.2, "Tasks with the best mean", va="center", fontsize=11.5, fontweight="bold", color=TEXT)
    for x, m in zip(xs, MODELS):
        ax.text(x, y + 0.2, str(wins[m]), ha="center", va="center", fontsize=13, fontweight="bold",
                color="#1f5fae" if m == "GBST-BERT" else TEXT)
    fig.text(0.01, 0.01, "Mean test score over 10 repeats (higher is better); small number = range (best minus worst "
             "repeat). Highlighted = best mean for that task (ties counted for each model). "
             "First five columns: Bio-xLSTM paper, Table 1.", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(DOWN / f"nt_table.{ext}", dpi=200)
    print(wins)


if __name__ == "__main__":
    main()
