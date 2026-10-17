"""Figure of the NT-benchmark results: GBST-BERT and Transformer++ next to the best Bio-xLSTM model.

Per task: mean test score over seeds (dot) and min-max over seeds (line), for our two models,
and the best mean of the five Bio-xLSTM Table 1 models (diamond, with its range). Left: MCC tasks;
right: F1 / accuracy tasks (different scale).

    python scripts_charna/plot_nt_comparison.py   # -> charna_project/results/downstream/nt_comparison.{png,pdf}
"""

import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_nt import DOWN, OURS, TASKS, our_scores

PAPER = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH"]
COLORS = {"GBST-BERT": "#2a78d6", "Transformer++": "#eb6834", "paper": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    paper = {r["task"]: r for r in csv.DictReader(open(DOWN / "bioxlstm_table1_nt.csv", newline=""))}
    groups = [[t for t in TASKS if paper[t]["metric"] == "MCC"], [t for t in TASKS if paper[t]["metric"] != "MCC"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), gridspec_kw={"width_ratios": [12, 7]})
    for ax, tasks, title in zip(axes, groups, ["Histone marks and enhancers (MCC)", "Promoters and splice sites (F1 / accuracy)"]):
        for i, task in enumerate(tasks):
            y = len(tasks) - 1 - i
            best = max(PAPER, key=lambda m: float(paper[task][f"{m} mean"]))
            mean, rng = float(paper[task][f"{best} mean"]), float(paper[task][f"{best} range"])
            ax.plot([mean - rng / 2, mean + rng / 2], [y + 0.25] * 2, color=COLORS["paper"], lw=2, solid_capstyle="round")
            ax.plot(mean, y + 0.25, marker="D", ms=8, mfc="white", mec=COLORS["paper"], mew=2, ls="none")
            ax.text(mean, y + 0.42, best, fontsize=7.5, color=MUTED, ha="center", va="bottom")
            for name, dy in zip(OURS, (0, -0.25)):
                s, _ = our_scores(task, OURS[name])
                ax.plot([min(s), max(s)], [y + dy] * 2, color=COLORS[name], lw=2, solid_capstyle="round")
                ax.plot(statistics.mean(s), y + dy, marker="o", ms=8, color=COLORS[name], mec="white", mew=1.5, ls="none")
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(reversed(tasks), fontsize=10, color=TEXT)
        ax.set_ylim(-0.6, len(tasks) - 0.35)
        ax.set_title(title, fontsize=12, color=TEXT, loc="left")
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        ax.tick_params(axis="y", length=0)
    axes[1].set_xlim(0.84, 1.0)
    handles = [plt.Line2D([], [], color=COLORS["GBST-BERT"], marker="o", ms=8, lw=2, label="GBST-BERT (ours)"),
               plt.Line2D([], [], color=COLORS["Transformer++"], marker="o", ms=8, lw=2, label="Transformer++ (ours)"),
               plt.Line2D([], [], color=COLORS["paper"], marker="D", ms=8, mfc="white", mew=2, lw=2,
                          label="Best model in Bio-xLSTM Table 1")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.99))
    fig.text(0.5, 0.01, "Dot / diamond: mean test score over 10 seeds; line: range (min to max). Higher is better. "
             "Right panel starts at 0.84.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(DOWN / f"nt_comparison.{ext}", dpi=200)
    print(DOWN / "nt_comparison.png")


if __name__ == "__main__":
    main()
