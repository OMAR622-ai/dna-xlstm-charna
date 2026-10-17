"""Simple version of the NT-benchmark figure: how much better (or worse) GBST-BERT is, per task.

Left: GBST-BERT minus Transformer++ (the same model without GBST). Right: GBST-BERT minus the best
of the five Bio-xLSTM Table 1 models. Mean test score over 10 seeds; a bar right of zero means
GBST-BERT is better.

    python scripts_charna/plot_nt_simple.py   # -> charna_project/results/downstream/nt_simple.{png,pdf}
"""

import csv
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_nt import DOWN, OURS, TASKS, our_scores

PAPER = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH"]
GROUP = {"enhancers": "Enhancer", "enhancers_types": "Enhancer", "promoter_all": "Promoter",
         "promoter_no_tata": "Promoter", "promoter_tata": "Promoter", "splice_sites_all": "Splice site",
         "splice_sites_acceptors": "Splice site", "splice_sites_donors": "Splice site"}
NAMES = {"enhancers": "enhancer or not", "enhancers_types": "enhancer type", "promoter_all": "all",
         "promoter_no_tata": "no TATA box", "promoter_tata": "with TATA box", "splice_sites_all": "all",
         "splice_sites_acceptors": "acceptor", "splice_sites_donors": "donor"}
BETTER, WORSE, TEXT, MUTED, GRID = "#2a78d6", "#a3a9b5", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    paper = {r["task"]: r for r in csv.DictReader(open(DOWN / "bioxlstm_table1_nt.csv", newline=""))}
    gbst = {t: statistics.mean(our_scores(t, OURS["GBST-BERT"])[0]) for t in TASKS}
    llama = {t: statistics.mean(our_scores(t, OURS["Transformer++"])[0]) for t in TASKS}
    best = {t: max(float(paper[t][f"{m} mean"]) for m in PAPER) for t in TASKS}
    labels = [f"{GROUP.get(t, 'Histone')}: {NAMES.get(t, t)}" for t in TASKS]
    panels = [("vs. the same model without GBST (Transformer++)", {t: gbst[t] - llama[t] for t in TASKS}),
              ("vs. the best published model (Bio-xLSTM paper)", {t: gbst[t] - best[t] for t in TASKS})]

    fig, axes = plt.subplots(1, 2, figsize=(14, 8.2), sharey=True)
    for ax, (title, diff) in zip(axes, panels):
        wins = sum(diff[t] > 0.0005 for t in TASKS)
        ties = sum(abs(diff[t]) <= 0.0005 for t in TASKS)
        for i, t in enumerate(TASKS):
            y = len(TASKS) - 1 - i
            d = diff[t]
            ax.barh(y, d, height=0.62, color=BETTER if d > 0.0005 else WORSE, edgecolor="white", linewidth=2)
            ax.text(d + (0.004 if d >= 0 else -0.004), y, "tie" if abs(d) <= 0.0005 else f"{d:+.3f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9.5, color=TEXT)
        ax.axvline(0, color=TEXT, lw=1.2)
        ax.set_xlim(-0.24, 0.24)
        tie_txt = f", tied on {ties}" if ties else ""
        ax.set_title(f"GBST-BERT {title}\nbetter on {wins} of 18 tasks{tie_txt}", fontsize=12.5, color=TEXT, loc="left")
        ax.text(0.01, -0.07, "◀ GBST-BERT worse", transform=ax.transAxes, fontsize=10, color=MUTED)
        ax.text(0.99, -0.07, "GBST-BERT better ▶", transform=ax.transAxes, fontsize=10, color=MUTED, ha="right")
        ax.grid(axis="x", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(TASKS)))
    axes[0].set_yticklabels(reversed(labels), fontsize=10.5, color=TEXT)
    fig.text(0.5, 0.012, "Bar = difference in mean test score over 10 repeats (MCC for histone and enhancer tasks, "
             "F1 or accuracy for promoters and splice sites). Blue = GBST-BERT better; grey = worse or equal.",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(DOWN / f"nt_simple.{ext}", dpi=200)
    print(DOWN / "nt_simple.png")


if __name__ == "__main__":
    main()
