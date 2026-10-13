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


# The full GBST-BERT run (R16) logged no step-10,000 validation (after resuming, Lightning counted
# one step ahead). That point was measured afterwards on the final model with the same validation
# settings (run R21, scripts_charna/server/final_val_charna.sh) and is added here.
FINAL_VAL = {"full_charna_mlm": REPO / "charna_project/results/full/final_val_charna/metrics.csv"}


def read(run):
    xs, ys = [], []
    with open(RUNS / run / "metrics.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("val/loss"):
                xs.append(int(row["tokens"]) / 1e9)
                ys.append(float(row["val/loss"]))
    if run in FINAL_VAL and FINAL_VAL[run].exists():
        with open(FINAL_VAL[run], newline="") as f:
            row = next(csv.DictReader(f))
        xs.append(10_000 * 1_048_576 / 1e9)
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


MARKER_GLYPH = {"o": "\u25cf", "s": "\u25a0", "^": "\u25b2", "D": "\u25c6", "v": "\u25bc"}


def last_val(run):
    return read(run)[1][-1]


def final_test():
    """Final test loss and accuracy of the two full models (run R18, evaluate_mlm.py)."""
    with open(REPO / "charna_project/results/full/eval_test.csv", newline="") as f:
        return {r["model"]: (float(r["loss"]), float(r["acc_ACGT"])) for r in csv.DictReader(f)}


def draw_table(ax):
    test = final_test()
    rows = [  # (series, steps trained, role)
        ("GBST-BERT, LR 2e-3", "500 + 10,000", "best in the check; used for the full run", "GBST-BERT"),
        ("GBST-BERT, LR 5e-3", "500", "second best", None),
        ("GBST-BERT, LR 1e-3", "500", "", None),
        ("GBST-BERT, LR 5e-4", "500", "ChaRNABERT paper value (Table 2)", None),
        ("Transformer++, LR 1e-2", "500 + 10,000", "comparison model; Bio-xLSTM value (Table A3)", "Transformer++"),
    ]
    header = ["", "Model", "Learning rate", "Steps trained", "Val. loss\nat 0.5 B bases",
              "Val. loss\nat 10.5 B bases", "Final test\nloss", "Final test\naccuracy", "Note"]
    cells = []
    for name, steps, note, full in rows:
        model, lr = name.split(", LR ")
        full_run = FULL.get(name)
        val84 = f"{read(full_run)[1][-1]:.3f}" if full_run else "\u2013"  # step 10,000
        t_loss, t_acc = (f"{test[full][0]:.4f}", f"{100 * test[full][1]:.2f}%") if full else ("\u2013", "\u2013")
        cells.append([MARKER_GLYPH[SERIES[name]["marker"]], model, lr, steps, f"{last_val(SMALL[name]):.3f}",
                      val84, t_loss, t_acc, note])
    cells.append(["", "Random guessing", "", "", "1.386", "1.386", "1.386", "25.00%", "reference"])
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=header, cellLoc="center", bbox=[0, 0.24, 1, 0.76],
                     colWidths=[0.03, 0.12, 0.08, 0.1, 0.09, 0.09, 0.08, 0.08, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.get_text().set_color(INK)
        if r == 0:
            cell.set_facecolor("#f0efea")
        else:
            cell.set_facecolor("#fcfcfb")
        if c == 8 and r > 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.02
        if c == 0 and 0 < r <= len(rows):
            cell.get_text().set_color(SERIES[rows[r - 1][0]]["color"])
            cell.get_text().set_fontsize(11)
        if r in (1, 5) and c in (6, 7):
            cell.get_text().set_fontweight("bold")
    ax.text(0.0, 0.0,
            "How to read it: loss = how wrong the guesses of hidden bases are (lower is better); accuracy = share of hidden bases "
            "guessed correctly on unseen test DNA.\n"
            "Left panel: 500-step check of four GBST-BERT learning rates (2e-3 best). Right panel: the full 10.5-billion-base runs, "
            "trained only at each model's chosen learning rate.\n"
            "Result: GBST-BERT 56.44% vs Transformer++ 56.33% test accuracy \u2014 a 0.1-point difference, too small to call a "
            "winner after one run per model.",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom", ha="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-table", action="store_true", help="add the results table and an explanation below the curves")
    args = ap.parse_args()
    if args.out is None:
        name = "lr_and_full_val_loss_table.png" if args.with_table else "lr_and_full_val_loss.png"
        args.out = str(REPO / "charna_project/results" / name)

    if args.with_table:
        fig = plt.figure(figsize=(12, 8.8))
        gs = fig.add_gridspec(3, 2, height_ratios=[4.6, 0.35, 3.4], width_ratios=[1, 1.15], hspace=0.32)
        left, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        legend_ax, table_ax = fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
    else:
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
    right.text(14.0, 1.055, "GBST-BERT's step-10,000 point was measured after training\n"
               "on the final model, with the same validation settings (see README).", fontsize=8, color=MUTED,
               va="top", ha="right")
    handles, labels = left.get_legend_handles_labels()
    fig.suptitle("Validation loss for every trained learning rate (lower is better; random guessing = 1.386)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    if args.with_table:
        legend_ax.axis("off")
        legend_ax.legend(handles, labels, loc="center", ncol=5, frameon=False, fontsize=9, labelcolor=INK)
        draw_table(table_ax)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.03)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(args.out, dpi=160, facecolor="#fcfcfb")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
