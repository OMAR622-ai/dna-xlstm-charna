"""Compare our Genomic Benchmarks results with Bio-xLSTM Table A2.

Reads the fine-tuning runs of scripts_charna/server/run_gb_benchmark.sh (copied into
charna_project/results/downstream/gb_cv5_ep10/): for every task and model, the test accuracy of the
best-validation epoch of each seed. Reports mean ± (max - min) over the 5 seeds, as in Bio-xLSTM
Table A2, next to the paper's values (charna_project/results/downstream/bioxlstm_tableA2_gb.csv).
Both of our models use the same 5 train/validation splits, so the seeds are also compared pairwise.

    python scripts_charna/compare_gb.py
"""

import csv
import statistics

from compare_nt import DOWN, run_scores

OURS = {"GBST-BERT": "full_charna_mlm_ph", "Transformer++": "full_llama_mlm_ph"}
PAPER = ["Mamba-PH-500k", "xLSTM-PH-500k", "Mamba-PS-500k", "xLSTM-PS-500k"]
SETTING = "LR-1e-3_BATCH_SIZE-128"
TASKS = ["dummy_mouse_enhancers_ensembl", "demo_coding_vs_intergenomic_seqs", "demo_human_or_worm",
         "human_enhancers_cohn", "human_enhancers_ensembl", "human_ensembl_regulatory", "human_ocr_ensembl",
         "human_nontata_promoters"]
NAMES = {"dummy_mouse_enhancers_ensembl": "Mouse enhancers", "demo_coding_vs_intergenomic_seqs": "Coding vs intergenomic",
         "demo_human_or_worm": "Human vs worm", "human_enhancers_cohn": "Human enhancers (Cohn)",
         "human_enhancers_ensembl": "Human enhancers (Ensembl)", "human_ensembl_regulatory": "Human regulatory (3 classes)",
         "human_ocr_ensembl": "Human open chromatin (OCR)", "human_nontata_promoters": "Human non-TATA promoters"}


def our_scores(task, folder):
    """{seed: test accuracy} of the finished runs."""
    out = {}
    for path in sorted((DOWN / "gb_cv5_ep10" / task / folder / SETTING).glob("seed-*/metrics.csv")):
        s = run_scores(path)
        if s:
            out[int(path.parent.name.split("-")[1])] = s[1]
    return out


def main():
    paper = {r["task"]: r for r in csv.DictReader(open(DOWN / "bioxlstm_tableA2_gb.csv", newline=""))}
    header = ["Task"] + PAPER + list(OURS) + ["GBST-BERT better (splits)", "seeds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rows_csv, wins = [], {m: 0 for m in PAPER + list(OURS)}
    for task in TASKS:
        p = paper[task]
        means = {m: (float(p[f"{m} mean"]), float(p[f"{m} range"])) for m in PAPER}
        seeds = {name: our_scores(task, folder) for name, folder in OURS.items()}
        for name, s in seeds.items():
            v = list(s.values())
            means[name] = (statistics.mean(v), max(v) - min(v))
        common = sorted(set(seeds["GBST-BERT"]) & set(seeds["Transformer++"]))
        better = sum(seeds["GBST-BERT"][k] > seeds["Transformer++"][k] for k in common)
        best = max(v[0] for v in means.values())
        for m, v in means.items():
            if abs(v[0] - best) < 5e-4:
                wins[m] += 1
        cells = []
        for m in PAPER + list(OURS):
            txt = f"{means[m][0]:.3f}±{means[m][1]:.3f}"
            cells.append(f"**{txt}**" if abs(means[m][0] - best) < 5e-4 else txt)
        n = min(len(s) for s in seeds.values())
        lines.append(f"| {NAMES[task]} | " + " | ".join(cells) + f" | {better} of {len(common)} | {n} |")
        rows_csv.append({"task": task, "name": NAMES[task],
                         **{f"{m} mean": f"{means[m][0]:.6f}" for m in means},
                         **{f"{m} range": f"{means[m][1]:.6f}" for m in means},
                         "GBST-BERT better splits": better, "splits": len(common), "seeds": n})
    lines.append("")
    lines.append("Top-1 test accuracy, mean ± (max − min) over 5 train/validation splits. Paper values: Bio-xLSTM "
                 "Table A2 (500k-parameter models; Mamba scores from Schiff et al. 2024). Ours: final pretrained "
                 "2M-parameter models (10,000 steps), PH, 10 epochs, one setting for every task (LR 1e-3, batch 128). "
                 "Bold = best mean per task.")
    lines.append("Tasks won (ties counted for each): " + ", ".join(f"{m} {w}" for m, w in wins.items()))
    table = "\n".join(lines)
    print(table)
    (DOWN / "gb_comparison.md").write_text(table + "\n")
    with open(DOWN / "gb_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
