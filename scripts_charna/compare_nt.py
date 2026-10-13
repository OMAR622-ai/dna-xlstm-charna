"""Compare our NT-benchmark results with Bio-xLSTM Table 1.

Reads the fine-tuning runs of scripts_charna/server/run_nt_benchmark.sh (copied into
charna_project/results/downstream/nt_cv10_ep20/ by collect_results.sh): for every task and model,
the test metric of the best-validation epoch of each seed. Reports mean ± (max - min) over seeds,
as in Bio-xLSTM Table 1, next to the paper's values (charna_project/results/downstream/
bioxlstm_table1_nt.csv). The best mean per task is in bold.

    python scripts_charna/compare_nt.py [--min-seeds 1]
"""

import argparse
import csv
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
DOWN = REPO / "charna_project/results/downstream"
OURS = {"GBST-BERT": "full_charna_mlm_ph", "Transformer++": "full_llama_mlm_ph"}
TASKS = ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac",
         "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata",
         "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors"]


def our_scores(task, folder):
    scores = []
    for path in sorted((DOWN / "nt_cv10_ep20" / task / folder).glob("*/seed-*/metrics.csv")):
        rows = list(csv.DictReader(open(path, newline="")))
        test = [(k, v) for k, v in (rows[-1].items() if rows else []) if k.startswith("test/") and k != "test/loss" and v]
        if test:
            scores.append(float(test[0][1]))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=1)
    args = ap.parse_args()
    paper = {r["task"]: r for r in csv.DictReader(open(DOWN / "bioxlstm_table1_nt.csv", newline=""))}
    paper_models = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH"]
    header = ["Task", "Metric"] + paper_models + list(OURS) + ["seeds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rows_csv, wins = [], {m: 0 for m in paper_models + list(OURS)}
    for task in TASKS:
        p = paper[task]
        means = {m: (float(p[f"{m} mean"]), float(p[f"{m} range"])) for m in paper_models}
        n_seeds = []
        for name, folder in OURS.items():
            s = our_scores(task, folder)
            n_seeds.append(len(s))
            if len(s) >= args.min_seeds:
                means[name] = (statistics.mean(s), max(s) - min(s))
        best = max(v[0] for v in means.values())
        for m, v in means.items():
            if abs(v[0] - best) < 5e-4:
                wins[m] += 1
        cells = []
        for m in paper_models + list(OURS):
            if m not in means:
                cells.append("–")
                continue
            txt = f"{means[m][0]:.3f}±{means[m][1]:.3f}"
            cells.append(f"**{txt}**" if abs(means[m][0] - best) < 5e-4 else txt)
        lines.append(f"| {task} | {p['metric']} | " + " | ".join(cells) + f" | {min(n_seeds)} |")
        rows_csv.append({"task": task, "metric": p["metric"], **{f"{m} mean": f"{means[m][0]:.4f}" for m in means},
                         **{f"{m} range": f"{means[m][1]:.4f}" for m in means}, "seeds": min(n_seeds)})
    lines.append("")
    lines.append("Mean ± (max − min) over seeds of the test metric. Paper values: Bio-xLSTM Table 1 (10 seeds, "
                 "per-task hyperparameter search). Ours: final pretrained models (10,000 steps), Bio-xLSTM's "
                 "DNA-xLSTM-PH settings per task (Table A5), PH. Bold = best mean per task.")
    lines.append("Tasks won (ties counted for each): " + ", ".join(f"{m} {w}" for m, w in wins.items()))
    table = "\n".join(lines)
    print(table)
    (DOWN / "nt_comparison.md").write_text(table + "\n")
    with open(DOWN / "nt_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
