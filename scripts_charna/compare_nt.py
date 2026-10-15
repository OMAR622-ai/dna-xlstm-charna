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


def run_scores(path):
    """(best validation score, test score) of one run, or None if it has no test score."""
    rows = list(csv.DictReader(open(path, newline="")))
    test = [v for k, v in (rows[-1].items() if rows else []) if k.startswith("test/") and k != "test/loss" and v]
    if not test:
        return None
    val_key = [k for k in rows[0] if k.startswith("val/") and k != "val/loss"][0]
    return max(float(r[val_key]) for r in rows if r.get(val_key)), float(test[0])


def our_scores(task, folder):
    """Test scores over seeds for the setting (LR, batch) with the best mean validation score.

    Only settings with the largest number of finished seeds compete. The test set is never used
    to choose the setting."""
    settings = {}
    for config in sorted((DOWN / "nt_cv10_ep20" / task / folder).glob("LR-*")):
        runs = [s for s in map(run_scores, sorted(config.glob("seed-*/metrics.csv"))) if s]
        if runs:
            settings[config.name] = runs
    if not settings:
        return [], None
    n = max(len(r) for r in settings.values())
    best = max((c for c in settings if len(settings[c]) == n),
               key=lambda c: statistics.mean(v for v, _ in settings[c]))
    return [t for _, t in settings[best]], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=1)
    args = ap.parse_args()
    paper = {r["task"]: r for r in csv.DictReader(open(DOWN / "bioxlstm_table1_nt.csv", newline=""))}
    paper_models = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH"]
    header = ["Task", "Metric"] + paper_models + list(OURS) + ["seeds"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    rows_csv, wins, chosen = [], {m: 0 for m in paper_models + list(OURS)}, []
    for task in TASKS:
        p = paper[task]
        means = {m: (float(p[f"{m} mean"]), float(p[f"{m} range"])) for m in paper_models}
        n_seeds = []
        for name, folder in OURS.items():
            s, setting = our_scores(task, folder)
            n_seeds.append(len(s))
            if setting:
                chosen.append(f"{task} {name}: {setting}")
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
                 "DNA-xLSTM-PH settings per task (Table A5), PH; for splice acceptors/donors also LR 1e-3 and 5e-4, chosen by validation score. Bold = best mean per task.")
    lines.append("Tasks won (ties counted for each): " + ", ".join(f"{m} {w}" for m, w in wins.items()))
    lines.append("")
    lines.append("Setting used per task and model (best mean validation score when several were run): "
                 + "; ".join(chosen))
    table = "\n".join(lines)
    print(table)
    (DOWN / "nt_comparison.md").write_text(table + "\n")
    with open(DOWN / "nt_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
