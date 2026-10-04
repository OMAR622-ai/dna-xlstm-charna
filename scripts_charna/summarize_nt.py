"""Summarise Nucleotide Transformer benchmark runs written by scripts_downstream/run_nucleotide_charna.sh.

Layout: <root>/<task>/<model>/<config>/seed-<s>/metrics.csv. Rows before the last hold the
per-epoch validation metric (val/<metric>); the last row holds the test metric of the
checkpoint with the best validation metric (test/<metric>), as logged by train.py.

Per task and model, the grid point (config) with the highest mean best-validation metric over
seeds is selected; for it the test metric is reported as mean, std (sample, ddof=1) and
max - min over seeds. Only configs with the full number of seeds are considered.

  python scripts_charna/summarize_nt.py outputs/downstream/nt_cv10_ep20 [--seeds 10] [--markdown out.md]
"""

import argparse
import csv
import pathlib
import statistics

TASK_ORDER = [
    "H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2", "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac",
    "enhancers", "enhancers_types", "promoter_all", "promoter_no_tata", "promoter_tata",
    "splice_sites_all", "splice_sites_acceptors", "splice_sites_donors",
]


def read_run(path):
    """Return (best validation metric, test metric, metric name) of one seed, or None if unfinished."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    test_keys = [k for k in (rows[-1] if rows else {}) if k.startswith("test/") and rows[-1][k] != ""]
    test_keys = [k for k in test_keys if k != "test/loss"]
    if len(rows) < 2 or not test_keys:
        return None
    metric = test_keys[0].split("/", 1)[1]
    val = [float(r[f"val/{metric}"]) for r in rows[:-1] if r.get(f"val/{metric}", "") != ""]
    if not val:
        return None
    return max(val), float(rows[-1][f"test/{metric}"]), metric


def summarize(root, n_seeds):
    results = {}  # (task, model) -> dict
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            best = None
            for config_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                runs = [read_run(m) for m in sorted(config_dir.glob("seed-*/metrics.csv"))]
                runs = [r for r in runs if r is not None]
                if len(runs) < n_seeds:
                    continue
                val_mean = statistics.mean(r[0] for r in runs)
                if best is None or val_mean > best["val_mean"]:
                    tests = [r[1] for r in runs]
                    best = {
                        "config": config_dir.name, "metric": runs[0][2], "n": len(runs),
                        "val_mean": val_mean, "test_mean": statistics.mean(tests),
                        "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
                        "test_range": max(tests) - min(tests),
                    }
            if best is not None:
                results[(task_dir.name, model_dir.name)] = best
    return results


def to_markdown(results):
    models = sorted({m for _, m in results})
    tasks = sorted({t for t, _ in results}, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99)
    lines = ["| Task | Metric | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    for task in tasks:
        metric = next(results[(task, m)]["metric"] for m in models if (task, m) in results)
        cells = []
        for m in models:
            r = results.get((task, m))
            cells.append("-" if r is None else
                         f"{r['test_mean']:.3f} ± {r['test_std']:.3f} (range {r['test_range']:.3f})")
        lines.append(f"| {task} | {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Mean ± std (ddof=1) of the test metric over seeds; range = max - min. "
                 "Selected grid point per task: " +
                 "; ".join(f"{t}/{m}: {r['config']} (n={r['n']})" for (t, m), r in sorted(results.items())))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--seeds", type=int, default=10, help="seeds required per grid point")
    parser.add_argument("--markdown", type=pathlib.Path, help="also write the table to this file")
    args = parser.parse_args()
    table = to_markdown(summarize(args.root, args.seeds))
    print(table)
    if args.markdown:
        args.markdown.write_text(table + "\n")


if __name__ == "__main__":
    main()
