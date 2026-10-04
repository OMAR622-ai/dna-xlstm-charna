"""Write one page per run (charna_project/runs/<id>.md) and the run index (runs/README.md).

Facts written by hand (purpose, command, status, outcome) come from charna_project/runs/runs.yaml;
settings and results are read from the run's files in charna_project/results/ (config.json,
model_config.json, metrics.csv, train.log, evaluation tables), so the pages always show what the
run actually used. Re-run after collecting new results:

  bash scripts_charna/collect_results.sh outputs charna_project/results   # on the server, then copy
  python scripts_charna/make_run_cards.py
"""

import csv
import json
import pathlib
import subprocess

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
PROJECT = REPO / "charna_project"
RESULTS = PROJECT / "results"
RUNS_DIR = PROJECT / "runs"
STATUS_ICON = {"done": "✅ done", "stopped": "⏹ stopped", "running": "⏳ running", "queued": "🕒 queued"}


def settings_rows(config):
    """Main training settings from a resolved Hydra config.json."""
    t, d, o, s, m = (config.get(k, {}) for k in ("trainer", "dataset", "optimizer", "scheduler", "model"))
    mc = m.get("config", {})
    devices, micro, accum = t.get("devices"), d.get("batch_size"), t.get("accumulate_grad_batches")
    rows = [
        ("Model", f"`{m.get('_name_')}`" + (f", tokenizer `{mc.get('tokenizer')}`" if mc.get("tokenizer") else "")),
        ("Shape", f"d_model {mc.get('d_model')}, {mc.get('n_layer')} blocks, {mc.get('n_heads')} heads, "
                  f"feed-forward {mc.get('hidden_dim')}"),
    ]
    if mc.get("tokenizer") == "gbst":
        rows.append(("GBST", f"max block {mc.get('gbst_max_block_size')}, offsets {mc.get('gbst_offsets')}, "
                             f"pooling {mc.get('gbst_pooling')}, pointwise layer {mc.get('gbst_pointwise_conv', False)}, "
                             f"calibration {mc.get('gbst_score_calibration')}, downsampling {mc.get('downsample_rate')}"))
    if "qk_norm" in mc:
        rows.append(("Norms", f"{mc.get('norm_type')}, QKNorm {mc.get('qk_norm')}"))
    rows += [
        ("Steps", f"{t.get('max_steps'):,} (≈{t.get('max_steps', 0) * 1048576 / 1e9:.2f} B bases)"),
        ("Batch", f"{devices} GPUs × {micro} sequences × {accum} accumulation = "
                  f"{(devices or 0) * (micro or 0) * (accum or 0):,} sequences of {d.get('max_length')} bases"),
        ("Optimizer", f"{o.get('_name_')}, LR {o.get('lr')}, weight decay {o.get('weight_decay')}, betas {o.get('betas')}"),
        ("Schedule", f"{s.get('_name_')}, warmup {s.get('warmup_t')} steps, minimum LR {s.get('lr_min')}"),
        ("Masking", f"{d.get('mlm_strategy', 'mlm')}, probability {d.get('mlm_probability')}; strand flipping (RC aug) {d.get('rc_aug')}"),
        ("Precision, clipping", f"fp{t.get('precision')}, gradient clip {t.get('gradient_clip_val')}"),
        ("Validation", validation_text(t)),
        ("Seed", str(config.get("train", {}).get("seed"))),
    ]
    return rows


def validation_text(t):
    every, accum, limit = t.get("val_check_interval"), t.get("accumulate_grad_batches") or 1, t.get("limit_val_batches")
    when = f"every {every / accum:g} steps" if every else "once per epoch"
    part = f"{limit:g} batches" if isinstance(limit, (int, float)) and limit > 1 else f"{100 * limit:g}% of the batches"
    return f"{when}, on {part} of the validation and test sets"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def metrics_table(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    if "tokens" in rows[0]:
        return table(["Step", "Bases seen", "Validation loss", "Test loss"],
                     [(r["global_step"], f"{int(r['tokens']) / 1e9:.3f} B",
                       f"{float(r['val/loss']):.4f}" if r.get("val/loss") else "–",
                       f"{float(r['test/loss']):.4f}" if r.get("test/loss") else "–") for r in rows])
    cols = [c for c in rows[0] if c != "global_step"]
    return table(["Step"] + cols, [[r["global_step"]] + [r[c][:7] if r[c] else "–" for c in cols] for r in rows])


def link(path):
    rel = path.relative_to(PROJECT)
    return f"[`{rel}`](../{rel.as_posix()})"


def run_page(run):
    lines = [f"# {run['id']}: {run['title']}", "",
             table(["", ""], [
                 ("Status", STATUS_ICON.get(run["status"], run["status"])),
                 ("Kind", run["kind"]),
                 ("Started", run.get("started", "–")),
                 ("Ended", run.get("ended", "–")),
                 ("Model", run.get("model", "–")),
             ]).replace("|  |  |\n|---|---|\n", "| | |\n|---|---|\n"),
             "", f"**Purpose.** {run['purpose']}", "",
             "**Command.**", "", "```bash", run["command"], "```", ""]
    if run.get("outcome"):
        lines += [f"**Outcome.** {run['outcome']}", ""]
    if run.get("notes"):
        lines += [f"**Notes.** {run['notes']}", ""]

    files, found_any = [], False
    for rel in run.get("results", []):
        base = RESULTS / rel
        if not base.exists():
            continue
        found_any = True
        if (base / "config.json").exists():
            lines += ["## Settings (from the run's `config.json`)", "",
                      table(["Setting", "Value"], settings_rows(json.load(open(base / "config.json")))), ""]
        if (base / "metrics.csv").exists():
            t = metrics_table(base / "metrics.csv")
            if t:
                lines += ["## Validation during training (`metrics.csv`)", "",
                          "Loss in nats per hidden base; 1.386 = random guess among A/C/G/T.", "", t, ""]
        for md in sorted(base.glob("eval_*.md")):
            lines += [f"## Accuracy evaluation ({md.stem.split('_', 1)[1]} split)", "", md.read_text().strip(), ""]
        for summary in [base / "summary.txt"] if (base / "summary.txt").exists() else []:
            lines += ["## Summary file", "", "```", summary.read_text().strip(), "```", ""]
        files += sorted(p for p in base.rglob("*") if p.is_file())
    if files:
        lines += ["## Files", ""] + [f"- {link(p)}" for p in files] + [""]
    elif run["status"] in ("running", "queued") and not found_any:
        lines += ["*No result files yet; this page is regenerated when the run produces them.*", ""]
    lines += ["[← All runs](README.md)", ""]
    return "\n".join(lines)


def commits():
    out = subprocess.run(["git", "log", "--reverse", "--date=format:%Y-%m-%d %H:%M", "--format=%ad|%h|%s",
                          "origin/main..HEAD"], cwd=REPO, capture_output=True, text=True).stdout
    return [line.split("|", 2) for line in out.splitlines() if line]


def main():
    register = yaml.safe_load(open(RUNS_DIR / "runs.yaml"))
    runs = register["runs"]
    for old in RUNS_DIR.glob("R*.md"):
        old.unlink()
    for run in runs:
        (RUNS_DIR / f"{run['id']}.md").write_text(run_page(run))

    index = ["# All runs and steps of the project", "",
             "One page per run (training, checks, evaluations). Hand-written facts come from "
             "[`runs.yaml`](runs.yaml); settings and results are read from "
             "[`../results/`](../results/) by `scripts_charna/make_run_cards.py`. Times are CEST.", "",
             "## Runs", "",
             table(["Run", "What", "Status", "Started", "Outcome"],
                   [(f"[{r['id']}]({r['id']}.md)", r["title"], STATUS_ICON.get(r["status"], r["status"]),
                     r.get("started", "–"), r["outcome"].split(". ")[0].rstrip(".") + "." if r.get("outcome") else "–")
                    for r in runs]),
             "", "## Project log (steps that are not runs)", "",
             table(["When", "What"], [(e["when"], e["what"]) for e in register["log"]]),
             "", "## Code changes (git commits on this branch)", "",
             table(["When", "Commit", "Message"], [(w, f"`{h}`", m) for w, h, m in commits()]), ""]
    (RUNS_DIR / "README.md").write_text("\n".join(index))
    print(f"wrote {len(runs)} run pages and the index to {RUNS_DIR}")


if __name__ == "__main__":
    main()
