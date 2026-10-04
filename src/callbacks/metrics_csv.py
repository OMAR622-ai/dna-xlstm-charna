"""Append validation metrics to a CSV file (rank 0), independent of wandb.

Enable with `+callbacks.metrics_csv.filename=metrics.csv` (relative to the Hydra run dir).
Each row: global_step, tokens seen (global_step * tokens_per_step, if given), and every logged
metric whose name starts with one of `prefixes` (exact float values, unlike the progress bar).
Rows may carry different metrics (e.g. the final test pass of the NT benchmark logs test/*
while the epochs before it log val/*); the header is the union of all columns seen so far.
"""

import csv
import os

import pytorch_lightning as pl


class MetricsCSV(pl.Callback):

    def __init__(self, filename="metrics.csv", tokens_per_step=None, prefixes=("val/", "test/")):
        super().__init__()
        self.filename = filename
        self.tokens_per_step = tokens_per_step
        self.prefixes = tuple(prefixes)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_rank != 0:
            return
        metrics = {
            k: float(v) for k, v in trainer.callback_metrics.items()
            if k.startswith(self.prefixes) and getattr(v, "numel", lambda: 1)() == 1
        }
        row = {"global_step": trainer.global_step}
        if self.tokens_per_step is not None:
            row["tokens"] = trainer.global_step * int(self.tokens_per_step)
        row.update(sorted(metrics.items()))
        append_row(self.filename, row)


def append_row(filename, row):
    """Append `row` to the CSV; rewrite the (small) file if `row` adds new columns."""
    rows, fieldnames = [], []
    if os.path.exists(filename):
        with open(filename, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    new_columns = [k for k in row if k not in fieldnames]
    if not new_columns and fieldnames:
        with open(filename, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        return
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + new_columns)
        writer.writeheader()
        writer.writerows(rows + [row])
