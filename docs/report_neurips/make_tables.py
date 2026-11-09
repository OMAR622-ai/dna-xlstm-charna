"""Writes the LaTeX tables of the report from the CSV files made by the notebook (notebooks/outputs/).

    python docs/report_neurips/make_tables.py
"""
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent.parent / "notebooks" / "outputs"
TAB = HERE / "tables"
TAB.mkdir(exist_ok=True)
esc = lambda s: s.replace("_", r"\_")


def cell(mean, spread, bold):
    txt = f"{mean:.3f}" + r"{\scriptsize$\,\pm\,$" + f"{spread:.3f}" + "}"
    return r"\textbf{" + txt + "}" if bold else txt


# Table: NT benchmark, all seven models
m = pd.read_csv(OUT / "nt_means.csv", index_col=0)
r = pd.read_csv(OUT / "nt_ranges.csv", index_col=0)
groups = [("Histone marks (MCC)", list(m.index[:10])), ("Enhancers (MCC)", list(m.index[10:12])),
          ("Promoters (F1)", list(m.index[12:15])), ("Splice sites (accuracy / F1)", list(m.index[15:]))]
models = list(m.columns)
wins = {c: 0 for c in models}
lines = []
for title, tasks in groups:
    lines.append(r"\multicolumn{8}{l}{\emph{" + title + r"}} \\")
    for t in tasks:
        best = m.loc[t].max()
        cells = []
        for c in models:
            b = abs(m.loc[t, c] - best) < 5e-4
            wins[c] += b
            cells.append(cell(m.loc[t, c], r.loc[t, c], b))
        lines.append(esc(t) + " & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
lines.append("Tasks with the best mean & " + " & ".join(str(int(wins[c])) for c in models) + r" \\")
head = "Task & " + " & ".join(models) + r" \\"
(TAB / "nt_all.tex").write_text("\n".join([r"\begin{tabular}{l" + "c" * 7 + "}", r"\toprule", head, r"\midrule"] + lines
                                          + [r"\bottomrule", r"\end{tabular}"]) + "\n")
print("NT wins", wins)

# Table: enhancer diagnosis
g = pd.read_csv(OUT / "validation_test_gap.csv", index_col=0)
b = pd.read_csv(OUT / "kmer_baseline.csv", index_col=0)
lines = []
for t in b.index:
    has_gap = t in g.index
    lines.append(" & ".join([esc(t), f"{g.loc[t, 'GBST-BERT validation']:.3f}", f"{g.loc[t, 'GBST-BERT test']:.3f}",
                             f"{g.loc[t, 'Transformer++ validation']:.3f}", f"{g.loc[t, 'Transformer++ test']:.3f}",
                             f"{b.loc[t, '4-mer MCC, validation']:.3f}", f"{b.loc[t, '4-mer MCC, test']:.3f}",
                             f"{b.loc[t, '4-mer MCC, test without copies']:.3f}",
                             f"{int(b.loc[t, 'test copies in train'])}/{int(b.loc[t, 'test sequences'])}"]) + r" \\")
head = (r"& \multicolumn{2}{c}{GBST-BERT} & \multicolumn{2}{c}{Transformer++} & \multicolumn{3}{c}{4-mer logistic regression} & Test seqs. \\"
        "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}" + "\n"
        r"Task & val. & test & val. & test & val. & test & test, no copies & in train \\")
(TAB / "enhancers.tex").write_text("\n".join([r"\begin{tabular}{lcccccccc}", r"\toprule", head, r"\midrule"] + lines
                                             + [r"\bottomrule", r"\end{tabular}"]) + "\n")
print(b.T)

# Tables: the tasks (NT sizes and settings from the notebook's output)
import json
import re

nb = json.load(open(HERE.parent.parent / "notebooks" / "GBST_BERT_full_pipeline.ipynb"))
html = [ "".join(o["data"]["text/html"]) for c in nb["cells"] if c["cell_type"] == "code" for o in c["outputs"]
         if "The 18 tasks" in "".join(o.get("data", {}).get("text/html", []))][0]
text = re.sub(r"\s+", " ", re.sub(r"<style.*?</style>|<[^>]+>", " ", html, flags=re.S))
rows = re.findall(r"(\S+) (histone mark|enhancer|promoter|splice site) (\d) (\d+) ([\d,]+) ([\d,]+) (MCC|F1|accuracy) (\S+) (\d+)", text)
assert len(rows) == 18
lines = [" & ".join([esc(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]]) + r" \\" for r in rows]
(TAB / "nt_tasks.tex").write_text("\n".join(
    [r"\begin{tabular}{llcrrrlcc}", r"\toprule",
     r"Task & Group & Classes & Length & Train & Test & Score & LR & Batch \\", r"\midrule"] + lines
    + [r"\bottomrule", r"\end{tabular}"]) + "\n")
