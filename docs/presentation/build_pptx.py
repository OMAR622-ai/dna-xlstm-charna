"""Fill the JKU DLfMI template with the GBST-BERT project and extend it."""
import copy
import csv
import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

W = pathlib.Path("/home/ubuntu/practical work/test 1/files")
FIG = W / "notebook_outputs"
TEMPLATE = W / "DLfMI_presentation_template.pptx"
OUT = W / "GBST_BERT_DLfMI_presentation.pptx"

JKU = RGBColor(0x00, 0x84, 0xBB)      # template title blue
GBST = RGBColor(0x2A, 0x78, 0xD6)     # GBST-BERT in all figures
TPP = RGBColor(0xC9, 0x4F, 0x1C)      # Transformer++ (figure orange, darkened for text contrast)
INK = RGBColor(0x1A, 0x1A, 0x1A)
SOFT = RGBColor(0x4A, 0x4A, 0x4A)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
CARD = RGBColor(0xEE, 0xF4, 0xF9)
BEST = RGBColor(0xD3, 0xE6, 0xF7)
LINE = RGBColor(0xD0, 0xD5, 0xDB)

prs = Presentation(TEMPLATE)
title_slide, problem, method, results = list(prs.slides)
CONTENT_LAYOUT = problem.slide_layout
SLDNUM = next(sp for sp in problem.shapes._spTree if sp.tag == qn("p:sp")
              and sp.find(".//" + qn("p:ph")) is not None and sp.find(".//" + qn("p:ph")).get("type") == "sldNum")


# ------------------------------------------------------------------------------ helpers
def ph(slide, idx):
    return next(s for s in slide.placeholders if s.placeholder_format.idx == idx)


def drop(shape):
    shape._element.getparent().remove(shape._element)


def new_slide(title):
    s = prs.slides.add_slide(CONTENT_LAYOUT)
    s.shapes.title.text = title
    s.shapes._spTree.append(copy.deepcopy(SLDNUM))
    for idx in (1, 2):
        try:
            drop(ph(s, idx))
        except StopIteration:
            pass
    return s


def fill_bullets(shape, items, size=18, space=8):
    tf = shape.text_frame
    first = tf.paragraphs[0]
    ppr = copy.deepcopy(first._p.find(qn("a:pPr")))
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for i, item in enumerate(items):
        p = first if i == 0 else tf.add_paragraph()
        if i and ppr is not None:
            p._p.insert(0, copy.deepcopy(ppr))
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        parts = item if isinstance(item, list) else [(item, False)]
        for text, bold in parts:
            run = p.add_run()
            run.text = text
            run.font.size = Pt(size)
            run.font.bold = bold
        p.space_before = Pt(space)


def text(slide, x, y, w, h, runs, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.1, name=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paragraphs = runs if isinstance(runs, list) and runs and isinstance(runs[0], list) else [runs]
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        if i:
            p.space_before = Pt(6)
        for part in (para if isinstance(para, list) else [para]):
            t, opts = (part, {}) if isinstance(part, str) else part
            r = p.add_run()
            r.text = t
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", bold)
            r.font.italic = opts.get("italic", False)
            r.font.color.rgb = opts.get("color", color)
            r.font.name = "Arial"
    if name:
        tb.name = name
    return tb


def card(slide, x, y, w, h, fill=CARD, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.5)
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                    Inches(w) if w else None, Inches(h) if h else None)


def stat(slide, x, y, w, number, label, color=GBST):
    text(slide, x, y, w, 0.75, number, size=40, bold=True, color=color)
    text(slide, x, y + 0.78, w, 0.8, label, size=14, color=SOFT)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def footnote(slide, t):
    text(slide, 0.53, 6.3, 12.1, 0.3, t, size=11, color=MUTED)


# ------------------------------------------------------------------------------ 1 title
tf = title_slide.shapes.title.text_frame
tf.paragraphs[0].runs[0].text = "Deep Learning for Medical Imaging"
tf.paragraphs[1].runs[0].text = "GBST-BERT: does a learned tokenizer help a small DNA language model?"
sub = ph(title_slide, 1).text_frame
sub.paragraphs[0].runs[0].text = "Practical work: pretraining on the human genome (hg38) and 18 prediction tasks"
sub.paragraphs[1].runs[0].text = "Presented by Omar Mohamed, K12103822"
notes(title_slide, "One sentence: I built a small DNA language model that learns how to cut DNA into pieces by "
      "itself (GBST), trained it on the human genome, and tested whether this helps on 18 biology tasks.")

# ------------------------------------------------------------------------------ 2 problem overview (template)
body = ph(problem, 1)
body.width = Inches(6.9)
fill_bullets(body, [
    [("Problem: ", True), ("a DNA language model must first cut DNA into tokens – single bases or fixed k-mers. "
                          "A fixed cut can split a motif in the wrong place.", False)],
    [("Question: ", True), ("does a learned tokenizer (GBST, from ChaRNABERT) help a small model learn the human "
                           "genome and predict biology?", False)],
    [("Relevance: ", True), ("models that read DNA predict promoters, enhancers, splice sites and histone marks "
                            "from sequence alone – a basis for interpreting genomes and variants.", False)],
    [("State of the art: ", True), ("Bio-xLSTM (2024) compares small DNA models – HyenaDNA, Mamba, xLSTM – "
                                   "on the 18-task Nucleotide Transformer benchmark.", False)],
], size=17, space=10)
ph(problem, 2).text_frame.paragraphs[0].runs[0].text = \
    "ChaRNABERT: Morales-Pastor et al. 2024, arXiv:2411.11808 · Bio-xLSTM: Schmidinger et al. 2024, arXiv:2411.04165"
for p in ph(problem, 2).text_frame.paragraphs:
    for r in p.runs:
        r.font.size = Pt(11)
        r.font.color.rgb = MUTED
x0 = 7.75
text(problem, x0, 1.8, 4.9, 0.4, "Three ways to cut the same DNA", size=15, bold=True, color=INK)
rows = [("Single bases", "T A T A A A G G C", "fine detail, no grouping", CARD, None),
        ("Fixed 3-mers", "TAT | AAA | GGC", "can split a motif (TATA box)", CARD, None),
        ("GBST (learned)", "TATAAA | G | GC", "block sizes 1–4 learned (illustration)", RGBColor(0xDC, 0xEB, 0xF8), GBST)]
for i, (label, seq, note, fill, line) in enumerate(rows):
    y = 2.3 + i * 1.3
    card(problem, x0, y, 4.9, 1.12, fill=fill, line=line)
    text(problem, x0 + 0.2, y + 0.12, 4.5, 0.3, label, size=14, bold=True, color=GBST if line else INK)
    text(problem, x0 + 0.2, y + 0.43, 4.5, 0.32, seq, size=17, color=INK)
    text(problem, x0 + 0.2, y + 0.8, 4.5, 0.26, note, size=12, color=SOFT)
notes(problem, "Tokenization is the first step of every language model. For DNA people use single bases or fixed "
      "k-mers; fixed k-mers can cut a motif like the TATA box in the wrong place. GBST looks at blocks of 1 to 4 "
      "bases at every position and learns how much to trust each block size.")

# ------------------------------------------------------------------------------ 3 method interpretation (template)
method.shapes.title.text_frame.paragraphs[0].runs[0].text = "Method Interpretation"
drop(ph(method, 1))
for idx in (2,):
    try:
        drop(ph(method, idx))
    except StopIteration:
        pass
picture(method, W / "model_visualizations/1_architecture.png", 1.6, 1.55, w=10.1)
text(method, 0.53, 5.98, 12.1, 0.45,
     [("How it works: ", {"bold": True}), "GBST mixes each base with its neighbours, scores candidate blocks of 1–4 bases "
      "and returns a weighted mix – one vector per base – which a bidirectional BERT encoder reads. "],
     size=14, color=INK)
notes(method, "Read the figure left to right. 1,024 bases go in, 15% are hidden. GBST: embedding, a 4-base "
      "convolution, 10 candidate blocks, learned scores, weighted mix (70,144 parameters). Then 4 encoder blocks "
      "with attention in both directions, RoPE and QKNorm, SwiGLU. The head guesses the hidden bases. "
      "2,107,392 parameters in total.")

# ------------------------------------------------------------------------------ 4 why this approach (new)
why = new_slide("Why this approach")
cards = [("Learned, not fixed", "GBST decides from the data how to group bases, while every base keeps its own "
          "position – a single-base change (SNP) still changes the input."),
         ("A clean comparison", "Control model Transformer++: the 2 M Transformer of Bio-xLSTM, same size, same "
          "training, plain lookup table instead of GBST."),
         ("A fair benchmark", "Pretraining and fine-tuning follow the Bio-xLSTM protocol exactly, so our numbers "
          "sit next to the published HyenaDNA, Mamba and xLSTM results.")]
for i, (head, body_t) in enumerate(cards):
    x = 0.53 + i * 4.1
    card(why, x, 1.8, 3.85, 3.0)
    text(why, x + 0.3, 2.05, 3.3, 0.4, f"{i + 1}", size=32, bold=True, color=GBST)
    text(why, x + 0.3, 2.75, 3.3, 0.4, head, size=18, bold=True, color=INK)
    text(why, x + 0.3, 3.25, 3.3, 1.5, body_t, size=14, color=SOFT, spacing=1.15)
stat(why, 0.53, 5.05, 3.8, "2,107,392", "parameters in GBST-BERT")
stat(why, 4.63, 5.05, 3.8, "70,144", "of them in GBST (3%)")
stat(why, 8.73, 5.05, 3.8, "2,036,992", "parameters in Transformer++", color=TPP)
notes(why, "Three reasons for the design. The only real difference between the two models is the tokenizer "
      "(plus small details like the normalisation type), so differences in results point at GBST.")

# ------------------------------------------------------------------------------ 5 data (new)
data = new_slide("Data: the human genome hg38")
picture(data, FIG / "data_chromosomes.png", 0.6, 1.6, w=11.9)
for i, (num, lab) in enumerate([("3.03 billion", "bases, chromosomes 1–22 and X (4.0% unknown, N)"),
                                ("2.33 billion", "distinct training bases (34,021 regions)"),
                                ("0.29 / 0.27", "billion validation / test bases – never seen in training")]):
    stat(data, 0.53 + i * 4.1, 5.3, 3.9, num, lab, color=GBST if i else INK)
notes(data, "The genome as prepared by the DNA-xLSTM repository. A BED file assigns regions to train, validation "
      "and test; each region is widened to 2^20 bases and cut into 1,024-base examples. Splits never overlap.")

# ------------------------------------------------------------------------------ 6 pretraining (new)
pre = new_slide("Pretraining: identical protocol for both models")
rows = [("Task", "masked language modelling: hide 15% of bases, guess them"),
        ("Batch", "1,024 sequences × 1,024 bases per step"),
        ("Length", "10,000 steps = 10.5 billion bases per model"),
        ("Optimiser", "AdamW (0.9, 0.95), weight decay 0.1, cosine schedule"),
        ("Learning rate", "GBST-BERT 2e-3 (checked: 5e-4 … 5e-3); Transformer++ 1e-2"),
        ("Hardware", "4 GPUs (JKU student servers), 32-bit precision")]
tbl = pre.shapes.add_table(len(rows), 2, Inches(0.53), Inches(1.8), Inches(12.1), Inches(0.58 * len(rows))).table
tbl.columns[0].width = Inches(2.6)
tbl.columns[1].width = Inches(9.5)
for r, (k, v) in enumerate(rows):
    for c, val in enumerate((k, v)):
        cell = tbl.cell(r, c)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = CARD if r % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
        cell.margin_left = Inches(0.15)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        for p in cell.text_frame.paragraphs:
            for run in p.runs:
                run.font.size = Pt(18)
                run.font.bold = c == 0
                run.font.color.rgb = INK
                run.font.name = "Arial"
tbl.first_row = False
text(pre, 0.53, 5.55, 12.1, 0.7,
     [[("Same data, same masks, same schedule: ", {"bold": True}),
       "only the tokenizer differs. Settings from Bio-xLSTM App. C.5 and Tables A3–A4; the learning rate of "
       "GBST-BERT was chosen by a 500-step check, Transformer++ uses the paper's value."]], size=15, color=SOFT)
notes(pre, "The pretraining protocol is the short-context protocol of Bio-xLSTM. Each model saw exactly 10.5 "
      "billion bases. Only GBST-BERT's learning rate was tuned.")

# ------------------------------------------------------------------------------ 7 pretraining results (new)
pr = new_slide("Pretraining result: almost level at the end")
picture(pr, FIG / "pretraining_curves.png", 0.45, 1.75, w=8.7)
x = 9.55
stat(pr, x, 1.8, 3.2, "56.44%", "GBST-BERT: hidden bases guessed right (2,048 test sequences)")
stat(pr, x, 3.25, 3.2, "56.33%", "Transformer++", color=TPP)
stat(pr, x, 4.4, 3.2, "25%", "random guessing", color=MUTED)
text(pr, 0.53, 5.4, 8.6, 0.8,
     [[("Left: ", {"bold": True}), "500-step learning-rate check – 2e-3 lowest loss. ",
       ("Right: ", {"bold": True}), "full runs – GBST-BERT ahead from 4 to 8 billion bases, final validation loss "
       "0.9728 vs 0.9755."]], size=14, color=SOFT)
footnote(pr, "A 0.1-point gap from one pretraining run per model is too small to call. Test loss 0.9753 vs 0.9774.")
notes(pr, "Loss = how surprised the model is by the true base; lower is better, 1.386 is random. Both models reach "
      "56% accuracy, more than twice chance. On this measure alone we cannot say GBST helps.")

# ------------------------------------------------------------------------------ 8 what GBST learned (new)
gl = new_slide("Inside the model: GBST chose single bases")
picture(gl, FIG / "gbst_block_sizes.png", 0.5, 1.75, w=7.6)
text(gl, 8.5, 1.85, 4.2, 3.6,
     [[("98%", {"size": 40, "bold": True, "color": GBST})],
      [("of GBST's weight is on single-base blocks in the final model", {"size": 15, "color": SOFT})],
      [("Early in training (LR 5e-4) it still used longer blocks: 72 / 22 / 5 / 1%.",
        {"size": 15, "color": SOFT})],
      [("Each base still sees its 3 neighbours through the convolution before the blocks.",
        {"size": 15, "color": SOFT})]], spacing=1.15)
text(gl, 0.53, 4.4, 7.6, 1.6,
     [[("Interpretation: ", {"bold": True}), "on human DNA, GBST does not learn longer “words”. It behaves like a "
       "single-base model with a small local window in front – a likely reason why the pretraining losses end so "
       "close."]], size=15, color=INK)
footnote(gl, "Average over 8 hg38 sequences from different chromosomes, 15% of bases hidden.")
notes(gl, "GBST weights per block size. The original idea of learning multi-base words is not what happens on "
      "human DNA: the model prefers single bases.")

# ------------------------------------------------------------------------------ 9 confidence (new)
cf = new_slide("Inside the model: it learned real DNA structure")
picture(cf, FIG / "confidence_chr21.png", 0.45, 1.9, w=9.3)
for i, (num, lab, col) in enumerate([("94.5%", "of hidden bases right inside tandem repeats", GBST),
                                     ("50.8%", "right in the rest of the window", INK),
                                     ("25%", "random guessing", MUTED)]):
    stat(cf, 10.0, 1.85 + i * 1.45, 2.75, num, lab, color=col)
text(cf, 0.53, 5.5, 9.2, 0.6, [[("chr21:33,045,057–33,046,080, ", {"bold": True}),
     "every base hidden once. Shaded: tandem repeats such as (GAAA)n."]], size=14, color=SOFT)
notes(cf, "Every base of a 1,024-base window of chromosome 21 was hidden once. Inside tandem repeats such as "
      "(GAAA)n the model predicts almost every base; in unique sequence it is much less sure.")

# ------------------------------------------------------------------------------ 10 benchmark setup (new)
bm = new_slide("Prediction tasks: the Nucleotide Transformer benchmark")
groups = [("10", "Histone marks", "Is this DNA on a nucleosome with a given mark (H3K4me3, …)? Score: MCC"),
          ("2", "Enhancers", "Is it an enhancer, and which type? Score: MCC"),
          ("3", "Promoters", "Is it a promoter (all, with / without TATA box)? Score: F1"),
          ("3", "Splice sites", "Donor, acceptor or neither? Score: F1 / accuracy")]
for i, (n, head, desc) in enumerate(groups):
    x = 0.53 + i * 3.07
    card(bm, x, 1.8, 2.87, 2.55)
    text(bm, x + 0.25, 1.95, 2.4, 0.7, n, size=36, bold=True, color=GBST)
    text(bm, x + 0.25, 2.7, 2.4, 0.4, head, size=17, bold=True, color=INK)
    text(bm, x + 0.25, 3.15, 2.4, 1.1, desc, size=13, color=SOFT, spacing=1.12)
fill = [("Same protocol as Bio-xLSTM: ", {"bold": True}),
        "pretrained model + mean pooling + linear layer, whole model fine-tuned for 20 epochs, both DNA strands "
        "averaged, best validation epoch tested, ", ("10 repeats per task", {"bold": True}),
        " (360 runs). Per-task learning rate and batch from the paper (Table A5); on splice acceptors / donors "
        "these made GBST-BERT collapse, so 1e-3 and 5e-4 were also tried and chosen by validation score."]
text(bm, 0.53, 4.6, 12.1, 1.4, [fill], size=15, color=INK, spacing=1.15)
footnote(bm, "MCC: 0 = guessing, 1 = perfect. Data: InstaDeepAI/nucleotide_transformer_downstream_tasks.")
notes(bm, "The standard downstream test of the Bio-xLSTM paper. Each repeat uses a new split of training and "
      "validation data, so the range over repeats shows how stable a result is. The test set never chooses "
      "anything.")

# ------------------------------------------------------------------------------ 11 results figure (new)
rf = new_slide("Results: GBST-BERT is furthest right on most tasks")
picture(rf, FIG / "nt_all_tasks.png", 0.45, 1.6, h=4.6)
x = 8.45
text(rf, x, 1.75, 4.25, 4.3,
     [[("How to read it", {"bold": True, "size": 16, "color": INK})],
      ["One row per task; further right is better."],
      ["Dot = mean of 10 repeats; line = worst to best repeat."],
      [("Blue", {"bold": True, "color": GBST}), " GBST-BERT, ", ("orange", {"bold": True, "color": TPP}),
       " Transformer++, green: best model of the paper."],
      [("Strongest: ", {"bold": True}), "splice sites and histone marks."],
      [("Weakest: ", {"bold": True}), "enhancers, for both of our models."]], size=14, color=SOFT, spacing=1.15)
notes(rf, "Example row H3K14ac: GBST-BERT 0.673, best paper model Mamba-PH 0.631, Transformer++ 0.569, with a short "
      "range. Splice sites bottom right: GBST-BERT best with short lines. Enhancers bottom left: both our models "
      "behind.")

# ------------------------------------------------------------------------------ 12-13 tables (new)
means = {r["task"] if "task" in r else r[""]: r for r in csv.DictReader(open(FIG / "nt_means.csv"))}
ranges = {r["task"] if "task" in r else r[""]: r for r in csv.DictReader(open(FIG / "nt_ranges.csv"))}
MODELS = ["HyenaDNA", "Mamba-PS", "Mamba-PH", "xLSTM-PS", "xLSTM-PH", "GBST-BERT", "Transformer++"]
TASKS = list(means)
wins = {m: 0 for m in MODELS}
for t in TASKS:
    best = max(float(means[t][m]) for m in MODELS)
    for m in MODELS:
        wins[m] += abs(float(means[t][m]) - best) < 5e-4


def results_table(slide, tasks, with_wins):
    n_rows = len(tasks) + 1 + with_wins
    top, row_h = 1.75, 0.405
    shape = slide.shapes.add_table(n_rows, 8, Inches(0.53), Inches(top), Inches(12.1), Inches(row_h * n_rows))
    t = shape.table
    t.columns[0].width = Inches(2.3)
    for c in range(1, 8):
        t.columns[c].width = Inches(1.4)
    header = ["Task"] + MODELS
    for c, h in enumerate(header):
        cell = t.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = JKU if c < 6 else (GBST if c == 6 else TPP)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size, r.font.bold, r.font.name = Pt(12), True, "Arial"
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for i, task in enumerate(tasks, start=1):
        best = max(float(means[task][m]) for m in MODELS)
        t.cell(i, 0).text = task
        for c, m in enumerate(MODELS, start=1):
            cell = t.cell(i, c)
            v, rg = float(means[task][m]), float(ranges[task][m])
            is_best = abs(v - best) < 5e-4
            tf = cell.text_frame
            tf.text = ""
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r1 = p.add_run(); r1.text = f"{v:.3f}"
            r1.font.size, r1.font.bold, r1.font.name = Pt(13), is_best, "Arial"
            r1.font.color.rgb = INK
            r2 = p.add_run(); r2.text = f" ±{rg:.3f}"
            r2.font.size, r2.font.name = Pt(9), "Arial"
            r2.font.color.rgb = MUTED
            cell.fill.solid()
            cell.fill.fore_color.rgb = BEST if is_best else (RGBColor(0xF6, 0xF8, 0xFA) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF))
        cell = t.cell(i, 0)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xF6, 0xF8, 0xFA) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF)
        for r in cell.text_frame.paragraphs[0].runs:
            r.font.size, r.font.name = Pt(13), "Arial"
            r.font.color.rgb = INK
    if with_wins:
        last = n_rows - 1
        t.cell(last, 0).text = "Best on (of 18)"
        for c, m in enumerate(MODELS, start=1):
            t.cell(last, c).text = str(wins[m])
        for c in range(8):
            cell = t.cell(last, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xDC, 0xEB, 0xF8)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size, r.font.bold, r.font.name = Pt(14), True, "Arial"
                r.font.color.rgb = GBST if c == 6 else INK
    for row in t.rows:
        row.height = Inches(row_h)
    for r_ in range(n_rows):
        for c in range(8):
            cell = t.cell(r_, c)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
    return t


t1 = new_slide("All models: histone marks (MCC)")
results_table(t1, TASKS[:10], False)
footnote(t1, "Mean over 10 repeats ± range (best − worst). Highlighted: best mean (ties marked for each). "
         "Columns 1–5: Bio-xLSTM Table 1.")
notes(t1, "GBST-BERT is best on 8 of 10 histone tasks, one a three-way tie (H3K79me3). xLSTM-PH is best on H3 and "
      "H4. Compare the last two columns: GBST-BERT is higher than Transformer++ on all 10.")
t2 = new_slide("All models: enhancers, promoters and splice sites")
results_table(t2, TASKS[10:], True)
footnote(t2, "MCC for enhancers, F1 for promoters and splice acceptors / donors, accuracy for splice_sites_all. "
         "Last row: all 18 tasks, ties counted for each model.")
notes(t2, "Enhancers: the paper's models are clearly better. Promoters: everybody near 0.95–0.97, GBST-BERT best on "
      "two of three. Splice sites: GBST-BERT best on all three. Overall GBST-BERT has the best mean on 13 of 18.")

# ------------------------------------------------------------------------------ 14 differences figure (new)
df = new_slide("The GBST effect, task by task")
picture(df, FIG / "nt_differences.png", 0.45, 1.6, h=4.6)
text(df, 8.75, 1.8, 3.95, 4.2,
     [[("16 / 18", {"size": 40, "bold": True, "color": GBST})],
      [("tasks where GBST-BERT beats the same model without GBST", {"size": 15, "color": SOFT})],
      [("12 + 1 tie", {"size": 40, "bold": True, "color": GBST})],
      [("tasks where it beats or ties the best published model", {"size": 15, "color": SOFT})],
      [("Biggest gains: H3K4me3 +0.19, H3K4me2 +0.14. Biggest loss: enhancers −0.21 vs the paper.",
        {"size": 13, "color": MUTED})]], spacing=1.1)
notes(df, "Each bar is a difference in mean test score; right of zero means GBST-BERT is better. Left panel: the "
      "effect of GBST. Right panel: the comparison with the best published model per task.")

# ------------------------------------------------------------------------------ 15 results interpretation (template)
results.shapes.title.text_frame.paragraphs[0].runs[0].text = "Results Interpretation"
rb = ph(results, 1)
rb.width = Inches(7.4)
fill_bullets(rb, [
    [("Pretraining: ", True), ("both models learned human DNA (56% vs 25% chance); the 0.1-point gap is too small "
                              "to call.", False)],
    [("Prediction: ", True), ("the difference is clear – GBST-BERT beats Transformer++ on 16 of 18 tasks and has "
                             "the best mean of all seven models on 13 of 18.", False)],
    [("Why? ", True), ("GBST chose single bases (98%), so the gain likely comes from its local window and mixing "
                      "layer – not yet proven, the models also differ in normalisation.", False)],
    [("Limits: ", True), ("enhancers are weak for both models; some repeats trained poorly (wide ranges); "
                         "fine-tuning settings were the paper's, not tuned for our models.", False)],
], size=16, space=10)
try:
    drop(ph(results, 2))
except StopIteration:
    pass
for i, (num, lab) in enumerate([("16 / 18", "better than the same model without GBST"),
                                ("13 / 18", "best mean of 7 models (xLSTM-PH: 4)"),
                                ("0.980", "splice donor F1 (best paper model 0.962)")]):
    card(results, 8.35, 1.8 + i * 1.45, 4.3, 1.3)
    text(results, 8.6, 1.92 + i * 1.45, 3.9, 0.6, num, size=32, bold=True, color=GBST)
    text(results, 8.6, 2.55 + i * 1.45, 3.9, 0.5, lab, size=13, color=SOFT)
notes(results, "Main message: pretraining loss could not separate the two models, but the prediction tasks can. "
      "Be honest about what is not shown yet: which part of GBST causes the gain.")

# ------------------------------------------------------------------------------ 16 strengths / weaknesses (new)
sw = new_slide("GBST-BERT: advantages and disadvantages")
cols = [("Advantages", GBST, ["Better than the same model without GBST on 16 of 18 tasks",
                              "Best mean of 7 models on 13 of 18, all 3 splice tasks",
                              "Keeps single-base resolution: a SNP still changes the input",
                              "Small cost: 70 k parameters, 3% of the model",
                              "No tokenizer to choose by hand; weights can be inspected"]),
        ("Disadvantages", TPP, ["About 18% slower to fine-tune (962 s vs 816 s on H3)",
                                "Less stable: collapsed at high learning rates on splice tasks",
                                "On DNA it chose single bases – no longer “words” learned",
                                "Weak on enhancers, like our other model",
                                "Almost no gain in pretraining loss (0.1 points)"])]
for i, (head, col, items) in enumerate(cols):
    x = 0.53 + i * 6.15
    card(sw, x, 1.8, 5.95, 4.25)
    text(sw, x + 0.3, 1.98, 5.4, 0.45, head, size=20, bold=True, color=col)
    tb = text(sw, x + 0.3, 2.6, 5.4, 3.3, [[f"{it}"] for it in items], size=15, color=INK, spacing=1.1)
    for p in tb.text_frame.paragraphs:
        p.space_before = Pt(9)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(Emu(Inches(0.25))))
        pPr.set("indent", str(-Emu(Inches(0.25))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
notes(sw, "The advantages are measured with 10 repeats per task. The fine-tuning time is from one timing run per "
      "model on task H3.")

# ------------------------------------------------------------------------------ 17 what was challenging (new)
ch = new_slide("What was challenging")
items = [("Genome reader bug", "Parallel data readers shared one file and read wrong bases (up to 33%). Found "
          "after a crash, measured, fixed with a test."),
         ("Benchmark loader bug", "Every task silently loaded the data of all 18 tasks. Fixed: one file per task."),
         ("Unstable fine-tuning", "The paper's learning rates collapsed GBST-BERT on splice tasks. Re-chosen by "
          "validation score only."),
         ("Long runs on shared GPUs", "10.5 billion bases per model, 384 fine-tuning runs; a full disk and a server "
          "error stopped the queue – fixed with checks and retries.")]
for i, (head, body_t) in enumerate(items):
    x = 0.53 + (i % 2) * 6.15
    y = 1.8 + (i // 2) * 2.25
    card(ch, x, y, 5.95, 2.05)
    text(ch, x + 0.3, y + 0.25, 5.4, 0.4, head, size=20, bold=True, color=INK)
    text(ch, x + 0.3, y + 0.8, 5.4, 1.15, body_t, size=16, color=SOFT, spacing=1.12)
footnote(ch, "The first two were bugs in the original repository; every fix is documented on GitHub.")
notes(ch, "The genome-reader bug: 0.35% wrong sequences with 2 workers, up to 33% with 12; wrong reads stayed in "
      "their own split, so no test DNA leaked. A Hugging Face server error made 9 jobs fail silently; they were "
      "re-run.")

# ------------------------------------------------------------------------------ 18 conclusion (new)
cc = new_slide("Conclusion and next steps")
text(cc, 0.53, 1.8, 12.1, 1.2,
     [[("With 3% more parameters, a learned tokenizer made a small DNA model better at 16 of 18 biology tasks",
        {"bold": True, "color": JKU})],
      [("– strongest on splice sites and histone marks, weak on enhancers.", {"color": INK})]],
     size=24, spacing=1.1)
steps = [("1", "Clean GBST test", "Same model, GBST replaced by a lookup table – proves the gain comes from GBST"),
         ("2", "Tune per model", "Learning-rate search for fine-tuning, as the paper did; fixes the wide ranges"),
         ("3", "Longer DNA context", "Longer pretraining or longer pieces, to address enhancers")]
for i, (n, head, body_t) in enumerate(steps):
    x = 0.53 + i * 4.1
    card(cc, x, 3.3, 3.85, 2.8)
    text(cc, x + 0.3, 3.5, 3.3, 0.6, n, size=32, bold=True, color=GBST)
    text(cc, x + 0.3, 4.2, 3.3, 0.4, head, size=19, bold=True, color=INK)
    text(cc, x + 0.3, 4.75, 3.3, 1.3, body_t, size=16, color=SOFT, spacing=1.12)
footnote(cc, "Code, models, notebook and results: github.com/OMAR622-ai/dna-xlstm-charna (branch charna-gbst)")
notes(cc, "End on the one-sentence message. The most valuable next experiment is the clean test with everything "
      "identical except GBST.")

# ------------------------------------------------------------------------------ order
order = [title_slide, problem, method, why, data, pre, pr, gl, cf, bm, rf, t1, t2, df, results, sw, ch, cc]
sld_ids = prs.slides._sldIdLst
by_slide = {id(s): el for s, el in zip(prs.slides, list(sld_ids))}
elements = [by_slide[id(s)] for s in order]
for el in list(sld_ids):
    sld_ids.remove(el)
for el in elements:
    sld_ids.append(el)
prs.save(OUT)
print("saved", OUT, len(order), "slides")
