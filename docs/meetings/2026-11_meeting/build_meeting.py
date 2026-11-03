"""Fill the JKU 'Progress Meeting Template' with Recap / Progress / Next steps for the next meeting."""
import copy
import sys

from pptx import Presentation
from pptx.util import Emu, Pt

SRC, OUT, FIG, TABLE = sys.argv[1:5]
p = Presentation(SRC)
title, recap, progress, nxt = p.slides

# title slide: same line breaks as the template, only the date and name change
runs = [r for r in title.shapes[1].text_frame.paragraphs[0].runs]
runs[1].text = "November 2026"
title.shapes[0].text_frame.paragraphs[0].runs[0].text = "Omar Mohamed – Practical Work (GBST-BERT)"


# bullet styles of the template's Progress slide: "-" at level 0 and level 1
TPL = {0: copy.deepcopy(progress.shapes[1].text_frame.paragraphs[0]._p),
       1: copy.deepcopy(progress.shapes[1].text_frame.paragraphs[2]._p)}
for el in TPL.values():
    for r in el.findall("{http://schemas.openxmlformats.org/drawingml/2006/main}r")[1:]:
        el.remove(r)


def clone(slide):
    new = p.slides.add_slide(slide.slide_layout)
    for sh in list(new.shapes):
        sh._element.getparent().remove(sh._element)
    for el in slide.shapes._spTree.iterchildren():
        if el.tag.endswith(("}sp", "}pic", "}grpSp")):
            new.shapes._spTree.append(copy.deepcopy(el))
    return new


def fill(slide, heading, items, size=16):
    """items: (level, text) or (level, bold, rest)."""
    slide.shapes[0].text_frame.paragraphs[0].runs[0].text = heading
    body = slide.shapes[1].text_frame
    for para in body.paragraphs:
        para._p.getparent().remove(para._p)
    for item in items:
        lvl, *parts = item
        el = copy.deepcopy(TPL[lvl])
        body._txBody.append(el)
        para = body.paragraphs[-1]
        first = para.runs[0]
        if len(parts) == 2:
            first.text, first.font.bold = parts[0], True
            rest = para.add_run()
            rest.text = parts[1]
            runs = [first, rest]
        else:
            first.text = parts[0]
            runs = [first]
        for r in runs:
            r.font.size = Pt(size if lvl == 0 else size - 2)
        para.space_after = Pt(4)


fill(recap, "Recap", [
    (0, "Status at the last meeting: ", "GBST-BERT = learned tokenizer (GBST) + small BERT encoder, 2.1M parameters"),
    (1, "Pretrained on the human genome (hg38) with the DNA-xLSTM short-context protocol: 10,000 steps"),
    (1, "Compared with the same model without GBST (Transformer++, 2.0M parameters)"),
    (0, "Next step decided: ", "test the model on real biology tasks"),
    (1, "Nucleotide Transformer benchmark (18 tasks), evaluated exactly like the Bio-xLSTM paper"),
])

p1 = clone(progress)
p2 = clone(progress)
p3 = clone(progress)
fill(progress, "Progress (1/4): pretraining and benchmark", [
    (0, "Pretraining (test set): ", "masked-base accuracy 56.44% vs 56.33% for Transformer++ (loss 0.975 vs 0.977)"),
    (0, "Nucleotide Transformer benchmark: ", "18 tasks × 10 repeats, Bio-xLSTM protocol"),
    (1, "Histone marks, enhancers, promoters, splice sites"),
    (1, "Paper's learning rate and batch size per task (Table A5); best validation epoch tested"),
    (0, "Result: ", "best of all 7 models on 13 of 18 tasks"),
    (1, "Better than Transformer++ on 16 of 18 tasks  →  the learned tokenizer helps"),
    (1, "Better than the best published model on 12 tasks, tied on 1"),
    (1, "Weakest task: enhancer or not (−0.21 vs best paper model)"),
])

fill(p1, "Progress (2/4): GBST-BERT vs other models", [
    (0, "Blue = GBST-BERT better"),
    (0, "Left: same model without GBST"),
    (0, "Right: best model in the Bio-xLSTM paper"),
    (0, "Score = mean test score over 10 repeats"),
], size=13)
body = p1.shapes[1]
body.width = Emu(2300000)
h = Emu(3550000)
w = Emu(int(h * 2800 / 1639))
p1.shapes.add_picture(FIG, Emu(p.slide_width - w - 300000), Emu(1080000), w, h)

# results table of all seven models: picture only, as large as fits next to the logo
p3.shapes[0].text_frame.paragraphs[0].runs[0].text = "Progress (3/4): results of all 7 models"
p3.shapes[1]._element.getparent().remove(p3.shapes[1]._element)
th = Emu(3950000)
tw = Emu(int(th * 3000 / 2256))
p3.shapes.add_picture(TABLE, Emu((p.slide_width - tw) // 2 + 300000), Emu(1050000), tw, th)

fill(p2, "Progress (4/4): findings and questions", [
    (0, "What GBST learned: ", "98% of its weight on 1-base blocks, so it hardly merges bases"),
    (1, "The gain probably comes from the extra convolution (local context), not from real \"words\""),
    (0, "Implemented / written: ", "one clean model file (gbst_bert.py, tested against the training code)"),
    (1, "Full notebook that re-runs every step on the server; report, poster, slides; all on GitHub"),
    (0, "Questions :", ""),
    (1, "Is this result enough for the practical work, or should I do more experiments?"),
    (1, "Learning-rate search per task, or keep the paper's settings for a fair comparison?"),
    (1, "Longer pretraining (20k steps) or longer sequences (2048)?"),
], size=15)

fill(nxt, "Next steps", [
    (0, "Before the next meeting :", ""),
    (1, "Small learning-rate / batch-size search on the weakest tasks (enhancers)"),
    (1, "Check whether GBST merges more bases with longer pretraining or longer sequences"),
    (1, "Finish the written report with the benchmark results"),
])

# order: title, recap, progress 1-4, next steps
ids = p.slides._sldIdLst
items = list(ids)
order = [items[0], items[1], items[2], items[4], items[6], items[5], items[3]]
for el in items:
    ids.remove(el)
for el in order:
    ids.append(el)
p.save(OUT)
