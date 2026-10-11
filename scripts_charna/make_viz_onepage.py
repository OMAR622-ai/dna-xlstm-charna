"""Put the four model visualizations on one A3-landscape page, each with a short explanation.

Run after scripts_charna/visualize_model.py (from the repository root):
    python scripts_charna/make_viz_onepage.py
Writes charna_project/results/model_viz/model_visualizations_onepage.{png,pdf}.
"""

import pathlib
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
VIZ = REPO / "charna_project/results/model_viz"
INK, MUTED, ACCENT, BG, SOFT = "#1f1f1e", "#5f5e58", "#1f5a85", "#fcfcfb", "#eef3f8"

PANELS = [
    ("1_architecture.png", "1  How the model is built",
     "DNA (1,024 bases, 15% hidden) → GBST (learned reading, 70,144 parameters) → 4 encoder blocks\n"
     "(attention looks both ways, 508,480 parameters each) → a guess for every base. Total 2,107,392 parameters."),
    ("2_gbst_blocks.png", "2  What GBST does along a real DNA sequence",
     "At every position GBST puts 98.2% of its weight on single bases. The small weight on 2-4-base blocks\n"
     "is the same inside and outside repeats (shaded) and does not follow GC content: GBST does not adapt."),
    ("3_confidence.png", "3  How sure the model is, base by base",
     "Each base hidden once. Inside the tandem repeat (shaded) 94.5% of bases are guessed correctly;\n"
     "elsewhere 50.8%. The model has learned that repeats are predictable (54.7% in this window)."),
    ("4_attention.png", "4  Where each base looks (attention)",
     "Layer 1: every head looks at one fixed nearby position (a few bases left or right). Deeper layers look\n"
     "further (47% → 29% of attention within ±8 bases). Layer 3: one position is looked at by all bases."),
]


def main():
    fig = plt.figure(figsize=(16.54, 11.69))  # A3 landscape, inches
    fig.patch.set_facecolor(BG)
    fig.text(0.02, 0.965, "Inside GBST-BERT: four views of the final model", fontsize=22, weight="bold", color=ACCENT)
    fig.text(0.02, 0.937, "Final model after 10,000 training steps (10.5 billion bases of hg38); example window "
             "chr21:33,045,056-33,046,080; tandem repeats marked by rule (unit of 1-6 bases repeated over ≥12 bases).",
             fontsize=11, color=MUTED)
    boxes = [(0.02, 0.50), (0.51, 0.50), (0.02, 0.035), (0.51, 0.035)]  # lower-left corner of each panel
    for (fname, title, caption), (x, y) in zip(PANELS, boxes):
        w, h = 0.47, 0.42
        frame = fig.add_axes([x, y, w, h])
        frame.set_xticks([]); frame.set_yticks([])
        frame.set_facecolor("white")
        for side in frame.spines.values():
            side.set_color("#e6e5e0")
        fig.text(x + 0.008, y + h - 0.028, title, fontsize=14, weight="bold", color=INK)
        img = mpimg.imread(VIZ / fname)
        ih, iw = img.shape[:2]
        # image area inside the panel: below the title, above the caption
        area_w, area_h = w - 0.016, h - 0.135
        aspect = (iw / ih) * (fig.get_figheight() / fig.get_figwidth())  # width/height in figure fractions
        img_w = min(area_w, area_h * aspect)
        img_h = img_w / aspect
        ax = fig.add_axes([x + (w - img_w) / 2, y + 0.085 + (area_h - img_h) / 2, img_w, img_h])
        ax.imshow(img)
        ax.axis("off")
        wrapped = textwrap.fill(caption.replace("\n", " "), width=100)
        fig.text(x + 0.008, y + 0.012, wrapped, fontsize=10.5, color=INK, va="bottom", linespacing=1.35,
                 bbox=dict(boxstyle="round,pad=0.45", fc=SOFT, ec="none"))
    fig.text(0.98, 0.012, "Main message: the model reads local context through the convolution and attention, "
             "not through GBST's block choice.", fontsize=11, color=ACCENT, ha="right", weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(VIZ / f"model_visualizations_onepage.{ext}", dpi=220 if ext == "png" else None)
    print("saved", VIZ / "model_visualizations_onepage.png", "and .pdf")


if __name__ == "__main__":
    main()
