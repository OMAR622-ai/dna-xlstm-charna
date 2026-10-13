"""Make a ~85-second motion-graphic explainer of the project (1920x1080, 30 fps, H.264 MP4).

Scenes: title, aim, data, the guessing game, the model, GBST, training curves, results,
what GBST learned + conclusions, next steps. All numbers are read from the project's result
files (charna_project/results/) or are the measured values documented in the README.

    python scripts_charna/make_explainer_video.py --ffmpeg /path/to/ffmpeg --out docs/video/explainer.mp4

Frames are drawn with Pillow and piped to ffmpeg (libx264, yuv420p, so it plays everywhere).
"""

import argparse
import csv
import math
import pathlib
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parent.parent
RES = REPO / "charna_project/results"
W, H, FPS = 1920, 1080, 30

BG = (252, 252, 251)
INK = (31, 31, 30)
MUTED = (107, 106, 100)
GRID = (230, 229, 224)
ACCENT = (31, 90, 133)
SOFT = (238, 243, 248)
GBST_C = (42, 120, 214)
TPP_C = (235, 104, 52)
GOOD = (0, 131, 0)
BAD = (227, 73, 72)

FONT_DIR = "/usr/share/fonts/truetype/dejavu/"


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono-Bold.ttf" if (mono and bold) else "DejaVuSansMono.ttf" if mono else \
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(FONT_DIR + name, size)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def appear(t, start, dur=0.6):
    """0 -> 1 between start and start + dur (eased)."""
    return ease((t - start) / dur)


def mix(color, a, bg=BG):
    return tuple(int(bg[i] + (color[i] - bg[i]) * a) for i in range(3))


def text(d, xy, s, size, color=INK, a=1.0, bold=False, mono=False, anchor="la"):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, bold, mono), fill=mix(color, a), anchor=anchor)


def header(d, t, title, step):
    a = appear(t, 0.0, 0.5)
    text(d, (110, 80), title, 54, ACCENT, a, bold=True)
    d.rectangle([110, 160, 110 + int(260 * a), 166], fill=mix(ACCENT, a))
    for i in range(9):  # progress dots
        c = ACCENT if i == step else GRID
        d.ellipse([W // 2 - 9 * 22 + i * 44, H - 60, W // 2 - 9 * 22 + i * 44 + 16, H - 44], fill=c)


# ------------------------------------------------------------------ data from the project
def read_curve(run):
    # GBST-BERT's step-10,000 validation was measured after training with the same settings (run R21)
    path = RES / "full/full_charna_mlm_curve_with_final_val.csv" if run == "full_charna_mlm" else \
        RES / "pretrain/hg38" / run / "metrics.csv"
    xs, ys = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("val/loss"):
                xs.append(int(r["tokens"]) / 1e9)
                ys.append(float(r["val/loss"]))
    return xs, ys


def final_test():
    with open(RES / "full/eval_test.csv", newline="") as f:
        return {r["model"]: (float(r["loss"]), 100 * float(r["acc_ACGT"])) for r in csv.DictReader(f)}


def dna_snippet(n=160):
    try:
        from pyfaidx import Fasta
        return str(Fasta(str(REPO / "data/hg38/hg38.ml.fa"))["chr21"][33_000_000:33_000_000 + n]).upper()
    except Exception:  # genome not present: a fixed piece of the same region
        return ("CGAGGTAAGCAGTGTAGGGCTAGAGTGGTTGCTTGACCAGCCATGTCCACTGGGACTCAG" * 3)[:n]


BASE_C = {"A": (27, 175, 122), "C": (42, 120, 214), "G": (237, 161, 0), "T": (227, 73, 72)}


# ------------------------------------------------------------------ scenes
def s_title(d, t, D):
    a = appear(t, 0.2, 0.8)
    text(d, (W // 2, 400), "Does a learned tokenizer", 84, ACCENT, a, bold=True, anchor="mm")
    text(d, (W // 2, 500), "help a small DNA model?", 84, ACCENT, a, bold=True, anchor="mm")
    b = appear(t, 1.3, 0.8)
    text(d, (W // 2, 620), "GBST-BERT (ChaRNABERT-style) trained on the human genome", 40, INK, b, anchor="mm")
    c = appear(t, 2.2, 0.8)
    text(d, (W // 2, 720), "Omar R.A Mohamed  ·  October 2026", 32, MUTED, c, anchor="mm")
    # a DNA strand sliding in
    seq = D["dna"][:48]
    for i, ch in enumerate(seq):
        x = 260 + i * 29 - int(400 * (1 - appear(t, 0.0, 1.5)))
        text(d, (x, 860), ch, 34, BASE_C[ch], appear(t, 0.0 + i * 0.02, 0.5), bold=True, mono=True)


def s_aim(d, t, D):
    header(d, t, "The aim", 0)
    lines = [
        ("Most DNA models fix how a sequence is cut into \"words\":", "single bases, codons or k-mers."),
        ("ChaRNABERT (2024) lets the model learn how to cut,", "at every position: GBST. It helped on RNA."),
        ("Our question: does this also help a small model", "of human DNA?"),
        ("We compare it with the same-size model that", "reads one base at a time: Transformer++."),
    ]
    for i, (l1, l2) in enumerate(lines):
        a = appear(t, 0.6 + i * 1.3, 0.7)
        y = 260 + i * 170
        d.ellipse([130, y + 14, 152, y + 36], fill=mix(ACCENT, a))
        text(d, (190, y), l1, 44, INK, a, bold=(i == 2))
        text(d, (190, y + 60), l2, 44, INK, a, bold=(i == 2))


def s_data(d, t, D):
    header(d, t, "The data: the human genome (hg38)", 1)
    # scrolling DNA
    seq = D["dna"]
    off = int(t * 90)
    a = appear(t, 0.3)
    for i in range(62):
        ch = seq[(i + off // 29) % len(seq)]
        x = 110 + i * 29 - (off % 29)
        if 100 <= x <= W - 120:
            text(d, (x, 230), ch, 34, BASE_C[ch], a, bold=True, mono=True)
    text(d, (110, 300), "3.03 billion bases · cut into pieces of 1,024 bases", 40, INK, appear(t, 1.0))
    # split bars
    parts = [("Training", 77, 2.33, ACCENT), ("Validation", 10, 0.29, (27, 175, 122)),
             ("Test (never seen in training)", 9, 0.27, TPP_C)]
    for i, (name, share, bases, col) in enumerate(parts):
        y = 420 + i * 140
        g = appear(t, 2.0 + i * 0.8, 1.0)
        text(d, (110, y), name, 40, INK, appear(t, 2.0 + i * 0.8), bold=True)
        bw = int(1300 * share / 100 * g)
        d.rounded_rectangle([110, y + 55, 110 + max(bw, 8), y + 105], radius=8, fill=mix(col, appear(t, 2.0 + i * 0.8)))
        text(d, (130 + bw, y + 58), f"{share}%  ({bases} B unique bases)", 36, INK, appear(t, 2.6 + i * 0.8))
    text(d, (110, 860), "The full training reads 10.5 billion bases = the training part about 4.5 times.",
         36, MUTED, appear(t, 5.2))


def s_game(d, t, D):
    header(d, t, "The training game: guess the hidden bases", 2)
    seq = "ACGTTGCAAGTC"
    hidden = {2, 9}
    x0, step = 330, 110
    for i, ch in enumerate(seq):
        x = x0 + i * step
        a = appear(t, 0.2 + i * 0.05, 0.4)
        is_h = i in hidden
        h = appear(t, 2.0, 0.6) if is_h else 0
        box = mix(SOFT, a) if not is_h else mix((255, 236, 210), h * a) if h > 0 else mix(SOFT, a)
        d.rounded_rectangle([x, 300, x + 90, 400], radius=14, fill=box)
        shown = "?" if (is_h and h > 0.5) else ch
        col = INK if shown == "?" else BASE_C[ch]
        text(d, (x + 45, 350), shown, 56, col, a, bold=True, mono=True, anchor="mm")
        if is_h:  # the model's guess
            g = appear(t, 4.2, 0.6)
            text(d, (x + 45, 470), ch, 56, BASE_C[ch], g, bold=True, mono=True, anchor="mm")
            text(d, (x + 45, 540), "✓", 50, GOOD, appear(t, 5.0, 0.4), bold=True, anchor="mm")
    text(d, (110, 300), "Real", 34, MUTED, appear(t, 0.2))
    text(d, (110, 340), "→ seen", 34, MUTED, appear(t, 2.0))
    text(d, (110, 450), "Guess", 34, MUTED, appear(t, 4.2))
    text(d, (110, 660), "15% of the bases are hidden. The model must guess them from both sides.", 40, INK, appear(t, 1.0))
    text(d, (110, 730), "No labels needed: the DNA itself is the teacher (masked language modelling).", 40, INK,
         appear(t, 3.0))
    text(d, (110, 800), "Score: loss = how wrong the guesses are (random guessing = 1.386).", 40, MUTED, appear(t, 5.5))


def s_model(d, t, D):
    header(d, t, "The model: GBST-BERT (2.1 million parameters)", 3)
    boxes = [("DNA", "ACGTTGCA...", SOFT), ("GBST", "learns how to read", (219, 233, 250)),
             ("BERT encoder", "4 blocks, looks both ways", (219, 233, 250)), ("Prediction", "guess hidden bases", SOFT)]
    bw, gap, y = 380, 80, 360
    x0 = (W - (4 * bw + 3 * gap)) // 2
    for i, (title, sub, fill) in enumerate(boxes):
        a = appear(t, 0.4 + i * 1.0, 0.7)
        x = x0 + i * (bw + gap) + int(60 * (1 - a))
        d.rounded_rectangle([x, y, x + bw, y + 200], radius=22, fill=mix(fill, a), outline=mix(ACCENT, a), width=4)
        text(d, (x + bw // 2, y + 75), title, 38, ACCENT, a, bold=True, anchor="mm")
        text(d, (x + bw // 2, y + 135), sub, 26, INK, a, anchor="mm")
        if i < 3:
            aa = appear(t, 0.9 + i * 1.0, 0.5)
            xa = x0 + i * (bw + gap) + bw + 12
            d.line([xa, y + 100, xa + int((gap - 30) * aa), y + 100], fill=mix(ACCENT, aa), width=8)
            if aa > 0.9:
                d.polygon([(xa + gap - 30, y + 84), (xa + gap - 6, y + 100), (xa + gap - 30, y + 116)], fill=ACCENT)
    # attention arcs inside the encoder box
    a = appear(t, 4.5, 0.8)
    ex = x0 + 2 * (bw + gap)
    for k in range(5):
        cx = ex + 40 + k * 65
        d.arc([cx - 30, y + 200 + 10, cx + 95, y + 200 + 90], 0, 180, fill=mix(ACCENT, a * 0.7), width=4)
    text(d, (ex + bw // 2, y + 320), "every base looks at every other base", 28, MUTED, a, anchor="mm")
    text(d, (110, 800), "Comparison model Transformer++: the same encoder idea, but reads one base at a time (no GBST).",
         36, INK, appear(t, 5.8))


def s_gbst(d, t, D):
    header(d, t, "GBST: choosing how to read, at every position", 4)
    seq = "ACGTTGCA"
    pos = 3
    x0, step = 140, 78
    for i, ch in enumerate(seq):
        a = appear(t, 0.2, 0.5)
        x = x0 + i * step
        hl = appear(t, 0.8, 0.5) if i == pos else 0
        if hl:
            d.rounded_rectangle([x - 6, 230, x + 64, 310], radius=12, fill=mix((255, 236, 210), hl))
        text(d, (x + 29, 270), ch, 52, BASE_C[ch], a, bold=True, mono=True, anchor="mm")
    text(d, (x0, 330), "position 3 (the first T): all blocks of 1–4 bases that contain it", 30, MUTED, appear(t, 0.8))
    blocks = [(1, [(3, 1)]), (2, [(3, 2), (2, 2)]), (3, [(3, 3), (2, 3), (1, 3)]), (4, [(3, 4), (2, 4), (1, 4), (0, 4)])]
    learned = {1: 98.0, 2: 1.9, 3: 0.1, 4: 0.0}
    equal = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}  # 1, 2, 3, 4 candidates x 10% each at the start
    learn = appear(t, 6.2, 2.2)
    for r, (size, cands) in enumerate(blocks):
        y = 400 + r * 115
        a = appear(t, 1.6 + r * 0.9, 0.6)
        text(d, (x0, y + 20), f"size {size}", 32, INK, a, bold=True)
        for j, (start, b) in enumerate(cands):
            bx = x0 + 160 + j * 260
            s = seq[start:start + b]
            d.rounded_rectangle([bx, y, bx + 230, y + 70], radius=12, fill=mix(SOFT, a), outline=mix(GRID, a), width=2)
            text(d, (bx + 115, y + 35), s, 38, INK, a, bold=True, mono=True, anchor="mm")
        # weight bar for this block size
        wv = equal[size] * (1 - learn) + learned[size] * learn
        bxx = 1350
        g = appear(t, 5.2, 0.6)
        d.rounded_rectangle([bxx, y + 10, bxx + 380, y + 60], radius=8, fill=mix(GRID, g))
        d.rounded_rectangle([bxx, y + 10, bxx + max(int(380 * wv / 100), 4), y + 60], radius=8, fill=mix(GBST_C, g))
        text(d, (bxx + 395, y + 35), f"{wv:4.1f}%", 32, INK, g, anchor="lm")
    text(d, (1350, 290), "weight GBST gives each block size", 28, MUTED, appear(t, 5.2))
    text(d, (1350, 330), "start of training → after training", 28, MUTED, appear(t, 6.2))
    text(d, (140, 880), "After training on human DNA, GBST put about 98% of its weight on single bases.", 38,
         ACCENT, appear(t, 8.8), bold=True)


def s_training(d, t, D):
    header(d, t, "Training: 10,000 steps, 10.5 billion bases, 4 GPUs", 5)
    # axes
    L, R, T, B = 230, 1450, 260, 860
    xmin, xmax, ymin, ymax = 1.5, 11, 0.965, 1.065
    a = appear(t, 0.2)
    d.line([L, B, R, B], fill=mix(GRID, a), width=3)
    d.line([L, T, L, B], fill=mix(GRID, a), width=3)
    for yv in (0.97, 0.99, 1.01, 1.03, 1.05):
        yy = B - (yv - ymin) / (ymax - ymin) * (B - T)
        d.line([L, yy, R, yy], fill=mix(GRID, a * 0.6), width=1)
        text(d, (L - 20, yy), f"{yv:.2f}", 26, MUTED, a, anchor="rm")
    for xv in (2, 4, 6, 8, 10):
        xx = L + (xv - xmin) / (xmax - xmin) * (R - L)
        text(d, (xx, B + 30), f"{xv}", 26, MUTED, a, anchor="mm")
    text(d, ((L + R) // 2, B + 75), "billions of bases seen", 28, MUTED, a, anchor="mm")
    text(d, (L - 140, T - 40), "loss (lower = better)", 28, MUTED, a)
    text(d, (R, T - 40), "random guessing would be 1.386 (far above this zoomed view)", 26, MUTED, appear(t, 0.8), anchor="ra")

    def px(x, y):
        return L + (x - xmin) / (xmax - xmin) * (R - L), B - (y - ymin) / (ymax - ymin) * (B - T)

    prog = appear(t, 1.6, 6.5)  # draw curves progressively
    xlim = 2.0 + prog * 8.6
    for name, (xs, ys), col in (("GBST-BERT", D["curve_gbst"], GBST_C), ("Transformer++", D["curve_tpp"], TPP_C)):
        xs2, ys2 = xs, ys  # measured validation points only
        pts = []
        for i in range(len(xs2) - 1):
            for k in range(21):
                x = xs2[i] + (xs2[i + 1] - xs2[i]) * k / 20
                if x > xlim:
                    break
                y = ys2[i] + (ys2[i + 1] - ys2[i]) * k / 20
                pts.append(px(x, y))
        if len(pts) > 1:
            d.line(pts, fill=col, width=6, joint="curve")
        for x, y in zip(xs, ys):
            if x <= xlim:
                cx, cy = px(x, y)
                d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=col, outline=BG, width=3)
        if xlim >= xs[-1]:
            cx, cy = px(xs[-1], ys[-1])
            if name == "GBST-BERT":
                text(d, (cx - 10, cy + 44), f"{name}  {ys[-1]:.3f}", 30, INK, appear(t, 8.2), anchor="rm")
            else:
                text(d, (cx + 24, cy + 30), f"{name}  {ys[-1]:.3f}", 30, INK, appear(t, 8.2), anchor="lm")
    steps = int(min(max(xlim, 0) / 10.49, 1) * 10000)
    text(d, (1530, 330), "step", 30, MUTED, a)
    text(d, (1530, 370), f"{steps:,}", 64, ACCENT, a, bold=True)
    text(d, (1530, 470), "validation every", 26, MUTED, a)
    text(d, (1530, 505), "2,000 steps", 26, MUTED, a)
    text(d, (1530, 540), "(zoomed y-axis)", 26, MUTED, a)
    text(d, (1530, 640), "GBST-BERT", 32, GBST_C, appear(t, 1.6), bold=True)
    text(d, (1530, 690), "Transformer++", 32, TPP_C, appear(t, 1.6), bold=True)


def s_results(d, t, D):
    header(d, t, "Result: accuracy on unseen test DNA", 6)
    test = D["test"]
    bars = [("Random guessing", 25.0, GRID), ("Always the most common base", 29.42, (200, 199, 192)),
            ("Transformer++", test["Transformer++"][1], TPP_C), ("GBST-BERT", test["GBST-BERT"][1], GBST_C)]
    for i, (name, val, col) in enumerate(bars):
        y = 280 + i * 150
        a = appear(t, 0.4 + i * 0.9, 0.5)
        g = appear(t, 0.6 + i * 0.9, 1.2)
        text(d, (110, y + 30), name, 38, INK, a, bold=(i >= 2), anchor="lm")
        bw = int(900 * val / 60 * g)
        d.rounded_rectangle([820, y, 820 + max(bw, 6), y + 62], radius=10, fill=mix(col, a))
        text(d, (840 + bw, y + 31), f"{val * g:.2f}%", 40, INK, a, bold=(i >= 2), anchor="lm")
    text(d, (110, 900), "Both models learned human DNA well; the difference is only 0.1 points: a tie.", 40, ACCENT,
         appear(t, 4.8), bold=True)


def s_conclusion(d, t, D):
    header(d, t, "What we learned", 7)
    items = [
        ("Both 2 M models learned human DNA: 56% accuracy, loss ≈ 0.975,", "like published models of the same size."),
        ("GBST-BERT was ahead for most of the training,", "but finished only 0.1 points ahead."),
        ("Inside the model, GBST learned to read almost only single bases (98%):", "that is why both models end so close."),
        ("Caveats: one run per model; only GBST-BERT's learning rate was tuned.", ""),
    ]
    for i, (l1, l2) in enumerate(items):
        a = appear(t, 0.5 + i * 1.5, 0.7)
        y = 250 + i * 165
        d.ellipse([130, y + 14, 152, y + 36], fill=mix(ACCENT if i < 3 else MUTED, a))
        text(d, (190, y), l1, 40, INK if i < 3 else MUTED, a)
        if l2:
            text(d, (190, y + 56), l2, 40, INK if i < 3 else MUTED, a)


def s_next(d, t, D):
    header(d, t, "Next steps", 8)
    items = ["A clean test: the same model with GBST replaced by a plain lookup table.",
             "Biology tasks: 18 Nucleotide Transformer tasks (promoters, enhancers, splice sites).",
             "Repeat runs with other seeds to measure run-to-run variation."]
    for i, s in enumerate(items):
        a = appear(t, 0.4 + i * 1.0, 0.7)
        y = 270 + i * 120
        text(d, (130, y), f"{i + 1}", 44, ACCENT, a, bold=True)
        text(d, (200, y), s, 40, INK, a)
    a = appear(t, 3.6, 0.8)
    d.rounded_rectangle([110, 700, W - 110, 900], radius=24, fill=mix(SOFT, a))
    text(d, (W // 2, 760), "Code, models, results and report", 40, ACCENT, a, bold=True, anchor="mm")
    text(d, (W // 2, 830), "github.com/OMAR622-ai/dna-xlstm-charna  (branch charna-gbst)", 36, INK, a, anchor="mm")


SCENES = [(s_title, 6), (s_aim, 7.5), (s_data, 9), (s_game, 9), (s_model, 9), (s_gbst, 11.5),
          (s_training, 11), (s_results, 9), (s_conclusion, 9), (s_next, 7)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out", default=str(REPO / "docs/video/explainer.mp4"))
    ap.add_argument("--preview", help="only write a PNG contact sheet of one frame per scene to this path")
    args = ap.parse_args()

    D = {"dna": dna_snippet(), "curve_gbst": read_curve("full_charna_mlm"),
         "curve_tpp": read_curve("full_llama_mlm"), "test": final_test()}

    def frame(scene, dur, t):
        img = Image.new("RGB", (W, H), BG)
        scene(ImageDraw.Draw(img), t, D)
        fade = min(ease(t / 0.35), ease((dur - t) / 0.35))  # fade in / out at the scene edges
        if fade < 1:
            img = Image.blend(Image.new("RGB", (W, H), BG), img, fade)
        return img

    if args.preview:
        thumbs = [frame(scene, dur, dur * 0.85).resize((480, 270)) for scene, dur in SCENES]
        sheet = Image.new("RGB", (480 * 4, 270 * math.ceil(len(thumbs) / 4)), BG)
        for i, th in enumerate(thumbs):
            sheet.paste(th, ((i % 4) * 480, (i // 4) * 270))
        sheet.save(args.preview)
        print(f"saved {args.preview}")
        return

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = 0
    for scene, dur in SCENES:
        for k in range(int(dur * FPS)):
            proc.stdin.write(np.asarray(frame(scene, dur, k / FPS)).tobytes())
            total += 1
    proc.stdin.close()
    proc.wait()
    print(f"saved {out}: {total} frames, {total / FPS:.1f} s")


if __name__ == "__main__":
    main()
