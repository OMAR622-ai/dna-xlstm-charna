"""Four visualizations of the final GBST-BERT model (no training needed; runs on the CPU).

  1. architecture  - the model's parts with data shapes and parameter counts
  2. gbst_blocks   - GBST's weight on blocks of 1/2/3/4 bases at every position of a real sequence
  3. confidence    - how sure the model is of the true base at every position (each base hidden once)
  4. attention     - which positions each base looks at, per layer and head

A real 1,024-base window of hg38 is used (chr21, the first window from 33.0 Mb on that contains
a simple tandem repeat of at least 30 bases). Tandem repeats (period 1-6, >= 12 bases) and the local GC content are
marked; the genome file has no repeat annotation (no lowercase soft-masking).

    python scripts_charna/visualize_model.py     # from the repository root; writes charna_project/results/model_viz/
"""

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import _shim  # noqa: E402,F401
from models.charna import CharnaConfig, CharnaForMaskedLM  # noqa: E402
from models.charna.modeling_charna import apply_rotary_emb  # noqa: E402

MODEL_DIR = REPO / "charna_project/models/full_charna_mlm"
OUT = REPO / "charna_project/results/model_viz"
TOKEN = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
MASK = 3
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
ACCENT, SOFT = "#1f5a85", "#eef3f8"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BASE_COLOR = {"A": AQUA, "C": BLUE, "G": YELLOW, "T": "#e34948"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": BG, "axes.facecolor": BG})


# ------------------------------------------------------------------ model and data
def load_model():
    config = json.load(open(MODEL_DIR / "model_config.json"))["config"]
    config.pop("_target_")
    model = CharnaForMaskedLM(CharnaConfig(**config))
    ckpt = torch.load(MODEL_DIR / "final_step10000.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")})
    return model.eval()


def tandem_repeats(seq, min_len=12):
    """Spans of simple tandem repeats: a unit of 1-6 bases repeated to cover >= min_len bases."""
    spans = []
    for period in range(1, 7):
        for m in re.finditer(r"(?=(([ACGT]{%d})\2{%d,}))" % (period, max(min_len // period - 1, 1)), seq):
            s, e = m.start(), m.start() + len(m.group(1))
            if e - s >= min_len:
                spans.append((s, e))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def pick_window():
    from pyfaidx import Fasta
    genome = Fasta(str(REPO / "data/hg38/hg38.ml.fa"))
    start = 33_000_000
    for _ in range(400):
        seq = str(genome["chr21"][start:start + 1024]).upper()
        if "N" not in seq and any(e - s >= 30 for s, e in tandem_repeats(seq)):
            return start, seq
        start += 1024
    raise RuntimeError("no window with a tandem repeat found")


def gc_track(seq, w=50):
    gc = np.array([c in "GC" for c in seq], dtype=float)
    return np.convolve(gc, np.ones(w) / w, mode="same")


def shade(ax, spans, label=True):
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color="#f6d7b8", alpha=0.7, lw=0, label="tandem repeat" if (label and i == 0) else None)


# ------------------------------------------------------------------ 1. architecture
def fig_architecture(model, pdf):
    groups = {"GBST": sum(p.numel() for p in model.backbone.gbst.parameters()),
              "block": sum(p.numel() for p in model.backbone.layers[0].parameters()),
              "norm": sum(p.numel() for p in model.backbone.norm.parameters()),
              "head": sum(p.numel() for p in model.lm_head.parameters())}
    total = sum(p.numel() for p in model.parameters())
    cfg = model.config
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(1.0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fill=SOFT, edge=ACCENT):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fill, ec=edge, lw=1.6))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12, color=ACCENT, weight="bold")
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.72 - i * 0.34, line, ha="center", va="center", fontsize=9.5, color=INK)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=ACCENT, lw=1.6))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label, ha="center", fontsize=8.5, color=MUTED)

    y = 3.3
    box(0.2, y, 1.9, 1.7, "Input", ["1,024 bases", "15% hidden", "token ids"], fill="#ffffff")
    box(2.6, y - 0.9, 3.0, 3.5, "GBST", ["embedding 12 → 256", "depthwise conv (4) + pointwise",
                                        "10 candidate blocks", "scores → softmax", "calibration (Eq. 6)",
                                        "weighted mix", f"{groups['GBST']:,} parameters"])
    box(6.1, y - 0.9, 3.4, 3.5, f"Encoder × {cfg.n_layer}", ["LayerNorm", f"attention, {cfg.n_heads} heads",
                                                        "(RoPE + QKNorm, both directions)", "LayerNorm",
                                                        f"SwiGLU {cfg.d_model} → {cfg.hidden_dim} → {cfg.d_model}",
                                                        "+ residual connections",
                                                        f"{groups['block']:,} parameters each"])
    box(10.0, y, 1.9, 1.7, "Head", ["final LayerNorm", "linear 256 → 12", f"{groups['head'] + groups['norm']:,} params"])
    box(12.1 - 0.1, y + 0.35, 0.95, 1.0, "", [], fill="#ffffff", edge=GRID)
    ax.text(12.47, y + 0.85, "guess\nper base", ha="center", va="center", fontsize=9, color=INK)
    arrow(2.1, y + 0.85, 2.6, y + 0.85, "(1024)")
    arrow(5.6, y + 0.85, 6.1, y + 0.85, "(1024×256)")
    arrow(9.5, y + 0.85, 10.0, y + 0.85, "(1024×256)")
    arrow(11.9, y + 0.85, 12.0, y + 0.85)
    ax.text(6.5, 6.85, f"GBST-BERT: {total:,} parameters (final model, 10,000 training steps)", ha="center",
            fontsize=15, color=INK, weight="bold")
    ax.text(6.5, 1.35, "Shapes in brackets: positions × numbers per position. Every base keeps its own position "
            "(no downsampling).\nTraining target: the true base at the hidden positions (loss = cross-entropy).",
            ha="center", fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "1_architecture.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)
    return groups, total


# ------------------------------------------------------------------ 2. GBST block choice
@torch.no_grad()
def gbst_weights(gbst, ids):
    x = gbst.embedding(ids)
    x = gbst.conv(F.pad(x.transpose(1, 2), gbst.conv_pad)).transpose(1, 2)
    if gbst.pointwise is not None:
        x = gbst.pointwise(x)
    ps = gbst.scorer(F.pad(x.cumsum(1), (0, 0, 1, 0))).squeeze(-1)
    scores = []
    for b, o in gbst.candidates:
        s, e = gbst._block_bounds(b, o, ids.size(1), ids.device)
        scores.append(ps[:, e] - ps[:, s])
    p = torch.stack(scores, -1).softmax(-1)
    if gbst.score_calibration:
        p = torch.softmax(p @ p.transpose(1, 2), -1) @ p
    sizes = torch.tensor([b for b, _ in gbst.candidates])
    return torch.stack([p[0][:, sizes == b].sum(-1) for b in range(1, 5)])  # (4, L)


def fig_gbst(model, seq, start, repeats, pdf):
    ids = torch.tensor([[TOKEN[c] for c in seq]])
    w = gbst_weights(model.backbone.gbst, ids).numpy()
    fig, axes = plt.subplots(3, 1, figsize=(13, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.4, 0.8]})
    im = axes[0].imshow(w, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                        extent=[0, len(seq), 4.5, 0.5])
    axes[0].set_yticks([1, 2, 3, 4], ["1 base", "2 bases", "3 bases", "4 bases"])
    axes[0].set_title("GBST's weight on each block size at every position (dark = high)", loc="left")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.015, pad=0.01)
    cb.set_label("weight")
    multi = 100 * (1 - w[0])
    shade(axes[1], repeats)
    axes[1].plot(multi, color=BLUE, lw=1)
    axes[1].set_ylabel("weight on blocks\nof 2-4 bases (%)")
    axes[1].set_title(f"Weight on multi-base blocks: mean {multi.mean():.1f}%, max {multi.max():.1f}% "
                      f"(single bases: mean {100 * w[0].mean():.1f}%)", loc="left")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(color=GRID, lw=0.6)
    gc = gc_track(seq)
    shade(axes[2], repeats, label=False)
    axes[2].plot(100 * gc, color=MUTED, lw=1)
    axes[2].set_ylabel("GC % (50 bp)")
    axes[2].set_xlabel(f"position in chr21:{start:,}-{start + 1024:,} (hg38)")
    axes[2].grid(color=GRID, lw=0.6)
    in_rep = np.zeros(len(seq), bool)
    for s, e in repeats:
        in_rep[s:e] = True
    fig.suptitle("2  What GBST does along a real DNA sequence", x=0.01, ha="left", fontsize=14, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2_gbst_blocks.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)
    return {"single_mean": float(w[0].mean()), "multi_in_repeat": float(multi[in_rep].mean()) if in_rep.any() else None,
            "multi_outside": float(multi[~in_rep].mean()), "corr_gc": float(np.corrcoef(multi, gc)[0, 1])}


# ------------------------------------------------------------------ 3. prediction confidence
@torch.no_grad()
def confidence(model, seq, stride=7):
    """Hide every 7th base (7 shifted passes, about 14% hidden each) so every base is hidden once."""
    ids = torch.tensor([[TOKEN[c] for c in seq]])
    p_true = np.zeros(len(seq))
    guess = np.zeros(len(seq), int)
    for k in range(stride):
        hidden = torch.zeros_like(ids, dtype=torch.bool)
        hidden[0, k::stride] = True
        logits = model(ids.masked_fill(hidden, MASK)).logits[0][:, 7:11]  # A, C, G, T
        probs = logits.softmax(-1)
        idx = torch.nonzero(hidden[0]).squeeze(-1)
        p_true[idx] = probs[idx, ids[0, idx] - 7].numpy()
        guess[idx] = probs[idx].argmax(-1).numpy()
    correct = guess == (ids[0].numpy() - 7)
    return p_true, correct


def fig_confidence(model, seq, start, repeats, pdf):
    p, correct = confidence(model, seq)
    in_rep = np.zeros(len(seq), bool)
    for s, e in repeats:
        in_rep[s:e] = True
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), gridspec_kw={"height_ratios": [2.2, 1]})
    ax = axes[0]
    shade(ax, repeats)
    ax.bar(np.arange(len(seq)), p, width=1.0, color=np.where(correct, BLUE, "#b8c9dd"), lw=0)
    smooth = np.convolve(p, np.ones(25) / 25, mode="same")
    ax.plot(smooth, color=ORANGE, lw=1.8, label="average over 25 bases")
    ax.axhline(0.25, color=MUTED, lw=1, ls="--")
    ax.text(len(seq) - 4, 0.255, "random guessing (25%)", ha="right", va="bottom", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=GRID), zorder=5)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(seq))
    ax.set_ylabel("probability given\nto the true base")
    ax.set_title(f"Each base hidden once (dark bars = best guess correct). Accuracy {100 * correct.mean():.1f}%; "
                 f"mean probability {100 * p.mean():.1f}%", loc="left")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    # zoom: the first tandem repeat with its letters
    s0, e0 = max(repeats, key=lambda r: r[1] - r[0])
    z0, z1 = max(s0 - 20, 0), min(e0 + 20, len(seq))
    ax2 = axes[1]
    for i in range(z0, z1):
        ax2.bar(i, p[i], color=BASE_COLOR[seq[i]], width=0.85)
        ax2.text(i, -0.08, seq[i], ha="center", va="top", fontsize=8, color=BASE_COLOR[seq[i]], family="monospace",
                 weight="bold")
    ax2.axvspan(s0 - 0.5, e0 - 0.5, color="#f6d7b8", alpha=0.7, lw=0)
    ax2.set_ylim(-0.18, 1)
    ax2.set_xlim(z0 - 0.5, z1 - 0.5)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_xticks([])
    ax2.set_title(f"Zoom on the longest tandem repeat, {s0}-{e0} (shaded): bars = probability for the true base", loc="left")
    fig.suptitle(f"3  How sure the model is, base by base (chr21:{start:,}-{start + 1024:,})", x=0.01, ha="left",
                 fontsize=14, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "3_confidence.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)
    return {"acc": float(correct.mean()), "p_mean": float(p.mean()),
            "p_repeat": float(p[in_rep].mean()), "p_other": float(p[~in_rep].mean()),
            "acc_repeat": float(correct[in_rep].mean()), "acc_other": float(correct[~in_rep].mean())}


# ------------------------------------------------------------------ 4. attention
@torch.no_grad()
def attention_maps(model, ids):
    bb = model.backbone
    h, _ = bb.gbst(ids)
    L = ids.size(1)
    cos, sin = bb.freqs_cos[:L], bb.freqs_sin[:L]
    maps = []
    for layer in bb.layers:
        att = layer.attention
        x = layer.attention_norm(h)
        q = att.q_norm(att.wq(x).view(1, L, att.n_heads, att.head_dim))
        k = att.k_norm(att.wk(x).view(1, L, att.n_heads, att.head_dim))
        q, k = apply_rotary_emb(q, k, cos, sin)
        a = torch.softmax(q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / att.head_dim ** 0.5, -1)
        maps.append(a[0].numpy())  # (heads, L, L)
        h = layer(h, cos, sin)
    return np.stack(maps)  # (layers, heads, L, L)


def fig_attention(model, seq, pdf):
    ids = torch.tensor([[TOKEN[c] for c in seq]])
    g = torch.Generator().manual_seed(0)
    hidden = torch.rand(ids.shape, generator=g) < 0.15
    maps = attention_maps(model, ids.masked_fill(hidden, MASK))
    n_layers, n_heads, L, _ = maps.shape
    rel = np.arange(-32, 33)
    fig, axes = plt.subplots(2, n_layers, figsize=(14, 6.6), gridspec_kw={"height_ratios": [1, 1.25]})
    head_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_heads))
    summary = []
    for li in range(n_layers):
        ax = axes[0, li]
        prof_all = []
        for hd in range(n_heads):
            m = maps[li, hd]
            prof = np.array([np.mean(np.diagonal(m, offset=r)) for r in rel])
            prof_all.append(prof)
            ax.plot(rel, prof, color=head_colors[hd], lw=1)
        prof_mean = np.mean(prof_all, 0)
        ax.plot(rel, prof_mean, color=INK, lw=2, label="mean of heads")
        ax.set_title(f"Layer {li + 1}: attention by distance", loc="left", fontsize=10)
        ax.set_xlabel("distance to the attended base")
        ax.grid(color=GRID, lw=0.6)
        if li == 0:
            ax.set_ylabel("average attention weight")
            ax.legend(frameon=False, fontsize=8)
        near = np.mean([np.sum(m[np.abs(np.subtract.outer(np.arange(L), np.arange(L))) <= 8]) / L for m in maps[li]])
        summary.append(near)
        ax2 = axes[1, li]
        z = maps[li].mean(0)[:64, :64]
        ax2.imshow(z, cmap="Blues", vmin=0, vmax=np.percentile(z, 99.5), interpolation="nearest")
        ax2.set_title(f"Layer {li + 1}: first 64 bases (mean of {n_heads} heads)\n"
                      f"{100 * near:.0f}% of attention within ±8 bases", loc="left", fontsize=9.5)
        ax2.set_xlabel("attended base")
        if li == 0:
            ax2.set_ylabel("base that looks")
    fig.suptitle("4  Where each base looks: attention per layer (thin lines = the 8 heads)", x=0.01, ha="left",
                 fontsize=14, color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "4_attention.png", dpi=160)
    pdf.savefig(fig)
    plt.close(fig)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_model()
    start, seq = pick_window()
    repeats = tandem_repeats(seq)
    print(f"window chr21:{start:,}-{start + 1024:,}; tandem repeats: {repeats}")
    with PdfPages(OUT / "model_visualizations.pdf") as pdf:
        groups, total = fig_architecture(model, pdf)
        print(f"1 architecture: total {total:,}; GBST {groups['GBST']:,}; per block {groups['block']:,}; "
              f"head {groups['head']:,}; final norm {groups['norm']:,}")
        g = fig_gbst(model, seq, start, repeats, pdf)
        print(f"2 GBST: single-base weight {100 * g['single_mean']:.1f}%; multi-base weight in repeats "
              f"{g['multi_in_repeat']}, outside {g['multi_outside']:.4f}; correlation with GC {g['corr_gc']:.2f}")
        c = fig_confidence(model, seq, start, repeats, pdf)
        print(f"3 confidence: accuracy {100 * c['acc']:.1f}%, mean p {100 * c['p_mean']:.1f}%; in repeats p "
              f"{100 * c['p_repeat']:.1f}% (acc {100 * c['acc_repeat']:.1f}%), elsewhere p {100 * c['p_other']:.1f}% "
              f"(acc {100 * c['acc_other']:.1f}%)")
        a = fig_attention(model, seq, pdf)
        print("4 attention within ±8 bases per layer: " + ", ".join(f"{100 * v:.0f}%" for v in a))
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
