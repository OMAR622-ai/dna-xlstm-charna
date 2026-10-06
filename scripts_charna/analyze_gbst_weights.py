"""How much weight does GBST give to blocks of 1, 2, 3 and 4 bases? Average over 8 hg38
sequences, for several saved models (before and after GBST's score calibration).

    python scripts_charna/analyze_gbst_weights.py      (from the repository root)
"""
import json, sys, pathlib, torch, torch.nn.functional as F
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))
import _shim  # noqa: F401
from models.charna import CharnaConfig, CharnaForMaskedLM
from pyfaidx import Fasta
M = REPO / "charna_project/models"
runs = [("small, LR 5e-4, step 500", M/"small_charna_500/last.ckpt", M/"small_charna_500/model_config.json"),
        ("small, LR 2e-3, step 500", M/"small_charna_500_lr2e-3/last.ckpt", M/"small_charna_500_lr2e-3/model_config.json"),
        ("full, step 2,000", M/"full_charna_mlm/best_test_loss.ckpt", M/"full_charna_mlm/model_config.json"),
        ("full, step 2,500", M/"full_charna_mlm/last.ckpt", M/"full_charna_mlm/model_config.json"),
        ("full, step 10,000 (final)", M/"full_charna_mlm/final_step10000.ckpt", M/"full_charna_mlm/model_config.json")]
g = Fasta(str(REPO / "data/hg38/hg38.ml.fa"))
T = {"A": 7, "C": 8, "G": 9, "T": 10}
seqs = []
for chrom, pos in [("chr21", 33_000_000), ("chr1", 50_000_000), ("chr7", 117_500_000), ("chr12", 6_500_000),
                   ("chr19", 1_000_000), ("chr2", 120_000_000), ("chrX", 70_000_000), ("chr17", 7_670_000)]:
    s = str(g[chrom][pos:pos + 1024]).upper()
    if "N" not in s: seqs.append([T[c] for c in s])
ids = torch.tensor(seqs)
torch.manual_seed(0)
ids_in = ids.masked_fill(torch.rand(ids.shape) < 0.15, 3)
for name, ck, cf in runs:
    c = json.load(open(cf))["config"]; c.pop("_target_")
    m = CharnaForMaskedLM(CharnaConfig(**c)); st = torch.load(ck, map_location="cpu", weights_only=False)
    m.load_state_dict({k[6:]: v for k, v in st["state_dict"].items() if k.startswith("model.")}); m.eval()
    gb = m.backbone.gbst
    with torch.no_grad():
        x = gb.embedding(ids_in); x = gb.conv(F.pad(x.transpose(1, 2), gb.conv_pad)).transpose(1, 2)
        if gb.pointwise is not None: x = gb.pointwise(x)
        ps = gb.scorer(F.pad(x.cumsum(1), (0, 0, 1, 0))).squeeze(-1)
        sc = []
        for b, o in gb.candidates:
            s0, e0 = gb._block_bounds(b, o, ids.size(1), ids.device); sc.append(ps[:, e0] - ps[:, s0])
        raw = torch.stack(sc, -1).softmax(-1)
        cal = torch.softmax(raw @ raw.transpose(1, 2), -1) @ raw
    sizes = torch.tensor([b for b, _ in gb.candidates])
    def per(p): return [100 * p[..., sizes == b].sum(-1).mean().item() for b in range(1, 5)]
    r, c2 = per(raw), per(cal)
    print(f"{name:27s} before calibration: " + " / ".join(f"{v:5.1f}%" for v in r) + "   after: " + " / ".join(f"{v:5.1f}%" for v in c2))
