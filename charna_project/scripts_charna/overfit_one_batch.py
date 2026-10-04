"""Sanity check: the 2M GBST + BERT model can memorise one fixed masked batch of real hg38.

The masked batch is drawn once and reused every step, so loss on the masked characters should
approach zero. Optimiser settings here are only for this check, not the pretraining recipe.

  python scripts_charna/overfit_one_batch.py --fasta .../hg38.ml.fa --bed .../human-sequences.bed
"""

import argparse
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests import _shim  # noqa: F401,E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.caduceus.tokenization_caduceus import CaduceusTokenizer  # noqa: E402
from models.charna import CharnaConfig, CharnaForMaskedLM  # noqa: E402
from src.dataloaders.datasets.hg38_dataset import HG38Dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--bed", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--target", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = CaduceusTokenizer(model_max_length=args.length)
    ds = HG38Dataset(split="train", bed_file=args.bed, fasta_file=args.fasta, max_length=args.length,
                     mlm=True, mlm_probability=0.15, tokenizer=tok,
                     tokenizer_name="char", add_eos=True)
    idx = torch.randint(len(ds), (args.batch,)).tolist()
    data, target = map(torch.stack, zip(*[ds[i] for i in idx]))
    n_masked = int((target != tok.pad_token_id).sum())
    print(f"batch {tuple(data.shape)}, masked characters: {n_masked}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, target = data.to(device), target.to(device)
    model = CharnaForMaskedLM(CharnaConfig(max_length=args.length)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,d}  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        logits = model(data).logits
        loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(), ignore_index=tok.pad_token_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if loss.item() < args.target:
            print(f"reached loss < {args.target} at step {step}")
            return 0
    print(f"did not reach loss < {args.target} in {args.steps} steps (final {loss.item():.4f})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
