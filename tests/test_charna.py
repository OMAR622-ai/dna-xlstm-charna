"""Unit tests for the GBST + BERT model and UL2 masking.

Run from the repo root:  python -m unittest tests.test_charna -v
Data tests use HG38_FASTA / HG38_BED (default: data/hg38/hg38.ml.fa, human-sequences.bed)
and are skipped if the files are missing.
"""

import os
import pathlib
import unittest

from tests import _shim  # noqa: F401  (must come before repo imports)

import torch
import torch.nn.functional as F

from models.caduceus.tokenization_caduceus import CaduceusTokenizer
from models.charna import CharnaConfig, CharnaForMaskedLM
from models.charna.gbst import GBST
from models.charna.modeling_charna import (
    conjoined_hidden_states, parameter_breakdown, rc_averaged_logits, reverse_complement_ids,
)
from src.dataloaders.utils.mlm import mlm_getitem
from src.dataloaders.utils.ul2 import UL2_DENOISERS, random_spans_noise_mask, ul2_getitem, ul2_noise_mask

REPO = pathlib.Path(__file__).resolve().parent.parent
HG38_FASTA = pathlib.Path(os.environ.get("HG38_FASTA", REPO / "data/hg38/hg38.ml.fa"))
HG38_BED = pathlib.Path(os.environ.get("HG38_BED", REPO / "data/hg38/human-sequences.bed"))

# The repo passes add_special_tokens=False here; transformers>=4.4x rejects that kwarg (the repo
# pins 4.38.1). It does not affect the vocabulary, so it is omitted for portability.
TOK = CaduceusTokenizer(model_max_length=1024)
MASK_ID, PAD_ID = TOK.mask_token_id, TOK.pad_token_id
BASE_IDS = torch.tensor([TOK._vocab_str_to_int[c] for c in "ACGT"])


def random_dna(batch, length, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BASE_IDS[torch.randint(0, 4, (batch, length), generator=g)]


def small_config(**kw):
    base = dict(d_model=32, n_layer=2, n_heads=4, hidden_dim=40, max_length=256)
    base.update(kw)
    return CharnaConfig(**base)


def naive_gbst(g: GBST, ids):
    """Loop-based reference of ChaRNABERT Eq. 1-7 (blocks containing i, zero outside the sequence)."""
    x = g.embedding(ids)
    x = g.conv(F.pad(x.transpose(1, 2), g.conv_pad)).transpose(1, 2)
    batch, length, dim = x.shape
    cands = []
    for b, o in g.candidates:
        rep = torch.zeros(batch, length, dim, dtype=x.dtype)
        for i in range(length):
            start = i - o if o is not None else (i // b) * b
            for j in range(start, start + b):
                if 0 <= j < length:
                    rep[:, i] += x[:, j]
        if g.pooling == "mean":
            rep = rep / b
        cands.append(rep)
    cands = torch.stack(cands, dim=2)  # B L K d
    probs = g.scorer(cands).squeeze(-1).softmax(-1)
    if g.score_calibration:
        probs = torch.softmax(probs @ probs.transpose(1, 2), -1) @ probs
    return (probs.unsqueeze(-1) * cands).sum(2)


class TestGBST(unittest.TestCase):

    def test_matches_naive_reference(self):
        torch.manual_seed(0)
        ids = random_dna(2, 37)
        for offsets in (True, False):
            for pooling in ("sum", "mean"):
                for calib in (True, False):
                    g = GBST(12, 8, max_block_size=4, offsets=offsets, pooling=pooling, score_calibration=calib).double()
                    out, char = g(ids)
                    ref = naive_gbst(g, ids)
                    self.assertTrue(torch.allclose(out, ref, atol=1e-10), (offsets, pooling, calib))
                    self.assertTrue(torch.equal(out, char))

    def test_candidate_count(self):
        self.assertEqual(GBST(12, 8, max_block_size=4, offsets=True).num_candidates, 10)  # M(M+1)/2
        self.assertEqual(GBST(12, 8, max_block_size=4, offsets=False).num_candidates, 4)

    def test_downsampling_shapes(self):
        ids = random_dna(3, 101)
        for rate in (1, 2, 4):
            g = GBST(12, 16, downsample_rate=rate)
            out, char = g(ids)
            self.assertEqual(out.shape, (3, -(-101 // rate), 16))
            self.assertEqual(char.shape, (3, 101, 16))


class TestModel(unittest.TestCase):

    def test_output_shapes(self):
        ids = random_dna(2, 130)
        for tokenizer, rate in (("gbst", 1), ("gbst", 2), ("gbst", 4), ("embedding", 1)):
            model = CharnaForMaskedLM(small_config(tokenizer=tokenizer, downsample_rate=rate)).eval()
            self.assertEqual(model(ids).logits.shape, (2, 130, 12))
            self.assertEqual(model.headless_forward(ids).shape, (2, -(-130 // rate), 32))

    def test_parameter_count_about_2m(self):
        model = CharnaForMaskedLM(CharnaConfig())
        breakdown = parameter_breakdown(model)
        print("\nGBST-BERT-2M parameter breakdown:")
        for k, v in breakdown.items():
            print(f"  {k:28s} {v:>10,d}")
        from models.llama.modelling_llama import TransformerConfig, TransformerLMHeadModel
        tpp = TransformerLMHeadModel(TransformerConfig(
            d_model=256, n_layer=4, n_heads=8, n_kv_heads=8, bidirectional=True, vocab_size=12,
            hidden_dim=320, multiple_of=8, norm_eps=1e-5, max_length=1024, dropout=0.0))
        tpp_total = sum(p.numel() for p in tpp.parameters())
        print(f"  Transformer++-2M (Bio-xLSTM Table A3, this repo): {tpp_total:,d}")
        self.assertTrue(1.9e6 < breakdown["total"] < 2.2e6, breakdown["total"])
        self.assertLess(abs(breakdown["total"] - tpp_total) / tpp_total, 0.01)

    def test_bidirectional(self):
        """Changing a token far to the right changes outputs on the left (no causal mask)."""
        model = CharnaForMaskedLM(small_config()).eval()
        ids = random_dna(1, 64)
        ids2 = ids.clone()
        ids2[0, 60] = BASE_IDS[(BASE_IDS == ids[0, 60]).nonzero().item() - 1]
        with torch.no_grad():
            self.assertFalse(torch.allclose(model(ids).logits[0, 0], model(ids2).logits[0, 0]))

    def test_loss_only_on_masked_positions(self):
        model = CharnaForMaskedLM(small_config()).eval()
        seq = random_dna(1, 128)[0]
        data, target = ul2_getitem(seq, tokenizer=TOK, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            logits = model(data[None]).logits[0]
        loss = F.cross_entropy(logits, target, ignore_index=PAD_ID)  # same call as the repo's LM task
        masked = target != PAD_ID
        manual = F.cross_entropy(logits[masked], seq[masked])
        self.assertTrue(torch.allclose(loss, manual))


class TestMasking(unittest.TestCase):

    def test_mlm_masking_positions(self):
        torch.manual_seed(0)
        seq = random_dna(1, 100_000)[0]
        data, target = mlm_getitem(seq, mlm_probability=0.15, tokenizer=TOK)
        masked = target != PAD_ID
        self.assertAlmostEqual(masked.float().mean().item(), 0.15, delta=0.005)
        self.assertTrue(torch.equal(target[masked], seq[masked]))
        self.assertTrue(torch.equal(data[~masked], seq[~masked]))  # unmasked inputs untouched
        frac_mask_token = (data[masked] == MASK_ID).float().mean().item()
        self.assertAlmostEqual(frac_mask_token, 0.8, delta=0.02)

    def test_random_spans_exact_counts(self):
        g = torch.Generator().manual_seed(0)
        for _, mu, r in [d for d in UL2_DENOISERS if d[0] != "S"]:
            mask = random_spans_noise_mask(1024, r, mu, g)
            self.assertEqual(int(mask.sum()), round(1024 * r))
            self.assertFalse(bool(mask[0]))  # T5 convention: starts with a non-noise span
            n_spans = int((mask[1:] & ~mask[:-1]).sum())
            self.assertEqual(n_spans, max(round(round(1024 * r) / mu), 1))

    def test_s_denoiser_masks_a_suffix(self):
        g = torch.Generator().manual_seed(0)
        lengths = []
        for _ in range(500):
            mask = ul2_noise_mask(1024, ("S", None, 0.25), g)
            k = int(mask.sum())
            self.assertTrue(1 <= k <= 512)
            self.assertTrue(bool(mask[-k:].all()) and not bool(mask[:-k].any()))
            lengths.append(k)
        self.assertAlmostEqual(sum(lengths) / len(lengths) / 1024, 0.25, delta=0.02)

    def test_ul2_never_leaks_masked_characters(self):
        """Corrupted input is identical whatever the original bases at masked positions were."""
        seq = random_dna(1, 1024, seed=3)[0]
        for seed in range(20):
            data, target = ul2_getitem(seq, tokenizer=TOK, generator=torch.Generator().manual_seed(seed))
            masked = target != PAD_ID
            self.assertTrue(bool((data[masked] == MASK_ID).all()))
            self.assertTrue(torch.equal(data[~masked], seq[~masked]))
            altered = seq.clone()
            altered[masked] = BASE_IDS[(torch.searchsorted(BASE_IDS, seq[masked]) + 1) % 4]
            data2, _ = ul2_getitem(altered, tokenizer=TOK, generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.equal(data, data2))


class TestReverseComplement(unittest.TestCase):
    cmap = TOK.complement_map

    def test_rc_ids(self):
        ids = torch.tensor([[TOK._vocab_str_to_int[c] for c in "AACGTN"]])
        rc = reverse_complement_ids(ids, self.cmap)
        self.assertEqual("".join(TOK._vocab_int_to_str[int(i)] for i in rc[0]), "NACGTT")
        self.assertTrue(torch.equal(reverse_complement_ids(rc, self.cmap), ids))

    def test_rc_averaged_logits_are_equivariant(self):
        model = CharnaForMaskedLM(small_config()).eval()
        ids = random_dna(2, 50)
        rc = reverse_complement_ids(ids, self.cmap)
        perm = torch.arange(12)
        for k, v in self.cmap.items():
            perm[k] = v
        with torch.no_grad():
            a = rc_averaged_logits(model, ids, self.cmap)
            b = rc_averaged_logits(model, rc, self.cmap)
        self.assertTrue(torch.allclose(a, b.flip(1)[..., perm], atol=1e-5))

    def test_posthoc_conjoining_is_rc_invariant(self):
        """Mean-pool -> head -> average over strands gives the same prediction for x and rc(x)."""
        model = CharnaForMaskedLM(small_config()).eval()
        head = torch.nn.Linear(32, 3)
        ids = random_dna(2, 60)
        rc = reverse_complement_ids(ids, self.cmap)

        def predict(x, x_rc):
            h = conjoined_hidden_states(model.headless_forward, torch.stack([x, x_rc], -1))
            pooled = h.mean(1)  # B d 2
            return (head(pooled[..., 0]) + head(pooled[..., 1])) / 2

        with torch.no_grad():
            self.assertTrue(torch.allclose(predict(ids, rc), predict(rc, ids), atol=1e-6))


@unittest.skipUnless(HG38_FASTA.exists() and HG38_BED.exists(), "hg38 files not found")
class TestHG38Dataset(unittest.TestCase):

    def _dataset(self, strategy):
        from src.dataloaders.datasets.hg38_dataset import HG38Dataset
        return HG38Dataset(split="valid", bed_file=HG38_BED, fasta_file=HG38_FASTA, max_length=1024,
                           mlm=True, mlm_probability=0.15, mlm_strategy=strategy, tokenizer=TOK,
                           tokenizer_name="char", add_eos=True)

    def test_items(self):
        torch.manual_seed(0)
        for strategy in ("mlm", "ul2"):
            ds = self._dataset(strategy)
            data, target = ds[12345]
            self.assertEqual(data.shape, (1024,))
            self.assertEqual(target.shape, (1024,))
            self.assertTrue(bool(((target == PAD_ID) | (target >= 7)).all()))
            self.assertFalse(bool((target == TOK._vocab_str_to_int["N"]).any()))  # N is never a target
            if strategy == "ul2":
                self.assertTrue(bool((data[target != PAD_ID] == MASK_ID).all()))

    def test_default_strategy_unchanged(self):
        from src.dataloaders.datasets.hg38_dataset import HG38Dataset
        ds = HG38Dataset(split="valid", bed_file=HG38_BED, fasta_file=HG38_FASTA, max_length=1024,
                         mlm=True, tokenizer=TOK, tokenizer_name="char", add_eos=True)
        self.assertEqual(ds.mlm_strategy, "mlm")


if __name__ == "__main__":
    unittest.main()
