"""Tests for the downstream (NT benchmark) wiring of GBST + BERT and Transformer++.

Needs the training environment (pytorch_lightning, hydra); skipped otherwise. Optional CUDA
packages (mamba_ssm, flash_attn) are replaced by the placeholders of scripts_charna/train_charna.py.
Run from the repo root:  python -m unittest tests.test_downstream -v
"""

import csv
import importlib.util
import pathlib
import sys
import tempfile
import unittest

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
HAVE_ENV = all(importlib.util.find_spec(m) is not None for m in ("pytorch_lightning", "hydra"))

if HAVE_ENV:
    sys.path.insert(0, str(REPO / "scripts_charna"))
    import train_charna  # noqa: E402

    train_charna.install_placeholders()
    train_charna.ensure_cuda_home()
    sys.path.insert(0, str(REPO))
    # tests/_shim.py (used by test_charna) registers empty stand-ins for `models` and `src`;
    # drop them so the real packages are imported here, whatever order the test modules load in.
    if not hasattr(sys.modules.get("models"), "__file__") or not hasattr(sys.modules.get("src"), "__file__"):
        for name in [n for n in sys.modules if n.split(".")[0] in ("models", "src")]:
            del sys.modules[name]
    from models.charna import CharnaConfig, CharnaForMaskedLM  # noqa: E402
    from models.llama.modelling_llama import TransformerConfig, TransformerLMHeadModel  # noqa: E402
    from src.callbacks.metrics_csv import append_row  # noqa: E402
    from src.models.sequence.dna_embedding import (  # noqa: E402
        DNAEmbeddingModelCharna, DNAEmbeddingModelTransformer, load_backbone,
    )
    sys.path.insert(0, str(REPO / "scripts_charna"))
    import summarize_nt  # noqa: E402

SMALL = dict(d_model=32, n_layer=2, n_heads=4, hidden_dim=40, max_length=64)


def small_transformer_config():
    return TransformerConfig(d_model=32, n_layer=2, bidirectional=True, max_length=64, vocab_size=12,
                             n_heads=4, n_kv_heads=4, hidden_dim=40, multiple_of=8, norm_eps=1e-5, dropout=0.0)


def random_ids(batch=2, length=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(7, 11, (batch, length), generator=g)


def pretrained_checkpoint_state(model):
    """State dict as saved by the Lightning pretraining run (SequenceLightningModule.model = model)."""
    return {f"model.{k}": v.clone() for k, v in model.state_dict().items()}


@unittest.skipUnless(HAVE_ENV, "needs the training environment (pytorch_lightning, hydra)")
class TestLoadBackbone(unittest.TestCase):

    def check_loads_all_backbone_weights(self, pretrained, wrapper):
        loaded = load_backbone(wrapper, pretrained_checkpoint_state(pretrained))
        wrapper.load_state_dict({k[len("model."):]: v for k, v in loaded.items()}, strict=True)
        source = pretrained.state_dict()
        for key, value in wrapper.state_dict().items():
            self.assertTrue(key.startswith("backbone.backbone."), key)
            torch.testing.assert_close(value, source[key[len("backbone."):]], msg=key)

    def test_charna(self):
        for downsample_rate in (1, 2):
            with self.subTest(downsample_rate=downsample_rate):
                config = CharnaConfig(**SMALL, downsample_rate=downsample_rate)
                torch.manual_seed(0)
                pretrained = CharnaForMaskedLM(config)
                torch.manual_seed(1)
                self.check_loads_all_backbone_weights(pretrained, DNAEmbeddingModelCharna(config))

    def test_transformer(self):
        config = small_transformer_config()
        torch.manual_seed(0)
        pretrained = TransformerLMHeadModel(config)
        torch.manual_seed(1)
        self.check_loads_all_backbone_weights(pretrained, DNAEmbeddingModelTransformer(config))


@unittest.skipUnless(HAVE_ENV, "needs the training environment (pytorch_lightning, hydra)")
class TestWrappers(unittest.TestCase):

    def test_transformer_headless_matches_lm(self):
        config = small_transformer_config()
        wrapper = DNAEmbeddingModelTransformer(config).eval()
        ids = random_ids()
        with torch.no_grad():
            hidden, _ = wrapper(ids)
            logits = wrapper.backbone(ids).logits
            torch.testing.assert_close(wrapper.backbone.backbone.output(hidden), logits)

    def test_conjoined_shapes(self):
        ids = random_ids()
        both = torch.stack([ids, ids.flip(1)], dim=-1)
        for wrapper in (DNAEmbeddingModelCharna(CharnaConfig(**SMALL), conjoin_test=True),
                        DNAEmbeddingModelTransformer(small_transformer_config(), conjoin_test=True)):
            with self.subTest(wrapper=type(wrapper).__name__), torch.no_grad():
                wrapper.eval()
                self.assertEqual(tuple(wrapper(both)[0].shape), (2, 50, 32, 2))
                wrapper.train()  # PH: no conjoining during training
                self.assertEqual(tuple(wrapper(ids)[0].shape), (2, 50, 32))


@unittest.skipUnless(HAVE_ENV, "needs the training environment (pytorch_lightning, hydra)")
class TestNTResults(unittest.TestCase):

    def test_metrics_csv_adds_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "metrics.csv"
            append_row(path, {"global_step": 1, "val/mcc": 0.1})
            append_row(path, {"global_step": 2, "val/mcc": 0.2})
            append_row(path, {"global_step": 2, "test/mcc": 0.3})
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["val/mcc"] for r in rows], ["0.1", "0.2", ""])
            self.assertEqual(rows[-1]["test/mcc"], "0.3")

    def test_summary_selects_config_by_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            # config A: better validation, worse test; config B: the opposite. A must be selected.
            for config, val, tests in [("A", 0.8, [0.5, 0.7]), ("B", 0.6, [0.9, 0.9])]:
                for seed, test in enumerate(tests, 1):
                    path = root / "H3" / "m" / config / f"seed-{seed}" / "metrics.csv"
                    path.parent.mkdir(parents=True)
                    append_row(path, {"global_step": 1, "val/mcc": val - 0.1})
                    append_row(path, {"global_step": 2, "val/mcc": val})
                    append_row(path, {"global_step": 2, "test/mcc": test})
            result = summarize_nt.summarize(root, n_seeds=2)[("H3", "m")]
            self.assertEqual(result["config"], "A")
            self.assertAlmostEqual(result["test_mean"], 0.6)
            self.assertAlmostEqual(result["test_range"], 0.2)
            self.assertAlmostEqual(result["test_std"], 0.1414, places=4)


if __name__ == "__main__":
    unittest.main()
