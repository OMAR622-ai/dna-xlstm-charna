"""The self-contained gbst_bert.py must stay identical to the training code in models/charna/."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _shim  # noqa: E402,F401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gbst_bert  # noqa: E402
from models.charna import CharnaConfig, CharnaForMaskedLM  # noqa: E402


def test_self_tests():
    gbst_bert.self_test()


def test_same_weights_same_outputs_as_training_code():
    torch.manual_seed(0)
    reference = CharnaForMaskedLM(CharnaConfig()).eval()
    model = gbst_bert.GBSTBertForMaskedLM().eval()
    model.load_state_dict(reference.state_dict())  # strict: identical parameter names and shapes
    ids = torch.randint(7, 12, (2, 300))
    ids[:, ::7] = gbst_bert.MASK_ID
    with torch.no_grad():
        assert torch.equal(model(ids), reference(ids).logits)
