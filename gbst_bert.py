"""GBST-BERT: a learned tokenizer (GBST) in front of a bidirectional Transformer, for DNA.

Self-contained reference implementation of the model built in this project. It needs only
PyTorch (>= 2.1) and reproduces the trained model exactly: the weights of the project's final
checkpoint load into it without changes, and its outputs equal those of the training code in
`models/charna/`.

Model (ChaRNABERT, Morales-Pastor et al. 2024, arXiv:2411.11808, Sec. 3):

    DNA (1,024 bases)
      -> GBST tokenizer   embedding, local convolution (4 bases), candidate blocks of 1-4
                          bases, learned block scores, weighted mix          (Eq. 1-7)
      -> encoder          4 pre-norm blocks: attention with RoPE and QKNorm, SwiGLU (Eq. 9-14)
      -> head             masked-base prediction (pretraining) or a task label (fine-tuning)

Size: 2,107,392 parameters, of which 70,144 in GBST. The encoder has the shape of the
2-million-parameter Transformer++ of Bio-xLSTM (Schmidinger et al. 2024, arXiv:2411.04165,
Table A3), which serves as the control model without GBST.

Usage:
    python gbst_bert.py --test                                   # self-tests, no files needed
    python gbst_bert.py --checkpoint final_step10000.ckpt        # predict hidden bases
    python gbst_bert.py --checkpoint final_step10000.ckpt --sequence ACGT...

Training (pretraining on hg38 and fine-tuning on the Nucleotide Transformer benchmark) ran with
the DNA-xLSTM repository's training pipeline; see README.md for every setting and result.
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------------------------
# Tokenizer: one token per base (the DNA-xLSTM repository's character tokenizer)
# ---------------------------------------------------------------------------------------------

VOCAB = {"[CLS]": 0, "[SEP]": 1, "[BOS]": 2, "[MASK]": 3, "[PAD]": 4, "[RESERVED]": 5, "[UNK]": 6,
         "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
ID_TO_TOKEN = {i: t for t, i in VOCAB.items()}
MASK_ID, PAD_ID = VOCAB["[MASK]"], VOCAB["[PAD]"]
BASE_IDS = [VOCAB[b] for b in "ACGT"]
COMPLEMENT = {VOCAB["A"]: VOCAB["T"], VOCAB["C"]: VOCAB["G"], VOCAB["G"]: VOCAB["C"], VOCAB["T"]: VOCAB["A"]}


def encode(sequence: str) -> torch.Tensor:
    """DNA string -> token ids (L,). Letters other than A, C, G, T become N."""
    return torch.tensor([VOCAB.get(base, VOCAB["N"]) for base in sequence.upper()])


def decode(ids: torch.Tensor) -> str:
    return "".join(ID_TO_TOKEN[int(i)] if int(i) in BASE_IDS + [VOCAB["N"]] else "?" for i in ids)


def reverse_complement(ids: torch.Tensor) -> torch.Tensor:
    """Reverse complement of token ids (..., L); non-base tokens keep their id."""
    lookup = torch.arange(len(VOCAB), device=ids.device)
    for base, comp in COMPLEMENT.items():
        lookup[base] = comp
    return lookup[ids].flip(-1)


# ---------------------------------------------------------------------------------------------
# Configuration (defaults = the trained model)
# ---------------------------------------------------------------------------------------------

@dataclass
class GBSTBertConfig:
    vocab_size: int = len(VOCAB)
    d_model: int = 256            # Bio-xLSTM Table A3 (Transformer++-2M)
    n_layer: int = 4
    n_heads: int = 8
    hidden_dim: int = 320         # SwiGLU width, 1.25 x d_model (Table A3)
    max_length: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0   # ChaRNABERT Eq. 13
    max_block_size: int = 4       # largest GBST block (Charformer; not stated in ChaRNABERT)
    tokenizer: str = "gbst"       # "gbst", or "embedding" for the ablation without GBST


# ---------------------------------------------------------------------------------------------
# GBST: Gradient-Based Subword Tokenization (ChaRNABERT Sec. 3.2)
# ---------------------------------------------------------------------------------------------

class GBST(nn.Module):
    """Learned soft tokenization at single-base resolution.

    For every position i, GBST forms candidate blocks of b = 1..M bases that contain i (one per
    offset, M(M+1)/2 = 10 candidates for M = 4), sums the base vectors in each block (Eq. 2-3),
    scores each block with a linear layer (Eq. 4), turns the scores into weights with a softmax
    (Eq. 5), calibrates them (Eq. 6), and returns the weighted sum of the blocks (Eq. 7).
    There is no downsampling, so the output has one vector per base.

    Two reading choices (documented in README.md, "Differences from the papers"):
      - Eq. 1 is a depthwise convolution; ChaRNABERT Table 9 counts d*M + d^2 weights for it,
        so a pointwise d x d layer follows the depthwise convolution.
      - Eq. 3 is read as the blocks that contain position i (X[i-o : i-o+b]).
    """

    def __init__(self, vocab_size: int, d_model: int, max_block_size: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        k = max_block_size
        self.conv_pad = ((k - 1) // 2, k - 1 - (k - 1) // 2)  # keep the sequence length
        self.conv = nn.Conv1d(d_model, d_model, k, groups=d_model)  # depthwise (Eq. 1)
        self.pointwise = nn.Linear(d_model, d_model, bias=False)     # pointwise (Table 9)
        self.scorer = nn.Linear(d_model, 1, bias=False)               # F_R (Eq. 4)
        self.candidates = [(b, o) for b in range(1, max_block_size + 1) for o in range(b)]

    def block_weights(self, x: torch.Tensor):
        """Base vectors after the convolution (B, L, d), their prefix sums, and the calibrated
        weight of every candidate block at every position (B, L, K)."""
        x = self.conv(F.pad(x.transpose(1, 2), self.conv_pad)).transpose(1, 2)
        x = self.pointwise(x)
        prefix = F.pad(x.cumsum(dim=1), (0, 0, 1, 0))       # prefix[:, j] = sum of x[:, :j]
        prefix_scores = self.scorer(prefix).squeeze(-1)    # the scorer is linear: scores telescope too
        positions = torch.arange(x.size(1), device=x.device)
        bounds, scores = [], []
        for b, o in self.candidates:
            start = (positions - o).clamp(0, x.size(1))   # block [i-o, i-o+b), cut at the ends
            end = (positions - o + b).clamp(0, x.size(1))
            bounds.append((start, end))
            scores.append(prefix_scores[:, end] - prefix_scores[:, start])
        weights = torch.stack(scores, dim=-1).softmax(dim=-1)                          # Eq. 5
        weights = torch.softmax(weights @ weights.transpose(1, 2), dim=-1) @ weights    # Eq. 6
        return prefix, bounds, weights

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        prefix, bounds, weights = self.block_weights(self.embedding(input_ids))
        out = 0
        for k, (start, end) in enumerate(bounds):                                      # Eq. 7
            out = out + weights[..., k:k + 1] * (prefix[:, end] - prefix[:, start])
        return out

    def block_size_weights(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Share of the weight on blocks of 1, 2, ..., M bases at each position (B, L, M)."""
        _, _, weights = self.block_weights(self.embedding(input_ids))
        sizes = torch.tensor([b for b, _ in self.candidates], device=weights.device)
        return torch.stack([weights[..., sizes == b].sum(-1) for b in range(1, int(sizes.max()) + 1)], dim=-1)


# ---------------------------------------------------------------------------------------------
# Encoder: pre-norm Transformer blocks with RoPE, QKNorm and SwiGLU (ChaRNABERT Fig. 1)
# ---------------------------------------------------------------------------------------------

def rotary_frequencies(head_dim: int, length: int, theta: float):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(length).float(), freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each pair of features of x (B, L, H, D) by a position-dependent angle (RoPE)."""
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    rotated = torch.stack([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
    return rotated.flatten(3).type_as(x)


class SelfAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE (Eq. 13) and QKNorm (Eq. 14)."""

    def __init__(self, config: GBSTBertConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=config.norm_eps, bias=False)

    def forward(self, x, cos, sin):
        b, length, _ = x.shape
        shape = (b, length, self.n_heads, self.head_dim)
        q = apply_rotary(self.q_norm(self.wq(x).view(shape)), cos, sin)
        k = apply_rotary(self.k_norm(self.wk(x).view(shape)), cos, sin)
        v = self.wv(x).view(shape)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.wo(out.transpose(1, 2).reshape(b, length, -1))


class SwiGLU(nn.Module):
    """Feed-forward layer w2(silu(w1 x) * w3 x) (Eq. 9, standard SwiGLU)."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderBlock(nn.Module):
    """x + attention(norm(x)), then x + SwiGLU(norm(x))."""

    def __init__(self, config: GBSTBertConfig):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=False)
        self.attention = SelfAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=False)
        self.feed_forward = SwiGLU(config.d_model, config.hidden_dim)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        return x + self.feed_forward(self.ffn_norm(x))


class GBSTBertEncoder(nn.Module):
    """Token ids (B, L) -> one vector per base (B, L, d)."""

    def __init__(self, config: GBSTBertConfig):
        super().__init__()
        self.config = config
        if config.tokenizer == "gbst":
            self.gbst = GBST(config.vocab_size, config.d_model, config.max_block_size)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps, bias=False)
        cos, sin = rotary_frequencies(config.d_model // config.n_heads, config.max_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.gbst(input_ids) if self.config.tokenizer == "gbst" else self.embedding(input_ids)
        length = input_ids.size(1)
        for layer in self.layers:
            h = layer(h, self.rope_cos[:length], self.rope_sin[:length])
        return self.norm(h)


# ---------------------------------------------------------------------------------------------
# Pretraining model: masked language modelling (MLM)
# ---------------------------------------------------------------------------------------------

class GBSTBertForMaskedLM(nn.Module):
    """Encoder + linear head that scores every vocabulary token at every position."""

    def __init__(self, config: GBSTBertConfig = GBSTBertConfig()):
        super().__init__()
        self.config = config
        self.backbone = GBSTBertEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 style scaling of residual projections
            if name.endswith(("w3.weight", "wo.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids (B, L) -> logits (B, L, vocab)."""
        return self.lm_head(self.backbone(input_ids))


def mask_for_mlm(ids: torch.Tensor, probability: float = 0.15, generator=None):
    """BERT masking as used in pretraining: choose 15% of positions; of these, 80% become [MASK],
    10% a random token and 10% stay unchanged. Returns (inputs, targets); targets are [PAD]
    (ignored by the loss) everywhere except at the chosen positions."""
    rand = lambda: torch.rand(ids.shape, generator=generator, device=ids.device)  # noqa: E731
    chosen = rand() < probability
    targets = ids.masked_fill(~chosen, PAD_ID)
    to_mask = (rand() < 0.8) & chosen
    to_random = (rand() < 0.5) & chosen & ~to_mask
    random_tokens = torch.randint(len(VOCAB), ids.shape, generator=generator, device=ids.device)
    inputs = ids.masked_fill(to_mask, MASK_ID)
    inputs = torch.where(to_random, random_tokens, inputs)
    return inputs, targets


def mlm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over the chosen positions (in nats; random guessing among 4 bases = 1.386)."""
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_ID)


def load_pretrained(checkpoint_path: str, config: GBSTBertConfig = GBSTBertConfig()) -> GBSTBertForMaskedLM:
    """Load a pretraining checkpoint of this project (PyTorch Lightning .ckpt or a plain state dict)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state):  # Lightning: keep the model's weights only
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    model = GBSTBertForMaskedLM(config)
    model.load_state_dict(state)  # strict: every weight must match
    return model.eval()


@torch.no_grad()
def predict_hidden_bases(model: GBSTBertForMaskedLM, ids: torch.Tensor, hidden: torch.Tensor):
    """Hide the bases where `hidden` is True and let the model guess them.
    Returns (guessed token ids (L,), probability of each base A/C/G/T (L, 4))."""
    logits = model(ids.masked_fill(hidden, MASK_ID)[None])[0][:, BASE_IDS]
    probs = logits.softmax(-1)
    return torch.tensor(BASE_IDS)[probs.argmax(-1)], probs


# ---------------------------------------------------------------------------------------------
# Fine-tuning model: one label per sequence (Nucleotide Transformer benchmark, Bio-xLSTM App. C.5)
# ---------------------------------------------------------------------------------------------

class GBSTBertForSequenceClassification(nn.Module):
    """Pretrained encoder + mean pooling over positions + linear layer.

    With `rc_average` (the "PH" setting of Bio-xLSTM), the prediction is the average of the
    predictions for the sequence and its reverse complement. In fine-tuning, the whole model is
    trained for 20 epochs and the epoch with the best validation score is tested.
    """

    def __init__(self, encoder: GBSTBertEncoder, num_classes: int, rc_average: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.d_model, num_classes)
        self.rc_average = rc_average

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.encoder(input_ids).mean(dim=1))
        if self.rc_average:
            logits = (logits + self.head(self.encoder(reverse_complement(input_ids)).mean(dim=1))) / 2
        return logits


# ---------------------------------------------------------------------------------------------
# Self-tests and demo
# ---------------------------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def self_test():
    torch.manual_seed(0)
    model = GBSTBertForMaskedLM().eval()
    gbst = model.backbone.gbst

    assert count_parameters(model) == 2_107_392, count_parameters(model)
    assert count_parameters(gbst) == 70_144  # 3,072 + 1,024 + 256 + 65,536 + 256
    assert len(gbst.candidates) == 10

    ids = encode("ACGTTGCAAN" * 20)[None]
    logits = model(ids)
    assert logits.shape == (1, 200, len(VOCAB))

    weights = gbst.block_size_weights(ids)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 200), atol=1e-5)

    # changing one base changes the output at that position (single-base resolution)
    snp = ids.clone()
    snp[0, 100] = VOCAB["G"] if snp[0, 100] != VOCAB["G"] else VOCAB["A"]
    assert not torch.allclose(model(snp)[0, 100], logits[0, 100])

    assert torch.equal(reverse_complement(encode("AACGTN")), encode("NACGTT"))

    inputs, targets = mask_for_mlm(torch.full((64, 1024), VOCAB["A"]), generator=torch.Generator().manual_seed(1))
    chosen = targets != PAD_ID
    assert abs(chosen.float().mean().item() - 0.15) < 0.01
    assert abs((inputs[chosen] == MASK_ID).float().mean().item() - 0.8) < 0.02
    loss = mlm_loss(model(inputs[:2]), targets[:2])
    assert abs(loss.item() - math.log(len(VOCAB))) < 0.1  # untrained model: close to uniform

    classifier = GBSTBertForSequenceClassification(model.backbone, num_classes=2)
    assert classifier(ids).shape == (1, 2)

    ablation = GBSTBertForMaskedLM(GBSTBertConfig(tokenizer="embedding"))
    assert count_parameters(ablation) == 2_107_392 - 70_144 + 3_072
    print("all self-tests passed: 2,107,392 parameters (GBST 70,144), shapes, GBST weights, "
          "single-base resolution, reverse complement, MLM masking, classifier, ablation")


def demo(checkpoint: str, sequence: str = None, seed: int = 0):
    model = load_pretrained(checkpoint)
    print(f"loaded {checkpoint}: {count_parameters(model):,} parameters")
    if sequence is None:  # hg38 chr21:33,045,477-33,045,596, with a (GAAA)n tandem repeat
        sequence = ("CCCTGTAAAAAGAAAAAGAAAGAAAGAGAGAGAGAGAAAGGAAGGACGGAAGGAAGGAAGAAAGAAAGAAAGAAAG"
                    "AAAGAAAGAAAGAAAGAAAGAAAGAAAGAAATTAACATGGTTAC")
    ids = encode(sequence[:1024])
    hidden = (torch.rand(ids.shape, generator=torch.Generator().manual_seed(seed)) < 0.15) & (ids != VOCAB["N"])
    guess, probs = predict_hidden_bases(model, ids, hidden)
    correct = (guess[hidden] == ids[hidden]).float().mean().item()
    print(f"hidden {int(hidden.sum())} of {len(ids)} bases; {100 * correct:.0f}% guessed right (random: 25%)")
    print("true    " + decode(ids))
    print("input   " + "".join("?" if h else c for h, c in zip(hidden, decode(ids))))
    print("guess   " + "".join(decode(g[None]) if h else " " for h, g in zip(hidden, guess)))
    size_share = model.backbone.gbst.block_size_weights(ids[None])[0].mean(0)
    print("GBST weight on blocks of 1/2/3/4 bases: " + " / ".join(f"{100 * w:.0f}%" for w in size_share))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test", action="store_true", help="run the self-tests")
    parser.add_argument("--checkpoint", help="pretrained checkpoint, e.g. final_step10000.ckpt")
    parser.add_argument("--sequence", help="DNA sequence for the demo (up to 1,024 bases)")
    args = parser.parse_args()
    if args.test or not args.checkpoint:
        self_test()
    if args.checkpoint:
        demo(args.checkpoint, args.sequence)
