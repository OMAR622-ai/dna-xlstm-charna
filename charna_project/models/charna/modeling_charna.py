"""ChaRNABERT-style model: GBST tokenizer + bidirectional BERT encoder (arXiv 2411.11808).

The encoder reuses the Transformer++ (Llama) building blocks of this repo (RoPE, SwiGLU
feed-forward, init scheme) and adds QKNorm (ChaRNABERT Eq. 14). With the default config it
has the same shape as DNA-Transformer++-2M (Bio-xLSTM Table A3), so differences against that
baseline come from GBST, QKNorm and the norm type.

The MLM head predicts every character position. When GBST downsamples (d_s > 1, a
Charformer-style ablation that ChaRNABERT does not use), encoder states are repeated d_s
times, added to GBST's character-level latents, normalised and projected. This upsampling
head is our own design: neither paper defines a character-level output after downsampling.
"""

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.llama.modelling_llama import FeedForward, RMSNorm, apply_rotary_emb, precompute_freqs_cis

from .configuration_charna import CharnaConfig
from .gbst import GBST

LMOutput = namedtuple("LMOutput", ["logits"])


def make_norm(config: CharnaConfig, dim: int):
    if config.norm_type == "rmsnorm":
        return RMSNorm(dim, eps=config.norm_eps)
    return nn.LayerNorm(dim, eps=config.norm_eps, bias=False)


class SelfAttention(nn.Module):
    """Bidirectional multi-head attention with RoPE and optional QKNorm."""

    def __init__(self, config: CharnaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)
        self.q_norm = make_norm(config, self.head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = make_norm(config, self.head_dim) if config.qk_norm else nn.Identity()

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape
        q = self.q_norm(self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim))
        k = self.k_norm(self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim))
        v = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        return self.wo(out.transpose(1, 2).reshape(bsz, seqlen, -1))


class EncoderBlock(nn.Module):
    """Pre-norm block (ChaRNABERT Fig. 1): norm -> attention -> residual, norm -> SwiGLU -> residual."""

    def __init__(self, config: CharnaConfig):
        super().__init__()
        self.attention_norm = make_norm(config, config.d_model)
        self.attention = SelfAttention(config)
        self.ffn_norm = make_norm(config, config.d_model)
        self.feed_forward = FeedForward(config.d_model, config.hidden_dim, multiple_of=8, dropout=config.dropout)

    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        return x + self.feed_forward(self.ffn_norm(x))


class CharnaModel(nn.Module):
    """Tokenizer (GBST or plain embedding) + encoder. Returns final-norm hidden states."""

    def __init__(self, config: CharnaConfig):
        super().__init__()
        self.config = config
        if config.tokenizer == "gbst":
            self.gbst = GBST(
                config.vocab_size, config.d_model,
                max_block_size=config.gbst_max_block_size,
                offsets=config.gbst_offsets,
                pooling=config.gbst_pooling,
                conv_kernel_size=config.gbst_conv_kernel_size,
                pointwise_conv=config.gbst_pointwise_conv,
                score_calibration=config.gbst_score_calibration,
                downsample_rate=config.downsample_rate,
            )
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(EncoderBlock(config) for _ in range(config.n_layer))
        self.norm = make_norm(config, config.d_model)
        freqs_cos, freqs_sin = precompute_freqs_cis(config.d_model // config.n_heads, config.max_length, config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, return_char_latents=False):
        if self.config.tokenizer == "gbst":
            h, char_latents = self.gbst(input_ids)
        else:
            h = char_latents = self.embedding(input_ids)
        h = self.dropout(h)
        seqlen = h.size(1)
        freqs_cos, freqs_sin = self.freqs_cos[:seqlen], self.freqs_sin[:seqlen]
        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)
        return (h, char_latents) if return_char_latents else h


class CharnaForMaskedLM(nn.Module):
    """Pretraining model: logits over the vocabulary at every character position."""

    def __init__(self, config: CharnaConfig, device=None, dtype=None):
        super().__init__()
        self.config = config
        self.backbone = CharnaModel(config)
        if config.downsample_rate > 1:
            self.upsample_norm = make_norm(config, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Same residual-projection scaling as the repo's Transformer++ (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith("w3.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if device is not None:
            self.to(device)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def headless_forward(self, input_ids):
        """Encoder states for downstream tasks: (B, ceil(L / d_s), d)."""
        return self.backbone(input_ids)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        h, char_latents = self.backbone(input_ids, return_char_latents=True)
        rate = self.config.downsample_rate
        if rate > 1:
            h = h.repeat_interleave(rate, dim=1)[:, :input_ids.size(1)]
            h = self.upsample_norm(h + char_latents)
        return LMOutput(logits=self.lm_head(h))


def reverse_complement_ids(input_ids, complement_map):
    """Reverse-complement token ids (B, L). `complement_map` maps token id -> complement id."""
    lut = torch.arange(max(complement_map) + 1, device=input_ids.device)
    for k, v in complement_map.items():
        lut[k] = v
    return lut[input_ids].flip(-1)


def rc_averaged_logits(model, input_ids, complement_map):
    """Post-hoc RC-averaged MLM logits: mean of f(x) and RC-mapped f(rc(x)).

    Position i of rc(x) corresponds to position L-1-i of x, and its prediction for base c
    is a prediction for complement(c) on the forward strand.
    """
    logits = model(input_ids).logits
    rc_logits = model(reverse_complement_ids(input_ids, complement_map)).logits.flip(1)
    perm = torch.arange(logits.size(-1), device=logits.device)
    for k, v in complement_map.items():
        perm[k] = v
    return (logits + rc_logits[..., perm]) / 2


def conjoined_hidden_states(backbone_fn, input_ids):
    """Post-hoc conjoining input (B, L, 2) = (forward, RC) -> hidden states (B, L', d, 2).

    Same contract as the repo's Caduceus/xLSTM embedding wrappers; the sequence decoder
    pools each channel, applies the head and averages the two.
    """
    assert input_ids.ndim == 3 and input_ids.size(-1) == 2, "expected (B, L, 2) forward/RC input"
    return torch.stack([backbone_fn(input_ids[..., 0]), backbone_fn(input_ids[..., 1])], dim=-1)


def parameter_breakdown(model: nn.Module):
    """Parameter counts grouped by component (top two name levels)."""
    groups = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:2]) if parts[0] == "backbone" and parts[1] != "layers" else parts[0]
        if parts[0] == "backbone" and parts[1] == "layers":
            key = f"backbone.layers.{parts[2]}"
        groups[key] = groups.get(key, 0) + p.numel()
    groups["total"] = sum(p.numel() for p in model.parameters())
    return groups
