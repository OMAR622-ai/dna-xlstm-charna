"""Gradient-Based Subsequence Tokenization (GBST).

Follows ChaRNABERT (arXiv 2411.11808) Sec. 3.2, which modifies Charformer's GBST
(arXiv 2106.12672) Sec. 2.1:

  1. character embedding X (L x d)
  2. depthwise 1D conv with kernel size M                                  (Eq. 1)
  3. candidate blocks for every size b in 1..M and every offset o in 0..b-1,
     pooled with a non-parametric sum                                      (Eq. 2-3)
  4. linear block scorer F_R: R^d -> R, softmax over all candidates        (Eq. 4-5)
  5. position-wise score calibration  P <- softmax(P P^T) P               (Eq. 6)
  6. latent representation  X_hat_i = sum_k P_ik X_k,i                     (Eq. 7)
  7. optional mean-pool downsampling by d_s (Charformer Sec. 2.1.5; ChaRNABERT
     removes this step, d_s = 1 is the paper setting)

Offsets: Eq. 3 literally reads X[i+o : i+o+b], which for o >= 1 gives position i a
block that does not contain i. We use the blocks that contain i at each alignment,
X[i-o : i-o+b], which is the sliding-window reading of the text and Fig. 1. With
offsets disabled we fall back to Charformer's strided blocks (start = floor(i/b)*b).
Positions outside the sequence contribute zero to a block.
"""

import torch
import torch.nn.functional as F
from torch import nn


class GBST(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_block_size: int = 4,
        offsets: bool = True,
        pooling: str = "sum",
        conv_kernel_size: int = None,
        score_calibration: bool = True,
        downsample_rate: int = 1,
    ):
        super().__init__()
        self.max_block_size = max_block_size
        self.offsets = offsets
        self.pooling = pooling
        self.score_calibration = score_calibration
        self.downsample_rate = downsample_rate

        self.embedding = nn.Embedding(vocab_size, d_model)
        kernel = conv_kernel_size or max_block_size
        self.conv_pad = ((kernel - 1) // 2, kernel - 1 - (kernel - 1) // 2)  # "same" length
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model)
        self.scorer = nn.Linear(d_model, 1, bias=False)

        # (block size, offset) pairs; with offsets the offset is fixed per candidate,
        # without offsets each block size has a single strided candidate.
        if offsets:
            self.candidates = [(b, o) for b in range(1, max_block_size + 1) for o in range(b)]
        else:
            self.candidates = [(b, None) for b in range(1, max_block_size + 1)]

    @property
    def num_candidates(self):
        return len(self.candidates)

    def _block_bounds(self, b, o, length, device):
        """Start/end indices (into prefix sums of length L+1) of the block for each position."""
        pos = torch.arange(length, device=device)
        start = pos - o if o is not None else (pos // b) * b
        end = start + b
        return start.clamp(0, length), end.clamp(0, length)

    def forward(self, input_ids):
        """Returns (latents, char_latents).

        latents: (B, ceil(L / d_s), d), the encoder input.
        char_latents: (B, L, d), the latents before downsampling (equal to latents when d_s = 1).
        """
        x = self.embedding(input_ids)  # B L d
        length = x.size(1)
        x = self.conv(F.pad(x.transpose(1, 2), self.conv_pad)).transpose(1, 2)

        prefix = F.pad(x.cumsum(dim=1), (0, 0, 1, 0))  # B (L+1) d, prefix[:, j] = sum x[:, :j]
        prefix_scores = self.scorer(prefix).squeeze(-1)  # scorer is linear, so scores also telescope

        bounds, scores = [], []
        for b, o in self.candidates:
            start, end = self._block_bounds(b, o, length, x.device)
            scale = 1.0 / b if self.pooling == "mean" else 1.0
            bounds.append((start, end, scale))
            scores.append((prefix_scores[:, end] - prefix_scores[:, start]) * scale)
        probs = torch.stack(scores, dim=-1).softmax(dim=-1)  # B L K

        if self.score_calibration:
            probs = torch.softmax(probs @ probs.transpose(1, 2), dim=-1) @ probs

        latents = 0
        for k, (start, end, scale) in enumerate(bounds):
            block = (prefix[:, end] - prefix[:, start]) * scale
            latents = latents + probs[..., k:k + 1] * block

        char_latents = latents
        if self.downsample_rate > 1:
            latents = downsample(latents, self.downsample_rate)
        return latents, char_latents


def downsample(x, rate):
    """Mean-pool non-overlapping windows of `rate` positions (last window may be partial)."""
    length = x.size(1)
    pad = (-length) % rate
    summed = F.pad(x, (0, 0, 0, pad)).view(x.size(0), -1, rate, x.size(-1)).sum(dim=2)
    counts = F.pad(torch.ones(length, device=x.device, dtype=x.dtype), (0, pad)).view(-1, rate).sum(-1)
    return summed / counts[None, :, None]
