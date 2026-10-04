"""UL2 mixture-of-denoisers (arXiv 2205.05131) adapted to in-place masking for encoder-only MLM.

Mixture (UL2 Table 1), sampled uniformly ("All tasks have an approximate equal participation"):
    R: (mu=3, r=0.15), (mu=8, r=0.15)
    S: one span ending at the sequence end, mu=L/4, r=0.25
    X: (mu=3, r=0.5), (mu=8, r=0.5), (mu=64, r=0.15), (mu=64, r=0.5)

R/X spans use T5's `random_spans_noise_mask`, which UL2's reference code calls (UL2 Sec. 9.2):
an exact number of noise tokens round(L*r), split into round(n/mu) spans by a random partition.

Adaptations for an encoder-only model (UL2 is seq2seq with sentinel targets):
  * every masked character is replaced in place by the shared [MASK] token (ChaRNABERT
    Sec. 4.1, "shared g-masked token"), so the sequence length is unchanged; no 80/10/10;
  * the S-denoiser masks a suffix whose length is uniform on [1, L/2] (mean ~L/4). UL2 only
    says "uniform distribution" with mean L/4, so this range is our choice;
  * no mode tokens ([R]/[S]/[X]): ChaRNABERT does not use them and they would change the vocab.
"""

import torch

UL2_DENOISERS = (
    ("R", 3.0, 0.15),
    ("R", 8.0, 0.15),
    ("S", None, 0.25),
    ("X", 3.0, 0.5),
    ("X", 8.0, 0.5),
    ("X", 64.0, 0.15),
    ("X", 64.0, 0.5),
)


def _random_segmentation(num_items, num_segments, generator=None):
    """Partition `num_items` into `num_segments` non-empty segments; returns segment lengths."""
    first_in_segment = torch.zeros(num_items - 1, dtype=torch.long)
    first_in_segment[: num_segments - 1] = 1
    first_in_segment = first_in_segment[torch.randperm(num_items - 1, generator=generator)]
    segment_id = torch.cat([torch.zeros(1, dtype=torch.long), first_in_segment.cumsum(0)])
    return torch.bincount(segment_id, minlength=num_segments)


def random_spans_noise_mask(length, noise_density, mean_noise_span_length, generator=None):
    """Port of t5.data.preprocessors.random_spans_noise_mask. Returns a bool mask of `length`.

    The sequence always starts with a non-noise span; spans alternate non-noise / noise.
    """
    orig_length = length
    length = max(length, 2)
    num_noise_tokens = int(round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = max(int(round(num_noise_tokens / mean_noise_span_length)), 1)
    num_nonnoise_tokens = length - num_noise_tokens
    num_noise_spans = min(num_noise_spans, num_nonnoise_tokens)

    noise_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, generator)
    nonnoise_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, generator)
    interleaved = torch.stack([nonnoise_lengths, noise_lengths], dim=1).reshape(-1)
    span_starts = interleaved.cumsum(0)[:-1]
    indicator = torch.zeros(length, dtype=torch.long)
    indicator[span_starts] = 1
    is_noise = indicator.cumsum(0) % 2 == 1
    return is_noise[:orig_length]


def suffix_noise_mask(length, generator=None):
    """S-denoiser: mask a suffix of length uniform on [1, length // 2]."""
    suffix = int(torch.randint(1, max(length // 2, 1) + 1, (1,), generator=generator))
    mask = torch.zeros(length, dtype=torch.bool)
    mask[length - suffix:] = True
    return mask


def ul2_noise_mask(length, denoiser, generator=None):
    kind, mean_span, rate = denoiser
    if kind == "S":
        return suffix_noise_mask(length, generator)
    return random_spans_noise_mask(length, rate, mean_span, generator)


def ul2_getitem(seq, contains_eos=False, tokenizer=None, denoisers=UL2_DENOISERS, generator=None):
    """UL2 counterpart of `mlm_getitem`: returns (data, target) with target = pad_id at unmasked positions."""
    data = seq[:-1].clone() if contains_eos else seq.clone()
    target = data.clone()
    choice = int(torch.randint(len(denoisers), (1,), generator=generator))
    mask = ul2_noise_mask(data.size(0), denoisers[choice], generator)
    target[~mask] = tokenizer.pad_token_id
    data[mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    return data, target
