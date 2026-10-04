"""Configuration for the ChaRNABERT-style GBST + BERT model (arXiv 2411.11808).

Defaults reproduce the DNA-Transformer++-2M backbone of Bio-xLSTM (Table A3: d_model 256,
4 blocks, 8 heads, SwiGLU up-projection 1.25:1, no biases, RoPE) so that a comparison
against Transformer++ isolates the tokenizer. GBST defaults follow ChaRNABERT Sec. 3.2
(sum pooling, offsets, score calibration, no downsampling) with max block size 4 from
Charformer (Fig. 2 / Sec. 7.4), since ChaRNABERT does not state M.
"""

from transformers import PretrainedConfig


class CharnaConfig(PretrainedConfig):

    model_type = "charna"

    def __init__(
        self,
        d_model: int = 256,
        n_layer: int = 4,
        n_heads: int = 8,
        hidden_dim: int = 320,  # SwiGLU width, 1.25 * d_model (Bio-xLSTM Table A3)
        vocab_size: int = 12,
        max_length: int = 1024,
        norm_type: str = "layernorm",  # "layernorm" (ChaRNABERT Fig. 1) | "rmsnorm" (Transformer++)
        norm_eps: float = 1e-5,
        qk_norm: bool = True,  # ChaRNABERT Eq. 14
        rope_theta: float = 10000.0,  # ChaRNABERT Eq. 13
        dropout: float = 0.0,
        tokenizer: str = "gbst",  # "gbst" | "embedding" (the paper's EM ablation)
        gbst_max_block_size: int = 4,
        gbst_offsets: bool = True,  # ChaRNABERT Eq. 3; False = Charformer strided blocks
        gbst_pooling: str = "sum",  # ChaRNABERT Eq. 2; Charformer uses "mean"
        gbst_conv_kernel_size: int = None,  # None = max block size (ChaRNABERT Eq. 1)
        gbst_score_calibration: bool = True,  # ChaRNABERT Eq. 6
        downsample_rate: int = 1,  # ChaRNABERT removes downsampling; >1 is the Charformer ablation
        gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert tokenizer in ("gbst", "embedding"), tokenizer
        assert norm_type in ("layernorm", "rmsnorm"), norm_type
        assert gbst_pooling in ("sum", "mean"), gbst_pooling
        assert d_model % n_heads == 0
        if tokenizer == "embedding":
            assert downsample_rate == 1, "downsampling requires the GBST tokenizer"
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.tokenizer = tokenizer
        self.gbst_max_block_size = gbst_max_block_size
        self.gbst_offsets = gbst_offsets
        self.gbst_pooling = gbst_pooling
        self.gbst_conv_kernel_size = gbst_conv_kernel_size
        self.gbst_score_calibration = gbst_score_calibration
        self.downsample_rate = downsample_rate
        self.gradient_checkpointing = gradient_checkpointing
