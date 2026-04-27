# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for HunYuanVL-MoT (Mixture of Tokens) model."""

from typing import Optional
from transformers import PretrainedConfig


class HunYuanVLMoTConfig(PretrainedConfig):
    """Config class for HunYuanVLMoT multimodal model.

    Corresponds to tencent/HY-Embodied-0.5 (model_type=hunyuan_vl_mot).
    """
    model_type = "hunyuan_vl_mot"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: Optional[int] = 120818,
        org_vocab_size: Optional[int] = 120818,
        hidden_size: Optional[int] = 2048,
        intermediate_size: Optional[int] = 6144,
        num_hidden_layers: Optional[int] = 32,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = 4,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 262144,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        pad_token_id: Optional[int] = 120002,
        bos_token_id: Optional[int] = 120000,
        eos_token_id: Optional[int] = 120020,
        tie_word_embeddings: Optional[bool] = True,
        pretraining_tp: Optional[int] = 1,
        # RoPE — config.json uses rope_theta + rope_scaling separately
        rope_theta: Optional[float] = 10000.0,
        rope_scaling: Optional[dict] = None,
        # kept for backward compat with older checkpoints
        rope_parameters: Optional[dict] = None,
        attention_bias: Optional[bool] = False,
        attention_dropout: Optional[float] = 0.0,
        head_dim: Optional[int] = 128,
        attention_head_dim: Optional[int] = 128,
        mlp_bias: Optional[bool] = False,
        # MoT-specific
        use_qk_norm: Optional[bool] = True,
        use_rotary_pos_emb: Optional[bool] = True,
        use_cla: Optional[bool] = False,
        cla_share_factor: Optional[int] = 1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.org_vocab_size = org_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.attention_head_dim = attention_head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        # RoPE: prefer rope_theta/rope_scaling style (matches config.json)
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        # legacy field — kept so older code paths that read rope_parameters still work
        #self.rope_parameters = rope_parameters
        self.use_qk_norm = use_qk_norm
        self.use_rotary_pos_emb = use_rotary_pos_emb
        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["HunYuanVLMoTConfig"]
