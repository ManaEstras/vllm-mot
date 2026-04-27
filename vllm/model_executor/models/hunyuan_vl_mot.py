# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# coding=utf-8
# Copyright 2024 The HunYuan team. Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Inference-only HunYuanVL-MoT (Mixture of Tokens) model."""

import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from transformers import BatchFeature
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize,
)
from transformers.video_utils import VideoMetadata

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)

from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    VideoItem,
)
from vllm.multimodal.parse import (
    ImageSize,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType
from vllm.vllm_flash_attn import flash_attn_varlen_func

from .interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .module_mapping import MultiModelKeys
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)


# ---------------------------------------------------------------------------
# Special token IDs (MoT variant)
# ---------------------------------------------------------------------------
IMAGE_TOKEN_ID = 120687
VIDEO_TOKEN_ID = 120688
NEWLINE_TOKEN_ID = 120689
VISION_START_TOKEN_ID = 120684
VISION_END_TOKEN_ID = 120685
LATENT_TOKEN_ID = 120690


# ---------------------------------------------------------------------------
# MoT routing utilities
# ---------------------------------------------------------------------------

def modality_mask_to_segments(mask: torch.Tensor) -> torch.Tensor:
    """Convert boolean modality mask to visual segment (start, end) pairs.

    Args:
        mask: shape (slen,) or (1, slen), True = vision token.
    Returns:
        Tensor of shape (N, 2) with [start, end) pairs for each visual segment.
    """
    if mask.dim() == 2:
        if mask.size(0) != 1:
            raise ValueError("Batch size greater than 1 not supported")
        mask = mask[0]
    mask = mask.to(torch.int64)
    slen = mask.numel()
    is_zero = (mask == 0).to(torch.int64)
    padded = torch.cat([
        torch.tensor([0], device=mask.device),
        is_zero,
        torch.tensor([0], device=mask.device),
    ])
    diff = padded[1:] - padded[:-1]
    zero_run_starts = (diff == 1).nonzero(as_tuple=True)[0]
    zero_run_ends = (diff == -1).nonzero(as_tuple=True)[0] - 1

    separators = []
    for s, e in zip(zero_run_starts, zero_run_ends):
        if (e - s + 1) >= 2:
            separators.append((s, e))

    segments: list[list[int]] = []
    seg_start = 0
    for s, e in separators:
        seg_end = s - 1
        if seg_end >= seg_start:
            segments.append([seg_start, seg_end])
        seg_start = e + 1
    if seg_start < slen:
        segments.append([seg_start, slen - 1])

    for i in range(len(segments)):
        segments[i][1] = segments[i][1] + 2  # make end exclusive
        if segments[i][1] > slen:
            segments[i][1] = slen

    return torch.tensor(segments, device=mask.device)


def mask_apply_no_batch(
    hidden_states: torch.Tensor,
    mask: Optional[torch.Tensor],
    text_func_list: list[Callable],
    vision_func_list: list[Callable],
    out_dims: Optional[list[int]] = None,
) -> list[torch.Tensor]:
    """Route tokens to modality-specific functions (no batch dimension).

    Args:
        hidden_states: (S, D)
        mask: (S,) bool/int — True/1 = vision token. None → text-only path.
        text_func_list: functions for text tokens.
        vision_func_list: functions for vision tokens.
        out_dims: output dimensions for each output tensor. None = same as input.
    Returns:
        List of output tensors, one per function pair.
    """
    if mask is None:
        results = []
        for func in text_func_list:
            r = func(hidden_states)
            results.append(r[0] if isinstance(r, tuple) else r)
        return results

    S, D = hidden_states.size()
    mask_flat = mask.reshape(S).bool()

    if out_dims is None:
        out_flat = [torch.empty_like(hidden_states) for _ in text_func_list]
    else:
        out_flat = [
            torch.empty(S, od, device=hidden_states.device, dtype=hidden_states.dtype)
            for od in out_dims
        ]

    text_idx = ~mask_flat
    hs_t = hidden_states[text_idx]
    if hs_t.shape[0] > 0:
        for i, func in enumerate(text_func_list):
            result = func(hs_t)
            if isinstance(result, tuple):
                result = result[0]
            out_flat[i][text_idx] = result

    vis_idx = mask_flat
    hs_v = hidden_states[vis_idx]
    if hs_v.shape[0] > 0:
        for i, func in enumerate(vision_func_list):
            result = func(hs_v)
            if isinstance(result, tuple):
                result = result[0]
            out_flat[i][vis_idx] = result

    return [o.view(S, -1) for o in out_flat]


def mask_apply_no_batch_dual(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    mask: Optional[torch.Tensor],
    text_func_list: list[Callable],
    vision_func_list: list[Callable],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens + residual to modality-specific functions.

    Args:
        hidden_states: (S, D)
        residual: (S, D)
        mask: (S,) bool/int. None → text-only path.
    Returns:
        Tuple of (hidden_states, residual) after routing.
    """
    if mask is None:
        return text_func_list[0](hidden_states, residual)

    S, D = hidden_states.size()
    mask_flat = mask.reshape(S).bool()

    out_flat = torch.empty_like(hidden_states)
    res_flat = torch.empty_like(residual)

    text_idx = ~mask_flat
    hs_t = hidden_states[text_idx]
    if hs_t.shape[0] > 0:
        h_t, r_t = text_func_list[0](hs_t, residual[text_idx])
        out_flat[text_idx] = h_t
        res_flat[text_idx] = r_t

    vis_idx = mask_flat
    hs_v = hidden_states[vis_idx]
    if hs_v.shape[0] > 0:
        h_v, r_v = vision_func_list[0](hs_v, residual[vis_idx])
        out_flat[vis_idx] = h_v
        res_flat[vis_idx] = r_v

    return out_flat.view(S, D), res_flat.view(S, D)


# ---------------------------------------------------------------------------
# MoT MLP
# ---------------------------------------------------------------------------

class HunYuanMoTMLP(nn.Module):
    """Feed-forward network used in MoT decoder layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            reduce_results=reduce_results,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoT Attention
# ---------------------------------------------------------------------------

def _get_cla_factor(config: Any) -> int:
    if not getattr(config, "use_cla", False):
        return 1
    return getattr(config, "cla_share_factor", 1)


def _build_rope_parameters(
    config: Any,
    rope_theta: float,
    rope_scaling: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Build rope_parameters dict for get_rope() from legacy theta/scaling args.

    The official vLLM get_rope() accepts a unified rope_parameters dict.
    If config already has rope_parameters, use that directly.
    Otherwise construct it from rope_theta and rope_scaling.
    """
    #if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
    #    return config.rope_parameters

    params: dict[str, Any] = {"rope_theta": rope_theta}
    if rope_scaling is not None:
        # rope_scaling typically has keys like 'type', 'factor',
        # 'original_max_position_embeddings', etc.
        params.update(rope_scaling)
        # Normalise 'type' -> 'rope_type' as expected by get_rope()
        if "type" in params and "rope_type" not in params:
            params["rope_type"] = params.pop("type")
    return params


class HunYuanMoTAttention(nn.Module):
    """Multi-headed attention with dual text/vision projection paths (MoT)."""

    def __init__(
        self,
        config: Any,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if hasattr(config, "head_dim") and config.head_dim:
            self.head_dim = config.head_dim
        elif hasattr(config, "attention_head_dim"):
            self.head_dim = config.attention_head_dim
        else:
            self.head_dim = hidden_size // self.total_num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        # Text path projections
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Vision path projections
        self.qkv_proj_v = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_v",
        )
        self.o_proj_v = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_v",
        )

        rope_parameters = _build_rope_parameters(config, rope_theta, rope_scaling)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @staticmethod
    def _apply_visual_noncausal_attn(
        attn_output: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        visual_segs: torch.Tensor,
    ) -> torch.Tensor:
        """Override causal attention output for visual segments with non-causal
        flash attention.  Mirrors the two-pass logic in vllm_embodied's
        attention/layer.py, but implemented at the model layer so the official
        vLLM Attention op is not modified.

        Strategy (same as vllm_embodied):
          1. self.attn(q, k, v) already ran causal attention for the full
             sequence (with KV-cache management handled by vLLM).
          2. For each visual segment [s, e), re-run flash_attn_varlen_func
             with causal=False and overwrite the corresponding rows in
             attn_output.

        Args:
            attn_output: (S, num_heads * head_dim) — causal attention output
                         already reshaped from self.attn().
            q, k, v:     (S, num_heads/kv_heads, head_dim) — reshaped tensors
                         as used in flash_attn (seq-first, no batch dim).
            visual_segs: (N, 2) int64 tensor of [start, end) pairs.
        Returns:
            attn_output with visual-segment rows replaced by non-causal attn.
        """

        visual_qs, visual_ks, visual_vs = [], [], []
        visual_mask = torch.zeros(
            q.shape[0], dtype=torch.bool, device=q.device
        )
        cu_v: list[int] = [0]
        max_v_len = 0

        for seg in visual_segs:
            s, e = int(seg[0].item()), int(seg[1].item())
            visual_qs.append(q[s:e])
            visual_ks.append(k[s:e])
            visual_vs.append(v[s:e])
            visual_mask[s:e] = True
            cu_v.append(cu_v[-1] + (e - s))
            if e - s > max_v_len:
                max_v_len = e - s

        if max_v_len == 0:
            return attn_output

        vq = torch.cat(visual_qs, dim=0)
        vk = torch.cat(visual_ks, dim=0)
        vv = torch.cat(visual_vs, dim=0)
        cu_v_seqlens = torch.tensor(
            cu_v, dtype=torch.int32, device=q.device
        )

        visual_attn_out = flash_attn_varlen_func(
            vq, vk, vv,
            cu_seqlens_q=cu_v_seqlens,
            cu_seqlens_k=cu_v_seqlens,
            max_seqlen_q=max_v_len,
            max_seqlen_k=max_v_len,
            causal=False,
        )  # (total_visual_tokens, num_heads, head_dim)

        # Reshape to match attn_output's last dim and overwrite
        S_v, H, D = visual_attn_out.shape
        attn_output = attn_output.clone()
        attn_output[visual_mask] = visual_attn_out.reshape(S_v, H * D)
        return attn_output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        visual_segs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Dual-path QKV projection
        qkv = mask_apply_no_batch(
            hidden_states,
            modality_mask,
            [lambda x: self.qkv_proj(x)],
            [lambda x: self.qkv_proj_v(x)],
            out_dims=[self.q_size + 2 * self.kv_size],
        )[0]

        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        ori_k = k

        if self.use_qk_norm:
            q = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous()
            )
            k = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
            )

        # Step 1: full-sequence causal attention via official vLLM (handles KV cache)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(q.shape[0], -1)

        # Step 2: override visual segments with non-causal flash attention.
        # This mirrors vllm_embodied/vllm/attention/layer.py's two-pass strategy,
        # implemented here at the model layer to avoid patching the Attention op.
        if visual_segs is not None and len(visual_segs) > 0:
            # Reshape q/k/v to (S, num_heads, head_dim) for flash_attn
            q_fa = q.view(q.shape[0], self.num_heads, self.head_dim)
            k_fa = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
            v_fa = v.view(v.shape[0], self.num_kv_heads, self.head_dim)
            # Expand k/v heads if GQA (num_kv_heads < num_heads)
            if self.num_kv_heads < self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_fa = k_fa.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(
                    k_fa.shape[0], self.num_heads, self.head_dim
                )
                v_fa = v_fa.unsqueeze(2).expand(-1, -1, n_rep, -1).reshape(
                    v_fa.shape[0], self.num_heads, self.head_dim
                )
            attn_output = self._apply_visual_noncausal_attn(
                attn_output, q_fa, k_fa, v_fa, visual_segs
            )

        # Dual-path output projection
        output = mask_apply_no_batch(
            attn_output,
            modality_mask,
            [lambda x: self.o_proj(x)],
            [lambda x: self.o_proj_v(x)],
        )[0]

        return output, (ori_k, v)


# ---------------------------------------------------------------------------
# MoT Decoder Layer
# ---------------------------------------------------------------------------

class HunYuanMoTDecoderLayer(nn.Module):
    """Transformer decoder layer with per-modality norm and MLP paths."""

    def __init__(
        self,
        config: Any,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: int = -1,
    ) -> None:
        super().__init__()
        assert layer_id >= 0
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size
            if isinstance(config.intermediate_size, int)
            else config.intermediate_size[layer_id]
        )

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )

        self.self_attn = HunYuanMoTAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            layer_id=layer_id,
        )

        # Text path
        self.mlp = HunYuanMoTMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Vision path (_v suffix matches checkpoint)
        self.mlp_v = HunYuanMoTMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp_v",
        )
        self.input_layernorm_v = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_v = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        modality_mask: Optional[torch.Tensor] = None,
        visual_segs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        # Pre-attention norm (dual path)
        if residual is None:
            residual = hidden_states
            hidden_states = mask_apply_no_batch(
                hidden_states,
                modality_mask,
                [lambda x: self.input_layernorm(x)],
                [lambda x: self.input_layernorm_v(x)],
            )[0]
        else:
            hidden_states, residual = mask_apply_no_batch_dual(
                hidden_states,
                residual,
                modality_mask,
                [lambda x, r: self.input_layernorm(x, r)],
                [lambda x, r: self.input_layernorm_v(x, r)],
            )

        hidden_states, ori_kv_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_states=kv_states,
            modality_mask=modality_mask,
            visual_segs=visual_segs,
        )

        # Post-attention norm (dual path)
        hidden_states, residual = mask_apply_no_batch_dual(
            hidden_states,
            residual,
            modality_mask,
            [lambda x, r: self.post_attention_layernorm(x, r)],
            [lambda x, r: self.post_attention_layernorm_v(x, r)],
        )

        # MLP (dual path)
        hidden_states = mask_apply_no_batch(
            hidden_states,
            modality_mask,
            [lambda x: self.mlp(x)],
            [lambda x: self.mlp_v(x)],
        )[0]

        return hidden_states, residual, ori_kv_states


# ---------------------------------------------------------------------------
# MoT Language Model
# ---------------------------------------------------------------------------

class HunYuanMoTModel(nn.Module):
    """Full MoT language model: embedding + N decoder layers + final norm."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HunYuanMoTDecoderLayer(
                config=config,
                layer_id=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        # Cache CLA factor to avoid repeated config attribute lookups per forward.
        self._cla_factor: int = _get_cla_factor(config)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        visual_segs = None
        if modality_mask is not None:
            visual_segs = modality_mask_to_segments(modality_mask)

        cla_factor = self._cla_factor
        prev_kv_states = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, kv_states = layer(
                positions,
                hidden_states,
                residual,
                prev_kv_states,
                modality_mask,
                visual_segs,
            )
            if cla_factor > 1 and (i - self.start_layer) % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _split_qkv_weight(self, qkv: torch.Tensor) -> torch.Tensor:
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size

        if hasattr(self.config, "head_dim"):
            attention_head_dim = self.config.head_dim
        elif hasattr(self.config, "attention_head_dim"):
            attention_head_dim = self.config.attention_head_dim
        else:
            attention_head_dim = hidden_size // num_attention_heads

        qkv = qkv.reshape(
            num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size
        )
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        import regex as re

        cla_factor = _get_cla_factor(self.config)
        stacked_params_mapping = [
            # Text path
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # Vision path (_v suffix in checkpoint)
            (".qkv_proj_v", ".q_proj_v", "q"),
            (".qkv_proj_v", ".k_proj_v", "k"),
            (".qkv_proj_v", ".v_proj_v", "v"),
            (".mlp_v.gate_up_proj", ".mlp_v.gate_proj", 0),
            (".mlp_v.gate_up_proj", ".mlp_v.up_proj", 1),
        ]
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj",
                ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self._split_qkv_weight,
            ),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if weight_name == ".q_proj":
                    match = re.search(r"layers\.\d+", name)
                    if match:
                        layer_id = int(match.group(0).split(".")[-1])
                        if cla_factor > 1 and layer_id % cla_factor != 0:
                            continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for param_name, weight_name, den, split_param, func in split_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                param = params_dict[name]
                weight_loader = param.weight_loader
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    if func:
                        weight_loader(param, func(loaded_weight)[offset:new_offset], shard_id)
                    else:
                        weight_loader(param, loaded_weight[offset:new_offset], shard_id)
                    offset = new_offset
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if "mlp.gate.wg." in name:
                    name = name.replace("wg.", "")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class _HunyuanV1MoTModelBase(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = HunYuanMoTModel(vllm_config=vllm_config, prefix="model")

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, modality_mask
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
                "residual": torch.zeros(
                    (batch_size, self.config.hidden_size), dtype=dtype, device=device
                ),
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)


class HunYuanDenseV1MoTForCausalLM(_HunyuanV1MoTModelBase):
    """Dense MoT language model (no MoE). Entry point for the language decoder."""
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
logger = init_logger(__name__)
_MAX_FRAMES_PER_VIDEO = 24576


# ---------------------------------------------------------------------------
# Multimodal Processing
# ---------------------------------------------------------------------------

class HunYuanVLMoTProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        from vllm.transformers_utils.configs.hunyuan_vl_mot import HunYuanVLMoTConfig
        return self.ctx.get_hf_config(HunYuanVLMoTConfig)

    def get_hf_processor(self, **kwargs: object):
        from vllm.transformers_utils.processors.hunyuan_vl_mot import HunYuanVLMoTProcessor
        return self.ctx.get_hf_processor(
            HunYuanVLMoTProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).video_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        max_image_tokens = self.get_max_image_tokens()
        max_video_tokens = self.get_max_video_tokens(seq_len, mm_counts)
        return {"image": max_image_tokens, "video": max_video_tokens}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor=None,
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        patch_size = 16
        merge_size = 2
        temporal_patch_size = 1

        if do_resize:
            resized_height, resized_width = image_smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = num_frames + num_frames % temporal_patch_size
        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size ** 2)
        return preprocessed_size, num_vision_tokens

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        image_processor=None,
    ) -> int:
        _, num_image_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=1,
            image_processor=image_processor,
        )
        return num_image_tokens

    def get_num_video_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int,
        image_processor=None,
    ) -> int:
        _, num_video_tokens = self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            image_processor=image_processor,
        )
        return num_video_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        max_image_size, _ = self._get_vision_info(
            image_width=512,
            image_height=8192,
            image_processor=image_processor,
        )
        return max_image_size

    def get_max_image_tokens(self) -> int:
        image_processor = self.get_image_processor()
        target_size = self.get_image_size_with_most_features()
        return self.get_num_image_tokens(
            image_width=target_size.width,
            image_height=target_size.height,
            image_processor=image_processor,
        )

    def _get_max_video_frames(self, max_tokens: int, start_num_frames: int = 2) -> int:
        image_processor = self.get_image_processor()
        target_size = self.get_image_size_with_most_features()

        num_frames = start_num_frames
        while True:
            next_num_frames = num_frames + 1
            next_max_tokens = self.get_num_video_tokens(
                image_width=target_size.width,
                image_height=target_size.height,
                num_frames=next_num_frames,
                image_processor=image_processor,
            )
            if next_max_tokens > max_tokens:
                break
            num_frames = next_num_frames
        return num_frames

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        max_frames_per_video: int = _MAX_FRAMES_PER_VIDEO,
    ) -> int:
        max_videos = mm_counts.get("video", 0)
        max_total_frames = self._get_max_video_frames(seq_len)
        max_frames_per_video = min(
            max_total_frames // max(max_videos, 1), max_frames_per_video
        )
        return max(max_frames_per_video, 1)

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        image_processor = self.get_image_processor()
        target_size = self.get_image_size_with_most_features()
        return self.get_num_video_tokens(
            image_width=target_size.width,
            image_height=target_size.height,
            num_frames=self.get_num_frames_with_most_features(seq_len, mm_counts),
            image_processor=image_processor,
        )

    def get_data_parser(self) -> MultiModalDataParser:
        from vllm.model_executor.models.qwen2_vl import Qwen2VLMultiModalDataParser
        return Qwen2VLMultiModalDataParser(
            spatial_merge_size=2,
            expected_hidden_size=self._get_expected_hidden_size(),
            video_needs_metadata=True,
        )

    def _get_expected_hidden_size(self) -> int:
        return self.get_hf_config().hidden_size


class HunYuanVLMoTDummyInputsBuilder(
    BaseDummyInputsBuilder["HunYuanVLMoTProcessingInfo"]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        image_token = (
            "<｜hy_place▁holder▁no▁666｜>"
            "<｜hy_place▁holder▁no▁669｜>"
            "<｜hy_place▁holder▁no▁667｜>"
        )
        video_token = (
            "<｜hy_place▁holder▁no▁666｜>"
            "<｜hy_place▁holder▁no▁670｜>"
            "<｜hy_place▁holder▁no▁667｜>"
        )
        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, object] = {},
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        target_size = self.info.get_image_size_with_most_features()
        target_num_frames = self.info.get_num_frames_with_most_features(
            seq_len, mm_counts
        )
        return {
            "image": self._get_dummy_images(
                width=target_size.width,
                height=target_size.height,
                num_images=num_images,
            ),
            "video": self._get_dummy_videos(
                width=target_size.width,
                height=target_size.height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[VideoItem]:
        num_frames = max(num_frames, 2)
        video = np.full((num_frames, height, width, 3), 255, dtype=np.uint8)
        video_items = []
        for _ in range(num_videos):
            metadata: dict[str, Any] = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": list(range(num_frames)),
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_items.append((video.copy(), metadata))
        return video_items


class HunYuanVLMoTMultiModalProcessor(
    BaseMultiModalProcessor["HunYuanVLMoTProcessingInfo"]
):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        if videos := mm_data.pop("videos", []):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            for item in videos:
                video_array, metadata = item
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False
                    )
                metadata_obj = VideoMetadata(
                    **{k: metadata[k] for k in metadata if k != "do_sample_frames"}
                )
                video_outputs = super()._call_hf_processor(
                    prompt=(
                        "<｜hy_place▁holder▁no▁666｜>"
                        "<｜hy_place▁holder▁no▁670｜>"
                        "<｜hy_place▁holder▁no▁667｜>"
                    ),
                    mm_data={
                        "videos": [[video_array]],
                        "video_metadata": [[metadata_obj]],
                    },
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )
                input_ids = video_outputs.pop("input_ids")
                video_placeholder = processor.tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace(
                    (
                        "<｜hy_place▁holder▁no▁666｜>"
                        "<｜hy_place▁holder▁no▁670｜>"
                        "<｜hy_place▁holder▁no▁667｜>"
                    ),
                    video_placeholder,
                    1,
                )
                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])

            video_outputs_combined: dict[str, Any] = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
            )
        else:
            video_outputs_combined = {}

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        return BatchFeature({**processed_outputs, **video_outputs_combined})

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
        image_grid_sizes = image_grid_thw.prod(-1)
        video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
        video_grid_sizes = video_grid_thw.prod(-1)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_embeds=MultiModalFieldConfig.flat_from_sizes(
                "image", image_grid_sizes
            ),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
            pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_embeds=MultiModalFieldConfig.flat_from_sizes(
                "video", video_grid_sizes
            ),
            video_grid_thw=MultiModalFieldConfig.batched("video"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_length = image_processor.merge_size ** 2

        def get_image_replacement(item_idx: int) -> list[int]:
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            rows = grid_thw[1].item() // image_processor.merge_size
            cols = grid_thw[2].item() // image_processor.merge_size
            T = grid_thw[0].item()
            one_row_ids = [IMAGE_TOKEN_ID] * cols + [NEWLINE_TOKEN_ID]
            return one_row_ids * (rows * T)

        def get_video_replacement(item_idx: int) -> PromptUpdateDetails:
            out_item = out_mm_kwargs["video"][item_idx]
            grid_thw = out_item["video_grid_thw"].data
            tokens_per_frame = int(grid_thw[1:].prod()) // merge_length
            placeholder = []
            for _ in range(int(grid_thw[0])):
                placeholder.extend(
                    [VISION_START_TOKEN_ID]
                    + [VIDEO_TOKEN_ID] * tokens_per_frame
                    + [VISION_END_TOKEN_ID]
                )
            return PromptUpdateDetails.select_token_id(placeholder, VIDEO_TOKEN_ID)

        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_TOKEN_ID],
                replacement=get_image_replacement,
            ),
            PromptReplacement(
                modality="video",
                target=(
                    "<｜hy_place▁holder▁no▁666｜>"
                    "<｜hy_place▁holder▁no▁670｜>"
                    "<｜hy_place▁holder▁no▁667｜>"
                ),
                replacement=get_video_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# HunYuanVLMoTForConditionalGeneration — Main Model Class
# ---------------------------------------------------------------------------

@MULTIMODAL_REGISTRY.register_processor(
    HunYuanVLMoTMultiModalProcessor,
    info=HunYuanVLMoTProcessingInfo,
    dummy_inputs=HunYuanVLMoTDummyInputsBuilder,
)
class HunYuanVLMoTForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
):
    """HunYuanVL-MoT: SigLIP vision encoder + MoT language decoder.

    Mixture of Tokens (MoT) uses separate QKV/MLP/LayerNorm parameters for
    text and vision tokens, routed by modality_mask.
    """

    supports_encoder_tp_data = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return (
                "<｜hy_place▁holder▁no▁666｜>"
                "<｜hy_place▁holder▁no▁669｜>"
                "<｜hy_place▁holder▁no▁667｜>"
            )
        if modality.startswith("video"):
            return (
                "<｜hy_place▁holder▁no▁666｜>"
                "<｜hy_place▁holder▁no▁670｜>"
                "<｜hy_place▁holder▁no▁667｜>"
            )
        raise ValueError("Only image or video modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm.model_executor.models.siglip_vit_anyres import SigLIPViTAnysizeWrapper
        from vllm.transformers_utils.configs.hunyuan_vl_mot import HunYuanVLMoTConfig

        config: HunYuanVLMoTConfig = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = SigLIPViTAnysizeWrapper('siglip_vit_anyres')

        with self._mark_language_model(vllm_config):
            self.language_model = HunYuanDenseV1MoTForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Buffer for modality mask
        max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "modality_mask",
            torch.zeros(max_batched_tokens, dtype=torch.bool),
        )
        # Python-side flag: True iff the current batch contains vision tokens.
        # Avoids a GPU→CPU sync (modality_mask.sum() > 0) every forward pass.
        self._has_visual_tokens: bool = False

    def _get_modality_mask(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Return the modality mask slice, or None if no vision tokens present."""
        if not self._has_visual_tokens:
            return None
        return self.modality_mask[:num_tokens]

    def _set_modality_mask(self, modality_mask: torch.Tensor) -> None:
        n = modality_mask.size(0)
        self.modality_mask[:n].copy_(modality_mask)
        self._has_visual_tokens = True

    def _clear_modality_mask(self, num_tokens: int) -> None:
        if num_tokens > 0:
            self.modality_mask[:num_tokens].zero_()
        self._has_visual_tokens = False

    def _validate_and_reshape_mm_tensor(
        self, mm_input: object, name: str
    ) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. Got type: {type(mm_input)}"
            )
        if isinstance(mm_input, torch.Tensor):
            if mm_input.ndim == 2:
                return mm_input
            if mm_input.ndim != 3:
                raise ValueError(
                    f"{name} should be 2D or batched 3D tensor. "
                    f"Got ndim: {mm_input.ndim} (shape={mm_input.shape})"
                )
            return torch.concat(list(mm_input))
        return torch.concat(mm_input)

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Optional[dict[str, Any]]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            pixel_values = self._validate_and_reshape_mm_tensor(
                pixel_values, "image pixel values"
            )
            image_grid_thw = self._validate_and_reshape_mm_tensor(
                image_grid_thw, "image grid_thw"
            )
            return {
                "type": "pixel_values",
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

        image_embeds = self._validate_and_reshape_mm_tensor(
            image_embeds, "image embeds"
        )
        image_grid_thw = self._validate_and_reshape_mm_tensor(
            image_grid_thw, "image grid_thw"
        )
        return {
            "type": "image_embeds",
            "image_embeds": image_embeds,
            "image_grid_thw": image_grid_thw,
        }

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Optional[dict[str, Any]]:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if pixel_values_videos is not None:
            pixel_values_videos = self._validate_and_reshape_mm_tensor(
                pixel_values_videos, "video pixel values"
            )
            video_grid_thw = self._validate_and_reshape_mm_tensor(
                video_grid_thw, "video grid_thw"
            )
            return {
                "type": "pixel_values_videos",
                "pixel_values_videos": pixel_values_videos,
                "video_grid_thw": video_grid_thw,
            }

        video_embeds = self._validate_and_reshape_mm_tensor(
            video_embeds, "video embeds"
        )
        video_grid_thw = self._validate_and_reshape_mm_tensor(
            video_grid_thw, "video grid_thw"
        )
        return {
            "type": "video_embeds",
            "video_embeds": video_embeds,
            "video_grid_thw": video_grid_thw,
        }

    def _process_image_input(
        self, image_input: dict[str, Any]
    ) -> list[torch.Tensor]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            return [image_input["image_embeds"].type(self.visual.dtype)]

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)

        # Reconstruct flat patches → per-image (T, C, H_full, W_full) tensors
        # (same as vllm_embodied _process_image_input)
        num_patches = grid_thw.prod(dim=-1).tolist()
        pixel_values = pixel_values.reshape(-1, 3, 16, 16)
        pixel_values_list = torch.split(pixel_values, [int(n) for n in num_patches], dim=0)
        recon_images = []
        for idx, cur_pixel_value in enumerate(pixel_values_list):
            T = int(grid_thw[idx][0].item())
            H = int(grid_thw[idx][1].item())
            W = int(grid_thw[idx][2].item())
            cur_pixel_value = (
                cur_pixel_value
                .reshape(T, H // 2, W // 2, 2, 2, 3, 16, 16)
                .permute(0, 1, 3, 2, 4, 6, 7, 5)
                .reshape(T, H, W, 16, 16, 3)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(T, H * 16, W * 16, 3)
                .permute(0, 3, 1, 2)
            )
            recon_images.append(cur_pixel_value)

        # Pass image_newline_embedding so SigLIPViTAnysizeWrapper can append
        # a newline token embedding at the end of each image row — matching
        # vllm_embodied's _process_image_input exactly.
        image_newline_embedding = self.language_model.model.get_input_embeddings(
            torch.tensor([NEWLINE_TOKEN_ID], device=pixel_values.device)
        )
        image_embeds = self.visual(
            recon_images, image_newline_embedding=image_newline_embedding
        )
        return image_embeds

    def _process_video_input(
        self, video_input: dict[str, Any]
    ) -> list[torch.Tensor]:
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            return [video_input["video_embeds"].type(self.visual.dtype)]

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype
        )

        # Reconstruct flat patches → per-video (T, C, H_full, W_full) tensors,
        # then encode each video independently — matching vllm_embodied exactly.
        pixel_values_videos = pixel_values_videos.reshape(-1, 3, 16, 16)
        num_patches = grid_thw.prod(dim=-1).tolist()
        pixel_values_list = torch.split(
            pixel_values_videos, [int(n) for n in num_patches], dim=0
        )
        video_features = []
        for idx, cur_pixel_value in enumerate(pixel_values_list):
            T = int(grid_thw[idx][0].item())
            H = int(grid_thw[idx][1].item())
            W = int(grid_thw[idx][2].item())
            cur_pixel_value = (
                cur_pixel_value
                .reshape(T, H // 2, W // 2, 2, 2, 3, 16, 16)
                .permute(0, 1, 3, 2, 4, 6, 7, 5)
                .reshape(T, H, W, 16, 16, 3)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(T, H * 16, W * 16, 3)
                .permute(0, 3, 1, 2)
            )
            video_embeds = self.visual(cur_pixel_value)
            video_features.extend(video_embeds)
        return video_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Extract visual embeddings (official vLLM SupportsMultiModal interface)."""
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        processed_keys: set[str] = set()

        for input_key in list(kwargs.keys()):
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in processed_keys
            ):
                image_input = self._parse_and_validate_image_input(**dict(kwargs))
                if image_input is not None:
                    image_embeddings = self._process_image_input(image_input)
                    multimodal_embeddings += tuple(image_embeddings)
                    processed_keys.add("image")
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in processed_keys
            ):
                video_input = self._parse_and_validate_video_input(**dict(kwargs))
                if video_input is not None:
                    video_embeddings = self._process_video_input(video_input)
                    multimodal_embeddings += tuple(video_embeddings)
                    processed_keys.add("video")

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Override to set modality_mask for MoT routing in the v1 runner path.

        The base class embed_input_ids (interfaces.py) does not call
        _set_modality_mask, so MoT routing in the language model layers is
        broken without this override.
        """
        has_mm = multimodal_embeddings is not None and len(multimodal_embeddings) > 0
        inputs_embeds = super().embed_input_ids(
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        # Set modality_mask for MoT routing: IMAGE and VIDEO positions only
        # (NEWLINE tokens get vision embeddings but do NOT use vision weights)
        if has_mm:
            union_mask = (input_ids == IMAGE_TOKEN_ID) | (
                input_ids == VIDEO_TOKEN_ID
            )
            self._set_modality_mask(union_mask)
        else:
            self._clear_modality_mask(inputs_embeds.size(0))
        return inputs_embeds

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            # Build vision token mask
            is_vision = (input_ids == IMAGE_TOKEN_ID) | (
                input_ids == VIDEO_TOKEN_ID
            ) | (input_ids == NEWLINE_TOKEN_ID)
            # Set modality mask before merge (only image/video, not newline)
            union_mask = (input_ids == IMAGE_TOKEN_ID) | (
                input_ids == VIDEO_TOKEN_ID
            )
            self._set_modality_mask(union_mask)
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_vision,
            )
        else:
            # No multimodal embeddings — clear mask
            self._clear_modality_mask(inputs_embeds.size(0))

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        if inputs_embeds is not None:
            modality_mask = self._get_modality_mask(inputs_embeds.size(0))
        else:
            modality_mask = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            modality_mask=modality_mask,
        )

        if inputs_embeds is not None:
            self._clear_modality_mask(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(
                ["lm_head.", "visual.vision_tower.attn_pool.",
                 "visual.vision_tower.norm"]
                if self.config.tie_word_embeddings
                else None
            ),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.merger",
            tower_model="visual.",
        )
