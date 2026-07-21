# patch.py
# Copyright (c) 2026.
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 -> Symmetric Bipolar Attention (SBA) monkey patch.

This file intentionally patches only Gemma4TextAttention modules. It leaves the
vision/audio towers and all MLP parameters untouched. The patch preserves Gemma
text attention mechanics that matter for a checkpoint-compatible conversion:

  * q/k/v/o projections remain the original modules, so PEFT LoRA wrappers work.
  * q_norm/k_norm/v_norm are applied in the same locations as Hugging Face Gemma4.
  * RoPE is applied to q/k before attention.
  * grouped-query attention repeats KV heads to query-head cardinality.
  * Gemma4 KV sharing for the final shared-KV layers is preserved.
  * attention logit softcapping is applied before the centered SBA bridge.

The centered SBA bridge is:

    w_j = sinh(b + tau * (s_j - mean_visible(s_i))) /
          sum_k cosh(b + tau * (s_k - mean_visible(s_i)))

where b is implemented as a scheduled per-head buffer plus a trainable per-head
delta. tau is a positive per-head trainable parameter. At high b, this bridge is
asymptotically equivalent to softmax over the centered scores; b=4.0 is the
preservation-stage initialization requested by the conversion recipe.
"""

from __future__ import annotations

import json
import math
import os
from types import MethodType
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SBA_BRIDGE_VERSION = 3
SBA_STATE_FILENAME = "sba_bridge_v3.pt"
SBA_CONFIG_FILENAME = "sba_bridge_v3_config.json"


def _inverse_softplus(y: float) -> float:
    y = float(y)
    if y <= 0.0:
        raise ValueError(f"softplus inverse requires y > 0, got {y}")
    return y + math.log(-math.expm1(-y))


def _first_tensor_device(module: nn.Module) -> torch.device:
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def _safe_zero_like(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().sum() * 0.0


def _align_2d_mask(mask: torch.Tensor, length: int, *, pad_value: bool) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D padding mask, got shape {tuple(mask.shape)}")
    if mask.shape[1] == length:
        return mask
    if mask.shape[1] > length:
        return mask[:, -length:]
    pad = torch.full((mask.shape[0], length - mask.shape[1]), pad_value, dtype=torch.bool, device=mask.device)
    return torch.cat([pad, mask], dim=1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 2,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _is_gemma4_text_attention(module: nn.Module) -> bool:
    cls_name = module.__class__.__name__
    if cls_name == "Gemma4TextAttention":
        return True

    cls_lower = cls_name.lower()
    if "vision" in cls_lower or "audio" in cls_lower:
        return False

    config = getattr(module, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type != "gemma4_text":
        return False

    required = (
        "q_proj",
        "o_proj",
        "q_norm",
        "head_dim",
        "num_key_value_groups",
        "layer_type",
        "scaling",
    )
    return all(hasattr(module, attr) for attr in required)


def iter_gemma4_text_attention_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if _is_gemma4_text_attention(module):
            yield name, module


def iter_sba_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if bool(getattr(module, "_sba_is_patched", False)):
            yield name, module


def _num_query_heads(module: nn.Module) -> int:
    config = getattr(module, "config", None)
    num_heads = getattr(config, "num_attention_heads", None)
    if num_heads is not None:
        return int(num_heads)

    q_proj = getattr(module, "q_proj", None)
    out_features = getattr(q_proj, "out_features", None)
    if out_features is None and hasattr(q_proj, "base_layer"):
        out_features = getattr(q_proj.base_layer, "out_features", None)
    if out_features is not None:
        return int(out_features) // int(module.head_dim)

    raise ValueError(f"cannot infer number of query heads for module {module}")


def _install_or_update_bridge_parameters(
    module: nn.Module,
    *,
    initial_bias: float,
    initial_tau: float,
    min_tau: float,
    max_tau: float,
    attention_softcap: Optional[float],
    row_l1_target: float,
    collect_structure_loss: bool,
) -> None:
    num_heads = _num_query_heads(module)
    device = _first_tensor_device(module)

    if not hasattr(module, "_sba_bias_base"):
        module.register_buffer(
            "_sba_bias_base",
            torch.full((num_heads,), float(initial_bias), dtype=torch.float32, device=device),
            persistent=True,
        )
    else:
        if module._sba_bias_base.numel() != num_heads:
            raise ValueError(
                f"existing SBA bias has {module._sba_bias_base.numel()} heads, expected {num_heads}"
            )
        module._sba_bias_base.data.fill_(float(initial_bias))

    if not hasattr(module, "_sba_bias_delta"):
        module.register_parameter(
            "_sba_bias_delta",
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device)),
        )

    if not hasattr(module, "_sba_log_tau"):
        init_log_tau = _inverse_softplus(float(initial_tau))
        module.register_parameter(
            "_sba_log_tau",
            nn.Parameter(torch.full((num_heads,), init_log_tau, dtype=torch.float32, device=device)),
        )

    if not hasattr(module, "_sba_min_tau"):
        module.register_buffer(
            "_sba_min_tau",
            torch.tensor(float(min_tau), dtype=torch.float32, device=device),
            persistent=False,
        )
    else:
        module._sba_min_tau.data.fill_(float(min_tau))

    if not hasattr(module, "_sba_max_tau"):
        module.register_buffer(
            "_sba_max_tau",
            torch.tensor(float(max_tau), dtype=torch.float32, device=device),
            persistent=False,
        )
    else:
        module._sba_max_tau.data.fill_(float(max_tau))

    softcap_value = float("nan") if attention_softcap is None else float(attention_softcap)
    if not hasattr(module, "_sba_attention_softcap"):
        module.register_buffer(
            "_sba_attention_softcap",
            torch.tensor(softcap_value, dtype=torch.float32, device=device),
            persistent=False,
        )
    else:
        module._sba_attention_softcap.data.fill_(softcap_value)

    if not hasattr(module, "_sba_row_l1_target"):
        module.register_buffer(
            "_sba_row_l1_target",
            torch.tensor(float(row_l1_target), dtype=torch.float32, device=device),
            persistent=False,
        )
    else:
        module._sba_row_l1_target.data.fill_(float(row_l1_target))

    module._sba_collect_structure_loss = bool(collect_structure_loss)
    module._sba_last_row_l1_loss = None
    module._sba_last_rbos_loss = None
    module._sba_last_row_l1_mean = None
    module._sba_last_rbos_mean = None
    module._sba_padding_mask_2d = None
    module._sba_bridge_version = SBA_BRIDGE_VERSION


def patch_gemma4_sba(
    model: nn.Module,
    *,
    initial_bias: float = 4.0,
    initial_tau: float = 1.0,
    min_tau: float = 0.05,
    max_tau: float = 8.0,
    attention_softcap: Optional[float] = 50.0,
    row_l1_target: float = 0.65,
    collect_structure_loss: bool = True,
) -> int:
    patched = 0
    for _, module in iter_gemma4_text_attention_modules(model):
        _install_or_update_bridge_parameters(
            module,
            initial_bias=initial_bias,
            initial_tau=initial_tau,
            min_tau=min_tau,
            max_tau=max_tau,
            attention_softcap=attention_softcap,
            row_l1_target=row_l1_target,
            collect_structure_loss=collect_structure_loss,
        )
        if not bool(getattr(module, "_sba_is_patched", False)):
            module._sba_original_forward = module.forward
            module.forward = MethodType(_sba_gemma4_text_attention_forward, module)
            module._sba_is_patched = True
        patched += 1

    if patched == 0:
        raise RuntimeError(
            "No Gemma4TextAttention modules were found. Load a Gemma 4 text/causal model "
            "before calling patch_gemma4_sba(), or update _is_gemma4_text_attention() "
            "for the local Transformers class names."
        )

    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False

    return patched


def unpatch_gemma4_sba(model: nn.Module) -> int:
    restored = 0
    for _, module in iter_sba_modules(model):
        original = getattr(module, "_sba_original_forward", None)
        if original is not None:
            module.forward = original
            module._sba_is_patched = False
            restored += 1
    return restored


def _mask_to_valid(attention_mask: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    batch, heads, q_len, k_len = scores.shape
    mask = attention_mask.to(device=scores.device)

    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask >= 0

    if valid.ndim == 2:
        valid = valid[:, None, None, :]
    elif valid.ndim == 3:
        valid = valid[:, None, :, :]
    elif valid.ndim == 4:
        pass
    else:
        raise ValueError(f"unsupported attention_mask shape {tuple(attention_mask.shape)}")

    if valid.shape[0] != batch:
        raise ValueError(f"attention mask batch {valid.shape[0]} does not match scores batch {batch}")

    if valid.shape[-1] != k_len:
        if valid.shape[-1] > k_len:
            valid = valid[..., -k_len:]
        else:
            pad = torch.ones(*valid.shape[:-1], k_len - valid.shape[-1], dtype=torch.bool, device=valid.device)
            valid = torch.cat([pad, valid], dim=-1)

    if valid.shape[-2] != q_len:
        if valid.shape[-2] == 1:
            pass
        elif valid.shape[-2] > q_len:
            valid = valid[..., -q_len:, :]
        else:
            pad = torch.ones(
                *valid.shape[:-2],
                q_len - valid.shape[-2],
                valid.shape[-1],
                dtype=torch.bool,
                device=valid.device,
            )
            valid = torch.cat([pad, valid], dim=-2)

    if valid.shape[1] not in (1, heads):
        raise ValueError(f"attention mask head dimension {valid.shape[1]} cannot broadcast to {heads}")

    return valid.expand(batch, heads, q_len, k_len)


def _fallback_causal_valid_mask(
    module: nn.Module,
    scores: torch.Tensor,
    sliding_window: Any = None,
) -> torch.Tensor:
    batch, heads, q_len, k_len = scores.shape
    valid = torch.ones((batch, heads, q_len, k_len), dtype=torch.bool, device=scores.device)

    if bool(getattr(module, "is_causal", False)):
        past_len = max(k_len - q_len, 0)
        q_pos = torch.arange(q_len, device=scores.device) + past_len
        k_pos = torch.arange(k_len, device=scores.device)
        causal = k_pos[None, :] <= q_pos[:, None]
        valid = valid & causal[None, None, :, :]

        window = sliding_window if sliding_window is not None else getattr(module, "sliding_window", None)
        if window:
            if isinstance(window, (tuple, list)):
                left = int(window[0])
            else:
                left = int(window)
            if left > 0:
                lower = k_pos[None, :] > (q_pos[:, None] - left)
                valid = valid & lower[None, None, :, :]

    return valid


def _build_valid_and_query_masks(
    module: nn.Module,
    scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    sliding_window: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    batch, heads, q_len, k_len = scores.shape

    if attention_mask is None:
        valid = _fallback_causal_valid_mask(module, scores, sliding_window=sliding_window)
    else:
        valid = _mask_to_valid(attention_mask, scores)

    query_valid = None
    key_valid_2d = None
    pad_mask_2d = getattr(module, "_sba_padding_mask_2d", None)
    if pad_mask_2d is not None:
        pad_mask_2d = pad_mask_2d.to(device=scores.device).bool()
        if pad_mask_2d.ndim == 2 and pad_mask_2d.shape[0] == batch:
            key_valid_2d = _align_2d_mask(pad_mask_2d, k_len, pad_value=True)
            query_valid_2d = _align_2d_mask(pad_mask_2d, q_len, pad_value=True)
            valid = valid & key_valid_2d[:, None, None, :]
            query_valid = query_valid_2d[:, None, :]

    if query_valid is None:
        query_valid = valid.any(dim=-1).any(dim=1, keepdim=True)

    return valid, query_valid, key_valid_2d


def _resolve_attention_softcap(module: nn.Module) -> Optional[float]:
    explicit = getattr(module, "_sba_attention_softcap", None)
    if explicit is not None:
        value = float(explicit.detach().float().cpu().item())
        if not math.isnan(value) and value > 0.0:
            return value

    config = getattr(module, "config", None)
    for attr in (
        "attention_logit_softcapping",
        "attn_logit_softcapping",
        "attention_logits_soft_cap",
        "attention_logit_cap",
    ):
        value = getattr(config, attr, None)
        if value is not None:
            value = float(value)
            if value > 0.0:
                return value
    return None


def _stable_sinh_over_sum_cosh(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    x = x.float()
    valid_f = valid.to(dtype=x.dtype)
    abs_x = x.abs()

    neg_inf = torch.full_like(abs_x, -float("inf"))
    row_max = torch.where(valid, abs_x, neg_inf).amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

    exp_scaled = torch.exp(torch.clamp(abs_x - row_max, max=0.0)) * valid_f
    exp_neg_2_abs = torch.exp(torch.clamp(-2.0 * abs_x, min=-80.0, max=0.0))

    sinh_scaled = torch.sign(x) * exp_scaled * (-torch.expm1(torch.clamp(-2.0 * abs_x, min=-80.0, max=0.0)))
    cosh_scaled = exp_scaled * (1.0 + exp_neg_2_abs)

    denom = cosh_scaled.sum(dim=-1, keepdim=True)
    denom = denom.clamp_min(torch.finfo(torch.float32).tiny)
    weights = sinh_scaled / denom
    return torch.where(valid, weights, torch.zeros_like(weights))


def _record_structure_losses(
    module: nn.Module,
    weights: torch.Tensor,
    valid: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid_2d: Optional[torch.Tensor],
    visible_count: torch.Tensor,
) -> None:
    if not bool(getattr(module, "_sba_collect_structure_loss", True)):
        module._sba_last_row_l1_loss = module._sba_last_rbos_loss = None
        module._sba_last_row_l1_mean = module._sba_last_rbos_mean = None
        return

    batch, heads, q_len, _ = weights.shape
    row_mask = query_valid.expand(batch, heads, q_len) & (visible_count.squeeze(-1) > 0)
    row_l1 = weights.abs().sum(dim=-1)
    target = module._sba_row_l1_target.to(device=weights.device, dtype=row_l1.dtype)

    row_mask_f = row_mask.to(dtype=row_l1.dtype)
    row_mask_sum = row_mask_f.sum().clamp_min(1.0)
    
    row_l1_loss = (((row_l1 - target) ** 2) * row_mask_f).sum() / row_mask_sum
    row_l1_mean = (row_l1.detach() * row_mask_f).sum() / row_mask_sum

    if key_valid_2d is not None:
        any_key = key_valid_2d.any(dim=-1)
        bos_idx = torch.where(any_key, key_valid_2d.to(torch.int64).argmax(dim=-1), torch.zeros_like(any_key, dtype=torch.int64))
    else:
        key_visible_any = valid.any(dim=1).any(dim=1)
        any_key = key_visible_any.any(dim=-1)
        bos_idx = torch.where(any_key, key_visible_any.to(torch.int64).argmax(dim=-1), torch.zeros_like(any_key, dtype=torch.int64))

    gather_idx = bos_idx.view(batch, 1, 1, 1).expand(batch, heads, q_len, 1)
    bos_abs = weights.abs().gather(dim=-1, index=gather_idx).squeeze(-1)
    bos_visible = valid.gather(dim=-1, index=gather_idx).squeeze(-1)

    ceiling = 2.0 / visible_count.squeeze(-1).clamp_min(1).to(dtype=weights.dtype)
    rbos_mask = row_mask & bos_visible
    rbos_mask_f = rbos_mask.to(dtype=weights.dtype)
    rbos_mask_sum = rbos_mask_f.sum().clamp_min(1.0)
    
    rbos_loss = (((F.relu(bos_abs - ceiling)) ** 2) * rbos_mask_f).sum() / rbos_mask_sum
    rbos_mean = (bos_abs.detach() * rbos_mask_f).sum() / rbos_mask_sum

    module._sba_last_row_l1_loss = row_l1_loss
    module._sba_last_rbos_loss = rbos_loss
    module._sba_last_row_l1_mean = row_l1_mean
    module._sba_last_rbos_mean = rbos_mean


def _sba_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(module.head_dim) ** -0.5

    key_states = _repeat_kv(key, int(module.num_key_value_groups))
    value_states = _repeat_kv(value, int(module.num_key_value_groups))

    scores = torch.matmul(query.float(), key_states.float().transpose(2, 3)) * float(scaling)

    softcap = _resolve_attention_softcap(module)
    if softcap is not None:
        scores = torch.tanh(scores / softcap) * softcap

    valid, query_valid, key_valid_2d = _build_valid_and_query_masks(
        module,
        scores,
        attention_mask,
        sliding_window=sliding_window,
    )

    visible_count = valid.sum(dim=-1, keepdim=True)
    visible_count_f = visible_count.clamp_min(1).to(dtype=scores.dtype)
    centered_mean = scores.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True) / visible_count_f

    bias = (module._sba_bias_base.float() + module._sba_bias_delta.float()).view(1, -1, 1, 1)
    tau = F.softplus(module._sba_log_tau.float()).clamp(
        min=float(module._sba_min_tau.item()),
        max=float(module._sba_max_tau.item()),
    )
    tau = tau.view(1, -1, 1, 1)

    bridge_scores = bias + tau * (scores - centered_mean)
    attn_weights = _stable_sinh_over_sum_cosh(bridge_scores, valid)

    _record_structure_losses(module, attn_weights, valid, query_valid, key_valid_2d, visible_count)

    attn_weights_for_value = attn_weights.to(dtype=value_states.dtype)
    if dropout and float(dropout) > 0.0:
        attn_weights_for_value = F.dropout(attn_weights_for_value, p=float(dropout), training=module.training)

    attn_output = torch.matmul(attn_weights_for_value, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights.to(dtype=query.dtype)


def _sba_gemma4_text_attention_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    shared_kv_states: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    past_key_values: Any = None,
    position_ids: Optional[torch.LongTensor] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if getattr(self, "_sba_bypass_patch", False):
        return self._sba_original_forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            shared_kv_states=shared_kv_states,
            past_key_values=past_key_values,
            position_ids=position_ids,
            **kwargs
        )

    del position_ids

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, int(self.head_dim))
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = _apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    if bool(getattr(self, "is_kv_shared_layer", False)):
        if shared_kv_states is None or self.layer_type not in shared_kv_states:
            raise KeyError(
                f"Gemma4 shared KV state for layer_type={self.layer_type!r} is missing. "
                "This usually means the patch was applied to an unsupported Gemma4 variant "
                "or the model forward did not pass shared_kv_states."
            )
        key_states, value_states = shared_kv_states[self.layer_type]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = _apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if bool(getattr(self, "store_full_length_kv", False)):
            if shared_kv_states is not None:
                shared_kv_states[self.layer_type] = (key_states, value_states)

    attn_output, attn_weights = _sba_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=float(self.attention_dropout) if self.training else 0.0,
        scaling=float(self.scaling),
        sliding_window=getattr(self, "sliding_window", None),
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def set_sba_patch_bypass(model: nn.Module, bypass: bool) -> None:
    for _, module in iter_sba_modules(model):
        module._sba_bypass_patch = bool(bypass)


def set_sba_bridge_bias(model: nn.Module, bias: float, *, zero_delta: bool = False) -> None:
    for _, module in iter_sba_modules(model):
        module._sba_bias_base.data.fill_(float(bias))
        if zero_delta:
            module._sba_bias_delta.data.zero_()


def set_sba_bridge_temperature(model: nn.Module, tau: float) -> None:
    init = _inverse_softplus(float(tau))
    for _, module in iter_sba_modules(model):
        module._sba_log_tau.data.fill_(init)


def reset_sba_bridge_deltas(model: nn.Module, *, reset_bias_delta: bool = True, reset_tau: Optional[float] = None) -> None:
    for _, module in iter_sba_modules(model):
        if reset_bias_delta:
            module._sba_bias_delta.data.zero_()
        if reset_tau is not None:
            module._sba_log_tau.data.fill_(_inverse_softplus(float(reset_tau)))


def set_sba_padding_mask(model: nn.Module, attention_mask_2d: Optional[torch.Tensor]) -> None:
    for _, module in iter_sba_modules(model):
        module._sba_padding_mask_2d = attention_mask_2d


def clear_sba_padding_mask(model: nn.Module) -> None:
    set_sba_padding_mask(model, None)


def set_sba_structure_loss_enabled(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_sba_modules(model):
        module._sba_collect_structure_loss = bool(enabled)


def clear_sba_attention_stats(model: nn.Module) -> None:
    for _, module in iter_sba_modules(model):
        module._sba_last_row_l1_loss = None
        module._sba_last_rbos_loss = None
        module._sba_last_row_l1_mean = None
        module._sba_last_rbos_mean = None


def sba_structure_loss(model: nn.Module) -> Dict[str, torch.Tensor]:
    row_losses = []
    rbos_losses = []
    row_means = []
    rbos_means = []
    zero = None

    for _, module in iter_sba_modules(model):
        if zero is None:
            for p in module.parameters(recurse=True):
                zero = _safe_zero_like(p)
                break

        row_loss = getattr(module, "_sba_last_row_l1_loss", None)
        rbos_loss = getattr(module, "_sba_last_rbos_loss", None)
        row_mean = getattr(module, "_sba_last_row_l1_mean", None)
        rbos_mean = getattr(module, "_sba_last_rbos_mean", None)

        if row_loss is not None:
            row_losses.append(row_loss)
        if rbos_loss is not None:
            rbos_losses.append(rbos_loss)
        if row_mean is not None:
            row_means.append(row_mean.float())
        if rbos_mean is not None:
            rbos_means.append(rbos_mean.float())

    if zero is None:
        raise RuntimeError("SBA structure loss requested before patch_gemma4_sba() installed any modules.")

    row_l1_loss = torch.stack(row_losses).mean() if row_losses else zero
    rbos_loss = torch.stack(rbos_losses).mean() if rbos_losses else zero
    row_l1_mean = torch.stack(row_means).mean() if row_means else zero.detach()
    rbos_mean = torch.stack(rbos_means).mean() if rbos_means else zero.detach()

    return {
        "row_l1_loss": row_l1_loss,
        "rbos_loss": rbos_loss,
        "row_l1_mean": row_l1_mean,
        "rbos_mean": rbos_mean,
    }


def named_sba_bridge_parameters(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
    for module_name, module in iter_sba_modules(model):
        for param_name in ("_sba_bias_delta", "_sba_log_tau"):
            param = getattr(module, param_name, None)
            if isinstance(param, nn.Parameter):
                yield f"{module_name}.{param_name}", param


def freeze_non_lora_non_sba_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        trainable = (
            ("lora_" in name)
            or (".lora_" in name)
            or ("_sba_bias_delta" in name)
            or ("_sba_log_tau" in name)
        )
        param.requires_grad_(trainable)


def get_sba_bridge_state(model: nn.Module) -> Dict[str, Any]:
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in iter_sba_modules(model):
        modules[name] = {
            "bias_base": module._sba_bias_base.detach().cpu(),
            "bias_delta": module._sba_bias_delta.detach().cpu(),
            "log_tau": module._sba_log_tau.detach().cpu(),
            "row_l1_target": module._sba_row_l1_target.detach().cpu(),
            "attention_softcap": module._sba_attention_softcap.detach().cpu(),
        }
    return {
        "version": SBA_BRIDGE_VERSION,
        "modules": modules,
    }


def save_sba_bridge_state(
    model: nn.Module,
    output_dir: str,
    *,
    save_function: Optional[Callable[[Any, str], None]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    state = get_sba_bridge_state(model)
    state_path = os.path.join(output_dir, SBA_STATE_FILENAME)

    if save_function is None:
        torch.save(state, state_path)
    else:
        save_function(state, state_path)

    config_path = os.path.join(output_dir, SBA_CONFIG_FILENAME)
    summary = {
        "version": SBA_BRIDGE_VERSION,
        "state_file": SBA_STATE_FILENAME,
        "num_patched_attention_modules": len(state["modules"]),
        "bridge_math": "sinh(b + tau * (score - visible_row_mean)) / sum_visible cosh(...) ",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return state_path


def _find_module_by_saved_name(model: nn.Module, saved_name: str) -> Optional[nn.Module]:
    modules = dict(model.named_modules())
    if saved_name in modules:
        return modules[saved_name]

    suffix_matches = [module for name, module in modules.items() if name.endswith(saved_name) or saved_name.endswith(name)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def load_sba_bridge_state(model: nn.Module, state_path: str) -> Dict[str, int]:
    payload = torch.load(state_path, map_location="cpu")
    if int(payload.get("version", -1)) != SBA_BRIDGE_VERSION:
        raise ValueError(f"unsupported SBA bridge state version: {payload.get('version')}")

    loaded = 0
    missing = 0

    for saved_name, values in payload["modules"].items():
        module = _find_module_by_saved_name(model, saved_name)
        if module is None or not bool(getattr(module, "_sba_is_patched", False)):
            missing += 1
            continue

        module._sba_bias_base.data.copy_(values["bias_base"].to(module._sba_bias_base.device))
        module._sba_bias_delta.data.copy_(values["bias_delta"].to(module._sba_bias_delta.device))
        module._sba_log_tau.data.copy_(values["log_tau"].to(module._sba_log_tau.device))

        if "row_l1_target" in values:
            module._sba_row_l1_target.data.copy_(values["row_l1_target"].to(module._sba_row_l1_target.device))
        if "attention_softcap" in values:
            module._sba_attention_softcap.data.copy_(values["attention_softcap"].to(module._sba_attention_softcap.device))

        loaded += 1

    return {"loaded": loaded, "missing": missing}