# conversion_diagnostics.py
"""
Shared diagnostic framework for SBA conversion experiments.
Measures BOS attention distribution, activation outlier statistics, and
provides pass/fail gate evaluation against SBA-native reference values.
"""
import math
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import ScratchConfig, ScratchLM, sba_weights, evaluate


DIAGNOSTIC_PROMPTS = {
    "math": "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n",
    "code_completion": (
        'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n'
        "    if n <= 1:\n        return n\n    return"
    ),
    "long_context": (
        "The transformer architecture was introduced in the paper "
        "'Attention Is All You Need' by Vaswani et al. in 2017. "
        "It relies on self-attention mechanisms to process input sequences "
        "in parallel, replacing recurrent neural networks. The key innovation "
        "was the multi-head attention mechanism, which allows the model to "
        "attend to different parts of the input simultaneously. Since then, "
        "transformers have become the dominant architecture for natural "
        "language processing tasks.\n\nQuestion: What was the key innovation "
        "of the transformer architecture?\n\nAnswer:\n"
    ),
    "conversational": (
        "User: Hey, can you explain what attention sinks are in large "
        "language models?\n\nAssistant:"
    ),
    "json_structured": (
        '{\n  "task": "classification",\n'
        '  "input": "The movie was absolutely terrible and a waste of time.",\n'
        '  "label":'
    ),
}

# ---------------------------------------------------------------------------
# Gate thresholds — derived from empirical SBA-native values
# ---------------------------------------------------------------------------
GATES = {
    "mean_bos_pct": 5.0,        # mean BOS attention across all layers/prompts
    "max_bos_pct": 15.0,        # worst-case BOS for any single layer/prompt
    "outlier_p999_max": 25.0,   # worst-layer p99.9 activation magnitude
    "activation_max": 50.0,     # worst-layer max absolute activation
    "val_loss_delta": 0.08,     # max acceptable gap from softmax baseline (1.0005)
}

SOFTMAX_BASELINE_LOSS = 1.0005  # verified baseline on 4k eval slice


def sba_weights_centered(
    logits: torch.Tensor,
    causal_mask: torch.Tensor,
    tau: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SBA attention weights with causal row-mean centering.

    Args:
        logits: Raw QK^T / sqrt(d) scores, shape [B, H, T, T].
        causal_mask: Boolean mask, True = visible, shape [1, 1, T, T] or similar.
        tau: Per-head temperature, shape [H] or [1].
        bias: Per-head bias, shape [H] or [1].

    Returns:
        weights: Signed attention weights [B, H, T, T].
        r_bos: Denominator share of BOS [B, H, T] (per-query position).
        row_l1: L1 norm of each attention row [B, H, T].
    """
    work = logits.float()

    # Causal row mean: μ_i = (1/i) Σ_{j≤i} s_ij
    # Set invisible positions to zero before computing mean
    masked_scores = torch.where(
        causal_mask.to(device=work.device), work, torch.zeros_like(work)
    )

    # Count visible positions per query
    visible_counts = causal_mask.float().to(device=work.device).sum(dim=-1, keepdim=True).clamp(min=1)
    row_means = masked_scores.sum(dim=-1, keepdim=True) / visible_counts

    # Center, scale, and bias
    # tau and bias have shape [H] → reshape to [1, H, 1, 1]
    tau_4d = tau.view(1, -1, 1, 1)
    bias_4d = bias.view(1, -1, 1, 1)
    centered = (work - row_means) / tau_4d.clamp(min=0.01) + bias_4d

    # Masked exp formulation for numerical stability (prevents positive infinity on masked positions)
    floor_val = torch.finfo(centered.dtype).min
    zeros = torch.zeros_like(centered)
    valid = causal_mask.to(device=centered.device)

    pos_logits = torch.where(valid, centered, torch.full_like(centered, floor_val))
    neg_logits = torch.where(valid, -centered, torch.full_like(centered, floor_val))

    row_max = torch.maximum(
        pos_logits.max(dim=-1, keepdim=True).values,
        neg_logits.max(dim=-1, keepdim=True).values,
    )

    e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
    e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)

    denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    weights = torch.where(valid, (e_pos - e_neg) / denom, zeros)

    # Compute r_BOS: cosh-denominator share of position 0
    cosh_vals = e_pos + e_neg  # [B, H, T, T]
    cosh_bos = cosh_vals[:, :, :, 0]  # [B, H, T]
    cosh_sum = cosh_vals.sum(dim=-1)  # [B, H, T]
    r_bos = cosh_bos / cosh_sum.clamp(min=1e-8)  # [B, H, T]

    # Row L1 norm
    row_l1 = weights.abs().sum(dim=-1)  # [B, H, T]

    return weights.to(dtype=logits.dtype), r_bos.to(dtype=logits.dtype), row_l1.to(dtype=logits.dtype)


def capture_attention_weights(
    model: ScratchLM,
    input_ids: torch.Tensor,
    config: ScratchConfig,
    device: torch.device,
    attn_type: str = "sba",
    scale: float | list[float] = 1.0,
) -> list[torch.Tensor]:
    """Run forward pass and capture per-layer attention weight matrices.

    Uses eager computation (explicit QK logits → weight function) regardless
    of the model's current forward implementation, so results are
    interpretable and comparable across experiments.

    Args:
        model: The ScratchLM model.
        input_ids: Token ids, shape [1, T].
        config: Model config.
        device: CUDA device.
        attn_type: "softmax", "sba", or "sba_centered".
        scale: Global logit scale (float) or per-layer list of scales (only for sba/softmax).

    Returns:
        List of attention weight tensors [1, H, T, T] per layer.
    """
    if isinstance(scale, (int, float)):
        per_layer_scale = [float(scale)] * config.n_layer
    else:
        per_layer_scale = [float(s) for s in scale]

    stored_weights: list[torch.Tensor] = []
    original_forwards: list[Any] = []

    def _make_capture_forward(layer_idx: int, layer_scale: float):
        def _forward(
            self_attn, x, attention_mask=None, past_key_value=None, use_cache=False
        ):
            batch, seq_len, channels = x.shape
            qkv = self_attn.qkv(x).view(
                batch, seq_len, 3, self_attn.n_head, self_attn.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            causal = self_attn.causal_mask[:, :, :seq_len, :seq_len]
            logits = (
                torch.matmul(q, k.transpose(-1, -2))
                / math.sqrt(self_attn.head_dim)
            )

            if attn_type == "sba_centered":
                tau = getattr(model, "tau_params", None)
                bias = getattr(model, "bias_params", None)
                if tau is None or bias is None:
                    # Fallback to standard sba
                    logits = logits * layer_scale
                    weights = sba_weights(logits, causal)
                else:
                    # Ignore layer_scale for centered attention as temperature is learned internally
                    weights, _, _ = sba_weights_centered(
                        logits, causal, tau[layer_idx], bias[layer_idx]
                    )
            elif attn_type == "sba":
                logits = logits * layer_scale
                weights = sba_weights(logits, causal)
            else:
                logits = logits * layer_scale
                floor_val = torch.finfo(logits.dtype).min
                floor = torch.full_like(logits, floor_val)
                masked = torch.where(
                    causal.to(device=logits.device), logits, floor
                )
                weights = F.softmax(masked, dim=-1)

            stored_weights.append(weights.detach().cpu().float())

            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))
            if use_cache:
                return out, (k, v)
            return out

        return _forward

    # Patch forwards
    for i, block in enumerate(model.blocks):
        original_forwards.append(block.attn.forward)
        block.attn.forward = _make_capture_forward(
            i, per_layer_scale[i]
        ).__get__(block.attn, type(block.attn))

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(input_ids)
    finally:
        # Restore original forwards
        for i, block in enumerate(model.blocks):
            block.attn.forward = original_forwards[i]

    return stored_weights


def compute_bos_pct_per_layer(
    attn_weights: list[torch.Tensor],
) -> list[float]:
    """Compute BOS attention percentage per layer from captured weights.

    BOS attention = mean attention weight from non-BOS positions to position 0,
    averaged across heads.
    """
    bos_per_layer = []
    for w in attn_weights:
        # w: [1, H, T, T]
        T = w.shape[-1]
        if T > 1:
            bos_pct = w[0, :, 1:, 0].mean().item() * 100.0
        else:
            bos_pct = 0.0
        bos_per_layer.append(bos_pct)
    return bos_per_layer


def capture_activation_stats(
    model: ScratchLM,
    input_ids: torch.Tensor,
    config: ScratchConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    """Capture per-layer activation outlier statistics.

    Hooks into each Block's output to measure hidden state magnitudes.
    """
    stats: list[dict[str, float]] = []
    hooks = []

    def _make_hook(layer_idx: int):
        def _hook(module, inp, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_flat = h.detach().float().abs().flatten()
            stats.append({
                "layer": layer_idx,
                "max_abs": h_flat.max().item(),
                "p999": torch.quantile(h_flat, 0.999).item(),
                "p99": torch.quantile(h_flat, 0.99).item(),
                "mean_abs": h_flat.mean().item(),
            })
        return _hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(_make_hook(i)))

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()

    return stats


def run_diagnostic(
    model: ScratchLM,
    config: ScratchConfig,
    device: torch.device,
    attn_type: str,
    tokenizer,
    val_loader=None,
    scale: float | list[float] = 1.0,
    label: str = "model",
) -> dict:
    """Run the full diagnostic suite.

    Returns a structured dict with BOS attention, outlier stats,
    optional val_loss, and gate evaluation.
    """
    model.eval()

    # --- BOS attention across prompts ---
    bos_results = {}
    for prompt_name, prompt_text in DIAGNOSTIC_PROMPTS.items():
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        weights = capture_attention_weights(
            model, encoded.input_ids, config, device, attn_type, scale
        )
        bos_per_layer = compute_bos_pct_per_layer(weights)
        bos_results[prompt_name] = {
            "n_tokens": encoded.input_ids.shape[1],
            "bos_per_layer": bos_per_layer,
        }

    # Aggregate BOS statistics
    all_bos = []
    for p_data in bos_results.values():
        all_bos.extend(p_data["bos_per_layer"])
    mean_bos = sum(all_bos) / len(all_bos) if all_bos else 0.0
    max_bos = max(all_bos) if all_bos else 0.0
    min_bos = min(all_bos) if all_bos else 0.0

    # --- Activation outliers (use the longest prompt for max stress) ---
    longest_prompt = max(DIAGNOSTIC_PROMPTS.values(), key=len)
    encoded = tokenizer(longest_prompt, return_tensors="pt").to(device)
    outlier_stats = capture_activation_stats(
        model, encoded.input_ids, config, device
    )
    worst_p999 = max(s["p999"] for s in outlier_stats) if outlier_stats else 0.0
    worst_max = max(s["max_abs"] for s in outlier_stats) if outlier_stats else 0.0

    # --- Validation loss ---
    val_loss = None
    if val_loader is not None:
        val_result = evaluate(model, val_loader, device, torch.bfloat16)
        val_loss = val_result["loss"]

    # --- Gate evaluation ---
    gates = {
        "mean_bos_pass": abs(mean_bos) < GATES["mean_bos_pct"],
        "max_bos_pass": abs(max_bos) < GATES["max_bos_pct"],
        "outlier_p999_pass": worst_p999 < GATES["outlier_p999_max"],
        "activation_max_pass": worst_max < GATES["activation_max"],
    }
    if val_loss is not None:
        gates["val_loss_pass"] = (
            val_loss - SOFTMAX_BASELINE_LOSS
        ) < GATES["val_loss_delta"]

    gates["all_pass"] = all(gates.values())

    result = {
        "label": label,
        "bos": {
            "per_prompt": bos_results,
            "mean_bos_pct": mean_bos,
            "max_bos_pct": max_bos,
            "min_bos_pct": min_bos,
        },
        "outliers": {
            "per_layer": outlier_stats,
            "worst_p999": worst_p999,
            "worst_max_abs": worst_max,
        },
        "val_loss": val_loss,
        "gates": gates,
    }
    return result


def print_diagnostic_summary(result: dict, references: dict | None = None) -> None:
    """Print a human-readable summary table."""
    label = result["label"]
    bos = result["bos"]
    outliers = result["outliers"]
    gates = result["gates"]

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC SUMMARY: {label}")
    print(f"{'='*70}")

    # BOS summary per prompt
    print(f"\n  BOS Attention (%) - per prompt, Layer 3 highlighted:")
    print(f"  {'Prompt':<20} {'Mean':>8} {'L3':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*52}")
    for prompt_name, p_data in bos["per_prompt"].items():
        layers = p_data["bos_per_layer"]
        l3 = layers[3] if len(layers) > 3 else 0.0
        print(
            f"  {prompt_name:<20} "
            f"{sum(layers)/len(layers):+7.2f}% "
            f"{l3:+7.2f}% "
            f"{min(layers):+7.2f}% "
            f"{max(layers):+7.2f}%"
        )

    print(f"\n  Aggregate BOS: mean={bos['mean_bos_pct']:+.2f}%  "
          f"max={bos['max_bos_pct']:+.2f}%  min={bos['min_bos_pct']:+.2f}%")

    # Outlier summary
    print(f"\n  Activation Outliers:")
    print(f"  {'Layer':<8} {'Max Abs':>10} {'p99.9':>10} {'p99':>10} {'Mean Abs':>10}")
    print(f"  {'-'*48}")
    for s in outliers["per_layer"]:
        print(
            f"  Layer {s['layer']:<2d}  "
            f"{s['max_abs']:>9.2f}  {s['p999']:>9.2f}  "
            f"{s['p99']:>9.2f}  {s['mean_abs']:>9.4f}"
        )
    print(f"\n  Worst p99.9: {outliers['worst_p999']:.2f}  "
          f"Worst max: {outliers['worst_max_abs']:.2f}")

    # Val loss
    if result["val_loss"] is not None:
        delta = result["val_loss"] - SOFTMAX_BASELINE_LOSS
        print(f"\n  Val Loss: {result['val_loss']:.6f}  "
              f"(Delta from baseline: {delta:+.4f})")

    # References
    if references:
        print(f"\n  Reference Comparison:")
        for ref_name, ref_data in references.items():
            print(f"    {ref_name}: mean_bos={ref_data.get('mean_bos', '?')}%, "
                  f"worst_p999={ref_data.get('worst_p999', '?')}")

    # Gates
    print(f"\n  Gate Results:")
    for gate_name, passed in gates.items():
        if gate_name == "all_pass":
            continue
        status = "[PASS]" if passed else "[FAIL]"
        print(f"    {gate_name}: {status}")
    overall = "[ALL PASS]" if gates["all_pass"] else "[SOME FAILED]"
    print(f"\n  Overall: {overall}")
    print(f"{'='*70}")


def save_diagnostic(result: dict, output_path: Path) -> None:
    """Save diagnostic results to JSON."""

    def _make_serializable(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(x) for x in obj]
        return obj

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(_make_serializable(result), f, indent=2)
    print(f"  Diagnostic saved to: {output_path}")
