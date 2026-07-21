# verify_rbos.py
"""
Critical diagnostic: Does SBA actually eliminate attention sinks?

The current diagnostic framework measures signed BOS weight:
    w_BOS = sinh(s_BOS) / Σ cosh(s_k)

The correct SBA sink metric is denominator share:
    r_BOS = cosh(s_BOS) / Σ cosh(s_k)

The relationship is:  w_BOS = tanh(s_BOS) × r_BOS

If s_BOS ≈ 0, then tanh(s_BOS) ≈ 0, so w_BOS ≈ 0 even if r_BOS is large.
That would mean the "sink" still dominates the denominator (suppressing all
other weights) but the signed weight to BOS happens to be near zero.

This script computes both metrics side-by-side on the from-scratch SBA and
softmax checkpoints to determine whether the sink-elimination claim is real
or an artifact of the wrong measurement.

For softmax: r_BOS = softmax weight to BOS = signed weight. Both agree.
For SBA:     r_BOS may differ significantly from signed weight.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import (
    ScratchConfig,
    ScratchLM,
)

PROMPTS = {
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


def capture_both_bos_metrics(
    model: ScratchLM,
    input_ids: torch.Tensor,
    config: ScratchConfig,
    device: torch.device,
    attn_type: str,
) -> tuple[list[float], list[float], list[float]]:
    """Capture signed-weight BOS%, r_BOS denominator share%, and raw BOS scores per layer.

    Returns:
        signed_bos: List of signed BOS weight % per layer
        rbos: List of r_BOS denominator share % per layer
        bos_scores: List of mean raw QK score to BOS per layer
    """
    signed_bos_per_layer: list[float] = []
    rbos_per_layer: list[float] = []
    bos_score_per_layer: list[float] = []
    original_forwards: list = []

    def _make_forward(layer_idx: int):
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

            if attn_type == "sba":
                work = logits.float()
                valid = causal.to(device=work.device)
                floor = torch.full_like(work, torch.finfo(work.dtype).min)
                zeros = torch.zeros_like(work)

                pos_logits = torch.where(valid, work, floor)
                neg_logits = torch.where(valid, -work, floor)
                row_max = torch.maximum(
                    pos_logits.max(dim=-1, keepdim=True).values,
                    neg_logits.max(dim=-1, keepdim=True).values,
                )
                e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
                e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)

                denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(1e-8)
                weights = torch.where(valid, (e_pos - e_neg) / denom, zeros)

                # r_BOS: cosh denominator share of position 0
                cosh_vals = e_pos + e_neg  # proportional to cosh (scaled by exp(-row_max))
                cosh_bos = cosh_vals[:, :, :, 0]   # [B, H, T]
                cosh_sum = cosh_vals.sum(dim=-1)    # [B, H, T]
                r_bos = cosh_bos / cosh_sum.clamp_min(1e-8)  # [B, H, T]
            else:
                # Softmax: r_BOS = softmax weight = signed weight (they're identical)
                floor_val = torch.finfo(logits.dtype).min
                floor = torch.full_like(logits, floor_val)
                masked = torch.where(
                    causal.to(device=logits.device), logits, floor
                )
                weights = F.softmax(masked.float(), dim=-1)
                r_bos = weights[:, :, :, 0].float()  # [B, H, T]

            T = weights.shape[-1]
            if T > 1:
                # Mean signed BOS weight across heads and non-BOS query positions
                signed_val = weights[0, :, 1:, 0].detach().cpu().float().mean().item() * 100.0
                # Mean r_BOS across heads and non-BOS query positions
                rbos_val = r_bos[0, :, 1:].detach().cpu().float().mean().item() * 100.0
                # Mean raw score to BOS (non-BOS queries, before SBA/softmax)
                raw_scores = logits.float()
                bos_score = raw_scores[0, :, 1:, 0].detach().cpu().mean().item()
            else:
                signed_val = 0.0
                rbos_val = 0.0
                bos_score = 0.0

            signed_bos_per_layer.append(signed_val)
            rbos_per_layer.append(rbos_val)
            bos_score_per_layer.append(bos_score)

            y = torch.matmul(weights.to(dtype=v.dtype), v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))

            if use_cache:
                return out, (k, v)
            return out

        return _forward

    for i, block in enumerate(model.blocks):
        original_forwards.append(block.attn.forward)
        block.attn.forward = _make_forward(i).__get__(
            block.attn, type(block.attn)
        )

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(input_ids)
    finally:
        for i, block in enumerate(model.blocks):
            block.attn.forward = original_forwards[i]

    return signed_bos_per_layer, rbos_per_layer, bos_score_per_layer


def analyze_model(
    model_name: str,
    ckpt: dict,
    attn_type: str,
    config: ScratchConfig,
    tokenizer,
    device: torch.device,
) -> dict:
    """Run full analysis on one model."""
    print(f"\n{'='*74}")
    print(f"  {model_name} Model — Signed BOS% vs r_BOS Denominator Share%")
    print(f"{'='*74}")

    model = ScratchLM(config, attn_type, sba_impl="sdpa")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    all_signed: list[float] = []
    all_rbos: list[float] = []

    print(
        f"\n  {'Prompt':<20} {'Signed BOS%':>14} {'r_BOS%':>14} "
        f"{'Ratio':>10} {'Tokens':>7}"
    )
    print(f"  {'-'*65}")

    prompt_results = {}
    for prompt_name, prompt_text in PROMPTS.items():
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        n_tokens = encoded.input_ids.shape[1]
        signed, rbos, scores = capture_both_bos_metrics(
            model, encoded.input_ids, config, device, attn_type
        )

        mean_signed = sum(signed) / len(signed)
        mean_rbos = sum(rbos) / len(rbos)
        all_signed.extend(signed)
        all_rbos.extend(rbos)

        ratio = mean_rbos / abs(mean_signed) if abs(mean_signed) > 0.001 else float("inf")
        ratio_str = f"{ratio:.1f}x" if ratio < 1000 else ">>>>"

        print(
            f"  {prompt_name:<20} {mean_signed:>+13.2f}% {mean_rbos:>13.2f}% "
            f"{ratio_str:>10} {n_tokens:>7}"
        )
        prompt_results[prompt_name] = {
            "signed": signed, "rbos": rbos, "scores": scores,
            "mean_signed": mean_signed, "mean_rbos": mean_rbos,
        }

    grand_signed = sum(all_signed) / len(all_signed)
    grand_rbos = sum(all_rbos) / len(all_rbos)
    grand_ratio = grand_rbos / abs(grand_signed) if abs(grand_signed) > 0.001 else float("inf")

    print(f"\n  AGGREGATE across all prompts and layers:")
    print(f"    Signed BOS (current metric):  {grand_signed:>+.4f}%")
    print(f"    r_BOS (denominator share):    {grand_rbos:>.4f}%")
    ratio_str = f"{grand_ratio:.1f}x" if grand_ratio < 1000 else ">>>>"
    print(f"    Ratio r_BOS / |signed|:       {ratio_str}")

    # Per-layer breakdown for the longest prompt
    longest_name = max(PROMPTS.keys(), key=lambda k: len(PROMPTS[k]))
    pr = prompt_results[longest_name]
    signed = pr["signed"]
    rbos = pr["rbos"]
    scores = pr["scores"]

    print(f"\n  Per-layer breakdown ({longest_name}):")
    print(
        f"  {'Layer':<8} {'Signed BOS%':>14} {'r_BOS%':>14} "
        f"{'Ratio':>10} {'Mean Score':>12}"
    )
    print(f"  {'-'*58}")
    for i, (s, r, sc) in enumerate(zip(signed, rbos, scores)):
        ratio = r / abs(s) if abs(s) > 0.001 else float("inf")
        ratio_str = f"{ratio:.1f}x" if ratio < 1000 else ">>>>"
        flag = " ⚠ SINK" if r > 15.0 else ""
        print(
            f"  Layer {i:<2d} {s:>+13.2f}% {r:>13.2f}% "
            f"{ratio_str:>10} {sc:>+11.4f}{flag}"
        )

    del model
    torch.cuda.empty_cache()

    return {
        "grand_signed": grand_signed,
        "grand_rbos": grand_rbos,
        "grand_ratio": grand_ratio,
        "prompt_results": prompt_results,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load checkpoints ---
    softmax_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/softmax"
        "/checkpoints/softmax_step015000_12l_12h_768d.pt"
    )
    sba_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/sba"
        "/checkpoints/sba_step015000_12l_12h_768d.pt"
    )

    for p, label in [(softmax_path, "Softmax"), (sba_path, "SBA")]:
        if not p.exists():
            print(f"ERROR: {label} checkpoint not found: {p}")
            sys.exit(1)

    softmax_ckpt = torch.load(softmax_path, map_location="cpu", weights_only=False)
    config = ScratchConfig(**softmax_ckpt["scratch_config"])
    sba_ckpt = torch.load(sba_path, map_location="cpu", weights_only=False)

    print(f"\nModel: {config.n_layer}L {config.n_head}H {config.n_embd}D")
    print(f"Head dim: {config.n_embd // config.n_head}")

    # --- Analyze both models ---
    softmax_result = analyze_model(
        "Softmax (baseline)", softmax_ckpt, "softmax", config, tokenizer, device
    )
    sba_result = analyze_model(
        "SBA (from-scratch)", sba_ckpt, "sba", config, tokenizer, device
    )

    # --- Verdict ---
    print(f"\n{'='*74}")
    print("  VERDICT: Is the SBA sink-elimination claim real?")
    print(f"{'='*74}")

    sba_rbos = sba_result["grand_rbos"]
    softmax_rbos = softmax_result["grand_rbos"]
    sba_signed = sba_result["grand_signed"]
    softmax_signed = softmax_result["grand_signed"]

    print(f"\n  Softmax baseline:")
    print(f"    Signed BOS: {softmax_signed:>+.4f}%   r_BOS: {softmax_rbos:>.4f}%")
    print(f"  SBA from-scratch:")
    print(f"    Signed BOS: {sba_signed:>+.4f}%   r_BOS: {sba_rbos:>.4f}%")

    print(f"\n  Reduction from Softmax → SBA:")
    print(f"    Signed BOS: {softmax_signed:.2f}% → {sba_signed:+.2f}%")
    print(f"    r_BOS:      {softmax_rbos:.2f}% → {sba_rbos:.2f}%")

    # Interpretation
    if sba_rbos < 10.0:
        print(f"\n  ✓ r_BOS is genuinely low ({sba_rbos:.2f}%).")
        print(f"    The sink-elimination claim appears VALID on the correct metric.")
        if sba_rbos < 2 * abs(sba_signed) + 1.0:
            print(f"    r_BOS and |signed| are close → measurement nuance only.")
        else:
            print(
                f"    r_BOS ({sba_rbos:.2f}%) > |signed| ({abs(sba_signed):.2f}%) "
                f"→ some denominator mass on BOS exists,"
            )
            print(f"    but it's within an acceptable range.")
    elif sba_rbos < 20.0:
        print(f"\n  ⚠ r_BOS is moderate ({sba_rbos:.2f}%).")
        print(f"    The sink is reduced but NOT eliminated.")
        print(
            f"    The signed-weight metric ({abs(sba_signed):.2f}%) "
            f"underestimates residual sink mass."
        )
        print(f"    The paper's claim needs qualification.")
    else:
        print(f"\n  ✗ r_BOS is HIGH ({sba_rbos:.2f}%).")
        print(
            f"    The signed-weight metric ({abs(sba_signed):.2f}%) "
            f"is MISLEADING."
        )
        print(f"    BOS still dominates the denominator → sink NOT eliminated.")
        print(f"    The paper's headline finding is WRONG on this metric.")

    print(f"\n{'='*74}")


if __name__ == "__main__":
    main()
