# verify_rbos_v1.py
import math
import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_scratch_tokenized_checkpoint import load_model, ScratchConfig, ScratchLM

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
    tau_vals: list[torch.Tensor],
    bias_vals: list[torch.Tensor]
) -> tuple[list[float], list[float], list[float]]:
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
                # Compute centered SBA with given tau and bias
                work = logits.float()
                
                # Causal row mean
                masked_scores = torch.where(
                    causal.to(device=work.device), work, torch.zeros_like(work)
                )
                visible_counts = causal.float().to(device=work.device).sum(dim=-1, keepdim=True).clamp(min=1)
                row_means = masked_scores.sum(dim=-1, keepdim=True) / visible_counts

                tau = tau_vals[layer_idx].view(1, -1, 1, 1).to(device=logits.device)
                bias = bias_vals[layer_idx].view(1, -1, 1, 1).to(device=logits.device)
                centered = (work - row_means) / tau.clamp(min=0.01) + bias

                floor_val = torch.finfo(centered.dtype).min
                zeros = torch.zeros_like(centered)
                valid = causal.to(device=centered.device)

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

                # r_BOS: cosh denominator share of position 0
                cosh_vals = e_pos + e_neg
                cosh_bos = cosh_vals[:, :, :, 0]
                cosh_sum = cosh_vals.sum(dim=-1)
                r_bos = cosh_bos / cosh_sum.clamp(min=1e-8)
            else:
                # Softmax: r_BOS = softmax weight = signed weight
                floor_val = torch.finfo(logits.dtype).min
                floor = torch.full_like(logits, floor_val)
                masked = torch.where(
                    causal.to(device=logits.device), logits, floor
                )
                weights = F.softmax(masked.float(), dim=-1)
                r_bos = weights[:, :, :, 0].float()

            T = weights.shape[-1]
            if T > 1:
                signed_val = weights[0, :, 1:, 0].detach().cpu().float().mean().item() * 100.0
                rbos_val = r_bos[0, :, 1:].detach().cpu().float().mean().item() * 100.0
                bos_score = logits.float()[0, :, 1:, 0].detach().cpu().mean().item()
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
        with torch.no_grad():
            model(input_ids)
    finally:
        for i, block in enumerate(model.blocks):
            block.attn.forward = original_forwards[i]

    return signed_bos_per_layer, rbos_per_layer, bos_score_per_layer

def analyze_model(
    model_name: str,
    checkpoint_path: Path,
    attn_type: str,
    tokenizer,
    device: torch.device,
) -> dict:
    print(f"\n{'='*74}")
    print(f"  {model_name} Model — Signed BOS% vs r_BOS Denominator Share%")
    print(f"{'='*74}")

    model, checkpoint = load_model(checkpoint_path, device)
    model.eval()

    n_layer = model.config.n_layer
    n_head = model.config.n_head

    # Set up tau/bias values
    tau_params = getattr(model, "tau_params", None)
    bias_params = getattr(model, "bias_params", None)
    if tau_params is None or bias_params is None:
        if "lora_history" in checkpoint:
            print("  LoRA conversion checkpoint format. Using tau = 1/1.5 and bias = 0.0.")
            tau_vals = [torch.full((n_head,), 1.0 / 1.5, device=device) for _ in range(n_layer)]
            bias_vals = [torch.zeros((n_head,), device=device) for _ in range(n_layer)]
        else:
            print("  Standard SBA checkpoint without learned parameters. Using default tau = 1.0 and bias = 0.0.")
            tau_vals = [torch.full((n_head,), 1.0, device=device) for _ in range(n_layer)]
            bias_vals = [torch.zeros((n_head,), device=device) for _ in range(n_layer)]
    else:
        tau_vals = tau_params
        bias_vals = bias_params

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
            model, encoded.input_ids, model.config, device, attn_type, tau_vals, bias_vals
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

    # Per-layer breakdown for conversational prompt
    conversational_name = "conversational"
    pr = prompt_results[conversational_name]
    signed = pr["signed"]
    rbos = pr["rbos"]
    scores = pr["scores"]

    print(f"\n  Per-layer breakdown ({conversational_name}):")
    print(
        f"  {'Layer':<8} {'Signed BOS%':>14} {'r_BOS%':>14} "
        f"{'Ratio':>10} {'Mean Score':>12}"
    )
    print(f"  {'-'*58}")
    for i, (s, r, sc) in enumerate(zip(signed, rbos, scores)):
        ratio = r / abs(s) if abs(s) > 0.001 else float("inf")
        ratio_str = f"{ratio:.1f}x" if ratio < 1000 else ">>>>"
        flag = " [SINK]" if r > 15.0 else ""
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

    # Checkpoint paths
    project_root = Path(__file__).resolve().parents[2]
    softmax_path = project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"
    sba_path = project_root / "sba_standalone/results/blackwell_125m_overnight/sba/checkpoints/sba_step015000_12l_12h_768d.pt"
    lora_path = project_root / "sba_standalone/results/lora_conversion/sba_lora_merged_step001000_12l_12h_768d.pt"

    print("Analyzing Softmax baseline model...")
    softmax_res = analyze_model("Softmax", softmax_path, "softmax", tokenizer, device)
    
    print("\nAnalyzing SBA (from-scratch) model...")
    sba_res = analyze_model("SBA From-Scratch", sba_path, "sba", tokenizer, device)
    
    print("\nAnalyzing LoRA Merged SBA (V1) model...")
    lora_res = analyze_model("LoRA Merged SBA (V1)", lora_path, "sba", tokenizer, device)

    # Save summary report
    out_dir = project_root / "sba_standalone/results/lora_conversion"
    report = {
        "softmax_rbos": softmax_res["grand_rbos"],
        "softmax_signed": softmax_res["grand_signed"],
        "sba_rbos": sba_res["grand_rbos"],
        "sba_signed": sba_res["grand_signed"],
        "lora_rbos": lora_res["grand_rbos"],
        "lora_signed": lora_res["grand_signed"],
        "lora_conversational_rbos_by_layer": lora_res["prompt_results"]["conversational"]["rbos"]
    }
    with open(out_dir / "verify_rbos_v1_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved r_BOS verification report to: {out_dir / 'verify_rbos_v1_results.json'}")

if __name__ == "__main__":
    main()
