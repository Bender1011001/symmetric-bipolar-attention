# verify_bos_multi_prompt.py
"""
Concern 3 verification: Is the Layer 3 -16.99% BOS suppression in the
LoRA-converted model a genuine architectural property, or a prompt-specific
artifact of the single math prompt used in the diagnostic?

Tests BOS attention across multiple diverse prompts to determine if the
suppression pattern is consistent.
"""
import os
import sys
import math
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import ScratchConfig, ScratchLM, sba_weights

PROMPTS = {
    "math": "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n",
    "code_completion": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n <= 1:\n        return n\n    return",
    "long_context": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It relies on self-attention mechanisms to process input sequences in parallel, replacing recurrent neural networks. The key innovation was the multi-head attention mechanism, which allows the model to attend to different parts of the input simultaneously. Since then, transformers have become the dominant architecture for natural language processing tasks.\n\nQuestion: What was the key innovation of the transformer architecture?\n\nAnswer:\n",
    "conversational": "User: Hey, can you explain what attention sinks are in large language models?\n\nAssistant:",
    "json_structured": "{\n  \"task\": \"classification\",\n  \"input\": \"The movie was absolutely terrible and a waste of time.\",\n  \"label\":",
}


def measure_bos_attention(model, input_ids, config, device, model_name, scale=1.0):
    """Run a forward pass with attention hooks and return per-layer BOS attention %."""
    attn_maps = []

    def make_hook(scale_val):
        def hook(self, x, attention_mask=None, past_key_value=None, use_cache=False):
            batch, seq_len, channels = x.shape
            qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            causal = self.causal_mask[:, :, :seq_len, :seq_len]
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            scaled_logits = logits * scale_val

            if model_name in ["sba", "lora"]:
                weights = sba_weights(scaled_logits, causal)
            else:
                floor = torch.full_like(logits, torch.finfo(logits.dtype).min)
                masked_logits = torch.where(causal.to(device=logits.device), logits, floor)
                weights = F.softmax(masked_logits, dim=-1)

            attn_maps.append(weights.detach().cpu())

            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self.dropout(self.proj(y))
            if use_cache:
                return out, (k, v)
            return out
        return hook

    # Install hooks
    for block in model.blocks:
        block.attn.forward = make_hook(scale).__get__(block.attn, type(block.attn))

    with torch.no_grad():
        model(input_ids)

    # Extract BOS attention per layer
    bos_per_layer = []
    for l in range(config.n_layer):
        attn = attn_maps[l][0]  # [H, T, T]
        T = attn.shape[-1]
        if T > 1:
            bos_pct = attn[:, 1:, 0].mean().item() * 100
        else:
            bos_pct = 0.0
        bos_per_layer.append(bos_pct)

    return bos_per_layer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent

    # Resolve paths
    def resolve(local_rel, remote_abs):
        p = project_root / local_rel
        if p.exists():
            return p
        return Path(remote_abs)

    softmax_path = resolve(
        "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt",
        "/root/sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"
    )
    sba_path = resolve(
        "sba_standalone/results/blackwell_125m_overnight/sba/checkpoints/sba_step015000_12l_12h_768d.pt",
        "/root/sba_standalone/results/blackwell_125m_overnight/sba/checkpoints/sba_step015000_12l_12h_768d.pt"
    )
    lora_path = resolve(
        "sba_standalone/results/lora_conversion/sba_lora_merged_step001000_12l_12h_768d.pt",
        "/root/sba_standalone/results/lora_conversion/sba_lora_merged_step001000_12l_12h_768d.pt"
    )

    for name, path in [("softmax", softmax_path), ("sba", sba_path), ("lora", lora_path)]:
        if not path.exists():
            print(f"ERROR: {name} checkpoint not found at {path}")
            sys.exit(1)

    # Load models
    ckpt = torch.load(softmax_path, map_location="cpu")
    config = ScratchConfig(**ckpt["scratch_config"])

    softmax_model = ScratchLM(config, "softmax")
    softmax_model.load_state_dict(ckpt["state_dict"])
    softmax_model.to(device).eval()

    sba_ckpt = torch.load(sba_path, map_location="cpu")
    sba_model = ScratchLM(config, "sba", sba_impl="sdpa")
    sba_model.load_state_dict(sba_ckpt["state_dict"])
    sba_model.to(device).eval()

    lora_ckpt = torch.load(lora_path, map_location="cpu")
    lora_model = ScratchLM(config, "sba", sba_impl="sdpa")
    lora_model.load_state_dict(lora_ckpt["state_dict"])
    lora_model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = encoded.input_ids
        n_tokens = input_ids.shape[1]
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt_name} ({n_tokens} tokens)")
        print(f"{'='*80}")

        sm_bos = measure_bos_attention(softmax_model, input_ids, config, device, "softmax", scale=1.0)
        sb_bos = measure_bos_attention(sba_model, input_ids, config, device, "sba", scale=1.0)
        lr_bos = measure_bos_attention(lora_model, input_ids, config, device, "lora", scale=1.5)

        results[prompt_name] = {
            "n_tokens": n_tokens,
            "softmax_bos": sm_bos,
            "sba_bos": sb_bos,
            "lora_bos": lr_bos,
        }

        # Print Layer 3 specifically (the concern layer)
        print(f"  Layer 3 BOS:  Softmax={sm_bos[3]:+7.2f}%  SBA={sb_bos[3]:+7.2f}%  LoRA={lr_bos[3]:+7.2f}%")
        # Print all layers in compact form
        print(f"  {'Layer':<8} {'Softmax':>10} {'SBA':>10} {'LoRA':>10}")
        print(f"  {'-'*38}")
        for l in range(config.n_layer):
            print(f"  Layer {l:<2d}  {sm_bos[l]:+8.2f}%  {sb_bos[l]:+8.2f}%  {lr_bos[l]:+8.2f}%")

    # Summary: Layer 3 across all prompts
    print(f"\n{'='*80}")
    print(f"SUMMARY: Layer 3 BOS Attention Across Prompts")
    print(f"{'='*80}")
    print(f"  {'Prompt':<20} {'Softmax':>10} {'SBA':>10} {'LoRA':>10}")
    print(f"  {'-'*50}")
    for prompt_name in PROMPTS:
        r = results[prompt_name]
        print(f"  {prompt_name:<20} {r['softmax_bos'][3]:+8.2f}%  {r['sba_bos'][3]:+8.2f}%  {r['lora_bos'][3]:+8.2f}%")

    # Overall BOS suppression summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: Mean BOS Attention Across All Layers and Prompts")
    print(f"{'='*80}")
    for model_name, key in [("Softmax", "softmax_bos"), ("SBA", "sba_bos"), ("LoRA", "lora_bos")]:
        all_vals = []
        for prompt_name in PROMPTS:
            all_vals.extend(results[prompt_name][key])
        mean_val = sum(all_vals) / len(all_vals)
        max_val = max(all_vals)
        min_val = min(all_vals)
        print(f"  {model_name:<10}: mean={mean_val:+6.2f}%  min={min_val:+7.2f}%  max={max_val:+7.2f}%")

    # Save
    output_dir = project_root / "sba_standalone/results/lora_conversion"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "multi_prompt_bos_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'multi_prompt_bos_results.json'}")


if __name__ == "__main__":
    main()
