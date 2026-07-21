import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can load local patch.py
sys.path.insert(0, "/root/sba_standalone")
import patch
from patch import patch_gemma4_sba, load_sba_bridge_state

PROMPTS = {
    "math": "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n",
    "code_completion": 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n    if n <= 1:\n        return n\n    return',
    "long_context": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It relies on self-attention mechanisms to process input sequences in parallel, replacing recurrent neural networks. The key innovation was the multi-head attention mechanism, which allows the model to attend to different parts of the input simultaneously. Since then, transformers have become the dominant architecture for natural language processing tasks.\n\nQuestion: What was the key innovation of the transformer architecture?\n\nAnswer:\n",
    "conversational": "User: Hey, can you explain what attention sinks are in large language models?\n\nAssistant:"
}

captured_layer_metrics = []
original_stable_sinh_over_sum_cosh = patch._stable_sinh_over_sum_cosh

def hooked_stable_sinh_over_sum_cosh(x, valid):
    weights = original_stable_sinh_over_sum_cosh(x, valid)
    
    # Process metrics ON-THE-FLY to prevent memory leaks from massive tensor storage
    with torch.no_grad():
        x_float = x.float()
        valid_f = valid.to(dtype=x_float.dtype)
        abs_x = x_float.abs()
        neg_inf = torch.full_like(abs_x, -float("inf"))
        
        row_max = torch.where(valid, abs_x, neg_inf).amax(dim=-1, keepdim=True)
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exp_scaled = torch.exp(torch.clamp(abs_x - row_max, max=0.0)) * valid_f
        exp_neg_2_abs = torch.exp(torch.clamp(-2.0 * abs_x, min=-80.0, max=0.0))
        cosh_scaled = exp_scaled * (1.0 + exp_neg_2_abs)
        
        denom = cosh_scaled.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
        r_bos = cosh_scaled[..., 0] / denom.squeeze(-1) 
        
        B, H, T, _ = weights.shape
        valid_queries = valid.any(dim=-1)
        
        mask_non_bos = valid_queries.clone()
        if T > 1:
            mask_non_bos[:, :, 0] = False
        valid_q_count = mask_non_bos.sum().clamp_min(1)
        
        mean_signed_bos = (weights[..., 0][mask_non_bos].sum() / valid_q_count).item() * 100.0
        mean_rbos = (r_bos[mask_non_bos].sum() / valid_q_count).item() * 100.0
        
        row_l1 = weights.abs().sum(dim=-1)
        mean_row_l1 = (row_l1[mask_non_bos].sum() / valid_q_count).item()
        
        # CORRECTED MATH: Divide strictly by the valid edge count, ignoring masked padding
        neg_mask = (weights < 0) & valid
        valid_edges = valid.sum().clamp_min(1)
        
        neg_fraction = neg_mask.sum().item() / valid_edges.item()
        neg_sum_abs = weights[neg_mask].abs().sum().item() / valid_edges.item()
        
        captured_layer_metrics.append({
            "mean_signed_bos_pct": mean_signed_bos,
            "mean_rbos_pct": mean_rbos,
            "mean_row_l1": mean_row_l1,
            "neg_fraction": neg_fraction,
            "neg_sum_abs": neg_sum_abs,
        })
    return weights

patch._stable_sinh_over_sum_cosh = hooked_stable_sinh_over_sum_cosh


def run_diagnostics_on_config(model, tokenizer, device, config_label):
    print(f"\n--- Diagnostics: {config_label} ---")
    results = {}
    for prompt_name, prompt_text in PROMPTS.items():
        captured_layer_metrics.clear()
        encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            model(encoded.input_ids, use_cache=False)
            
        layers_metrics = []
        for idx, layer_data in enumerate(captured_layer_metrics):
            m = layer_data.copy()
            m["layer"] = idx
            layers_metrics.append(m)
            
        L = len(layers_metrics)
        if L > 0:
            mean_signed_bos = sum(m["mean_signed_bos_pct"] for m in layers_metrics) / L
            mean_rbos = sum(m["mean_rbos_pct"] for m in layers_metrics) / L
            mean_row_l1 = sum(m["mean_row_l1"] for m in layers_metrics) / L
            mean_neg_fraction = sum(m["neg_fraction"] for m in layers_metrics) / L
            mean_neg_sum_abs = sum(m["neg_sum_abs"] for m in layers_metrics) / L
        else:
            mean_signed_bos, mean_rbos, mean_row_l1, mean_neg_fraction, mean_neg_sum_abs = 0.0, 0.0, 1.0, 0.0, 0.0
            
        print(f"  Prompt: {prompt_name:<18} | Signed BOS: {mean_signed_bos:>+6.2f}% | r_BOS: {mean_rbos:>5.2f}% | Row L1: {mean_row_l1:.4f} | Neg Frac: {mean_neg_fraction*100:5.2f}%")
        results[prompt_name] = {
            "mean_signed_bos_pct": mean_signed_bos,
            "mean_rbos_pct": mean_rbos,
            "mean_row_l1": mean_row_l1,
            "neg_fraction": mean_neg_fraction,
            "neg_sum_abs": mean_neg_sum_abs,
            "layers": layers_metrics,
        }
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "google/gemma-4-E4B"
    results_dir = "/root/sba_standalone/results/gemma4_e4b_sba_bridge"
    output_file = "/root/sba_standalone/results/gemma4_diagnostics_report.json"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    diagnostics_output = {}
    
    print("\nLoading Base Model ONCE into RAM...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    
    # Run Softmax Baseline
    patch_gemma4_sba(base_model) 
    diagnostics_output["softmax"] = run_diagnostics_on_config(base_model, tokenizer, device, "Softmax Baseline")
    
    checkpoints = []
    if os.path.exists(results_dir):
        for item in sorted(os.listdir(results_dir)):
            item_path = os.path.join(results_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                try:
                    checkpoints.append((int(item.split("-")[1]), item_path))
                except ValueError:
                    pass
    checkpoints.sort()
    
    if checkpoints:
        # Wrap PEFT dynamic hot-swapper
        peft_model = PeftModel.from_pretrained(base_model, checkpoints[0][1], adapter_name="eval_adapter")
        for step, ckpt_dir in checkpoints:
            print(f"\nEvaluating SBA Checkpoint-{step}...")
            
            # Instant memory-efficient adapter hot-swapping
            peft_model.load_adapter(ckpt_dir, adapter_name=f"c_{step}")
            peft_model.set_adapter(f"c_{step}")
            
            bridge_state_path = os.path.join(ckpt_dir, "sba_bridge_v3.pt")
            if os.path.exists(bridge_state_path):
                load_sba_bridge_state(peft_model, bridge_state_path)
            
            diagnostics_output[f"checkpoint_{step}"] = run_diagnostics_on_config(peft_model, tokenizer, device, f"SBA Checkpoint-{step}")
            peft_model.delete_adapter(f"c_{step}")  # Prevent sequential RAM accumulation

    with open(output_file, "w") as f:
        json.dump(diagnostics_output, f, indent=2)
    print(f"\nDiagnostics report successfully saved to: {output_file}")


if __name__ == "__main__":
    main()
