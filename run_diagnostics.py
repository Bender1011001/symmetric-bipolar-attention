# run_diagnostics.py
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent

    # Try local paths first, then remote paths
    paths = {
        "softmax": [
            project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt",
            Path("/root/sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt")
        ],
        "sba": [
            project_root / "sba_standalone/results/blackwell_125m_overnight/sba/checkpoints/sba_step015000_12l_12h_768d.pt",
            Path("/root/sba_standalone/results/blackwell_125m_overnight/sba/checkpoints/sba_step015000_12l_12h_768d.pt")
        ],
        "lora_merged": [
            project_root / "sba_standalone/results/lora_conversion/sba_lora_merged_step001000_12l_12h_768d.pt",
            Path("/root/sba_standalone/results/lora_conversion/sba_lora_merged_step001000_12l_12h_768d.pt")
        ]
    }

    resolved_paths = {}
    for name, options in paths.items():
        found = False
        for opt in options:
            if opt.exists():
                resolved_paths[name] = opt
                found = True
                break
        if not found:
            print(f"Warning: Checkpoint {name} not found locally. We will look for it on remote if running remote.")
            # Do not exit immediately; if we are creating this file to sync to remote, the remote paths will exist there.
            resolved_paths[name] = options[1] # fallback to remote path representation

    print("Resolved checkpoint paths:")
    for name, p in resolved_paths.items():
        print(f"  {name}: {p}")

    # Load models if they exist, otherwise raise friendly error
    def load_model_from_path(name, arch_type, sba_impl=None):
        path = resolved_paths[name]
        if not path.exists():
            print(f"Error: {name} checkpoint does not exist at {path}.")
            sys.exit(1)
        ckpt = torch.load(path, map_location="cpu")
        config = ScratchConfig(**ckpt["scratch_config"])
        model = ScratchLM(config, arch_type, sba_impl=sba_impl) if sba_impl else ScratchLM(config, arch_type)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model, config

    softmax_model, config = load_model_from_path("softmax", "softmax")
    sba_model, _ = load_model_from_path("sba", "sba", sba_impl="sdpa")
    lora_model, _ = load_model_from_path("lora_merged", "sba", sba_impl="sdpa")

    # Diagnostic Setup
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    sample_text = "Question:\nSolve for x: 2x + 5 = 17.\n\nAnswer:\n"
    encoded = tokenizer(sample_text, return_tensors="pt").to(device)
    input_ids = encoded.input_ids
    print(f"\nEvaluating with input: {input_ids.shape[1]} tokens")

    # Hook store
    attn_maps = {"softmax": [], "sba": [], "lora": []}
    activations = {"softmax": {}, "sba": {}, "lora": {}}

    # Hooks for attention maps
    def make_attn_hook(model_name, scale=1.0):
        def hook(self, x, attention_mask=None, past_key_value=None, use_cache=False):
            batch, seq_len, channels = x.shape
            qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            causal = self.causal_mask[:, :, :seq_len, :seq_len]
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            # Apply scale (1.5x for LoRA merged model)
            scaled_logits = logits * scale
            
            if model_name in ["sba", "lora"]:
                weights = sba_weights(scaled_logits, causal)
            else:
                floor = torch.full_like(logits, torch.finfo(logits.dtype).min)
                masked_logits = torch.where(causal.to(device=logits.device), logits, floor)
                weights = F.softmax(masked_logits, dim=-1)
                
            attn_maps[model_name].append(weights.detach().cpu())
            
            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self.dropout(self.proj(y))
            if use_cache:
                return out, (k, v)
            return out
        return hook

    # Override forward methods
    for block in softmax_model.blocks:
        block.attn.forward = make_attn_hook("softmax").__get__(block.attn, type(block.attn))
    for block in sba_model.blocks:
        block.attn.forward = make_attn_hook("sba").__get__(block.attn, type(block.attn))
    for block in lora_model.blocks:
        # Scale queries/keys by 1.5 for LoRA model to match logit scaling
        block.attn.forward = make_attn_hook("lora", scale=1.5).__get__(block.attn, type(block.attn))

    # Hooks for activations
    def get_activation_hook(name, store_dict):
        def hook(module, input, output):
            if isinstance(output, tuple):
                val = output[0].detach().cpu().float()
            else:
                val = output.detach().cpu().float()
            store_dict[name] = val
        return hook

    for i, block in enumerate(softmax_model.blocks):
        block.register_forward_hook(get_activation_hook(f"layer_{i}", activations["softmax"]))
    for i, block in enumerate(sba_model.blocks):
        block.register_forward_hook(get_activation_hook(f"layer_{i}", activations["sba"]))
    for i, block in enumerate(lora_model.blocks):
        block.register_forward_hook(get_activation_hook(f"layer_{i}", activations["lora"]))

    # Forward passes
    with torch.no_grad():
        softmax_model(input_ids)
        sba_model(input_ids)
        lora_model(input_ids)

    # 1. Analyze Attention Sinks (BOS Attention)
    print("\n" + "="*80)
    print(f"{'Layer':<6} | {'Softmax BOS %':<15} | {'SBA BOS %':<15} | {'LoRA BOS %':<15}")
    print("-"*80)
    
    bos_stats = {"softmax": [], "sba": [], "lora": []}
    
    for l in range(config.n_layer):
        sm_bos = attn_maps["softmax"][l][0][:, 1:, 0].mean().item() * 100
        sb_bos = attn_maps["sba"][l][0][:, 1:, 0].mean().item() * 100
        lr_bos = attn_maps["lora"][l][0][:, 1:, 0].mean().item() * 100
        
        bos_stats["softmax"].append(sm_bos)
        bos_stats["sba"].append(sb_bos)
        bos_stats["lora"].append(lr_bos)
        
        print(f"Layer {l:<2d} | {sm_bos:12.2f}% | {sb_bos:12.2f}% | {lr_bos:12.2f}%")

    # 2. Analyze Activation Outliers (p99.9 and Max)
    print("\n" + "="*80)
    print(f"{'Layer':<6} | {'Softmax Max / p99.9':<22} | {'SBA Max / p99.9':<22} | {'LoRA Max / p99.9':<22}")
    print("-"*80)
    
    outlier_stats = {"softmax_max": [], "softmax_p999": [], "sba_max": [], "sba_p999": [], "lora_max": [], "lora_p999": []}
    
    for l in range(config.n_layer):
        sm_act = activations["softmax"][f"layer_{l}"].abs()
        sb_act = activations["sba"][f"layer_{l}"].abs()
        lr_act = activations["lora"][f"layer_{l}"].abs()
        
        sm_max = sm_act.max().item()
        sb_max = sb_act.max().item()
        lr_max = lr_act.max().item()
        
        sm_p999 = torch.quantile(sm_act, 0.999).item()
        sb_p999 = torch.quantile(sb_act, 0.999).item()
        lr_p999 = torch.quantile(lr_act, 0.999).item()
        
        outlier_stats["softmax_max"].append(sm_max)
        outlier_stats["softmax_p999"].append(sm_p999)
        outlier_stats["sba_max"].append(sb_max)
        outlier_stats["sba_p999"].append(sb_p999)
        outlier_stats["lora_max"].append(lr_max)
        outlier_stats["lora_p999"].append(lr_p999)
        
        print(f"Layer {l:<2d} | {sm_max:7.2f} / {sm_p999:7.2f}  | {sb_max:7.2f} / {sb_p999:7.2f}  | {lr_max:7.2f} / {lr_p999:7.2f}")

    # Save outputs to JSON
    output_dir = project_root / "sba_standalone/results/lora_conversion"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    diagnostics = {
        "bos_stats": bos_stats,
        "outlier_stats": outlier_stats
    }
    
    with open(output_dir / "diagnostic_results.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nDiagnostic results saved to: {output_dir / 'diagnostic_results.json'}")

if __name__ == "__main__":
    main()
