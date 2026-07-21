# gemma4_gradient_audit.py
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Ensure we can load local patch.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import patch
from patch import patch_gemma4_sba, load_sba_bridge_state

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Section 5 Gradient Audit on Gemma 4 SBA checkpoint")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-4-E4B")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="PEFT SBA adapter checkpoint directory")
    parser.add_argument("--validation_file", type=str, default="data/ecot_validation.jsonl")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to run audit on")
    parser.add_argument("--output_file", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )

    print("Patching model with SBA...")
    patch_gemma4_sba(model)

    print(f"Loading PEFT adapter weights from {args.checkpoint_dir}...")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir, adapter_name="eval_adapter")

    bridge_state_path = os.path.join(args.checkpoint_dir, "sba_bridge_v3.pt")
    if os.path.exists(bridge_state_path):
        print(f"Loading SBA bridge parameters from {bridge_state_path}...")
        load_sba_bridge_state(model, bridge_state_path)
    else:
        raise FileNotFoundError(f"SBA bridge state not found at {bridge_state_path}")

    # Enable gradients for PEFT adapter and SBA bridge parameters
    print("Enabling gradients for PEFT and SBA parameters...")
    for name, param in model.named_parameters():
        if "lora" in name or "sba" in name:
            param.requires_grad = True

    # Set up hooks to capture attention weights and retain grads
    captured_weights = []
    original_stable = patch._stable_sinh_over_sum_cosh

    def hooked_stable(x, valid):
        w = original_stable(x, valid)
        if w.requires_grad:
            w.retain_grad()
            captured_weights.append(w)
        return w

    patch._stable_sinh_over_sum_cosh = hooked_stable

    # Load validation data
    print(f"Loading validation samples from {args.validation_file}...")
    samples = []
    with open(args.validation_file, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            # We want samples that contain <INHIBIT> and <PROPOSE> tags
            text = d.get("text", "")
            if "<INHIBIT>" in text and "<PROPOSE>" in text:
                samples.append(d)
                if len(samples) >= args.num_samples:
                    break

    print(f"Loaded {len(samples)} valid validation samples for auditing.")
    if len(samples) == 0:
        print("No valid samples with both <INHIBIT> and <PROPOSE> found!")
        return

    results = []
    vocab_size = getattr(model.config, "vocab_size", None) or model.get_input_embeddings().weight.shape[0]


    for idx, sample in enumerate(samples):
        text = sample["text"]
        
        # Locate block spans in characters
        p_start = text.find("<PROPOSE>") + len("<PROPOSE>")
        p_end = text.find("</PROPOSE>")
        
        i_start = text.find("<INHIBIT>") + len("<INHIBIT>")
        i_end = text.find("</INHIBIT>")
        
        if p_start == -1 or p_end == -1 or i_start == -1 or i_end == -1:
            continue
            
        # Tokenize with offsets
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        offsets = encoding["offset_mapping"][0].cpu().tolist()
        
        seq_len = input_ids.shape[1]
        
        propose_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        inhibit_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        for t, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                continue
            if p_start <= start < p_end:
                propose_mask[t] = True
            if i_start <= start < i_end:
                inhibit_mask[t] = True
                
        # We need at least some tokens in both masks
        if propose_mask.sum() == 0 or inhibit_mask.sum() == 0:
            continue
            
        # --- PROPOSE GRADIENT EVALUATION ---
        model.zero_grad()
        captured_weights.clear()
        
        # Forward pass (grads enabled)
        with torch.enable_grad():
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits

        
        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()
        shift_propose_mask = propose_mask[..., 1:].contiguous()
        
        propose_active_labels = shift_labels[shift_propose_mask]
        propose_active_logits = shift_logits[shift_propose_mask]
        
        if propose_active_labels.numel() == 0:
            continue
            
        loss_propose = F.cross_entropy(propose_active_logits, propose_active_labels)
        loss_propose.backward()
        
        # Extract gradients of negative weights
        grad_neg_propose = []
        for w in captured_weights:
            if w.grad is not None:
                # Find elements where attention weight is negative
                neg_w_mask = w < 0.0
                if neg_w_mask.any():
                    grad_neg_propose.append(w.grad[neg_w_mask].flatten().cpu())
                    
        if len(grad_neg_propose) == 0:
            continue
            
        grad_neg_propose = torch.cat(grad_neg_propose)
        norm_neg_propose = torch.norm(grad_neg_propose).item()
        mean_abs_neg_propose = grad_neg_propose.abs().mean().item()
        
        # --- INHIBIT GRADIENT EVALUATION ---
        model.zero_grad()
        captured_weights.clear()
        
        # Forward pass (grads enabled)
        with torch.enable_grad():
            outputs = model(input_ids, use_cache=False)
            logits = outputs.logits
        
        shift_logits = logits[0, :-1, :].contiguous()
        shift_labels = input_ids[0, 1:].contiguous()
        shift_inhibit_mask = inhibit_mask[..., 1:].contiguous()
        
        inhibit_active_labels = shift_labels[shift_inhibit_mask]
        inhibit_active_logits = shift_logits[shift_inhibit_mask]
        
        if inhibit_active_labels.numel() == 0:
            continue
            
        loss_inhibit = F.cross_entropy(inhibit_active_logits, inhibit_active_labels)
        loss_inhibit.backward()
        
        # Extract gradients of negative weights
        grad_neg_inhibit = []
        for w in captured_weights:
            if w.grad is not None:
                neg_w_mask = w < 0.0
                if neg_w_mask.any():
                    grad_neg_inhibit.append(w.grad[neg_w_mask].flatten().cpu())
                    
        if len(grad_neg_inhibit) == 0:
            continue
            
        grad_neg_inhibit = torch.cat(grad_neg_inhibit)
        norm_neg_inhibit = torch.norm(grad_neg_inhibit).item()
        mean_abs_neg_inhibit = grad_neg_inhibit.abs().mean().item()
        
        # R_grad ratios
        r_grad_norm = norm_neg_inhibit / max(norm_neg_propose, 1e-9)
        r_grad_mean_abs = mean_abs_neg_inhibit / max(mean_abs_neg_propose, 1e-9)
        
        results.append({
            "sample_id": sample.get("id", f"sample_{idx}"),
            "num_propose_tokens": int(propose_mask.sum().item()),
            "num_inhibit_tokens": int(inhibit_mask.sum().item()),
            "propose_loss": float(loss_propose.item()),
            "inhibit_loss": float(loss_inhibit.item()),
            "norm_neg_propose": norm_neg_propose,
            "norm_neg_inhibit": norm_neg_inhibit,
            "mean_abs_neg_propose": mean_abs_neg_propose,
            "mean_abs_neg_inhibit": mean_abs_neg_inhibit,
            "r_grad_norm": r_grad_norm,
            "r_grad_mean_abs": r_grad_mean_abs
        })
        
        print(f"Sample {idx+1}/{len(samples)} | ID: {sample.get('id', 'N/A')[:15]}... | R_grad (Norm): {r_grad_norm:.4f} | R_grad (Mean Abs): {r_grad_mean_abs:.4f}")

    if len(results) == 0:
        print("No samples could be successfully audited.")
        return

    avg_r_grad_norm = sum(r["r_grad_norm"] for r in results) / len(results)
    avg_r_grad_mean_abs = sum(r["r_grad_mean_abs"] for r in results) / len(results)
    
    print("\n=========================================")
    print("GRADIENT AUDIT SUMMARY:")
    print(f"  Processed Samples: {len(results)}")
    print(f"  Average R_grad (L2 Norm): {avg_r_grad_norm:.4f}")
    print(f"  Average R_grad (Mean Abs): {avg_r_grad_mean_abs:.4f}")
    print(f"  Passes threshold (R_grad >= 3.0): {avg_r_grad_norm >= 3.0}")
    print("=========================================")

    output_data = {
        "model_name_or_path": args.model_name_or_path,
        "checkpoint_dir": args.checkpoint_dir,
        "average_r_grad_norm": avg_r_grad_norm,
        "average_r_grad_mean_abs": avg_r_grad_mean_abs,
        "passes_threshold": bool(avg_r_grad_norm >= 3.0),
        "samples": results
    }

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"Audit results saved to {args.output_file}")

if __name__ == "__main__":
    main()
