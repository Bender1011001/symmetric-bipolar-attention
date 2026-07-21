# run_lora_conversion.py
import os
import sys
import math
import time
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import ScratchConfig, ScratchLM, evaluate
from data import build_dataloaders
from artifacts import write_json, utc_now

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if original_linear.bias is not None:
            self.original.bias.requires_grad = False
        
        in_features, out_features = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank
        
        # Initialization
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / rank)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return original_out + lora_out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    softmax_path = project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"
    output_dir = project_root / "sba_standalone/results/lora_conversion"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not softmax_path.exists():
        print(f"Pre-trained Softmax checkpoint not found at: {softmax_path}")
        sys.exit(1)

    print("Loading pre-trained Softmax checkpoint...")
    checkpoint = torch.load(softmax_path, map_location="cpu")
    config = ScratchConfig(**checkpoint["scratch_config"])

    # Build dataloaders (using local subset)
    print("Building dataloaders for open_reason_code_mix_25k.jsonl...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = project_root / "data/open_reason_code_mix_25k.jsonl"
    if not dataset_path.exists():
        dataset_path = Path("/root/data/open_reason_code_mix_25k.jsonl")
    print(f"Loading dataset from: {dataset_path}")
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        data_files=[str(dataset_path)],
        max_length=config.block_size,
        train_limit_per_file=21000,
        eval_limit_per_file=4000,
        batch_size=16, # batch_size=16 matches original config
        seed=42
    )

    # Initialize SBA model with Softmax weights
    print("Initializing SBA model with pre-trained Softmax weights...")
    model = ScratchLM(config, "sba", sba_impl="sdpa")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # Apply logit scaling factor of 1.5 to q_dbl inside CausalSelfAttention
    # We patch forward to scale the queries, which scales pre-softmax logits by 1.5
    def sba_forward_hook_scaled(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        causal = self.causal_mask[:, :, :seq_len, :seq_len]
        if attention_mask is not None:
            attn_mask_bool = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
            combined_mask = causal & attn_mask_bool
        else:
            combined_mask = causal

        # Doubled-KV SDPA logic with 1.5x logit scaling
        B, H, T, D = k.shape
        k_dbl = torch.stack([k, -k], dim=-2).view(B, H, 2 * T, D)
        v_dbl = torch.stack([v, -v], dim=-2).view(B, H, 2 * T, D)
        q_dbl = torch.stack([q, q], dim=-2).view(B, H, 2 * T, D)
        
        # Scaling q_dbl by 1.5 multiplies pre-softmax logits by 1.5
        q_dbl = q_dbl * 1.5

        if combined_mask is not None:
            combined_mask_dbl = combined_mask.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        else:
            combined_mask_dbl = None

        y_dbl = F.scaled_dot_product_attention(
            q_dbl, k_dbl, v_dbl, attn_mask=combined_mask_dbl, dropout_p=0.0, is_causal=False
        )
        y = y_dbl[:, :, 1::2, :]

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.dropout(self.proj(y))
        if use_cache:
            return out, (k, v)
        return out

    for block in model.blocks:
        block.attn.forward = sba_forward_hook_scaled.__get__(block.attn, type(block.attn))

    # Evaluate step 0 (initial swap loss)
    print("Evaluating Step 0 (Initial Direct Swap Loss with 1.5x Scaling)...")
    val_loss_step_0 = evaluate(model, val_loader, device, torch.bfloat16)["loss"]
    print(f"Step 0 Val Loss: {val_loss_step_0:.4f}")

    # Set up LoRA wrapping
    print("Wrapping self-attention projection layers in LoRA (rank=16)...")
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to qkv and proj
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank=16)
        block.attn.proj = LoRALinear(block.attn.proj, rank=16)

    model.to(device)

    # Optimizer (only optimize parameters that require gradients - i.e., LoRA)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in lora_params)}")
    
    # LoRA learning rate is typically slightly higher, let's use 3e-4 with cosine decay
    optimizer = torch.optim.AdamW(lora_params, lr=3e-4, weight_decay=0.01)

    steps = 1000
    log_every = 100
    started = time.perf_counter()
    history = []

    # Record step 0
    history.append({
        "step": 0,
        "val_loss": val_loss_step_0,
        "elapsed_seconds": 0.0
    })

    print("Starting LoRA fine-tuning for 1000 steps...")
    step = 0
    train_iter = iter(train_loader)

    model.train()
    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Gradient accumulation = 4
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(4):
            # Split batch into microbatches
            # Since dataloader batch is 16, we split into 4 microbatches of size 4
            start_idx = micro_step * 4
            end_idx = start_idx + 4
            micro_input = input_ids[start_idx:end_idx]
            micro_mask = attention_mask[start_idx:end_idx]
            micro_labels = labels[start_idx:end_idx]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(micro_input, attention_mask=micro_mask, labels=micro_labels)
                scaled_loss = loss / 4
            scaled_loss.backward()
            loss_accum += float(scaled_loss.item())

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        step += 1

        if step % log_every == 0 or step == steps:
            # Evaluate
            model.eval()
            val = evaluate(model, val_loader, device, torch.bfloat16)
            elapsed = time.perf_counter() - started
            print(f"Step {step:4d} | Train Loss: {loss_accum:.4f} | Val Loss: {val['loss']:.4f} | Elapsed: {elapsed:.1f}s")
            history.append({
                "step": step,
                "train_loss": loss_accum,
                "val_loss": val["loss"],
                "elapsed_seconds": elapsed
            })
            model.train()

    # Save results
    results = {
        "format": "sba-lora-conversion-v1",
        "finished_at": utc_now(),
        "history": history
    }
    write_json(output_dir / "results.json", results)
    print("\n=== LoRA Conversion Completed ===")
    print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")

    # Merge and save clean model checkpoint
    print("Merging LoRA weights back into base weights...")
    clean_state_dict = {}
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        if ".original.weight" in key:
            prefix = key.replace(".original.weight", "")
            lora_A = state_dict[f"{prefix}.lora_A"]
            lora_B = state_dict[f"{prefix}.lora_B"]
            submodule = model.get_submodule(prefix)
            scale = submodule.scale
            # Merge W_merged = W_original + (lora_A @ lora_B * scale)^T
            merged_weight = value + (lora_A @ lora_B * scale).t()
            clean_state_dict[f"{prefix}.weight"] = merged_weight
            print(f"  Merged weight for {prefix}")
        elif ".original.bias" in key:
            prefix = key.replace(".original.bias", "")
            clean_state_dict[f"{prefix}.bias"] = value
        elif ".lora_A" in key or ".lora_B" in key:
            continue
        else:
            clean_state_dict[key] = value

    # Construct checkpoint matching the original format
    checkpoint_to_save = {
        "state_dict": clean_state_dict,
        "scratch_config": checkpoint["scratch_config"],
        "attention": "sba",
        "sba_impl": "sdpa",
        "step": 1000,
        "val_loss": history[-1]["val_loss"],
        "lora_history": history
    }
    torch.save(checkpoint_to_save, output_dir / "sba_lora_merged_step001000_12l_12h_768d.pt")
    print(f"Merged model checkpoint saved to: {output_dir / 'sba_lora_merged_step001000_12l_12h_768d.pt'}")

if __name__ == "__main__":
    main()

