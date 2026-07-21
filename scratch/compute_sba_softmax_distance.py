import math
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate_scratch_tokenized_checkpoint import load_model
from data import build_dataloaders
from conversion_diagnostics import sba_weights_centered

def parse_args():
    parser = argparse.ArgumentParser(description="Compute mathematical distance between SBA and Softmax weights")
    parser.add_argument("--checkpoint", default="sba_standalone/results/v3_softmax_bridge/sba_bridge_v3_step003000_12l_12h_768d.pt")
    parser.add_argument("--data-file", default="data/open_reason_code_mix_25k.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model from: {args.checkpoint}")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    model.eval()

    # Reconstruct tau and bias from model or checkpoint
    tau_params = getattr(model, "tau_params", None)
    bias_params = getattr(model, "bias_params", None)
    if tau_params is None or bias_params is None:
        if "lora_history" in checkpoint:
            print("Detected LoRA conversion checkpoint. Using tau = 1/1.5 and bias = 0.0.")
            tau_params = [torch.full((model.config.n_head,), 1.0 / 1.5, device=device) for _ in range(model.config.n_layer)]
            bias_params = [torch.zeros((model.config.n_head,), device=device) for _ in range(model.config.n_layer)]
        else:
            print("Detected standard SBA checkpoint without learned parameters. Using default tau = 1.0 and bias = 0.0.")
            tau_params = [torch.full((model.config.n_head,), 1.0, device=device) for _ in range(model.config.n_layer)]
            bias_params = [torch.zeros((model.config.n_head,), device=device) for _ in range(model.config.n_layer)]

    n_layer = model.config.n_layer
    n_head = model.config.n_head
    print(f"Model loaded: {n_layer}L, {n_head}H, {model.config.n_embd}D")
    
    # Print average bias and tau per layer
    print("\nLearned Parameters:")
    for i in range(n_layer):
        b_mean = bias_params[i].mean().item()
        tau_mean = tau_params[i].mean().item()
        print(f"  Layer {i:2d}: Mean b = {b_mean:.4f} | Mean tau = {tau_mean:.4f}")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading dataset: {args.data_file}")
    train_loader, val_loader, manifest = build_dataloaders(
        tokenizer=tokenizer,
        data_files=[args.data_file],
        max_length=model.config.block_size,
        train_limit_per_file=100,
        eval_limit_per_file=100,
        batch_size=4,
        seed=42,
    )

    # Accumulators for metrics
    # List of dicts, one per layer
    layer_stats = [{
        "mae_total": 0.0,
        "max_diff_max": 0.0,
        "rmse_total": 0.0,
        "neg_pct_total": 0.0,
        "zero_pct_total": 0.0,
        "row_l1_total": 0.0,
        "n_samples": 0
    } for _ in range(n_layer)]

    original_forwards = []

    def _make_compare_forward(layer_idx: int):
        def _forward(self_attn, x, attention_mask=None, past_key_value=None, use_cache=False):
            batch, seq_len, channels = x.shape
            qkv = self_attn.qkv(x).view(
                batch, seq_len, 3, self_attn.n_head, self_attn.head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            causal = self_attn.causal_mask[:, :, :seq_len, :seq_len]
            if attention_mask is not None:
                attn_mask_bool = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
                combined_mask = causal & attn_mask_bool
            else:
                combined_mask = causal

            logits = (
                torch.matmul(q, k.transpose(-1, -2))
                / math.sqrt(self_attn.head_dim)
            )

            # Compute SBA weights
            tau = tau_params[layer_idx].to(device=logits.device)
            bias = bias_params[layer_idx].to(device=logits.device)
            w_sba, _, _ = sba_weights_centered(logits, combined_mask, tau, bias)

            # Compute Softmax weights
            floor_val = torch.finfo(logits.dtype).min
            floor = torch.full_like(logits, floor_val)
            valid = combined_mask.to(device=logits.device)
            masked = torch.where(valid, logits, floor)
            w_softmax = F.softmax(masked, dim=-1)

            # Masked difference calculations (only count visible positions)
            valid_mask = valid.expand_as(logits)
            
            sba_flat = w_sba[valid_mask].detach().float()
            softmax_flat = w_softmax[valid_mask].detach().float()

            abs_diff = torch.abs(sba_flat - softmax_flat)
            mae = abs_diff.mean().item()
            max_diff = abs_diff.max().item()
            rmse = torch.sqrt((abs_diff ** 2).mean()).item()

            neg_pct = (sba_flat < 0.0).float().mean().item() * 100.0
            zero_pct = (sba_flat == 0.0).float().mean().item() * 100.0
            
            # Row L1 norms (weights are already masked, sum over key dim)
            if attention_mask is not None:
                qmask = attention_mask.to(device=logits.device, dtype=torch.bool)[:, None, :].expand(-1, self_attn.n_head, -1)
                row_l1 = w_sba.abs().sum(dim=-1)[qmask].detach().float().mean().item()
            else:
                row_l1 = w_sba.abs().sum(dim=-1).detach().float().mean().item()

            # Accumulate layer stats
            stats = layer_stats[layer_idx]
            stats["mae_total"] += mae
            stats["max_diff_max"] = max(stats["max_diff_max"], max_diff)
            stats["rmse_total"] += rmse
            stats["neg_pct_total"] += neg_pct
            stats["zero_pct_total"] += zero_pct
            stats["row_l1_total"] += row_l1
            stats["n_samples"] += 1

            # Complete forward
            y = torch.matmul(w_sba, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))
            if use_cache:
                return out, (k, v)
            return out

        return _forward

    # Patch forwards
    for i, block in enumerate(model.blocks):
        original_forwards.append(block.attn.forward)
        block.attn.forward = _make_compare_forward(i).__get__(block.attn, type(block.attn))

    # Run batches
    print(f"\nRunning {args.batches} verification batches...")
    batch_count = 0
    for batch in val_loader:
        if batch_count >= args.batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        batch_count += 1

    # Restore forwards
    for i, block in enumerate(model.blocks):
        block.attn.forward = original_forwards[i]

    # Print mathematical distance report
    print(f"\n{'='*80}")
    print("  MATHEMATICAL DISTANCE: CENTERED SBA (b ~ 3.45) vs. STANDARD SOFTMAX")
    print(f"{'='*80}")
    print(f"  {'Layer':<6} | {'Mean Abs Diff':<13} | {'Max Abs Diff':<13} | {'RMSE':<8} | {'Neg Wt %':<8} | {'Zero Wt %':<9} | {'Row L1':<6}")
    print(f"  {'-'*78}")
    
    total_mae = 0.0
    total_rmse = 0.0
    max_diff_all = 0.0
    total_neg_pct = 0.0
    total_zero_pct = 0.0
    total_row_l1 = 0.0

    for i, stats in enumerate(layer_stats):
        n = max(stats["n_samples"], 1)
        mae = stats["mae_total"] / n
        max_diff = stats["max_diff_max"]
        rmse = stats["rmse_total"] / n
        neg_pct = stats["neg_pct_total"] / n
        zero_pct = stats["zero_pct_total"] / n
        row_l1 = stats["row_l1_total"] / n

        total_mae += mae
        total_rmse += rmse
        max_diff_all = max(max_diff_all, max_diff)
        total_neg_pct += neg_pct
        total_zero_pct += zero_pct
        total_row_l1 += row_l1

        print(f"  Layer {i:<2d} | {mae:<13.6f} | {max_diff:<13.6f} | {rmse:<8.6f} | {neg_pct:<8.2f}% | {zero_pct:<9.2f}% | {row_l1:<6.4f}")

    avg_mae = total_mae / n_layer
    avg_rmse = total_rmse / n_layer
    avg_neg_pct = total_neg_pct / n_layer
    avg_zero_pct = total_zero_pct / n_layer
    avg_row_l1 = total_row_l1 / n_layer

    print(f"  {'-'*78}")
    print(f"  Average| {avg_mae:<13.6f} | {max_diff_all:<13.6f} | {avg_rmse:<8.6f} | {avg_neg_pct:<8.2f}% | {avg_zero_pct:<9.2f}% | {avg_row_l1:<6.4f}")
    print(f"{'='*80}")

    print("\nSummary Interpretation:")
    print(f"  - The average absolute difference per attention weight is {avg_mae:.6f} ({avg_mae*100:.4f}% of attention weight scale).")
    print(f"  - The maximum discrepancy between any attention weight and Softmax is {max_diff_all:.6f}.")
    print(f"  - On average, only {avg_neg_pct:.2f}% of the attention weights in the model are negative.")
    print(f"  - The row L1-norm of SBA is {avg_row_l1:.4f} (compared to 1.0000 for standard Softmax).")
    
    # Save mathematical distance results
    out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "avg_mae": avg_mae,
        "max_diff_all": max_diff_all,
        "avg_rmse": avg_rmse,
        "avg_neg_pct": avg_neg_pct,
        "avg_zero_pct": avg_zero_pct,
        "avg_row_l1": avg_row_l1,
        "learned_params": [
            {
                "layer": i,
                "mean_b": bias_params[i].mean().item(),
                "mean_tau": tau_params[i].mean().item()
            }
            for i in range(n_layer)
        ]
    }
    with open(out_dir / "mathematical_distance.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved mathematical distance report to: {out_dir / 'mathematical_distance.json'}")

if __name__ == "__main__":
    main()
