# run_lora_conversion_v2.py
"""
Experiment 1: Anti-Sink Regularized LoRA Conversion

Converts a pre-trained softmax checkpoint to SBA attention using LoRA
with compound loss: CE + BOS attention penalty + activation outlier penalty.
Uses learnable per-layer logit temperature instead of hardcoded 1.5× scaling.
Includes cosine LR schedule with warmup.

Key differences from v1:
  - Compound loss forces SBA-native routing (no BOS sinks)
  - Learnable per-layer temperature (init=1.0, learned during training)
  - Cosine LR with 100-step warmup
  - Periodic diagnostic evaluation (every 500 steps)
  - Eager SBA computation for gradient flow through BOS penalty
"""
import os
import sys
import math
import time
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import (
    ScratchConfig,
    ScratchLM,
    sba_weights,
    evaluate,
)
from data import build_dataloaders
from artifacts import write_json, utc_now
from conversion_diagnostics import (
    run_diagnostic,
    print_diagnostic_summary,
    save_diagnostic,
)


# ---------------------------------------------------------------------------
# LoRA Wrapper
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""

    def __init__(
        self, original: nn.Linear, rank: int = 16, alpha: float = 32.0
    ) -> None:
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        if original.bias is not None:
            self.original.bias.requires_grad = False

        in_f, out_f = original.in_features, original.out_features
        self.lora_A = nn.Parameter(torch.zeros(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.scale = alpha / rank

        # Kaiming-uniform init for A (standard LoRA), zeros for B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


# ---------------------------------------------------------------------------
# Conversion Config
# ---------------------------------------------------------------------------
@dataclass
class ConversionConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    total_steps: int = 3000
    warmup_steps: int = 100
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 16
    grad_accum: int = 4
    lambda_bos: float = 0.5
    lambda_outlier: float = 0.05
    outlier_threshold: float = 30.0
    log_every: int = 100
    diag_every: int = 500
    init_temp: float = 1.0


# ---------------------------------------------------------------------------
# Cosine LR with linear warmup
# ---------------------------------------------------------------------------
def make_lr_schedule(cfg: ConversionConfig):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(
            1, cfg.total_steps - cfg.warmup_steps
        )
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Attention weight capture via eager SBA forward
# ---------------------------------------------------------------------------
def install_sba_forward_with_capture(
    model: ScratchLM,
    layer_temps: nn.ParameterList,
    attn_storage: list,
    hidden_storage: list,
) -> None:
    """Replace each attention block's forward with eager SBA that stores
    attention weights (in the computation graph) and hidden states."""

    def _make_sba_forward(layer_idx: int):
        def _forward(
            self_attn,
            x,
            attention_mask=None,
            past_key_value=None,
            use_cache=False,
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
            # Apply learnable per-layer temperature
            logits = logits * layer_temps[layer_idx]

            # Eager SBA weights — kept in computation graph for BOS penalty
            weights = sba_weights(logits, causal)
            attn_storage[layer_idx] = weights

            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))

            if use_cache:
                return out, (k, v)
            return out

        return _forward

    def _make_hidden_hook(layer_idx: int):
        def _hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            hidden_storage[layer_idx] = h
        return _hook

    for i, block in enumerate(model.blocks):
        block.attn.forward = _make_sba_forward(i).__get__(
            block.attn, type(block.attn)
        )
        block.register_forward_hook(_make_hidden_hook(i))


# ---------------------------------------------------------------------------
# Penalty losses
# ---------------------------------------------------------------------------
def compute_bos_penalty(attn_storage: list) -> torch.Tensor:
    """Mean |attention to BOS| across layers and positions.

    Penalizes the model for concentrating attention on position 0 (BOS),
    forcing it to find SBA-native routing that distributes or suppresses
    attention at position 0.
    """
    total = torch.tensor(0.0, device=attn_storage[0].device)
    count = 0
    for w in attn_storage:
        if w is None:
            continue
        T = w.shape[-1]
        if T > 1:
            # Attention from non-BOS positions to BOS position
            bos_attn = w[:, :, 1:, 0]  # [B, H, T-1]
            total = total + bos_attn.abs().mean()
            count += 1
    return total / max(count, 1)


def compute_outlier_penalty(
    hidden_storage: list, threshold: float = 30.0
) -> torch.Tensor:
    """Mean excess activation magnitude above threshold.

    Penalizes hidden state values that exceed the threshold, pushing
    the model toward SBA-native flat activation profiles.
    """
    total = torch.tensor(0.0, device=hidden_storage[0].device)
    count = 0
    for h in hidden_storage:
        if h is None:
            continue
        excess = F.relu(h.abs() - threshold)
        total = total + excess.mean()
        count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Merge LoRA weights into clean checkpoint
# ---------------------------------------------------------------------------
def merge_lora_weights(model: ScratchLM) -> dict:
    """Merge LoRA adapters into base weights and return clean state_dict."""
    clean = {}
    state_dict = model.state_dict()
    for key, value in state_dict.items():
        if ".original.weight" in key:
            prefix = key.replace(".original.weight", "")
            lora_A = state_dict[f"{prefix}.lora_A"]
            lora_B = state_dict[f"{prefix}.lora_B"]
            submodule = model.get_submodule(prefix)
            scale = submodule.scale
            merged = value + (lora_A @ lora_B * scale).t()
            clean[f"{prefix}.weight"] = merged
        elif ".original.bias" in key:
            prefix = key.replace(".original.bias", "")
            clean[f"{prefix}.bias"] = value
        elif ".lora_A" in key or ".lora_B" in key:
            continue
        else:
            clean[key] = value
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = ConversionConfig()
    project_root = Path(__file__).resolve().parent.parent

    # --- Paths ---
    softmax_path = (
        project_root
        / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints"
        / "softmax_step015000_12l_12h_768d.pt"
    )
    if not softmax_path.exists():
        softmax_path = Path(
            "/root/sba_standalone/results/blackwell_125m_overnight/softmax"
            "/checkpoints/softmax_step015000_12l_12h_768d.pt"
        )
    output_dir = project_root / "sba_standalone/results/lora_conversion_v2"
    if not output_dir.exists():
        output_dir = Path("/root/sba_standalone/results/lora_conversion_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not softmax_path.exists():
        print(f"ERROR: Softmax checkpoint not found: {softmax_path}")
        sys.exit(1)

    # --- Load checkpoint & config ---
    print(f"Loading softmax checkpoint: {softmax_path}")
    checkpoint = torch.load(softmax_path, map_location="cpu")
    model_config = ScratchConfig(**checkpoint["scratch_config"])

    # --- Tokenizer & data ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = project_root / "data/open_reason_code_mix_25k.jsonl"
    if not dataset_path.exists():
        dataset_path = Path("/root/data/open_reason_code_mix_25k.jsonl")
    print(f"Loading dataset: {dataset_path}")
    train_loader, val_loader, manifest = build_dataloaders(
        tokenizer=tokenizer,
        data_files=[str(dataset_path)],
        max_length=model_config.block_size,
        train_limit_per_file=21000,
        eval_limit_per_file=4000,
        batch_size=cfg.batch_size,
        seed=42,
    )
    print(
        f"Train: {manifest['train_examples']} examples, "
        f"Eval: {manifest['eval_examples']} examples"
    )

    # --- Initialize SBA model with softmax weights ---
    print("Initializing SBA model with softmax weights...")
    model = ScratchLM(model_config, "sba", sba_impl="sdpa")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    # --- Learnable per-layer temperatures ---
    layer_temps = nn.ParameterList(
        [nn.Parameter(torch.tensor(cfg.init_temp)) for _ in range(model_config.n_layer)]
    ).to(device)

    # --- Install custom SBA forward with capture ---
    attn_storage = [None] * model_config.n_layer
    hidden_storage = [None] * model_config.n_layer
    install_sba_forward_with_capture(model, layer_temps, attn_storage, hidden_storage)

    # --- Evaluate Step 0 (pre-LoRA, with learnable temps at init) ---
    print("Evaluating Step 0 (direct swap, no LoRA)...")
    step0_loss = evaluate(model, val_loader, device, torch.bfloat16)["loss"]
    print(f"Step 0 Val Loss: {step0_loss:.4f}")

    # --- Apply LoRA ---
    print(f"Applying LoRA (rank={cfg.lora_rank}) to QKV and proj...")
    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
        block.attn.proj = LoRALinear(
            block.attn.proj, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
    model.to(device)

    # Trainable params: LoRA + temperatures
    lora_params = [p for p in model.parameters() if p.requires_grad]
    temp_params = list(layer_temps.parameters())
    all_trainable = lora_params + temp_params
    n_trainable = sum(p.numel() for p in all_trainable)
    n_total = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in layer_temps.parameters()
    )
    print(f"Trainable parameters: {n_trainable:,} ({100*n_trainable/n_total:.2f}%)")
    print(f"  LoRA: {sum(p.numel() for p in lora_params):,}")
    print(f"  Temperatures: {sum(p.numel() for p in temp_params):,} ({model_config.n_layer} scalars)")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        all_trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(optimizer, make_lr_schedule(cfg))

    # --- Training loop ---
    history = [{"step": 0, "val_loss": step0_loss, "elapsed_seconds": 0.0}]
    started = time.perf_counter()
    step = 0
    train_iter = iter(train_loader)

    print(
        f"\nStarting training: {cfg.total_steps} steps, "
        f"λ_bos={cfg.lambda_bos}, λ_outlier={cfg.lambda_outlier}, "
        f"outlier_threshold={cfg.outlier_threshold}"
    )

    model.train()
    while step < cfg.total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss_ce_accum = 0.0
        loss_bos_accum = 0.0
        loss_outlier_accum = 0.0

        micro_bs = cfg.batch_size // cfg.grad_accum
        for micro in range(cfg.grad_accum):
            s = micro * micro_bs
            e = s + micro_bs
            micro_ids = input_ids[s:e]
            micro_mask = attention_mask[s:e]
            micro_labels = labels[s:e]

            # Clear storages
            for i in range(model_config.n_layer):
                attn_storage[i] = None
                hidden_storage[i] = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, ce_loss = model(
                    micro_ids, attention_mask=micro_mask, labels=micro_labels
                )

                # Compound loss
                bos_loss = compute_bos_penalty(attn_storage)
                outlier_loss = compute_outlier_penalty(
                    hidden_storage, cfg.outlier_threshold
                )

                total_loss = (
                    ce_loss
                    + cfg.lambda_bos * bos_loss
                    + cfg.lambda_outlier * outlier_loss
                ) / cfg.grad_accum

            total_loss.backward()

            loss_ce_accum += ce_loss.item() / cfg.grad_accum
            loss_bos_accum += bos_loss.item() / cfg.grad_accum
            loss_outlier_accum += outlier_loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        # --- Logging ---
        if step % cfg.log_every == 0 or step == cfg.total_steps:
            model.eval()
            val = evaluate(model, val_loader, device, torch.bfloat16)
            elapsed = time.perf_counter() - started
            current_lr = scheduler.get_last_lr()[0]
            temps = [layer_temps[i].item() for i in range(model_config.n_layer)]
            mean_temp = sum(temps) / len(temps)

            print(
                f"Step {step:4d} | CE: {loss_ce_accum:.4f} | "
                f"BOS: {loss_bos_accum:.4f} | Outlier: {loss_outlier_accum:.4f} | "
                f"Val: {val['loss']:.4f} | LR: {current_lr:.2e} | "
                f"Temp: {mean_temp:.3f} | {elapsed:.0f}s"
            )

            entry = {
                "step": step,
                "train_ce": loss_ce_accum,
                "train_bos": loss_bos_accum,
                "train_outlier": loss_outlier_accum,
                "val_loss": val["loss"],
                "lr": current_lr,
                "temps": temps,
                "elapsed_seconds": elapsed,
            }
            history.append(entry)
            model.train()

        # --- Diagnostic evaluation ---
        if step % cfg.diag_every == 0 or step == cfg.total_steps:
            model.eval()
            temps_for_diag = [layer_temps[i].item() for i in range(model_config.n_layer)]
            diag = run_diagnostic(
                model,
                model_config,
                device,
                attn_type="sba",
                tokenizer=tokenizer,
                val_loader=val_loader,
                scale=temps_for_diag,
                label=f"LoRA-v2 step {step}",
            )
            print_diagnostic_summary(diag)
            save_diagnostic(
                diag, output_dir / f"diagnostic_step{step:04d}.json"
            )

            # Early success check
            if diag["gates"]["all_pass"]:
                print(f"\n✓ ALL GATES PASSED at step {step}!")

            model.train()

    # --- Final results ---
    elapsed = time.perf_counter() - started
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"{'='*60}")

    # Print learned temperatures
    print("\nLearned per-layer temperatures:")
    for i in range(model_config.n_layer):
        print(f"  Layer {i:2d}: {layer_temps[i].item():.4f}")

    # --- Save results ---
    results = {
        "format": "sba-lora-conversion-v2",
        "experiment": "anti-sink-regularized-lora",
        "finished_at": utc_now(),
        "config": {
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "total_steps": cfg.total_steps,
            "warmup_steps": cfg.warmup_steps,
            "lr": cfg.lr,
            "lambda_bos": cfg.lambda_bos,
            "lambda_outlier": cfg.lambda_outlier,
            "outlier_threshold": cfg.outlier_threshold,
            "init_temp": cfg.init_temp,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
        },
        "history": history,
        "final_temps": [layer_temps[i].item() for i in range(model_config.n_layer)],
    }
    write_json(output_dir / "results.json", results)

    # --- Merge LoRA and save clean checkpoint ---
    print("\nMerging LoRA weights...")
    clean_state = merge_lora_weights(model)
    ckpt = {
        "state_dict": clean_state,
        "scratch_config": checkpoint["scratch_config"],
        "attention": "sba",
        "sba_impl": "eager",
        "step": cfg.total_steps,
        "val_loss": history[-1]["val_loss"],
        "learned_temps": [layer_temps[i].item() for i in range(model_config.n_layer)],
        "conversion_config": {
            "lambda_bos": cfg.lambda_bos,
            "lambda_outlier": cfg.lambda_outlier,
            "outlier_threshold": cfg.outlier_threshold,
        },
        "lora_history": history,
    }
    ckpt_name = (
        f"sba_lora_v2_merged_step{cfg.total_steps:06d}"
        f"_{model_config.n_layer}l_{model_config.n_head}h_{model_config.n_embd}d.pt"
    )
    torch.save(ckpt, output_dir / ckpt_name)
    print(f"Merged checkpoint saved: {output_dir / ckpt_name}")

    # --- Final diagnostic ---
    print("\n--- FINAL DIAGNOSTIC ---")
    model.eval()
    final_temps = [layer_temps[i].item() for i in range(model_config.n_layer)]
    final_diag = run_diagnostic(
        model,
        model_config,
        device,
        attn_type="sba",
        tokenizer=tokenizer,
        val_loader=val_loader,
        scale=final_temps,
        label="LoRA-v2 FINAL",
    )
    print_diagnostic_summary(final_diag)
    save_diagnostic(final_diag, output_dir / "diagnostic_final.json")

    if final_diag["gates"]["all_pass"]:
        print("\n🎉 EXPERIMENT 1 SUCCESS — All gates passed!")
    else:
        failed = [k for k, v in final_diag["gates"].items() if not v and k != "all_pass"]
        print(f"\n⚠ EXPERIMENT 1 — Failed gates: {', '.join(failed)}")
        print("Consider: adjusting λ_bos/λ_outlier, more steps, or Experiment 2.")


if __name__ == "__main__":
    main()
