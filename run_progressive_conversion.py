# run_progressive_conversion.py
"""
Experiment 3: Progressive Layer-by-Layer SBA Conversion

Converts a pre-trained softmax model to score-centered SBA layer-by-layer
(bottom-to-top) using local MSE alignment, random-probe attention matching,
and routing statistics matching, followed by a global fine-tuning stage.

Teacher forcing isolates each layer, preventing downstream error propagation
during training, and forces the adapter to correct for earlier representation drift.
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
    sba_weights_centered,
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

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scale


# ---------------------------------------------------------------------------
# Progressive Conversion Config
# ---------------------------------------------------------------------------
@dataclass
class ProgressiveConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    # Steps per layer
    steps_per_layer: int = 500
    layer_lr: float = 5e-4
    # Global fine-tuning
    global_steps: int = 1000
    global_lr: float = 2e-4
    # Loss weights
    lambda_mse: float = 1.0
    lambda_probe: float = 0.5
    lambda_stats: float = 0.3
    lambda_rbos: float = 0.3
    lambda_bias_reg: float = 0.01
    n_probes: int = 32
    # Shared settings
    warmup_steps: int = 50
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 16
    grad_accum: int = 4
    log_every: int = 100
    init_tau: float = 1.0
    init_bias: float = 0.0


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------
def make_lr_schedule(lr, min_lr, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Forwards Hooks for Capture
# ---------------------------------------------------------------------------
def install_centered_sba_forward(
    model: ScratchLM,
    tau_params: nn.ParameterList,
    bias_params: nn.ParameterList,
    attn_weight_storage: list,
    r_bos_storage: list,
    row_l1_storage: list,
):
    def _make_forward(layer_idx):
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
            if attention_mask is not None:
                attn_mask_bool = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
                combined_mask = causal & attn_mask_bool
            else:
                combined_mask = causal

            logits = (
                torch.matmul(q, k.transpose(-1, -2))
                / math.sqrt(self_attn.head_dim)
            )

            # Score-centered SBA
            weights, r_bos, row_l1 = sba_weights_centered(
                logits, combined_mask,
                tau_params[layer_idx],
                bias_params[layer_idx],
            )

            # Store for loss computation
            attn_weight_storage[layer_idx] = weights
            r_bos_storage[layer_idx] = r_bos
            row_l1_storage[layer_idx] = row_l1

            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))

            if use_cache:
                return out, (k, v)
            return out

        return _forward

    for i, block in enumerate(model.blocks):
        block.attn.forward = _make_forward(i).__get__(
            block.attn, type(block.attn)
        )


def install_sba_capture_forward(model: ScratchLM, attn_storage: list):
    def _make_forward(layer_idx):
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
            if attention_mask is not None:
                attn_mask_bool = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
                combined_mask = causal & attn_mask_bool
            else:
                combined_mask = causal

            logits = (
                torch.matmul(q, k.transpose(-1, -2))
                / math.sqrt(self_attn.head_dim)
            )

            weights = sba_weights(logits, combined_mask)
            attn_storage[layer_idx] = weights.detach()

            y = torch.matmul(weights, v)
            y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
            out = self_attn.dropout(self_attn.proj(y))

            if use_cache:
                return out, (k, v)
            return out

        return _forward

    for i, block in enumerate(model.blocks):
        block.attn.forward = _make_forward(i).__get__(
            block.attn, type(block.attn)
        )


# ---------------------------------------------------------------------------
# Merge LoRA
# ---------------------------------------------------------------------------
def merge_lora_weights(model: ScratchLM) -> dict:
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

    cfg = ProgressiveConfig()

    # --- Paths ---
    softmax_ckpt_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/softmax"
        "/checkpoints/softmax_step015000_12l_12h_768d.pt"
    )
    sba_ckpt_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/sba"
        "/checkpoints/sba_step015000_12l_12h_768d.pt"
    )
    output_dir = Path("/root/sba_standalone/results/progressive_conversion")
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [(softmax_ckpt_path, "Softmax"), (sba_ckpt_path, "SBA")]:
        if not p.exists():
            print(f"ERROR: {label} checkpoint not found: {p}")
            sys.exit(1)

    # --- Load checkpoints ---
    print(f"Loading softmax checkpoint: {softmax_ckpt_path}")
    softmax_ckpt = torch.load(softmax_ckpt_path, map_location="cpu")
    model_config = ScratchConfig(**softmax_ckpt["scratch_config"])

    print(f"Loading SBA teacher checkpoint: {sba_ckpt_path}")
    sba_ckpt = torch.load(sba_ckpt_path, map_location="cpu")

    # --- Tokenizer & data ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # --- Softmax Teacher (original, frozen) ---
    print("Loading Softmax original teacher model (frozen)...")
    softmax_teacher = ScratchLM(model_config, "softmax")
    softmax_teacher.load_state_dict(softmax_ckpt["state_dict"])
    softmax_teacher.to(device)
    softmax_teacher.eval()
    for p in softmax_teacher.parameters():
        p.requires_grad = False

    # --- SBA Teacher (frozen, captures attention weights) ---
    print("Loading SBA teacher model (frozen)...")
    sba_teacher = ScratchLM(model_config, "sba", sba_impl="sdpa")
    sba_teacher.load_state_dict(sba_ckpt["state_dict"])
    sba_teacher.to(device)
    sba_teacher.eval()
    for p in sba_teacher.parameters():
        p.requires_grad = False

    teacher_attn_storage = [None] * model_config.n_layer
    install_sba_capture_forward(sba_teacher, teacher_attn_storage)

    # --- Student (softmax weights + score-centered SBA + LoRA) ---
    print("Initializing student model (softmax weights → score-centered SBA)...")
    student = ScratchLM(model_config, "sba", sba_impl="sdpa")
    student.load_state_dict(softmax_ckpt["state_dict"])
    student.to(device)

    # Per-head learnable τ and b (n_layer × n_head)
    n_head = model_config.n_head
    n_layer = model_config.n_layer
    tau_params = nn.ParameterList([
        nn.Parameter(torch.full((n_head,), cfg.init_tau))
        for _ in range(n_layer)
    ]).to(device)
    bias_params = nn.ParameterList([
        nn.Parameter(torch.full((n_head,), cfg.init_bias))
        for _ in range(n_layer)
    ]).to(device)

    # Attach to student so diagnostic framework can access them
    student.tau_params = tau_params
    student.bias_params = bias_params

    # Storage for student attention data
    student_attn_storage = [None] * n_layer
    student_r_bos_storage = [None] * n_layer
    student_row_l1_storage = [None] * n_layer

    install_centered_sba_forward(
        student, tau_params, bias_params,
        student_attn_storage, student_r_bos_storage, student_row_l1_storage,
    )

    # Apply LoRA linear adapters everywhere initially, but freeze them
    print(f"Wrapping student attention modules with LoRA (rank={cfg.lora_rank})...")
    for block in student.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
        block.attn.proj = LoRALinear(
            block.attn.proj, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
    student.to(device)

    # Initially freeze all parameters
    for p in student.parameters():
        p.requires_grad = False
    for p in tau_params.parameters():
        p.requires_grad = False
    for p in bias_params.parameters():
        p.requires_grad = False

    history = []
    started = time.perf_counter()
    train_iter = iter(train_loader)

    # ===================================================================
    # STAGE 1: Progressive Layer-by-Layer Alignment
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STAGE 1: Progressive Layer-by-Layer Alignment")
    print(f"Training bottom-to-top: {cfg.steps_per_layer} steps per layer")
    print(f"{'='*70}")

    for layer_idx in range(n_layer):
        print(f"\n--- Converted Layer {layer_idx} / {n_layer - 1} ---")

        # Enable gradients ONLY for the current layer's parameters
        current_block = student.blocks[layer_idx]
        current_block.attn.qkv.lora_A.requires_grad = True
        current_block.attn.qkv.lora_B.requires_grad = True
        current_block.attn.proj.lora_A.requires_grad = True
        current_block.attn.proj.lora_B.requires_grad = True
        tau_params[layer_idx].requires_grad = True
        bias_params[layer_idx].requires_grad = True

        layer_trainable = [
            current_block.attn.qkv.lora_A,
            current_block.attn.qkv.lora_B,
            current_block.attn.proj.lora_A,
            current_block.attn.proj.lora_B,
            tau_params[layer_idx],
            bias_params[layer_idx],
        ]

        optimizer = torch.optim.AdamW(
            layer_trainable, lr=cfg.layer_lr, weight_decay=cfg.weight_decay
        )
        scheduler = LambdaLR(
            optimizer, make_lr_schedule(cfg.layer_lr, cfg.min_lr, cfg.warmup_steps, cfg.steps_per_layer)
        )

        student.train()
        for step in range(1, cfg.steps_per_layer + 1):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()

            mse_accum = 0.0
            probe_accum = 0.0
            stats_accum = 0.0

            micro_bs = cfg.batch_size // cfg.grad_accum
            for micro in range(cfg.grad_accum):
                s = micro * micro_bs
                e = s + micro_bs
                micro_ids = input_ids[s:e]
                micro_mask = attention_mask[s:e]

                # Clear captured attention weight storages
                student_attn_storage[layer_idx] = None
                student_r_bos_storage[layer_idx] = None
                student_row_l1_storage[layer_idx] = None
                teacher_attn_storage[layer_idx] = None

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # 1. Run SBA Teacher to capture targets
                    with torch.no_grad():
                        sba_teacher(micro_ids, attention_mask=micro_mask)

                    # 2. Run Softmax Teacher up to output of block layer_idx (frozen target)
                    with torch.no_grad():
                        pos_t = torch.arange(micro_ids.shape[1], device=device)
                        x_t = softmax_teacher.token_emb(micro_ids) + softmax_teacher.pos_emb(pos_t)[None, :, :]
                        for idx in range(layer_idx + 1):
                            x_t = softmax_teacher.blocks[idx](x_t, attention_mask=micro_mask)

                    # 3. Run Student prefix up to block layer_idx (frozen)
                    with torch.no_grad():
                        pos_s = torch.arange(micro_ids.shape[1], device=device)
                        x_s = student.token_emb(micro_ids) + student.pos_emb(pos_s)[None, :, :]
                        for idx in range(layer_idx):
                            x_s = student.blocks[idx](x_s, attention_mask=micro_mask)

                    # Detach prefix states so gradients only flow through current layer
                    x_s = x_s.detach().requires_grad_(True)

                    # 4. Run current Student layer
                    y_s = student.blocks[layer_idx](x_s, attention_mask=micro_mask)

                    # --- Compute local progressive losses ---
                    # Local output MSE (student matches original teacher forced residual basis)
                    mse_loss = F.mse_loss(y_s, x_t)

                    # Random-probe attention matching
                    s_attn = student_attn_storage[layer_idx]
                    t_attn = teacher_attn_storage[layer_idx]
                    probe_loss = torch.tensor(0.0, device=device)
                    if s_attn is not None and t_attn is not None:
                        T = s_attn.shape[-1]
                        R = torch.randn(T, cfg.n_probes, device=device, dtype=s_attn.dtype) / math.sqrt(cfg.n_probes)
                        diff = s_attn.float() - t_attn.float()
                        probe_loss = (diff @ R).pow(2).mean()

                    # Routing stats matching (suppress sinks and softmax mimics)
                    stats_loss = torch.tensor(0.0, device=device)
                    r_bos = student_r_bos_storage[layer_idx]
                    row_l1 = student_row_l1_storage[layer_idx]
                    if r_bos is not None and row_l1 is not None:
                        r_bos_excess = F.relu(r_bos[:, :, 1:] - 0.08)
                        l1_excess = F.relu(row_l1 - 0.5)
                        stats_loss = r_bos_excess.mean() + l1_excess.mean() * 0.5

                    # Bias regularization
                    bias_reg = bias_params[layer_idx].pow(2).mean()

                    total = (
                        cfg.lambda_mse * mse_loss
                        + cfg.lambda_probe * probe_loss
                        + cfg.lambda_stats * stats_loss
                        + cfg.lambda_bias_reg * bias_reg
                    ) / cfg.grad_accum

                total.backward()
                mse_accum += mse_loss.item() / cfg.grad_accum
                probe_accum += probe_loss.item() / cfg.grad_accum
                stats_accum += stats_loss.item() / cfg.grad_accum

            torch.nn.utils.clip_grad_norm_(layer_trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every == 0 or step == cfg.steps_per_layer:
                elapsed = time.perf_counter() - started
                lr = scheduler.get_last_lr()[0]
                t_val = tau_params[layer_idx].mean().item()
                b_val = bias_params[layer_idx].mean().item()
                print(
                    f"L{layer_idx} | Step {step:3d}/{cfg.steps_per_layer} | "
                    f"MSE: {mse_accum:.6f} | Probe: {probe_accum:.6f} | "
                    f"Stats: {stats_accum:.4f} | τ̄: {t_val:.3f} | b̄: {b_val:.4f} | "
                    f"LR: {lr:.2e} | {elapsed:.0f}s"
                )

        # Freeze the layer's parameters before moving to the next
        current_block.attn.qkv.lora_A.requires_grad = False
        current_block.attn.qkv.lora_B.requires_grad = False
        current_block.attn.proj.lora_A.requires_grad = False
        current_block.attn.proj.lora_B.requires_grad = False
        tau_params[layer_idx].requires_grad = False
        bias_params[layer_idx].requires_grad = False

        # Run diagnostic on the intermediate model (using the centered configuration)
        student.eval()
        diag = run_diagnostic(
            student, model_config, device, "sba_centered", tokenizer,
            val_loader=val_loader, label=f"Post-Layer {layer_idx}",
        )
        print_diagnostic_summary(diag)
        save_diagnostic(diag, output_dir / f"diagnostic_l{layer_idx}.json")

    # ===================================================================
    # STAGE 2: Global CE Fine-tuning
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"STAGE 2: Global Quality Fine-tuning ({cfg.global_steps} steps)")
    print(f"Unfreezing all LoRA and temperature/bias parameters")
    print(f"Loss = CE + λ_rbos * r_BOS_penalty + λ_bias * |b|²")
    print(f"{'='*70}")

    # Unfreeze all adapters
    for block in student.blocks:
        block.attn.qkv.lora_A.requires_grad = True
        block.attn.qkv.lora_B.requires_grad = True
        block.attn.proj.lora_A.requires_grad = True
        block.attn.proj.lora_B.requires_grad = True
    for p in tau_params.parameters():
        p.requires_grad = True
    for p in bias_params.parameters():
        p.requires_grad = True

    all_trainable = [p for p in student.parameters() if p.requires_grad] + \
                    list(tau_params.parameters()) + \
                    list(bias_params.parameters())

    optimizer = torch.optim.AdamW(
        all_trainable, lr=cfg.global_lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(
        optimizer, make_lr_schedule(cfg.global_lr, cfg.min_lr, cfg.warmup_steps, cfg.global_steps)
    )

    student.train()
    for gs_step in range(1, cfg.global_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        ce_accum = 0.0
        rbos_accum = 0.0

        micro_bs = cfg.batch_size // cfg.grad_accum
        for micro in range(cfg.grad_accum):
            sl = micro * micro_bs
            el = sl + micro_bs
            micro_ids = input_ids[sl:el]
            micro_mask = attention_mask[sl:el]
            micro_labels = labels[sl:el]

            # Clear captured attention weight storages
            for i in range(n_layer):
                student_attn_storage[i] = None
                student_r_bos_storage[i] = None
                student_row_l1_storage[i] = None

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Student forward (runs through patched centered SBA layers)
                _, ce_loss = student(
                    micro_ids, attention_mask=micro_mask, labels=micro_labels
                )

                # Direct r_BOS penalty (prevent absolute-score sinks globally)
                rbos_penalty = torch.tensor(0.0, device=device)
                for r_bos in student_r_bos_storage:
                    if r_bos is not None:
                        rbos_penalty = rbos_penalty + F.relu(
                            r_bos[:, :, 1:] - 0.1
                        ).mean()
                rbos_penalty = rbos_penalty / n_layer

                # Bias regularization
                bias_reg = sum(b.pow(2).mean() for b in bias_params) / n_layer

                total = (
                    ce_loss
                    + cfg.lambda_rbos * rbos_penalty
                    + cfg.lambda_bias_reg * bias_reg
                ) / cfg.grad_accum

            total.backward()
            ce_accum += ce_loss.item() / cfg.grad_accum
            rbos_accum += rbos_penalty.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if gs_step % cfg.log_every == 0 or gs_step == cfg.global_steps:
            student.eval()
            val = evaluate(student, val_loader, device, torch.bfloat16)
            elapsed = time.perf_counter() - started
            lr = scheduler.get_last_lr()[0]
            mean_tau = sum(tau_params[i].mean().item() for i in range(n_layer)) / n_layer
            mean_bias = sum(bias_params[i].mean().item() for i in range(n_layer)) / n_layer
            print(
                f"Global {gs_step:4d}/{cfg.global_steps} | "
                f"CE: {ce_accum:.4f} | rBOS: {rbos_accum:.4f} | "
                f"Val: {val['loss']:.4f} | τ̄: {mean_tau:.3f} | b̄: {mean_bias:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.0f}s"
            )
            history.append({
                "global_step": gs_step,
                "val_loss": val["loss"],
                "mean_tau": mean_tau,
                "mean_bias": mean_bias,
                "elapsed": elapsed,
            })
            student.train()

        if gs_step % cfg.log_every == 0:
            student.eval()
            diag = run_diagnostic(
                student, model_config, device, "sba_centered", tokenizer,
                val_loader=val_loader, label=f"Global step {gs_step}",
            )
            print_diagnostic_summary(diag)
            save_diagnostic(diag, output_dir / f"diagnostic_global_{gs_step:04d}.json")
            student.train()

    # --- Save final results ---
    elapsed = time.perf_counter() - started
    print(f"\n{'='*70}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"{'='*70}")

    # Merge LoRA and save clean checkpoint
    print("\nMerging LoRA weights...")
    clean_state = merge_lora_weights(student)
    ckpt = {
        "state_dict": clean_state,
        "scratch_config": softmax_ckpt["scratch_config"],
        "attention": "sba_centered",
        "sba_impl": "eager",
        "step": cfg.global_steps,
        "val_loss": history[-1]["val_loss"],
        "conversion_method": "progressive-layer-wise-distillation",
        "learned_tau": {
            f"layer_{i}": tau_params[i].detach().cpu().tolist()
            for i in range(n_layer)
        },
        "learned_bias": {
            f"layer_{i}": bias_params[i].detach().cpu().tolist()
            for i in range(n_layer)
        },
    }
    ckpt_name = (
        f"sba_centered_progressive_step{cfg.global_steps:06d}"
        f"_{n_layer}l_{n_head}h_{model_config.n_embd}d.pt"
    )
    torch.save(ckpt, output_dir / ckpt_name)
    print(f"Merged checkpoint saved: {output_dir / ckpt_name}")

    # Save training metrics
    results = {
        "format": "sba-progressive-conversion",
        "experiment": "progressive-layer-wise-distillation",
        "finished_at": utc_now(),
        "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        "history": history,
        "learned_tau": {
            f"layer_{i}": tau_params[i].detach().cpu().tolist()
            for i in range(n_layer)
        },
        "learned_bias": {
            f"layer_{i}": bias_params[i].detach().cpu().tolist()
            for i in range(n_layer)
        },
    }
    write_json(output_dir / "results.json", results)

    # Final diagnostic
    print("\n--- FINAL DIAGNOSTIC ---")
    student.eval()
    final_diag = run_diagnostic(
        student, model_config, device, "sba_centered", tokenizer,
        val_loader=val_loader, label="Progressive FINAL",
    )
    print_diagnostic_summary(final_diag)
    save_diagnostic(final_diag, output_dir / "diagnostic_final.json")

    if final_diag["gates"]["all_pass"]:
        print("\n🎉 EXPERIMENT 3 SUCCESS — All gates passed!")
    else:
        failed = [k for k, v in final_diag["gates"].items() if not v and k != "all_pass"]
        print(f"\n⚠ EXPERIMENT 3 — Failed gates: {', '.join(failed)}")


if __name__ == "__main__":
    main()
