# run_dual_teacher_conversion.py
"""
Experiment 2: Score-Centered SBA with Random-Probe Attention Distillation

Based on deep-think mathematical analysis of SBA conversion:

Key architectural change:
    t_ij = (s_ij - μ_i) / τ_h + b_h
    w_ij = sinh(t_ij) / Σ_m cosh(t_im)

  where μ_i is causal row mean of raw scores, τ_h and b_h are learnable
  per-head parameters. Score centering restores meaningful zero (softmax
  models never learned this), and τ_h controls the nonlinearity regime.

Key loss change:
    Random-probe attention matching instead of attention output MSE.
    L_probe = ||(A_student - A_teacher) R||_F²
    where R is random, independent of model V, preventing the student
    from exploiting V null space to hide sinky attention.

Key metric change:
    r_BOS = cosh(s_BOS) / Σ cosh(s_j) — the absolute-score denominator
    share, which is the correct sink metric for SBA (not signed BOS weight).

Two-stage training:
    Stage 1: Random-probe attention alignment + routing stats matching
    Stage 2: CE loss + decaying alignment + r_BOS regularization
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
# Config
# ---------------------------------------------------------------------------
@dataclass
class DualTeacherConfig:
    lora_rank: int = 16
    lora_alpha: float = 32.0
    # Stage 1: attention alignment
    stage1_steps: int = 1500
    stage1_lr: float = 5e-4
    # Stage 2: quality + alignment
    stage2_steps: int = 2000
    stage2_lr: float = 3e-4
    alpha_start: float = 1.0
    alpha_end: float = 0.1
    # Probe matching
    n_probes: int = 32
    lambda_probe: float = 1.0
    lambda_stats: float = 0.5
    lambda_rbos: float = 0.3
    lambda_bias_reg: float = 0.01  # prevent b_h from going large positive
    # Shared
    warmup_steps: int = 50
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 16
    grad_accum: int = 4
    log_every: int = 100
    diag_every: int = 500
    # Temperature init
    init_tau: float = 1.0
    init_bias: float = 0.0

    @property
    def total_steps(self) -> int:
        return self.stage1_steps + self.stage2_steps


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
# Score-Centered SBA Forward with Capture
# ---------------------------------------------------------------------------
def install_centered_sba_forward(
    model: ScratchLM,
    tau_params: nn.Parameter,
    bias_params: nn.Parameter,
    attn_weight_storage: list,
    r_bos_storage: list,
    row_l1_storage: list,
):
    """Install score-centered SBA forward on all attention blocks.

    Captures attention weight matrices (for probe matching), r_BOS
    (for sink metric), and row L1 norms.
    """

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


# ---------------------------------------------------------------------------
# Capture teacher attention weights (standard SBA, no centering)
# ---------------------------------------------------------------------------
def install_sba_capture_forward(model: ScratchLM, attn_storage: list):
    """Install SBA forward on teacher that captures attention weight matrices."""

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
# Random-Probe Attention Matching
# ---------------------------------------------------------------------------
def compute_probe_loss(
    student_attn: list[torch.Tensor],
    teacher_attn: list[torch.Tensor],
    n_probes: int = 32,
) -> torch.Tensor:
    """Random-probe attention matrix matching.

    ||(A_student - A_teacher) R||_F^2

    R is random, independent of model V, preventing the student from
    exploiting V null space to hide sinky attention.
    """
    total = torch.tensor(0.0, device=student_attn[0].device)
    count = 0

    for s_attn, t_attn in zip(student_attn, teacher_attn):
        if s_attn is None or t_attn is None:
            continue
        # s_attn, t_attn: [B, H, T, T]
        T = s_attn.shape[-1]
        # Random probes: [T, n_probes], fresh each call
        R = torch.randn(T, n_probes, device=s_attn.device, dtype=s_attn.dtype)
        R = R / math.sqrt(n_probes)  # normalize

        diff = s_attn.float() - t_attn.float()  # [B, H, T, T]
        projected = torch.matmul(diff, R)  # [B, H, T, n_probes]
        total = total + (projected ** 2).mean()
        count += 1

    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Routing Statistics Loss
# ---------------------------------------------------------------------------
def compute_stats_loss(
    student_r_bos: list[torch.Tensor],
    student_row_l1: list[torch.Tensor],
    target_r_bos_mean: float = 0.08,  # ~1/12 = uniform share for 12-token avg
    target_row_l1_mean: float = 0.5,  # SBA native is well below 1.0
) -> torch.Tensor:
    """Match routing statistics to SBA-native targets.

    Penalizes:
    - r_BOS exceeding target (absolute-score sink)
    - row L1 approaching 1.0 (softmax boundary)
    """
    total = torch.tensor(0.0, device=student_r_bos[0].device)
    count = 0

    for r_bos, row_l1 in zip(student_r_bos, student_row_l1):
        if r_bos is None or row_l1 is None:
            continue
        # Penalize r_BOS above target (only non-BOS queries, position >= 1)
        r_bos_excess = F.relu(r_bos[:, :, 1:] - target_r_bos_mean)
        total = total + r_bos_excess.mean()

        # Penalize row L1 approaching 1.0
        l1_excess = F.relu(row_l1 - target_row_l1_mean)
        total = total + l1_excess.mean() * 0.5
        count += 1

    return total / max(count, 1)


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

    cfg = DualTeacherConfig()

    # --- Paths ---
    softmax_ckpt_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/softmax"
        "/checkpoints/softmax_step015000_12l_12h_768d.pt"
    )
    sba_ckpt_path = Path(
        "/root/sba_standalone/results/blackwell_125m_overnight/sba"
        "/checkpoints/sba_step015000_12l_12h_768d.pt"
    )
    output_dir = Path("/root/sba_standalone/results/dual_teacher_conversion")
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

    # Evaluate Step 0
    print("Evaluating Step 0 (direct swap, no LoRA)...")
    student.eval()
    step0_loss = evaluate(student, val_loader, device, torch.bfloat16)["loss"]
    print(f"Step 0 Val Loss: {step0_loss:.4f}")

    # Apply LoRA
    print(f"Applying LoRA (rank={cfg.lora_rank}) to QKV and proj...")
    for param in student.parameters():
        param.requires_grad = False
    for param in tau_params.parameters():
        param.requires_grad = True
    for param in bias_params.parameters():
        param.requires_grad = True
    for block in student.blocks:
        block.attn.qkv = LoRALinear(
            block.attn.qkv, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
        block.attn.proj = LoRALinear(
            block.attn.proj, rank=cfg.lora_rank, alpha=cfg.lora_alpha
        )
    student.to(device)

    lora_params = [p for p in student.parameters() if p.requires_grad]
    sba_head_params = list(tau_params.parameters()) + list(bias_params.parameters())
    # Exclude sba_head_params from lora_params to avoid duplicate exposure in optimizer
    sba_head_set = {id(p) for p in sba_head_params}
    lora_params = [p for p in lora_params if id(p) not in sba_head_set]
    all_trainable = lora_params + sba_head_params
    n_trainable = sum(p.numel() for p in all_trainable)
    n_total = sum(p.numel() for p in student.parameters())
    print(f"Trainable: {n_trainable:,} ({100*n_trainable/n_total:.2f}%)")
    print(f"  LoRA: {sum(p.numel() for p in lora_params):,}")
    print(f"  τ params: {sum(p.numel() for p in tau_params.parameters()):,}")
    print(f"  b params: {sum(p.numel() for p in bias_params.parameters()):,}")
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"VRAM: {vram_mb:.0f} MB")

    # --- Training ---
    history = [{"step": 0, "stage": 0, "val_loss": step0_loss, "elapsed": 0.0}]
    started = time.perf_counter()
    global_step = 0
    train_iter = iter(train_loader)

    def clear_storages():
        for i in range(n_layer):
            student_attn_storage[i] = None
            student_r_bos_storage[i] = None
            student_row_l1_storage[i] = None
            teacher_attn_storage[i] = None

    # ===================================================================
    # STAGE 1: Random-Probe Attention Alignment + Routing Stats
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 1: Attention Alignment ({cfg.stage1_steps} steps)")
    print(f"Loss = λ_probe * probe_match + λ_stats * routing_stats + λ_bias * |b|²")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        all_trainable, lr=cfg.stage1_lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(
        optimizer, make_lr_schedule(cfg.stage1_lr, cfg.min_lr, cfg.warmup_steps, cfg.stage1_steps)
    )

    student.train()
    for s1_step in range(1, cfg.stage1_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()

        probe_accum = 0.0
        stats_accum = 0.0

        micro_bs = cfg.batch_size // cfg.grad_accum
        for micro in range(cfg.grad_accum):
            s = micro * micro_bs
            e = s + micro_bs
            micro_ids = input_ids[s:e]
            clear_storages()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Teacher forward (frozen)
                with torch.no_grad():
                    sba_teacher(micro_ids)

                # Student forward
                student(micro_ids)

                # Random-probe attention matching
                probe_loss = compute_probe_loss(
                    student_attn_storage, teacher_attn_storage, cfg.n_probes
                )

                # Routing statistics loss
                stats_loss = compute_stats_loss(
                    student_r_bos_storage, student_row_l1_storage
                )

                # Bias regularization: prevent b_h from going large positive
                bias_reg = sum(b.pow(2).mean() for b in bias_params) / n_layer

                total = (
                    cfg.lambda_probe * probe_loss
                    + cfg.lambda_stats * stats_loss
                    + cfg.lambda_bias_reg * bias_reg
                ) / cfg.grad_accum

            total.backward()
            probe_accum += probe_loss.item() / cfg.grad_accum
            stats_accum += stats_loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if s1_step % cfg.log_every == 0 or s1_step == cfg.stage1_steps:
            student.eval()
            val = evaluate(student, val_loader, device, torch.bfloat16)
            elapsed = time.perf_counter() - started
            lr = scheduler.get_last_lr()[0]
            mean_tau = sum(tau_params[i].mean().item() for i in range(n_layer)) / n_layer
            mean_bias = sum(bias_params[i].mean().item() for i in range(n_layer)) / n_layer
            print(
                f"S1 {s1_step:4d}/{cfg.stage1_steps} | "
                f"Probe: {probe_accum:.6f} | Stats: {stats_accum:.4f} | "
                f"Val: {val['loss']:.4f} | τ̄: {mean_tau:.3f} | b̄: {mean_bias:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.0f}s"
            )
            history.append({
                "step": global_step, "stage": 1, "stage_step": s1_step,
                "probe_loss": probe_accum, "stats_loss": stats_accum,
                "val_loss": val["loss"], "mean_tau": mean_tau,
                "mean_bias": mean_bias, "lr": lr, "elapsed": elapsed,
            })
            student.train()

        if s1_step % cfg.diag_every == 0:
            student.eval()
            diag = run_diagnostic(
                student, model_config, device, "sba_centered", tokenizer,
                val_loader=val_loader, label=f"S1 step {s1_step}",
            )
            print_diagnostic_summary(diag)
            save_diagnostic(diag, output_dir / f"diagnostic_s1_{s1_step:04d}.json")
            student.train()

    # ===================================================================
    # STAGE 2: Quality + Decaying Alignment + r_BOS Regularization
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Quality Fine-tuning ({cfg.stage2_steps} steps)")
    print(f"Loss = CE + α(t)×probe_match + λ_stats×stats + λ_rbos×r_BOS_penalty")
    print(f"α decays: {cfg.alpha_start} → {cfg.alpha_end}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(
        all_trainable, lr=cfg.stage2_lr, weight_decay=cfg.weight_decay
    )
    scheduler = LambdaLR(
        optimizer, make_lr_schedule(cfg.stage2_lr, cfg.min_lr, cfg.warmup_steps, cfg.stage2_steps)
    )

    student.train()
    for s2_step in range(1, cfg.stage2_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        decay_progress = s2_step / cfg.stage2_steps
        alpha = cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * decay_progress

        optimizer.zero_grad()
        ce_accum = 0.0
        probe_accum = 0.0
        stats_accum = 0.0
        rbos_accum = 0.0

        micro_bs = cfg.batch_size // cfg.grad_accum
        for micro in range(cfg.grad_accum):
            sl = micro * micro_bs
            el = sl + micro_bs
            micro_ids = input_ids[sl:el]
            micro_mask = attention_mask[sl:el]
            micro_labels = labels[sl:el]
            clear_storages()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    sba_teacher(micro_ids)

                _, ce_loss = student(
                    micro_ids, attention_mask=micro_mask, labels=micro_labels
                )

                probe_loss = compute_probe_loss(
                    student_attn_storage, teacher_attn_storage, cfg.n_probes
                )
                stats_loss = compute_stats_loss(
                    student_r_bos_storage, student_row_l1_storage
                )

                # Direct r_BOS penalty
                rbos_penalty = torch.tensor(0.0, device=device)
                for r_bos in student_r_bos_storage:
                    if r_bos is not None:
                        rbos_penalty = rbos_penalty + F.relu(
                            r_bos[:, :, 1:] - 0.1
                        ).mean()
                rbos_penalty = rbos_penalty / n_layer

                bias_reg = sum(b.pow(2).mean() for b in bias_params) / n_layer

                total = (
                    ce_loss
                    + alpha * cfg.lambda_probe * probe_loss
                    + cfg.lambda_stats * stats_loss
                    + cfg.lambda_rbos * rbos_penalty
                    + cfg.lambda_bias_reg * bias_reg
                ) / cfg.grad_accum

            total.backward()
            ce_accum += ce_loss.item() / cfg.grad_accum
            probe_accum += probe_loss.item() / cfg.grad_accum
            stats_accum += stats_loss.item() / cfg.grad_accum
            rbos_accum += rbos_penalty.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(all_trainable, cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if s2_step % cfg.log_every == 0 or s2_step == cfg.stage2_steps:
            student.eval()
            val = evaluate(student, val_loader, device, torch.bfloat16)
            elapsed = time.perf_counter() - started
            lr = scheduler.get_last_lr()[0]
            mean_tau = sum(tau_params[i].mean().item() for i in range(n_layer)) / n_layer
            mean_bias = sum(bias_params[i].mean().item() for i in range(n_layer)) / n_layer
            print(
                f"S2 {s2_step:4d}/{cfg.stage2_steps} | "
                f"CE: {ce_accum:.4f} | Probe: {probe_accum:.6f} | "
                f"rBOS: {rbos_accum:.4f} | α: {alpha:.3f} | "
                f"Val: {val['loss']:.4f} | τ̄: {mean_tau:.3f} | b̄: {mean_bias:.4f} | "
                f"LR: {lr:.2e} | {elapsed:.0f}s"
            )
            history.append({
                "step": global_step, "stage": 2, "stage_step": s2_step,
                "ce_loss": ce_accum, "probe_loss": probe_accum,
                "stats_loss": stats_accum, "rbos_penalty": rbos_accum,
                "alpha": alpha, "val_loss": val["loss"],
                "mean_tau": mean_tau, "mean_bias": mean_bias,
                "lr": lr, "elapsed": elapsed,
            })
            student.train()

        if s2_step % cfg.diag_every == 0 or s2_step == cfg.stage2_steps:
            student.eval()
            diag = run_diagnostic(
                student, model_config, device, "sba_centered", tokenizer,
                val_loader=val_loader, label=f"S2 step {s2_step}",
            )
            print_diagnostic_summary(diag)
            save_diagnostic(diag, output_dir / f"diagnostic_s2_{s2_step:04d}.json")
            if diag["gates"]["all_pass"]:
                print(f"\n✓ ALL GATES PASSED at global step {global_step}!")
            student.train()

    # --- Final ---
    elapsed = time.perf_counter() - started
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed:.1f}s")
    print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"{'='*60}")

    # Print learned τ and b per layer
    print("\nLearned per-head parameters:")
    for i in range(n_layer):
        tau_vals = tau_params[i].detach().cpu().tolist()
        bias_vals = bias_params[i].detach().cpu().tolist()
        print(f"  Layer {i:2d}: τ=[{', '.join(f'{t:.3f}' for t in tau_vals)}]")
        print(f"            b=[{', '.join(f'{b:.4f}' for b in bias_vals)}]")

    # Save results
    results = {
        "format": "sba-dual-teacher-v2",
        "experiment": "score-centered-sba-probe-distillation",
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

    # Merge and save
    print("\nMerging LoRA weights...")
    clean_state = merge_lora_weights(student)
    ckpt = {
        "state_dict": clean_state,
        "scratch_config": softmax_ckpt["scratch_config"],
        "attention": "sba_centered",
        "sba_impl": "eager",
        "step": cfg.total_steps,
        "val_loss": history[-1]["val_loss"],
        "conversion_method": "score-centered-probe-distillation",
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
        f"sba_centered_dual_step{cfg.total_steps:06d}"
        f"_{n_layer}l_{n_head}h_{model_config.n_embd}d.pt"
    )
    torch.save(ckpt, output_dir / ckpt_name)
    print(f"Merged checkpoint saved: {output_dir / ckpt_name}")

    # Final diagnostic
    print("\n--- FINAL DIAGNOSTIC ---")
    student.eval()
    final_diag = run_diagnostic(
        student, model_config, device, "sba_centered", tokenizer,
        val_loader=val_loader, label="Dual-Teacher FINAL",
    )
    print_diagnostic_summary(final_diag)
    save_diagnostic(final_diag, output_dir / "diagnostic_final.json")

    if final_diag["gates"]["all_pass"]:
        print("\n🎉 EXPERIMENT 2 SUCCESS — All gates passed!")
    else:
        failed = [k for k, v in final_diag["gates"].items() if not v and k != "all_pass"]
        print(f"\n⚠ EXPERIMENT 2 — Failed gates: {', '.join(failed)}")


if __name__ == "__main__":
    main()
