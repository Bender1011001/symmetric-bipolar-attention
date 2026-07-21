# run_homotopy_conversion.py
"""
Equation Homotopy Conversion for Symmetric Bipolar Attention (SBA) (v4)

Performs full-parameter pre-training (unfreezing MLPs, Norms, and Attention)
to transition a pre-trained Softmax checkpoint to centered-SBA routing.

Mathematically bridges Softmax and SBA:
  Y_out = (1 - alpha) * Y_softmax + alpha * Y_sba
where alpha is hard-scheduled 0.0 -> 1.0 over the training run.
"""
import math
import sys
import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_scratch_tokenized_compare import (
    ScratchConfig,
    ScratchLM,
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

MIN_TAU = 0.03
SOFTMAX_BASELINE_LOSS = 1.0005


@dataclass
class HomotopyConfig:
    """Configuration for equation homotopy conversion."""
    total_steps: int = 4000
    warmup_steps: int = 200
    lr: float = 2e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_size: int = 8
    grad_accum: int = 4
    log_every: int = 100
    diag_every: int = 500
    save_checkpoints: bool = True
    checkpoint_every: int = 1000

    # Temperature and bias initialization
    init_tau: float = 1.0
    init_bias: float = 4.0
    bias_target: float = 1.0

    # Distillation teachers
    kl_weight_start: float = 1.0
    kl_weight_final: float = 0.1
    kl_temperature: float = 2.0

    # Structural regularizers
    lambda_rbos: float = 0.2
    lambda_row_l1: float = 0.1
    lambda_bias_anneal: float = 0.1

    # Row L1 schedule
    row_l1_target_start: float = 1.05
    row_l1_target_final: float = 0.65

    # r_BOS row ceiling schedule
    rbos_multiplier: float = 2.0
    max_rbos_target: float = 0.25


def make_lr_schedule(cfg: HomotopyConfig):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return max(cfg.min_lr / cfg.lr, cosine_decay)
    return lr_lambda


def masked_next_token_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    valid = labels[:, 1:].ne(-100)
    
    # VRAM FIX: Flatten to just valid tokens BEFORE casting to float32
    s_valid = student_logits[:, :-1, :][valid].float() / temperature
    t_valid = teacher_logits[:, :-1, :][valid].float() / temperature
    
    s_logp = F.log_softmax(s_valid, dim=-1)
    t_prob = F.softmax(t_valid, dim=-1)
    
    # PyTorch batchmean cleanly handles the flattened 2D tensor over N_valid
    kl = F.kl_div(s_logp, t_prob, reduction="batchmean") * (temperature ** 2)
    return kl


# Hook/global variables to store attention metrics during training forward pass
r_bos_storage = []
row_l1_storage = []


def install_homotopy_forward(
    model: ScratchLM,
    raw_tau_params: nn.ParameterList,
) -> None:
    global r_bos_storage, row_l1_storage
    n_layers = len(model.blocks)
    r_bos_storage = [None] * n_layers
    row_l1_storage = [None] * n_layers

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
            if attention_mask is not None:
                combined_mask = causal & attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
            else:
                combined_mask = causal

            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self_attn.head_dim)

            # --- SBA attention output ---
            tau = MIN_TAU + F.softplus(raw_tau_params[layer_idx])
            current_bias = getattr(model, "current_bias", 4.0)
            weights_sba, r_bos, row_l1 = sba_weights_centered(
                logits, combined_mask, tau, torch.tensor(current_bias, device=logits.device)
            )

            # Save in global lists for loss calculation
            r_bos_storage[layer_idx] = r_bos
            row_l1_storage[layer_idx] = row_l1

            y = torch.matmul(weights_sba, v)
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


def _install_inference_forward(
    model: ScratchLM,
    tau_values: list[torch.Tensor],
    bias_values: list[torch.Tensor],
) -> None:
    """Install centered SBA forward for inference with KV cache support."""

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

            # Concatenate past KV cache
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

            kv_len = k.shape[-2]
            if seq_len == 1:
                causal = torch.ones((1, 1, 1, kv_len), dtype=torch.bool, device=self_attn.causal_mask.device)
            else:
                causal = self_attn.causal_mask[:, :, :seq_len, :kv_len]
            if attention_mask is not None:
                attn_mask_bool = attention_mask.to(
                    dtype=torch.bool
                ).view(batch, 1, 1, kv_len)
                combined_mask = causal & attn_mask_bool
            else:
                combined_mask = causal

            logits = (
                torch.matmul(q, k.transpose(-1, -2))
                / math.sqrt(self_attn.head_dim)
            )

            tau = tau_values[layer_idx].to(device=logits.device)
            bias = bias_values[layer_idx].to(device=logits.device)

            weights, _, _ = sba_weights_centered(
                logits, combined_mask, tau, bias
            )

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


def compute_structural_loss(
    attention_mask: torch.Tensor,
    row_l1_tgt: float,
    rbos_multiplier: float = 2.0,
    max_rbos_target: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    global r_bos_storage, row_l1_storage

    first = next((x for x in r_bos_storage if x is not None), None)
    if first is None:
        device = attention_mask.device
        return torch.zeros((), device=device), torch.zeros((), device=device)

    device = first.device
    qmask = attention_mask.to(device=device, dtype=torch.bool).clone()
    qmask[:, 0] = False  # Skip BOS query position

    visible_counts = attention_mask.to(device=device, dtype=torch.float32).cumsum(dim=-1).clamp_min(1.0)
    rbos_ceiling = (rbos_multiplier / visible_counts).clamp(max=max_rbos_target)

    rbos_total = torch.zeros((), device=device)
    l1_total = torch.zeros((), device=device)
    count = 0

    for r_bos, row_l1 in zip(r_bos_storage, row_l1_storage):
        if r_bos is None or row_l1 is None:
            continue
        valid = qmask[:, None, :].expand_as(r_bos)
        n_valid = valid.sum().clamp_min(1.0).float()

        rbos_excess = F.relu(r_bos.float() - rbos_ceiling[:, None, :])
        rbos_total = rbos_total + (rbos_excess * valid.float()).sum() / n_valid

        l1_excess = F.relu(row_l1.float() - row_l1_tgt)
        l1_total = l1_total + (l1_excess * valid.float()).sum() / n_valid
        count += 1

    return rbos_total / max(count, 1), l1_total / max(count, 1)


def save_checkpoint(
    model: ScratchLM,
    raw_tau_params: nn.ParameterList,
    current_bias: float,
    output_dir: Path,
    step: int,
    config: ScratchConfig,
    args: argparse.Namespace,
    metrics: dict,
) -> str:
    path = output_dir / f"homotopy_step{step:06d}_{config.n_layer}l_{config.n_head}h_{config.n_embd}d.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare standard state dict without LoRA wrappers (we did full parameter fine-tuning)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint = {
        "format": "sba-standalone-scratch-lm-v3-tokenized",
        "attention": "sba",
        "sba_impl": "eager",
        "step": step,
        "scratch_config": asdict(config),
        "args": vars(args),
        "metrics": metrics,
        "state_dict": state_dict,
        "raw_tau": {f"layer_{i}": p.detach().cpu().item() for i, p in enumerate(raw_tau_params)},
        "learned_bias": {f"layer_{i}": current_bias for i in range(config.n_layer)},
        "learned_tau": {f"layer_{i}": (MIN_TAU + F.softplus(p)).detach().cpu().item() for i, p in enumerate(raw_tau_params)},
        "min_tau": MIN_TAU,
        "saved_at": utc_now(),
    }
    torch.save(checkpoint, path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Equation Homotopy Softmax-to-SBA Conversion")
    parser.add_argument("--data-files", nargs="+", default=["data/open_reason_code_mix_50k.jsonl"])
    parser.add_argument("--output-dir", default="sba_standalone/results/homotopy_conversion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--diag-every", type=int, default=500)
    parser.add_argument("--save-checkpoints", action="store_true", default=True)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = HomotopyConfig(
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        diag_every=args.diag_every,
        checkpoint_every=args.checkpoint_every,
    )

    project_root = Path(__file__).resolve().parent.parent
    softmax_path = project_root / "sba_standalone/results/blackwell_125m_overnight/softmax/checkpoints/softmax_step015000_12l_12h_768d.pt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not softmax_path.exists():
        print(f"ERROR: Softmax baseline checkpoint not found: {softmax_path}")
        sys.exit(1)

    print(f"Loading pre-trained Softmax checkpoint: {softmax_path}")
    softmax_ckpt = torch.load(softmax_path, map_location="cpu")
    model_config = ScratchConfig(**softmax_ckpt["scratch_config"])

    # Load BPE Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build Dataloaders on curated dataset mix
    max_len = args.max_length if args.max_length is not None else model_config.block_size
    train_loader, val_loader, data_manifest = build_dataloaders(
        tokenizer=tokenizer,
        data_files=args.data_files,
        max_length=max_len,
        train_limit_per_file=45000,
        eval_limit_per_file=5000,
        batch_size=cfg.batch_size,
        seed=args.seed,
    )

    # Initialize frozen teacher model for KL divergence
    print("Initializing frozen Softmax teacher...")
    teacher = ScratchLM(model_config, "softmax")
    teacher.load_state_dict(softmax_ckpt["state_dict"])
    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Initialize student model (fully unfrozen)
    print("Initializing student model (fully unfrozen)...")
    student = ScratchLM(model_config, "sba", sba_impl="eager")
    student.load_state_dict(softmax_ckpt["state_dict"])
    student.to(device)

    # Verify baseline loss
    print("Evaluating baseline Softmax model...")
    baseline_val = evaluate(teacher, val_loader, device, torch.bfloat16)
    print(f"Softmax Baseline Val Loss: {baseline_val['loss']:.4f}")

    # Set up learnable scale (tau) parameters for centered-SBA (bias is hardcoded)
    raw_tau_params = nn.ParameterList(
        [nn.Parameter(torch.tensor(0.0)) for _ in range(model_config.n_layer)]
    ).to(device)

    student.current_bias = cfg.init_bias
    install_homotopy_forward(student, raw_tau_params)

    # Hook up optimizer for full model + tau parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": student.parameters(), "weight_decay": cfg.weight_decay},
            {"params": list(raw_tau_params.parameters()), "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    scheduler = LambdaLR(optimizer, make_lr_schedule(cfg))

    history = []
    started = time.perf_counter()
    train_iter = iter(train_loader)

    print(f"\nStarting Homotopy Conversion Sweep over {cfg.total_steps} steps...")
    
    student.train()
    for step in range(1, cfg.total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Dynamic updates to Homotopy Schedule based on global step
        progress = step / cfg.total_steps
        student.current_bias = cfg.init_bias - progress * (cfg.init_bias - cfg.bias_target)
        current_row_l1_target = cfg.row_l1_target_start - progress * (cfg.row_l1_target_start - cfg.row_l1_target_final)
        current_kl_weight = cfg.kl_weight_start - progress * (cfg.kl_weight_start - cfg.kl_weight_final)

        # Clear global metrics storage
        for idx in range(model_config.n_layer):
            r_bos_storage[idx] = None
            row_l1_storage[idx] = None

        optimizer.zero_grad(set_to_none=True)
        
        loss_ce_accum = 0.0
        loss_kl_accum = 0.0
        loss_rbos_accum = 0.0
        loss_l1_accum = 0.0

        micro_bs = cfg.batch_size // cfg.grad_accum
        for micro in range(cfg.grad_accum):
            s = micro * micro_bs
            e = s + micro_bs
            micro_ids = input_ids[s:e]
            micro_mask = attention_mask[s:e]
            micro_labels = labels[s:e]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                student_logits, ce_loss = student(
                    micro_ids, attention_mask=micro_mask, labels=micro_labels
                )
                with torch.no_grad():
                    teacher_logits, _ = teacher(
                        micro_ids, attention_mask=micro_mask, labels=micro_labels
                    )

                kl_loss = masked_next_token_kl(
                    student_logits, teacher_logits, micro_labels, cfg.kl_temperature
                )
                rbos_loss, l1_loss = compute_structural_loss(
                    micro_mask, current_row_l1_target, cfg.rbos_multiplier, cfg.max_rbos_target
                )

                # Total combined loss (bias is non-negotiable schedule, no bias_loss)
                total_loss = (
                    ce_loss
                    + current_kl_weight * kl_loss
                    + progress * (cfg.lambda_rbos * rbos_loss + cfg.lambda_row_l1 * l1_loss)
                ) / cfg.grad_accum

            total_loss.backward()

            loss_ce_accum += ce_loss.item() / cfg.grad_accum
            loss_kl_accum += kl_loss.item() / cfg.grad_accum
            loss_rbos_accum += rbos_loss.item() / cfg.grad_accum
            loss_l1_accum += l1_loss.item() / cfg.grad_accum

        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Log metrics periodically
        if step == 1 or step % cfg.log_every == 0 or step == cfg.total_steps:
            student.eval()
            with torch.no_grad():
                val = evaluate(student, val_loader, device, torch.bfloat16)
            
            elapsed = time.perf_counter() - started
            current_lr = scheduler.get_last_lr()[0]
            mean_bias = student.current_bias
            mean_tau = sum((MIN_TAU + F.softplus(p)).item() for p in raw_tau_params) / len(raw_tau_params)

            print(
                f"Step {step:4d} | progress: {progress:.2f} | CE: {loss_ce_accum:.4f} | "
                f"KL: {loss_kl_accum:.4f} | rBOS: {loss_rbos_accum:.4f} | L1: {loss_l1_accum:.4f} | "
                f"Val Loss: {val['loss']:.4f} | Bias: {mean_bias:.2f} (tgt: {student.current_bias:.2f}) | "
                f"tau: {mean_tau:.3f} | {elapsed:.0f}s"
            )
            
            entry = {
                "step": step,
                "progress": progress,
                "train_ce": loss_ce_accum,
                "train_kl": loss_kl_accum,
                "train_rbos": loss_rbos_accum,
                "train_row_l1": loss_l1_accum,
                "train_bias": mean_bias,
                "val_loss": val["loss"],
                "val_perplexity": val["perplexity"],
                "lr": current_lr,
                "mean_bias": mean_bias,
                "mean_tau": mean_tau,
                "elapsed_seconds": elapsed,
            }
            history.append(entry)
            student.train()

        # Run diagnostics periodically
        if step % cfg.diag_every == 0 or step == cfg.total_steps:
            student.eval()
            
            # Briefly attach detached tau/bias params for diagnostic framework
            student.tau_params = [
                MIN_TAU + F.softplus(p).detach() for p in raw_tau_params
            ]
            student.bias_params = [
                torch.tensor(student.current_bias, device=device) for _ in range(model_config.n_layer)
            ]

            # Re-install inference forward to bypass homotopy outputs at diagnostic point
            original_forwards = [block.attn.forward for block in student.blocks]
            inference_tau = [MIN_TAU + F.softplus(p).detach() for p in raw_tau_params]
            inference_bias = [torch.tensor(student.current_bias, device=device) for _ in range(model_config.n_layer)]
            _install_inference_forward(student, inference_tau, inference_bias)

            diag = run_diagnostic(
                student,
                model_config,
                device,
                attn_type="sba_centered",
                tokenizer=tokenizer,
                val_loader=val_loader,
                label=f"Homotopy step {step}",
            )
            print_diagnostic_summary(diag)
            save_diagnostic(diag, output_dir / f"diagnostic_step{step:04d}.json")

            # Restore homotopy training forward
            for idx, block in enumerate(student.blocks):
                block.attn.forward = original_forwards[idx]
            
            student.train()

        # Save checkpoints
        if cfg.save_checkpoints and (step % cfg.checkpoint_every == 0 or step == cfg.total_steps):
            checkpoint_metrics = {
                "train_ce": loss_ce_accum,
                "val_loss": history[-1]["val_loss"],
                "progress": progress,
            }
            save_checkpoint(
                student, raw_tau_params, student.current_bias, output_dir, step, model_config, args, checkpoint_metrics
            )

    # --- Closeout ---
    elapsed = time.perf_counter() - started
    print(f"\n{'='*70}")
    print(f"Homotopy conversion sweep completed in {elapsed:.1f}s")
    print(f"Final Val Loss: {history[-1]['val_loss']:.4f}")
    print(f"{'='*70}")

    # Write summary run data
    results = {
        "format": "sba-homotopy-conversion-v5",
        "experiment": "full-parameter-homotopy-distillation-hardcoded-bias",
        "finished_at": utc_now(),
        "config": asdict(cfg),
        "history": history,
        "final_bias": [student.current_bias] * model_config.n_layer,
        "final_tau": [(MIN_TAU + F.softplus(p)).item() for p in raw_tau_params],
    }
    write_json(output_dir / "results.json", results)


if __name__ == "__main__":
    main()
