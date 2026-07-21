from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import git_revision, utc_now, write_json
from data import build_dataloaders

SBAImpl = Literal["eager", "sdpa"]


@dataclass(frozen=True)
class ScratchConfig:
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    mlp_ratio: int = 4


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float()
        normed = work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * normed).to(dtype=x.dtype)


def sba_weights(logits: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
    work = logits.float()
    if causal_mask is not None:
        valid = causal_mask.to(device=logits.device)
        floor = torch.full_like(work, torch.finfo(work.dtype).min)
        zeros = torch.zeros_like(work)
        pos_logits = torch.where(valid, work, floor)
        neg_logits = torch.where(valid, -work, floor)
        row_max = torch.maximum(
            pos_logits.max(dim=-1, keepdim=True).values,
            neg_logits.max(dim=-1, keepdim=True).values,
        )
        e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
        e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)
        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work.dtype).tiny)
        weights = torch.where(valid, (e_pos - e_neg) / denom, zeros)
    else:
        row_max = torch.maximum(
            work.max(dim=-1, keepdim=True).values,
            (-work).max(dim=-1, keepdim=True).values,
        )
        e_pos = torch.exp(work - row_max)
        e_neg = torch.exp(-work - row_max)
        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work.dtype).tiny)
        weights = (e_pos - e_neg) / denom
    return weights.to(dtype=logits.dtype)


def diff_sba_weights(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    lamb: torch.Tensor,
    causal_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    work1 = logits1.float()
    work2 = logits2.float()
    lamb_val = lamb.float()
    if causal_mask is not None:
        valid = causal_mask.to(device=logits1.device)
        floor = torch.full_like(work1, torch.finfo(work1.dtype).min)
        zeros = torch.zeros_like(work1)
        l1_pos = torch.where(valid, work1, floor)
        l1_neg = torch.where(valid, -work1, floor)
        l2_pos = torch.where(valid, work2, floor)
        l2_neg = torch.where(valid, -work2, floor)
        row_max = torch.maximum(
            torch.maximum(l1_pos.max(dim=-1, keepdim=True).values, l1_neg.max(dim=-1, keepdim=True).values),
            torch.maximum(l2_pos.max(dim=-1, keepdim=True).values, l2_neg.max(dim=-1, keepdim=True).values),
        )
        e1_pos = torch.where(valid, torch.exp(l1_pos - row_max), zeros)
        e1_neg = torch.where(valid, torch.exp(l1_neg - row_max), zeros)
        e2_pos = torch.where(valid, torch.exp(l2_pos - row_max), zeros)
        e2_neg = torch.where(valid, torch.exp(l2_neg - row_max), zeros)
        denom = (e1_pos + e1_neg + e2_pos + e2_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work1.dtype).tiny)
        weights = torch.where(valid, ((e1_pos - e1_neg) - lamb_val * (e2_pos - e2_neg)) / denom, zeros)
    else:
        row_max = torch.maximum(
            torch.maximum(work1.max(dim=-1, keepdim=True).values, (-work1).max(dim=-1, keepdim=True).values),
            torch.maximum(work2.max(dim=-1, keepdim=True).values, (-work2).max(dim=-1, keepdim=True).values),
        )
        e1_pos = torch.exp(work1 - row_max)
        e1_neg = torch.exp(-work1 - row_max)
        e2_pos = torch.exp(work2 - row_max)
        e2_neg = torch.exp(-work2 - row_max)
        denom = (e1_pos + e1_neg + e2_pos + e2_neg).sum(dim=-1, keepdim=True).clamp_min(torch.finfo(work1.dtype).tiny)
        weights = ((e1_pos - e1_neg) - lamb_val * (e2_pos - e2_neg)) / denom
    return weights.to(dtype=logits1.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str, sba_impl: SBAImpl = "sdpa") -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if attention not in {"softmax", "sba", "diff_sba"}:
            raise ValueError("attention must be softmax, sba, or diff_sba")
        if sba_impl not in {"eager", "sdpa"}:
            raise ValueError("sba_impl must be eager or sdpa")
        self.attention = attention
        self.sba_impl = sba_impl
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        if attention == "diff_sba":
            self.q1_k1_q2_k2_v = nn.Linear(config.n_embd, 5 * config.n_embd, bias=False)
            self.raw_lambda = nn.Parameter(torch.full((1, self.n_head, 1, 1), math.log(0.8)))
        else:
            self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, channels = x.shape
        if self.attention == "diff_sba":
            qkv5 = self.q1_k1_q2_k2_v(x).view(batch, seq_len, 5, self.n_head, self.head_dim)
            q1, k1, q2, k2, v = qkv5.unbind(dim=2)
            q1 = q1.transpose(1, 2)
            k1 = k1.transpose(1, 2)
            q2 = q2.transpose(1, 2)
            k2 = k2.transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = q1, k1
        else:
            qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        if past_key_value is not None:
            if self.attention == "diff_sba":
                k_cache = torch.cat([past_key_value[0], torch.cat([k1, k2], dim=-1)], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)
                k1, k2 = k_cache.chunk(2, dim=-1)
                q, k = q1, k1
            else:
                # Store NORMAL k and v in the cache for BOTH Softmax and SBA
                k = torch.cat([past_key_value[0], k], dim=-2)
                v = torch.cat([past_key_value[1], v], dim=-2)

        if use_cache:
            if self.attention == "diff_sba":
                present_key_value = (torch.cat([k1, k2], dim=-1), v)
            else:
                # Return standard k and v to the cache (do NOT double them here)
                present_key_value = (k, v)
        else:
            present_key_value = None

        if seq_len == 1 and past_key_value is not None:
            if attention_mask is not None:
                combined_mask = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, k.shape[-2])
            else:
                combined_mask = None
        else:
            causal = self.causal_mask[:, :, :seq_len, :seq_len]
            if attention_mask is not None:
                attn_mask_bool = attention_mask.to(dtype=torch.bool).view(batch, 1, 1, seq_len)
                combined_mask = causal & attn_mask_bool
            else:
                combined_mask = causal

        if self.attention == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=combined_mask, dropout_p=0.0, is_causal=False)
        elif self.attention == "diff_sba":
            if self.sba_impl == "sdpa":
                # Zero-Overhead, FlashAttention (SDPA) formulation for Diff-SBA
                q = torch.cat([q1, q2], dim=-1)
                q_scaled = q * math.sqrt(2.0)
                zeros_d = torch.zeros_like(k1)
                k1_pos = torch.cat([k1, zeros_d], dim=-1)
                k1_neg = torch.cat([-k1, zeros_d], dim=-1)
                k2_pos = torch.cat([zeros_d, k2], dim=-1)
                k2_neg = torch.cat([zeros_d, -k2], dim=-1)
                k_concat = torch.cat([k1_pos, k1_neg, k2_pos, k2_neg], dim=-2)
                lamb_val = torch.exp(self.raw_lambda)
                v_concat = torch.cat([v, -v, -lamb_val * v, lamb_val * v], dim=-2)
                if combined_mask is not None:
                    combined_mask_concat = torch.cat([combined_mask, combined_mask, combined_mask, combined_mask], dim=-1)
                else:
                    combined_mask_concat = None
                y = F.scaled_dot_product_attention(
                    q_scaled, k_concat, v_concat, attn_mask=combined_mask_concat, dropout_p=0.0, is_causal=False
                )
            else:
                lamb = torch.exp(self.raw_lambda)
                logits1 = torch.matmul(q1, k1.transpose(-1, -2)) / math.sqrt(self.head_dim)
                logits2 = torch.matmul(q2, k2.transpose(-1, -2)) / math.sqrt(self.head_dim)
                weights = diff_sba_weights(logits1, logits2, lamb, combined_mask)
                y = torch.matmul(weights, v)
        elif self.sba_impl == "sdpa":
            B_k, H_k, T_k, D_k = k.shape
            
            # Double K and V dynamically just-in-time
            k_dbl = torch.stack([k, -k], dim=-2).view(B_k, H_k, 2 * T_k, D_k)
            v_dbl = torch.stack([v, -v], dim=-2).view(B_k, H_k, 2 * T_k, D_k)
            
            if combined_mask is not None:
                # Double the Key sequence dimension of the mask (T -> 2T)
                combined_mask_dbl = combined_mask.repeat_interleave(2, dim=-1)
            else:
                combined_mask_dbl = None
                
            # Native SDPA: Q is length T, K is length 2T. Output is perfectly (B, H, T, D)!
            y = F.scaled_dot_product_attention(
                q, k_dbl, v_dbl, attn_mask=combined_mask_dbl, dropout_p=0.0, is_causal=False
            )
        else:
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            weights = sba_weights(logits, combined_mask)
            y = torch.matmul(weights, v)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.dropout(self.proj(y))

        if use_cache:
            return out, present_key_value
        return out


class Block(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str, sba_impl: SBAImpl = "sdpa") -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, attention, sba_impl=sba_impl)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if use_cache:
            attn_out, present_kv = self.attn(
                self.norm_1(x), attention_mask=attention_mask, past_key_value=past_key_value, use_cache=True
            )
            x = x + attn_out
            x = x + self.mlp(self.norm_2(x))
            return x, present_kv
        else:
            x = x + self.attn(self.norm_1(x), attention_mask=attention_mask)
            return x + self.mlp(self.norm_2(x))


class ScratchLM(nn.Module):
    def __init__(self, config: ScratchConfig, attention: str, sba_impl: SBAImpl = "sdpa") -> None:
        super().__init__()
        self.config = config
        self.attention = attention
        self.sba_impl = sba_impl
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config, attention, sba_impl=sba_impl) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
    ):
        batch, seq_len = input_ids.shape
        if position_ids is not None:
            positions = position_ids
        elif past_key_values is not None:
            T = past_key_values[0][0].shape[-2]
            positions = torch.tensor([[T]], device=input_ids.device)
        else:
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present_kv = block(
                    x, attention_mask=attention_mask, past_key_value=layer_past, use_cache=True
                )
                present_key_values.append(present_kv)
            else:
                x = block(x, attention_mask=attention_mask)

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        if use_cache:
            return logits, loss, present_key_values
        return logits, loss


@torch.no_grad()
def evaluate(
    model: ScratchLM,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(loss.item()))
    loss_value = sum(losses) / max(len(losses), 1)
    model.train()
    return {"loss": loss_value, "perplexity": math.exp(min(loss_value, 20.0))}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return 0.1 + 0.9 * cosine_decay


def should_save_checkpoint(step: int, final_step: int, args: argparse.Namespace) -> bool:
    if not getattr(args, "save_checkpoints", False):
        return False
    if step == final_step:
        return True
    interval = int(getattr(args, "checkpoint_every", 0) or 0)
    return interval > 0 and step % interval == 0


def sba_impl_for_attention(attention: str, args: argparse.Namespace | None = None) -> SBAImpl | None:
    if attention in {"softmax", "diff_sba"}:
        return None
    value = getattr(args, "sba_impl", "sdpa") if args is not None else "sdpa"
    if value not in {"eager", "sdpa"}:
        raise ValueError(f"unsupported sba_impl {value!r}")
    return value


def checkpoint_path_for_step(output_dir: Path, attention: str, step: int, config: ScratchConfig) -> Path:
    filename = f"{attention}_step{step:06d}_{config.n_layer}l_{config.n_head}h_{config.n_embd}d.pt"
    return output_dir / attention / "checkpoints" / filename


def save_checkpoint(
    model: ScratchLM,
    output_dir: Path,
    attention: str,
    step: int,
    config: ScratchConfig,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> str:
    path = checkpoint_path_for_step(output_dir, attention, step, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "sba-standalone-scratch-lm-v2-tokenized",
        "attention": attention,
        "sba_impl": sba_impl_for_attention(attention, args=args),
        "step": step,
        "scratch_config": asdict(config),
        "args": vars(args),
        "metrics": metrics,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "saved_at": utc_now(),
    }
    torch.save(checkpoint, path)
    return str(path)


def train_arm(
    attention: str,
    config: ScratchConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    arm_dir = output_dir / attention
    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    sba_impl = sba_impl_for_attention(attention, args=args)
    model = ScratchLM(config, attention, sba_impl=sba_impl or "sdpa").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    parameter_count = count_parameters(model)
    started = time.perf_counter()
    metrics_path = arm_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    best_val = float("inf")
    last_train_loss = None
    saved_checkpoints: list[str] = []

    train_iter = iter(train_loader)

    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Learning rate schedule
        lr_mult = get_lr_multiplier(step, args.steps, args.warmup_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr * lr_mult

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            scaled_loss = loss / args.grad_accum_steps
        scaled_loss.backward()

        if step % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last_train_loss = float(loss.item())

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            val = evaluate(model, val_loader, device, amp_dtype)
            best_val = min(best_val, val["loss"])
            event = {
                "step": step,
                "train_loss": last_train_loss,
                "val_loss": val["loss"],
                "val_perplexity": val["perplexity"],
                "best_val_loss": best_val,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                "sba_impl": sba_impl,
                "lr": args.lr * lr_mult,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            write_json(arm_dir / "latest.json", event)
        if step < args.steps and should_save_checkpoint(step, args.steps, args):
            saved_checkpoints.append(
                save_checkpoint(
                    model=model,
                    output_dir=output_dir,
                    attention=attention,
                    step=step,
                    config=config,
                    args=args,
                    metrics={"train_loss": last_train_loss, "best_val_loss_seen": best_val},
                )
            )

    final_val = evaluate(model, val_loader, device, amp_dtype)
    if should_save_checkpoint(args.steps, args.steps, args):
        saved_checkpoints.append(
            save_checkpoint(
                model=model,
                output_dir=output_dir,
                attention=attention,
                step=args.steps,
                config=config,
                args=args,
                metrics={
                    "train_loss": last_train_loss,
                    "best_val_loss_seen": best_val,
                    "final_val_loss": final_val["loss"],
                    "final_val_perplexity": final_val["perplexity"],
                },
            )
        )
    result = {
        "attention": attention,
        "sba_impl": sba_impl,
        "valid": bool(math.isfinite(final_val["loss"])),
        "parameter_count": parameter_count,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "block_size": config.block_size,
        "last_train_loss": last_train_loss,
        "final_val_loss": final_val["loss"],
        "final_val_perplexity": final_val["perplexity"],
        "best_val_loss_seen": best_val,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "elapsed_seconds": time.perf_counter() - started,
        "finished_at": utc_now(),
        "saved_checkpoints": saved_checkpoints,
    }
    write_json(arm_dir / "result.json", result)
    del model
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train matched from-scratch softmax and SBA tokenized LMs")
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[
            "data/task0_gsm8k_cot.jsonl",
            "data/task0_logic_cot.jsonl",
            "data/task1_arc_cot.jsonl",
            "data/task1_decomp_cot.jsonl",
            "data/task2_action_cot.jsonl",
            "data/task2_csqa_cot.jsonl",
        ]
    )
    parser.add_argument("--output-dir", default="sba_standalone/results/tokenized_30m_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--train-limit-per-file", type=int, default=120)
    parser.add_argument("--eval-limit-per-file", type=int, default=20)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    parser.add_argument("--arms", choices=["all", "both", "softmax", "sba", "diff_sba"], default="both")
    parser.add_argument(
        "--sba-impl",
        choices=["eager", "sdpa"],
        default="sdpa",
        help="SBA implementation for the sba arm: eager dense reference or exact doubled-KV SDPA",
    )
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    amp_dtype = torch.bfloat16 if args.dtype in {"auto", "bf16"} and torch.cuda.is_bf16_supported() else torch.float16

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ScratchConfig(
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataloaders
    train_loader, val_loader, data_manifest = build_dataloaders(
        tokenizer=tokenizer,
        data_files=args.data_files,
        max_length=args.block_size,
        train_limit_per_file=args.train_limit_per_file,
        eval_limit_per_file=args.eval_limit_per_file,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    write_json(
        output_dir / "config.json",
        {
            "script": "run_scratch_tokenized_compare.py",
            "scratch_config": asdict(config),
            "args": vars(args),
            "sba_impl": args.sba_impl,
            "amp_dtype": str(amp_dtype),
            "cuda_device": torch.cuda.get_device_name(0),
            "git_revision": git_revision(),
            "data_manifest": data_manifest,
            "started_at": utc_now(),
        },
    )

    results: dict[str, Any] = {}
    if args.arms in {"all", "both", "softmax"}:
        results["softmax"] = train_arm("softmax", config, train_loader, val_loader, args, output_dir, amp_dtype)
    if args.arms in {"all", "both", "sba"}:
        results["sba"] = train_arm("sba", config, train_loader, val_loader, args, output_dir, amp_dtype)
    if args.arms in {"all", "diff_sba"}:
        results["diff_sba"] = train_arm("diff_sba", config, train_loader, val_loader, args, output_dir, amp_dtype)

    if len(results) >= 2:
        comparison = {
            "valid": all(r["valid"] for r in results.values()),
            "finished_at": utc_now(),
        }
        for arm_name, arm_res in results.items():
            comparison[f"{arm_name}_final_val_loss"] = arm_res["final_val_loss"]
            comparison[f"{arm_name}_best_val_loss"] = arm_res["best_val_loss_seen"]
            comparison[f"{arm_name}_peak_vram_mb"] = arm_res["peak_vram_mb"]

        if "softmax" in results and "sba" in results:
            comparison["softmax_final_val_loss"] = results["softmax"]["final_val_loss"]
            comparison["sba_final_val_loss"] = results["sba"]["final_val_loss"]
            comparison["sba_impl"] = results["sba"].get("sba_impl")
            comparison["delta_val_loss_sba_minus_softmax"] = results["sba"]["final_val_loss"] - results["softmax"]["final_val_loss"]
            comparison["softmax_best_val_loss"] = results["softmax"]["best_val_loss_seen"]
            comparison["sba_best_val_loss"] = results["sba"]["best_val_loss_seen"]
            comparison["delta_best_val_loss_sba_minus_softmax"] = results["sba"]["best_val_loss_seen"] - results["softmax"]["best_val_loss_seen"]
            comparison["softmax_peak_vram_mb"] = results["softmax"]["peak_vram_mb"]
            comparison["sba_peak_vram_mb"] = results["sba"]["peak_vram_mb"]
            comparison["parameter_count"] = results["softmax"]["parameter_count"]

        write_json(output_dir / "comparison.json", comparison)
        print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
