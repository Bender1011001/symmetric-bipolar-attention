#!/usr/bin/env python3
"""
Quick end-to-end training step benchmark: softmax vs SBA (fused Triton kernel).
Measures real step time with the same 125M architecture as the sweep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys
import os

sys.path.insert(0, '/root/sba_standalone')
from sba_triton_kernel import sba_attention_fused


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attention_mode="softmax"):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.attention_mode == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        elif self.attention_mode == "sba_triton":
            y = sba_attention_fused(q, k, v, causal=True)
        elif self.attention_mode == "sba_doubled_kv":
            _B, H, _T, D = k.shape
            k_dbl = torch.stack([k, -k], dim=-2).view(_B, H, 2 * _T, D)
            v_dbl = torch.stack([v, -v], dim=-2).view(_B, H, 2 * _T, D)
            q_dbl = torch.stack([q, q], dim=-2).view(_B, H, 2 * _T, D)
            mask_dbl = self.causal_mask[:, :, :_T, :_T].repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
            y_dbl = F.scaled_dot_product_attention(
                q_dbl, k_dbl, v_dbl, attn_mask=mask_dbl, dropout_p=0.0, is_causal=False
            )
            y = y_dbl[:, :, 1::2, :]
        else:
            raise ValueError(f"Unknown attention mode: {self.attention_mode}")

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attention_mode="softmax"):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attention_mode)
        self.norm2 = RMSNorm(n_embd)
        hidden = 4 * n_embd
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, n_embd, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, attention_mode="softmax"):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, attention_mode) for _ in range(n_layer)
        ])
        self.norm = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


def benchmark_training(attention_mode, steps=500, warmup=20):
    """Run training loop and measure step time."""
    # Config matching our 125M sweep
    n_embd = 768
    n_head = 12
    n_layer = 12
    block_size = 1024
    vocab_size = 32000
    batch_size = 16
    lr = 6e-4
    dtype = torch.bfloat16

    device = torch.device("cuda")

    # Build model
    model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size, attention_mode).to(device).to(dtype)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Synthetic data (random tokens — we only care about step time, not loss)
    torch.manual_seed(42)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {attention_mode}")
    print(f"  Model: {param_count/1e6:.1f}M params, {n_layer}L/{n_head}H/{n_embd}D")
    print(f"  Batch: {batch_size} × {block_size} tokens")
    print(f"  Steps: {warmup} warmup + {steps} timed")
    print(f"{'='*60}")

    step_times = []

    for step in range(warmup + steps):
        # Generate random batch each step
        input_ids = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, block_size), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size).float(), targets.view(-1))

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000

        if step >= warmup:
            step_times.append(elapsed_ms)
            if (step - warmup) % 50 == 0:
                print(f"  Step {step - warmup}/{steps}: {elapsed_ms:.1f} ms, loss={loss.item():.4f}")

    # Stats
    step_times_t = torch.tensor(step_times)
    mean_ms = step_times_t.mean().item()
    std_ms = step_times_t.std().item()
    median_ms = step_times_t.median().item()
    p95_ms = step_times_t.quantile(0.95).item()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n  Results for {attention_mode}:")
    print(f"    Mean step:   {mean_ms:.1f} ± {std_ms:.1f} ms")
    print(f"    Median step: {median_ms:.1f} ms")
    print(f"    P95 step:    {p95_ms:.1f} ms")
    print(f"    Peak VRAM:   {peak_vram:.2f} GB")

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return {
        "mode": attention_mode,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "peak_vram_gb": peak_vram,
        "param_count": param_count,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Timed steps per config")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup steps")
    parser.add_argument("--modes", nargs="+", default=["softmax", "sba_triton"],
                        choices=["softmax", "sba_triton", "sba_doubled_kv"])
    args = parser.parse_args()

    results = []
    for mode in args.modes:
        r = benchmark_training(mode, steps=args.steps, warmup=args.warmup)
        results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<20} | {'Mean (ms)':>10} | {'Median (ms)':>12} | {'VRAM (GB)':>10} | {'vs Softmax':>10}")
    print("-" * 70)
    sm_mean = next((r['mean_ms'] for r in results if r['mode'] == 'softmax'), None)
    for r in results:
        ratio = f"{r['mean_ms']/sm_mean:.3f}x" if sm_mean else "—"
        print(f"{r['mode']:<20} | {r['mean_ms']:>10.1f} | {r['median_ms']:>12.1f} | {r['peak_vram_gb']:>10.2f} | {ratio:>10}")
    print("=" * 70)
