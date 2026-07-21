#!/usr/bin/env python3
"""Timing stability: run the SBA fused benchmark 3 times."""
import sys
sys.path.insert(0, '/root/sba_standalone')

import torch
import torch.nn.functional as F
import time

from sba_triton_kernel import sba_attention_fused

B, H, T, D = 16, 12, 1024, 64
dtype = torch.bfloat16

print("TIMING STABILITY CHECK — 3 rounds")
print(f"Config: B={B}, H={H}, T={T}, D={D}, dtype={dtype}, causal=True")
print()

sba_times = []
sdpa_times = []

for round_num in range(3):
    torch.manual_seed(123 + round_num)
    q = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    go = torch.randn_like(q)

    # Warmup
    for _ in range(10):
        q.grad = k.grad = v.grad = None
        y = sba_attention_fused(q, k, v, causal=True)
        y.backward(go)
    torch.cuda.synchronize()

    # SBA timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(50):
        q.grad = k.grad = v.grad = None
        y = sba_attention_fused(q, k, v, causal=True)
        y.backward(go)
    end.record()
    torch.cuda.synchronize()
    sba_ms = start.elapsed_time(end) / 50

    # SDPA warmup
    for _ in range(10):
        q.grad = k.grad = v.grad = None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y.backward(go)
    torch.cuda.synchronize()

    # SDPA timing
    start.record()
    for _ in range(50):
        q.grad = k.grad = v.grad = None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y.backward(go)
    end.record()
    torch.cuda.synchronize()
    sdpa_ms = start.elapsed_time(end) / 50

    ratio = sba_ms / sdpa_ms
    print(f"  Round {round_num+1}: SDPA={sdpa_ms:.3f}ms  SBA={sba_ms:.3f}ms  ratio={ratio:.3f}x")
    sba_times.append(sba_ms)
    sdpa_times.append(sdpa_ms)

print()
sba_avg = sum(sba_times) / len(sba_times)
sdpa_avg = sum(sdpa_times) / len(sdpa_times)
sba_std = (sum((t - sba_avg)**2 for t in sba_times) / len(sba_times))**0.5
ratio_avg = sba_avg / sdpa_avg
print(f"  SBA mean: {sba_avg:.3f} ± {sba_std:.3f} ms")
print(f"  SDPA mean: {sdpa_avg:.3f} ms")
print(f"  Ratio mean: {ratio_avg:.3f}x")
print(f"  Timing variation: ±{sba_std/sba_avg*100:.1f}%")
