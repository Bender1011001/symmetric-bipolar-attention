#!/usr/bin/env python3
"""Quick V3 correctness + training step benchmark."""
import sys
sys.path.insert(0, '/root/sba_standalone')

import torch
import torch.nn.functional as F
import math
import time

from sba_triton_kernel import sba_attention_fused, sba_attention_eager_reference

print("=" * 60)
print("V3 KERNEL: CORRECTNESS + TRAINING BENCHMARK")
print("=" * 60)

# --- Correctness ---
print("\n--- Correctness Tests ---")
old_tf32 = torch.backends.cuda.matmul.allow_tf32
torch.backends.cuda.matmul.allow_tf32 = False

for dtype in [torch.float32, torch.bfloat16, torch.float16]:
    for causal in [True, False]:
        torch.manual_seed(42)
        B, H, T, D = 2, 4, 256, 64
        ip = 'ieee' if dtype == torch.float32 else 'tf32'

        q = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
        k = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
        v = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
        go = torch.randn_like(q)

        y = sba_attention_fused(q, k, v, causal=causal, input_precision=ip)
        y.backward(go)
        dq1, dk1, dv1 = q.grad.clone(), k.grad.clone(), v.grad.clone()
        q.grad = k.grad = v.grad = None

        y2 = sba_attention_eager_reference(q, k, v, causal=causal)
        y2.backward(go)

        y_diff = (y.float() - y2.float()).abs().max().item()
        dq_diff = (dq1.float() - q.grad.float()).abs().max().item()
        dk_diff = (dk1.float() - k.grad.float()).abs().max().item()
        dv_diff = (dv1.float() - v.grad.float()).abs().max().item()

        if dtype == torch.float32:
            tol = 2e-3
        elif dtype == torch.float16:
            tol = 6e-2
        else:
            tol = 8e-2

        ok = all(d < tol for d in [y_diff, dq_diff, dk_diff, dv_diff])
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {str(dtype):20s} causal={causal}: Y={y_diff:.2e} dQ={dq_diff:.2e} dK={dk_diff:.2e} dV={dv_diff:.2e}")

torch.backends.cuda.matmul.allow_tf32 = old_tf32

# --- Forward+Backward micro-benchmark ---
print("\n--- Forward+Backward Micro-Benchmark ---")
for T in [256, 512, 1024]:
    B, H, D = 16, 12, 64
    dtype = torch.bfloat16

    q = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    go = torch.randn_like(q)

    # Warmup
    for _ in range(5):
        q.grad = k.grad = v.grad = None
        y = sba_attention_fused(q, k, v, causal=True)
        y.backward(go)
    torch.cuda.synchronize()

    # SBA fused
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        q.grad = k.grad = v.grad = None
        y = sba_attention_fused(q, k, v, causal=True)
        y.backward(go)
    end.record()
    torch.cuda.synchronize()
    sba_ms = start.elapsed_time(end) / 20

    # Softmax SDPA
    for _ in range(5):
        q.grad = k.grad = v.grad = None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y.backward(go)
    torch.cuda.synchronize()

    start.record()
    for _ in range(20):
        q.grad = k.grad = v.grad = None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y.backward(go)
    end.record()
    torch.cuda.synchronize()
    sdpa_ms = start.elapsed_time(end) / 20

    print(f"  T={T:4d}: SDPA={sdpa_ms:.3f}ms  SBA_fused={sba_ms:.3f}ms  ratio={sba_ms/sdpa_ms:.2f}x")

print("\nDone. Now run benchmark_step_time.py for full model step times.")
