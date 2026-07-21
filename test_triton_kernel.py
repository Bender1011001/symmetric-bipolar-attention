#!/usr/bin/env python3
"""Test the Triton SBA kernel for correctness and performance."""
import sys
sys.path.insert(0, '/root/sba_standalone')

import torch
import torch.nn.functional as F
import time
import math

from sba_triton_kernel import sba_attention_fused
from sba_kernel_spec import (
    sba_attention_eager_reference,
    sba_attention_doubled_kv,
    test_correctness,
    test_backward,
    benchmark,
    softmax_attention_baseline,
)

device = 'cuda'

print("=" * 70)
print("SBA TRITON KERNEL — FULL VALIDATION")
print("=" * 70)

# --- Correctness ---
print("\n--- Correctness Tests ---")
for T in [128, 256, 512, 1024]:
    test_correctness(sba_attention_fused, f'fused_T{T}', device=device, seq_len=T)

# Also test with fp16
print("\n--- FP16 Correctness ---")
test_correctness(sba_attention_fused, 'fused_fp16_T512', device=device, seq_len=512,
                 dtype=torch.float16)

# --- Backward ---
print("\n--- Backward Tests ---")
test_backward(sba_attention_fused, 'fused', device=device)

# Also test backward gradients match reference numerically
print("\n--- Backward Gradient Accuracy ---")
torch.manual_seed(42)
T, B, H, D = 128, 2, 4, 64
q = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)
k = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)
v = torch.randn(B, H, T, D, dtype=torch.float32, device=device, requires_grad=True)

y_fused = sba_attention_fused(q, k, v, causal=True)
loss_fused = y_fused.sum()
loss_fused.backward()
dq_fused, dk_fused, dv_fused = q.grad.clone(), k.grad.clone(), v.grad.clone()
q.grad, k.grad, v.grad = None, None, None

y_ref = sba_attention_eager_reference(q, k, v, causal=True)
loss_ref = y_ref.sum()
loss_ref.backward()
dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

for name, g_fused, g_ref in [('dQ', dq_fused, dq_ref), ('dK', dk_fused, dk_ref), ('dV', dv_fused, dv_ref)]:
    max_diff = (g_fused - g_ref).abs().max().item()
    mean_diff = (g_fused - g_ref).abs().mean().item()
    print(f"  {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {'✅' if max_diff < 1e-4 else '❌'}")

# --- Benchmarks ---
print("\n--- Benchmarks (forward only) ---")
results = {}
for T in [256, 512, 1024, 2048]:
    print(f"\n  Sequence length: {T}")
    t_softmax = benchmark(softmax_attention_baseline, 'softmax', device=device, seq_len=T)
    t_doubled = benchmark(sba_attention_doubled_kv, 'doubled_kv', device=device, seq_len=T)
    t_fused = benchmark(sba_attention_fused, 'fused', device=device, seq_len=T)
    results[T] = (t_softmax, t_doubled, t_fused)
    print(f"  Ratios: doubled_kv={t_doubled/t_softmax:.2f}x, fused={t_fused/t_softmax:.2f}x vs softmax")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'T':>6} | {'Softmax (ms)':>14} | {'Doubled-KV (ms)':>16} | {'Fused (ms)':>12} | {'DKV/SM':>8} | {'Fused/SM':>8}")
print("-" * 80)
for T, (ts, td, tf) in results.items():
    print(f"{T:>6} | {ts:>14.3f} | {td:>16.3f} | {tf:>12.3f} | {td/ts:>7.2f}x | {tf/ts:>7.2f}x")
print("=" * 70)
