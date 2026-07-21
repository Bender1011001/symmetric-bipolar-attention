#!/usr/bin/env python3
"""
V3 Kernel Correctness Verification — Non-Negotiable Pre-Publication Check

Tests the V3 fused Triton kernel (forward AND backward) against the eager
reference implementation at fp32 with ieee precision. This catches any
subtle gradient bugs in the fused backward that could flatter training
results.

Runs at multiple sizes and seeds. Reports max/mean diffs for Y, dQ, dK, dV.
"""
import sys
sys.path.insert(0, '/root/sba_standalone')

import torch
import math
from sba_triton_kernel import sba_attention_fused, sba_attention_eager_reference

torch.backends.cuda.matmul.allow_tf32 = False

print("=" * 70)
print("V3 KERNEL CORRECTNESS VERIFICATION")
print("fp32, input_precision='ieee', causal=True")
print("=" * 70)

configs = [
    # (B, H, T, D, seed)
    (2, 4, 128, 64, 42),
    (2, 4, 256, 64, 42),
    (2, 4, 512, 64, 42),
    (2, 4, 256, 64, 123),
    (2, 4, 256, 64, 999),
    (4, 12, 256, 64, 42),   # match sweep head config
    (2, 4, 1024, 64, 42),   # match sweep seq length
    (1, 1, 64, 32, 42),     # minimal
    (4, 12, 1024, 64, 123), # full production config, different seed
]

all_passed = True
worst_diffs = {"Y": 0, "dQ": 0, "dK": 0, "dV": 0}

for B, H, T, D, seed in configs:
    torch.manual_seed(seed)
    dtype = torch.float32

    q = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device='cuda', requires_grad=True)
    go = torch.randn_like(q)

    # V3 fused
    y_fused = sba_attention_fused(q, k, v, causal=True, input_precision='ieee')
    y_fused.backward(go)
    dq_fused = q.grad.clone()
    dk_fused = k.grad.clone()
    dv_fused = v.grad.clone()
    q.grad = k.grad = v.grad = None

    # Eager reference
    y_ref = sba_attention_eager_reference(q, k, v, causal=True)
    y_ref.backward(go)
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    diffs = {
        "Y": (y_fused.float() - y_ref.float()).abs().max().item(),
        "dQ": (dq_fused.float() - dq_ref.float()).abs().max().item(),
        "dK": (dk_fused.float() - dk_ref.float()).abs().max().item(),
        "dV": (dv_fused.float() - dv_ref.float()).abs().max().item(),
    }

    # Track worst case across all configs
    for k_name in diffs:
        worst_diffs[k_name] = max(worst_diffs[k_name], diffs[k_name])

    tol = 2e-5  # strict fp32 tolerance
    passed = all(d < tol for d in diffs.values())
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False

    print(f"  [{status}] B={B} H={H} T={T} D={D} seed={seed}: "
          f"Y={diffs['Y']:.2e} dQ={diffs['dQ']:.2e} dK={diffs['dK']:.2e} dV={diffs['dV']:.2e}")

print()
print("WORST-CASE DIFFS ACROSS ALL CONFIGS:")
for name, diff in worst_diffs.items():
    print(f"  {name}: {diff:.2e}")

print()
if all_passed:
    print("RESULT: ALL PASSED. V3 kernel backward is mathematically correct.")
else:
    print("RESULT: SOME TESTS FAILED. Investigate gradient bugs before proceeding.")

# Additional: verify gradient norms are comparable (catches scale bugs)
print()
print("=" * 70)
print("GRADIENT NORM COMPARISON (catch scale bugs)")
print("=" * 70)
torch.manual_seed(42)
B, H, T, D = 4, 12, 256, 64

q = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
go = torch.randn_like(q)

y = sba_attention_fused(q, k, v, causal=True, input_precision='ieee')
y.backward(go)
norms_fused = {
    "dQ": q.grad.norm().item(),
    "dK": k.grad.norm().item(),
    "dV": v.grad.norm().item(),
}
q.grad = k.grad = v.grad = None

y2 = sba_attention_eager_reference(q, k, v, causal=True)
y2.backward(go)
norms_ref = {
    "dQ": q.grad.norm().item(),
    "dK": k.grad.norm().item(),
    "dV": v.grad.norm().item(),
}

print(f"  {'':5s} | {'Fused':>12s} | {'Eager':>12s} | {'Ratio':>8s}")
print(f"  {'-'*45}")
for name in ["dQ", "dK", "dV"]:
    ratio = norms_fused[name] / norms_ref[name]
    print(f"  {name:5s} | {norms_fused[name]:12.6f} | {norms_ref[name]:12.6f} | {ratio:8.6f}")
