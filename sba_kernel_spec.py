"""
SBA Kernel Specification — Reference Implementation & Test Harness
==================================================================
This file contains:
  1. The EXACT eager PyTorch reference implementation of SBA attention (ground truth)
  2. The current SLOW Doubled-KV SDPA workaround (what we want to replace)
  3. A drop-in test harness that validates any new kernel against the reference
  4. Benchmark code that measures throughput vs standard softmax attention

YOUR TASK: Write a Triton (preferred) or CUDA kernel that replaces the Doubled-KV
workaround with a native fused implementation that operates at sequence length T
(not 2T), bringing SBA's per-step latency close to standard softmax attention.
"""

import torch
import torch.nn.functional as F
import math
import time

# ==============================================================================
# SECTION 1: MATHEMATICAL SPECIFICATION
# ==============================================================================
#
# Standard Softmax Attention:
#   S_ij = q_i · k_j / sqrt(d)
#   A_ij = exp(S_ij) / Σ_k exp(S_ik)           ← weights in [0, 1], sum to 1
#   Y_i  = Σ_j A_ij * v_j
#
# SBA (Symmetric Bipolar Attention):
#   S_ij = q_i · k_j / sqrt(d)                  ← same logit computation
#   A_ij = sinh(S_ij) / Σ_k cosh(S_ik)          ← weights in [-1, 1], |sum| ≤ 1
#   Y_i  = Σ_j A_ij * v_j
#
# Expanded form (what the kernel actually computes):
#   A_ij = (exp(S_ij) - exp(-S_ij)) / Σ_k (exp(S_ik) + exp(-S_ik))
#
# Numerically stable form (subtract row-max before exp):
#   m_i  = max_k max(S_ik, -S_ik) = max_k |S_ik|
#   e_pos_ij = exp(S_ij - m_i)
#   e_neg_ij = exp(-S_ij - m_i)
#   A_ij = (e_pos_ij - e_neg_ij) / Σ_k (e_pos_ik + e_neg_ik)
#   Y_i  = Σ_j A_ij * v_j
#
# KEY INSIGHT: This is extremely similar to standard softmax attention.
# The ONLY differences are:
#   1. Two exp() calls per element instead of one (exp(S) AND exp(-S))
#   2. Numerator is (exp(S) - exp(-S)) instead of exp(S)
#   3. Denominator is Σ(exp(S) + exp(-S)) instead of Σ exp(S)
#
# A FlashAttention-style tiled implementation would look nearly identical to
# standard FlashAttention, but tracking two running sums per row instead of one.
#
# CAUSAL MASKING: For masked positions (j > i), set A_ij = 0.
# This means: skip those positions in both numerator and denominator sums.
# ==============================================================================


# ==============================================================================
# SECTION 2: REFERENCE IMPLEMENTATION (EAGER PyTorch — GROUND TRUTH)
# ==============================================================================

def sba_attention_eager_reference(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    v: torch.Tensor,  # (B, H, T, D)
    causal: bool = True,
    scale: float | None = None,
) -> torch.Tensor:   # (B, H, T, D)
    """
    Ground-truth SBA attention. O(T^2) memory. Use only for correctness testing.
    """
    B, H, T, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Compute logits: (B, H, T, T)
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    work = logits.float()
    
    if causal:
        # Build causal mask
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        floor = torch.finfo(work.dtype).min
        zeros = torch.zeros_like(work)
        
        pos_logits = torch.where(mask, work, floor)
        neg_logits = torch.where(mask, -work, floor)
        
        # Numerically stable: subtract row-max
        row_max = torch.maximum(
            pos_logits.max(dim=-1, keepdim=True).values,
            neg_logits.max(dim=-1, keepdim=True).values,
        )
        
        e_pos = torch.where(mask, torch.exp(pos_logits - row_max), zeros)
        e_neg = torch.where(mask, torch.exp(neg_logits - row_max), zeros)
        
        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(1e-30)
        weights = torch.where(mask, (e_pos - e_neg) / denom, zeros)
    else:
        row_max = torch.maximum(
            work.max(dim=-1, keepdim=True).values,
            (-work).max(dim=-1, keepdim=True).values,
        )
        e_pos = torch.exp(work - row_max)
        e_neg = torch.exp(-work - row_max)
        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(1e-30)
        weights = (e_pos - e_neg) / denom
    
    weights = weights.to(dtype=q.dtype)
    return torch.matmul(weights, v)


# ==============================================================================
# SECTION 3: CURRENT WORKAROUND (SLOW — Doubled-KV via standard SDPA)
# ==============================================================================

def sba_attention_doubled_kv(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    v: torch.Tensor,  # (B, H, T, D)
    causal: bool = True,
) -> torch.Tensor:   # (B, H, T, D)
    """
    Current production implementation. Maps SBA to standard softmax by doubling
    the KV sequence length from T to 2T. This is correct but ~1.5x slower than
    standard softmax because FlashAttention sees sequence length 2T.
    
    Math proof of equivalence:
      softmax([Q·K^T, Q·(-K)^T]) · [V; -V]
      = Σ_j [ exp(S_ij)/(Σ_k exp(S_ik) + exp(-S_ik)) * v_j 
            + exp(-S_ij)/(Σ_k exp(S_ik) + exp(-S_ik)) * (-v_j) ]
      = Σ_j (exp(S_ij) - exp(-S_ij)) / (Σ_k exp(S_ik) + exp(-S_ik)) * v_j
      = SBA(Q, K, V)
    """
    B, H, T, D = q.shape
    
    # Double the KV sequence: [K, -K] and [V, -V]
    k_dbl = torch.stack([k, -k], dim=-2).reshape(B, H, 2 * T, D)
    v_dbl = torch.stack([v, -v], dim=-2).reshape(B, H, 2 * T, D)
    
    # Double the query to match (we only keep odd-indexed outputs)
    q_dbl = torch.stack([q, q], dim=-2).reshape(B, H, 2 * T, D)
    
    # Build doubled causal mask if needed
    if causal:
        # Original causal: mask[i,j] = (j <= i), shape (T, T)
        # Doubled: mask[2i, 2j] = mask[2i, 2j+1] = mask[2i+1, 2j] = mask[2i+1, 2j+1] = (j <= i)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        mask_dbl = causal_mask.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        mask_dbl = mask_dbl.unsqueeze(0).unsqueeze(0)  # (1, 1, 2T, 2T)
    else:
        mask_dbl = None
    
    y_dbl = F.scaled_dot_product_attention(
        q_dbl, k_dbl, v_dbl, attn_mask=mask_dbl, dropout_p=0.0, is_causal=False
    )
    
    # Take only the odd-indexed rows (the "real" query positions)
    return y_dbl[:, :, 1::2, :]


# ==============================================================================
# SECTION 4: YOUR KERNEL GOES HERE
# ==============================================================================

def sba_attention_fused(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    v: torch.Tensor,  # (B, H, T, D)
    causal: bool = True,
) -> torch.Tensor:   # (B, H, T, D)
    """
    *** REPLACE THIS WITH YOUR FUSED TRITON/CUDA KERNEL ***
    
    Requirements:
      - Input shapes: Q, K, V are all (B, H, T, D) where B=batch, H=heads, T=seq_len, D=head_dim
      - Output shape: (B, H, T, D)
      - Must support causal masking (lower-triangular)
      - Must support bf16 and fp16 inputs
      - Must be numerically stable (use row-max subtraction like FlashAttention)
      - Must match sba_attention_eager_reference() to within fp32 tolerance (~1e-6)
      - Must operate at sequence length T (NOT 2T)
      - Must support autograd (backward pass)
    
    Performance target:
      - Within 1.2x of standard F.scaled_dot_product_attention latency
      - Currently we are at 1.52x — closing this gap is the entire goal
    
    The algorithm (FlashAttention-style tiling for SBA):
      For each tile of Q rows (block_q):
        Initialize per-row accumulators:
          m_i = -inf          (running row-max of |S_ij|)
          l_pos_i = 0         (running sum of exp(S_ij - m_i) for valid j)
          l_neg_i = 0         (running sum of exp(-S_ij - m_i) for valid j)
          o_i = 0             (running output accumulator)
        
        For each tile of K/V columns (block_kv):
          Compute tile logits: S = Q_tile @ K_tile^T * scale     # (block_q, block_kv)
          
          # Update running max (need max of |S|)
          m_new = max(m_i, max(|S_tile|))
          
          # Rescale old accumulators
          correction = exp(m_old - m_new)
          l_pos_i *= correction
          l_neg_i *= correction
          o_i *= correction   # NOTE: this rescales the numerator accumulator
          
          # Compute new exponentials
          e_pos = exp(S_tile - m_new)    # exp(+logit)
          e_neg = exp(-S_tile - m_new)   # exp(-logit)
          
          # Apply causal mask (zero out invalid positions)
          
          # Accumulate
          l_pos_i += sum(e_pos, axis=-1)
          l_neg_i += sum(e_neg, axis=-1)
          o_i += (e_pos - e_neg) @ V_tile    # <-- KEY: signed weights times values
        
        # Final normalization
        Y_tile = o_i / (l_pos_i + l_neg_i)
    
    This is essentially FlashAttention with:
      - Two exp() per element instead of one
      - Two running denominators (l_pos, l_neg) instead of one (l)
      - Numerator weights are (e_pos - e_neg) instead of e_pos
      - Row-max tracks max(|S|) instead of max(S)
    """
    # Placeholder — currently falls back to reference
    return sba_attention_eager_reference(q, k, v, causal=causal)


# ==============================================================================
# SECTION 5: TEST HARNESS
# ==============================================================================

def test_correctness(
    impl_fn,
    name: str = "fused",
    batch: int = 2,
    heads: int = 12,
    seq_len: int = 512,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    causal: bool = True,
):
    """Test that impl_fn matches the eager reference."""
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Reference (always fp32 internally)
    y_ref = sba_attention_eager_reference(q, k, v, causal=causal)
    
    # Implementation under test
    y_test = impl_fn(q, k, v, causal=causal)
    
    # Compare in fp32
    max_diff = (y_ref.float() - y_test.float()).abs().max().item()
    mean_diff = (y_ref.float() - y_test.float()).abs().mean().item()
    
    # Tolerance depends on dtype
    tol = 3e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-5
    passed = max_diff < tol
    
    print(f"[{name}] seq_len={seq_len}, dtype={dtype}")
    print(f"  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, tol={tol:.0e}")
    print(f"  {'✅ PASSED' if passed else '❌ FAILED'}")
    return passed


def test_backward(
    impl_fn,
    name: str = "fused",
    batch: int = 2,
    heads: int = 12,
    seq_len: int = 256,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
):
    """Test that impl_fn supports autograd backward pass."""
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    try:
        y = impl_fn(q, k, v, causal=True)
        loss = y.sum()
        loss.backward()
        
        grads_ok = all(x.grad is not None and x.grad.shape == x.shape for x in [q, k, v])
        grads_finite = all(torch.isfinite(x.grad).all() for x in [q, k, v])
        passed = grads_ok and grads_finite
        print(f"[{name}] backward: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    except Exception as e:
        print(f"[{name}] backward: ❌ FAILED ({e})")
        return False


def benchmark(
    impl_fn,
    name: str,
    batch: int = 4,
    heads: int = 12,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    warmup: int = 10,
    repeats: int = 50,
):
    """Benchmark forward pass latency."""
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = impl_fn(q, k, v, causal=True)
    torch.cuda.synchronize()
    
    # Timed
    start = time.perf_counter()
    for _ in range(repeats):
        _ = impl_fn(q, k, v, causal=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000  # ms
    
    print(f"[{name}] B={batch}, H={heads}, T={seq_len}, D={head_dim}: {elapsed:.3f} ms/call")
    return elapsed


def softmax_attention_baseline(q, k, v, causal=True):
    """Standard softmax for benchmarking comparison."""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


# ==============================================================================
# SECTION 6: MAIN — Run all tests and benchmarks
# ==============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("SBA KERNEL VALIDATION SUITE")
    print("=" * 70)
    
    # --- Correctness Tests ---
    print("\n--- Correctness Tests ---")
    
    # Test doubled-KV (current production) against reference
    test_correctness(sba_attention_doubled_kv, "doubled_kv", device=device, seq_len=256)
    test_correctness(sba_attention_doubled_kv, "doubled_kv", device=device, seq_len=512)
    
    # Test YOUR kernel against reference
    test_correctness(sba_attention_fused, "fused", device=device, seq_len=256)
    test_correctness(sba_attention_fused, "fused", device=device, seq_len=512)
    test_correctness(sba_attention_fused, "fused", device=device, seq_len=1024)
    
    # --- Backward Tests ---
    print("\n--- Backward Tests ---")
    test_backward(sba_attention_fused, "fused", device=device)
    
    # --- Benchmarks ---
    print("\n--- Benchmarks (forward only) ---")
    for T in [256, 512, 1024, 2048]:
        print(f"\n  Sequence length: {T}")
        t_softmax = benchmark(softmax_attention_baseline, "softmax", device=device, seq_len=T)
        t_doubled = benchmark(sba_attention_doubled_kv, "doubled_kv", device=device, seq_len=T)
        t_fused = benchmark(sba_attention_fused, "fused", device=device, seq_len=T)
        
        print(f"  Ratios: doubled_kv={t_doubled/t_softmax:.2f}x, fused={t_fused/t_softmax:.2f}x vs softmax")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
