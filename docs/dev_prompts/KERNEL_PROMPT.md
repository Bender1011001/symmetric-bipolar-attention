# Triton Kernel Request: Fused SBA (Symmetric Bipolar Attention)

## What I Need

A **Triton kernel** (OpenAI Triton, not Triton Inference Server) that implements Symmetric Bipolar Attention (SBA) as a fused, FlashAttention-style tiled operation. This kernel must operate at the native sequence length T, not the current 2T workaround.

## Why

SBA is an attention variant where weights can be negative (signed), enabling inhibitory routing. We have empirical evidence (n=3 seeds, 125M parameter GPT-2, 15k training steps) that SBA produces **1.28% lower validation loss** than standard softmax attention on reasoning/code datasets. But our current implementation uses a mathematical trick that doubles the KV sequence length to map SBA onto standard FlashAttention — this makes each step **1.52× slower** than softmax. When given equal wall-clock compute, softmax catches up by running 52% more steps.

**If we close the throughput gap to <1.2×, SBA wins on both per-step AND per-dollar metrics.** The math allows this — the overhead should be ~2 extra exp() calls per attention element, not 2× the sequence length.

## The Math

### Standard Softmax Attention
```
S_ij = q_i · k_j / sqrt(d)
A_ij = exp(S_ij) / Σ_k exp(S_ik)
Y_i  = Σ_j A_ij * v_j
```

### SBA (what the kernel must compute)
```
S_ij = q_i · k_j / sqrt(d)           ← identical logit computation
A_ij = sinh(S_ij) / Σ_k cosh(S_ik)   ← signed weights, |row sum| ≤ 1
Y_i  = Σ_j A_ij * v_j
```

### Expanded (what the kernel actually implements)
```
A_ij = (exp(S_ij) - exp(-S_ij)) / Σ_k (exp(S_ik) + exp(-S_ik))
```

### Numerically stable form (required for the kernel)
```
m_i      = max_k |S_ik|                              ← row-max of absolute logits
e_pos_ij = exp(+S_ij - m_i)                           ← standard exp
e_neg_ij = exp(-S_ij - m_i)                            ← extra exp (the only new cost)
A_ij     = (e_pos_ij - e_neg_ij) / Σ_k (e_pos_ik + e_neg_ik)
Y_i      = Σ_j A_ij * v_j
```

## How It Maps to FlashAttention Tiling

Standard FlashAttention tracks per-row:
- `m_i`: running max of logits
- `l_i`: running sum of exp(S - m) (the softmax denominator)  
- `o_i`: running output accumulator (numerator × V, rescaled)

**SBA needs almost the same thing, with these changes:**

| Component | Standard FlashAttention | SBA FlashAttention |
|:---|:---|:---|
| Row max | `m_i = max_k S_ik` | `m_i = max_k \|S_ik\|` (absolute value) |
| Denominator | `l_i = Σ_k exp(S_ik - m_i)` | `l_i = Σ_k (exp(S_ik - m_i) + exp(-S_ik - m_i))` |
| Weight × V | `o_i += exp(S - m) @ V` | `o_i += (exp(S - m) - exp(-S - m)) @ V` |
| Final output | `Y_i = o_i / l_i` | `Y_i = o_i / l_i` (same) |
| Rescaling | `o_i *= exp(m_old - m_new)` | `o_i *= exp(m_old - m_new)` (same) |

**That's it.** One extra exp() per element, one extra subtraction, and the row-max uses absolute values. Everything else — the tiling, the online rescaling, the causal masking, the memory access pattern — is identical to FlashAttention.

## Tiled Algorithm Pseudocode (FlashAttention-2 style)

```python
# For each block of query rows (block_q rows at a time):
for i_block in range(0, T, BLOCK_Q):
    # Initialize accumulators for this block of rows
    m_i = -inf                    # (BLOCK_Q,) running row-max of |S|
    l_i = 0                       # (BLOCK_Q,) running denominator sum
    o_i = 0                       # (BLOCK_Q, D) running output accumulator
    
    # For each block of KV columns:
    for j_block in range(0, T_valid, BLOCK_KV):     # T_valid = j_block+BLOCK_KV for causal
        # Load tiles
        Q_tile = Q[i_block : i_block+BLOCK_Q]       # (BLOCK_Q, D)
        K_tile = K[j_block : j_block+BLOCK_KV]      # (BLOCK_KV, D)
        V_tile = V[j_block : j_block+BLOCK_KV]      # (BLOCK_KV, D)
        
        # Compute logits
        S_tile = Q_tile @ K_tile.T * scale           # (BLOCK_Q, BLOCK_KV)
        
        # Apply causal mask (set masked positions to 0 in both e_pos and e_neg)
        
        # Update running max: max of |S| across all tiles seen so far
        m_new = max(m_i, max(|S_tile|, axis=-1))     # (BLOCK_Q,)
        
        # Rescale old accumulators to new max
        alpha = exp(m_old - m_new)                    # (BLOCK_Q,)
        l_i = l_i * alpha
        o_i = o_i * alpha[:, None]
        
        # Compute exponentials
        e_pos = exp(+S_tile - m_new[:, None])         # (BLOCK_Q, BLOCK_KV)
        e_neg = exp(-S_tile - m_new[:, None])         # (BLOCK_Q, BLOCK_KV)
        
        # Accumulate denominator: Σ(exp(S) + exp(-S))
        l_i += (e_pos + e_neg).sum(axis=-1)           # (BLOCK_Q,)
        
        # Accumulate numerator × V: (exp(S) - exp(-S)) @ V
        weights = e_pos - e_neg                       # (BLOCK_Q, BLOCK_KV) — SIGNED!
        o_i += weights @ V_tile                       # (BLOCK_Q, D)
        
        m_i = m_new
    
    # Final normalization
    Y[i_block : i_block+BLOCK_Q] = o_i / l_i[:, None]
```

## Inputs & Outputs

```
Forward:
  Input:  Q, K, V — each (batch, heads, seq_len, head_dim), dtype bf16 or fp16
  Input:  causal (bool) — whether to apply causal (lower-triangular) masking
  Output: Y — (batch, heads, seq_len, head_dim), same dtype as input

Backward:
  Input:  dY — gradient of output, same shape/dtype
  Output: dQ, dK, dV — gradients of Q, K, V
```

## Constraints & Requirements

1. **Correctness**: Must match the eager reference implementation to within bf16 tolerance (~1e-2 max absolute error). The reference implementation is in `sba_kernel_spec.py`, function `sba_attention_eager_reference()`.

2. **Performance**: Target ≤1.2× the latency of `torch.nn.functional.scaled_dot_product_attention` for the same shapes. Current doubled-KV workaround is 1.52×.

3. **Shapes we care about**:
   - batch=4-16, heads=12, seq_len=1024, head_dim=64 (our 125M model)
   - batch=4-8, heads=16, seq_len=2048, head_dim=64 (future 350M model)
   - batch=1-4, heads=32, seq_len=4096, head_dim=128 (future 1.5B model)

4. **Causal masking**: Must support causal (lower-triangular) attention. Non-causal is a bonus.

5. **Autograd**: Must support torch autograd backward pass (dQ, dK, dV gradients).

6. **GPU targets**: NVIDIA Ampere (A100) and Blackwell (RTX PRO 4500). CUDA compute capability ≥ 8.0.

7. **Dependencies**: Triton (OpenAI) preferred. Raw CUDA C++ acceptable. No external kernel libraries beyond PyTorch + Triton.

## File Structure

```
sba_kernel_spec.py          ← Reference impl + test harness + benchmarks (PROVIDED)
sba_triton_kernel.py        ← YOUR OUTPUT: The Triton kernel implementation
```

Put your kernel in `sba_triton_kernel.py`. It must export a function with this signature:

```python
def sba_attention_fused(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    v: torch.Tensor,  # (B, H, T, D)
    causal: bool = True,
) -> torch.Tensor:    # (B, H, T, D)
```

Then in `sba_kernel_spec.py`, I will replace the placeholder `sba_attention_fused` with an import from your file and run the test suite.

## Backward Pass Notes

The backward pass for SBA follows the same structure as FlashAttention's backward. The key differences mirror the forward:

For the backward, you need to compute dQ, dK, dV given dO (gradient of output Y).

The standard FlashAttention backward computes:
```
dV = A^T @ dO
dA = dO @ V^T  
dS = A * (dA - (A * dA).sum(dim=-1, keepdim=True))   ← softmax Jacobian
dQ = dS @ K * scale
dK = dS^T @ Q * scale
```

For SBA, the Jacobian of A w.r.t. S is different because A = sinh(S)/Σcosh(S) instead of exp(S)/Σexp(S). The derivative is:

```
∂A_ij/∂S_ij = cosh(S_ij)/Σ_k cosh(S_ik) - sinh²(S_ij)/(Σ_k cosh(S_ik))²
            = (cosh(S_ij) * Σ_k cosh(S_ik) - sinh²(S_ij)) / (Σ_k cosh(S_ik))²
```

In terms of the exponentials we already have:
```
∂A_ij/∂S_ij = (e_pos + e_neg) / denom - ((e_pos - e_neg) / denom)²
            = (e_pos + e_neg) / denom - A_ij²
```

And the off-diagonal:
```
∂A_ij/∂S_ik (k≠j) = -sinh(S_ij) * sinh(S_ik) / (Σ cosh)²
                   = -A_ij * A_ik    ... wait, not quite.
```

Actually, let me write it cleanly. Define:
```
c_ij = cosh(S_ij) = (e_pos + e_neg) / 2
s_ij = sinh(S_ij) = (e_pos - e_neg) / 2
D_i  = Σ_k cosh(S_ik)
A_ij = s_ij / D_i
```

Then:
```
∂A_ij/∂S_ij = c_ij/D_i - s_ij² * (-1/D_i²) * ... 
```

Actually, it's simpler to use the chain rule directly:
```
dS_ij = dA_ij * (cosh(S_ij)/D_i) - A_ij * Σ_k dA_ik * sinh(S_ik)/D_i
```

If the backward pass is too complex for a first implementation, a `torch.autograd.Function` with the forward kernel and PyTorch autograd for backward (via the eager reference) is acceptable as a starting point. We can optimize the backward later.

## How to Validate

```bash
# Run the test suite
python sba_kernel_spec.py

# Expected output for a correct, fast implementation:
# [fused] seq_len=256:  ✅ PASSED
# [fused] seq_len=512:  ✅ PASSED
# [fused] seq_len=1024: ✅ PASSED
# [fused] backward:     ✅ PASSED
# Ratios: fused=1.1x-1.2x vs softmax   ← THIS IS THE GOAL
```

## Reference Material

- Flash Attention 2 paper: https://arxiv.org/abs/2307.08691
- Triton Flash Attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- Our paper on SBA math: The key identity is sinh(x)/Σcosh = (exp(x)-exp(-x))/Σ(exp(x)+exp(-x))
