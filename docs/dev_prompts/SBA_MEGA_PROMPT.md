# SBA (Symmetric Bipolar Attention) — Complete Technical Analysis & Kernel Request

You are being given a complete technical package for a novel attention mechanism called **Symmetric Bipolar Attention (SBA)**. This is real research with real training results. I need you to complete 5 tasks in one pass. Be rigorous — if any of our claims are wrong, say so.

---

## CONTEXT: What SBA Is

Standard Transformer self-attention uses softmax, which forces attention weights to be non-negative:
```
A_ij = exp(S_ij) / Σ_k exp(S_ik)     where S_ij = q_i · k_j / √d
```

SBA replaces softmax with hyperbolic normalization, allowing signed (negative) attention weights:
```
A_ij = sinh(S_ij) / Σ_k cosh(S_ik)
     = (exp(S_ij) - exp(-S_ij)) / Σ_k (exp(S_ik) + exp(-S_ik))
```

Properties:
- A_ij ∈ [-1, 1] (signed weights)
- Σ_j |A_ij| ≤ 1 (L1-bounded rows)
- Infinitely differentiable everywhere, including at S=0
- Enables "inhibitory routing" — tokens can subtract/cancel other tokens' representations

### Current Implementation: Doubled-KV Trick

We currently map SBA onto standard FlashAttention/SDPA by concatenating:
```
K_cat = [K, -K]  ∈ ℝ^(2T × d)
V_cat = [V, -V]  ∈ ℝ^(2T × d)
```
Then calling standard `F.scaled_dot_product_attention(Q, K_cat, V_cat)`.

This is mathematically correct (proof below) but makes FlashAttention see sequence length 2T instead of T, causing **1.52× step latency overhead**.

### Empirical Results (n=3 seeds, 125M parameter GPT-2, 15k steps, bug-free)

| Seed | Softmax Val Loss | SBA Val Loss | Δ |
|:---:|:---:|:---:|:---:|
| 42 | 1.3682 | 1.3469 | +0.0213 (SBA wins) |
| 43 | 1.3628 | 1.3508 | +0.0120 (SBA wins) |
| 44 | 1.3664 | 1.3473 | +0.0191 (SBA wins) |
| **Mean** | **1.3658 ± 0.0027** | **1.3483 ± 0.0021** | **+0.0175 (−1.28%)** |

Hardware overhead (seed 42):
- VRAM: 18.01 GB (softmax) vs 20.55 GB (SBA) = 1.14×
- Step time: 804.8 ms vs 1226.3 ms = **1.52×**

Iso-clock test (same wall-clock budget):
- SBA 15,000 steps → val loss 1.3469
- Softmax 22,760 steps → val loss **1.2617** (softmax wins with more steps)

**Bottom line: SBA has better representation capacity per gradient update, but the 1.52× overhead erases the advantage under fixed compute budgets.**

---

## TASK 1: TRITON KERNEL (Primary)

Write a **Triton kernel** that implements SBA attention as a fused, FlashAttention-2-style tiled operation at native sequence length T (not 2T).

### The Algorithm

The tiling structure is identical to FlashAttention-2, with these modifications:

| Component | Standard FlashAttention | SBA FlashAttention |
|:---|:---|:---|
| Row max | `m_i = max_k S_ik` | `m_i = max_k |S_ik|` |
| Denominator | `l_i = Σ_k exp(S_ik - m_i)` | `l_i = Σ_k [exp(S_ik - m_i) + exp(-S_ik - m_i)]` |
| Weight × V | `o_i += exp(S - m) @ V` | `o_i += [exp(S - m) - exp(-S - m)] @ V` |
| Final output | `Y_i = o_i / l_i` | `Y_i = o_i / l_i` |
| Rescaling on max update | `o_i *= exp(m_old - m_new)` | `o_i *= exp(m_old - m_new)` (same) |

### Detailed Pseudocode

```python
# Forward pass (FlashAttention-2 style, outer loop over Q blocks)
for i_block in range(0, T, BLOCK_Q):
    # Load Q tile: (BLOCK_Q, D)
    Q_tile = Q[i_block : i_block + BLOCK_Q]
    
    # Initialize per-row accumulators
    m_i = -inf            # (BLOCK_Q,) running max of |S|
    l_i = 0.0             # (BLOCK_Q,) running denominator Σ(e_pos + e_neg)
    o_i = zeros(BLOCK_Q, D)  # running numerator accumulator
    
    # Inner loop over KV blocks
    kv_end = min(i_block + BLOCK_Q, T) if causal else T  # causal: only up to diagonal
    for j_block in range(0, kv_end, BLOCK_KV):
        K_tile = K[j_block : j_block + BLOCK_KV]   # (BLOCK_KV, D)
        V_tile = V[j_block : j_block + BLOCK_KV]   # (BLOCK_KV, D)
        
        # Compute logits
        S = Q_tile @ K_tile.T * scale               # (BLOCK_Q, BLOCK_KV)
        
        # Apply causal mask: zero out positions where j > i
        if causal and j_block + BLOCK_KV > i_block:
            # mask out future positions
            ...
        
        # New row-max: max(|S|) across this tile
        tile_max = max(abs(S), axis=-1)              # (BLOCK_Q,)
        m_new = max(m_i, tile_max)                   # (BLOCK_Q,)
        
        # Rescale old accumulators to new max
        alpha = exp(m_i - m_new)                     # (BLOCK_Q,)
        l_i *= alpha
        o_i *= alpha[:, None]                        # broadcast over D
        
        # Compute exponentials
        e_pos = exp(S - m_new[:, None])              # (BLOCK_Q, BLOCK_KV)
        e_neg = exp(-S - m_new[:, None])             # (BLOCK_Q, BLOCK_KV)
        
        # Zero out masked positions in both e_pos and e_neg
        
        # Accumulate denominator
        l_i += (e_pos + e_neg).sum(axis=-1)          # (BLOCK_Q,)
        
        # Accumulate output: SIGNED weights × values
        weights = e_pos - e_neg                      # (BLOCK_Q, BLOCK_KV)
        o_i += weights @ V_tile                      # (BLOCK_Q, D)
        
        m_i = m_new
    
    # Final normalization
    Y[i_block : i_block + BLOCK_Q] = o_i / l_i[:, None]
```

### Requirements

- **Inputs**: Q, K, V each `(batch, heads, seq_len, head_dim)`, dtype bf16 or fp16
- **Output**: Y same shape/dtype
- **Causal masking**: lower-triangular (j ≤ i)
- **Numerical stability**: row-max subtraction (as shown above)
- **Autograd**: must support backward pass (see Task 5 for Jacobian)
- **Performance target**: ≤1.2× latency of `F.scaled_dot_product_attention`
- **Target shapes**: B=4-16, H=12, T=1024, D=64 (our 125M model)

### Reference Implementation for Correctness Testing

```python
def sba_attention_eager_reference(q, k, v, causal=True, scale=None):
    """Ground truth. O(T²) memory. Use only for testing."""
    B, H, T, D = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    work = logits.float()
    
    if causal:
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        floor = torch.finfo(work.dtype).min
        zeros = torch.zeros_like(work)
        
        pos_logits = torch.where(mask, work, floor)
        neg_logits = torch.where(mask, -work, floor)
        
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
    
    return torch.matmul(weights.to(q.dtype), v)
```

### Output Format for Task 1

Provide a complete, runnable `sba_triton_kernel.py` file that exports:
```python
def sba_attention_fused(q, k, v, causal=True) -> torch.Tensor
```

---

## TASK 2: HYBRID-LAYER OVERHEAD ANALYSIS

Both the Cogformer (Lv et al., 2024) and Integral Transformer (Kobyzev et al., EMNLP 2025) papers found that alternative attention is most beneficial in the **upper layers** of a Transformer, while vanilla softmax in lower layers aids training stability.

We propose a hybrid architecture: **bottom 50% of layers use softmax, top 50% use SBA**.

### Questions to Answer

1. **Derive the exact training overhead formula** for a hybrid model where `f` fraction of layers use SBA (with overhead factor `r` per SBA attention block) and the rest use softmax. Account for the fact that attention is only a fraction of each layer's total compute (the MLP and layer norms are identical).

   For our 125M model:
   - 12 layers total
   - Each layer: attention (~1/3 of FLOPs) + MLP (~2/3 of FLOPs)
   - SBA attention overhead: 1.52× (current doubled-KV) or ~1.15× (with fused kernel)
   - Embedding/output projection: identical

2. **What is the predicted wall-clock overhead** for:
   - f=0.5 (6/12 layers SBA) with doubled-KV (1.52× attention overhead)?
   - f=0.5 with fused kernel (1.15× attention overhead)?
   - f=0.25 (3/12 layers SBA) with doubled-KV?

3. **Does the hybrid approach change the iso-clock calculus?** Our current iso-clock test showed softmax winning at 22,760 steps vs SBA at 15,000 steps. If hybrid SBA only adds ~1.10× overhead instead of 1.52×, does SBA win per-dollar?

---

## TASK 3: MATHEMATICAL AUDIT

Verify or refute these claims. For each, state: CORRECT, INCORRECT (with explanation), or UNVERIFIABLE.

### Claim A: L1 Bound
We claim: For SBA weights A_ij = sinh(S_ij) / Σ_k cosh(S_ik):
$$\sum_{j=1}^{T} |A_{ij}| \leq 1$$

**Proof sketch**:
$$\sum_j |A_{ij}| = \frac{\sum_j |\sinh(S_{ij})|}{\sum_k \cosh(S_{ik})} \leq \frac{\sum_j \cosh(S_{ij})}{\sum_k \cosh(S_{ik})} = 1$$

using |sinh(x)| ≤ cosh(x) for all real x.

Is this proof correct? When is equality achieved?

### Claim B: Doubled-KV Isomorphism
We claim that:
```
softmax(Q @ [K, -K]^T / √d) @ [V, -V] = SBA(Q, K, V)
```

Specifically, for the i-th query:
$$Y_i = \sum_{j=1}^{T} \frac{e^{S_{ij}} \cdot v_j + e^{-S_{ij}} \cdot (-v_j)}{\sum_{k=1}^{T} (e^{S_{ik}} + e^{-S_{ik}})} = \sum_{j=1}^{T} \frac{(e^{S_{ij}} - e^{-S_{ij}}) v_j}{\sum_{k=1}^{T} (e^{S_{ik}} + e^{-S_{ik}})} = \text{SBA}(Q, K, V)_i$$

Is this proof correct? Are there edge cases where it breaks (e.g., causal masking, numerical precision)?

### Claim C: Smoothness Advantage over Cog Attention
We claim SBA has an advantage over Cog Attention (Lv et al., 2024) because:
- Cog Attention uses: A_ij = sign(S_ij) · exp(|S_ij|) / Σ_k exp(|S_ik|)
- The sign() function creates a step discontinuity at S=0
- SBA's sinh/cosh formulation passes smoothly through zero

Is this characterization correct? Specifically:
1. Does Cog Attention actually have a discontinuity in the forward pass at S=0, or only in the gradient?
2. Is SBA's sinh(S)/Σcosh(S) truly C∞ (infinitely differentiable) at S=0?
3. Does this smoothness difference matter in practice for training stability, or is it a theoretical distinction without practical impact?

### Claim D: BOS Sink Defense
The Integral Transformer (Kobyzev et al., EMNLP 2025) critiques signed attention: they show 50% of Cog and 41% of Diff attention heads assign negative weights to the [BOS] token, suppressing "attention sinks."

We argue SBA mitigates this because:
1. Hyperbolic smoothness allows near-zero negative weights rather than binary sign flips
2. A hybrid layer-wise architecture (softmax in bottom layers) preserves BOS anchors

**Is this defense sound?** Specifically:
- Does SBA's continuous formulation actually prevent BOS suppression, or does gradient descent still converge to strongly negative BOS weights?
- Is there a mathematical reason to expect SBA to behave differently from Cog/Diff attention w.r.t. special token attention patterns?

---

## TASK 4: QUANTIZATION ANALYSIS

We have claimed that SBA's L1-bounded, signed attention weights may benefit low-bit quantization (INT4/INT8) by reducing activation outliers.

### Questions

1. In standard softmax attention, the output Y_i is a convex combination of value vectors (weights sum to 1, all non-negative). In SBA, Y_i is a signed combination with |weights| summing to ≤1. **Does this reduce the dynamic range of the output activations?** Derive or bound the ratio of ||Y_softmax||/||Y_SBA|| under reasonable assumptions.

2. Softmax attention is known to produce "activation outliers" — large-magnitude hidden states that cause quantization difficulties (see SmoothQuant, Dettmers et al.). **Does SBA's signed cancellation reduce or increase the likelihood of outlier activations?** Provide a theoretical argument.

3. For KV-cache quantization during inference: SBA doubles the KV-cache size (storing [K,-K] and [V,-V]). **Is there a way to quantize the SBA KV-cache at half precision without doubling storage?** (Hint: you only need to store K and V, then negate at runtime.)

---

## TASK 5: BACKWARD PASS JACOBIAN DERIVATION

For the Triton kernel backward pass, derive the exact gradients dQ, dK, dV given dO (gradient of output Y w.r.t. loss).

### Setup

Let:
```
S_ij = q_i · k_j * scale                    (logits)
e_pos_ij = exp(S_ij - m_i)                  (positive exponential)
e_neg_ij = exp(-S_ij - m_i)                 (negative exponential)
D_i = Σ_k (e_pos_ik + e_neg_ik)             (denominator, scalar per row)
A_ij = (e_pos_ij - e_neg_ij) / D_i          (SBA weight)
Y_i = Σ_j A_ij · v_j                        (output)
```

### Derive

1. ∂L/∂V (should be straightforward: A^T @ dO)

2. ∂L/∂A_ij (intermediate: dO @ V^T)

3. ∂A_ij/∂S_ij (the SBA Jacobian — this is the key difference from softmax)
   - Diagonal term: ∂A_ij/∂S_ij
   - Off-diagonal term: ∂A_ij/∂S_ik for k ≠ j
   - Express in terms of A_ij and the cosh/sinh values we already have

4. ∂L/∂S (chain rule combining 2 and 3)

5. ∂L/∂Q and ∂L/∂K (from ∂L/∂S, since S = Q @ K^T * scale)

Express the final results in a form suitable for tiled computation (similar to FlashAttention-2's backward pass), using only quantities available during the forward pass (A_ij, e_pos, e_neg, D_i).

---

## OUTPUT FORMAT

Please structure your response as:

### Task 1: Triton Kernel
```python
# Complete sba_triton_kernel.py file
```

### Task 2: Hybrid-Layer Analysis
- Overhead formula derivation
- Numerical predictions for our model
- Iso-clock implications

### Task 3: Mathematical Audit
- Claim A: [CORRECT/INCORRECT] — explanation
- Claim B: [CORRECT/INCORRECT] — explanation
- Claim C: [CORRECT/INCORRECT] — explanation
- Claim D: [CORRECT/INCORRECT/PARTIALLY CORRECT] — explanation

### Task 4: Quantization Analysis
- Answers to questions 1-3

### Task 5: Backward Pass
- Complete Jacobian derivation
- Tiled computation form
