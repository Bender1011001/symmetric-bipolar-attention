# SBA Fused Backward Kernel — Triton Implementation Request

You previously wrote a highly successful **forward** Triton kernel for Symmetric Bipolar Attention (SBA). The forward kernel achieves **1.32× softmax** latency at T=1024 on H100 — excellent.

However, the **backward pass is still eager PyTorch**, materializing full T×T attention matrices. This makes training **4.68× slower** than softmax — far worse than even the naive doubled-KV approach (1.96×).

**Your task: write a fused Triton backward kernel for SBA that mirrors FlashAttention-2's backward pass structure.**

---

## THE PROBLEM (why the backward is slow)

The current eager backward does this for EVERY layer (×12), at B=16, H=12, T=1024:

```python
# Step 1: Recompute full T×T logits — 200M float32 elements
logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # (16, 12, 1024, 1024)

# Step 2: Recompute attention weights — another 400M elements  
e_pos = torch.exp(pos_logits - row_max)   # (16, 12, 1024, 1024)
e_neg = torch.exp(neg_logits - row_max)   # (16, 12, 1024, 1024)
A = (e_pos - e_neg) / denom              # (16, 12, 1024, 1024)
C = (e_pos + e_neg) / denom              # (16, 12, 1024, 1024)

# Step 3: Compute gradient intermediates — more T×T matrices
dP = torch.matmul(dout, v.transpose(-1, -2))  # (16, 12, 1024, 1024)
dS = (dP * C - A * Delta) * scale             # (16, 12, 1024, 1024)

# Step 4: Final gradients
dq = torch.matmul(dS, k)      # (16, 12, 1024, 64)
dk = torch.matmul(dS^T, q)    # (16, 12, 1024, 64)
dv = torch.matmul(A^T, dout)  # (16, 12, 1024, 64)
```

Total: ~800M float32 elements per layer in GPU global memory. This is O(T²) memory, destroying performance.

## MEASURED TRAINING STEP TIMES (B=16, T=1024, 125M params, H100)

| Mode | Step Time | vs Softmax |
|:---|:---:|:---:|
| softmax (FlashAttention) | 200.2 ms | 1.00× |
| SBA doubled-KV (SDPA) | 392.3 ms | 1.96× |
| SBA fused fwd + eager bwd | 936.8 ms | **4.68×** |

The forward kernel contributes ~1ms per step. The backward contributes ~735ms. **The backward is 99% of the overhead.**

---

## THE SOLUTION: FlashAttention-2 Style Backward

FlashAttention-2's backward pass avoids materializing the T×T attention matrix by recomputing attention weights tile-by-tile in SRAM during the backward pass. It uses two passes:

### Pass 1: Compute dV and dK (outer loop over KV blocks)
For each KV block, iterate over all Q blocks, recompute the attention tile, and accumulate dV and dK.

### Pass 2: Compute dQ (outer loop over Q blocks)  
For each Q block, iterate over all KV blocks, recompute the attention tile, and accumulate dQ.

### SBA Modification

For standard softmax, the backward computes:
```
dS_ij = P_ij * (dP_ij - Delta_i)    where Delta_i = sum_j(dO_i * Y_i)
```

For SBA, the Jacobian (derived and verified) is:
```
dS_ij = dP_ij * C_ij - A_ij * Delta_i    where Delta_i = dot(dO_i, Y_i)
```

Where:
- `A_ij = (e_pos - e_neg) / D_i` — the SBA attention weight (sinh/cosh normalized)
- `C_ij = (e_pos + e_neg) / D_i` — the auxiliary "cosh fraction"
- `e_pos = exp(S_ij - m_i)`, `e_neg = exp(-S_ij - m_i)`
- `D_i = sum_j(e_pos_ij + e_neg_ij)` — the denominator
- `dP_ij = dO_i · v_j` — gradient w.r.t. attention weights

Then:
```
dV = A^T @ dO        (same structure as softmax)
dK = dS^T @ Q * scale
dQ = dS @ K * scale
```

---

## YOUR WORKING FORWARD KERNEL (for reference)

This is the V2 forward kernel that passes all tests at 1.32× softmax. Use the same tiling structure, autotuning, exp2 optimization, and causal block skipping.

```python
@triton.autotune(configs=get_autotune_configs(), key=['N_CTX'])
@triton.jit
def _sba_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr, IS_CAUSAL: tl.constexpr,
    DTYPE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    o_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_DMODEL), other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    k_ptrs = K + off_hz * stride_kh + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M) if IS_CAUSAL else N_CTX

    LOG2_E: tl.constexpr = 1.4426950408889634
    scale = sm_scale * LOG2_E

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        k_mask = (offs_n_curr[None, :] < N_CTX) & (offs_k[:, None] < BLOCK_DMODEL)
        v_mask = (offs_n_curr[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_DMODEL)

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= scale

        need_mask = (IS_CAUSAL and (start_n + BLOCK_N > start_m * BLOCK_M)) or ((start_n + BLOCK_N) > N_CTX)

        if need_mask:
            valid = offs_n_curr[None, :] < N_CTX
            if IS_CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n_curr[None, :])
            qk_pos = tl.where(valid, qk, -float('inf'))
            qk_neg = tl.where(valid, -qk, -float('inf'))
        else:
            qk_pos = qk
            qk_neg = -qk

        m_ij = tl.maximum(tl.max(qk_pos, 1), tl.max(qk_neg, 1))
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha
        acc = acc * alpha[:, None]

        p_pos = tl.math.exp2(qk_pos - m_new[:, None])
        p_neg = tl.math.exp2(qk_neg - m_new[:, None])

        l_i += tl.sum(p_pos + p_neg, 1)

        weights = (p_pos - p_neg).to(DTYPE)
        acc += tl.dot(weights, v)

        m_i = m_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    denom = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / denom[:, None]
    tl.store(o_ptrs, acc.to(DTYPE), mask=(offs_m[:, None] < N_CTX) & (offs_k[None, :] < BLOCK_DMODEL))
```

---

## FORWARD PASS: WHAT MUST BE SAVED FOR BACKWARD

The forward must save enough information to reconstruct the attention weights tile-by-tile during the backward without materializing the full T×T matrix. Following FlashAttention-2:

Save from forward: `Q, K, V, Out, m_i (row max), l_i (row denominator)`

The per-row `m_i` and `l_i` vectors are (B, H, T) — negligible memory. During backward, each tile recomputes:
```
S_tile = Q_block @ K_block^T * scale
e_pos = exp2(S_tile * log2e - m_i)    # recomputed on-the-fly
e_neg = exp2(-S_tile * log2e - m_i)   # recomputed on-the-fly
A_tile = (e_pos - e_neg) / l_i        # attention weights
C_tile = (e_pos + e_neg) / l_i        # auxiliary
```

**IMPORTANT**: The forward kernel must be modified to also store `m_i` and `l_i` to global memory. Currently it does not.

---

## BACKWARD KERNEL PSEUDOCODE

### Precomputation (in Python, before launching kernels):
```python
# Delta_i = row-wise dot(dO, Y) — (B, H, T) vector, cheap to compute
Delta = (dout * out).sum(dim=-1)  # (B, H, T)
```

### Kernel 1: dV and dK (outer loop over KV blocks)
```python
# Each program handles one KV block
for each KV_block j:
    dv_acc = zeros(BLOCK_N, D)    # accumulator for dV[j]
    dk_acc = zeros(BLOCK_N, D)    # accumulator for dK[j]
    
    k_tile = K[j]                  # (BLOCK_N, D)
    v_tile = V[j]                  # (BLOCK_N, D)
    
    # For causal: only Q blocks where i >= j (block-level check)
    for each Q_block i (where i >= j for causal):
        q_tile = Q[i]              # (BLOCK_M, D)
        do_tile = dO[i]            # (BLOCK_M, D)
        o_tile = Out[i]            # (BLOCK_M, D)
        m_tile = m[i]              # (BLOCK_M,) — saved from forward
        l_tile = l[i]              # (BLOCK_M,) — saved from forward
        delta_tile = Delta[i]      # (BLOCK_M,)
        
        # Recompute attention tile
        S = q_tile @ k_tile^T * scale              # (BLOCK_M, BLOCK_N)
        qk_pos = S * log2e       # (already scaled by log2e in forward)
        qk_neg = -S * log2e
        # Apply causal mask if needed
        
        e_pos = exp2(qk_pos - m_tile[:, None])     # recomputed
        e_neg = exp2(qk_neg - m_tile[:, None])     # recomputed
        A_tile = (e_pos - e_neg) / l_tile[:, None]  # SBA weights
        C_tile = (e_pos + e_neg) / l_tile[:, None]  # auxiliary
        
        # dV contribution: A^T @ dO
        dv_acc += A_tile^T @ do_tile               # (BLOCK_N, D)
        
        # dS for this tile
        dP_tile = do_tile @ v_tile^T               # (BLOCK_M, BLOCK_N)
        dS_tile = (dP_tile * C_tile - A_tile * delta_tile[:, None]) * sm_scale
        
        # dK contribution: dS^T @ Q
        dk_acc += dS_tile^T @ q_tile               # (BLOCK_N, D)
    
    dV[j] = dv_acc
    dK[j] = dk_acc
```

### Kernel 2: dQ (outer loop over Q blocks)
```python
# Each program handles one Q block  
for each Q_block i:
    dq_acc = zeros(BLOCK_M, D)     # accumulator for dQ[i]
    
    q_tile = Q[i]                   # (BLOCK_M, D)
    do_tile = dO[i]                 # (BLOCK_M, D)
    m_tile = m[i]                   # (BLOCK_M,)
    l_tile = l[i]                   # (BLOCK_M,)
    delta_tile = Delta[i]           # (BLOCK_M,)
    
    # For causal: only KV blocks where j <= i (block-level check)
    for each KV_block j (where j <= i for causal):
        k_tile = K[j]              # (BLOCK_N, D)
        v_tile = V[j]              # (BLOCK_N, D)
        
        # Recompute attention tile (same as above)
        S = q_tile @ k_tile^T * scale
        # ... same exp2, masking ...
        
        dP_tile = do_tile @ v_tile^T
        dS_tile = (dP_tile * C_tile - A_tile * delta_tile[:, None]) * sm_scale
        
        # dQ contribution: dS @ K
        dq_acc += dS_tile @ k_tile
    
    dQ[i] = dq_acc
```

---

## IMPORTANT TRITON IMPLEMENTATION NOTES

1. **Use `tl.math.exp2`** with pre-scaled logits (multiply by log2(e)), matching the forward kernel.

2. **The forward kernel must be modified** to write `m_i` and `l_i` to global memory:
   ```python
   # At end of forward kernel, after computing final acc:
   m_ptrs = M + off_hz * N_CTX + offs_m
   l_ptrs = L + off_hz * N_CTX + offs_m
   tl.store(m_ptrs, m_i, mask=offs_m < N_CTX)
   tl.store(l_ptrs, l_i, mask=offs_m < N_CTX)
   ```
   Add `M` and `L` output buffers of shape `(B*H, T)` to the forward.

3. **DTYPE handling**: Use the same `DTYPE: tl.constexpr` pattern as the forward kernel. Cast dot-product operands to DTYPE before `tl.dot`.

4. **Causal block skipping**: In Kernel 1, skip Q blocks where `start_m * BLOCK_M < start_n` (fully above diagonal). In Kernel 2, skip KV blocks where `start_n > (start_m + 1) * BLOCK_M` (fully below query range).

5. **The scale factor**: The forward uses `scale = sm_scale * LOG2_E` for the exp2 path. The backward's `dS` must be multiplied by `sm_scale` (NOT `sm_scale * LOG2_E`), because the Jacobian operates in natural-log space.

6. **Autotuning**: Use the same autotune config pattern as the forward. The backward kernels' BLOCK_M/BLOCK_N don't need to match the forward's.

---

## OUTPUT REQUIREMENTS

### Provide a complete, runnable `sba_triton_kernel.py` that includes:

1. **Modified forward kernel** that also stores `m_i` and `l_i`
2. **Backward kernel 1** (`_sba_bwd_dkdv_kernel`): computes dK and dV  
3. **Backward kernel 2** (`_sba_bwd_dq_kernel`): computes dQ
4. **Updated `SBA_Fused` autograd Function** that:
   - Forward: runs the Triton forward, saves Q, K, V, Out, M, L
   - Backward: precomputes Delta, launches both backward kernels
5. **`sba_attention_fused(q, k, v, causal=True)`** — drop-in API

### Correctness validation:

The backward must match the eager reference to within bf16 tolerance (3e-2 max diff). Test code:

```python
torch.manual_seed(42)
B, H, T, D = 2, 4, 256, 64
q = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
k = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)
v = torch.randn(B, H, T, D, dtype=torch.float32, device='cuda', requires_grad=True)

# Your kernel
y = sba_attention_fused(q, k, v, causal=True)
y.sum().backward()
dq_fused, dk_fused, dv_fused = q.grad.clone(), k.grad.clone(), v.grad.clone()
q.grad, k.grad, v.grad = None, None, None

# Eager reference
y_ref = sba_attention_eager_reference(q, k, v, causal=True)
y_ref.sum().backward()
dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

for name, fused, ref in [('dQ', dq_fused, dq_ref), ('dK', dk_fused, dk_ref), ('dV', dv_fused, dv_ref)]:
    print(f"{name}: max_diff={(fused-ref).abs().max():.2e}")
```

### Performance target:
- Full training step (forward + backward) at B=16, H=12, T=1024, D=64: **≤1.3× softmax**
- Currently: forward is 1.32×, backward is the bottleneck at ~735ms vs softmax's ~150ms

### Eager reference (ground truth for testing):

```python
def sba_attention_eager_reference(q, k, v, causal=True, scale=None):
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
