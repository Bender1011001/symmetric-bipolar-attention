import math
import os

import torch
import triton
import triton.language as tl




_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
]


_BWD_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
]


def _triton_dtype_and_id(dtype: torch.dtype):
    if dtype is torch.float32:
        return tl.float32, 0
    if dtype is torch.float16:
        return tl.float16, 1
    if dtype is torch.bfloat16:
        return tl.bfloat16, 2
    raise TypeError(f"SBA fused attention supports fp32, fp16, and bf16; got {dtype}")


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _block_d(head_dim: int) -> int:
    return max(16, _next_power_of_2(head_dim))


def _normalize_input_precision(input_precision):
    if input_precision is None:
        return "tf32"
    if input_precision not in ("tf32", "tf32x3", "ieee"):
        raise ValueError(
            "input_precision must be one of {'tf32', 'tf32x3', 'ieee'}; "
            f"got {input_precision!r}"
        )
    return input_precision


@triton.autotune(
    configs=_FWD_CONFIGS,
    key=["N_CTX", "HEAD_DIM", "BLOCK_D", "CAUSAL", "DTYPE_ID", "INPUT_PRECISION"],
)
@triton.jit
def _sba_fwd_kernel(
    Q,
    K,
    V,
    Out,
    M,
    L,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    DTYPE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base = off_hz * N_CTX * HEAD_DIM

    q = tl.load(
        Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(DTYPE)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    LOG2_E: tl.constexpr = 1.4426950408889634
    scale_log2 = sm_scale * LOG2_E

    hi_n = N_CTX
    if CAUSAL:
        hi_n = (pid_m + 1) * BLOCK_M
        hi_n = tl.multiple_of(hi_n, BLOCK_M)

    for start_n in range(0, hi_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        k = tl.load(
            K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        v = tl.load(
            V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale_log2

        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        s_pos = tl.where(valid, s, -float("inf"))
        s_neg = tl.where(valid, -s, -float("inf"))

        m_block = tl.maximum(tl.max(s_pos, axis=1), tl.max(s_neg, axis=1))
        m_new = tl.maximum(m_i, m_block)

        alpha = tl.where(m_i == -float("inf"), 0.0, tl.math.exp2(m_i - m_new))

        e_pos = tl.where(valid, tl.math.exp2(s - m_new[:, None]), 0.0)
        e_neg = tl.where(valid, tl.math.exp2(-s - m_new[:, None]), 0.0)

        p = e_pos - e_neg
        c = e_pos + e_neg

        l_i = l_i * alpha + tl.sum(c, axis=1)
        acc = acc * alpha[:, None] + tl.dot(
            p.to(DTYPE),
            v,
            input_precision=INPUT_PRECISION,
        )
        m_i = m_new

    l_safe = tl.maximum(l_i, 1.0e-30)
    out = acc / l_safe[:, None]

    tl.store(
        Out + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )

    tl.store(M + off_hz * N_CTX + offs_m, m_i, mask=offs_m < N_CTX)
    tl.store(L + off_hz * N_CTX + offs_m, l_safe, mask=offs_m < N_CTX)


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_CTX", "HEAD_DIM", "BLOCK_D", "CAUSAL", "DTYPE_ID", "INPUT_PRECISION"],
)
@triton.jit
def _sba_bwd_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    M,
    L,
    Delta,
    DK,
    DV,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    DTYPE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_n = tl.program_id(0)
    off_hz = tl.program_id(1)

    start_n = pid_n * BLOCK_N
    start_n = tl.multiple_of(start_n, BLOCK_N)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base = off_hz * N_CTX * HEAD_DIM

    k = tl.load(
        K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(DTYPE)

    v = tl.load(
        V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(DTYPE)

    dk_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv_acc = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    LOG2_E: tl.constexpr = 1.4426950408889634
    scale_log2 = sm_scale * LOG2_E

    lo_m = 0
    if CAUSAL:
        lo_m = (start_n // BLOCK_M) * BLOCK_M
        lo_m = tl.multiple_of(lo_m, BLOCK_M)

    for start_m in range(lo_m, N_CTX, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(
            Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        do = tl.load(
            DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        m_i = tl.load(
            M + off_hz * N_CTX + offs_m,
            mask=offs_m < N_CTX,
            other=-float("inf"),
        )
        l_i = tl.load(
            L + off_hz * N_CTX + offs_m,
            mask=offs_m < N_CTX,
            other=1.0,
        )
        delta_i = tl.load(
            Delta + off_hz * N_CTX + offs_m,
            mask=offs_m < N_CTX,
            other=0.0,
        )

        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale_log2

        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        e_pos = tl.where(valid, tl.math.exp2(s - m_i[:, None]), 0.0)
        e_neg = tl.where(valid, tl.math.exp2(-s - m_i[:, None]), 0.0)

        inv_l = 1.0 / tl.maximum(l_i, 1.0e-30)
        a = (e_pos - e_neg) * inv_l[:, None]
        c = (e_pos + e_neg) * inv_l[:, None]

        dv_acc += tl.dot(tl.trans(a.to(DTYPE)), do, input_precision=INPUT_PRECISION)

        d_p = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)

        d_s = (d_p * c - a * delta_i[:, None]) * sm_scale

        dk_acc += tl.dot(tl.trans(d_s.to(DTYPE)), q, input_precision=INPUT_PRECISION)

    tl.store(
        DK + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        dk_acc,
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )
    tl.store(
        DV + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
        dv_acc,
        mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["N_CTX", "HEAD_DIM", "BLOCK_D", "CAUSAL", "DTYPE_ID", "INPUT_PRECISION"],
)
@triton.jit
def _sba_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    M,
    L,
    Delta,
    DQ,
    sm_scale,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    DTYPE: tl.constexpr,
    DTYPE_ID: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_m = tl.multiple_of(start_m, BLOCK_M)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base = off_hz * N_CTX * HEAD_DIM

    q = tl.load(
        Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(DTYPE)

    do = tl.load(
        DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    ).to(DTYPE)

    m_i = tl.load(
        M + off_hz * N_CTX + offs_m,
        mask=offs_m < N_CTX,
        other=-float("inf"),
    )
    l_i = tl.load(
        L + off_hz * N_CTX + offs_m,
        mask=offs_m < N_CTX,
        other=1.0,
    )
    delta_i = tl.load(
        Delta + off_hz * N_CTX + offs_m,
        mask=offs_m < N_CTX,
        other=0.0,
    )

    dq_acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    LOG2_E: tl.constexpr = 1.4426950408889634
    scale_log2 = sm_scale * LOG2_E
    inv_l = 1.0 / tl.maximum(l_i, 1.0e-30)

    hi_n = N_CTX
    if CAUSAL:
        hi_n = (pid_m + 1) * BLOCK_M
        hi_n = tl.multiple_of(hi_n, BLOCK_M)

    for start_n in range(0, hi_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        k = tl.load(
            K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        v = tl.load(
            V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(offs_n[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        ).to(DTYPE)

        s = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION) * scale_log2

        valid = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < N_CTX)
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        e_pos = tl.where(valid, tl.math.exp2(s - m_i[:, None]), 0.0)
        e_neg = tl.where(valid, tl.math.exp2(-s - m_i[:, None]), 0.0)

        a = (e_pos - e_neg) * inv_l[:, None]
        c = (e_pos + e_neg) * inv_l[:, None]

        d_p = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)

        d_s = (d_p * c - a * delta_i[:, None]) * sm_scale

        dq_acc += tl.dot(d_s.to(DTYPE), k, input_precision=INPUT_PRECISION)

    tl.store(
        DQ + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        dq_acc,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


class SBA_Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=True, scale=None, input_precision="tf32"):
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise RuntimeError("SBA_Fused requires CUDA tensors.")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError("q, k, and v must have shape (B, H, T, D).")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical shapes.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise TypeError("q, k, and v must have the same dtype.")
        if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise TypeError(f"Unsupported dtype {q.dtype}; expected fp32, fp16, or bf16.")

        B, H, T, D = q.shape
        if T <= 0:
            raise ValueError("Sequence length T must be positive.")
        if D <= 0 or D > 256:
            raise ValueError("This implementation supports 0 < HEAD_DIM <= 256.")

        if scale is None:
            scale = 1.0 / math.sqrt(D)
        scale = float(scale)
        input_precision = _normalize_input_precision(input_precision)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out = torch.empty_like(q)
        m = torch.empty((B * H, T), device=q.device, dtype=torch.float32)
        l = torch.empty((B * H, T), device=q.device, dtype=torch.float32)

        block_d = _block_d(D)
        dtype, dtype_id = _triton_dtype_and_id(q.dtype)

        grid = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)

        _sba_fwd_kernel[grid](
            q,
            k,
            v,
            out,
            m,
            l,
            scale,
            T,
            D,
            BLOCK_D=block_d,
            CAUSAL=bool(causal),
            DTYPE=dtype,
            DTYPE_ID=dtype_id,
            INPUT_PRECISION=input_precision,
        )

        ctx.save_for_backward(q, k, v, out, m, l)
        ctx.causal = bool(causal)
        ctx.scale = scale
        ctx.block_d = block_d
        ctx.input_precision = input_precision

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, m, l = ctx.saved_tensors
        B, H, T, D = q.shape

        dout = dout.contiguous()
        delta = (dout.float() * out.float()).sum(dim=-1).contiguous().view(B * H, T)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        dtype, dtype_id = _triton_dtype_and_id(q.dtype)

        grid_dkdv = lambda META: (triton.cdiv(T, META["BLOCK_N"]), B * H)
        _sba_bwd_dkdv_kernel[grid_dkdv](
            q,
            k,
            v,
            dout,
            m,
            l,
            delta,
            dk,
            dv,
            ctx.scale,
            T,
            D,
            BLOCK_D=ctx.block_d,
            CAUSAL=ctx.causal,
            DTYPE=dtype,
            DTYPE_ID=dtype_id,
            INPUT_PRECISION=ctx.input_precision,
        )

        grid_dq = lambda META: (triton.cdiv(T, META["BLOCK_M"]), B * H)
        _sba_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            dout,
            m,
            l,
            delta,
            dq,
            ctx.scale,
            T,
            D,
            BLOCK_D=ctx.block_d,
            CAUSAL=ctx.causal,
            DTYPE=dtype,
            DTYPE_ID=dtype_id,
            INPUT_PRECISION=ctx.input_precision,
        )

        return dq, dk, dv, None, None, None


def sba_attention_fused(q, k, v, causal=True, scale=None, input_precision="tf32"):
    return SBA_Fused.apply(q, k, v, causal, scale, input_precision)


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

        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
        weights = torch.where(mask, (e_pos - e_neg) / denom, zeros)
    else:
        row_max = torch.maximum(
            work.max(dim=-1, keepdim=True).values,
            (-work).max(dim=-1, keepdim=True).values,
        )

        e_pos = torch.exp(work - row_max)
        e_neg = torch.exp(-work - row_max)

        denom = (e_pos + e_neg).sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
        weights = (e_pos - e_neg) / denom

    return torch.matmul(weights.to(q.dtype), v)
