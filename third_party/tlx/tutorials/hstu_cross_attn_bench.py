# Self-contained HSTU cross-attention (jagged, masked) forward + backward Triton
# kernels with a benchmark. No fbcode / hammer / generative_recommenders deps --
# only torch, triton, triton.language, and triton.tools.tensor_descriptor.
#
# HSTU cross-attention is a ragged/jagged attention where Q and K/V each have
# their own per-batch sequence offsets (cross attention => different lengths).
# Tensors are laid out as [total_tokens, H, D] with per-batch segments given by
# the offsets. For each (batch z, head h):
#   S = alpha * (Q @ K^T)               # [Lq, Lkv], with validity mask
#   SiLU variant   (SOFTMAX=False):  P = silu(S) * mask ;  O = (P @ V) * attn_scale
#   Softmax variant(SOFTMAX=True):   P = softmax_kv(masked S) ;  O = P @ V
#
# This is an all-or-nothing softmax: either every head is softmax or every head
# is silu, selected by a single constexpr SOFTMAX so there is no runtime
# scf.if/else in the activation branch (matters for auto warp-specialization).

import os
from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# small host helpers
# ---------------------------------------------------------------------------
def _next_power_of_2(n: int) -> int:
    n = int(n)
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def _alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


# dq is reduced across KV blocks. TMA store_reduce="add" is the fast path; an
# atomic_add fallback is kept for environments where TMA reduce is unavailable.
USE_TMA_DQ = os.environ.get("HSTU_TMA_DQ", "1") == "1"


# ---------------------------------------------------------------------------
# device helpers (inlined, no external deps)
# ---------------------------------------------------------------------------
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _silu(x):
    # silu(x) = x * sigmoid(x)
    return x / (1.0 + tl.exp(-x))


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


# ---------------------------------------------------------------------------
# FORWARD
# ---------------------------------------------------------------------------
def _get_fwd_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=ns, num_warps=nw)
        for bm in [32, 64, 128]
        for bn in [32, 64]
        for ns in [2, 3]
        for nw in [4]
    ]


@triton.autotune(configs=_get_fwd_configs(), key=["H", "DimQ", "DimV", "AUTOTUNE_MAX_Q_LEN", "SOFTMAX"])
@triton.jit
def _hstu_cross_attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    M,  # logsumexp (log2) for softmax heads, [total_seq_len_q, H]; unused for silu
    seq_offsets_q,
    seq_offsets,
    alpha,
    attn_scale,  # scalar tensor (silu output scale, e.g. 1/max_seq_len)
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    stride_mm,
    H,
    AUTOTUNE_MAX_Q_LEN: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    SOFTMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = seq_end_q - seq_start_q
    if start_m >= seq_len_q:
        return
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = seq_end_kv - seq_start_kv

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, DimQ)
    offs_v_d = tl.arange(0, DimV)
    mask_m = offs_m < seq_len_q

    q_ptrs = (
        Q
        + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_qm
        + (off_h * stride_qh + offs_qk_d)[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    acc = tl.zeros([BLOCK_M, DimV], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    scale = tl.load(attn_scale).to(tl.float32)

    for start_n in tl.range(0, seq_len_kv, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len_kv
        k_ptrs = (
            K
            + (seq_start_kv + offs_n).to(tl.int64)[:, None] * stride_kn
            + (off_h * stride_kh + offs_qk_d)[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_ptrs = (
            V
            + (seq_start_kv + offs_n).to(tl.int64)[:, None] * stride_vn
            + (off_h * stride_vh + offs_v_d)[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k))
        valid_mask = mask_m[:, None] & mask_n[None, :]
        if SOFTMAX:
            qk = qk * (alpha * _LOG2E)
            qk = tl.where(valid_mask, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
            p = tl.math.exp2(qk)
            corr = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            acc = acc * corr[:, None]
            l_i = l_i * corr + l_ij
            m_i = m_ij
            acc += tl.dot(p.to(v.dtype), v)
        else:
            masked_alpha = tl.where(valid_mask, alpha, 0.0)
            qk = qk * masked_alpha
            p = _silu(qk)
            acc += tl.dot(p.to(v.dtype), v)

    if SOFTMAX:
        acc = acc / l_i[:, None]
        m_final = m_i + tl.math.log2(l_i)
        m_ptrs = M + (seq_start_q + offs_m) * stride_mm + off_h
        tl.store(m_ptrs, m_final, mask=mask_m)
    else:
        acc = acc * scale

    o_ptrs = (
        Out
        + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_om
        + (off_h * stride_oh + offs_v_d)[None, :]
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None])


def hstu_cross_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    alpha: float,
    attn_scale: torch.Tensor,
    max_q_len: int,
    num_softmax_heads: int,
):
    q = _switch_to_contiguous_if_needed(q)
    k = _switch_to_contiguous_if_needed(k)
    v = _switch_to_contiguous_if_needed(v)
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    SOFTMAX = num_softmax_heads > 0
    if SOFTMAX:
        assert num_softmax_heads == H, "all-or-nothing softmax expected"
    out = torch.empty((total_seq_len_q, H, DimV), device=q.device, dtype=q.dtype)
    if SOFTMAX:
        M = torch.empty((total_seq_len_q, H), device=q.device, dtype=torch.float32)
        stride_mm = M.stride(0)
    else:
        M = torch.empty(1, device=q.device, dtype=torch.float32)
        stride_mm = 0
    if total_seq_len_q == 0:
        return out, M, stride_mm

    Z = seq_offsets.numel() - 1
    grid = lambda meta: (triton.cdiv(max_q_len, meta["BLOCK_M"]), Z * H)  # noqa: E731
    _hstu_cross_attn_fwd_kernel[grid](
        Q=q,
        K=k,
        V=v,
        Out=out,
        M=M,
        seq_offsets_q=seq_offsets_q,
        seq_offsets=seq_offsets,
        alpha=alpha,
        attn_scale=attn_scale,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        stride_mm=stride_mm,
        H=H,
        AUTOTUNE_MAX_Q_LEN=_next_power_of_2(max_q_len),
        DimQ=DimQ,
        DimV=DimV,
        SOFTMAX=SOFTMAX,
    )
    return out, M, stride_mm


# ---------------------------------------------------------------------------
# BACKWARD preprocess (Delta = rowsum(O * dO)) for softmax heads
# ---------------------------------------------------------------------------
@triton.jit
def _bwd_preprocess_kernel(
    Out,
    DOut,
    Delta,
    total_seq_len_q,
    H,
    stride_om,
    stride_oh,
    DimV: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    off_h = tl.program_id(1)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < total_seq_len_q
    offs_d = tl.arange(0, DimV)
    o_ptrs = Out + offs_m[:, None] * stride_om + (off_h * stride_oh + offs_d)[None, :]
    do_ptrs = DOut + offs_m[:, None] * stride_om + (off_h * stride_oh + offs_d)[None, :]
    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + offs_m * H + off_h, delta, mask=mask_m)


# ---------------------------------------------------------------------------
# BACKWARD (reduce_dq): grid one program per (z, h). Outer KV loop, inner Q loop.
# dk/dv accumulated in-loop and stored once per KV block; dq via TMA store_reduce
# add. A single constexpr SOFTMAX picks the activation branch -> no runtime
# scf.if/else. WARP_SPECIALIZE puts warp_specialize on the inner Q loop.
# ---------------------------------------------------------------------------
def _bwd_redq_pre_hook(nargs):
    # dq is reduced (add) across KV blocks, so it MUST be zeroed before every
    # (autotuned) launch; otherwise repeated autotune trials accumulate.
    nargs["DQ"].zero_()


def _get_bwd_configs():
    return [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn},
            num_stages=ns,
            num_warps=nw,
            pre_hook=_bwd_redq_pre_hook,
        )
        for bm in [32, 64]
        for bn in [64, 128]
        for ns in [2, 3]
        for nw in [4]
        if not (bn == 128 and ns == 3)
    ]


@triton.autotune(configs=_get_bwd_configs(), key=["H", "DimQ", "DimV", "AUTOTUNE_MAX_Q_LEN", "SOFTMAX"])
@triton.jit
def _hstu_cross_attn_bwd_redq_kernel(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    DQ,
    M,
    Delta,
    seq_offsets_q,
    seq_offsets,
    total_seq_len_q,
    total_seq_len_kv,
    alpha,
    attn_scale,  # scalar tensor (silu output scale)
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    stride_mm,
    H,
    AUTOTUNE_MAX_Q_LEN: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    SOFTMAX: tl.constexpr,
    SHARED_KV: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    USE_TMA_DQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = seq_end_kv - seq_start_kv
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = seq_end_q - seq_start_q
    if seq_len_kv == 0 or seq_len_q == 0:
        return

    if USE_TMA_DQ:
        desc_dq = tl.make_tensor_descriptor(
            DQ,
            shape=[(seq_start_q + seq_len_q).to(tl.int32), DimQ * H],
            strides=[DimQ * H, 1],
            block_shape=[BLOCK_M, DimQ],
        )

    offs_qk_d = tl.arange(0, DimQ)
    offs_v_d = tl.arange(0, DimV)
    scale = tl.load(attn_scale).to(tl.float32)

    M_off = M + off_h + seq_start_q * stride_mm
    Delta_off = Delta + off_h + seq_start_q * stride_mm

    for start_n in tl.range(0, seq_len_kv, BLOCK_N, num_stages=1):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len_kv
        k_ptrs = (
            K
            + (seq_start_kv + offs_n).to(tl.int64)[:, None] * stride_kn
            + (off_h * stride_kh + offs_qk_d)[None, :]
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        if SHARED_KV:
            v = k
        else:
            v_ptrs = (
                V
                + (seq_start_kv + offs_n).to(tl.int64)[:, None] * stride_vn
                + (off_h * stride_vh + offs_v_d)[None, :]
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
        if not SHARED_KV:
            dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)

        for start_m in tl.range(0, seq_len_q, BLOCK_M, num_stages=1, warp_specialize=WARP_SPECIALIZE):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < seq_len_q
            q_offset = (seq_start_q + start_m).to(tl.int32)
            q_ptrs = (
                Q
                + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_qm
                + (off_h * stride_qh + offs_qk_d)[None, :]
            )
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            do_ptrs = (
                DO
                + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_dom
                + (off_h * stride_doh + offs_v_d)[None, :]
            )
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

            valid_mask_trans = mask_n[:, None] & mask_m[None, :]  # [BLOCK_N, BLOCK_M]
            qk_trans = tl.dot(k, tl.trans(q))  # [BLOCK_N, BLOCK_M]

            if SOFTMAX:
                qk_trans = qk_trans * (alpha * _LOG2E)
                m = tl.load(M_off + offs_m * stride_mm, mask=mask_m, other=0.0)
                pT = tl.math.exp2(qk_trans - m[None, :])
                pT = tl.where(valid_mask_trans, pT, 0.0)
                act_qk_trans = pT.to(k.dtype)
            else:
                masked_alpha = tl.where(valid_mask_trans, alpha, 0.0)
                qk_trans = qk_trans * masked_alpha
                sigT = _sigmoid(qk_trans)
                siluT = qk_trans * sigT * scale
                pT = sigT
                act_qk_trans = siluT.to(k.dtype)

            # accumulate dv (or fold into dk when shared_kv)
            if SHARED_KV:
                dk += tl.dot(act_qk_trans, do)
            else:
                dv += tl.dot(act_qk_trans, do)

            dact_qk_trans = tl.dot(v, tl.trans(do))  # [BLOCK_N, BLOCK_M]
            if SOFTMAX:
                Di = tl.load(Delta_off + offs_m * stride_mm, mask=mask_m, other=0.0)
                dqk_trans = pT * (dact_qk_trans - Di[None, :])
            else:
                dqk_trans = dact_qk_trans * pT * (1 + qk_trans * (1 - pT)) * scale
                dqk_trans = tl.where(valid_mask_trans, dqk_trans, 0.0)

            dqk_trans = dqk_trans.to(k.dtype)
            dk += tl.dot(dqk_trans, q) * alpha
            dq_trans = tl.dot(tl.trans(k), dqk_trans) * alpha  # [DimQ, BLOCK_M]
            dq = tl.trans(dq_trans)  # [BLOCK_M, DimQ]
            if USE_TMA_DQ:
                desc_dq.store([q_offset, DimQ * off_h], dq.to(desc_dq.dtype), store_reduce="add")
            else:
                dq_ptrs = (
                    DQ
                    + (seq_start_q + offs_m).to(tl.int64)[:, None] * (DimQ * H)
                    + (off_h * DimQ + offs_qk_d)[None, :]
                )
                tl.atomic_add(dq_ptrs, dq, mask=mask_m[:, None], sem="relaxed")

        DK_off = (
            DK
            + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64)
            + off_h * tl.cast(stride_dkh, tl.int64)
        )
        dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
        tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=mask_n[:, None])
        if not SHARED_KV:
            DV_off = (
                DV
                + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64)
                + off_h * tl.cast(stride_dvh, tl.int64)
            )
            dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
            tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=mask_n[:, None])


def hstu_cross_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    out: torch.Tensor,
    M: torch.Tensor,
    stride_mm: int,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    alpha: float,
    attn_scale: torch.Tensor,
    max_q_len: int,
    num_softmax_heads: int,
    shared_kv: bool,
    warp_specialize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = _switch_to_contiguous_if_needed(q)
    k = _switch_to_contiguous_if_needed(k)
    v = _switch_to_contiguous_if_needed(v)
    do = _switch_to_contiguous_if_needed(do)
    total_seq_len_q, H, DimQ = q.shape
    total_seq_len_kv, _, DimV = v.shape
    SOFTMAX = num_softmax_heads > 0
    if SOFTMAX:
        assert num_softmax_heads == H, "all-or-nothing softmax expected"

    triton.set_allocator(_alloc_fn)

    # dq is float32 because TMA store_reduce="add" accumulates across KV blocks.
    dq = torch.zeros((total_seq_len_q, H, DimQ), device=q.device, dtype=torch.float32)
    dk = torch.empty((total_seq_len_kv, H, DimQ), device=k.device, dtype=k.dtype)
    if shared_kv:
        dv = dk
    else:
        dv = torch.empty((total_seq_len_kv, H, DimV), device=v.device, dtype=v.dtype)

    if total_seq_len_q == 0:
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    if SOFTMAX:
        Delta = torch.empty((total_seq_len_q, H), device=q.device, dtype=torch.float32)
        pre_grid = (triton.cdiv(total_seq_len_q, 128), H)
        _bwd_preprocess_kernel[pre_grid](
            Out=out,
            DOut=do,
            Delta=Delta,
            total_seq_len_q=total_seq_len_q,
            H=H,
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            DimV=DimV,
            BLOCK_M=128,
        )
    else:
        Delta = torch.empty(1, device=q.device, dtype=torch.float32)

    Z = seq_offsets.numel() - 1
    grid = lambda meta: (Z * H,)  # noqa: E731
    _hstu_cross_attn_bwd_redq_kernel[grid](
        Q=q,
        K=k,
        V=v,
        DO=do,
        DK=dk,
        DV=dv,
        DQ=dq,
        M=M,
        Delta=Delta,
        seq_offsets_q=seq_offsets_q,
        seq_offsets=seq_offsets,
        total_seq_len_q=total_seq_len_q,
        total_seq_len_kv=total_seq_len_kv,
        alpha=alpha,
        attn_scale=attn_scale,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=do.stride(0),
        stride_doh=do.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        stride_mm=stride_mm,
        H=H,
        AUTOTUNE_MAX_Q_LEN=_next_power_of_2(max_q_len),
        DimQ=DimQ,
        DimV=DimV,
        SOFTMAX=SOFTMAX,
        SHARED_KV=shared_kv,
        WARP_SPECIALIZE=warp_specialize,
        USE_TMA_DQ=USE_TMA_DQ,
    )
    dq = dq.to(q.dtype)
    if shared_kv:
        # dv is folded into dk; return a shape-valid zero for the v-arg slot so
        # that when k and v are the same tensor autograd's grad sum stays correct.
        return dq, dk, torch.zeros_like(dk)
    return dq, dk, dv


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------
class HSTUCrossAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        alpha,
        attn_scale,
        max_q_len,
        num_softmax_heads,
        shared_kv,
        warp_specialize,
    ):
        out, M, stride_mm = hstu_cross_attn_fwd(
            q, k, v, seq_offsets, seq_offsets_q, alpha, attn_scale, max_q_len, num_softmax_heads
        )
        ctx.save_for_backward(q, k, v, seq_offsets, seq_offsets_q, attn_scale, out, M)
        ctx.alpha = alpha
        ctx.max_q_len = max_q_len
        ctx.num_softmax_heads = num_softmax_heads
        ctx.shared_kv = shared_kv
        ctx.warp_specialize = warp_specialize
        ctx.stride_mm = stride_mm
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, seq_offsets, seq_offsets_q, attn_scale, out, M = ctx.saved_tensors
        dq, dk, dv = hstu_cross_attn_bwd(
            q,
            k,
            v,
            do,
            out,
            M,
            ctx.stride_mm,
            seq_offsets,
            seq_offsets_q,
            ctx.alpha,
            attn_scale,
            ctx.max_q_len,
            ctx.num_softmax_heads,
            ctx.shared_kv,
            ctx.warp_specialize,
        )
        return (dq, dk, dv, None, None, None, None, None, None, None, None)


def hstu_cross_attn(
    q,
    k,
    v,
    seq_offsets,
    seq_offsets_q,
    alpha,
    attn_scale,
    max_q_len,
    num_softmax_heads=0,
    shared_kv=False,
    warp_specialize=False,
):
    return HSTUCrossAttnFunction.apply(
        q,
        k,
        v,
        seq_offsets,
        seq_offsets_q,
        alpha,
        attn_scale,
        max_q_len,
        num_softmax_heads,
        shared_kv,
        warp_specialize,
    )


# ---------------------------------------------------------------------------
# torch reference (jagged loop). Returns out; supports autograd for dq/dk/dv.
# ---------------------------------------------------------------------------
def _ref_fwd(q, k, v, seq_offsets, seq_offsets_q, alpha, attn_scale, num_softmax_heads):
    H = q.shape[1]
    DimV = v.shape[2]
    SOFTMAX = num_softmax_heads > 0
    out = torch.zeros((q.shape[0], H, DimV), device=q.device, dtype=torch.float32)
    Z = seq_offsets.numel() - 1
    qf, kf, vf = q.float(), k.float(), v.float()
    for z in range(Z):
        qs, qe = int(seq_offsets_q[z]), int(seq_offsets_q[z + 1])
        ks, ke = int(seq_offsets[z]), int(seq_offsets[z + 1])
        if qe == qs or ke == ks:
            continue
        for h in range(H):
            qh = qf[qs:qe, h]  # [Lq, D]
            kh = kf[ks:ke, h]  # [Lkv, D]
            vh = vf[ks:ke, h]  # [Lkv, Dv]
            s = alpha * (qh @ kh.t())  # [Lq, Lkv]
            if SOFTMAX:
                p = torch.softmax(s, dim=-1)
                o = p @ vh
            else:
                p = torch.sigmoid(s) * s  # silu(s)
                o = (p @ vh) * float(attn_scale)
            out[qs:qe, h] = o
    return out


# ---------------------------------------------------------------------------
# input generation
# ---------------------------------------------------------------------------
def _gen_inputs(Z, H, D, max_kv, max_q, dtype, device, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    lengths_kv = torch.randint(1, max_kv + 1, (Z,), device=device)
    lengths_q = torch.randint(1, max_q + 1, (Z,), device=device)
    seq_offsets = torch.zeros(Z + 1, dtype=torch.int64, device=device)
    seq_offsets[1:] = torch.cumsum(lengths_kv, 0)
    seq_offsets_q = torch.zeros(Z + 1, dtype=torch.int64, device=device)
    seq_offsets_q[1:] = torch.cumsum(lengths_q, 0)
    total_q = int(seq_offsets_q[-1].item())
    total_kv = int(seq_offsets[-1].item())
    q = torch.empty((total_q, H, D), dtype=dtype, device=device).uniform_(-0.1, 0.1).requires_grad_(requires_grad)
    k = torch.empty((total_kv, H, D), dtype=dtype, device=device).uniform_(-0.1, 0.1).requires_grad_(requires_grad)
    v = torch.empty((total_kv, H, D), dtype=dtype, device=device).uniform_(-0.1, 0.1).requires_grad_(requires_grad)
    return q, k, v, seq_offsets, seq_offsets_q, max_kv, max_q


# ---------------------------------------------------------------------------
# correctness + benchmark
# ---------------------------------------------------------------------------
def _check_correctness(num_softmax_heads, warp_specialize, shared_kv=False):
    device = "cuda"
    dtype = torch.bfloat16
    Z, H, D = 4, 2, 128
    max_kv, max_q = 256, 256
    q, k, v, seq_offsets, seq_offsets_q, max_kv, max_q = _gen_inputs(
        Z, H, D, max_kv, max_q, dtype, device, seed=123
    )
    alpha = 1.0 / D
    max_seq_len = max_kv
    attn_scale = torch.tensor(1.0 / max_seq_len, device=device, dtype=torch.float32)

    # reference (double-backward via autograd on float reference). With shared_kv,
    # V aliases K so there is a single kv leaf whose grad = dk + dv.
    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    if shared_kv:
        vr = kr
    else:
        vr = v.detach().clone().requires_grad_(True)
    ref_out = _ref_fwd(qr, kr, vr, seq_offsets, seq_offsets_q, alpha, attn_scale, num_softmax_heads)
    do = torch.randn_like(ref_out) * 0.1
    ref_out.backward(do)
    ref_dq, ref_dk, ref_dv = qr.grad, kr.grad, (None if shared_kv else vr.grad)

    # triton
    qt = q.detach().clone().requires_grad_(True)
    kt = k.detach().clone().requires_grad_(True)
    if shared_kv:
        vt = kt
    else:
        vt = v.detach().clone().requires_grad_(True)
    tri_out = hstu_cross_attn(
        qt, kt, vt, seq_offsets, seq_offsets_q, alpha, attn_scale, max_q,
        num_softmax_heads=num_softmax_heads, shared_kv=shared_kv, warp_specialize=warp_specialize,
    )
    tri_out.backward(do.to(tri_out.dtype))

    tag = ("softmax" if num_softmax_heads > 0 else "silu") + (" ws" if warp_specialize else "") + (" shared_kv" if shared_kv else "")

    def rel_l2(a, b):
        a, b = a.float(), b.float()
        return ((a - b).norm() / (b.norm() + 1e-12)).item()

    # bf16 jagged attention: per-element rtol/atol is noisy on near-zero grad
    # entries, so validate with a relative-L2 (Frobenius) error, the standard
    # check for low-precision attention kernels.
    TOL = 2e-2
    checks = [("out", tri_out, ref_out), ("dq", qt.grad, ref_dq), ("dk", kt.grad, ref_dk)]
    if not shared_kv:
        checks.append(("dv", vt.grad, ref_dv))
    stats = []
    for name, a, b in checks:
        e = rel_l2(a, b)
        stats.append(f"{name}={e:.2e}")
        assert e < TOL, f"{tag} {name} rel-L2 {e:.3e} >= {TOL}"
    print(f"[OK] correctness ({tag}): rel-L2 vs torch ref  " + "  ".join(stats))


def bench():
    device = "cuda"
    dtype = torch.bfloat16
    Z, H, D = 256, 2, 128
    max_kv, max_q = 1024, 256
    for num_softmax_heads in [0, H]:
        q, k, v, seq_offsets, seq_offsets_q, max_kv_, max_q_ = _gen_inputs(
            Z, H, D, max_kv, max_q, dtype, device, requires_grad=True, seed=7
        )
        alpha = 1.0 / D
        attn_scale = torch.tensor(1.0 / max_kv, device=device, dtype=torch.float32)
        tag = "softmax" if num_softmax_heads > 0 else "silu"

        fwd_fn = lambda: hstu_cross_attn(  # noqa: E731
            q, k, v, seq_offsets, seq_offsets_q, alpha, attn_scale, max_q,
            num_softmax_heads=num_softmax_heads, warp_specialize=False,
        )
        fwd_ms = triton.testing.do_bench(fwd_fn, warmup=25, rep=100)
        o = fwd_fn()
        do = torch.randn_like(o)
        bwd_fn = lambda: o.backward(do, retain_graph=True)  # noqa: E731
        bwd_ms = triton.testing.do_bench(bwd_fn, warmup=25, rep=100)
        print(f"[bench {tag}] Z={Z} H={H} D={D} max_kv={max_kv} max_q={max_q}: "
              f"fwd={fwd_ms:.3f} ms  bwd(redq)={bwd_ms:.3f} ms")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("=== correctness (non warp-specialized) ===")
    _check_correctness(num_softmax_heads=0, warp_specialize=False)
    _check_correctness(num_softmax_heads=2, warp_specialize=False)
    _check_correctness(num_softmax_heads=0, warp_specialize=False, shared_kv=True)

    if os.environ.get("HSTU_TEST_WS", "0") == "1":
        print("=== correctness (warp-specialized) ===")
        try:
            _check_correctness(num_softmax_heads=0, warp_specialize=True)
            _check_correctness(num_softmax_heads=2, warp_specialize=True)
        except Exception as e:  # noqa: BLE001
            print(f"[autows] FAILED: {type(e).__name__}: {e}")

    print("=== latency ===")
    bench()
