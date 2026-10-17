"""gfx950 MXFP8 Flash Attention training forward.

Backward needs per-row softmax statistics and a forward whose probabilities
match the ones it recomputes, so training does not use the inference kernels
in gfx950.py (whose ping-pong kernel uses an approximate exp2 and saves no
statistics). Training runs a hardware-exp loop kernel on the same MXFP8 recipe:
Q and K are E4M3 with an E8M0 scale per 32x32 block, V is E4M3 with an E8M0
scale per 32 keys of each head column, and P gets an E8M0 scale per 32 keys of
each row. Quantization shares gfx950_quant's scale rules with backward and
uses 64-bit tensor offsets.

The forward saves separate FP32 row maxima and log2 normalization terms.
Backward uses native scaled MFMA for all five unique products, fixed-scale P
and transpose-compatible 32x32 dS quantization. The general path uses separate
Q and KV owners that recompute QK and dP. For contiguous B4/H32/D128 MHA at
scale 0.5 and N=1024/2048/4096/8192, causal or noncausal, a specialized
backward shares square32 Q/K/dO quantization and materializes FP8 dS for reuse
by the Q owner. It also supports N8192 noncausal at scale 1.3. Each gradient
has a single writer and accumulates deterministically in FP32. Whole, aligned
storage is required. Temporary storage is 128*N*N + 128*(N//32)**2 + 34816*N
bytes, up to 8 GiB + 280 MiB at N8192, excluding gradients and saved inputs.
Causal calls allocate the same dense workspace. Other configurations use the
general implementation; allocation or kernel failures are not retried.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .gfx950_quant import _decode_scale, _rounded_mul, _scale_exponent
from .gfx950_bwd_shared import _saved_qk_arena_segments

_FP8 = torch.float8_e4m3fn
_LOG2E = 1.4426950408889634
_HEAD_DIM = 128


def _default_config(causal, n_ctx):
    # The hardware-exp loop kernel, measured on gfx950 at B=4 H=32 D=128.
    del n_ctx
    if causal:
        return {"block_m": 128, "block_n": 64, "num_warps": 4, "num_stages": 2}
    return {"block_m": 128, "block_n": 128, "num_warps": 4, "num_stages": 2}


@triton.jit
def _mx_scale(amax):
    # Keep the result signed for the subsequent (byte - 127) exponent.
    return _scale_exponent(amax).to(tl.int32)


@triton.jit
def _quantize_mxfp8_kernel(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    # Each 32x32 block of x (32 rows by 32 head elements) gets the E8M0 byte
    # _mx_scale(amax) for its max magnitude amax, stored for each of its rows,
    # and E4M3 data x / 2**(byte - 127). The outputs are read once, by the
    # attention kernel, so the stores stream past the caches.
    # Tensor bases can exceed signed-i32 offsets even when one tile is small.
    pid_bh = tl.program_id(1).to(tl.int64)
    offs_n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_b = tl.arange(0, HEAD_DIM // 32)
    rows = offs_n[:, None] < N_CTX
    x = tl.load(
        X + (pid_bh // H) * stride_xz + (pid_bh % H) * stride_xh + offs_n[:, None] * stride_xn +
        offs_d[None, :] * stride_xd, mask=rows, other=0.0).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM // 32, 32))
    byte = _mx_scale(tl.max(tl.max(tl.abs(x), 3), 1))
    inv = _decode_scale(byte, RECIPROCAL=True)
    q = tl.clamp(x * inv[:, None, :, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], tl.reshape(q, (BLOCK_N, HEAD_DIM)),
             mask=rows, cache_modifier=".cs")
    s = tl.reshape(tl.broadcast_to(byte[:, None, :], (BLOCK_N // 32, 32, HEAD_DIM // 32)), (BLOCK_N, HEAD_DIM // 32))
    tl.store(Scale + (pid_bh * N_CTX + offs_n[:, None]) * (HEAD_DIM // 32) + offs_b[None, :], s.to(tl.uint8), mask=rows,
             cache_modifier=".cs")


def quantize_mxfp8_head(x):
    """E4M3 data and E8M0 scales of a ``[Z, H, N_CTX, HEAD_DIM]`` tensor with
    a scale per 32x32 block (32 rows by 32 head elements). The scales are
    ``[Z, H, N_CTX, HEAD_DIM // 32]``, each row holding its block's."""
    z, h, n, d = x.shape
    quant = torch.empty((z, h, n, d), device=x.device, dtype=_FP8)
    scale = torch.empty((z, h, n, d // 32), device=x.device, dtype=torch.uint8)
    _quantize_mxfp8_kernel[(triton.cdiv(n, 64), z * h)](x, quant, scale, x.stride(0), x.stride(1), x.stride(2),
                                                        x.stride(3), h, n, HEAD_DIM=d, BLOCK_N=64, num_warps=4)
    return quant, scale


@triton.jit
def _quantize_saved_qk_arena(X, Arena, stride_xz, stride_xh, stride_xn, stride_xd, ARENA_N: tl.constexpr,
                             KEY: tl.constexpr):
    q, k, qs, ks = _saved_qk_arena_segments(Arena, ARENA_N)
    if KEY:
        out, scale = k, ks
    else:
        out, scale = q, qs
    _quantize_mxfp8_kernel(X, out, scale, stride_xz, stride_xh, stride_xn, stride_xd, 32, ARENA_N, HEAD_DIM=128,
                           BLOCK_N=64)


@triton.jit
def _fa_loop(
    acc,
    l_i,
    m_i,
    q,
    q_scale,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    stride_kn,
    stride_kd,
    stride_ksn,
    stride_vn,
    stride_vd,
    stride_vsn,
    offs_m,
    lo,
    hi,
    N_CTX,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    SCALE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    MASK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    NEGATIVE_SCALE: tl.constexpr,
    CENTER_LOGITS: tl.constexpr = False,
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SCALE_K)
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
        offs_n = start_n.to(tl.int64) + tl.arange(0, BLOCK_N)
        if MASK_N:
            n_mask = offs_n < N_CTX
            k = tl.load(k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn, mask=n_mask[None, :],
                        other=0.0)
            ks = tl.load(ks_ptr + offs_n[:, None] * stride_ksn + offs_s[None, :], mask=n_mask[:, None], other=127)
            v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd, mask=n_mask[:, None],
                        other=0.0)
        else:
            k = tl.load(k_ptr + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
            ks = tl.load(ks_ptr + offs_n[:, None] * stride_ksn + offs_s[None, :])
            v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        # V's [head, key block] scales; blocks past the end read as 1.0 (their P is 0).
        offs_vb = start_n // 32 + tl.arange(0, BLOCK_N // 32)
        vb_mask = offs_vb[None, :] < tl.cdiv(N_CTX, 32)
        vs = tl.load(vs_ptr + offs_vb[None, :] * stride_vsn + offs_d[:, None], mask=vb_mask, other=127)
        qk = tl.dot_scaled(q, q_scale, "e4m3", k, ks, "e4m3", fast_math=True)
        if CENTER_LOGITS or NEGATIVE_SCALE:
            # Scale before masking to handle zero scales and large negative scores.
            qk = _rounded_mul(qk, qk_scale) if CENTER_LOGITS else qk * qk_scale
        if CAUSAL_MASK or MASK_N:
            valid = offs_m[:, None] >= offs_n[None, :] if CAUSAL_MASK else offs_m[:, None] < N_CTX
            if MASK_N:
                valid = valid & (offs_n[None, :] < N_CTX)
            if CAUSAL_MASK:
                valid = valid & (offs_m[:, None] < N_CTX)
            qk = tl.where(valid, qk, -float("inf") if CENTER_LOGITS or NEGATIVE_SCALE else -1.0e6)
        # Split launches and negative-scale launches already use logit units.
        # Other launches scale block maxima here and scores inside exp.
        qk = tl.reshape(qk, (BLOCK_M, BLOCK_N // 32, 32))
        if CENTER_LOGITS or NEGATIVE_SCALE:
            tmax = tl.max(qk, 2)
        else:
            tmax = tl.max(qk, 2) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(tmax, 1))
        # P to MXFP8: a block's max P is exp2(its max score - m).
        # The scale's exponent joins the exp argument, so P comes out divided by it.
        byte = _mx_scale(tl.math.exp2(tmax - m_ij[:, None]))
        if CENTER_LOGITS:
            centered = qk - m_ij[:, None, None]
            p = tl.math.exp2(centered - (byte - 127).to(tl.float32)[:, :, None])
        elif NEGATIVE_SCALE:
            p = tl.math.exp2(qk - (m_ij[:, None] + (byte - 127).to(tl.float32))[:, :, None])
        else:
            p = tl.math.exp2(qk * qk_scale - (m_ij[:, None] + (byte - 127).to(tl.float32))[:, :, None])
        if CAUSAL_MASK or MASK_N:
            p = tl.where(tl.reshape(valid, (BLOCK_M, BLOCK_N // 32, 32)), p, 0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(tl.sum(p, 2) * _decode_scale(byte), 1)
        acc = acc * alpha[:, None]
        p = tl.reshape(p, (BLOCK_M, BLOCK_N)).to(tl.float8e4nv)
        acc = tl.dot_scaled(p, byte.to(tl.uint8), "e4m3", v, vs, "e4m3", acc=acc, fast_math=True)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _mxfp8_fa_fwd(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
    LSE,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_qsb,
    stride_qsh,
    stride_qsm,
    stride_ksb,
    stride_ksh,
    stride_ksn,
    stride_vsb,
    stride_vsh,
    stride_vsn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    H,
    N_CTX,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CAUSAL: tl.constexpr,
    EVEN_N: tl.constexpr,
    SAVE_LSE: tl.constexpr,
    NEGATIVE_SCALE: tl.constexpr,
    LSE_SPLIT: tl.constexpr = False,
):
    SCALE_K: tl.constexpr = HEAD_DIM // 32
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SCALE_K)
    row_mask = offs_m < N_CTX
    if LSE_SPLIT:
        # Use the same FP32 coefficient calculation as backward.
        qk_scale = qk_scale * 1.4426950408889634

    q = tl.load(Q + off_z * stride_qb + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                mask=row_mask[:, None], other=0.0)
    q_scale = tl.load(Qs + off_z * stride_qsb + off_h * stride_qsh + offs_m[:, None] * stride_qsm + offs_s[None, :],
                      mask=row_mask[:, None], other=127)
    k_ptr = K + off_z * stride_kb + off_h * stride_kh
    v_ptr = V + off_z * stride_vb + off_h * stride_vh
    ks_ptr = Ks + off_z * stride_ksb + off_h * stride_ksh
    vs_ptr = Vs + off_z * stride_vsb + off_h * stride_vsh
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    if CAUSAL:
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, 0, start_m * BLOCK_M, N_CTX,
                                 qk_scale, HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, False, False,
                                 NEGATIVE_SCALE, CENTER_LOGITS=LSE_SPLIT)
        diag_hi = tl.minimum(start_m * BLOCK_M + BLOCK_M, N_CTX)
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, start_m * BLOCK_M, diag_hi,
                                 N_CTX, qk_scale, HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, True, True,
                                 NEGATIVE_SCALE, CENTER_LOGITS=LSE_SPLIT)
    else:
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, 0, N_CTX, N_CTX, qk_scale,
                                 HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, not EVEN_N, False, NEGATIVE_SCALE,
                                 CENTER_LOGITS=LSE_SPLIT)
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0)
    tl.store(Out + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             out.to(tl.bfloat16), mask=row_mask[:, None])
    if SAVE_LSE:
        if LSE_SPLIT:
            plane = tl.num_programs(1).to(tl.int64) * N_CTX
            tl.store(LSE + off_hz * N_CTX + offs_m, m_i, row_mask)
            tl.store(LSE + plane + off_hz * N_CTX + offs_m, tl.log2(l_i), row_mask)
        else:
            tl.store(LSE + off_hz * N_CTX + offs_m, m_i + tl.log2(l_i), row_mask)


@triton.jit
def _quantize_mxfp8_v_kernel(V, Out, Scale, stride_vb, stride_vh, stride_vn, stride_vd, H, N_CTX,
                             HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr):
    # V to MXFP8 along the keys (a 32-key block per column), by
    # _quantize_mxfp8_kernel's rule: [key, head] data and [key block, head]
    # scales.
    pid_bh = tl.program_id(1).to(tl.int64)
    offs_n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    rows = offs_n[:, None] < N_CTX
    x = tl.load(
        V + (pid_bh // H) * stride_vb + (pid_bh % H) * stride_vh + offs_n[:, None] * stride_vn +
        offs_d[None, :] * stride_vd, mask=rows, other=0.0).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM))
    byte = _mx_scale(tl.max(tl.abs(x), 1))
    inv = _decode_scale(byte, RECIPROCAL=True)
    q = tl.reshape(tl.clamp(x * inv[:, None, :], -448.0, 448.0).to(tl.float8e4nv), (BLOCK_N, HEAD_DIM))
    tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], q, mask=rows, cache_modifier=".cs")
    offs_b = tl.program_id(0) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
    n_blocks = tl.cdiv(N_CTX, 32)
    blocks = offs_b[:, None] < n_blocks
    tl.store(Scale + (pid_bh * n_blocks + offs_b[:, None]) * HEAD_DIM + offs_d[None, :], byte.to(tl.uint8), mask=blocks,
             cache_modifier=".cs")


def quantize_mxfp8_v(v):
    """E4M3 data and E8M0 scales of ``v`` along the sequence, a scale per 32
    keys of each head column: data ``[Z, H, N_CTX, HEAD_DIM]``, scales
    ``[Z, H, cdiv(N_CTX, 32), HEAD_DIM]``."""
    z, h, n, d = v.shape
    out = torch.empty((z, h, n, d), device=v.device, dtype=_FP8)
    scale = torch.empty((z, h, triton.cdiv(n, 32), d), device=v.device, dtype=torch.uint8)
    _quantize_mxfp8_v_kernel[(triton.cdiv(n, 64), z * h)](v, out, scale, v.stride(0), v.stride(1), v.stride(2),
                                                          v.stride(3), h, n, HEAD_DIM=d, BLOCK_N=64, num_warps=1)
    return out, scale


def _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, *, return_lse=False,
                      split_lse=False):
    """Launch the loop kernel on inputs from ``quantize_mxfp8_head`` and
    ``quantize_mxfp8_v``. ``return_lse=True`` returns ``(output, base2_lse)``.
    ``split_lse=True`` returns FP32 ``[2,B,H,N]`` row maxima and log2
    normalization terms instead of their sum."""
    if not math.isfinite(sm_scale):
        raise ValueError("gfx950 flash_attn_mxfp8 expects a finite sm_scale")
    batch, heads, n_ctx, _ = q_fp8.shape
    if split_lse and not return_lse:
        raise ValueError("Split softmax statistics require return_lse=True")
    qk_scale = sm_scale if split_lse else sm_scale * _LOG2E
    out = torch.empty(q_fp8.shape, device=q_fp8.device, dtype=torch.bfloat16)
    cfg = _default_config(causal, n_ctx)
    grid = (triton.cdiv(n_ctx, cfg["block_m"]), batch * heads)
    lse_shape = (2, batch, heads, n_ctx) if split_lse else (batch, heads, n_ctx)
    lse = torch.empty(lse_shape, device=q_fp8.device, dtype=torch.float32) if return_lse else None
    v_scale = v_scale.contiguous()
    _mxfp8_fa_fwd[grid](
        q_fp8,
        k_fp8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        out,
        lse,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        k_fp8.stride(0),
        k_fp8.stride(1),
        k_fp8.stride(2),
        k_fp8.stride(3),
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        q_scale.stride(0),
        q_scale.stride(1),
        q_scale.stride(2),
        k_scale.stride(0),
        k_scale.stride(1),
        k_scale.stride(2),
        v_scale.stride(0),
        v_scale.stride(1),
        v_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        heads,
        n_ctx,
        qk_scale,
        HEAD_DIM=_HEAD_DIM,
        BLOCK_M=cfg["block_m"],
        BLOCK_N=cfg["block_n"],
        NUM_STAGES=cfg["num_stages"],
        CAUSAL=causal,
        EVEN_N=(n_ctx % cfg["block_n"] == 0),
        SAVE_LSE=return_lse,
        NEGATIVE_SCALE=sm_scale < 0,
        LSE_SPLIT=split_lse,
        enable_fp_fusion=not split_lse,
        num_warps=cfg["num_warps"],
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-ieee", "false"), ),
    )
    return (out, lse) if return_lse else out


def _forward_helper_contract(function):
    return (function, function.__code__, function.__defaults__, tuple((function.__kwdefaults__ or {}).items()))


_PUBLIC_QK_ARENA_LENGTHS = (1024, 2048)
_PUBLIC_QK_FORWARD_HELPERS = tuple(
    (name, _forward_helper_contract(globals()[name]))
    for name in ("quantize_mxfp8_head", "quantize_mxfp8_v", "_launch_quantized", "_default_config"))
_PUBLIC_QK_JIT_HELPERS = tuple((name, globals()[name], globals()[name].src)
                               for name in ("_quantize_mxfp8_kernel", "_mxfp8_fa_fwd", "_fa_loop", "_mx_scale",
                                            "_decode_scale", "_scale_exponent", "_rounded_mul"))


def _can_save_public_qk_arena(q, k, v, sm_scale, causal):
    from .gfx950_bwd_shared import _arena_launch_controls, _is_gfx950

    if (type(causal) is not bool or type(sm_scale) is not float or sm_scale != 0.5 or _FP8 != torch.float8_e4m3fn
            or _HEAD_DIM != 128 or any(type(tensor) is not torch.Tensor for tensor in (q, k, v))):
        return False
    shape = tuple(q.shape)
    if len(shape) != 4 or shape[:2] != (4, 32) or shape[2] not in _PUBLIC_QK_ARENA_LENGTHS or shape[3] != 128:
        return False
    if any(tensor.shape != shape or tensor.dtype != torch.bfloat16 or tensor.device != q.device
           or tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.is_conj() or tensor.is_neg()
           for tensor in (q, k, v)):
        return False
    # Keep originally unsupported V storage on the separate-tensor path.
    # Saved-tensor hooks may later restore valid dense offset views; those
    # remain supported by the general backward after fresh metadata checks.
    storage = v.untyped_storage()
    pointer = v.data_ptr()
    if (v.storage_offset() != 0 or storage.nbytes() != 256 * shape[2] * 128 or pointer != storage.data_ptr()
            or pointer % 16 != 0 or not _arena_launch_controls()):
        return False
    for name, expected in _PUBLIC_QK_FORWARD_HELPERS:
        current = globals()[name]
        if (current is not expected[0] or current.__code__ is not expected[1]
                or current.__defaults__ is not expected[2]):
            return False
        keywords = current.__kwdefaults__
        if keywords is None:
            keywords = {}
        if (type(keywords) is not dict or len(keywords) != len(expected[3])
                or any(key not in keywords or type(keywords[key]) is not type(value) or keywords[key] != value
                       for key, value in expected[3])):
            return False
    for name, expected, source in _PUBLIC_QK_JIT_HELPERS:
        current = globals()[name]
        if (current is not expected or current.src != source or current.pre_run_hooks
                or current.launch_metadata is not None or current.debug):
            return False
    return q.device.type == "cuda" and _is_gfx950(q.device)


class _MXFP8Attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        from .gfx950_bwd_shared import (_SAVED_QK_ARENA_FORMAT, _check_saved_qk_arena, _saved_qk_arena_layout,
                                        _saved_qk_arena_views)

        packed = _can_save_public_qk_arena(q, k, v, sm_scale, causal)
        if packed:
            n = q.shape[2]
            arena = torch.empty((_saved_qk_arena_layout(n)[-1], ), device=q.device, dtype=torch.uint8)
            _check_saved_qk_arena(arena, q.device, n)
            for tensor, key in ((q, False), (k, True)):
                _quantize_saved_qk_arena[(n // 64, 128)](tensor, arena, *tensor.stride(), ARENA_N=n, KEY=key,
                                                         num_warps=4)
            # Forward reads these controlled Q/K views.
            q_fp8, k_fp8, q_scale, k_scale = _saved_qk_arena_views(arena, n)
        else:
            q_fp8, q_scale = quantize_mxfp8_head(q)
            k_fp8, k_scale = quantize_mxfp8_head(k)
        v_fp8, v_scale = quantize_mxfp8_v(v)
        out, lse = _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, return_lse=True,
                                     split_lse=True)
        if packed:
            ctx.save_for_backward(q, k, v, arena, out, lse)
            ctx.saved_format = _SAVED_QK_ARENA_FORMAT
        else:
            ctx.save_for_backward(q, k, v, q_fp8, k_fp8, out, lse, q_scale, k_scale)
            ctx.saved_format = None
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.lse_split = True
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        from .gfx950_bwd_shared import (_SAVED_QK_ARENA_FORMAT, _check_saved_qk_backward_tensors, _saved_qk_arena_views,
                                        _try_launch_backward_shared_square)

        if ctx.saved_format == _SAVED_QK_ARENA_FORMAT:
            q, k, v, qk_arena, out, lse = ctx.saved_tensors
            if qk_arena.device != q.device or q.shape != k.shape or q.shape != v.shape:
                raise ValueError("Saved Q/K arena must match the live input device and shape")
            q_fp8, k_fp8, q_scale, k_scale = qk_arena, None, None, None
        elif ctx.saved_format is None:
            q, k, v, q_fp8, k_fp8, out, lse, q_scale, k_scale = ctx.saved_tensors
        else:
            raise ValueError("Unknown MXFP8 saved tensor format")
        # Autograd may execute after the caller changes the current device.
        with torch.cuda.device(q.device):
            # Unpack hooks preserve values but can restore different strides or negative flags.
            q, k, v, out, lse = (tensor.resolve_neg().contiguous() for tensor in (q, k, v, out, lse))
            if ctx.saved_format is None:
                q_fp8, k_fp8, q_scale, k_scale = (tensor.resolve_neg().contiguous()
                                                  for tensor in (q_fp8, k_fp8, q_scale, k_scale))
            lse_split = getattr(ctx, "lse_split", False)
            do_bf16 = do.resolve_neg().to(torch.bfloat16).contiguous()
            # Saved Q/K have unpacked square32 scales. Softmax statistics and
            # O come from the same hardware-exp forward. Keep BF16 V/dO for
            # backward preparation rather than requantizing saved FP8 data.
            shared_args = (q_fp8, k_fp8, q_scale, k_scale, v, do_bf16, out, lse, ctx.sm_scale)
            shared_grads = _try_launch_backward_shared_square(*shared_args, causal=ctx.causal, lse_split=lse_split)
            if shared_grads is not None:
                dq, dk, dv = shared_grads
                return dq, dk, dv, None, None
            if ctx.saved_format == _SAVED_QK_ARENA_FORMAT:
                # General fallback uses controlled views only after fresh
                # whole-arena checks, never the separate shared admission.
                _check_saved_qk_backward_tensors(qk_arena, v, do_bf16, out, lse, allow_views=True, lse_split=lse_split)
                q_fp8, k_fp8, q_scale, k_scale = _saved_qk_arena_views(qk_arena, q.shape[2])
            from .gfx950_bwd import launch_backward
            from .gfx950_quant import quantize_backward_operands

            (q_dk, q_dk_scale), (k_dq,
                                 k_dq_scale), (v_bwd,
                                               v_scale), (do_fp8,
                                                          do_scale), (do_dv, do_dv_scale) = (quantize_backward_operands(
                                                              q, k, v, do_bf16, transpose_storage=True))
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            delta = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
            launch_backward(do_fp8, do_dv, q_fp8, q_dk, k_fp8, k_dq, v_bwd, out, lse, q_scale, q_dk_scale, k_scale,
                            k_dq_scale, v_scale, do_scale, do_dv_scale, ctx.sm_scale, do_bf16, dq, dk, dv, delta,
                            causal=ctx.causal, lse_split=lse_split)
        return dq, dk, dv, None, None


def flash_attn_mxfp8_train(q, k, v, causal, sm_scale):
    """Differentiable gfx950 MXFP8 attention; gfx950.flash_attn_mxfp8 calls
    it when autograd records Q, K or V."""
    if not math.isfinite(sm_scale):
        from triton.tlx.ops import InvalidInput

        raise InvalidInput("gfx950 flash_attn_mxfp8 expects a finite sm_scale")
    # Materialize lazy-negative masters before pointer-based quantization.
    q, k, v = (tensor.resolve_neg() for tensor in (q, k, v))
    return _MXFP8Attention.apply(q, k, v, sm_scale, causal)
