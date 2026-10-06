"""gfx950 MXFP8 Flash Attention.

BF16 ``(Z, H, N_CTX, HEAD_DIM)`` in, BF16 out. Following the gfx950 forward
implementation, Q and K are E4M3 with an E8M0 scale per 32x32 block (32 rows
by 32 head elements), V is E4M3 with an E8M0 scale per 32 keys of each head
column, and the softmax probabilities P get an E8M0 scale per 32 keys of each
row, all by RCEIL of the block max. Q is quantized unscaled, and sm_scale *
log2(e) multiplies the scores. Both MFMAs are scaled.
The Blackwell public wrapper also uses 32x32 groups for Q/K.

Non-causal runs a persistent two-warp-group ping-pong kernel (256x128 tiles,
8 warps, Q/K/V staged in LDS by direct copies, K/V double buffered). It feeds
the row offset in as the QK accumulator, computes P with a bit-trick exp2
instead of the hardware exp, and only rescales the accumulator when a row max
grows past a threshold. P's block scales are applied by its scaled FP8
conversion (v_cvt_scalef32_pk_fp8_bf16); its row sums run on the matrix cores
from the FP8 P per block and take the block scales on the vector ALUs. Causal
runs a single software-pipelined loop with the hardware exp, where P's block
scales join the exp argument.

The bit-trick exp2 is specific to gfx950 (Blackwell uses the hardware exp2):
the hardware exp runs at a quarter of the VALU rate, in the softmax stage
that bounds the non-causal kernel's speed. It gives 2**x with a linear
mantissa, up to 6.1% high and 4.1% high on average. P and its row sums share
that bias, so after the normalization P is between about 4% low and 2% high,
within the error of rounding P to E4M3 (up to 6.25%).
``_launch_quantized(..., fast_exp=False)`` runs the kernel with the hardware
exp instead.

Training uses the hardware-exp loop kernel and saves its base-2 logsumexp
before P quantization. Backward uses native scaled MFMA for all five unique
products, fixed-scale P and transpose-compatible 32x32 dS quantization.
The general path uses separate Q and KV owners that recompute QK and dP.
For contiguous B4/H32/D128 MHA at scale 0.5 and N=1024/2048/4096/8192,
causal or noncausal, a specialized backward shares square32 Q/K/dO
quantization and materializes FP8 dS for reuse by the Q owner. It also
supports N8192 noncausal at scale 1.3. Each gradient has a single writer
and accumulates deterministically in FP32. Whole, aligned storage is
required. Temporary storage is 128*N*N + 128*(N//32)**2 + 35840*N bytes,
up to 8 GiB + 288 MiB at N8192, excluding gradients and saved inputs.
Causal calls allocate the same dense workspace. Other configurations use
the general implementation; allocation or kernel failures are not retried.
"""

import math

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from .gfx950_quant import _decode_scale, _scale_exponent

_FP8 = torch.float8_e4m3fn
_LOG2E = 1.4426950408889634
_HEAD_DIM = 128
_NUM_CUS = 256  # gfx950 as one whole device (compute partition SPX)


def _default_config(causal, n_ctx, *, training=False):
    # Measured on gfx950, B=4 H=32 D=128. The ping-pong kernel is built around
    # its tiles and 8 warps and pipelines by hand (no num_stages), so it takes
    # only the non-causal lengths its block_m divides; the loop kernel takes
    # causal and the rest.
    if not causal and n_ctx % 256 == 0 and not training:
        return {"pingpong": True, "block_m": 256, "block_n": 128, "num_warps": 8}
    if causal:
        return {"pingpong": False, "block_m": 128, "block_n": 64, "num_warps": 4, "num_stages": 2}
    return {"pingpong": False, "block_m": 128, "block_n": 128, "num_warps": 4, "num_stages": 2}


@triton.jit
def _mx_scale(amax):
    # Keep the result signed for the subsequent (byte - 127) exponent.
    return _scale_exponent(amax).to(tl.int32)


@triton.jit
def _quantize_mxfp8_kernel(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM: tl.constexpr,
                           BLOCK_N: tl.constexpr, PACK_K: tl.constexpr):
    # Each 32x32 block of x (32 rows by 32 head elements) gets the E8M0 byte
    # _mx_scale(amax) for its max magnitude amax, stored for each of its rows,
    # and E4M3 data x / 2**(byte - 127). PACK_K writes the scales in
    # quantize_mxfp8_head's pack_k order. The outputs are read once, by the
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
    s = s.to(tl.uint8)
    if PACK_K:
        s = tl.reshape(tl.permute(tl.reshape(s, (BLOCK_N // 64, 2, 32, 2, 2)), (0, 4, 2, 3, 1)), (BLOCK_N, 4))
    tl.store(Scale + (pid_bh * N_CTX + offs_n[:, None]) * (HEAD_DIM // 32) + offs_b[None, :], s, mask=rows,
             cache_modifier=".cs")


def quantize_mxfp8_head(x, *, pack_k=False):
    """E4M3 data and E8M0 scales of a ``[Z, H, N_CTX, HEAD_DIM]`` tensor with
    a scale per 32x32 block (32 rows by 32 head elements). The scales are
    ``[Z, H, N_CTX, HEAD_DIM // 32]``, each
    row holding its block's. ``pack_k`` orders K's scales as the non-causal
    kernel reads them: per 64 keys, word ``32 * b + k`` holds keys ``k`` and
    ``k + 32`` at d-block ``b`` of both head-dim halves."""
    z, h, n, d = x.shape
    if pack_k and (d != 128 or n % 64 != 0):
        raise ValueError("packed K scales need HEAD_DIM 128 and N_CTX % 64 == 0")
    quant = torch.empty((z, h, n, d), device=x.device, dtype=_FP8)
    scale = torch.empty((z, h, n, d // 32), device=x.device, dtype=torch.uint8)
    _quantize_mxfp8_kernel[(triton.cdiv(n, 64), z * h)](x, quant, scale, x.stride(0), x.stride(1), x.stride(2),
                                                        x.stride(3), h, n, HEAD_DIM=d, BLOCK_N=64, PACK_K=pack_k,
                                                        num_warps=4)
    return quant, scale


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
        if NEGATIVE_SCALE:
            # Multiplication reverses extrema for a negative scale. Scale
            # first, then apply the mask in logit units before reducing max.
            qk = qk * qk_scale
        if CAUSAL_MASK or MASK_N:
            valid = offs_m[:, None] >= offs_n[None, :] if CAUSAL_MASK else offs_m[:, None] < N_CTX
            if MASK_N:
                valid = valid & (offs_n[None, :] < N_CTX)
            if CAUSAL_MASK:
                valid = valid & (offs_m[:, None] < N_CTX)
            qk = tl.where(valid, qk, -float("inf") if NEGATIVE_SCALE else -1.0e6)
        # qk_scale takes the scores to log2 units: the block maxima are scaled
        # directly, each score inside the exp argument.
        qk = tl.reshape(qk, (BLOCK_M, BLOCK_N // 32, 32))
        if NEGATIVE_SCALE:
            tmax = tl.max(qk, 2)
        else:
            tmax = tl.max(qk, 2) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(tmax, 1))
        # P to MXFP8: a block's max P is exp2(its max score - m).
        # The scale's exponent joins the exp argument, so P comes out divided by it.
        byte = _mx_scale(tl.math.exp2(tmax - m_ij[:, None]))
        if NEGATIVE_SCALE:
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
                                 NEGATIVE_SCALE)
        diag_hi = tl.minimum(start_m * BLOCK_M + BLOCK_M, N_CTX)
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, start_m * BLOCK_M, diag_hi,
                                 N_CTX, qk_scale, HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, True, True,
                                 NEGATIVE_SCALE)
    else:
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, 0, N_CTX, N_CTX, qk_scale,
                                 HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, not EVEN_N, False, NEGATIVE_SCALE)
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0)
    tl.store(Out + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             out.to(tl.bfloat16), mask=row_mask[:, None])
    if SAVE_LSE:
        tl.store(LSE + off_hz * N_CTX + offs_m, m_i + tl.log2(l_i), row_mask)


# Padded LDS layouts the AMD pipeliner picks for these FP8 dot operands with
# warpsPerCTA [8, 1]: K stored [head 128, key 64] with the head contiguous, V
# [key 64, head 128] with the keys contiguous (V comes transposed).
_K_LDS_BASES = tl.constexpr([[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 8], [0, 16], [0, 32],
                             [0, 1], [0, 2], [0, 4]])
_V_LDS_BASES = tl.constexpr([[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8], [0, 16], [0, 32], [0, 1],
                             [0, 2], [0, 64]])


@triton.jit
def _exp2_bf16_bits(t):
    # t = (x + 127) * 128 / 65535. v_cvt_pknorm_u16_f32 rounds clamp(t, 0, 1) *
    # 65535, the bf16 bit pattern of 2**x with a linear mantissa (at most 6.1%
    # high), two per register.
    return tl.inline_asm_elementwise("v_cvt_pknorm_u16_f32 $0, $1, $2", "=v,v,v", [t], dtype=tl.bfloat16, is_pure=True,
                                     pack=2)


# The softmax sees scores t = (x - m_i + OFF) * C, x the log2-domain logit and
# m_i the stale row max; P = 2**(x - m_i + 4) stays below 2**8 while the max
# holds, and each 32-key block of it then gets its own E8M0 scale. With the
# bit-trick exp, C = 2**-9 comes from lowering Q's E8M0 scales and OFF = 127 + 4
# makes t the bf16 exponent position scaled for v_cvt_pknorm_u16_f32
# (65535 / 2**16 slope error, common to P and its sum). With the hardware exp,
# t is the exponent itself.
@triton.jit
def _score_scale(FAST_EXP: tl.constexpr):
    return 0.001953125 if FAST_EXP else 1.0


@triton.jit
def _score_offset(FAST_EXP: tl.constexpr):
    return 131.0 if FAST_EXP else 4.0


@triton.jit
def _relabel_keys(p):
    # PV sums over keys, so any key order works if P and V agree. Swapping
    # key-index bits 2 and 4 puts P, as the QK MFMA leaves it, in the PV
    # operand layout register for register; quantize_mxfp8_v(transposed=True)
    # stores V's keys in that order.
    x = tl.reshape(p, (p.shape[0], 2, 2, 2, 2, 4))
    x = tl.permute(x, (0, 1, 4, 3, 2, 5))
    return tl.reshape(x, (p.shape[0], p.shape[1]))


@triton.jit
def _block_max(s):
    # Each row's max score per 32-key block. In the transposed 32x32 MFMA
    # layout a lane holds one row, and key 32t + 8j + 4h + i sits in register
    # (t, j, i) of lane half h: a reduction over registers, then one across the
    # two halves.
    return tl.max(tl.reshape(s, (s.shape[0], 2, 32)), 2)


@triton.jit
def _per_block(x):
    # [row, block] values over each block's 32 keys.
    return tl.reshape(tl.broadcast_to(x[:, :, None], (x.shape[0], 2, 32)), (x.shape[0], 64))


@triton.jit
def _p_scales(tmax, FAST_EXP: tl.constexpr):
    # P's E8M0 scale per 32-key block from the block's max score (P rises with
    # t), by _mx_scale's rule. Returns the byte and an fp32 with the byte as
    # its exponent field, the scale operand of _to_e4m3.
    if FAST_EXP:
        # Max P's bf16 bits b give byte ((b + 31) >> 7) - 8, clamped at 0.
        b = _exp2_bf16_bits(tmax).to(tl.uint16, bitcast=True).to(tl.int32)
        byte = tl.maximum(((b + 31) >> 7) - 8, 0)
    else:
        byte = _mx_scale(tl.math.exp2(tmax))
    # The bytes go to the PV MFMA and _scale_row_sums, which both take block b
    # of row r from lane r % 32 + 32b: pinned there, as P carries them across
    # the loop (left free they get a blocked layout and an LDS round trip).
    halves: tl.constexpr = tlx.layout(shape=((32, 2, tmax.shape[0] // 32), (1, )), stride=((2, 1, 64), (1, )))
    return tlx.require_layout(byte.to(tl.uint8), halves), _decode_scale(byte)


@triton.jit
def _to_e4m3(p, scale):
    # E4M3 of bf16 P divided by 2**(e - 127), e the exponent field of scale:
    # v_cvt_scalef32_pk_fp8_bf16 converts two keys, so two fill a register of
    # four. The asm returns that register as two u16 (AMDGPU inline asm has no
    # register type for four i8); its second output is unused.
    w = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_bf16 $0, $2, $4\n"
        "v_cvt_scalef32_pk_fp8_bf16 $0, $3, $6 op_sel:[0,0,1]", "=&v,=v,v,v,v,v,v,v", [p, scale], dtype=tl.uint16,
        is_pure=True, pack=4)
    rows: tl.constexpr = p.shape[0]
    cols: tl.constexpr = p.shape[1]
    w, _ = tl.split(tl.permute(tl.reshape(w, (rows, cols // 4, 2, 2)), (0, 1, 3, 2)))
    b = tl.join(w.to(tl.uint8), (w >> 8).to(tl.uint8))
    return tl.reshape(b, (rows, cols)).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _as_rowsum_a(p):
    # FP8 P in the PV operand layout, read register for register as the A
    # operand of a 16x16x128 MFMA: A'[i, k'] = P[(i & 15) + 16 * k'_4 +
    # 32 * (i >> 4), key], key = (k' & 15) + 16 * k'_5 + 32 * k'_6.
    x = tl.reshape(p, (p.shape[0] // 32, 2, 16, 2, 2, 16))
    x = tl.permute(x, (0, 2, 3, 4, 1, 5))
    return tl.reshape(x, (p.shape[0] // 2, 128))


@triton.jit
def _rowsum_b():
    # Routes k'_4 (P row bit 4) and k'_6 (the key's 32-key block b) to output
    # columns 4 * (2b + k'_4) .. + 3: unscaled sums per row and block.
    k = tl.arange(0, 128)[:, None]
    j = tl.arange(0, 16)[None, :]
    return tl.where(2 * ((k >> 6) & 1) + ((k >> 4) & 1) == (j >> 2), 1.0, 0.0).to(tl.float8e4nv)


@triton.jit
def _block_sums(t):
    # _rowsum_b's [16w + i, 4 * (2b + q) + r] as [row 32w + 16q + i, block b].
    # In the transposed 16x16 MFMA layout lane l holds A' row l % 16 and column
    # group l // 16, so row 32w + l % 32 of block l // 32 is already in its
    # lane; pinned there (left free, the multiply by the scales moves the
    # block into registers with a lane exchange).
    halves: tl.constexpr = tlx.layout(shape=((32, 2, t.shape[0] // 16), (1, )), stride=((2, 1, 64), (1, )))
    x = tl.permute(tl.reshape(t, (t.shape[0] // 16, 16, 2, 2, 2, 2)), (0, 3, 1, 2, 4, 5))
    x0, _ = tl.split(tl.reshape(x, (t.shape[0] * 2, 2, 2, 2)))
    x00, _ = tl.split(x0)
    return tlx.require_layout(x00, halves)


@triton.jit
def _scale_row_sums(t0, t1, p0, p1, l_i):
    # l_i [row, block half] plus P's row sums per block half: 2**(byte - 127)
    # times the unscaled sums t0, t1 of P's two key halves.
    f0 = _decode_scale(p0[1])
    f1 = _decode_scale(p1[1])
    return l_i + _block_sums(t0) * f0 + _block_sums(t1) * f1


@triton.jit
def _p_half(s, tmax, FAST_EXP: tl.constexpr):
    # One 64-key half of P in MXFP8 from scores s whose 32-key blocks peak at
    # tmax: (E4M3 in the PV operand order, E8M0 per 32 keys of each row). The
    # scales are pinned to the scores' MFMA layout (the reshape leaves them in
    # an equivalent linear one); otherwise the scores get converted to it and
    # the accumulators lose the MFMA layout.
    mfma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [8, 1])
    byte, scale = _p_scales(tmax, FAST_EXP)
    if FAST_EXP:
        bits = _exp2_bf16_bits(s)
    else:
        bits = tl.math.exp2(s).to(tl.bfloat16)
    p = _to_e4m3(bits, tlx.require_layout(_per_block(scale), mfma))
    return _relabel_keys(p), byte


@triton.jit
def _pingpong_softmax_first(s0, s1, FAST_EXP: tl.constexpr):
    # First tile: no offset in the scores yet (t = x * C).
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    tmax0 = _block_max(s0)
    tmax1 = _block_max(s1)
    m_i = tl.maximum(tl.max(s0, 1), tl.max(s1, 1)) * (1.0 / C)
    # The same shift in the scores' layout and in the block maxima's.
    shift = ((OFF - m_i) * C)[:, None]
    tshift = (OFF * C - tl.maximum(tl.max(tmax0, 1), tl.max(tmax1, 1)))[:, None]
    p0 = _p_half(s0 + shift, tmax0 + tshift, FAST_EXP)
    p1 = _p_half(s1 + shift, tmax1 + tshift, FAST_EXP)
    return p0, p1, m_i, tl.zeros([s0.shape[0], 2], dtype=tl.float32)


@triton.jit
def _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP: tl.constexpr):
    # s0, s1 carry the offset for m_i from the QK accumulator, and s0 comes
    # multiplied by qk_scale (see _mfma_stage); acc0..acc3 are the 32-column
    # chunks of the output accumulator and l_i the row sums per block half of
    # _row_sums.
    #
    # Keep m_i until some row in this warp grows past it by more than 2**4,
    # i.e. t > (OFF + 4) * C. The vote predicate is pinned to the layout the
    # block maxima leave it in (row r on lanes r % 32 and r % 32 + 32 of warp
    # r // 32); otherwise warp_any gets a default layout and a shuffle.
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    vote_layout: tl.constexpr = tlx.layout(shape=((32, 2, s0.shape[0] // 32), (1, )), stride=((1, 0, 32), (1, )))
    s1 = s1 * qk_scale
    tmax0 = _block_max(s0)
    tmax1 = _block_max(s1)
    row = tl.maximum(tl.max(tmax0, 1), tl.max(tmax1, 1))
    tshift = tl.zeros_like(row)
    if tlx.warp_any(tlx.require_layout(row > (OFF + 4.0) * C, vote_layout)):
        # m_i moves up by max(row / C - OFF, 0). For the scores, m_i and the
        # accumulators it is taken again from the scores, in the MFMA layout's
        # rows; from the block maxima the accumulators would end up carried in
        # a blocked layout and converted around every PV. Shift the scores in
        # place: computing P in both branches keeps P and the scores live
        # together and pushes the kernel past 256 VGPRs.
        up = tl.maximum(tl.maximum(tl.max(s0, 1), tl.max(s1, 1)) - OFF * C, 0.0)
        alpha = tl.math.exp2(up * (-1.0 / C))[:, None]
        acc0 = acc0 * alpha
        acc1 = acc1 * alpha
        acc2 = acc2 * alpha
        acc3 = acc3 * alpha
        s0 = s0 - up[:, None]
        s1 = s1 - up[:, None]
        m_i = m_i + up * (1.0 / C)
        tshift = -tl.maximum(row - OFF * C, 0.0)
    # The block maxima and the row sums follow outside the branch (a shift of
    # 0 without a rescale): inside, their layouts keep the branch's results out
    # of the MFMA layout.
    l_i = l_i * tl.math.exp2(tshift * (1.0 / C))[:, None]
    p0 = _p_half(s0, tmax0 + tshift[:, None], FAST_EXP)
    p1 = _p_half(s1, tmax1 + tshift[:, None], FAST_EXP)
    return p0, p1, m_i, l_i, acc0, acc1, acc2, acc3


@triton.jit
def _issue_tile(k_ptr, v_ptr, ks_ptr, vs_ptr, ka, kb, va, vb, ksa, ksb, vsa, vsb, kt, vt, k_step, v_step, ks_step, k_n1,
                v_n1, ks_n1, COPY_K: tl.constexpr, COPY_V: tl.constexpr):
    # V's packed E8M0 words go with V; they are laid out like K's, so they
    # share its offsets and steps.
    if COPY_K:
        tlx.async_load(k_ptr + kt * k_step, ka)
        tlx.async_load(k_ptr + kt * k_step + k_n1, kb)
        tlx.async_load(ks_ptr + kt * ks_step, ksa)
        tlx.async_load(ks_ptr + kt * ks_step + ks_n1, ksb)
    if COPY_V:
        tlx.async_load(v_ptr + vt * v_step, va)
        tlx.async_load(v_ptr + vt * v_step + v_n1, vb)
        tlx.async_load(vs_ptr + vt * ks_step, vsa)
        tlx.async_load(vs_ptr + vt * ks_step + ks_n1, vsb)
    if COPY_K or COPY_V:
        tlx.async_load_commit_group()


@triton.jit
def _ld_k(kbuf, half: tl.constexpr):
    # One head-dim half of a K tile.
    return tlx.local_load(tlx.local_slice(kbuf, [64 * half, 0], [64, kbuf.shape[1]]), relaxed=True)


@triton.jit
def _ld_ks(ksbuf):
    # The E8M0 scales of 64 keys in the packed K order: lane l's word holds
    # keys l % 32 and l % 32 + 32 at d-block l // 32 of both head-dim halves,
    # so one word per lane feeds both QK MFMA pairs. Returns the scales of the
    # low and high head-dim halves, [key, block] each.
    view: tl.constexpr = tlx.shared_linear_layout_encoding([[32, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0], [4, 0, 0],
                                                            [8, 0, 0], [16, 0, 0], [0, 1, 0]])
    lanes: tl.constexpr = tlx.layout(shape=((32, 2, 8), (2, 2)), stride=((4, 2, 0), (128, 1)))
    ks = tlx.local_reinterpret(ksbuf, tl.uint8, [64, 2, 2], layout=view)
    return tl.split(tlx.local_load(ks, layout=lanes, relaxed=True))


@triton.jit
def _ld_v(vbuf, c: tl.constexpr):
    return tlx.local_load(tlx.local_slice(vbuf, [0, 32 * c], [vbuf.shape[0], 32]), relaxed=True)


@triton.jit
def _ld_vs(vsbuf):
    # V's E8M0 scales for 64 keys in the transposed V order: lane l's word
    # holds column l % 32 of key block l // 32 for each 32-column V chunk, the
    # lane order the scaled PV MFMA takes. Returns one [column, block] scale
    # per chunk.
    view: tl.constexpr = tlx.shared_linear_layout_encoding([[0, 0, 1], [0, 0, 2], [1, 0, 0], [2, 0, 0], [4, 0, 0],
                                                            [8, 0, 0], [16, 0, 0], [0, 1, 0]])
    lanes: tl.constexpr = tlx.layout(shape=((32, 2, 8), (4, )), stride=((8, 4, 0), (1, )))
    vs = tlx.local_load(tlx.local_reinterpret(vsbuf, tl.uint8, [32, 2, 4], layout=view), layout=lanes, relaxed=True)
    even, odd = tl.split(tl.reshape(vs, (32, 2, 2, 2)))
    s0, s2 = tl.split(even)
    s1, s3 = tl.split(odd)
    return s0, s1, s2, s3


@triton.jit
def _pv(p, v, vs, acc):
    # p is _p_half's (E4M3, E8M0) pair.
    return tl.dot_scaled(p[0], p[1], "e4m3", v, vs, "e4m3", acc=acc, fast_math=True)


@triton.jit
def _mfma_stage(q_lo, q_hi, qs_lo, qs_hi, ka, kb, ksa, ksb, va, vb, vsa, vsb, p0, p1, acc0, acc1, acc2, acc3, l_i,
                offset, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, kd_a, kd_b, vd_a, vd_b, ksd_a, ksd_b, vsd_a, vsd_b, kt,
                vt, k_step, v_step, ks_step, k_n1, v_n1, ks_n1, COPY_K: tl.constexpr, COPY_V: tl.constexpr):
    # QK(j + 1) and PV(j) with operands read from LDS in MFMA-sized pieces. Each
    # read is issued right after the MFMA that frees the previous piece and is
    # consumed one MFMA later, so its latency hides behind an MFMA while at most
    # one K half and one V chunk are live. The scheduling barriers keep LLVM
    # from sinking the reads to their uses. A per-row offset, divided by
    # qk_scale, starts the QK accumulator. s0 leaves multiplied by qk_scale,
    # behind the last PV MFMAs, and s1 is multiplied in the softmax stage:
    # split between the two stages, the multiplies cost the least.
    #
    # P's row sums go first: they need only registers, so the matrix cores get
    # work right after the stage barrier, and the first reads and the next
    # tile's global->LDS copies (those COPY_K / COPY_V select) issue behind
    # them. Their block scales are applied after the first QK MFMA, when the
    # sums are in.
    init = tl.broadcast_to(offset[:, None], (q_lo.shape[0], ka.shape[1]))
    ones = _rowsum_b()
    t0 = tl.dot(_as_rowsum_a(p0[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32))
    kA0 = _ld_k(ka, 0)
    sA0, sA1 = _ld_ks(ksa)
    vA0 = _ld_v(va, 0)
    vsA0, vsA1, vsA2, vsA3 = _ld_vs(vsa)
    tlx.amd_sched_barrier(0)
    t1 = tl.dot(_as_rowsum_a(p1[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32))
    _issue_tile(k_ptr, v_ptr, ks_ptr, vs_ptr, kd_a, kd_b, vd_a, vd_b, ksd_a, ksd_b, vsd_a, vsd_b, kt, vt, k_step,
                v_step, ks_step, k_n1, v_n1, ks_n1, COPY_K, COPY_V)
    tlx.amd_sched_barrier(0)
    s0 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kA0, sA0, "e4m3", acc=init, fast_math=True)
    vA1 = _ld_v(va, 1)
    tlx.amd_sched_barrier(0)
    l_i = _scale_row_sums(t0, t1, p0, p1, l_i)
    acc0 = _pv(p0, vA0, vsA0, acc0)
    kA1 = _ld_k(ka, 1)
    tlx.amd_sched_barrier(0)
    acc1 = _pv(p0, vA1, vsA1, acc1)
    vA2 = _ld_v(va, 2)
    tlx.amd_sched_barrier(0)
    s0 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kA1, sA1, "e4m3", acc=s0, fast_math=True)
    vA3 = _ld_v(va, 3)
    tlx.amd_sched_barrier(0)
    acc2 = _pv(p0, vA2, vsA2, acc2)
    kB0 = _ld_k(kb, 0)
    sB0, sB1 = _ld_ks(ksb)
    tlx.amd_sched_barrier(0)
    acc3 = _pv(p0, vA3, vsA3, acc3)
    vB0 = _ld_v(vb, 0)
    vsB0, vsB1, vsB2, vsB3 = _ld_vs(vsb)
    tlx.amd_sched_barrier(0)
    s1 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", acc=init, fast_math=True)
    vB1 = _ld_v(vb, 1)
    tlx.amd_sched_barrier(0)
    acc0 = _pv(p1, vB0, vsB0, acc0)
    kB1 = _ld_k(kb, 1)
    tlx.amd_sched_barrier(0)
    acc1 = _pv(p1, vB1, vsB1, acc1)
    vB2 = _ld_v(vb, 2)
    tlx.amd_sched_barrier(0)
    s1 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kB1, sB1, "e4m3", acc=s1, fast_math=True)
    vB3 = _ld_v(vb, 3)
    tlx.amd_sched_barrier(0)
    acc2 = _pv(p1, vB2, vsB2, acc2)
    acc3 = _pv(p1, vB3, vsB3, acc3)
    return s0 * qk_scale, s1, acc0, acc1, acc2, acc3, l_i


@triton.jit
def _qk_first(q_lo, q_hi, qs_lo, qs_hi, ka, kb, ksa, ksb, qk_scale):
    kA0 = _ld_k(ka, 0)
    kA1 = _ld_k(ka, 1)
    kB0 = _ld_k(kb, 0)
    kB1 = _ld_k(kb, 1)
    sA0, sA1 = _ld_ks(ksa)
    sB0, sB1 = _ld_ks(ksb)
    s0 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kA0, sA0, "e4m3", fast_math=True)
    s0 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kA1, sA1, "e4m3", acc=s0, fast_math=True)
    s1 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", fast_math=True)
    s1 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kB1, sB1, "e4m3", acc=s1, fast_math=True)
    return s0 * qk_scale, s1 * qk_scale


@triton.jit
def _pv_last(p0, p1, va, vb, vsa, vsb, acc0, acc1, acc2, acc3, l_i):
    ones = _rowsum_b()
    t0 = tl.dot(_as_rowsum_a(p0[0]), ones, tl.zeros([p0[0].shape[0] // 2, 16], dtype=tl.float32))
    t1 = tl.dot(_as_rowsum_a(p1[0]), ones, tl.zeros([p1[0].shape[0] // 2, 16], dtype=tl.float32))
    sa0, sa1, sa2, sa3 = _ld_vs(vsa)
    sb0, sb1, sb2, sb3 = _ld_vs(vsb)
    acc0 = _pv(p0, _ld_v(va, 0), sa0, acc0)
    acc1 = _pv(p0, _ld_v(va, 1), sa1, acc1)
    acc2 = _pv(p0, _ld_v(va, 2), sa2, acc2)
    acc3 = _pv(p0, _ld_v(va, 3), sa3, acc3)
    acc0 = _pv(p1, _ld_v(vb, 0), sb0, acc0)
    acc1 = _pv(p1, _ld_v(vb, 1), sb1, acc1)
    acc2 = _pv(p1, _ld_v(vb, 2), sb2, acc2)
    acc3 = _pv(p1, _ld_v(vb, 3), sb3, acc3)
    return acc0, acc1, acc2, acc3, _scale_row_sums(t0, t1, p0, p1, l_i)


@triton.jit
def _issue_q(q_ptr, qs_ptr, q_n1, qa, qsa):
    tlx.async_load(q_ptr, qa[0])
    tlx.async_load(q_ptr + q_n1, qa[1])
    tlx.async_load(qs_ptr, qsa)
    tlx.async_load_commit_group()


@triton.jit
def _mxfp8_fa_fwd_pingpong(
    Q,
    K,
    V,
    Qs,
    Ks,
    Vs,
    Out,
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
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    Z,
    H,
    N_CTX,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MFMA_PRIO: tl.constexpr,
    SOFTMAX_PRIO: tl.constexpr,
    FAST_EXP: tl.constexpr,
):
    # Non-causal, N_CTX % (2 * BLOCK_N) == 0, contiguous [row, 4] Q scales, K
    # from quantize_mxfp8_head(pack_k=True), V and its scales from
    # quantize_mxfp8_v(transposed=True) (V's strides given as for [key, head],
    # its scales strided like K's).
    # Persistent: each workgroup loops over (Q tile, head) items. Two warp
    # groups run one stage apart: one issues QK(j + 1) and PV(j) while the
    # other does softmax(j + 1). Each K/V tile is two key
    # halves in LDS, double buffered with compile-time buffer indices; tile
    # j + 1 is copied during step j. Q and its scales are staged in LDS as two
    # head-dim halves; the next item's Q and first K/V tiles are copied during
    # this item's epilogue.
    HN: tl.constexpr = BLOCK_N // 2
    HD: tl.constexpr = HEAD_DIM // 2
    tl.static_assert(HN == 64 and HEAD_DIM == 128)
    pid = tl.program_id(0)
    n_ctas = tl.num_programs(0)
    n_m = N_CTX // BLOCK_M
    n_items = n_m * H * Z
    # Workgroups go round-robin to the 8 XCDs. Give each XCD a contiguous run
    # of items so the Q tiles sharing a K/V stream share that XCD's L2.
    if (n_ctas % 8 == 0) and (n_items % 8 == 0):
        per_xcd = n_items // 8
        stride_item = n_ctas // 8
        base = (pid % 8) * per_xcd + pid // 8
        n_mine = (per_xcd - pid // 8 + stride_item - 1) // stride_item
    else:
        stride_item = n_ctas
        base = pid
        n_mine = (n_items - pid + n_ctas - 1) // n_ctas

    offs_d = tl.arange(0, HEAD_DIM)
    offs_hd = tl.arange(0, HD)
    offs_n = tl.arange(0, HN)
    offs_mm = tl.arange(0, BLOCK_M)
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    # The row offset starts the QK accumulator, before the scores' qk_scale.
    c_qk = C / qk_scale

    k_off = offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
    v_off = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd.to(tl.int64)
    ks_off = offs_n[:, None] * stride_ksn // 4
    q_off = offs_mm[:, None] * stride_qm + offs_hd[None, :] * stride_qd
    qs_off = offs_mm[:, None] * stride_qsm // 4
    k_step = BLOCK_N * tl.full((), stride_kn, tl.int64)
    v_step = BLOCK_N * tl.full((), stride_vn, tl.int64)
    ks_step = BLOCK_N * tl.full((), stride_ksn, tl.int64) // 4
    k_n1 = HN * stride_kn
    v_n1 = HN * stride_vn
    ks_n1 = HN * stride_ksn // 4
    q_n1 = HD * stride_qd
    n_tiles = N_CTX // BLOCK_N

    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(1024, 16)], _K_LDS_BASES, [HEAD_DIM, HN])
    v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(1024, 16)], _V_LDS_BASES, [HN, HEAD_DIM])
    ka = tlx.local_alloc((HEAD_DIM, HN), tl.float8e4nv, 2, layout=k_layout)
    kb = tlx.local_alloc((HEAD_DIM, HN), tl.float8e4nv, 2, layout=k_layout)
    va = tlx.local_alloc((HN, HEAD_DIM), tl.float8e4nv, 2, layout=v_layout)
    vb = tlx.local_alloc((HN, HEAD_DIM), tl.float8e4nv, 2, layout=v_layout)
    ksa = tlx.local_alloc((HN, 1), tl.int32, 2)
    ksb = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsa = tlx.local_alloc((HN, 1), tl.int32, 2)
    vsb = tlx.local_alloc((HN, 1), tl.int32, 2)
    qa = tlx.local_alloc((BLOCK_M, HD), tl.float8e4nv, 2)
    qsa = tlx.local_alloc((BLOCK_M, 1), tl.int32, 1)

    # First item: Q, then K(0) into buffer 1, then K(1) and V(0) into buffer 0.
    start_m = (base % n_m).to(tl.int64)
    off_hz = (base // n_m).to(tl.int64)
    q_base = (off_hz // H) * stride_qb + (off_hz % H) * stride_qh + start_m * BLOCK_M * stride_qm
    qs_base = ((off_hz // H) * stride_qsb + (off_hz % H) * stride_qsh + start_m * BLOCK_M * stride_qsm) // 4
    k_ptr = K + (off_hz // H) * stride_kb + (off_hz % H) * stride_kh + k_off
    v_ptr = V + (off_hz // H) * stride_vb + (off_hz % H) * stride_vh + v_off
    ks_ptr = Ks.to(tl.pointer_type(tl.int32)) + ((off_hz // H) * stride_ksb + (off_hz % H) * stride_ksh) // 4 + ks_off
    vs_ptr = Vs.to(tl.pointer_type(tl.int32)) + ((off_hz // H) * stride_vsb + (off_hz % H) * stride_vsh) // 4 + ks_off
    _issue_q(Q + q_base + q_off, Qs.to(tl.pointer_type(tl.int32)) + qs_base + qs_off, q_n1, qa, qsa[0])
    tlx.async_load(k_ptr, ka[1])
    tlx.async_load(k_ptr + k_n1, kb[1])
    tlx.async_load(ks_ptr, ksa[1])
    tlx.async_load(ks_ptr + ks_n1, ksb[1])
    tlx.async_load_commit_group()
    _issue_tile(k_ptr, v_ptr, ks_ptr, vs_ptr, ka[0], kb[0], va[0], vb[0], ksa[0], ksb[0], vsa[0], vsb[0], 1, 0, k_step,
                v_step, ks_step, k_n1, v_n1, ks_n1, True, True)

    for w in tl.range(0, n_mine, num_stages=1):
        item = base + w * stride_item
        start_m = (item % n_m).to(tl.int64)
        off_hz = (item // n_m).to(tl.int64)
        off_z = off_hz // H
        off_h = off_hz % H
        k_ptr = K + off_z * stride_kb + off_h * stride_kh + k_off
        v_ptr = V + off_z * stride_vb + off_h * stride_vh + v_off
        ks_ptr = Ks.to(tl.pointer_type(tl.int32)) + (off_z * stride_ksb + off_h * stride_ksh) // 4 + ks_off
        vs_ptr = Vs.to(tl.pointer_type(tl.int32)) + (off_z * stride_vsb + off_h * stride_vsh) // 4 + ks_off

        tlx.async_load_wait_group(1)
        q_lo = tlx.local_load(qa[0])
        q_hi = tlx.local_load(qa[1])
        qs4 = tlx.local_reinterpret(qsa[0], tl.uint8, [BLOCK_M, 4])
        qs_lo = tlx.local_load(tlx.local_slice(qs4, [0, 0], [BLOCK_M, 2]))
        qs_hi = tlx.local_load(tlx.local_slice(qs4, [0, 2], [BLOCK_M, 2]))
        if FAST_EXP:
            # Scores in units of 2**-9 (_score_scale); E8M0 byte 0 is the floor.
            qs_lo = (tl.maximum(qs_lo, 9) - 9).to(tl.uint8)
            qs_hi = (tl.maximum(qs_hi, 9) - 9).to(tl.uint8)
        acc0 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc2 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        acc3 = tl.zeros([BLOCK_M, 32], dtype=tl.float32)
        s0, s1 = _qk_first(q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], qk_scale)
        p0, p1, m_i, l_i = _pingpong_softmax_first(s0, s1, FAST_EXP)
        tlx.async_load_wait_group(0)

        for it in tl.range(0, (n_tiles - 2) // 2, num_stages=1):
            # Step j = 2 * it: QK(j + 1) and PV(j) from buffer 0, tile j + 1 into buffer 1.
            with tlx.warp_pipeline_stage("mfma", priority=MFMA_PRIO):
                t = 2 * it + 2
                s0, s1, acc0, acc1, acc2, acc3, l_i = _mfma_stage(
                    q_lo, q_hi, qs_lo, qs_hi, ka[0], kb[0], ksa[0], ksb[0], va[0], vb[0], vsa[0], vsb[0], p0, p1, acc0,
                    acc1, acc2, acc3, l_i, (OFF - m_i) * c_qk, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, ka[1], kb[1],
                    va[1], vb[1], ksa[1], ksb[1], vsa[1], vsb[1], t, t - 1, k_step, v_step, ks_step, k_n1, v_n1, ks_n1,
                    True, True)
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
                p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3,
                                                                             qk_scale, FAST_EXP)
            # Step j + 1: same with the buffers swapped.
            with tlx.warp_pipeline_stage("mfma", priority=MFMA_PRIO):
                t = 2 * it + 3
                s0, s1, acc0, acc1, acc2, acc3, l_i = _mfma_stage(
                    q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], va[1], vb[1], vsa[1], vsb[1], p0, p1, acc0,
                    acc1, acc2, acc3, l_i, (OFF - m_i) * c_qk, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, ka[0], kb[0],
                    va[0], vb[0], ksa[0], ksb[0], vsa[0], vsb[0], t, t - 1, k_step, v_step, ks_step, k_n1, v_n1, ks_n1,
                    True, True)
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
                p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3,
                                                                             qk_scale, FAST_EXP)

        # Last step j = n_tiles - 2 from buffer 0; V(n_tiles - 1) goes to buffer 1.
        s0, s1, acc0, acc1, acc2, acc3, l_i = _mfma_stage(q_lo, q_hi, qs_lo, qs_hi, ka[0], kb[0], ksa[0], ksb[0], va[0],
                                                          vb[0], vsa[0], vsb[0], p0, p1, acc0, acc1, acc2, acc3, l_i,
                                                          (OFF - m_i) * c_qk, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr,
                                                          ka[1], kb[1], va[1], vb[1], ksa[1], ksb[1], vsa[1], vsb[1], 0,
                                                          n_tiles - 1, k_step, v_step, ks_step, k_n1, v_n1, ks_n1,
                                                          False, True)
        # Buffer 0 and buffer 1's K half are free now: copy the next item's Q
        # and first tiles behind the rest of this epilogue (the last item
        # re-copies its own, which nothing reads).
        nxt = tl.where(w + 1 < n_mine, item + stride_item, item)
        n_start = (nxt % n_m).to(tl.int64)
        n_hz = (nxt // n_m).to(tl.int64)
        nq_base = (n_hz // H) * stride_qb + (n_hz % H) * stride_qh + n_start * BLOCK_M * stride_qm
        nqs_base = ((n_hz // H) * stride_qsb + (n_hz % H) * stride_qsh + n_start * BLOCK_M * stride_qsm) // 4
        nk_ptr = K + (n_hz // H) * stride_kb + (n_hz % H) * stride_kh + k_off
        nv_ptr = V + (n_hz // H) * stride_vb + (n_hz % H) * stride_vh + v_off
        nks_ptr = Ks.to(tl.pointer_type(tl.int32)) + ((n_hz // H) * stride_ksb + (n_hz % H) * stride_ksh) // 4 + ks_off
        nvs_ptr = Vs.to(tl.pointer_type(tl.int32)) + ((n_hz // H) * stride_vsb + (n_hz % H) * stride_vsh) // 4 + ks_off
        _issue_q(Q + nq_base + q_off, Qs.to(tl.pointer_type(tl.int32)) + nqs_base + qs_off, q_n1, qa, qsa[0])
        tlx.async_load(nk_ptr, ka[1])
        tlx.async_load(nk_ptr + k_n1, kb[1])
        tlx.async_load(nks_ptr, ksa[1])
        tlx.async_load(nks_ptr + ks_n1, ksb[1])
        tlx.async_load_commit_group()
        _issue_tile(nk_ptr, nv_ptr, nks_ptr, nvs_ptr, ka[0], kb[0], va[0], vb[0], ksa[0], ksb[0], vsa[0], vsb[0], 1, 0,
                    k_step, v_step, ks_step, k_n1, v_n1, ks_n1, True, True)
        tlx.async_load_wait_group(3)
        p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale,
                                                                     FAST_EXP)
        acc0, acc1, acc2, acc3, l_i = _pv_last(p0, p1, va[1], vb[1], vsa[1], vsb[1], acc0, acc1, acc2, acc3, l_i)

        scale = tlx.require_layout(1.0 / tl.sum(l_i, 1),
                                   tlx.slice_layout(tlx.amd_mfma_layout(4, [32, 32, 64], True, [8, 1]), 1))[:, None]
        offs_c = tl.arange(0, 32)
        o_rows = Out + off_z * stride_ob + off_h * stride_oh + (start_m * BLOCK_M + offs_mm[:, None]) * stride_om
        tl.store(o_rows + offs_c[None, :] * stride_od, (acc0 * scale).to(tl.bfloat16))
        tl.store(o_rows + (offs_c[None, :] + 32) * stride_od, (acc1 * scale).to(tl.bfloat16))
        tl.store(o_rows + (offs_c[None, :] + 64) * stride_od, (acc2 * scale).to(tl.bfloat16))
        tl.store(o_rows + (offs_c[None, :] + 96) * stride_od, (acc3 * scale).to(tl.bfloat16))
    tlx.async_load_wait_group(0)


@triton.jit
def _quantize_mxfp8_v_kernel(V, Out, Scale, stride_vb, stride_vh, stride_vn, stride_vd, H, N_CTX,
                             HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr):
    # V to MXFP8 along the keys (a 32-key block per column), by
    # _quantize_mxfp8_kernel's rule. TRANSPOSED writes quantize_mxfp8_v's
    # transposed layout; otherwise [key, head] data and [key block, head]
    # scales.
    pid_bh = tl.program_id(1).to(tl.int64)
    offs_n = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    if TRANSPOSED:
        src_n = (offs_n & ~20) | (((offs_n >> 2) & 1) << 4) | (((offs_n >> 4) & 1) << 2)
    else:
        src_n = offs_n
    rows = src_n[:, None] < N_CTX
    x = tl.load(
        V + (pid_bh // H) * stride_vb + (pid_bh % H) * stride_vh + src_n[:, None] * stride_vn +
        offs_d[None, :] * stride_vd, mask=rows, other=0.0).to(tl.float32)
    # The relabeling permutes keys within each 32-key block, so blocks keep their keys.
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM))
    byte = _mx_scale(tl.max(tl.abs(x), 1))
    inv = _decode_scale(byte, RECIPROCAL=True)
    q = tl.reshape(tl.clamp(x * inv[:, None, :], -448.0, 448.0).to(tl.float8e4nv), (BLOCK_N, HEAD_DIM))
    s = byte.to(tl.uint8)
    if TRANSPOSED:
        tl.store(Out + (pid_bh * HEAD_DIM + offs_d[None, :]) * N_CTX + offs_n[:, None], q, cache_modifier=".cs")
        s = tl.reshape(tl.permute(tl.reshape(s, (BLOCK_N // 32, HEAD_DIM // 32, 32)), (0, 2, 1)),
                       (BLOCK_N, HEAD_DIM // 32))
        offs_c = tl.arange(0, HEAD_DIM // 32)
        tl.store(Scale + (pid_bh * N_CTX + offs_n[:, None]) * (HEAD_DIM // 32) + offs_c[None, :], s,
                 cache_modifier=".cs")
    else:
        tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], q, mask=rows,
                 cache_modifier=".cs")
        offs_b = tl.program_id(0) * (BLOCK_N // 32) + tl.arange(0, BLOCK_N // 32)
        n_blocks = tl.cdiv(N_CTX, 32)
        blocks = offs_b[:, None] < n_blocks
        tl.store(Scale + (pid_bh * n_blocks + offs_b[:, None]) * HEAD_DIM + offs_d[None, :], s, mask=blocks,
                 cache_modifier=".cs")


def quantize_mxfp8_v(v, *, transposed=False):
    """E4M3 data and E8M0 scales of ``v`` along the sequence, a scale per 32
    keys of each head column: data ``[Z, H, N_CTX, HEAD_DIM]``, scales
    ``[Z, H, cdiv(N_CTX, 32), HEAD_DIM]``. ``transposed`` lays both out as the
    non-causal kernel reads them: the data as ``[Z, H, HEAD_DIM, N_CTX]`` with
    its keys in the order P takes for the PV MFMA (key-index bits 2 and 4
    swapped), the scales as ``[Z, H, N_CTX, HEAD_DIM // 32]`` where, per 64
    keys, word ``32 * b + j`` holds key block ``b``'s scales of columns ``j``,
    ``j + 32``, ``j + 64`` and ``j + 96``."""
    z, h, n, d = v.shape
    if transposed:
        if d != 128 or n % 64 != 0:
            raise ValueError("transposed V needs HEAD_DIM 128 and N_CTX % 64 == 0")
        out = torch.empty((z, h, d, n), device=v.device, dtype=_FP8)
        scale = torch.empty((z, h, n, d // 32), device=v.device, dtype=torch.uint8)
    else:
        out = torch.empty((z, h, n, d), device=v.device, dtype=_FP8)
        scale = torch.empty((z, h, triton.cdiv(n, 32), d), device=v.device, dtype=torch.uint8)
    _quantize_mxfp8_v_kernel[(triton.cdiv(n, 64), z * h)](v, out, scale, v.stride(0), v.stride(1), v.stride(2),
                                                          v.stride(3), h, n, HEAD_DIM=d, BLOCK_N=64,
                                                          TRANSPOSED=transposed, num_warps=1)
    return out, scale


def _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, *, fast_exp=True,
                      return_lse=False):
    """Launch on already-quantized inputs: ``q_fp8`` and ``q_scale`` from
    ``quantize_mxfp8_head(q)``, K from ``quantize_mxfp8_head(k,
    pack_k=pingpong)`` and V from ``quantize_mxfp8_v(v, transposed=pingpong)``,
    with ``pingpong`` from ``_default_config``. On the ping-pong path,
    ``fast_exp`` computes P with a linear-mantissa exp2 bit trick;
    ``fast_exp=False`` uses the hardware exp. ``return_lse=True`` selects
    the hardware-exp loop kernel and returns ``(output, base2_lse)``; it
    requires unpacked K scales and non-transposed V, regardless of causal.
    Nonpositive scales also require that loop-kernel layout."""
    if not math.isfinite(sm_scale):
        raise ValueError("gfx950 flash_attn_mxfp8 expects a finite sm_scale")
    batch, heads, n_ctx, _ = q_fp8.shape
    qk_scale = sm_scale * _LOG2E
    out = torch.empty(q_fp8.shape, device=q_fp8.device, dtype=torch.bfloat16)
    cfg = _default_config(causal, n_ctx, training=return_lse or sm_scale <= 0)
    if cfg["pingpong"]:
        # Each row's four E8M0 bytes are read as one int32, and V's scale words
        # are addressed with K's offsets.
        q_scale = q_scale.contiguous()
        v_scale = v_scale.contiguous()
        if v_scale.shape != k_scale.shape or v_scale.stride(2) != k_scale.stride(2):
            raise ValueError("V's scales must be laid out like K's")
        n_items = n_ctx // cfg["block_m"] * batch * heads
        grid = (min(n_items, _NUM_CUS), )
        args = (
            q_fp8,
            k_fp8,
            v_fp8,
            q_scale,
            k_scale,
            v_scale,
            out,
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
            v_fp8.stride(3),
            v_fp8.stride(2),
            q_scale.stride(0),
            q_scale.stride(1),
            q_scale.stride(2),
            k_scale.stride(0),
            k_scale.stride(1),
            k_scale.stride(2),
            v_scale.stride(0),
            v_scale.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            batch,
            heads,
            n_ctx,
            qk_scale,
        )
        kwargs = dict(
            HEAD_DIM=_HEAD_DIM,
            BLOCK_M=cfg["block_m"],
            BLOCK_N=cfg["block_n"],
            MFMA_PRIO=1,
            SOFTMAX_PRIO=0,
            FAST_EXP=fast_exp,
            num_warps=cfg["num_warps"],
            num_stages=1,
            # Accumulators and scores stay in arch VGPRs for the softmax VALU
            # work; the group-barrier scheduler interleaves LDS reads with MFMAs.
            # Without IEEE mode, max needs no canonicalizing v_max x, x of its
            # operands (NaN scores would still come out NaN).
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ("amdgpu-ieee", "false")),
            enable_sched_group_barrier_scheduler=True,
            sched_group_barrier_mfma_per_dwordx4=4,
        )
        _mxfp8_fa_fwd_pingpong[grid](*args, **kwargs)
        return out
    grid = (triton.cdiv(n_ctx, cfg["block_m"]), batch * heads)
    lse = torch.empty((batch, heads, n_ctx), device=q_fp8.device, dtype=torch.float32) if return_lse else None
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
        num_warps=cfg["num_warps"],
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-ieee", "false"), ),
    )
    return (out, lse) if return_lse else out


class _MXFP8Attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, causal):
        q_fp8, q_scale = quantize_mxfp8_head(q)
        k_fp8, k_scale = quantize_mxfp8_head(k)
        v_fp8, v_scale = quantize_mxfp8_v(v)
        out, lse = _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, return_lse=True)
        ctx.save_for_backward(q, k, v, q_fp8, k_fp8, out, lse, q_scale, k_scale)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        from .gfx950_bwd import launch_backward
        from .gfx950_bwd_shared import can_use_shared_square, launch_backward_shared_square
        from .gfx950_quant import quantize_backward_operands

        q, k, v, q_fp8, k_fp8, out, lse, q_scale, k_scale = ctx.saved_tensors
        # Autograd may execute after the caller changes the current device.
        with torch.cuda.device(q.device):
            do_bf16 = do.to(torch.bfloat16).contiguous()
            # Saved Q/K have unpacked square32 scales; O/LSE come from the
            # same hardware-exp training forward. Keep original BF16 V/dO for
            # backward preparation rather than requantizing saved FP8 data.
            shared_args = (q_fp8, k_fp8, q_scale, k_scale, v, do_bf16, out, lse, ctx.sm_scale)
            if can_use_shared_square(*shared_args, causal=ctx.causal):
                dq, dk, dv = launch_backward_shared_square(*shared_args, causal=ctx.causal)
                return dq, dk, dv, None, None
            (q_dk, q_dk_scale), (k_dq,
                                 k_dq_scale), (v_bwd,
                                               v_scale), (do_fp8,
                                                          do_scale), (do_dv, do_dv_scale) = (quantize_backward_operands(
                                                              q, k, v, do_bf16, transpose_storage=True))
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            delta = torch.empty_like(lse)
            launch_backward(do_fp8, do_dv, q_fp8, q_dk, k_fp8, k_dq, v_bwd, out, lse, q_scale, q_dk_scale, k_scale,
                            k_dq_scale, v_scale, do_scale, do_dv_scale, ctx.sm_scale, do_bf16, dq, dk, dv, delta,
                            causal=ctx.causal)
        return dq, dk, dv, None, None


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """BF16 MXFP8 attention. ``space`` selects no separate gfx950 config yet."""
    del space
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    if not math.isfinite(sm_scale):
        from triton.tlx.ops import InvalidInput

        raise InvalidInput("gfx950 flash_attn_mxfp8 expects a finite sm_scale")
    if q.dtype != torch.bfloat16:
        raise ValueError(f"gfx950 flash_attn_mxfp8 expects bf16, got {q.dtype}")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"gfx950 flash_attn_mxfp8 expects head dim {_HEAD_DIM}")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        return _MXFP8Attention.apply(q, k, v, sm_scale, causal)
    pingpong = _default_config(causal, q.shape[2], training=sm_scale <= 0)["pingpong"]
    q_fp8, q_scale = quantize_mxfp8_head(q)
    k_fp8, k_scale = quantize_mxfp8_head(k, pack_k=pingpong)
    v_fp8, v_scale = quantize_mxfp8_v(v, transposed=pingpong)
    return _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale)
