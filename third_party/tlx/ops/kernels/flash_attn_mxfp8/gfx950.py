"""gfx950 MXFP8 Flash Attention forward.

BF16 ``(Z, H, N_CTX, HEAD_DIM)`` in, BF16 out. When autograd records the
inputs, the call goes to the training forward in gfx950_train.py, which saves
what backward needs; this module's kernels run inference. The quantization
is Blackwell's: Q and K are E4M3 with an E8M0 scale per 32x32 block (32 rows
by 32 head elements), V is E4M3 with an E8M0 scale per 32 keys of each head
column, and the softmax probabilities P get an E8M0 scale per 32 keys of each
row, all by RCEIL of the block max. Q is quantized unscaled, and sm_scale *
log2(e) multiplies the scores. Both MFMAs are scaled. A call makes two
launches: one quantizes Q, K and V, the other runs the attention.

A persistent two-warp-group ping-pong kernel (256x128 tiles, 8 warps, Q/K/V
staged in LDS by direct copies, K/V double buffered) runs both cases. It feeds
the row offset in as the QK accumulator, computes P with a bit-trick exp2
instead of the hardware exp, and only rescales the accumulator when a row max
grows past a threshold. P's block scales are applied by its scaled FP8
conversion (v_cvt_scalef32_pk_fp8_bf16); its row sums run on the matrix cores
from the FP8 P per block and take the block scales on the vector ALUs. Causal
Q tiles stop at the diagonal, whose two K/V tiles start the QK accumulator at
-inf above it, and go to the workgroups in pairs of equal total length. For
sm_scale = 0, where the scale multiply would turn the row offset and the -inf
into NaN, a separately compiled variant applies both after the multiply.
Lengths the ping-pong tiles do not divide run a single software-pipelined
loop with the hardware exp, where P's block scales join the exp argument.

The bit-trick exp2 is specific to gfx950 (Blackwell uses the hardware exp2):
the hardware exp runs at a quarter of the VALU rate, in the softmax stage
that bounds the ping-pong kernel's speed. It gives 2**x with a linear
mantissa, up to 6.1% high and 4.1% high on average. P and its row sums share
that bias, so after the normalization P is between about 4% low and 2% high,
within the error of rounding P to E4M3 (up to 6.25%).
``_launch_quantized(..., fast_exp=False)`` runs the kernel with the hardware
exp instead.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

_FP8 = torch.float8_e4m3fn
_LOG2E = 1.4426950408889634
_HEAD_DIM = 128
_NUM_CUS = 256  # gfx950 as one whole device (compute partition SPX)


def _default_config(causal, n_ctx):
    # Measured on gfx950, B=4 H=32 D=128. The ping-pong kernel is built around
    # its tiles and 8 warps and pipelines by hand (no num_stages), so it takes
    # only the lengths its block_m divides, and causal ones only in pairs of Q
    # tiles; the loop kernel takes the rest.
    if n_ctx % (512 if causal else 256) == 0:
        return {"pingpong": True, "block_m": 256, "block_n": 128, "num_warps": 8}
    if causal:
        return {"pingpong": False, "block_m": 128, "block_n": 64, "num_warps": 4, "num_stages": 2}
    return {"pingpong": False, "block_m": 128, "block_n": 128, "num_warps": 4, "num_stages": 2}


@triton.jit
def _mx_scale(amax):
    # E8M0 byte of RCEIL(amax * fp32(1 / 448)), 0 for amax 0: the rule of
    # Blackwell's quantizers (cvt.rp.satfinite.ue8m0x2.f32) below its top byte,
    # which BF16 data and P's block maxes stay far from.
    return ((amax * (1.0 / 448.0)).to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23


@triton.jit
def _quantize_head(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM: tl.constexpr,
                   BLOCK_N: tl.constexpr, PACK_K: tl.constexpr):
    # Each 32x32 block of x (32 rows by 32 head elements) gets the E8M0 byte
    # _mx_scale(amax) for its max magnitude amax, stored for each of its rows,
    # and E4M3 data x / 2**(byte - 127). PACK_K writes the scales in
    # quantize_mxfp8_head's pack_k order. The outputs are read once, by the
    # attention kernel, so the stores stream past the caches.
    pid_bh = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_b = tl.arange(0, HEAD_DIM // 32)
    rows = offs_n[:, None] < N_CTX
    x = tl.load(
        X + (pid_bh // H) * stride_xz + (pid_bh % H) * stride_xh + offs_n[:, None] * stride_xn +
        offs_d[None, :] * stride_xd, mask=rows, other=0.0).to(tl.float32)
    x = tl.reshape(x, (BLOCK_N // 32, 32, HEAD_DIM // 32, 32))
    byte = _mx_scale(tl.max(tl.max(tl.abs(x), 3), 1))
    inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(x * inv[:, None, :, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(Out + (pid_bh * N_CTX + offs_n[:, None]) * HEAD_DIM + offs_d[None, :], tl.reshape(q, (BLOCK_N, HEAD_DIM)),
             mask=rows, cache_modifier=".cs")
    s = tl.reshape(tl.broadcast_to(byte[:, None, :], (BLOCK_N // 32, 32, HEAD_DIM // 32)), (BLOCK_N, HEAD_DIM // 32))
    s = s.to(tl.uint8)
    if PACK_K:
        s = tl.reshape(tl.permute(tl.reshape(s, (BLOCK_N // 64, 2, 32, 2, 2)), (0, 4, 2, 3, 1)), (BLOCK_N, 4))
    tl.store(Scale + (pid_bh * N_CTX + offs_n[:, None]) * (HEAD_DIM // 32) + offs_b[None, :], s, mask=rows,
             cache_modifier=".cs")


@triton.jit
def _quantize_mxfp8_kernel(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM: tl.constexpr,
                           BLOCK_N: tl.constexpr, PACK_K: tl.constexpr):
    _quantize_head(X, Out, Scale, stride_xz, stride_xh, stride_xn, stride_xd, H, N_CTX, HEAD_DIM, BLOCK_N, PACK_K)


def quantize_mxfp8_head(x, *, pack_k=False):
    """E4M3 data and E8M0 scales of a ``[Z, H, N_CTX, HEAD_DIM]`` tensor with
    a scale per 32x32 block (32 rows by 32 head elements), as Blackwell
    quantizes Q and K. The scales are ``[Z, H, N_CTX, HEAD_DIM // 32]``, each
    row holding its block's. ``pack_k`` orders K's scales as the ping-pong
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
):
    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, SCALE_K)
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N)
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
        if CAUSAL_MASK or MASK_N:
            valid = offs_m[:, None] >= offs_n[None, :] if CAUSAL_MASK else offs_m[:, None] < N_CTX
            if MASK_N:
                valid = valid & (offs_n[None, :] < N_CTX)
            if CAUSAL_MASK:
                valid = valid & (offs_m[:, None] < N_CTX)
            qk = tl.where(valid, qk, -1.0e6)
        # qk_scale takes the scores to log2 units: the block maxima are scaled
        # directly, each score inside the exp argument.
        qk = tl.reshape(qk, (BLOCK_M, BLOCK_N // 32, 32))
        tmax = tl.max(qk, 2) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(tmax, 1))
        # P to MXFP8: a block's max P is exp2(its max score - m).
        # The scale's exponent joins the exp argument, so P comes out divided by it.
        byte = _mx_scale(tl.math.exp2(tmax - m_ij[:, None]))
        p = tl.math.exp2(qk * qk_scale - (m_ij[:, None] + (byte - 127).to(tl.float32))[:, :, None])
        if CAUSAL_MASK or MASK_N:
            p = tl.where(tl.reshape(valid, (BLOCK_M, BLOCK_N // 32, 32)), p, 0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(tl.sum(p, 2) * (byte << 23).to(tl.float32, bitcast=True), 1)
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
):
    SCALE_K: tl.constexpr = HEAD_DIM // 32
    tl.static_assert(BLOCK_M % BLOCK_N == 0)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
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
                                 qk_scale, HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, False, False)
        diag_hi = tl.minimum(start_m * BLOCK_M + BLOCK_M, N_CTX)
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, start_m * BLOCK_M, diag_hi,
                                 N_CTX, qk_scale, HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, True, True)
    else:
        acc, l_i, m_i = _fa_loop(acc, l_i, m_i, q, q_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, stride_kn, stride_kd,
                                 stride_ksn, stride_vn, stride_vd, stride_vsn, offs_m, 0, N_CTX, N_CTX, qk_scale,
                                 HEAD_DIM, SCALE_K, BLOCK_M, BLOCK_N, NUM_STAGES, not EVEN_N, False)
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0)
    tl.store(Out + off_z * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
             out.to(tl.bfloat16), mask=row_mask[:, None])


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
    return tlx.require_layout(byte.to(tl.uint8), halves), (byte << 23).to(tl.float32, bitcast=True)


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
    f0 = (p0[1].to(tl.int32) << 23).to(tl.float32, bitcast=True)
    f1 = (p1[1].to(tl.int32) << 23).to(tl.float32, bitcast=True)
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
def _causal_init(init, KEY_BASE: tl.constexpr):
    # The QK accumulator start for keys KEY_BASE + [0, 64) of a diagonal K/V
    # tile: row r of the Q tile sees key k of its own span iff k <= r, so the
    # rest start at -inf, which gives P = 0 and leaves the row maxima to the
    # visible keys. Masking the start rather than the scores keeps the scores'
    # path to the PV MFMA free of branches, which the MFMA layout choice for
    # the QK dot depends on.
    return _causal_mask(init, tl.arange(0, init.shape[0]), KEY_BASE)


@triton.jit
def _causal_mask(x, lim, KEY_BASE: tl.constexpr):
    # x with -inf where key KEY_BASE + c of its 64 keys (counted in the Q
    # tile's span) is past row r's last visible key lim[r].
    # Key 32t + 8j + 4h + i sits in register (t, j, i) of lane half h (see
    # _block_max), so each test is a register's constant against one per-lane
    # limit. KEY_BASE reaches the limit through an asm LLVM cannot see into:
    # from a constant, the tests would be hoisted out of the loops and their
    # lane masks spill.
    R: tl.constexpr = x.shape[0]
    base = tl.inline_asm_elementwise("s_mov_b32 $0, $1", "=s,s", [KEY_BASE], dtype=tl.int32, is_pure=False, pack=1)
    t = tl.arange(0, 2)[None, :, None, None, None]
    j = tl.arange(0, 4)[None, None, :, None, None]
    h = tl.arange(0, 2)[None, None, None, :, None]
    i = tl.arange(0, 4)[None, None, None, None, :]
    lim = lim[:, None, None, None, None] - 4 * h - base
    m = tl.where(32 * t + 8 * j + i <= lim, tl.reshape(x, (R, 2, 4, 2, 4)), float("-inf"))
    return tl.reshape(m, (R, x.shape[1]))


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
def _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3, qk_scale, FAST_EXP: tl.constexpr,
                      ZERO_SCALE: tl.constexpr):
    # s0, s1 carry the offset for m_i from the QK accumulator, and s0 comes
    # multiplied by qk_scale (with ZERO_SCALE, s1 too; see _mfma_stage);
    # acc0..acc3 are the 32-column chunks of the output accumulator and l_i
    # the row sums per block half of _row_sums.
    #
    # Keep m_i until some row in this warp grows past it by more than 2**4,
    # i.e. t > (OFF + 4) * C. The vote predicate is pinned to the layout the
    # block maxima leave it in (row r on lanes r % 32 and r % 32 + 32 of warp
    # r // 32); otherwise warp_any gets a default layout and a shuffle.
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    vote_layout: tl.constexpr = tlx.layout(shape=((32, 2, s0.shape[0] // 32), (1, )), stride=((1, 0, 32), (1, )))
    if not ZERO_SCALE:
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
                vt, k_step, v_step, ks_step, k_n1, v_n1, ks_n1, COPY_K: tl.constexpr, COPY_V: tl.constexpr, diag,
                KEY_BASE: tl.constexpr, ZERO_SCALE: tl.constexpr, MASK: tl.constexpr):
    # QK(j + 1) and PV(j) with operands read from LDS in MFMA-sized pieces. Each
    # read is issued right after the MFMA that frees the previous piece and is
    # consumed one MFMA later, so its latency hides behind an MFMA while at most
    # one K half and one V chunk are live. The scheduling barriers keep LLVM
    # from sinking the reads to their uses. A per-row offset, divided by
    # qk_scale, starts the QK accumulator. s0 leaves multiplied by qk_scale,
    # behind the last PV MFMAs, and s1 is multiplied in the softmax stage:
    # split between the two stages, the multiplies cost the least.
    #
    # With ZERO_SCALE (qk_scale = 0) the offset comes undivided and the
    # accumulator starts at 0: both halves are multiplied here, the offset
    # joins by an FMA, and where MASK (a causal step) the keys diag hides go
    # to -inf after it. Their per-row limit, past every key off the diagonal,
    # keeps no masked tile live across the MFMAs.
    #
    # P's row sums go first: they need only registers, so the matrix cores get
    # work right after the stage barrier, and the first reads and the next
    # tile's global->LDS copies (those COPY_K / COPY_V select) issue behind
    # them. Their block scales are applied after the first QK MFMA, when the
    # sums are in. diag masks K/V tile j + 1, keys KEY_BASE + [0, 128) of the
    # Q tile's span (see _causal_init).
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
    if ZERO_SCALE:
        init0 = tl.zeros_like(init)
    else:
        init0 = init
        if diag:
            init0 = _causal_init(init, KEY_BASE)
    s0 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kA0, sA0, "e4m3", acc=init0, fast_math=True)
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
    if ZERO_SCALE:
        init1 = tl.zeros_like(init)
    else:
        init1 = init
        if diag:
            init1 = _causal_init(init, KEY_BASE + 64)
    s1 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", acc=init1, fast_math=True)
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
    if ZERO_SCALE:
        s0 = tl.fma(s0, qk_scale, init)
        s1 = tl.fma(s1, qk_scale, init)
        if MASK:
            lim = tl.where(diag, tl.arange(0, q_lo.shape[0]), 1 << 20)
            s0 = _causal_mask(s0, lim, KEY_BASE)
            s1 = _causal_mask(s1, lim, KEY_BASE + 64)
    else:
        s0 = s0 * qk_scale
    return s0, s1, acc0, acc1, acc2, acc3, l_i


@triton.jit
def _qk_first(q_lo, q_hi, qs_lo, qs_hi, ka, kb, ksa, ksb, qk_scale, diag, ZERO_SCALE: tl.constexpr):
    # QK(0); diag masks it as the diagonal tile of Q tile 0. With ZERO_SCALE
    # the start (0, or -inf on the diagonal) is added after qk_scale instead.
    kA0 = _ld_k(ka, 0)
    kA1 = _ld_k(ka, 1)
    kB0 = _ld_k(kb, 0)
    kB1 = _ld_k(kb, 1)
    sA0, sA1 = _ld_ks(ksa)
    sB0, sB1 = _ld_ks(ksb)
    init0 = tl.zeros([q_lo.shape[0], ka.shape[1]], dtype=tl.float32)
    init1 = init0
    if diag:
        init0 = _causal_init(init0, 0)
        init1 = _causal_init(init1, 64)
    s0 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kA0, sA0, "e4m3", acc=tl.zeros_like(init0) if ZERO_SCALE else init0,
                       fast_math=True)
    s0 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kA1, sA1, "e4m3", acc=s0, fast_math=True)
    s1 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", acc=tl.zeros_like(init1) if ZERO_SCALE else init1,
                       fast_math=True)
    s1 = tl.dot_scaled(q_hi, qs_hi, "e4m3", kB1, sB1, "e4m3", acc=s1, fast_math=True)
    s0 = s0 * qk_scale
    s1 = s1 * qk_scale
    if ZERO_SCALE:
        s0 = s0 + init0
        s1 = s1 + init1
    return s0, s1


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
def _item(base, w, stride, n_m, CAUSAL: tl.constexpr):
    # A workgroup's w-th (Q tile, batch * head) item. Causal items come in
    # pairs, Q tiles n_m - 1 - i and i of one head, the long one first: every
    # pair walks 2 * n_m + 2 K/V tiles, so equal pair counts balance the load.
    if CAUSAL:
        unit = base + (w // 2) * stride
        i = unit % (n_m // 2)
        start_m = tl.where(w % 2 == 0, n_m - 1 - i, i)
        off_hz = unit // (n_m // 2)
    else:
        item = base + w * stride
        start_m = item % n_m
        off_hz = item // n_m
    return start_m, off_hz


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
    CAUSAL: tl.constexpr,
    ZERO_SCALE: tl.constexpr,
):
    # N_CTX % (2 * BLOCK_N) == 0 (causal: % (2 * BLOCK_M)), contiguous [row, 4]
    # Q scales, K from quantize_mxfp8_head(pack_k=True), V and its scales from
    # quantize_mxfp8_v(transposed=True) (V's strides given as for [key, head],
    # its scales strided like K's).
    # Persistent: each workgroup loops over (Q tile, head) items. Two warp
    # groups run one stage apart: one issues QK(j + 1) and PV(j) while the
    # other does softmax(j + 1). Each K/V tile is two key
    # halves in LDS, double buffered with compile-time buffer indices; tile
    # j + 1 is copied during step j. Q and its scales are staged in LDS as two
    # head-dim halves; the next item's Q and first K/V tiles are copied during
    # this item's epilogue. Causal Q tile m walks K/V tiles 0 .. 2m + 1, the
    # last two masked.
    HN: tl.constexpr = BLOCK_N // 2
    HD: tl.constexpr = HEAD_DIM // 2
    tl.static_assert(HN == 64 and HEAD_DIM == 128)
    tl.static_assert(BLOCK_M == 2 * BLOCK_N)
    pid = tl.program_id(0)
    n_ctas = tl.num_programs(0)
    n_m = N_CTX // BLOCK_M
    # Work units: items, or causal pairs of items (see _item).
    n_units = (n_m // 2 if CAUSAL else n_m) * H * Z
    # Workgroups go round-robin to the 8 XCDs. Give each XCD a contiguous run
    # of units so the Q tiles sharing a K/V stream share that XCD's L2.
    if (n_ctas % 8 == 0) and (n_units % 8 == 0):
        per_xcd = n_units // 8
        stride_item = n_ctas // 8
        base = (pid % 8) * per_xcd + pid // 8
        n_mine = (per_xcd - pid // 8 + stride_item - 1) // stride_item
    else:
        stride_item = n_ctas
        base = pid
        n_mine = (n_units - pid + n_ctas - 1) // n_ctas
    if CAUSAL:
        n_mine = 2 * n_mine

    offs_d = tl.arange(0, HEAD_DIM)
    offs_hd = tl.arange(0, HD)
    offs_n = tl.arange(0, HN)
    offs_mm = tl.arange(0, BLOCK_M)
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    # The row offset starts the QK accumulator, before the scores' qk_scale;
    # with ZERO_SCALE it is added after it (see _mfma_stage).
    c_qk = C if ZERO_SCALE else C / qk_scale

    k_off = offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
    v_off = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    ks_off = offs_n[:, None] * stride_ksn // 4
    q_off = offs_mm[:, None] * stride_qm + offs_hd[None, :] * stride_qd
    qs_off = offs_mm[:, None] * stride_qsm // 4
    k_step = BLOCK_N * stride_kn
    v_step = BLOCK_N * stride_vn
    ks_step = BLOCK_N * stride_ksn // 4
    k_n1 = HN * stride_kn
    v_n1 = HN * stride_vn
    ks_n1 = HN * stride_ksn // 4
    q_n1 = HD * stride_qd

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
    start_m, off_hz = _item(base, 0, stride_item, n_m, CAUSAL)
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
        start_m, off_hz = _item(base, w, stride_item, n_m, CAUSAL)
        n_tiles = 2 * start_m + 2 if CAUSAL else N_CTX // BLOCK_N
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
        s0, s1 = _qk_first(q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], qk_scale, CAUSAL and start_m == 0,
                           ZERO_SCALE)
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
                    True, True, False, 0, ZERO_SCALE, False)
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
                p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3,
                                                                             qk_scale, FAST_EXP, ZERO_SCALE)
            # Step j + 1: same with the buffers swapped.
            with tlx.warp_pipeline_stage("mfma", priority=MFMA_PRIO):
                t = 2 * it + 3
                s0, s1, acc0, acc1, acc2, acc3, l_i = _mfma_stage(
                    q_lo, q_hi, qs_lo, qs_hi, ka[1], kb[1], ksa[1], ksb[1], va[1], vb[1], vsa[1], vsb[1], p0, p1, acc0,
                    acc1, acc2, acc3, l_i, (OFF - m_i) * c_qk, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr, ka[0], kb[0],
                    va[0], vb[0], ksa[0], ksb[0], vsa[0], vsb[0], t, t - 1, k_step, v_step, ks_step, k_n1, v_n1, ks_n1,
                    True, True, CAUSAL and it == start_m - 1, 0, ZERO_SCALE, CAUSAL)
            tlx.async_load_wait_group(0)
            with tlx.warp_pipeline_stage("softmax", priority=SOFTMAX_PRIO):
                p0, p1, m_i, l_i, acc0, acc1, acc2, acc3 = _pingpong_softmax(s0, s1, m_i, l_i, acc0, acc1, acc2, acc3,
                                                                             qk_scale, FAST_EXP, ZERO_SCALE)

        # Last step j = n_tiles - 2 from buffer 0; V(n_tiles - 1) goes to buffer 1.
        s0, s1, acc0, acc1, acc2, acc3, l_i = _mfma_stage(q_lo, q_hi, qs_lo, qs_hi, ka[0], kb[0], ksa[0], ksb[0], va[0],
                                                          vb[0], vsa[0], vsb[0], p0, p1, acc0, acc1, acc2, acc3, l_i,
                                                          (OFF - m_i) * c_qk, qk_scale, k_ptr, v_ptr, ks_ptr, vs_ptr,
                                                          ka[1], kb[1], va[1], vb[1], ksa[1], ksb[1], vsa[1], vsb[1], 0,
                                                          n_tiles - 1, k_step, v_step, ks_step, k_n1, v_n1, ks_n1,
                                                          False, True, CAUSAL, BLOCK_N, ZERO_SCALE, CAUSAL)
        # Buffer 0 and buffer 1's K half are free now: copy the next item's Q
        # and first tiles behind the rest of this epilogue (the last item
        # re-copies its own, which nothing reads).
        n_start, n_hz = _item(base, tl.minimum(w + 1, n_mine - 1), stride_item, n_m, CAUSAL)
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
                                                                     FAST_EXP, ZERO_SCALE)
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
def _quantize_v(V, Out, Scale, stride_vb, stride_vh, stride_vn, stride_vd, H, N_CTX, HEAD_DIM: tl.constexpr,
                BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr):
    # V to MXFP8 along the keys (a 32-key block per column), by
    # _quantize_head's rule. TRANSPOSED writes quantize_mxfp8_v's
    # transposed layout; otherwise [key, head] data and [key block, head]
    # scales.
    pid_bh = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
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
    inv = ((254 - byte) << 23).to(tl.float32, bitcast=True)
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


@triton.jit
def _quantize_mxfp8_v_kernel(V, Out, Scale, stride_vb, stride_vh, stride_vn, stride_vd, H, N_CTX,
                             HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr):
    _quantize_v(V, Out, Scale, stride_vb, stride_vh, stride_vn, stride_vd, H, N_CTX, HEAD_DIM, BLOCK_N, TRANSPOSED)


@triton.jit
def _quantize_mxfp8_qkv_kernel(Q, K, V, Q_out, Q_scale, K_out, K_scale, V_out, V_scale, stride_qz, stride_qh, stride_qn,
                               stride_qd, stride_kz, stride_kh, stride_kn, stride_kd, stride_vz, stride_vh, stride_vn,
                               stride_vd, H, N_CTX, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                               PINGPONG: tl.constexpr):
    # The three quantizers in one launch, picked by the grid's third axis: each
    # launch costs more host time than the quantizers take on the GPU.
    which = tl.program_id(2)
    if which == 0:
        _quantize_head(Q, Q_out, Q_scale, stride_qz, stride_qh, stride_qn, stride_qd, H, N_CTX, HEAD_DIM, BLOCK_N,
                       False)
    elif which == 1:
        _quantize_head(K, K_out, K_scale, stride_kz, stride_kh, stride_kn, stride_kd, H, N_CTX, HEAD_DIM, BLOCK_N,
                       PINGPONG)
    else:
        _quantize_v(V, V_out, V_scale, stride_vz, stride_vh, stride_vn, stride_vd, H, N_CTX, HEAD_DIM, BLOCK_N,
                    PINGPONG)


def quantize_mxfp8_v(v, *, transposed=False):
    """E4M3 data and E8M0 scales of ``v`` along the sequence, a scale per 32
    keys of each head column: data ``[Z, H, N_CTX, HEAD_DIM]``, scales
    ``[Z, H, cdiv(N_CTX, 32), HEAD_DIM]``. ``transposed`` lays both out as the
    ping-pong kernel reads them: the data as ``[Z, H, HEAD_DIM, N_CTX]`` with
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


def _quantize_qkv(q, k, v, pingpong):
    """``quantize_mxfp8_head(q)``, ``quantize_mxfp8_head(k, pack_k=pingpong)``
    and ``quantize_mxfp8_v(v, transposed=pingpong)`` in one launch, as
    ``(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)``."""
    z, h, n, d = q.shape
    q_fp8 = torch.empty((z, h, n, d), device=q.device, dtype=_FP8)
    k_fp8 = torch.empty((z, h, n, d), device=q.device, dtype=_FP8)
    q_scale = torch.empty((z, h, n, d // 32), device=q.device, dtype=torch.uint8)
    k_scale = torch.empty((z, h, n, d // 32), device=q.device, dtype=torch.uint8)
    if pingpong:
        v_fp8 = torch.empty((z, h, d, n), device=q.device, dtype=_FP8)
        v_scale = torch.empty((z, h, n, d // 32), device=q.device, dtype=torch.uint8)
    else:
        v_fp8 = torch.empty((z, h, n, d), device=q.device, dtype=_FP8)
        v_scale = torch.empty((z, h, triton.cdiv(n, 32), d), device=q.device, dtype=torch.uint8)
    # One warp suits V's per-column blocks; the head quantizer runs as fast on one.
    _quantize_mxfp8_qkv_kernel[(triton.cdiv(n, 64), z * h, 3)](q, k, v, q_fp8, q_scale, k_fp8, k_scale, v_fp8, v_scale,
                                                               *q.stride(), *k.stride(), *v.stride(), h, n, HEAD_DIM=d,
                                                               BLOCK_N=64, PINGPONG=pingpong, num_warps=1)
    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


def _launch_quantized(q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale, causal, sm_scale, *, fast_exp=True):
    """Launch on already-quantized inputs: ``q_fp8`` and ``q_scale`` from
    ``quantize_mxfp8_head(q)``, K from ``quantize_mxfp8_head(k,
    pack_k=pingpong)`` and V from ``quantize_mxfp8_v(v, transposed=pingpong)``,
    with ``pingpong`` from ``_default_config``. On the ping-pong path,
    causal or not, ``fast_exp`` computes P with a linear-mantissa exp2 bit
    trick; ``fast_exp=False`` uses the hardware exp."""
    batch, heads, n_ctx, _ = q_fp8.shape
    qk_scale = sm_scale * _LOG2E
    out = torch.empty(q_fp8.shape, device=q_fp8.device, dtype=torch.bfloat16)
    cfg = _default_config(causal, n_ctx)
    if cfg["pingpong"]:
        # Each row's four E8M0 bytes are read as one int32, and V's scale words
        # are addressed with K's offsets.
        q_scale = q_scale.contiguous()
        v_scale = v_scale.contiguous()
        if v_scale.shape != k_scale.shape or v_scale.stride(2) != k_scale.stride(2):
            raise ValueError("V's scales must be laid out like K's")
        n_units = n_ctx // cfg["block_m"] // (2 if causal else 1) * batch * heads
        grid = (min(n_units, _NUM_CUS), )
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
            CAUSAL=causal,
            ZERO_SCALE=sm_scale == 0,
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
    v_scale = v_scale.contiguous()
    _mxfp8_fa_fwd[grid](
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
        num_warps=cfg["num_warps"],
        num_stages=1,
        llvm_fn_attrs=(("amdgpu-ieee", "false"), ),
    )
    return out


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full", p_expcast=False):
    """BF16 MXFP8 attention. ``space`` selects no separate gfx950 config yet."""
    del space
    if p_expcast:
        raise ValueError("gfx950 flash_attn_mxfp8 does not support p_expcast")
    if sm_scale is None:
        sm_scale = q.shape[-1]**-0.5
    if q.dtype != torch.bfloat16:
        raise ValueError(f"gfx950 flash_attn_mxfp8 expects bf16, got {q.dtype}")
    if q.shape[-1] != _HEAD_DIM:
        raise ValueError(f"gfx950 flash_attn_mxfp8 expects head dim {_HEAD_DIM}")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        # Training saves softmax statistics for backward (see gfx950_train.py).
        from .gfx950_train import flash_attn_mxfp8_train

        return flash_attn_mxfp8_train(q, k, v, causal, sm_scale)
    pingpong = _default_config(causal, q.shape[2])["pingpong"]
    return _launch_quantized(*_quantize_qkv(q, k, v, pingpong), causal, sm_scale)
