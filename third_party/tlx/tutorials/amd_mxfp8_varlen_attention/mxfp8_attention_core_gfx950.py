# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Scaled-MFMA MXFP8 attention building blocks for gfx950 (from D122867768).

One warp per 32 query rows (eight warps, 256 rows in D122867768); each 128-key tile is two 64-key halves.
Q and K carry one E8M0 scale per 32x32 (row, feature) tile, V one per
feature and 32 keys, P one per row and 32 keys. Two warp groups ping-pong:
one runs QK(j + 1) and PV(j) on the matrix cores while the other runs the
softmax of tile j + 1.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _mx_scale(amax):
    # E8M0 byte of RCEIL(amax * fp32(1 / 448)), 0 for amax 0: the rule of
    # Blackwell's quantizers (cvt.rp.satfinite.ue8m0x2.f32) below its top byte,
    # which BF16 data and P's block maxes stay far from.
    return ((amax * (1.0 / 448.0)).to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23


_K_LDS_BASES = tl.constexpr(
    [
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 0],
        [16, 0],
        [32, 0],
        [64, 0],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 1],
        [0, 2],
        [0, 4],
    ]
)
_V_LDS_BASES = tl.constexpr(
    [
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 0],
        [16, 0],
        [32, 0],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 1],
        [0, 2],
        [0, 64],
    ]
)


@triton.jit
def _exp2_bf16_bits(t):
    # t = (x + 127) * 128 / 65535. v_cvt_pknorm_u16_f32 rounds clamp(t, 0, 1) *
    # 65535, the bf16 bit pattern of 2**x with a linear mantissa (at most 6.1%
    # high), two per register.
    return tl.inline_asm_elementwise(
        "v_cvt_pknorm_u16_f32 $0, $1, $2",
        "=v,v,v",
        [t],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


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
    return tl.reshape(
        tl.broadcast_to(x[:, :, None], (x.shape[0], 2, 32)), (x.shape[0], 64)
    )


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
    halves: tl.constexpr = tlx.layout(
        shape=((32, 2, tmax.shape[0] // 32), (1,)), stride=((2, 1, 64), (1,))
    )
    return tlx.require_layout(byte.to(tl.uint8), halves), (byte << 23).to(
        tl.float32, bitcast=True
    )


@triton.jit
def _to_e4m3(p, scale):
    # E4M3 of bf16 P divided by 2**(e - 127), e the exponent field of scale:
    # v_cvt_scalef32_pk_fp8_bf16 converts two keys, so two fill a register of
    # four. The asm returns that register as two u16 (AMDGPU inline asm has no
    # register type for four i8); its second output is unused.
    w = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_bf16 $0, $2, $4\n"
        "v_cvt_scalef32_pk_fp8_bf16 $0, $3, $6 op_sel:[0,0,1]",
        "=&v,=v,v,v,v,v,v,v",
        [p, scale],
        dtype=tl.uint16,
        is_pure=True,
        pack=4,
    )
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
    return tl.where(2 * ((k >> 6) & 1) + ((k >> 4) & 1) == (j >> 2), 1.0, 0.0).to(
        tl.float8e4nv
    )


@triton.jit
def _block_sums(t):
    # _rowsum_b's [16w + i, 4 * (2b + q) + r] as [row 32w + 16q + i, block b].
    # In the transposed 16x16 MFMA layout lane l holds A' row l % 16 and column
    # group l // 16, so row 32w + l % 32 of block l // 32 is already in its
    # lane; pinned there (left free, the multiply by the scales moves the
    # block into registers with a lane exchange).
    halves: tl.constexpr = tlx.layout(
        shape=((32, 2, t.shape[0] // 16), (1,)), stride=((2, 1, 64), (1,))
    )
    x = tl.permute(
        tl.reshape(t, (t.shape[0] // 16, 16, 2, 2, 2, 2)), (0, 3, 1, 2, 4, 5)
    )
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
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
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
def _pingpong_softmax(s0, s1, mi, li, a0, a1, a2, a3, qk_scale, FAST_EXP: tl.constexpr):
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    vote: tl.constexpr = tlx.layout(
        shape=((32, 2, s0.shape[0] // 32), (1,)), stride=((1, 0, 32), (1,))
    )
    s1 = s1 * qk_scale
    tmax0, tmax1 = _block_max(s0), _block_max(s1)
    row = tl.maximum(tl.max(tmax0, 1), tl.max(tmax1, 1))
    tshift = tl.zeros_like(row)
    if tlx.warp_any(tlx.require_layout(row > (OFF + 4.0) * C, vote)):
        # Preserve the original rescaling threshold and score reduction.
        up = tl.maximum(tl.maximum(tl.max(s0, 1), tl.max(s1, 1)) - OFF * C, 0.0)
        alpha = tl.math.exp2(up * (-1.0 / C))[:, None]
        a0, a1, a2, a3 = a0 * alpha, a1 * alpha, a2 * alpha, a3 * alpha
        s0, s1 = s0 - up[:, None], s1 - up[:, None]
        mi = mi + up * (1.0 / C)
        tshift = -tl.maximum(row - OFF * C, 0.0)
        # The non-rescaling branch has tshift==0, so its old multiply is identity.
        li = li * tl.math.exp2(tshift * (1.0 / C))[:, None]
    p0, p1 = (
        _p_half(s0, tmax0 + tshift[:, None], FAST_EXP),
        _p_half(s1, tmax1 + tshift[:, None], FAST_EXP),
    )
    return p0, p1, mi, li, a0, a1, a2, a3


@triton.jit
def _quantize_q_half(values):
    rows: tl.constexpr = values.shape[0]
    grouped = tl.reshape(values, (rows // 32, 32, 2, 32))
    magnitude = (grouped.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.uint16)
    maximum_bits = tl.max(tl.max(magnitude, 3), 1)
    maximum = maximum_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    byte = _mx_scale(maximum)
    operand_scale = (byte << 23).to(tl.float32, bitcast=True)
    operand_scale = tl.broadcast_to(
        operand_scale[:, None, :, None], (rows // 32, 32, 2, 32)
    ).reshape((rows, 64))
    quantized = _to_e4m3(values, operand_scale)
    scale = tl.reshape(
        tl.broadcast_to(byte.to(tl.uint8)[:, None, :], (rows // 32, 32, 2)), (rows, 2)
    )
    q_layout: tl.constexpr = tlx.dot_operand_layout(
        0, tlx.amd_mfma_layout(4, [32, 32, 64], True, [tlx.num_warps(), 1]), k_width=16
    )
    s_layout: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), (1,)), stride=((2, 1, 64), (1,))
    )
    return (
        tlx.require_layout(tl.reshape(quantized, (rows, 64)), q_layout),
        tlx.require_layout(scale, s_layout),
    )


@triton.jit
def _issue_tile(
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    ka,
    kb,
    va,
    vb,
    ksa,
    ksb,
    vsa,
    vsb,
    kt,
    vt,
    k_step,
    v_step,
    ks_step,
    k_n1,
    v_n1,
    ks_n1,
    COPY_K: tl.constexpr,
    COPY_V: tl.constexpr,
):
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
    return tlx.local_load(
        tlx.local_slice(kbuf, [64 * half, 0], [64, kbuf.shape[1]]), relaxed=True
    )


@triton.jit
def _ld_ks(ksbuf):
    # The E8M0 scales of 64 keys in the packed K order: lane l's word holds
    # keys l % 32 and l % 32 + 32 at d-block l // 32 of both head-dim halves,
    # so one word per lane feeds both QK MFMA pairs. Returns the scales of the
    # low and high head-dim halves, [key, block] each.
    view: tl.constexpr = tlx.shared_linear_layout_encoding(
        [
            [32, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [8, 0, 0],
            [16, 0, 0],
            [0, 1, 0],
        ]
    )
    lanes: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), (2, 2)), stride=((4, 2, 0), (128, 1))
    )
    ks = tlx.local_reinterpret(ksbuf, tl.uint8, [64, 2, 2], layout=view)
    return tl.split(tlx.local_load(ks, layout=lanes, relaxed=True))


@triton.jit
def _ld_v(vbuf, c: tl.constexpr):
    return tlx.local_load(
        tlx.local_slice(vbuf, [0, 32 * c], [vbuf.shape[0], 32]), relaxed=True
    )


@triton.jit
def _ld_vs(vsbuf):
    # V's E8M0 scales for 64 keys in the transposed V order: lane l's word
    # holds column l % 32 of key block l // 32 for each 32-column V chunk, the
    # lane order the scaled PV MFMA takes. Returns one [column, block] scale
    # per chunk.
    view: tl.constexpr = tlx.shared_linear_layout_encoding(
        [
            [0, 0, 1],
            [0, 0, 2],
            [1, 0, 0],
            [2, 0, 0],
            [4, 0, 0],
            [8, 0, 0],
            [16, 0, 0],
            [0, 1, 0],
        ]
    )
    lanes: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), (4,)), stride=((8, 4, 0), (1,))
    )
    vs = tlx.local_load(
        tlx.local_reinterpret(vsbuf, tl.uint8, [32, 2, 4], layout=view),
        layout=lanes,
        relaxed=True,
    )
    even, odd = tl.split(tl.reshape(vs, (32, 2, 2, 2)))
    s0, s2 = tl.split(even)
    s1, s3 = tl.split(odd)
    return s0, s1, s2, s3


@triton.jit
def _pv(p, v, vs, acc):
    # p is _p_half's (E4M3, E8M0) pair.
    return tl.dot_scaled(p[0], p[1], "e4m3", v, vs, "e4m3", acc=acc, fast_math=True)


@triton.jit
def _mfma_stage(
    q_lo,
    q_hi,
    qs_lo,
    qs_hi,
    ka,
    kb,
    ksa,
    ksb,
    va,
    vb,
    vsa,
    vsb,
    p0,
    p1,
    acc0,
    acc1,
    acc2,
    acc3,
    l_i,
    offset,
    qk_scale,
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    kd_a,
    kd_b,
    vd_a,
    vd_b,
    ksd_a,
    ksd_b,
    vsd_a,
    vsd_b,
    kt,
    vt,
    k_step,
    v_step,
    ks_step,
    k_n1,
    v_n1,
    ks_n1,
    COPY_K: tl.constexpr,
    COPY_V: tl.constexpr,
    rem=None,
):
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
    #
    # With rem, keys of the QK tile at or past rem start at -1e30 or less
    # (arithmetic: selects would take an SGPR pair per element).
    init = tl.broadcast_to(offset[:, None], (q_lo.shape[0], ka.shape[1]))
    init_b = init
    if rem is not None:
        # Integer first: LLVM then hoists only the lane's base key.
        keys = (rem - 1) - tl.arange(0, ka.shape[1])[None, :]
        init_b = init + tl.minimum(keys - ka.shape[1], 0).to(tl.float32) * 1e30
        init = init + tl.minimum(keys, 0).to(tl.float32) * 1e30
    ones = _rowsum_b()
    # Triton TR011: FP8 operands do not use the TF32 input mode.
    t0 = tl.dot(  # noqa: TR011
        _as_rowsum_a(p0[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32)
    )
    kA0 = _ld_k(ka, 0)
    sA0, sA1 = _ld_ks(ksa)
    vA0 = _ld_v(va, 0)
    vsA0, vsA1, vsA2, vsA3 = _ld_vs(vsa)
    tlx.amd_sched_barrier(0)
    # Triton TR011: FP8 operands do not use the TF32 input mode.
    t1 = tl.dot(  # noqa: TR011
        _as_rowsum_a(p1[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32)
    )
    _issue_tile(
        k_ptr,
        v_ptr,
        ks_ptr,
        vs_ptr,
        kd_a,
        kd_b,
        vd_a,
        vd_b,
        ksd_a,
        ksd_b,
        vsd_a,
        vsd_b,
        kt,
        vt,
        k_step,
        v_step,
        ks_step,
        k_n1,
        v_n1,
        ks_n1,
        COPY_K,
        COPY_V,
    )
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
    s1 = tl.dot_scaled(
        q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", acc=init_b, fast_math=True
    )
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
    # Triton TR011: FP8 operands do not use the TF32 input mode.
    t0 = tl.dot(  # noqa: TR011
        _as_rowsum_a(p0[0]), ones, tl.zeros([p0[0].shape[0] // 2, 16], dtype=tl.float32)
    )
    # Triton TR011: FP8 operands do not use the TF32 input mode.
    t1 = tl.dot(  # noqa: TR011
        _as_rowsum_a(p1[0]), ones, tl.zeros([p1[0].shape[0] // 2, 16], dtype=tl.float32)
    )
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
