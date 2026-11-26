# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""Scaled-MFMA MXFP8 attention building blocks for the gfx950 4-wave kernel.

One warp covers 32 query rows; each 128-key tile is two 64-key halves.
Q and K carry one E8M0 scale per 32x32 (row, feature) tile, V one per
feature and 32 keys, P one per row and 32 keys. The compute loop alternates
QK/PV work with softmax updates while double-buffered LDS tiles are refilled.
"""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from ..flash_attn_mxfp8.gfx950 import (
    _as_rowsum_a,
    _exp2_bf16_bits,
    _mx_scale,
    _p_scales,
    _pv,
    _relabel_keys,
    _rowsum_b,
    _scale_row_sums,
    _score_offset,
    _score_scale,
    _to_e4m3,
)

# Units of the scores ``s`` that the softmax helpers below take, shared with
# flash_attn_mxfp8.gfx950 (explained at its _score_scale): a score is
# t = (x - m_i + OFF) * C, with x the logit in the log2 domain, m_i the row max
# over the earlier tiles, C = _score_scale and OFF = _score_offset. The
# probability is then P = 2**(x - m_i + 4).


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
def _p_half(s, tmax, FAST_EXP: tl.constexpr):
    # One 64-key half of P in MXFP8 from scores s whose 32-key blocks peak at
    # tmax: (E4M3 in the PV operand order, E8M0 per 32 keys of each row). The
    # scales are pinned to the scores' MFMA layout (the reshape leaves them in
    # an equivalent linear one); otherwise the scores get converted to it and
    # the accumulators lose the MFMA layout.
    mfma: tl.constexpr = tlx.amd_mfma_layout(4, [32, 32, 64], True, [tlx.num_warps(), 1])
    byte, scale = _p_scales(tmax, FAST_EXP)
    if FAST_EXP:
        bits = _exp2_bf16_bits(s)
    else:
        bits = tl.math.exp2(s).to(tl.bfloat16)
    p = _to_e4m3(bits, tlx.require_layout(_per_block(scale), mfma))
    return _relabel_keys(p), byte


@triton.jit
def _softmax_first(s0, s1, FAST_EXP: tl.constexpr):
    # P of an item's first tile from the scores of its two 64-key halves, which
    # carry no offset yet (t = x * C). Returns (p0, p1, m_i, l_i): the two
    # halves of P as _p_half pairs, the row max in the log2 domain and zeroed
    # row sums [row, block half].
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
def _softmax_stats(s0, s1, mi, li, a0, a1, a2, a3, qk_scale, FAST_EXP: tl.constexpr):
    # The online-softmax update ahead of P's quantization. s0 arrives as t and
    # s1 still lacks its multiply by qk_scale (see _mfma_stage). Returns the
    # scores, their per-block maxima and the per-row shift tshift the maxima
    # still need, with the row max mi, the row sums li and the PV accumulators
    # rescaled where a row's max rose.
    C: tl.constexpr = _score_scale(FAST_EXP)
    OFF: tl.constexpr = _score_offset(FAST_EXP)
    vote: tl.constexpr = tlx.layout(shape=((32, 2, s0.shape[0] // 32), (1, )), stride=((1, 0, 32), (1, )))
    s1 = s1 * qk_scale
    tmax0, tmax1 = _block_max(s0), _block_max(s1)
    row = tl.maximum(tl.max(tmax0, 1), tl.max(tmax1, 1))
    tshift = tl.zeros_like(row)
    # A warp rescales only when one of its rows passes the stale max by more
    # than 4 (P would reach 2**8); it then moves each row whose max rose to its
    # new max.
    if tlx.warp_any(tlx.require_layout(row > (OFF + 4.0) * C, vote)):
        # The same shift in the scores' layout (up) and, below, in the block
        # maxima's (tshift).
        up = tl.maximum(tl.maximum(tl.max(s0, 1), tl.max(s1, 1)) - OFF * C, 0.0)
        alpha = tl.math.exp2(up * (-1.0 / C))[:, None]
        a0, a1, a2, a3 = a0 * alpha, a1 * alpha, a2 * alpha, a3 * alpha
        s0, s1 = s0 - up[:, None], s1 - up[:, None]
        mi = mi + up * (1.0 / C)
        tshift = -tl.maximum(row - OFF * C, 0.0)
        # A row whose max did not rise has tshift zero: its multiply is an
        # identity.
        li = li * tl.math.exp2(tshift * (1.0 / C))[:, None]
    return s0, s1, tmax0, tmax1, tshift, mi, li, a0, a1, a2, a3


@triton.jit
def _softmax_quantize(s0, s1, tmax0, tmax1, tshift, FAST_EXP: tl.constexpr):
    # Both 64-key halves of P from _softmax_stats' scores and block maxima.
    p0, p1 = (
        _p_half(s0, tmax0 + tshift[:, None], FAST_EXP),
        _p_half(s1, tmax1 + tshift[:, None], FAST_EXP),
    )
    return p0, p1


@triton.jit
def _softmax(s0, s1, mi, li, a0, a1, a2, a3, qk_scale, FAST_EXP: tl.constexpr):
    # One tile's softmax: the online update, then P in MXFP8. Returns
    # (p0, p1, mi, li, a0, a1, a2, a3).
    s0, s1, tmax0, tmax1, tshift, mi, li, a0, a1, a2, a3 = _softmax_stats(s0, s1, mi, li, a0, a1, a2, a3, qk_scale,
                                                                          FAST_EXP)
    p0, p1 = _softmax_quantize(s0, s1, tmax0, tmax1, tshift, FAST_EXP)
    return p0, p1, mi, li, a0, a1, a2, a3


@triton.jit
def _quantize_q_half(values):
    # One 64-feature half of a BF16 query tile to MXFP8 in registers, by
    # _mx_scale's rule per 32x32 (row, feature) tile. Returns (E4M3 in the QK
    # MFMA's A operand layout, E8M0 [row, feature block] in the layout the
    # scaled MFMA takes its scales in).
    rows: tl.constexpr = values.shape[0]
    grouped = tl.reshape(values, (rows // 32, 32, 2, 32))
    magnitude = (grouped.to(tl.uint16, bitcast=True) & 0x7FFF).to(tl.uint16)
    maximum_bits = tl.max(tl.max(magnitude, 3), 1)
    maximum = maximum_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    byte = _mx_scale(maximum)
    operand_scale = (byte << 23).to(tl.float32, bitcast=True)
    operand_scale = tl.broadcast_to(operand_scale[:, None, :, None], (rows // 32, 32, 2, 32)).reshape((rows, 64))
    quantized = _to_e4m3(values, operand_scale)
    scale = tl.reshape(tl.broadcast_to(byte.to(tl.uint8)[:, None, :], (rows // 32, 32, 2)), (rows, 2))
    q_layout: tl.constexpr = tlx.dot_operand_layout(0, tlx.amd_mfma_layout(4, [32, 32, 64], True, [tlx.num_warps(), 1]),
                                                    k_width=16)
    s_layout: tl.constexpr = tlx.layout(shape=((32, 2, tlx.num_warps()), (1, )), stride=((2, 1, 64), (1, )))
    return (
        tlx.require_layout(tl.reshape(quantized, (rows, 64)), q_layout),
        tlx.require_layout(scale, s_layout),
    )


@triton.jit
def _pv_last(p0, p1, va, vb, vsa, vsb, acc0, acc1, acc2, acc3, l_i):
    # PV and P's row sums for an item's last tile, which has no QK left to
    # interleave. Returns (acc0, acc1, acc2, acc3, l_i).
    ones = _rowsum_b()
    t0 = tl.dot(
        _as_rowsum_a(p0[0]),
        ones,
        tl.zeros([p0[0].shape[0] // 2, 16], dtype=tl.float32),
    )
    t1 = tl.dot(
        _as_rowsum_a(p1[0]),
        ones,
        tl.zeros([p1[0].shape[0] // 2, 16], dtype=tl.float32),
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
):
    # Issues the global->LDS copies of K tile kt and V tile vt, with their E8M0
    # words, as two commit groups: the first 64-key halves (ka, va, ksa, vsa),
    # then the second (kb, vb, ksb, vsb), which lie k_n1, v_n1 and ks_n1 further
    # on. The K or the V destinations may be None (no such tile left to fetch).
    #
    # V's packed E8M0 words go with V; they are laid out like K's, so they
    # share its offsets and steps.
    if ka is not None:
        tlx.async_load(k_ptr + kt * k_step, ka)
        tlx.async_load(ks_ptr + kt * ks_step, ksa)
    if va is not None:
        tlx.async_load(v_ptr + vt * v_step, va)
        tlx.async_load(vs_ptr + vt * ks_step, vsa)
    tlx.async_load_commit_group()
    if ka is not None:
        tlx.async_load(k_ptr + kt * k_step + k_n1, kb)
        tlx.async_load(ks_ptr + kt * ks_step + ks_n1, ksb)
    if va is not None:
        tlx.async_load(v_ptr + vt * v_step + v_n1, vb)
        tlx.async_load(vs_ptr + vt * ks_step + ks_n1, vsb)
    tlx.async_load_commit_group()


@triton.jit
def _ld_k(kbuf, half: tl.constexpr, RELAXED: tl.constexpr = True):
    # One head-dim half of a K tile.
    return tlx.local_load(tlx.local_slice(kbuf, [64 * half, 0], [64, kbuf.shape[1]]), relaxed=RELAXED)


@triton.jit
def _ld_ks(ksbuf, RELAXED: tl.constexpr = True):
    # The E8M0 scales of 64 keys in the packed K order: lane l's word holds
    # keys l % 32 and l % 32 + 32 at d-block l // 32 of both head-dim halves,
    # so one word per lane feeds both QK MFMA pairs. Returns the scales of the
    # low and high head-dim halves, [key, block] each.
    view: tl.constexpr = tlx.shared_linear_layout_encoding([
        [32, 0, 0],
        [0, 0, 1],
        [1, 0, 0],
        [2, 0, 0],
        [4, 0, 0],
        [8, 0, 0],
        [16, 0, 0],
        [0, 1, 0],
    ])
    lanes: tl.constexpr = tlx.layout(shape=((32, 2, tlx.num_warps()), (2, 2)), stride=((4, 2, 0), (128, 1)))
    ks = tlx.local_reinterpret(ksbuf, tl.uint8, [64, 2, 2], layout=view)
    return tl.split(tlx.local_load(ks, layout=lanes, relaxed=RELAXED))


@triton.jit
def _ld_v(vbuf, c: tl.constexpr, RELAXED: tl.constexpr = True):
    # One 32-feature chunk of a V half tile.
    return tlx.local_load(tlx.local_slice(vbuf, [0, 32 * c], [vbuf.shape[0], 32]), relaxed=RELAXED)


@triton.jit
def _ld_vs(vsbuf, RELAXED: tl.constexpr = True):
    # V's E8M0 scales for 64 keys in the transposed V order: lane l's word
    # holds column l % 32 of key block l // 32 for each 32-column V chunk, the
    # lane order the scaled PV MFMA takes. Returns one [column, block] scale
    # per chunk.
    view: tl.constexpr = tlx.shared_linear_layout_encoding([
        [0, 0, 1],
        [0, 0, 2],
        [1, 0, 0],
        [2, 0, 0],
        [4, 0, 0],
        [8, 0, 0],
        [16, 0, 0],
        [0, 1, 0],
    ])
    lanes: tl.constexpr = tlx.layout(shape=((32, 2, tlx.num_warps()), (4, )), stride=((8, 4, 0), (1, )))
    vs = tlx.local_load(
        tlx.local_reinterpret(vsbuf, tl.uint8, [32, 2, 4], layout=view),
        layout=lanes,
        relaxed=RELAXED,
    )
    even, odd = tl.split(tl.reshape(vs, (32, 2, 2, 2)))
    s0, s2 = tl.split(even)
    s1, s3 = tl.split(odd)
    return s0, s1, s2, s3


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
    rem=None,
):
    # QK(j + 1) and PV(j) with operands read from LDS in MFMA-sized pieces. Each
    # read is issued right after the MFMA that frees the previous piece and is
    # consumed one MFMA later, so its latency hides behind an MFMA while at most
    # one K half and one V chunk are live. The scheduling barriers keep LLVM
    # from sinking the reads to their uses. A per-row offset, divided by
    # qk_scale, starts the QK accumulator. s0 leaves multiplied by qk_scale,
    # behind the last PV MFMAs, and s1 is multiplied during the softmax update;
    # splitting the multiplies keeps both parts short.
    #
    # The operands come from one LDS slot: K tile j + 1 and V tile j, each as
    # two 64-key halves (ka, kb, va, vb) with their E8M0 words (ksa, ksb, vsa,
    # vsb). At most two commit groups are outstanding on entry, the slot's
    # first halves and its second: the wait for one outstanding group lets the
    # first halves land. The stage then issues two more groups, K tile kt and
    # V tile vt into the other slot (kd_*, vd_*, ksd_*, vsd_*) through
    # _issue_tile, so the wait for two outstanding groups ahead of the second
    # halves' reads lets those land.
    #
    # P's row sums go first: they need only registers, so the matrix cores get
    # work as the stage starts, and the first reads and the next tile's
    # global->LDS copies issue behind them. Their block scales are applied after
    # the first QK MFMA, when the sums are in.
    #
    # With rem, keys of the QK tile at or past rem start at -1e30 or less
    # (arithmetic: selects would take an SGPR pair per element).
    #
    # Returns (s0, s1, acc0, acc1, acc2, acc3, l_i).
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
        _as_rowsum_a(p0[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32))
    tlx.async_load_wait_group(1)
    kA0 = _ld_k(ka, 0)
    sA0, sA1 = _ld_ks(ksa)
    vA0 = _ld_v(va, 0)

    vsA0, vsA1, vsA2, vsA3 = _ld_vs(vsa)

    tlx.amd_sched_barrier(0)
    # Triton TR011: FP8 operands do not use the TF32 input mode.
    t1 = tl.dot(  # noqa: TR011
        _as_rowsum_a(p1[0]), ones, tl.zeros([q_lo.shape[0] // 2, 16], dtype=tl.float32))
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
    tlx.async_load_wait_group(2)
    kB0 = _ld_k(kb, 0)
    sB0, sB1 = _ld_ks(ksb)
    tlx.amd_sched_barrier(0)
    acc3 = _pv(p0, vA3, vsA3, acc3)
    vB0 = _ld_v(vb, 0)
    vsB0, vsB1, vsB2, vsB3 = _ld_vs(vsb)
    tlx.amd_sched_barrier(0)
    s1 = tl.dot_scaled(q_lo, qs_lo, "e4m3", kB0, sB0, "e4m3", acc=init_b, fast_math=True)
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
    # QK of an item's first tile: no PV to interleave and no offset yet, so
    # the accumulators start at zero and both halves leave multiplied by
    # qk_scale.
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
