# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Packed-varlen MXFP8 attention backward on scaled MFMAs (H4, D128).

STE gradients of tlx_mxfp8_varlen_pingpong_gfx950.forward, from its output
and natural-log LSE. Two kernels share the forward's operand layouts:

- dQ, query-owned: recompute S = Q K^T and dP = dO V^T over the head
  dimension, P = exp2(S - LSE) exactly, dS = P (dP - delta) quantized per
  row and 32 keys in registers, dQ += dS K. Padded keys are masked out of
  S: their P = exp(-LSE) would otherwise set the scales of the 32-key
  blocks they share with real keys.
- dK and dV, key-owned, the same with the roles swapped: S^T and dP^T,
  dV += P^T dO and dK += dS^T Q.

Preparation kernels write one FP8 copy of K, V, Q and dO in the forward's
K layout (row-major, one E8M0 scale per 32x32 block). The square blocks
scale both orientations, so the products read K, Q and dO transposed from
LDS (ds_read_tr8). Keys are padded like the forward's; queries are padded
to 128 at the end. Padded rows are zero and padded queries get an infinite
LSE, so they contribute nothing to dK/dV.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from . import mxfp8_attention_core_gfx950 as core
from . import mxfp8_varlen_pingpong_gfx950 as fwd

_HEADS = 4
_HEAD_DIM = 128
_LOG2E = 1.4426950408889634
_SHORT_DS_SCALE = 0.03125
_SHORT_DS_SCALE_BYTE = 122
_SHORT_DS_MAX_SEQLEN_K = 2048
# 128 rows on four warps: LDS admits one workgroup per CU anyway, and one
# wave per SIMD gets all 512 registers for the accumulators.
_BLOCK = 128
_LAUNCH_OPTIONS = {
    "num_warps": 4,
    "num_stages": 1,
    "llvm_fn_attrs": (("amdgpu-ieee", "false"),),
}
# dQ runs two waves per SIMD, all 256 registers as VGPRs.
_DQ_LAUNCH_OPTIONS = {
    **_LAUNCH_OPTIONS,
    "llvm_fn_attrs": (
        ("amdgpu-ieee", "false"),
        ("amdgpu-waves-per-eu", "2,2"),
        ("amdgpu-agpr-alloc", "0,0"),
    ),
}


@triton.jit
def _prepare_kv(
    K,
    V,
    K8,
    KS,
    V8,
    VS,
    CuKV,
    CuPad,
    TileSequence,
    CAPACITY,
    BATCH,
    NPAD,
    HEADS: tl.constexpr,
):
    """K and V in the K layout, padded as the forward;
    four extra programs zero the spare last 256 keys."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    if tile >= CAPACITY:
        dst = head * NPAD + NPAD - 256 + (tile - CAPACITY) * 64
        fwd._rows_tile(K, K8, KS, 0, dst, 0, 0, HEADS)
        fwd._rows_tile(V, V8, VS, 0, dst, 0, 0, HEADS)
        return
    sequence = tl.load(TileSequence + tile, mask=tile < CAPACITY, other=BATCH)
    if sequence >= BATCH:
        return
    first = tl.load(CuKV + sequence)
    length = tl.load(CuKV + sequence + 1) - first
    pad_start = tl.load(CuPad + sequence)
    row = tile * 64 - pad_start
    key = row - fwd._front(length, tl.load(CuPad + sequence + 1) - pad_start)
    src = (first + key).to(tl.int64) * (HEADS * 128) + head * 128
    dst = head * NPAD + pad_start + row
    fwd._rows_tile(K, K8, KS, src, dst, -key, length - key, HEADS)
    fwd._rows_tile(V, V8, VS, src, dst, -key, length - key, HEADS)


@triton.jit
def _load_rows(X, src, lo, hi, HEADS: tl.constexpr):
    # 64 rows of one head, zero outside [lo, hi).
    rows = tl.arange(0, 64)
    return tlx.buffer_load(
        X + src,
        rows[:, None] * (HEADS * 128) + tl.arange(0, 128)[None, :],
        ((rows >= lo) & (rows < hi))[:, None],
        0.0,
        contiguity=8,
    )


@triton.jit
def _prepare_q(
    Q,
    DO,
    O,
    LSE,
    Q8,
    QS,
    D8,
    DS,
    LSEP,
    DELTAP,
    CuQ,
    CuQPad,
    TileSequence,
    CAPACITY,
    BATCH,
    NQPAD,
    TOTAL_Q,
    HEADS: tl.constexpr,
):
    """Q and dO in the K layout, log2 LSE (infinite on padding) and -delta;
    two extra programs write the spare last 128 rows (zero data, infinite
    LSE), which serve the dK/dV work items without queries."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    if tile >= CAPACITY:
        dst = head * NQPAD + NQPAD - 128 + (tile - CAPACITY) * 64
        fwd._rows_tile(Q, Q8, QS, 0, dst, 0, 0, HEADS)
        fwd._rows_tile(DO, D8, DS, 0, dst, 0, 0, HEADS)
        rows = tl.arange(0, 64)
        tl.store(DELTAP + dst + rows, tl.zeros([64], tl.float32))
        tl.store(LSEP + dst + rows, tl.full([64], float("inf"), tl.float32))
        return
    sequence = tl.load(TileSequence + tile, mask=tile < CAPACITY, other=BATCH)
    if sequence >= BATCH:
        return
    first = tl.load(CuQ + sequence)
    length = tl.load(CuQ + sequence + 1) - first
    pad_start = tl.load(CuQPad + sequence)
    row = tile * 64 - pad_start
    src = (first + row).to(tl.int64) * (HEADS * 128) + head * 128
    dst = head * NQPAD + pad_start + row
    # All loads issue before the stores (which may alias them for the
    # compiler).
    q = _load_rows(Q, src, -row, length - row, HEADS)
    do = _load_rows(DO, src, -row, length - row, HEADS)
    o = _load_rows(O, src, -row, length - row, HEADS)
    rows = tl.arange(0, 64)
    valid = rows < length - row
    lse = tl.load(
        LSE + head * TOTAL_Q + first + row + rows, mask=valid, other=float("inf")
    )
    fwd._rows_store(q, Q8, QS, dst)
    fwd._rows_store(do, D8, DS, dst)
    # Stored negated: dP's accumulator starts from it.
    tl.store(DELTAP + dst + rows, -tl.sum(do.to(tl.float32) * o.to(tl.float32), 1))
    tl.store(LSEP + dst + rows, lse * 1.4426950408889634)


@triton.jit
def _issue_tile(ptr, s_ptr, buf_a, buf_b, s_a, s_b, tile):
    # One 128-row tile as two 64-row halves; both layouts store a 64-row half
    # in 64 * 128 contiguous bytes.
    tlx.async_load(ptr + tile * (128 * 128), buf_a)
    tlx.async_load(ptr + tile * (128 * 128) + 64 * 128, buf_b)
    tlx.async_load(s_ptr + tile * 128, s_a)
    tlx.async_load(s_ptr + tile * 128 + 64, s_b)


@triton.jit
def _half_scores(a_lo, as_lo, a_hi, as_hi, kbuf, ksbuf, init):
    k0 = core._ld_k(kbuf, 0)
    k1 = core._ld_k(kbuf, 1)
    s_lo, s_hi = core._ld_ks(ksbuf)
    s = tl.dot_scaled(a_lo, as_lo, "e4m3", k0, s_lo, "e4m3", acc=init, fast_math=True)
    return tl.dot_scaled(a_hi, as_hi, "e4m3", k1, s_hi, "e4m3", acc=s, fast_math=True)


@triton.jit
def _to_e4m3_f32(x, scale):
    # core._to_e4m3 straight from FP32 (v_cvt_scalef32_pk_fp8_f32), without
    # the BF16 conversions.
    w = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp8_f32 $0, $2, $3, $6\n"
        "v_cvt_scalef32_pk_fp8_f32 $0, $4, $5, $8 op_sel:[0,0,0,1]",
        "=&v,=v,v,v,v,v,v,v,v,v",
        [x, scale],
        dtype=tl.uint16,
        is_pure=True,
        pack=4,
    )
    rows: tl.constexpr = x.shape[0]
    cols: tl.constexpr = x.shape[1]
    w, _ = tl.split(tl.permute(tl.reshape(w, (rows, cols // 4, 2, 2)), (0, 1, 3, 2)))
    b = tl.join(w.to(tl.uint8), (w >> 8).to(tl.uint8))
    return tl.reshape(b, (rows, cols)).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _quantize_blocks(x, DIRECT: tl.constexpr = False):
    # [rows, 64] FP32 in the MFMA layout -> (E4M3 in the next MFMA's operand
    # order, E8M0 per row and 32 columns), as core._p_half does for P.
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
    halves: tl.constexpr = tlx.layout(
        shape=((32, 2, x.shape[0] // 32), (1,)), stride=((2, 1, 64), (1,))
    )
    byte = core._mx_scale(core._block_max(tl.abs(x)))
    scale = (byte << 23).to(tl.float32, bitcast=True)
    if DIRECT:
        # Straight from FP32: fewer instructions where dS's layout allows.
        q = _to_e4m3_f32(x, tlx.require_layout(core._per_block(scale), mfma))
    else:
        q = core._to_e4m3(
            x.to(tl.bfloat16), tlx.require_layout(core._per_block(scale), mfma)
        )
    return core._relabel_keys(q), tlx.require_layout(byte.to(tl.uint8), halves)


@triton.jit
def _quantize_unit(p):
    # Probabilities in [0, 1] with the fixed E8M0 scale 2**-8 (byte 119): the
    # block maxima only matter for blocks below 2**-8, which add little.
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
    halves: tl.constexpr = tlx.layout(
        shape=((32, 2, p.shape[0] // 32), (1,)), stride=((2, 1, 64), (1,))
    )
    scale = tlx.require_layout(tl.full(p.shape, 0.00390625, tl.float32), mfma)
    byte = tl.full((p.shape[0], 2), 119, tl.uint8)
    return core._relabel_keys(_to_e4m3_f32(p, scale)), tlx.require_layout(byte, halves)


@triton.jit
def _quantize_ds(ds, STATIC: tl.constexpr):
    if not STATIC:
        return _quantize_blocks(ds, True)
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
    halves: tl.constexpr = tlx.layout(
        shape=((32, 2, ds.shape[0] // 32), (1,)), stride=((2, 1, 64), (1,))
    )
    scale = tlx.require_layout(tl.full(ds.shape, _SHORT_DS_SCALE, tl.float32), mfma)
    byte = tl.full((ds.shape[0], 2), _SHORT_DS_SCALE_BYTE, tl.uint8)
    return core._relabel_keys(_to_e4m3_f32(ds, scale)), tlx.require_layout(byte, halves)


@triton.jit
def _product_half(x, vbuf, vsbuf, acc):
    # acc += x vbuf over the four 32-column chunks of a 64-row V-layout tile.
    s0, s1, s2, s3 = core._ld_vs(vsbuf)
    a0, a1, a2, a3 = acc
    a0 = core._pv(x, core._ld_v(vbuf, 0), s0, a0)
    a1 = core._pv(x, core._ld_v(vbuf, 1), s1, a1)
    a2 = core._pv(x, core._ld_v(vbuf, 2), s2, a2)
    a3 = core._pv(x, core._ld_v(vbuf, 3), s3, a3)
    return a0, a1, a2, a3


# LDS layout of the dK/dV kernels' Q and dO tiles ([feature, query], offset
# bit i -> coordinate): no bank conflicts (SQ_LDS_BANK_CONFLICT) for either
# the scores' 16-byte row reads or the products' transposed 8-byte reads;
# with core's K layout the latter conflicted on about half the LDS cycles.
_Q_LDS_BASES = tl.constexpr(
    [
        [1, 0],
        [2, 0],
        [4, 0],
        [8, 0],
        [16, 0],
        [32, 0],
        [64, 0],
        [0, 1],
        [0, 8],
        [0, 32],
        [0, 4],
        [0, 2],
        [0, 16],
    ]
)
# The same, transposed to [query, feature], with query-index bits 2 and 4
# swapped: the relabeled query order _relabel_keys gives P and dS.
_Q_LDS_BASES_T = tl.constexpr(
    [
        [0, 1],
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [0, 32],
        [0, 64],
        [1, 0],
        [8, 0],
        [32, 0],
        [16, 0],
        [2, 0],
        [4, 0],
    ]
)


@triton.jit
def _ld_vt(kbuf, c: tl.constexpr):
    """The product's [64 query, 32 feature] B operand, chunk c, read
    transposed from a K-layout half tile in the relabeled query order."""
    # A direct transposed view: memdesc_trans of a reinterpreted padded
    # layout drops the reinterpretation's bases.
    view: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], _Q_LDS_BASES_T, [64, 128]
    )
    t = tlx.local_reinterpret(kbuf, tl.float8e4nv, [64, 128], view)
    return tlx.local_load(tlx.local_slice(t, [0, 32 * c], [64, 32]), relaxed=True)


@triton.jit
def _ld_vs_square(ksbuf):
    # core._ld_vs from K-layout scales: one byte per 32x32 block, the same
    # for every column of a chunk. Chunk c, query block g is byte
    # 2 (c // 2) + g of row 32 (c % 2): two broadcast words.
    lanes: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), ()), stride=((2, 1, 0), ())
    )
    g = tlx.require_layout(
        tl.broadcast_to(tl.arange(0, 2)[None, :], (32, 2)), lanes, pin=False
    )
    w0 = tlx.local_load(tlx.local_slice(ksbuf, [0, 0], [1, 1]), relaxed=True)
    w1 = tlx.local_load(tlx.local_slice(ksbuf, [32, 0], [1, 1]), relaxed=True)
    s0 = ((w0 >> (8 * g)) & 0xFF).to(tl.uint8)
    s1 = ((w1 >> (8 * g)) & 0xFF).to(tl.uint8)
    s2 = ((w0 >> (8 * g + 16)) & 0xFF).to(tl.uint8)
    s3 = ((w1 >> (8 * g + 16)) & 0xFF).to(tl.uint8)
    return s0, s1, s2, s3


@triton.jit
def _b_operands(kbuf, ksbuf):
    # The product's four B chunks and scales, read transposed from a K-layout
    # half tile.
    s0, s1, s2, s3 = _ld_vs_square(ksbuf)
    return (
        _ld_vt(kbuf, 0),
        _ld_vt(kbuf, 1),
        _ld_vt(kbuf, 2),
        _ld_vt(kbuf, 3),
        s0,
        s1,
        s2,
        s3,
    )


@triton.jit
def _product_b(x, ops, acc):
    # acc += x B over the four 32-column chunks of _b_operands.
    a0, a1, a2, a3 = acc
    a0 = core._pv(x, ops[0], ops[4], a0)
    a1 = core._pv(x, ops[1], ops[5], a1)
    a2 = core._pv(x, ops[2], ops[6], a2)
    a3 = core._pv(x, ops[3], ops[7], a3)
    return a0, a1, a2, a3


@triton.jit
def _load_operand(X8, XS, row, ROWS: tl.constexpr):
    """ROWS rows of a K-layout array at row `row` (a multiple of 64) as the
    scaled MFMA's A operand: two head-dim halves and their E8M0 scales."""
    mfma: tl.constexpr = tlx.amd_mfma_layout(
        4, [32, 32, 64], True, [tlx.num_warps(), 1]
    )
    a_layout: tl.constexpr = tlx.dot_operand_layout(0, mfma, k_width=16)
    s_layout: tl.constexpr = tlx.layout(
        shape=((32, 2, tlx.num_warps()), (1,)), stride=((2, 1, 64), (1,))
    )
    r = tl.arange(0, ROWS)[:, None]
    offsets = tlx.require_layout(
        (r * 128 + tl.arange(0, 64)[None, :]).to(tl.int32), a_layout
    )
    base = X8 + row * 128
    lo = tlx.require_layout(tlx.buffer_load(base, offsets, contiguity=16), a_layout)
    hi = tlx.require_layout(
        tlx.buffer_load(base + 64, offsets, contiguity=16), a_layout
    )
    # Undo _rows_tile's scale permutation: the byte of 32-row group g and
    # 32-column block b sits at row 64 * blk + 32 * (b % 2) + i, column
    # 2 * (b // 2) + g of the [rows, 4] scale array.
    j = tl.arange(0, 2)[None, :]
    s_lo = ((r // 64) * 64 + j * 32 + r % 32) * 4 + (r % 64) // 32
    s_base = XS + row * 4
    lo_s = tlx.buffer_load(s_base, tlx.require_layout(s_lo.to(tl.int32), s_layout))
    hi_s = tlx.buffer_load(
        s_base, tlx.require_layout((s_lo + 2).to(tl.int32), s_layout)
    )
    return lo, lo_s, hi, hi_s


@triton.jit
def _store_rows(Out, base, acc, scale, lo, hi, ROW: tl.constexpr):
    """Rows [lo, hi) of four [rows, 32] accumulator chunks, times scale, with
    eight contiguous BF16 per lane (the MFMA layout gives four)."""
    n: tl.constexpr = acc[0].shape[0]
    warps: tl.constexpr = tlx.num_warps()
    wide: tl.constexpr = tlx.layout(
        shape=((4, 16, warps), (8, n // (16 * warps))),
        stride=((8, 32, 512), (1, 512 * warps)),
    )
    rows = tl.arange(0, n)[:, None]
    columns = tl.arange(0, 32)[None, :]
    mask = tlx.require_layout(
        (rows >= lo) & (rows < hi) & (columns >= 0), wide, pin=False
    )
    ptr = tlx.require_layout(Out + base + rows * ROW + columns, wide, pin=False)
    for c in tl.static_range(4):
        value = tlx.require_layout((acc[c] * scale).to(tl.bfloat16), wide, pin=False)
        tl.store(ptr + 32 * c, value, mask=mask)


@triton.jit
def _add_rows(Out, base, acc, scale, hi, ROW: tl.constexpr):
    """FP32 atomic add of rows [0, hi) of four [rows, 32] chunks, times scale."""
    rows = tl.arange(0, acc[0].shape[0])[:, None]
    columns = tl.arange(0, 32)[None, :]
    mask = rows < hi
    ptr = Out + base + rows * ROW + columns
    tl.atomic_add(ptr, acc[0] * scale, mask=mask, sem="relaxed")
    tl.atomic_add(ptr + 32, acc[1] * scale, mask=mask, sem="relaxed")
    tl.atomic_add(ptr + 64, acc[2] * scale, mask=mask, sem="relaxed")
    tl.atomic_add(ptr + 96, acc[3] * scale, mask=mask, sem="relaxed")


@triton.jit
def _store_rows_fp32(Out, base, acc, scale, hi, ROW: tl.constexpr):
    rows = tl.arange(0, acc[0].shape[0])[:, None]
    columns = tl.arange(0, 32)[None, :]
    mask = rows < hi
    ptr = Out + base + rows * ROW + columns
    tl.store(ptr, acc[0] * scale, mask=mask)
    tl.store(ptr + 32, acc[1] * scale, mask=mask)
    tl.store(ptr + 64, acc[2] * scale, mask=mask)
    tl.store(ptr + 96, acc[3] * scale, mask=mask)


@triton.jit
def _dq_partial_reduce(Partial, DQ, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    value = tl.load(Partial + offsets, mask=valid, other=0.0)
    value += tl.load(Partial + N + offsets, mask=valid, other=0.0)
    tl.store(DQ + offsets, value.to(tl.bfloat16), mask=valid)


@triton.jit
def _persistent(n_items):
    # Each XCD takes a contiguous run of items.
    pid = tl.program_id(0)
    n_ctas = tl.num_programs(0)
    base = pid % 8 * (n_ctas // 8) + pid // 8
    return base, n_ctas, (n_items - base + n_ctas - 1) // n_ctas


@triton.jit
def _sload(ptr):
    # A scalar load: counted in lgkmcnt, so waiting for it does not wait for
    # the vector memory operations (dK/dV stores, tile copies) before it.
    return tl.inline_asm_elementwise(
        "s_load_dword $0, $1, 0x0\ns_waitcnt lgkmcnt(0)",
        "=s,s",
        [ptr],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sload_pairs(a, b, c, d):
    """Elements [0] and [1] of four int32 arrays as eight scalars, with one
    scalar-memory round trip (see _sload)."""
    x, y, z, w = tl.inline_asm_elementwise(
        "s_load_dwordx2 $0, $4, 0x0\n"
        "s_load_dwordx2 $1, $5, 0x0\n"
        "s_load_dwordx2 $2, $6, 0x0\n"
        "s_load_dwordx2 $3, $7, 0x0\n"
        "s_waitcnt lgkmcnt(0)",
        # Early-clobber: a result may land before a later load reads its
        # address.
        "=&s,=&s,=&s,=&s,s,s,s,s",
        [a, b, c, d],
        dtype=(tl.int64, tl.int64, tl.int64, tl.int64),
        is_pure=True,
        pack=1,
    )
    return (
        x.to(tl.int32),
        (x >> 32).to(tl.int32),
        y.to(tl.int32),
        (y >> 32).to(tl.int32),
        z.to(tl.int32),
        (z >> 32).to(tl.int32),
        w.to(tl.int32),
        (w >> 32).to(tl.int32),
    )


@triton.jit
def _padded_item(
    item, TileSequence, CuPad, NP, HEADS: tl.constexpr, ROWS: tl.constexpr = 128
):
    # Items are (padded ROWS-row tile, head); TileSequence maps 64-row tiles.
    tile = item // HEADS
    head = (item % HEADS).to(tl.int64)
    sequence = _sload(TileSequence + (ROWS // 64) * tile)
    return (
        sequence,
        head,
        tile * ROWS - _sload(CuPad + sequence),
        head * NP + tile * ROWS,
    )


@triton.jit
def _zeros4(rows: tl.constexpr):
    return (
        tl.zeros([rows, 32], dtype=tl.float32),
        tl.zeros([rows, 32], dtype=tl.float32),
        tl.zeros([rows, 32], dtype=tl.float32),
        tl.zeros([rows, 32], dtype=tl.float32),
    )


@triton.jit
def _views8(bufs, slot):
    """One slot of each double-buffered tile buffer of the dQ kernel."""
    return (
        tlx.local_view(bufs[0], slot),
        tlx.local_view(bufs[1], slot),
        tlx.local_view(bufs[2], slot),
        tlx.local_view(bufs[3], slot),
        tlx.local_view(bufs[4], slot),
        tlx.local_view(bufs[5], slot),
        tlx.local_view(bufs[6], slot),
        tlx.local_view(bufs[7], slot),
    )


@triton.jit
def _views10(bufs, slot):
    """One slot of each double-buffered tile buffer of the dK/dV kernels."""
    return (
        tlx.local_view(bufs[0], slot),
        tlx.local_view(bufs[1], slot),
        tlx.local_view(bufs[2], slot),
        tlx.local_view(bufs[3], slot),
        tlx.local_view(bufs[4], slot),
        tlx.local_view(bufs[5], slot),
        tlx.local_view(bufs[6], slot),
        tlx.local_view(bufs[7], slot),
        tlx.local_view(bufs[8], slot),
        tlx.local_view(bufs[9], slot),
    )


@triton.jit
def _key_mask(init_s, keys, kv_len):
    # Arithmetic like the forward's _mask_front: -1e30 or less outside
    # [0, kv_len).
    return (
        init_s
        + (tl.minimum(keys, 0) + tl.minimum(kv_len - 1 - keys, 0)).to(tl.float32) * 1e30
    )


@triton.jit
def _issue_k_part(src, b, PART: tl.constexpr):
    # Half of _issue_k_tile; the second half commits the group.
    K8, KS, V8, VS, row, k_off, s_off = src
    i32: tl.constexpr = tl.pointer_type(tl.int32)
    if PART == 0:
        _issue_tile(
            K8 + row * 128 + k_off, KS.to(i32) + row + s_off, b[0], b[1], b[2], b[3], 0
        )
    else:
        _issue_tile(
            V8 + row * 128 + k_off, VS.to(i32) + row + s_off, b[4], b[5], b[6], b[7], 0
        )
        tlx.async_load_commit_group()
    tlx.amd_sched_barrier(0)


@triton.jit
def _dq_tile(
    q,
    d,
    init_a,
    init_b,
    delta,
    qk_scale,
    b,
    acc,
    src,
    nb,
    STATIC_DS: tl.constexpr,
):
    """dQ += dS K over one 128-key tile, one 64-key half at a time (two waves
    per SIMD then fit in 256 registers); init_a / init_b start the halves'
    scores (padded keys masked), delta is a per-row column. The next tile's
    copies issue between the score MFMAs: issued back to back they stall on
    the memory pipeline. K is read transposed from its K-layout copy."""
    ka, kb, ksa, ksb, va, vb, vsa, vsb = b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    tlx.amd_sched_barrier(0)
    s_a = _half_scores(q[0], q[1], q[2], q[3], ka, ksa, init_a)
    _issue_k_part(src, nb, 0)
    dp_a = _half_scores(d[0], d[1], d[2], d[3], va, vsa, None)
    _issue_k_part(src, nb, 1)
    acc = _product_b(
        _quantize_ds(tl.math.exp2(s_a * qk_scale) * (dp_a + delta), STATIC_DS),
        _b_operands(ka, ksa),
        acc,
    )
    tlx.amd_sched_barrier(0)
    s_b = _half_scores(q[0], q[1], q[2], q[3], kb, ksb, init_b)
    dp_b = _half_scores(d[0], d[1], d[2], d[3], vb, vsb, None)
    return _product_b(
        _quantize_ds(tl.math.exp2(s_b * qk_scale) * (dp_b + delta), STATIC_DS),
        _b_operands(kb, ksb),
        acc,
    )


@triton.jit
def _issue_k_tile(K8, KS, V8, VS, row, k_off, s_off, b):
    # K and V in the K layout for the 128-key tile at padded row `row`, as
    # one commit group.
    _issue_k_part((K8, KS, V8, VS, row, k_off, s_off), b, 0)
    _issue_k_part((K8, KS, V8, VS, row, k_off, s_off), b, 1)


@triton.jit
def _dq_item(
    row,
    Q8,
    QS,
    D8,
    DS,
    LSEP,
    DELTAP,
    qk_scale,
    BLOCK: tl.constexpr,
):
    q = _load_operand(Q8, QS, row, BLOCK)
    d = _load_operand(D8, DS, row, BLOCK)
    rows = tl.arange(0, BLOCK)
    # Triton TR003: row addresses a host-padded BLOCK-row allocation.
    lse = tl.load(LSEP + row + rows) * (-1.0 / qk_scale)  # noqa: TR003
    delta = tl.load(DELTAP + row + rows)  # noqa: TR003
    return q, d, lse, delta


@triton.jit
def _dq_meta(
    item,
    CuQ,
    CuKV,
    CuPad,
    CuQPad,
    QTileSequence,
    NPAD,
    NQPAD,
    SPLITS,
    HEADS: tl.constexpr,
):
    # Items are (query tile, head, key split).
    split = item % SPLITS
    tile = item // SPLITS // HEADS
    head = (item // SPLITS % HEADS).to(tl.int64)
    sequence = _sload(QTileSequence + 2 * tile)
    cq0, cq1, pad_start, pad_end, kv0, kv1, qpad0, _ = _sload_pairs(
        CuQ + sequence, CuPad + sequence, CuKV + sequence, CuQPad + sequence
    )
    row = head * NQPAD + tile * 128
    q0 = cq0 + tile * 128 - qpad0
    q_rows = cq1 - q0
    all_tiles = (pad_end - pad_start) // 128
    per_split = (all_tiles + SPLITS - 1) // SPLITS
    first_tile = tl.minimum(split * per_split, all_tiles)
    n_tiles = tl.minimum(per_split, all_tiles - first_tile)
    kv_len = kv1 - kv0
    key = first_tile * 128 - fwd._front(kv_len, all_tiles * 128)
    # A split without tiles takes one fully masked tile of the spare keys.
    kv_len = tl.where(n_tiles > 0, kv_len, 0)
    pad_start = tl.where(n_tiles > 0, pad_start + first_tile * 128, NPAD - 256)
    n_tiles = tl.maximum(n_tiles, 1)
    meta = (
        q0.to(tl.int64) * (HEADS * 128) + head * 128,
        q_rows,
        head * NPAD + pad_start,
        n_tiles,
        key,
        kv_len,
    )
    return meta, row


@triton.jit
def _dq_kernel(
    Q8,
    QS,
    D8,
    DS,
    K8,
    KS,
    V8,
    VS,
    LSEP,
    DELTAP,
    DQ,
    CuQ,
    CuKV,
    CuPad,
    CuQPad,
    QTileSequence,
    BATCH,
    NPAD,
    NQPAD,
    qk_scale,
    sm_scale,
    SPLITS,
    PARTIAL_STRIDE,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
    ATOMIC: tl.constexpr,
    PARTIAL: tl.constexpr,
    STATIC_DS: tl.constexpr,
):
    """dQ per (query tile, head); with ATOMIC, SPLITS workgroups share a
    query tile's keys and add FP32 partials into DQ."""
    D: tl.constexpr = 128
    HN: tl.constexpr = 64
    n_items = _sload(CuQPad + BATCH) // 128 * HEADS * SPLITS
    base, n_ctas, n_mine = _persistent(n_items)
    if base >= n_items:
        return
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, HN)
    k_off = offs_d[:, None] + offs_n[None, :] * D
    s_off = offs_n[:, None]
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], _Q_LDS_BASES, [D, HN]
    )
    bufs = (
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
    )
    meta, row = _dq_meta(
        base, CuQ, CuKV, CuPad, CuQPad, QTileSequence, NPAD, NQPAD, SPLITS, HEADS
    )
    _issue_k_tile(K8, KS, V8, VS, meta[2], k_off, s_off, _views8(bufs, 0))
    step = 0
    for w in tl.range(0, n_mine, num_stages=1):
        item = base + w * n_ctas
        nxt = tl.where(w + 1 < n_mine, item + n_ctas, item)
        # Loaded at the item start: the other wave on the SIMD hides the
        # latency, and prefetched they would hold 40 registers across the item.
        next_meta, next_row = _dq_meta(
            nxt, CuQ, CuKV, CuPad, CuQPad, QTileSequence, NPAD, NQPAD, SPLITS, HEADS
        )
        q, d, lse, delta = _dq_item(
            row,
            Q8,
            QS,
            D8,
            DS,
            LSEP,
            DELTAP,
            qk_scale,
            BLOCK,
        )
        q_base, q_rows, key_row, n_tiles, key, kv_len = meta
        init_s = tl.broadcast_to(lse[:, None], (BLOCK, HN))
        acc = _zeros4(BLOCK)
        for t in tl.range(0, n_tiles, num_stages=1):
            slot = step % 2
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            row = tl.where(t + 1 < n_tiles, key_row + (t + 1) * 128, next_meta[2])
            first = key + t * 128
            init_a = init_s
            init_b = init_s
            # Only the first and last tiles of a sequence hold padded keys.
            if (first < 0) | (first + 128 > kv_len):
                keys = first + tl.arange(0, HN)[None, :]
                init_a = _key_mask(init_s, keys, kv_len)
                init_b = _key_mask(init_s, keys + HN, kv_len)
            acc = _dq_tile(
                q,
                d,
                init_a,
                init_b,
                delta[:, None],
                qk_scale,
                _views8(bufs, slot),
                acc,
                (K8, KS, V8, VS, row, k_off, s_off),
                _views8(bufs, 1 - slot),
                STATIC_DS,
            )
            step += 1
        if PARTIAL:
            split = item % SPLITS
            _store_rows_fp32(
                DQ,
                split.to(tl.int64) * PARTIAL_STRIDE + q_base,
                acc,
                sm_scale,
                q_rows,
                HEADS * D,
            )
        elif ATOMIC:
            _add_rows(DQ, q_base, acc, sm_scale, q_rows, HEADS * D)
        else:
            _store_rows(DQ, q_base, acc, sm_scale, 0, q_rows, HEADS * D)
        meta = next_meta
        row = next_row
    tlx.async_load_wait_group(0)


@triton.jit
def _issue_q_tile(Q8, QS, D8, DS, LSEP, DELTAP, row, k_off, s_off, b):
    # Q and dO of the 128-query tile at padded row `row` (head included), in
    # the K layout, and its log2 LSE and delta, as one commit group. The
    # products read Q and dO transposed (_ld_vt). The statistics go through
    # LDS: carried in registers, each broadcast row vector would hold 32
    # VGPRs across the tile.
    _issue_q_part((Q8, QS, D8, DS, LSEP, DELTAP, row, k_off, s_off), b, 0)
    _issue_q_part((Q8, QS, D8, DS, LSEP, DELTAP, row, k_off, s_off), b, 1)
    _issue_q_part((Q8, QS, D8, DS, LSEP, DELTAP, row, k_off, s_off), b, 2)


@triton.jit
def _issue_q_part(src, b, PART: tl.constexpr):
    # A third of _issue_q_tile; the last third commits the group.
    Q8, QS, D8, DS, LSEP, DELTAP, row, k_off, s_off = src
    i32: tl.constexpr = tl.pointer_type(tl.int32)
    if PART == 0:
        _issue_tile(
            Q8 + row * 128 + k_off, QS.to(i32) + row + s_off, b[0], b[1], b[2], b[3], 0
        )
    elif PART == 1:
        _issue_tile(
            D8 + row * 128 + k_off, DS.to(i32) + row + s_off, b[4], b[5], b[6], b[7], 0
        )
    elif PART == 2:
        cols = tl.arange(0, 128)
        tlx.async_load(LSEP + row + cols, b[8])
        tlx.async_load(DELTAP + row + cols, b[9])
        tlx.async_load_commit_group()
    tlx.amd_sched_barrier(0)


@triton.jit
def _copy_step(src, nb, PART: tl.constexpr, COPY: tl.constexpr):
    # With COPY, a quarter of the next tile's copies; either way a scheduling
    # boundary, which keeps loads for later phases from being hoisted.
    if COPY:
        _issue_q_part(src, nb, PART)
    else:
        tlx.amd_sched_barrier(0)


@triton.jit
def _dkdv_tile(
    k,
    v,
    qk_scale,
    b,
    dk,
    dv,
    src,
    nb,
    COPY: tl.constexpr,
    STATIC_DS: tl.constexpr,
):
    """dV += P^T dO and dK += dS^T Q over one 128-query tile, one 64-query
    half at a time (fewer live registers than issuing both halves' score
    MFMAs first). With COPY the next tile's copies (src into the views nb)
    issue spread over the tile: issued back to back they stall on the memory
    pipeline."""
    ROWS: tl.constexpr = k[0].shape[0]
    qa, qb, qsa, qsb, da, db, dsa, dsb = b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    lse_buf, delta_buf = b[8], b[9]
    tlx.amd_sched_barrier(0)
    # Scores start from zero; the scale and LSE fold into one FMA.
    s_a = _half_scores(k[0], k[1], k[2], k[3], qa, qsa, None)
    _copy_step(src, nb, 0, COPY)
    # dP starts from -delta: the accumulator registers take it in the MFMA
    # layout, sparing packed adds that pair registers out of order.
    dp_a = _half_scores(
        v[0],
        v[1],
        v[2],
        v[3],
        da,
        dsa,
        tl.broadcast_to(
            tlx.local_load(tlx.local_slice(delta_buf, [0], [64]))[None, :],
            (ROWS, 64),
        ),
    )
    _copy_step(src, nb, 1, COPY)
    # Each broadcast LSE half holds 32 VGPRs: loaded where used.
    lse_a = tlx.local_load(tlx.local_slice(lse_buf, [0], [64]))[None, :]
    p = tl.math.exp2(s_a * qk_scale - lse_a)
    dv = _product_b(_quantize_unit(p), _b_operands(da, dsa), dv)
    ds = p * dp_a
    dk = _product_b(_quantize_ds(ds, STATIC_DS), _b_operands(qa, qsa), dk)
    _copy_step(src, nb, 2, COPY)
    s_b = _half_scores(k[0], k[1], k[2], k[3], qb, qsb, None)
    dp_b = _half_scores(
        v[0],
        v[1],
        v[2],
        v[3],
        db,
        dsb,
        tl.broadcast_to(
            tlx.local_load(tlx.local_slice(delta_buf, [64], [64]))[None, :],
            (ROWS, 64),
        ),
    )
    _copy_step(src, nb, 3, COPY)
    lse_b = tlx.local_load(tlx.local_slice(lse_buf, [64], [64]))[None, :]
    p = tl.math.exp2(s_b * qk_scale - lse_b)
    dv = _product_b(_quantize_unit(p), _b_operands(db, dsb), dv)
    ds = p * dp_b
    dk = _product_b(_quantize_ds(ds, STATIC_DS), _b_operands(qb, qsb), dk)
    return dk, dv


@triton.jit
def _kv_item(
    item,
    K8,
    KS,
    V8,
    VS,
    CuKV,
    CuPad,
    CuQPad,
    KTileSequence,
    NPAD,
    NQPAD,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = item // HEADS
    head = (item % HEADS).to(tl.int64)
    sequence = _sload(KTileSequence + (BLOCK // 64) * tile)
    pad0, pad1, kv0, kv1, q_start, q_end, _, _ = _sload_pairs(
        CuPad + sequence, CuKV + sequence, CuQPad + sequence, CuQPad + sequence
    )
    row = head * NPAD + tile * BLOCK
    kv_len = kv1 - kv0
    # Original key of the tile's first row (negative inside front padding).
    key = tile * BLOCK - pad0 - fwd._front(kv_len, pad1 - pad0)
    n_tiles = (q_end - q_start) // 128
    # Without queries, one tile of the spare (zero data, infinite LSE) rows
    # contributes nothing and keeps every item's control flow alike.
    q_start = tl.where(n_tiles > 0, q_start, NQPAD - 128)
    n_tiles = tl.maximum(n_tiles, 1)
    out_base = (kv0 + key).to(tl.int64) * (HEADS * 128) + head * 128
    return out_base, key, kv_len, head * NQPAD + q_start, n_tiles, row


@triton.jit
def _dkdv_kernel(
    K8,
    KS,
    V8,
    VS,
    Q8,
    QS,
    D8,
    DS,
    LSEP,
    DELTAP,
    DK,
    DV,
    CuKV,
    CuPad,
    CuQPad,
    KTileSequence,
    BATCH,
    NPAD,
    NQPAD,
    qk_scale,
    sm_scale,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
    PREFETCH: tl.constexpr,
    STATIC_DS: tl.constexpr,
):
    D: tl.constexpr = 128
    HN: tl.constexpr = 64
    n_items = _sload(CuPad + BATCH) // BLOCK * HEADS
    base, n_ctas, n_mine = _persistent(n_items)
    if base >= n_items:
        return
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, HN)
    k_off = offs_d[:, None] + offs_n[None, :] * D
    s_off = offs_n[:, None]
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], _Q_LDS_BASES, [D, HN]
    )
    bufs = (
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((128,), tl.float32, 2),
        tlx.local_alloc((128,), tl.float32, 2),
    )
    meta = _kv_item(
        base,
        K8,
        KS,
        V8,
        VS,
        CuKV,
        CuPad,
        CuQPad,
        KTileSequence,
        NPAD,
        NQPAD,
        HEADS,
        BLOCK,
    )
    if PREFETCH:
        k = _load_operand(K8, KS, meta[5], BLOCK)
        v = _load_operand(V8, VS, meta[5], BLOCK)
    _issue_q_tile(
        Q8,
        QS,
        D8,
        DS,
        LSEP,
        DELTAP,
        meta[3],
        k_off,
        s_off,
        _views10(bufs, 0),
    )
    step = 0
    # An item's dK/dV stores issue behind the next item's first tile copy:
    # stores also count in vmcnt, so stored before a wait they would stall it.
    # Nothing to store before the first item: rows [-2, -1). The initial
    # values must be distinct SSA values from each other and from meta's, or
    # the frontend merges their loop-carried slots.
    prev_dk = _zeros4(BLOCK)
    prev_dv = _zeros4(BLOCK)
    prev_base = tl.zeros((), tl.int64)
    prev_key = tl.full((), 2, tl.int32)
    prev_len = tl.full((), 1, tl.int32)
    for w in tl.range(0, n_mine, num_stages=1):
        item = base + w * n_ctas
        nxt = tl.where(w + 1 < n_mine, item + n_ctas, item)
        # The next item's metadata loads behind this item's tiles. Its K/V
        # operands do too with PREFETCH (short query loops), at the cost of
        # the tiles' VGPRs.
        next_meta = _kv_item(
            nxt,
            K8,
            KS,
            V8,
            VS,
            CuKV,
            CuPad,
            CuQPad,
            KTileSequence,
            NPAD,
            NQPAD,
            HEADS,
            BLOCK,
        )
        out_base, key, kv_len, q_row, n_tiles, k_row = meta
        if PREFETCH:
            next_k = _load_operand(K8, KS, next_meta[5], BLOCK)
            next_v = _load_operand(V8, VS, next_meta[5], BLOCK)
        else:
            k = _load_operand(K8, KS, k_row, BLOCK)
            v = _load_operand(V8, VS, k_row, BLOCK)
        # First tile, peeled so the previous item's accumulators die before
        # the remaining tiles.
        slot = step % 2
        tlx.async_load_wait_group(0)
        tl.debug_barrier()
        row = tl.where(n_tiles > 1, q_row + 128, next_meta[3])
        _store_rows(
            DK, prev_base, prev_dk, sm_scale, -prev_key, prev_len - prev_key, HEADS * D
        )
        _store_rows(
            DV, prev_base, prev_dv, 1.0, -prev_key, prev_len - prev_key, HEADS * D
        )
        dk, dv = _dkdv_tile(
            k,
            v,
            qk_scale,
            _views10(bufs, slot),
            _zeros4(BLOCK),
            _zeros4(BLOCK),
            (
                Q8,
                QS,
                D8,
                DS,
                LSEP,
                DELTAP,
                row,
                k_off,
                s_off,
            ),
            _views10(bufs, 1 - slot),
            True,
            STATIC_DS,
        )
        step += 1
        for t in tl.range(1, n_tiles, num_stages=1):
            slot = step % 2
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            row = tl.where(t + 1 < n_tiles, q_row + (t + 1) * 128, next_meta[3])
            dk, dv = _dkdv_tile(
                k,
                v,
                qk_scale,
                _views10(bufs, slot),
                dk,
                dv,
                (
                    Q8,
                    QS,
                    D8,
                    DS,
                    LSEP,
                    DELTAP,
                    row,
                    k_off,
                    s_off,
                ),
                _views10(bufs, 1 - slot),
                True,
                STATIC_DS,
            )
            step += 1
        prev_dk, prev_dv = dk, dv
        prev_base, prev_key, prev_len = out_base, key, kv_len
        meta = next_meta
        if PREFETCH:
            k, v = next_k, next_v
    tlx.async_load_wait_group(0)
    _store_rows(
        DK, prev_base, prev_dk, sm_scale, -prev_key, prev_len - prev_key, HEADS * D
    )
    _store_rows(DV, prev_base, prev_dv, 1.0, -prev_key, prev_len - prev_key, HEADS * D)


@triton.jit
def _stream_item(
    item,
    n_tiles,
    CuKV,
    CuPad,
    CuQPad,
    KTileSequence,
    BATCH,
    NPAD,
    NQPAD,
    HEADS: tl.constexpr,
    KEY_ROWS: tl.constexpr,
):
    # Items are (head, key tile), head-major: a run of items shares its
    # sequence's query tiles. Spare tiles get no queries and store nothing.
    head = item // n_tiles
    tile = item % n_tiles
    sequence = _sload(KTileSequence + (KEY_ROWS // 64) * tile)
    valid = sequence < BATCH
    sequence = tl.minimum(sequence, BATCH - 1)
    pad_start, pad_end, kv0, kv1, q_start, q_end, _, _ = _sload_pairs(
        CuPad + sequence, CuKV + sequence, CuQPad + sequence, CuQPad + sequence
    )
    kv_len = tl.where(valid, kv1 - kv0, 0)
    key = tile * KEY_ROWS - pad_start - fwd._front(kv_len, pad_end - pad_start)
    n_q = tl.where(valid, (q_end - q_start) // 128, 0)
    out_base = (kv0 + key).to(tl.int64) * (HEADS * 128) + head * 128
    return (
        sequence + head * BATCH,
        out_base,
        key,
        kv_len,
        head.to(tl.int64) * NQPAD + q_start,
        n_q,
        head.to(tl.int64) * NPAD + tile * KEY_ROWS,
    )


@triton.jit
def _dkdv_stream_kernel(
    K8,
    KS,
    V8,
    VS,
    Q8,
    QS,
    D8,
    DS,
    LSEP,
    DELTAP,
    DK,
    DV,
    CuKV,
    CuPad,
    CuQPad,
    KTileSequence,
    BATCH,
    NPAD,
    NQPAD,
    qk_scale,
    sm_scale,
    HEADS: tl.constexpr,
    STATIC_DS: tl.constexpr,
    KEY_ROWS: tl.constexpr,
):
    """dK and dV for sequences of at most two query tiles: each workgroup
    takes a contiguous run of key tiles, keeps the current sequence's query
    tiles in LDS and streams the keys, prefetching the next K/V operands."""
    D: tl.constexpr = 128
    HN: tl.constexpr = 64
    n_tiles = _sload(CuPad + BATCH) // KEY_ROWS
    n_items = n_tiles * HEADS
    per = tl.cdiv(n_items, tl.num_programs(0))
    start = tl.program_id(0) * per
    end = tl.minimum(start + per, n_items)
    if start >= end:
        return
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, HN)
    k_off = offs_d[:, None] + offs_n[None, :] * D
    s_off = offs_n[:, None]
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(1024, 16)], _Q_LDS_BASES, [D, HN]
    )
    # Slot j holds query tile j of the current sequence.
    bufs = (
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((D, HN), tl.float8e4nv, 2, layout=k_layout),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((HN, 1), tl.int32, 2),
        tlx.local_alloc((128,), tl.float32, 2),
        tlx.local_alloc((128,), tl.float32, 2),
    )
    meta = _stream_item(
        start,
        n_tiles,
        CuKV,
        CuPad,
        CuQPad,
        KTileSequence,
        BATCH,
        NPAD,
        NQPAD,
        HEADS,
        KEY_ROWS,
    )
    k = _load_operand(K8, KS, meta[6], KEY_ROWS)
    v = _load_operand(V8, VS, meta[6], KEY_ROWS)
    loaded = tl.full((), -1, tl.int32)
    for item in tl.range(start, end, num_stages=1):
        owner, out_base, key, kv_len, q_row, n_q, k_row = meta
        if owner != loaded:
            tl.debug_barrier()
            for j in tl.range(0, n_q, num_stages=1):
                _issue_q_tile(
                    Q8,
                    QS,
                    D8,
                    DS,
                    LSEP,
                    DELTAP,
                    q_row + j * 128,
                    k_off,
                    s_off,
                    _views10(bufs, j),
                )
            tlx.async_load_wait_group(0)
            tl.debug_barrier()
            loaded = owner
        # The next item's operands load behind this item's tiles.
        next_item = tl.minimum(item + 1, end - 1)
        if next_item == item:
            next_meta = meta
        else:
            next_head = next_item // n_tiles
            next_tile = next_item % n_tiles
            next_sequence = _sload(KTileSequence + (KEY_ROWS // 64) * next_tile)
            same_owner = (next_head == owner // BATCH) & (
                next_sequence == owner % BATCH
            )
            if same_owner:
                next_meta = (
                    owner,
                    out_base + KEY_ROWS * HEADS * D,
                    key + KEY_ROWS,
                    kv_len,
                    q_row,
                    n_q,
                    k_row + KEY_ROWS,
                )
            else:
                next_meta = _stream_item(
                    next_item,
                    n_tiles,
                    CuKV,
                    CuPad,
                    CuQPad,
                    KTileSequence,
                    BATCH,
                    NPAD,
                    NQPAD,
                    HEADS,
                    KEY_ROWS,
                )
        next_k = _load_operand(K8, KS, next_meta[6], KEY_ROWS)
        next_v = _load_operand(V8, VS, next_meta[6], KEY_ROWS)
        dk = _zeros4(KEY_ROWS)
        dv = _zeros4(KEY_ROWS)
        for j in tl.range(0, n_q, num_stages=1):
            dk, dv = _dkdv_tile(
                k,
                v,
                qk_scale,
                _views10(bufs, j),
                dk,
                dv,
                None,
                None,
                False,
                STATIC_DS,
            )
        _store_rows(DK, out_base, dk, sm_scale, -key, kv_len - key, HEADS * D)
        _store_rows(DV, out_base, dv, 1.0, -key, kv_len - key, HEADS * D)
        meta = next_meta
        k, v = next_k, next_v


def _operands(npad, device, v_layout=True):
    """FP8 data and E8M0 scales in the K layout and (optionally, else None)
    in the V layout."""
    k8 = torch.empty(
        (_HEADS, npad, _HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty((_HEADS, npad, 4), device=device, dtype=torch.uint8)
    if not v_layout:
        return k8, None, scales, None
    v8 = torch.empty(
        (_HEADS, npad // 64, _HEAD_DIM, 64), device=device, dtype=torch.float8_e4m3fn
    )
    return k8, v8, scales, torch.empty_like(scales)


def backward(
    do,
    q,
    k,
    v,
    out,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    sm_scale,
    kv_state=None,
    kv_shape=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    """(dq, dk, dv) for packed contiguous BF16 [tokens, 4, 128] tensors.

    kv_state is the forward's with backward_state=True; given it, K and V
    may be None, with their shape in kv_shape. A max_seqlen_q of at most 256
    (cross-attention) streams dK/dV over the keys."""
    device = q.device
    batch = cu_seqlens_k.numel() - 1
    kv_shape = k.shape if k is not None else kv_shape
    total_q, total_kv = q.shape[0], kv_shape[0]
    do = do.contiguous()
    qmeta = fwd._metadata(cu_seqlens_q, cu_seqlens_k, total_q, total_kv, _BLOCK)
    # Roles swapped: query padding, key tiles as items, query preparation tiles.
    kmeta = fwd._metadata(
        cu_seqlens_k, cu_seqlens_q, total_kv, total_q, _BLOCK, pad=_BLOCK
    )
    npad, nqpad = qmeta["npad"], kmeta["npad"]
    if kv_state is None:
        k8, _, ks, _ = _operands(npad, device, v_layout=False)
        v8 = torch.empty_like(k8)
        vs = torch.empty_like(ks)
        cast(Any, _prepare_kv)[(qmeta["tile_capacity"] + 4, _HEADS)](
            k,
            v,
            k8,
            ks,
            v8,
            vs,
            cu_seqlens_k,
            qmeta["cu_pad"],
            qmeta["tile_sequence"],
            qmeta["tile_capacity"],
            batch,
            npad,
            HEADS=_HEADS,
            num_warps=2,
        )
    else:
        k8, ks, _, _, v8, vs = kv_state
    q8, _, qs, _ = _operands(nqpad, device, v_layout=False)
    d8, _, ds, _ = _operands(nqpad, device, v_layout=False)
    lsep = torch.empty((_HEADS, nqpad), device=device, dtype=torch.float32)
    deltap = torch.empty_like(lsep)
    cast(Any, _prepare_q)[(kmeta["tile_capacity"] + 2, _HEADS)](
        q,
        do,
        out,
        lse,
        q8,
        qs,
        d8,
        ds,
        lsep,
        deltap,
        cu_seqlens_q,
        kmeta["cu_pad"],
        kmeta["tile_sequence"],
        kmeta["tile_capacity"],
        batch,
        nqpad,
        total_q,
        HEADS=_HEADS,
        num_warps=2,
    )
    qk_scale = sm_scale * _LOG2E
    # Split each query tile's keys when query tiles alone cannot fill the GPU
    # (cross-attention to long KV).
    query_items = (total_q // 128 + batch) * _HEADS
    splits = max(1, min(8, triton.cdiv(4 * 256, query_items)))
    splits = max(1, min(splits, total_kv // max(1, batch * 128 * 4)))
    if max_seqlen_k is not None and max_seqlen_k > _SHORT_DS_MAX_SEQLEN_K:
        splits = min(splits, 2)
    static_ds = max_seqlen_k is not None
    partial_dq = (
        splits == 2
        and max_seqlen_k is not None
        and max_seqlen_k > _SHORT_DS_MAX_SEQLEN_K
    )
    partial_stride = q.numel()
    dq = (
        torch.empty_like(q)
        if partial_dq or splits == 1
        else torch.zeros(q.shape, device=device, dtype=torch.float32)
    )
    dq_acc = (
        torch.empty(2 * partial_stride, device=device, dtype=torch.float32)
        if partial_dq
        else dq
    )
    # Two workgroups per CU (two waves per SIMD, 256 registers each).
    grid = min(512, triton.cdiv(nqpad // 128 * _HEADS * splits, 8) * 8)
    cast(Any, _dq_kernel)[(grid,)](
        q8,
        qs,
        d8,
        ds,
        k8,
        ks,
        v8,
        vs,
        lsep,
        deltap,
        dq_acc,
        cu_seqlens_q,
        cu_seqlens_k,
        qmeta["cu_pad"],
        kmeta["cu_pad"],
        kmeta["tile_sequence"],
        batch,
        npad,
        nqpad,
        qk_scale,
        sm_scale,
        splits,
        partial_stride,
        HEADS=_HEADS,
        BLOCK=_BLOCK,
        ATOMIC=splits > 1 and not partial_dq,
        PARTIAL=partial_dq,
        STATIC_DS=static_ds,
        **_DQ_LAUNCH_OPTIONS,
    )
    if partial_dq:
        block = 1024
        cast(Any, _dq_partial_reduce)[(triton.cdiv(partial_stride, block),)](
            dq_acc,
            dq,
            partial_stride,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
    elif splits > 1:
        dq = dq.to(q.dtype)
    dk = torch.empty(kv_shape, device=device, dtype=q.dtype)
    dv = torch.empty_like(dk)
    if max_seqlen_q is not None and max_seqlen_q <= 256:
        key_rows = 64 if static_ds else 128
        long_cross = max_seqlen_k is not None and max_seqlen_k > _SHORT_DS_MAX_SEQLEN_K
        stream_grid = min(
            512 if long_cross else 1024 if static_ds else 256,
            npad // key_rows * _HEADS,
        )
        cast(Any, _dkdv_stream_kernel)[(stream_grid,)](
            k8,
            ks,
            v8,
            vs,
            q8,
            qs,
            d8,
            ds,
            lsep,
            deltap,
            dk,
            dv,
            cu_seqlens_k,
            qmeta["cu_pad"],
            kmeta["cu_pad"],
            qmeta["tile_sequence"],
            batch,
            npad,
            nqpad,
            qk_scale,
            sm_scale,
            HEADS=_HEADS,
            STATIC_DS=static_ds,
            KEY_ROWS=key_rows,
            **{**_LAUNCH_OPTIONS, "num_warps": key_rows // 32},
        )
        return dq, dk, dv
    grid = min(256, triton.cdiv(npad // _BLOCK * _HEADS, 8) * 8)
    cast(Any, _dkdv_kernel)[(grid,)](
        k8,
        ks,
        v8,
        vs,
        q8,
        qs,
        d8,
        ds,
        lsep,
        deltap,
        dk,
        dv,
        cu_seqlens_k,
        qmeta["cu_pad"],
        kmeta["cu_pad"],
        qmeta["tile_sequence"],
        batch,
        npad,
        nqpad,
        qk_scale,
        sm_scale,
        HEADS=_HEADS,
        BLOCK=_BLOCK,
        # Short query loops prefetch the next K/V operands.
        PREFETCH=total_q < 512 * batch,
        STATIC_DS=False,
        **_LAUNCH_OPTIONS,
    )
    return dq, dk, dv
