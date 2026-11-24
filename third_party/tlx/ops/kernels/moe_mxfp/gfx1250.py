"""MXFP4 A16W4 routed MoE on gfx1250: BF16 activations, MXFP4 weights, TDM-pipelined grouped expert GEMMs.

Same contract and numerics as ``gfx950_a16w4.moe_mxfp4_a16w4``: the stage-1 gate/up GEMM, the SiLU and the stage-2
down GEMM take BF16 inputs, only the stage-1 output is rounded to BF16 between the GEMMs, and the finalize (optional
post-norm and routing-weighted sum) rounds at the same points as ``moe.gfx950._finalize_kernel``. It reads one slot
row at a time instead of all top-k rows at once, which on gfx1250 is 2.6x faster (62 vs 160 us at T = 8192, top-8,
H = 4096).

The weights stay packed in LDS. Each wave loads its words straight into the layout whose decoded values are the BF16
WMMA B operand, decodes them in registers with ``tlx.amd_scaled_upcast_fp4`` (``v_cvt_scale_pk8_bf16_fp4``, E8M0
scale applied in the convert) and runs ``v_wmma_f32_16x16x32_bf16``; four waves sit side by side along N, so no weight
is decoded twice in a program.

Routing runs on the device without a host readback: an atomic count gives every local slot its rank in its expert,
a grouped tile map gives every program one (expert, M tile, N tile), and the slots' activation rows are copied into
one contiguous block per expert. That copy is a ``[T * topk, K]`` BF16 scratch buffer (512 MiB at T = 8192, K = 4096,
top-8), sized for every slot because only the device knows how many are local; the stage-2 output reuses it when H <= K.
Gathering the rows in the A-tile load would remove it, but TLX exposes no AMD TDM gather, and a gather could not share
the fused A / weight TDM op. Each program streams its operands global -> LDS through a ring of ``NUM_BUFFERS``
stages, each one fused TDM op (waves 0-1 load the A tile, waves 2-3 the weight tiles), and loads its weight-scale
panel once; one workgroup barrier per K step makes a stage visible, and the stage refilled after it is the one read in
the previous step. Descriptor shapes start at the program's tile, so TDM bounds checks zero-fill rows past the
expert's last row. With ``HALF_TAIL`` an expert's last tile with at most BLOCK_M / 2 rows runs a half-height tile
from the same ring. Stage 2 stores each row at its routing slot, and the finalize reads only each token's local slots.

Weights are read in the checkpoint's natural (Quark) layout: ``w13`` uint8 ``[E, 2I, K/2]`` with ``w13_scale`` uint8
``[E, 2I, K/32]`` (gate rows first), and ``w2`` uint8 ``[E, H, I/2]`` with ``w2_scale`` uint8 ``[E, H, I/32]``.
Packed bytes hold the even element in the low nibble.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.jit import constexpr_function

from triton.tlx.ops.kernels.moe_mxfp.gfx950_a16w4 import _check_inputs, _word_strided

_LOG2E = tl.constexpr(1.4426950408889634)
_BLOCK_N = 128
_NUM_WARPS = 4


def _alloc(shape, dtype, device):
    """Workspace allocation (tests replace it to place guard regions around the op's buffers)."""
    return torch.empty(shape, dtype=dtype, device=device)


@triton.jit
def _route_count_kernel(ids_ptr, pos_ptr, counts_ptr, n, expert_start, E: tl.constexpr, BLOCK: tl.constexpr):
    """Each routing slot's rank within its local expert (atomic, so not in slot order: every row is computed
    independently, so the row order inside an expert does not change any result); -1 for slots routed elsewhere."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = offs < n
    local = tl.load(ids_ptr + offs, mask=in_range, other=-1).to(tl.int32) - expert_start
    ok = in_range & (local >= 0) & (local < E)
    pos = tl.atomic_add(counts_ptr + tl.where(ok, local, 0), 1, mask=ok, sem="relaxed")
    tl.store(pos_ptr + offs, tl.where(ok, pos, -1), mask=in_range)


@triton.jit
def _tile_map_kernel(split_sizes, group_base, tile_meta0, tile_meta1, tile_count, NUM_GROUPS: tl.constexpr,
                     GROUP_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, MAX_TILES_PER_GROUP: tl.constexpr):
    """``grouped_gemm_mxfp``'s device tile map, plus each group's first row."""
    group = tl.program_id(0)
    offsets = tl.arange(0, GROUP_BLOCK)
    sizes = tl.load(split_sizes + offsets, mask=offsets < NUM_GROUPS, other=0)
    tiles = tl.cdiv(sizes, BLOCK_M)
    before = offsets < group
    row_base = tl.sum(tl.where(before, sizes, 0), axis=0).to(tl.int64)
    tile_base = tl.sum(tl.where(before, tiles, 0), axis=0).to(tl.int64)
    group_rows = tl.load(split_sizes + group).to(tl.int64)
    group_tiles = tl.cdiv(group_rows, BLOCK_M)
    tl.store(group_base + group, row_base.to(tl.int32))
    local_tile = tl.arange(0, MAX_TILES_PER_GROUP)
    mask = local_tile < group_tiles
    meta0 = row_base | (group_rows << 32)
    meta1 = group.to(tl.int64) | (local_tile.to(tl.int64) << 32)
    tl.store(tile_meta0 + tile_base + local_tile, meta0, mask=mask)
    tl.store(tile_meta1 + tile_base + local_tile, meta1, mask=mask)
    if group == NUM_GROUPS - 1:
        tl.store(tile_count, tile_base + group_tiles)
        tl.store(tile_count + 1, row_base + group_rows)


@triton.jit
def _place_rows_kernel(x_ptr, xs_ptr, order_ptr, ids_ptr, pos_ptr, group_base, n, expert_start, stride_xm,
                       K: tl.constexpr, TOP_K: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_K: tl.constexpr):
    """Row of slot s (local expert e, rank p) = group_base[e] + p: copy token s // TOP_K's activations there and record
    the slot (stage 2 stores row r at slot order[r])."""
    slots = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    in_range = slots < n
    pos = tl.load(pos_ptr + slots, mask=in_range, other=-1)
    ok = pos >= 0
    local = tl.load(ids_ptr + slots, mask=ok, other=0).to(tl.int32) - expert_start
    row = (tl.load(group_base + local, mask=ok, other=0) + pos).to(tl.int64)
    tl.store(order_ptr + row, slots.to(tl.int64), mask=ok)
    token = (slots // TOP_K).to(tl.int64)
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = ok[:, None] & (offs < K)[None, :]
        v = tl.load(x_ptr + token[:, None] * stride_xm + offs[None, :], mask=m)
        tl.store(xs_ptr + row[:, None] * K + offs[None, :], v, mask=m)


@constexpr_function
def _linear(shape, lane, warp, reg):
    """tlx.layout from per-bit lane / warp / register coordinates of a row-major tile."""
    strides = []
    for d in range(len(shape)):
        s = 1
        for later in shape[d + 1:]:
            s *= later
        strides.append(s)

    def flat(c):
        return sum(ci * si for ci, si in zip(c, strides))

    th = tuple(flat(c) for c in list(lane) + list(warp))
    va = tuple(flat(c) for c in reg)
    return tlx.layout(shape=((2, ) * len(th), (2, ) * len(va)), stride=(th, va))


@constexpr_function
def _bits(n):
    return [1 << i for i in range(n.bit_length() - 1)]


@constexpr_function
def _wmma_layout():
    # Four waves side by side along N, 16 columns each, repeating every 64 columns.
    return tlx.amd_wmma_layout(warp_bases=[[0, 1], [0, 2]], reg_bases=[], instr_shape=(16, 16, 32), version=3,
                               transposed=True)


@constexpr_function
def _w_layout(BN, BKW, kind):
    """Layouts of a [BN, BKW] int32 tile of packed weights (eight E2M1 values per word along K) that land the decoded
    BF16 values in the WMMA B-operand layout: a lane holds column n = lane % 16 (+16 per wave, +64 per register), and
    the two half-waves hold alternate words of each 16-word (128-value) K step.

    kind 0: [BN, BKW] words; kind 1: [BN, BKW, 8] decoded values; kind 2: [BN, BKS, 4] scale broadcast (word = 4 * ks +
    i, one E8M0 byte per four words).
    """
    n_lane = [(1, 0), (2, 0), (4, 0), (8, 0)]
    n_warp = [(16, 0), (32, 0)]
    n_rep = [(64 * b, 0) for b in _bits(BN // 64)]
    w_hi = [(0, b) for b in _bits(BKW)][1:]
    if kind == 0:
        return _linear((BN, BKW), n_lane + [(0, 1)], n_warp, w_hi + n_rep)
    if kind == 1:
        return _linear((BN, BKW, 8), [(a, b, 0) for a, b in n_lane] + [(0, 1, 0)], [(a, b, 0) for a, b in n_warp],
                       [(0, 0, 1), (0, 0, 2), (0, 0, 4)] + [(a, b, 0) for a, b in w_hi + n_rep])
    assert BKW >= 8
    ks_hi = [(0, b, 0) for b in _bits(BKW // 4)]
    return _linear((BN, BKW // 4, 4), [(a, b, 0) for a, b in n_lane] + [(0, 0, 1)], [(a, b, 0) for a, b in n_warp],
                   [(0, 0, 2)] + ks_hi + [(a, b, 0) for a, b in n_rep])


@constexpr_function
def _sliced(parent, shape):
    """``parent`` (a 3-D layout of ``shape``) sliced along its last dim."""

    class _Slice(tlx.slice_layout):

        def to_ir(self, builder, shape_=None, element_type=None):
            return builder.make_slice_encoding_attr(2, self.parent.to_ir(builder, shape, element_type))

    return _Slice(parent, 2)


@triton.jit
def _load_b(w_buf, s_buf, slot, k, BN: tl.constexpr, BKW: tl.constexpr, BKS: tl.constexpr, b_op: tl.constexpr):
    """[BN, BKW] packed words of slot ``slot`` and the K step's E8M0 scales -> BF16 [BKW * 8, BN] B operand, exact."""
    l_w: tl.constexpr = _w_layout(BN, BKW, 0)
    l_s3: tl.constexpr = _w_layout(BN, BKW, 2)
    w = tlx.local_load(tlx.local_view(w_buf, slot), layout=l_w)
    s = tlx.local_load(tlx.local_slice(tlx.local_view(s_buf, 0), [0, k * BKS], [BN, BKS]),
                       layout=_sliced(l_s3, (BN, BKS, 4)))
    # Split each word into its bytes in memory order, so its eight values stay in one thread's consecutive registers.
    lo = tl.join(w.to(tl.uint8), (w >> 16).to(tl.uint8))
    hi = tl.join((w >> 8).to(tl.uint8), (w >> 24).to(tl.uint8))
    v = tlx.amd_scaled_upcast_fp4(tl.reshape(tl.join(lo, hi), (BN, BKW * 4)), s, tl.bfloat16, axis=1)
    return tlx.require_layout(tl.trans(v), b_op)


@triton.jit
def _a_view(a_buf, slot, BM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    v = tlx.local_view(a_buf, slot)
    if BM < BLOCK_M:
        v = tlx.local_slice(v, [0, 0], [BM, BLOCK_K])
    return v


@triton.jit
def _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, k_iter, pred, BM: tl.constexpr, BLOCK_M: tl.constexpr,
                 BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr, FUSE_SILU_MUL: tl.constexpr):
    """One fused TDM op for K step ``k_iter``: waves 0-1 load the A tile, waves 2-3 the weight tile(s)."""
    slot = k_iter % NUM_BUFFERS
    BKW: tl.constexpr = BLOCK_K // 8
    a_d = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, k_iter * BLOCK_K], pred=pred, clamp_bounds=True)
    b_d = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, k_iter * BKW], pred=pred, clamp_bounds=True)
    a_dst = _a_view(a_buf, slot, BM, BLOCK_M, BLOCK_K)
    if FUSE_SILU_MUL:
        b2_d = tlx.update_tensor_descriptor(b2_desc, add_offsets=[0, k_iter * BKW], pred=pred, clamp_bounds=True)
        tlx.async_amd_descriptor_load_fused([(a_d, a_dst, 3), (b_d, tlx.local_view(b_buf, slot), 4),
                                             (b2_d, tlx.local_view(b2_buf, slot), 8)])
    else:
        tlx.async_amd_descriptor_load_fused([(a_d, a_dst, 3), (b_d, tlx.local_view(b_buf, slot), 12)])


@triton.jit
def _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc, a_buf, b_buf, bs_buf,
          b2_buf, bs2_buf, K: tl.constexpr, BM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr, FUSE_SILU_MUL: tl.constexpr, STORE_ROWS: tl.constexpr):
    """One [BM, BLOCK_N] output tile (BM <= BLOCK_M, the rows of the shared LDS A ring)."""
    K_ITERS: tl.constexpr = K // BLOCK_K
    BKS: tl.constexpr = BLOCK_K // 32
    BKW: tl.constexpr = BLOCK_K // 8
    mma: tl.constexpr = _wmma_layout()
    a_op: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    b_op: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_desc = tl.make_tensor_descriptor(a_ptr + row0 * K, shape=[rows_left, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BM, BLOCK_K])
    tlx.async_amd_descriptor_load(bs_desc, tlx.local_view(bs_buf, 0), [0, 0])
    if FUSE_SILU_MUL:
        tlx.async_amd_descriptor_load(bs2_desc, tlx.local_view(bs2_buf, 0), [0, 0])
    for s in tl.static_range(NUM_BUFFERS - 1):
        _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, s, s < K_ITERS, BM, BLOCK_M, BLOCK_K, NUM_BUFFERS,
                     FUSE_SILU_MUL)
    acc = tlx.zeros((BM, BLOCK_N), tl.float32, layout=mma)
    acc2 = tlx.zeros((BM, BLOCK_N), tl.float32, layout=mma)
    for k in range(K_ITERS):
        tlx.async_amd_descriptor_wait(NUM_BUFFERS - 2)
        tlx.workgroup_barrier()
        slot = k % NUM_BUFFERS
        a = tlx.local_load(_a_view(a_buf, slot, BM, BLOCK_M, BLOCK_K), layout=a_op)
        b = _load_b(b_buf, bs_buf, slot, k, BLOCK_N, BKW, BKS, b_op)
        if FUSE_SILU_MUL:
            b2 = _load_b(b2_buf, bs2_buf, slot, k, BLOCK_N, BKW, BKS, b_op)
        nk = k + NUM_BUFFERS - 1
        _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, nk, nk < K_ITERS, BM, BLOCK_M, BLOCK_K, NUM_BUFFERS,
                     FUSE_SILU_MUL)
        acc = tl.dot(a, b, acc)
        if FUSE_SILU_MUL:
            acc2 = tl.dot(a, b2, acc2)
    tlx.async_amd_descriptor_wait(0)
    if FUSE_SILU_MUL:
        out = acc / (1.0 + tl.exp2(acc * -_LOG2E)) * acc2
    else:
        out = acc
    offs_m = tl.arange(0, BM)
    offs_n = off_n + tl.arange(0, BLOCK_N)
    rmask = offs_m < rows_left
    if STORE_ROWS:
        dst = tl.load(store_rows + row0 + offs_m, mask=rmask, other=0).to(tl.int64)
    else:
        dst = row0 + offs_m
    mask = rmask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptr + dst[:, None] * N + offs_n[None, :], out.to(tl.bfloat16), mask=mask)


@triton.jit
def _a16w4_grouped_kernel(
    a_ptr,  # bf16 [rows, K], rows sorted by expert
    b_ptr,  # int32 [E, NB, K/8]: packed E2M1 words
    bs_ptr,  # uint8 [E, NB, K/32]
    c_ptr,  # bf16 [rows, N] (STORE_ROWS: [slots, N], row r stored at slot store_rows[r])
    store_rows,
    tile_meta0,
    tile_meta1,
    tile_count,
    N,  # output columns; stage 1 (FUSE_SILU_MUL): I, with the up rows starting at row N of B
    stride_be,  # in words
    stride_bse,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    FUSE_SILU_MUL: tl.constexpr,
    STORE_ROWS: tl.constexpr,
    HALF_TAIL: tl.constexpr,
):
    tl.static_assert(K % BLOCK_K == 0 and BLOCK_K % 64 == 0 and BLOCK_N % 64 == 0 and BLOCK_N <= 128)
    tl.static_assert(NUM_BUFFERS >= 2)
    KS: tl.constexpr = K // 32
    KW: tl.constexpr = K // 8
    BKW: tl.constexpr = BLOCK_K // 8

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    m_tile = pid // num_pid_n
    pid_n = pid % num_pid_n
    if m_tile >= tl.load(tile_count):
        return
    meta0 = tl.load(tile_meta0 + m_tile)
    meta1 = tl.load(tile_meta1 + m_tile)
    row_base = meta0 & 0xFFFFFFFF
    group_rows = meta0 >> 32
    expert = (meta1 & 0xFFFFFFFF).to(tl.int64)
    pid_m = meta1 >> 32
    row0 = row_base + pid_m * BLOCK_M
    rows_left = group_rows - pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    b_base = b_ptr + expert * stride_be
    bs_base = bs_ptr + expert * stride_bse
    b_desc = tl.make_tensor_descriptor(b_base + off_n * KW, shape=[N - off_n, KW], strides=[KW, tl.constexpr(1)],
                                       block_shape=[BLOCK_N, BKW])
    bs_desc = tl.make_tensor_descriptor(bs_base + off_n * KS, shape=[N - off_n, KS], strides=[KS, tl.constexpr(1)],
                                        block_shape=[BLOCK_N, KS])
    if FUSE_SILU_MUL:
        b2_desc = tl.make_tensor_descriptor(b_base + (N + off_n) * KW, shape=[N - off_n, KW],
                                            strides=[KW, tl.constexpr(1)], block_shape=[BLOCK_N, BKW])
        bs2_desc = tl.make_tensor_descriptor(bs_base + (N + off_n) * KS, shape=[N - off_n, KS],
                                             strides=[KS, tl.constexpr(1)], block_shape=[BLOCK_N, KS])
    else:
        b2_desc = b_desc
        bs2_desc = bs_desc

    w_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BKW, 4]], [BLOCK_N, BKW], [1, 0])
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                 [1, 0])
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=a_layout)
    b_buf = tlx.local_alloc((BLOCK_N, BKW), tl.int32, NUM_BUFFERS, layout=w_layout)
    s_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[KS, 16]], [BLOCK_N, KS], [1, 0])
    bs_buf = tlx.local_alloc((BLOCK_N, KS), tl.uint8, 1, layout=s_layout)
    if FUSE_SILU_MUL:
        b2_buf = tlx.local_alloc((BLOCK_N, BKW), tl.int32, NUM_BUFFERS, layout=w_layout)
        bs2_buf = tlx.local_alloc((BLOCK_N, KS), tl.uint8, 1, layout=s_layout)
    else:
        b2_buf = b_buf
        bs2_buf = bs_buf
    if HALF_TAIL and BLOCK_M >= 32 and rows_left <= BLOCK_M // 2:
        _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc, a_buf, b_buf,
              bs_buf, b2_buf, bs2_buf, K, BLOCK_M // 2, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL,
              STORE_ROWS)
    else:
        _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc, a_buf, b_buf,
              bs_buf, b2_buf, bs2_buf, K, BLOCK_M, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, STORE_ROWS)


@triton.jit
def _finalize_kernel(y_ptr, w_ptr, scores_ptr, ids_ptr, out_ptr, H, local_start, num_local, eps, gain,
                     TOP_K: tl.constexpr, HAS_NORM: tl.constexpr, BLOCK_H: tl.constexpr):
    """out[m] = sum over token m's local slots of score * post_norm(y[slot]); one token per program, one slot row at a
    time."""
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if HAS_NORM:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0)
        scale = (w.to(tl.float32) + gain).to(w_ptr.dtype.element_ty).to(tl.float32)
    acc = tl.zeros((BLOCK_H, ), dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        e = tl.load(ids_ptr + m * TOP_K + j)
        if (e >= local_start) & (e < local_start + num_local):
            sc = tl.load(scores_ptr + m * TOP_K + j).to(tl.float32)
            y = tl.load(y_ptr + (m * TOP_K + j).to(tl.int64) * H + offs, mask=mask, other=0.0).to(tl.float32)
            if HAS_NORM:
                y = (y * scale).to(tl.bfloat16).to(tl.float32)
                inv = 1.0 / tl.sqrt(tl.sum(y * y, axis=0) / H + eps)
                acc += y * (sc * inv)
            else:
                acc += y * sc
    tl.store(out_ptr + m.to(tl.int64) * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def pick_config(T):
    """Tile choice by token count: rows per expert are small in decode and ~T * topk / E_total in prefill. Keys of
    ``stage2`` override the stage-2 (down GEMM) tiles."""
    if T <= 64:
        return dict(BLOCK_M=16, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=False)
    if T <= 512:
        return dict(BLOCK_M=32, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=False)
    return dict(BLOCK_M=64, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=True, stage2=dict(BLOCK_M=128, BLOCK_K=64))


def _tile_map(split_sizes, rows_max, block_m):
    groups = split_sizes.numel()
    max_tiles = triton.cdiv(rows_max, block_m) + groups
    meta0 = torch.empty(max_tiles, dtype=torch.int64, device=split_sizes.device)
    meta1 = torch.empty_like(meta0)
    count = torch.zeros(2, dtype=torch.int64, device=split_sizes.device)
    group_base = torch.empty(groups, dtype=torch.int32, device=split_sizes.device)
    _tile_map_kernel[(groups, )](split_sizes, group_base, meta0, meta1, count, NUM_GROUPS=groups,
                                 GROUP_BLOCK=triton.next_power_of_2(groups), BLOCK_M=block_m,
                                 MAX_TILES_PER_GROUP=triton.next_power_of_2(triton.cdiv(rows_max, block_m)))
    return meta0, meta1, count, max_tiles, group_base


def _launch(a, w, w_scale, out, store_rows, tmap, n_out, k, fuse_silu_mul, cfg):
    meta0, meta1, count, max_tiles = tmap[:4]
    grid = (max_tiles * (n_out // _BLOCK_N), )
    _a16w4_grouped_kernel[grid](a, w.view(torch.int32), w_scale,
                                out, store_rows if store_rows is not None else count, meta0, meta1, count, n_out,
                                w.stride(0) // 4, w_scale.stride(0), K=k, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=_BLOCK_N,
                                BLOCK_K=cfg["BLOCK_K"], NUM_BUFFERS=cfg["NUM_BUFFERS"], FUSE_SILU_MUL=fuse_silu_mul,
                                STORE_ROWS=store_rows is not None, HALF_TAIL=cfg.get("HALF_TAIL",
                                                                                     False), num_warps=_NUM_WARPS)


def _place_rows(x, xs, order, ids, pos, group_base, expert_start):
    """Copy each local slot's token row of ``x`` to its expert-grouped row of ``xs`` and record the slot in ``order``."""
    num_valid, K = xs.shape
    xg = x if x.stride(1) == 1 else x.contiguous()
    block_r = 32
    _place_rows_kernel[(triton.cdiv(num_valid,
                                    block_r), )](xg, xs, order, ids, pos, group_base, num_valid, expert_start,
                                                 xg.stride(0), K=K, TOP_K=ids.shape[1], BLOCK_R=block_r,
                                                 BLOCK_K=min(triton.next_power_of_2(K), 512), num_warps=8)


def _check_post_norm(op, post_norm_weight, H, device):
    """The finalize reads ``H`` elements of ``post_norm_weight`` and rounds the scale to its dtype."""
    if post_norm_weight is not None and (post_norm_weight.shape != (H, ) or not post_norm_weight.dtype.is_floating_point
                                         or post_norm_weight.device != device):
        raise ValueError(f"{op}: post_norm_weight must be a floating-point tensor of shape ({H},) on {device}")


def _finalize(y, topk_weights, ids, out, expert_start, num_local, post_norm_weight, post_norm_eps, post_norm_gain):
    """out[t] = sum over token t's local slots of routing weight * post_norm(y[slot]), y being [T * topk, H]."""
    T, topk = ids.shape
    H = out.shape[1]
    has_norm = post_norm_weight is not None
    _finalize_kernel[(T, )](y, post_norm_weight.contiguous() if has_norm else out,
                            topk_weights.to(torch.float32).contiguous(), ids, out, H, expert_start, num_local,
                            float(post_norm_eps), float(post_norm_gain), TOP_K=topk, HAS_NORM=has_norm,
                            BLOCK_H=triton.next_power_of_2(H), num_warps=8)


def moe_mxfp4_a16w4(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, expert_start=0, post_norm_weight=None,
                    post_norm_eps=1e-6, post_norm_gain=0.0, out=None, cfg=None):
    """BF16 x [T, K] through the local experts [expert_start, expert_start + E) with natural-layout MXFP4 weights.
    Returns BF16 [T, H]: the routing-weighted sum over local experts (slots routed elsewhere contribute nothing).
    ``cfg`` overrides ``pick_config`` (keys BLOCK_M, BLOCK_K, NUM_BUFFERS, HALF_TAIL, stage2)."""
    _check_inputs(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale)
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    _check_post_norm("moe_mxfp4_a16w4", post_norm_weight, H, x.device)
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return out
    cfg = cfg or pick_config(T)
    cfg2 = dict(cfg, **cfg.get("stage2", {}))
    for dim, size, block in (("K", K, cfg["BLOCK_K"]), ("I", I, cfg2["BLOCK_K"]), ("I", I, _BLOCK_N), ("H", H,
                                                                                                       _BLOCK_N)):
        if size % block:
            raise ValueError(f"moe_mxfp4_a16w4: {dim}={size} is not a multiple of the tile ({block})")
    for dim, size in (("K", K), ("I", I)):
        if size % 512:
            raise ValueError(f"moe_mxfp4_a16w4: {dim}={size} is not a multiple of 512 (the TDM-loaded scale rows, "
                             f"{dim} / 32 bytes, must be whole 16-byte units)")
    for name, w in (("w13", w13), ("w13_scale", w13_scale), ("w2", w2), ("w2_scale", w2_scale)):
        if w.stride(1) != w.shape[2]:
            raise ValueError(f"moe_mxfp4_a16w4: {name} rows must be contiguous (stride(1) == {w.shape[2]})")
    for name, w in (("w13", w13), ("w2", w2)):
        if not _word_strided(w):
            raise ValueError(f"moe_mxfp4_a16w4: {name} is read as int32 words; its storage offset and expert stride "
                             f"must be multiples of 4 bytes")

    ids = topk_ids.contiguous()
    pos = torch.empty(num_valid, dtype=torch.int32, device=dev)
    split_sizes = torch.zeros(E, dtype=torch.int32, device=dev)
    block = 1024
    _route_count_kernel[(triton.cdiv(num_valid, block), )](ids, pos, split_sizes, num_valid, expert_start, E=E,
                                                           BLOCK=block, num_warps=4)
    tmap = _tile_map(split_sizes, num_valid, cfg["BLOCK_M"])
    order = torch.empty(num_valid, dtype=torch.int64, device=dev)
    xs = _alloc((num_valid, K), torch.bfloat16, dev)
    _place_rows(x, xs, order, ids, pos, tmap[4], expert_start)
    inter = _alloc((num_valid, I), torch.bfloat16, dev)
    _launch(xs, w13, w13_scale, inter, None, tmap, I, K, True, cfg)
    # xs is dead after stage 1.
    y = xs.view(-1)[:num_valid * H].view(num_valid, H) if H <= K else _alloc((num_valid, H), torch.bfloat16, dev)
    tmap2 = tmap if cfg2["BLOCK_M"] == cfg["BLOCK_M"] else _tile_map(split_sizes, num_valid, cfg2["BLOCK_M"])
    _launch(inter, w2, w2_scale, y, order, tmap2, H, I, False, cfg2)
    _finalize(y, topk_weights, ids, out, expert_start, E, post_norm_weight, post_norm_eps, post_norm_gain)
    return out
