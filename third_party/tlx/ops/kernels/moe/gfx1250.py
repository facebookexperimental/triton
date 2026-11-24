"""Scalar-scaled FP8 routed MoE on gfx1250, using a fixed two-buffer TDM/WMMA pipeline.

The GEMM schedule and SiLU/FP8 operation boundaries follow the validated
``third_party/amd/python/examples/gluon/moe_cdna5.py`` baseline schedule.
This implementation owns GPU routing and scratch allocations and has no dependency
on examples, experimental runners, or triton_kernels. Model weights are consumed
on every invocation, so graph replay observes updated weight contents as well as
updated activations, route IDs and router weights.
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna5 import tdm
from triton.experimental.gluon.language.extra import libdevice

from ..._catalog import InvalidInput


@triton.jit
def _route(Ids, Gather, Inverse, Tiles, R: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr, START: tl.constexpr,
           PAD: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr, HIST: tl.constexpr, TILE_BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    # Compare in int64 before narrowing: sentinel / remote IDs must not wrap into a local expert.
    ids = tl.load(Ids + slots, slots < R, other=-1).to(tl.int64) - START
    local = (slots < R) & (ids >= 0) & (ids < E)
    hist = tl.histogram(tl.where(local, ids, E).to(tl.int32), HIST)
    experts = tl.arange(0, HIST)
    start = tl.sum(tl.where(experts < expert, hist, 0))
    total = tl.sum(tl.where(experts < E, hist, 0))
    mask = local & (ids == expert)
    count = tl.sum(mask.to(tl.int32))
    within = tl.cumsum(mask.to(tl.int32)) - 1
    tl.store(Gather + start + within, slots // TOPK, mask)
    tl.store(Inverse + slots, start + within, mask)
    if expert == 0:
        tl.store(Tiles + CAP * 4, total)
        tl.store(Inverse + slots, -1, (slots < R) & ~local)
        rows = tl.arange(0, triton.next_power_of_2(PAD))
        tl.store(Gather + rows, 0, (rows >= total) & (rows < PAD))
    blocks = tl.cdiv(hist, 64)
    tile_start = tl.sum(tl.where(experts < expert, blocks, 0))
    tile_total = tl.sum(tl.where(experts < E, blocks, 0))
    tiles = tl.arange(0, TILE_BLOCK)
    valid = tiles < tl.cdiv(count, 64)
    base = Tiles + (tile_start + tiles) * 4
    tl.store(base, expert, valid)
    tl.store(base + 1, start, valid)
    tl.store(base + 2, count, valid)
    tl.store(base + 3, tiles * 64, valid)
    if expert == 0:
        tail = (tiles >= tile_total) & (tiles < CAP)
        tl.store(Tiles + tiles * 4, -1, tail)


@triton.jit
def _route_counts(Ids, Counts, R: tl.constexpr, E: tl.constexpr, START: tl.constexpr, BLOCK: tl.constexpr,
                  HIST: tl.constexpr):
    chunk = tl.program_id(0)
    slots = chunk * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Ids + slots, slots < R, other=-1).to(tl.int64) - START
    local = (slots < R) & (ids >= 0) & (ids < E)
    hist = tl.histogram(tl.where(local, ids, E).to(tl.int32), HIST)
    experts = tl.arange(0, HIST)
    tl.store(Counts + chunk * HIST + experts, hist)


@triton.jit
def _route_scatter(Ids, Counts, Gather, Inverse, Tiles, R: tl.constexpr, TOPK: tl.constexpr, E: tl.constexpr,
                   START: tl.constexpr, PAD: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr, HIST: tl.constexpr,
                   CHUNKS: tl.constexpr, CHUNK_BLOCK: tl.constexpr, TILE_BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    expert = tl.program_id(1)
    chunks = tl.arange(0, CHUNK_BLOCK)
    experts = tl.arange(0, HIST)
    counts = tl.load(Counts + chunks[:, None] * HIST + experts[None, :], chunks[:, None] < CHUNKS, other=0)
    hist = tl.sum(counts, axis=0)
    start = tl.sum(tl.where(experts < expert, hist, 0))
    total = tl.sum(tl.where(experts < E, hist, 0))
    count = tl.sum(tl.where(experts == expert, hist, 0))
    chunk_prefix = tl.sum(tl.sum(tl.where((chunks[:, None] < chunk) & (experts[None, :] == expert), counts, 0), axis=0),
                          axis=0)
    slots = chunk * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Ids + slots, slots < R, other=-1).to(tl.int64) - START
    local = (slots < R) & (ids >= 0) & (ids < E)
    mask = local & (ids == expert)
    within = tl.cumsum(mask.to(tl.int32)) - 1
    row = start + chunk_prefix + within
    tl.store(Gather + row, slots // TOPK, mask)
    tl.store(Inverse + slots, row, mask)
    if expert == 0:
        tl.store(Inverse + slots, -1, (slots < R) & ~local)
        # Cover padding in parallel, including the extra gather tile.
        rows = chunk * BLOCK + tl.arange(0, BLOCK)
        tl.store(Gather + rows, 0, (rows >= total) & (rows < PAD))
        if chunk == CHUNKS - 1:
            tail_rows = CHUNKS * BLOCK + tl.arange(0, BLOCK)
            tl.store(Gather + tail_rows, 0, (tail_rows >= total) & (tail_rows < PAD))
            tl.store(Tiles + CAP * 4, total)
    if chunk == 0:
        blocks = tl.cdiv(hist, 64)
        tile_start = tl.sum(tl.where(experts < expert, blocks, 0))
        tile_total = tl.sum(tl.where(experts < E, blocks, 0))
        tiles = tl.arange(0, TILE_BLOCK)
        valid = tiles < tl.cdiv(count, 64)
        base = Tiles + (tile_start + tiles) * 4
        tl.store(base, expert, valid)
        tl.store(base + 1, start, valid)
        tl.store(base + 2, count, valid)
        tl.store(base + 3, tiles * 64, valid)
        if expert == 0:
            tl.store(Tiles + tiles * 4, -1, (tiles >= tile_total) & (tiles < CAP))


def _chunked_routes(ids, gather, inverse, tiles, experts, expert_start, *, chunk_size=128):
    """Stable expert grouping: chunk histograms, then prefix-based scatter without atomics."""
    routes = ids.numel()
    hist = triton.next_power_of_2(experts + 1)
    chunks = triton.cdiv(routes, chunk_size)
    counts = torch.empty((chunks, hist), device=ids.device, dtype=torch.int32)
    _route_counts[(chunks, )](ids, counts, routes, experts, expert_start, chunk_size, hist, num_warps=4)
    cap = tiles.shape[0] - 1
    _route_scatter[(chunks, experts)](ids, counts, gather, inverse, tiles, routes, ids.shape[1], experts, expert_start,
                                      gather.numel(), cap, chunk_size, hist, chunks, triton.next_power_of_2(chunks),
                                      triton.next_power_of_2(cap), num_warps=4)


@gluon.jit
def _loads(XDesc, WDesc, XBuf, WBuf, GatherRows, Index, PRED, GATHER: gl.constexpr):
    if GATHER:
        desc = tdm.update_tensor_descriptor(XDesc, add_offsets=[0, Index * 256], pred=PRED, clamp_bounds=True)
        tdm.async_gather(desc, GatherRows, XBuf.index(Index % 2))
    else:
        tdm.async_load(XDesc, [0, Index * 256], XBuf.index(Index % 2), pred=PRED)
    tdm.async_load(WDesc, [0, Index * 256], WBuf.index(Index % 2), pred=PRED)


@gluon.jit
def _project(X, W, Gather, Tiles, Y, M: gl.constexpr, K: gl.constexpr, N: gl.constexpr, PAD: gl.constexpr,
             S: gl.constexpr, SH: gl.constexpr, GATE_UP: gl.constexpr):
    BM: gl.constexpr = 64
    BN: gl.constexpr = 128
    BK: gl.constexpr = 256
    tile = gl.program_id(0)
    col_tile = gl.program_id(1)
    expert = gl.load(Tiles + tile * 4)
    if expert == -1:
        return
    start = gl.load(Tiles + tile * 4 + 1)
    count = gl.load(Tiles + tile * 4 + 2)
    first = gl.load(Tiles + tile * 4 + 3)
    mma: gl.constexpr = gl.amd.AMDWMMALayout(3, True, [[0, 1], [1, 0]], [], [16, 16, 128])
    x_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BM, BK], [1, 0])
    w_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[256, 16]], [BN, BK], [1, 0])
    x_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 16)
    w_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 16)
    if GATE_UP:
        idx_layout: gl.constexpr = gl.SliceLayout(1, gl.BlockedLayout([BM, 1], [1, 32], [1, 4], [1, 0]))
        rows = gl.load(Gather + start + first + gl.arange(0, BM, idx_layout))
        xbase = X
    else:
        rows = gl.constexpr(0)
        xbase = X + (start + first) * K
    xd = tdm.make_tensor_descriptor(xbase, (M, K), (K, 1), (BM, BK), x_shared)
    wd = tdm.make_tensor_descriptor(W + expert * N * K + col_tile * BN * K, (N, K), (K, 1), (BN, BK), w_shared)
    xb = gl.allocate_shared_memory(X.dtype.element_ty, [2, BM, BK], x_shared)
    wb = gl.allocate_shared_memory(W.dtype.element_ty, [2, BN, BK], w_shared)
    _loads(xd, wd, xb, wb, rows, 0, True, GATE_UP)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in range(K // BK):
        _loads(xd, wd, xb, wb, rows, i + 1, i + 1 < K // BK, GATE_UP)
        tdm.async_wait(2)
        a = xb.index(i % 2).load(x_layout)
        b = wb.index(i % 2).permute([1, 0]).load(w_layout)
        zero = 0
        zero = zero.to(gl.uint8)
        acc = gl.amd.cdna5.wmma_scaled(a, zero, "e4m3", b, zero, "e4m3", acc)
    acc = acc + 0.0
    if GATE_UP:
        gate, up = gl.split(gl.reshape(acc, (BM, BN // 2, 2)))
        gate = gate * S
        up = up * S
        silu = gl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        value = gl.div_rn(silu * up, SH)
        WIDTH: gl.constexpr = BN // 2
        # Same store layout as the example's get_blocked_layout([64,64], FP8).
        store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [2, 2], [1, 0])
    else:
        value = acc * S
        WIDTH: gl.constexpr = BN
        store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [2, 2], [1, 0])
    value = gl.convert_layout(value.to(Y.dtype.element_ty), store_layout)
    rr = first + gl.arange(0, BM, gl.SliceLayout(1, store_layout))
    cc = col_tile * WIDTH + gl.arange(0, WIDTH, gl.SliceLayout(0, store_layout))
    out_n: gl.constexpr = N // 2 if GATE_UP else N
    offsets = (start + rr[:, None]) * out_n + cc[None, :]
    gl.amd.cdna5.buffer_store(value, Y, offsets, mask=(rr[:, None] < count) & (cc[None, :] < out_n))
    if GATE_UP:
        # The first actual tile owns the tail; no other tile writes these rows.
        if tile == 0:
            # Store the last valid row count, supplied in a reserved metadata word.
            actual = gl.load(Tiles + gl.num_programs(0) * 4)
            for begin in range(actual, PAD, BM):
                pr = begin + gl.arange(0, BM, gl.SliceLayout(1, store_layout))
                gl.amd.cdna5.buffer_store(gl.full((BM, WIDTH), 0, Y.dtype.element_ty,
                                                  store_layout), Y, pr[:, None] * out_n + cc[None, :],
                                          mask=(pr[:, None] < PAD) & (cc[None, :] < out_n))


@triton.jit
def _combine(Y, Inverse, Weights, Out, H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK, ), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        row = tl.load(Inverse + token * TOPK + slot)
        weight = tl.load(Weights + token * TOPK + slot)
        expert = tl.load(Y + tl.maximum(row, 0) * H + cols, (row >= 0) & (cols < H), other=0).to(tl.float32)
        value = value + expert * tl.where(row >= 0, weight, 0)
    tl.store(Out + token * H + cols, value.to(Out.dtype.element_ty), cols < H)


def moe(x, weights, ids, w13, w2, *, expert_start, fp8_scales, out):
    m, d = x.shape
    e, two_i, _ = w13.shape
    i, h, topk = two_i // 2, w2.shape[1], ids.shape[1]
    if not (1 <= e <= 16 and topk in (2, 4) and m * topk <= 32768):
        raise InvalidInput("tlx.ops.moe gfx1250 FP8 requires E<=16, top-k 2/4 and at most 32768 route slots")
    if expert_start > (1 << 62):
        raise InvalidInput("tlx.ops.moe gfx1250 expert_start must be <=2**62")
    if any(not t.is_contiguous() for t in (x, weights, ids, w13, w2)) or weights.dtype != torch.float32:
        raise InvalidInput("tlx.ops.moe gfx1250 FP8 requires contiguous inputs and FP32 router weights")
    if any(t.numel() and t.data_ptr() % 16 for t in (x, w13, w2)):
        raise InvalidInput("tlx.ops.moe gfx1250 FP8 payloads require 16-byte aligned bases")
    if out is None:
        out = torch.empty((m, h), device=x.device, dtype=torch.bfloat16)
    if m == 0:
        return out
    routes = m * topk
    pad = triton.cdiv(routes, 64) * 64 + 64
    cap = triton.cdiv(routes, 64) + e
    gather = torch.empty(pad, device=x.device, dtype=torch.int32)
    inverse = torch.empty(routes, device=x.device, dtype=torch.int32)
    tiles = torch.empty((cap + 1, 4), device=x.device, dtype=torch.int32)
    if routes >= 8192:
        # The extra launch pays off from 8192 route slots. Smaller chunks bound
        # prefix-scan state; larger expert/count tables favor 256-slot chunks.
        _chunked_routes(ids, gather, inverse, tiles, e, expert_start, chunk_size=256 if
                        (routes > 16384 or e > 8 or (e == 8 and routes >= 16384)) else 128)
    else:
        _route[(e, )](ids, gather, inverse, tiles, routes, topk, e, expert_start, pad, cap,
                      triton.next_power_of_2(routes), triton.next_power_of_2(e + 1), triton.next_power_of_2(cap),
                      num_warps=4)
    # Gate/up pairs share an epilogue lane; byte permutation preserves FP8 payloads.
    interleaved = (w13.view(torch.uint8).reshape(e, 2, i,
                                                 d).transpose(1, 2).contiguous().reshape(e, two_i,
                                                                                         d).view(torch.float8_e4m3fn))
    hidden = torch.empty((pad, i), device=x.device, dtype=x.dtype)
    down = torch.empty((pad, h), device=x.device, dtype=torch.bfloat16)
    sx, sw13, sh, sw2 = fp8_scales
    _project[(cap, triton.cdiv(two_i, 128))](x, interleaved, gather, tiles, hidden, m, d, two_i, pad, sx * sw13, sh,
                                             True, num_warps=4, enable_fp_fusion=False)
    _project[(cap, triton.cdiv(h, 128))](hidden, w2, None, tiles, down, pad, i, h, pad, sh * sw2, sh, False,
                                         num_warps=4, enable_fp_fusion=False)
    _combine[(m, triton.cdiv(h, 256))](down, inverse, weights, out, h, topk, 256, num_warps=4, enable_fp_fusion=False)
    return out
