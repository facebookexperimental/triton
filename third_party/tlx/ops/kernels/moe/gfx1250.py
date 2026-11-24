"""BF16 routed MoE on gfx1250: BF16 activations and weights, TDM-pipelined grouped expert GEMMs.

Same contract and numerics as ``gfx950.moe``: the stage-1 gate/up GEMM and the stage-2 down GEMM accumulate in FP32,
only the stage-1 output (SiLU(gate) * up) is rounded to BF16 between the GEMMs, and the finalize (optional post-norm
and routing-weighted sum) rounds at the same points as the gfx950 finalize.

Routing, the tile map, the row placement and the finalize are the A16W4 op's (``moe_mxfp.gfx1250``), so the op never
reads device values on the host. As there, the row placement copies every slot's token row into a ``[T * topk, K]`` BF16
scratch buffer (512 MiB at T = 8192, K = 4096, top-8), because only the device knows how many slots are local; the
stage-2 output reuses it when H <= K. Each program streams one (expert, M tile, N tile) through a ring of ``NUM_BUFFERS``
``BLOCK_K``-wide stages, each one fused TDM op (waves 0-1 load the A tile, waves 2-3 the weight tiles). The GEMM
schedule follows ``grouped_gemm.gfx1250``: a stage is read as 32-wide K subtiles with the next subtile's LDS load in
flight during the current subtile's WMMAs, and one TDM wait + workgroup barrier per stage both publishes the next stage
and frees the one just read for refill. Weight tiles are loaded as ``[BLOCK_N, BLOCK_K]`` from the checkpoint's
K-contiguous layout and transposed in LDS for the WMMA B operand, so the weights are used in place. Descriptor shapes
start at the program's tile, so TDM bounds checks zero-fill rows past the expert's last row; an expert's last tile with
at most a quarter or half of ``BLOCK_M`` rows runs a shorter tile (``QUARTER_TAIL`` / ``HALF_TAIL``) from a ring that
aliases the full one. The GEMMs launch with ``enable_wmma_queuing`` (WMMAs queue behind independent scalar and memory
instructions).

Weights: ``w13`` ``[E, 2I, K]`` (gate rows first) and ``w2`` ``[E, H, I]``, both BF16 with contiguous rows. ``K``, ``I``
and ``H`` are multiples of 128; tiles the tuned ones do not divide shrink (``_fit``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.tlx.ops.kernels.moe_mxfp.gfx1250 import (
    _check_post_norm,
    _finalize,
    _place_rows,
    _route_count_kernel,
    _tile_map,
)

_NUM_WARPS = 4
_SUBTILE_K = 32


def _alloc(shape, dtype, device):
    """Workspace allocation (tests replace it to place guard regions around the op's buffers)."""
    return torch.empty(shape, dtype=dtype, device=device)


@triton.jit
def _ld_sub(a_buf, b_buf, b2_buf, slot, j: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, SUB: tl.constexpr,
            FUSE_SILU_MUL: tl.constexpr):
    """Subtile ``j`` (K columns [j * SUB, (j + 1) * SUB)) of ring slot ``slot`` as WMMA operands."""
    a = tlx.local_load(tlx.local_slice(tlx.local_view(a_buf, slot), [0, j * SUB], [BM, SUB]))
    b = tlx.local_load(tlx.local_trans(tlx.local_slice(tlx.local_view(b_buf, slot), [0, j * SUB], [BLOCK_N, SUB])))
    b2 = b
    if FUSE_SILU_MUL:
        b2 = tlx.local_load(tlx.local_trans(tlx.local_slice(tlx.local_view(b2_buf, slot), [0, j * SUB],
                                                            [BLOCK_N, SUB])))
    return a, b, b2


@triton.jit
def _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, k_iter, pred, BLOCK_K: tl.constexpr,
                 NUM_BUFFERS: tl.constexpr, FUSE_SILU_MUL: tl.constexpr):
    """One fused TDM op for K block ``k_iter``: waves 0-1 load the A tile, waves 2-3 the weight tile(s)."""
    slot = k_iter % NUM_BUFFERS
    a_d = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, k_iter * BLOCK_K], pred=pred, clamp_bounds=True)
    b_d = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, k_iter * BLOCK_K], pred=pred, clamp_bounds=True)
    if FUSE_SILU_MUL:
        b2_d = tlx.update_tensor_descriptor(b2_desc, add_offsets=[0, k_iter * BLOCK_K], pred=pred, clamp_bounds=True)
        tlx.async_amd_descriptor_load_fused([(a_d, tlx.local_view(a_buf, slot), 3), (b_d, tlx.local_view(b_buf,
                                                                                                         slot), 4),
                                             (b2_d, tlx.local_view(b2_buf, slot), 8)])
    else:
        tlx.async_amd_descriptor_load_fused([(a_d, tlx.local_view(a_buf, slot), 3),
                                             (b_d, tlx.local_view(b_buf, slot), 12)])


@triton.jit
def _k_block(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, k, a0, b0, c0, acc, acc2, K_ITERS: tl.constexpr,
             BM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr,
             FUSE_SILU_MUL: tl.constexpr, SUB: tl.constexpr):
    """K block ``k``, whose first subtile is already in registers; returns block k + 1's first subtile."""
    slot = k % NUM_BUFFERS
    a, b, c = a0, b0, c0
    for j in tl.static_range(1, BLOCK_K // SUB):
        an, bn, cn = _ld_sub(a_buf, b_buf, b2_buf, slot, j, BM, BLOCK_N, SUB, FUSE_SILU_MUL)
        acc = tl.dot(a, b, acc)
        if FUSE_SILU_MUL:
            acc2 = tl.dot(a, c, acc2)
        # Bound the live operand tiles; the next subtile's LDS load stays in flight during these dots.
        tlx.amd_sched_barrier()
        a, b, c = an, bn, cn
    # Every wave has read slot k (all its subtiles are in registers) and block k + 1 is the oldest TDM op in
    # flight: one wait + barrier publishes block k + 1 and frees slot k for block k + NUM_BUFFERS.
    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 2)
    tlx.workgroup_barrier()
    nk = k + NUM_BUFFERS
    _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, nk, nk < K_ITERS, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL)
    a0, b0, c0 = _ld_sub(a_buf, b_buf, b2_buf, (k + 1) % NUM_BUFFERS, 0, BM, BLOCK_N, SUB, FUSE_SILU_MUL)
    acc = tl.dot(a, b, acc)
    if FUSE_SILU_MUL:
        acc2 = tl.dot(a, c, acc2)
    tlx.amd_sched_barrier()
    return a0, b0, c0, acc, acc2


@triton.jit
def _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, b2_desc, a_buf, b_buf, b2_buf, K: tl.constexpr,
          BM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr,
          FUSE_SILU_MUL: tl.constexpr, STORE_ROWS: tl.constexpr, SUB: tl.constexpr):
    """One [BM, BLOCK_N] output tile; ``a_buf`` is a ring of [BM, BLOCK_K] A tiles."""
    K_ITERS: tl.constexpr = K // BLOCK_K
    a_desc = tl.make_tensor_descriptor(a_ptr + row0 * K, shape=[rows_left, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BM, BLOCK_K])
    for s in tl.static_range(NUM_BUFFERS):
        _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, s, True, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL)
    acc = tl.zeros((BM, BLOCK_N), tl.float32)
    acc2 = tl.zeros((BM, BLOCK_N), tl.float32)
    tlx.async_amd_descriptor_wait(NUM_BUFFERS - 1)
    tlx.workgroup_barrier()
    a0, b0, c0 = _ld_sub(a_buf, b_buf, b2_buf, 0, 0, BM, BLOCK_N, SUB, FUSE_SILU_MUL)
    # Two K blocks per iteration, so the loop-carried first subtile alternates between two register sets.
    for k in range(0, K_ITERS, 2):
        a1, b1, c1, acc, acc2 = _k_block(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, k, a0, b0, c0, acc, acc2,
                                         K_ITERS, BM, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, SUB)
        a0, b0, c0, acc, acc2 = _k_block(a_desc, b_desc, b2_desc, a_buf, b_buf, b2_buf, k + 1, a1, b1, c1, acc, acc2,
                                         K_ITERS, BM, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, SUB)
    tlx.async_amd_descriptor_wait(0)
    if FUSE_SILU_MUL:
        # SiLU via exp2 and the fast divide; a literal log2(e), since a module-level constant bypasses Triton's C
        # launch fast path.
        out = tl.math.fdiv(acc, 1.0 + tl.exp2(acc * -1.4426950408889634), approx=True) * acc2
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
def _bf16_grouped_kernel(
    a_ptr,  # bf16 [rows, K], rows sorted by expert
    b_ptr,  # bf16 [E, NB, K]
    c_ptr,  # bf16 [rows, N] (STORE_ROWS: [slots, N], row r stored at slot store_rows[r])
    store_rows,
    tile_meta0,
    tile_meta1,
    tile_count,
    N,  # output columns; stage 1 (FUSE_SILU_MUL): I, with the up rows starting at row N of B
    stride_be,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    FUSE_SILU_MUL: tl.constexpr,
    STORE_ROWS: tl.constexpr,
    HALF_TAIL: tl.constexpr,
    QUARTER_TAIL: tl.constexpr,
    SUB: tl.constexpr,
):
    tl.static_assert(BLOCK_K % SUB == 0 and BLOCK_N % 64 == 0)
    tl.static_assert(K % (2 * BLOCK_K) == 0 and K // BLOCK_K >= NUM_BUFFERS and NUM_BUFFERS >= 2)
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
    b_desc = tl.make_tensor_descriptor(b_base + off_n * K, shape=[N - off_n, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_N, BLOCK_K])
    if FUSE_SILU_MUL:
        b2_desc = tl.make_tensor_descriptor(b_base + (N + off_n) * K, shape=[N - off_n, K],
                                            strides=[K, tl.constexpr(1)], block_shape=[BLOCK_N, BLOCK_K])
    else:
        b2_desc = b_desc

    # One 8-element pad per BLOCK_K row keeps the WMMA operand reads free of bank conflicts.
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_K],
                                                                                 [1, 0])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]], [BLOCK_N, BLOCK_K],
                                                                                 [1, 0])
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=a_layout)
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=b_layout)
    if FUSE_SILU_MUL:
        b2_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=b_layout)
    else:
        b2_buf = b_buf
    # Short tiles get their own ring of [BM, BLOCK_K] A tiles aliasing the full one (a K subtile of a row slice of
    # the full ring is not expressible).
    a2_buf = a_buf
    a4_buf = a_buf
    if HALF_TAIL and BLOCK_M >= 32:
        a2_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]],
                                                                                      [BLOCK_M // 2, BLOCK_K], [1, 0])
        a2_buf = tlx.local_alloc((BLOCK_M // 2, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=a2_layout, reuse=a_buf)
    if QUARTER_TAIL and BLOCK_M >= 64:
        a4_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[BLOCK_K, 8]],
                                                                                      [BLOCK_M // 4, BLOCK_K], [1, 0])
        a4_buf = tlx.local_alloc((BLOCK_M // 4, BLOCK_K), tl.bfloat16, NUM_BUFFERS, layout=a4_layout, reuse=a_buf)
    if QUARTER_TAIL and BLOCK_M >= 64 and rows_left <= BLOCK_M // 4:
        _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, b2_desc, a4_buf, b_buf, b2_buf, K,
              BLOCK_M // 4, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, STORE_ROWS, SUB)
    elif HALF_TAIL and BLOCK_M >= 32 and rows_left <= BLOCK_M // 2:
        _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, b2_desc, a2_buf, b_buf, b2_buf, K,
              BLOCK_M // 2, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, STORE_ROWS, SUB)
    else:
        _tile(a_ptr, c_ptr, store_rows, row0, rows_left, off_n, N, b_desc, b2_desc, a_buf, b_buf, b2_buf, K, BLOCK_M,
              BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL, STORE_ROWS, SUB)


def pick_config(T):
    """Tile choice by token count, tuned on gfx1250 (96 CUs) at 64 local experts of 256, top-8, K = I = H = 4096,
    where an expert receives ~T / 32 rows. ``stage2`` keys override the stage-2 (down GEMM) tiles; a
    ``waves_per_eu`` of 0 leaves the compiler default."""
    if T <= 64:
        return dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=64, NUM_BUFFERS=4)
    if T <= 512:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=64, NUM_BUFFERS=3)
    if T <= 1536:
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=True)
    if T <= 6144:
        return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=True, QUARTER_TAIL=True,
                    waves_per_eu=1, stage2=dict(BLOCK_M=128, BLOCK_N=256, BLOCK_K=128, NUM_BUFFERS=3, waves_per_eu=0))
    return dict(BLOCK_M=256, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=2, HALF_TAIL=True, QUARTER_TAIL=True, waves_per_eu=1,
                stage2=dict(BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, NUM_BUFFERS=2))


def _fit(cfg, k, n):
    """``cfg`` with tiles that divide a reduction dim ``k`` and an output dim ``n`` the tuned tiles do not: ``BLOCK_K``
    halves down to 64 (the K loop runs pairs of stages, and a two-stage loop allocates an extra stage of LDS, which
    overflows at BLOCK_M = 256), 256-wide ``BLOCK_N`` falls back to 128, and the ring is no deeper than the K loop.
    Multiples of 128 always fit."""
    bk, bn = cfg["BLOCK_K"], cfg["BLOCK_N"]
    while bk > 64 and (k % (2 * bk) or k // bk <= 2):
        bk //= 2
    while bn > 128 and n % bn:
        bn //= 2
    return dict(cfg, BLOCK_K=bk, BLOCK_N=bn, NUM_BUFFERS=max(1, min(cfg["NUM_BUFFERS"], k // bk)))


def _launch(a, w, out, store_rows, tmap, n_out, k, fuse_silu_mul, cfg):
    meta0, meta1, count, max_tiles = tmap[:4]
    grid = (max_tiles * (n_out // cfg["BLOCK_N"]), )
    extra = {"waves_per_eu": cfg["waves_per_eu"]} if cfg.get("waves_per_eu", 0) else {}
    _bf16_grouped_kernel[grid](a, w, out, store_rows if store_rows is not None else count, meta0, meta1, count, n_out,
                               w.stride(0), K=k, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
                               NUM_BUFFERS=cfg["NUM_BUFFERS"], FUSE_SILU_MUL=fuse_silu_mul, STORE_ROWS=store_rows
                               is not None, HALF_TAIL=cfg.get("HALF_TAIL",
                                                              False), QUARTER_TAIL=cfg.get("QUARTER_TAIL", False),
                               SUB=_SUBTILE_K, num_warps=_NUM_WARPS, enable_wmma_queuing=True, **extra)


def _check_inputs(x, topk_weights, topk_ids, w13, w2):
    if x.dim() != 2 or w13.dim() != 3 or w2.dim() != 3:
        raise ValueError("moe: x must be [T, K], w13 [E, 2I, K] and w2 [E, H, I]")
    if x.dtype != torch.bfloat16 or w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("moe: x, w13 and w2 must be bfloat16")
    E, two_i, K = w13.shape
    if x.shape[1] != K or two_i % 2 or w2.shape[0] != E or w2.shape[2] != two_i // 2:
        raise ValueError(f"moe: shape mismatch x {tuple(x.shape)} w13 {tuple(w13.shape)} w2 {tuple(w2.shape)}")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
        raise ValueError("moe: topk_ids and topk_weights must both be [T, top_k]")
    for name, w in (("w13", w13), ("w2", w2)):
        if w.stride(2) != 1 or w.stride(1) != w.shape[2]:
            raise ValueError(f"moe: {name} rows must be contiguous")
        if w.data_ptr() % 16 or w.stride(0) % 8:
            raise ValueError(f"moe: {name} is TDM-loaded; its base and expert stride must be multiples of 16 bytes")


def moe(x, topk_weights, topk_ids, w13, w2, expert_start=0, post_norm_weight=None, post_norm_eps=1e-6,
        post_norm_gain=0.0, out=None, cfg=None):
    """BF16 x [T, K] through the local experts [expert_start, expert_start + E) with BF16 w13 [E, 2I, K] and w2
    [E, H, I]. Returns BF16 [T, H]: the routing-weighted sum of the (optionally post-normed) local expert outputs;
    slots routed elsewhere contribute nothing. ``cfg`` overrides ``pick_config``."""
    _check_inputs(x, topk_weights, topk_ids, w13, w2)
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    _check_post_norm("moe", post_norm_weight, H, x.device)
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    if T == 0:
        return out
    cfg = cfg or pick_config(T)
    cfg2 = _fit(dict(cfg, **cfg.get("stage2", {})), I, H)
    cfg = _fit(cfg, K, I)
    for dim, size, block in (("K", K, 2 * cfg["BLOCK_K"]), ("I", I, 2 * cfg2["BLOCK_K"]), ("I", I, cfg["BLOCK_N"]),
                             ("H", H, cfg2["BLOCK_N"])):
        if size % block:
            raise ValueError(f"moe: {dim}={size} is not a multiple of {block}")
    for dim, size, c in (("K", K, cfg), ("I", I, cfg2)):
        if size // c["BLOCK_K"] < c["NUM_BUFFERS"]:
            raise ValueError(f"moe: {dim}={size} is shorter than the TDM ring ({c['NUM_BUFFERS']} x {c['BLOCK_K']})")

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
    _launch(xs, w13, inter, None, tmap, I, K, True, cfg)
    # xs is dead after stage 1.
    y = xs.view(-1)[:num_valid * H].view(num_valid, H) if H <= K else _alloc((num_valid, H), torch.bfloat16, dev)
    tmap2 = tmap if cfg2["BLOCK_M"] == cfg["BLOCK_M"] else _tile_map(split_sizes, num_valid, cfg2["BLOCK_M"])
    _launch(inter, w2, y, order, tmap2, H, I, False, cfg2)
    _finalize(y, topk_weights, ids, out, expert_start, E, post_norm_weight, post_norm_eps, post_norm_gain)
    return out
