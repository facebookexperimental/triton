"""MXFP4 W4A4 routed MoE on gfx1250: MXFP4 activations x MXFP4 weights on the native scaled FP4 WMMA.

Activations are quantized to MXFP4 (E2M1 + one E8M0 scale per 32 along K, AITER / Quark "even" rounding: scale
2^(floor(log2(amax rounded to 2 mantissa bits)) - 2), E2M1 round to nearest even, saturating at 6) before each GEMM, and
both GEMMs run ``v_wmma_scale_f32_32x16x128_f4`` through ``tlx.dot_scaled(e2m1, e2m1)``. One call runs: the routing of
``gfx1250.moe_mxfp4_a16w4`` (atomic per-expert ranks and a device tile map, no host readback), a placement kernel that
quantizes every token row once and writes it to the expert-sorted row of each of its local slots, the stage-1 gate/up
GEMM, the stage-2 down GEMM, and the A16W4 op's finalize (optional post-norm and routing-weighted sum).

Stage 1 rounds gate and up to BF16 (the A16W4 op's rounding point), stages them through LDS into a layout in which every
lane holds whole 32-value groups, and requantizes SiLU(gate) * up to MXFP4 in registers; the E2M1 packing uses
``v_cvt_scalef32_pk8_fp4_f32``. Stage 2 stores each BF16 row at its routing slot.

Each GEMM program owns one (expert, M tile, N tile). Operands stream global -> LDS through a ring of ``NUM_BUFFERS``
fused TDM stages (waves 0-1 load the A tile, waves 2-3 the weight tile; stage 1 loads the gate and up rows into one
stage), the A and weight scale panels are loaded once per tile, and one workgroup barrier per K step makes a stage
visible. Descriptor shapes start at the program's tile, so TDM bounds checks zero-fill rows past the expert's last row.
With ``HALF_TAIL`` an expert's last tile with at most BLOCK_M / 2 rows runs a half-height tile from the same ring.

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

from triton.tlx.ops.kernels.moe_mxfp import gfx1250 as a16w4
from triton.tlx.ops.kernels.moe_mxfp.gfx950_a16w4 import _check_inputs

_LOG2E = tl.constexpr(1.4426950408889634)
_OP = "moe_mxfp4_w4a4"


def _alloc(shape, dtype, device):
    """Workspace allocation (tests replace it to place guard regions around the op's buffers)."""
    return torch.empty(shape, dtype=dtype, device=device)


@constexpr_function
def _group_layout(R, C, W):
    """[R, C] layout in which every lane holds 32 consecutive columns (one MXFP4 group) in registers, so the group
    amax and the nibble packing need no data exchange."""
    cols = [(0, 1 << i) for i in range(5)]
    spread = [(0, 32 << i) for i in range((C // 32).bit_length() - 1)] + [(1 << i, 0) for i in range(R.bit_length() - 1)]
    nw = W.bit_length() - 1
    lane, warp, rest = spread[:5], spread[5:5 + nw], spread[5 + nw:]
    lane += [(0, 0)] * (5 - len(lane))
    warp += [(0, 0)] * (nw - len(warp))
    return a16w4._linear((R, C), lane, warp, cols + rest)


@constexpr_function
def _pk8_asm(base):
    movs = "".join(f"v_mov_b32 v{base + i}, ${i + 1}\n" for i in range(8))
    return movs + f"v_cvt_scalef32_pk8_fp4_f32 $0, v[{base}:{base + 7}], $9\ns_wait_alu depctr_va_vdst(0)"


@constexpr_function
def _pk8_constraints(base):
    return "=&v," + ",".join(["v"] * 9) + "," + ",".join(f"~{{v{base + i}}}" for i in range(8))


@triton.jit
def _pk8(v0, v1, v2, v3, v4, v5, v6, v7, BASE: tl.constexpr):
    """Eight scaled fp32 values -> one int32 of E2M1 nibbles (v0 lowest): ``v_cvt_scalef32_pk8_fp4_f32`` (round to
    nearest even, saturating) with scale 1. Its source is an 8-VGPR tuple, which inline asm cannot bind, so the values
    are moved into the fixed clobbered tuple v[BASE:BASE+7] first (BASE must be free in the calling kernel). The
    compiler cannot see the conversion's latency, so the asm waits for the VALU write before its result is read;
    without the wait, back-to-back conversions read stale destinations."""
    return tl.inline_asm_elementwise(_pk8_asm(BASE), _pk8_constraints(BASE),
                                     [v0, v1, v2, v3, v4, v5, v6, v7, tl.full(v0.shape, 1.0, tl.float32)],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _quant32(v, R: tl.constexpr, C: tl.constexpr, BASE: tl.constexpr):
    """MXFP4 of fp32 v [R, C]: packed int32 [R, C/8] (byte j = columns 2j, 2j+1, even column in the low nibble) and
    E8M0 uint8 [R, C/32]."""
    vb = tl.reshape(v, (R, C // 32, 32))
    amax = tl.max(tl.abs(vb), axis=2)
    e = ((amax.to(tl.int32, bitcast=True) + 0x200000) & 0x7F800000) >> 23
    unb = tl.minimum(tl.maximum(e - 129, -127), 127)
    q = tl.reshape(vb * tl.exp2((-unb).to(tl.float32))[:, :, None], (R, C // 8, 2, 2, 2))
    s0, s1 = tl.split(q)
    s00, s01 = tl.split(s0)
    s10, s11 = tl.split(s1)
    e0, e4 = tl.split(s00)
    e2, e6 = tl.split(s01)
    e1, e5 = tl.split(s10)
    e3, e7 = tl.split(s11)
    return _pk8(e0, e1, e2, e3, e4, e5, e6, e7, BASE), (unb + 127).to(tl.uint8)


@triton.jit
def _slot_rows(ids_ptr, pos_ptr, group_base, slot, t_ok, expert_start):
    """Expert-sorted row of each slot (local expert e, rank p -> group_base[e] + p) and whether the slot is local."""
    pos = tl.load(pos_ptr + slot, mask=t_ok, other=-1)
    ok = pos >= 0
    local = tl.load(ids_ptr + slot, mask=ok, other=0).to(tl.int32) - expert_start
    return (tl.load(group_base + local, mask=ok, other=0) + pos).to(tl.int64), ok


@triton.jit
def _quant_place_kernel(x_ptr, xq_ptr, xs_ptr, order_ptr, ids_ptr, pos_ptr, group_base, T, expert_start, stride_xm,
                        K: tl.constexpr, TOP_K: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr):
    """Each token's activations are read and quantized once, then written to the expert-sorted row of each of its
    local slots; order[row] = slot (stage 2 stores row r at slot order[r])."""
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    k0 = tl.program_id(1) * BLOCK_K
    t_ok = tokens < T
    if tl.program_id(1) == 0:
        for j in tl.static_range(TOP_K):
            slot = tokens * TOP_K + j
            row, ok = _slot_rows(ids_ptr, pos_ptr, group_base, slot, t_ok, expert_start)
            tl.store(order_ptr + row, slot.to(tl.int64), mask=ok)
    gl: tl.constexpr = _group_layout(BLOCK_T, BLOCK_K, tlx.num_warps())
    offs = tlx.require_layout(tokens.to(tl.int64)[:, None] * stride_xm + (k0 + tl.arange(0, BLOCK_K))[None, :], gl)
    msk = tlx.require_layout(tl.broadcast_to(t_ok[:, None], (BLOCK_T, BLOCK_K)), gl)
    v = tl.load(x_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    q, s = _quant32(v, BLOCK_T, BLOCK_K, 88)
    qo = k0 // 8 + tl.arange(0, BLOCK_K // 8)
    so = k0 // 32 + tl.arange(0, BLOCK_K // 32)
    for j in tl.static_range(TOP_K):
        row, ok = _slot_rows(ids_ptr, pos_ptr, group_base, tokens * TOP_K + j, t_ok, expert_start)
        tl.store(xq_ptr.to(tl.pointer_type(tl.int32)) + row[:, None] * (K // 8) + qo[None, :], q, mask=ok[:, None])
        tl.store(xs_ptr + row[:, None] * (K // 32) + so[None, :], s, mask=ok[:, None])


@triton.jit
def _rows(buf, slot, BM: tl.constexpr, BLOCK_M: tl.constexpr, COLS: tl.constexpr):
    v = tlx.local_view(buf, slot)
    if BM < BLOCK_M:
        v = tlx.local_slice(v, [0, 0], [BM, COLS])
    return v


@triton.jit
def _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, k_iter, pred, BM: tl.constexpr, BLOCK_M: tl.constexpr,
                 BN: tl.constexpr, BKP: tl.constexpr, NUM_BUFFERS: tl.constexpr, FUSE_SILU_MUL: tl.constexpr):
    """One fused TDM op for K step ``k_iter``: waves 0-1 load the A tile, waves 2-3 the weight tile (stage 1: the gate
    rows into the first BN rows of the stage, the up rows into the second)."""
    slot = k_iter % NUM_BUFFERS
    # With 8 waves, waves w and w + 4 share each load member.
    REP: tl.constexpr = 0x11 if tlx.num_warps() == 8 else 1
    a_d = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, k_iter * BKP], pred=pred, clamp_bounds=True)
    b_d = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, k_iter * BKP], pred=pred, clamp_bounds=True)
    a_dst = _rows(a_buf, slot, BM, BLOCK_M, BKP)
    if FUSE_SILU_MUL:
        b2_d = tlx.update_tensor_descriptor(b2_desc, add_offsets=[0, k_iter * BKP], pred=pred, clamp_bounds=True)
        bv = tlx.local_view(b_buf, slot)
        tlx.async_amd_descriptor_load_fused([(a_d, a_dst, 3 * REP),
                                             (b_d, tlx.local_slice(bv, [0, 0], [BN, BKP]), 4 * REP),
                                             (b2_d, tlx.local_slice(bv, [BN, 0], [BN, BKP]), 8 * REP)])
    else:
        tlx.async_amd_descriptor_load_fused([(a_d, a_dst, 3 * REP), (b_d, tlx.local_view(b_buf, slot), 12 * REP)])


@triton.jit
def _tile(a_ptr, as_ptr, c_ptr, cs_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc,
          a_buf, as_buf, b_buf, bs_buf, K: tl.constexpr, BM: tl.constexpr, BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, NUM_BUFFERS: tl.constexpr, FUSE_SILU_MUL: tl.constexpr,
          STORE_ROWS: tl.constexpr):
    """One [BM, BLOCK_N] output tile (BM <= BLOCK_M, the rows of the shared LDS A ring)."""
    K_ITERS: tl.constexpr = K // BLOCK_K
    KP: tl.constexpr = K // 2
    BKP: tl.constexpr = BLOCK_K // 2
    KS: tl.constexpr = K // 32
    BKS: tl.constexpr = BLOCK_K // 32
    NB: tl.constexpr = 2 * BLOCK_N if FUSE_SILU_MUL else BLOCK_N
    a_desc = tl.make_tensor_descriptor(a_ptr + row0 * KP, shape=[rows_left, KP], strides=[KP, tl.constexpr(1)],
                                       block_shape=[BM, BKP])
    as_desc = tl.make_tensor_descriptor(as_ptr + row0 * KS, shape=[rows_left, KS], strides=[KS, tl.constexpr(1)],
                                        block_shape=[BM, KS])
    as_view = _rows(as_buf, 0, BM, BLOCK_M, KS)
    bs_view = tlx.local_view(bs_buf, 0)
    tlx.async_amd_descriptor_load(as_desc, as_view, [0, 0])
    if FUSE_SILU_MUL:
        tlx.async_amd_descriptor_load(bs_desc, tlx.local_slice(bs_view, [0, 0], [BLOCK_N, KS]), [0, 0])
        tlx.async_amd_descriptor_load(bs2_desc, tlx.local_slice(bs_view, [BLOCK_N, 0], [BLOCK_N, KS]), [0, 0])
    else:
        tlx.async_amd_descriptor_load(bs_desc, bs_view, [0, 0])
    for s in tl.static_range(NUM_BUFFERS - 1):
        _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, s, s < K_ITERS, BM, BLOCK_M, BLOCK_N, BKP, NUM_BUFFERS,
                     FUSE_SILU_MUL)
    acc = tl.zeros((BM, NB), dtype=tl.float32)
    for k in range(K_ITERS):
        tlx.async_amd_descriptor_wait(NUM_BUFFERS - 2)
        tlx.workgroup_barrier()
        slot = k % NUM_BUFFERS
        a = tlx.local_load(_rows(a_buf, slot, BM, BLOCK_M, BKP))
        sa = tlx.local_load(tlx.local_slice(as_view, [0, k * BKS], [BM, BKS]))
        b = tlx.local_load(tlx.local_trans(tlx.local_view(b_buf, slot)))
        sb = tlx.local_load(tlx.local_slice(bs_view, [0, k * BKS], [NB, BKS]))
        nk = k + NUM_BUFFERS - 1
        _issue_stage(a_desc, b_desc, b2_desc, a_buf, b_buf, nk, nk < K_ITERS, BM, BLOCK_M, BLOCK_N, BKP, NUM_BUFFERS,
                     FUSE_SILU_MUL)
        acc = tlx.dot_scaled(a, sa, "e2m1", b, sb, "e2m1", acc)
    tlx.async_amd_descriptor_wait(0)
    if FUSE_SILU_MUL:
        # The BF16-rounded [gate | up] accumulator goes through the (now idle) weight ring and comes back in the
        # group-local layout of the requantization, 32 rows at a time.
        c_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[NB, 8]], [BM, NB], [1, 0])
        c_view = tlx.local_view(tlx.local_alloc((BM, NB), tl.bfloat16, 1, layout=c_layout, reuse=b_buf), 0)
        tlx.workgroup_barrier()
        tlx.local_store(c_view, acc.to(tl.bfloat16))
        tlx.workgroup_barrier()
        QR: tl.constexpr = 32
        gl: tl.constexpr = _group_layout(QR, BLOCK_N, tlx.num_warps())
        qo = off_n // 8 + tl.arange(0, BLOCK_N // 8)
        so = off_n // 32 + tl.arange(0, BLOCK_N // 32)
        for r0 in tl.static_range(0, BM, QR):
            g = tlx.local_load(tlx.local_slice(c_view, [r0, 0], [QR, BLOCK_N]), layout=gl).to(tl.float32)
            u = tlx.local_load(tlx.local_slice(c_view, [r0, BLOCK_N], [QR, BLOCK_N]), layout=gl).to(tl.float32)
            q, s = _quant32(g / (1.0 + tl.exp2(g * -_LOG2E)) * u, QR, BLOCK_N, 240)
            rows = r0 + tl.arange(0, QR)
            rm = rows < rows_left
            dst = (row0 + rows).to(tl.int64)
            tl.store(c_ptr.to(tl.pointer_type(tl.int32)) + dst[:, None] * (N // 8) + qo[None, :], q, mask=rm[:, None])
            tl.store(cs_ptr + dst[:, None] * (N // 32) + so[None, :], s, mask=rm[:, None])
    else:
        offs_m = tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BLOCK_N)
        rmask = offs_m < rows_left
        if STORE_ROWS:
            dst = tl.load(store_rows + row0 + offs_m, mask=rmask, other=0).to(tl.int64)
        else:
            dst = (row0 + offs_m).to(tl.int64)
        mask = rmask[:, None] & (offs_n[None, :] < N)
        tl.store(c_ptr + dst[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _w4a4_grouped_kernel(
    a_ptr,  # uint8 [rows, K/2]: packed E2M1 activations, rows sorted by expert
    as_ptr,  # uint8 [rows, K/32]: E8M0
    b_ptr,  # uint8 [E, NB, K/2]: packed E2M1 weights
    bs_ptr,  # uint8 [E, NB, K/32]
    c_ptr,  # stage 1: uint8 [rows, N/2] (MXFP4 of SiLU(gate) * up); stage 2: bf16 [slots, N]
    cs_ptr,  # stage 1: uint8 [rows, N/32]
    store_rows,  # stage 2: int64 [rows], the slot of each row
    tile_meta0,
    tile_meta1,
    tile_count,
    N,  # output columns; stage 1 (FUSE_SILU_MUL): I, with the up rows starting at row N of B
    stride_be,
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
    tl.static_assert(K % BLOCK_K == 0 and BLOCK_K % 128 == 0 and BLOCK_N % 64 == 0 and BLOCK_M >= 32)
    tl.static_assert(NUM_BUFFERS >= 2)
    KP: tl.constexpr = K // 2
    KS: tl.constexpr = K // 32
    BKP: tl.constexpr = BLOCK_K // 2

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
    b_desc = tl.make_tensor_descriptor(b_base + off_n * KP, shape=[N - off_n, KP], strides=[KP, tl.constexpr(1)],
                                       block_shape=[BLOCK_N, BKP])
    bs_desc = tl.make_tensor_descriptor(bs_base + off_n * KS, shape=[N - off_n, KS], strides=[KS, tl.constexpr(1)],
                                        block_shape=[BLOCK_N, KS])
    if FUSE_SILU_MUL:
        b2_desc = tl.make_tensor_descriptor(b_base + (N + off_n) * KP, shape=[N - off_n, KP],
                                            strides=[KP, tl.constexpr(1)], block_shape=[BLOCK_N, BKP])
        bs2_desc = tl.make_tensor_descriptor(bs_base + (N + off_n) * KS, shape=[N - off_n, KS],
                                             strides=[KS, tl.constexpr(1)], block_shape=[BLOCK_N, KS])
    else:
        b2_desc = b_desc
        bs2_desc = bs_desc

    NB: tl.constexpr = 2 * BLOCK_N if FUSE_SILU_MUL else BLOCK_N
    op_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [BLOCK_M, BKP], [1, 0])
    op_b: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [NB, BKP], [1, 0])
    sc_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], [BLOCK_M, KS], [1, 0])
    sc_b: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], [NB, KS], [1, 0])
    a_buf = tlx.local_alloc((BLOCK_M, BKP), tl.uint8, NUM_BUFFERS, layout=op_a)
    as_buf = tlx.local_alloc((BLOCK_M, KS), tl.uint8, 1, layout=sc_a)
    b_buf = tlx.local_alloc((NB, BKP), tl.uint8, NUM_BUFFERS, layout=op_b)
    bs_buf = tlx.local_alloc((NB, KS), tl.uint8, 1, layout=sc_b)
    if HALF_TAIL and BLOCK_M >= 64 and rows_left <= BLOCK_M // 2:
        _tile(a_ptr, as_ptr, c_ptr, cs_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc,
              a_buf, as_buf, b_buf, bs_buf, K, BLOCK_M // 2, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL,
              STORE_ROWS)
    else:
        _tile(a_ptr, as_ptr, c_ptr, cs_ptr, store_rows, row0, rows_left, off_n, N, b_desc, bs_desc, b2_desc, bs2_desc,
              a_buf, as_buf, b_buf, bs_buf, K, BLOCK_M, BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, FUSE_SILU_MUL,
              STORE_ROWS)


def pick_config(T):
    """Tile choice by token count (rows per expert are ~T * topk / E_total). Keys of ``stage2`` override the stage-2
    (down GEMM) tiles."""
    if T <= 512:
        return dict(BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=False)
    if T <= 2048:
        return dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=True)
    return dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_BUFFERS=3, HALF_TAIL=True, stage2=dict(BLOCK_N=256))


def _launch(a, a_s, w, w_s, c, c_s, store_rows, tmap, n_out, k, fuse_silu_mul, cfg):
    meta0, meta1, count, max_tiles = tmap[:4]
    grid = (max_tiles * (n_out // cfg["BLOCK_N"]), )
    _w4a4_grouped_kernel[grid](a, a_s, w, w_s, c, c_s if c_s is not None else count,
                               store_rows if store_rows is not None else count, meta0, meta1, count, n_out,
                               w.stride(0), w_s.stride(0), K=k, BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"],
                               BLOCK_K=cfg["BLOCK_K"], NUM_BUFFERS=cfg["NUM_BUFFERS"], FUSE_SILU_MUL=fuse_silu_mul,
                               STORE_ROWS=store_rows is not None, HALF_TAIL=cfg.get("HALF_TAIL", False), num_warps=4)


def moe_mxfp4_w4a4(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, expert_start=0, post_norm_weight=None,
                   post_norm_eps=1e-6, post_norm_gain=0.0, out=None, cfg=None):
    """BF16 x [T, K] through the local experts [expert_start, expert_start + E) with natural-layout MXFP4 weights and
    MXFP4 activations (quantized per token before stage 1; the stage-1 output requantized before stage 2). Returns
    BF16 [T, H]: the routing-weighted sum over local experts (slots routed elsewhere contribute nothing). ``cfg``
    overrides ``pick_config`` (keys BLOCK_M, BLOCK_N, BLOCK_K, NUM_BUFFERS, HALF_TAIL, stage2)."""
    _check_inputs(x, topk_weights, topk_ids, w13, w13_scale, w2, w2_scale, op=_OP)
    T, K = x.shape
    E, two_i, _ = w13.shape
    I = two_i // 2
    H = w2.shape[1]
    topk = topk_ids.shape[1]
    num_valid = T * topk
    dev = x.device
    a16w4._check_post_norm(_OP, post_norm_weight, H, dev)
    if out is None:
        out = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    elif out.shape != (T, H) or out.dtype != torch.bfloat16 or not out.is_contiguous() or out.device != dev:
        raise ValueError(f"{_OP}: out must be a contiguous bf16 [{T}, {H}] tensor on {dev}")
    if T == 0:
        return out
    cfg = cfg or pick_config(T)
    cfg2 = dict(cfg, **cfg.get("stage2", {}))
    for dim, size, block in (("K", K, cfg["BLOCK_K"]), ("I", I, cfg2["BLOCK_K"]), ("I", I, cfg["BLOCK_N"]),
                             ("H", H, cfg2["BLOCK_N"])):
        if size % block:
            raise ValueError(f"{_OP}: {dim}={size} is not a multiple of the tile ({block})")
    for dim, size in (("K", K), ("I", I)):
        if size % 512:
            raise ValueError(f"{_OP}: {dim}={size} is not a multiple of 512 (the TDM-loaded scale rows, {dim} / 32 "
                             f"bytes, must be whole 16-byte units)")
    for name, w in (("w13", w13), ("w13_scale", w13_scale), ("w2", w2), ("w2_scale", w2_scale)):
        if not w.is_contiguous():
            raise ValueError(f"{_OP}: {name} must be contiguous")

    ids = topk_ids.contiguous()
    pos = torch.empty(num_valid, dtype=torch.int32, device=dev)
    split_sizes = torch.zeros(E, dtype=torch.int32, device=dev)
    a16w4._route_count_kernel[(triton.cdiv(num_valid, 1024), )](ids, pos, split_sizes, num_valid, expert_start, E=E,
                                                                 BLOCK=1024, num_warps=4)
    tmap = a16w4._tile_map(split_sizes, num_valid, cfg["BLOCK_M"])
    order = torch.empty(num_valid, dtype=torch.int64, device=dev)
    xq = _alloc((num_valid, K // 2), torch.uint8, dev)
    xs = _alloc((num_valid, K // 32), torch.uint8, dev)
    xg = x if x.stride(1) == 1 else x.contiguous()
    block_t = 16
    block_k = min(K, 128 if T <= 64 else (256 if T <= 1024 else 512))
    while K % block_k:
        block_k //= 2
    _quant_place_kernel[(triton.cdiv(T, block_t), K // block_k)](xg, xq, xs, order, ids, pos, tmap[4], T, expert_start,
                                                                  xg.stride(0), K=K, TOP_K=topk, BLOCK_T=block_t,
                                                                  BLOCK_K=block_k, num_warps=block_k // 64)
    hq = _alloc((num_valid, I // 2), torch.uint8, dev)
    hs = _alloc((num_valid, I // 32), torch.uint8, dev)
    _launch(xq, xs, w13, w13_scale, hq, hs, None, tmap, I, K, True, cfg)
    y = _alloc((num_valid, H), torch.bfloat16, dev)
    tmap2 = tmap if cfg2["BLOCK_M"] == cfg["BLOCK_M"] else a16w4._tile_map(split_sizes, num_valid, cfg2["BLOCK_M"])
    _launch(hq, hs, w2, w2_scale, y, None, order, tmap2, H, I, False, cfg2)

    a16w4._finalize(y, topk_weights, ids, out, expert_start, E, post_norm_weight, post_norm_eps, post_norm_gain)
    return out
