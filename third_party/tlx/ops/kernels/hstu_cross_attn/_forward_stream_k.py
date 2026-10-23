"""Persistent stream-K forward for gfx950 HSTU cross attention (BM256 / BN64 / D128, non-causal).

Work is the flattened space of (sequence, head, q tile, 64-row KV block) units. One program per CU takes an equal
contiguous range of a cost space in which every q tile is ``ph`` units of fixed cost followed by its KV blocks, and
walks it as segments of single q tiles. A segment covering a whole tile stores the output; a program's first
segment, when it starts mid-tile, stores a normalized fp32 partial (and base-2 LSE) in slot p and raises flag p.
The program that holds the tile's first block merges those partials in registers and stores the output.

Owners wait only on higher-numbered programs, so every program must be resident at once: the grid is one
workgroup per CU, and each workgroup takes a CU's LDS. The flags are self-cleaning (each is reset by its consumer),
so one zeroed buffer per (device, stream) serves every eager launch, and each captured graph keeps its own across
replays.
"""

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.libdevice import fast_dividef
from triton.runtime import driver

from ..flash_attn.gfx950 import (
    LAZY_RESCALE_THRESHOLD,
    SoftmaxState,
    _attn_dot_pv_mfma,
    _attn_war_barrier,
    _concat_cols,
    _split_cols,
)

BLOCK_M = 256
BLOCK_N = 64
HEAD_DIM = 128
# Static segment copies per program; ranges spanning more q tiles finish in a non-pipelined loop.
MAX_SEGMENTS = 5
# Sequence tables are register vectors of next_power_of_2(Z) entries.
MAX_Z = 64
# Fewest unmasked KV blocks that take the pipelined loop.
PIPE_STAGES = 4
# Fixed cost of a q tile, in KV blocks, when the cost space charges one (see `_phantom`).
PHANTOM_BLOCKS = 3
LOG2E: tl.constexpr = tl.constexpr(1.4426950408889634)
_VGPR_ONLY = (("amdgpu-agpr-alloc", "0,0"), )


@triton.jit
def _mma():
    return tlx.amd_mfma_layout(version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[tlx.num_warps(), 1])


@triton.jit
def _qk_mfma(q, kt):
    mma: tl.constexpr = _mma()
    q = tlx.require_layout(q, tlx.dot_operand_layout(0, mma, k_width=8), pin=False)
    kt = tlx.require_layout(kt, tlx.dot_operand_layout(1, mma, k_width=8), pin=False)
    return tl.dot(q, kt, tlx.zeros((q.shape[0], kt.shape[1]), tl.float32, layout=mma))


@triton.jit
def _lanes(N: tl.constexpr):
    """`tl.arange(0, N)` that LLVM can neither hoist out of a loop nor merge with another segment's copy: either
    keeps the lane offsets (and every address derived from them) live, and spilled, across the pipeline."""
    x = tl.inline_asm_elementwise("; lanes $0", "=v,0", [tl.arange(0, N)], dtype=tl.int32, is_pure=False, pack=1)
    return tl.max_contiguous(tl.multiple_of(x, N), N)


@triton.jit
def _wait_stores(x):
    """Per-wave wait for outstanding global stores (the workgroup barrier alone does not wait for them)."""
    return tl.inline_asm_elementwise("s_waitcnt vmcnt(0)", "=v,0", [x], dtype=x.dtype, is_pure=False, pack=1)


# --- Softmax: the dense FA lazy-rescale pipeline, with the row max taken before scaling. --------------------------
# For QK_SCALE > 0, fl(max(qk) * s) == max(fl(qk * s)) because rounding is monotonic, so the frame is unchanged
# while the full-tile multiply shrinks to one multiply per row.


@triton.jit
def _lazy_max(state, qk, QK_SCALE: tl.constexpr, RESCALE_THRESHOLD: tl.constexpr):
    if QK_SCALE > 0:
        m_ij = SoftmaxState._row_max(qk) * QK_SCALE
    else:
        m_ij = SoftmaxState._row_max(qk * QK_SCALE)
    score_delta = m_ij - state.m_i
    advance = score_delta > RESCALE_THRESHOLD
    m_new = tl.where(advance, m_ij, state.m_i)
    return SoftmaxState(state.acc, state.l_i, m_new), score_delta, advance


@triton.jit
def _vec1_lazy_split(state, qk, QK_SCALE: tl.constexpr, RESCALE_THRESHOLD: tl.constexpr):
    state, score_delta, advance = _lazy_max(state, qk, QK_SCALE, RESCALE_THRESHOLD)
    return state, state.vec1_lazy_exp_split(qk, QK_SCALE), score_delta, advance


@triton.jit
def _dot_pv_vec1_lazy(state, p_dot, v_dot, qk, QK_SCALE: tl.constexpr, RESCALE_THRESHOLD: tl.constexpr):
    """`_attn_dot_pv_vec1_lazy` (BM256, split VEC1) with the cheaper row max."""
    tl.static_assert(state.acc.shape[0] == 256)
    mma: tl.constexpr = _mma()
    p_dot = tlx.require_layout(p_dot, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    v_dot = tlx.require_layout(v_dot, tlx.dot_operand_layout(1, mma, k_width=4), pin=False)
    acc = tlx.require_layout(state.acc, mma, pin=False)
    p0 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 0])
    p1 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 16])
    p2 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 32])
    p3 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 48])
    v0 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [0, 0])
    v1 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [16, 0])
    v2 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [32, 0])
    v3 = tlx.extract_slice(v_dot, [16, v_dot.shape[1]], [48, 0])
    acc = tl.dot(p0, v0, acc)
    max_state, score_delta, advance = _lazy_max(state, qk, QK_SCALE, RESCALE_THRESHOLD)
    acc = tl.dot(p1, v1, acc)
    acc = tl.dot(p2, v2, acc)
    acc = tl.dot(p3, v3, acc)
    state = SoftmaxState(acc, state.l_i, max_state.m_i)
    return state, state.vec1_lazy_exp_split(qk, QK_SCALE), score_delta, advance


@triton.jit
def _lazy_step(state, p_c, q, kt_dot, k_ptrs, v_ptrs, block_n, k_buf, v_buf, stride_kn, stride_vn,
               QK_SCALE: tl.constexpr, BLOCK_N: tl.constexpr, CUR_SLOT: tl.constexpr, NEXT_SLOT: tl.constexpr,
               RESCALE_THRESHOLD: tl.constexpr):
    """`_attn_inner_pipelined_lazy_step` (split VEC1, PV-woven VEC1) with the cheaper row max, and fp32 row sums
    for fp16 as well as bf16."""
    with tlx.warp_pipeline_stage("dot1"):
        qk = tl.dot(q, kt_dot)
        state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM=False)
    tlx.async_load_wait_group(1)
    with tlx.warp_pipeline_stage("mem1"):
        v_dot = tlx.local_load(tlx.local_view(v_buf, CUR_SLOT), relaxed=True)
        tok_k = tlx.async_load(k_ptrs + (block_n + 3) * BLOCK_N * stride_kn, tlx.local_view(k_buf, NEXT_SLOT))
        tlx.async_load_commit_group([tok_k])
    with tlx.warp_pipeline_stage("dot2"):
        state, p_c, delta_c, advance_c = _dot_pv_vec1_lazy(state, p_dot, v_dot, qk, QK_SCALE, RESCALE_THRESHOLD)
    tlx.async_load_wait_group(1)
    with tlx.warp_pipeline_stage("mem2"):
        kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, CUR_SLOT)), relaxed=True)
        tok_v = tlx.async_load(v_ptrs + (block_n + 2) * BLOCK_N * stride_vn, tlx.local_view(v_buf, CUR_SLOT))
        tlx.async_load_commit_group([tok_v])
        state = state.rescale_lazy(delta_c, advance_c)
    return state, p_c, kt_dot


@triton.jit
def _softmax_pipelined(state, q, k_ptrs, v_ptrs, n_blocks, k_buf, v_buf, stride_kn, stride_vn, QK_SCALE: tl.constexpr,
                       BLOCK_N: tl.constexpr, RESCALE_THRESHOLD: tl.constexpr):
    """Lazy-rescale pipeline over `n_blocks` (>= PIPE_STAGES, odd) unmasked KV blocks."""
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0))
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_k0])
    tlx.async_load_commit_group([tok_v0])
    tok_k1 = tlx.async_load(k_ptrs + BLOCK_N * stride_kn, tlx.local_view(k_buf, 1))
    tlx.async_load_commit_group([tok_k1])

    wait0 = tlx.async_load_wait_group(2)
    kt0 = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait0, relaxed=True)
    qk = tl.dot(q, kt0)
    state, p_c, _, _ = _vec1_lazy_split(state, qk, QK_SCALE, RESCALE_THRESHOLD)
    # The first frame starts from an empty accumulator, so its correction is a zero denominator.
    state = SoftmaxState(state.acc, tl.zeros([q.shape[0]], tl.float32), state.m_i)
    tl.debug_barrier()
    tok_k2 = tlx.async_load(k_ptrs + 2 * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0))
    tlx.async_load_commit_group([tok_k2])
    wait1 = tlx.async_load_wait_group(1)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 1)), token=wait1, relaxed=True)
    tok_v1 = tlx.async_load(v_ptrs + BLOCK_N * stride_vn, tlx.local_view(v_buf, 1))
    tlx.async_load_commit_group([tok_v1])

    # Pairs keep both ring slots static.
    for pair_idx in tl.range(0, (n_blocks - 3) // 2, num_stages=1):
        block_n = pair_idx * 2
        state, p_c, kt_dot = _lazy_step(state, p_c, q, kt_dot, k_ptrs, v_ptrs, block_n, k_buf, v_buf, stride_kn,
                                        stride_vn, QK_SCALE, BLOCK_N, 0, 1, RESCALE_THRESHOLD)
        state, p_c, kt_dot = _lazy_step(state, p_c, q, kt_dot, k_ptrs, v_ptrs, block_n + 1, k_buf, v_buf, stride_kn,
                                        stride_vn, QK_SCALE, BLOCK_N, 1, 0, RESCALE_THRESHOLD)
    # Drain the last three blocks without further prefetches.
    nm1 = n_blocks - 1
    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(2)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)
    state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM=False)
    state, p_c, delta_c, advance_c = _dot_pv_vec1_lazy(state, p_dot, v_dot, qk, QK_SCALE, RESCALE_THRESHOLD)
    state = state.rescale_lazy(delta_c, advance_c)
    _attn_war_barrier(False)
    tok_vlast = tlx.async_load(v_ptrs + nm1 * BLOCK_N * stride_vn, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_vlast])
    tlx.async_load_wait_group(2)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), relaxed=True)

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(1)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 1), relaxed=True)
    state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM=False)
    state, p_c, delta_c, advance_c = _dot_pv_vec1_lazy(state, p_dot, v_dot, qk, QK_SCALE, RESCALE_THRESHOLD)
    state = state.rescale_lazy(delta_c, advance_c)

    tlx.async_load_wait_group(0)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)
    state, p_dot = state.vec2_lazy_split(p_c, QK_SCALE, q.dtype, FP16_CAST_ROWSUM=False)
    acc = _attn_dot_pv_mfma(state.acc, p_dot, v_dot)
    return SoftmaxState(acc, state.l_i, state.m_i)


@triton.jit
def _softmax_block_masked(state, q, k_ptrs, v_ptrs, start_n, kv_len, k_buf, v_buf, stride_kn, stride_vn,
                          QK_SCALE: tl.constexpr, BLOCK_N: tl.constexpr):
    """One exact (non-lazy) KV block, masked to `kv_len`."""
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = (start_n + offs_n) < kv_len
    tl.debug_barrier()
    tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=mask_n[:, None])
    tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=mask_n[:, None])
    tlx.async_load_commit_group([tok_k, tok_v])
    wait = tlx.async_load_wait_group(0)
    kt = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait)
    qk = _qk_mfma(q, kt) * QK_SCALE
    qk = tl.where(mask_n[None, :], qk, float("-inf"))
    m_new = tl.maximum(state.m_i, tl.max(qk, 1))
    # A fully masked row keeps m = -inf; anchor it at 0 so no -inf - -inf forms.
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.math.exp2(qk - m_safe[:, None])
    corr = tl.math.exp2(state.m_i - m_safe)
    l_i = state.l_i * corr + tl.sum(p, 1)
    v = tlx.local_load(tlx.local_view(v_buf, 0), token=wait)
    acc = _attn_dot_pv_mfma(state.acc * corr[:, None], p.to(q.dtype), v)
    return SoftmaxState(acc, l_i, m_new)


# --- SiLU: the same four-stage ping-pong, SiLU of each block split between its PV stage and the next QK stage. ----
# Each dot stage is one wave issuing 16 MFMAs and half a block of SiLU in order, so the SiLU is woven between the
# MFMAs op by op: four 8-column chunks whose dependent ops sit one to three MFMAs apart, with a scheduling fence
# after every MFMA. Packed fp32 ops stall while the SIMD runs an MFMA; LLVM unpacks them only next to one.


@triton.jit
def _silu(qk, ALPHA: tl.constexpr):
    """`silu(qk * ALPHA)` as `qk * rcp(fma(e, 1/ALPHA, 1/ALPHA))`, with e = exp2(-qk * ALPHA * log2(e))."""
    NEG_ALPHA_LOG2E: tl.constexpr = -ALPHA * LOG2E
    INV_ALPHA: tl.constexpr = 1.0 / ALPHA
    return qk * fast_dividef(1.0, tl.fma(tl.math.exp2(qk * NEG_ALPHA_LOG2E), INV_ALPHA, INV_ALPHA))


@triton.jit
def _kt_piece(k_buf, SLOT: tl.constexpr, K: tl.constexpr, N: tl.constexpr, token=None):
    kt = tlx.local_load(tlx.local_slice(tlx.local_trans(tlx.local_view(k_buf, SLOT)), [16 * K, 32 * N], [16, 32]),
                        token=token, relaxed=True)
    return tlx.require_layout(kt, tlx.dot_operand_layout(1, _mma(), k_width=8), pin=False)


@triton.jit
def _kt_pieces(k_buf, SLOT: tl.constexpr, token=None):
    """K^T as the B operands of the 16 MFMAs of `tl.dot(q, kt)`, [K16 step][N32 tile], each loaded on its own
    (register slices of an LDS-loaded operand miscompile)."""
    return (
        _kt_piece(k_buf, SLOT, 0, 0, token),
        _kt_piece(k_buf, SLOT, 0, 1, token),
        _kt_piece(k_buf, SLOT, 1, 0, token),
        _kt_piece(k_buf, SLOT, 1, 1, token),
        _kt_piece(k_buf, SLOT, 2, 0, token),
        _kt_piece(k_buf, SLOT, 2, 1, token),
        _kt_piece(k_buf, SLOT, 3, 0, token),
        _kt_piece(k_buf, SLOT, 3, 1, token),
        _kt_piece(k_buf, SLOT, 4, 0, token),
        _kt_piece(k_buf, SLOT, 4, 1, token),
        _kt_piece(k_buf, SLOT, 5, 0, token),
        _kt_piece(k_buf, SLOT, 5, 1, token),
        _kt_piece(k_buf, SLOT, 6, 0, token),
        _kt_piece(k_buf, SLOT, 6, 1, token),
        _kt_piece(k_buf, SLOT, 7, 0, token),
        _kt_piece(k_buf, SLOT, 7, 1, token),
    )


@triton.jit
def _v_piece(v_buf, SLOT: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    v = tlx.local_load(tlx.local_slice(tlx.local_view(v_buf, SLOT), [16 * K, 32 * N], [16, 32]), relaxed=True)
    return tlx.require_layout(v, tlx.dot_operand_layout(1, _mma(), k_width=4), pin=False)


@triton.jit
def _v_pieces(v_buf, SLOT: tl.constexpr):
    """V as the B operands of the 16 MFMAs of the PV dot, [K16 step][D32 tile], each loaded on its own."""
    return (
        _v_piece(v_buf, SLOT, 0, 0),
        _v_piece(v_buf, SLOT, 0, 1),
        _v_piece(v_buf, SLOT, 0, 2),
        _v_piece(v_buf, SLOT, 0, 3),
        _v_piece(v_buf, SLOT, 1, 0),
        _v_piece(v_buf, SLOT, 1, 1),
        _v_piece(v_buf, SLOT, 1, 2),
        _v_piece(v_buf, SLOT, 1, 3),
        _v_piece(v_buf, SLOT, 2, 0),
        _v_piece(v_buf, SLOT, 2, 1),
        _v_piece(v_buf, SLOT, 2, 2),
        _v_piece(v_buf, SLOT, 2, 3),
        _v_piece(v_buf, SLOT, 3, 0),
        _v_piece(v_buf, SLOT, 3, 1),
        _v_piece(v_buf, SLOT, 3, 2),
        _v_piece(v_buf, SLOT, 3, 3),
    )


@triton.jit
def _mfma(a, b, acc):
    acc = tl.dot(a, b, acc)
    tlx.amd_sched_barrier(0)
    return acc


@triton.jit
def _chunks(x):
    lo, hi = _split_cols(x)
    a, b = _split_cols(lo)
    c, d = _split_cols(hi)
    return a, b, c, d


@triton.jit
def _unchunk(a, b, c, d):
    return _concat_cols(_concat_cols(a, b), _concat_cols(c, d))


@triton.jit
def _qk_pieces(q, kt):
    """`tl.dot(q, kt)` from `_kt_pieces`: each N32 tile's K16 steps in order, as `tl.dot` issues them."""
    mma: tl.constexpr = _mma()
    q = tlx.require_layout(q, tlx.dot_operand_layout(0, mma, k_width=8), pin=False)
    a = tlx.zeros((q.shape[0], 32), tl.float32, layout=mma)
    b = tlx.zeros((q.shape[0], 32), tl.float32, layout=mma)
    for k in tl.static_range(8):
        a = tl.dot(tlx.extract_slice(q, [q.shape[0], 16], [0, 16 * k]), kt[2 * k], a)
        b = tl.dot(tlx.extract_slice(q, [q.shape[0], 16], [0, 16 * k]), kt[2 * k + 1], b)
    return _concat_cols(a, b)


@triton.jit
def _qk_silu(q, kt, h, ALPHA: tl.constexpr):
    """`_qk_pieces(q, kt)` and `_silu(h)` (as `q.dtype`), woven."""
    NEG_ALPHA_LOG2E: tl.constexpr = -ALPHA * LOG2E
    INV_ALPHA: tl.constexpr = 1.0 / ALPHA
    mma: tl.constexpr = _mma()
    q = tlx.require_layout(q, tlx.dot_operand_layout(0, mma, k_width=8), pin=False)
    q0 = tlx.extract_slice(q, [q.shape[0], 16], [0, 0])
    q1 = tlx.extract_slice(q, [q.shape[0], 16], [0, 16])
    q2 = tlx.extract_slice(q, [q.shape[0], 16], [0, 32])
    q3 = tlx.extract_slice(q, [q.shape[0], 16], [0, 48])
    q4 = tlx.extract_slice(q, [q.shape[0], 16], [0, 64])
    q5 = tlx.extract_slice(q, [q.shape[0], 16], [0, 80])
    q6 = tlx.extract_slice(q, [q.shape[0], 16], [0, 96])
    q7 = tlx.extract_slice(q, [q.shape[0], 16], [0, 112])
    ha, hb, hc, hd = _chunks(h)
    a = tlx.zeros((q.shape[0], 32), tl.float32, layout=mma)
    b = tlx.zeros((q.shape[0], 32), tl.float32, layout=mma)
    a = _mfma(q0, kt[0], a)
    ta = ha * NEG_ALPHA_LOG2E
    b = _mfma(q0, kt[1], b)
    ea = tl.math.exp2(ta)
    a = _mfma(q1, kt[2], a)
    da = tl.fma(ea, INV_ALPHA, INV_ALPHA)
    tb = hb * NEG_ALPHA_LOG2E
    b = _mfma(q1, kt[3], b)
    ra = fast_dividef(1.0, da)
    a = _mfma(q2, kt[4], a)
    pa = ha * ra
    eb = tl.math.exp2(tb)
    b = _mfma(q2, kt[5], b)
    db = tl.fma(eb, INV_ALPHA, INV_ALPHA)
    tc = hc * NEG_ALPHA_LOG2E
    a = _mfma(q3, kt[6], a)
    rb = fast_dividef(1.0, db)
    b = _mfma(q3, kt[7], b)
    pb = hb * rb
    ec = tl.math.exp2(tc)
    a = _mfma(q4, kt[8], a)
    dc = tl.fma(ec, INV_ALPHA, INV_ALPHA)
    td = hd * NEG_ALPHA_LOG2E
    b = _mfma(q4, kt[9], b)
    rc = fast_dividef(1.0, dc)
    a = _mfma(q5, kt[10], a)
    pc = hc * rc
    ed = tl.math.exp2(td)
    b = _mfma(q5, kt[11], b)
    dd = tl.fma(ed, INV_ALPHA, INV_ALPHA)
    a = _mfma(q6, kt[12], a)
    rd = fast_dividef(1.0, dd)
    b = _mfma(q6, kt[13], b)
    pd = hd * rd
    a = _mfma(q7, kt[14], a)
    pa = pa.to(q.dtype)
    pb = pb.to(q.dtype)
    b = _mfma(q7, kt[15], b)
    pc = pc.to(q.dtype)
    pd = pd.to(q.dtype)
    return _concat_cols(a, b), _unchunk(pa, pb, pc, pd)


@triton.jit
def _pv_silu(acc, p_dot, v, h, ALPHA: tl.constexpr):
    """`_attn_dot_pv_mfma(acc, p_dot, v)` from `_v_pieces` (each D32 tile's K16 steps in order, bit for bit) and
    `_silu(h)` (as `p_dot.dtype`), woven."""
    NEG_ALPHA_LOG2E: tl.constexpr = -ALPHA * LOG2E
    INV_ALPHA: tl.constexpr = 1.0 / ALPHA
    mma: tl.constexpr = _mma()
    p_dot = tlx.require_layout(p_dot, tlx.dot_operand_layout(0, mma, k_width=4), pin=False)
    acc = tlx.require_layout(acc, mma, pin=False)
    p0 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 0])
    p1 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 16])
    p2 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 32])
    p3 = tlx.extract_slice(p_dot, [p_dot.shape[0], 16], [0, 48])
    c0 = tlx.extract_slice(acc, [acc.shape[0], 32], [0, 0])
    c1 = tlx.extract_slice(acc, [acc.shape[0], 32], [0, 32])
    c2 = tlx.extract_slice(acc, [acc.shape[0], 32], [0, 64])
    c3 = tlx.extract_slice(acc, [acc.shape[0], 32], [0, 96])
    ha, hb, hc, hd = _chunks(h)
    c0 = _mfma(p0, v[0], c0)
    ta = ha * NEG_ALPHA_LOG2E
    c1 = _mfma(p0, v[1], c1)
    ea = tl.math.exp2(ta)
    c2 = _mfma(p0, v[2], c2)
    da = tl.fma(ea, INV_ALPHA, INV_ALPHA)
    tb = hb * NEG_ALPHA_LOG2E
    c3 = _mfma(p0, v[3], c3)
    ra = fast_dividef(1.0, da)
    c0 = _mfma(p1, v[4], c0)
    pa = ha * ra
    eb = tl.math.exp2(tb)
    c1 = _mfma(p1, v[5], c1)
    db = tl.fma(eb, INV_ALPHA, INV_ALPHA)
    tc = hc * NEG_ALPHA_LOG2E
    c2 = _mfma(p1, v[6], c2)
    rb = fast_dividef(1.0, db)
    c3 = _mfma(p1, v[7], c3)
    pb = hb * rb
    ec = tl.math.exp2(tc)
    c0 = _mfma(p2, v[8], c0)
    dc = tl.fma(ec, INV_ALPHA, INV_ALPHA)
    td = hd * NEG_ALPHA_LOG2E
    c1 = _mfma(p2, v[9], c1)
    rc = fast_dividef(1.0, dc)
    c2 = _mfma(p2, v[10], c2)
    pc = hc * rc
    ed = tl.math.exp2(td)
    c3 = _mfma(p2, v[11], c3)
    dd = tl.fma(ed, INV_ALPHA, INV_ALPHA)
    c0 = _mfma(p3, v[12], c0)
    rd = fast_dividef(1.0, dd)
    c1 = _mfma(p3, v[13], c1)
    pd = hd * rd
    c2 = _mfma(p3, v[14], c2)
    pa = pa.to(p_dot.dtype)
    pb = pb.to(p_dot.dtype)
    c3 = _mfma(p3, v[15], c3)
    pc = pc.to(p_dot.dtype)
    pd = pd.to(p_dot.dtype)
    return _unchunk(c0, c1, c2, c3), _unchunk(pa, pb, pc, pd)


@triton.jit
def _silu_step(acc, p_lo, qk_hi, q, kt_dot, k_ptrs, v_ptrs, block_n, k_buf, v_buf, stride_kn, stride_vn,
               ALPHA: tl.constexpr, BLOCK_N: tl.constexpr, CUR_SLOT: tl.constexpr, NEXT_SLOT: tl.constexpr):
    with tlx.warp_pipeline_stage("dot1"):
        qk, p_hi = _qk_silu(q, kt_dot, qk_hi, ALPHA)
    tlx.async_load_wait_group(1)
    with tlx.warp_pipeline_stage("mem1"):
        p_dot = _concat_cols(p_lo, p_hi)
        v_dot = _v_pieces(v_buf, CUR_SLOT)
        tok_k = tlx.async_load(k_ptrs + (block_n + 3) * BLOCK_N * stride_kn, tlx.local_view(k_buf, NEXT_SLOT))
        tlx.async_load_commit_group([tok_k])
    with tlx.warp_pipeline_stage("dot2"):
        qk_lo, qk_hi = _split_cols(qk)
        acc, p_lo = _pv_silu(acc, p_dot, v_dot, qk_lo, ALPHA)
    tlx.async_load_wait_group(1)
    with tlx.warp_pipeline_stage("mem2"):
        kt_dot = _kt_pieces(k_buf, CUR_SLOT)
        tok_v = tlx.async_load(v_ptrs + (block_n + 2) * BLOCK_N * stride_vn, tlx.local_view(v_buf, CUR_SLOT))
        tlx.async_load_commit_group([tok_v])
    return acc, p_lo, qk_hi, kt_dot


@triton.jit
def _silu_pipelined(acc, q, k_ptrs, v_ptrs, n_blocks, k_buf, v_buf, stride_kn, stride_vn, ALPHA: tl.constexpr,
                    BLOCK_N: tl.constexpr):
    """Pipeline over `n_blocks` (>= PIPE_STAGES, odd) unmasked KV blocks."""
    tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0))
    tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_k0])
    tlx.async_load_commit_group([tok_v0])
    tok_k1 = tlx.async_load(k_ptrs + BLOCK_N * stride_kn, tlx.local_view(k_buf, 1))
    tlx.async_load_commit_group([tok_k1])

    wait0 = tlx.async_load_wait_group(2)
    kt0 = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait0, relaxed=True)
    qk = tl.dot(q, kt0)
    qk_lo, qk_hi = _split_cols(qk)
    p_lo = _silu(qk_lo, ALPHA).to(q.dtype)
    tl.debug_barrier()
    tok_k2 = tlx.async_load(k_ptrs + 2 * BLOCK_N * stride_kn, tlx.local_view(k_buf, 0))
    tlx.async_load_commit_group([tok_k2])
    wait1 = tlx.async_load_wait_group(1)
    kt_dot = _kt_pieces(k_buf, 1, wait1)
    tok_v1 = tlx.async_load(v_ptrs + BLOCK_N * stride_vn, tlx.local_view(v_buf, 1))
    tlx.async_load_commit_group([tok_v1])

    for pair_idx in tl.range(0, (n_blocks - 3) // 2, num_stages=1):
        block_n = pair_idx * 2
        acc, p_lo, qk_hi, kt_dot = _silu_step(acc, p_lo, qk_hi, q, kt_dot, k_ptrs, v_ptrs, block_n, k_buf, v_buf,
                                              stride_kn, stride_vn, ALPHA, BLOCK_N, 0, 1)
        acc, p_lo, qk_hi, kt_dot = _silu_step(acc, p_lo, qk_hi, q, kt_dot, k_ptrs, v_ptrs, block_n + 1, k_buf, v_buf,
                                              stride_kn, stride_vn, ALPHA, BLOCK_N, 1, 0)
    p_c = _concat_cols(p_lo, _silu(qk_hi, ALPHA).to(q.dtype))
    nm1 = n_blocks - 1
    qk = _qk_pieces(q, kt_dot)
    tlx.async_load_wait_group(2)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)
    acc = _attn_dot_pv_mfma(acc, p_c, v_dot)
    p_c = _silu(qk, ALPHA)
    _attn_war_barrier(False)
    tok_vlast = tlx.async_load(v_ptrs + nm1 * BLOCK_N * stride_vn, tlx.local_view(v_buf, 0))
    tlx.async_load_commit_group([tok_vlast])
    tlx.async_load_wait_group(2)
    kt_dot = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), relaxed=True)

    qk = tl.dot(q, kt_dot)
    tlx.async_load_wait_group(1)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 1), relaxed=True)
    acc = _attn_dot_pv_mfma(acc, p_c.to(q.dtype), v_dot)
    p_c = _silu(qk, ALPHA)

    tlx.async_load_wait_group(0)
    v_dot = tlx.local_load(tlx.local_view(v_buf, 0), relaxed=True)
    return _attn_dot_pv_mfma(acc, p_c.to(q.dtype), v_dot)


@triton.jit
def _silu_block_masked(acc, q, k_ptrs, v_ptrs, start_n, kv_len, k_buf, v_buf, stride_kn, stride_vn, ALPHA: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = (start_n + offs_n) < kv_len
    tl.debug_barrier()
    tok_k = tlx.async_load(k_ptrs + start_n * stride_kn, tlx.local_view(k_buf, 0), mask=mask_n[:, None])
    tok_v = tlx.async_load(v_ptrs + start_n * stride_vn, tlx.local_view(v_buf, 0), mask=mask_n[:, None])
    tlx.async_load_commit_group([tok_k, tok_v])
    wait = tlx.async_load_wait_group(0)
    kt = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, 0)), token=wait)
    p = tl.where(mask_n[None, :], _silu(_qk_mfma(q, kt), ALPHA), 0.0)
    v = tlx.local_load(tlx.local_view(v_buf, 0), token=wait)
    return _attn_dot_pv_mfma(acc, p.to(q.dtype), v)


# --- Stream-K partition and fixup. ---------------------------------------------------------------------------------


@triton.jit
def _seq_tables(seq_offsets_q, seq_offsets, Z, H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                Z_PAD: tl.constexpr):
    """Per-sequence KV blocks and q tiles, and the exclusive prefix of the unit counts (H * tiles * blocks).

    A q tile with no KV still takes one (empty) block, so its output is written."""
    zs = tl.arange(0, Z_PAD)
    ok = zs < Z
    lq = (tl.load(seq_offsets_q + zs + 1, mask=ok, other=0) - tl.load(seq_offsets_q + zs, mask=ok, other=0)).to(
        tl.int32)
    lk = (tl.load(seq_offsets + zs + 1, mask=ok, other=0) - tl.load(seq_offsets + zs, mask=ok, other=0)).to(tl.int32)
    nb = tl.maximum(tl.cdiv(lk, BLOCK_N), 1)
    tiles = tl.where(ok, H * tl.cdiv(lq, BLOCK_M), 0)
    units = tiles * nb
    return zs, tl.cumsum(units, 0) - units, units, nb, tiles


@triton.jit
def _seq_kv_blocks(seq_offsets, z, BLOCK_N: tl.constexpr):
    lk = (tl.load(seq_offsets + z + 1) - tl.load(seq_offsets + z)).to(tl.int32)
    return tl.maximum(tl.cdiv(lk, BLOCK_N), 1)


@triton.jit
def _owner(v, W, P):
    """Program whose range [p*W//P, (p+1)*W//P) contains cost-space position v."""
    return ((v + 1).to(tl.int64) * P + W - 1) // W - 1


@triton.jit
def _first_unit(tab, v, Z, W, ph, seq_offsets, Z_PAD: tl.constexpr, LOG2_Z_PAD: tl.constexpr, BLOCK_N: tl.constexpr):
    """First unit at or after cost-space position v (a tile's fixed-cost prefix maps to its first block)."""
    lo = 0
    hi = Z - 1
    for _ in tl.static_range(LOG2_Z_PAD):
        mid = (lo + hi + 1) // 2
        take = tl.load(tab + Z_PAD + 2 + mid) <= v
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    rel = v - tl.load(tab + Z_PAD + 2 + lo)
    nb = _seq_kv_blocks(seq_offsets, lo, BLOCK_N)
    t = rel // (nb + ph)
    u = tl.load(tab + lo) + t * nb + tl.maximum(rel - t * (nb + ph) - ph, 0)
    return tl.where(v >= W, tl.load(tab + Z_PAD), u)


@triton.jit
def _part_ptrs(base):
    """A [256, 128] fp32 partial stored as each thread's MFMA registers: dims (16 register groups, 8 warps, 2 lane
    halves, 32 lanes, 4 columns), so every access is one 16-byte vector per thread and 1 KB contiguous per
    instruction, with no relayout on either side."""
    g = tl.arange(0, 16)[:, None, None, None, None]
    w = tl.arange(0, 8)[None, :, None, None, None]
    lh = tl.arange(0, 2)[None, None, :, None, None]
    ll = tl.arange(0, 32)[None, None, None, :, None]
    j = tl.arange(0, 4)[None, None, None, None, :]
    return base + (g * 2048 + w * 256 + lh * 128 + ll * 4 + j)


@triton.jit
def _to_part(acc):
    # (row = 32 w + lane, col = 32 gn + 8 gi + 4 lh + j) -> (gn gi, w, lh, lane, j)
    x = tl.reshape(acc, [8, 32, 4, 4, 2, 4])
    x = tl.permute(x, (2, 3, 0, 4, 1, 5))
    return tl.reshape(x, [16, 8, 2, 32, 4])


@triton.jit
def _from_part(x):
    x = tl.reshape(x, [4, 4, 8, 2, 32, 4])
    x = tl.permute(x, (2, 4, 0, 1, 3, 5))
    return tl.reshape(x, [256, 128])


@triton.jit
def _segment(Q, K, V, Out, M, Part, PartLse, seq_offsets_q, seq_offsets, attn_scale, tab, Flags, Z, W, ph,
             P: tl.constexpr, p, u, u1, k_buf, v_buf, stride_qm: tl.constexpr, stride_qh: tl.constexpr,
             stride_kn: tl.constexpr, stride_kh: tl.constexpr, stride_vn: tl.constexpr, stride_vh: tl.constexpr,
             stride_om: tl.constexpr, stride_oh: tl.constexpr, stride_mm: tl.constexpr, G: tl.constexpr,
             QK_SCALE: tl.constexpr, IS_SOFTMAX: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr, Z_PAD: tl.constexpr, LOG2_Z_PAD: tl.constexpr, PIPELINE: tl.constexpr,
             PIPE_STAGES: tl.constexpr, RESCALE_THRESHOLD: tl.constexpr):
    """Process units [u, min(u1, end of u's q tile)) of one q tile; return the next unit."""
    offs_n = _lanes(BLOCK_N)
    offs_d = _lanes(HEAD_DIM)
    local_m = _lanes(BLOCK_M)
    # Largest z with starts[z] <= u (padding rows start at W, past every unit).
    lo = 0
    hi = Z - 1
    for _ in tl.static_range(LOG2_Z_PAD):
        mid = (lo + hi + 1) // 2
        take = tl.load(tab + mid) <= u
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid - 1)
    z = lo
    rel = u - tl.load(tab + z)
    lq_z = tl.load(seq_offsets_q + z + 1).to(tl.int32) - tl.load(seq_offsets_q + z).to(tl.int32)
    lk_z = tl.load(seq_offsets + z + 1).to(tl.int32) - tl.load(seq_offsets + z).to(tl.int32)
    nb = tl.maximum(tl.cdiv(lk_z, BLOCK_N), 1)
    nt = tl.cdiv(lq_z, BLOCK_M)
    t_lin = rel // nb
    b0 = rel - t_lin * nb
    b1 = tl.minimum(nb, b0 + (u1 - u))
    off_h = t_lin // nt
    start_m = (t_lin - off_h * nt) * BLOCK_M
    off_h_kv = off_h // G

    seq_start_q = tl.load(seq_offsets_q + z).to(tl.int32)
    seq_len_q = tl.load(seq_offsets_q + z + 1).to(tl.int32) - seq_start_q
    seq_start_kv = tl.load(seq_offsets + z).to(tl.int32)
    seq_len_kv = tl.load(seq_offsets + z + 1).to(tl.int32) - seq_start_kv
    kv_lo = b0 * BLOCK_N
    kv_len = tl.minimum(b1 * BLOCK_N, seq_len_kv) - kv_lo

    row_ok = (start_m + local_m) < seq_len_q
    row0 = (seq_start_q + start_m).to(tl.int64)
    q_ptrs = Q + row0 * stride_qm + off_h * stride_qh + (local_m[:, None] * stride_qm + offs_d[None, :])
    if start_m + BLOCK_M <= seq_len_q:
        q = tl.load(q_ptrs, cache_modifier=".cg")
    else:
        q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0, cache_modifier=".cg")
    kv_base = (seq_start_kv + kv_lo).to(tl.int64)
    k_ptrs = K + kv_base * stride_kn + off_h_kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    v_ptrs = V + kv_base * stride_vn + off_h_kv * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]
    n_full = kv_len // BLOCK_N
    # The previous segment's last LDS reads must finish before this segment's first copies land.
    tl.debug_barrier()

    if IS_SOFTMAX:
        state = SoftmaxState.create(BLOCK_M, HEAD_DIM)
        # One masked-tail loop per branch: a shared loop after the branch costs ~10% (the merged state).
        if PIPELINE and n_full >= PIPE_STAGES:
            # The paired steady state takes an odd block count; an extra block joins the masked tail.
            n_pipe = n_full - (n_full - 3) % 2
            state = _softmax_pipelined(state, q, k_ptrs, v_ptrs, n_pipe, k_buf, v_buf, stride_kn, stride_vn, QK_SCALE,
                                       BLOCK_N, RESCALE_THRESHOLD)
            for start_n in tl.range(n_pipe * BLOCK_N, kv_len, BLOCK_N, num_stages=1):
                state = _softmax_block_masked(state, q, k_ptrs, v_ptrs, start_n, kv_len, k_buf, v_buf, stride_kn,
                                              stride_vn, QK_SCALE, BLOCK_N)
        else:
            for start_n in tl.range(0, kv_len, BLOCK_N, num_stages=1):
                state = _softmax_block_masked(state, q, k_ptrs, v_ptrs, start_n, kv_len, k_buf, v_buf, stride_kn,
                                              stride_vn, QK_SCALE, BLOCK_N)
        mma: tl.constexpr = _mma()
        l_recip = tlx.require_layout(1.0 / state.l_i, tlx.slice_layout(mma, 1))
        acc = state.acc * l_recip[:, None]
        lse = state.m_i + tl.math.log2(state.l_i)
    else:
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        if PIPELINE and n_full >= PIPE_STAGES:
            n_pipe = n_full - (n_full - 3) % 2
            acc = _silu_pipelined(acc, q, k_ptrs, v_ptrs, n_pipe, k_buf, v_buf, stride_kn, stride_vn, QK_SCALE, BLOCK_N)
            n_done = n_pipe
        else:
            n_done = 0
        for start_n in tl.range(n_done * BLOCK_N, kv_len, BLOCK_N, num_stages=1):
            acc = _silu_block_masked(acc, q, k_ptrs, v_ptrs, start_n, kv_len, k_buf, v_buf, stride_kn, stride_vn,
                                     QK_SCALE, BLOCK_N)
        acc = acc * tl.load(attn_scale).to(tl.float32)
        lse = tl.zeros([BLOCK_M], dtype=tl.float32)

    local_m = _lanes(BLOCK_M)
    row_ok = (start_m + local_m) < seq_len_q
    offs_d = _lanes(HEAD_DIM)
    mma_st: tl.constexpr = _mma()
    # Fixup: the program holding a tile's first block folds in the partials of the later programs covering the
    # rest of it (each such partial is that program's first segment, in slot p).
    mid = b0 > 0
    v_last = tl.load(tab + Z_PAD + 2 + z) + t_lin * (nb + ph) + ph + nb - 1
    p_hi = tl.where((b0 == 0) & (b1 < nb), _owner(v_last, W, P).to(tl.int32), p)
    if IS_SOFTMAX:
        m = tlx.release_layout(lse)
        l = tl.full([BLOCK_M], 1.0, tl.float32)
    for pp in range(p + 1, p_hi + 1):
        # With fewer units than programs some ranges are empty and raise no flag. (With the fixed tile cost every
        # range spans >= 6 cost positions, so none is.)
        if (W >= P) or ((pp + 1) * W // P > pp * W // P):
            while tl.atomic_cas(Flags + pp, 1, 0, sem="acquire", scope="gpu") != 1:
                pass
            tl.debug_barrier()
            x = _from_part(tl.load(_part_ptrs(Part + pp.to(tl.int64) * (BLOCK_M * HEAD_DIM))))
            if IS_SOFTMAX:
                lse_s = tl.load(PartLse + pp.to(tl.int64) * BLOCK_M + local_m, mask=row_ok, other=0.0)
                m_new = tl.maximum(m, lse_s)
                c_old = tl.math.exp2(m - m_new)
                c_s = tl.math.exp2(lse_s - m_new)
                acc = acc * c_old[:, None] + x * c_s[:, None]
                l = l * c_old + c_s
                m = m_new
            else:
                acc = acc + x
    if IS_SOFTMAX:
        acc = acc * (1.0 / l)[:, None]
        lse = m + tl.math.log2(l)
    final = b0 == 0
    slot = p.to(tl.int64)
    # Both stores are issued with disjoint masks; a branch here costs registers across the epilogue.
    # Register-local store layout (as the dense kernel's epilogue): no LDS staging, no staging addresses.
    out_layout: tl.constexpr = tlx.layout(shape=((32, 2, 8), (8, 8)), stride=((128, 8, 4096), (1, 16)))
    o_ptrs = Out + row0 * stride_om + off_h * stride_oh + (local_m[:, None] * stride_om + offs_d[None, :])
    o_mask = (row_ok & final)[:, None] & (offs_d[None, :] < HEAD_DIM)
    tl.store(tlx.require_layout(o_ptrs, out_layout), tlx.require_layout(acc.to(Out.dtype.element_ty), out_layout),
             mask=tlx.require_layout(o_mask, out_layout))
    # Rows past the sequence hold finite values (their q rows load as zeros) that the owner never stores.
    tl.store(_part_ptrs(Part + slot * (BLOCK_M * HEAD_DIM)), _to_part(tlx.require_layout(acc, mma_st)), mask=not final)
    if IS_SOFTMAX:
        tl.store(M + row0 * stride_mm + off_h + local_m * stride_mm, lse, mask=row_ok & final)
        tl.store(PartLse + slot * BLOCK_M + local_m, lse, mask=row_ok & (not final))
    local_m = _wait_stores(local_m)
    tl.debug_barrier()
    tl.atomic_xchg(Flags + p, 1, sem="release", scope="gpu", mask=mid)
    return u + (b1 - b0)


@triton.jit
def _hstu_cross_attn_fwd_stream_k(
    Q,
    K,
    V,
    Out,
    M,
    Part,
    PartLse,
    seq_offsets_q,
    seq_offsets,
    attn_scale,
    Tab,
    Flags,
    Z,
    P: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_mm: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    QK_SCALE: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Z_PAD: tl.constexpr,
    LOG2_Z_PAD: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    PHANTOM_BLOCKS: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    pid = tl.program_id(0)
    # Workgroups go round-robin over the 8 XCDs; give each XCD a contiguous range so the programs sharing a
    # sequence's K/V share an L2.
    p = (pid % 8) * (P // 8) + pid // 8
    # Round-trip the tables through this program's scratch row: values reduced from register tensors are not
    # provably wave-uniform, and every scalar derived from them (pointers, loop bounds) would land in VGPRs.
    # Row: unit starts [0, Z_PAD), W, 0, cost-space starts [Z_PAD + 2, 2 Z_PAD + 2), cost-space total, phantom.
    zs, starts, units, _, tiles = _seq_tables(seq_offsets_q, seq_offsets, Z, H, BLOCK_M, BLOCK_N, Z_PAD)
    # Charging each q tile a fixed cost evens out the per-segment overhead once most programs cover more than
    # one tile; below that it only moves split points.
    ph = tl.where(2 * tl.sum(tiles, 0) >= 3 * P, PHANTOM_BLOCKS, 0)
    vstarts = starts + ph * (tl.cumsum(tiles, 0) - tiles)
    tab = Tab + pid.to(tl.int64) * (2 * Z_PAD + 4)
    first = zs == 0
    tl.store(tab + zs, starts)
    tl.store(tab + Z_PAD + zs * 0, tl.sum(units, 0) + zs * 0, mask=first)
    tl.store(tab + Z_PAD + 1 + zs * 0, zs * 0, mask=first)
    tl.store(tab + Z_PAD + 2 + zs, vstarts)
    tl.store(tab + 2 * Z_PAD + 2 + zs * 0, tl.sum(units + ph * tiles, 0) + zs * 0, mask=first)
    tl.store(tab + 2 * Z_PAD + 3 + zs * 0, ph + zs * 0, mask=first)
    tl.debug_barrier()
    W = tl.load(tab + 2 * Z_PAD + 2)
    ph = tl.load(tab + 2 * Z_PAD + 3)
    u = (p.to(tl.int64) * W // P).to(tl.int32)
    u1 = ((p + 1).to(tl.int64) * W // P).to(tl.int32)
    if ph != 0:
        u = _first_unit(tab, u, Z, W, ph, seq_offsets, Z_PAD, LOG2_Z_PAD, BLOCK_N)
        u1 = _first_unit(tab, u1, Z, W, ph, seq_offsets, Z_PAD, LOG2_Z_PAD, BLOCK_N)

    kv_offset_bases: tl.constexpr = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0],
                                     [1, 0], [2, 0], [4, 0], [8, 0]]
    k_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 8)], kv_offset_bases,
                                                                          [BLOCK_N, HEAD_DIM])
    v_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 32)], kv_offset_bases,
                                                                          [BLOCK_N, HEAD_DIM])
    k_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), 2, layout=k_layout)
    v_buf = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(V), 2, layout=v_layout)
    # Statically unrolled: in a runtime loop LLVM hoists the per-segment address math out of it and spills.
    for _ in tl.static_range(MAX_SEGMENTS):
        if u < u1:
            u = _segment(Q, K, V, Out, M, Part, PartLse, seq_offsets_q, seq_offsets, attn_scale, tab, Flags, Z, W, ph,
                         P, p, u, u1, k_buf, v_buf, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh,
                         stride_om, stride_oh, stride_mm, G, QK_SCALE, IS_SOFTMAX, HEAD_DIM, BLOCK_M, BLOCK_N, Z_PAD,
                         LOG2_Z_PAD, True, PIPE_STAGES, RESCALE_THRESHOLD)
    # Rare: ranges spanning more than MAX_SEGMENTS q tiles (short KV sequences).
    while u < u1:
        u = _segment(Q, K, V, Out, M, Part, PartLse, seq_offsets_q, seq_offsets, attn_scale, tab, Flags, Z, W, ph, P, p,
                     u, u1, k_buf, v_buf, stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_om,
                     stride_oh, stride_mm, G, QK_SCALE, IS_SOFTMAX, HEAD_DIM, BLOCK_M, BLOCK_N, Z_PAD, LOG2_Z_PAD,
                     False, PIPE_STAGES, RESCALE_THRESHOLD)


_FLAGS = {}


def _flags(device: torch.device, num_programs: int) -> torch.Tensor:
    """Zeroed handoff flags for launches on the current stream; every raised flag is consumed (reset to 0)."""
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        # Traced through (export under AOTI_LOWER): a cached tensor would leak out of the trace. Captured: graphs
        # share a capture stream but may replay concurrently, so each takes flags from its own pool.
        return torch.zeros(num_programs, device=device, dtype=torch.int32)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, driver.active.get_current_stream(index), num_programs)
    flags = _FLAGS.get(key)
    if flags is None:
        flags = _FLAGS[key] = torch.zeros(num_programs, device=device, dtype=torch.int32)
    return flags


def stream_k_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, Z: int, num_softmax_heads: int,
                       shared_kv: bool, causal: bool, truncate_method: str, num_cus: int) -> bool:
    H = q.shape[1]
    return (not causal and truncate_method == "none" and not shared_kv and 0 < Z <= MAX_Z and num_cus % 8 == 0
            and q.dtype in (torch.float16, torch.bfloat16) and q.shape[2] == HEAD_DIM and v.shape[2] == HEAD_DIM
            and num_softmax_heads in (0, H) and q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1)


def stream_k_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_offsets: torch.Tensor,
                 seq_offsets_q: torch.Tensor, alpha: float, attn_scale: torch.Tensor, G: int, out: torch.Tensor,
                 M: Optional[torch.Tensor], num_cus: int) -> None:
    """Writes `out` and, for softmax (`M` given), the base-2 LSE `M`. Launches one program per CU: they must all
    become resident (no persistent kernel or CU mask keeping CUs from this launch), or the fixup waits forever."""
    H, D = q.shape[1], q.shape[2]
    Z = seq_offsets.numel() - 1
    is_softmax = M is not None
    if not is_softmax and alpha == 0:
        # `_silu` divides by ALPHA; silu(0) is 0 everywhere.
        out.zero_()
        return
    P = num_cus
    z_pad = 1 << (max(Z, 2) - 1).bit_length()
    part = torch.empty((P, BLOCK_M, D), device=q.device, dtype=torch.float32)
    part_lse = torch.empty((P, BLOCK_M), device=q.device, dtype=torch.float32)
    tab = torch.empty((P, 2 * z_pad + 4), device=q.device, dtype=torch.int32)
    m_arg = M if is_softmax else part_lse
    _hstu_cross_attn_fwd_stream_k[(P, )](
        q, k, v, out, m_arg, part, part_lse, seq_offsets_q, seq_offsets, attn_scale, tab, _flags(q.device, P), Z, P=P,
        stride_qm=q.stride(0), stride_qh=q.stride(1), stride_kn=k.stride(0), stride_kh=k.stride(1),
        stride_vn=v.stride(0), stride_vh=v.stride(1), stride_om=out.stride(0), stride_oh=out.stride(1),
        stride_mm=M.stride(0) if is_softmax else 0, H=H, G=G, QK_SCALE=alpha * LOG2E.value if is_softmax else alpha,
        IS_SOFTMAX=is_softmax, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, Z_PAD=z_pad,
        LOG2_Z_PAD=max(1, (z_pad - 1).bit_length()), MAX_SEGMENTS=MAX_SEGMENTS, PHANTOM_BLOCKS=PHANTOM_BLOCKS,
        PIPE_STAGES=PIPE_STAGES, RESCALE_THRESHOLD=LAZY_RESCALE_THRESHOLD.value, num_warps=8, waves_per_eu=2,
        llvm_fn_attrs=_VGPR_ONLY, enable_sched_group_barrier_scheduler=False)
