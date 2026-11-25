"""Jagged softmax attention forward and backward on Blackwell (sm100).

Ported from the ads_mkl TLX jagged flash attention (fbcode/ads_mkl/ops/oss/tlx_jfa), with its symmetric sliding
window generalized to flash-attn's bottom-right aligned ``(left, right)`` band, so causal masking is ``right == 0``.
"""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.cuda.inline_ptx_lib import _fma_f32x2, _mul_f32x2
from triton.language.extra.tlx.warp_spec import get_bufidx_phase as _get_bufidx_phase

from ._sm100_common import _apply_column_range, _join_n, _kv_range, _reduce_or, _row_has_key, _split_n, _unmasked_kv_range
from .sm100_bwd import _BWD_CONFIG, _attn_bwd_ws

# One config each; the kernels launch without the autotuner, whose per-call dispatch dominates small calls.
_FWD_CONFIG = dict(BLOCK_M=256, BLOCK_N=64, NUM_BUFFERS_Q=1, NUM_BUFFERS_KV=4, NUM_BUFFERS_QK=1, NUM_MMA_GROUPS=2,
                   NUM_MMA_SLICES=2, NUM_REGS_SFM=184, NUM_REGS_MMA=24, NUM_REGS_LOAD=32, NUM_REGS_EPILOG=24,
                   USE_CLC=True, RESCALE_OPT=True, num_warps=4, num_stages=0)
_PREPROC_CONFIG = dict(BLOCK_M=128, num_warps=8, num_stages=2)


@triton.jit  # pragma: no cover
def _compute_seq_len(
    tile_idx,
    n_tile_num,
    Q_offsets,
    K_offsets,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    BLOCK_M: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m_start,
    valid_tiles_m_end,
    valid_tiles_h,
    sm_offsets,
    cur_sm_offset,
    Z_order,
):
    if ENABLE_LOAD_BALANCING:
        off_h = tl.load(valid_tiles_h + cur_sm_offset).to(tl.int64)
        off_z = tl.load(valid_tiles_b + cur_sm_offset).to(tl.int64)
        if not BROADCAST_Q:
            begin_q = tl.load(valid_tiles_m_start + cur_sm_offset).to(tl.int64)
            end_q = tl.load(valid_tiles_m_end + cur_sm_offset).to(tl.int64)
            start_m = 0
        else:
            begin_q = tl.load(Q_offsets)
            end_q = tl.load(Q_offsets + 1)
            m_start_abs = tl.load(valid_tiles_m_start + cur_sm_offset).to(tl.int64)
            start_m = (m_start_abs - begin_q) // BLOCK_M
    else:
        off_hz = tile_idx // n_tile_num
        # Z_order lists the sequences longest-first, so the longest tiles start first and finish
        # before the short ones, instead of leaving a tail.
        if Z_order is not None:
            off_z = tl.load(Z_order + off_hz // H).to(off_hz.dtype)
        else:
            off_z = off_hz // H
        off_h = off_hz % H
        if not BROADCAST_Q:
            off_q_z = off_z
        else:
            off_q_z = 0
        begin_q = tl.load(Q_offsets + off_q_z)
        end_q = tl.load(Q_offsets + off_q_z + 1)
        start_m = tile_idx % n_tile_num

    qlen = end_q - begin_q
    qlen = tl.minimum(qlen, N_CTX)

    begin_k = tl.load(K_offsets + off_z)
    end_k = tl.load(K_offsets + off_z + 1)
    klen = end_k - begin_k

    return begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m


@triton.jit  # pragma: no cover
def _softmax_inner_iter(
    cid,
    accum_cnt_qk,
    m_i,
    l_i,
    qk_scale,
    klen,
    start_n,
    start_m,
    diag,
    qk_fulls,
    qk_tiles,
    alpha_empties,
    alpha_fulls,
    alpha_tiles,
    p_fulls,
    p_tiles,
    out_dtype,
    BLOCK_M_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    RESCALE_OPT: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    APPLY_MASK: tl.constexpr,
    APPLY_BAND: tl.constexpr = True,
    mask_tile=True,
):
    # Per-iteration body of the softmax warpgroup. APPLY_MASK is constexpr so
    # the bulk vs tail loops generate different code: bulk emits no mask `where`
    # at all, tail emits the klen-bound mask.
    _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
    tlx.barrier_wait(qk_fulls[cid], qk_phase)
    qk = tlx.local_load(qk_tiles[cid])
    HAS_BAND: tl.constexpr = APPLY_BAND and (WINDOW_LEFT is not None or WINDOW_RIGHT is not None)
    if APPLY_MASK or HAS_BAND:
        # Row r of the tile keeps columns [r + lo0, min(r + hi0, cap)).
        diag_col = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + diag - start_n
        lo0 = -BLOCK_M_SPLIT
        hi0 = BLOCK_N + BLOCK_M_SPLIT
        cap = BLOCK_N
        if APPLY_MASK:
            cap = klen - start_n
        if HAS_BAND:
            if WINDOW_RIGHT is not None:
                hi0 = diag_col + WINDOW_RIGHT + 1
            if WINDOW_LEFT is not None:
                lo0 = diag_col - WINDOW_LEFT
        if mask_tile:
            qk = _apply_column_range(qk, lo0, hi0, cap, -1.0e9, BLOCK_M_SPLIT, BLOCK_N)

    if RESCALE_OPT:
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
    else:
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

    if RESCALE_OPT:
        alpha_ = (m_i - m_ij) * qk_scale
        alpha = tl.math.exp2(alpha_)
        rescale_mask = alpha_ >= -8.0
        alpha = tl.where(rescale_mask, 1.0, alpha)
        m_ij = tl.where(rescale_mask, m_i, m_ij)
    else:
        alpha = tl.math.exp2(m_i - m_ij)
    tlx.barrier_wait(alpha_empties[cid], qk_phase ^ 1)
    tlx.local_store(alpha_tiles[cid], alpha[:, None])
    tlx.barrier_arrive(alpha_fulls[cid])

    if RESCALE_OPT:
        m_scaled = m_ij * qk_scale
        qk = _fma_f32x2(qk, qk_scale, -m_scaled[:, None])
    else:
        qk = _fma_f32x2(qk, qk_scale, -m_ij[:, None])
    qks = _split_n(qk, NUM_MMA_SLICES)
    ps = ()
    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
        p_bufIdx = cid * NUM_MMA_SLICES + slice_id
        p_i = tl.math.exp2(qks[slice_id])
        tlx.local_store(p_tiles[p_bufIdx], p_i.to(out_dtype))
        tlx.barrier_arrive(p_fulls[slice_id + cid * NUM_MMA_SLICES])
        ps = ps + (p_i, )

    p = _join_n(ps)
    l_ij = tl.sum(p, 1)
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    accum_cnt_qk += 1
    return m_i, l_i, accum_cnt_qk


@triton.jit  # pragma: no cover
def _attn_fwd_ws(  # noqa: C901, TR001
    Q,
    Q_offsets,
    K,
    K_offsets,
    V,
    Out,
    M,
    sm_scale,
    stride_qh,
    stride_kh,
    stride_oh,
    stride_mh,
    Z,  # Batch size
    H,  # number of q heads.
    G,  # number of q head in each group. number of k v head will be H//G
    N_CTX,
    total_len_q,
    total_len_kv,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_D: tl.constexpr,  #
    USE_ON_DEVICE_TMA: tl.constexpr,
    NUM_BUFFERS_Q: tl.constexpr,
    NUM_BUFFERS_KV: tl.constexpr,
    NUM_BUFFERS_QK: tl.constexpr,
    NUM_MMA_GROUPS: tl.constexpr,
    NUM_MMA_SLICES: tl.constexpr,
    BROADCAST_Q: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
    ENABLE_LOAD_BALANCING: tl.constexpr,
    NUM_REGS_SFM: tl.constexpr,
    NUM_REGS_MMA: tl.constexpr,
    NUM_REGS_LOAD: tl.constexpr,
    NUM_REGS_EPILOG: tl.constexpr,
    valid_tiles_b,
    valid_tiles_m_start,
    valid_tiles_m_end,
    valid_tiles_h,
    sm_offsets,
    Z_order,
    USE_I64_IDX: tl.constexpr,
    USE_CLC: tl.constexpr,
    RESCALE_OPT: tl.constexpr = True,
):
    # Load balancing cannot work with CLC due to SM allocation
    LOAD_BALANCING_VALID: tl.constexpr = ENABLE_LOAD_BALANCING and not USE_CLC
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // 2

    n_tile_num = tl.cdiv(N_CTX, BLOCK_M)
    prog_id = tl.program_id(0)
    if USE_I64_IDX:
        prog_id = prog_id.to(tl.int64)
    num_progs = tl.num_programs(0)
    tile_idx = prog_id
    if LOAD_BALANCING_VALID:
        sm_start = tl.load(sm_offsets + prog_id)
        sm_end = tl.load(sm_offsets + prog_id + 1)

        tiles_per_sm = sm_end - sm_start
    else:
        total_tiles = n_tile_num * Z * H

        tiles_per_sm = total_tiles // num_progs
        if prog_id < total_tiles % num_progs:
            tiles_per_sm += 1

        # unused declarations to make compiler happy
        sm_start = -1
        sm_end = -1

    # on-device TMA
    if USE_ON_DEVICE_TMA:
        desc_q = tl.make_tensor_descriptor(
            Q,
            shape=[total_len_q, HEAD_DIM * H],
            strides=[HEAD_DIM * H, 1],
            block_shape=[BLOCK_M_SPLIT, BLOCK_D],
        )
        desc_k = tl.make_tensor_descriptor(
            K,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
        desc_v = tl.make_tensor_descriptor(
            V,
            shape=[total_len_kv, HEAD_DIM * H // G],
            strides=[HEAD_DIM * H // G, 1],
            block_shape=[BLOCK_N, BLOCK_D],
        )
    else:
        desc_q = Q
        desc_k = K
        desc_v = V

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_q), NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_tiles = tlx.local_alloc((BLOCK_N, BLOCK_D), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)
    o_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_D), tlx.dtype_of(desc_v), NUM_MMA_GROUPS)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    q_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_Q)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    o_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    o_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    # allocate TMEM buffers and barriers
    qk_p_l_m_storage_alias = tlx.storage_alias_spec(storage=tlx.storage_kind.tmem)
    qk_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
        reuse=qk_p_l_m_storage_alias,
    )
    # Shared buffer for QK, P and Alpha, l, and m.
    # A single QK buffer is split evenly:
    #   - First half  : stores P
    #   - Second half  : stores Alpha, l, and m
    #     QK : |                              BLK_M/2 * BLOCK_N * fp32                  |
    #     P:                                                |  BLK_M/2 * BLOCK_N * fp16 |
    #  Alpha : |BLK_M/2*1*fp32|
    #     l :                 |BLK_M/2*1*fp32|
    #     m :                                |BLK_M/2*1*fp32|
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N // NUM_MMA_SLICES),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_MMA_SLICES,
        tlx.storage_kind.tmem,
        reuse=qk_p_l_m_storage_alias,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
        reuse=qk_p_l_m_storage_alias,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
        reuse=qk_p_l_m_storage_alias,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        NUM_MMA_GROUPS,
        tlx.storage_kind.tmem,
        reuse=qk_p_l_m_storage_alias,
    )
    # QK shares each group with P's slices plus alpha/l/m packed after them.
    qk_p_l_m_storage_alias.set_buffer_overlap(
        tlx.reuse_group(
            qk_tiles,
            tlx.reuse_group(
                tlx.reuse_group(p_tiles, group_size=NUM_MMA_SLICES),
                alpha_tiles,
                l_tiles,
                m_tiles,
                group_type=tlx.reuse_group_type.distinct,
            ),
            group_type=tlx.reuse_group_type.shared,
        ))

    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_D), tl.float32, NUM_MMA_GROUPS, tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    qk_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_MMA_SLICES)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    if USE_CLC:
        clc_context = tlx.clc_create_context(4 + NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            accum_cnt = 0
            accum_cnt1 = 0
            accum_cnt_temp = 0
            accum_cnt_temp1 = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            phase = 0
            has_more_tile = True
            i = 0
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (_compute_seq_len(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    H,
                    N_CTX,
                    BROADCAST_Q,
                    LOAD_BALANCING_VALID,
                    BLOCK_M,
                    valid_tiles_b,
                    valid_tiles_m_start,
                    valid_tiles_m_end,
                    valid_tiles_h,
                    sm_offsets,
                    sm_start + i,
                    Z_order,
                ))
                out_offset = off_h.to(tl.int64) * stride_oh
                lo, hi = _kv_range(start_m, qlen, klen, BLOCK_M, BLOCK_N, WINDOW_LEFT, WINDOW_RIGHT)
                if start_m * BLOCK_M < qlen and lo < hi:
                    for start_n in tl.range(lo, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        _, phase = _get_bufidx_phase(accum_cnt, 1)
                        _, phase1 = _get_bufidx_phase(accum_cnt1, 1)
                        for cid in tl.static_range(0, NUM_MMA_GROUPS):
                            # Group 1 runs only on tiles with queries past the first BLOCK_M_SPLIT rows, so
                            # its barriers count their own phases.
                            if cid == 0:
                                group_phase = phase
                            else:
                                group_phase = phase1
                            if start_m * BLOCK_M + cid * BLOCK_M_SPLIT < qlen:
                                # -- update output accumulator --
                                tlx.barrier_wait(alpha_fulls[cid], group_phase)
                                # Use alpha[0] for cid=0, and alpha[HEAD_DIM] for cid=1
                                alpha_1 = tlx.local_load(alpha_tiles[cid])
                                tlx.barrier_arrive(alpha_empties[cid])
                                # Ballot skip: when alpha was forced to 1.0
                                # in the softmax warp (no row needs rescale), skip the
                                # entire TMEM load/mul/store loop.
                                if RESCALE_OPT:
                                    pred = alpha_1 < 1.0
                                    ballot_result = tlx.vote_ballot_sync(0xFFFFFFFF, pred)
                                    should_rescale = ballot_result != 0
                                    should_rescale_red = tl.reduce(should_rescale, axis=0, combine_fn=_reduce_or)
                                    should_rescale_scalar = tl.reshape(should_rescale_red, ())
                                if not RESCALE_OPT or should_rescale_scalar:
                                    for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                        subslice = tlx.subslice(
                                            acc_tiles[cid],
                                            BLOCK_D * slice_id // NUM_MMA_SLICES,
                                            BLOCK_D // NUM_MMA_SLICES,
                                        )
                                        acc = tlx.local_load(subslice)
                                        # acc = acc * alpha_1
                                        acc = _mul_f32x2(acc, alpha_1)
                                        tlx.local_store(subslice, acc)
                                tlx.barrier_arrive(acc_fulls[cid])
                        accum_cnt += 1
                        if start_m * BLOCK_M + BLOCK_M_SPLIT < qlen:
                            accum_cnt1 += 1

                    _, phase = _get_bufidx_phase(accum_cnt_temp, 1)
                    _, phase1 = _get_bufidx_phase(accum_cnt_temp1, 1)
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        if cid == 0:
                            group_phase = phase
                        else:
                            group_phase = phase1
                        if start_m * BLOCK_M + cid * BLOCK_M_SPLIT < qlen:
                            # epilogue
                            tlx.barrier_wait(l_fulls[cid], group_phase)
                            # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                            # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                            l_i_epilogue = tlx.local_load(l_tiles[cid])
                            m = tlx.local_load(m_tiles[cid])
                            tlx.barrier_arrive(qk_empties[cid])
                            # When RESCALE_OPT is on, m_tiles holds UNSCALED row-max.
                            # The bwd kernel reads logsumexp = (m * sm_scale * log2_e)
                            # + log2(l), so we scale here.
                            if RESCALE_OPT:
                                m = m * sm_scale * 1.44269504
                            m += tl.math.log2(l_i_epilogue)
                            offs_m = (start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT))
                            if BROADCAST_Q:
                                begin_o = qlen * off_z
                                m_ptrs = M + begin_o + offs_m + off_h * stride_mh
                            else:
                                m_ptrs = M + begin_q + offs_m + off_h * stride_mh
                            tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]), mask=offs_m < qlen)

                            tlx.barrier_wait(acc_empties[cid], group_phase)
                            tlx.barrier_wait(o_empties[cid], group_phase ^ 1)
                            scale = 1 / l_i_epilogue
                            if WINDOW_LEFT is not None or WINDOW_RIGHT is not None:
                                # A row whose band holds no key averaged only masked keys; attention over no keys is 0.
                                has_key = _row_has_key(offs_m, klen - qlen, klen, WINDOW_LEFT, WINDOW_RIGHT)
                                scale = tl.where(has_key[:, None], scale, 0.0)
                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                subslice = tlx.subslice(
                                    acc_tiles[cid],
                                    HEAD_DIM * slice_id // NUM_MMA_SLICES,
                                    HEAD_DIM // NUM_MMA_SLICES,
                                )
                                acc = tlx.local_load(subslice)
                                acc = _mul_f32x2(acc, scale)
                                acc = acc.to(tlx.dtype_of(Out))
                                subslice_o = tlx.local_slice(
                                    o_tiles[cid],
                                    [0, HEAD_DIM * slice_id // NUM_MMA_SLICES],
                                    [BLOCK_M_SPLIT, HEAD_DIM // NUM_MMA_SLICES],
                                )
                                tlx.local_store(subslice_o, acc)
                            tlx.barrier_arrive(o_fulls[cid])
                    accum_cnt_temp += 1
                    if start_m * BLOCK_M + BLOCK_M_SPLIT < qlen:
                        accum_cnt_temp1 += 1
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # softmax groups
        with tlx.async_task(num_warps=4, registers=NUM_REGS_SFM, replicate=NUM_MMA_GROUPS):
            accum_cnt_qk = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            phase = 0
            has_more_tile = True
            i = 0
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (_compute_seq_len(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    H,
                    N_CTX,
                    BROADCAST_Q,
                    LOAD_BALANCING_VALID,
                    BLOCK_M,
                    valid_tiles_b,
                    valid_tiles_m_start,
                    valid_tiles_m_end,
                    valid_tiles_h,
                    sm_offsets,
                    sm_start + i,
                    Z_order,
                ))
                # initialize pointer to m and l
                m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M_SPLIT, BLOCK_D], dtype=tl.float32)
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                cid = tlx.async_task_replica_id()
                lo, hi = _kv_range(start_m, qlen, klen, BLOCK_M, BLOCK_N, WINDOW_LEFT, WINDOW_RIGHT)
                if start_m * BLOCK_M + cid * BLOCK_M_SPLIT < qlen and lo < hi:
                    # Loop peeling: with no band, split the KV loop into a bulk
                    # pass with no klen mask and a 0-or-1 iteration tail with the
                    # mask. With a band, only the tiles crossing its edges (or
                    # klen) take the mask; the tiles between them run unmasked.
                    if WINDOW_LEFT is None and WINDOW_RIGHT is None:
                        hi_aligned = (klen // BLOCK_N) * BLOCK_N
                        for start_n in tl.range(lo, hi_aligned, BLOCK_N):
                            start_n = tl.multiple_of(start_n, BLOCK_N)
                            m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                cid,
                                accum_cnt_qk,
                                m_i,
                                l_i,
                                qk_scale,
                                klen,
                                start_n,
                                start_m,
                                klen - qlen,
                                qk_fulls,
                                qk_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                p_fulls,
                                p_tiles,
                                tlx.dtype_of(desc_v),
                                BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                BLOCK_M=BLOCK_M,
                                BLOCK_N=BLOCK_N,
                                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                NUM_MMA_SLICES=NUM_MMA_SLICES,
                                RESCALE_OPT=RESCALE_OPT,
                                WINDOW_LEFT=WINDOW_LEFT,
                                WINDOW_RIGHT=WINDOW_RIGHT,
                                APPLY_MASK=False,
                            )
                        for start_n in tl.range(hi_aligned, klen, BLOCK_N):
                            start_n = tl.multiple_of(start_n, BLOCK_N)
                            m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                cid,
                                accum_cnt_qk,
                                m_i,
                                l_i,
                                qk_scale,
                                klen,
                                start_n,
                                start_m,
                                klen - qlen,
                                qk_fulls,
                                qk_tiles,
                                alpha_empties,
                                alpha_fulls,
                                alpha_tiles,
                                p_fulls,
                                p_tiles,
                                tlx.dtype_of(desc_v),
                                BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                BLOCK_M=BLOCK_M,
                                BLOCK_N=BLOCK_N,
                                NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                NUM_MMA_SLICES=NUM_MMA_SLICES,
                                RESCALE_OPT=RESCALE_OPT,
                                WINDOW_LEFT=WINDOW_LEFT,
                                WINDOW_RIGHT=WINDOW_RIGHT,
                                APPLY_MASK=True,
                            )
                    else:
                        mid_lo, mid_hi = _unmasked_kv_range(
                            start_m * BLOCK_M + cid * BLOCK_M_SPLIT,
                            lo,
                            hi,
                            qlen,
                            klen,
                            BLOCK_M_SPLIT,
                            BLOCK_N,
                            WINDOW_LEFT,
                            WINDOW_RIGHT,
                        )
                        if WINDOW_LEFT is None:
                            for start_n in tl.range(mid_lo, mid_hi, BLOCK_N):
                                start_n = tl.multiple_of(start_n, BLOCK_N)
                                m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                    cid,
                                    accum_cnt_qk,
                                    m_i,
                                    l_i,
                                    qk_scale,
                                    klen,
                                    start_n,
                                    start_m,
                                    klen - qlen,
                                    qk_fulls,
                                    qk_tiles,
                                    alpha_empties,
                                    alpha_fulls,
                                    alpha_tiles,
                                    p_fulls,
                                    p_tiles,
                                    tlx.dtype_of(desc_v),
                                    BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                    BLOCK_M=BLOCK_M,
                                    BLOCK_N=BLOCK_N,
                                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                    NUM_MMA_SLICES=NUM_MMA_SLICES,
                                    RESCALE_OPT=RESCALE_OPT,
                                    WINDOW_LEFT=WINDOW_LEFT,
                                    WINDOW_RIGHT=WINDOW_RIGHT,
                                    APPLY_MASK=False,
                                    APPLY_BAND=False,
                                )
                            for start_n in tl.range(mid_hi, hi, BLOCK_N):
                                start_n = tl.multiple_of(start_n, BLOCK_N)
                                m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                    cid,
                                    accum_cnt_qk,
                                    m_i,
                                    l_i,
                                    qk_scale,
                                    klen,
                                    start_n,
                                    start_m,
                                    klen - qlen,
                                    qk_fulls,
                                    qk_tiles,
                                    alpha_empties,
                                    alpha_fulls,
                                    alpha_tiles,
                                    p_fulls,
                                    p_tiles,
                                    tlx.dtype_of(desc_v),
                                    BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                    BLOCK_M=BLOCK_M,
                                    BLOCK_N=BLOCK_N,
                                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                    NUM_MMA_SLICES=NUM_MMA_SLICES,
                                    RESCALE_OPT=RESCALE_OPT,
                                    WINDOW_LEFT=WINDOW_LEFT,
                                    WINDOW_RIGHT=WINDOW_RIGHT,
                                    APPLY_MASK=True,
                                )
                        else:
                            # Edge tiles take the mask through a runtime branch inside one loop: separate
                            # masked and unmasked loops cost more than the mask once the band is narrow.
                            for start_n in tl.range(lo, hi, BLOCK_N):
                                start_n = tl.multiple_of(start_n, BLOCK_N)
                                m_i, l_i, accum_cnt_qk = _softmax_inner_iter(
                                    cid,
                                    accum_cnt_qk,
                                    m_i,
                                    l_i,
                                    qk_scale,
                                    klen,
                                    start_n,
                                    start_m,
                                    klen - qlen,
                                    qk_fulls,
                                    qk_tiles,
                                    alpha_empties,
                                    alpha_fulls,
                                    alpha_tiles,
                                    p_fulls,
                                    p_tiles,
                                    tlx.dtype_of(desc_v),
                                    BLOCK_M_SPLIT=BLOCK_M_SPLIT,
                                    BLOCK_M=BLOCK_M,
                                    BLOCK_N=BLOCK_N,
                                    NUM_MMA_GROUPS=NUM_MMA_GROUPS,
                                    NUM_MMA_SLICES=NUM_MMA_SLICES,
                                    RESCALE_OPT=RESCALE_OPT,
                                    WINDOW_LEFT=WINDOW_LEFT,
                                    WINDOW_RIGHT=WINDOW_RIGHT,
                                    APPLY_MASK=True,
                                    mask_tile=(start_n < mid_lo) | (start_n >= mid_hi),
                                )

                    # prepare l_i for the epilog
                    # Use l[1]/l[1+HEAD_DIM] and m[2][2 + HEAD_DIM]
                    # to disambigulate from alpha[0]/alpha[HEAD_DIM]
                    tlx.local_store(l_tiles[cid], l_i[:, None])
                    tlx.local_store(m_tiles[cid], m_i[:, None])
                    tlx.barrier_arrive(l_fulls[cid])
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # mma group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_MMA):
            accum_cnt_kv = 0
            accum_cnt_qk = 0
            accum_cnt_qk1 = 0
            accum_cnt_q = 0
            accum_cnt_q1 = 0
            clc_phase_consumer = 0
            j = 0
            has_more_tile = True
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (_compute_seq_len(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    H,
                    N_CTX,
                    BROADCAST_Q,
                    LOAD_BALANCING_VALID,
                    BLOCK_M,
                    valid_tiles_b,
                    valid_tiles_m_start,
                    valid_tiles_m_end,
                    valid_tiles_h,
                    sm_offsets,
                    sm_start + j,
                    Z_order,
                ))
                lo, hi = _kv_range(start_m, qlen, klen, BLOCK_M, BLOCK_N, WINDOW_LEFT, WINDOW_RIGHT)
                if start_m * BLOCK_M < qlen and lo < hi:
                    q_bufIdx, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                    if start_m * BLOCK_M + BLOCK_M_SPLIT < qlen:
                        q1_bufIdx, q1_phase = _get_bufidx_phase(accum_cnt_q1, NUM_BUFFERS_Q)
                        q1_bufIdx += NUM_BUFFERS_Q
                        # Group 1 has run on fewer KV steps than group 0; its barrier phases differ by the
                        # parity of that gap, which is fixed within the tile.
                        g1_gap = accum_cnt_qk - accum_cnt_qk1
                        g1_parity = g1_gap & 1
                        # wait for the K buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)

                        # wait for the Q buffer to be populated by the producer
                        tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)

                        # -- compute q0 @ k ----
                        k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                        tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
                        tlx.async_dot(
                            q_tiles[0],
                            k_tile,
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0]],
                        )

                        # -- compute q1 @ k ----
                        tlx.barrier_wait(q_fulls[q1_bufIdx], q1_phase)
                        tlx.barrier_wait(qk_empties[1], q1_phase ^ 1)
                        tlx.async_dot(
                            q_tiles[1],
                            k_tile,
                            qk_tiles[1],
                            use_acc=False,
                            mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                        )

                        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                        # -- compute p0 @ v ----
                        # wait for the V buffer to be populated by the producer
                        tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                        tlx.barrier_wait(acc_fulls[0], qk_phase)
                        # Use p[slice_id] for cid=0, and p[NUM_MMA_SLICES + slice_id] for cid=1
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase)
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                            )
                            p_bufIdx = slice_id
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[0],
                                use_acc=slice_id > 0,
                                force_async=True,
                            )

                        acc1_init = False

                        for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                            start_n = tl.multiple_of(i, BLOCK_N)
                            v_bufIdx_prev = v_bufIdx
                            qk_phase_prev = qk_phase

                            accum_cnt_qk += 1
                            accum_cnt_kv += 2
                            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                            # -- compute q0 @ k ----
                            # wait for the K buffer to be populated by the producer
                            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                            k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                            _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)

                            tlx.async_dot(
                                q_tiles[0],
                                k_tile,
                                qk_tiles[0],
                                use_acc=False,
                                mBarriers=[qk_fulls[0]],
                            )

                            # -- compute p1 @ v from the previous iteration----
                            tlx.barrier_wait(acc_fulls[1], qk_phase_prev ^ g1_parity)
                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                tlx.barrier_wait(p_fulls[slice_id + 1 * NUM_MMA_SLICES], qk_phase_prev ^ g1_parity)
                                kv_slice = tlx.local_slice(
                                    kv_tiles[v_bufIdx_prev],
                                    [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                    [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                                )
                                p_bufIdx = NUM_MMA_SLICES + slice_id
                                use_acc = acc1_init if slice_id == 0 else True
                                mBarriers = ([kv_empties[v_bufIdx_prev]] if slice_id == NUM_MMA_SLICES - 1 else [])
                                tlx.async_dot(
                                    p_tiles[p_bufIdx],
                                    kv_slice,
                                    acc_tiles[1],
                                    use_acc=use_acc,
                                    mBarriers=mBarriers,
                                    force_async=True,
                                )

                            acc1_init = True

                            # -- compute q1 @ k ----
                            tlx.async_dot(
                                q_tiles[1],
                                k_tile,
                                qk_tiles[1],
                                use_acc=False,
                                mBarriers=[qk_fulls[1], kv_empties[k_bufIdx]],
                            )

                            # -- compute p0 @ v ----
                            # wait for the V buffer to be populated by the producer
                            tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)

                            tlx.barrier_wait(acc_fulls[0], qk_phase)
                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                tlx.barrier_wait(p_fulls[slice_id + 0 * NUM_MMA_SLICES], qk_phase)
                                # Use p[1] for cid=0, and p[3] for cid=1
                                kv_slice = tlx.local_slice(
                                    kv_tiles[v_bufIdx],
                                    [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                    [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                                )
                                p_bufIdx = slice_id
                                tlx.async_dot(
                                    p_tiles[p_bufIdx],
                                    kv_slice,
                                    acc_tiles[0],
                                    use_acc=True,
                                    force_async=True,
                                )

                        tlx.tcgen05_commit(q_empties[q_bufIdx])
                        tlx.tcgen05_commit(q_empties[q1_bufIdx])
                        tlx.tcgen05_commit(acc_empties[0])

                        # -- compute p1 @ v ----
                        tlx.barrier_wait(acc_fulls[1], qk_phase ^ g1_parity)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(p_fulls[slice_id + NUM_MMA_SLICES], qk_phase ^ g1_parity)
                            # Use p[1] for cid=0, and p[3] for cid=1
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                            )
                            p_bufIdx = NUM_MMA_SLICES + slice_id
                            use_acc = acc1_init if slice_id == 0 else True
                            mBarriers = ([acc_empties[1], kv_empties[v_bufIdx]] if slice_id == NUM_MMA_SLICES -
                                         1 else [])
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_slice,
                                acc_tiles[1],
                                use_acc=use_acc,
                                mBarriers=mBarriers,
                                force_async=True,
                            )

                        accum_cnt_qk1 = accum_cnt_qk + 1 - g1_gap
                        accum_cnt_q1 += 1
                    else:
                        # Every query of the tile falls in its first BLOCK_M_SPLIT rows: run group 0 only, and
                        # release each K/V buffer from group 0's MMAs.
                        tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                        tlx.barrier_wait(q_fulls[q_bufIdx], q_phase)
                        tlx.barrier_wait(qk_empties[0], q_phase ^ 1)
                        tlx.async_dot(
                            q_tiles[0],
                            tlx.local_trans(kv_tiles[k_bufIdx]),
                            qk_tiles[0],
                            use_acc=False,
                            mBarriers=[qk_fulls[0], kv_empties[k_bufIdx]],
                        )
                        _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                        tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                        tlx.barrier_wait(acc_fulls[0], qk_phase)
                        for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                            tlx.barrier_wait(p_fulls[slice_id], qk_phase)
                            kv_slice = tlx.local_slice(
                                kv_tiles[v_bufIdx],
                                [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                            )
                            tlx.async_dot(
                                p_tiles[slice_id],
                                kv_slice,
                                acc_tiles[0],
                                use_acc=slice_id > 0,
                                mBarriers=[kv_empties[v_bufIdx]] if slice_id == NUM_MMA_SLICES - 1 else [],
                                force_async=True,
                            )
                        for i in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                            accum_cnt_qk += 1
                            accum_cnt_kv += 2
                            k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                            v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                            tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                            tlx.async_dot(
                                q_tiles[0],
                                tlx.local_trans(kv_tiles[k_bufIdx]),
                                qk_tiles[0],
                                use_acc=False,
                                mBarriers=[qk_fulls[0], kv_empties[k_bufIdx]],
                            )
                            _, qk_phase = _get_bufidx_phase(accum_cnt_qk, 1)
                            tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                            tlx.barrier_wait(acc_fulls[0], qk_phase)
                            for slice_id in tl.static_range(0, NUM_MMA_SLICES):
                                tlx.barrier_wait(p_fulls[slice_id], qk_phase)
                                kv_slice = tlx.local_slice(
                                    kv_tiles[v_bufIdx],
                                    [BLOCK_N * slice_id // NUM_MMA_SLICES, 0],
                                    [BLOCK_N // NUM_MMA_SLICES, HEAD_DIM],
                                )
                                tlx.async_dot(
                                    p_tiles[slice_id],
                                    kv_slice,
                                    acc_tiles[0],
                                    use_acc=True,
                                    mBarriers=[kv_empties[v_bufIdx]] if slice_id == NUM_MMA_SLICES - 1 else [],
                                    force_async=True,
                                )
                        tlx.tcgen05_commit(q_empties[q_bufIdx])
                        tlx.tcgen05_commit(acc_empties[0])
                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
                    accum_cnt_q += 1
                j += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = j < tiles_per_sm

        # load
        with tlx.async_task(num_warps=1, registers=NUM_REGS_LOAD):
            accum_cnt_kv = 0
            accum_cnt_q = 0
            accum_cnt_q1 = 0
            clc_phase_consumer = 0
            i = 0
            has_more_tile = True
            while has_more_tile:
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (_compute_seq_len(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    H,
                    N_CTX,
                    BROADCAST_Q,
                    LOAD_BALANCING_VALID,
                    BLOCK_M,
                    valid_tiles_b,
                    valid_tiles_m_start,
                    valid_tiles_m_end,
                    valid_tiles_h,
                    sm_offsets,
                    sm_start + i,
                    Z_order,
                ))
                off_h_kv = off_h // G
                q_offset = off_h.to(tl.int64) * stride_qh
                kv_offset = off_h_kv.to(tl.int64) * stride_kh

                lo, hi = _kv_range(start_m, qlen, klen, BLOCK_M, BLOCK_N, WINDOW_LEFT, WINDOW_RIGHT)
                if start_m * BLOCK_M < qlen and lo < hi:
                    # load q0
                    q_bufIdx, q_phase = _get_bufidx_phase(accum_cnt_q, NUM_BUFFERS_Q)
                    tlx.barrier_wait(q_empties[q_bufIdx], q_phase ^ 1)
                    tlx.barrier_expect_bytes(q_fulls[q_bufIdx], 2 * BLOCK_M_SPLIT * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_q,
                        q_tiles[q_bufIdx],
                        [
                            (begin_q + start_m * BLOCK_M).to(tl.int32),
                            (q_offset).to(tl.int32),
                        ],
                        q_fulls[q_bufIdx],
                    )

                    # loop over loading k, v
                    k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    # wait for the K buffer to be released by the consumer
                    k_empty = tlx.local_view(kv_empties, k_bufIdx)
                    tlx.barrier_wait(k_empty, k_phase ^ 1)

                    # load K
                    k_full = tlx.local_view(kv_fulls, k_bufIdx)
                    k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                    tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D)  # float16
                    start_n = lo
                    tlx.async_descriptor_load(
                        desc_k,
                        k_tile,
                        [
                            (begin_k + start_n).to(tl.int32),
                            (kv_offset).to(tl.int32),
                        ],
                        k_full,
                    )

                    # load q1, unless every query of the tile falls in its first half
                    if start_m * BLOCK_M + BLOCK_M_SPLIT < qlen:
                        q1_bufIdx, q1_phase = _get_bufidx_phase(accum_cnt_q1, NUM_BUFFERS_Q)
                        q1_bufIdx += NUM_BUFFERS_Q
                        tlx.barrier_wait(q_empties[q1_bufIdx], q1_phase ^ 1)
                        tlx.barrier_expect_bytes(q_fulls[q1_bufIdx], 2 * BLOCK_M_SPLIT * BLOCK_D)  # float16
                        tlx.async_descriptor_load(
                            desc_q,
                            q_tiles[q1_bufIdx],
                            [
                                (begin_q + start_m * BLOCK_M + BLOCK_M_SPLIT).to(tl.int32),
                                (q_offset).to(tl.int32),
                            ],
                            q_fulls[q1_bufIdx],
                        )
                        accum_cnt_q1 += 1

                    v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                    # wait for the V buffer to be released by the consumer
                    v_empty = tlx.local_view(kv_empties, v_bufIdx)
                    tlx.barrier_wait(v_empty, v_phase ^ 1)
                    # load V
                    v_full = tlx.local_view(kv_fulls, v_bufIdx)
                    v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                    tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D)  # float16
                    tlx.async_descriptor_load(
                        desc_v,
                        v_tile,
                        [(begin_k + start_n).to(tl.int32), (kv_offset).to(tl.int32)],
                        v_full,
                    )

                    accum_cnt_kv += 2

                    for start_n in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                        start_n = tl.multiple_of(start_n, BLOCK_N)
                        k_bufIdx, k_phase = _get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                        # wait for the K buffer to be released by the consumer
                        k_empty = tlx.local_view(kv_empties, k_bufIdx)
                        tlx.barrier_wait(k_empty, k_phase ^ 1)
                        # load K
                        k_full = tlx.local_view(kv_fulls, k_bufIdx)
                        k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                        tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * BLOCK_D)  # float16
                        tlx.async_descriptor_load(
                            desc_k,
                            k_tile,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            k_full,
                        )

                        v_bufIdx, v_phase = _get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                        # wait for the V buffer to be released by the consumer
                        v_empty = tlx.local_view(kv_empties, v_bufIdx)
                        tlx.barrier_wait(v_empty, v_phase ^ 1)
                        # load V
                        v_full = tlx.local_view(kv_fulls, v_bufIdx)
                        v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                        tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * BLOCK_D)  # float16
                        tlx.async_descriptor_load(
                            desc_v,
                            v_tile,
                            [
                                (begin_k + start_n).to(tl.int32),
                                (kv_offset).to(tl.int32),
                            ],
                            v_full,
                        )

                        accum_cnt_kv += 2
                    accum_cnt_q += 1
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm

        # epilog group
        with tlx.async_task(num_warps=1, registers=NUM_REGS_EPILOG):
            accum_cnt = 0
            accum_cnt1 = 0
            clc_phase_consumer = 0
            clc_phase_producer = 1
            i = 0
            has_more_tile = True
            while has_more_tile:
                if USE_CLC:
                    tlx.clc_producer(clc_context, clc_phase_producer)
                    clc_phase_producer = clc_phase_producer ^ 1
                # initialize offsets
                begin_q, end_q, begin_k, end_k, qlen, klen, off_z, off_h, start_m = (_compute_seq_len(
                    tile_idx,
                    n_tile_num,
                    Q_offsets,
                    K_offsets,
                    H,
                    N_CTX,
                    BROADCAST_Q,
                    LOAD_BALANCING_VALID,
                    BLOCK_M,
                    valid_tiles_b,
                    valid_tiles_m_start,
                    valid_tiles_m_end,
                    valid_tiles_h,
                    sm_offsets,
                    sm_start + i,
                    Z_order,
                ))
                out_offset = off_h.to(tl.int64) * stride_oh
                if not BROADCAST_Q:
                    begin_o = begin_q
                    end_o = end_q
                else:
                    begin_o = qlen * off_z
                    end_o = qlen * (off_z + 1)
                _, phase = _get_bufidx_phase(accum_cnt, 1)
                _, phase1 = _get_bufidx_phase(accum_cnt1, 1)
                lo, hi = _kv_range(start_m, qlen, klen, BLOCK_M, BLOCK_N, WINDOW_LEFT, WINDOW_RIGHT)
                if start_m * BLOCK_M < qlen and lo < hi:
                    desc_o = tl.make_tensor_descriptor(
                        Out,
                        shape=[end_o.to(tl.int32), HEAD_DIM * H],
                        strides=[HEAD_DIM * H, 1],
                        block_shape=[BLOCK_M_SPLIT, BLOCK_D],
                    )
                    for cid in tl.static_range(0, NUM_MMA_GROUPS):
                        if cid == 0:
                            group_phase = phase
                        else:
                            group_phase = phase1
                        if start_m * BLOCK_M + cid * BLOCK_M_SPLIT < qlen:
                            tlx.barrier_wait(o_fulls[cid], group_phase)
                            tlx.fence_async_shared()
                            tlx.async_descriptor_store(
                                desc_o,
                                o_tiles[cid],
                                [
                                    (begin_o + start_m * BLOCK_M + cid * BLOCK_M_SPLIT).to(tl.int32),
                                    (out_offset).to(tl.int32),
                                ],
                            )
                            tlx.async_descriptor_store_wait(0)
                            tlx.barrier_arrive(o_empties[cid])
                    accum_cnt += 1
                    if start_m * BLOCK_M + BLOCK_M_SPLIT < qlen:
                        accum_cnt1 += 1
                elif start_m * BLOCK_M < qlen:
                    # No key in [lo, hi), so the other groups skipped the tile.
                    # Attention over no keys is 0: write its zero O and M rows,
                    # one row per iteration to keep this 1-warp group's registers.
                    offs_d = tl.arange(0, BLOCK_D)
                    zero_o = tl.zeros([BLOCK_D], Out.dtype.element_ty)
                    row_end = tl.minimum(qlen, (start_m + 1) * BLOCK_M)
                    for row in tl.range(start_m * BLOCK_M, row_end):
                        o_row = (begin_o + row).to(tl.int64) * (HEAD_DIM * H)
                        tl.store(
                            Out + o_row + out_offset + offs_d,
                            zero_o,
                            mask=offs_d < HEAD_DIM,
                        )
                    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    tl.store(
                        M + begin_o + offs_m + off_h * stride_mh,
                        tl.zeros([BLOCK_M], tl.float32),
                        mask=offs_m < qlen,
                    )
                i += 1
                if USE_CLC:
                    if USE_I64_IDX:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer).to(tl.int64)
                    else:
                        tile_idx = tlx.clc_consumer(clc_context, clc_phase_consumer)
                    clc_phase_consumer = clc_phase_consumer ^ 1
                    has_more_tile = tile_idx != -1
                else:
                    if not ENABLE_LOAD_BALANCING:
                        tile_idx += num_progs
                    has_more_tile = i < tiles_per_sm


@triton.jit  # pragma: no cover
def _attn_bwd_preprocess(  # noqa: TR001
    O,
    O_offsets,
    DO,  #
    Delta,  #
    stride_lh,
    H,
    N_CTX,  #
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    start_m = pid0 * BLOCK_M
    off_m = start_m + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1).to(tl.int64)
    off_d = tl.arange(0, BLOCK_D)
    off_h = off_hz % H
    off_z = off_hz // H
    begin_o = tl.load(O_offsets + off_z).to(tl.int64)
    end_o = tl.load(O_offsets + off_z + 1).to(tl.int64)
    olen = end_o - begin_o
    olen = tl.minimum(olen, N_CTX)
    if start_m > olen:
        return
    stride_om = HEAD_DIM * H
    o_offsets = (begin_o * stride_om + off_h * HEAD_DIM + off_m[:, None] * stride_om + off_d[None, :])
    o_mask = (off_m[:, None] < olen) & (off_d[None, :] < HEAD_DIM)
    # load
    o = tl.load(O + o_offsets, mask=o_mask, other=0.0)
    do = tl.load(DO + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_h * stride_lh + begin_o + off_m, delta, mask=off_m < olen)


def _band_bounds(causal, window_size):
    """flash-attn's ``window_size`` (-1 = unbounded) and ``causal`` as the kernels' (WINDOW_LEFT, WINDOW_RIGHT)."""
    left, right = window_size
    left = None if left < 0 else left
    right = None if right < 0 else right
    if causal:
        right = 0
    return left, right


def _alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _use_i64(*tensors):
    return any(t.numel() >= 2**31 for t in tensors)


@triton.jit
def _longest_first_kernel(cu_seqlens, order, B, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    valid = i < B
    lens = tl.load(cu_seqlens + i + 1, mask=valid, other=0) - tl.load(cu_seqlens + i, mask=valid, other=0)
    # Sort (length, reversed index) descending, so equal lengths keep their index order.
    key = tl.where(valid, (lens.to(tl.int64) << 32) | (BLOCK - 1 - i).to(tl.int64), -1)
    key = tl.sort(key, descending=True)
    tl.store(order + i, (BLOCK - 1 - (key & 0xFFFFFFFF)).to(tl.int32), mask=valid)


_LONGEST_FIRST_MAX_BATCH = 2048


def _longest_first(cu_seqlens_k, batch, tokens_k, max_seqlen_k):
    """Sequence order by descending key length, or None to keep index order.

    Ordering pays only when a few long sequences would otherwise run last; for similar lengths (and for large
    batches of fixed-size blocks) the extra launch costs more than it saves.
    """
    if batch > _LONGEST_FIRST_MAX_BATCH or max_seqlen_k * batch <= 2 * tokens_k:
        return None
    order = torch.empty(batch, device=cu_seqlens_k.device, dtype=torch.int32)
    _longest_first_kernel[(1, )](cu_seqlens_k, order, batch, BLOCK=max(triton.next_power_of_2(batch), 16), num_warps=4)
    return order


def _forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale, left, right):
    tokens, heads, head_dim = q.shape
    batch = cu_seqlens_q.numel() - 1
    out = torch.empty_like(q)
    lse = torch.empty((heads, tokens), device=q.device, dtype=torch.float32)
    if tokens == 0 or batch == 0:
        return out, lse
    # The grid covers max_seqlen_q queries per sequence; a longer sequence would silently lose its tail rows.
    # Checked on the device so the host does not sync.
    torch._assert_async((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max() <= max_seqlen_q,
                        "flash_attn_varlen: max_seqlen_q must bound every query length")
    if max_seqlen_q == 0:
        return out, lse
    if k.shape[0] == 0:
        return out.zero_(), lse.zero_()
    triton.set_allocator(_alloc_fn)
    z_order = _longest_first(cu_seqlens_k, batch, k.shape[0], max_seqlen_k)

    block_m = _FWD_CONFIG["BLOCK_M"]
    block_d = max(triton.next_power_of_2(head_dim), 16)
    grid = (triton.cdiv(max_seqlen_q, block_m) * batch * heads, )
    _attn_fwd_ws[grid](
        q,
        cu_seqlens_q,
        k,
        cu_seqlens_k,
        v,
        out,
        lse,
        sm_scale,
        q.stride(1),
        k.stride(1),
        out.stride(1),
        lse.stride(0),
        batch,
        heads,
        heads // k.shape[1],
        N_CTX=max_seqlen_q,
        total_len_q=tokens,
        total_len_kv=k.shape[0],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        USE_ON_DEVICE_TMA=True,
        BROADCAST_Q=False,
        WINDOW_LEFT=left,
        WINDOW_RIGHT=right,
        ENABLE_LOAD_BALANCING=False,
        valid_tiles_b=None,
        valid_tiles_m_start=None,
        valid_tiles_m_end=None,
        valid_tiles_h=None,
        sm_offsets=None,
        Z_order=z_order,
        USE_I64_IDX=_use_i64(q, k, v, out),
        **_FWD_CONFIG,
    )
    return out, lse


def _backward(do, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale, left, right):
    group = q.shape[1] // k.shape[1]
    if group > 1:
        # The kernel takes one query head per K/V head; differentiate through the repeated K/V and sum the group.
        k_rep, v_rep = (t.repeat_interleave(group, dim=1) for t in (k, v))
        dq, dk, dv = _backward(do, q, k_rep, v_rep, out, lse, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               sm_scale, left, right)
        tokens_k, kv_heads, head_dim = k.shape
        dk = dk.view(tokens_k, kv_heads, group, head_dim).sum(2)
        dv = dv.view(tokens_k, kv_heads, group, head_dim).sum(2)
        return dq, dk, dv
    tokens, heads, head_dim = q.shape
    batch = cu_seqlens_q.numel() - 1
    do = do.contiguous()
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    if tokens == 0 or batch == 0 or k.shape[0] == 0:
        return dq, dk.zero_(), dv.zero_()
    # The grid covers max_seqlen_k keys per sequence; a longer sequence would silently lose its dK/dV tail.
    # Checked on the device so the host does not sync.
    torch._assert_async((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max() <= max_seqlen_k,
                        "flash_attn_varlen: max_seqlen_k must bound every key length")
    block_d = max(triton.next_power_of_2(head_dim), 32)
    delta = torch.empty_like(lse)

    preproc_grid = (triton.cdiv(max_seqlen_q, _PREPROC_CONFIG["BLOCK_M"]), batch * heads)
    _attn_bwd_preprocess[preproc_grid](
        out,
        cu_seqlens_q,
        do,
        delta,
        lse.stride(0),
        heads,
        max_seqlen_q,
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
        **_PREPROC_CONFIG,
    )
    triton.set_allocator(_alloc_fn)
    grid = (triton.cdiv(max_seqlen_k, _BWD_CONFIG["BLOCK_N1"]) * batch * heads, )
    _attn_bwd_ws[grid](
        q,
        cu_seqlens_q,
        k,
        cu_seqlens_k,
        v,
        do,
        cu_seqlens_q,
        dq,
        dq,
        dk,
        dv,
        sm_scale,
        lse,
        delta,
        k.stride(0),
        q.stride(1),
        k.stride(1),
        lse.stride(0),
        q.stride(2),
        batch,
        heads,
        1,
        N_CTX_KV=max_seqlen_k,
        tile_to_batch=None,
        tile_to_head=None,
        tile_to_block=None,
        total_valid_tiles=0,
        total_len_q=tokens,
        total_len_kv=k.shape[0],
        BLOCK_D=block_d,
        HEAD_DIM=head_dim,
        BROADCAST_Q=False,
        WINDOW_LEFT=left,
        WINDOW_RIGHT=right,
        USE_I64_IDX=_use_i64(q, k, v, out),
        ENABLE_LB=False,
        USE_ON_DEVICE_TMA=True,
        **_BWD_CONFIG,
    )
    return dq, dk, dv


class _FlashAttnVarlen(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale, left, right):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        out, lse = _forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale, left, right)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.args = (max_seqlen_q, max_seqlen_k, sm_scale, left, right)
        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = _backward(do, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, *ctx.args)
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, sm_scale, causal, *, max_seqlen_k,
                      window_size=(-1, -1)):
    """Jagged attention with autograd; inputs are validated by ``tlx.ops.flash_attn_varlen``."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    left, right = _band_bounds(causal, window_size)
    if not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)):
        # No graph to record: skip autograd.Function's per-call dispatch.
        return _forward(q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q),
                        int(max_seqlen_k), float(sm_scale), left, right)[0]
    return _FlashAttnVarlen.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), int(max_seqlen_k),
                                  float(sm_scale), left, right)
