"""Parameterized Blackwell FMHA-forward skeleton (SKC Phase A).

Faithful extraction of third_party/tlx/tutorials/blackwell_fa_ws.py: same
five-role structure (TMA load / MMA issuer / softmax x k / correction), same
TMEM reuse layout and mbarrier protocol.  Everything the ScheduleBinder can
bind is a constexpr parameter; the synchronization topology is fixed — it is
the verified 651+ TFLOPS protocol, not a degree of freedom.

Roles and their solver-side fingerprints:
  load       1 warp,  REGS_LOAD  regs — tt.descriptor_load (TMA) only
  mma        1 warp,  REGS_MMA   regs — tcgen05 issue + barriers only (R1)
  softmax*k  4 warps, REGS_SOFTMAX regs each, k = NUM_MMA_GROUPS
  correction default task (leftover regs) — alpha rescale + epilogue
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

# Bindable parameter set and skeleton defaults (== the tuned tutorial config).
# MMA_PV_SKEW pipelines the mma issue loop: with skew 1 the issuer emits
# QK(i) before PV(i-1) each iteration (the solver's steady-state interleave),
# which requires NUM_BUFFERS_QK >= 2.  Skew 0 is the original tutorial loop.
DEFAULT_PARAMS = {
    "BLOCK_M": 256,
    "BLOCK_N": 128,
    "NUM_BUFFERS_KV": 3,
    "NUM_BUFFERS_QK": 1,
    "NUM_MMA_GROUPS": 2,
    "MMA_PV_SKEW": 0,
    "REGS_SOFTMAX": 152,
    "REGS_MMA": 24,
    "REGS_LOAD": 24,
}

# Per-CTA register budget check (R2): 64K 32-bit registers per SM, and the
# tutorial protocol runs one CTA per SM.  Correction (default task) receives
# the leftover budget; 4 warps * 24 regs is its floor.
_REG_FILE = 65536
_WARP = 32


def check_register_budget(p):
    softmax = p["NUM_MMA_GROUPS"] * 4 * _WARP * p["REGS_SOFTMAX"]
    mma = 1 * _WARP * p["REGS_MMA"]
    load = 1 * _WARP * p["REGS_LOAD"]
    correction_floor = 4 * _WARP * 24
    total = softmax + mma + load + correction_floor
    if total > _REG_FILE:
        raise ValueError(
            f"register quota table exceeds the {_REG_FILE}-reg file: "
            f"softmax {softmax} + mma {mma} + load {load} + "
            f"correction floor {correction_floor} = {total}")
    return total


@triton.jit
def _compute_offsets(H, N_CTX, BLOCK_M):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    lo, hi = 0, N_CTX
    kv_offset_y = offset_y + lo
    return start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y


@triton.jit
def _attn_fwd_skc(sm_scale, M,  #
                  Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                  HEAD_DIM: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  NUM_BUFFERS_KV: tl.constexpr,  #
                  NUM_BUFFERS_QK: tl.constexpr,  #
                  NUM_MMA_GROUPS: tl.constexpr,  #
                  MMA_PV_SKEW: tl.constexpr,  #
                  REGS_SOFTMAX: tl.constexpr,  #
                  REGS_MMA: tl.constexpr,  #
                  REGS_LOAD: tl.constexpr,  #
                  ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(MMA_PV_SKEW == 0 or MMA_PV_SKEW == 1)
    tl.static_assert(NUM_BUFFERS_QK > MMA_PV_SKEW,
                     "PV skew needs one extra QK buffer in flight")
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_M // NUM_MMA_GROUPS

    # allocate SMEM buffers and barriers
    q_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tlx.dtype_of(desc_q), NUM_MMA_GROUPS)
    kv_tiles = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(desc_k), NUM_BUFFERS_KV)

    q_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)
    kv_fulls = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)
    kv_empties = tlx.alloc_barriers(num_barriers=NUM_BUFFERS_KV)

    # allocate TMEM buffers and barriers.  A qk tile is (SPLIT, BLOCK_N) —
    # identical to the tutorial's (SPLIT, HEAD_DIM) when BLOCK_N == HEAD_DIM,
    # narrower under the solver's sub-tiled geometry.
    qk_tiles = tlx.local_alloc((BLOCK_M_SPLIT, BLOCK_N), tl.float32, NUM_MMA_GROUPS * NUM_BUFFERS_QK,
                               tlx.storage_kind.tmem)
    # Shared buffer for QK, P and Alpha, l, and m.
    # Alpha/l/m lives in the lower half of qk_buf, and P lives in the upper half.
    p_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, BLOCK_N),
        tlx.dtype_of(desc_v),
        NUM_MMA_GROUPS * NUM_BUFFERS_QK * 2,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    alpha_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    l_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )
    m_tiles = tlx.local_alloc(
        (BLOCK_M_SPLIT, 1),
        tl.float32,
        BLOCK_N * NUM_MMA_GROUPS * NUM_BUFFERS_QK,
        tlx.storage_kind.tmem,
        reuse=qk_tiles,
    )

    # One accumulator per softmax chain: PV dots accumulate in place across
    # iterations, so the acc ring never deepens with NUM_BUFFERS_QK (the
    # acc_fulls/empties barrier rings still cycle with the QK ring).
    acc_tiles = tlx.local_alloc((BLOCK_M_SPLIT, HEAD_DIM), tl.float32, NUM_MMA_GROUPS,
                                tlx.storage_kind.tmem)

    qk_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    p_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    acc_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)

    alpha_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    alpha_empties = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS * NUM_BUFFERS_QK)
    l_fulls = tlx.alloc_barriers(num_barriers=NUM_MMA_GROUPS)

    with tlx.async_tasks():
        # correction group
        with tlx.async_task("default"):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)
            accum_cnt = 0
            buf_idx = 0
            phase = 0

            for _ in tl.range(lo, hi, BLOCK_N):
                buf_idx, phase = get_bufidx_phase(accum_cnt, NUM_BUFFERS_QK)
                for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    buf_idx_2 = buf_idx + cid * NUM_BUFFERS_QK

                    # -- update output accumulator --
                    tlx.barrier_wait(alpha_fulls[buf_idx_2], phase)
                    # alpha for ring slot s lives at column-slot s*BLOCK_N of
                    # the reused qk region (for NUM_BUFFERS_QK=1 and BLOCK_N ==
                    # HEAD_DIM this is the tutorial's alpha[0]/alpha[HEAD_DIM]).
                    alpha_1 = tlx.local_load(alpha_tiles[buf_idx_2 * BLOCK_N])
                    tlx.barrier_arrive(alpha_empties[buf_idx_2])

                    if MMA_PV_SKEW == 1:
                        # With PV skew the issuer emits QK(i) before PV(i-1),
                        # so alpha_fulls(i) no longer implies PV(i-1) has
                        # landed in acc (the tutorial's implicit tensor-core
                        # ordering).  Wait for PV(i-1)'s completion signal
                        # before rescaling.
                        if accum_cnt > 0:
                            prev_idx, prev_phase = get_bufidx_phase(accum_cnt - 1, NUM_BUFFERS_QK)
                            tlx.barrier_wait(acc_empties[prev_idx + cid * NUM_BUFFERS_QK], prev_phase)

                    acc = tlx.local_load(acc_tiles[cid])
                    acc = acc * alpha_1
                    tlx.local_store(acc_tiles[cid], acc)
                    tlx.barrier_arrive(acc_fulls[buf_idx_2])
                accum_cnt += 1

            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                # epilogue
                tlx.barrier_wait(l_fulls[cid], 0)
                # l/m sit at columns 1/2 of the chain's first ring slot,
                # clear of that slot's alpha column (multiple of BLOCK_N).
                l = tlx.local_load(l_tiles[cid * BLOCK_N * NUM_BUFFERS_QK + 1])
                m = tlx.local_load(m_tiles[cid * BLOCK_N * NUM_BUFFERS_QK + 2])
                m += tl.math.log2(l)
                offs_m = start_m * BLOCK_M + cid * BLOCK_M_SPLIT + tl.arange(0, BLOCK_M_SPLIT)
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, tl.reshape(m, [BLOCK_M_SPLIT]))

                # Reuse the phase from the last iteration, i.e., accum_cnt - 1, so no need
                # to flip the phase.
                tlx.barrier_wait(acc_empties[buf_idx + cid * NUM_BUFFERS_QK], phase)
                acc = tlx.local_load(acc_tiles[cid])
                acc = acc / l
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                desc_o.store([qo_offset_y_split, 0], acc.to(tlx.dtype_of(desc_o)))

        # softmax groups
        with tlx.async_task(num_warps=4, registers=REGS_SOFTMAX, replicate=NUM_MMA_GROUPS):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)
            # initialize pointer to m and l
            m_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) - float("inf")
            l_i = tl.zeros([BLOCK_M_SPLIT], dtype=tl.float32) + 1.0
            acc = tl.zeros([BLOCK_M_SPLIT, HEAD_DIM], dtype=tl.float32)
            qk_scale = sm_scale
            qk_scale *= 1.44269504  # 1/log(2)

            accum_cnt_qk = 0
            cid = tlx.async_task_replica_id()
            for _ in tl.range(lo, hi, BLOCK_N):
                qk_bufIdx, qk_phase = get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                qk_bufIdx += cid * NUM_BUFFERS_QK

                tlx.barrier_wait(qk_fulls[qk_bufIdx], qk_phase)
                qk = tlx.local_load(qk_tiles[qk_bufIdx])

                # compute m_i, p in registers
                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)

                # -- compute correction factor
                alpha = tl.math.exp2(m_i - m_ij)
                tlx.barrier_wait(alpha_empties[qk_bufIdx], qk_phase ^ 1)
                # alpha for ring slot s -> column-slot s*BLOCK_N (tutorial
                # layout for depth 1 at BLOCK_N == HEAD_DIM).
                tlx.local_store(alpha_tiles[qk_bufIdx * BLOCK_N], alpha[:, None])
                tlx.barrier_arrive(alpha_fulls[qk_bufIdx])

                qk = qk * qk_scale - m_ij[:, None]
                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                p = p.to(tlx.dtype_of(desc_v))

                # prepare p for the v dot: p occupies the upper half of ring
                # slot s's qk region -> fp16 slot 2*s+1 ({1, 3} for depth 1).
                p_bufIdx = 2 * qk_bufIdx + 1
                tlx.local_store(p_tiles[p_bufIdx], p)
                tlx.barrier_arrive(p_fulls[qk_bufIdx])

                l_i = l_i * alpha + l_ij
                m_i = m_ij
                accum_cnt_qk += 1

            # prepare l_i for the epilog
            # l/m sit at columns 1/2 of the chain's first ring slot, clear
            # of that slot's alpha column (multiple of BLOCK_N).
            tlx.local_store(l_tiles[cid * BLOCK_N * NUM_BUFFERS_QK + 1], l_i[:, None])
            tlx.local_store(m_tiles[cid * BLOCK_N * NUM_BUFFERS_QK + 2], m_i[:, None])
            tlx.barrier_arrive(l_fulls[cid])

        # mma group
        with tlx.async_task(num_warps=1, registers=REGS_MMA):
            _, _, lo, hi, _, _ = _compute_offsets(H, N_CTX, BLOCK_M)

            # wait for the Q buffer to be populated by the producer
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                tlx.barrier_wait(q_fulls[cid], 0)

            if MMA_PV_SKEW == 0:
                # loop over k, v and update accumulator (tutorial issue order:
                # QK(i) then PV(i) each iteration)
                accum_cnt_kv = 0
                accum_cnt_qk = 0
                for i in tl.range(lo, hi, BLOCK_N):
                    k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                    v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)

                    # -- compute q @ k ----
                    # wait for the K buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    qk_bufIdx, qk_phase = get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                    for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                        qk_bufIdx_2 = qk_bufIdx + cid * NUM_BUFFERS_QK
                        if cid == NUM_MMA_GROUPS - 1:
                            tlx.async_dot(
                                q_tiles[cid],
                                k_tile,
                                qk_tiles[qk_bufIdx_2],
                                use_acc=False,
                                mBarriers=[qk_fulls[qk_bufIdx_2], kv_empties[k_bufIdx]],
                            )
                        else:
                            tlx.async_dot(
                                q_tiles[cid],
                                k_tile,
                                qk_tiles[qk_bufIdx_2],
                                use_acc=False,
                                mBarriers=[qk_fulls[qk_bufIdx_2]],
                            )

                    # -- compute p @ v ----
                    # wait for the V buffer to be populated by the producer
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                    for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                        qk_bufIdx_2 = qk_bufIdx + cid * NUM_BUFFERS_QK
                        tlx.barrier_wait(p_fulls[qk_bufIdx_2], qk_phase)
                        tlx.barrier_wait(acc_fulls[qk_bufIdx_2], qk_phase)
                        p_bufIdx = 2 * qk_bufIdx_2 + 1
                        if cid == NUM_MMA_GROUPS - 1:
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_tiles[v_bufIdx],
                                acc_tiles[cid],
                                use_acc=i > 0,
                                mBarriers=[acc_empties[qk_bufIdx_2], kv_empties[v_bufIdx]],
                            )
                        else:
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_tiles[v_bufIdx],
                                acc_tiles[cid],
                                use_acc=i > 0,
                                mBarriers=[acc_empties[qk_bufIdx_2]],
                            )

                    accum_cnt_qk += 1
                    accum_cnt_kv += 2
            else:
                # Solver steady-state issue order (one PV stage of skew):
                # each iteration issues QK(i) before PV(i-1), so the QK dot
                # for the next tile runs while softmax processes the current
                # one.  Ring math is identical; PV lags by one index.

                # prologue: QK(0)
                k_bufIdx, k_phase = get_bufidx_phase(0, NUM_BUFFERS_KV)
                tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    qk_bufIdx_2 = cid * NUM_BUFFERS_QK
                    if cid == NUM_MMA_GROUPS - 1:
                        tlx.async_dot(
                            q_tiles[cid],
                            k_tile,
                            qk_tiles[qk_bufIdx_2],
                            use_acc=False,
                            mBarriers=[qk_fulls[qk_bufIdx_2], kv_empties[k_bufIdx]],
                        )
                    else:
                        tlx.async_dot(
                            q_tiles[cid],
                            k_tile,
                            qk_tiles[qk_bufIdx_2],
                            use_acc=False,
                            mBarriers=[qk_fulls[qk_bufIdx_2]],
                        )

                # body: QK(i), PV(i-1)
                accum_cnt_qk = 1
                for _ in tl.range(lo + BLOCK_N, hi, BLOCK_N):
                    # -- compute q @ k for iteration accum_cnt_qk ----
                    k_bufIdx, k_phase = get_bufidx_phase(2 * accum_cnt_qk, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_fulls[k_bufIdx], k_phase)
                    k_tile = tlx.local_trans(kv_tiles[k_bufIdx])
                    qk_bufIdx, qk_phase = get_bufidx_phase(accum_cnt_qk, NUM_BUFFERS_QK)
                    for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                        qk_bufIdx_2 = qk_bufIdx + cid * NUM_BUFFERS_QK
                        if cid == NUM_MMA_GROUPS - 1:
                            tlx.async_dot(
                                q_tiles[cid],
                                k_tile,
                                qk_tiles[qk_bufIdx_2],
                                use_acc=False,
                                mBarriers=[qk_fulls[qk_bufIdx_2], kv_empties[k_bufIdx]],
                            )
                        else:
                            tlx.async_dot(
                                q_tiles[cid],
                                k_tile,
                                qk_tiles[qk_bufIdx_2],
                                use_acc=False,
                                mBarriers=[qk_fulls[qk_bufIdx_2]],
                            )

                    # -- compute p @ v for iteration accum_cnt_qk - 1 ----
                    pv_i = accum_cnt_qk - 1
                    v_bufIdx, v_phase = get_bufidx_phase(2 * pv_i + 1, NUM_BUFFERS_KV)
                    tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                    pv_bufIdx, pv_phase = get_bufidx_phase(pv_i, NUM_BUFFERS_QK)
                    for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                        qk_bufIdx_2 = pv_bufIdx + cid * NUM_BUFFERS_QK
                        tlx.barrier_wait(p_fulls[qk_bufIdx_2], pv_phase)
                        tlx.barrier_wait(acc_fulls[qk_bufIdx_2], pv_phase)
                        p_bufIdx = 2 * qk_bufIdx_2 + 1
                        if cid == NUM_MMA_GROUPS - 1:
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_tiles[v_bufIdx],
                                acc_tiles[cid],
                                use_acc=pv_i > 0,
                                mBarriers=[acc_empties[qk_bufIdx_2], kv_empties[v_bufIdx]],
                            )
                        else:
                            tlx.async_dot(
                                p_tiles[p_bufIdx],
                                kv_tiles[v_bufIdx],
                                acc_tiles[cid],
                                use_acc=pv_i > 0,
                                mBarriers=[acc_empties[qk_bufIdx_2]],
                            )
                    accum_cnt_qk += 1

                # epilogue: PV(last)
                pv_i = accum_cnt_qk - 1
                v_bufIdx, v_phase = get_bufidx_phase(2 * pv_i + 1, NUM_BUFFERS_KV)
                tlx.barrier_wait(kv_fulls[v_bufIdx], v_phase)
                pv_bufIdx, pv_phase = get_bufidx_phase(pv_i, NUM_BUFFERS_QK)
                for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                    qk_bufIdx_2 = pv_bufIdx + cid * NUM_BUFFERS_QK
                    tlx.barrier_wait(p_fulls[qk_bufIdx_2], pv_phase)
                    tlx.barrier_wait(acc_fulls[qk_bufIdx_2], pv_phase)
                    p_bufIdx = 2 * qk_bufIdx_2 + 1
                    if cid == NUM_MMA_GROUPS - 1:
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_tiles[v_bufIdx],
                            acc_tiles[cid],
                            use_acc=pv_i > 0,
                            mBarriers=[acc_empties[qk_bufIdx_2], kv_empties[v_bufIdx]],
                        )
                    else:
                        tlx.async_dot(
                            p_tiles[p_bufIdx],
                            kv_tiles[v_bufIdx],
                            acc_tiles[cid],
                            use_acc=pv_i > 0,
                            mBarriers=[acc_empties[qk_bufIdx_2]],
                        )

        # load
        with tlx.async_task(num_warps=1, registers=REGS_LOAD):
            # initialize offsets
            start_m, off_hz, lo, hi, qo_offset_y, kv_offset_y = _compute_offsets(H, N_CTX, BLOCK_M)

            # load q: it will stay in SRAM throughout
            for cid in tl.range(0, NUM_MMA_GROUPS, loop_unroll_factor=NUM_MMA_GROUPS):
                tlx.barrier_expect_bytes(q_fulls[cid], 2 * BLOCK_M_SPLIT * HEAD_DIM)  # float16
                qo_offset_y_split = qo_offset_y + cid * BLOCK_M_SPLIT
                tlx.async_descriptor_load(desc_q, q_tiles[cid], [qo_offset_y_split, 0], q_fulls[cid])

            # loop over loading k, v
            accum_cnt_kv = 0
            for _ in tl.range(lo, hi, BLOCK_N):
                k_bufIdx, k_phase = get_bufidx_phase(accum_cnt_kv, NUM_BUFFERS_KV)
                # wait for the K buffer to be released by the consumer
                k_empty = tlx.local_view(kv_empties, k_bufIdx)
                tlx.barrier_wait(k_empty, k_phase ^ 1)
                # load K
                k_full = tlx.local_view(kv_fulls, k_bufIdx)
                k_tile = tlx.local_view(kv_tiles, k_bufIdx)
                tlx.barrier_expect_bytes(k_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_k, k_tile, [kv_offset_y, 0], k_full)

                v_bufIdx, v_phase = get_bufidx_phase(accum_cnt_kv + 1, NUM_BUFFERS_KV)
                # wait for the V buffer to be released by the consumer
                v_empty = tlx.local_view(kv_empties, v_bufIdx)
                tlx.barrier_wait(v_empty, v_phase ^ 1)
                # load V
                v_full = tlx.local_view(kv_fulls, v_bufIdx)
                v_tile = tlx.local_view(kv_tiles, v_bufIdx)
                tlx.barrier_expect_bytes(v_full, 2 * BLOCK_N * HEAD_DIM)  # float16
                tlx.async_descriptor_load(desc_v, v_tile, [kv_offset_y, 0], v_full)

                kv_offset_y += BLOCK_N
                accum_cnt_kv += 2


def attention(q, k, v, sm_scale, params=None):
    """Launch the skeleton with an explicit parameter binding (no autotuner)."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    check_register_budget(p)

    HEAD_DIM_K = q.shape[-1]
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    o = torch.empty_like(q)
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

    split = p["BLOCK_M"] // p["NUM_MMA_GROUPS"]
    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[split, HEAD_DIM_K])
    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                              block_shape=[p["BLOCK_N"], HEAD_DIM_K])
    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                              block_shape=[p["BLOCK_N"], HEAD_DIM_K])
    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=[split, HEAD_DIM_K])

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    grid = (triton.cdiv(q.shape[2], p["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
    _attn_fwd_skc[grid](
        sm_scale,
        M,
        q.shape[0],
        q.shape[1],
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=p["BLOCK_M"],
        BLOCK_N=p["BLOCK_N"],
        NUM_BUFFERS_KV=p["NUM_BUFFERS_KV"],
        NUM_BUFFERS_QK=p["NUM_BUFFERS_QK"],
        NUM_MMA_GROUPS=p["NUM_MMA_GROUPS"],
        MMA_PV_SKEW=p["MMA_PV_SKEW"],
        REGS_SOFTMAX=p["REGS_SOFTMAX"],
        REGS_MMA=p["REGS_MMA"],
        REGS_LOAD=p["REGS_LOAD"],
        num_warps=4,
        num_stages=1,
    )
    return o, M
