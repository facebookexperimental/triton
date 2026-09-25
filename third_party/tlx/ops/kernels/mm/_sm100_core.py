"""Shared Blackwell GEMM JIT core for TLX.ops and TorchTLX.

The direct ``sm100`` kernel and the Inductor template intentionally keep their
own launch and epilogue shells.  This module owns the common tile mapping,
MMA, and TMA producer implementation so those two entry points cannot drift.
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS: tl.constexpr,
    CAST_TO_INT32: tl.constexpr = False,
):
    """Compute the persistent grid shared by all three async tasks."""
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if CAST_TO_INT32:
        num_pid_m = num_pid_m.to(tl.int32)
        num_pid_n = num_pid_n.to(tl.int32)
    # Pad M tiles to the cluster width so CTA pairs stay together.
    num_pid_m = (num_pid_m + NUM_CTAS - 1) // NUM_CTAS * NUM_CTAS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    if CAST_TO_INT32:
        k_tiles_total = k_tiles_total.to(tl.int32)
    return start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_mn_tiles, num_tiles, k_tiles_total


@triton.jit
def _compute_k_tile_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K: tl.constexpr):
    if SPLIT_K == 1:
        return 0, k_tiles_total
    split_id = tile_id // num_mn_tiles
    k_tiles_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
    k_tile_start = split_id * k_tiles_per_split
    k_tile_end = min(k_tile_start + k_tiles_per_split, k_tiles_total)
    return k_tile_start, k_tile_end


@triton.jit
def _process_tile_mma_inner(
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    cur_tmem_buf,
    tmem_empty_bars,
    tmem_write_phase,
    smem_accum_cnt,
    NUM_CTAS,
    cta_bars=None,
    pred_cta0=None,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    EXPLICIT_CTA_SYNC: tl.constexpr = False,
):
    """Run one tile's MMA pipeline for either persistent scheduling shell.

    The TLX.ops CLC kernel uses collaborative TMA plus ``tcgen05_commit``.
    TorchTLX's static persistent shell retains its explicit CTA rendezvous and
    completion arrivals.  ``EXPLICIT_CTA_SYNC`` specializes those protocol
    differences away while sharing the actual tile pipeline.
    """
    local_k_tiles = k_tile_end - k_tile_start
    buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    if not EXPLICIT_CTA_SYNC and NUM_MMA_GROUPS == 1:
        tlx.barrier_wait(tmem_empty_bars[cur_tmem_buf], tmem_write_phase ^ 1)

    # The CLC single-group path aliases B_smem_full_bars to A_smem_full_bars.
    # The static template keeps them separate and therefore waits on both.
    tlx.barrier_wait(B_smem_full_bars[buf], phase)

    for group_id in tl.static_range(NUM_MMA_GROUPS):
        a_buf = group_id * NUM_SMEM_BUFFERS + buf
        acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

        if EXPLICIT_CTA_SYNC or NUM_MMA_GROUPS > 1:
            tlx.barrier_wait(A_smem_full_bars[a_buf], phase)
            cur_barrier_idx = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_wait(tmem_empty_bars[cur_barrier_idx], tmem_write_phase ^ 1)

        if EXPLICIT_CTA_SYNC and NUM_CTAS == 2:
            tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
            tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

        a_operand = tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
        b_operand = tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
        tlx.async_dot(
            a_operand,
            b_operand,
            tmem_buffers[acc_buf],
            use_acc=False,
            mBarriers=[A_smem_empty_bars[a_buf]],
            two_ctas=NUM_CTAS == 2,
            out_dtype=tl.float32,
        )

    smem_accum_cnt += 1

    for _ in range(1, local_k_tiles):
        if EXPLICIT_CTA_SYNC:
            buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)
        else:
            # Avoid a non-power-of-two divide/modulo in the CLC hot loop.
            buf += 1
            if buf == NUM_SMEM_BUFFERS:
                buf = 0
                phase ^= 1

        tlx.barrier_wait(B_smem_full_bars[buf], phase)

        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + buf
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf

            if EXPLICIT_CTA_SYNC or NUM_MMA_GROUPS > 1:
                tlx.barrier_wait(A_smem_full_bars[a_buf], phase)

            if EXPLICIT_CTA_SYNC and NUM_CTAS == 2:
                tlx.barrier_arrive(cta_bars[a_buf], arrive_count=1, remote_cta_rank=0)
                tlx.barrier_wait(cta_bars[a_buf], phase=phase, pred=pred_cta0)

            a_operand = tlx.local_trans(buffers_A[a_buf]) if not A_ROW_MAJOR else buffers_A[a_buf]
            b_operand = tlx.local_trans(buffers_B[buf]) if not B_ROW_MAJOR else buffers_B[buf]
            tlx.async_dot(
                a_operand,
                b_operand,
                tmem_buffers[acc_buf],
                use_acc=True,
                mBarriers=[A_smem_empty_bars[a_buf]],
                two_ctas=NUM_CTAS == 2,
                out_dtype=tl.float32,
            )

        smem_accum_cnt += 1

    if EXPLICIT_CTA_SYNC:
        last_buf, last_phase = get_bufidx_phase(smem_accum_cnt - 1, NUM_SMEM_BUFFERS)
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            a_buf = group_id * NUM_SMEM_BUFFERS + last_buf
            tlx.barrier_wait(A_smem_empty_bars[a_buf], last_phase)
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.barrier_arrive(tmem_full_bars[acc_buf], 1)
    else:
        # Track completion asynchronously so the MMA task can begin its next
        # tile while this tile's final MMAs drain.
        for group_id in tl.static_range(NUM_MMA_GROUPS):
            acc_buf = group_id * NUM_TMEM_BUFFERS + cur_tmem_buf
            tlx.tcgen05_commit(tmem_full_bars[acc_buf], two_ctas=NUM_CTAS == 2)

    return smem_accum_cnt


@triton.jit
def _expect_and_load(
    desc,
    dst,
    offset0,
    offset1,
    full_bar,
    expected_bytes,
    is_leader,
    MULTICAST_TMA: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    EXPECT_BYTES: tl.constexpr = True,
):
    if EXPECT_BYTES:
        if MULTICAST_TMA:
            tlx.barrier_expect_bytes(full_bar, expected_bytes, pred=is_leader)
        else:
            tlx.barrier_expect_bytes(full_bar, expected_bytes)
    if TRANSPOSED:
        tlx.async_descriptor_load(
            desc,
            dst,
            [offset1, offset0],
            full_bar,
            eviction_policy="evict_last",
            two_ctas=MULTICAST_TMA,
        )
    else:
        tlx.async_descriptor_load(
            desc,
            dst,
            [offset0, offset1],
            full_bar,
            eviction_policy="evict_last",
            two_ctas=MULTICAST_TMA,
        )


@triton.jit
def _process_tile_producer_inner(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    num_mn_tiles,
    GROUP_SIZE_M,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    NUM_MMA_GROUPS,
    k_tile_start,
    k_tile_end,
    NUM_SMEM_BUFFERS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    smem_accum_cnt,
    NUM_CTAS,
    cluster_cta_rank,
    SPLIT_K: tl.constexpr,
    A_ROW_MAJOR: tl.constexpr = True,
    B_ROW_MAJOR: tl.constexpr = True,
    MULTICAST_TMA: tl.constexpr = True,
):
    """Load one tile's operands for either persistent scheduling shell."""
    mn_tile_id = tile_id if SPLIT_K == 1 else tile_id % num_mn_tiles
    pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    dsize: tl.constexpr = tlx.size_of(tlx.dtype_of(b_desc))
    BLOCK_M_SPLIT: tl.constexpr = BLOCK_SIZE_M // NUM_MMA_GROUPS
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // NUM_CTAS)
    b_expected_bytes: tl.constexpr = dsize * BLOCK_SIZE_N * BLOCK_SIZE_K // NUM_CTAS
    a_expected_bytes: tl.constexpr = dsize * BLOCK_M_SPLIT * BLOCK_SIZE_K
    use_multicast: tl.constexpr = MULTICAST_TMA and NUM_CTAS == 2
    is_leader = cluster_cta_rank == 0

    local_k_tiles = k_tile_end - k_tile_start
    buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    for k_idx in range(0, local_k_tiles):
        k = k_tile_start + k_idx
        offs_k = k * BLOCK_SIZE_K
        offs_am = pid_m * BLOCK_SIZE_M

        if NUM_MMA_GROUPS == 1 and MULTICAST_TMA:
            tlx.barrier_wait(A_smem_empty_bars[buf], phase ^ 1)
            combined_bytes: tl.constexpr = b_expected_bytes + a_expected_bytes
            _expect_and_load(
                b_desc,
                buffers_B[buf],
                offs_k,
                offs_bn,
                A_smem_full_bars[buf],
                combined_bytes * NUM_CTAS if use_multicast else combined_bytes,
                is_leader,
                use_multicast,
                not B_ROW_MAJOR,
            )
            _expect_and_load(
                a_desc,
                buffers_A[buf],
                offs_am,
                offs_k,
                A_smem_full_bars[buf],
                combined_bytes * NUM_CTAS if use_multicast else combined_bytes,
                is_leader,
                use_multicast,
                not A_ROW_MAJOR,
                EXPECT_BYTES=False,
            )
        else:
            a0_buf = buf
            tlx.barrier_wait(A_smem_empty_bars[a0_buf], phase ^ 1)
            _expect_and_load(
                a_desc,
                buffers_A[a0_buf],
                offs_am,
                offs_k,
                A_smem_full_bars[a0_buf],
                a_expected_bytes * NUM_CTAS if use_multicast else a_expected_bytes,
                is_leader,
                use_multicast,
                not A_ROW_MAJOR,
            )

            last_a_buf = (NUM_MMA_GROUPS - 1) * NUM_SMEM_BUFFERS + buf
            tlx.barrier_wait(A_smem_empty_bars[last_a_buf], phase ^ 1)
            _expect_and_load(
                b_desc,
                buffers_B[buf],
                offs_k,
                offs_bn,
                B_smem_full_bars[buf],
                b_expected_bytes * NUM_CTAS if use_multicast else b_expected_bytes,
                is_leader,
                use_multicast,
                not B_ROW_MAJOR,
            )

            for group_id in tl.static_range(1, NUM_MMA_GROUPS):
                a_buf = group_id * NUM_SMEM_BUFFERS + buf
                if not MULTICAST_TMA or group_id != NUM_MMA_GROUPS - 1:
                    tlx.barrier_wait(A_smem_empty_bars[a_buf], phase ^ 1)
                offs_am_group = offs_am + group_id * BLOCK_M_SPLIT
                _expect_and_load(
                    a_desc,
                    buffers_A[a_buf],
                    offs_am_group,
                    offs_k,
                    A_smem_full_bars[a_buf],
                    a_expected_bytes * NUM_CTAS if use_multicast else a_expected_bytes,
                    is_leader,
                    use_multicast,
                    not A_ROW_MAJOR,
                )

        smem_accum_cnt += 1
        if MULTICAST_TMA:
            # The direct kernel keeps this counter incrementally to avoid
            # non-power-of-two division in the hot loop.
            buf += 1
            if buf == NUM_SMEM_BUFFERS:
                buf = 0
                phase ^= 1
        else:
            buf, phase = get_bufidx_phase(smem_accum_cnt, NUM_SMEM_BUFFERS)

    return smem_accum_cnt


@triton.jit
def _run_static_mma_task(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS,
    NUM_SMS,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    NUM_TMEM_BUFFERS,
    buffers_A,
    buffers_B,
    tmem_buffers,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    tmem_full_bars,
    tmem_empty_bars,
    cta_bars,
    pred_cta0,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    (
        start_pid,
        _,
        _,
        _,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    ) = _compute_grid_info(
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        SPLIT_K,
        NUM_CTAS,
        CAST_TO_INT32=True,
    )

    tmem_accum_cnt = 0
    smem_accum_cnt = 0
    tile_id = start_pid
    while tile_id < num_tiles:
        k_tile_start, k_tile_end = _compute_k_tile_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
        if SPLIT_K == 1 or k_tile_end > k_tile_start:
            cur_tmem_buf, tmem_write_phase = get_bufidx_phase(tmem_accum_cnt, NUM_TMEM_BUFFERS)
            smem_accum_cnt = _process_tile_mma_inner(
                k_tile_start,
                k_tile_end,
                NUM_SMEM_BUFFERS,
                NUM_MMA_GROUPS,
                NUM_TMEM_BUFFERS,
                buffers_A,
                buffers_B,
                tmem_buffers,
                A_smem_full_bars,
                B_smem_full_bars,
                A_smem_empty_bars,
                tmem_full_bars,
                cur_tmem_buf,
                tmem_empty_bars,
                tmem_write_phase,
                smem_accum_cnt,
                NUM_CTAS,
                cta_bars,
                pred_cta0,
                A_ROW_MAJOR,
                B_ROW_MAJOR,
                EXPLICIT_CTA_SYNC=True,
            )
            tmem_accum_cnt += 1
        tile_id += NUM_SMS


@triton.jit
def _run_static_producer_task(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    SPLIT_K,
    NUM_CTAS,
    NUM_SMS,
    NUM_SMEM_BUFFERS,
    NUM_MMA_GROUPS,
    a_desc,
    b_desc,
    buffers_A,
    buffers_B,
    A_smem_full_bars,
    B_smem_full_bars,
    A_smem_empty_bars,
    cluster_cta_rank,
    A_ROW_MAJOR: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    (
        start_pid,
        num_pid_m,
        _,
        num_pid_in_group,
        num_mn_tiles,
        num_tiles,
        k_tiles_total,
    ) = _compute_grid_info(
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        SPLIT_K,
        NUM_CTAS,
        CAST_TO_INT32=True,
    )

    smem_accum_cnt = 0
    tile_id = start_pid
    while tile_id < num_tiles:
        k_tile_start, k_tile_end = _compute_k_tile_range(tile_id, num_mn_tiles, k_tiles_total, SPLIT_K)
        if SPLIT_K == 1 or k_tile_end > k_tile_start:
            smem_accum_cnt = _process_tile_producer_inner(
                tile_id,
                num_pid_in_group,
                num_pid_m,
                num_mn_tiles,
                GROUP_SIZE_M,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                NUM_MMA_GROUPS,
                k_tile_start,
                k_tile_end,
                NUM_SMEM_BUFFERS,
                a_desc,
                b_desc,
                buffers_A,
                buffers_B,
                A_smem_full_bars,
                B_smem_full_bars,
                A_smem_empty_bars,
                smem_accum_cnt,
                NUM_CTAS,
                cluster_cta_rank,
                SPLIT_K,
                A_ROW_MAJOR,
                B_ROW_MAJOR,
                MULTICAST_TMA=False,
            )
        tile_id += NUM_SMS
