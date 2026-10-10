"""Shared Hopper cooperative GEMM pipeline for TLX ops and Inductor."""

from __future__ import annotations

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _allocate_buffers(a_desc, b_desc, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM_STAGES: tl.constexpr,
                      NUM_MMA_GROUPS: tl.constexpr, A_ROW_MAJOR: tl.constexpr, B_ROW_MAJOR: tl.constexpr):
    block_m_split: tl.constexpr = BM // NUM_MMA_GROUPS

    if A_ROW_MAJOR:
        a = tlx.local_alloc(
            (block_m_split, BK),
            tlx.dtype_of(a_desc),
            NUM_STAGES * NUM_MMA_GROUPS,
        )
    else:
        a = tlx.local_alloc(
            (BK, block_m_split),
            tlx.dtype_of(a_desc),
            NUM_STAGES * NUM_MMA_GROUPS,
        )
    if B_ROW_MAJOR:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)
    else:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_desc), NUM_STAGES)
    bars_empty_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    bars_full_a = tlx.alloc_barriers(
        num_barriers=NUM_STAGES * NUM_MMA_GROUPS,
        arrive_count=1,
    )
    bars_empty_b = tlx.alloc_barriers(
        num_barriers=NUM_STAGES,
        arrive_count=NUM_MMA_GROUPS,
    )
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    return a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b


@triton.jit
def _run_producer(M, N, K, a_desc, b_desc, a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b,
                  NUM_SMS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, NUM_STAGES: tl.constexpr, NUM_MMA_GROUPS: tl.constexpr,
                  A_ROW_MAJOR: tl.constexpr, B_ROW_MAJOR: tl.constexpr, A_EVICTION: tl.constexpr):
    block_m_split: tl.constexpr = BM // NUM_MMA_GROUPS
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    phase = 1
    buf = 0

    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m
        offset_am = pid_m * BM
        offset_bn = pid_n * BN

        for k in range(0, tl.cdiv(K, BK)):
            offset_k = k * BK

            empty_a_0 = tlx.local_view(bars_empty_a, buf)
            full_a_0 = tlx.local_view(bars_full_a, buf)
            # Triton TR051: bounded unrolling misses the matching release
            # after the nested persistent loop revisits this ring slot.
            tlx.barrier_wait(empty_a_0, phase)  # noqa: TR051
            tlx.barrier_expect_bytes(
                full_a_0,
                block_m_split * BK * tlx.size_of(tlx.dtype_of(a_desc)),
            )
            data_a_0 = tlx.local_view(a, buf)
            if A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_0,
                    [offset_am, offset_k],
                    full_a_0,
                    eviction_policy=A_EVICTION,
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_0,
                    [offset_k, offset_am],
                    full_a_0,
                    eviction_policy=A_EVICTION,
                )

            empty_b = tlx.local_view(bars_empty_b, buf)
            full_b = tlx.local_view(bars_full_b, buf)
            tlx.barrier_wait(empty_b, phase)
            tlx.barrier_expect_bytes(
                full_b,
                BN * BK * tlx.size_of(tlx.dtype_of(b_desc)),
            )
            data_b = tlx.local_view(b, buf)
            if B_ROW_MAJOR:
                tlx.async_descriptor_load(
                    b_desc,
                    data_b,
                    [offset_k, offset_bn],
                    full_b,
                )
            else:
                tlx.async_descriptor_load(
                    b_desc,
                    data_b,
                    [offset_bn, offset_k],
                    full_b,
                )

            a_1_index = buf + NUM_STAGES
            empty_a_1 = tlx.local_view(bars_empty_a, a_1_index)
            full_a_1 = tlx.local_view(bars_full_a, a_1_index)
            tlx.barrier_wait(empty_a_1, phase)
            tlx.barrier_expect_bytes(
                full_a_1,
                block_m_split * BK * tlx.size_of(tlx.dtype_of(a_desc)),
            )
            data_a_1 = tlx.local_view(a, a_1_index)
            if A_ROW_MAJOR:
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_1,
                    [offset_am + block_m_split, offset_k],
                    full_a_1,
                    eviction_policy=A_EVICTION,
                )
            else:
                tlx.async_descriptor_load(
                    a_desc,
                    data_a_1,
                    [offset_k, offset_am + block_m_split],
                    full_a_1,
                    eviction_policy=A_EVICTION,
                )

            phase = phase ^ (buf == NUM_STAGES - 1)
            buf = (buf + 1) % NUM_STAGES


@triton.jit
def _consume_tile(K, a, b, bars_empty_a, bars_full_a, bars_empty_b, bars_full_b, buf, phase, consumer_id: tl.constexpr,
                  BK: tl.constexpr, NUM_STAGES: tl.constexpr, A_ROW_MAJOR: tl.constexpr, B_ROW_MAJOR: tl.constexpr):
    last_buf = buf
    a_index = buf + NUM_STAGES * consumer_id
    full_a = tlx.local_view(bars_full_a, a_index)
    full_b = tlx.local_view(bars_full_b, buf)
    tlx.barrier_wait(full_a, phase)
    tlx.barrier_wait(full_b, phase)

    data_a = tlx.local_view(a, a_index)
    data_b = tlx.local_view(b, buf)
    a_operand = data_a if A_ROW_MAJOR else tlx.local_trans(data_a)
    b_operand = data_b if B_ROW_MAJOR else tlx.local_trans(data_b)
    acc = tlx.async_dot(a_operand, b_operand)

    phase = phase ^ (buf == NUM_STAGES - 1)
    buf = (buf + 1) % NUM_STAGES

    for _ in range(1, tl.cdiv(K, BK)):
        a_index = buf + NUM_STAGES * consumer_id
        full_a = tlx.local_view(bars_full_a, a_index)
        full_b = tlx.local_view(bars_full_b, buf)
        tlx.barrier_wait(full_a, phase)
        # Triton TR051: bounded unrolling misses the producer's TMA
        # completion after the shared B ring wraps.
        tlx.barrier_wait(full_b, phase)  # noqa: TR051

        data_a = tlx.local_view(a, a_index)
        data_b = tlx.local_view(b, buf)
        a_operand = data_a if A_ROW_MAJOR else tlx.local_trans(data_a)
        b_operand = data_b if B_ROW_MAJOR else tlx.local_trans(data_b)
        acc = tlx.async_dot(a_operand, b_operand, acc)
        acc = tlx.async_dot_wait(1, acc)

        empty_a = tlx.local_view(
            bars_empty_a,
            last_buf + NUM_STAGES * consumer_id,
        )
        empty_b = tlx.local_view(bars_empty_b, last_buf)
        tlx.barrier_arrive(empty_a)
        tlx.barrier_arrive(empty_b)

        last_buf = buf
        phase = phase ^ (buf == NUM_STAGES - 1)
        buf = (buf + 1) % NUM_STAGES

    acc = tlx.async_dot_wait(0, acc)
    empty_a = tlx.local_view(
        bars_empty_a,
        last_buf + NUM_STAGES * consumer_id,
    )
    empty_b = tlx.local_view(bars_empty_b, last_buf)
    tlx.barrier_arrive(empty_a)
    tlx.barrier_arrive(empty_b)

    return acc, buf, phase


@triton.jit
def _split_acc(acc, block_m_split: tl.constexpr, BN: tl.constexpr):
    acc = tl.reshape(acc, (block_m_split, 2, BN // 2))
    acc = tl.permute(acc, (0, 2, 1))
    acc_lo, acc_hi = tl.split(acc)
    acc_lo = tl.permute(tl.reshape(acc_lo, (block_m_split, 2, BN // 4)), (0, 2, 1))
    acc_0, acc_1 = tl.split(acc_lo)
    acc_hi = tl.permute(tl.reshape(acc_hi, (block_m_split, 2, BN // 4)), (0, 2, 1))
    acc_2, acc_3 = tl.split(acc_hi)
    return acc_0, acc_1, acc_2, acc_3


@triton.jit
def _store_subtile(c_desc, slot, acc, offset_m, offset_n):
    # A slot is reused every second store; waiting until at most one store is
    # pending guarantees the TMA engine has finished reading this slot.
    tlx.async_descriptor_store_wait(1)
    tlx.local_store(slot, acc.to(tlx.dtype_of(c_desc)))
    tlx.fence("async_shared")
    tlx.async_descriptor_store(c_desc, slot, [offset_m, offset_n])
