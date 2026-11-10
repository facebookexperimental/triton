"""Hopper (sm90) ``tlx.ops.addmm`` implementation.

Fused ``input + a @ b`` via TMA descriptors and TLX warp specialization. The
kernel is persistent over the M/N tile grid with two roles -- a default
producer task that TMA-loads A/B tiles plus two replicated 4-warp consumers
running wgmma -- taken from the single-CTA path of
``third_party/tlx/tutorials/hopper_gemm_ws.py``. The epilogue folds the bias
in with one masked ``tl.load`` per tile, so every broadcastable bias shape
rides the WS path with no ``BROADCAST_Y`` specialization (the idiom from
``third_party/tlx/tutorials/amd_addmm_gfx942.py``: a 1-D bias arrives with a
zero row stride, which broadcasts it down the rows for free).

``NUM_CTAS`` selects one CTA per cluster or the tutorial's 2-CTA multicast
path, where each CTA TMA-loads half of B and multicasts it to its partner.
Column-major B runs through a transposed descriptor with an SMEM transpose
before the MMA. With ``EPILOGUE_PARTITION`` the TMA store moves to its own
2-replica task: each wgmma partition hands its C half-tile to a private
SMEM ring, and each epilogue replica cycles wait/load/store/arrive over
its ring, overlapping the store of tile t with the MMA of tile t+1.
Inputs that cannot use TMA (misaligned bases, A without a unit inner
stride, B without a unit stride in either dimension, or K/N not multiples
of 16) fall back to the direct-load baseline kernel, shared with the sm100
implementation.
"""

import functools
import math

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.language.extra.tlx.warp_spec import get_bufidx_phase
from triton.tools.tensor_descriptor import TensorDescriptor

from ..mm import sm90 as _mm

# Handoff ring depth per M-half group for the partitioned epilogue. Depth
# 2 measures the same as depth 1 (the store keeps up), so this stays 1.


@triton.jit
def addmm_kernel_tma_ws(
    a_desc,
    b_desc,
    c_desc,
    y_ptr,
    M,
    N,
    K,
    stride_ym,
    stride_yn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    EPILOGUE_PARTITION: tl.constexpr,
    USE_PINGPONG: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    """Persistent 2-role WS addmm: TMA producer + replicated wgmma consumers.

    Each CTA walks the M/N tile grid with stride ``NUM_SMS``. The producer and
    the MMA loop match the Hopper GEMM tutorial exactly; the two consumer
    replicas each own half the M-tile (``BM // 2`` rows) and, once their
    accumulator is drained, add their bias tile and TMA-store the result.
    OOB M/N elements clip in the TMA store; the bias load is explicitly
    masked. With ``EPILOGUE_SUBTILE`` the store (and the accumulator it is
    sliced from) halves along N. With ``NUM_CTAS == 2`` each CTA loads its
    N-half of B and multicasts it to its cluster partner, so both CTAs
    observe the full B tile. Partners loop over the pair's max tile count
    so the per-k sync never strands: the shorter partner ghosts (sync
    only) while the longer loads both B halves alone on its extra tile.
    Pairs split across N-tiles (odd M-groups pair consecutive tiles from
    different N-tiles) also load both halves without multicasting, since
    the partners need different B tiles. Column-major B arrives through a
    transposed descriptor and is transposed back in SMEM before the MMA.
    With ``EPILOGUE_PARTITION`` the consumers hand their converted C
    half-tiles to private SMEM rings instead of TMA-storing, and a third,
    2-replica task drains the rings (each replica owns one M-half's ring).
    With ``USE_PINGPONG`` the two consumers alternate wgmma issue via
    named barriers 9/10 (consumer 1 primes 9 once up front).
    """
    BLOCK_M_SPLIT: tl.constexpr = BM // 2
    # Consumer counts: fused matches both groups at the pool max
    # ((65536 - producer 3072) / 256 threads, down to %8 = 240).
    # Partitioned keeps consumer 1 at 184 with consumer 0 auto: the
    # backend splits a fixed 392/pair budget complementarily, so
    # symmetric caps flatten to static-96 (-18%), while 208/184
    # stays dynamic.
    _CONS_REGS_C0: tl.constexpr = None if EPILOGUE_PARTITION else 240
    _CONS_REGS_C1: tl.constexpr = 184 if EPILOGUE_PARTITION else 240
    # Epilogue ring depth, kept kernel-local (not a module global) so the
    # kernel stays eligible for the C fast dispatch path: any global it
    # reads bypasses the C cache and costs ~50us of host overhead/launch.
    EPI_STAGES: tl.constexpr = 1

    a = tlx.local_alloc((BLOCK_M_SPLIT, BK), tlx.dtype_of(a_desc), NUM_STAGES * 2)
    if not B_ROW_MAJOR:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_desc), NUM_STAGES)
    else:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2, arrive_count=1)
    bars_empty_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)
    bars_full_a = tlx.alloc_barriers(num_barriers=NUM_STAGES * 2, arrive_count=1)
    bars_full_b = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)

    # Handoff rings for the partitioned epilogue: one ring of half-tiles
    # per wgmma partition, with full/empty barriers per ring slot.
    if EPILOGUE_PARTITION:
        epi = tlx.local_alloc((BLOCK_M_SPLIT, BN), tlx.dtype_of(c_desc), EPI_STAGES * 2)
        epi_empty = tlx.alloc_barriers(num_barriers=EPI_STAGES * 2, arrive_count=1)
        epi_full = tlx.alloc_barriers(num_barriers=EPI_STAGES * 2, arrive_count=1)

    # Barriers for cross-CTA synchronization before multicast TMA loads.
    if NUM_CTAS == 2:
        cta_bars = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)

    with tlx.async_tasks(exclusive=True):

        # Consumer 0: wgmma + bias epilogue + TMA store, one replica per M-half.
        with tlx.async_task("default", registers=_CONS_REGS_C0):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            epi_cnt = 0
            cid = 0
            while tile_id < num_tiles:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BLOCK_M_SPLIT, BN], dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * cid)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    data_a = tlx.local_view(a, buf + NUM_STAGES * cid)
                    data_b = tlx.local_view(b, buf)
                    # Transpose SMEM buffers if inputs were column-major
                    b_operand = tlx.local_trans(data_b) if not B_ROW_MAJOR else data_b
                    if USE_PINGPONG:
                        tlx.named_barrier_wait(9, 256)
                    acc = tlx.async_dot(data_a, b_operand, acc)
                    if USE_PINGPONG:
                        tlx.named_barrier_arrive(10, 256)
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * cid)
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    smem_accum_cnt += 1

                offset_cm = offset_am + BLOCK_M_SPLIT * cid
                offs_m = offset_cm + tl.arange(0, BLOCK_M_SPLIT)
                offs_n = offset_bn + tl.arange(0, BN)
                bias_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
                bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
                c = (acc + bias.to(tl.float32)).to(tlx.dtype_of(c_desc))
                if EPILOGUE_PARTITION:
                    eslot = EPI_STAGES * cid + epi_cnt % EPI_STAGES
                    ephase = (epi_cnt // EPI_STAGES) & 1
                    tlx.barrier_wait(bar=tlx.local_view(epi_empty, eslot), phase=ephase ^ 1)
                    tlx.local_store(tlx.local_view(epi, eslot), c)
                    tlx.barrier_arrive(tlx.local_view(epi_full, eslot))
                    epi_cnt += 1
                elif EPILOGUE_SUBTILE:
                    c = tl.reshape(c, (BLOCK_M_SPLIT, 2, BN // 2))
                    c = tl.permute(c, (0, 2, 1))
                    c0, c1 = tl.split(c)
                    c_desc.store([offset_cm, offset_bn], c0)
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], c)

                tile_id += NUM_SMS

        # Consumers 1: wgmma + bias epilogue + TMA store, one replica per M-half.
        with tlx.async_task(num_warps=4, registers=_CONS_REGS_C1):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            epi_cnt = 0
            cid = 1
            if USE_PINGPONG:
                # Prime barrier 9 so consumer 0's first wait passes.
                tlx.named_barrier_arrive(9, 256)
            while tile_id < num_tiles:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BLOCK_M_SPLIT, BN], dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = get_bufidx_phase(smem_accum_cnt, NUM_STAGES)

                    full_a = tlx.local_view(bars_full_a, buf + NUM_STAGES * cid)
                    full_b = tlx.local_view(bars_full_b, buf)
                    tlx.barrier_wait(bar=full_a, phase=p)
                    tlx.barrier_wait(bar=full_b, phase=p)

                    data_a = tlx.local_view(a, buf + NUM_STAGES * cid)
                    data_b = tlx.local_view(b, buf)
                    # Transpose SMEM buffers if inputs were column-major
                    b_operand = tlx.local_trans(data_b) if not B_ROW_MAJOR else data_b
                    if USE_PINGPONG:
                        tlx.named_barrier_wait(10, 256)
                    acc = tlx.async_dot(data_a, b_operand, acc)
                    if USE_PINGPONG:
                        tlx.named_barrier_arrive(9, 256)
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    empty_a = tlx.local_view(bars_empty_a, buf + NUM_STAGES * cid)
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    tlx.barrier_arrive(empty_a)
                    tlx.barrier_arrive(empty_b)

                    smem_accum_cnt += 1

                offset_cm = offset_am + BLOCK_M_SPLIT * cid
                offs_m = offset_cm + tl.arange(0, BLOCK_M_SPLIT)
                offs_n = offset_bn + tl.arange(0, BN)
                bias_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
                bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
                c = (acc + bias.to(tl.float32)).to(tlx.dtype_of(c_desc))
                if EPILOGUE_PARTITION:
                    eslot = EPI_STAGES * cid + epi_cnt % EPI_STAGES
                    ephase = (epi_cnt // EPI_STAGES) & 1
                    tlx.barrier_wait(bar=tlx.local_view(epi_empty, eslot), phase=ephase ^ 1)
                    tlx.local_store(tlx.local_view(epi, eslot), c)
                    tlx.barrier_arrive(tlx.local_view(epi_full, eslot))
                    epi_cnt += 1
                elif EPILOGUE_SUBTILE:
                    c = tl.reshape(c, (BLOCK_M_SPLIT, 2, BN // 2))
                    c = tl.permute(c, (0, 2, 1))
                    c0, c1 = tl.split(c)
                    c_desc.store([offset_cm, offset_bn], c0)
                    c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                else:
                    c_desc.store([offset_cm, offset_bn], c)

                tile_id += NUM_SMS

        # Producer: TMA loads of A/B tiles.
        with tlx.async_task(num_warps=1, num_regs=24):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_tiles = num_pid_m * num_pid_n

            if NUM_CTAS == 2:
                # Partners must run the same number of producer iterations:
                # the per-k cta sync deadlocks if one partner exits early.
                # Strided sharding can strand one CTA of a pair with one tile
                # fewer (e.g. 3771 tiles over 132 CTAs leaves pair (74, 75)
                # at 29 vs 28 tiles), so both loop over the pair's max.
                count_me = tl.cdiv(num_tiles - sm_id, NUM_SMS)
                count_peer = tl.cdiv(num_tiles - (sm_id ^ 1), NUM_SMS)
                pair_count = tl.maximum(count_me, count_peer)
                tile_end = sm_id + pair_count * NUM_SMS
            else:
                tile_end = num_tiles

            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < tile_end:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                if NUM_CTAS == 2:
                    do_a = tile_id < num_tiles
                    # The partner's simultaneous tile decides the B-load form:
                    # a ghosted partner (solo tile) or a partner on another
                    # N-tile (split pair) means both halves come from plain
                    # loads; otherwise the halves are multicast. The peer
                    # group math is garbage when the partner is ghosted, but
                    # the clamped divisor keeps it fault-free and `split`
                    # masks it out.
                    peer_tile = (sm_id ^ 1) + ((tile_id - sm_id) // NUM_SMS) * NUM_SMS
                    peer_active = peer_tile < num_tiles
                    peer_group_id = peer_tile // num_pid_in_group
                    peer_first_pid_m = peer_group_id * GROUP_SIZE_M
                    peer_group_size_m = tl.maximum(1, min(num_pid_m - peer_first_pid_m, GROUP_SIZE_M))
                    peer_pid_n = (peer_tile % num_pid_in_group) // peer_group_size_m
                    solo = do_a & ~peer_active
                    split = do_a & peer_active & (pid_n != peer_pid_n)
                    need_both = solo | split
                else:
                    do_a = True
                    need_both = False

                for k in range(0, tl.cdiv(K, BK)):
                    buf, p = get_bufidx_phase(smem_accum_cnt, NUM_STAGES)
                    offset_k = k * BK

                    # Views are compile-time aliases, so they stay outside the
                    # predicated regions (names defined under a runtime `if`
                    # are not visible after it); only the waits and TMAs below
                    # are skipped on ghost iterations.
                    empty_a_1st = tlx.local_view(bars_empty_a, buf)
                    full_a_1st = tlx.local_view(bars_full_a, buf)
                    data_a_1st = tlx.local_view(a, buf)
                    empty_b = tlx.local_view(bars_empty_b, buf)
                    full_b = tlx.local_view(bars_full_b, buf)
                    data_b = tlx.local_view(b, buf)
                    empty_a_2nd = tlx.local_view(bars_empty_a, buf + NUM_STAGES)
                    full_a_2nd = tlx.local_view(bars_full_a, buf + NUM_STAGES)
                    data_a_2nd = tlx.local_view(a, buf + NUM_STAGES)

                    if do_a:
                        tlx.barrier_wait(bar=empty_a_1st, phase=p ^ 1)
                        tlx.barrier_expect_bytes(full_a_1st, BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                        tlx.async_descriptor_load(a_desc, data_a_1st, [offset_am, offset_k], full_a_1st)

                        tlx.barrier_wait(bar=empty_b, phase=p ^ 1)
                        tlx.barrier_expect_bytes(full_b, BK * BN * tlx.size_of(tlx.dtype_of(b_desc)))

                    if NUM_CTAS == 2:
                        # Sync cluster: both CTAs' buffers are drained (the
                        # wait above) and ready for multicast. Ghosts run this
                        # sync alone, with no TMA after.
                        cta_id = tlx.cluster_cta_rank()
                        cta_bar = tlx.local_view(cta_bars, buf)
                        tlx.barrier_arrive(cta_bar, 1)
                        tlx.barrier_arrive(cta_bar, 1, remote_cta_rank=cta_id ^ 1)
                        tlx.barrier_wait(bar=cta_bar, phase=p)

                    if do_a:
                        if NUM_CTAS == 2:
                            if need_both:
                                # Solo or split: the partner needs no (or other)
                                # data, so both halves come from plain loads
                                # with no cross-CTA side effects. The two
                                # half-arrivals sum to the expected bytes.
                                if not B_ROW_MAJOR:
                                    bb_lo = tlx.local_slice(data_b, [0, 0], [BN // 2, BK])
                                    bb_hi = tlx.local_slice(data_b, [BN // 2, 0], [BN // 2, BK])
                                    off_lo = [offset_bn, offset_k]
                                    off_hi = [offset_bn + BN // 2, offset_k]
                                else:
                                    bb_lo = tlx.local_slice(data_b, [0, 0], [BK, BN // 2])
                                    bb_hi = tlx.local_slice(data_b, [0, BN // 2], [BK, BN // 2])
                                    off_lo = [offset_k, offset_bn]
                                    off_hi = [offset_k, offset_bn + BN // 2]
                                tlx.async_descriptor_load(b_desc, bb_lo, off_lo, full_b)
                                tlx.async_descriptor_load(b_desc, bb_hi, off_hi, full_b)
                            else:
                                # Each CTA loads half of B and multicasts to both CTAs.
                                if not B_ROW_MAJOR:
                                    if cta_id == 0:
                                        bb = tlx.local_slice(data_b, [0, 0], [BN // 2, BK])
                                    else:
                                        bb = tlx.local_slice(data_b, [BN // 2, 0], [BN // 2, BK])
                                    off = [offset_bn + cta_id * BN // 2, offset_k]
                                else:
                                    if cta_id == 0:
                                        bb = tlx.local_slice(data_b, [0, 0], [BK, BN // 2])
                                    else:
                                        bb = tlx.local_slice(data_b, [0, BN // 2], [BK, BN // 2])
                                    off = [offset_k, offset_bn + cta_id * BN // 2]
                                tlx.async_descriptor_load(
                                    b_desc,
                                    bb,
                                    off,
                                    full_b,
                                    multicast_targets=[cta_id, cta_id ^ 1],
                                )
                        else:
                            if not B_ROW_MAJOR:
                                tlx.async_descriptor_load(b_desc, data_b, [offset_bn, offset_k], full_b)
                            else:
                                tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full_b)

                        tlx.barrier_wait(bar=empty_a_2nd, phase=p ^ 1)
                        tlx.barrier_expect_bytes(bar=full_a_2nd,
                                                 size=BLOCK_M_SPLIT * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                        tlx.async_descriptor_load(a_desc, data_a_2nd, [offset_am + BLOCK_M_SPLIT, offset_k], full_a_2nd)

                    smem_accum_cnt += 1

                tile_id += NUM_SMS

        if EPILOGUE_PARTITION:
            # Epilogue partition: replica g drains the TMA stores for the
            # M-half-g handoff ring, overlapping them with the next tile's
            # MMA. The subtile split (if any) happens here, on the loaded
            # half-tile, exactly as in the fused epilogue.
            with tlx.async_task(num_warps=4, num_regs=32, replicate=2):
                sm_id = tl.program_id(axis=0)
                num_pid_m = tl.cdiv(M, BM)
                num_pid_n = tl.cdiv(N, BN)
                num_pid_in_group = GROUP_SIZE_M * num_pid_n
                num_tiles = num_pid_m * num_pid_n
                rid = tlx.async_task_replica_id()

                tile_id = sm_id
                epi_cnt = 0
                while tile_id < num_tiles:
                    pid = tile_id
                    group_id = pid // num_pid_in_group
                    first_pid_m = group_id * GROUP_SIZE_M
                    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                    pid_m = first_pid_m + (pid % group_size_m)
                    pid_n = (pid % num_pid_in_group) // group_size_m
                    offset_cm = pid_m * BM + BLOCK_M_SPLIT * rid
                    offset_bn = pid_n * BN

                    eslot = EPI_STAGES * rid + epi_cnt % EPI_STAGES
                    ephase = (epi_cnt // EPI_STAGES) & 1
                    tlx.barrier_wait(bar=tlx.local_view(epi_full, eslot), phase=ephase)
                    c_tile = tlx.local_load(tlx.local_view(epi, eslot))
                    if EPILOGUE_SUBTILE:
                        c_tile = tl.reshape(c_tile, (BLOCK_M_SPLIT, 2, BN // 2))
                        c_tile = tl.permute(c_tile, (0, 2, 1))
                        c0, c1 = tl.split(c_tile)
                        c_desc.store([offset_cm, offset_bn], c0)
                        c_desc.store([offset_cm, offset_bn + BN // 2], c1)
                    else:
                        c_desc.store([offset_cm, offset_bn], c_tile)
                    tlx.barrier_arrive(tlx.local_view(epi_empty, eslot))

                    epi_cnt += 1
                    tile_id += NUM_SMS


@triton.jit
def addmm_kernel_direct(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BROADCAST_Y: tl.constexpr,
):
    pid_0, pid_1 = tl.program_id(axis=0), tl.program_id(axis=1)
    pid = pid_0 * tl.num_programs(axis=1) + pid_1
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z_mask = mask_m & mask_n
    if BROADCAST_Y:
        # y is a vector, broadcast to add to z
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=mask_n)
    else:
        y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
        y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
        y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
        y = tl.load(y_ptrs, mask=z_mask)
    z = (accumulator + y.to(tl.float32)).to(z_ptr.dtype.element_ty)
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)


@triton.jit
def addmm_kernel_tma_np(
    a_desc,
    b_desc,
    c_desc,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ym,
    stride_yn,
    stride_cm,
    stride_cn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SMS: tl.constexpr,
    B_ROW_MAJOR: tl.constexpr,
):
    """Non-persistent WS addmm: one CTA per tile, TMA loads, direct store.

    Same TMA-producer/wgmma-consumer structure as the persistent kernel
    with a single M partition, so BM=64 tiles fit wgmma. One CTA owns
    each M/N tile (grid == num_tiles), which suits few-tile shapes
    where a persistent grid would idle SMs. The C store is a masked
    direct store: cheaper than TMA issue at small tiles, and it takes
    M/N tails for free. The A/B barrier pair is fused into one full
    (arrive_count=2, both TMA completions) and one empty barrier per
    round, halving the per-round sync ops.
    """
    a = tlx.local_alloc((BM, BK), tlx.dtype_of(a_desc), NUM_STAGES)
    if not B_ROW_MAJOR:
        b = tlx.local_alloc((BN, BK), tlx.dtype_of(b_desc), NUM_STAGES)
    else:
        b = tlx.local_alloc((BK, BN), tlx.dtype_of(b_desc), NUM_STAGES)

    bars_empty = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=1)
    bars_full = tlx.alloc_barriers(num_barriers=NUM_STAGES, arrive_count=2)

    with tlx.async_tasks(exclusive=True):
        with tlx.async_task("default"):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < sm_id + 1:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                acc = tl.zeros([BM, BN], dtype=tl.float32)
                for k in range(0, tl.cdiv(K, BK)):
                    buf = smem_accum_cnt % NUM_STAGES
                    p = (smem_accum_cnt // NUM_STAGES) & 1

                    full = tlx.local_view(bars_full, buf)
                    tlx.barrier_wait(bar=full, phase=p)

                    data_a = tlx.local_view(a, buf)
                    data_b = tlx.local_view(b, buf)
                    b_operand = tlx.local_trans(data_b) if not B_ROW_MAJOR else data_b
                    acc = tlx.async_dot(data_a, b_operand, acc)
                    acc = tlx.async_dot_wait(tl.constexpr(0), acc)

                    empty = tlx.local_view(bars_empty, buf)
                    tlx.barrier_arrive(empty)

                    smem_accum_cnt += 1

                offs_m = offset_am + tl.arange(0, BM)
                offs_n = offset_bn + tl.arange(0, BN)
                bias_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
                bias = tl.load(bias_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
                c = (acc + bias.to(tl.float32)).to(tlx.dtype_of(c_desc))
                offs_cm = offset_am + tl.arange(0, BM)
                offs_cn = offset_bn + tl.arange(0, BN)
                c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

                tile_id += NUM_SMS

        with tlx.async_task(num_warps=1, num_regs=24):
            sm_id = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BM)
            num_pid_n = tl.cdiv(N, BN)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n

            tile_id = sm_id
            smem_accum_cnt = 0
            while tile_id < sm_id + 1:
                pid = tile_id
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m
                offset_am = pid_m * BM
                offset_bn = pid_n * BN

                for k in range(0, tl.cdiv(K, BK)):
                    buf = smem_accum_cnt % NUM_STAGES
                    p = (smem_accum_cnt // NUM_STAGES) & 1
                    offset_k = k * BK

                    empty = tlx.local_view(bars_empty, buf)
                    full = tlx.local_view(bars_full, buf)
                    tlx.barrier_wait(bar=empty, phase=p ^ 1)
                    # One expect per TMA: a single expect of the summed
                    # bytes leaves the second TMA unarmed and the wait
                    # below hangs.
                    tlx.barrier_expect_bytes(full, BM * BK * tlx.size_of(tlx.dtype_of(a_desc)))
                    data_a = tlx.local_view(a, buf)
                    tlx.async_descriptor_load(a_desc, data_a, [offset_am, offset_k], full)

                    tlx.barrier_expect_bytes(full, BK * BN * tlx.size_of(tlx.dtype_of(b_desc)))
                    data_b = tlx.local_view(b, buf)
                    if not B_ROW_MAJOR:
                        tlx.async_descriptor_load(b_desc, data_b, [offset_bn, offset_k], full)
                    else:
                        tlx.async_descriptor_load(b_desc, data_b, [offset_k, offset_bn], full)

                    smem_accum_cnt += 1
                tile_id += NUM_SMS


def _tma_block_hook(nargs):
    BM = nargs["BM"]
    BN = nargs["BN"]
    BK = nargs["BK"]
    NUM_CTAS = nargs.get("NUM_CTAS", 1)
    nargs["a_desc"].block_shape = [BM // 2, BK]
    # For column-major inputs, TMA descriptor block shape matches the transposed view
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BK, BN // NUM_CTAS]
    else:
        nargs["b_desc"].block_shape = [BN // NUM_CTAS, BK]
    if nargs["EPILOGUE_SUBTILE"]:
        nargs["c_desc"].block_shape = [BM // 2, BN // 2]
    else:
        nargs["c_desc"].block_shape = [BM // 2, BN]


def _config(block_m, block_n, block_k, group_m, num_warps, num_stages, pre_hook=None, num_smem_buffers=None):
    kwargs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
    }
    if num_smem_buffers is not None:
        kwargs["NUM_SMEM_BUFFERS"] = num_smem_buffers
    return triton.Config(
        kwargs,
        num_warps=num_warps,
        num_stages=num_stages,
        pre_hook=pre_hook,
    )


def _ws_config(block_m, block_n, block_k, group_m, stages, subtile, num_ctas=1, partition=False, pingpong=False):
    return triton.Config(
        {
            "BM": block_m,
            "BN": block_n,
            "BK": block_k,
            "GROUP_SIZE_M": group_m,
            "NUM_STAGES": stages,
            "EPILOGUE_SUBTILE": subtile,
            "NUM_CTAS": num_ctas,
            "EPILOGUE_PARTITION": partition,
            "USE_PINGPONG": pingpong,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_tma_block_hook,
        ctas_per_cga=(num_ctas, 1, 1),
    )


def _ws_prune_configs(configs, named_args, **kwargs):
    """Drop WS configs that cannot fit the H100 resource budget or the shape.

    Safety net over the hand-picked space below: smem (A/B operands +
    barriers + the ``BM * BN`` epilogue staging tile the TMA store lowers
    through) under 232448 bytes, wgmma M-split in {64, 128}, an even N
    whenever the epilogue halves it, and tiles within 2x of the problem.
    The staging term is measured, not modeled: without it the estimator
    undercounts by exactly one output tile (e.g. ``Required: 328008`` for
    ``(256, 256, 64, s=3)`` against a 196752-byte operand estimate).
    2-CTA configs additionally need an even N (the multicast halves B along
    N), at least 8 tiles (below that the cluster sync costs more than the
    multicast saves), and a grid divisible by 2 (CUDA forbids partial
    clusters, and the cross-CTA arrive would hang without its partner).
    Partitioned configs add the two half-tile handoff rings plus one more
    measured half-tile of store-path staging, and four more barriers.
    """
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    out = []
    for conf in configs:
        kw = conf.kwargs
        BM = kw["BM"]
        BN = kw["BN"]
        BK = kw["BK"]
        s = kw["NUM_STAGES"]
        split = BM // 2
        if split > 128 or split < 64:
            continue
        if kw["EPILOGUE_SUBTILE"] and BN % 2 != 0:
            continue
        if BK > K:
            continue
        if BM > M * 2 or BN > N * 2:
            continue
        smem_a = 2 * s * split * BK * 2
        smem_b = s * BK * BN * 2
        smem_barriers = 6 * s * 8
        smem_c = BM * BN * 2
        smem_epi = 0
        if kw.get("NUM_CTAS", 1) == 2:
            if BN % 2 != 0:
                continue
            tiles = math.ceil(M / BM) * math.ceil(N / BN)
            if tiles < 8:
                continue
            if min(tiles, _get_num_sms()) % 2 != 0:
                continue
            smem_barriers += s * 8
        if kw.get("EPILOGUE_PARTITION", False):
            # Two M-half handoff rings plus the measured extra half-tile:
            # without the extra term the estimator passes (256,64,64,s4)
            # partitioned at 229568 bytes, but it OORs at 246192.
            smem_epi = 3 * split * BN * 2
            smem_barriers += 4 * 8
        if smem_a + smem_b + smem_barriers + smem_c + smem_epi > 232448:
            continue
        out.append(conf)
    return out


def _ws_configs():
    """Persistent WS space: (BM, BN, BK, stages, subtile, num_ctas, partition).

    Two MMA groups, so BM >= 128 keeps the wgmma M-split at 64+. The
    subtile entries halve the TMA store along N; the 2-CTA twins multicast
    B across the cluster; the partitioned twins move the TMA store to a
    2-replica epilogue task (only small tiles fit the extra handoff smem).
    The TMA block hook rides on each config: the autotuner only replays
    per-config hooks on cache-hit launches, and the descriptors are
    rebuilt fresh on every call.
    """
    cfgs = [
        (128, 256, 64, 3, False, 1, False),
        (128, 256, 64, 3, True, 1, False),
        (128, 128, 64, 3, False, 1, False),
        (128, 128, 64, 3, True, 1, False),
        (128, 128, 64, 4, False, 1, False),
        (256, 128, 64, 3, False, 1, False),
        (256, 128, 64, 3, True, 1, False),
        (128, 64, 64, 4, False, 1, False),
        (256, 64, 64, 3, False, 1, False),
        (256, 64, 64, 3, True, 1, False),
        (128, 128, 128, 2, False, 1, False),
        (256, 64, 128, 2, False, 1, False),
        (128, 256, 64, 3, False, 2, False),
        (128, 256, 64, 3, True, 2, False),
        (128, 128, 64, 3, False, 2, False),
        (256, 128, 64, 3, False, 2, False),
        # Sweep winners point at deeper stages and wider K: the s4 family
        # completion, BK=128 with s3, and one s8 probe (all fit the smem
        # budget by the prune estimator; autotune skips any that do not).
        (256, 64, 64, 4, False, 1, False),
        (128, 128, 128, 3, False, 1, False),
        (128, 64, 128, 3, False, 1, False),
        (128, 64, 64, 8, False, 1, False),
        # Partitioned twins: the store-overlap wins on deep pipelines and
        # large M, measured on (128,128,64,s4); wide twins OOR (smem and
        # ptxas registers) and stay out.
        (128, 128, 64, 3, False, 1, True),
        (128, 128, 64, 4, False, 1, True),
        (128, 64, 64, 4, False, 1, True),
        (128, 64, 128, 3, False, 1, True),
    ]
    return [_ws_config(bm, bn, bk, 8, s, sub, nc, part) for bm, bn, bk, s, sub, nc, part in cfgs]


def _ws_smoke_configs():
    cfgs = [
        (128, 128, 64, 2, False, 1, False),
        (128, 64, 64, 2, False, 1, False),
        (128, 128, 64, 2, False, 2, False),
        (128, 128, 64, 2, False, 1, True),
        (128, 128, 64, 2, True, 1, True),
        (128, 128, 64, 2, False, 2, True),
    ]
    return [_ws_config(bm, bn, bk, 8, s, sub, nc, part) for bm, bn, bk, s, sub, nc, part in cfgs]


def _ws_heuristic_config(M, N, K, num_ctas=1):
    """Choose one warp-specialized configuration without runtime autotuning.

    Rules fit to per-config sweeps over the 13-shape trace suite (at the
    full-space best within noise on all 13): BK=128 wins once the
    pipeline fills (K>=512), small problems take BN=64, large-M takes
    partitioned s4 (always for K<512; for K>=512 once N>=1024),
    small-M takes the wide shallow-K tile, and only giant-M keeps the
    wide tile (on two CTAs past M=524288).
    Every output is a member of the full space, so autotune can only
    tie or beat the heuristic. ``num_ctas=2`` pins the same tile choice to
    its 2-CTA twin for the ``heuristic_2cta`` space.
    """
    # Giant-M keeps the wide tile (M=705448 runs 6% faster wide than on
    # BK=128; single-point fit, nothing else in-suite comes close). Past
    # M=524288 the wide tile's 2-CTA twin wins (8-16% at M=705448,
    # confirmed in three runs; at/below M=524288 it ties or loses,
    # measured at 163840/262144/524288). A pruned twin (odd grid)
    # degrades to one CTA instead of erroring.
    if M >= 262144 and N >= 1024:
        if M > 524288:
            twin = _ws_config(128, 256, 64, 8, 3, False, 2)
            if _ws_prune_configs([twin], {"M": M, "N": N, "K": K}):
                return [twin]
        return [_ws_config(128, 256, 64, 8, 3, False, num_ctas)]
    # Tiny-N keeps the narrow tile at s4 (no suite shape covers it, but
    # s4 meets-or-beats s3 on every measured BK=64 tile, and this keeps the
    # heuristic inside the full space).
    if N <= 64:
        return [_ws_config(128, 64, 64, 8, 4, False, num_ctas)]
    if K >= 512:
        BK, s = 128, 3
    else:
        # K<64 would need BK<64 for TMA legality, but no eligible shape
        # has K<64 (K%16==0 and K<64 is empty in L1 and the suite).
        BK, s = 64, 4
    BN = 64 if (M <= 1024 and N <= 768) else 128
    # Large-M s4 pocket: partitioned (128,128,64,s4) beats fused s4
    # (K<512) and, once N>=1024, fused BK=128 (K>=512, +4-8% at
    # M>=21347, confirmed in repeat runs); at N<=768 the BK=128 tile
    # still wins (3-6%), so those shapes keep it. Neutral at M<=16384
    # either way.
    if M > 16384 and BN == 128 and (K < 512 or N >= 1024):
        return [_ws_config(128, 128, 64, 8, 4, False, num_ctas, True)]
    # Small-M wide pocket: below M=65536 with K>=512 and N<1024 the
    # wide shallow-K tile beats fused BK=128 (+2.7% at M=21197, Buck
    # full-space winner confirmed by local bakeoff; single-point fit,
    # M=160853 still ties on BK=128). N>=128 keeps BN=256 in-prune.
    if M > 16384 and M < 65536 and K >= 512 and 128 <= N < 1024:
        return [_ws_config(128, 256, 64, 8, 3, False, num_ctas)]
    return [_ws_config(128, BN, BK, 8, s, False, num_ctas)]


def _ws_heuristic_2cta_config(M, N, K):
    """2-CTA twin of the heuristic tile, with a 1-CTA fallback.

    A lone 2-CTA config would leave autotune with nothing when the prune
    drops it (few tiles or an odd grid), so a pruned twin degrades to the
    same tile on one CTA instead of erroring.
    """
    cfg = _ws_heuristic_config(M, N, K, num_ctas=2)[0]
    if _ws_prune_configs([cfg], {"M": M, "N": N, "K": K}):
        return [cfg]
    return _ws_heuristic_config(M, N, K, num_ctas=1)


def _np_block_hook(nargs):
    BM = nargs["BM"]
    BN = nargs["BN"]
    BK = nargs["BK"]
    nargs["a_desc"].block_shape = [BM, BK]
    if nargs.get("B_ROW_MAJOR", True):
        nargs["b_desc"].block_shape = [BK, BN]
    else:
        nargs["b_desc"].block_shape = [BN, BK]
    nargs["c_desc"].block_shape = [BM, BN]


def _np_config(block_m, block_n, block_k, stages):
    return triton.Config(
        {
            "BM": block_m,
            "BN": block_n,
            "BK": block_k,
            "GROUP_SIZE_M": 8,
            "NUM_STAGES": stages,
        },
        num_warps=4,
        num_stages=1,
        pre_hook=_np_block_hook,
    )


def _np_prune_configs(configs, named_args, **kwargs):
    """Drop NP configs outside the H100 budget or the shape.

    Single-partition operand smem plus barriers under 232448 bytes; no
    epilogue staging term (the store is direct). BM itself must fit
    wgmma (64-128), and tiles stay within 2x of the problem.
    """
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    out = []
    for conf in configs:
        kw = conf.kwargs
        BM = kw["BM"]
        BN = kw["BN"]
        BK = kw["BK"]
        s = kw["NUM_STAGES"]
        if BM > 128 or BM < 64:
            continue
        if BK > K:
            continue
        if BM > M * 2 or BN > N * 2:
            continue
        smem = s * BM * BK * 2 + s * BK * BN * 2
        if smem + 2 * s * 8 > 232448:
            continue
        out.append(conf)
    return out


def _np_configs():
    """Non-persistent space: small single-partition tiles.

    128-tiles lose 15%+ to 64s on every few-tile shape (halved
    occupancy, no per-tile gain), so the search stays small. The
    BN=32/BK=256 family serves long-K and tiny-N shapes, where
    narrow N buys CTA count and deep K cuts K-round overhead.
    """
    cfgs = [
        (64, 64, 128, 3),
        (64, 128, 128, 3),
        (64, 128, 64, 4),
        (64, 32, 256, 4),
        (64, 32, 256, 3),
        (64, 64, 256, 3),
    ]
    return [_np_config(bm, bn, bk, s) for bm, bn, bk, s in cfgs]


def _np_smoke_configs():
    cfgs = [
        (64, 128, 64, 2),
        (128, 128, 64, 2),
    ]
    return [_np_config(bm, bn, bk, s) for bm, bn, bk, s in cfgs]


def _np_heuristic_config(M, N, K):
    """Choose one non-persistent configuration without runtime autotuning.

    BN=128 wins on per-tile efficiency while it still covers the
    machine; below a wave of BN=32 tiles the narrow-N/deep-K tile
    wins back CTA count (+30% at (160,256,2560)); at/above a wave
    BN=64 with deep K wins (+3% at (652,768,768)).
    """
    if -(-M // 64) * -(-N // 128) >= 72:
        return [_np_config(64, 128, 128, 3)]
    if K >= 256:
        if -(-M // 64) * -(-N // 32) < 132:
            return [_np_config(64, 32, 256, 4)]
        return [_np_config(64, 64, 256, 3)]
    return [_np_config(64, 64, 128, 3)]


def _direct_configs():
    """The sm100-filtered CUDA branch of the baseline's ``get_mm_configs``.

    Shared with the sm100 implementation: the baseline's set minus its two
    ``num_warps=2`` configs, which keeps the direct path identical across
    archs at a small tuning-coverage cost on sm90.
    """
    return [
        _config(128, 256, 64, 8, 8, 3),
        _config(64, 256, 32, 8, 4, 4),
        _config(128, 128, 32, 8, 4, 4),
        _config(128, 64, 32, 8, 4, 4),
        _config(64, 128, 32, 8, 4, 4),
        _config(128, 32, 32, 8, 4, 4),
        _config(128, 256, 64, 8, 4, 3),
    ]


CONFIGS = _direct_configs


def _direct_smoke_configs():
    return [_config(128, 128, 32, 8, 4, 4), _config(64, 128, 32, 8, 4, 4)]


SMOKE_CONFIGS = _direct_smoke_configs


def heuristic_config(M, N, K):
    """Choose one direct-load configuration without runtime autotuning.

    The first two branches serve shapes the WS path routes away (see
    ``_use_direct_path``).
    """
    if K <= 256 and M <= 4096:
        BN = 32 if M <= 1024 else 64
        return [_config(128, BN, 32, 8, 4, 4)]
    if K <= 512 and M <= 4096 and N > 64 and (M > 1024 or N > 768):
        return [_config(64, 128, 32, 8, 4, 4)]
    # Skinny-M: short row tiles keep masked rows low while reusing B.
    if M <= 64:
        return [_config(64, 128, 32, 8, 4, 4)]
    # Skinny-N: narrow column tiles keep masked columns low.
    if N <= 64:
        return [_config(128, 64, 32, 8, 4, 4)]
    # Large squares have enough tiles for the wide high-reuse shape.
    if M >= 1024 and N >= 1024:
        return [_config(128, 256, 64, 8, 8, 3)]
    return [_config(128, 128, 32, 8, 4, 4)]


def _space_configs(space, shape, configs_fn, smoke_fn, heuristic_fn):
    if space == "heuristic":
        return heuristic_fn(*shape) or smoke_fn()
    if space == "full":
        return configs_fn()
    if space == "smoke":
        return smoke_fn()
    raise ValueError(f"Unknown sm90 addmm search space: {space!r}")


@functools.lru_cache(maxsize=None)
def _get_num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@functools.lru_cache(maxsize=None)
def _tuned_ws(space, shape):
    """Autotuned warp-specialized kernel per search space."""
    if space == "heuristic_2cta":
        configs = _ws_heuristic_2cta_config(*shape)
    else:
        configs = _space_configs(space, shape, _ws_configs, _ws_smoke_configs, _ws_heuristic_config)
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K", "B_ROW_MAJOR"],
        prune_configs_by={"early_config_prune": _ws_prune_configs},
    )(addmm_kernel_tma_ws)


@functools.lru_cache(maxsize=None)
def _tuned_direct(space, shape=None):
    """Autotuned direct-load kernel per search space (non-TMA fallback)."""
    if space == "heuristic_2cta":
        # The direct kernel has no cluster mode; the 2-CTA request only
        # affects the WS path, so fall back to the plain heuristic here.
        space = "heuristic"
    configs = _space_configs(space, shape, _direct_configs, _direct_smoke_configs, heuristic_config)
    return triton.autotune(configs=configs, key=["M", "N", "K", "BROADCAST_Y"])(addmm_kernel_direct)


@functools.lru_cache(maxsize=None)
def _tuned_np(space, shape):
    """Autotuned non-persistent kernel per search space."""
    if space == "heuristic_2cta":
        # The NP kernel has no cluster mode; fall back to the plain
        # heuristic the way the direct path does.
        space = "heuristic"
    configs = _space_configs(space, shape, _np_configs, _np_smoke_configs, _np_heuristic_config)
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K", "B_ROW_MAJOR"],
        prune_configs_by={"early_config_prune": _np_prune_configs},
    )(addmm_kernel_tma_np)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    M, K = a.shape
    N = b.shape[1]
    if out is not None:
        if out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({M}, {N}) on A's device")
    return M, N, K


def _bias_strides(bias, M, N, a):
    if bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if bias.ndim == 1:
        if bias.shape[0] != N:
            raise ValueError(f"1-D addmm input must have shape ({N},), got {tuple(bias.shape)}")
        return 0, bias.stride(0)
    if bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N):
        return (0 if bias.shape[0] == 1 else bias.stride(0), 0 if bias.shape[1] == 1 else bias.stride(1))
    raise ValueError(f"addmm input with shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})")


def _tma_eligible(a, b, out, N, K):
    """Whether the warp-specialized TMA path can serve these operands.

    TMA needs 16-element-aligned K/N plus 16-byte-aligned bases; A needs
    a unit inner stride while B may be row- or column-major (column-major
    B runs through a transposed descriptor). Anything else runs on the
    direct-load kernel instead. The bias is direct-loaded in the
    epilogue, so it carries no TMA constraint -- every broadcastable bias
    rides the WS path.
    """
    if K % 16 != 0 or N % 16 != 0:
        return False
    if a.data_ptr() % 16 != 0 or a.stride(1) != 1:
        return False
    # TensorDescriptor requires a contiguous inner dim plus 16-byte-aligned
    # outer strides. Row-major B is checked as before; column-major B is
    # judged by its transposed view (unit inner stride s0, aligned outer s1).
    if b.data_ptr() % 16 != 0:
        return False
    if b.stride(1) == 1:
        pass
    elif b.stride(0) == 1 and (b.stride(1) * b.element_size()) % 16 == 0:
        pass
    else:
        return False
    return out.data_ptr() % 16 == 0


def _use_direct_path(M, N, K):
    """Route small shapes to the direct kernel even when TMA-eligible.

    Direct wins where K/M are small enough that WS tiles stay coarse.
    """
    if M > 4096:
        return False
    if K <= 256:
        return True
    return K <= 512 and (M > 1024 or N > 768)


def _use_np_path(M, N, K):
    """Route few-tile shapes to the non-persistent kernel.

    Below ~2 waves of 128-tiles a persistent grid idles SMs, so one
    CTA per tile wins. The floors keep the heuristic tiles inside the
    prune (BM within 2x of M, BK within K).
    """
    if M < 32 or N < 64 or K < 128:
        return False
    return -(-M // 128) * -(-N // 128) < 256


def _launch_ws(bias, a, b, c, M, N, K, stride_ym, stride_yn, space):
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    # A column-major (K, N) tensor has strides (1, K); its .T is row-major
    # (N, K), which is what the transposed descriptor is built on.
    b_row_major = b.stride(1) == 1
    b_src = b if b_row_major else b.T
    b_desc = TensorDescriptor(b_src, b_src.shape, b_src.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    num_sms = _get_num_sms()

    def grid(meta):
        tiles = triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"])
        return (min(tiles, num_sms), )

    kernel = _tuned_ws(space, (M, N, K) if space in ("heuristic", "heuristic_2cta") else None)
    kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        bias,
        M,
        N,
        K,
        stride_ym,
        stride_yn,
        NUM_SMS=num_sms,
        B_ROW_MAJOR=b_row_major,
    )
    return c


def _launch_direct(bias, a, b, c, M, N, K, stride_ym, stride_yn, space):

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    kernel = _tuned_direct(space, (M, N, K) if space in ("heuristic", "heuristic_2cta") else None)
    kernel[grid](
        a,
        b,
        bias,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_ym,
        stride_yn,
        c.stride(0),
        c.stride(1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=bias.ndim == 1,
    )
    return c


def _launch_np(bias, a, b, c, M, N, K, stride_ym, stride_yn, space):
    dummy_block = [1, 1]
    a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
    b_row_major = b.stride(1) == 1
    b_src = b if b_row_major else b.T
    b_desc = TensorDescriptor(b_src, b_src.shape, b_src.stride(), dummy_block)
    c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
    num_sms = _get_num_sms()

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), )

    kernel = _tuned_np(space, (M, N, K) if space in ("heuristic", "heuristic_2cta") else None)
    kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        bias,
        c,
        M,
        N,
        K,
        stride_ym,
        stride_yn,
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
        B_ROW_MAJOR=b_row_major,
    )
    return c


# --- GEMM path shared with tlx.ops.mm -----------------------------------------
#
# TMA-capable inputs with a bias constant along M run the sm90 mm kernels, which
# take an optional (N,) bias in their epilogue; see `_mm._run`.


def _mm_path_eligible(a, b, c):
    """Operands the mm kernels' TMA descriptors can address: row- or
    column-major A and B, contiguous C, 16-byte-aligned bases and rows."""
    if not (a.is_contiguous() or a.T.is_contiguous()) or not (b.is_contiguous() or b.T.is_contiguous()):
        return False
    if not c.is_contiguous():
        return False
    for t in (a if a.is_contiguous() else a.T, b if b.is_contiguous() else b.T, c):
        if t.data_ptr() % 16 != 0 or (t.shape[1] * t.element_size()) % 16 != 0:
            return False
    return True


# --- N = 1 (matrix-vector) ---------------------------------------------------
#
# A single output column has no MMA shape and no TMA-legal row stride; it is a
# bandwidth-bound row reduction over A, so each program reduces BLOCK_M rows.


def _gemv_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_K": bk}, num_warps=w)
        for bm, bk, w in ((8, 128, 4), (16, 128, 4), (16, 256, 4), (32, 128, 4), (32, 64, 2), (64, 64, 4))
    ]


@functools.lru_cache(maxsize=None)
def _tuned_gemv(space):
    configs = _gemv_configs() if space == "full" else [triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=4)]
    return triton.autotune(configs=configs, key=["M", "K"])(addmm_kernel_gemv)


# c_cache=False for the reason given at mm's `_ws_kernel_bias`.
@triton.jit(c_cache=False)
def addmm_kernel_gemv(a_ptr, b_ptr, bias_ptr, c_ptr, M, K, stride_am, stride_ak, stride_bk, stride_cm,
                      BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k * BLOCK_K + tl.arange(0, BLOCK_K)
        col_mask = cols < K
        a = tl.load(a_ptr + rows[:, None] * stride_am + cols[None, :] * stride_ak,
                    mask=row_mask[:, None] & col_mask[None, :], other=0.0)
        w = tl.load(b_ptr + cols * stride_bk, mask=col_mask, other=0.0)
        acc += a.to(tl.float32) * w.to(tl.float32)[None, :]
    # With N = 1 an (N,) bias is a single value.
    out = tl.sum(acc, axis=1) + tl.load(bias_ptr).to(tl.float32)
    tl.store(c_ptr + rows * stride_cm, out.to(c_ptr.dtype.element_ty), mask=row_mask)


def _launch_gemv(bias, a, b, c, M, K, space):
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), )  # noqa: E731
    _tuned_gemv(space)[grid](a, b, bias, c, M, K, a.stride(0), a.stride(1), b.stride(0), c.stride(0))
    return c


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Fused ``input + a @ b`` on Hopper.

    ``input`` may be ``(N,)`` or two-dimensional and broadcastable to the
    ``(M, N)`` result. Matrix and input scale factors are both one. This op is
    currently forward-only.

    In the "full" and "heuristic" spaces, a bias that is constant along M --
    ``(N,)``, ``(1, N)`` or a scalar -- with row- or column-major A and B
    with 16-byte-aligned bases and rows runs the tlx.ops.mm kernels with the
    bias added in their epilogue (see ``_mm._run``), and N = 1 runs a
    matrix-vector kernel (``_launch_gemv``). Other inputs (e.g. an ``(M, 1)``
    or ``(M, N)`` bias), and the "smoke" and "heuristic_2cta" spaces, run the
    original kernels below: the warp-specialized kernel, the non-persistent
    kernel for few-tile shapes (``_use_np_path``) or the direct-load kernel.

    `space` selects the search space -- "full" (the default via the catalog)
    autotunes, "heuristic" picks one config, "smoke" covers the original
    kernels' lowering paths, and "heuristic_2cta" pins the original kernel's
    heuristic tile to two CTAs (with a 1-CTA fallback when clustering is
    illegal for the shape).
    """
    M, N, K = _validate_operands(a, b, out)
    stride_ym, stride_yn = _bias_strides(input, M, N, a)
    c = out if out is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c

    if space in ("full", "heuristic") and K > 0 and stride_ym == 0:
        if N == 1:
            return _launch_gemv(input, a, b, c, M, K, space)
        if _mm_path_eligible(a, b, c):
            return _mm._run(a, b, c, M, N, K, space, bias=input, stride_bias=stride_yn)
    if _tma_eligible(a, b, c, N, K):
        if _use_np_path(M, N, K):
            return _launch_np(input, a, b, c, M, N, K, stride_ym, stride_yn, space)
        if not _use_direct_path(M, N, K):
            return _launch_ws(input, a, b, c, M, N, K, stride_ym, stride_yn, space)
    return _launch_direct(input, a, b, c, M, N, K, stride_ym, stride_yn, space)
