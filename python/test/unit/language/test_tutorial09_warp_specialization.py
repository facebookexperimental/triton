"""
Explicit unit tests for all warp-specialized variations of Tutorial 09 (Persistent Matmul).

These tests validate the warp specialization feature for persistent matmul kernels
with both Flatten=True and Flatten=False configurations. Tests cover both
Blackwell and Hopper GPUs.
"""

from typing import NamedTuple

import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_blackwell, is_hopper
from triton.language.extra.subtile_ops import _split_n_2D
from triton.tools.tensor_descriptor import TensorDescriptor


# Helper function from tutorial 09
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ============================================================================
# Kernel 1: matmul_kernel_tma - TMA-based matmul with warp specialization
# This kernel uses warp_specialize in the K-loop (inner loop)
# ============================================================================
@triton.jit
def matmul_kernel_tma_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SEPARATE_EPILOGUE_STORE: tl.constexpr,
):
    """TMA-based matmul with warp specialization in K-loop (always enabled)."""
    dtype = tl.float16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Always use warp_specialize=True
    for k in tl.range(
            k_tiles,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        offs_k = k * BLOCK_SIZE_K
        if A_COL_MAJOR:
            a = a_desc.load([offs_k, offs_am]).T
        else:
            a = a_desc.load([offs_am, offs_k])
        if B_COL_MAJOR:
            b = b_desc.load([offs_k, offs_bn]).T
        else:
            b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(dtype)

    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


# ============================================================================
# Kernel 2: matmul_kernel_tma_persistent - Persistent TMA matmul with warp spec
# This kernel uses warp_specialize in the outer tile loop with flatten parameter
# ============================================================================
@triton.jit
def matmul_kernel_tma_persistent_ws(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FLATTEN: tl.constexpr,
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SEPARATE_EPILOGUE_STORE: tl.constexpr,
):
    """Persistent TMA matmul with warp specialization (always enabled)."""
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Always use warp_specialize=True with configurable flatten
    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if A_COL_MAJOR:
                a = a_desc.load([offs_k, offs_am]).T
            else:
                a = a_desc.load([offs_am, offs_k])
            if B_COL_MAJOR:
                b = b_desc.load([offs_k, offs_bn]).T
            else:
                b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )


# ============================================================================
# Kernel 2b: matmul_kernel_tma_static_persistent_ws_while
# Static persistent TMA matmul whose persistent outer loop is a while loop.
# Work assignment is still static: each CTA processes tile_id += NUM_SMS.
# ============================================================================
@triton.jit
def matmul_kernel_tma_static_persistent_ws_while(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    tile_id = start_pid

    while tl.condition(tile_id < num_tiles, warp_specialize=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )
        tile_id += NUM_SMS


# ============================================================================
# Kernel 2c: matmul_kernel_tma_dynamic_persistent_ws_while
# Dynamic (work-stealing) persistent TMA matmul whose persistent outer loop is a
# while loop. Each CTA starts at its program id, then grabs the next tile from a
# global counter via atomic_add (no constant per-CTA increment), so tiles are
# distributed dynamically rather than statically strided by NUM_SMS.
# ============================================================================
@triton.jit
def matmul_kernel_tma_dynamic_persistent_ws_while(
    a_desc,
    b_desc,
    c_desc,
    tile_counter,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Each CTA seeds its first tile with its program id; subsequent tiles are
    # claimed dynamically from the shared counter (initialized to NUM_SMS).
    tile_id = start_pid

    while tile_id < num_tiles:
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in tl.range(k_tiles, warp_specialize=True):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )
        # Dynamically claim the next tile (work stealing): no constant increment.
        tile_id = tl.atomic_add(tile_counter, 1)


# ============================================================================
# Kernel 2d: matmul_kernel_tma_clc_persistent_ws_while
# Dynamic (work-stealing) persistent TMA matmul whose while-loop tile id is
# claimed via the core CLC tile scheduler (tl.clc_tile_scheduler) instead of a
# global atomic counter. The grid is launched with one cluster per tile so
# running CTAs can cancel/steal pending clusters (Blackwell hardware CLC).
# ============================================================================
@triton.jit
def matmul_kernel_tma_clc_persistent_ws_while(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = tl.float16
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    sched = tl.clc_tile_scheduler()
    while sched.is_valid():
        tile_id = sched.tile_id[0]
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in tl.range(k_tiles, warp_specialize=True):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )
        # Claim the next tile via CLC hardware work-stealing.
        sched = sched.advance()


# ============================================================================
# Kernel 2e: matmul_kernel_tma_unified_persistent_ws_while
# A single outer-loop AutoWS kernel for the unified schedules.
# ============================================================================
class _MatmulTileArgs(NamedTuple):
    """Named ``lowering_args`` for the schedule -- the fields ``_unified_num_tiles``
    reads by name to compute the tile count."""

    M: tl.tensor
    N: tl.tensor
    BLOCK_SIZE_M: tl.constexpr
    BLOCK_SIZE_N: tl.constexpr
    NUM_CTAS: tl.constexpr


@triton.jit
def _unified_num_tiles(lowering_args):
    num_tiles = tl.cdiv(lowering_args.M, lowering_args.BLOCK_SIZE_M) * tl.cdiv(lowering_args.N,
                                                                               lowering_args.BLOCK_SIZE_N)
    return tl.cdiv(num_tiles, lowering_args.NUM_CTAS) * lowering_args.NUM_CTAS


@triton.jit
def matmul_kernel_tma_unified_persistent_ws_while(
    a_desc,
    b_desc,
    c_desc,
    tile_counter,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SCHEDULE: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    TWO_CTAS: tl.constexpr,
    SMEM_ALLOC_ALGO: tl.constexpr,
    SEPARATE_EPILOGUE_STORE: tl.constexpr,
):
    dtype = tl.float16
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    lowering_args = _MatmulTileArgs(M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_CTAS)
    sched = SCHEDULE.initialize(lowering_args, _unified_num_tiles, tile_counter)
    while tl.condition(
            sched.is_valid(),
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
            smem_alloc_algo=SMEM_ALLOC_ALGO,
    ):
        pid_m, pid_n = _compute_pid(sched.tile_id[0], num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator, two_ctas=TWO_CTAS)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )
        # Claim the next tile -- how depends entirely on the selected schedule.
        sched = sched.advance()


# ============================================================================
# Kernel 3: matmul_kernel_descriptor_persistent - Device-side TMA descriptors
# Uses warp_specialize with flatten in outer tile loop
# ============================================================================
@triton.jit
def matmul_kernel_descriptor_persistent_ws(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FLATTEN: tl.constexpr,
    A_COL_MAJOR: tl.constexpr,
    B_COL_MAJOR: tl.constexpr,
    DATA_PARTITION_FACTOR: tl.constexpr,
    SEPARATE_EPILOGUE_STORE: tl.constexpr,
):
    """Persistent matmul with device-side TMA descriptors and warp specialization (always enabled)."""
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    if A_COL_MAJOR:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[K, M],
            strides=[M, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_M],
        )
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        )
    if B_COL_MAJOR:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[K, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        )
    else:
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N // EPILOGUE_SUBTILE,
        ],
    )

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Always use warp_specialize=True with configurable flatten
    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            data_partition_factor=DATA_PARTITION_FACTOR,
            separate_epilogue_store=SEPARATE_EPILOGUE_STORE,
    ):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            if A_COL_MAJOR:
                a = a_desc.load([offs_k, offs_am]).T
            else:
                a = a_desc.load([offs_am, offs_k])
            if B_COL_MAJOR:
                b = b_desc.load([offs_k, offs_bn]).T
            else:
                b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            c_desc.store(
                [offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )


# ============================================================================
# Kernel 4: matmul_kernel_tma_persistent_ws_splitk
# Persistent TMA matmul + warp specialization + deterministic Split-K.
# Mirrors Kernel 2 but expands the persistent grid by SPLIT_K. Each split
# writes its partial sum into a (SPLIT_K * M, N) workspace at row split_id*M;
# a separate _reduce_k_kernel folds the slabs into C in fp32.
# Requires SPLIT_K > 1 — the data-parallel case is already covered by Kernel 2.
# ============================================================================
@triton.jit
def matmul_kernel_tma_persistent_ws_splitk(
    a_desc,
    b_desc,
    workspace_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr,
    NUM_SMS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    FLATTEN: tl.constexpr,
):
    """Persistent TMA matmul with warp specialization + deterministic Split-K.

    Caller must guarantee cdiv(k_tiles, SPLIT_K) * (SPLIT_K - 1) < k_tiles
    so every split has at least one K tile — otherwise the warp-specialized
    inner loop runs zero iterations and the producer/consumer partition can
    deadlock waiting on barriers that are never armed.
    """
    tl.static_assert(SPLIT_K > 1, "splitk kernel requires SPLIT_K > 1")
    dtype = tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles_total = tl.cdiv(K, BLOCK_SIZE_K)
    num_mn_tiles = num_pid_m * num_pid_n
    num_tiles = num_mn_tiles * SPLIT_K

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(
            start_pid,
            num_tiles,
            NUM_SMS,
            flatten=FLATTEN,
            warp_specialize=True,
            separate_epilogue_store=True,
    ):
        split_id = tile_id // num_mn_tiles
        mn_tile_id = tile_id % num_mn_tiles
        k_per_split = tl.cdiv(k_tiles_total, SPLIT_K)
        k_start = split_id * k_per_split
        k_end = tl.minimum(k_start + k_per_split, k_tiles_total)

        pid_m, pid_n = _compute_pid(mn_tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_start, k_end):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        row_base = split_id * M
        acc_slices = _split_n_2D(accumulator, EPILOGUE_SUBTILE)
        slice_size: tl.constexpr = BLOCK_SIZE_N // EPILOGUE_SUBTILE
        for slice_id in tl.static_range(0, EPILOGUE_SUBTILE):
            workspace_desc.store(
                [row_base + offs_am, offs_bn + slice_id * slice_size],
                acc_slices[slice_id].to(dtype),
            )


@triton.jit
def _reduce_k_kernel(
    workspace_ptr,
    c_ptr,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Fold SPLIT_K partial-sum slabs from workspace into C, accumulating in fp32."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    tl.store(c_ptr + base, acc.to(OUTPUT_DTYPE), mask=mask)


# ============================================================================
# Test 1: matmul_kernel_tma warp specialization (K-loop based)
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("generate_subtiled_region", [True, False])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    generate_subtiled_region,
    separate_epilogue_store,
):
    """Test matmul_kernel_tma with warp_specialize=True (K-loop based)."""

    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors (swap dims for col-major so contiguous dim is last)
        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

        kernel = matmul_kernel_tma_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            generate_subtiled_region=generate_subtiled_region,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 2: matmul_kernel_tma_persistent warp specialization (tile-loop based)
# Tests both Flatten=True and Flatten=False
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("generate_subtiled_region", [True, False])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    FLATTEN,
    EPILOGUE_SUBTILE,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    generate_subtiled_region,
    separate_epilogue_store,
):
    """Test matmul_kernel_tma_persistent with warp_specialize=True for both Flatten values."""

    if FLATTEN:
        pytest.skip("FLATTEN will not WarpSpecialize although it will otherwise pass.")

    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: tensor memory exceeded (BLOCK_SIZE_M=256 with num_stages=3 and FLATTEN)")

    if DATA_PARTITION_FACTOR == 2 and BLOCK_SIZE_M == 256 and FLATTEN and EPILOGUE_SUBTILE == 4 and num_stages == 3:
        pytest.skip("Out of resources: tensor memory exceeded")

    if DATA_PARTITION_FACTOR == 2 and BLOCK_SIZE_M == 256 and FLATTEN and EPILOGUE_SUBTILE in (1, 2):
        pytest.skip("Out of resources: tensor memory exceeded")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        # Set up tensor descriptors (swap dims for col-major so contiguous dim is last)
        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(
            C,
            C.shape,
            C.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            generate_subtiled_region=generate_subtiled_region,
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 2b: Static persistent matmul with a while-loop outer loop
# ============================================================================
@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
def test_tutorial09_matmul_tma_static_persistent_while_loop_warp_specialize(EPILOGUE_SUBTILE):
    """Test a static persistent matmul whose persistent outer loop is a while loop."""
    M, N, K = 2048, 2048, 256
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_static_persistent_ws_while[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        ttgir = kernel.asm["ttgir"]
        # The static-persistent outer loop is countable (`tile_id < num_tiles`,
        # `tile_id += NUM_SMS`), so triton-uplift-while-to-for rewrites the
        # `scf.while` into an `scf.for` at the TTIR stage.
        assert "scf.for" in ttgir, "Expected countable persistent outer loop to uplift to scf.for"
        assert "scf.while" not in ttgir, "Expected the countable while to be uplifted away"
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        # Blackwell lowers to tcgen5 MMA; Hopper lowers to wgmma (warp_group_dot).
        assert "ttng.tc_gen5_mma" in ttgir or "ttng.warp_group_dot" in ttgir, "Expected an MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"
        assert "ttng.clc_" not in ttgir, "Expected static persistent scheduling, not CLC"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 2c: Dynamic (work-stealing) persistent matmul with a while-loop outer loop
# ============================================================================
@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
def test_tutorial09_matmul_tma_dynamic_persistent_while_loop_warp_specialize(EPILOGUE_SUBTILE):
    """Dynamic persistent matmul: the while-loop tile id is claimed via atomic_add."""
    M, N, K = 2048, 2048, 256
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)
        # Shared work counter: CTAs claim initial tiles 0..NUM_SMS-1 by program id,
        # then atomically claim NUM_SMS, NUM_SMS+1, ... so the counter starts at
        # NUM_SMS.
        tile_counter = torch.full((1, ), NUM_SMS, dtype=torch.int32, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_dynamic_persistent_ws_while[grid](
            a_desc,
            b_desc,
            c_desc,
            tile_counter,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        ttgir = kernel.asm["ttgir"]
        assert "scf.while" in ttgir, "Expected persistent outer loop to lower to scf.while"
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        # Blackwell lowers to tcgen5 MMA; Hopper lowers to wgmma (warp_group_dot).
        assert "ttng.tc_gen5_mma" in ttgir or "ttng.warp_group_dot" in ttgir, "Expected an MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"
        assert "atomic" in ttgir, "Expected an atomic op driving the dynamic tile id"
        assert "ttng.clc_" not in ttgir, "Expected dynamic atomic scheduling, not CLC"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not is_blackwell(), reason="CLC requires Blackwell (SM100+)")
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
def test_tutorial09_matmul_tma_clc_persistent_while_loop_warp_specialize(EPILOGUE_SUBTILE):
    """Dynamic persistent matmul whose while-loop tile id is claimed via the core
    CLC tile scheduler (tl.clc_tile_scheduler) and warp-specialized (Blackwell)."""
    M, N, K = 2048, 2048, 256
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])

        # CLC launches one cluster per tile (over-subscribed) so running CTAs can
        # cancel/steal pending clusters.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

        kernel = matmul_kernel_tma_clc_persistent_ws_while[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        ttgir = kernel.asm["ttgir"]
        assert "scf.while" in ttgir, "Expected persistent outer loop to lower to scf.while"
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected a Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"
        assert "ttng.clc_try_cancel" in ttgir, "Expected CLC scheduling in IR"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 2e: Outer-loop AutoWS through a unified scheduler while.
# ============================================================================
_UNIFIED_OUTER_AUTOWS_CONFIGS = [
    pytest.param(tl.NonPersistentScheduler, 128, 128, 1, 1, 1, False, False, id="nonpersistent-baseline"),
    pytest.param(
        tl.NonPersistentScheduler,
        128,
        128,
        4,
        1,
        1,
        False,
        True,
        id="nonpersistent-epilogue-subtile-4",
    ),
    pytest.param(
        tl.NonPersistentScheduler,
        256,
        128,
        2,
        2,
        1,
        False,
        True,
        id="nonpersistent-data-partition-2-subtile-2",
    ),
    pytest.param(
        tl.NonPersistentScheduler,
        256,
        256,
        1,
        2,
        2,
        False,
        True,
        id="nonpersistent-2cta-data-partition-2",
    ),
    pytest.param(tl.StaticPersistent1DScheduler, 128, 128, 1, 1, 1, False, False, id="static-baseline"),
    pytest.param(
        tl.StaticPersistent1DScheduler,
        128,
        128,
        4,
        1,
        1,
        True,
        True,
        id="static-epilogue-subtile-4",
    ),
    pytest.param(
        tl.StaticPersistent1DScheduler,
        256,
        128,
        2,
        2,
        1,
        True,
        True,
        id="static-data-partition-2-subtile-2",
    ),
    pytest.param(
        tl.StaticPersistent1DScheduler,
        256,
        256,
        1,
        2,
        2,
        False,
        True,
        id="static-2cta-data-partition-2",
    ),
    pytest.param(tl.DynamicPersistent1DScheduler, 128, 128, 1, 1, 1, False, False, id="dynamic-subtile-1"),
    pytest.param(tl.DynamicPersistent1DScheduler, 128, 128, 2, 1, 1, False, False, id="dynamic-subtile-2"),
    pytest.param(tl.DynamicPersistent1DScheduler, 128, 128, 4, 1, 1, False, False, id="dynamic-subtile-4"),
    pytest.param(tl.ClcTileScheduler, 128, 128, 1, 1, 1, False, True, id="clc-subtile-1"),
    pytest.param(tl.ClcTileScheduler, 128, 128, 2, 1, 1, False, True, id="clc-subtile-2"),
    pytest.param(tl.ClcTileScheduler, 128, 128, 4, 1, 1, False, True, id="clc-subtile-4"),
]


@pytest.mark.skipif(not (is_hopper() or is_blackwell()), reason="Requires Hopper or Blackwell")
@pytest.mark.parametrize(
    "SCHEDULE,BLOCK_SIZE_M,BLOCK_SIZE_N,EPILOGUE_SUBTILE,DATA_PARTITION_FACTOR,NUM_CTAS,"
    "generate_subtiled_region,blackwell_only",
    _UNIFIED_OUTER_AUTOWS_CONFIGS,
)
def test_tutorial09_matmul_tma_unified_persistent_while_loop_warp_specialize(
    SCHEDULE,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    EPILOGUE_SUBTILE,
    DATA_PARTITION_FACTOR,
    NUM_CTAS,
    generate_subtiled_region,
    blackwell_only,
):
    """Exercise outer-loop AutoWS with each unified scheduler."""
    if blackwell_only and not is_blackwell():
        pytest.skip("Subtiled regions, BLOCK_M=256 data partitioning, and 2-CTA require Blackwell")
    if NUM_CTAS == 2 and SCHEDULE in (tl.DynamicPersistent1DScheduler, tl.ClcTileScheduler):
        pytest.skip("2-CTA is not supported for dynamic or CLC scheduling")

    M, N, K = 2048, 2048, 256
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    num_stages = 3
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.float16
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        num_tiles = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N // NUM_CTAS)
        num_tiles = triton.cdiv(num_tiles, NUM_CTAS) * NUM_CTAS
        if SCHEDULE in (tl.NonPersistentScheduler, tl.ClcTileScheduler):
            grid_size = num_tiles
        else:
            grid_size = min(NUM_SMS, num_tiles)
        tile_counter = torch.full((1, ), grid_size, dtype=torch.int32, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE])

        launch_options = {
            "num_stages": num_stages,
            "num_warps": num_warps,
            "generate_subtiled_region": generate_subtiled_region,
        }
        if NUM_CTAS == 2:
            launch_options["ctas_per_cga"] = (2, 1, 1)

        kernel = matmul_kernel_tma_unified_persistent_ws_while[(grid_size, )](
            a_desc,
            b_desc,
            c_desc,
            tile_counter,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            SCHEDULE=SCHEDULE,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            NUM_CTAS=NUM_CTAS,
            TWO_CTAS=NUM_CTAS == 2,
            SMEM_ALLOC_ALGO=1 if NUM_CTAS == 2 else None,
            SEPARATE_EPILOGUE_STORE=SCHEDULE is tl.StaticPersistent1DScheduler,
            **launch_options,
        )

        ttgir = kernel.asm["ttgir"]
        if SCHEDULE is tl.NonPersistentScheduler:
            # The non-persistent schedule's outer loop provably runs exactly once
            # (`_valid` flips True->False), so triton-simplify-single-trip-while
            # optimizes the `scf.while` away after TTGIR loop scheduling.
            assert "scf.while" not in ttgir, "Expected single-trip outer while to be optimized away"
        elif SCHEDULE is tl.StaticPersistent1DScheduler:
            # The static-persistent schedule is a countable loop (`_x < num_tiles`,
            # `_x += num_programs`); triton-uplift-while-to-for (after LICM hoists
            # num_tiles out of the before-region) rewrites it into an `scf.for`.
            assert "scf.while" not in ttgir, "Expected countable static-persistent while to uplift to scf.for"
        else:
            assert "scf.while" in ttgir, "Expected dynamic and CLC schedules to retain their outer while"
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir or "ttng.warp_group_dot" in ttgir, "Expected an MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"
        if SCHEDULE is tl.ClcTileScheduler:
            assert "ttng.clc_try_cancel" in ttgir, "Expected CLC scheduling in IR"
        else:
            assert "ttng.clc_" not in ttgir, "Expected non-CLC scheduling"
        if SCHEDULE is tl.DynamicPersistent1DScheduler:
            assert "atomic" in ttgir, "Expected an atomic op driving the dynamic tile id"

        mma_op = "ttng.tc_gen5_mma" if is_blackwell() else "ttng.warp_group_dot"
        assert ttgir.count(mma_op) >= DATA_PARTITION_FACTOR, "Expected one MMA per data partition"
        expected_stores = EPILOGUE_SUBTILE * DATA_PARTITION_FACTOR
        assert ttgir.count("ttng.async_tma_copy_local_to_global") >= expected_stores, (
            "Expected every epilogue subtile and data partition to emit a TMA store")
        if NUM_CTAS == 2:
            assert ttgir.count("two_ctas") >= DATA_PARTITION_FACTOR, "Expected 2-CTA MMA per data partition"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 3: matmul_kernel_descriptor_persistent warp specialization (device-side TMA)
# Tests both Flatten=True and Flatten=False
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("FLATTEN", [True, False])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("generate_subtiled_region", [True, False])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_descriptor_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    FLATTEN,
    EPILOGUE_SUBTILE,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    generate_subtiled_region,
    separate_epilogue_store,
):
    """Test matmul_kernel_descriptor_persistent with warp_specialize=True for both Flatten values."""

    if FLATTEN:
        pytest.skip("FLATTEN will not WarpSpecialize although it will otherwise pass.")

    # DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 256:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 256")

    if DATA_PARTITION_FACTOR == 1 and BLOCK_SIZE_M == 256 and num_stages == 3 and FLATTEN:
        pytest.skip("Out of resources: shared memory and/or tensor memory exceeded")

    # Use scope() to set use_meta_ws and automatically restore on exit
    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_descriptor_persistent_ws[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            generate_subtiled_region=generate_subtiled_region,
        )

        # Verify IR contains expected ops
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 4: Multi-copy epilogue buffers with epilogue subtiling
# Focused test for the Phase 4.5 memory planner feature: with algo 1 and
# numBuffers capped at 2, 4 epilogue channels share 2 buffer copies.
# FLATTEN=True is not supported because the flattened loop generates
# scf.IfOp with else blocks, which the autoWS pass cannot handle yet.
# ============================================================================
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_multi_epilogue_subtile():
    """Test multi-copy epilogue buffers: 4 epilogue channels with 2 buffer copies."""
    M, N, K = 8192, 8192, 8192
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    EPILOGUE_SUBTILE = 4
    num_stages = 2
    num_warps = 4

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(
            C,
            C.shape,
            C.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=False,
            A_COL_MAJOR=False,
            B_COL_MAJOR=False,
            DATA_PARTITION_FACTOR=1,
            SEPARATE_EPILOGUE_STORE=False,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Verify warp specialization actually ran (ttg.warp_return is only
        # emitted by the WS code partition pass)
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_return" in ttgir, "Expected warp specialization to run"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"

        # Verify correctness
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test 5: matmul_kernel_tma_persistent_ws_splitk (deterministic Split-K)
# Targets large-K, undersaturated-MN shapes where Split-K is the right call.
# Config matrix is intentionally narrow: one (BM, BN, BK) tile, FLATTEN=False,
# fixed num_stages/num_warps — vary only the Split-K-relevant axes.
# ============================================================================
@pytest.mark.parametrize(
    "M, N, K",
    [
        (256, 256, 65536),
    ],
)
@pytest.mark.parametrize("SPLIT_K", [2, 4, 8])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [1, 2, 4])
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_tutorial09_matmul_tma_persistent_warp_specialize_splitk(
    M,
    N,
    K,
    SPLIT_K,
    EPILOGUE_SUBTILE,
):
    """Test deterministic Split-K variant: workspace partial sums + reduce."""
    pytest.skip("FLATTEN=True temporarily disabled")
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    GROUP_SIZE_M = 8
    FLATTEN = False
    num_stages = 3
    num_warps = 4

    # Empty-trailing-split guard: kernel deadlocks if any split has 0 K-tiles.
    k_tiles = triton.cdiv(K, BLOCK_SIZE_K)
    k_per_split = triton.cdiv(k_tiles, SPLIT_K)
    if k_per_split * (SPLIT_K - 1) >= k_tiles:
        pytest.skip("SPLIT_K leaves trailing split empty (would deadlock kernel)")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        # TritonBench-style scaling: (randn + 1) / K keeps |C| ~ O(1)
        # regardless of K, so error doesn't grow with K and we can use
        # standard fp16 tolerances. The +1 avoids denormals.
        A = (torch.randn((M, K), dtype=dtype, device=device) + 1) / K
        B = (torch.randn((N, K), dtype=dtype, device=device) + 1) / K
        C = torch.empty((M, N), dtype=dtype, device=device)
        workspace = torch.empty((SPLIT_K * M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        a_desc = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
        b_desc = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
        ws_desc = TensorDescriptor(
            workspace,
            workspace.shape,
            workspace.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N // EPILOGUE_SUBTILE],
        )

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]) * META["SPLIT_K"],
        ), )

        kernel = matmul_kernel_tma_persistent_ws_splitk[grid](
            a_desc,
            b_desc,
            ws_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            SPLIT_K=SPLIT_K,
            FLATTEN=FLATTEN,
            num_stages=num_stages,
            num_warps=num_warps,
        )

        # Reduce SPLIT_K partial-sum slabs into final C.
        REDUCE_BM, REDUCE_BN = 32, 32
        reduce_grid = (triton.cdiv(M, REDUCE_BM), triton.cdiv(N, REDUCE_BN))
        _reduce_k_kernel[reduce_grid](
            workspace,
            C,
            M,
            N,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=REDUCE_BM,
            BLOCK_SIZE_N=REDUCE_BN,
            OUTPUT_DTYPE=tl.float16,
            num_warps=4,
        )

        # Verify IR contains warp_specialize
        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        # Verify correctness — TritonBench fp16 tolerances. Inputs are
        # scaled by 1/K so |C| ~ O(1) and error doesn't grow with K.
        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=1e-2, rtol=1e-1)


# ============================================================================
# Hopper Tests
# ============================================================================


# ============================================================================
# Hopper Test 1: matmul_kernel_tma warp specialization (K-loop based)
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [64, 128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("enable_pingpong", [False, True])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_hopper_matmul_tma_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    enable_pingpong,
    separate_epilogue_store,
):
    """Test matmul_kernel_tma with warp_specialize=True on Hopper (K-loop based)."""
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 128:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 128")

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )

        kernel = matmul_kernel_tma_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            pingpongAutoWS=enable_pingpong,
            maxRegAutoWS=208 if DATA_PARTITION_FACTOR > 1 else 248,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.warp_group_dot" in ttgir, "Expected Hopper MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Hopper Test 2: matmul_kernel_tma_persistent warp specialization (tile-loop)
# Hopper constraints: FLATTEN=False, EPILOGUE_SUBTILE=1
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [64, 128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("enable_pingpong", [False, True])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_hopper_matmul_tma_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    enable_pingpong,
    separate_epilogue_store,
):
    """Test matmul_kernel_tma_persistent with warp_specialize=True on Hopper.

    Hopper constraints: FLATTEN=False (not supported with WS), EPILOGUE_SUBTILE=1 (no TMEM).
    """
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 128:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 128")

    FLATTEN = False
    EPILOGUE_SUBTILE = 1

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        if A_col_major:
            a_desc = TensorDescriptor(A, [K, M], [M, 1], [BLOCK_SIZE_K, BLOCK_SIZE_M])
        else:
            a_desc = TensorDescriptor(A, [M, K], [K, 1], [BLOCK_SIZE_M, BLOCK_SIZE_K])
        if B_col_major:
            b_desc = TensorDescriptor(B, [K, N], [N, 1], [BLOCK_SIZE_K, BLOCK_SIZE_N])
        else:
            b_desc = TensorDescriptor(B, [N, K], [K, 1], [BLOCK_SIZE_N, BLOCK_SIZE_K])
        c_desc = TensorDescriptor(
            C,
            C.shape,
            C.stride(),
            [BLOCK_SIZE_M, BLOCK_SIZE_N],
        )

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_tma_persistent_ws[grid](
            a_desc,
            b_desc,
            c_desc,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            pingpongAutoWS=enable_pingpong,
            maxRegAutoWS=208 if DATA_PARTITION_FACTOR > 1 else 248,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.warp_group_dot" in ttgir, "Expected Hopper MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Hopper Test 3: matmul_kernel_descriptor_persistent warp specialization
# (device-side TMA descriptors)
# Hopper constraints: FLATTEN=False, EPILOGUE_SUBTILE=1
# ============================================================================
@pytest.mark.parametrize("M, N, K", [(8192, 8192, 1024)])
@pytest.mark.parametrize("BLOCK_SIZE_M", [64, 128])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128])
@pytest.mark.parametrize("BLOCK_SIZE_K", [64])
@pytest.mark.parametrize("num_stages", [3])
@pytest.mark.parametrize("num_warps", [4])
@pytest.mark.parametrize("A_col_major", [False, True])
@pytest.mark.parametrize("B_col_major", [False, True])
@pytest.mark.parametrize("DATA_PARTITION_FACTOR", [1, 2])
@pytest.mark.parametrize("enable_pingpong", [False, True])
@pytest.mark.parametrize("separate_epilogue_store", [True, False])
@pytest.mark.skipif(not is_hopper(), reason="Requires Hopper")
def test_hopper_matmul_descriptor_persistent_warp_specialize(
    M,
    N,
    K,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    num_stages,
    num_warps,
    A_col_major,
    B_col_major,
    DATA_PARTITION_FACTOR,
    enable_pingpong,
    separate_epilogue_store,
):
    """Test matmul_kernel_descriptor_persistent with warp_specialize=True on Hopper.

    Hopper constraints: FLATTEN=False (not supported with WS), EPILOGUE_SUBTILE=1 (no TMEM).
    """
    if DATA_PARTITION_FACTOR != 1 and BLOCK_SIZE_M != 128:
        pytest.skip("DATA_PARTITION_FACTOR != 1 requires BLOCK_SIZE_M == 128")

    FLATTEN = False
    EPILOGUE_SUBTILE = 1

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        if A_col_major:
            A = torch.randn((K, M), dtype=dtype, device=device).t()
        else:
            A = torch.randn((M, K), dtype=dtype, device=device)
        if B_col_major:
            B = torch.randn((K, N), dtype=dtype, device=device).t()
        else:
            B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_descriptor_persistent_ws[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
            NUM_SMS=NUM_SMS,
            FLATTEN=FLATTEN,
            A_COL_MAJOR=A_col_major,
            B_COL_MAJOR=B_col_major,
            DATA_PARTITION_FACTOR=DATA_PARTITION_FACTOR,
            SEPARATE_EPILOGUE_STORE=separate_epilogue_store,
            num_stages=num_stages,
            num_warps=num_warps,
            pingpongAutoWS=enable_pingpong,
            maxRegAutoWS=208 if DATA_PARTITION_FACTOR > 1 else 248,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.warp_group_dot" in ttgir, "Expected Hopper MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)


# ============================================================================
# Test: num_warps=8 warp specialization
# Smoke test to verify WS works with num_warps=8.
# ============================================================================
@pytest.mark.skipif(not is_blackwell(), reason="Requires Blackwell")
def test_warp_specialize_num_warps_8():
    """Test warp specialization with num_warps=8."""
    M, N, K = 512, 512, 256
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64

    with triton.knobs.nvidia.scope():
        triton.knobs.nvidia.use_meta_ws = True
        triton.knobs.nvidia.use_meta_partition = True

        dtype = torch.float16
        GROUP_SIZE_M = 8
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        device = "cuda"

        torch.manual_seed(42)
        A = torch.randn((M, K), dtype=dtype, device=device)
        B = torch.randn((N, K), dtype=dtype, device=device)
        C = torch.empty((M, N), dtype=dtype, device=device)

        def alloc_fn(size, align, stream):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        grid = lambda META: (min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ), )

        kernel = matmul_kernel_descriptor_persistent_ws[grid](
            A,
            B,
            C,
            M,
            N,
            K,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            EPILOGUE_SUBTILE=1,
            NUM_SMS=NUM_SMS,
            FLATTEN=False,
            A_COL_MAJOR=False,
            B_COL_MAJOR=False,
            DATA_PARTITION_FACTOR=1,
            SEPARATE_EPILOGUE_STORE=False,
            num_stages=2,
            num_warps=8,
        )

        ttgir = kernel.asm["ttgir"]
        assert "ttg.warp_specialize" in ttgir, "Expected warp specialization in IR"
        assert "ttng.tc_gen5_mma" in ttgir, "Expected Blackwell MMA instruction"
        assert "ttng.async_tma_copy_global_to_local" in ttgir, "Expected TMA copy"

        ref_out = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(dtype)
        torch.testing.assert_close(ref_out, C, atol=0.03, rtol=0.03)
