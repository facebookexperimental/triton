# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic utility functions for TLX Triton kernels.

This module provides reusable @triton.jit helper functions for common patterns
across TLX kernels, including:
- Tile and grid computation utilities
- Phase and buffer index management
- Accumulator initialization and loading
- Async load/store patterns
- Barrier management utilities

These utilities are designed to reduce code duplication across TLX tutorials
and can be imported into any Triton kernel that uses the TLX extensions.

Usage:
    from tlx_kernel_utils import (
        compute_tile_position,
        compute_grid_info,
        get_buffer_index_and_phase,
        flip_phase_on_boundary,
        ...
    )
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


# =============================================================================
# Tile and Grid Computation Utilities
# =============================================================================


@triton.jit
def compute_tile_position(
    tile_id,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute tile position (pid_m, pid_n) from a flattened tile_id using grouped ordering.

    This implements the swizzled tile ordering pattern commonly used in GEMM kernels
    to improve L2 cache locality by grouping tiles that access nearby memory regions.

    Args:
        tile_id: Flattened tile index (0 to num_tiles-1)
        num_pid_in_group: Number of tiles per group (GROUP_SIZE_M * num_pid_n)
        num_pid_m: Total number of tiles in M dimension
        GROUP_SIZE_M: Number of M-tiles per group (constexpr)

    Returns:
        tuple: (pid_m, pid_n) - tile coordinates in M and N dimensions
    """
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def compute_grid_info(
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Compute common grid dimension information used across async tasks.

    This centralizes the grid computation logic that is typically duplicated
    across producer, consumer, and epilogue tasks in warp-specialized kernels.

    Args:
        M, N, K: Matrix dimensions
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes (constexpr)
        GROUP_SIZE_M: Number of M-tiles per group for swizzling (constexpr)

    Returns:
        tuple: (start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles)
    """
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    return start_pid, num_pid_m, num_pid_n, num_pid_in_group, num_tiles, k_tiles


@triton.jit
def compute_tile_offsets(
    pid_m,
    pid_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Compute memory offsets from tile coordinates.

    Args:
        pid_m, pid_n: Tile coordinates
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes (constexpr)

    Returns:
        tuple: (offs_am, offs_bn) - byte offsets for A's M-dimension and B's N-dimension
    """
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    return offs_am, offs_bn


@triton.jit
def compute_tile_offsets_2cta(
    pid_m,
    pid_n,
    cluster_cta_rank,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Compute memory offsets for 2-CTA mode where B is split between CTAs.

    In 2-CTA mode, each CTA loads half of the B matrix, so the offset is adjusted
    based on the CTA rank within the cluster.

    Args:
        pid_m, pid_n: Tile coordinates
        cluster_cta_rank: Rank of this CTA within the cluster (0 or 1)
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile sizes (constexpr)

    Returns:
        tuple: (offs_am, offs_bn) - byte offsets adjusted for 2-CTA split
    """
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N + cluster_cta_rank * (BLOCK_SIZE_N // 2)
    return offs_am, offs_bn


# =============================================================================
# Phase and Buffer Index Management
# =============================================================================


@triton.jit
def get_buffer_index_and_phase(accum_cnt, NUM_BUFFERS: tl.constexpr):
    """
    Map an accumulation counter to a buffer index and phase.

    This is the standard pattern for multi-buffered producer-consumer synchronization.
    The phase flips each time we complete a full round through all buffers.

    Args:
        accum_cnt: Running count of operations (tiles processed, k-iterations, etc.)
        NUM_BUFFERS: Number of buffers in the ring (constexpr)

    Returns:
        tuple: (buf_idx, phase)
            - buf_idx: Buffer index in [0, NUM_BUFFERS)
            - phase: Current phase (0 or 1), flips each full round
    """
    buf_idx = accum_cnt % NUM_BUFFERS
    phase = (accum_cnt // NUM_BUFFERS) & 1
    return buf_idx, phase


@triton.jit
def flip_phase_on_boundary(current_phase, buf_idx, NUM_BUFFERS: tl.constexpr):
    """
    Flip the phase when buffer index wraps around to complete a round.

    This is the common phase update pattern: phase only flips when we've
    completed a full cycle through all buffers (buf_idx == NUM_BUFFERS - 1).

    Args:
        current_phase: Current phase value (0 or 1)
        buf_idx: Current buffer index
        NUM_BUFFERS: Total number of buffers (constexpr)

    Returns:
        New phase value (flipped if at boundary, unchanged otherwise)
    """
    return current_phase ^ (buf_idx == NUM_BUFFERS - 1)


@triton.jit
def compute_k_offset(k_iter, BLOCK_SIZE_K: tl.constexpr):
    """
    Compute the K-dimension offset for a given iteration.

    Args:
        k_iter: Current K iteration index
        BLOCK_SIZE_K: Tile size in K dimension (constexpr)

    Returns:
        K-dimension offset in elements
    """
    return k_iter * BLOCK_SIZE_K


# =============================================================================
# Accumulator Utilities
# =============================================================================


@triton.jit
def init_tmem_accumulator(
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Allocate and initialize a single TMEM accumulator buffer with zeros.

    This is the common pattern for non-persistent kernels that need one accumulator.

    Args:
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile dimensions (constexpr)

    Returns:
        acc_tmem: TMEM buffer view initialized to zeros
    """
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    buffers = tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        tl.constexpr(1),
        tlx.storage_kind.tmem,
    )
    acc_tmem = tlx.local_view(buffers, 0)
    tlx.local_store(acc_tmem, accumulator)
    return acc_tmem


@triton.jit
def alloc_multi_tmem_buffers(
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """
    Allocate multiple TMEM accumulator buffers for overlapping MMA and epilogue.

    Used in persistent/warp-specialized kernels where we overlap computation
    of one tile with epilogue processing of another.

    Args:
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile dimensions (constexpr)
        NUM_BUFFERS: Number of TMEM buffers to allocate (constexpr)

    Returns:
        tmem_buffers: Multi-buffer TMEM allocation
    """
    return tlx.local_alloc(
        (BLOCK_SIZE_M, BLOCK_SIZE_N),
        tl.float32,
        NUM_BUFFERS,
        tlx.storage_kind.tmem,
    )


@triton.jit
def load_and_convert_accumulator(acc_tmem):
    """
    Load accumulator from TMEM to registers and convert to float16.

    This is the standard epilogue pattern: load fp32 accumulator and
    downcast to fp16 for output.

    Args:
        acc_tmem: TMEM buffer containing fp32 accumulator

    Returns:
        fp16 tensor ready for storage
    """
    result = tlx.local_load(acc_tmem)
    return result.to(tl.float16)


# =============================================================================
# SMEM Buffer Allocation
# =============================================================================


@triton.jit
def alloc_smem_buffers_ab(
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """
    Allocate SMEM ring buffers for A and B matrices.

    Args:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions (constexpr)
        NUM_BUFFERS: Number of buffers in the ring (constexpr)

    Returns:
        tuple: (buffers_A, buffers_B)
    """
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tl.float16, NUM_BUFFERS)
    return buffers_A, buffers_B


@triton.jit
def alloc_smem_buffers_ab_2cta(
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """
    Allocate SMEM ring buffers for A and B matrices in 2-CTA mode.

    In 2-CTA mode, each CTA only loads half of B, so B buffers are half-sized.

    Args:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions (constexpr)
        NUM_BUFFERS: Number of buffers in the ring (constexpr)

    Returns:
        tuple: (buffers_A, buffers_B) where B is half-sized for 2-CTA
    """
    buffers_A = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)
    buffers_B = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N // 2), tl.float16, NUM_BUFFERS)
    return buffers_A, buffers_B


# =============================================================================
# Barrier Allocation Utilities
# =============================================================================


@triton.jit
def alloc_producer_consumer_barriers(NUM_BUFFERS: tl.constexpr):
    """
    Allocate full/empty barrier pairs for producer-consumer synchronization.

    This is the standard double-buffering barrier pattern:
    - full_bars: Producer signals when buffer is full (ready to consume)
    - empty_bars: Consumer signals when buffer is empty (ready to reuse)

    Args:
        NUM_BUFFERS: Number of barrier pairs to allocate (constexpr)

    Returns:
        tuple: (full_bars, empty_bars)
    """
    full_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    empty_bars = tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=1)
    return full_bars, empty_bars


@triton.jit
def alloc_2cta_sync_barriers(NUM_BUFFERS: tl.constexpr):
    """
    Allocate barriers for 2-CTA synchronization.

    These barriers coordinate between the two CTAs in a cluster, typically
    used to ensure both CTAs have loaded their portion of data before MMA.

    Args:
        NUM_BUFFERS: Number of barriers to allocate (constexpr)

    Returns:
        cta_bars: Barriers with arrive_count=2 for 2-CTA sync
    """
    return tlx.alloc_barriers(num_barriers=NUM_BUFFERS, arrive_count=2)


# =============================================================================
# Async Load Utilities
# =============================================================================


@triton.jit
def async_load_ab_tiles(
    a_desc,
    b_desc,
    buf_a,
    buf_b,
    offs_am,
    offs_k,
    offs_bn,
    load_bar,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Issue async TMA loads for both A and B tiles with barrier signaling.

    This is the common pattern for loading a (k-iteration's worth of) A and B tiles
    together with proper barrier setup.

    Args:
        a_desc, b_desc: TMA tensor descriptors for A and B
        buf_a, buf_b: SMEM buffer views to load into
        offs_am: A matrix M-dimension offset
        offs_k: K-dimension offset (shared by A and B)
        offs_bn: B matrix N-dimension offset
        load_bar: Barrier to signal when loads complete
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions (constexpr)
    """
    tlx.barrier_expect_bytes(
        load_bar,
        2 * (BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K,
    )  # float16
    tlx.async_descriptor_load(a_desc, buf_a, [offs_am, offs_k], load_bar)
    tlx.async_descriptor_load(b_desc, buf_b, [offs_k, offs_bn], load_bar)


@triton.jit
def async_load_ab_tiles_2cta(
    a_desc,
    b_desc,
    buf_a,
    buf_b,
    offs_am,
    offs_k,
    offs_bn,
    load_bar,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Issue async TMA loads for A and B tiles in 2-CTA mode.

    In 2-CTA mode, each CTA loads half of B, so the expected bytes are adjusted.

    Args:
        a_desc, b_desc: TMA tensor descriptors
        buf_a, buf_b: SMEM buffer views (B is half-sized)
        offs_am, offs_k, offs_bn: Tile offsets (offs_bn already adjusted for 2-CTA)
        load_bar: Barrier to signal when loads complete
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions (constexpr)
    """
    tlx.barrier_expect_bytes(
        load_bar,
        2 * (BLOCK_SIZE_M + BLOCK_SIZE_N // 2) * BLOCK_SIZE_K,
    )  # float16
    tlx.async_descriptor_load(a_desc, buf_a, [offs_am, offs_k], load_bar)
    tlx.async_descriptor_load(b_desc, buf_b, [offs_k, offs_bn], load_bar)


# =============================================================================
# CLC (Cluster Launch Control) Utilities
# =============================================================================


@triton.jit
def clc_advance_producer(
    clc_context,
    clc_buf,
    clc_phase,
    NUM_CLC_STAGES: tl.constexpr,
    multi_ctas: tl.constexpr = False,
):
    """
    Produce work item via CLC and advance phase.

    Args:
        clc_context: CLC context handle
        clc_buf: Current CLC buffer index
        clc_phase: Current CLC phase
        NUM_CLC_STAGES: Number of CLC stages (constexpr)
        multi_ctas: Whether using multi-CTA mode (constexpr)

    Returns:
        New phase value after producing
    """
    tlx.clc_producer(clc_context, clc_buf, clc_phase, multi_ctas=multi_ctas)
    return clc_phase ^ (clc_buf == (NUM_CLC_STAGES - 1))


@triton.jit
def clc_advance_consumer(
    clc_context,
    clc_buf,
    clc_phase,
    NUM_CLC_STAGES: tl.constexpr,
    multi_ctas: tl.constexpr = False,
):
    """
    Consume work item via CLC and advance phase/buffer.

    Args:
        clc_context: CLC context handle
        clc_buf: Current CLC buffer index
        clc_phase: Current CLC phase
        NUM_CLC_STAGES: Number of CLC stages (constexpr)
        multi_ctas: Whether using multi-CTA mode (constexpr)

    Returns:
        tuple: (tile_id, new_phase, new_buf)
            - tile_id: Next tile to process (-1 if done)
            - new_phase: Updated phase value
            - new_buf: Updated buffer index
    """
    tile_id = tlx.clc_consumer(clc_context, clc_buf, clc_phase, multi_ctas=multi_ctas)
    new_phase = clc_phase ^ (clc_buf == (NUM_CLC_STAGES - 1))
    new_buf = (clc_buf + 1) % NUM_CLC_STAGES
    return tile_id, new_phase, new_buf


# =============================================================================
# Epilogue Store Utilities
# =============================================================================


@triton.jit
def epilogue_store_full(
    acc_tmem,
    c_desc,
    offs_am,
    offs_bn,
):
    """
    Standard full epilogue: load accumulator and store entire tile.

    Args:
        acc_tmem: TMEM accumulator buffer
        c_desc: TMA descriptor for C matrix
        offs_am, offs_bn: Output tile offsets
    """
    result = tlx.local_load(acc_tmem)
    c = result.to(tl.float16)
    c_desc.store([offs_am, offs_bn], c)


@triton.jit
def epilogue_store_subtiled(
    acc_tmem,
    c_desc,
    offs_am,
    offs_bn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NUM_SUBTILES: tl.constexpr,
):
    """
    Subtiled epilogue: load and store in chunks to reduce SMEM pressure.

    Splits the output tile into NUM_SUBTILES slices along N dimension,
    loading and storing each slice sequentially to reduce peak SMEM usage.

    Args:
        acc_tmem: TMEM accumulator buffer
        c_desc: TMA descriptor for C matrix
        offs_am, offs_bn: Output tile offsets
        BLOCK_SIZE_M, BLOCK_SIZE_N: Tile dimensions (constexpr)
        NUM_SUBTILES: Number of subtiles to split into (constexpr)
    """
    slice_size: tl.constexpr = BLOCK_SIZE_N // NUM_SUBTILES
    for slice_id in tl.static_range(NUM_SUBTILES):
        acc_tmem_subslice = tlx.local_slice(
            acc_tmem,
            [0, slice_id * slice_size],
            [BLOCK_SIZE_M, slice_size],
        )
        result = tlx.local_load(acc_tmem_subslice)
        c = result.to(tl.float16)
        c_desc.store([offs_am, offs_bn + slice_id * slice_size], c)


@triton.jit
def epilogue_store_half_half(
    acc_tmem,
    c_desc,
    offs_am,
    offs_bn,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Two-pass epilogue: load and store in two halves.

    Uses tlx.subslice (column-based) instead of local_slice for simpler
    half-by-half processing. Reduces SMEM pressure compared to full store.

    Args:
        acc_tmem: TMEM accumulator buffer
        c_desc: TMA descriptor for C matrix
        offs_am, offs_bn: Output tile offsets
        BLOCK_SIZE_N: Tile N dimension (constexpr)
    """
    # First half
    acc_tmem_subslice1 = tlx.subslice(acc_tmem, 0, BLOCK_SIZE_N // 2)
    result = tlx.local_load(acc_tmem_subslice1)
    c = result.to(tl.float16)
    c_desc.store([offs_am, offs_bn], c)

    # Second half
    acc_tmem_subslice2 = tlx.subslice(acc_tmem, BLOCK_SIZE_N // 2, BLOCK_SIZE_N // 2)
    result = tlx.local_load(acc_tmem_subslice2)
    c = result.to(tl.float16)
    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c)


# =============================================================================
# MMA (Matrix Multiply Accumulate) Utilities
# =============================================================================


@triton.jit
def async_mma_with_barrier(
    buf_a,
    buf_b,
    acc_tmem,
    use_acc,
    done_barrier,
    two_ctas: tl.constexpr = False,
):
    """
    Issue async MMA operation with barrier signaling on completion.

    Args:
        buf_a, buf_b: SMEM buffers containing A and B tiles
        acc_tmem: TMEM accumulator buffer
        use_acc: Whether to accumulate (True) or overwrite (False for k=0)
        done_barrier: Barrier to signal when MMA completes
        two_ctas: Whether using 2-CTA mode (constexpr)
    """
    tlx.async_dot(
        buf_a,
        buf_b,
        acc_tmem,
        use_acc=use_acc,
        mBarriers=[done_barrier],
        two_ctas=two_ctas,
        out_dtype=tl.float32,
    )


@triton.jit
def async_mma_first_iter(
    buf_a,
    buf_b,
    acc_tmem,
    done_barrier,
    two_ctas: tl.constexpr = False,
):
    """
    Issue async MMA for first K iteration (no accumulation).

    Args:
        buf_a, buf_b: SMEM buffers containing A and B tiles
        acc_tmem: TMEM accumulator buffer
        done_barrier: Barrier to signal when MMA completes
        two_ctas: Whether using 2-CTA mode (constexpr)
    """
    tlx.async_dot(
        buf_a,
        buf_b,
        acc_tmem,
        use_acc=False,
        mBarriers=[done_barrier],
        two_ctas=two_ctas,
        out_dtype=tl.float32,
    )


@triton.jit
def async_mma_accumulate(
    buf_a,
    buf_b,
    acc_tmem,
    done_barrier,
    two_ctas: tl.constexpr = False,
):
    """
    Issue async MMA with accumulation (for k > 0 iterations).

    Args:
        buf_a, buf_b: SMEM buffers containing A and B tiles
        acc_tmem: TMEM accumulator buffer
        done_barrier: Barrier to signal when MMA completes
        two_ctas: Whether using 2-CTA mode (constexpr)
    """
    tlx.async_dot(
        buf_a,
        buf_b,
        acc_tmem,
        use_acc=True,
        mBarriers=[done_barrier],
        two_ctas=two_ctas,
        out_dtype=tl.float32,
    )
