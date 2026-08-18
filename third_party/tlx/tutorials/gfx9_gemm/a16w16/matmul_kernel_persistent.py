"""Persistent gfx950 FP16 GEMM family for K64 and 32x32 fragments.

Tile geometry and the fine-grained pipeline order are compile-time specs.  Each
four-wave workgroup computes two adjacent N tiles with a K64 double-buffered
pipeline, decomposing N into N128 LDS groups plus unpadded N32 tails.
"""

import os

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


TILE = tl.constexpr(32)
BLOCK_K = tl.constexpr(64)
K_BLOCKS = tl.constexpr(96)
N_GROUP_FRAGMENTS = tl.constexpr(4)

# (BLOCK_M, BLOCK_N, NUM_PID_N, NUM_PROGRAMS, TILES_PER_PROGRAM)
_MT256X160_TILE_SPEC = (256, 160, 128, 256, 2)
_MT256X192_TILE_SPEC = (256, 192, 128, 256, 2)

# The pipeline schedule is compile-time data, separate from the reusable load,
# LDS, MFMA, persistent traversal, and epilogue machinery below.
_M8_A1_READ_PLAN = (
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 3),
    (3, 3),
    (6, 2),
    (0, 0),
)
_MT256X160_B_PUBLISH_PLAN = (
    ((2, 0, 3), (-1, 0, 0)),
    ((2, 3, 2), (3, 0, 1)),
    ((3, 1, 4), (-1, 0, 0)),
    ((4, 0, 3), (-1, 0, 0)),
    ((1, 0, 2), (4, 3, 2)),
)
_MT256X160_FINISH_MFMA_PLAN = (
    7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
)
_MT256X160_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    4,
    _MT256X160_B_PUBLISH_PLAN,
    _MT256X160_FINISH_MFMA_PLAN,
)

# N192 reuses the same N128-plus-tail decomposition with two independent N32
# tail images.  B publishes cover rows 1-3 and the first four columns of row 4;
# the final LDS reads cover the rest of row 4 and rows 5-6.
_MT256X192_B_PUBLISH_PLAN = (
    ((1, 0, 4), (-1, 0, 0)),
    ((1, 4, 2), (2, 0, 2)),
    ((2, 2, 4), (-1, 0, 0)),
    ((3, 0, 4), (-1, 0, 0)),
    ((3, 4, 2), (4, 0, 2)),
    ((4, 2, 2), (-1, 0, 0)),
)
_MT256X192_FINISH_MFMA_PLAN = tuple(range(28, 42))
_MT256X192_PIPELINE_SPEC = (
    _M8_A1_READ_PLAN,
    5,
    _MT256X192_B_PUBLISH_PLAN,
    _MT256X192_FINISH_MFMA_PLAN,
)

_SPECIALIZATIONS = {
    "mt256x160": (
        (1024, 20480, 6144),
        _MT256X160_TILE_SPEC,
        _MT256X160_PIPELINE_SPEC,
    ),
    "mt256x192": (
        (1024, 24576, 6144),
        _MT256X192_TILE_SPEC,
        _MT256X192_PIPELINE_SPEC,
    ),
}
_SHAPE_DEFAULTS = {
    (1024, 20480, 6144): "mt256x160",
    (1024, 24576, 6144): "mt256x192",
}

# Every complete group of four N32 fragments shares one N128 LDS image.  Any
# remaining fragments use separate N32 images, avoiding power-of-two padding.
_B_BASES = tl.constexpr(
    [[1 << bit, 0] for bit in range(6)]
    + [[0, 1 << bit] for bit in (4, 5, 6, 0, 1, 2, 3)]
)

# Four waves jointly own the first four N32 accumulator fragments as one 32x128
# region.  Each lane gets sixteen contiguous N values, allowing four narrow
# stores to become two 128-bit stores after one layout conversion.
_C_STORE_32X128_LAYOUT = tlx.layout(
    shape=((16, 4, 2, 2), (16,)),
    stride=((128, 16, 2048, 64), (1,)),
)

@triton.jit
def _global_loads(source, kb, tile_spec: tl.constexpr):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    a_ptr += kb * BLOCK_K * stride_ak
    b_ptr += kb * BLOCK_K * stride_bk
    a = [
        tlx.buffer_load(a_ptr, a_offsets[mi], contiguity=8)
        for mi in range(tl.constexpr(tile_spec[0] // TILE))
    ]
    b = [
        tlx.buffer_load(b_ptr, b_offsets[nj], contiguity=8)
        for nj in range(tl.constexpr(tile_spec[1] // TILE))
    ]
    return tl.tuple(a + b)


@triton.jit
def _local_store_one(stage, value, index: tl.constexpr, tile_spec: tl.constexpr):
    a_buffers, b_buffers = stage
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    if index < m_fragments:
        tlx.local_store(tlx.local_view(a_buffers[index], 0), value)
    else:
        nj: tl.constexpr = index - m_fragments
        view = tlx.local_slice(
            tlx.local_view(
                b_buffers[
                    nj // N_GROUP_FRAGMENTS
                    if nj < n_main_fragments
                    else n_main_groups + nj - n_main_fragments
                ],
                0,
            ),
            [
                0,
                (nj % N_GROUP_FRAGMENTS) * TILE
                if nj < n_main_fragments
                else 0,
            ],
            [BLOCK_K, TILE],
        )
        tlx.local_store(view, value)


@triton.jit
def _local_store_all(stage, values, tile_spec: tl.constexpr):
    load_groups: tl.constexpr = tl.constexpr(
        tile_spec[0] // TILE + tile_spec[1] // TILE
    )
    for index in tl.static_range(load_groups):
        _local_store_one(stage, values[index], index, tile_spec)


@triton.jit
def _local_load_a(
    stage, kh: tl.constexpr, mi: tl.constexpr, dot_a: tl.constexpr
):
    a_buffers = stage[0]
    view = tlx.local_slice(
        tlx.local_view(a_buffers[mi], 0),
        [0, kh * TILE],
        [TILE, TILE],
    )
    return tlx.require_layout(tlx.local_load(view, relaxed=True), dot_a, pin=False)


@triton.jit
def _local_load_b(
    stage,
    kh: tl.constexpr,
    nj: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_buffers = stage[1]
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    view = tlx.local_slice(
        tlx.local_view(
            b_buffers[
                nj // N_GROUP_FRAGMENTS
                if nj < n_main_fragments
                else n_main_groups + nj - n_main_fragments
            ],
            0,
        ),
        [
            kh * TILE,
            (nj % N_GROUP_FRAGMENTS) * TILE
            if nj < n_main_fragments
            else 0,
        ],
        [TILE, TILE],
    )
    value = tlx.local_load(view, relaxed=True)
    return tlx.require_layout(value, dot_b, pin=False)


@triton.jit
def _local_load_b_row(
    stage,
    kh: tl.constexpr,
    dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return tl.tuple([
        _local_load_b(stage, kh, nj, dot_b, tile_spec)
        for nj in range(tl.constexpr(tile_spec[1] // TILE))
    ])


@triton.jit
def _mfma_part(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    first_nj: tl.constexpr,
    count: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    accumulators: tl.constexpr = tl.constexpr(
        tile_spec[0] // TILE * tile_spec[1] // TILE
    )
    return tl.tuple([
        tlx.amd_scheduled_mfma(
            tlx.require_layout(a_operand, dot_a, pin=False),
            tlx.require_layout(
                b_operands[index % n_fragments], dot_b, pin=False
            ),
            tlx.require_layout(acc[index], mma, pin=False),
            accumulator_role="persistent",
            resident_operand=None,
            initialize=initialize,
        )
        if (
            index // n_fragments == mi
            and index % n_fragments >= first_nj
            and index % n_fragments < first_nj + count
        )
        else acc[index]
        for index in range(accumulators)
    ])


@triton.jit
def _mfma_row(
    a_operand,
    b_operands,
    acc,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    tile_spec: tl.constexpr,
):
    return _mfma_part(
        a_operand, b_operands, acc, mi, 0,
        tl.constexpr(tile_spec[1] // TILE),
        mma, dot_a, dot_b, initialize, tile_spec,
    )


@triton.jit
def _global_prefetch_one(
    source, kb, index: tl.constexpr, tile_spec: tl.constexpr
):
    a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets = source
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    if index < m_fragments:
        return tlx.buffer_load(
            a_ptr + kb * BLOCK_K * stride_ak,
            a_offsets[index],
            contiguity=8,
        )
    else:
        return tlx.buffer_load(
            b_ptr + kb * BLOCK_K * stride_bk,
            b_offsets[index - m_fragments],
            contiguity=8,
        )


@triton.jit
def _publish_a_row(
    a_operand,
    b_operands,
    acc,
    old_value,
    next_stage,
    source,
    future_kb,
    mi: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance one A fragment across three K64 pipeline generations.

    ``a_operand`` and ``b_operands`` compute one accumulator row for K(t),
    ``old_value`` is A(K(t+1)) already prefetched in VGPRs, and ``future``
    becomes A(K(t+2)) in VGPRs.  The K(t) MFMA row is split around the future
    global load to balance load-latency coverage against VGPR lifetime.
    """
    # One accumulator row contains BLOCK_N / 32 logical C[32, 32]
    # fragments.  This is 5 fragments for N160 and 6 for N192.
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # K(t+1): retire the previously prefetched A[32, 64] value from VGPRs
    # into the next LDS stage.
    _local_store_one(next_stage, old_value, mi, tile_spec)

    # K(t): compute columns [0, mfmas_before_prefetch) of accumulator row mi.
    acc = _mfma_part(
        a_operand, b_operands, acc, mi, 0,
        tl.constexpr(pipeline_spec[1]),
        mma, dot_a, dot_b, initialize, tile_spec,
    )

    # K(t+2): issue this row's next global A[32, 64] load into VGPRs.
    future = _global_prefetch_one(source, future_kb, mi, tile_spec)

    # K(t): compute the remaining columns
    # [mfmas_before_prefetch, n_fragments).  For N160 the split is 4 + 1;
    # for N192 it is 5 + 1.
    acc = _mfma_part(
        a_operand, b_operands, acc, mi,
        tl.constexpr(pipeline_spec[1]),
        n_fragments - tl.constexpr(pipeline_spec[1]),
        mma, dot_a, dot_b, initialize, tile_spec,
    )
    return acc, future


@triton.jit
def _publish_b_fragment(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    nj: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    index: tl.constexpr = m_fragments + nj
    _local_store_one(next_stage, prefetched[index], index, tile_spec)
    acc = _mfma_part(
        a_operands[0], b_operands, acc,
        0, nj, 1, mma, dot_a, dot_b, False, tile_spec,
    )

    future = _global_prefetch_one(source, future_kb, index, tile_spec)

    # Spread the MFMA rows across publish groups according to a tile-specific
    # compile-time plan, so the pipeline core itself is independent of 8x5.
    for part in tl.static_range(
        tl.constexpr(len(pipeline_spec[2][nj]))
    ):
        if tl.constexpr(pipeline_spec[2][nj][part][2]) > 0:
            acc = _mfma_part(
                a_operands[tl.constexpr(pipeline_spec[2][nj][part][0])],
                b_operands,
                acc,
                tl.constexpr(pipeline_spec[2][nj][part][0]),
                tl.constexpr(pipeline_spec[2][nj][part][1]),
                tl.constexpr(pipeline_spec[2][nj][part][2]),
                mma,
                dot_a,
                dot_b,
                False,
                tile_spec,
            )
    return acc, future


@triton.jit
def _publish_b_rows(
    a_operands,
    b_operands,
    acc,
    prefetched,
    next_stage,
    source,
    future_kb,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Publish B fragments and issue their tile-specific K1 MFMA spans."""
    future = tl.tuple([])
    for nj in tl.static_range(tl.constexpr(tile_spec[1] // TILE)):
        acc, value = _publish_b_fragment(
            a_operands, b_operands, acc, prefetched,
            next_stage, source, future_kb, nj, mma, dot_a, dot_b,
            pipeline_spec, tile_spec,
        )
        future += tl.tuple([value])
    return acc, future


@triton.jit
def _finish_read(
    next_stage,
    a_operands,
    b_operands,
    acc,
    read: tl.constexpr,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    if read < n_fragments:
        value = _local_load_b(next_stage, 0, read, dot_b, tile_spec)
    else:
        value = _local_load_a(next_stage, 0, read - n_fragments, dot_a)
    flat: tl.constexpr = tl.constexpr(pipeline_spec[3][read])
    acc = _mfma_part(
        a_operands[flat // n_fragments], b_operands, acc,
        flat // n_fragments, flat % n_fragments, 1,
        mma, dot_a, dot_b, False, tile_spec,
    )
    return acc, value


@triton.jit
def _finish_iteration(
    next_stage,
    a_operands,
    b_operands,
    acc,
    early_prefetched,
    late_prefetched,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Pair next-stage LDS reads with the tile-specific final MFMA sequence."""
    next_a = tl.tuple([])
    next_b = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for read in tl.static_range(m_fragments + n_fragments):
        acc, value = _finish_read(
            next_stage, a_operands, b_operands,
            acc, read, mma, dot_a, dot_b, pipeline_spec, tile_spec,
        )
        if read < n_fragments:
            next_b += tl.tuple([value])
        else:
            next_a += tl.tuple([value])
    acc = _mfma_row(
        a_operands[m_fragments - 1], b_operands, acc, m_fragments - 1,
        mma, dot_a, dot_b, False, tile_spec,
    )
    return acc, early_prefetched + late_prefetched, next_a, next_b


@triton.jit
def _pipeline(
    source,
    future_kb,
    current_stage,
    next_stage,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Advance the manual pipeline by one K64 block.

    Entry state:
      * ``current_stage`` contains K(t), while ``current_a``/``current_b``
        already hold its first K32 half (kh=0) in operand VGPRs.
      * ``prefetched`` contains the complete K(t+1) A/B tile in VGPRs.
      * ``next_stage`` is available to receive K(t+1).

    Exit state:
      * K(t) has been fully accumulated.
      * ``next_stage`` contains K(t+1), whose kh=0 operands are returned.
      * the complete K(t+2) A/B tile is returned in prefetch VGPRs.
    """
    # Future global prefetches for K(t+2).  ``a1`` and ``b1`` are not K(t+1):
    # they are the second K32 half (kh=1) of the current K(t) LDS stage.
    future_a = tl.tuple([])
    a1 = tl.tuple([])
    b1 = tl.tuple([])
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    for mi in tl.static_range(m_fragments):
        # K(t).kh1: spread the smaller B operand window across the first
        # n_fragments loop iterations instead of issuing one late LDS burst.
        if mi < n_fragments:
            b1 += tl.tuple([
                _local_load_b(current_stage, 1, mi, dot_b, tile_spec)
            ])

        # In one interleaved step: publish A(mi,K(t+1)) from VGPRs to the next
        # LDS stage, compute row mi of K(t).kh0, and prefetch A(mi,K(t+2)).
        acc, future = _publish_a_row(
            current_a[mi], current_b, acc, prefetched[mi],
            next_stage, source, future_kb,
            mi, mma, dot_a, dot_b, initialize, pipeline_spec, tile_spec,
        )
        future_a += tl.tuple([future])

        # K(t).kh1: read the configured A operand span from current LDS.  The
        # plan covers every A fragment exactly once while controlling lifetime.
        if tl.constexpr(pipeline_spec[0][mi][1]) > 0:
            a1 += tl.tuple([
                _local_load_a(current_stage, 1, index, dot_a)
                for index in range(
                    tl.constexpr(pipeline_spec[0][mi][0]),
                    tl.constexpr(pipeline_spec[0][mi][0])
                    + tl.constexpr(pipeline_spec[0][mi][1]),
                )
            ])

    # Publish B(K(t+1)) to next LDS and prefetch B(K(t+2)), while a1/b1
    # compute most of the current K(t).kh1 accumulator updates.
    acc, late_prefetched = _publish_b_rows(
        a1, b1, acc, prefetched, next_stage,
        source, future_kb, mma, dot_a, dot_b, pipeline_spec, tile_spec,
    )

    # All A/B fragments of K(t+1) are now in next_stage.  Make them visible
    # before reading its kh=0 operands; pair those reads with the remaining
    # K(t).kh1 MFMA updates in _finish_iteration.
    tl.debug_barrier()
    return _finish_iteration(
        next_stage, a1, b1, acc, future_a, late_prefetched,
        mma, dot_a, dot_b, pipeline_spec, tile_spec,
    )


@triton.jit
def _pipeline_pair(
    kb,
    source,
    stage0,
    stage1,
    current_a,
    current_b,
    prefetched,
    acc,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    initialize: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Consume K(kb) and K(kb+1), restoring the LDS stage orientation.

    On entry, stage0/current_a/current_b represent K(kb), and ``prefetched``
    is K(kb+1).  The first call advances to K(kb+1), swapping the current LDS
    stage from stage0 to stage1.  The second advances to K(kb+2), swapping it
    back to stage0.  The numeric argument passed to ``_pipeline`` is the
    future global-prefetch block, not the block currently being consumed.
    """
    # Consume K(kb), publish K(kb+1) into stage1, and prefetch K(kb+2).
    acc, prefetched, current_a, current_b = _pipeline(
        source, kb + 2, stage0, stage1, current_a, current_b,
        prefetched, acc, mma, dot_a, dot_b, initialize,
        pipeline_spec, tile_spec,
    )
    # Consume K(kb+1), publish K(kb+2) into stage0, and prefetch K(kb+3).
    # Accumulators were initialized by the first call, so initialize=False.
    return _pipeline(
        source, kb + 3, stage1, stage0, current_a, current_b,
        prefetched, acc, mma, dot_a, dot_b, False,
        pipeline_spec, tile_spec,
    )


@triton.jit
def _make_global_tile_load_addresses(
    a_ptr,
    b_ptr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    tile_id,
    tile_spec: tl.constexpr,
):
    """Build the A/B base pointers and fragment offsets for one output tile."""
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    rk = tl.arange(0, BLOCK_K)
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    block_m_offset = pid_m * block_m
    block_n_offset = pid_n * block_n
    a_offsets = tl.tuple([
        (block_m_offset + mi * TILE + tl.arange(0, TILE))[:, None] * stride_am
        + rk[None, :] * stride_ak
        for mi in range(m_fragments)
    ])
    b_offsets = tl.tuple([
        rk[:, None] * stride_bk
        + (block_n_offset + nj * TILE + tl.arange(0, TILE))[None, :] * stride_bn
        for nj in range(n_fragments)
    ])
    return tl.tuple([a_ptr, b_ptr, stride_ak, stride_bk, a_offsets, b_offsets])


@triton.jit
def _consume_k32(
    stage, kh: tl.constexpr, acc,
    mma: tl.constexpr, dot_a: tl.constexpr, dot_b: tl.constexpr,
    tile_spec: tl.constexpr,
):
    b_operands = _local_load_b_row(stage, kh, dot_b, tile_spec)
    for mi in tl.static_range(tl.constexpr(tile_spec[0] // TILE)):
        acc = _mfma_row(
            _local_load_a(stage, kh, mi, dot_a), b_operands, acc,
            mi, mma, dot_a, dot_b, False, tile_spec,
        )
    return acc


@triton.jit
def _compute_full_tile(
    source,
    preloaded_k0,
    stage0,
    stage1,
    mma: tl.constexpr,
    dot_a: tl.constexpr,
    dot_b: tl.constexpr,
    pipeline_spec: tl.constexpr,
    tile_spec: tl.constexpr,
):
    """Compute one output tile across all 96 K64 blocks.

    ``preloaded_k0`` is the already-issued global load of K0 held in VGPRs.
    The prologue publishes K0 to stage0 and prefetches K1.  Pipeline pairs
    consume K0..K93 and leave K94 in stage0 plus K95 in prefetch VGPRs.  The
    explicit epilogue drains K94/K95 without issuing the out-of-range K96/K97
    prefetches a final pair would require.
    """
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)

    # Prologue: publish the caller's preloaded K0 from VGPRs to stage0, read
    # K0.kh0 into operand VGPRs, and prefetch the complete K1 into VGPRs.
    _local_store_all(stage0, preloaded_k0, tile_spec)
    tl.debug_barrier()
    current_b = _local_load_b_row(stage0, 0, dot_b, tile_spec)
    current_a = tl.tuple([
        _local_load_a(stage0, 0, mi, dot_a)
        for mi in range(m_fragments)
    ])
    prefetched = _global_loads(source, 1, tile_spec)

    # Create one persistent FP32 accumulator for every logical C[32, 32]
    # fragment.  The first pair consumes K0/K1 and prepares K2/K3.
    zero = tlx.zeros((TILE, TILE), tl.float32, layout=mma)
    acc = tl.tuple([zero for _ in range(m_fragments * n_fragments)])
    acc, prefetched, current_a, current_b = _pipeline_pair(
        0, source,
        stage0, stage1, current_a, current_b, prefetched, acc,
        mma, dot_a, dot_b, True, pipeline_spec, tile_spec,
    )

    # Steady state: pair(kb) consumes K(kb)/K(kb+1) and prepares
    # K(kb+2)/K(kb+3).  kb takes 2,4,...,92, so this covers K2..K93 and
    # exits with K94 in stage0/current operands plus K95 in prefetch VGPRs.
    for kb in tl.range(2, K_BLOCKS - 2, 2, num_stages=1):
        acc, prefetched, current_a, current_b = _pipeline_pair(
            kb, source,
            stage0, stage1, current_a, current_b, prefetched, acc,
            mma, dot_a, dot_b, False, pipeline_spec, tile_spec,
        )

    # Epilogue: publish K95 into stage1 while consuming the already-resident
    # K94.kh0 operands, then consume K94.kh1 from stage0.  After the barrier,
    # both K32 halves of K95 are safe to read and accumulate from stage1.
    _local_store_all(stage1, prefetched, tile_spec)
    for mi in tl.static_range(m_fragments):
        acc = _mfma_row(
            current_a[mi], current_b, acc, mi,
            mma, dot_a, dot_b, False, tile_spec,
        )
    acc = _consume_k32(stage0, 1, acc, mma, dot_a, dot_b, tile_spec)
    tl.debug_barrier()
    for kh in tl.static_range(2):
        acc = _consume_k32(
            stage1, kh, acc, mma, dot_a, dot_b, tile_spec
        )
    return acc


@triton.jit
def _global_store_output(
    c_ptr,
    acc,
    stride_cm,
    stride_cn,
    tile_id,
    mma: tl.constexpr,
    tile_spec: tl.constexpr,
):
    block_m: tl.constexpr = tl.constexpr(tile_spec[0])
    block_n: tl.constexpr = tl.constexpr(tile_spec[1])
    num_pid_n: tl.constexpr = tl.constexpr(tile_spec[2])
    m_fragments: tl.constexpr = block_m // TILE
    n_fragments: tl.constexpr = block_n // TILE
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_main_fragments: tl.constexpr = n_main_groups * N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = n_fragments - n_main_fragments
    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n
    rm = pid_m * block_m + tl.arange(0, TILE)
    rn = pid_n * block_n + tl.arange(0, TILE)
    for mi in tl.static_range(m_fragments):
        for group in tl.static_range(n_main_groups):
            rn_wide = (
                pid_n * block_n
                + group * N_GROUP_FRAGMENTS * TILE
                + tl.arange(0, N_GROUP_FRAGMENTS * TILE)
            )
            offsets = (
                c_ptr
                + (rm + mi * TILE)[:, None] * stride_cm
                + rn_wide[None, :] * stride_cn
            )
            lo = tl.cat(
                tlx.require_layout(
                    acc[
                        mi * n_fragments
                        + group * N_GROUP_FRAGMENTS
                    ],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[
                        mi * n_fragments
                        + group * N_GROUP_FRAGMENTS
                        + 1
                    ],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            hi = tl.cat(
                tlx.require_layout(
                    acc[
                        mi * n_fragments
                        + group * N_GROUP_FRAGMENTS
                        + 2
                    ],
                    mma,
                    pin=False,
                ),
                tlx.require_layout(
                    acc[
                        mi * n_fragments
                        + group * N_GROUP_FRAGMENTS
                        + 3
                    ],
                    mma,
                    pin=False,
                ),
                dim=1,
            )
            value = tl.cat(lo, hi, dim=1)
            value = tlx.require_layout(
                value.to(c_ptr.dtype.element_ty), _C_STORE_32X128_LAYOUT
            )
            tlx.assert_same_layout(value, _C_STORE_32X128_LAYOUT)
            tl.store(offsets, value)
        for tail in tl.static_range(n_tail_fragments):
            offsets = tlx.require_layout(
                c_ptr
                + (rm + mi * TILE)[:, None] * stride_cm
                + (
                    rn
                    + (n_main_fragments + tail) * TILE
                )[None, :] * stride_cn,
                mma,
                pin=False,
            )
            value = tlx.require_layout(
                acc[mi * n_fragments + n_main_fragments + tail],
                mma,
                pin=False,
            )
            tl.store(offsets, value)


@triton.jit
def _commit_accumulators(acc, mma: tl.constexpr, tile_spec: tl.constexpr):
    accumulators: tl.constexpr = tl.constexpr(
        tile_spec[0] // TILE * tile_spec[1] // TILE
    )
    values = [
        tlx.require_layout(acc[index], mma, pin=False)
        for index in range(accumulators)
    ]
    return tlx.amd_mfma_commit(tl.tuple(values))


@triton.jit
def _local_alloc_stage(
    a_layout: tl.constexpr,
    b_main_layout: tl.constexpr,
    b_tail_layout: tl.constexpr,
    tile_spec: tl.constexpr,
):
    m_fragments: tl.constexpr = tl.constexpr(tile_spec[0] // TILE)
    n_fragments: tl.constexpr = tl.constexpr(tile_spec[1] // TILE)
    n_main_groups: tl.constexpr = n_fragments // N_GROUP_FRAGMENTS
    n_tail_fragments: tl.constexpr = (
        n_fragments - n_main_groups * N_GROUP_FRAGMENTS
    )
    a_buffers = tl.tuple([
        tlx.local_alloc((TILE, BLOCK_K), tl.float16, 1, layout=a_layout)
        for _ in range(m_fragments)
    ])
    b_buffers = tl.tuple(
        [
            tlx.local_alloc(
                (BLOCK_K, N_GROUP_FRAGMENTS * TILE),
                tl.float16,
                1,
                layout=b_main_layout,
            )
            for _ in range(n_main_groups)
        ]
        + [
            tlx.local_alloc(
                (BLOCK_K, TILE), tl.float16, 1, layout=b_tail_layout
            )
            for _ in range(n_tail_fragments)
        ]
    )
    return tl.tuple([a_buffers, b_buffers])


@triton.jit
def _kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    TILE_SPEC: tl.constexpr,
    PIPELINE_SPEC: tl.constexpr,
):
    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)
    a_layout: tl.constexpr = (
        tlx.padded_shared_layout_encoding.with_identity_for(
            [(64, 16)], [TILE, BLOCK_K], order=[1, 0]
        )
    )
    b_main_layout: tl.constexpr = (
        tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)], _B_BASES, [BLOCK_K, 128]
        )
    )
    b_tail_layout: tl.constexpr = (
        tlx.padded_shared_layout_encoding.with_identity_for(
            [(256, 16)], [BLOCK_K, TILE], order=[0, 1]
        )
    )
    stage0 = _local_alloc_stage(
        a_layout, b_main_layout, b_tail_layout, TILE_SPEC
    )
    stage1 = _local_alloc_stage(
        a_layout, b_main_layout, b_tail_layout, TILE_SPEC
    )

    # Persistently assign a compile-time group of adjacent N tiles to each
    # program.  For example, 128 N tiles with two tiles/program produce 64
    # programs per M row: program 0 -> tiles 0/1, ..., program 63 -> 126/127.
    program = tl.program_id(0)
    num_pid_n: tl.constexpr = tl.constexpr(TILE_SPEC[2])
    tiles_per_program: tl.constexpr = tl.constexpr(TILE_SPEC[4])
    tl.static_assert(tiles_per_program > 0)
    tl.static_assert(num_pid_n % tiles_per_program == 0)
    programs_per_m = num_pid_n // tiles_per_program
    program_m = program // programs_per_m
    program_n = program % programs_per_m
    first_tile = (
        program_m * num_pid_n + program_n * tiles_per_program
    )

    # Prime the persistent traversal with the first tile's complete K0 slice.
    tile_load_addresses = _make_global_tile_load_addresses(
        a_ptr, b_ptr, stride_am, stride_ak,
        stride_bk, stride_bn, first_tile, TILE_SPEC,
    )
    prefetched_k0 = _global_loads(tile_load_addresses, 0, TILE_SPEC)

    for tile_offset in tl.static_range(tiles_per_program):
        tile_id = first_tile + tile_offset
        acc_tile = _compute_full_tile(
            tile_load_addresses, prefetched_k0, stage0, stage1,
            mma, dot_a, dot_b, PIPELINE_SPEC, TILE_SPEC,
        )

        # Before the current tile's epilogue, issue the next tile's K0 loads.
        # At most one completed accumulator tile and one future K0 prefetch are
        # live together, independent of tiles_per_program.
        if tile_offset + 1 < tiles_per_program:
            next_tile_load_addresses = _make_global_tile_load_addresses(
                a_ptr, b_ptr, stride_am, stride_ak,
                stride_bk, stride_bn, tile_id + 1, TILE_SPEC,
            )
            next_prefetched_k0 = _global_loads(
                next_tile_load_addresses, 0, TILE_SPEC
            )

        acc_tile = _commit_accumulators(acc_tile, mma, TILE_SPEC)
        _global_store_output(
            c_ptr, acc_tile, stride_cm, stride_cn,
            tile_id, mma, TILE_SPEC,
        )

        if tile_offset + 1 < tiles_per_program:
            tile_load_addresses = next_tile_load_addresses
            prefetched_k0 = next_prefetched_k0


def _validate_specialization(m, n, tile_spec, pipeline_spec):
    block_m, block_n, num_pid_n, num_programs, tiles_per_program = tile_spec
    m_fragments = block_m // int(TILE)
    n_fragments = block_n // int(TILE)
    a1_read_plan, mfmas_before_prefetch, b_publish_plan, finish_plan = (
        pipeline_spec
    )
    assert m % block_m == 0 and n % block_n == 0
    assert num_pid_n == n // block_n
    assert tiles_per_program > 0
    assert num_pid_n % tiles_per_program == 0
    assert num_programs * tiles_per_program == (m // block_m) * num_pid_n
    assert len(a1_read_plan) == m_fragments
    assert 0 <= mfmas_before_prefetch <= n_fragments
    assert len(b_publish_plan) == n_fragments
    assert len(finish_plan) == m_fragments + n_fragments

    a1_reads = []
    for first, count in a1_read_plan:
        assert 0 <= first <= m_fragments and 0 <= count <= m_fragments - first
        a1_reads.extend(range(first, first + count))
    assert sorted(a1_reads) == list(range(m_fragments))

    # K1 coverage consists of row0's mandatory B publishes, the extra spans
    # attached to each publish, the final-read plan, and the last MFMA row.
    mfma_coverage = list(range(n_fragments))
    for parts in b_publish_plan:
        for mi, first_nj, count in parts:
            if count == 0:
                continue
            assert 0 <= mi < m_fragments
            assert 0 <= first_nj < n_fragments
            assert first_nj + count <= n_fragments
            mfma_coverage.extend(
                mi * n_fragments + nj
                for nj in range(first_nj, first_nj + count)
            )
    assert all(0 <= flat < m_fragments * n_fragments for flat in finish_plan)
    mfma_coverage.extend(finish_plan)
    mfma_coverage.extend(
        range((m_fragments - 1) * n_fragments, m_fragments * n_fragments)
    )
    assert sorted(mfma_coverage) == list(range(m_fragments * n_fragments))


def supports(a, b):
    """Return whether a and b select one of the persistent specializations."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return False
    m, k = a.shape
    _, n = b.shape
    return (
        a.dtype == torch.float16
        and b.dtype == torch.float16
        and (m, n, k) in _SHAPE_DEFAULTS
    )


def matmul(a, b, out=None, specialization=None):
    """Run a compile-time gfx950 persistent GEMM specialization."""
    assert a.ndim == 2 and b.ndim == 2
    m, k = a.shape
    kb, n = b.shape
    assert k == kb
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if specialization is None:
        specialization = os.environ.get("TLX_GFX950_TILE")
    if specialization is None:
        assert (m, n, k) in _SHAPE_DEFAULTS
        specialization = _SHAPE_DEFAULTS[(m, n, k)]
    assert specialization in _SPECIALIZATIONS
    expected_shape, tile_spec, pipeline_spec = _SPECIALIZATIONS[specialization]
    assert (m, n, k) == expected_shape
    _validate_specialization(m, n, tile_spec, pipeline_spec)
    _kernel[(tile_spec[3],)](
        a,
        b,
        out,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        TILE_SPEC=tile_spec,
        PIPELINE_SPEC=pipeline_spec,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_sched_group_barrier_scheduler=True,
        sched_group_barrier_mfma_per_dwordx4=1,
        regclass_priority_trumps_globalness=True,
        reverse_local_assignment=True,
    )
    return out
