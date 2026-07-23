"""TLX rendering of Wave's specialized gfx950 4-wave f16 GEMM.

The kernel keeps the upstream 256x256x64 CTA shape, four-wave 2x2 MFMA
topology, 8x8 instruction grid per wave, column-major accumulator ownership,
and double-buffered direct-to-LDS pipeline.  A and B are refilled only after
their phase-one LDS reads and phase-zero register uses are complete.  The
resulting memory/value dependencies are part of the source program; Wave's
multi-wave specialization remains a backend compilation policy.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _load_a_fragment(region, phase: tl.constexpr, tile: tl.constexpr, ready):
    fragment_view = tlx.local_slice(
        region,
        [0, phase, 0, tile, 0, 0],
        [1, 1, 2, 1, 16, 32],
    )
    fragment_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 4],
        ],
        lane_bases=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 4, 0],
            [0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 0, 16],
        ],
        # Warp bit zero is the N ownership bit and is broadcast for A; warp
        # bit one selects the two M groups carried by this ordinary tensor.
        warp_bases=[
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        block_bases=[],
        shape=[1, 1, 2, 1, 16, 32],
    )
    fragment = tlx.require_layout(
        tlx.local_load(fragment_view, token=ready), fragment_layout)
    fragment = tl.reshape(fragment, (32, 32))
    return tlx.release_layout(fragment)


@triton.jit
def _load_b_fragment(region, phase: tl.constexpr, tile: tl.constexpr, ready):
    fragment_view = tlx.local_slice(
        region,
        [1, phase, 0, tile, 0, 0],
        [1, 1, 2, 1, 32, 16],
    )
    fragment_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 4, 0],
        ],
        lane_bases=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 8],
            [0, 0, 0, 0, 8, 0],
            [0, 0, 0, 0, 16, 0],
        ],
        # B has the complementary ownership: warp bit zero selects N and the
        # M bit is broadcast through LDS.
        warp_bases=[
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        block_bases=[],
        shape=[1, 1, 2, 1, 32, 16],
    )
    fragment = tlx.require_layout(
        tlx.local_load(fragment_view, token=ready), fragment_layout)
    fragment = tl.permute(fragment, (0, 1, 4, 3, 2, 5))
    fragment = tl.reshape(fragment, (32, 32))
    return tlx.release_layout(fragment)


@triton.jit
def _load_a_phase(region, phase: tl.constexpr, ready):
    return tl.tuple([
        _load_a_fragment(region, phase, tile, ready)
        for tile in range(8)
    ])


@triton.jit
def _load_b_phase(region, phase: tl.constexpr, ready):
    return tl.tuple([
        _load_b_fragment(region, phase, tile, ready)
        for tile in range(8)
    ])


@triton.jit
def _prefetch_tile_subpanels(
    smem_a,
    smem_b,
    buffer,
    a_ptr,
    b_ptr,
    a_subpanel0_off,
    a_subpanel1_request_off,
    b_subpanel0_off,
    b_subpanel1_request_off,
    tile_k,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
):
    """Issue one tile as the same three async subpanels as the hot loop."""
    a0_view = tlx.local_slice(
        smem_a[buffer], [0, 0, 0, 0, 0, 0], [1, 2, 1, 8, 16, 32])
    a0 = tlx.buffer_load_to_local(
        a0_view, a_ptr, a_subpanel0_off + tile_k * stride_ak)
    a1_early_view = tlx.local_slice(
        smem_a[buffer], [0, 0, 1, 0, 0, 0], [1, 2, 1, 8, 4, 32])
    a1_early = tlx.buffer_load_to_local(
        a1_early_view, a_ptr,
        a_subpanel1_request_off + tile_k * stride_ak)
    tlx.async_load_commit_group(tokens=[a0, a1_early])

    a1_late1_view = tlx.local_slice(
        smem_a[buffer], [0, 0, 1, 0, 4, 0], [1, 2, 1, 8, 4, 32])
    a1_late2_view = tlx.local_slice(
        smem_a[buffer], [0, 0, 1, 0, 8, 0], [1, 2, 1, 8, 4, 32])
    a1_late3_view = tlx.local_slice(
        smem_a[buffer], [0, 0, 1, 0, 12, 0], [1, 2, 1, 8, 4, 32])
    b0_view = tlx.local_slice(
        smem_b[buffer], [1, 0, 0, 0, 0, 0], [1, 2, 1, 8, 32, 16])
    b1_early_view = tlx.local_slice(
        smem_b[buffer], [1, 0, 1, 0, 0, 0], [1, 2, 1, 8, 32, 4])
    a1_late1 = tlx.buffer_load_to_local(
        a1_late1_view, a_ptr,
        a_subpanel1_request_off + 32 * stride_am + tile_k * stride_ak)
    a1_late2 = tlx.buffer_load_to_local(
        a1_late2_view, a_ptr,
        a_subpanel1_request_off + 64 * stride_am + tile_k * stride_ak)
    a1_late3 = tlx.buffer_load_to_local(
        a1_late3_view, a_ptr,
        a_subpanel1_request_off + 96 * stride_am + tile_k * stride_ak)
    b0 = tlx.buffer_load_to_local(
        b0_view, b_ptr, b_subpanel0_off + tile_k * stride_bk)
    b1_early = tlx.buffer_load_to_local(
        b1_early_view, b_ptr,
        b_subpanel1_request_off + tile_k * stride_bk)
    tlx.async_load_commit_group(tokens=[
        a1_late1, a1_late2, a1_late3, b0, b1_early])

    b1_late1_view = tlx.local_slice(
        smem_b[buffer], [1, 0, 1, 0, 0, 4], [1, 2, 1, 8, 32, 4])
    b1_late2_view = tlx.local_slice(
        smem_b[buffer], [1, 0, 1, 0, 0, 8], [1, 2, 1, 8, 32, 4])
    b1_late3_view = tlx.local_slice(
        smem_b[buffer], [1, 0, 1, 0, 0, 12], [1, 2, 1, 8, 32, 4])
    b1_late1 = tlx.buffer_load_to_local(
        b1_late1_view, b_ptr,
        b_subpanel1_request_off + 32 * stride_bn + tile_k * stride_bk)
    b1_late2 = tlx.buffer_load_to_local(
        b1_late2_view, b_ptr,
        b_subpanel1_request_off + 64 * stride_bn + tile_k * stride_bk)
    b1_late3 = tlx.buffer_load_to_local(
        b1_late3_view, b_ptr,
        b_subpanel1_request_off + 96 * stride_bn + tile_k * stride_bk)
    tlx.async_load_commit_group(tokens=[b1_late1, b1_late2, b1_late3])


@triton.jit
def _mma_range(a, b, acc, begin: tl.constexpr, end: tl.constexpr):
    """Emit Wave's B-major serpentine MFMA interval.

    Coalesced-output MFMAs reverse the operand roles at the instruction
    boundary.  Keep one B fragment fixed while walking all A fragments, and
    reverse every other row, exactly like Wave's subpanel traversal.  The
    accumulators remain stored j-major regardless of traversal direction.
    """
    mma_layout: tl.constexpr = tlx.amd_mfma_layout_encoding.make(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    a_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        0, mma_layout, 8)
    b_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(
        1, mma_layout, 8)
    a = tl.tuple([tlx.require_layout(value, a_layout) for value in a])
    b = tl.tuple([tlx.require_layout(value, b_layout) for value in b])
    acc = tl.tuple([tlx.require_layout(value, mma_layout) for value in acc])

    # ``ordinal`` names the operation sequence, while ``acc_index`` names the
    # stable j-major accumulator slot.  Odd B rows walk A backwards so the
    # last operand of one row is the first operand of the next row.
    emitted = tl.tuple([
        tlx.dot(
            a[(ordinal % 8) if (ordinal // 8) % 2 == 0 else 7 - ordinal % 8],
            b[ordinal // 8],
            acc[(ordinal // 8) * 8 + (
                (ordinal % 8) if (ordinal // 8) % 2 == 0
                else 7 - ordinal % 8)],
            tiles_per_warp=[1, 1],
        )
        if begin <= ordinal and ordinal < end
        else acc[(ordinal // 8) * 8 + (
            (ordinal % 8) if (ordinal // 8) % 2 == 0
            else 7 - ordinal % 8)]
        for ordinal in range(64)
    ])
    return tl.tuple([
        tlx.release_layout(emitted[
            (index // 8) * 8 + (
                (index % 8) if (index // 8) % 2 == 0
                else 7 - index % 8)])
        for index in range(64)
    ])


@triton.jit
def _compute_tile_and_refill(
    a0,
    b0,
    acc,
    current_a,
    current_b,
    next_a,
    next_b,
    refill_a_ptr,
    refill_b_ptr,
    a_subpanel0_off,
    a_subpanel1_request_off,
    b_subpanel0_off,
    b_subpanel1_request_off,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    current_ready,
):
    # Match Wave's coalesced subpanel cadence.  A1 is read during the first 16
    # MFMAs and A can be released after MFMA 22.
    acc = _mma_range(a0, b0, acc, 0, 1)
    a10 = _load_a_fragment(current_a, 1, 0, current_ready)
    acc = _mma_range(a0, b0, acc, 1, 3)
    a11 = _load_a_fragment(current_a, 1, 1, current_ready)
    acc = _mma_range(a0, b0, acc, 3, 5)
    a12 = _load_a_fragment(current_a, 1, 2, current_ready)
    acc = _mma_range(a0, b0, acc, 5, 7)
    a13 = _load_a_fragment(current_a, 1, 3, current_ready)
    acc = _mma_range(a0, b0, acc, 7, 9)
    a14 = _load_a_fragment(current_a, 1, 4, current_ready)
    acc = _mma_range(a0, b0, acc, 9, 11)
    a15 = _load_a_fragment(current_a, 1, 5, current_ready)
    acc = _mma_range(a0, b0, acc, 11, 13)
    a16 = _load_a_fragment(current_a, 1, 6, current_ready)
    acc = _mma_range(a0, b0, acc, 13, 15)
    a17 = _load_a_fragment(current_a, 1, 7, current_ready)
    acc = _mma_range(a0, b0, acc, 15, 22)
    a1 = (a10, a11, a12, a13, a14, a15, a16, a17)

    # Four groups are live after the first refill group is committed.  Retain
    # all four: this publication point closes the completed A read epoch before
    # the first subpanel reuses its LDS range, but must not wait for DMA.
    tlx.async_load_wait_group(4)
    refill_a0 = tlx.local_slice(
        current_a,
        [0, 0, 0, 0, 0, 0],
        [1, 2, 1, 8, 16, 32],
    )
    refill_a0_token = tlx.buffer_load_to_local(
        refill_a0, refill_a_ptr, a_subpanel0_off)
    refill_a1_early = tlx.local_slice(
        current_a,
        [0, 0, 1, 0, 0, 0],
        [1, 2, 1, 8, 4, 32],
    )
    refill_a1_early_token = tlx.buffer_load_to_local(
        refill_a1_early, refill_a_ptr, a_subpanel1_request_off)
    tlx.async_load_commit_group(
        tokens=[refill_a0_token, refill_a1_early_token])

    # B1 spans 21 MFMAs.  Once its read and phase-zero uses are complete at
    # MFMA 52, the B half of the ring is independently reusable.
    acc = _mma_range(a0, b0, acc, 22, 23)
    b10 = _load_b_fragment(current_b, 1, 0, current_ready)
    acc = _mma_range(a0, b0, acc, 23, 25)
    b11 = _load_b_fragment(current_b, 1, 1, current_ready)
    acc = _mma_range(a0, b0, acc, 25, 28)
    b12 = _load_b_fragment(current_b, 1, 2, current_ready)
    acc = _mma_range(a0, b0, acc, 28, 30)
    b13 = _load_b_fragment(current_b, 1, 3, current_ready)
    acc = _mma_range(a0, b0, acc, 30, 33)
    b14 = _load_b_fragment(current_b, 1, 4, current_ready)
    acc = _mma_range(a0, b0, acc, 33, 36)
    b15 = _load_b_fragment(current_b, 1, 5, current_ready)
    acc = _mma_range(a0, b0, acc, 36, 38)
    b16 = _load_b_fragment(current_b, 1, 6, current_ready)
    acc = _mma_range(a0, b0, acc, 38, 41)
    b17 = _load_b_fragment(current_b, 1, 7, current_ready)
    acc = _mma_range(a0, b0, acc, 41, 52)
    b1 = (b10, b11, b12, b13, b14, b15, b16, b17)

    # The second refill raises the live queue to five groups.  This is another
    # LDS-reuse publication point, not a DMA-completion point, so retain all
    # five groups.  The wait_group(2) below is the sole readiness wait.
    tlx.async_load_wait_group(5)
    refill_a1_late1 = tlx.local_slice(
        current_a,
        [0, 0, 1, 0, 4, 0],
        [1, 2, 1, 8, 4, 32],
    )
    refill_a1_late2 = tlx.local_slice(
        current_a,
        [0, 0, 1, 0, 8, 0],
        [1, 2, 1, 8, 4, 32],
    )
    refill_a1_late3 = tlx.local_slice(
        current_a,
        [0, 0, 1, 0, 12, 0],
        [1, 2, 1, 8, 4, 32],
    )
    refill_b0 = tlx.local_slice(
        current_b,
        [1, 0, 0, 0, 0, 0],
        [1, 2, 1, 8, 32, 16],
    )
    refill_b1_early = tlx.local_slice(
        current_b,
        [1, 0, 1, 0, 0, 0],
        [1, 2, 1, 8, 32, 4],
    )
    refill_a1_late1_token = tlx.buffer_load_to_local(
        refill_a1_late1, refill_a_ptr,
        a_subpanel1_request_off + 32 * stride_am)
    refill_a1_late2_token = tlx.buffer_load_to_local(
        refill_a1_late2, refill_a_ptr,
        a_subpanel1_request_off + 64 * stride_am)
    refill_a1_late3_token = tlx.buffer_load_to_local(
        refill_a1_late3, refill_a_ptr,
        a_subpanel1_request_off + 96 * stride_am)
    refill_b0_token = tlx.buffer_load_to_local(
        refill_b0, refill_b_ptr, b_subpanel0_off)
    refill_b1_early_token = tlx.buffer_load_to_local(
        refill_b1_early, refill_b_ptr, b_subpanel1_request_off)
    tlx.async_load_commit_group(tokens=[
        refill_a1_late1_token,
        refill_a1_late2_token,
        refill_a1_late3_token,
        refill_b0_token,
        refill_b1_early_token,
    ])

    acc = _mma_range(a0, b0, acc, 52, 64)
    acc = _mma_range(a1, b1, acc, 0, 29)

    # Drain the two remaining groups of the older tile while retaining the two
    # refill groups just issued.  This token therefore makes the whole next
    # tile visible without waiting for the refill of the tile after it.
    next_ready = tlx.async_load_wait_group(2)
    refill_b1_late1 = tlx.local_slice(
        current_b,
        [1, 0, 1, 0, 0, 4],
        [1, 2, 1, 8, 32, 4],
    )
    refill_b1_late2 = tlx.local_slice(
        current_b,
        [1, 0, 1, 0, 0, 8],
        [1, 2, 1, 8, 32, 4],
    )
    refill_b1_late3 = tlx.local_slice(
        current_b,
        [1, 0, 1, 0, 0, 12],
        [1, 2, 1, 8, 32, 4],
    )
    refill_b1_late1_token = tlx.buffer_load_to_local(
        refill_b1_late1, refill_b_ptr,
        b_subpanel1_request_off + 32 * stride_bn)
    refill_b1_late2_token = tlx.buffer_load_to_local(
        refill_b1_late2, refill_b_ptr,
        b_subpanel1_request_off + 64 * stride_bn)
    refill_b1_late3_token = tlx.buffer_load_to_local(
        refill_b1_late3, refill_b_ptr,
        b_subpanel1_request_off + 96 * stride_bn)
    tlx.async_load_commit_group(tokens=[
        refill_b1_late1_token,
        refill_b1_late2_token,
        refill_b1_late3_token,
    ])
    acc = _mma_range(a1, b1, acc, 29, 30)
    na0 = _load_a_fragment(next_a, 0, 0, next_ready)
    acc = _mma_range(a1, b1, acc, 30, 32)
    nb0 = _load_b_fragment(next_b, 0, 0, next_ready)
    acc = _mma_range(a1, b1, acc, 32, 34)
    na1 = _load_a_fragment(next_a, 0, 1, next_ready)
    acc = _mma_range(a1, b1, acc, 34, 36)
    nb1 = _load_b_fragment(next_b, 0, 1, next_ready)
    acc = _mma_range(a1, b1, acc, 36, 38)
    na2 = _load_a_fragment(next_a, 0, 2, next_ready)
    acc = _mma_range(a1, b1, acc, 38, 40)
    nb2 = _load_b_fragment(next_b, 0, 2, next_ready)
    acc = _mma_range(a1, b1, acc, 40, 43)
    na3 = _load_a_fragment(next_a, 0, 3, next_ready)
    acc = _mma_range(a1, b1, acc, 43, 45)
    nb3 = _load_b_fragment(next_b, 0, 3, next_ready)
    acc = _mma_range(a1, b1, acc, 45, 47)
    na4 = _load_a_fragment(next_a, 0, 4, next_ready)
    acc = _mma_range(a1, b1, acc, 47, 49)
    nb4 = _load_b_fragment(next_b, 0, 4, next_ready)
    acc = _mma_range(a1, b1, acc, 49, 51)
    na5 = _load_a_fragment(next_a, 0, 5, next_ready)
    acc = _mma_range(a1, b1, acc, 51, 54)
    nb5 = _load_b_fragment(next_b, 0, 5, next_ready)
    acc = _mma_range(a1, b1, acc, 54, 56)
    na6 = _load_a_fragment(next_a, 0, 6, next_ready)
    acc = _mma_range(a1, b1, acc, 56, 58)
    nb6 = _load_b_fragment(next_b, 0, 6, next_ready)
    acc = _mma_range(a1, b1, acc, 58, 60)
    na7 = _load_a_fragment(next_a, 0, 7, next_ready)
    acc = _mma_range(a1, b1, acc, 60, 62)
    nb7 = _load_b_fragment(next_b, 0, 7, next_ready)
    acc = _mma_range(a1, b1, acc, 62, 64)
    return acc, (na0, na1, na2, na3, na4, na5, na6, na7), (
        nb0, nb1, nb2, nb3, nb4, nb5, nb6, nb7), next_ready


@triton.jit
def _compute_tile(a0, b0, acc, region_a, region_b, ready):
    # The drain path retains upstream's B1-read cadence in the first MFMA row.
    acc = _mma_range(a0, b0, acc, 0, 1)
    b10 = _load_b_fragment(region_b, 1, 0, ready)
    acc = _mma_range(a0, b0, acc, 1, 2)
    b11 = _load_b_fragment(region_b, 1, 1, ready)
    acc = _mma_range(a0, b0, acc, 2, 3)
    b12 = _load_b_fragment(region_b, 1, 2, ready)
    acc = _mma_range(a0, b0, acc, 3, 4)
    b13 = _load_b_fragment(region_b, 1, 3, ready)
    acc = _mma_range(a0, b0, acc, 4, 5)
    b14 = _load_b_fragment(region_b, 1, 4, ready)
    acc = _mma_range(a0, b0, acc, 5, 6)
    b15 = _load_b_fragment(region_b, 1, 5, ready)
    acc = _mma_range(a0, b0, acc, 6, 7)
    b16 = _load_b_fragment(region_b, 1, 6, ready)
    acc = _mma_range(a0, b0, acc, 7, 8)
    b17 = _load_b_fragment(region_b, 1, 7, ready)
    acc = _mma_range(a0, b0, acc, 8, 64)
    a1 = _load_a_phase(region_a, 1, ready)
    b1 = (b10, b11, b12, b13, b14, b15, b16, b17)
    return _mma_range(a1, b1, acc, 0, 64)


@triton.jit
def _store_acc_strip(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn,
                     local_m: tl.constexpr):
    # This is the ordinary distributed form of one transposed MFMA result.
    # Requiring it is a packet-structural boundary: no lane or wave changes
    # ownership.  Concatenation then adds only register dimensions for the
    # eight N tiles.  N is the contiguous axis of TLX's ordinary row-major
    # result, so symbolic scatter can form the same eight-wide stores as the
    # upstream kernel without changing the public output convention.
    fragment_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 1],
            [0, 2],
        ],
        lane_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 4],
            [0, 8],
        ],
        warp_bases=[
            [0, 16],
            [16, 0],
        ],
        block_bases=[],
        shape=[32, 32],
    )
    acc = tl.tuple([
        tlx.require_layout(value, fragment_layout) for value in acc
    ])
    pair0 = tl.cat(acc[0], acc[1], dim=1)
    pair1 = tl.cat(acc[2], acc[3], dim=1)
    pair2 = tl.cat(acc[4], acc[5], dim=1)
    pair3 = tl.cat(acc[6], acc[7], dim=1)
    quad0 = tl.cat(pair0, pair1, dim=1)
    quad1 = tl.cat(pair2, pair3, dim=1)
    acc = tl.cat(quad0, quad1, dim=1)

    m = tl.arange(0, 32)
    n = tl.arange(0, 256)
    within_tile_n = n % 32
    tile_n = n // 32
    offs_m = pid_m * 256 + (m // 16) * 128 + (m % 16) * 8 + local_m
    offs_n = (pid_n * 256 + (within_tile_n // 16) * 128 +
              (within_tile_n % 16) * 8 + tile_n)
    offsets = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    store_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[
            [0, 128],
            [0, 64],
            [0, 32],
            [0, 1],
            [0, 2],
        ],
        lane_bases=[
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [0, 4],
            [0, 8],
        ],
        warp_bases=[
            [0, 16],
            [16, 0],
        ],
        block_bases=[],
        shape=[32, 256],
    )
    offsets = tlx.require_layout(offsets, store_layout)
    acc = tlx.require_layout(acc, store_layout)

    # Isolate the two MFMA register-coordinate bits before conversion.  Each
    # result owns one scalar from every N tile, so its eight f16 values form a
    # single 16-byte store.  Keeping the conversion adjacent to that store
    # avoids making all 32 VGPR results live at once.
    acc = tl.reshape(acc, (32, 128, 2))
    offsets = tl.reshape(offsets, (32, 128, 2))
    acc_even, acc_odd = tl.split(acc)
    offsets_even, offsets_odd = tl.split(offsets)

    acc_even = tl.reshape(acc_even, (32, 64, 2))
    acc_odd = tl.reshape(acc_odd, (32, 64, 2))
    offsets_even = tl.reshape(offsets_even, (32, 64, 2))
    offsets_odd = tl.reshape(offsets_odd, (32, 64, 2))
    acc_0, acc_2 = tl.split(acc_even)
    acc_1, acc_3 = tl.split(acc_odd)
    offsets_0, offsets_2 = tl.split(offsets_even)
    offsets_1, offsets_3 = tl.split(offsets_odd)

    tlx.buffer_store(
        tlx.cast_preserve_layout(acc_0, tl.float16), c_ptr, offsets_0)
    tlx.buffer_store(
        tlx.cast_preserve_layout(acc_1, tl.float16), c_ptr, offsets_1)
    tlx.buffer_store(
        tlx.cast_preserve_layout(acc_2, tl.float16), c_ptr, offsets_2)
    tlx.buffer_store(
        tlx.cast_preserve_layout(acc_3, tl.float16), c_ptr, offsets_3)


@triton.jit
def _store_accumulators(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn):
    for i in tl.static_range(8):
        _store_acc_strip(
            c_ptr,
            tl.tuple([acc[j * 8 + i] for j in range(8)]),
            pid_m,
            pid_n,
            stride_cm,
            stride_cn,
            i,
        )


@triton.jit
def wave_4wave_specialized(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    raw_pid_n = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    tl.assume(raw_pid_m >= 0)
    tl.assume(raw_pid_m < num_pid_m)
    tl.assume(raw_pid_n >= 0)
    tl.assume(raw_pid_n < num_pid_n)

    pid = raw_pid_n * num_pid_m + raw_pid_m
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if tall_xcds == 0:
            pid = xcd * pids_per_xcd + local_pid
        else:
            if xcd < tall_xcds:
                pid = xcd * pids_per_xcd + local_pid
            else:
                pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        if M % (BLOCK_M * GROUP_SIZE_M) == 0:
            group_size_m: tl.constexpr = GROUP_SIZE_M
        else:
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # Match Wave's coalesced MFMA staging layout exactly.  Each 512-element
    # logical line holds all 8 tiles and both K32 phases for one MFMA row, and
    # carries 8 f16 elements of padding.  The 520-element stride rotates the
    # LDS bank phase between rows.  A and B differ only in which operand axis
    # is the contiguous K dimension.
    a_load_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [2, 2, 2, 8, 16, 32],
        order=[5, 1, 3, 4, 2, 0])
    b_load_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(512, 8)], [2, 2, 2, 8, 32, 16],
        order=[4, 1, 3, 5, 2, 0])

    # Two 65 KiB rings, each split into disjoint padded A/B regions.  Both
    # views describe the same physical allocation without leaking an
    # instruction-fragment representation through the source operation graph.
    smem_a = tlx.local_alloc(
        (2, 2, 2, 8, 16, 32), tl.float16, 2, layout=a_load_layout)
    smem_b = tlx.local_alloc(
        (2, 2, 2, 8, 32, 16), tl.float16, 2, reuse=smem_a,
        layout=b_load_layout)

    # Build offsets in the same logical coordinate spaces as the destination
    # subviews.  Layout inference assigns those coordinates to the four waves
    # and emits the upstream eight 16-byte DMA requests per operand and wave.
    a_phase = tl.arange(0, 2)[None, :, None, None, None, None]
    a_tile = tl.arange(0, 8)[None, None, None, :, None, None]
    a_row = tl.arange(0, 16)[None, None, None, None, :, None]
    a_request_row = tl.arange(0, 4)[None, None, None, None, :, None]
    a_k = tl.arange(0, 32)[None, None, None, None, None, :]
    a_subpanel0_off = (
        (pid_m * BLOCK_M + a_row * 8 + a_tile)
        * stride_am
        + (a_phase * 32 + a_k) * stride_ak
    )
    a_subpanel1_request_off = (
        (pid_m * BLOCK_M + 128 + a_request_row * 8 + a_tile)
        * stride_am
        + (a_phase * 32 + a_k) * stride_ak
    )

    b_phase = tl.arange(0, 2)[None, :, None, None, None, None]
    b_tile = tl.arange(0, 8)[None, None, None, :, None, None]
    b_k = tl.arange(0, 32)[None, None, None, None, :, None]
    b_col = tl.arange(0, 16)[None, None, None, None, None, :]
    b_request_col = tl.arange(0, 4)[None, None, None, None, None, :]
    b_subpanel0_off = (
        (b_phase * 32 + b_k) * stride_bk
        + (pid_n * BLOCK_N + b_col * 8 + b_tile)
        * stride_bn
    )
    b_subpanel1_request_off = (
        (b_phase * 32 + b_k) * stride_bk
        + (pid_n * BLOCK_N + 128 + b_request_col * 8 + b_tile)
        * stride_bn
    )

    zero = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.tuple([zero for _ in range(64)])
    iter_max: tl.constexpr = K // BLOCK_K

    _prefetch_tile_subpanels(
        smem_a, smem_b, 0, a_ptr, b_ptr,
        a_subpanel0_off, a_subpanel1_request_off,
        b_subpanel0_off, b_subpanel1_request_off,
        0, stride_am, stride_ak, stride_bk, stride_bn)
    _prefetch_tile_subpanels(
        smem_a, smem_b, 1, a_ptr, b_ptr,
        a_subpanel0_off, a_subpanel1_request_off,
        b_subpanel0_off, b_subpanel1_request_off,
        BLOCK_K, stride_am, stride_ak, stride_bk, stride_bn)

    current_ready = tlx.async_load_wait_group(3)
    current_a = smem_a[0]
    current_b = smem_b[0]
    a0 = _load_a_phase(current_a, 0, current_ready)
    b0 = _load_b_phase(current_b, 0, current_ready)

    # Carry the preceding tile's scalar bases and advance them at the loop
    # head.  The Wave bridge turns these structural pointer recurrences into
    # the per-request buffer-pointer carries consumed by the DMA operations.
    refill_a_ptr = a_ptr + BLOCK_K * stride_ak
    refill_b_ptr = b_ptr + BLOCK_K * stride_bk
    for k in tl.range(0, iter_max - 2, num_stages=1):
        refill_a_ptr += BLOCK_K * stride_ak
        refill_b_ptr += BLOCK_K * stride_bk
        next_buffer = 1 - k % 2
        next_a = smem_a[next_buffer]
        next_b = smem_b[next_buffer]
        acc, a0, b0, current_ready = _compute_tile_and_refill(
            a0, b0, acc, current_a, current_b, next_a, next_b,
            refill_a_ptr, refill_b_ptr,
            a_subpanel0_off, a_subpanel1_request_off,
            b_subpanel0_off, b_subpanel1_request_off,
            stride_am, stride_bn,
            current_ready)
        current_a = next_a
        current_b = next_b

    next_buffer: tl.constexpr = 1 - (iter_max - 2) % 2
    next_a = smem_a[next_buffer]
    next_b = smem_b[next_buffer]

    acc = _compute_tile(
        a0, b0, acc, current_a, current_b, current_ready)

    last_ready = tlx.async_load_wait_group(0)
    last_a0 = _load_a_phase(next_a, 0, last_ready)
    last_b0 = _load_b_phase(next_b, 0, last_ready)
    acc = _compute_tile(
        last_a0, last_b0, acc, next_a, next_b, last_ready)
    _store_accumulators(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert b.stride(0) == 1, (
        "wave_4wave_specialized expects a K-contiguous transposed-B view")
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n
    wave_4wave_specialized[(grid_m, grid_n)](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=4,
        NUM_XCDS=8,
        GRID_MN=grid_mn,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
