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
def _load_a_fragment(smem_a, buffer, phase: tl.constexpr,
                     tile: tl.constexpr, ready):
    region = smem_a[buffer]
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
def _load_b_fragment(smem_b, buffer, phase: tl.constexpr,
                     tile: tl.constexpr, ready):
    region = smem_b[buffer]
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
def _load_a_phase(smem_a, buffer, phase: tl.constexpr, ready):
    return tl.tuple([
        _load_a_fragment(smem_a, buffer, phase, tile, ready)
        for tile in range(8)
    ])


@triton.jit
def _load_b_phase(smem_b, buffer, phase: tl.constexpr, ready):
    return tl.tuple([
        _load_b_fragment(smem_b, buffer, phase, tile, ready)
        for tile in range(8)
    ])


@triton.jit
def _mma_range(a, b, acc, begin: tl.constexpr, end: tl.constexpr):
    """Emit an i-major MFMA interval while keeping accumulators j-major."""
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

    # Upstream emits MFMAs with M as the outer traversal but coalesces output
    # registers by N.  Reorder only the tuple, never the operation sequence.
    emitted = tl.tuple([
        tlx.dot(
            a[ordinal // 8],
            b[ordinal % 8],
            acc[(ordinal % 8) * 8 + ordinal // 8],
            tiles_per_warp=[1, 1],
        )
        if begin <= ordinal and ordinal < end
        else acc[(ordinal % 8) * 8 + ordinal // 8]
        for ordinal in range(64)
    ])
    return tl.tuple([
        tlx.release_layout(emitted[(index % 8) * 8 + index // 8])
        for index in range(64)
    ])


@triton.jit
def _compute_tile_and_refill(
    a0,
    b0,
    acc,
    smem_a,
    smem_b,
    consumed,
    next_buffer,
    a_ptr,
    b_ptr,
    a_off,
    b_off,
    refill_k,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    current_ready,
):
    # Match Wave's coalesced subpanel cadence.  A1 is read during the first 16
    # MFMAs and A can be released after MFMA 22.
    acc = _mma_range(a0, b0, acc, 0, 1)
    a10 = _load_a_fragment(smem_a, consumed, 1, 0, current_ready)
    acc = _mma_range(a0, b0, acc, 1, 3)
    a11 = _load_a_fragment(smem_a, consumed, 1, 1, current_ready)
    acc = _mma_range(a0, b0, acc, 3, 5)
    a12 = _load_a_fragment(smem_a, consumed, 1, 2, current_ready)
    acc = _mma_range(a0, b0, acc, 5, 7)
    a13 = _load_a_fragment(smem_a, consumed, 1, 3, current_ready)
    acc = _mma_range(a0, b0, acc, 7, 9)
    a14 = _load_a_fragment(smem_a, consumed, 1, 4, current_ready)
    acc = _mma_range(a0, b0, acc, 9, 11)
    a15 = _load_a_fragment(smem_a, consumed, 1, 5, current_ready)
    acc = _mma_range(a0, b0, acc, 11, 13)
    a16 = _load_a_fragment(smem_a, consumed, 1, 6, current_ready)
    acc = _mma_range(a0, b0, acc, 13, 15)
    a17 = _load_a_fragment(smem_a, consumed, 1, 7, current_ready)
    acc = _mma_range(a0, b0, acc, 15, 22)
    a1 = (a10, a11, a12, a13, a14, a15, a16, a17)

    # A zero-drain is unnecessary here: the older payload was already made
    # visible by current_ready, while the next buffer must remain in flight.
    # Repeating wait_group(1) supplies the standard TLX CTA publication point
    # that closes the completed A read epoch before its aliased DMA write.
    tlx.async_load_wait_group(1)
    refill_a = tlx.local_slice(
        smem_a[consumed],
        [0, 0, 0, 0, 0, 0],
        [1, 2, 2, 8, 16, 32],
    )
    refill_a_token = tlx.buffer_load_to_local(
        refill_a, a_ptr, a_off + refill_k * stride_ak)

    # B1 spans 21 MFMAs.  Once its read and phase-zero uses are complete at
    # MFMA 52, the B half of the ring is independently reusable.
    acc = _mma_range(a0, b0, acc, 22, 23)
    b10 = _load_b_fragment(smem_b, consumed, 1, 0, current_ready)
    acc = _mma_range(a0, b0, acc, 23, 25)
    b11 = _load_b_fragment(smem_b, consumed, 1, 1, current_ready)
    acc = _mma_range(a0, b0, acc, 25, 28)
    b12 = _load_b_fragment(smem_b, consumed, 1, 2, current_ready)
    acc = _mma_range(a0, b0, acc, 28, 30)
    b13 = _load_b_fragment(smem_b, consumed, 1, 3, current_ready)
    acc = _mma_range(a0, b0, acc, 30, 33)
    b14 = _load_b_fragment(smem_b, consumed, 1, 4, current_ready)
    acc = _mma_range(a0, b0, acc, 33, 36)
    b15 = _load_b_fragment(smem_b, consumed, 1, 5, current_ready)
    acc = _mma_range(a0, b0, acc, 36, 38)
    b16 = _load_b_fragment(smem_b, consumed, 1, 6, current_ready)
    acc = _mma_range(a0, b0, acc, 38, 41)
    b17 = _load_b_fragment(smem_b, consumed, 1, 7, current_ready)
    acc = _mma_range(a0, b0, acc, 41, 52)
    b1 = (b10, b11, b12, b13, b14, b15, b16, b17)

    tlx.async_load_wait_group(1)
    refill_b = tlx.local_slice(
        smem_b[consumed],
        [1, 0, 0, 0, 0, 0],
        [1, 2, 2, 8, 32, 16],
    )
    refill_b_token = tlx.buffer_load_to_local(
        refill_b, b_ptr, b_off + refill_k * stride_bk)
    tlx.async_load_commit_group(tokens=[refill_a_token, refill_b_token])

    acc = _mma_range(a0, b0, acc, 52, 64)
    acc = _mma_range(a1, b1, acc, 0, 29)

    # This is the sole synchronization of the older DMA group.  The refill
    # above remains outstanding and overlaps the rest of phase one.
    next_ready = tlx.async_load_wait_group(1)
    acc = _mma_range(a1, b1, acc, 29, 30)
    na0 = _load_a_fragment(smem_a, next_buffer, 0, 0, next_ready)
    acc = _mma_range(a1, b1, acc, 30, 32)
    nb0 = _load_b_fragment(smem_b, next_buffer, 0, 0, next_ready)
    acc = _mma_range(a1, b1, acc, 32, 34)
    na1 = _load_a_fragment(smem_a, next_buffer, 0, 1, next_ready)
    acc = _mma_range(a1, b1, acc, 34, 36)
    nb1 = _load_b_fragment(smem_b, next_buffer, 0, 1, next_ready)
    acc = _mma_range(a1, b1, acc, 36, 38)
    na2 = _load_a_fragment(smem_a, next_buffer, 0, 2, next_ready)
    acc = _mma_range(a1, b1, acc, 38, 40)
    nb2 = _load_b_fragment(smem_b, next_buffer, 0, 2, next_ready)
    acc = _mma_range(a1, b1, acc, 40, 43)
    na3 = _load_a_fragment(smem_a, next_buffer, 0, 3, next_ready)
    acc = _mma_range(a1, b1, acc, 43, 45)
    nb3 = _load_b_fragment(smem_b, next_buffer, 0, 3, next_ready)
    acc = _mma_range(a1, b1, acc, 45, 47)
    na4 = _load_a_fragment(smem_a, next_buffer, 0, 4, next_ready)
    acc = _mma_range(a1, b1, acc, 47, 49)
    nb4 = _load_b_fragment(smem_b, next_buffer, 0, 4, next_ready)
    acc = _mma_range(a1, b1, acc, 49, 51)
    na5 = _load_a_fragment(smem_a, next_buffer, 0, 5, next_ready)
    acc = _mma_range(a1, b1, acc, 51, 54)
    nb5 = _load_b_fragment(smem_b, next_buffer, 0, 5, next_ready)
    acc = _mma_range(a1, b1, acc, 54, 56)
    na6 = _load_a_fragment(smem_a, next_buffer, 0, 6, next_ready)
    acc = _mma_range(a1, b1, acc, 56, 58)
    nb6 = _load_b_fragment(smem_b, next_buffer, 0, 6, next_ready)
    acc = _mma_range(a1, b1, acc, 58, 60)
    na7 = _load_a_fragment(smem_a, next_buffer, 0, 7, next_ready)
    acc = _mma_range(a1, b1, acc, 60, 62)
    nb7 = _load_b_fragment(smem_b, next_buffer, 0, 7, next_ready)
    acc = _mma_range(a1, b1, acc, 62, 64)
    return acc, (na0, na1, na2, na3, na4, na5, na6, na7), (
        nb0, nb1, nb2, nb3, nb4, nb5, nb6, nb7), next_ready


@triton.jit
def _compute_tile(a0, b0, acc, smem_a, smem_b, buffer, ready):
    # The drain path retains upstream's B1-read cadence in the first MFMA row.
    acc = _mma_range(a0, b0, acc, 0, 1)
    b10 = _load_b_fragment(smem_b, buffer, 1, 0, ready)
    acc = _mma_range(a0, b0, acc, 1, 2)
    b11 = _load_b_fragment(smem_b, buffer, 1, 1, ready)
    acc = _mma_range(a0, b0, acc, 2, 3)
    b12 = _load_b_fragment(smem_b, buffer, 1, 2, ready)
    acc = _mma_range(a0, b0, acc, 3, 4)
    b13 = _load_b_fragment(smem_b, buffer, 1, 3, ready)
    acc = _mma_range(a0, b0, acc, 4, 5)
    b14 = _load_b_fragment(smem_b, buffer, 1, 4, ready)
    acc = _mma_range(a0, b0, acc, 5, 6)
    b15 = _load_b_fragment(smem_b, buffer, 1, 5, ready)
    acc = _mma_range(a0, b0, acc, 6, 7)
    b16 = _load_b_fragment(smem_b, buffer, 1, 6, ready)
    acc = _mma_range(a0, b0, acc, 7, 8)
    b17 = _load_b_fragment(smem_b, buffer, 1, 7, ready)
    acc = _mma_range(a0, b0, acc, 8, 64)
    a1 = _load_a_phase(smem_a, buffer, 1, ready)
    b1 = (b10, b11, b12, b13, b14, b15, b16, b17)
    return _mma_range(a1, b1, acc, 0, 64)


@triton.jit
def _store_acc_tile(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn,
                    local_m: tl.constexpr, local_n: tl.constexpr):
    m = tl.arange(0, 32)
    n = tl.arange(0, 32)
    offs_m = pid_m * 256 + (m // 16) * 128 + local_m * 16 + m % 16
    offs_n = pid_n * 256 + (n // 16) * 128 + local_n * 16 + n % 16
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float16))


@triton.jit
def _store_accumulators(c_ptr, acc, pid_m, pid_n, stride_cm, stride_cn):
    for i in tl.static_range(8):
        for j in tl.static_range(8):
            _store_acc_tile(
                c_ptr, acc[j * 8 + i], pid_m, pid_n, stride_cm, stride_cn, i, j)


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
    a_group = tl.arange(0, 2)[None, None, :, None, None, None]
    a_tile = tl.arange(0, 8)[None, None, None, :, None, None]
    a_row = tl.arange(0, 16)[None, None, None, None, :, None]
    a_k = tl.arange(0, 32)[None, None, None, None, None, :]
    a_off = (
        (pid_m * BLOCK_M + a_group * 128 + a_tile * 16 + a_row)
        * stride_am
        + (a_phase * 32 + a_k) * stride_ak
    )

    b_phase = tl.arange(0, 2)[None, :, None, None, None, None]
    b_group = tl.arange(0, 2)[None, None, :, None, None, None]
    b_tile = tl.arange(0, 8)[None, None, None, :, None, None]
    b_k = tl.arange(0, 32)[None, None, None, None, :, None]
    b_col = tl.arange(0, 16)[None, None, None, None, None, :]
    b_off = (
        (b_phase * 32 + b_k) * stride_bk
        + (pid_n * BLOCK_N + b_group * 128 + b_tile * 16 + b_col)
        * stride_bn
    )

    zero = tl.zeros((32, 32), dtype=tl.float32)
    acc = tl.tuple([zero for _ in range(64)])
    iter_max: tl.constexpr = K // BLOCK_K

    a_dma0_view = tlx.local_slice(
        smem_a[0], [0, 0, 0, 0, 0, 0], [1, 2, 2, 8, 16, 32])
    b_dma0_view = tlx.local_slice(
        smem_b[0], [1, 0, 0, 0, 0, 0], [1, 2, 2, 8, 32, 16])
    a_dma0 = tlx.buffer_load_to_local(a_dma0_view, a_ptr, a_off)
    b_dma0 = tlx.buffer_load_to_local(b_dma0_view, b_ptr, b_off)
    tlx.async_load_commit_group(tokens=[a_dma0, b_dma0])

    a_dma1_view = tlx.local_slice(
        smem_a[1], [0, 0, 0, 0, 0, 0], [1, 2, 2, 8, 16, 32])
    b_dma1_view = tlx.local_slice(
        smem_b[1], [1, 0, 0, 0, 0, 0], [1, 2, 2, 8, 32, 16])
    a_dma1 = tlx.buffer_load_to_local(
        a_dma1_view, a_ptr, a_off + BLOCK_K * stride_ak)
    b_dma1 = tlx.buffer_load_to_local(
        b_dma1_view, b_ptr, b_off + BLOCK_K * stride_bk)
    tlx.async_load_commit_group(tokens=[a_dma1, b_dma1])

    current_ready = tlx.async_load_wait_group(1)
    a0 = _load_a_phase(smem_a, 0, 0, current_ready)
    b0 = _load_b_phase(smem_b, 0, 0, current_ready)

    for k in tl.range(0, iter_max - 2, num_stages=1):
        consumed = k % 2
        next_buffer = 1 - consumed
        refill_k = (k + 2) * BLOCK_K
        acc, a0, b0, current_ready = _compute_tile_and_refill(
            a0, b0, acc, smem_a, smem_b, consumed, next_buffer,
            a_ptr, b_ptr, a_off, b_off, refill_k,
            stride_ak, stride_bk, current_ready)

    current_buffer: tl.constexpr = (iter_max - 2) % 2
    acc = _compute_tile(
        a0, b0, acc, smem_a, smem_b, current_buffer, current_ready)

    last_ready = tlx.async_load_wait_group(0)
    last_buffer: tl.constexpr = (iter_max - 1) % 2
    last_a0 = _load_a_phase(smem_a, last_buffer, 0, last_ready)
    last_b0 = _load_b_phase(smem_b, last_buffer, 0, last_ready)
    acc = _compute_tile(
        last_a0, last_b0, acc, smem_a, smem_b, last_buffer, last_ready)
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
