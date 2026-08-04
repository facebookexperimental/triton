"""TLX MXFP4 GEMM for gfx950 -- 128x128 tile, for occupancy-starved shapes.

Companion to the 256x256 kernel in matmul_kernel.py. Same ABI, same intra-wave
structure; the only reason it exists is grid fill.

A 256x256 tile is one workgroup per CU, so on a 256-CU MI350X it needs
M*N >= 256*65536 to fill the machine. Below that it strands CUs: 256x4096x4096
launches 16 workgroups on 256 CUs. Measured against AITER on the four small
shapes the 256x256 kernel lands at 0.385-0.562 -- the loop body is the same one
that reaches parity on large shapes, it is simply running on 6-25% of the GPU.

AITER's tuned configs make the rule explicit: it selects 32x128 / 64x128 /
128x128 / 128x128 on those four shapes, i.e. whichever tile makes the grid come
out at exactly 256. This kernel supplies the 128x128 point of that curve:

    512x8192x4096   ->  4 x 64 = 256 workgroups   (exact fill)
    256x8192x4096   ->  2 x 64 = 128
    512x4096x4096   ->  4 x 32 = 128
    256x4096x4096   ->  2 x 32 =  64

The narrower tiles (and/or split-K) are what the two smallest shapes still need.

Differences from the 256x256 kernel, all consequences of BLOCK_N:
  * No left/right N split. BLOCK_N=128 is one MFMA group, so there is a single
    accumulator and one tl.dot_scaled per k-step instead of two.
  * No hand-pinned tlx.layout constants. Those in matmul_kernel.py are derived
    for the 256x256x256 tile specifically and do not carry over; this kernel
    lets the compiler choose, which is correct but not yet tuned. Use
    tlx.dump_layout() to read back what it picked before pinning anything.
  * Scales still take the LDS round-trip. The pre-shuffle + direct-to-VGPR path
    (shuffle_a_scale in matmul_kernel.py) needs its own derivation per tile.

Inputs use the same ABI as the Gluon a4w4 tutorial:
  * A: packed e2m1, shape (M, K // 2), K-contiguous
  * B: packed e2m1, shape (N, K // 2), K-contiguous; computes A @ B.T
  * scales: e8m0 uint8, shapes (M, K // 32) and (N, K // 32),
    contiguous along M/N so scale tiles are coalesced
  * C: bfloat16, shape (M, N)
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 256
NUM_CU = 256
# Launch knobs, module-level so a sweep can set them without editing the source.
NUM_WARPS = 4
GROUP_SIZE_M = 4
NUM_XCDS = 8


@triton.jit
def _a4w4_kernel_128(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    # Without these the compiler picks sizePerThread=[1,1] for the scale tiles and
    # the whole scale path scalarizes: 20 ds_write_b8 + 16 ds_read_u8 +
    # 8 buffer_load_ubyte + 14 v_perm per iteration, ~96 overhead instructions
    # against 64 MFMAs of real work. The A/B path does not need pinning -- it
    # already comes out as ds_read_b128 / buffer_load_dwordx4.
    # Both scale tiles are (128, 8), the same shape as the 256x256 kernel's
    # B-scale tile, so scale_b_layout carries over unchanged.
    # Epilogue layouts, read off the TTGIR. NOTE warpsPerCTA here is [2, 2], not
    # the [1, 4] of the 64x128 kernel -- the wave grid really is 2x2 for this
    # tile, so the warp dims land in BOTH the row and column halves. Do not carry
    # the 64x128 versions over.
    #   #mma     = amd_mfma<{warpsPerCTA=[2,2], instrShape=[16,16,128], isTransposed}>
    #   #blocked = sizePerThread=[1,8], threadsPerWarp=[4,16], warpsPerCTA=[4,1]
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 8)),
        stride=((8, 512), (1, 2048)),
    )
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (4, 4, 4)),
        stride=((128, 4, 16, 2048), (1, 32, 4096)),
    )
    scale_mfma_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (2, 4)),
        stride=((8, 1, 128, 0), (4, 256)),
    )
    # A and B tiles are both [128, 128] here, which is exactly the shape the
    # 256x256 kernel derives shared_layout_b for -- so its padded layout carries
    # over verbatim to both. Left on the compiler default these allocs gave
    # LDSBankConflict = 18.9% with MfmaUtil at 12%, i.e. the MFMA unit idle 88% of
    # the time while MemUnitStalled was ~0: LDS-latency bound, not memory bound.
    # The [[1024, 32]] pad is what breaks the conflicting stride.
    shared_layout_ab: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [[1024, 32]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [BLOCK_M, BLOCK_K // 2],
    )
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Spread consecutive pids across XCDs so each XCD gets a contiguous stripe
    # of the (m, n) grid rather than a strided one.
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
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
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m > 0)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    # 2 x 16 KB for A/B plus 2 x 1 KB of scale scratch = 66 KB, so unlike the
    # 256x256 kernel (138 KB) this fits two workgroups per CU on the 160 KB
    # CDNA4 LDS -- which is the point, since these shapes cannot fill the grid.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED), tlx.dtype_of(a_ptr), 2, layout=shared_layout_ab)
    smem_b = tlx.local_alloc((BLOCK_N, BLOCK_K_PACKED), tlx.dtype_of(b_ptr), 2, layout=shared_layout_ab)
    # 2-deep, not 1: a 1-deep scratchpad is store->load->(next iter)store on the
    # same buffer, a cross-iteration WAR across the 4 warps that showed up as a
    # nondeterministic result (same inputs, different output run to run).

    offs_am = tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K_PACKED)
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_offsets_next = a_offsets + BLOCK_K_PACKED * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am

    offs_bn = tl.arange(0, BLOCK_N)
    offs_bk = tl.arange(0, BLOCK_K_PACKED)
    b_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_offsets_next = b_offsets + BLOCK_K_PACKED * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Pre-shuffled scale addressing. A and B scale tiles are both
    # (128, BLOCK_K_SCALE), so one offsets tensor serves both. Within a tile the
    # byte for (row, k) sits at
    #   (row%16)*8 + (k%4)*128 + ((row//16)%2)*512 + (k//4) + (row//32)*2
    # which is scale_mfma_layout's (lane, register) -> element map linearised with
    # the register index innermost -- so each lane's 8 bytes are contiguous and the
    # load is one buffer_load_dwordx2 straight into MFMA operand layout. No LDS.
    # shuffle_a_scale / shuffle_b_scale below are the host side of this formula.
    TILE_S: tl.constexpr = BLOCK_M * BLOCK_K_SCALE
    offs_srow = tl.arange(0, BLOCK_M)
    offs_sk = tl.arange(0, BLOCK_K_SCALE)
    s_row = (offs_srow % 16) * 8 + ((offs_srow // 16) % 2) * 512 + (offs_srow // 32) * 2
    s_col = (offs_sk % 4) * 128 + (offs_sk // 4)
    scale_offsets = tl.add(s_row[:, None], s_col[None, :], sanitize_overflow=False)
    scale_offsets = tlx.require_layout(scale_offsets, scale_mfma_layout)
    scale_offsets_next = tl.add(scale_offsets, TILE_S, sanitize_overflow=False)
    a_scales_base = a_scales_ptr + pid_m * (K // 32 // BLOCK_K_SCALE) * TILE_S
    b_scales_base = b_scales_ptr + pid_n * (K // 32 // BLOCK_K_SCALE) * TILE_S

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Runtime trip count so every supported K shares one compiled kernel.
    iter_max = K // BLOCK_K
    tl.assume(iter_max > 1)

    # Prime both stages, one async group each.
    tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
    tlx.buffer_load_to_local(smem_b[0], b_base, b_offsets)
    a_sc_buf0 = tlx.require_layout(tlx.buffer_load(a_scales_base, scale_offsets, contiguity=8), scale_mfma_layout)
    b_sc_buf0 = tlx.require_layout(tlx.buffer_load(b_scales_base, scale_offsets, contiguity=8), scale_mfma_layout)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
    tlx.buffer_load_to_local(smem_b[1], b_base, b_offsets_next)
    a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, scale_offsets_next, contiguity=8), scale_mfma_layout)
    b_sc_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, scale_offsets_next, contiguity=8), scale_mfma_layout)
    tlx.async_load_commit_group()

    a_base += BLOCK_K_PACKED * stride_ak * 2
    b_base += BLOCK_K_PACKED * stride_bk * 2
    a_scales_base += TILE_S * 2
    b_scales_base += TILE_S * 2

    tlx.async_load_wait_group(1)
    a = tlx.local_load(smem_a[0])
    b = tlx.local_load(tlx.local_trans(smem_b[0]))
    a_sc = a_sc_buf0
    b_sc = b_sc_buf0

    # Unrolled by 2 so the two LDS stages alternate without an index variable.
    # Issue the refill for the stage we just consumed BEFORE waiting on the other
    # stage. With only two stages there is exactly one group per stage, so if the
    # refill came after the wait there would be a single group outstanding and
    # async_load_wait_group(1) would be satisfied by 1 <= 1 without waiting for
    # anything -- the next local_load would then read a buffer whose DMA was still
    # in flight. That raced: ~20% of the elements of a tile (one warp's share)
    # differed run to run. Refill-then-wait keeps two groups in flight, so the
    # wait actually retires the one being read and still overlaps the other.
    for _ in tl.range(0, iter_max - 2, 2, num_stages=1):
        acc = tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
        tlx.buffer_load_to_local(smem_b[0], b_base, b_offsets)
        a_sc_buf0 = tlx.require_layout(tlx.buffer_load(a_scales_base, scale_offsets, contiguity=8), scale_mfma_layout)
        b_sc_buf0 = tlx.require_layout(tlx.buffer_load(b_scales_base, scale_offsets, contiguity=8), scale_mfma_layout)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(1)
        a_n = tlx.local_load(smem_a[1])
        b_n = tlx.local_load(tlx.local_trans(smem_b[1]))
        a_sc_n = a_sc_buf1
        b_sc_n = b_sc_buf1

        acc = tl.dot_scaled(a_n, a_sc_n, "e2m1", b_n, b_sc_n, "e2m1", acc)
        tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
        tlx.buffer_load_to_local(smem_b[1], b_base, b_offsets_next)
        a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, scale_offsets_next, contiguity=8),
                                       scale_mfma_layout)
        b_sc_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, scale_offsets_next, contiguity=8),
                                       scale_mfma_layout)
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(1)
        a = tlx.local_load(smem_a[0])
        b = tlx.local_load(tlx.local_trans(smem_b[0]))
        a_sc = a_sc_buf0
        b_sc = b_sc_buf0

        a_base += BLOCK_K_PACKED * stride_ak * 2
        b_base += BLOCK_K_PACKED * stride_bk * 2
        a_scales_base += TILE_S * 2
        b_scales_base += TILE_S * 2

    # Drain the two in-flight stages.
    acc = tl.dot_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)
    tlx.async_load_wait_group(0)
    a_n = tlx.local_load(smem_a[1])
    b_n = tlx.local_load(tlx.local_trans(smem_b[1]))
    a_sc_n = a_sc_buf1
    b_sc_n = b_sc_buf1
    acc = tl.dot_scaled(a_n, a_sc_n, "e2m1", b_n, b_sc_n, "e2m1", acc)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = tl.add(
        tl.mul(stride_cm, offs_cm, sanitize_overflow=False)[:, None],
        tl.mul(stride_cn, offs_cn, sanitize_overflow=False)[None, :],
        sanitize_overflow=False,
    )
    c_offsets = tlx.require_layout(c_offsets, store_layout_c)
    c_tile_base = c_ptr + pid_m * BLOCK_M * stride_cm
    # Pin the accumulator before narrowing so the store-layout requirement
    # redistributes bf16 rather than propagating back through the truncf to f32.
    # Unpinned the TTGIR reads convert_layout(f32, #mma -> #blocked) then truncf,
    # i.e. 4 bytes per element through LDS instead of 2.
    acc = tlx.require_layout(acc, accumulator_layout)
    c = acc.to(c_ptr.dtype.element_ty)
    c = tlx.require_layout(c, store_layout_c)
    tlx.buffer_store(c, c_tile_base, c_offsets)


def _shuffle_scale_128(scales):
    """Permute raw (rows, K//32) e8m0 scales into the layout the kernel loads.

    Host-side, done once, mirroring what AITER gets from
    aiter.ops.shuffle.shuffle_scale. Raw scales are row-contiguous, so the 8 e8m0
    a lane needs for one MFMA are scattered; reordering into (row_tile, k_tile)
    tiles of 128*BLOCK_K_SCALE bytes makes each lane's 8 bytes contiguous, so the
    kernel reads them with one buffer_load and never touches LDS.
    Pure permutation; shape and dtype preserved.
    """
    rows, nk = scales.shape
    assert rows % 128 == 0 and nk % 8 == 0, "scales must tile by 128 x 8"
    t = scales.contiguous().reshape(rows // 128, 4, 2, 16, nk // 8, 2, 4)
    #        (row_tile, row//32, (row//16)%2, row%16, k_tile, k//4, k%4)
    # ->     (row_tile, k_tile, (row//16)%2, k%4, row%16, row//32, k//4)
    return t.permute(0, 4, 2, 6, 3, 1, 5).contiguous().reshape(rows, nk)


def shuffle_a_scale(a_scales):
    return _shuffle_scale_128(a_scales)


def shuffle_b_scale(b_scales):
    return _shuffle_scale_128(b_scales)


def matmul(a, b, a_scales, b_scales):
    assert a.dtype is torch.uint8
    assert b.dtype is torch.uint8
    assert a_scales.dtype is torch.uint8
    assert b_scales.dtype is torch.uint8
    assert a.is_cuda and b.is_cuda and a_scales.is_cuda and b_scales.is_cuda

    M = a.shape[0]
    K_packed = a.shape[1]
    K = K_packed * 2
    N = b.shape[0]

    assert b.shape[1] == K_packed, "B must have shape (N, K // 2)"
    assert a_scales.shape == (M, K // 32), "A scales must have shape (M, K // 32)"
    assert b_scales.shape == (N, K // 32), "B scales must have shape (N, K // 32)"
    assert a_scales.is_contiguous(), "A scales must be pre-shuffled (see shuffle_a_scale)"
    assert b_scales.is_contiguous(), "B scales must be pre-shuffled (see shuffle_b_scale)"

    assert M % BLOCK_M == 0, "M must be a multiple of 128"
    assert N % BLOCK_N == 0, "N must be a multiple of 128"
    assert K >= 2 * BLOCK_K and K % (2 * BLOCK_K) == 0, "K must be at least 512 and a multiple of 512"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _a4w4_kernel_128[(grid_mn, )](
        a,
        b,
        c,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=grid_mn,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
