"""TLX MXFP4 GEMM for gfx950 -- 128x256 tile.

Exists for 2048x4096x8192, where it is the only tile that fills the machine:
  128x256 -> (2048/128) x (4096/256) = 16 x 16 = 256 workgroups = the CU count
  128x128 -> 16 x 32 = 512 (two per CU, half the arithmetic intensity each)
  256x256 ->  8 x 16 = 128 (half the machine idle)
AITER selects its own 128x256 kernel here for the same reason.

Derived from matmul_kernel.py (256x256) with BLOCK_M halved. Only the
M-dependent pieces change -- accumulator M-repeat 8->4, store M-repeat 16->8,
scale_a_layout M-repeat 8->4 (so 8 e8m0 per lane, not 16, hence contiguity=8
and halved A-scale offset strides), shared_layout_a drops the [128, 0] base,
and the host shuffle tiles by 128 rows. Everything on the B/HALF_N side is
independent of BLOCK_M and carries over verbatim.

Inherit the parent's config: LLIR sched + force-AGPR TOGETHER, i.e.
  TRITON_ENABLE_LLIR_SCHED=1 TRITON_LLVM_OPTS=amdgpu-mfma-vgpr-form=false
Note the flag has NO leading dash -- the dash form is silently ignored.
Verify against plain once measured; this tile may sit below the 510/512 VGPR
ceiling that makes the pairing mandatory on the 256x256 parent.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _a4w4_kernel_128x256(
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
    BLOCK_K_PACKED: tl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_GROUP_SIZE
    HALF_N: tl.constexpr = BLOCK_N // 2

    g_load_layout_a: tl.constexpr = tlx.layout(
        shape=((8, 8, 4), (16, 4, 2)),
        stride=((16, 2048, 128), (1, 512, 16384)),
    )
    g_load_layout_b: tl.constexpr = tlx.layout(
        shape=((8, 8, 4), (16, 4)),
        stride=((16, 2048, 128), (1, 512)),
    )
    scale_load_layout_b: tl.constexpr = tlx.layout(
        shape=((32, 2, 4), (4, )),
        stride=((32, 1, 2), (8, )),
    )
    shared_layout_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
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
        [BLOCK_M, BLOCK_K_PACKED],
    )
    shared_layout_b: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
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
        [HALF_N, BLOCK_K_PACKED],
    )
    shared_scales: tl.constexpr = tlx.swizzled_layout(0, 0, 0, order=[0, 1])
    scale_a_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (2, 4)),
        stride=((8, 1, 0, 128), (4, 256)),
    )
    scale_b_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (2, 4)),
        stride=((8, 1, 128, 0), (4, 256)),
    )
    store_layout_c: tl.constexpr = tlx.layout(
        shape=((64, 4), (8, 8)),
        stride=((8, 512), (1, 2048)),
    )
    # Generalized Shape:Stride form of the gfx950 16x16x128 MFMA
    # accumulator layout for one [BLOCK_M, HALF_N] result tile.
    accumulator_layout: tl.constexpr = tlx.layout(
        shape=((16, 4, 2, 2), (4, 4, 4)),
        stride=((128, 4, 16, 2048), (1, 32, 4096)),
    )

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

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

    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED), tlx.dtype_of(a_ptr), 2, layout=shared_layout_a)
    smem_b_left = tlx.local_alloc((HALF_N, BLOCK_K_PACKED), tlx.dtype_of(b_ptr), 2, layout=shared_layout_b)
    smem_b_right = tlx.local_alloc((HALF_N, BLOCK_K_PACKED), tlx.dtype_of(b_ptr), 2, layout=shared_layout_b)
    smem_bs_left = tlx.local_alloc((HALF_N, BLOCK_K_SCALE), tlx.dtype_of(b_scales_ptr), 2, layout=shared_scales)
    smem_bs_right = tlx.local_alloc((HALF_N, BLOCK_K_SCALE), tlx.dtype_of(b_scales_ptr), 2, layout=shared_scales)

    offs_am = tl.arange(0, BLOCK_M)
    offs_ak = tl.arange(0, BLOCK_K_PACKED)
    a_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_offsets_next = a_offsets + BLOCK_K_PACKED * stride_ak
    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    a_offsets = tlx.require_layout(a_offsets, g_load_layout_a)
    a_offsets_next = tlx.require_layout(a_offsets_next, g_load_layout_a)

    offs_bn = tl.arange(0, HALF_N)
    offs_bk = tl.arange(0, BLOCK_K_PACKED)
    b_left_offsets = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    b_right_offsets = b_left_offsets + HALF_N * stride_bn
    b_left_offsets_next = b_left_offsets + BLOCK_K_PACKED * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K_PACKED * stride_bk
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn
    b_left_offsets = tlx.require_layout(b_left_offsets, g_load_layout_b)
    b_right_offsets = tlx.require_layout(b_right_offsets, g_load_layout_b)
    b_left_offsets_next = tlx.require_layout(b_left_offsets_next, g_load_layout_b)
    b_right_offsets_next = tlx.require_layout(b_right_offsets_next, g_load_layout_b)

    # Pre-shuffled A-scale addressing. The buffer is a flat sequence of
    # (BLOCK_M x BLOCK_K_SCALE) = 2048 B tiles in (m_tile, k_tile) order; within a
    # tile the byte for (m, k) lives at
    #   (m%16)*16 + (k%4)*256 + ((m//16)%2)*1024 + (k//4) + (m//32)*2
    # which is exactly scale_a_layout's (lane, register) -> element map linearized
    # with the register index innermost. That is what makes each lane's 16 bytes
    # contiguous; shuffle_a_scale() below is the host side of the same formula.
    TILE_AS: tl.constexpr = BLOCK_M * BLOCK_K_SCALE
    offs_asm = tl.arange(0, BLOCK_M)
    offs_sk_a = tl.arange(0, BLOCK_K_SCALE)
    a_scale_m_offsets = (offs_asm % 16) * 8 + ((offs_asm // 16) % 2) * 512 + (offs_asm // 32) * 2
    a_scale_k_offsets = (offs_sk_a % 4) * 128 + (offs_sk_a // 4)
    a_scale_offsets = tl.add(a_scale_m_offsets[:, None], a_scale_k_offsets[None, :], sanitize_overflow=False)
    a_scale_offsets = tlx.require_layout(a_scale_offsets, scale_a_layout)
    a_scale_offsets_next = tl.add(a_scale_offsets, TILE_AS, sanitize_overflow=False)
    offs_sk_b = tl.arange(0, BLOCK_K_SCALE)
    offs_bsn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    b_scale_n_offsets = tl.mul(offs_bsn[:, None], stride_bsn, sanitize_overflow=False)
    b_scale_k_offsets = tl.mul(offs_sk_b[None, :], stride_bsk, sanitize_overflow=False)
    b_scale_left_offsets = tl.add(b_scale_n_offsets, b_scale_k_offsets, sanitize_overflow=False)
    b_scale_left_offsets = tlx.require_layout(b_scale_left_offsets, scale_load_layout_b)
    b_scale_right_step = tl.mul(stride_bsn, HALF_N, sanitize_overflow=False)
    b_scale_right_offsets = tl.add(b_scale_left_offsets, b_scale_right_step, sanitize_overflow=False)
    b_scale_next_delta = BLOCK_K_SCALE * stride_bsk
    b_scale_left_offsets_next = tl.add(b_scale_left_offsets, b_scale_next_delta, sanitize_overflow=False)
    b_scale_right_offsets_next = tl.add(b_scale_right_offsets, b_scale_next_delta, sanitize_overflow=False)
    a_scales_base = a_scales_ptr + pid_m * (K // 32 // BLOCK_K_SCALE) * TILE_AS
    b_scales_base = b_scales_ptr

    acc_left = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    # Keep the trip count runtime so all supported K sizes share one compiled kernel.
    iter_max = K // BLOCK_K
    tl.assume(iter_max > 3)

    tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
    tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
    a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8), scale_a_layout)
    tlx.buffer_load_to_local(smem_bs_left[0], b_scales_base, b_scale_left_offsets)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
    tlx.buffer_load_to_local(smem_bs_right[0], b_scales_base, b_scale_right_offsets)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
    tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
    a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8), scale_a_layout)
    tlx.buffer_load_to_local(smem_bs_left[1], b_scales_base, b_scale_left_offsets_next)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
    tlx.buffer_load_to_local(smem_bs_right[1], b_scales_base, b_scale_right_offsets_next)
    tlx.async_load_commit_group()

    a_base += BLOCK_K_PACKED * stride_ak * 2
    b_base += BLOCK_K_PACKED * stride_bk * 2
    a_scales_base += TILE_AS * 2
    b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

    tlx.async_load_wait_group(3)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
    a_sc_reg_buf0 = a_sc_buf1
    b_sc_left_reg_buf0 = tlx.local_load(smem_bs_left[0], layout=scale_b_layout)

    # K = 1024 per loop body, matching AITER's 128x256 kernel (256 MFMA / 8
    # s_barrier / 48 buffer_load_dwordx4 per body, against our 128 / 4 / 24).
    #
    # Done by unrolling the 4-quarter block twice, NOT by doubling BLOCK_K the way
    # the 64x128 kernel does it. BLOCK_K=512 here would want A 64 KB + B 128 KB +
    # scales 8 KB = 205 KB against a 160 KB CU, so it does not fit. Unrolling keeps
    # LDS byte-identical and buys loop-overhead amortisation plus a 2x bigger
    # scheduling window -- it does NOT halve barrier density the way the 64x128
    # change did, because the buffer count is unchanged so barriers scale with the
    # body. AITER sits at that same 8 barriers per 1024 K.
    #
    # The loop covers iter_max - 4 iterations in steps of 4 (so K % 1024 == 0) and
    # one 4-quarter block is peeled AFTER it, not before: the loop's live-in state
    # has to come from the prologue. Peeling ahead of the loop instead makes layout
    # inference resolve the accumulators to #blocked and the A scales to a second,
    # narrower linear layout, and the kernel fails to compile with
    #   size mismatch when packing elements for LLVM struct expected 4 but got 8
    for _ in tl.range(0, iter_max - 4, 4, num_stages=1):
        acc_left = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
        b_sc_right_reg_buf0 = tlx.local_load(smem_bs_right[0], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
        tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
        a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8), scale_a_layout)
        tlx.buffer_load_to_local(smem_bs_left[0], b_scales_base, b_scale_left_offsets)
        tlx.async_load_commit_group()

        acc_right = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
        tlx.async_load_wait_group(2)
        a_next = tlx.local_load(smem_a[1], relaxed=True)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
        a_sc_reg_buf2 = a_sc_buf3
        b_sc_left_reg_buf2 = tlx.local_load(smem_bs_left[1], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
        tlx.buffer_load_to_local(smem_bs_right[0], b_scales_base, b_scale_right_offsets)
        tlx.async_load_commit_group()

        acc_left = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
        b_sc_right_reg_buf2 = tlx.local_load(smem_bs_right[1], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
        tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
        a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8),
                                       scale_a_layout)
        tlx.buffer_load_to_local(smem_bs_left[1], b_scales_base, b_scale_left_offsets_next)
        tlx.async_load_commit_group()

        acc_right = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
        tlx.async_load_wait_group(2)
        a = tlx.local_load(smem_a[0], relaxed=True)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
        a_sc_reg_buf0 = a_sc_buf1
        b_sc_left_reg_buf0 = tlx.local_load(smem_bs_left[0], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
        tlx.buffer_load_to_local(smem_bs_right[1], b_scales_base, b_scale_right_offsets_next)
        tlx.async_load_commit_group()

        a_base += BLOCK_K_PACKED * stride_ak * 2
        b_base += BLOCK_K_PACKED * stride_bk * 2
        a_scales_base += TILE_AS * 2
        b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

        acc_left = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
        b_sc_right_reg_buf0 = tlx.local_load(smem_bs_right[0], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
        tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
        a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8), scale_a_layout)
        tlx.buffer_load_to_local(smem_bs_left[0], b_scales_base, b_scale_left_offsets)
        tlx.async_load_commit_group()

        acc_right = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
        tlx.async_load_wait_group(2)
        a_next = tlx.local_load(smem_a[1], relaxed=True)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
        a_sc_reg_buf2 = a_sc_buf3
        b_sc_left_reg_buf2 = tlx.local_load(smem_bs_left[1], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
        tlx.buffer_load_to_local(smem_bs_right[0], b_scales_base, b_scale_right_offsets)
        tlx.async_load_commit_group()

        acc_left = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
        tlx.async_load_wait_group(2)
        b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
        b_sc_right_reg_buf2 = tlx.local_load(smem_bs_right[1], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
        tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
        a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8),
                                       scale_a_layout)
        tlx.buffer_load_to_local(smem_bs_left[1], b_scales_base, b_scale_left_offsets_next)
        tlx.async_load_commit_group()

        acc_right = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
        tlx.async_load_wait_group(2)
        a = tlx.local_load(smem_a[0], relaxed=True)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
        a_sc_reg_buf0 = a_sc_buf1
        b_sc_left_reg_buf0 = tlx.local_load(smem_bs_left[0], layout=scale_b_layout)
        tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
        tlx.buffer_load_to_local(smem_bs_right[1], b_scales_base, b_scale_right_offsets_next)
        tlx.async_load_commit_group()

        a_base += BLOCK_K_PACKED * stride_ak * 2
        b_base += BLOCK_K_PACKED * stride_bk * 2
        a_scales_base += TILE_AS * 2
        b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

    # Peeled 4-quarter block: the loop leaves 2 iterations beyond the 2 the
    # epilogue drains.
    acc_left = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
    tlx.async_load_wait_group(2)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
    b_sc_right_reg_buf0 = tlx.local_load(smem_bs_right[0], layout=scale_b_layout)
    tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
    tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
    a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets, contiguity=8), scale_a_layout)
    tlx.buffer_load_to_local(smem_bs_left[0], b_scales_base, b_scale_left_offsets)
    tlx.async_load_commit_group()

    acc_right = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
    tlx.async_load_wait_group(2)
    a_next = tlx.local_load(smem_a[1], relaxed=True)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
    a_sc_reg_buf2 = a_sc_buf3
    b_sc_left_reg_buf2 = tlx.local_load(smem_bs_left[1], layout=scale_b_layout)
    tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
    tlx.buffer_load_to_local(smem_bs_right[0], b_scales_base, b_scale_right_offsets)
    tlx.async_load_commit_group()

    acc_left = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
    tlx.async_load_wait_group(2)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
    b_sc_right_reg_buf2 = tlx.local_load(smem_bs_right[1], layout=scale_b_layout)
    tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
    tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
    a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next, contiguity=8), scale_a_layout)
    tlx.buffer_load_to_local(smem_bs_left[1], b_scales_base, b_scale_left_offsets_next)
    tlx.async_load_commit_group()

    acc_right = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
    tlx.async_load_wait_group(2)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
    a_sc_reg_buf0 = a_sc_buf1
    b_sc_left_reg_buf0 = tlx.local_load(smem_bs_left[0], layout=scale_b_layout)
    tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
    tlx.buffer_load_to_local(smem_bs_right[1], b_scales_base, b_scale_right_offsets_next)
    tlx.async_load_commit_group()

    a_base += BLOCK_K_PACKED * stride_ak * 2
    b_base += BLOCK_K_PACKED * stride_bk * 2
    a_scales_base += TILE_AS * 2
    b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

    acc_left = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
    tlx.async_load_wait_group(2)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
    b_sc_right_reg_buf0 = tlx.local_load(smem_bs_right[0], layout=scale_b_layout)

    acc_right = tl.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
    tlx.async_load_wait_group(1)
    a_next = tlx.local_load(smem_a[1], relaxed=True)
    b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
    a_sc_reg_buf2 = a_sc_buf3
    b_sc_left_reg_buf2 = tlx.local_load(smem_bs_left[1], layout=scale_b_layout)

    acc_left = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
    b_sc_right_reg_buf2 = tlx.local_load(smem_bs_right[1], layout=scale_b_layout)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    c_row_offsets = tl.mul(stride_cm, offs_cm, sanitize_overflow=False)
    c_col_offsets = tl.mul(stride_cn, offs_cn_left, sanitize_overflow=False)
    c_left_offsets = tl.add(c_row_offsets[:, None], c_col_offsets[None, :], sanitize_overflow=False)
    c_left_offsets = tlx.require_layout(c_left_offsets, store_layout_c)
    c_right_delta = tl.mul(HALF_N, stride_cn, sanitize_overflow=False)
    c_right_offsets = tl.add(c_left_offsets, c_right_delta, sanitize_overflow=False)
    c_tile_base = c_ptr + pid_m * BLOCK_M * stride_cm
    # Pin the accumulator layout before narrowing so that the store-layout
    # requirement redistributes bf16 rather than propagating back to f32
    # (+32 LDS writes and +32 LDS reads).
    acc_left = tlx.require_layout(acc_left, accumulator_layout)
    c_left = acc_left.to(c_ptr.dtype.element_ty)
    c_left = tlx.require_layout(c_left, store_layout_c)
    tlx.buffer_store(c_left, c_tile_base, c_left_offsets)

    acc_right = tl.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
    c_right_offsets = tlx.require_layout(c_right_offsets, store_layout_c)
    acc_right = tlx.require_layout(acc_right, accumulator_layout)
    c_right = acc_right.to(c_ptr.dtype.element_ty)
    c_right = tlx.require_layout(c_right, store_layout_c)
    tlx.buffer_store(c_right, c_tile_base, c_right_offsets)


def shuffle_a_scale(a_scales):
    """Permute raw (M, K//32) e8m0 A scales into the layout the kernel loads.

    Host-side and done once, mirroring what AITER gets from
    ``aiter.ops.shuffle.shuffle_scale`` on its A scales.

    Raw scales are M-contiguous, so the 16 e8m0 a lane needs for one MFMA are
    scattered (strides of 32 and 4*M) -- the kernel used to pay a
    ``ds_write`` + ``ds_read_b64_tr_b8`` LDS round-trip purely to transpose them.
    Reordering to (m_tile, k_tile) tiles of BLOCK_M*BLOCK_K_SCALE bytes, with
    (m, k) at ``(m%16)*8 + (k%4)*128 + ((m//16)%2)*512 + (k//4) + (m//32)*2``,
    makes each lane's 8 bytes contiguous -> one buffer_load, no LDS. (Halved
    from the 256x256 parent: BLOCK_M=128 gives 8 e8m0 per lane, not 16.)

    Shape and dtype are preserved; this is a pure permutation.
    """
    return _shuffle_scale_128(a_scales)


def _shuffle_scale_128(scales):
    rows, nk = scales.shape
    assert rows % 128 == 0 and nk % 8 == 0, "scales must tile by 128 x 8"
    t = scales.contiguous().reshape(rows // 128, 4, 2, 16, nk // 8, 2, 4)
    #            (tile, r//32, (r//16)%2, r%16, k_tile, k//4, k%4)
    # ->         (tile, k_tile, (r//16)%2, k%4, r%16, r//32, k//4)
    return t.permute(0, 4, 2, 6, 3, 1, 5).contiguous().reshape(rows, nk)


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
    # Both scale operands must be pre-shuffled by shuffle_a_scale() /
    # shuffle_b_scale() -- one-off host-side permutations, the same deal AITER gets
    # from aiter.ops.shuffle.shuffle_scale. That is what lets the kernel read them
    # straight into MFMA operand layout instead of routing them through LDS.
    assert a_scales.is_contiguous(), "A scales must be pre-shuffled (see shuffle_a_scale)"
    assert b_scales.stride(0) == 1, "B scales must be contiguous along N"

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256
    assert M % BLOCK_M == 0, "M must be a multiple of 128"
    assert N % BLOCK_N == 0, "N must be a multiple of 256"
    assert K >= 4 * BLOCK_K and K % (4 * BLOCK_K) == 0, "K must be at least 1024 and a multiple of 1024"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _a4w4_kernel_128x256[(grid_mn, )](
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
        GROUP_SIZE_M=4,
        NUM_XCDS=8,
        GRID_MN=grid_mn,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
