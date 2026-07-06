"""TLX MXFP4 GEMM for gfx950.

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


@triton.jit
def _a4w4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K: tl.constexpr,
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

    g_load_layout_a: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0], [128, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K_PACKED],
    )
    g_load_layout_b: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [4, 0], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[HALF_N, BLOCK_K_PACKED],
    )
    blocked_scales_a: tl.constexpr = tlx.blocked_layout_encoding.make(
        [8, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )
    blocked_scales_b: tl.constexpr = tlx.blocked_layout_encoding.make(
        [4, 1],
        [32, 2],
        [1, 4],
        [0, 1],
    )
    shared_layout_a: tl.constexpr = tlx.padded_shared_layout_encoding.with_offsets(
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
            [128, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K_PACKED],
    )
    shared_layout_b: tl.constexpr = tlx.padded_shared_layout_encoding.with_offsets(
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
        [],
        [HALF_N, BLOCK_K_PACKED],
    )
    shared_scales: tl.constexpr = tlx.swizzled_shared_layout_encoding.with_order([0, 1])
    mfma_layout: tl.constexpr = tlx.amd_mfma_layout_encoding.make(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(0, mfma_layout, 16)
    dot_b_layout: tl.constexpr = tlx.dot_operand_layout_encoding.make(1, mfma_layout, 16)
    scale_a_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[[0, 4], [32, 0], [64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
        warp_bases=[[0, 0], [16, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_K_SCALE],
    )
    scale_b_layout: tl.constexpr = tlx.distributed_linear_layout_encoding.make(
        reg_bases=[[0, 4], [32, 0], [64, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
        warp_bases=[[16, 0], [0, 0]],
        block_bases=[],
        shape=[HALF_N, BLOCK_K_SCALE],
    )
    store_layout_c: tl.constexpr = tlx.blocked_layout_encoding.make(
        [1, 8],
        [4, 16],
        [4, 1],
        [1, 0],
    )
    store_layout_c_m: tl.constexpr = tlx.slice_layout_encoding.make(1, store_layout_c)
    store_layout_c_n: tl.constexpr = tlx.slice_layout_encoding.make(0, store_layout_c)
    scale_a_layout_k: tl.constexpr = tlx.slice_layout_encoding.make(0, blocked_scales_a)
    scale_a_layout_m: tl.constexpr = tlx.slice_layout_encoding.make(1, blocked_scales_a)
    scale_b_layout_k: tl.constexpr = tlx.slice_layout_encoding.make(0, blocked_scales_b)
    scale_b_layout_n: tl.constexpr = tlx.slice_layout_encoding.make(1, blocked_scales_b)

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
    smem_as = tlx.local_alloc((BLOCK_M, BLOCK_K_SCALE), tlx.dtype_of(a_scales_ptr), 1, layout=shared_scales)
    smem_bs = tlx.local_alloc((HALF_N, BLOCK_K_SCALE), tlx.dtype_of(b_scales_ptr), 1, layout=shared_scales)

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

    offs_sk_a = tlx.require_layout(tl.arange(0, BLOCK_K_SCALE), scale_a_layout_k)
    offs_asm = tlx.require_layout(tl.arange(0, BLOCK_M), scale_a_layout_m)
    offs_asm_base = tlx.require_layout(tl.full((BLOCK_M, ), pid_m * BLOCK_M, tl.int32), scale_a_layout_m)
    offs_asm = tl.add(offs_asm_base, offs_asm, sanitize_overflow=False)
    a_scale_m_stride = tlx.require_layout(tl.full((BLOCK_M, 1), stride_asm, tl.int32), blocked_scales_a)
    a_scale_m_offsets = tl.mul(offs_asm[:, None], a_scale_m_stride, sanitize_overflow=False)
    a_scale_k_stride = tlx.require_layout(tl.full((1, BLOCK_K_SCALE), stride_ask, tl.int32), blocked_scales_a)
    a_scale_k_offsets = tl.mul(offs_sk_a[None, :], a_scale_k_stride, sanitize_overflow=False)
    a_scale_offsets = tl.add(a_scale_m_offsets, a_scale_k_offsets, sanitize_overflow=False)
    a_scale_next_delta = tlx.require_layout(
        tl.full((BLOCK_M, BLOCK_K_SCALE), BLOCK_K_SCALE * stride_ask, tl.int32),
        blocked_scales_a,
    )
    a_scale_offsets_next = tl.add(a_scale_offsets, a_scale_next_delta, sanitize_overflow=False)
    offs_sk_b = tlx.require_layout(tl.arange(0, BLOCK_K_SCALE), scale_b_layout_k)
    offs_bsn = tlx.require_layout(tl.arange(0, HALF_N), scale_b_layout_n)
    offs_bsn_base = tlx.require_layout(tl.full((HALF_N, ), pid_n * BLOCK_N, tl.int32), scale_b_layout_n)
    offs_bsn = tl.add(offs_bsn_base, offs_bsn, sanitize_overflow=False)
    b_scale_n_stride = tlx.require_layout(tl.full((HALF_N, 1), stride_bsn, tl.int32), blocked_scales_b)
    b_scale_n_offsets = tl.mul(offs_bsn[:, None], b_scale_n_stride, sanitize_overflow=False)
    b_scale_k_stride = tlx.require_layout(tl.full((1, BLOCK_K_SCALE), stride_bsk, tl.int32), blocked_scales_b)
    b_scale_k_offsets = tl.mul(offs_sk_b[None, :], b_scale_k_stride, sanitize_overflow=False)
    b_scale_left_offsets = tl.add(b_scale_n_offsets, b_scale_k_offsets, sanitize_overflow=False)
    b_scale_right_step = tl.mul(stride_bsn, HALF_N, sanitize_overflow=False)
    b_scale_right_delta = tlx.require_layout(
        tl.full((HALF_N, BLOCK_K_SCALE), b_scale_right_step, tl.int32),
        blocked_scales_b,
    )
    b_scale_right_offsets = tl.add(b_scale_left_offsets, b_scale_right_delta, sanitize_overflow=False)
    b_scale_next_delta = tlx.require_layout(
        tl.full((HALF_N, BLOCK_K_SCALE), BLOCK_K_SCALE * stride_bsk, tl.int32),
        blocked_scales_b,
    )
    b_scale_left_offsets_next = tl.add(b_scale_left_offsets, b_scale_next_delta, sanitize_overflow=False)
    b_scale_right_offsets_next = tl.add(b_scale_right_offsets, b_scale_next_delta, sanitize_overflow=False)
    a_scales_base = a_scales_ptr
    b_scales_base = b_scales_ptr

    acc_left = tlx.require_layout(tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32), mfma_layout)
    acc_right = tlx.require_layout(tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32), mfma_layout)

    iter_max: tl.constexpr = K // BLOCK_K
    tl.assume(iter_max > 3)

    tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
    tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
    a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets), blocked_scales_a)
    b_sc_left_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_left_offsets), blocked_scales_b)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
    b_sc_right_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_right_offsets), blocked_scales_b)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
    tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
    a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next), blocked_scales_a)
    b_sc_left_buf3 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_left_offsets_next), blocked_scales_b)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
    b_sc_right_buf3 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_right_offsets_next), blocked_scales_b)
    tlx.async_load_commit_group()

    a_base += BLOCK_K_PACKED * stride_ak * 2
    b_base += BLOCK_K_PACKED * stride_bk * 2
    a_scales_base += BLOCK_K_SCALE * stride_ask * 2
    b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

    tlx.async_load_wait_group(3)
    a = tlx.require_layout(tlx.local_load(smem_a[0], relaxed=True), dot_a_layout)
    b_left = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True), dot_b_layout)
    tlx.local_store(smem_as[0], a_sc_buf1)
    tlx.local_store(smem_bs[0], b_sc_left_buf1)
    a_sc_reg_buf0 = tlx.require_layout(tlx.local_load(smem_as[0]), scale_a_layout)
    b_sc_left_reg_buf0 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)

    for _ in tl.range(0, iter_max - 2, 2, num_stages=1):
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tlx.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True), dot_b_layout)
            tlx.local_store(smem_bs[0], b_sc_right_buf1)
            b_sc_right_reg_buf0 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)
            tlx.buffer_load_to_local(smem_a[0], a_base, a_offsets)
            tlx.buffer_load_to_local(smem_b_left[0], b_base, b_left_offsets)
            a_sc_buf1 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets), blocked_scales_a)
            b_sc_left_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_left_offsets),
                                                blocked_scales_b)
            tlx.async_load_commit_group()

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tlx.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a_next = tlx.require_layout(tlx.local_load(smem_a[1], relaxed=True), dot_a_layout)
            b_left = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True), dot_b_layout)
            tlx.local_store(smem_as[0], a_sc_buf3)
            tlx.local_store(smem_bs[0], b_sc_left_buf3)
            a_sc_reg_buf2 = tlx.require_layout(tlx.local_load(smem_as[0]), scale_a_layout)
            b_sc_left_reg_buf2 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)
            tlx.buffer_load_to_local(smem_b_right[0], b_base, b_right_offsets)
            b_sc_right_buf1 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_right_offsets),
                                                 blocked_scales_b)
            tlx.async_load_commit_group()

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tlx.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True), dot_b_layout)
            tlx.local_store(smem_bs[0], b_sc_right_buf3)
            b_sc_right_reg_buf2 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)
            tlx.buffer_load_to_local(smem_a[1], a_base, a_offsets_next)
            tlx.buffer_load_to_local(smem_b_left[1], b_base, b_left_offsets_next)
            a_sc_buf3 = tlx.require_layout(tlx.buffer_load(a_scales_base, a_scale_offsets_next),
                                           blocked_scales_a)
            b_sc_left_buf3 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_left_offsets_next),
                                                blocked_scales_b)
            tlx.async_load_commit_group()

        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tlx.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a = tlx.require_layout(tlx.local_load(smem_a[0], relaxed=True), dot_a_layout)
            b_left = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True), dot_b_layout)
            tlx.local_store(smem_as[0], a_sc_buf1)
            tlx.local_store(smem_bs[0], b_sc_left_buf1)
            a_sc_reg_buf0 = tlx.require_layout(tlx.local_load(smem_as[0]), scale_a_layout)
            b_sc_left_reg_buf0 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)
            tlx.buffer_load_to_local(smem_b_right[1], b_base, b_right_offsets_next)
            b_sc_right_buf3 = tlx.require_layout(tlx.buffer_load(b_scales_base, b_scale_right_offsets_next),
                                                 blocked_scales_b)
            tlx.async_load_commit_group()

        a_base += BLOCK_K_PACKED * stride_ak * 2
        b_base += BLOCK_K_PACKED * stride_bk * 2
        a_scales_base += BLOCK_K_SCALE * stride_ask * 2
        b_scales_base += BLOCK_K_SCALE * stride_bsk * 2

    acc_left = tlx.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_left, b_sc_left_reg_buf0, "e2m1", acc_left)
    tlx.async_load_wait_group(2)
    b_right = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True), dot_b_layout)
    tlx.local_store(smem_bs[0], b_sc_right_buf1)
    b_sc_right_reg_buf0 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)

    acc_right = tlx.dot_scaled(a, a_sc_reg_buf0, "e2m1", b_right, b_sc_right_reg_buf0, "e2m1", acc_right)
    tlx.async_load_wait_group(1)
    a_next = tlx.require_layout(tlx.local_load(smem_a[1], relaxed=True), dot_a_layout)
    b_left = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True), dot_b_layout)
    tlx.local_store(smem_as[0], a_sc_buf3)
    tlx.local_store(smem_bs[0], b_sc_left_buf3)
    a_sc_reg_buf2 = tlx.require_layout(tlx.local_load(smem_as[0]), scale_a_layout)
    b_sc_left_reg_buf2 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)

    acc_left = tlx.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_left, b_sc_left_reg_buf2, "e2m1", acc_left)
    tlx.async_load_wait_group(0)
    b_right = tlx.require_layout(tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True), dot_b_layout)
    tlx.local_store(smem_bs[0], b_sc_right_buf3)
    b_sc_right_reg_buf2 = tlx.require_layout(tlx.local_load(smem_bs[0]), scale_b_layout)

    offs_cm = tl.arange(0, BLOCK_M)
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    c_row_offsets = tlx.require_layout(tl.mul(stride_cm, offs_cm, sanitize_overflow=False), store_layout_c_m)
    c_col_offsets = tlx.require_layout(tl.mul(stride_cn, offs_cn_left, sanitize_overflow=False), store_layout_c_n)
    c_left_offsets = tl.add(c_row_offsets[:, None], c_col_offsets[None, :], sanitize_overflow=False)
    c_right_delta = tlx.require_layout(
        tl.mul(tl.full((BLOCK_M, HALF_N), HALF_N, tl.int32), stride_cn, sanitize_overflow=False),
        store_layout_c,
    )
    c_right_offsets = tl.add(c_left_offsets, c_right_delta, sanitize_overflow=False)
    c_tile_base = c_ptr + pid_m * BLOCK_M * stride_cm
    c_left_offsets = tlx.require_layout(c_left_offsets, store_layout_c)
    c_left = tlx.require_layout(tlx.cast_preserve_layout(acc_left, c_ptr.dtype.element_ty), store_layout_c)
    tlx.buffer_store(c_left, c_tile_base, c_left_offsets)

    acc_right = tlx.dot_scaled(a_next, a_sc_reg_buf2, "e2m1", b_right, b_sc_right_reg_buf2, "e2m1", acc_right)
    c_right_offsets = tlx.require_layout(c_right_offsets, store_layout_c)
    c_right = tlx.require_layout(tlx.cast_preserve_layout(acc_right, c_ptr.dtype.element_ty), store_layout_c)
    tlx.buffer_store(c_right, c_tile_base, c_right_offsets)


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
    assert a_scales.stride(0) == 1, "A scales must be contiguous along M"
    assert b_scales.stride(0) == 1, "B scales must be contiguous along N"

    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 256
    assert M % BLOCK_M == 0, "M must be a multiple of 256"
    assert N % BLOCK_N == 0, "N must be a multiple of 256"
    assert K >= 4 * BLOCK_K and K % (2 * BLOCK_K) == 0, "K must be at least 1024 and a multiple of 512"

    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _a4w4_kernel[(grid_mn, )](
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
