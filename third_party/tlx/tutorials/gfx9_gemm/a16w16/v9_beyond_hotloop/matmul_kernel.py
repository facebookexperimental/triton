"""v9_beyond_hotloop — v8 + grid-level scheduling (PID remap + workgroup swizzle).

The hot loop is fully tuned by v8; the remaining headroom is *outside* it. This
step adds two grid-level optimizations that change only which output tile each
workgroup computes (the loop body is identical to v8):
  * NUM_XCDS remap   — round-robin program ids across the 8 XCDs so consecutive
                       tiles land on different XCDs (better global/L2 balance).
  * GROUP_SIZE_M swizzle — group tiles in M so neighbouring workgroups reuse the
                       same B columns out of L2.
This is the final step and the TLX-only finale (Gluon stops at beyond_hotloop).
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# Inherited from v8: keep the LLVM post-RA machine scheduler from re-ordering the
# warp_pipeline_stage mem/MFMA interleave. Equivalent to TRITON_DISABLE_POST_MISCHED=1,
# baked in so the kernel ships with its intended schedule (~+1-2%). setdefault()
# lets an explicit env override win. See python/src/llvm.cc (enable-post-misched=false).
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")


@triton.jit
def v9_beyond_hotloop(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # ── Grid-level scheduling (the only change vs v8) ──
    # Remap pid across XCDs, then swizzle into GROUP_SIZE_M-tall column groups.
    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
        pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)

    HALF_N: tl.constexpr = BLOCK_N // 2

    # The bank-conflict-avoiding padded shared layout (shown explicitly in v3)
    # is now inferred by the compiler from how these buffers feed tl.dot.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 2)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tl.float16, 2)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tl.float16, 2)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_off = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    bl_off = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    br_off = bl_off + HALF_N * stride_bn
    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    acc_left = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_M, HALF_N), dtype=tl.float32)

    iterMax: tl.constexpr = K // BLOCK_K

    # ── Prologue: fill both buffers ──
    tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr, bl_off + b_k)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_ptr, br_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    tlx.buffer_load_to_local(smem_a[1], a_ptr, a_off + a_k)
    tlx.buffer_load_to_local(smem_b_left[1], b_ptr, bl_off + b_k)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_ptr, br_off + b_k)
    tlx.async_load_commit_group()
    a_k += BLOCK_K * stride_ak
    b_k += BLOCK_K * stride_bk

    tlx.async_load_wait_group(3)
    a = tlx.local_load(smem_a[0], relaxed=True)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)

    # ── Main loop: step 2, four regions per body (identical to v8) ──
    for k in tl.range(0, iterMax - 2, 2, num_stages=1):
        # ──── Region 0 ────
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tl.dot(a, b_left, acc_left)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a[0], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smem_b_left[0], b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

        # ──── Region 1 ────
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tl.dot(a, b_right, acc_right)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a = tlx.local_load(smem_a[1], relaxed=True)
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_ptr, br_off + b_k)
            tlx.async_load_commit_group()

        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

        # ──── Region 2 ────
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_left = tl.dot(a, b_left, acc_left)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a[1], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smem_b_left[1], b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

        # ──── Region 3 ────
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_right = tl.dot(a, b_right, acc_right)
        with tlx.warp_pipeline_stage("mem", priority=1):
            tlx.async_load_wait_group(2)
            a = tlx.local_load(smem_a[0], relaxed=True)
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_ptr, br_off + b_k)
            tlx.async_load_commit_group()

        a_k += BLOCK_K * stride_ak
        b_k += BLOCK_K * stride_bk

    # ── Epilogue: drain the last two K iterations ──
    acc_left = tl.dot(a, b_left, acc_left)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(smem_b_right[0], relaxed=True)

    acc_right = tl.dot(a, b_right, acc_right)
    a = tlx.local_load(smem_a[1], relaxed=True)
    b_left = tlx.local_load(smem_b_left[1], relaxed=True)

    acc_left = tl.dot(a, b_left, acc_left)
    b_right = tlx.local_load(smem_b_right[1], relaxed=True)

    # Store left half
    c_left = acc_left.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    c_left_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn_left[None, :])
    tl.store(c_left_ptrs, c_left, mask=(offs_cm[:, None] < M) & (offs_cn_left[None, :] < N))

    acc_right = tl.dot(a, b_right, acc_right)

    # Store right half
    c_right = acc_right.to(tl.float16)
    offs_cn_right = pid_n * BLOCK_N + HALF_N + tl.arange(0, HALF_N)
    c_right_ptrs = (c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn_right[None, :])
    tl.store(
        c_right_ptrs,
        c_right,
        mask=(offs_cm[:, None] < M) & (offs_cn_right[None, :] < N),
    )


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    NUM_XCDS = 8
    GROUP_SIZE_M = 4
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    v9_beyond_hotloop[(GRID_MN, )](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_XCDS=NUM_XCDS,
        GRID_MN=GRID_MN,
        num_warps=8,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return c
