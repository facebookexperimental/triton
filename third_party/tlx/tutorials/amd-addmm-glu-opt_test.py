"""
AMD TLX fused addmm + GLU (out = x + x*y, x = A@B + bias).

Benchmarks the baseline and optimized TLX kernels against a PyTorch reference
(naive matmul + add, then GLU) and ROCBLAS (pure matmul, no add/GLU).

Usage:
  # Sweep all K, both kernels (default)
  python amd-addmm-glu-opt_test.py

  # Single K, optimized kernel only
  python amd-addmm-glu-opt_test.py -K 1024 --kernel tlx_optimized

  # A couple of sizes, both kernels
  python amd-addmm-glu-opt_test.py -K 256 512 --kernel tlx_baseline tlx_optimized
"""
import argparse

import pytest
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

DEVICE = triton.runtime.driver.active.get_active_torch_device()

M, N = 1024, 21568

# gfx950 has 8 XCDs per chip
NUM_XCDS = 8

# Per-row padding (fp16 elements) on the async kernel's Y LDS buffer
Y_PAD = 4

# Autotuning winning configurations for K=256, K=512, and K=1024
BEST_CONFIG = {
    256:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=4,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    512:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, XCD_CHUNK=4, num_warps=4,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    1024:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
}

# Autotuning winning configurations for the baseline kernel (K=256, K=512, K=1024)
BASELINE_BEST_CONFIG = {
    256:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, XCD_CHUNK=4, NUM_STAGES=2, num_warps=4,
         matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
    512:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, XCD_CHUNK=4, NUM_STAGES=2, num_warps=4,
         matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
    1024:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, XCD_CHUNK=4, NUM_STAGES=2, num_warps=4,
         matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
}

# Autotuning winning configurations for the async-epilogue kernel (K=256, K=512, K=1024)
ASYNC_BEST_CONFIG = {
    256:
    dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, NUM_BUFFERS=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    512:
    dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, NUM_BUFFERS=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    1024:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, XCD_CHUNK=4, NUM_BUFFERS=3, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
}

# Autotuning winning configurations for the persistent warp-pipelined kernel
# BN=256 with BK=32 (72KB LDS) beats BN=128 with BK=64 (96KB) at all K values
# BN=256 with BK=64 (144KB) is rejected by hardware due to warp-pipeline overhead
PERSISTENT_BEST_CONFIG = {
    256:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    512:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    1024:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
}

# Configs for the simple (non-pipelined) kernel with async-prefetched Y
SIMPLE_BEST_CONFIG = {
    256:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    512:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
    1024:
    dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, XCD_CHUNK=4, num_warps=8,
         matrix_instr_nonkdim=16, waves_per_eu=0),
}


# L2 swizzling
@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """Group adjacent PIDs onto the same XCD in chunks of chunk_size for L2 reuse.

    PIDs in the trailing remainder (not a multiple of num_xcds*chunk_size) pass through.
    """
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.constexpr_function
def _y_padded_layout(BM, BN, PAD):
    # Pad each row of the (BM, BN) Y tile
    return tlx.padded_shared_layout_encoding.with_identity_for([[BN, PAD]], [BM, BN], [1, 0])


# Warp-pipelined async direct-to-LDS fused addmm + GLU optimized kernel
@triton.jit
def tlx_addmm_glu_kernel_optimized(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)

    # Num pids along M dim
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # Num pids along N dim
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # How many pids we have in total grid
    grid_mn = num_pid_m * num_pid_n

    # Remap pids via L2 swizzle
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)

    # Num pids in the group (of shared rows along M dim pids)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Which group we are in
    group_id = pid // num_pid_in_group

    # Index of first pid in this group
    first_pid_m = group_id * GROUP_SIZE_M

    # Either GROUP_SIZE_M or smaller due to last group cutting off early
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Which pid along m dim
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    # Which pid along n dim
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Get offsets this pid is responsible for in m dim
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    # Get offsets this pid is responsible for in n dim
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    # Get offsets this pid is responsible for in k dim
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers for A along m dim
    a_base_off = offs_m[:, None] * sa0

    # Base pointers for B along n dim
    b_base_off = offs_n[None, :] * sb1

    # The number of buffers for this kernel is fixed to be 3 which allows
    # for the async_load_wait_group in the hotloop to not wait on the directly
    # previous global read (GR) to finish because the GR that it actually
    # needs was fetched much earlier (possibly in the prologue)
    NUM_BUFFERS: tl.constexpr = 3

    # How many times we need to iterate through blocks of size BLOCK_SIZE_K
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    # Create multibuffered shared memory arrays
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # GR (Global Read) 0
    k_start = 0
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off

    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 0), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 0), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # GR 1
    k_start = BLOCK_SIZE_K
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off

    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 1), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 1), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # GR 2
    k_start = BLOCK_SIZE_K * 2
    a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
    b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off

    tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 2), mask=offs_k[None, :] < K - k_start)
    tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 2), mask=offs_k[:, None] < K - k_start)
    tlx.async_load_commit_group([tok_a, tok_b])

    # Make GR0 and GR1 finish
    tlx.async_load_wait_group(1)

    # LR (Local Read) 0 where we transfer GR0 to registers
    a_tile = tlx.local_load(tlx.local_view(smemA, 0), relaxed=True)
    b_tile = tlx.local_load(tlx.local_view(smemB, 0), relaxed=True)

    # Create accumulator array
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Hot loop: warp-pipelined MFMA / async-prefetch
    for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
        # Index within the multibuffered circular SMEM where we are storing the global prefetch
        prefetch_buf = i % NUM_BUFFERS

        # Index within the multibuffered circular SMEM where we will load from to transfer into regs
        next_buf = (i + 1) % NUM_BUFFERS

        # Which tile we need to prefetch globally along the k dim
        k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

        # Execute MFMA with the data we already have loaded into registers
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        with tlx.warp_pipeline_stage("mem", priority=1):
            # Perform global prefetching (GR i + NUM_BUFFERS)
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
            b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off

            tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf), mask=offs_k[None, :]
                                   < K - k_prefetch)
            tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf), mask=offs_k[:, None]
                                   < K - k_prefetch)

            tlx.async_load_commit_group([tok_a, tok_b])

            # Perform local prefetching (LR i + 1)
            a_tile = tlx.local_load(tlx.local_view(smemA, next_buf), relaxed=True)
            b_tile = tlx.local_load(tlx.local_view(smemB, next_buf), relaxed=True)

        # Most recently committed buffers (GR i + NUM_BUFFERS) can be in flight
        tlx.async_load_wait_group(1)

    # Can potentially async load the y and bias here

    # Epilogue: drain the remaining in-flight tiles
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Wait for all buffers to be loaded before draining in epilogue loop
    tlx.async_load_wait_group(0)

    # Finish final set of LRs and MFMAs
    for i in tl.static_range(0, NUM_BUFFERS - 1):
        buf = (k_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS

        a_tile = tlx.local_load(tlx.local_view(smemA, buf), relaxed=True)
        b_tile = tlx.local_load(tlx.local_view(smemB, buf), relaxed=True)

        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Fused epilogue: addmm + GLU
    # Add bias (broadcast over M), then GLU: X = X + X*Y
    # Streaming cache hints (.cs) on Y and C: both are touched once and never
    # reused, so we avoid polluting L2 with this 44MB+44MB epilogue traffic

    bias = tl.load(bias_ptr + offs_n).to(tl.float32)

    y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1

    y = tl.load(y_ptrs, cache_modifier=".cs").to(tl.float32)

    x = acc + bias[None, :]

    out = x + x * y

    c_ptrs = c_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1

    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), cache_modifier=".cs")


def run_kernel_optimized(a, b, bias, y, out, cfg):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), )
    return tlx_addmm_glu_kernel_optimized[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


# Warp-pipelined async direct-to-LDS fused addmm + GLU kernel, async-epilogue variant
# Stages the epilogue Y tile into a padded LDS buffer via async_load so its 44MB HBM
# read overlaps GEMM compute
@triton.jit
def tlx_addmm_glu_kernel_optimized_async(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    Y_PAD: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)

    # Num pids along M dim
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)

    # Num pids along N dim
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # How many pids we have in total grid
    grid_mn = num_pid_m * num_pid_n

    # Remap pids via L2 swizzle
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)

    # Num pids in the group (of shared rows along M dim pids)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Which group we are in
    group_id = pid // num_pid_in_group

    # Index of first pid in this group
    first_pid_m = group_id * GROUP_SIZE_M

    # Either GROUP_SIZE_M or smaller due to last group cutting off early
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # Which pid along m dim
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)

    # Which pid along n dim
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Get offsets this pid is responsible for in m dim
    offs_m = offs_cm % M

    # Get offsets this pid is responsible for in n dim
    offs_n = offs_cn % N

    # Get offsets this pid is responsible for in k dim
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Base pointers for A along m dim
    a_base_off = offs_m[:, None] * sa0

    # Base pointers for B along n dim
    b_base_off = offs_n[None, :] * sb1

    # How many times we need to iterate through blocks of size BLOCK_SIZE_K
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    # Create multibuffered shared memory arrays
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: issue NUM_BUFFERS global reads
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        k_start = i * BLOCK_SIZE_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off

        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, i), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, i), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

    # Drain down to NUM_BUFFERS-2 in flight so buffers 0 and 1 are ready
    tlx.async_load_wait_group(NUM_BUFFERS - 2)

    # LR (Local Read) 0 where we transfer GR0 to registers
    a_tile = tlx.local_load(tlx.local_view(smemA, 0), relaxed=True)
    b_tile = tlx.local_load(tlx.local_view(smemB, 0), relaxed=True)

    # Create accumulator array
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Hot loop: warp-pipelined MFMA / async-prefetch
    for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
        # Index within the multibuffered circular SMEM where we are storing the global prefetch
        prefetch_buf = i % NUM_BUFFERS

        # Index within the multibuffered circular SMEM where we will load from to transfer into regs
        next_buf = (i + 1) % NUM_BUFFERS

        # Which tile we need to prefetch globally along the k dim
        k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

        # Execute MFMA with the data we already have loaded into registers
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        with tlx.warp_pipeline_stage("mem", priority=1):
            # Perform global prefetching (GR i + NUM_BUFFERS)
            a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
            b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off

            tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf), mask=offs_k[None, :]
                                   < K - k_prefetch)
            tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf), mask=offs_k[:, None]
                                   < K - k_prefetch)

            tlx.async_load_commit_group([tok_a, tok_b])

            # Perform local prefetching (LR i + 1)
            a_tile = tlx.local_load(tlx.local_view(smemA, next_buf), relaxed=True)
            b_tile = tlx.local_load(tlx.local_view(smemB, next_buf), relaxed=True)

        # Keep the NUM_BUFFERS-2 most-recent prefetches in flight (pipeline depth)
        tlx.async_load_wait_group(NUM_BUFFERS - 2)

    # Async-prefetch Y into a padded LDS buffer
    y_layout: tl.constexpr = _y_padded_layout(BLOCK_SIZE_M, BLOCK_SIZE_N, Y_PAD)
    smemY = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tlx.dtype_of(y_ptr), 1, layout=y_layout)
    y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1
    y_tok = tlx.async_load(y_ptrs, tlx.local_view(smemY, 0))
    tlx.async_load_commit_group([y_tok])

    # Epilogue: drain the remaining in-flight tiles
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Keep Y (newest committed group) in flight; drain all A/B.
    tlx.async_load_wait_group(1)

    # Finish final set of LRs and MFMAs
    for i in tl.static_range(0, NUM_BUFFERS - 1):
        buf = (k_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS

        a_tile = tlx.local_load(tlx.local_view(smemA, buf), relaxed=True)
        b_tile = tlx.local_load(tlx.local_view(smemB, buf), relaxed=True)

        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    # Fused epilogue: addmm + GLU. Add bias (broadcast over M), then GLU: X = X + X*Y.
    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    x = acc + bias[None, :]

    # Wait for Y to finish loading, then read it back from the padded LDS buffer
    tlx.async_load_wait_group(0)
    y_regs = tlx.local_load(tlx.local_view(smemY, 0), relaxed=True).to(tl.float32)
    out = x + x * y_regs

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1

    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


def run_kernel_optimized_async(a, b, bias, y, out, cfg):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), )
    return tlx_addmm_glu_kernel_optimized_async[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        Y_PAD=Y_PAD,
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


# Simple (non-pipelined) GEMM with async-prefetched Y. The K-loop does a plain
# blocking load of A and B each iteration (no multibuffering, no warp-pipeline,
# no prefetch). Only the epilogue Y tile is staged async into LDS before the loop
# so its HBM read latency can overlap the whole GEMM
@triton.jit
def tlx_addmm_glu_kernel_simple_async(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    Y_PAD: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid_mn = num_pid_m * num_pid_n
    pid = chiplet_transform_chunked(pid, grid_mn, NUM_XCDS, XCD_CHUNK)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = offs_cm % M
    offs_n = offs_cn % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Async-prefetch Y into a padded LDS buffer before the K-loop
    y_layout: tl.constexpr = _y_padded_layout(BLOCK_SIZE_M, BLOCK_SIZE_N, Y_PAD)
    smemY = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_N), tlx.dtype_of(y_ptr), 1, layout=y_layout)
    y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1
    y_tok = tlx.async_load(y_ptrs, tlx.local_view(smemY, 0))
    tlx.async_load_commit_group([y_tok])

    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1

    # Plain unpipelined K-loop: grab A and B for this iteration, dot, advance
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)
    for k in tl.range(0, k_iters):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * sa1
        b_ptrs += BLOCK_SIZE_K * sb0

    # Fused epilogue: addmm + GLU
    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    x = acc + bias[None, :]

    # Wait for Y to finish loading, then read it back from the padded LDS buffer
    tlx.async_load_wait_group(0)
    y_regs = tlx.local_load(tlx.local_view(smemY, 0), relaxed=True).to(tl.float32)
    out = x + x * y_regs

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


# Persistent warp-pipelined fused addmm + GLU kernel.
# Each CU is launched once and loops over its share of (num_pid_m * num_pid_n) tiles
@triton.jit
def tlx_addmm_glu_kernel_persistent(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    NUM_CUS: tl.constexpr,
):
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    NUM_BUFFERS: tl.constexpr = 3

    start_pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # LDS buffers are allocated once and reused across tiles (fully drained between tiles)
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smemB = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Distribute tiles among CUs
    for tile_id in tl.range(start_pid, num_tiles, NUM_CUS):
        # Apply XCD swizzle to the flat tile_id for L2 locality
        swizzled = chiplet_transform_chunked(tile_id, num_tiles, NUM_XCDS, XCD_CHUNK)

        # Delinearize swizzled tile_id
        group_id = swizzled // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((swizzled % num_pid_in_group) % group_size_m)
        pid_n = (swizzled % num_pid_in_group) // group_size_m

        offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        a_base_off = offs_m[:, None] * sa0
        b_base_off = offs_n[None, :] * sb1

        k_iters = tl.cdiv(K, BLOCK_SIZE_K)

        # Prologue: issue 3 global reads
        k_start = 0
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 0), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 0), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        k_start = BLOCK_SIZE_K
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 1), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 1), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        k_start = BLOCK_SIZE_K * 2
        a_offs = a_base_off + (k_start + offs_k[None, :]) * sa1
        b_offs = (k_start + offs_k[:, None]) * sb0 + b_base_off
        tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, 2), mask=offs_k[None, :] < K - k_start)
        tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, 2), mask=offs_k[:, None] < K - k_start)
        tlx.async_load_commit_group([tok_a, tok_b])

        # Make GR0 and GR1 finish
        tlx.async_load_wait_group(1)

        a_tile = tlx.local_load(tlx.local_view(smemA, 0), relaxed=True)
        b_tile = tlx.local_load(tlx.local_view(smemB, 0), relaxed=True)

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        # Hot loop: warp-pipelined MFMA / async-prefetch
        for i in tl.range(0, k_iters - NUM_BUFFERS, loop_unroll_factor=0):
            prefetch_buf = i % NUM_BUFFERS
            next_buf = (i + 1) % NUM_BUFFERS
            k_prefetch = (i + NUM_BUFFERS) * BLOCK_SIZE_K

            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                a_offs = a_base_off + (k_prefetch + offs_k[None, :]) * sa1
                b_offs = (k_prefetch + offs_k[:, None]) * sb0 + b_base_off

                tok_a = tlx.async_load(a_ptr + a_offs, tlx.local_view(smemA, prefetch_buf), mask=offs_k[None, :]
                                       < K - k_prefetch)
                tok_b = tlx.async_load(b_ptr + b_offs, tlx.local_view(smemB, prefetch_buf), mask=offs_k[:, None]
                                       < K - k_prefetch)

                tlx.async_load_commit_group([tok_a, tok_b])

                a_tile = tlx.local_load(tlx.local_view(smemA, next_buf), relaxed=True)
                b_tile = tlx.local_load(tlx.local_view(smemB, next_buf), relaxed=True)

            tlx.async_load_wait_group(1)

        # Epilogue: drain remaining in-flight tiles
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        tlx.async_load_wait_group(0)

        for i in tl.static_range(0, NUM_BUFFERS - 1):
            buf = (k_iters - (NUM_BUFFERS - 1) + i) % NUM_BUFFERS
            a_tile = tlx.local_load(tlx.local_view(smemA, buf), relaxed=True)
            b_tile = tlx.local_load(tlx.local_view(smemB, buf), relaxed=True)
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

        # Fused addmm + GLU epilogue
        bias = tl.load(bias_ptr + offs_n).to(tl.float32)
        y_ptrs = y_ptr + offs_m[:, None] * sy0 + offs_n[None, :] * sy1
        y = tl.load(y_ptrs, cache_modifier=".cs").to(tl.float32)
        x = acc + bias[None, :]
        out = x + x * y

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        c_ptrs = c_ptr + offs_cm[:, None] * sc0 + offs_cn[None, :] * sc1
        tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask, cache_modifier=".cs")


def run_kernel_persistent(a, b, bias, y, out, cfg, num_cus):
    M, K = a.shape
    _, N = b.shape
    grid = (min(num_cus, triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"])), )
    return tlx_addmm_glu_kernel_persistent[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        NUM_CUS=num_cus,
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


def run_kernel_simple_async(a, b, bias, y, out, cfg):
    M, K = a.shape
    _, N = b.shape
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), )
    return tlx_addmm_glu_kernel_simple_async[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        Y_PAD=Y_PAD,
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )


def pytorch_baseline(bias, a, b, y):
    # Reference: matmul (a@b) and bias add done separately, then GLU (out = x + x*y)
    x = torch.matmul(a, b).to(torch.float32)
    x = x + bias.to(torch.float32)[None, :]
    return (x + x * y.to(torch.float32)).to(torch.float16)


# Baseline Addmm kernel
@triton.jit
def tlx_addmm_glu_kernel_baseline(
    a_ptr,
    b_ptr,
    bias_ptr,
    y_ptr,
    c_ptr,
    M,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sy0,
    sy1,
    sc0,
    sc1,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(sa0 > 0)
    tl.assume(sa1 > 0)
    tl.assume(sb0 > 0)
    tl.assume(sb1 > 0)
    tl.assume(sy0 > 0)
    tl.assume(sy1 > 0)
    tl.assume(sc0 > 0)
    tl.assume(sc1 > 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * sa0 + offs_k[None, :] * sa1
    b_ptrs = b_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
    k_iters = tl.cdiv(K, BLOCK_SIZE_K)

    buffers_a = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tlx.dtype_of(a_ptr), NUM_STAGES - 1)
    buffers_b = tlx.local_alloc((BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(b_ptr), NUM_STAGES - 1)

    # TLX prologue: prefetch the first stages from HBM into LDS.
    for stage in tl.range(0, NUM_STAGES - 1, loop_unroll_factor=NUM_STAGES - 1):
        a_smem = tlx.local_view(buffers_a, stage)
        b_smem = tlx.local_view(buffers_b, stage)
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - stage * BLOCK_SIZE_K, other=0.0)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - stage * BLOCK_SIZE_K, other=0.0)
        tlx.local_store(a_smem, a_reg)
        tlx.local_store(b_smem, b_reg)
        a_ptrs += BLOCK_SIZE_K * sa1
        b_ptrs += BLOCK_SIZE_K * sb0

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(NUM_STAGES - 1, k_iters, num_stages=0):
        a_smem = tlx.local_view(buffers_a, k % (NUM_STAGES - 1))
        b_smem = tlx.local_view(buffers_b, k % (NUM_STAGES - 1))
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        prev = (k % (NUM_STAGES - 1))
        a_prev = tlx.local_load(tlx.local_view(buffers_a, prev), relaxed=True)
        b_prev = tlx.local_load(tlx.local_view(buffers_b, prev), relaxed=True)
        acc = tl.dot(a_prev, b_prev, acc)

        tlx.local_store(a_smem, a_reg)
        tlx.local_store(b_smem, b_reg)
        a_ptrs += BLOCK_SIZE_K * sa1
        b_ptrs += BLOCK_SIZE_K * sb0

    for k in tl.range(k_iters - (NUM_STAGES - 1), k_iters, loop_unroll_factor=NUM_STAGES - 1):
        buf = k % (NUM_STAGES - 1)
        a_prev = tlx.local_load(tlx.local_view(buffers_a, buf), relaxed=True)
        b_prev = tlx.local_load(tlx.local_view(buffers_b, buf), relaxed=True)
        acc = tl.dot(a_prev, b_prev, acc)

    # === Fused epilogue: addmm + GLU ===
    ocm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ocn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cmask = (ocm[:, None] < M) & (ocn[None, :] < N)

    bias = tl.load(bias_ptr + ocn, mask=ocn < N, other=0.0).to(tl.float32)
    x = acc + bias[None, :]

    y_ptrs = y_ptr + ocm[:, None] * sy0 + ocn[None, :] * sy1
    y = tl.load(y_ptrs, mask=cmask, other=0.0).to(tl.float32)
    out = x + x * y

    c_ptrs = c_ptr + ocm[:, None] * sc0 + ocn[None, :] * sc1
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=cmask)


def run_kernel_baseline(bias, a, b, y, cfg):
    M, K = a.shape
    K2, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]), )
    tlx_addmm_glu_kernel_baseline[grid](
        a,
        b,
        bias,
        y,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_STAGES=cfg["NUM_STAGES"],
        num_warps=cfg["num_warps"],
        matrix_instr_nonkdim=cfg.get("matrix_instr_nonkdim", 0),
        waves_per_eu=cfg.get("waves_per_eu", 0),
        kpack=cfg.get("kpack", 1),
    )
    return out


def run_baseline(a, b, bias, y):
    K = b.shape[0]
    return run_kernel_baseline(bias, a, b, y, BASELINE_BEST_CONFIG[K])


def run_optimized(a, b, bias, y):
    K = b.shape[0]
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    run_kernel_optimized(a, b, bias, y, out, BEST_CONFIG[K])
    return out


def run_optimized_async(a, b, bias, y):
    K = b.shape[0]
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    run_kernel_optimized_async(a, b, bias, y, out, ASYNC_BEST_CONFIG[K])
    return out


def run_persistent(a, b, bias, y):
    K = b.shape[0]
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    num_cus = torch.cuda.get_device_properties(a.device).multi_processor_count
    run_kernel_persistent(a, b, bias, y, out, PERSISTENT_BEST_CONFIG[K], num_cus)
    return out


def run_simple_async(a, b, bias, y):
    K = b.shape[0]
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    run_kernel_simple_async(a, b, bias, y, out, SIMPLE_BEST_CONFIG[K])
    return out


KERNEL_REGISTRY = {
    "tlx_baseline": run_baseline,
    "tlx_simple_async": run_simple_async,
    "tlx_optimized_async": run_optimized_async,
    "tlx_optimized": run_optimized,
    "tlx_persistent": run_persistent,
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name!r}. "
                         f"Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name]


def verify(name, got, ref, atol=2e-2, rtol=2e-2, log=True):
    diff = (got.float() - ref.float()).abs()
    ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    status = "PASS" if ok else "FAIL"
    if log:
        print(f"  {name:<28} {status}  max={max_err:.6f}  mean={mean_err:.6f}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Performance regression guard
# ═══════════════════════════════════════════════════════════════════════════

# Baseline TFLOPS measured on gfx950/MI350 (fp16, M=1024, N=21568). A kernel must
# stay within PERF_TOL below its baseline or the perf test fails (catches
# regressions). Regenerate the numbers with:
#   python amd-addmm-glu-opt_test.py
# then copy the printed Summary table values here.
PERF_TOL = 0.15  # allow 15% slack below baseline (run-to-run noise + minor drift)

# key: (kernel, K) -> TFLOPS
PERF_BASELINE_TFLOPS = {
    ("tlx_baseline", 256): 237,
    ("tlx_baseline", 512): 355,
    ("tlx_baseline", 1024): 459,
    ("tlx_simple_async", 256): 169,
    ("tlx_simple_async", 512): 269,
    ("tlx_simple_async", 1024): 349,
    ("tlx_optimized_async", 256): 181,
    ("tlx_optimized_async", 512): 279,
    ("tlx_optimized_async", 1024): 406,
    ("tlx_optimized", 256): 280,
    ("tlx_optimized", 512): 442,
    ("tlx_optimized", 1024): 597,
    ("tlx_persistent", 256): 298,
    ("tlx_persistent", 512): 449,
    ("tlx_persistent", 1024): 613,
}


def measure_tflops(kernel_fn, M, N, K):
    """Verify correctness then return achieved TFLOPS for one config."""
    torch.manual_seed(0)
    a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
    b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
    y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
    ref = pytorch_baseline(bias, a, b, y)
    fn = lambda: kernel_fn(a, b, bias, y)
    assert verify("", fn(), ref, log=False), "correctness check failed"
    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    total_flops = 2.0 * M * N * K
    return total_flops / ms * 1e-9


@pytest.mark.parametrize("K", sorted(BEST_CONFIG))
@pytest.mark.parametrize("kernel_name", list(KERNEL_REGISTRY))
def test_addmm_glu_performance(kernel_name, K):
    """Guard against perf regressions: achieved TFLOPS must stay within PERF_TOL
    of the recorded gfx950 baseline (M=1024, N=21568, fp16)."""
    baseline = PERF_BASELINE_TFLOPS[(kernel_name, K)]
    floor = baseline * (1 - PERF_TOL)
    got = measure_tflops(get_kernel(kernel_name), M, N, K)
    print(f"  {kernel_name:20s} K={K:5d} {got:7.1f} TFLOPS (baseline {baseline}, floor {floor:.1f})")
    assert got >= floor, (f"perf regression: {kernel_name} K={K} -> "
                          f"{got:.1f} TFLOPS < floor {floor:.1f} (baseline {baseline}, tol {PERF_TOL:.0%})")


def run_perf_test(args):
    """CLI entry: run the perf guard over all baseline configs, print a report,
    and return True iff every kernel is within PERF_TOL of its baseline."""
    tol = args.perf_tol
    all_ok = True
    print(f"\nPerformance regression test (floor = baseline * (1 - {tol:.0%}))")
    print("-" * 78)
    for (kernel_name, K), baseline in sorted(PERF_BASELINE_TFLOPS.items()):
        floor = baseline * (1 - tol)
        try:
            got = measure_tflops(get_kernel(kernel_name), M, N, K)
        except Exception as e:
            all_ok = False
            print(f"  [FAIL] {kernel_name:20s} K={K:5d} -> ERROR ({e})")
            continue
        ok = got >= floor
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {kernel_name:20s} K={K:5d} "
              f"{got:7.1f} TFLOPS (baseline {baseline}, floor {floor:.1f})")
    print("-" * 78)
    print("RESULT:", "PASS" if all_ok else "FAIL")
    assert all_ok


def print_summary_table(results, providers):
    """Print a markdown-style summary table of benchmark results."""
    rows = [(f"M={m} N={n} K={k}", results[(m, n, k)]) for (m, n, k) in sorted(results.keys())]

    cfg_w = max(len("Config"), *(len(lbl) for lbl, _ in rows)) if rows else len("Config")
    col_w = max(14, *(len(p) for p in providers))

    hdr = f"| {'Config':<{cfg_w}} |" + "".join(f" {p:>{col_w}} |" for p in providers)
    sep = f"|{'-' * (cfg_w + 2)}|" + "".join(f"{'-' * (col_w + 2)}|" for _ in providers)

    print(f"\n{'=' * len(sep)}")
    print("Summary (TFLOPS)   fp16")
    print(f"{'=' * len(sep)}")
    print(hdr)
    print(sep)

    for label, prov in rows:
        vals = (f"{prov[p]['tflops']:>{col_w}.1f}" if p in prov else f"{'—':>{col_w}}" for p in providers)
        print(f"| {label:<{cfg_w}} |" + "".join(f" {v} |" for v in vals))

    print(f"{'=' * len(sep)}\n")


def run_benchmark(args):
    results = {}

    for M in args.M:
        for N in args.N:
            for K in args.K:
                torch.manual_seed(0)
                a = torch.randn(M, K, device=DEVICE, dtype=torch.float16)
                b = torch.randn(K, N, device=DEVICE, dtype=torch.float16)
                bias = torch.randn(N, device=DEVICE, dtype=torch.float16)
                y = torch.randn(M, N, device=DEVICE, dtype=torch.float16)

                total_flops = 2.0 * M * N * K
                ref = pytorch_baseline(bias, a, b, y)

                key = (M, N, K)
                results[key] = {}

                # PyTorch reference: naive matmul + add, then GLU
                pt_lambda = lambda: pytorch_baseline(bias, a, b, y)
                ms = triton.testing.do_bench(pt_lambda, warmup=25, rep=100)
                results[key]["PyTorch_naive"] = {"ms": ms, "tflops": total_flops / ms * 1e-9}

                # Selected TLX kernels (verified against PyTorch reference)
                for kernel_name in args.kernel:
                    kernel_fn = get_kernel(kernel_name)
                    try:
                        k_lambda = lambda fn=kernel_fn: fn(a, b, bias, y)
                        out = k_lambda()
                        assert verify("", out, ref, log=False)
                    except Exception as e:
                        print(f"  {kernel_name:20s} M={M:5d} N={N:5d} K={K:5d} -> SKIPPED ({e})")
                        continue
                    ms = triton.testing.do_bench(k_lambda, warmup=25, rep=100)
                    results[key][kernel_name] = {"ms": ms, "tflops": total_flops / ms * 1e-9}

                # ROCBLAS reference: pure matmul (no add/GLU)
                rb_lambda = lambda: torch.matmul(a, b)
                ms = triton.testing.do_bench(rb_lambda, warmup=25, rep=100)
                results[key]["ROCBLAS (matmul only)"] = {"ms": ms, "tflops": total_flops / ms * 1e-9}

    providers = ["PyTorch_naive"] + list(args.kernel) + ["ROCBLAS (matmul only)"]
    print_summary_table(results, providers)


def parse_args():
    k_choices = sorted(BEST_CONFIG)
    kernel_choices = list(KERNEL_REGISTRY)
    p = argparse.ArgumentParser(prog="AMD TLX Addmm+GLU")
    p.add_argument("-M", type=int, nargs="+", default=[M], help="M (rows of A / output) sizes to benchmark")
    p.add_argument("-N", type=int, nargs="+", default=[N], help="N (cols of B / output) sizes to benchmark")
    p.add_argument("-K", type=int, nargs="+", default=k_choices, choices=k_choices,
                   help="K (contraction) sizes to benchmark; only stored-config sizes are supported")
    p.add_argument("--kernel", type=str, nargs="+", default=kernel_choices, choices=kernel_choices,
                   help="Kernel variants to benchmark")
    p.add_argument("--mode", choices=["benchmark", "perf_test"], default="benchmark",
                   help="benchmark: print TFLOPS table; perf_test: assert TFLOPS within PERF_TOL of baseline")
    p.add_argument("--perf-tol", dest="perf_tol", type=float, default=PERF_TOL,
                   help="perf_test slack below baseline (e.g. 0.15 = allow 15%% regression)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "perf_test":
        run_perf_test(args)
    else:
        run_benchmark(args)
