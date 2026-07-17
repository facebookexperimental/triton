"""8-wave warp-pipeline FP16/BF16 GEMM — TLX port of the Gluon `inter_wave/a16w16`.

This is a TLX (instead of Gluon) recreation of ROCm/gfx950-gluon-tutorials
`kernels/gemm/inter_wave/a16w16`. It takes the 4-wave `a16w16/v9` hot loop and
runs it with 8 warps (2 waves/SIMD) using wave-level `warp_pipeline_stage`
scheduling. The 256x256 output tile is sliced into a 2x2 grid of [128x128]
quadrants, each operand half-tile living in its OWN double-buffered LDS
allocation (smemA_top/bot, smemB_left/right). K is unrolled 2x → 8 regions.

The counter-intuitive bit (see the reference README): because the two co-resident
wave groups run a full stage apart, each `async_load_wait_group` is hoisted to
*before* its mfma cluster, so the LDS producer→consumer hazard (which spans two
stages across the two groups) is closed a stage early.

STATUS / PERF
-------------
Correctness matches torch on all shapes. On gfx950 (MI350), fp16, do_bench,
4096x4096xK, this kernel matches / slightly beats rocBLAS:

    M     N     K       rocBLAS   this kernel
    4096  4096  4096    ~1113T    ~1154T
    4096  4096  8192    ~1210T    ~1185T
    4096  4096  16384   ~1211T    ~1204T

The hand-scheduled hot loop emits tokenless `async_wait {num=N}` to keep N load
groups in flight across the two co-resident wave groups, so loads overlap the
MFMAs and the one-stage-apart overlap the design relies on materializes.
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

# Keep the LLVM post-RA machine scheduler from re-ordering the
# warp_pipeline_stage mem/MFMA interleave (equivalent to TRITON_DISABLE_POST_MISCHED=1).
os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

MIN_K = 4 * BLOCK_K
KERNEL_NAME = "a16w16_8wave"


@triton.jit
def a16w16_8wave(
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

    # ── Grid-level scheduling: XCD PID remap + GROUP_SIZE_M swizzle (v9-style) ──
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

    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Four separate double-buffered LDS allocations — one per operand half-tile.
    # Pin the *swizzled* padded_shared layout (row/col-permuted offset bases,
    # identical to the Gluon reference) so the ds_reads feeding the MFMAs are
    # bank-conflict-free. The default inferred padded layout ({order, shape})
    # conflicts on CDNA4 (measured 50M SQ_LDS_BANK_CONFLICT vs 0 for this one).
    a_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]],
        [HALF_M, BLOCK_K],
    )
    b_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
        [(512, 16)],
        [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]],
        [BLOCK_K, HALF_N],
    )
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), tl.float16, 2, layout=a_shared)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), tl.float16, 2, layout=a_shared)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tl.float16, 2, layout=b_shared)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tl.float16, 2, layout=b_shared)

    # The direct-to-LDS buffer_load write is coalesced only when each offset
    # tensor's #linear layout matches the swizzled LDS layout above. We pin only
    # the shared layouts; the matching offset layouts are inferred from them by
    # tlx-insert-require-layout (no explicit offset_layout= needed).
    offs_am = pid_m * BLOCK_M + tl.arange(0, HALF_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_top_off = offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = a_top_off + HALF_M * stride_am
    b_left_off = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_off = b_left_off + HALF_N * stride_bn

    # _next = the K+1 buffer (offset one BLOCK_K along K).
    a_top_off_n = a_top_off + BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + BLOCK_K * stride_ak
    b_left_off_n = b_left_off + BLOCK_K * stride_bk
    b_right_off_n = b_right_off + BLOCK_K * stride_bk

    ka = tl.zeros([], dtype=tl.int32)
    kb = tl.zeros([], dtype=tl.int32)

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    iterMax: tl.constexpr = K // BLOCK_K

    # ── Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits) ──
    tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
    tlx.async_load_commit_group()

    tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
    tlx.async_load_commit_group()
    tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
    tlx.async_load_commit_group()

    ka += BLOCK_K * stride_ak * 2
    kb += BLOCK_K * stride_bk * 2

    tlx.async_load_wait_group(6)
    b_left = tlx.local_load(smem_b_left[0], relaxed=True)
    a_top = tlx.local_load(smem_a_top[0], relaxed=True)

    # ── Main loop (2x unrolled): 8 (mfma + local_load + async refill) regions ──
    for k in tl.range(0, iterMax - 2, 2, num_stages=1):
        # --- sub-iter 0 (buffer 0) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + kb)
            tlx.async_load_commit_group()

        # --- sub-iter 1 (buffer 1, _next offsets) ---
        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tl = tl.dot(a_top, b_left, acc_tl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
            tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off_n + kb)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_right = tlx.local_load(smem_b_right[1], relaxed=True)
            tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_tr = tl.dot(a_top, b_right, acc_tr)
        with tlx.warp_pipeline_stage("mem", priority=1):
            b_left = tlx.local_load(smem_b_left[0], relaxed=True)
            tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off_n + ka)
            tlx.async_load_commit_group()

        tlx.async_load_wait_group(5)
        with tlx.warp_pipeline_stage("mfma", priority=0):
            acc_br = tl.dot(a_bot, b_right, acc_br)
        with tlx.warp_pipeline_stage("mem", priority=1):
            a_top = tlx.local_load(smem_a_top[0], relaxed=True)
            tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off_n + kb)
            tlx.async_load_commit_group()
            ka += BLOCK_K * stride_ak * 2
            kb += BLOCK_K * stride_bk * 2

    # ── Epilogue: last 2 K-steps, drain, 4-quadrant store ──
    # iter iterMax-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx = 1 - l_idx
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    # iter iterMax-1: finish ALL four mfmas before converting/storing so the dot
    # operands die and the store phase holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)

    offs_cm_top = pid_m * BLOCK_M + tl.arange(0, HALF_M)
    offs_cm_bot = offs_cm_top + HALF_M
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_left[None, :], c_tl,
             mask=(offs_cm_top[:, None] < M) & (offs_cn_left[None, :] < N))
    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_left[None, :], c_bl,
             mask=(offs_cm_bot[:, None] < M) & (offs_cn_left[None, :] < N))
    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_right[None, :], c_tr,
             mask=(offs_cm_top[:, None] < M) & (offs_cn_right[None, :] < N))
    c_br = acc_br.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_right[None, :], c_br,
             mask=(offs_cm_bot[:, None] < M) & (offs_cn_right[None, :] < N))


def matmul(a, b):
    """C = A @ B. `a` is (M, K), `b` is (K, N)."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    a16w16_8wave[(GRID_MN, )](
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
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma). Essential to match the reference perf.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    return c
