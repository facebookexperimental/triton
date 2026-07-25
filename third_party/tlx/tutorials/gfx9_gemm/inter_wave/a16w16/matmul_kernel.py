"""8-wave inter-wave warp-pipelined FP16/BF16 GEMM for gfx950 (CDNA4).

Runs a 256x256 output tile on 8 warps (2 waves/SIMD). Key ideas:

  * 2x2 quadrant tiling: the tile is split into four [128x128] quadrants, and
    each operand half-tile gets its OWN double-buffered LDS allocation
    (smem_a_top/bot, smem_b_left/right) so the four MFMAs stay independent.
  * Inter-wave software pipeline: the two co-resident wave groups run a full
    stage apart, so each `async_load_wait_group` is hoisted *before* its MFMA
    cluster -- closing the LDS producer->consumer hazard a stage early and
    keeping N load groups in flight to overlap loads with MFMAs.
  * Swizzled LDS layout pinned via `padded_shared_layout_encoding.with_bases`
    to make the direct-to-LDS loads bank-conflict-free on CDNA4.

Adapted from ROCm/gfx950-gluon-tutorials `kernels/gemm/inter_wave/a16w16`
(the 4-wave `a16w16/v9` hot loop, run here on 8 warps via `warp_pipeline_stage`).
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

MIN_K = 2 * BLOCK_K  # pipeline prefetches 2 whole K-tiles; the rest goes to the masked tail
KERNEL_NAME = "a16w16_8wave"


@triton.jit
def a16w16_8wave(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
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

    # The pipeline consumes K in pairs of BLOCK_K tiles (prologue prefetches 2,
    # the loop 2/iter, the epilogue drains 2), so it covers only an EVEN number of
    # whole K-tiles: n_pipe. Any leftover -- an odd whole tile and/or a partial
    # final tile (K not a multiple of BLOCK_K) -- is handled by the masked scalar
    # tail after the epilogue. K is a runtime arg (not constexpr) so distinct K
    # values reuse one compile; n_pipe even keeps the epilogue's buffer parity
    # fixed (reads buffer 0 first (l_idx) then buffer 1 (g_idx), constexpr below).
    n_full = K // BLOCK_K
    n_pipe = (n_full // 2) * 2

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
    for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
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

    # ── Epilogue: last 2 pipelined K-steps, drain LDS loads ──
    # iter n_pipe-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (n_pipe - 2) % 2, always 0 since n_pipe is even
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1  # 1 - l_idx
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

    # iter n_pipe-1: finish ALL four mfmas before the tail/store so the dot
    # operands die and the store phase holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)

    # ── Masked scalar tail: K columns past the pipelined region (an odd leftover
    # tile and/or a partial final tile). Plain masked tl.load + tl.dot -- no LDS,
    # no pipeline. The K-mask zeros the missing contraction elements (they add 0
    # to C = sum_k A*B), so this is correct for arbitrary K. Runs 0-2 iterations;
    # the whole-tile even hot path (n_pipe*BLOCK_K == K) skips it entirely.
    offs_am_bot = offs_am + HALF_M
    offs_bn_right = offs_bn + HALF_N
    for kk in tl.range(n_pipe * BLOCK_K, K, BLOCK_K, num_stages=1):
        offs_kt = kk + offs_k
        k_mask = offs_kt < K
        a_top_t = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                          mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
        a_bot_t = tl.load(a_ptr + offs_am_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                          mask=(offs_am_bot[:, None] < M) & k_mask[None, :], other=0.0)
        b_left_t = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                           mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
        b_right_t = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_bn_right[None, :] * stride_bn,
                            mask=k_mask[:, None] & (offs_bn_right[None, :] < N), other=0.0)
        acc_tl = tl.dot(a_top_t, b_left_t, acc_tl)
        acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl)
        acc_tr = tl.dot(a_top_t, b_right_t, acc_tr)
        acc_br = tl.dot(a_bot_t, b_right_t, acc_br)

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
    # The pipeline needs at least 2 whole K-tiles (it prefetches 2 up front); any
    # K beyond that -- odd tile count and/or a partial final tile -- is handled by
    # the kernel's masked scalar tail. K must still be a multiple of 16: the
    # pinned swizzled LDS layout coalesces the direct-to-LDS loads over 16-element
    # groups, so the K-major strides (stride_am, stride_bn = K) must be 16-aligned.
    assert K >= 2 * BLOCK_K, f"K={K} must be at least {2 * BLOCK_K}"
    assert K % 16 == 0, f"K={K} must be a multiple of 16 (swizzled-layout coalescing)"
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
