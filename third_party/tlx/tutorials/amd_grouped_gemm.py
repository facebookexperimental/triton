"""AMD TLX grouped GEMM for gfx950 -- 2x2 quadrant inter-wave pipeline.

Storage (both operands row-major with K, the contraction dim, innermost):
  A[i]: [M, K] row-major.
  B[i]: [N, K] row-major, handed to the kernel as a [K, N] .t() view
        (logically column-major [K, N]) -- the natural weight layout.

The B operand keeps the ``[outer=N, K]`` orientation such that B tiles
load coalesced along the contiguous K axis and the transpose the MFMA wants is
folded into the LDS read (``tlx.local_load(tlx.local_trans(smemB))``). Keeping K
innermost is what lets a plain masked K-tail loop handle a partial last K-tile,
so this kernel supports fully ragged M, N and K.

The 256x256 output tile is split into four 128x128 quadrants (2x2 subtiling
along M and N):
  * Each operand half-tile (a_top/a_bot, b_left/b_right) gets its own
    double-buffered LDS allocation so the four MFMAs stay independent.
  * Inter-wave software pipeline (8 warps): the hot loop is 2x-unrolled into 8
    (mfma + local_load + async refill) regions, each ``async_load_wait_group``
    hoisted a stage ahead via ``warp_pipeline_stage`` to keep loads overlapping
    the MFMAs.

Ragged handling (mask-free hot loop):
  * M and N are wrapped with a modulo, so every hot-loop read is in bounds.
    Garbage rows/cols are dropped by the masked C store.
  * The pipeline covers an even number of whole K-tiles (n_pipe). Any leftover
    (an odd whole tile and/or a partial final tile) is a cold masked tl.load
    tail. Small K (< 2 whole tiles) skips the pipeline entirely and runs only
    the tail.
"""
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

os.environ.setdefault("TRITON_DISABLE_POST_MISCHED", "1")

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# gfx950 has 8 XCDs
NUM_XCDS = 8


def num_sms():
    return torch.cuda.get_device_properties(DEVICE).multi_processor_count


@triton.jit
def chiplet_transform_chunked(pid, num_workgroups, num_xcds: tl.constexpr, chunk_size: tl.constexpr):
    """Permute program ids so adjacent-in-chunk pids land on the same XCD (L2 reuse)."""
    aligned = (num_workgroups // (num_xcds * chunk_size)) * (num_xcds * chunk_size)
    if pid >= aligned:
        return pid
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    return ((local_pid // chunk_size) * num_xcds * chunk_size + xcd * chunk_size + (local_pid % chunk_size))


@triton.jit
def _grouped_gemm_tile(
    # tile identity within this GEMM's [num_m_tiles, num_n_tiles] grid
    pid_m,
    pid_n,
    # this GEMM's base pointers
    a_ptr,
    b_ptr,
    c_ptr,
    # this GEMM's sizes <M, N, K>
    gm,
    gn,
    gk,
    # this GEMM's strides <A row-stride, B N-stride, C row-stride>
    stride_am,
    stride_bn,
    stride_cm,
    # LDS double buffers (one per operand half-tile), allocated once by the scheduler
    smem_a_top,
    smem_a_bot,
    smem_b_left,
    smem_b_right,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    HAS_K_TAIL: tl.constexpr,
):
    """Compute one [BLOCK_SIZE_M, BLOCK_SIZE_N] output tile of ``A @ B`` as four
    128x128 quadrants and store it to C."""
    tl.static_assert(NUM_BUFFERS == 2, "the 2x2 quadrant inter-wave pipeline is double-buffered")

    HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
    HALF_N: tl.constexpr = BLOCK_SIZE_N // 2

    # A rows and B columns are wrapped so every hot-loop read stays in bounds and
    # keeps vectorized loads along K. Garbage from wrapped lanes is dropped by the
    # masked C store.
    offs_am_top = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, HALF_M)) % gm, HALF_M)
    offs_am_bot = tl.multiple_of((pid_m * BLOCK_SIZE_M + HALF_M + tl.arange(0, HALF_M)) % gm, HALF_M)
    offs_bn_left = tl.multiple_of((pid_n * BLOCK_SIZE_N + tl.arange(0, HALF_N)) % gn, HALF_N)
    offs_bn_right = tl.multiple_of((pid_n * BLOCK_SIZE_N + HALF_N + tl.arange(0, HALF_N)) % gn, HALF_N)

    # K is the contiguous/innermost axis of both A and (col-major) B tiles.
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

    # Full 2-D tile offsets. With the pinned swizzle, tlx infers a compact
    # offset layout, so keeping these offsets live is cheaper than rematerializing
    # (which adds VALU/reload pressure). K stride is 1 for both A and col-major B,
    # so the running K position is a scalar (ka) stepping by BLOCK_K.
    a_top_off = offs_am_top[:, None] * stride_am + offs_k[None, :]
    a_bot_off = offs_am_bot[:, None] * stride_am + offs_k[None, :]
    b_left_off = offs_bn_left[:, None] * stride_bn + offs_k[None, :]
    b_right_off = offs_bn_right[:, None] * stride_bn + offs_k[None, :]

    # Buffer 1 holds the K+1 tile. We derive it with a scalar (+ BLOCK_K) rather
    # than persisting four more 2-D offset tensors.
    kb1: tl.constexpr = BLOCK_SIZE_K

    ka = tl.zeros([], dtype=tl.int32)  # K position of buffer 0's in-flight tile

    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

    # The pipeline consumes K in pairs of BLOCK_K tiles, so it covers only an EVEN
    # number of whole K-tiles: n_pipe. Any leftover (odd whole tile and/or a
    # partial final tile) is handled by the masked scalar tail below.
    n_full = gk // BLOCK_SIZE_K
    n_pipe = (n_full // 2) * 2

    if n_full >= 2:
        # Prologue: prefetch K-steps 0,1 into buffers 0,1 (8 commits)
        tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + ka)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + ka)
        tlx.async_load_commit_group()

        tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off + (ka + kb1))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off + (ka + kb1))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off + (ka + kb1))
        tlx.async_load_commit_group()
        tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off + (ka + kb1))
        tlx.async_load_commit_group()

        ka += BLOCK_SIZE_K * 2

        tlx.async_load_wait_group(6)
        b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
        a_top = tlx.local_load(smem_a_top[0], relaxed=True)

        # Main loop (2x unrolled): 8 (mfma + local_load + async refill) regions
        for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
            # sub-iter 0 (buffer 0, K = ka)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(smem_a_bot[0], relaxed=True)
                tlx.buffer_load_to_local(smem_b_left[0], b_ptr, b_left_off + ka)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[0]), relaxed=True)
                tlx.buffer_load_to_local(smem_a_top[0], a_ptr, a_top_off + ka)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[1]), relaxed=True)
                tlx.buffer_load_to_local(smem_a_bot[0], a_ptr, a_bot_off + ka)
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[1], relaxed=True)
                tlx.buffer_load_to_local(smem_b_right[0], b_ptr, b_right_off + ka)
                tlx.async_load_commit_group()

            # sub-iter 1 (buffer 1, K = ka + BLOCK_K)
            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_bot = tlx.local_load(smem_a_bot[1], relaxed=True)
                tlx.buffer_load_to_local(smem_b_left[1], b_ptr, b_left_off + (ka + kb1))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_right = tlx.local_load(tlx.local_trans(smem_b_right[1]), relaxed=True)
                tlx.buffer_load_to_local(smem_a_top[1], a_ptr, a_top_off + (ka + kb1))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                b_left = tlx.local_load(tlx.local_trans(smem_b_left[0]), relaxed=True)
                tlx.buffer_load_to_local(smem_a_bot[1], a_ptr, a_bot_off + (ka + kb1))
                tlx.async_load_commit_group()

            tlx.async_load_wait_group(5)
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
            with tlx.warp_pipeline_stage("mem", priority=1):
                a_top = tlx.local_load(smem_a_top[0], relaxed=True)
                tlx.buffer_load_to_local(smem_b_right[1], b_ptr, b_right_off + (ka + kb1))
                tlx.async_load_commit_group()
                ka += BLOCK_SIZE_K * 2

        # Epilogue: last 2 pipelined K-steps, drain LDS loads
        # iter n_pipe-2 (buffer 0)
        acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        tlx.async_load_wait_group(5)
        l_idx: tl.constexpr = 0  # (n_pipe - 2) % 2, always 0 since n_pipe is even
        a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

        acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        tlx.async_load_wait_group(4)
        b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, l_idx)), relaxed=True)

        acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        tlx.async_load_wait_group(3)
        g_idx: tl.constexpr = 1  # 1 - l_idx
        b_left = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_left, g_idx)), relaxed=True)

        acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)
        tlx.async_load_wait_group(2)
        a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

        # iter n_pipe-1 (buffer 1): finish all four mfmas before the tail/store.
        acc_tl = tl.dot(a_top, b_left, acc_tl, allow_tf32=False)
        tlx.async_load_wait_group(1)
        a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

        acc_bl = tl.dot(a_bot, b_left, acc_bl, allow_tf32=False)
        tlx.async_load_wait_group(0)
        b_right = tlx.local_load(tlx.local_trans(tlx.local_view(smem_b_right, g_idx)), relaxed=True)

        acc_tr = tl.dot(a_top, b_right, acc_tr, allow_tf32=False)
        acc_br = tl.dot(a_bot, b_right, acc_br, allow_tf32=False)

    # Masked scalar tail: whole K-tiles past the pipelined region (an odd
    # leftover tile when n_full is odd) plus a partial final tile (gk % BLOCK_K).
    # Uses the same wrapped M/N offsets, only K is masked. Runs 0-2 iterations
    # (and covers the whole GEMM when small K skipped the pipeline).
    #
    # Compiled out entirely when the host can prove no group needs it, which
    # reduces register pressure.
    if HAS_K_TAIL:
        for kk in tl.range(n_pipe * BLOCK_SIZE_K, gk, BLOCK_SIZE_K, num_stages=1):
            k_mask = offs_k < gk - kk
            a_top_t = tl.load(a_ptr + a_top_off + kk, mask=k_mask[None, :], other=0.0)
            a_bot_t = tl.load(a_ptr + a_bot_off + kk, mask=k_mask[None, :], other=0.0)
            b_left_t = tl.load(b_ptr + b_left_off + kk, mask=k_mask[None, :], other=0.0)
            b_right_t = tl.load(b_ptr + b_right_off + kk, mask=k_mask[None, :], other=0.0)
            b_left_t = tl.trans(b_left_t)
            b_right_t = tl.trans(b_right_t)
            acc_tl = tl.dot(a_top_t, b_left_t, acc_tl, allow_tf32=False)
            acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl, allow_tf32=False)
            acc_tr = tl.dot(a_top_t, b_right_t, acc_tr, allow_tf32=False)
            acc_br = tl.dot(a_bot_t, b_right_t, acc_br, allow_tf32=False)

    # Store the four quadrants, mask out OOB rows and columns
    offs_cm_top = pid_m * BLOCK_SIZE_M + tl.arange(0, HALF_M)
    offs_cm_bot = offs_cm_top + HALF_M
    offs_cn_left = pid_n * BLOCK_SIZE_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N

    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm_top[:, None] * stride_cm + offs_cn_left[None, :], c_tl,
             mask=(offs_cm_top[:, None] < gm) & (offs_cn_left[None, :] < gn))
    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm_bot[:, None] * stride_cm + offs_cn_left[None, :], c_bl,
             mask=(offs_cm_bot[:, None] < gm) & (offs_cn_left[None, :] < gn))
    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm_top[:, None] * stride_cm + offs_cn_right[None, :], c_tr,
             mask=(offs_cm_top[:, None] < gm) & (offs_cn_right[None, :] < gn))
    c_br = acc_br.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm_bot[:, None] * stride_cm + offs_cn_right[None, :], c_br,
             mask=(offs_cm_bot[:, None] < gm) & (offs_cn_right[None, :] < gn))


@triton.jit
def _grouped_gemm_tile_generic(
    pid_m,
    pid_n,
    a_ptr,
    b_ptr,
    c_ptr,
    gm,
    gn,
    gk,
    stride_am,
    stride_bn,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    HAS_K_TAIL: tl.constexpr,
):
    """One [BLOCK_SIZE_M, BLOCK_SIZE_N] output tile, compiler-pipelined.

    Tile-shape generic, unlike the 2x2 quadrant path which is pinned to
    256x256/BK=64. Small-M problems (MoE-style) do not have enough 256x256 tiles
    to fill 256 CUs, so they need a tile that actually fits them. This is that
    path. B is read through its column-major [K, N] view, so K is contiguous on
    axis 0, the dot consumes it with no transpose, and the K-tail is a plain
    axis-0 mask.
    """
    offs_am = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % gn
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    n_full = gk // BLOCK_SIZE_K
    for _ in tl.range(0, n_full, num_stages=NUM_STAGES):
        tl.multiple_of(a_ptrs, [16, 16])
        tl.multiple_of(b_ptrs, [16, 16])
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    # Partial last K-tile. Offsets are rematerialized rather than reusing the
    # loop-carried pointers, to keep them out of the pipelined loop's live set.
    if HAS_K_TAIL and n_full * BLOCK_SIZE_K < gk:
        k_start = n_full * BLOCK_SIZE_K
        k_mask = offs_k < gk - k_start
        a_t = tl.load(a_ptr + offs_am[:, None] * stride_am + (k_start + offs_k[None, :]), mask=k_mask[None, :],
                      other=0.0)
        b_t = tl.load(b_ptr + (k_start + offs_k[:, None]) + offs_bn[None, :] * stride_bn, mask=k_mask[:, None],
                      other=0.0)
        acc = tl.dot(a_t, b_t, acc, allow_tf32=False)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :], c,
             mask=(offs_cm[:, None] < gm) & (offs_cn[None, :] < gn))


@triton.jit
def grouped_gemm_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # how many tiles along M to group by
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    # 0 = 2x2 quadrant inter-wave pipeline (pinned 256x256/BK=64, for large tiles)
    # 1 = tile-shape-generic compiler-pipelined path (for small-M problems)
    TILE_MODE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    # False when the host can prove gk is exactly covered by the pipelined
    # region for every group, letting the masked K-tail be compiled out.
    HAS_K_TAIL: tl.constexpr,
):
    """Persistent, XCD-grouped scheduler over the whole group of GEMMs.
    The per-tile compute is selected by TILE_MODE."""
    pid = tl.program_id(0)

    # Program id after L2 remapping
    pid = chiplet_transform_chunked(pid, NUM_SM, NUM_XCDS, XCD_CHUNK)

    if TILE_MODE == 0:
        HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
        HALF_N: tl.constexpr = BLOCK_SIZE_N // 2

        # Swizzled (row/col-permuted) LDS layout pinned to kill bank conflicts.
        # All four half-tiles are [128, 64].
        tl.static_assert(HALF_M == 128 and HALF_N == 128 and BLOCK_SIZE_K == 64,
                         "pinned swizzle bases are hardcoded for [128, 64] half-tiles")
        smem_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
            [(512, 16)],
            [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0],
             [8, 0]],
            [HALF_M, BLOCK_SIZE_K],
        )

        # Four double-buffered LDS allocations, one per operand half-tile, reused for
        # every tile this program computes. Both A and B halves are [outer, K]
        smem_a_top = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS, layout=smem_layout)
        smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS, layout=smem_layout)
        smem_b_left = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS, layout=smem_layout)
        smem_b_right = tlx.local_alloc((HALF_N, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS, layout=smem_layout)

    # Which global output tile we are computing
    tile_idx = pid

    # The global tile id for where the current group begins
    last_problem_end = 0

    for g in range(group_size):
        # Load base pointers
        a_ptr = tl.multiple_of(tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        b_ptr = tl.multiple_of(tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

        # Load gemm sizes
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gn = tl.multiple_of(gn, 16)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)

        # Load strides
        stride_am = tl.load(g_lds + g * 3)  # A row stride
        stride_bn = tl.multiple_of(tl.load(g_lds + g * 3 + 1), 16)  # B N-stride
        stride_cm = tl.load(g_lds + g * 3 + 2)  # C row stride

        # How many tiles are necessary to compute this specific gemm
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        # This program owns the tiles where tile_idx lies within this group's
        # tiles in the range [last_problem_end, +num_tiles)
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            # GROUP_SIZE_M swizzle within this group's tile grid
            local = tile_idx - last_problem_end
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles
            group_id = local // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            # Compute this (pid_m, pid_n) output tile. Scheduler-agnostic logic.
            if TILE_MODE == 0:
                _grouped_gemm_tile(pid_m, pid_n, a_ptr, b_ptr, c_ptr, gm, gn, gk, stride_am, stride_bn, stride_cm,
                                   smem_a_top, smem_a_bot, smem_b_left, smem_b_right, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                   BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_BUFFERS=NUM_BUFFERS,
                                   HAS_K_TAIL=HAS_K_TAIL)
            else:
                _grouped_gemm_tile_generic(pid_m, pid_n, a_ptr, b_ptr, c_ptr, gm, gn, gk, stride_am, stride_bn,
                                           stride_cm, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                                           BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_STAGES=NUM_STAGES, HAS_K_TAIL=HAS_K_TAIL)

            # Program p owns tiles p, p+NUM_SM, p+2*NUM_SM, and so on
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# Tuned config for the large-tile quadrant path (TILE_MODE 0).
_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "NUM_BUFFERS": 2,
    "XCD_CHUNK": 16,
    "num_warps": 8,
    "TILE_MODE": 0,
    "NUM_STAGES": 1,
}


def _cdiv(a, b):
    return -(-a // b)


def _n_tiles(shapes, bm, bn):
    return sum(_cdiv(M, bm) * _cdiv(N, bn) for (M, N, K) in shapes)


def _needs_k_tail(shapes, cfg):
    """Does any group have K the pipelined region cannot cover exactly?

    Quadrant path consumes K two BLOCK_K tiles at a time, so it covers only an
    even number of whole tiles. The generic path covers every whole tile.
    """
    if cfg["TILE_MODE"] == 0:
        return True
    bk = cfg["BLOCK_SIZE_K"]
    for (_, _, K) in shapes:
        if K > (K // bk) * bk:
            return True
    return False


# Saturated per-tile throughput in TFLOP/s, measured on 16x4096^3 at a fixed
# 1104 MHz deterministic clock. That shape puts 4096 equal-sized tiles on 256
# CUs, so machine utilization and padding efficiency are both exactly 1.0 and
# the number isolates how fast the tile itself runs. Values are (rate, BLOCK_K,
# num_warps, NUM_STAGES) with the best (BLOCK_K, warps, stages) for that tile.
_QUAD_RATE = 782.7
_GENERIC_RATES = {
    (256, 256): (683.5, 32, 8, 3),
    (128, 256): (567.9, 32, 8, 3),
    (256, 128): (559.6, 64, 8, 3),
    (128, 128): (542.9, 64, 8, 3),
    (128, 64): (441.2, 128, 8, 3),
    (64, 128): (426.5, 128, 8, 3),
    (64, 64): (347.7, 128, 8, 3),
}


def _tile_score(shapes, bm, bn, rate, nsm):
    """Predicted delivered throughput for tiling `shapes` as bm x bn tiles.

    Three independent factors, each measurable:
      rate  -- how fast one tile runs when the machine is saturated.
      util  -- fraction of CUs doing work. A partial final wave leaves CUs idle,
               which is what cripples 256x256 on small-M MoE shapes.
      pad   -- useful FLOPs / computed FLOPs. Rounding M up to bm and N up to bn
               is real MFMA work that is then masked away by the C store.
    """
    tiles = sum(_cdiv(M, bm) * _cdiv(N, bn) for (M, N, _) in shapes)
    if tiles == 0:
        return 0.0
    util = tiles / (_cdiv(tiles, nsm) * nsm)
    useful = sum(M * N * K for (M, N, K) in shapes)
    padded = sum(_cdiv(M, bm) * bm * _cdiv(N, bn) * bn * K for (M, N, K) in shapes)
    return rate * util * (useful / padded)


def _pick_config(shapes):
    """Choose a launch config from the group's shapes.

    The quadrant path is by far the fastest per-tile engine (782.7 vs 683.5
    TFLOP/s for the best generic tile) but it only exists at 256x256. On small-M
    MoE shapes that tile is too coarse in two ways. It leaves most CUs idle, and
    it rounds every M up to 256. Score both engines on the same footing and take
    the winner.
    """
    nsm = num_sms()

    def entry(bm, bn, rate, cfg):
        tiles = sum(_cdiv(M, bm) * _cdiv(N, bn) for (M, N, _) in shapes)
        util = tiles / (_cdiv(tiles, nsm) * nsm) if tiles else 0.0
        return (_tile_score(shapes, bm, bn, rate, nsm), bm * bn / (bm + bn), util, cfg)

    # XCD_CHUNK sets how many consecutive tiles land on one chiplet. When there
    # are many tiles per CU a long chunk keeps an A/B panel resident in that
    # XCD's L2 across several tiles; when there is barely one wave, a long chunk
    # just unbalances the chiplets and a short one spreads the work.
    quad = dict(_CONFIG)
    quad_tiles = sum(_cdiv(M, 256) * _cdiv(N, 256) for (M, N, _) in shapes)
    quad["XCD_CHUNK"] = 32 if quad_tiles >= 2 * nsm else 8
    cands = [entry(256, 256, _QUAD_RATE, quad)]
    for (bm, bn), (rate, bk, warps, stages) in _GENERIC_RATES.items():
        cands.append(
            entry(
                bm, bn, rate, {
                    "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8, "NUM_BUFFERS": 2,
                    "XCD_CHUNK": 16, "num_warps": warps, "TILE_MODE": 1, "NUM_STAGES": stages
                }))

    # c[0] = predicted throughput
    # c[1] = arithmetic intensity
    # c[2] = machine utilization
    # c[3] = the config dict
    # The score model carries roughly +-10% error, so keep any candidates within
    # 10% of the top score as selected candidates. Then break the "tie" using:
    #   1. arithmetic intensity bm*bn/(bm+bn): least data moved per FLOP, so the
    #      most headroom against whatever the model is not capturing.
    #   2. machine utilization -- the score charges padding waste and idle CUs
    #      equally, but a padded lane at least keeps its CU busy. 128x256 and
    #      256x128 have identical intensity and score on A_deepK, but the latter
    #      fills every CU and measures 3.7% faster.
    top = max(c[0] for c in cands)
    return max((c for c in cands if c[0] >= 0.9 * top), key=lambda c: (c[1], c[2]))[3]


def _make_grouped_gemm_args(group_A, group_B, config=None):
    """Construct every device tensor the kernel needs (pointer / size / stride
    arrays and output buffers). This is the host-side setup. _bench keeps it out
    of the timed region.

    group_A[i]: fp16 [M_i, K_i] row-major (K contiguous).
    group_B[i]: fp16 [K_i, N_i] column-major (K contiguous) == [N_i, K_i].t().
    """
    shapes = [(A.shape[0], B.shape[1], A.shape[1]) for A, B in zip(group_A, group_B)]
    cfg = _pick_config(shapes)
    if config:
        cfg.update(config)
    if not config or "HAS_K_TAIL" not in config:
        cfg["HAS_K_TAIL"] = _needs_k_tail(shapes, cfg)

    G = len(group_A)
    assert len(group_B) == G, "group_A / group_B length mismatch"

    A_addrs, B_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"K mismatch: A has K={K}, B has K={Kb}"
        assert B.stride(0) == 1, "B must be column-major [K, N] (K contiguous, stride(0)==1)"
        C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        # lda = A row stride (K); ldb = B N-stride (== K for col-major); ldc = C row stride (N)
        g_lds += [A.stride(0), B.stride(1), C.stride(0)]

    d_a_ptrs = torch.tensor(A_addrs, dtype=torch.int64, device=DEVICE)
    d_b_ptrs = torch.tensor(B_addrs, dtype=torch.int64, device=DEVICE)
    d_c_ptrs = torch.tensor(C_addrs, dtype=torch.int64, device=DEVICE)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

    return d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg, group_C


def _perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg):
    """Forward already-constructed tensors straight to the kernel (no host-side
    setup), so it can be timed on its own."""
    NUM_SM = num_sms()
    grouped_gemm_kernel[(NUM_SM, )](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        G,
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        NUM_XCDS=NUM_XCDS,
        XCD_CHUNK=cfg["XCD_CHUNK"],
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        TILE_MODE=cfg["TILE_MODE"],
        NUM_STAGES=cfg["NUM_STAGES"],
        HAS_K_TAIL=cfg["HAS_K_TAIL"],
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma)
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )


def grouped_gemm(group_A, group_B, config=None):
    """group_A[i]: fp16 [M_i, K_i] row-major (K contiguous).
       group_B[i]: fp16 [K_i, N_i] COLUMN-major (K contiguous) == [N_i, K_i].t().
    """
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg, group_C = _make_grouped_gemm_args(
        group_A, group_B, config)
    _perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg)
    return group_C


def _rand_groups(shape_spec, seed=0):
    """A[M,K] row-major; B[K,N] column-major (K contiguous) via a [N,K].t() view."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    group_A, group_B = [], []
    for (M, N, K) in shape_spec:
        group_A.append(torch.randn((M, K), device=DEVICE, dtype=torch.float16, generator=g))
        Bt = torch.randn((N, K), device=DEVICE, dtype=torch.float16, generator=g)  # [N,K] row-major
        group_B.append(Bt.t())  # [K,N] view, stride (1, K) == column-major
    return group_A, group_B


def _check(shape_spec, label):
    group_A, group_B = _rand_groups(shape_spec)
    group_C = grouped_gemm(group_A, group_B)
    for i, (A, B) in enumerate(zip(group_A, group_B)):
        ref = torch.matmul(A, B)
        torch.testing.assert_close(group_C[i], ref, atol=1e-2, rtol=1e-2)
    print(f"  [PASS] {label} ({len(shape_spec)} groups)")


def test_op():
    _check([(1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128)], "ragged M=N=K")
    _check([(4096, 4096, 4096), (2048, 4096, 4096), (1000, 4096, 4096), (333, 4096, 4096)],
           "ragged-M (MoE-style), N=K=4096")
    _check([(512, 300, 4000), (333, 1000, 1500), (128, 128, 100), (256, 704, 320)], "k/n-unaligned")
    _check([(1, 64, 64), (33, 128, 50)], "tiny")
    print("test_op: all correctness checks passed")


def _bench():

    def tflops(ms, total_flops):
        return total_flops * 1e-12 / (ms * 1e-3)

    n = 16
    spec = [(4096, 4096, 4096)] * n
    group_A, group_B = _rand_groups(spec)
    total_flops = sum(2 * M * N * K for (M, N, K) in spec)

    # Host-side setup (pointer/size/stride tensors, output buffers) is built once,
    # outside the timed region
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg, _ = _make_grouped_gemm_args(group_A, group_B)
    ms = triton.testing.do_bench(lambda: _perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg), rep=100)
    print(f"  fast grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ group_B[i] for i in range(n)], rep=100)
    print(f"  torch loop        : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
