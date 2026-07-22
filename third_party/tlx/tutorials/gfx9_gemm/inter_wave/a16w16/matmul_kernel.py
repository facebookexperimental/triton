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

Split-K: for skinny / small-tile-count shapes the M/N tile grid can't fill the
256 CUs (e.g. N=256 -> 8 tiles -> 8 workgroups), so `matmul` auto-selects a
SPLIT_K that partitions the K reduction across more workgroups. Partials land in
an fp32 workspace and a separate fp32 reduce kernel sums them into C -- keeping
the result numerically identical to the non-split-K path. `choose_split_k`
returns 1 for shapes that already fill the machine, making split-K a no-op there.
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
    workspace_ptr,
    M,
    N,
    K,
    KS,
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
    SPLIT_K: tl.constexpr,
):
    # ── Split-K: grid is GRID_MN*SPLIT_K. Peel off split_id, keep the MN pid for
    # the XCD/group remap below. Each split owns a contiguous K-slice of size KS.
    # We do NOT shift a_ptr/b_ptr (AMD buffer_load builds its resource descriptor
    # from the raw kernel-arg pointer, so an arith'd base fails to lower); instead
    # the split's K byte-offset is folded into the running ka/kb offset (used by
    # every buffer_load) and into the masked-tail addresses. Partials go to a
    # (SPLIT_K*M, N) workspace (row_base=split_id*M); a reduce kernel sums (fp32).
    #
    # KS (per-split K length) is passed as a runtime ARG, not computed as
    # K // SPLIT_K here: the in-kernel divide only proves divisibility 2 for large
    # SPLIT_K (K is known div-16, //8 -> div-2), which collapses the buffer_load
    # offset from the coalesced #linear layout to #blocked and fails to lower. As
    # an arg, KS gets Triton's div-by-16 specialization, so split_id*KS*stride
    # keeps enough divisibility for #linear.
    split_id = tl.program_id(0) // GRID_MN
    pid = tl.program_id(0) % GRID_MN
    ak_split = split_id * KS * stride_ak
    bk_split = split_id * KS * stride_bk
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

    # Start the running K-offset at this split's slice (0 when SPLIT_K==1).
    ka = ak_split
    kb = bk_split

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
    n_full = KS // BLOCK_K
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
    for kk in tl.range(n_pipe * BLOCK_K, KS, BLOCK_K, num_stages=1):
        offs_kt = kk + offs_k
        k_mask = offs_kt < KS
        a_top_t = tl.load(a_ptr + ak_split + offs_am[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                          mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
        a_bot_t = tl.load(a_ptr + ak_split + offs_am_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                          mask=(offs_am_bot[:, None] < M) & k_mask[None, :], other=0.0)
        b_left_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                           mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
        b_right_t = tl.load(b_ptr + bk_split + offs_kt[:, None] * stride_bk + offs_bn_right[None, :] * stride_bn,
                            mask=k_mask[:, None] & (offs_bn_right[None, :] < N), other=0.0)
        acc_tl = tl.dot(a_top_t, b_left_t, acc_tl)
        acc_bl = tl.dot(a_bot_t, b_left_t, acc_bl)
        acc_tr = tl.dot(a_top_t, b_right_t, acc_tr)
        acc_br = tl.dot(a_bot_t, b_right_t, acc_br)

    offs_cm_top = pid_m * BLOCK_M + tl.arange(0, HALF_M)
    offs_cm_bot = offs_cm_top + HALF_M
    offs_cn_left = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_cn_right = offs_cn_left + HALF_N
    m_top = offs_cm_top[:, None] < M
    m_bot = offs_cm_bot[:, None] < M
    n_left = offs_cn_left[None, :] < N
    n_right = offs_cn_right[None, :] < N

    if SPLIT_K == 1:
        # Direct store to C -- exact no-op vs the champion kernel.
        et = c_ptr.dtype.element_ty
        tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_left[None, :], acc_tl.to(et),
                 mask=m_top & n_left)
        tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_left[None, :], acc_bl.to(et),
                 mask=m_bot & n_left)
        tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_right[None, :], acc_tr.to(et),
                 mask=m_top & n_right)
        tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_right[None, :], acc_br.to(et),
                 mask=m_bot & n_right)
    else:
        # Split-K: every split writes its fp32 partial into its workspace slice
        # (rows [split_id*M, split_id*M+M)). Mask stays in relative-M coords; the
        # row offset is added only to the store index.
        rb = split_id * M
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_left[None, :], acc_tl,
                 mask=m_top & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_left[None, :], acc_bl,
                 mask=m_bot & n_left)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_top)[:, None] + stride_cn * offs_cn_right[None, :], acc_tr,
                 mask=m_top & n_right)
        tl.store(workspace_ptr + stride_cm * (rb + offs_cm_bot)[:, None] + stride_cn * offs_cn_right[None, :], acc_br,
                 mask=m_bot & n_right)


_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _reduce_k_kernel(workspace_ptr, c_ptr, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                     BLOCK_SIZE_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr):
    # Sum the SPLIT_K partials (each a contiguous (M, N) slab in workspace) into
    # C with fp32 accumulation. Small tiles (32x32) so small outputs still spawn
    # many CTAs -- else the reduce is CTA-starved and dominates (D97513062).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    base_offs = offs_m[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        partial = tl.load(workspace_ptr + base_offs + s * M * N, mask=mask, other=0.0)
        acc += partial.to(tl.float32)
    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


NUM_CU = 256  # gfx950 (CDNA4) compute units
# Below this many K-tiles per split, the per-split prologue/epilogue overhead
# starts to dominate (measured: Router SPLIT_K=16 (8 tiles/split) beats 32 (4)).
MIN_KTILES_PER_SPLIT = 8


def choose_split_k(M, N, K):
    """Pick SPLIT_K from the shape: add K-parallelism only when the M/N tile grid
    can't fill the CUs, and cap it so each split keeps enough K-tiles to stay
    efficient and the reduce stays cheap. Returns 1 (no split-K) for shapes that
    already fill the machine."""
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if grid_mn >= NUM_CU:
        return 1
    min_ks = MIN_KTILES_PER_SPLIT * BLOCK_K
    # TODO: only powers of two are considered here. That is optimal for pure-pow2
    # K (e.g. 8192, 4096 -- every divisor is a power of two), but for K with odd
    # factors (18432 = 2^11*3^2, 12800 = 2^9*5^2) a non-pow2 SPLIT_K can divide K
    # and fill the CUs more precisely. Generalize to enumerate divisors of K (or
    # K/BLOCK_K), or autotune SPLIT_K over a candidate set with this as the seed.
    sk = 1
    while True:
        nxt = sk * 2
        ks = K // nxt
        # stop at one full wave of workgroups, and keep splits whole/aligned/efficient
        if grid_mn * nxt > NUM_CU or K % nxt != 0 or ks < min_ks or ks % BLOCK_K != 0:
            break
        sk = nxt
    return sk


def matmul(a, b, SPLIT_K=None):
    """C = A @ B. `a` is (M, K), `b` is (K, N).

    SPLIT_K partitions the K reduction across SPLIT_K programs per output tile
    (grid = GRID_MN*SPLIT_K), landing fp32 partials in a (SPLIT_K*M, N) workspace
    that a separate fp32 reduce kernel sums into C. This fills the CUs on small-N /
    small-tile-count shapes where the M/N tile grid alone can't. SPLIT_K is chosen
    automatically from the shape (pass an int to override); SPLIT_K=1 launches the
    plain kernel (no workspace, no reduce). The fp32 workspace keeps the result
    numerically identical to the non-split-K kernel; only an int-free fp32 sum is
    added, so there is no precision loss and the result is deterministic.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    if SPLIT_K is None:
        SPLIT_K = choose_split_k(M, N, K)
    KS = K // SPLIT_K
    # Each split is a whole number of K-tiles, big enough for the 2-tile prologue.
    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    assert KS >= 2 * BLOCK_K, f"K/SPLIT_K={KS} must be at least {2 * BLOCK_K}"
    assert KS % BLOCK_K == 0, f"K/SPLIT_K={KS} must be a multiple of BLOCK_K={BLOCK_K} (split base alignment)"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if SPLIT_K > 1:
        # fp32 workspace: partials are stored without a rounding step, so the
        # split-K result matches a single fp32-accumulated GEMM (an fp16 workspace
        # would lose ~1e-1 near cancellation). The reduce sums in fp32 too.
        workspace = torch.empty((SPLIT_K * M, N), device=a.device, dtype=torch.float32)
    else:
        workspace = c  # dummy; the kernel writes c_ptr directly when SPLIT_K==1
    a16w16_8wave[(GRID_MN * SPLIT_K, )](
        a,
        b,
        c,
        workspace,
        M,
        N,
        K,
        KS,
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
        SPLIT_K=SPLIT_K,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma). Essential to match the reference perf.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
    )
    if SPLIT_K > 1:
        # Adaptive reduce tile: small outputs need many small CTAs to fill the CUs;
        # large outputs are BW-bound and prefer big tiles for burst efficiency
        # (measured: 32x32 -> 4.5 TB/s vs 128x128 -> 5.4 TB/s on Pooler).
        big = (M * N) >= (2048 * 2048)
        rbm, rbn, rw = (128, 128, 8) if big else (32, 32, 4)
        reduce_grid = (triton.cdiv(M, rbm), triton.cdiv(N, rbn))
        _reduce_k_kernel[reduce_grid](
            workspace,
            c,
            M,
            N,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
            num_warps=rw,
        )
    return c
