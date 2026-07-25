"""AMD TLX grouped GEMM for gfx950.

Storage (both operands row-major with K, the contraction dim, innermost):
  A[i]: [M, K] row-major.
  B[i]: [N, K] row-major, handed to the kernel as a [K, N] .t() view
        (logically column-major [K, N]), the natural weight layout.

With K contiguous for both A and B, the A and B tiles can load identically
as [outer, K] tiles via plain coalesced `async_load`. B is loaded directly into a [BLOCK_K, BLOCK_N] LDS tile (K on dim0, which is
contiguous for the column-major B), so the dot is `tl.dot(a, b)`.

The hot loop is mask-free. M is the outer axis of A and N is the outer axis
of B, so both dimensions are wrapped with a modulo to prevent out of bounds
reads. Garbage rows and columns are discarded at the final store to C. The
hot loop runs full K-tiles that it can load with wide `async_load` instructions.
The partial last K-tile (if it exists) is a cold masked `tl.load`. The only
constraint is that gn % 8 = 0 and stride_bn % 16 = 0 (for wide vector loads
and no masking).
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
    # LDS ring buffers, allocated once by the scheduler and reused per tile
    smemA,
    smemB_left,
    smemB_right,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """
    Compute one [BLOCK_SIZE_M, BLOCK_SIZE_N] output tile of ``A @ B`` for a
    single GEMM and store it to C.

    N-split pipeline (ported from v9_beyond_hotloop): B is split into left/right
    [BLOCK_K, HALF_N] halves fed to two independent accumulators. Each K tile is
    two async groups (A + B_left, then B_right) so the left/right MFMAs alternate
    with the local/global memory ops (four regions per two-K-tile body).
    """
    tl.static_assert(NUM_BUFFERS == 2, "the four-region N-split pipeline requires two LDS buffers")

    # How many K-tile iterations we have where each tile is full BLOCK_SIZE_K size
    k_full_chunk_iters = gk // BLOCK_SIZE_K

    HALF_N: tl.constexpr = BLOCK_SIZE_N // 2

    # A rows and B columns are wrapped to keep all reads in bound and maintain
    # vectorized loads along the K dimension. The garbage from wrapped lanes
    # is dropped by the masked C store
    offs_am = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % gm, BLOCK_SIZE_M)
    offs_bn_left = tl.multiple_of((pid_n * BLOCK_SIZE_N + tl.arange(0, HALF_N)) % gn, HALF_N)
    offs_bn_right = tl.multiple_of((pid_n * BLOCK_SIZE_N + HALF_N + tl.arange(0, HALF_N)) % gn, HALF_N)

    # K is the contiguous/innermost axis of both A and B tiles
    offs_k = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK_SIZE_K), BLOCK_SIZE_K), BLOCK_SIZE_K)

    # Full tile offsets computed once. The running K position is advanced with the
    # scalars a_k / b_k (K stride is 1 for both A and col-major B). B halves are
    # loaded directly as [BLOCK_K, HALF_N] (K on dim0, contiguous for col-major B):
    # feeds tl.dot as [K, N] with NO local_trans.
    a_off = offs_am[:, None] * stride_am + offs_k[None, :]
    bl_off = offs_k[:, None] + offs_bn_left[None, :] * stride_bn
    br_off = offs_k[:, None] + offs_bn_right[None, :] * stride_bn

    a_k = tl.zeros([], dtype=tl.int32)
    b_k = tl.zeros([], dtype=tl.int32)

    # Two independent accumulators, one per N half
    acc_left = tl.zeros((BLOCK_SIZE_M, HALF_N), dtype=tl.float32)
    acc_right = tl.zeros((BLOCK_SIZE_M, HALF_N), dtype=tl.float32)

    if k_full_chunk_iters >= NUM_BUFFERS:
        # ── Prologue: fill both buffers (four async groups) ──
        for pi in tl.static_range(0, NUM_BUFFERS):
            tlx.buffer_load_to_local(smemA[pi], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smemB_left[pi], b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

            tlx.buffer_load_to_local(smemB_right[pi], b_ptr, br_off + b_k)
            tlx.async_load_commit_group()

            a_k += BLOCK_SIZE_K
            b_k += BLOCK_SIZE_K

        # Wait for group 0 (A + B_left in buffer 0).
        tlx.async_load_wait_group(3)

        a_tile = tlx.local_load(smemA[0], relaxed=True)
        b_left_tile = tlx.local_load(smemB_left[0], relaxed=True)

        # Leave two loaded K tiles for the common epilogue. If the number of
        # full K tiles is odd, leave the final tile for a separate cold drain.
        hot_loop_end = k_full_chunk_iters - NUM_BUFFERS - (k_full_chunk_iters % 2)

        # ── Main loop: step 2, four regions per body ──
        for k in tl.range(0, hot_loop_end, 2, num_stages=1):
            # ──── Region 0 ────
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                tlx.async_load_wait_group(2)
                b_right_tile = tlx.local_load(smemB_right[0], relaxed=True)

                tlx.buffer_load_to_local(smemA[0], a_ptr, a_off + a_k)
                tlx.buffer_load_to_local(smemB_left[0], b_ptr, bl_off + b_k)
                tlx.async_load_commit_group()

            # ──── Region 1 ────
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                tlx.async_load_wait_group(2)
                a_tile = tlx.local_load(smemA[1], relaxed=True)
                b_left_tile = tlx.local_load(smemB_left[1], relaxed=True)

                tlx.buffer_load_to_local(smemB_right[0], b_ptr, br_off + b_k)
                tlx.async_load_commit_group()

            a_k += BLOCK_SIZE_K
            b_k += BLOCK_SIZE_K

            # ──── Region 2 ────
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                tlx.async_load_wait_group(2)
                b_right_tile = tlx.local_load(smemB_right[1], relaxed=True)

                tlx.buffer_load_to_local(smemA[1], a_ptr, a_off + a_k)
                tlx.buffer_load_to_local(smemB_left[1], b_ptr, bl_off + b_k)
                tlx.async_load_commit_group()

            # ──── Region 3 ────
            with tlx.warp_pipeline_stage("mfma", priority=0):
                acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

            with tlx.warp_pipeline_stage("mem", priority=1):
                tlx.async_load_wait_group(2)
                a_tile = tlx.local_load(smemA[0], relaxed=True)
                b_left_tile = tlx.local_load(smemB_left[0], relaxed=True)

                tlx.buffer_load_to_local(smemB_right[1], b_ptr, br_off + b_k)
                tlx.async_load_commit_group()

            a_k += BLOCK_SIZE_K
            b_k += BLOCK_SIZE_K

        # ── Epilogue: drain the final two pipelined K tiles ──
        acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)
        tlx.async_load_wait_group(0)
        b_right_tile = tlx.local_load(smemB_right[0], relaxed=True)

        acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)
        a_tile = tlx.local_load(smemA[1], relaxed=True)
        b_left_tile = tlx.local_load(smemB_left[1], relaxed=True)

        acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)
        b_right_tile = tlx.local_load(smemB_right[1], relaxed=True)
        acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

        # Grouped GEMM accepts arbitrary runtime K, so drain one additional full
        # tile when k_full_chunk_iters is odd. a_k / b_k already point at the
        # final full tile here.
        if (k_full_chunk_iters % 2) != 0:
            tlx.buffer_load_to_local(smemA[0], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smemB_left[0], b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

            tlx.buffer_load_to_local(smemB_right[0], b_ptr, br_off + b_k)
            tlx.async_load_commit_group()
            tlx.async_load_wait_group(0)

            a_tile = tlx.local_load(smemA[0], relaxed=True)
            b_left_tile = tlx.local_load(smemB_left[0], relaxed=True)
            b_right_tile = tlx.local_load(smemB_right[0], relaxed=True)
            acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)
            acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

            a_k += BLOCK_SIZE_K
            b_k += BLOCK_SIZE_K
    else:
        # Small-K path: there can be at most one full K tile.
        for i in tl.range(0, k_full_chunk_iters):
            tlx.buffer_load_to_local(smemA[i], a_ptr, a_off + a_k)
            tlx.buffer_load_to_local(smemB_left[i], b_ptr, bl_off + b_k)
            tlx.async_load_commit_group()

            tlx.buffer_load_to_local(smemB_right[i], b_ptr, br_off + b_k)
            tlx.async_load_commit_group()

            a_k += BLOCK_SIZE_K
            b_k += BLOCK_SIZE_K

        tlx.async_load_wait_group(0)

        for i in tl.range(0, k_full_chunk_iters):
            a_tile = tlx.local_load(smemA[i], relaxed=True)
            b_left_tile = tlx.local_load(smemB_left[i], relaxed=True)
            b_right_tile = tlx.local_load(smemB_right[i], relaxed=True)
            acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)
            acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

    # Peel the partial last K-tile (gk % BLOCK_SIZE_K != 0). a_k / b_k point here.
    if k_full_chunk_iters * BLOCK_SIZE_K < gk:
        k_start = k_full_chunk_iters * BLOCK_SIZE_K
        k_mask = offs_k < gk - k_start
        a_tile = tl.load(a_ptr + a_off + a_k, mask=offs_k[None, :] < gk - k_start, other=0.0)
        b_left_tile = tl.load(b_ptr + bl_off + b_k, mask=k_mask[:, None], other=0.0)
        b_right_tile = tl.load(b_ptr + br_off + b_k, mask=k_mask[:, None], other=0.0)
        acc_left = tl.dot(a_tile, b_left_tile, acc_left, allow_tf32=False)
        acc_right = tl.dot(a_tile, b_right_tile, acc_right, allow_tf32=False)

    # Store the two N halves and mask out OOB rows and columns.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn_left = pid_n * BLOCK_SIZE_N + tl.arange(0, HALF_N)
    c_left_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn_left[None, :]
    c_left_mask = (offs_cm[:, None] < gm) & (offs_cn_left[None, :] < gn)
    tl.store(c_left_ptrs, acc_left.to(c_ptr.dtype.element_ty), mask=c_left_mask, cache_modifier=".cs")

    offs_cn_right = pid_n * BLOCK_SIZE_N + HALF_N + tl.arange(0, HALF_N)
    c_right_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn_right[None, :]
    c_right_mask = (offs_cm[:, None] < gm) & (offs_cn_right[None, :] < gn)
    tl.store(c_right_ptrs, acc_right.to(c_ptr.dtype.element_ty), mask=c_right_mask, cache_modifier=".cs")


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
):
    """
    Persistent, XCD-grouped scheduler over the whole group of GEMMs.

    Launches a fixed NUM_SM programs; each walks the flattened tile space of all
    GEMMs (tiles ``pid, pid + NUM_SM, ...`` after an L2-locality remap), applies a
    GROUP_SIZE_M swizzle to pick the (pid_m, pid_n) tile within a GEMM, and hands
    the actual tile computation to ``_grouped_gemm_tile``.
    """
    pid = tl.program_id(0)

    # Program id after L2 remapping
    pid = chiplet_transform_chunked(pid, NUM_SM, NUM_XCDS, XCD_CHUNK)

    HALF_N: tl.constexpr = BLOCK_SIZE_N // 2

    # LDS ring buffers reused throughout the entire program. B is split into two
    # [BLOCK_K, HALF_N] halves for the four-region N-split pipeline.
    smemA = tlx.local_alloc((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float16, NUM_BUFFERS)
    smemB_left = tlx.local_alloc((BLOCK_SIZE_K, HALF_N), tl.float16, NUM_BUFFERS)
    smemB_right = tlx.local_alloc((BLOCK_SIZE_K, HALF_N), tl.float16, NUM_BUFFERS)

    # Which global output tile we are computing
    tile_idx = pid

    # The global tile id for where the current group begins
    last_problem_end = 0

    for g in range(group_size):
        # Load base pointers
        a_ptr = tl.multiple_of(tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        b_ptr = tl.multiple_of(tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16)), 16)
        c_ptr = tl.multiple_of(tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16)), 16)

        # Load gemm sizes
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gn = tl.multiple_of(gn, 8)
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
            # Which tile we own relative to the current gemm
            local = tile_idx - last_problem_end

            # How many tiles are in the swizzle group
            num_pid_in_group = GROUP_SIZE_M * num_n_tiles

            # Which swizzle group the tile is in
            group_id = local // num_pid_in_group

            first_pid_m = group_id * GROUP_SIZE_M

            # Last group might have num rows less than GROUP_SIZE_M because
            # GROUP_SIZE_M does not evenly divide gm
            group_size_m = min(num_m_tiles - first_pid_m, GROUP_SIZE_M)

            # Ensure that consecutive PIDs are assigned tiles along m dimension in
            # groups of size GROUP_SIZE_M
            pid_m = first_pid_m + ((local % num_pid_in_group) % group_size_m)
            pid_n = (local % num_pid_in_group) // group_size_m

            # Compute this (pid_m, pid_n) output tile; scheduler-agnostic logic.
            _grouped_gemm_tile(pid_m, pid_n, a_ptr, b_ptr, c_ptr, gm, gn, gk, stride_am, stride_bn, stride_cm, smemA,
                               smemB_left, smemB_right, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                               BLOCK_SIZE_K=BLOCK_SIZE_K, NUM_BUFFERS=NUM_BUFFERS)

            # Program p owns tiles p, p+NUM_SM, p+2*NUM_SM, and so on
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles


# Best config
_CONFIG = {
    "BLOCK_SIZE_M": 256,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 4,
    "NUM_BUFFERS": 2,
    "XCD_CHUNK": 16,
    "num_warps": 8,
}


def _make_grouped_gemm_args(group_A, group_B, config=None):
    """Construct every device tensor the kernel needs (pointer / size / stride
    arrays and output buffers). This is the host-side setup; _bench keeps it out
    of the timed region, matching blackwell-grouped-gemm_test.py.

    group_A[i]: fp16 [M_i, K_i] row-major (K contiguous).
    group_B[i]: fp16 [K_i, N_i] COLUMN-major (K contiguous) == [N_i, K_i].t().
    """
    cfg = dict(_CONFIG)
    if config:
        cfg.update(config)

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
    setup), so it can be timed on its own like blackwell's triton_perf_fn."""
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
        num_warps=cfg["num_warps"],
        num_stages=1,
        matrix_instr_nonkdim=16,
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
    _check([(1024, 1024, 1024), (512, 512, 512), (256, 256, 256), (128, 128, 128)], "blackwell-ragged M=N=K")
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
    # outside the timed region; do_bench times only _perf_fn (the kernel launch),
    # matching blackwell-grouped-gemm_test.py.
    d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg, _ = _make_grouped_gemm_args(group_A, group_B)
    ms = triton.testing.do_bench(lambda: _perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, G, cfg), rep=100)
    print(f"  v3 grouped GEMM : {tflops(ms, total_flops):7.1f} TFLOPS ({ms:.3f} ms)")

    ms_torch = triton.testing.do_bench(lambda: [group_A[i] @ group_B[i] for i in range(n)], rep=100)
    print(f"  torch loop      : {tflops(ms_torch, total_flops):7.1f} TFLOPS ({ms_torch:.3f} ms)")


if __name__ == "__main__":
    test_op()
    print("\n16 x 4096 x 4096 x 4096:")
    _bench()
