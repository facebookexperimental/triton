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

The additive FP16/BF16 `streamk_matmul` schedule uses an extracted `matmul_tile`
load/LDS/MFMA pipeline with persistent full tiles plus a TritonBLAS-style
variable-work Stream-K tail, including an even-work fast path for the two
performance-critical production shapes.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
NUM_WARPS = 8
GROUP_SIZE_M = 4
NUM_XCDS = 8

MIN_K = 2 * BLOCK_K  # pipeline prefetches 2 whole K-tiles; the rest goes to the masked tail
KERNEL_NAME = "a16w16_8wave"
_LLVM_ATTRS = (("amdgpu-agpr-alloc", "0,0"), )
_READY_VALUE = 3


def _prune_register_configs(configs, named_args, **_):
    k = named_args["K"]
    if 128 <= k < 256 and named_args["M"] >= 16384 and named_args["N"] <= 128:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 1 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 64 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 4
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 1536 and named_args["M"] == 3072 and named_args["N"] == 3072:
        preferred = [
            config for config in configs if config.kwargs["NUM_XCDS"] == 8 and config.kwargs["BLOCK_M"] == 128
            and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 64 and config.kwargs["GROUP_M"] == 16
            and config.kwargs["waves_per_eu"] == 0 and config.num_warps == 4 and config.num_stages == 2
        ]
        if preferred:
            return preferred
    if k == 256 and named_args["M"] <= 1024 and named_args["N"] >= 16384:
        preferred = [
            config for config in configs
            if config.kwargs["BLOCK_M"] == 256 and config.kwargs["BLOCK_N"] == 128 and config.kwargs["BLOCK_K"] == 32
            and config.kwargs["GROUP_M"] == 4 and config.kwargs["waves_per_eu"] == 2 and config.num_warps == 4
        ]
        if preferred:
            return preferred
    return configs


# Triton TR001: autotune the register path across stock ROCm tiles and the
# deeper software-pipelined tiles used by the TorchTLX register path.
_REGISTER_CONFIGS = [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=2,
    ) for block_m, block_n, block_k, group_m, num_warps, waves_per_eu in (
        (16, 16, 256, 4, 4, 2),
        (32, 16, 256, 4, 4, 0),
        (32, 32, 16, 8, 4, 2),
        (32, 32, 128, 8, 4, 0),
        (32, 64, 64, 8, 4, 0),
        (64, 16, 128, 8, 4, 2),
        (64, 32, 32, 8, 4, 0),
        (64, 32, 64, 8, 4, 0),
        (64, 32, 64, 8, 8, 0),
        (64, 32, 128, 8, 4, 0),
        (64, 64, 16, 8, 4, 0),
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 16, 8, 0),
        (64, 64, 256, 4, 8, 0),
        (64, 128, 32, 4, 4, 2),
        (64, 128, 32, 8, 8, 0),
        (64, 128, 64, 4, 8, 0),
        (64, 128, 128, 4, 8, 0),
        (128, 32, 32, 8, 4, 0),
        (128, 32, 64, 8, 4, 0),
        (128, 64, 32, 8, 4, 2),
        (128, 64, 64, 16, 4, 0),
        (128, 64, 128, 4, 8, 0),
        (128, 128, 32, 16, 4, 2),
        (128, 128, 32, 16, 8, 0),
        (128, 128, 32, 16, 8, 2),
        (128, 128, 64, 16, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (128, 128, 128, 16, 8, 0),
        (128, 256, 32, 16, 4, 2),
        (128, 256, 64, 4, 8, 0),
        (256, 64, 64, 4, 8, 0),
        (256, 128, 32, 4, 4, 2),
        (256, 128, 32, 16, 8, 0),
        (256, 128, 64, 4, 8, 0),
        (256, 256, 64, 4, 8, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 1,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 64, 64, 16, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (128, 256, 64, 8, 8, 3, 0),
        (128, 256, 64, 8, 8, 3, 1),
        (256, 128, 32, 4, 4, 3, 2),
        (256, 128, 32, 4, 4, 4, 2),
        (256, 256, 64, 4, 8, 3, 0),
        (256, 256, 64, 4, 8, 4, 0),
        (128, 256, 32, 8, 8, 3, 0),
    )
]

_REGISTER_CONFIGS += [
    triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": 8,
            "matrix_instr_nonkdim": 16,
            "waves_per_eu": waves_per_eu,
            "kpack": 1,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    ) for block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu in (
        (128, 64, 64, 4, 4, 2, 0),
        (128, 64, 64, 4, 4, 3, 0),
        (128, 128, 64, 8, 4, 2, 0),
        (128, 128, 64, 8, 4, 3, 0),
        (128, 128, 64, 16, 4, 2, 0),
        (128, 128, 64, 16, 4, 3, 0),
        (256, 128, 32, 4, 4, 3, 2),
    )
]

# Coalesced SIMD register layout for the [HALF_M, HALF_N] = [128, 128] fp16 quadrant
# store (num_warps=8, warp_size=64): each thread holds 8 contiguous N elements ->
# 128-bit buffer_store_dwordx4. Applied to the epilogue store via tlx.require_layout
# so tritongpu-coalesce sets the store to this #linear layout and AMD
# OptimizeEpilogue leaves it alone (it only rewrites #blocked stores) -- keeping the
# wide coalesced store instead of the narrow MMA-accumulator (dwordx2) fallback.
_C_STORE_SIMD_LAYOUT = tlx.layout(shape=((16, 4, 8), (8, 4)), stride=((8, 128, 512), (1, 4096)))


def _swz_offset_bases(shape, contig_dim):
    """Padded-shared swizzle offset bases for a 2D fp16 half-tile, derived from the
    tile shape so both tile sizes share one path (no per-size branch).

    `contig_dim` is the K-contiguous axis (0 or 1); its bits come first (fastest),
    then the free axis contributes its high bits (>= bit 4) before its low bits --
    the row/col permutation that makes the direct-to-LDS ds_reads bank-conflict-free
    on the 128x64 / 64x128 halves. A 128-wide free axis simply carries the extra top
    bit ([64,0] resp. [0,64]) that a 64-wide one omits. Used for both operands: the
    a half-tile [HALF_M, BLOCK_K] has K on dim 1, the b half-tile [BLOCK_K, HALF_N]
    has K on dim 0."""

    def basis(dim, i):
        return [1 << i, 0] if dim == 0 else [0, 1 << i]

    free_dim = 1 - contig_dim
    # log2 of each extent: int(n).bit_length() - 1 == floor(log2(n)), exact for the
    # power-of-two tile extents here (integer math, no float log2).
    cb = int(shape[contig_dim]).bit_length() - 1
    fb = int(shape[free_dim]).bit_length() - 1
    contig = [basis(contig_dim, i) for i in range(cb)]
    free = ([basis(free_dim, i) for i in range(4, fb)] + [basis(free_dim, i) for i in range(min(4, fb))])
    return contig + free


# Swizzle offset bases per (square) tile size, computed once from the tile shape by
# _swz_offset_bases. The @jit body can't call the generator (only constexpr module
# values are referenceable inside @jit), so precompute the base lists here and build
# the layout in-body, selecting by the constexpr tile size.
# The bases are built for a half-tile (2x2 quadrant tiling): HALF = tile // 2.
_HALF_256 = 256 // 2  # half of the 256x256 tile
_HALF_128 = 128 // 2  # half of the 128x128 tile
_A_BASES_256 = tl.constexpr(_swz_offset_bases([_HALF_256, BLOCK_K], 1))
_A_BASES_128 = tl.constexpr(_swz_offset_bases([_HALF_128, BLOCK_K], 1))
_B_BASES_256 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_256], 0))
_B_BASES_128 = tl.constexpr(_swz_offset_bases([BLOCK_K, _HALF_128], 0))
# Direct-to-LDS offset layouts inferred by the aligned 256x256 path. Pinning
# these keeps a merely 16-byte-aligned leading stride from falling back to a
# blocked layout that the AMD buffer-load lowering cannot consume.
_A_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((8, 1024, 64), (1, 512)))
_B_OFFSET_LAYOUT_256 = tlx.layout(shape=((8, 8, 8), (8, 2)), stride=((1024, 16, 1), (128, 8)))


@triton.autotune(
    configs=_REGISTER_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_register_configs},
)
@triton.jit
def _register_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    ADD_BIAS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int32)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_mn = grid_m * grid_n

    xcd_chunk: tl.constexpr = 4
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * xcd_chunk)) * (NUM_XCDS * xcd_chunk)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = (local_pid // xcd_chunk) * NUM_XCDS * xcd_chunk + xcd * xcd_chunk + local_pid % xcd_chunk

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_idx * BLOCK_K
        a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
        if K % BLOCK_K == 0:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    idx_m = rows[:, None]
    idx_n = cols[None, :]
    mask = (idx_m < M) & (idx_n < N)
    if ADD_BIAS:
        bias_offsets = idx_m * stride_bias_m + idx_n * stride_bias_n
        bias = tl.load(bias_ptr + bias_offsets, mask=mask, eviction_policy="evict_last")
        acc += bias.to(tl.float32)
    output_offsets = idx_m * N + idx_n
    tl.store(c_ptr + output_offsets, acc, mask=mask)


def _launch_register(a, b, bias=None, config=None):
    """Launch the register-resident gfx950 GEMM path.

    ``config=None`` autotunes over `_REGISTER_CONFIGS`. Passing an explicit
    config dict bypasses the autotuner and launches that config directly --
    the same convention the Blackwell/Hopper tutorials use so correctness
    tests can pin a config instead of paying for a sweep.
    """
    M, K = a.shape
    b_k, N = b.shape
    if K != b_k:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if bias is not None:
        if bias.shape != (M, N):
            raise ValueError(f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}")
        if bias.device != a.device or bias.dtype != a.dtype:
            raise ValueError("Bias and matrix operands must have matching device and dtype")
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    disable_agpr = (K == 256 and N > 256) or (K > 512 and (K % BLOCK_K != 0 or M * N <= 2 * 1024 * 1024))
    launch_options = {"llvm_fn_attrs": (("amdgpu-agpr-alloc", "0,0"), )} if disable_agpr else {}
    bias_ptr = bias if bias is not None else out
    args = (
        a,
        b,
        bias_ptr,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    if config is not None:
        grid = (triton.cdiv(M, config["BLOCK_M"]) * triton.cdiv(N, config["BLOCK_N"]), )
        _register_kernel.fn[grid](*args, ADD_BIAS=bias is not None, **config, **launch_options)
    else:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
        _register_kernel[grid](*args, ADD_BIAS=bias is not None, **launch_options)
    return out


@triton.jit
def matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, a_top_off, a_bot_off, b_left_off,
                b_right_off, ka, kb, n_steps, stride_ak, stride_bk, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    """Compute one output tile over an even contiguous range of K64 steps.

    ``ka`` and ``kb`` are the initial element offsets along K. ``n_steps`` must
    be even and at least two. Both the original data-centric kernel and the new
    Stream-K kernel use this same LDS/MFMA pipeline.
    """
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Keep the direct-to-LDS producer contract local to this extracted helper.
    # K is contiguous in A's second tensor dimension and B's first tensor
    # dimension. The helper boundary otherwise hides those width/alignment
    # facts from AxisInfo and buffer-load lowering falls back to an illegal
    # scalar copy.
    a_top_off = tl.max_contiguous(tl.multiple_of(a_top_off, (1, 8)), (1, 8))
    a_bot_off = tl.max_contiguous(tl.multiple_of(a_bot_off, (1, 8)), (1, 8))
    b_left_off = tl.max_contiguous(tl.multiple_of(b_left_off, (8, 1)), (8, 1))
    b_right_off = tl.max_contiguous(tl.multiple_of(b_right_off, (8, 1)), (8, 1))

    k_step_a = BLOCK_K * stride_ak
    k_step_b = BLOCK_K * stride_bk
    a_top_off_n = a_top_off + k_step_a
    a_bot_off_n = a_bot_off + k_step_a
    b_left_off_n = b_left_off + k_step_b
    b_right_off_n = b_right_off + k_step_b
    acc_tl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_bl = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_tr = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)
    acc_br = tl.zeros((HALF_M, HALF_N), dtype=tl.float32)

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
    for k in tl.range(0, n_steps - 2, 2, num_stages=1):
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
    # iter n_steps-2
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0  # (n_steps - 2) % 2, always 0 since n_steps is even
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

    # iter n_steps-1: finish ALL four mfmas before returning so the dot operands
    # die and the caller holds only the four f32 accumulators.
    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(1)
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, g_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(0)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, g_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    acc_br = tl.dot(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


def _streamk_schedule(M, N, K, block_m=BLOCK_M, block_n=BLOCK_N):
    """Build a persistent or variable-work Stream-K schedule."""
    num_pid_m = M // block_m
    num_pid_n = N // block_n
    total_tiles = num_pid_m * num_pid_n
    n_full = K // BLOCK_K
    k_pipe_steps = (n_full // 2) * 2
    k_pipe_pairs = k_pipe_steps // 2
    streamk_tiles = total_tiles % NUM_CU
    total_streamk_units = streamk_tiles * k_pipe_pairs
    # The optimized fixup assumes one resident full-tile wave followed by the
    # distributed tail. Multi-wave full-tile loops currently put the helper's
    # async waits inside an outer warp-pipeline region on the AMD pipeline pass.
    use_streamk = (K == k_pipe_steps * BLOCK_K and total_tiles - streamk_tiles == NUM_CU and streamk_tiles > 0
                   and total_streamk_units >= NUM_CU)
    units_per_program = total_streamk_units // NUM_CU if use_streamk else 0
    remainder_units = total_streamk_units % NUM_CU if use_streamk else 0

    return {
        "HAS_STREAMK": use_streamk,
        "HAS_K_TAIL": K != k_pipe_steps * BLOCK_K,
        "NUM_PROGRAMS": NUM_CU if use_streamk else min(NUM_CU, total_tiles),
        "NUM_FULL_TILES": total_tiles - streamk_tiles if use_streamk else total_tiles,
        "NUM_PID_M": num_pid_m,
        "NUM_PID_N": num_pid_n,
        "K_PIPE_STEPS": k_pipe_steps,
        "K_PIPE_PAIRS": k_pipe_pairs,
        "UNITS_PER_PROGRAM": units_per_program,
        "REMAINDER_UNITS": remainder_units,
    }


@triton.jit
def a16w16_8wave(
    a_ptr,
    b_ptr,
    bias_ptr,
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
    stride_bias_m,
    stride_bias_n,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    GRID_MN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    HAS_REGISTER_TAIL: tl.constexpr,
    PIN_OFFSET_LAYOUT: tl.constexpr,
    DEFER_EPILOGUE: tl.constexpr,
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
    if PIN_OFFSET_LAYOUT:
        stride_am = tl.multiple_of(stride_am, 8)
        stride_bn = tl.multiple_of(stride_bn, 8)

    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2

    # Four separate double-buffered LDS allocations — one per operand half-tile.
    # Pin the *swizzled* padded_shared layout (row/col-permuted offset bases) so
    # the ds_reads feeding the MFMAs are bank-conflict-free. The default inferred
    # padded layout ({order, shape})
    # conflicts on CDNA4 (measured 50M SQ_LDS_BANK_CONFLICT vs 0 for this one).
    # Swizzle bases are derived from the half-tile shape (_swz_offset_bases), so the
    # 256x256 (128x64 / 64x128 halves) and thin-N 128x128 (64x64 halves) tiles share
    # one path -- the 64-wide free axis just drops the top bit the 128-wide one adds.
    # TODO(perf): the 64x64 swizzle still shows ~1.5M SQ_LDS_BANK_CONFLICT (10%
    # LDS stall) vs 0 for 128x64. It can't be made conflict-free as a padded layout
    # (direct-to-LDS needs pad interval >=512, but 64x64 lacks a high offset bit for
    # the 4th MFMA row-bit); a swizzled_shared layout is conflict-free but slower
    # (gfx950 has no direct-to-LDS scattering -> extra write swizzle). Net: this
    # padded layout is the fastest option and still beats vendor -- the stall is the
    # price of the cheap direct-to-LDS write on a small square tile.
    a_bases: tl.constexpr = _A_BASES_256 if BLOCK_M == 256 else _A_BASES_128
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [HALF_M, BLOCK_K])
    b_shared: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_shared)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), tlx.dtype_of(a_ptr), 2, layout=a_shared)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), tlx.dtype_of(b_ptr), 2, layout=b_shared)

    # The direct-to-LDS buffer_load write is coalesced only when each offset
    # tensor's #linear layout matches the swizzled LDS layout above. We pin only
    # the shared layouts; the matching offset layouts are inferred from them by
    # tlx-insert-require-layout (no explicit offset_layout= needed).
    offs_am = pid_m * BLOCK_M + tl.arange(0, HALF_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_row_off = offs_am[:, None] * stride_am
    b_col_off = offs_bn[None, :] * stride_bn
    if PIN_OFFSET_LAYOUT:
        a_row_off = tl.multiple_of(a_row_off, (8, 8))
        b_col_off = tl.multiple_of(b_col_off, (8, 8))
    a_top_off = a_row_off + offs_k[None, :] * stride_ak
    a_bot_off = a_top_off + HALF_M * stride_am
    b_left_off = offs_k[:, None] * stride_bk + b_col_off
    b_right_off = b_left_off + HALF_N * stride_bn
    if PIN_OFFSET_LAYOUT:
        a_top_off = tlx.require_layout(a_top_off, _A_OFFSET_LAYOUT_256)
        a_bot_off = tlx.require_layout(a_bot_off, _A_OFFSET_LAYOUT_256)
        b_left_off = tlx.require_layout(b_left_off, _B_OFFSET_LAYOUT_256)
        b_right_off = tlx.require_layout(b_right_off, _B_OFFSET_LAYOUT_256)

    # Keep this pipeline inline: its K-contiguous B producer layout is inferred
    # together with the bank-conflict-free LDS layout. Moving it through a JIT
    # helper boundary loses that relationship on current layout propagation.
    a_top_off_n = a_top_off + BLOCK_K * stride_ak
    a_bot_off_n = a_bot_off + BLOCK_K * stride_ak
    b_left_off_n = b_left_off + BLOCK_K * stride_bk
    b_right_off_n = b_right_off + BLOCK_K * stride_bk

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
    # tail after the epilogue.
    n_full = KS // BLOCK_K
    n_pipe = (n_full // 2) * 2

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

    for k in tl.range(0, n_pipe - 2, 2, num_stages=1):
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

    acc_tl = tl.dot(a_top, b_left, acc_tl)
    tlx.async_load_wait_group(5)
    l_idx: tl.constexpr = 0
    a_bot = tlx.local_load(tlx.local_view(smem_a_bot, l_idx), relaxed=True)

    acc_bl = tl.dot(a_bot, b_left, acc_bl)
    tlx.async_load_wait_group(4)
    b_right = tlx.local_load(tlx.local_view(smem_b_right, l_idx), relaxed=True)

    acc_tr = tl.dot(a_top, b_right, acc_tr)
    tlx.async_load_wait_group(3)
    g_idx: tl.constexpr = 1
    b_left = tlx.local_load(tlx.local_view(smem_b_left, g_idx), relaxed=True)

    acc_br = tl.dot(a_bot, b_right, acc_br)
    tlx.async_load_wait_group(2)
    a_top = tlx.local_load(tlx.local_view(smem_a_top, g_idx), relaxed=True)

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
    if HAS_REGISTER_TAIL:
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

    if SPLIT_K == 1 and not DEFER_EPILOGUE:
        if ADD_BIAS:
            acc_tl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_top & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_bl += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_left[None, :],
                mask=m_bot & n_left,
                other=0.0,
            ).to(tl.float32)
            acc_tr += tl.load(
                bias_ptr + stride_bias_m * offs_cm_top[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_top & n_right,
                other=0.0,
            ).to(tl.float32)
            acc_br += tl.load(
                bias_ptr + stride_bias_m * offs_cm_bot[:, None] + stride_bias_n * offs_cn_right[None, :],
                mask=m_bot & n_right,
                other=0.0,
            ).to(tl.float32)

        # Direct store to C.
        et = c_ptr.dtype.element_ty
        if HALF_M == 128 and HALF_N == 128:
            # Stop the wide epilogue-store layout from propagating backward
            # through the extracted tile function. Split-K and the 128 tile keep
            # the original inferred accumulator layout.
            acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                           warps_per_cta=[2, 4])
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
            # Pin each 128x128 quadrant to the coalesced SIMD #linear layout (no LDS
            # staging) so OptimizeEpilogue keeps the wide dwordx4 store.
            # _C_STORE_SIMD_LAYOUT is derived for the 128x128 quadrant, so it only
            # applies to the 256x256 tile; a smaller tile (64x64 quadrant) uses a
            # plain store.
            L: tl.constexpr = _C_STORE_SIMD_LAYOUT
            c_tl = tlx.require_layout(acc_tl.to(et), L)
            # Static guard (no device code): the pin must survive coalesce /
            # remove-layout-conversions / AMD optimize-epilogue so the store stays
            # a wide dwordx4. Fails compilation if a future change drops the pin.
            tlx.assert_same_layout(c_tl, L)
            tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_left[None, :], c_tl,
                     mask=m_top & n_left)
            tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_left[None, :],
                     tlx.require_layout(acc_bl.to(et), L), mask=m_bot & n_left)
            tl.store(c_ptr + stride_cm * offs_cm_top[:, None] + stride_cn * offs_cn_right[None, :],
                     tlx.require_layout(acc_tr.to(et), L), mask=m_top & n_right)
            tl.store(c_ptr + stride_cm * offs_cm_bot[:, None] + stride_cn * offs_cn_right[None, :],
                     tlx.require_layout(acc_br.to(et), L), mask=m_bot & n_right)
        else:
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


@triton.jit
def _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                      HAS_K_TAIL: tl.constexpr, c_layout: tl.constexpr):
    """Compute and store one complete output tile."""
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_m_top = pid_m * BLOCK_M + offs_m
    offs_m_bot = offs_m_top + HALF_M
    offs_n_left = pid_n * BLOCK_N + offs_n
    offs_n_right = offs_n_left + HALF_N
    a_top_off = offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_bot_off = offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_left_off = offs_k[:, None] * stride_bk + offs_n_left[None, :] * stride_bn
    b_right_off = offs_k[:, None] * stride_bk + offs_n_right[None, :] * stride_bn
    acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                 a_top_off, a_bot_off, b_left_off, b_right_off, 0, 0, K_PIPE_STEPS,
                                                 stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)
    if HAS_K_TAIL:
        # Mask odd full and/or partial K64 steps left after the even pipelined prefix.
        for kk in tl.range(K_PIPE_STEPS * BLOCK_K, K, BLOCK_K, num_stages=1):
            offs_kt = kk + offs_k
            k_mask = offs_kt < K
            a_top = tl.load(a_ptr + offs_m_top[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            a_bot = tl.load(a_ptr + offs_m_bot[:, None] * stride_am + offs_kt[None, :] * stride_ak,
                            mask=k_mask[None, :], other=0.0)
            b_left = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_left[None, :] * stride_bn,
                             mask=k_mask[:, None], other=0.0)
            b_right = tl.load(b_ptr + offs_kt[:, None] * stride_bk + offs_n_right[None, :] * stride_bn,
                              mask=k_mask[:, None], other=0.0)
            acc_tl = tl.dot(a_top, b_left, acc_tl)
            acc_bl = tl.dot(a_bot, b_left, acc_bl)
            acc_tr = tl.dot(a_top, b_right, acc_tr)
            acc_br = tl.dot(a_bot, b_right, acc_br)
    et: tl.constexpr = c_ptr.dtype.element_ty
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_tl.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_left[None, :] * stride_cn,
             tlx.require_layout(acc_bl.to(et), c_layout))
    tl.store(c_ptr + offs_m_top[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_tr.to(et), c_layout))
    tl.store(c_ptr + offs_m_bot[:, None] * stride_cm + offs_n_right[None, :] * stride_cn,
             tlx.require_layout(acc_br.to(et), c_layout))


@triton.jit
def _grouped_tile_coords(tile_id, NUM_PID_M: tl.constexpr, NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Map a linear tile ID to the grouped M-major output grid."""
    tiles_per_group: tl.constexpr = GROUP_SIZE_M * NUM_PID_N
    group_id = tile_id // tiles_per_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(NUM_PID_M - first_pid_m, GROUP_SIZE_M)
    tile_in_group = tile_id % tiles_per_group
    pid_m = first_pid_m + tile_in_group % group_size_m
    pid_n = tile_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _wait_for_streamk_partial(locks_ptr, slot, ready_value):
    """Wait until one producer has published its partial tile."""
    while tl.load(locks_ptr + slot, cache_modifier=".cv", volatile=True) != ready_value:
        pass


@triton.jit
def _reduce_and_store_streamk_quadrant(
    partials_ptr,
    c_ptrs,
    first_contributor,
    num_contributors: tl.constexpr,
    partial_off,
    tile_elems: tl.constexpr,
    acc_layout: tl.constexpr,
    c_layout: tl.constexpr,
):
    """Reduce one quadrant across a tile's contributors and store it."""
    acc = tlx.require_layout(tl.load(partials_ptr + first_contributor * tile_elems + partial_off, cache_modifier=".cv"),
                             acc_layout, pin=False)
    for peer in range(1, num_contributors):
        acc += tlx.require_layout(
            tl.load(partials_ptr + (first_contributor + peer) * tile_elems + partial_off, cache_modifier=".cv"),
            acc_layout, pin=False)
    tl.store(c_ptrs, tlx.require_layout(acc.to(c_ptrs.dtype.element_ty), c_layout))


@triton.jit
def streamk_kernel(a_ptr, b_ptr, c_ptr, partials_ptr, locks_ptr, ready_value, K, stride_am, stride_ak, stride_bk,
                   stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                   NUM_XCDS: tl.constexpr, NUM_CU: tl.constexpr, NUM_PROGRAMS: tl.constexpr, HAS_STREAMK: tl.constexpr,
                   NUM_FULL_TILES: tl.constexpr, HAS_K_TAIL: tl.constexpr, NUM_PID_M: tl.constexpr,
                   NUM_PID_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, K_PIPE_STEPS: tl.constexpr,
                   K_PIPE_PAIRS: tl.constexpr, UNITS_PER_PROGRAM: tl.constexpr, REMAINDER_UNITS: tl.constexpr):
    """Persistent full tiles plus owner or distributed Stream-K fixup."""
    pid = tl.program_id(0)
    contributors_per_tile: tl.constexpr = K_PIPE_PAIRS // max(UNITS_PER_PROGRAM, 1)
    # When every flattened interval is one equal segment of one tile, all
    # contributors can participate in fixup instead of serializing it in the
    # tile owner. This is a reduction optimization, not a separate schedule.
    DISTRIBUTED_FIXUP: tl.constexpr = (HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS and REMAINDER_UNITS == 0
                                       and K_PIPE_PAIRS % max(UNITS_PER_PROGRAM, 1) == 0
                                       and (contributors_per_tile == 2 or contributors_per_tile == 4))
    HALF_M: tl.constexpr = BLOCK_M // 2
    HALF_N: tl.constexpr = BLOCK_N // 2
    acc_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                   warps_per_cta=[2, 4])
    a_bases: tl.constexpr = _A_BASES_256 if BLOCK_M == 256 else _A_BASES_128
    b_bases: tl.constexpr = _B_BASES_256 if BLOCK_N == 256 else _B_BASES_128
    a_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], a_bases, [HALF_M, BLOCK_K])
    b_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases([(512, 16)], b_bases, [BLOCK_K, HALF_N])
    et: tl.constexpr = a_ptr.dtype.element_ty
    smem_a_top = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_a_bot = tlx.local_alloc((HALF_M, BLOCK_K), et, 2, layout=a_layout)
    smem_b_left = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    smem_b_right = tlx.local_alloc((BLOCK_K, HALF_N), et, 2, layout=b_layout)
    offs_m = tl.arange(0, HALF_M)
    offs_n = tl.arange(0, HALF_N)
    offs_k = tl.arange(0, BLOCK_K)
    C: tl.constexpr = _C_STORE_SIMD_LAYOUT if BLOCK_M == 256 and BLOCK_N == 256 else acc_layout
    tile_elems: tl.constexpr = BLOCK_M * BLOCK_N
    partial_tl_off = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    partial_bl_off = partial_tl_off + HALF_M * BLOCK_N
    partial_tr_off = partial_tl_off + HALF_N
    partial_br_off = partial_bl_off + HALF_N
    stream_pid = pid
    if HAS_STREAMK:
        stream_pid = (pid % NUM_XCDS) * (NUM_CU // NUM_XCDS) + pid // NUM_XCDS
        # Fuse TritonBLAS-style lock initialization into the resident kernel;
        # each program clears the slot it may later publish.
        tl.store(locks_ptr + stream_pid, 0, cache_modifier=".wt")
        tl.debug_barrier()

    if HAS_STREAMK and NUM_FULL_TILES == NUM_PROGRAMS:
        # Fast path: each Stream-K program owns one full tile, so no loop is needed.
        head_pid_m = stream_pid % NUM_PID_M
        head_pid_n = stream_pid // NUM_PID_M
        _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, head_pid_m,
                          head_pid_n, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M,
                          BLOCK_N, BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    else:
        # General path for both persistent and generic Stream-K.
        pids_per_xcd: tl.constexpr = (NUM_FULL_TILES + NUM_XCDS - 1) // NUM_XCDS
        remainder_xcds: tl.constexpr = NUM_FULL_TILES % NUM_XCDS
        tall_xcds: tl.constexpr = NUM_XCDS if remainder_xcds == 0 else remainder_xcds
        for virtual_pid in range(pid, NUM_FULL_TILES, NUM_PROGRAMS):
            xcd = virtual_pid % NUM_XCDS
            local_pid = virtual_pid // NUM_XCDS
            if xcd < tall_xcds:
                tile_id = xcd * pids_per_xcd + local_pid
            else:
                tile_id = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            _matmul_full_tile(a_ptr, b_ptr, c_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right, pid_m, pid_n, K,
                              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M, BLOCK_N,
                              BLOCK_K, K_PIPE_STEPS, HAS_K_TAIL, C)
    if not HAS_STREAMK:
        return

    if DISTRIBUTED_FIXUP:
        # Unlike owner-based (standard Stream-K) fixup, every contributor publishes
        # its partial tile, synchronizes, and reduces a disjoint output region.
        # This lets all contributing CTAs actively reduce instead of relying on a
        # single owner CTA, as in standard Stream-K.
        contributor_id = stream_pid % contributors_per_tile
        tail_tile = NUM_FULL_TILES + stream_pid // contributors_per_tile

        tail_pid_m, tail_pid_n = _grouped_tile_coords(tail_tile, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
        tail_offs_m_top = tail_pid_m * BLOCK_M + offs_m
        tail_offs_m_bot = tail_offs_m_top + HALF_M
        tail_offs_n_left = tail_pid_n * BLOCK_N + offs_n
        tail_offs_n_right = tail_offs_n_left + HALF_N
        tail_a_top_off = tail_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_a_bot_off = tail_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
        tail_b_left_off = offs_k[:, None] * stride_bk + tail_offs_n_left[None, :] * stride_bn
        tail_b_right_off = offs_k[:, None] * stride_bk + tail_offs_n_right[None, :] * stride_bn
        segment_k_steps: tl.constexpr = UNITS_PER_PROGRAM * 2
        segment_k_offset = contributor_id * segment_k_steps * BLOCK_K
        acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left, smem_b_right,
                                                     tail_a_top_off, tail_a_bot_off, tail_b_left_off, tail_b_right_off,
                                                     segment_k_offset * stride_ak, segment_k_offset * stride_bk,
                                                     segment_k_steps, stride_ak, stride_bk, BLOCK_M, BLOCK_N, BLOCK_K)

        # Pin and publish all four partial quadrants before any contributor waits.
        # This avoids cyclic dependencies and lets the MFMA accumulators die before
        # fixup. Publishing a runtime-selected quadrant instead costs more VGPR and
        # select work than it saves in workspace traffic on gfx950.
        acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
        acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
        acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
        acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)
        partial_base = partials_ptr + stream_pid * tile_elems
        partial_tl_ptr = tlx.require_layout(partial_base + partial_tl_off, acc_layout, pin=False)
        partial_bl_ptr = tlx.require_layout(partial_base + partial_bl_off, acc_layout, pin=False)
        partial_tr_ptr = tlx.require_layout(partial_base + partial_tr_off, acc_layout, pin=False)
        partial_br_ptr = tlx.require_layout(partial_base + partial_br_off, acc_layout, pin=False)
        tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
        tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
        tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
        tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
        tl.debug_barrier()
        tl.store(locks_ptr + stream_pid, ready_value, cache_modifier=".wt")

        # Cooperatively reduce the tile, assigning consecutive output quadrants to
        # each contributor so all programs remain useful during fixup.
        first_contributor = stream_pid - contributor_id
        for peer in range(contributors_per_tile):
            _wait_for_streamk_partial(locks_ptr, first_contributor + peer, ready_value)
        # Two contributors own the left and right halves; four contributors
        # own one quadrant each.
        bottom_left_owner: tl.constexpr = 0 if contributors_per_tile == 2 else 1
        top_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 2
        bottom_right_owner: tl.constexpr = 1 if contributors_per_tile == 2 else 3
        if contributor_id == 0:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tl_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_left_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_left[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_bl_off, tile_elems, acc_layout, C)
        if contributor_id == top_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_top[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_tr_off, tile_elems, acc_layout, C)
        if contributor_id == bottom_right_owner:
            _reduce_and_store_streamk_quadrant(
                partials_ptr, c_ptr + tail_offs_m_bot[:, None] * stride_cm + tail_offs_n_right[None, :] * stride_cn,
                first_contributor, contributors_per_tile, partial_br_off, tile_elems, acc_layout, C)
    else:
        # Standard Stream-K fixup path: Divide remaining flattened pairs of K
        # pipeline steps almost evenly across CUs.
        logical_pid = stream_pid
        streamk_base = NUM_FULL_TILES * K_PIPE_PAIRS
        start_unit = (streamk_base + logical_pid * UNITS_PER_PROGRAM + min(logical_pid, REMAINDER_UNITS))
        last_unit = (streamk_base + (logical_pid + 1) * UNITS_PER_PROGRAM + min(logical_pid + 1, REMAINDER_UNITS))

        while start_unit < last_unit:
            tile_id = start_unit // K_PIPE_PAIRS
            tile_start = tile_id * K_PIPE_PAIRS
            tile_end = tile_start + K_PIPE_PAIRS
            segment_end = min(last_unit, tile_end)
            pid_m, pid_n = _grouped_tile_coords(tile_id, NUM_PID_M, NUM_PID_N, GROUP_SIZE_M)
            tile_offs_m_top = pid_m * BLOCK_M + offs_m
            tile_offs_m_bot = tile_offs_m_top + HALF_M
            tile_offs_n_left = pid_n * BLOCK_N + offs_n
            tile_offs_n_right = tile_offs_n_left + HALF_N
            tile_a_top_off = tile_offs_m_top[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_a_bot_off = tile_offs_m_bot[:, None] * stride_am + offs_k[None, :] * stride_ak
            tile_b_left_off = offs_k[:, None] * stride_bk + tile_offs_n_left[None, :] * stride_bn
            tile_b_right_off = offs_k[:, None] * stride_bk + tile_offs_n_right[None, :] * stride_bn
            k_step = (start_unit - tile_start) * 2 * BLOCK_K
            acc_tl, acc_bl, acc_tr, acc_br = matmul_tile(a_ptr, b_ptr, smem_a_top, smem_a_bot, smem_b_left,
                                                         smem_b_right, tile_a_top_off, tile_a_bot_off, tile_b_left_off,
                                                         tile_b_right_off, k_step * stride_ak, k_step * stride_bk,
                                                         (segment_end - start_unit) * 2, stride_ak, stride_bk, BLOCK_M,
                                                         BLOCK_N, BLOCK_K)
            acc_tl = tlx.require_layout(acc_tl, acc_layout, pin=False)
            acc_bl = tlx.require_layout(acc_bl, acc_layout, pin=False)
            acc_tr = tlx.require_layout(acc_tr, acc_layout, pin=False)
            acc_br = tlx.require_layout(acc_br, acc_layout, pin=False)

            if start_unit != tile_start:
                # A contributor publishes one partial. Its range can then start
                # the next tile, so continue instead of returning immediately.
                base = logical_pid * tile_elems
                partial_tl_ptr = tlx.require_layout(partials_ptr + base + partial_tl_off, acc_layout, pin=False)
                partial_bl_ptr = tlx.require_layout(partials_ptr + base + partial_bl_off, acc_layout, pin=False)
                partial_tr_ptr = tlx.require_layout(partials_ptr + base + partial_tr_off, acc_layout, pin=False)
                partial_br_ptr = tlx.require_layout(partials_ptr + base + partial_br_off, acc_layout, pin=False)
                tl.store(partial_tl_ptr, acc_tl, cache_modifier=".wt")
                tl.store(partial_bl_ptr, acc_bl, cache_modifier=".wt")
                tl.store(partial_tr_ptr, acc_tr, cache_modifier=".wt")
                tl.store(partial_br_ptr, acc_br, cache_modifier=".wt")
                tl.debug_barrier()
                tl.store(locks_ptr + logical_pid, ready_value, cache_modifier=".wt")
            else:
                # The program owning the first work unit of a tile collects the
                # following contributors, exactly as the TritonBLAS fixup does.
                covered_end = segment_end
                next_pid = logical_pid + 1
                while covered_end < tile_end:
                    _wait_for_streamk_partial(locks_ptr, next_pid, ready_value)
                    peer_base = next_pid * tile_elems
                    acc_tl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_bl += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_bl_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_tr += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_tr_off, cache_modifier=".cv"), acc_layout, pin=False)
                    acc_br += tlx.require_layout(
                        tl.load(partials_ptr + peer_base + partial_br_off, cache_modifier=".cv"), acc_layout, pin=False)
                    covered_end += UNITS_PER_PROGRAM + (next_pid < REMAINDER_UNITS)
                    next_pid += 1
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_tl.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_left[None, :] * stride_cn,
                         tlx.require_layout(acc_bl.to(et), C))
                tl.store(c_ptr + tile_offs_m_top[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_tr.to(et), C))
                tl.store(c_ptr + tile_offs_m_bot[:, None] * stride_cm + tile_offs_n_right[None, :] * stride_cn,
                         tlx.require_layout(acc_br.to(et), C))
            start_unit = segment_end


_TORCH_TO_TL = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def _reduce_k_kernel(workspace_ptr, bias_ptr, c_ptr, M, N, stride_bias_m, stride_bias_n, SPLIT_K: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, OUTPUT_DTYPE: tl.constexpr,
                     ADD_BIAS: tl.constexpr):
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
    if ADD_BIAS:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bias_m + offs_n[None, :] * stride_bias_n,
            mask=mask,
            other=0.0,
        )
        acc += bias.to(tl.float32)
    tl.store(c_ptr + base_offs, acc.to(OUTPUT_DTYPE), mask=mask)


NUM_CU = 256  # gfx950 (CDNA4) compute units
# Minimum K-tiles per split. Two forces set this floor: (1) below it the per-split
# prologue/epilogue overhead dominates the shrinking K work; (2) more splits means
# a proportionally larger fp32 workspace for the reduce to stream back (reduce cost
# ~ SPLIT_K*M*N), so an over-split that only marginally improves the GEMM loses the
# gain to the reduce. Every measured production optimum uses >= 16 tiles/split
# (e.g. K=12288 wants SPLIT_K=12 (16 tiles) not 16 (12 tiles); the latter fills the
# CUs but its extra reduce traffic makes it net slower).
MIN_KTILES_PER_SPLIT = 16

# Tile candidates, largest first. The big tile is the tuned default; the smaller
# one is used only when the big tile can't fill the CUs (see choose_tile).
# choose_tile scans the fallbacks generically, so adding another tile here (e.g.
# (64, 64)) needs no logic change; today only the 128x128 fallback is used.
TILE_CANDIDATES = ((256, 256), (128, 128))


def _split_k_for(grid_mn, K):
    """Largest SPLIT_K keeping grid_mn*SK within one CU wave and each split a whole,
    BLOCK_K-aligned chunk of >= MIN_KTILES_PER_SPLIT tiles.

    All divisors of K are considered, not just powers of two: for K with odd factors
    (e.g. 22272 = 64*348) a non-pow2 SPLIT_K divides K and fills the CUs far more
    precisely than the nearest pow2 (SPLIT_K=12 -> 192 CUs vs SPLIT_K=4 -> 64 CUs).
    The scan is <= NUM_CU/grid_mn (~20) iterations, negligible at compile time."""
    min_ks = MIN_KTILES_PER_SPLIT * BLOCK_K
    best = 1
    for sk in range(2, NUM_CU // grid_mn + 1):  # grid_mn*sk <= NUM_CU
        ks = K // sk
        if K % sk == 0 and ks >= min_ks and ks % BLOCK_K == 0:
            best = sk  # fill = grid_mn*sk grows with sk, so the last valid sk wins
    return best


def choose_tile(M, N, K):
    """Pick (BLOCK_M, BLOCK_N, SPLIT_K) by CU fill -- no shape hardcoding.

    Prefer the tuned 256x256 tile; it is more MFMA-efficient per work-group than the
    128x128 tile. Fall back to the smaller tile only when the 256 grid leaves most of
    the machine idle even after split-K (fill < NUM_CU/2) -- the genuinely thin-N /
    small-tile-count shapes (e.g. N=256, gmn=8): the 4x-denser MN grid then reaches
    occupancy the big tile can't. When the big tile fills at least half the CUs, its
    efficiency beats a full grid of small tiles, so it is kept (comparing raw
    work-group counts across tile sizes is apples-to-oranges -- a 128 tile does 1/4
    the work -- so a bigger small-tile count does not mean it is faster)."""
    bm, bn = TILE_CANDIDATES[0]
    gmn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = _split_k_for(gmn, K)
    best_fill = gmn * sk
    if best_fill < NUM_CU // 2:  # big tile leaves most CUs idle even with split-K
        for cbm, cbn in TILE_CANDIDATES[1:]:
            g = triton.cdiv(M, cbm) * triton.cdiv(N, cbn)
            s = _split_k_for(g, K)
            if g * s > best_fill:  # smaller tile fills the machine better
                bm, bn, sk, best_fill = cbm, cbn, s, g * s
    return bm, bn, sk


def choose_split_k(M, N, K):
    """Back-compat: SPLIT_K for the auto-chosen tile."""
    return choose_tile(M, N, K)[2]


def _launch(a, b, bias=None, SPLIT_K=None, TILE=None, K_LIMIT=None, DEFER_EPILOGUE=False):
    """Launch the shared gfx950 GEMM core, optionally with a fused bias."""
    M, input_k = a.shape
    b_k, N = b.shape
    assert input_k == b_k, "Incompatible dimensions"
    K = input_k if K_LIMIT is None else K_LIMIT
    assert 0 < K <= input_k, f"K_LIMIT={K} must be in (0, {input_k}]"
    if bias is not None:
        assert bias.shape == (M, N), f"Bias must expand to ({M}, {N}), got {tuple(bias.shape)}"
        assert bias.device == a.device, "Bias and matrix operands must be on the same device"
        assert bias.dtype == a.dtype, "Bias and matrix operands must have the same dtype"
    if TILE is not None:
        BM, BN = TILE
        grid_mn = triton.cdiv(M, BM) * triton.cdiv(N, BN)
        SPLIT_K = _split_k_for(grid_mn, K) if SPLIT_K is None else SPLIT_K
    elif SPLIT_K is None:
        BM, BN, SPLIT_K = choose_tile(M, N, K)
    else:
        BM, BN = BLOCK_M, BLOCK_N  # explicit SPLIT_K override keeps the default tile
    KS = K // SPLIT_K
    # Each split is big enough for the 2-tile prologue and starts on a 16-byte
    # boundary. Full BLOCK_K tiles use direct-to-LDS; the remainder is handled by
    # the masked register tail in the kernel.
    assert K % SPLIT_K == 0, f"K={K} must be divisible by SPLIT_K={SPLIT_K}"
    assert KS >= 2 * BLOCK_K, f"K/SPLIT_K={KS} must be at least {2 * BLOCK_K}"
    assert KS * a.element_size() % 16 == 0, f"K/SPLIT_K={KS} must preserve 16-byte split alignment"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    GRID_MN = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    if SPLIT_K > 1 or DEFER_EPILOGUE:
        # fp32 workspace: partials are stored without a rounding step, so the
        # split-K result matches a single fp32-accumulated GEMM (an fp16 workspace
        # would lose ~1e-1 near cancellation). The reduce sums in fp32 too.
        workspace = torch.empty((SPLIT_K * M, N), device=a.device, dtype=torch.float32)
    else:
        workspace = c  # dummy; the kernel writes c_ptr directly when SPLIT_K==1
    bias_ptr = bias if bias is not None else c
    stride_bias_m = bias.stride(0) if bias is not None else 0
    stride_bias_n = bias.stride(1) if bias is not None else 0
    a16w16_8wave[(GRID_MN * SPLIT_K, )](
        a,
        b,
        bias_ptr,
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
        stride_bias_m,
        stride_bias_n,
        c.stride(0),
        c.stride(1),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=4 if M == N and K >= 8192 else (2 if M <= 1024 and N >= 16384 else GROUP_SIZE_M),
        NUM_XCDS=1 if M == N and K >= 8192 else NUM_XCDS,
        GRID_MN=GRID_MN,
        SPLIT_K=SPLIT_K,
        ADD_BIAS=bias is not None,
        HAS_REGISTER_TAIL=KS % (2 * BLOCK_K) != 0,
        PIN_OFFSET_LAYOUT=K_LIMIT is not None,
        DEFER_EPILOGUE=DEFER_EPILOGUE,
        num_warps=NUM_WARPS,
        num_stages=1,
        matrix_instr_nonkdim=16,
        # Forbid AGPRs: f32 accumulators write VGPRs directly (packs tighter, no
        # v_accvgpr moves around each mfma). Essential to match the reference perf.
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"), ),
        enable_sched_group_barrier_scheduler=True,
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
            bias_ptr,
            c,
            M,
            N,
            stride_bias_m,
            stride_bias_n,
            SPLIT_K=SPLIT_K,
            BLOCK_SIZE_M=rbm,
            BLOCK_SIZE_N=rbn,
            OUTPUT_DTYPE=_TORCH_TO_TL[a.dtype],
            ADD_BIAS=bias is not None,
            num_warps=rw,
        )
    if DEFER_EPILOGUE:
        return workspace, c
    return c


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
    return _launch(a, b, SPLIT_K=SPLIT_K)


def _validate_streamk(a, b):
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype and a.dtype in (torch.float16, torch.bfloat16), \
        "streamk_matmul requires matching FP16 or BF16 operands"
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, \
        f"M and N must be multiples of {BLOCK_M}"
    assert K >= MIN_K, f"K must be at least {MIN_K}"
    assert K % (2 * BLOCK_K) == 0, f"K must be a multiple of {2 * BLOCK_K}"
    return M, N, K


def _choose_streamk_tile(M, N):
    """Use a smaller persistent tile only when the default grid underfills the GPU."""
    BM, BN = TILE_CANDIDATES[0]
    grid_mn = (M // BM) * (N // BN)
    if grid_mn < NUM_CU // 2:
        for candidate_m, candidate_n in TILE_CANDIDATES[1:]:
            candidate_grid = (M // candidate_m) * (N // candidate_n)
            if grid_mn < candidate_grid <= NUM_CU:
                BM, BN, grid_mn = candidate_m, candidate_n, candidate_grid
    return BM, BN


def streamk_matmul(a, b):
    """Run one persistent kernel with a variable-work Stream-K tail when profitable."""
    M, N, K = _validate_streamk(a, b)
    BM, BN = _choose_streamk_tile(M, N)
    schedule = _streamk_schedule(M, N, K, block_m=BM, block_n=BN)
    # Keep the extracted async pipeline out of an outer multi-wave tile loop;
    # the AMD warp-pipeline pass cannot nest its waits in that region. The
    # data-centric kernel keeps the same pipeline inline for this case.
    if not schedule["HAS_STREAMK"] and schedule["NUM_FULL_TILES"] > schedule["NUM_PROGRAMS"]:
        return matmul(a, b)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if schedule["HAS_STREAMK"]:
        partials = torch.empty((NUM_CU, BM, BN), device=a.device, dtype=torch.float32)
        locks = torch.empty((NUM_CU, ), device=a.device, dtype=torch.int32)
    else:
        partials = locks = c
    streamk_kernel[(schedule["NUM_PROGRAMS"], )](a, b, c, partials, locks, _READY_VALUE, K, a.stride(0), a.stride(1),
                                                 b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BM,
                                                 BLOCK_N=BN, BLOCK_K=BLOCK_K, NUM_XCDS=NUM_XCDS, NUM_CU=NUM_CU,
                                                 GROUP_SIZE_M=GROUP_SIZE_M, **schedule, num_warps=NUM_WARPS,
                                                 num_stages=1, matrix_instr_nonkdim=16, llvm_fn_attrs=_LLVM_ATTRS)
    return c
