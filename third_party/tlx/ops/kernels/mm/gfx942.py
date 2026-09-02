"""MI300X (gfx942 / CDNA3) GEMM -- the `tlx.ops.mm` implementation.

Promoted from `tutorials/amd_gemm_gfx942.py`, which is now frozen. The kernel
below is that file's kernel verbatim; what is new here is the search-space
plumbing every op needs -- lazy `_tuned`, a `heuristic_config` so a first call
does not autotune, and a `smoke` space.

The operand path is register-staged, which is what separates CDNA3 from the
gfx950 kernels next door:

    global --tl.load--> VGPR --tlx.local_store--> LDS --tlx.local_load--> MFMA

Three things the kernel does, and why:

* **Sized to the 64 KB CDNA3 LDS budget**, not CDNA4's 160 KB. The gfx950
  kernels' 256x256x64 two-buffer ring wants ~128 KB and cannot be made to fit.
  `_prune_configs` drops anything over budget before it is compiled.
* **Program ids remapped across the 8 XCDs.** Measured *neutral* here -- 0.86x
  aten with the remap vs 0.87x without at 4096^3, inside the noise -- because
  the GROUP_M swizzle already captures that reuse. Kept as the standard MI300X
  grid transform; set `NUM_XCDS=1` to A/B it.
* **`matrix_instr_nonkdim=16`** -- gfx942 fp16 MFMA is 16x16x16 / 32x32x8, half
  the K of CDNA4's 16x16x32 / 32x32x16.

Unlike sm100, this op has no TMA, so it reads operands through plain strided
pointers and imposes no alignment constraint. That is why it admits the odd-K
production shape that sm100 declines.
"""
import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ._shapes import GFX942_FOCUS

#: The shapes `bench_mm.py` gates on for this arch. Correctness runs the union
#: of every arch's list; perf runs only its own.
PERF_SHAPES = GFX942_FOCUS

# MI300X: 8 XCDs, 304 CUs. Consecutive program ids are dispatched round-robin
# across the XCDs, so the remap below undoes that to restore tile locality.
NUM_XCDS = 8

#: Compute units on an MI300X.
NUM_CUS = 304

#: Workgroup count the heuristic requires before it will accept a wider tile.
#:
#: Deliberately *below* `NUM_CUS`, which is the counter-intuitive part. Three of
#: the six autotuned winners below land on exactly 256 workgroups -- 0.84 of a
#: wave, leaving ~48 CUs idle -- and beat the next tile down, which fills the
#: chip several times over. The wider tile's MFMA efficiency is worth more than
#: the idle CUs. A `>= NUM_CUS` threshold was tried first and mispredicted both
#: 2048^3 and 4096^3 by one rung.
_MIN_WORKGROUPS = 256

# Per-workgroup LDS on CDNA3. Configs are checked against this so an oversized
# tile is dropped before compilation instead of failing out-of-resources.
CDNA3_LDS_BYTES = 64 * 1024


@triton.jit
def _xcd_remap(pid, grid_mn, num_xcds: tl.constexpr):
    """Undo the hardware's round-robin XCD dispatch of consecutive program ids.

    Workgroup ``pid`` runs on XCD ``pid % num_xcds``. Left alone, tiles that
    should share B columns land on different chiplets with different L2s. This
    maps each XCD's slice of the grid back to a contiguous range of tile ids.
    Handles a grid that is not a multiple of ``num_xcds``: the first
    ``grid_mn % num_xcds`` XCDs get one extra tile.
    """
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    tall_xcds = grid_mn % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if xcd < tall_xcds:
        return xcd * pids_per_xcd + local_pid
    return tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid


@triton.jit
def matmul_kernel_gfx942(
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
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """C = A @ B, staging global -> VGPR -> LDS (the fastest operand path on CDNA3)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = _xcd_remap(pid, num_pid_m * num_pid_n, NUM_XCDS)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Wrap the row/column offsets so an edge tile re-reads valid memory; the
    # epilogue store is masked, so the duplicated work is discarded. This keeps
    # the hot loop's loads unmasked in M/N -- only K needs a mask.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    K_ITERS = tl.cdiv(K, BLOCK_K)

    # The bank-conflict-avoiding padded shared layout is inferred by the compiler
    # from how these buffers feed tl.dot -- see gfx9_gemm/a16w16 v3 vs v4 for the
    # explicit form and why the inferred one is identical.
    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    # Prologue: fill the whole LDS ring.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.local_store(tlx.local_view(smem_a, i), a_reg)
        tlx.local_store(tlx.local_view(smem_b, i), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop. Iteration k multiplies K tile ``k - NUM_BUFFERS``, which lives in
    # buffer ``k % NUM_BUFFERS`` (since ``(k - NUM_BUFFERS) % NUM_BUFFERS ==
    # k % NUM_BUFFERS``), and refills that same buffer with tile k. Both global
    # loads are issued first so their latency overlaps the MFMA burst below; the
    # local_store then lands on a buffer the dot has already consumed.
    #
    # num_stages=1 disables the automatic software pipeliner: the ring and the
    # prefetch distance are managed by hand.
    for k in tl.range(NUM_BUFFERS, K_ITERS, num_stages=1):
        buf = k % NUM_BUFFERS
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K)

        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

        tlx.local_store(tlx.local_view(smem_a, buf), a_reg)
        tlx.local_store(tlx.local_view(smem_b, buf), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Epilogue: drain the NUM_BUFFERS tiles still sitting in the ring. Tile
    # ``K_ITERS - NUM_BUFFERS + i`` is in buffer ``(K_ITERS + i) % NUM_BUFFERS``.
    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        buf = (K_ITERS + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def lds_bytes(block_m, block_n, block_k, num_buffers, elem_bytes=2):
    """LDS footprint of a tile, ignoring shared-layout padding."""
    return (block_m * block_k + block_k * block_n) * elem_bytes * num_buffers


def _config(block_m, block_n, block_k, group_m, num_buffers, num_warps):
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_BUFFERS": num_buffers,
            "NUM_XCDS": NUM_XCDS,
            "waves_per_eu": 0,
        },
        num_warps=num_warps,
        # The manual LDS ring does the pipelining, so the automatic software
        # pipeliner must bail out -- which it does at num_stages=1.
        num_stages=1,
    )


def _configs():
    """Tiles worth trying on MI300X. Used by `space="full"`.

    Deliberately small: with 304 CUs the useful range runs from a 64x64 tile
    (enough workgroups to fill the chip on a 1024^2 output, which only
    decomposes into 16 tiles of 256x256) up to 256x256 (which needs a 4096^2
    output before it saturates). BLOCK_K is mostly 32 because the 64 KB budget
    will not hold a deep K tile at the wide end -- 256x256x64 would want 128 KB.

    NUM_BUFFERS spans 1..3, so the single-buffered ring is in the space as the
    degenerate case rather than as a separate kernel. Depth is not the lever it
    looks like: it costs LDS linearly, and on a 64 KB budget that LDS is often
    better spent on a wider tile.
    """
    tiles = [
        (64, 64, 64, 4),
        (128, 128, 32, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (256, 128, 32, 8),
        (128, 256, 32, 8),
        (256, 256, 32, 8),
    ]
    return [_config(bm, bn, bk, gm, nb, warps) for (bm, bn, bk, warps) in tiles for gm in (4, 8) for nb in (1, 2, 3)]


CONFIGS = _configs


def _smoke_configs():
    """One config per distinct lowering path, for shapes the heuristic declines.

    The paths that actually differ here are ring depth (1 is the degenerate
    no-double-buffer case, >1 rotates), the XCD remap, and the two warp counts.
    """
    return [
        _config(64, 64, 64, 4, 1, 4),
        _config(128, 128, 32, 8, 2, 4),
        _config(128, 128, 32, 8, 3, 8),
    ]


SMOKE_CONFIGS = _smoke_configs

#: Tile ladder for `heuristic_config`, widest first, each with the ring depth,
#: group size and warp count that won for it during autotuning.
#:
#: Calibrated against a full-space autotune sweep on MI300X, fp16. Measured
#: winners, and what this ladder picks:
#:
#:     shape               autotuned winner              ladder picks
#:     1024^3              64x64x64   GM4 nb1 w4         same
#:     2048^3              128x128x64 GM8 nb2 w8         same
#:     4096^3              256x256x32 GM8 nb2 w8         same
#:     8192^3              256x256x32 GM8 nb2 w8         same
#:     8192x1024x8192      128x128x32 GM8 nb1 w4         same
#:     1024x8192x8192      128x128x32 GM4 nb1 w4         GM8 (only GROUP_M differs)
#:
#: Five of six exact. Six points is not a lot of calibration, which is why
#: `space="full"` stays available and the perf suite gates the heuristic.
_TILE_LADDER = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_BUFFERS, num_warps)
    (256, 256, 32, 8, 2, 8),
    (128, 128, 64, 8, 2, 8),
    (64, 64, 64, 4, 1, 4),
]

#: The tile every narrow shape gets. Both measured rectangular cases chose it
#: over the wider tile with the same workgroup count, so grid size alone does
#: not decide: at BLOCK_M=256 a shape with M=1024 has only four M-tiles, and the
#: GROUP_M swizzle has too little to work with to keep B resident in L2.
_NARROW_TILE = (128, 128, 32, 8, 1, 4)

#: A shape is "narrow" when one side is this small while the other is large.
_NARROW_SIDE = 1024
_WIDE_SIDE = 4096


def heuristic_config(M, N, K):
    """The shape-picked config as a one-element space, or None if it declines.

    Two rules, both read off the sweep in `_TILE_LADDER`:

    1. A narrow shape -- one side <= 1024 while the other is >= 4096 -- takes
       `_NARROW_TILE` regardless of how many workgroups a wider tile would make.
    2. Otherwise take the widest tile that still produces at least
       `_MIN_WORKGROUPS`, falling back to the narrowest tile in the ladder.

    Returns None when nothing in the ladder fits the LDS budget at this K, which
    sends the caller to the smoke space rather than off a cliff.
    """
    if min(M, N) <= _NARROW_SIDE <= _WIDE_SIDE <= max(M, N):
        candidates = [_NARROW_TILE]
    else:
        candidates = [t for t in _TILE_LADDER if triton.cdiv(M, t[0]) * triton.cdiv(N, t[1]) >= _MIN_WORKGROUPS]
        candidates = candidates or [_TILE_LADDER[-1]]

    for block_m, block_n, block_k, group_m, num_buffers, num_warps in candidates:
        # K must supply at least one tile per buffer, and the ring must fit.
        depth = min(num_buffers, max(triton.cdiv(K, block_k), 1))
        if lds_bytes(block_m, block_n, block_k, depth) > CDNA3_LDS_BYTES:
            continue
        return [_config(block_m, block_n, block_k, group_m, depth, num_warps)]
    return None


def _prune_configs(configs, named_args, **kwargs):
    """Drop tiles that cannot run on this shape before anything is compiled."""
    K = named_args["K"]
    elem_bytes = named_args["a_ptr"].element_size()
    kept = []
    for config in configs:
        bm = config.kwargs["BLOCK_M"]
        bn = config.kwargs["BLOCK_N"]
        bk = config.kwargs["BLOCK_K"]
        nb = config.kwargs["NUM_BUFFERS"]
        # K must supply at least one tile per buffer.
        if triton.cdiv(K, bk) < nb:
            continue
        if lds_bytes(bm, bn, bk, nb, elem_bytes) > CDNA3_LDS_BYTES:
            continue
        kept.append(config)
    if not kept:
        raise RuntimeError(f"No config fits K={K} within the {CDNA3_LDS_BYTES} B gfx942 LDS budget")
    return kept


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    """Autotuned kernel per search space; `shape` keys only the heuristic one."""
    if space == "heuristic":
        configs = heuristic_config(*shape) or SMOKE_CONFIGS()
    else:
        configs = {"full": CONFIGS, "smoke": SMOKE_CONFIGS}[space]()
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )(matmul_kernel_gfx942)


def mm(a, b, *, space="heuristic"):
    """Matrix multiply ``a @ b`` on MI300X.

    `space` selects the search space -- "full" for perf, "heuristic" (one
    config) for a first call that stays interactive, "smoke" for path coverage.
    Not exposed on `tlx.ops.mm`.

    Either operand may be column-major: the kernel indexes through explicit
    strides, so a transposed view costs nothing and needs no copy.
    """
    assert a.shape[1] == b.shape[0], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )  # noqa: E731
    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None)
    kernel[grid](
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
        # 16x16x16 MFMA on gfx942 fp16.
        matrix_instr_nonkdim=16,
    )
    return c


#: The kernel-optimization agent loads a source file and calls `matmul(a, b)`
#: (`harnesses/gfx942/targets/amd_gemm_gfx942/harness.py`). Aliasing it here
#: means the agent can be pointed at the shipped op rather than at the frozen
#: tutorial, so a win it finds lands on the code users run.
#:
#: Note this reaches `mm`'s default `space="heuristic"`, so the agent measures
#: one config rather than an autotuned winner. That is the right target -- the
#: heuristic is what a caller gets -- but it means an agent result is not
#: comparable to a `space="full"` number.
matmul = mm
