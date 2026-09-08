"""MI300X (gfx942 / CDNA3) GEMM -- the `tlx.ops.mm` implementation.

Kernel promoted verbatim from the now-frozen `tutorials/amd_gemm_gfx942.py`;
what is new is the search-space plumbing -- lazy `_tuned`, a `heuristic_config`
so a first call does not autotune, and a `smoke` space.

The operand path is register-staged, which is what separates CDNA3 from the
gfx950 kernels next door:

    global --tl.load--> VGPR --tlx.local_store--> LDS --tlx.local_load--> MFMA

* Sized to CDNA3's 64 KB LDS, not CDNA4's 160 KB; `_prune_configs` drops
  anything over budget before it is compiled.
* XCD remap measured *neutral* (the GROUP_M swizzle already captures that
  reuse); kept as the standard MI300X transform. `NUM_XCDS=1` to A/B it.
* `matrix_instr_nonkdim=16` -- gfx942 fp16 MFMA is 16x16x16, half CDNA4's K.

No TMA, so operands are read through plain strided pointers with no alignment
constraint -- which is why this arch admits the odd-K shape sm100 declines.
"""
import functools

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ._shapes import GFX942_FOCUS

#: Shapes `bench_mm.py` gates on for this arch.
PERF_SHAPES = GFX942_FOCUS

NUM_XCDS = 8
NUM_CUS = 304

#: Deliberately below `NUM_CUS`: three autotuned winners land on 256 workgroups
#: and still beat the next tile down. `>= NUM_CUS` mispredicted 2048^3 and
#: 4096^3 by one rung.
_MIN_WORKGROUPS = 256

_LONG_K_TAIL_THRESHOLD = 16 * 1024

CDNA3_LDS_BYTES = 64 * 1024

#: Unified VGPR+AGPR entries per lane on one CDNA3 SIMD. `waves_per_eu=N` hands
#: each of N resident waves 512/N of them.
CDNA3_VGPRS_PER_SIMD = 512

#: `waves_per_eu` values `space="full"` searches. 4 is the only nonzero value
#: measured to win -- 1.02x on the long-K rung, 1.15x on 64x64x64 -- while 1 and
#: 3 cost up to 0.74x and 0.21x by landing on a worse occupancy step, and >= 5
#: spills on every rung.
_WAVES_PER_EU_SPACE = (0, 4)


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

    # Wrapping lets an edge tile re-read valid memory, so the hot loop's loads
    # need no M/N mask; the masked epilogue store discards the duplicated work.
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    K_ITERS = tl.cdiv(K, BLOCK_K)

    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.local_store(tlx.local_view(smem_a, i), a_reg)
        tlx.local_store(tlx.local_view(smem_b, i), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iteration k multiplies tile ``k - NUM_BUFFERS`` out of buffer
    # ``k % NUM_BUFFERS``, then refills that same buffer with tile k.
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

    # Drain the NUM_BUFFERS tiles still in the ring.
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


def acc_vgprs(block_m, block_n, num_warps):
    """Lanes' fp32 accumulator registers -- the floor under any register budget."""
    return block_m * block_n // (num_warps * 64)


def _fits_waves_per_eu(block_m, block_n, num_warps, waves_per_eu):
    """Whether 512/waves_per_eu leaves the accumulator room for operands too.

    The 256x256 tile's accumulator is already 128 registers, so pinning it to a
    128-register budget spills 370 and runs at 0.06x. Half the budget is the
    cheapest cutoff that rejects those without touching a measured winner.
    """
    if waves_per_eu == 0:
        return True
    return acc_vgprs(block_m, block_n, num_warps) * 2 <= CDNA3_VGPRS_PER_SIMD // waves_per_eu


def _config(block_m, block_n, block_k, group_m, num_buffers, num_warps, waves_per_eu=0):
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_BUFFERS": num_buffers,
            "NUM_XCDS": NUM_XCDS,
            "waves_per_eu": waves_per_eu,
        },
        num_warps=num_warps,
        # num_stages=1 keeps the automatic pipeliner out of a hand-managed ring.
        num_stages=1,
    )


def _configs():
    """Tiles worth trying on MI300X. Used by `space="full"`.

    Deliberately small: 64x64 fills the chip on a 1024^2 output, 256x256 needs
    4096^2 to saturate, and BLOCK_K stays at 32 at the wide end because the
    64 KB budget will not hold a deeper one. NUM_BUFFERS spans 1..3 so the
    single-buffered ring is the degenerate case, not a separate kernel; depth
    costs LDS linearly and is usually better spent on a wider tile.

    Crossed with `_WAVES_PER_EU_SPACE`, minus the pins whose budget the tile's
    accumulator alone would blow.
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
    return [
        _config(bm, bn, bk, gm, nb, warps, wpe)
        for (bm, bn, bk, warps) in tiles
        for gm in (4, 8)
        for nb in (1, 2, 3)
        for wpe in _WAVES_PER_EU_SPACE
        if _fits_waves_per_eu(bm, bn, warps, wpe)
    ]


CONFIGS = _configs


def _smoke_configs():
    """One config per distinct lowering path, for shapes the heuristic declines.

    The paths that differ: ring depth (1 is degenerate, >1 rotates), the XCD
    remap, and the two warp counts.
    """
    return [
        _config(64, 64, 64, 4, 1, 4),
        _config(128, 128, 32, 8, 2, 4),
        _config(128, 128, 32, 8, 3, 8),
    ]


SMOKE_CONFIGS = _smoke_configs

#: Tile ladder for `heuristic_config`, widest first, each carrying the settings
#: that won for it in a full-space autotune sweep on MI300X, fp16:
#:
#:     shape               autotuned winner              ladder picks
#:     1024^3              64x64x64   GM4 nb1 w4         same
#:     2048^3              128x128x64 GM8 nb2 w8         same
#:     4096^3              256x256x32 GM8 nb2 w8         same
#:     8192^3              256x256x32 GM8 nb2 w8         same
#:     8192x1024x8192      128x128x32 GM8 nb1 w4         same
#:     1024x8192x8192      128x128x32 GM4 nb1 w4         GM8 (only GROUP_M differs)
#:
#: Five of six exact -- not a lot of calibration, which is why `space="full"`
#: stays available and the perf suite gates the heuristic.
_TILE_LADDER = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, NUM_BUFFERS, num_warps, waves_per_eu)
    # waves_per_eu=0 means unset: the backend only emits the attribute when nonzero.
    (256, 256, 32, 8, 2, 8, 0),
    (128, 128, 64, 8, 2, 8, 0),
    # wpe=4 on the small rung is 1.04-1.21x over six shapes that select it, and
    # bit-identical. Not an occupancy change: registers and waves are unmoved at
    # 90 and 4, so this is purely the pinned budget steadying the scheduler.
    # (LLVM's reported occupancy says 5 -- it is register-only, because Triton
    # passes LDS at launch so `.group_segment_fixed_size` is 0.)
    (64, 64, 64, 4, 1, 4, 4),
]

#: The tile every narrow shape gets. Both measured rectangular cases chose it
#: over the wider tile with the same workgroup count.
_NARROW_TILE = (128, 128, 32, 8, 1, 4, 0)

#: The next rung's geometry with one LDS buffer, so a grid just over one device
#: wave decomposes into several better-filled ones.
#:
#: Its `waves_per_eu` buys a register budget here, not occupancy -- `num_warps`
#: and the LDS footprint are the direct occupancy levers. EU = SIMD, and `N`
#: pins the allocator to 512/N of the per-SIMD register file. LDS caps residency
#: at 4 waves/SIMD regardless, and the default stops at 106 VGPRs, so the 128
#: that 4 waves affords is free -- the scheduler spends it batching the
#: `ds_read`s behind counted `lgkmcnt` waits. 1.03-1.06x, and only here, because
#: NUM_BUFFERS=1 is what puts `ds_read` -> `mfma` on the critical path.
_LONG_K_TAIL_TILE = (128, 128, 64, 8, 1, 8, 4)

_NARROW_SIDE = 1024
_WIDE_SIDE = 4096


def heuristic_config(M, N, K):
    """The shape-picked config as a one-element space, or None if it declines.

    Two rules, both read off the sweep in `_TILE_LADDER`:

    1. A narrow shape -- one side <= 1024 while the other is >= 4096 -- takes
       `_NARROW_TILE` regardless of how many workgroups a wider tile would make.
    2. Otherwise take the widest tile that still produces at least
       `_MIN_WORKGROUPS`, except that an extreme-K grid between one and two
       full device waves takes the single-buffer specialization of the next
       rung to avoid a long under-filled tail. Fall back to the narrowest tile.

    The tile carries its own `waves_per_eu`: M/N/K are runtime arguments, so
    every shape on a rung compiles to the same binary and the pin belongs to the
    tile, not the shape. Note the autotune spaces do not vary it, so only a rung
    can supply one. Only `_LONG_K_TAIL_TILE` does; see the note there.

    Returns None when nothing in the ladder fits the LDS budget at this K, which
    sends the caller to the smoke space rather than off a cliff.
    """
    if min(M, N) <= _NARROW_SIDE <= _WIDE_SIDE <= max(M, N):
        candidates = [_NARROW_TILE]
    else:
        candidates = []
        long_k_tail = False
        for tile in _TILE_LADDER:
            workgroups = triton.cdiv(M, tile[0]) * triton.cdiv(N, tile[1])
            if workgroups < _MIN_WORKGROUPS:
                continue
            if K >= _LONG_K_TAIL_THRESHOLD and NUM_CUS < workgroups < 2 * NUM_CUS:
                long_k_tail = True
                continue
            candidates.append(tile)
        if long_k_tail:
            candidates = [_LONG_K_TAIL_TILE]
        candidates = candidates or [_TILE_LADDER[-1]]

    for block_m, block_n, block_k, group_m, num_buffers, num_warps, waves_per_eu in candidates:
        depth = min(num_buffers, max(triton.cdiv(K, block_k), 1))
        if lds_bytes(block_m, block_n, block_k, depth) > CDNA3_LDS_BYTES:
            continue
        return [_config(block_m, block_n, block_k, group_m, depth, num_warps, waves_per_eu)]
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
        matrix_instr_nonkdim=16,
    )
    return c


#: Entry point for the kernel-optimization agent. Reaches `mm`'s default
#: `space="heuristic"`, so results are not comparable to a `space="full"` number.
matmul = mm
